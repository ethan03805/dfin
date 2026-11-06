//! The stage inference engine: one OS thread that owns the llama `Context`.
//!
//! `Context` is `Send` but not `Sync`, and `llama_decode` is blocking CPU/GPU work
//! that must not run on the tokio runtime. Connection tasks therefore send [`Job`]s
//! over an mpsc channel; the engine thread processes them strictly FIFO and replies
//! through a oneshot. This is also where micro-batching across sessions will go.
//!
//! Session slots: each `(connection_id, session_id)` pair maps to one llama
//! sequence id in `0..n_seq_max`. A `Forward` for an unknown session allocates a
//! free slot; closing the session (or its connection) clears the slot's KV and
//! returns it to the free list.

use dfin_llama::{Context, OutputMode, SeqId};
use dfin_proto::{ActDtype, Activations, ErrorCode, Response};
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

/// Work for the engine thread. Every variant except `CloseConn` carries a reply channel.
pub enum Job {
    Forward {
        conn: u64,
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
        reply: oneshot::Sender<Response>,
    },
    SeqRm {
        conn: u64,
        session: u64,
        p0: i32,
        p1: i32,
        reply: oneshot::Sender<Response>,
    },
    Close {
        conn: u64,
        session: u64,
        reply: oneshot::Sender<Response>,
    },
    /// A connection went away: free every session it owned.
    CloseConn { conn: u64 },
}

pub struct Engine {
    ctx: Context,
    act_dtype: ActDtype,
    /// KV capacity of one session (`n_ctx / n_seq_max`).
    n_ctx_seq: i64,
    /// Max tokens per `decode` call.
    n_batch: usize,
    slots: HashMap<(u64, u64), SeqId>,
    free: Vec<SeqId>,
}

fn err(code: ErrorCode, message: impl Into<String>) -> Response {
    Response::Error {
        code,
        message: message.into(),
    }
}

impl Engine {
    pub fn new(ctx: Context, act_dtype: ActDtype, n_batch: usize) -> Engine {
        let n_seq_max = ctx.n_seq_max().max(1);
        let n_ctx_seq = (ctx.n_ctx() / n_seq_max) as i64;
        // Pop from the back, so slot 0 is handed out first.
        let free: Vec<SeqId> = (0..n_seq_max as SeqId).rev().collect();
        Engine {
            ctx,
            act_dtype,
            n_ctx_seq,
            n_batch,
            slots: HashMap::new(),
            free,
        }
    }

    /// Start the engine thread. It stops when every sender is dropped.
    pub fn spawn(self, queue: usize) -> std::io::Result<(mpsc::Sender<Job>, std::thread::JoinHandle<()>)> {
        let (tx, rx) = mpsc::channel(queue);
        let handle = std::thread::Builder::new()
            .name("dfin-engine".into())
            .spawn(move || self.run(rx))?;
        Ok((tx, handle))
    }

    fn run(mut self, mut rx: mpsc::Receiver<Job>) {
        while let Some(job) = rx.blocking_recv() {
            match job {
                Job::Forward {
                    conn,
                    session,
                    seq,
                    pos0,
                    act,
                    reply,
                } => {
                    let r = self.forward(conn, session, seq, pos0, &act);
                    let _ = reply.send(r);
                }
                Job::SeqRm {
                    conn,
                    session,
                    p0,
                    p1,
                    reply,
                } => {
                    let r = match self.slots.get(&(conn, session)) {
                        None => err(ErrorCode::UnknownSession, format!("unknown session {session}")),
                        Some(&slot) => {
                            if self.ctx.seq_rm(slot, p0, p1) {
                                Response::Ok
                            } else {
                                err(ErrorCode::BadRequest, format!("seq_rm({p0}, {p1}) not possible"))
                            }
                        }
                    };
                    let _ = reply.send(r);
                }
                Job::Close { conn, session, reply } => {
                    self.free_session(conn, session);
                    let _ = reply.send(Response::Ok);
                }
                Job::CloseConn { conn } => {
                    let keys: Vec<(u64, u64)> = self.slots.keys().filter(|k| k.0 == conn).copied().collect();
                    for (c, s) in keys {
                        self.free_session(c, s);
                    }
                }
            }
        }
        tracing::debug!("engine thread stopping");
    }

    fn free_session(&mut self, conn: u64, session: u64) {
        if let Some(slot) = self.slots.remove(&(conn, session)) {
            self.ctx.seq_rm(slot, -1, -1);
            self.free.push(slot);
            tracing::debug!(conn, session, slot, "session closed");
        }
    }

    fn forward(&mut self, conn: u64, session: u64, seq: u64, pos0: i32, act: &Activations) -> Response {
        let n_embd = self.ctx.n_embd();
        let n_tokens = act.n_tokens as usize;
        if act.n_embd as usize != n_embd {
            return err(
                ErrorCode::BadRequest,
                format!("activation width {} != model n_embd {n_embd}", act.n_embd),
            );
        }
        if n_tokens == 0 {
            return err(ErrorCode::BadRequest, "empty activation batch");
        }
        if n_tokens > self.n_batch {
            return err(
                ErrorCode::BadRequest,
                format!("{n_tokens} tokens exceed the stage batch limit {}", self.n_batch),
            );
        }
        if pos0 < 0 {
            return err(ErrorCode::BadRequest, format!("negative position {pos0}"));
        }
        let end = pos0 as i64 + n_tokens as i64;
        if end > self.n_ctx_seq {
            return err(
                ErrorCode::ContextFull,
                format!(
                    "positions {pos0}..{end} exceed the per-session context {}",
                    self.n_ctx_seq
                ),
            );
        }
        let hidden = match act.to_f32() {
            Ok(h) => h,
            Err(e) => return err(ErrorCode::BadRequest, e.to_string()),
        };

        // Contiguity rule: the next position this session expects is `cur`.
        // `pos0 < cur` rolls the session back to `pos0` first (prefix caching, retries);
        // `pos0 > cur` would leave a hole in the KV cache and is refused.
        let key = (conn, session);
        let existing = self.slots.get(&key).copied();
        let cur = existing.map(|s| self.ctx.seq_pos_max(s) + 1).unwrap_or(0);
        if pos0 > cur {
            return err(
                ErrorCode::BadRequest,
                format!("position gap: session {session} is at {cur}, forward starts at {pos0}"),
            );
        }
        let slot = match existing {
            Some(s) => s,
            None => match self.free.pop() {
                Some(s) => {
                    self.slots.insert(key, s);
                    tracing::debug!(conn, session, slot = s, "session opened");
                    s
                }
                None => {
                    return err(
                        ErrorCode::SessionLimit,
                        format!("all {} session slots are in use", self.ctx.n_seq_max()),
                    )
                }
            },
        };
        if pos0 < cur && !self.ctx.seq_rm(slot, pos0, -1) {
            return err(ErrorCode::Internal, format!("could not roll session {session} back to {pos0}"));
        }

        let t = Instant::now();
        let r = self.ctx.decode_embd(&hidden, pos0, slot, OutputMode::All);
        let compute_ms = t.elapsed().as_secs_f32() * 1000.0;
        match r {
            Ok(_) => {}
            Err(dfin_llama::Error::ContextFull) => return err(ErrorCode::ContextFull, "KV cache full"),
            Err(e) => return err(ErrorCode::Decode, e.to_string()),
        }
        let out = self.ctx.embeddings();
        if out.len() != n_tokens * n_embd {
            return err(
                ErrorCode::Internal,
                format!("decode produced {} values, expected {}", out.len(), n_tokens * n_embd),
            );
        }
        let act = Activations::from_f32(out, n_tokens, n_embd, self.act_dtype);
        tracing::debug!(
            session,
            seq,
            pos0,
            n_tokens,
            bytes = act.byte_len(),
            compute_ms = format!("{compute_ms:.2}"),
            "forward"
        );
        Response::ForwardOk {
            session,
            seq,
            act,
            compute_ms,
        }
    }
}
