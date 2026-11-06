//! `dfin stage`: serve one contiguous middle layer range `[a, b)` of a GGUF over TCP.
//!
//! Protocol (see `dfin-proto`): the first frame on a connection must be
//! `Request::Hello`; after a successful `HelloOk` the client sends requests and the
//! stage answers each one, in order. Requests may be pipelined: a reader task
//! queues work for the engine thread and a writer task emits replies in request
//! order, so receiving the next activation overlaps with compute.

mod engine;

use crate::config::LayerRange;
use anyhow::{bail, Context as _};
use clap::{Args, ValueEnum};
use dfin_llama::{Context, ContextParams, FlashAttn, KvType, Model, ModelParams};
use dfin_proto::{
    read_frame, write_frame, ActDtype, ErrorCode, Request, Response, PROTOCOL_VERSION,
};
use engine::{Engine, Job};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FlashAttnArg {
    Auto,
    On,
    Off,
}

impl From<FlashAttnArg> for FlashAttn {
    fn from(v: FlashAttnArg) -> Self {
        match v {
            FlashAttnArg::Auto => FlashAttn::Auto,
            FlashAttnArg::On => FlashAttn::On,
            FlashAttnArg::Off => FlashAttn::Off,
        }
    }
}

#[derive(Args, Debug)]
pub struct StageArgs {
    /// GGUF model file (the full file; only this stage's layers are loaded).
    #[arg(long)]
    pub model: PathBuf,
    /// Half-open layer range `a:b` this stage serves. Must not include layer 0's
    /// embedding or the LM head (those run on the client).
    #[arg(long)]
    pub layers: LayerRange,
    /// Address to listen on.
    #[arg(long, default_value = "0.0.0.0:7001")]
    pub listen: String,
    /// Total KV capacity in tokens, shared by all session slots.
    #[arg(long, default_value_t = 8192)]
    pub n_ctx: u32,
    /// Concurrent session slots; each gets `n_ctx / n_seq_max` tokens.
    #[arg(long, default_value_t = 8)]
    pub n_seq_max: u32,
    /// Layers to offload to the GPU (`-1` = all of this stage's layers).
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    pub n_gpu_layers: i32,
    /// Threads for single-token decode.
    #[arg(long)]
    pub threads: Option<i32>,
    /// Threads for batch (prefill) decode.
    #[arg(long)]
    pub threads_batch: Option<i32>,
    /// Element type of the activations this stage sends back.
    #[arg(long, default_value = "f16")]
    pub act_dtype: ActDtype,
    /// KV cache element type (K and V).
    #[arg(long, default_value = "f16")]
    pub kv_type: KvType,
    #[arg(long, value_enum, default_value_t = FlashAttnArg::Auto)]
    pub flash_attn: FlashAttnArg,
    /// Regex of tensor names to keep in system RAM (e.g. `exps`). Repeatable.
    #[arg(long = "cpu-tensors")]
    pub cpu_tensors: Vec<String>,
    /// Read the weights into memory at load instead of memory-mapping the file. Slower
    /// start, but no first-request page-fault storm; recommended for a node that runs a
    /// single piece (llama.cpp's own advice when experts live on the CPU).
    #[arg(long)]
    pub no_mmap: bool,
    /// Lock the mapped weights in RAM at load so the first request is not slowed by page
    /// faults. Needs free memory for the whole piece.
    #[arg(long)]
    pub mlock: bool,
    /// Max tokens per forward request.
    #[arg(long, default_value_t = 2048)]
    pub n_batch: u32,
    /// Physical micro-batch size.
    #[arg(long, default_value_t = 512)]
    pub n_ubatch: u32,
}

/// What every connection needs to know about this stage.
struct StageInfo {
    fingerprint: String,
    layer_begin: i32,
    layer_end: i32,
    n_embd: u32,
    n_ctx_seq: u32,
    n_seq_max: u32,
    backend: String,
}

/// Summarise the devices this stage computes on, e.g. `"Vulkan0 (AMD Radeon RX 7800 XT), CPU"`.
pub fn backend_summary(n_gpu_layers: i32) -> String {
    let mut parts = Vec::new();
    for d in dfin_llama::devices() {
        if d.is_gpu {
            if n_gpu_layers != 0 {
                parts.push(format!("{} ({})", d.name, d.description));
            }
        } else {
            parts.push(d.name.clone());
        }
    }
    if parts.is_empty() {
        "CPU".into()
    } else {
        parts.join(", ")
    }
}

pub fn log_devices() {
    for d in dfin_llama::devices() {
        info!(
            "device {}: {} ({}, {} / {} MiB free)",
            d.name,
            d.description,
            if d.is_gpu { "gpu" } else { "cpu/accel" },
            d.memory_free >> 20,
            d.memory_total >> 20
        );
    }
}

pub async fn run(args: StageArgs) -> anyhow::Result<()> {
    dfin_llama::init();
    if args.layers.begin == 0 || args.layers.end == -1 {
        bail!(
            "stage range {} is not a middle piece: stages never hold the embedding or the LM head; the client runs those",
            args.layers
        );
    }
    if args.n_seq_max == 0 {
        bail!("--n-seq-max must be at least 1");
    }

    let model_path = args.model.clone();
    let fingerprint = {
        let p = model_path.clone();
        tokio::task::spawn_blocking(move || dfin_proto::model_fingerprint(&p))
            .await?
            .with_context(|| format!("fingerprinting {}", model_path.display()))?
    };

    let mparams = ModelParams {
        n_gpu_layers: args.n_gpu_layers,
        layer_begin: args.layers.begin,
        layer_end: args.layers.end,
        cpu_tensor_patterns: args.cpu_tensors.clone(),
        use_mmap: !args.no_mmap,
        use_mlock: args.mlock,
        ..ModelParams::default()
    };
    log_devices();
    info!("loading layers {} of {}", args.layers, model_path.display());
    let model = {
        let p = model_path.clone();
        tokio::task::spawn_blocking(move || Model::load(&p, &mparams))
            .await?
            .with_context(|| format!("loading {}", model_path.display()))?
    };
    let model = Arc::new(model);
    if model.has_head() || model.has_tail() {
        bail!(
            "resolved range {:?} of {} layers includes the embedding or the LM head: stages never hold the embedding or the LM head; the client runs those",
            model.layer_range(),
            model.n_layer()
        );
    }
    let (layer_begin, layer_end) = model.layer_range();
    info!(
        "model: {} | {:.1} MiB resident | layers {}:{} of {} | n_embd {}",
        model.desc(),
        model.size_bytes() as f64 / (1024.0 * 1024.0),
        layer_begin,
        layer_end,
        model.n_layer(),
        model.n_embd()
    );

    let mut cparams = ContextParams {
        n_ctx: args.n_ctx,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_seq_max: args.n_seq_max,
        embeddings: true,
        kv_unified: false,
        flash_attn: args.flash_attn.into(),
        type_k: args.kv_type,
        type_v: args.kv_type,
        ..ContextParams::default()
    };
    if let Some(t) = args.threads {
        cparams.n_threads = t;
        cparams.n_threads_batch = t;
    }
    if let Some(t) = args.threads_batch {
        cparams.n_threads_batch = t;
    }
    let ctx = Context::new(model.clone(), &cparams).context("creating llama context")?;
    let n_seq_max = ctx.n_seq_max().max(1);
    let n_ctx_seq = ctx.n_ctx() / n_seq_max;
    info!(
        "context: n_ctx {} total, {} slots, {} tokens per session, threads {}/{}",
        ctx.n_ctx(),
        n_seq_max,
        n_ctx_seq,
        cparams.n_threads,
        cparams.n_threads_batch
    );

    let info = Arc::new(StageInfo {
        fingerprint,
        layer_begin,
        layer_end,
        n_embd: model.n_embd() as u32,
        n_ctx_seq,
        n_seq_max,
        backend: backend_summary(args.n_gpu_layers),
    });

    let (jobs, _engine_thread) =
        Engine::new(ctx, args.act_dtype, args.n_batch as usize).spawn(256)?;

    let listener = TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    let addr = listener.local_addr()?;
    info!(
        "stage ready: layers {}:{} on {} ({})",
        layer_begin, layer_end, addr, info.backend
    );

    let next_conn = AtomicU64::new(1);
    let accept_loop = async {
        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(x) => x,
                Err(e) => {
                    warn!("accept failed: {e}");
                    continue;
                }
            };
            let conn = next_conn.fetch_add(1, Ordering::Relaxed);
            let info = info.clone();
            let jobs = jobs.clone();
            tokio::spawn(async move {
                debug!(conn, %peer, "connection opened");
                if let Err(e) = handle_conn(stream, conn, info, jobs).await {
                    warn!(conn, %peer, "connection error: {e:#}");
                }
                debug!(conn, %peer, "connection closed");
            });
        }
    };

    tokio::select! {
        _ = accept_loop => Ok(()),
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: stage exiting");
            Ok(())
        }
    }
}

fn error(code: ErrorCode, message: impl Into<String>) -> Response {
    Response::Error {
        code,
        message: message.into(),
    }
}

/// Check a `Hello` against this stage. `Ok` carries `HelloOk`, `Err` the refusal.
fn check_hello(req: &Request, info: &StageInfo) -> Result<Response, Response> {
    let Request::Hello {
        version,
        model_fingerprint,
        layer_begin,
        layer_end,
    } = req
    else {
        return Err(error(ErrorCode::BadRequest, "first frame must be Hello"));
    };
    if *version != PROTOCOL_VERSION {
        return Err(error(
            ErrorCode::VersionMismatch,
            format!("protocol version {version}, stage speaks {PROTOCOL_VERSION}"),
        ));
    }
    if *model_fingerprint != info.fingerprint {
        return Err(error(
            ErrorCode::ModelMismatch,
            format!(
                "model fingerprint {model_fingerprint} does not match this stage's {}",
                info.fingerprint
            ),
        ));
    }
    if (*layer_begin, *layer_end) != (info.layer_begin, info.layer_end) {
        return Err(error(
            ErrorCode::LayerRangeMismatch,
            format!(
                "requested layers {layer_begin}:{layer_end}, this stage serves {}:{}",
                info.layer_begin, info.layer_end
            ),
        ));
    }
    Ok(Response::HelloOk {
        layer_begin: info.layer_begin,
        layer_end: info.layer_end,
        n_embd: info.n_embd,
        n_ctx: info.n_ctx_seq,
        n_seq_max: info.n_seq_max,
        backend: info.backend.clone(),
    })
}

/// A reply slot in request order: either already known or pending on the engine.
enum Pending {
    Ready(Response),
    Wait(oneshot::Receiver<Response>),
}

async fn handle_conn(
    stream: TcpStream,
    conn: u64,
    info: Arc<StageInfo>,
    jobs: mpsc::Sender<Job>,
) -> anyhow::Result<()> {
    stream.set_nodelay(true)?;
    let (mut rd, mut wr) = stream.into_split();

    // Handshake.
    let first: Request = match read_frame(&mut rd).await {
        Ok(r) => r,
        Err(e) => {
            // Port probes (e.g. `dfin pod` waiting for readiness) connect and leave.
            debug!(conn, "no hello: {e}");
            return Ok(());
        }
    };
    match check_hello(&first, &info) {
        Ok(ok) => write_frame(&mut wr, &ok).await?,
        Err(refusal) => {
            if let Response::Error { code, message } = &refusal {
                warn!(conn, ?code, "refusing connection: {message}");
            }
            write_frame(&mut wr, &refusal).await?;
            return Ok(());
        }
    }

    let (ptx, mut prx) = mpsc::channel::<Pending>(64);
    let writer = tokio::spawn(async move {
        while let Some(p) = prx.recv().await {
            let resp = match p {
                Pending::Ready(r) => r,
                Pending::Wait(rx) => rx
                    .await
                    .unwrap_or_else(|_| error(ErrorCode::Internal, "engine stopped")),
            };
            write_frame(&mut wr, &resp).await?;
        }
        Ok::<_, std::io::Error>(())
    });

    let read_result: anyhow::Result<()> = async {
        loop {
            let req: Request = match read_frame(&mut rd).await {
                Ok(r) => r,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                    let _ = ptx
                        .send(Pending::Ready(error(
                            ErrorCode::BadRequest,
                            format!("bad frame: {e}"),
                        )))
                        .await;
                    return Err(anyhow::Error::new(e).context("bad frame"));
                }
                Err(e) => return Err(e.into()),
            };
            let pending = match req {
                Request::Hello { .. } => {
                    Pending::Ready(error(ErrorCode::BadRequest, "duplicate Hello"))
                }
                Request::Ping => Pending::Ready(Response::Pong),
                Request::Forward {
                    session,
                    seq,
                    pos0,
                    act,
                } => {
                    let (reply, rx) = oneshot::channel();
                    jobs.send(Job::Forward {
                        conn,
                        session,
                        seq,
                        pos0,
                        act,
                        reply,
                    })
                    .await
                    .map_err(|_| anyhow::anyhow!("engine stopped"))?;
                    Pending::Wait(rx)
                }
                Request::SeqRm { session, p0, p1 } => {
                    let (reply, rx) = oneshot::channel();
                    jobs.send(Job::SeqRm {
                        conn,
                        session,
                        p0,
                        p1,
                        reply,
                    })
                    .await
                    .map_err(|_| anyhow::anyhow!("engine stopped"))?;
                    Pending::Wait(rx)
                }
                Request::SessionClose { session } => {
                    let (reply, rx) = oneshot::channel();
                    jobs.send(Job::Close {
                        conn,
                        session,
                        reply,
                    })
                    .await
                    .map_err(|_| anyhow::anyhow!("engine stopped"))?;
                    Pending::Wait(rx)
                }
            };
            if ptx.send(pending).await.is_err() {
                // Writer died (peer gone); stop reading.
                return Ok(());
            }
        }
    }
    .await;

    // Free every session of this connection (queued after its in-flight work).
    let _ = jobs.send(Job::CloseConn { conn }).await;
    drop(ptx);
    let write_result = writer.await?;
    read_result?;
    if let Err(e) = write_result {
        if !matches!(
            e.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
        ) {
            return Err(e.into());
        }
    }
    Ok(())
}
