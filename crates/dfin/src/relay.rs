//! `dfin relay`: a dumb stream pipe between a client and its first hop, and between
//! the last hop and the client, keyed by a token the client chose.
//!
//! The relay sees the client's address, the first hop's address, and bytes. Route
//! records are sealed to node keys it does not hold, and the only plaintext it could
//! read, the `Hello` the client sends through it, names a model fingerprint and a
//! layer range. It keeps no record of sessions. Protocol: `docs/phase1-protocol.md`
//! section 6.

use anyhow::Context as _;
use clap::Args;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey, RecvStream, SendStream};
use dfin_proto::{read_frame_max, ErrorCode, Request, Response};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// A client session that nobody attached to within this long is forgotten.
const UNATTACHED_TTL: Duration = Duration::from_secs(120);

#[derive(Args, Debug)]
pub struct RelayArgs {
    /// Address to listen on (UDP, QUIC).
    #[arg(long, default_value = "0.0.0.0:7000")]
    pub listen: String,
    /// Node key file (64 hex characters). Created if missing.
    #[arg(long, default_value = "dfin-relay.key")]
    pub identity: PathBuf,
    /// Client sessions held open at once.
    #[arg(long, default_value_t = 1024)]
    pub max_sessions: usize,
}

struct Session {
    client: Connection,
    first_hop: Connection,
    opened: Instant,
}

struct Relay {
    endpoint: Endpoint,
    sessions: Mutex<HashMap<[u8; 16], Arc<Session>>>,
    max_sessions: usize,
}

pub async fn run(args: RelayArgs) -> anyhow::Result<()> {
    let key = NodeKey::load_or_create(&args.identity)
        .with_context(|| format!("node key {}", args.identity.display()))?;
    let listen: SocketAddr = tokio::net::lookup_host(&args.listen)
        .await
        .with_context(|| format!("resolving {}", args.listen))?
        .next()
        .with_context(|| format!("{} resolved to nothing", args.listen))?;
    let endpoint =
        Endpoint::server(key, listen).with_context(|| format!("listening on {listen}"))?;
    info!(
        "relay ready on {} node {}",
        endpoint.local_addr()?,
        endpoint.node_id()
    );
    let relay = Arc::new(Relay {
        endpoint: endpoint.clone(),
        sessions: Mutex::new(HashMap::new()),
        max_sessions: args.max_sessions.max(1),
    });

    let accept_loop = async {
        while let Some(incoming) = endpoint.accept().await {
            let conn = match incoming {
                Ok(c) => c,
                Err(e) => {
                    debug!("handshake failed: {e}");
                    continue;
                }
            };
            let relay = relay.clone();
            tokio::spawn(async move {
                if let Err(e) = relay.serve(conn).await {
                    debug!("connection: {e}");
                }
            });
        }
    };
    tokio::select! {
        _ = accept_loop => Ok(()),
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: relay exiting");
            endpoint.close();
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

impl Relay {
    /// The first stream says what the connection is for; the rest are piped.
    async fn serve(self: &Arc<Self>, conn: Connection) -> anyhow::Result<()> {
        let (mut send, mut recv) = conn.accept_bi().await?;
        let req: Request = tokio::time::timeout(
            CONTROL_TIMEOUT,
            read_frame_max(&mut recv, MAX_CONTROL_FRAME),
        )
        .await
        .context("first frame")??;
        match req {
            Request::Ping => {
                reply(&mut send, &Response::Pong).await?;
                linger(&conn).await;
                Ok(())
            }
            Request::RelayOpen { token, first_hop } => {
                self.expire();
                if self.sessions.lock().len() >= self.max_sessions {
                    reply(&mut send, &error(ErrorCode::SessionLimit, "relay is full")).await?;
                    linger(&conn).await;
                    return Ok(());
                }
                let hop = match self
                    .endpoint
                    .connect(&first_hop.addr, NodeId(first_hop.node_id))
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        reply(
                            &mut send,
                            &error(ErrorCode::Downstream, format!("first hop: {e}")),
                        )
                        .await?;
                        linger(&conn).await;
                        return Ok(());
                    }
                };
                let session = Arc::new(Session {
                    client: conn.clone(),
                    first_hop: hop.clone(),
                    opened: Instant::now(),
                });
                self.sessions.lock().insert(token, session);
                reply(&mut send, &Response::Ack).await?;
                debug!(peer = %conn.remote_address(), hop = %first_hop.addr, "client session opened");
                let r = self.pipe_client(&conn, &hop).await;
                self.sessions.lock().remove(&token);
                hop.close("client gone");
                r
            }
            Request::RelayAttach { token } => {
                let Some(session) = self.sessions.lock().get(&token).cloned() else {
                    reply(&mut send, &error(ErrorCode::UnknownRoute, "unknown relay token")).await?;
                    linger(&conn).await;
                    return Ok(());
                };
                reply(&mut send, &Response::Ack).await?;
                debug!(peer = %conn.remote_address(), "last hop attached");
                self.pipe_returns(&conn, &session.client).await
            }
            _ => {
                reply(
                    &mut send,
                    &error(ErrorCode::BadRequest, "relay expects RelayOpen or RelayAttach"),
                )
                .await?;
                linger(&conn).await;
                Ok(())
            }
        }
    }

    /// Every request stream the client opens becomes a request stream to the first hop.
    async fn pipe_client(&self, client: &Connection, hop: &Connection) -> anyhow::Result<()> {
        loop {
            let (c_send, c_recv) = match client.accept_bi().await {
                Ok(x) => x,
                Err(e) => {
                    debug!("client connection closed: {e}");
                    return Ok(());
                }
            };
            let (h_send, h_recv) = match hop.open_bi().await {
                Ok(x) => x,
                Err(e) => {
                    warn!("first hop connection lost: {e}");
                    client.close("first hop lost");
                    return Ok(());
                }
            };
            tokio::spawn(pipe(c_recv, h_send));
            tokio::spawn(pipe(h_recv, c_send));
        }
    }

    /// Every one-way stream the last hop opens becomes a one-way stream to the client.
    async fn pipe_returns(&self, hop: &Connection, client: &Connection) -> anyhow::Result<()> {
        loop {
            let h_recv = match hop.accept_uni().await {
                Ok(s) => s,
                Err(e) => {
                    debug!("last hop connection closed: {e}");
                    return Ok(());
                }
            };
            let c_send = match client.open_uni().await {
                Ok(s) => s,
                Err(e) => {
                    debug!("client gone while returning: {e}");
                    return Ok(());
                }
            };
            tokio::spawn(pipe(h_recv, c_send));
        }
    }

    fn expire(&self) {
        let now = Instant::now();
        self.sessions.lock().retain(|_, s| {
            !(s.client.is_closed() || now.duration_since(s.opened) > UNATTACHED_TTL && s.first_hop.is_closed())
        });
    }
}

/// Keep a connection alive until the peer closes it (or a short timeout), so a
/// reply written just before is delivered rather than discarded with the connection.
async fn linger(conn: &Connection) {
    let _ = tokio::time::timeout(CONTROL_TIMEOUT, conn.closed()).await;
}

/// Copy one stream into another until the source ends, then finish the sink.
async fn pipe(mut from: RecvStream, mut to: SendStream) {
    if let Err(e) = tokio::io::copy(&mut from, &mut to).await {
        debug!("pipe ended: {e}");
    }
    let _ = to.finish();
}
