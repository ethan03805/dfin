//! `dfin relay`: a dumb stream pipe between a client and its first hop, and between
//! the last hop and the client, keyed by a token the client chose.
//!
//! The relay terminates transport encryption and can inspect application frames,
//! including model states in both directions. Sealed route records remain encrypted
//! to their node keys. Direct mode also exposes the client's IP; Tor hides it from
//! this relay. Sessions live in memory and are not persisted by this implementation.
//! Protocol: `docs/protocol.md` section 6.

use anyhow::Context as _;
use clap::Args;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey, RecvStream, SendStream};
use dfin_proto::{read_frame_max, ErrorCode, Request, Response};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// A client session that nobody attached to within this long is forgotten.
const UNATTACHED_TTL: Duration = Duration::from_secs(120);

#[derive(Args, Debug)]
pub struct RelayArgs {
    /// Local listener (TCP in Tor mode, UDP in direct mode).
    #[arg(long, default_value = "127.0.0.1:7000")]
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
    /// None while dialing; pending sessions still reserve their token and capacity.
    first_hop: Mutex<Option<Connection>>,
    attached: AtomicBool,
    opened: Instant,
}

/// Owns a reservation through dialing, piping, errors, and cancellation.
struct SessionLease<'a> {
    relay: &'a Relay,
    token: [u8; 16],
    session: Arc<Session>,
}

impl Drop for SessionLease<'_> {
    fn drop(&mut self) {
        self.session.client.close("relay session ended");
        if let Some(hop) = self.session.first_hop.lock().take() {
            hop.close("relay session ended");
        }
        let mut sessions = self.relay.sessions.lock();
        // A closed reservation may have been swept and its token reused. Its
        // late cleanup must never remove the newer owner's reservation.
        if sessions
            .get(&self.token)
            .is_some_and(|s| Arc::ptr_eq(s, &self.session))
        {
            sessions.remove(&self.token);
        }
    }
}

struct CloseOnDrop(Connection);
impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close("relay return path ended");
    }
}

struct Relay {
    endpoint: Endpoint,
    sessions: Mutex<HashMap<[u8; 16], Arc<Session>>>,
    max_sessions: usize,
}

pub async fn run(args: RelayArgs, network: crate::network::Network) -> anyhow::Result<()> {
    network.listener(&args.listen)?;
    let key = NodeKey::load_or_create(&args.identity)
        .with_context(|| format!("node key {}", args.identity.display()))?;
    let endpoint = network.server(key, &args.listen).await?;
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
        // Include waiting for the first stream: authenticating without sending
        // any stream must not reserve a listener slot indefinitely.
        // The request stream's read half stays open while the connection is served.
        let (mut send, _recv, req) = tokio::time::timeout(CONTROL_TIMEOUT, async {
            let (send, mut recv) = conn.accept_bi().await?;
            let req: Request = read_frame_max(&mut recv, MAX_CONTROL_FRAME).await?;
            Ok::<_, anyhow::Error>((send, recv, req))
        })
        .await
        .context("first frame")??;
        match req {
            Request::Ping => {
                reply(&mut send, &Response::Pong).await?;
                linger(&conn).await;
                Ok(())
            }
            Request::RelayOpen { token, first_hop } => {
                let lease = match self.reserve_session(&conn, token) {
                    Ok(lease) => lease,
                    Err(response) => {
                        reply(&mut send, &response).await?;
                        linger(&conn).await;
                        return Ok(());
                    }
                };
                let session = &lease.session;
                let dial = tokio::select! {
                    _ = conn.closed() => return Ok(()),
                    result = self.endpoint.connect(&first_hop.addr, NodeId(first_hop.node_id)) => result,
                };
                let hop = match dial {
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
                *session.first_hop.lock() = Some(hop.clone());
                reply(&mut send, &Response::Ack).await?;
                debug!(peer = ?conn.remote_address(), hop = %first_hop.addr, "client session opened");
                let unattached = async {
                    tokio::time::sleep_until((session.opened + UNATTACHED_TTL).into()).await;
                    if session.attached.load(Ordering::Acquire) {
                        std::future::pending::<()>().await;
                    }
                };
                tokio::select! {
                    r = self.pipe_client(&conn, &hop) => r,
                    _ = unattached => Ok(()),
                }
            }
            Request::RelayAttach { token } => {
                let Some(session) = self.sessions.lock().get(&token).cloned() else {
                    reply(
                        &mut send,
                        &error(ErrorCode::UnknownRoute, "unknown relay token"),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                };
                if session.client.is_closed() || session.first_hop.lock().is_none() {
                    reply(
                        &mut send,
                        &error(ErrorCode::UnknownRoute, "relay session is not ready"),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                }
                if session.attached.swap(true, Ordering::AcqRel) {
                    reply(
                        &mut send,
                        &error(
                            ErrorCode::BadRequest,
                            "relay session already has a return path",
                        ),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                }
                // A lost last hop invalidates this session; keeping its client
                // alive would leave it waiting for results that cannot arrive.
                let _close_client = CloseOnDrop(session.client.clone());
                reply(&mut send, &Response::Ack).await?;
                debug!(peer = ?conn.remote_address(), "last hop attached");
                self.pipe_returns(&conn, &session.client).await
            }
            _ => {
                reply(
                    &mut send,
                    &error(
                        ErrorCode::BadRequest,
                        "relay expects RelayOpen or RelayAttach",
                    ),
                )
                .await?;
                linger(&conn).await;
                Ok(())
            }
        }
    }

    fn reserve_session(
        &self,
        client: &Connection,
        token: [u8; 16],
    ) -> Result<SessionLease<'_>, Response> {
        let mut sessions = self.sessions.lock();
        sessions.retain(|_, s| !s.client.is_closed());
        if sessions.contains_key(&token) {
            return Err(error(
                ErrorCode::BadRequest,
                "relay token is already in use",
            ));
        }
        if sessions.len() >= self.max_sessions {
            return Err(error(ErrorCode::SessionLimit, "relay is full"));
        }
        let session = Arc::new(Session {
            client: client.clone(),
            first_hop: Mutex::new(None),
            attached: AtomicBool::new(false),
            opened: Instant::now(),
        });
        sessions.insert(token, session.clone());
        Ok(SessionLease {
            relay: self,
            token,
            session,
        })
    }

    /// Every request stream the client opens becomes a request stream to the first hop.
    async fn pipe_client(&self, client: &Connection, hop: &Connection) -> anyhow::Result<()> {
        let mut pipes = tokio::task::JoinSet::new();
        loop {
            let incoming = tokio::select! {
                _ = client.closed() => return Ok(()),
                _ = hop.closed() => return Ok(()),
                Some(_) = pipes.join_next(), if !pipes.is_empty() => continue,
                result = client.accept_bi(), if pipes.len() < 128 => result,
            };
            let (c_send, c_recv) = match incoming {
                Ok(x) => x,
                Err(e) => {
                    debug!("client connection closed: {e}");
                    return Ok(());
                }
            };
            let outgoing = tokio::select! {
                _ = client.closed() => return Ok(()),
                result = hop.open_bi() => result,
            };
            let (h_send, h_recv) = match outgoing {
                Ok(x) => x,
                Err(e) => {
                    warn!("first hop connection lost: {e}");
                    client.close("first hop lost");
                    return Ok(());
                }
            };
            pipes.spawn(pipe(c_recv, h_send));
            pipes.spawn(pipe(h_recv, c_send));
        }
    }

    /// Every one-way stream the last hop opens becomes a one-way stream to the client.
    async fn pipe_returns(&self, hop: &Connection, client: &Connection) -> anyhow::Result<()> {
        let mut pipes = tokio::task::JoinSet::new();
        loop {
            let incoming = tokio::select! {
                _ = client.closed() => return Ok(()),
                _ = hop.closed() => return Ok(()),
                Some(_) = pipes.join_next(), if !pipes.is_empty() => continue,
                result = hop.accept_uni(), if pipes.len() < 64 => result,
            };
            let h_recv = match incoming {
                Ok(s) => s,
                Err(e) => {
                    debug!("last hop connection closed: {e}");
                    return Ok(());
                }
            };
            let outgoing = tokio::select! {
                _ = hop.closed() => return Ok(()),
                result = client.open_uni() => result,
            };
            let c_send = match outgoing {
                Ok(s) => s,
                Err(e) => {
                    debug!("client gone while returning: {e}");
                    return Ok(());
                }
            };
            pipes.spawn(pipe(h_recv, c_send));
        }
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
    let _ = tokio::io::AsyncWriteExt::shutdown(&mut to).await;
}
