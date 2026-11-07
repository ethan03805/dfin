//! `dfin relay`: a dumb stream pipe between a client and its first hop, and between
//! the last hop and the client, keyed by a token the client chose.
//!
//! The relay terminates transport encryption and can inspect application frames,
//! including model states in both directions. Sealed route records remain encrypted
//! to their node keys. Direct mode also exposes the client's IP; Tor hides it from
//! this relay. Sessions live in memory and are not persisted by this implementation.
//! Protocol: `docs/phase1-protocol.md`
//! section 6.
//!
//! It also hosts **cloaked nodes** (section 9): a node that opens no port of its own
//! sends `RelayListen`; the relay binds a public UDP port for it and forwards every
//! datagram down the stream, tagged with a synthetic address in 100.64.0.0/10 that
//! stands in for the real sender, so the node learns no addresses at all. Dialers
//! complete an ordinary end-to-end QUIC handshake with the node's key through that
//! port; the relay sees only ciphertext.

use anyhow::Context as _;
use clap::Args;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::tunnel::{read_packet, write_packet, Packet};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey, RecvStream, SendStream};
use dfin_proto::{read_frame_max, ErrorCode, Request, Response};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
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
    /// Host name or address dialers use for the ports allocated to cloaked nodes
    /// (default: the `--listen` address, which must then not be unspecified).
    #[arg(long)]
    pub public_host: Option<String>,
    /// UDP ports for cloaked nodes, `first-last` (default: any free port).
    #[arg(long, value_parser = parse_port_range)]
    pub port_range: Option<(u16, u16)>,
    /// Cloaked nodes hosted at once.
    #[arg(long, default_value_t = 256)]
    pub max_cloaked: usize,
}

fn parse_port_range(s: &str) -> Result<(u16, u16), String> {
    let (a, b) = s
        .split_once('-')
        .ok_or_else(|| format!("port range {s:?} must look like 7200-7299"))?;
    let a: u16 = a.trim().parse().map_err(|_| format!("bad port {a:?}"))?;
    let b: u16 = b.trim().parse().map_err(|_| format!("bad port {b:?}"))?;
    if a == 0 || b < a {
        return Err(format!("port range {s:?} is empty"));
    }
    Ok((a, b))
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
    listen_ip: IpAddr,
    public_host: Option<String>,
    port_range: Option<(u16, u16)>,
    /// Cloaked nodes and the public address allocated to each.
    cloaked: Mutex<HashMap<NodeId, String>>,
    max_cloaked: usize,
}

pub async fn run(args: RelayArgs, network: crate::network::Network) -> anyhow::Result<()> {
    network.listener(&args.listen)?;
    anyhow::ensure!(
        !network.private() || (args.public_host.is_none() && args.port_range.is_none()),
        "UDP relay cloaking options are unavailable in Tor mode"
    );
    let key = NodeKey::load_or_create(&args.identity)
        .with_context(|| format!("node key {}", args.identity.display()))?;
    let endpoint = network.server(key, &args.listen).await?;
    let listen = endpoint.local_addr()?;
    info!(
        "relay ready on {} node {}",
        endpoint.local_addr()?,
        endpoint.node_id()
    );
    let public_host = args
        .public_host
        .clone()
        .or_else(|| (!listen.ip().is_unspecified()).then(|| listen.ip().to_string()));
    if public_host.is_none() {
        warn!("no --public-host and --listen is unspecified: cloaked nodes will be refused");
    }
    let relay = Arc::new(Relay {
        endpoint: endpoint.clone(),
        sessions: Mutex::new(HashMap::new()),
        max_sessions: args.max_sessions.max(1),
        listen_ip: listen.ip(),
        public_host,
        port_range: args.port_range,
        cloaked: Mutex::new(HashMap::new()),
        max_cloaked: args.max_cloaked.max(1),
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
        let (mut send, recv, req) = tokio::time::timeout(CONTROL_TIMEOUT, async {
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
            Request::RelayListen => {
                if self.endpoint.is_private() {
                    reply(
                        &mut send,
                        &error(
                            ErrorCode::BadRequest,
                            "UDP cloaking is disabled in Tor mode",
                        ),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                }
                let node = conn.peer_id();
                let Some(host) = self.public_host.clone() else {
                    reply(
                        &mut send,
                        &error(
                            ErrorCode::BadRequest,
                            "this relay hosts no cloaked nodes (no public host)",
                        ),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                };
                let full = {
                    let cloaked = self.cloaked.lock();
                    !cloaked.contains_key(&node) && cloaked.len() >= self.max_cloaked
                };
                if full {
                    reply(
                        &mut send,
                        &error(ErrorCode::SessionLimit, "relay hosts no more cloaked nodes"),
                    )
                    .await?;
                    linger(&conn).await;
                    return Ok(());
                }
                let sock = match bind_in_range(self.listen_ip, self.port_range).await {
                    Ok(s) => s,
                    Err(e) => {
                        reply(
                            &mut send,
                            &error(ErrorCode::Internal, format!("no port free: {e}")),
                        )
                        .await?;
                        linger(&conn).await;
                        return Ok(());
                    }
                };
                let port = sock.local_addr()?.port();
                let public = format!("{host}:{port}");
                self.cloaked.lock().insert(node, public.clone());
                // Written without finishing: this stream now carries the tunnel.
                dfin_proto::write_frame(
                    &mut send,
                    &Response::RelayListening {
                        addr: public.clone(),
                    },
                )
                .await?;
                info!(node = %node.short(), peer = ?conn.remote_address(), public = %public, "cloaked node listening");
                let r = forward_udp(sock, send, recv).await;
                self.cloaked.lock().remove(&node);
                debug!(node = %node.short(), "cloaked node gone: {r:?}");
                Ok(())
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

async fn bind_in_range(ip: IpAddr, range: Option<(u16, u16)>) -> std::io::Result<UdpSocket> {
    match range {
        None => UdpSocket::bind(SocketAddr::new(ip, 0)).await,
        Some((a, b)) => {
            let mut last = std::io::Error::new(std::io::ErrorKind::AddrInUse, "range exhausted");
            for port in a..=b {
                match UdpSocket::bind(SocketAddr::new(ip, port)).await {
                    Ok(s) => return Ok(s),
                    Err(e) => last = e,
                }
            }
            Err(last)
        }
    }
}

/// Real addresses of dialers and the synthetic ones the cloaked node sees instead.
#[derive(Default)]
struct SynthMap {
    to_synth: HashMap<SocketAddr, SocketAddr>,
    to_real: HashMap<SocketAddr, SocketAddr>,
    next: u32,
}

impl SynthMap {
    const MAX: usize = 65_536;

    fn synth_for(&mut self, real: SocketAddr) -> Option<SocketAddr> {
        if let Some(s) = self.to_synth.get(&real) {
            return Some(*s);
        }
        if self.to_synth.len() >= Self::MAX {
            return None;
        }
        self.next += 1;
        let n = self.next;
        let synth = SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::new(
                100,
                64 + ((n >> 16) & 0x3f) as u8,
                ((n >> 8) & 0xff) as u8,
                (n & 0xff) as u8,
            ),
            4000 + (n % 60_000) as u16,
        ));
        self.to_synth.insert(real, synth);
        self.to_real.insert(synth, real);
        Some(synth)
    }

    fn real_for(&self, synth: SocketAddr) -> Option<SocketAddr> {
        self.to_real.get(&synth).copied()
    }
}

/// Shuttle datagrams between a public UDP socket and a cloaked node's tunnel stream
/// until either side fails.
async fn forward_udp(
    sock: UdpSocket,
    mut send: SendStream,
    mut recv: RecvStream,
) -> std::io::Result<()> {
    let sock = Arc::new(sock);
    let map = Arc::new(Mutex::new(SynthMap::default()));
    let up = {
        let sock = sock.clone();
        let map = map.clone();
        async move {
            let mut buf = vec![0u8; 65_535];
            loop {
                let (n, real) = sock.recv_from(&mut buf).await?;
                let synth = {
                    let mut m = map.lock();
                    m.synth_for(real)
                };
                let Some(synth) = synth else {
                    continue;
                };
                write_packet(
                    &mut send,
                    &Packet {
                        addr: synth,
                        data: buf[..n].to_vec(),
                    },
                )
                .await?;
            }
        }
    };
    let down = async move {
        loop {
            let p = read_packet(&mut recv).await?;
            let real = {
                let m = map.lock();
                m.real_for(p.addr)
            };
            if let Some(real) = real {
                sock.send_to(&p.data, real).await?;
            }
        }
    };
    tokio::select! {
        r = up => r,
        r = down => r,
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
