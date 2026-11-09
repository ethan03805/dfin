//! QUIC endpoints and connections with node-id authentication.
//!
//! One request is one bidirectional stream: the opener writes a frame, the acceptor
//! writes one frame back, both finish. Senders of unsolicited results open
//! unidirectional streams. Frames are `dfin-proto` frames.

use crate::identity::{NodeId, NodeKey};
use crate::tls;
use quinn::{RecvStream, SendStream, VarInt};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub const ALPN: &[u8] = b"dfin/1";

/// Idle connections are dropped after this; keep-alives go every fifth of it.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Time allowed to read a small control frame.
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
/// Largest control frame (anything that is not a Forward or a result).
pub const MAX_CONTROL_FRAME: u32 = 64 << 10;

#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("connect {addr}: {err}")]
    Connect { addr: String, err: String },
    #[error("address {0} did not resolve")]
    Resolve(String),
    #[error("connection closed: {0}")]
    Closed(String),
    #[error("stream: {0}")]
    Io(#[from] std::io::Error),
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    #[error("{0}")]
    Other(String),
}

impl From<quinn::ConnectionError> for NetError {
    fn from(e: quinn::ConnectionError) -> Self {
        NetError::Closed(e.to_string())
    }
}

impl From<quinn::WriteError> for NetError {
    fn from(e: quinn::WriteError) -> Self {
        NetError::Closed(e.to_string())
    }
}

impl From<quinn::ReadError> for NetError {
    fn from(e: quinn::ReadError) -> Self {
        NetError::Closed(e.to_string())
    }
}

impl From<quinn::ClosedStream> for NetError {
    fn from(e: quinn::ClosedStream) -> Self {
        NetError::Closed(e.to_string())
    }
}

fn transport_config() -> Arc<quinn::TransportConfig> {
    let mut t = quinn::TransportConfig::default();
    t.max_idle_timeout(Some(IDLE_TIMEOUT.try_into().expect("idle timeout fits")));
    t.keep_alive_interval(Some(IDLE_TIMEOUT / 5));
    // Prefill chunks are tens of megabytes; windows must not throttle them.
    t.stream_receive_window(VarInt::from_u32(64 << 20));
    t.receive_window(VarInt::from_u32(256 << 20));
    t.send_window(256 << 20);
    // A peer may hold this many requests open at once; each can carry one forward
    // frame, so this bounds what a connection can make the receiver buffer.
    t.max_concurrent_bidi_streams(VarInt::from_u32(32));
    t.max_concurrent_uni_streams(VarInt::from_u32(32));
    Arc::new(t)
}

/// A QUIC endpoint that can accept (if bound with [`Endpoint::server`]) and dial.
#[derive(Clone)]
pub struct Endpoint {
    inner: quinn::Endpoint,
    key: NodeKey,
    cert: CertificateDer<'static>,
    pkey: Arc<PrivateKeyDer<'static>>,
}

impl Endpoint {
    /// A dial-only endpoint on an ephemeral port.
    pub fn client(key: NodeKey) -> anyhow::Result<Endpoint> {
        let bind: SocketAddr = "[::]:0".parse().unwrap();
        let inner = match quinn::Endpoint::client(bind) {
            Ok(e) => e,
            // Hosts without IPv6: fall back to IPv4.
            Err(_) => quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())?,
        };
        Self::wrap(inner, key)
    }

    /// A listening endpoint that can also dial.
    pub fn server(key: NodeKey, listen: SocketAddr) -> anyhow::Result<Endpoint> {
        let (cert, pkey) = tls::self_signed(&key)?;
        let provider = tls::provider();
        let mut cfg = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(Arc::new(tls::AnyClient { provider }))
            .with_single_cert(vec![cert.clone()], pkey.clone_key())?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        let qsc = quinn::crypto::rustls::QuicServerConfig::try_from(cfg)?;
        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(qsc));
        server_config.transport_config(transport_config());
        let inner = quinn::Endpoint::server(server_config, listen)?;
        Ok(Endpoint {
            inner,
            key,
            cert,
            pkey: Arc::new(pkey),
        })
    }

    /// A listening endpoint whose datagrams travel through a relay tunnel instead of a
    /// socket of its own (see the `tunnel` module). It cannot usefully dial.
    pub fn over_tunnel(
        key: NodeKey,
        socket: Arc<crate::tunnel::TunnelSocket>,
    ) -> anyhow::Result<Endpoint> {
        let (cert, pkey) = tls::self_signed(&key)?;
        let provider = tls::provider();
        let mut cfg = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(Arc::new(tls::AnyClient { provider }))
            .with_single_cert(vec![cert.clone()], pkey.clone_key())?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        let qsc = quinn::crypto::rustls::QuicServerConfig::try_from(cfg)?;
        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(qsc));
        server_config.transport_config(transport_config());
        let inner = quinn::Endpoint::new_with_abstract_socket(
            quinn::EndpointConfig::default(),
            Some(server_config),
            socket,
            Arc::new(quinn::TokioRuntime),
        )?;
        Ok(Endpoint {
            inner,
            key,
            cert,
            pkey: Arc::new(pkey),
        })
    }

    fn wrap(inner: quinn::Endpoint, key: NodeKey) -> anyhow::Result<Endpoint> {
        let (cert, pkey) = tls::self_signed(&key)?;
        Ok(Endpoint {
            inner,
            key,
            cert,
            pkey: Arc::new(pkey),
        })
    }

    pub fn key(&self) -> &NodeKey {
        &self.key
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    fn client_config(&self, expect: NodeId) -> anyhow::Result<quinn::ClientConfig> {
        let provider = tls::provider();
        let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(tls::PinnedServer { expect, provider }))
            .with_client_auth_cert(vec![self.cert.clone()], self.pkey.clone_key())?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        let qcc = quinn::crypto::rustls::QuicClientConfig::try_from(cfg)?;
        let mut client_config = quinn::ClientConfig::new(Arc::new(qcc));
        client_config.transport_config(transport_config());
        Ok(client_config)
    }

    /// Dial `addr` (host:port) and require the peer to be `expect`.
    pub async fn connect(&self, addr: &str, expect: NodeId) -> Result<Connection, NetError> {
        let targets: Vec<SocketAddr> = tokio::net::lookup_host(addr)
            .await
            .map_err(|e| NetError::Connect {
                addr: addr.to_string(),
                err: e.to_string(),
            })?
            .collect();
        if targets.is_empty() {
            return Err(NetError::Resolve(addr.to_string()));
        }
        let cfg = self
            .client_config(expect)
            .map_err(|e| NetError::Other(e.to_string()))?;
        let mut last = String::from("no address");
        for target in targets {
            let target = map_for_socket(target, self.inner.local_addr().ok());
            let connecting = match self
                .inner
                .connect_with(cfg.clone(), target, tls::DUMMY_NAME)
            {
                Ok(c) => c,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            match tokio::time::timeout(CONTROL_TIMEOUT, connecting).await {
                Ok(Ok(conn)) => {
                    return Ok(Connection {
                        inner: conn,
                        peer: expect,
                    })
                }
                Ok(Err(e)) => last = e.to_string(),
                Err(_) => last = "handshake timed out".into(),
            }
        }
        Err(NetError::Connect {
            addr: addr.to_string(),
            err: last,
        })
    }

    /// Accept the next connection. `None` when the endpoint is closed.
    pub async fn accept(&self) -> Option<Result<Connection, NetError>> {
        let incoming = self.inner.accept().await?;
        Some(
            async move {
                let conn = incoming.await?;
                let peer = peer_id(&conn)?;
                Ok(Connection { inner: conn, peer })
            }
            .await,
        )
    }

    pub fn close(&self) {
        self.inner.close(VarInt::from_u32(0), b"shutdown");
    }
}

/// An IPv4 target dialled from a dual-stack `[::]` socket must be written as v4-mapped.
fn map_for_socket(target: SocketAddr, local: Option<SocketAddr>) -> SocketAddr {
    match (target, local) {
        (SocketAddr::V4(v4), Some(SocketAddr::V6(_))) => {
            SocketAddr::new(v4.ip().to_ipv6_mapped().into(), v4.port())
        }
        _ => target,
    }
}

fn peer_id(conn: &quinn::Connection) -> Result<NodeId, NetError> {
    let identity = conn
        .peer_identity()
        .ok_or_else(|| NetError::Other("peer presented no certificate".into()))?;
    let certs = identity
        .downcast::<Vec<CertificateDer<'static>>>()
        .map_err(|_| NetError::Other("unexpected peer identity type".into()))?;
    let cert = certs
        .first()
        .ok_or_else(|| NetError::Other("peer certificate chain is empty".into()))?;
    tls::node_id_from_cert(cert).map_err(|e| NetError::Other(e.to_string()))
}

/// An authenticated connection. Cheap to clone.
#[derive(Clone)]
pub struct Connection {
    inner: quinn::Connection,
    peer: NodeId,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Connection({} at {})",
            self.peer.short(),
            self.inner.remote_address()
        )
    }
}

impl Connection {
    pub fn peer_id(&self) -> NodeId {
        self.peer
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    /// A stable id for this connection on this endpoint.
    pub fn stable_id(&self) -> usize {
        self.inner.stable_id()
    }

    pub fn is_closed(&self) -> bool {
        self.inner.close_reason().is_some()
    }

    /// Resolves when the connection is gone.
    pub async fn closed(&self) -> NetError {
        self.inner.closed().await.into()
    }

    pub fn close(&self, reason: &str) {
        self.inner.close(VarInt::from_u32(0), reason.as_bytes());
    }

    /// Server side: wait for the next request stream.
    pub async fn accept_bi(&self) -> Result<(SendStream, RecvStream), NetError> {
        Ok(self.inner.accept_bi().await?)
    }

    /// Wait for the next unsolicited one-way stream.
    pub async fn accept_uni(&self) -> Result<RecvStream, NetError> {
        Ok(self.inner.accept_uni().await?)
    }

    /// Open a raw bidirectional stream (relays pipe these).
    pub async fn open_bi(&self) -> Result<(SendStream, RecvStream), NetError> {
        Ok(self.inner.open_bi().await?)
    }

    pub async fn open_uni(&self) -> Result<SendStream, NetError> {
        Ok(self.inner.open_uni().await?)
    }
}
