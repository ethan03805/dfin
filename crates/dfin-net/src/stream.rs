//! TLS 1.3 + yamux over Tor SOCKS. Node keys authenticate both ends of every connection.

use crate::transport::{NetError, RecvStream, SendStream, CONTROL_TIMEOUT};
use crate::{privacy, tls, NodeId, NodeKey};
use futures::future::poll_fn;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::Poll,
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpListener,
    sync::{mpsc, oneshot, Mutex, OwnedSemaphorePermit, Semaphore},
};
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tokio_util::{
    compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt},
    sync::CancellationToken,
};

const ALPN: &[u8] = b"dfin/1-tor-yamux";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
type Pair = (SendStream, RecvStream);
type Open = oneshot::Sender<yamux::Stream>;

pub(crate) struct Endpoint {
    key: NodeKey,
    cert: CertificateDer<'static>,
    pkey: PrivateKeyDer<'static>,
    proxy: SocketAddr,
    local: SocketAddr,
    incoming: Mutex<mpsc::Receiver<Result<Connection, NetError>>>,
    cancel: CancellationToken,
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl Endpoint {
    pub fn client(key: NodeKey, proxy: SocketAddr) -> anyhow::Result<Self> {
        privacy::local_proxy(proxy)?;
        let (cert, pkey) = tls::self_signed(&key)?;
        let (_, incoming) = mpsc::channel(32);
        Ok(Self {
            key,
            cert,
            pkey,
            proxy,
            local: "127.0.0.1:0".parse()?,
            incoming: Mutex::new(incoming),
            cancel: CancellationToken::new(),
        })
    }
    pub async fn server(
        key: NodeKey,
        listen: SocketAddr,
        proxy: SocketAddr,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            listen.ip().is_loopback(),
            "Tor service listeners must bind numeric loopback addresses"
        );
        let mut endpoint = Self::client(key, proxy)?;
        let provider = tls::provider();
        let mut cfg = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(Arc::new(tls::AnyClient { provider }))
            .with_single_cert(vec![endpoint.cert.clone()], endpoint.pkey.clone_key())?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        let acceptor = TlsAcceptor::from(Arc::new(cfg));
        let listener = TcpListener::bind(listen).await?;
        endpoint.local = listener.local_addr()?;
        let (tx, rx) = mpsc::channel(32);
        endpoint.incoming = Mutex::new(rx);
        let cancel = endpoint.cancel.clone();
        // Bound TLS work and established connections. Hold a permit until the last
        // connection handle is dropped; a slow handshake cannot block all accepts.
        let permits = Arc::new(Semaphore::new(32));
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    _ = cancel.cancelled() => break,
                    r = listener.accept() => r,
                };
                let (socket, _) = match accepted {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let Ok(permit) = permits.clone().try_acquire_owned() else {
                    continue;
                };
                let acceptor = acceptor.clone();
                let tx = tx.clone();
                let cancel = cancel.clone();
                tokio::spawn(async move {
                    let handshake = async {
                        socket.set_nodelay(true)?;
                        let io = acceptor.accept(socket).await?;
                        let session = &io.get_ref().1;
                        anyhow::ensure!(
                            session.alpn_protocol() == Some(ALPN),
                            "unexpected Tor transport ALPN"
                        );
                        let cert = session
                            .peer_certificates()
                            .and_then(|cs| cs.first())
                            .ok_or_else(|| anyhow::anyhow!("missing client certificate"))?;
                        let peer = tls::node_id_from_cert(cert)?;
                        Ok::<_, anyhow::Error>(Connection::start(
                            io,
                            peer,
                            yamux::Mode::Server,
                            cancel.child_token(),
                            Some(permit),
                        ))
                    };
                    let result = tokio::select! {
                        _ = cancel.cancelled() => return,
                        r = tokio::time::timeout(CONTROL_TIMEOUT, handshake) => match r {
                            Ok(r) => r.map_err(other), Err(_) => Err(NetError::Timeout(CONTROL_TIMEOUT)),
                        },
                    };
                    // Full queue drops the connection and its driver, never grows memory.
                    let _ = tx.try_send(result);
                });
            }
        });
        Ok(endpoint)
    }
    pub fn proxy(&self) -> SocketAddr {
        self.proxy
    }
    pub fn key(&self) -> &NodeKey {
        &self.key
    }
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }
    pub fn close(&self) {
        self.cancel.cancel();
    }
    pub async fn accept(&self) -> Option<Result<Connection, NetError>> {
        tokio::select! { _ = self.cancel.cancelled() => None, r = async { self.incoming.lock().await.recv().await } => r }
    }
    pub async fn connect(&self, addr: &str, expect: NodeId) -> Result<Connection, NetError> {
        // Validate before constructing a socket or doing any resolution.
        privacy::onion_target(addr).map_err(other)?;
        let provider = tls::provider();
        let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(other)?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(tls::PinnedServer { expect, provider }))
            .with_client_auth_cert(vec![self.cert.clone()], self.pkey.clone_key())
            .map_err(other)?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        // No session resumption across identities or requests.
        cfg.resumption = rustls::client::Resumption::disabled();
        let connector = TlsConnector::from(Arc::new(cfg));
        let dial = async {
            let socket = privacy::connect(self.proxy, addr).await?;
            let io = connector
                .connect(ServerName::try_from(tls::DUMMY_NAME)?, socket)
                .await?;
            anyhow::ensure!(
                io.get_ref().1.alpn_protocol() == Some(ALPN),
                "unexpected Tor transport ALPN"
            );
            Ok::<_, anyhow::Error>(Connection::start(
                io,
                expect,
                yamux::Mode::Client,
                self.cancel.child_token(),
                None,
            ))
        };
        tokio::select! {
            _ = self.cancel.cancelled() => Err(closed()),
            r = tokio::time::timeout(CONNECT_TIMEOUT, dial) => match r {
                Ok(r) => r.map_err(other), Err(_) => Err(NetError::Timeout(CONNECT_TIMEOUT)),
            },
        }
    }
}

fn other(e: impl std::fmt::Display) -> NetError {
    NetError::Other(e.to_string())
}
fn closed() -> NetError {
    NetError::Closed("Tor stream transport closed".into())
}

struct Handle {
    _permit: Option<OwnedSemaphorePermit>,
    open: mpsc::Sender<Open>,
    bidi: Mutex<mpsc::Receiver<Pair>>,
    uni: Mutex<mpsc::Receiver<RecvStream>>,
    cancel: CancellationToken,
    peer: NodeId,
    id: usize,
}
impl Drop for Handle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Clone)]
pub(crate) struct Connection(Arc<Handle>);
impl Connection {
    fn start<T>(
        io: T,
        peer: NodeId,
        mode: yamux::Mode,
        cancel: CancellationToken,
        permit: Option<OwnedSemaphorePermit>,
    ) -> Self
    where
        T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        let mut config = yamux::Config::default();
        config
            .set_max_num_streams(64)
            .set_max_connection_receive_window(Some(64 << 20));
        let mut mux = yamux::Connection::new(io.compat(), config, mode);
        let (open_tx, mut open_rx) = mpsc::channel::<Open>(32);
        let (bidi_tx, bidi_rx) = mpsc::channel(32);
        let (uni_tx, uni_rx) = mpsc::channel(32);
        let conn = Self(Arc::new(Handle {
            _permit: permit,
            open: open_tx,
            bidi: Mutex::new(bidi_rx),
            uni: Mutex::new(uni_rx),
            cancel: cancel.clone(),
            peer,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }));
        tokio::spawn(async move {
            // Wake consumers and terminate classifiers on every exit, including panic.
            let _guard = cancel.clone().drop_guard();
            let permits = Arc::new(Semaphore::new(32));
            let mut pending: Option<Open> = None;
            enum Event {
                Incoming(yamux::Stream),
                Outgoing(yamux::Stream),
                End,
            }
            loop {
                let event = tokio::select! {
                    _ = cancel.cancelled() => break,
                    event = poll_fn(|cx| {
                        // Always drive incoming frames: outbound flow control depends on it.
                        let incoming = mux.poll_next_inbound(cx);
                        if pending.is_none() {
                            match open_rx.poll_recv(cx) {
                                Poll::Ready(Some(p)) => pending = Some(p),
                                Poll::Ready(None) => return Poll::Ready(Event::End),
                                Poll::Pending => {},
                            }
                        }
                        if let Poll::Ready(r) = incoming {
                            return Poll::Ready(match r { Some(Ok(s)) => Event::Incoming(s), _ => Event::End });
                        }
                        if pending.is_some() {
                            if let Poll::Ready(r) = mux.poll_new_outbound(cx) {
                                return Poll::Ready(match r { Ok(s) => Event::Outgoing(s), Err(_) => Event::End });
                            }
                        }
                        Poll::Pending
                    }) => event,
                };
                match event {
                    Event::End => break,
                    Event::Outgoing(s) => {
                        if let Some(tx) = pending.take() {
                            let _ = tx.send(s);
                        }
                    }
                    Event::Incoming(s) => {
                        let Ok(permit) = permits.clone().try_acquire_owned() else {
                            drop(s);
                            continue;
                        };
                        let bidi = bidi_tx.clone();
                        let uni = uni_tx.clone();
                        let cancel = cancel.clone();
                        tokio::spawn(async move {
                            let _permit = permit;
                            let classify = async {
                                let mut io = s.compat();
                                let kind = io.read_u8().await?;
                                let (recv, send) = tokio::io::split(io);
                                match kind {
                                    0 => {
                                        let _ = bidi.try_send((
                                            Box::new(send) as SendStream,
                                            Box::new(recv) as RecvStream,
                                        ));
                                    }
                                    1 => {
                                        let mut send = send;
                                        send.shutdown().await?;
                                        let _ = uni.try_send(Box::new(recv) as RecvStream);
                                    }
                                    _ => cancel.cancel(),
                                }
                                Ok::<_, std::io::Error>(())
                            };
                            tokio::select! {
                                _ = cancel.cancelled() => {},
                                _ = tokio::time::timeout(CONTROL_TIMEOUT, classify) => {},
                            }
                        });
                    }
                }
            }
        });
        conn
    }
    pub fn peer_id(&self) -> NodeId {
        self.0.peer
    }
    pub fn stable_id(&self) -> usize {
        self.0.id
    }
    pub fn is_closed(&self) -> bool {
        self.0.cancel.is_cancelled()
    }
    pub fn close(&self) {
        self.0.cancel.cancel();
    }
    pub async fn closed(&self) -> NetError {
        self.0.cancel.cancelled().await;
        closed()
    }
    async fn open(&self, kind: u8) -> Result<Pair, NetError> {
        let op = async {
            let (tx, rx) = oneshot::channel();
            self.0.open.send(tx).await.map_err(|_| closed())?;
            let mut io = rx.await.map_err(|_| closed())?.compat();
            io.write_u8(kind).await?;
            io.flush().await?;
            let (recv, send) = tokio::io::split(io);
            Ok((Box::new(send) as SendStream, Box::new(recv) as RecvStream))
        };
        tokio::select! { _ = self.0.cancel.cancelled() => Err(closed()), r = op => r }
    }
    pub async fn open_bi(&self) -> Result<Pair, NetError> {
        self.open(0).await
    }
    pub async fn open_uni(&self) -> Result<SendStream, NetError> {
        Ok(self.open(1).await?.0)
    }
    pub async fn accept_bi(&self) -> Result<Pair, NetError> {
        tokio::select! { _ = self.0.cancel.cancelled() => Err(closed()), r = async { self.0.bidi.lock().await.recv().await } => r.ok_or_else(closed) }
    }
    pub async fn accept_uni(&self) -> Result<RecvStream, NetError> {
        tokio::select! { _ = self.0.cancel.cancelled() => Err(closed()), r = async { self.0.uni.lock().await.recv().await } => r.ok_or_else(closed) }
    }
}
