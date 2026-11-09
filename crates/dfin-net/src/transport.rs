//! Authenticated transports. Tor is a separate TCP/TLS transport, never a QUIC fallback.

use crate::{quic, stream, NodeId, NodeKey};
use dfin_proto::{read_frame_max, write_frame};
use serde::{de::DeserializeOwned, Serialize};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

pub use quic::{NetError, ALPN, CONTROL_TIMEOUT, IDLE_TIMEOUT, MAX_CONTROL_FRAME};
pub type SendStream = Box<dyn AsyncWrite + Send + Unpin>;
pub type RecvStream = Box<dyn AsyncRead + Send + Unpin>;

#[derive(Clone)]
pub struct Endpoint(EndpointKind);
#[derive(Clone)]
enum EndpointKind {
    Direct(quic::Endpoint),
    Tor(Arc<stream::Endpoint>),
}

impl Endpoint {
    /// Explicit direct transport for LANs and development. Does not provide anonymity.
    pub fn direct_client(key: NodeKey) -> anyhow::Result<Self> {
        Ok(Self(EndpointKind::Direct(quic::Endpoint::client(key)?)))
    }
    /// Explicit direct listener. Does not provide anonymity.
    pub fn direct_server(key: NodeKey, listen: SocketAddr) -> anyhow::Result<Self> {
        Ok(Self(EndpointKind::Direct(quic::Endpoint::server(
            key, listen,
        )?)))
    }
    pub fn tor_client(key: NodeKey, proxy: SocketAddr) -> anyhow::Result<Self> {
        Ok(Self(EndpointKind::Tor(Arc::new(stream::Endpoint::client(
            key, proxy,
        )?))))
    }
    pub async fn tor_server(
        key: NodeKey,
        listen: SocketAddr,
        proxy: SocketAddr,
    ) -> anyhow::Result<Self> {
        Ok(Self(EndpointKind::Tor(Arc::new(
            stream::Endpoint::server(key, listen, proxy).await?,
        ))))
    }
    pub fn over_tunnel(
        key: NodeKey,
        socket: Arc<crate::tunnel::TunnelSocket>,
    ) -> anyhow::Result<Self> {
        Ok(Self(EndpointKind::Direct(quic::Endpoint::over_tunnel(
            key, socket,
        )?)))
    }
    pub fn is_private(&self) -> bool {
        matches!(self.0, EndpointKind::Tor(_))
    }
    /// Separate credentials and node identity for another privacy context (e.g. discovery).
    pub fn isolated_client(&self) -> anyhow::Result<Self> {
        match &self.0 {
            EndpointKind::Direct(_) => Self::direct_client(NodeKey::generate()),
            EndpointKind::Tor(e) => Self::tor_client(NodeKey::generate(), e.proxy()),
        }
    }
    pub fn node_id(&self) -> NodeId {
        self.key().id()
    }
    pub fn key(&self) -> &NodeKey {
        match &self.0 {
            EndpointKind::Direct(e) => e.key(),
            EndpointKind::Tor(e) => e.key(),
        }
    }
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        match &self.0 {
            EndpointKind::Direct(e) => e.local_addr(),
            EndpointKind::Tor(e) => Ok(e.local_addr()),
        }
    }
    pub async fn connect(&self, addr: &str, expect: NodeId) -> Result<Connection, NetError> {
        // Protocol fields can construct NodeId directly, bypassing text parsing.
        // Refuse impossible pins before DNS, SOCKS, TLS, or a UDP packet.
        expect
            .verifying_key()
            .map_err(|_| NetError::Other("invalid destination node identity".into()))?;
        match &self.0 {
            EndpointKind::Direct(e) => e
                .connect(addr, expect)
                .await
                .map(|c| Connection(ConnectionKind::Direct(c))),
            EndpointKind::Tor(e) => e
                .connect(addr, expect)
                .await
                .map(|c| Connection(ConnectionKind::Tor(c))),
        }
    }
    pub async fn accept(&self) -> Option<Result<Connection, NetError>> {
        match &self.0 {
            EndpointKind::Direct(e) => e
                .accept()
                .await
                .map(|r| r.map(|c| Connection(ConnectionKind::Direct(c)))),
            EndpointKind::Tor(e) => e
                .accept()
                .await
                .map(|r| r.map(|c| Connection(ConnectionKind::Tor(c)))),
        }
    }
    pub fn close(&self) {
        match &self.0 {
            EndpointKind::Direct(e) => e.close(),
            EndpointKind::Tor(e) => e.close(),
        }
    }
}

#[derive(Clone)]
pub struct Connection(ConnectionKind);
#[derive(Clone)]
enum ConnectionKind {
    Direct(quic::Connection),
    Tor(stream::Connection),
}
impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("peer", &self.peer_id())
            .finish()
    }
}
impl Connection {
    pub fn peer_id(&self) -> NodeId {
        match &self.0 {
            ConnectionKind::Direct(c) => c.peer_id(),
            ConnectionKind::Tor(c) => c.peer_id(),
        }
    }
    /// Tor peers have no observable IP address.
    pub fn remote_address(&self) -> Option<SocketAddr> {
        match &self.0 {
            ConnectionKind::Direct(c) => Some(c.remote_address()),
            ConnectionKind::Tor(_) => None,
        }
    }
    pub fn stable_id(&self) -> usize {
        match &self.0 {
            ConnectionKind::Direct(c) => c.stable_id(),
            ConnectionKind::Tor(c) => c.stable_id(),
        }
    }
    pub fn is_closed(&self) -> bool {
        match &self.0 {
            ConnectionKind::Direct(c) => c.is_closed(),
            ConnectionKind::Tor(c) => c.is_closed(),
        }
    }
    pub async fn closed(&self) -> NetError {
        match &self.0 {
            ConnectionKind::Direct(c) => c.closed().await,
            ConnectionKind::Tor(c) => c.closed().await,
        }
    }
    pub fn close(&self, reason: &str) {
        match &self.0 {
            ConnectionKind::Direct(c) => c.close(reason),
            ConnectionKind::Tor(c) => c.close(),
        }
    }
    pub async fn open_bi(&self) -> Result<(SendStream, RecvStream), NetError> {
        match &self.0 {
            ConnectionKind::Direct(c) => {
                let (s, r) = c.open_bi().await?;
                Ok((Box::new(s), Box::new(r)))
            }
            ConnectionKind::Tor(c) => c.open_bi().await,
        }
    }
    pub async fn accept_bi(&self) -> Result<(SendStream, RecvStream), NetError> {
        match &self.0 {
            ConnectionKind::Direct(c) => {
                let (s, r) = c.accept_bi().await?;
                Ok((Box::new(s), Box::new(r)))
            }
            ConnectionKind::Tor(c) => c.accept_bi().await,
        }
    }
    pub async fn open_uni(&self) -> Result<SendStream, NetError> {
        match &self.0 {
            ConnectionKind::Direct(c) => Ok(Box::new(c.open_uni().await?)),
            ConnectionKind::Tor(c) => c.open_uni().await,
        }
    }
    pub async fn accept_uni(&self) -> Result<RecvStream, NetError> {
        match &self.0 {
            ConnectionKind::Direct(c) => Ok(Box::new(c.accept_uni().await?)),
            ConnectionKind::Tor(c) => c.accept_uni().await,
        }
    }
    pub async fn call<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        req: &Req,
        max_reply: u32,
        timeout: Duration,
    ) -> Result<Resp, NetError> {
        tokio::time::timeout(timeout, async {
            let (mut send, mut recv) = self.open_bi().await?;
            write_frame(&mut send, req).await?;
            send.shutdown().await?;
            Ok(read_frame_max(&mut recv, max_reply).await?)
        })
        .await
        .map_err(|_| NetError::Timeout(timeout))?
    }
    pub async fn send_uni<T: Serialize>(&self, msg: &T) -> Result<(), NetError> {
        let mut send = self.open_uni().await?;
        write_frame(&mut send, msg).await?;
        send.shutdown().await?;
        Ok(())
    }
}
pub async fn reply<T: Serialize>(send: &mut SendStream, msg: &T) -> Result<(), NetError> {
    write_frame(send, msg).await?;
    send.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum Msg {
        Hi(String),
        Echo(Vec<u8>),
    }

    async fn serve(ep: Endpoint) {
        while let Some(Ok(conn)) = ep.accept().await {
            tokio::spawn(async move {
                while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                    let m: Msg = read_frame_max(&mut recv, 8 << 20).await.unwrap();
                    let r = match m {
                        Msg::Hi(s) => Msg::Hi(format!("hello {s} from {}", conn.peer_id().short())),
                        Msg::Echo(v) => Msg::Echo(v),
                    };
                    reply(&mut send, &r).await.unwrap();
                }
            });
        }
    }

    #[tokio::test]
    async fn pinned_connection_roundtrip() {
        let server_key = NodeKey::generate();
        let server =
            Endpoint::direct_server(server_key.clone(), "127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(serve(server));

        let client_key = NodeKey::generate();
        let client = Endpoint::direct_client(client_key.clone()).unwrap();
        let conn = client
            .connect(&addr.to_string(), server_key.id())
            .await
            .unwrap();
        assert_eq!(conn.peer_id(), server_key.id());
        let r: Msg = conn
            .call(&Msg::Hi("there".into()), MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(
            r,
            Msg::Hi(format!("hello there from {}", client_key.id().short()))
        );
        let big = vec![7u8; 5 << 20];
        let r: Msg = conn
            .call(&Msg::Echo(big.clone()), 8 << 20, Duration::from_secs(30))
            .await
            .unwrap();
        assert_eq!(r, Msg::Echo(big));
    }

    #[tokio::test]
    async fn wrong_node_id_is_refused() {
        let server_key = NodeKey::generate();
        let server = Endpoint::direct_server(server_key, "127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(serve(server));
        let client = Endpoint::direct_client(NodeKey::generate()).unwrap();
        let err = client
            .connect(&addr.to_string(), NodeKey::generate().id())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
    }

    #[tokio::test]
    async fn weak_raw_destination_id_never_sends_a_direct_packet() {
        let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let client = Endpoint::direct_client(NodeKey::generate()).unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            client.connect(&socket.local_addr().unwrap().to_string(), NodeId([0; 32])),
        )
        .await;
        assert!(matches!(result, Ok(Err(NetError::Other(_)))));
        let mut packet = [0u8; 2048];
        assert!(
            tokio::time::timeout(Duration::from_millis(30), socket.recv_from(&mut packet))
                .await
                .is_err()
        );
    }
}
