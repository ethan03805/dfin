//! A QUIC endpoint over a relayed stream, for cloaked nodes.
//!
//! A cloaked node keeps one bidirectional stream open to a relay. The relay binds a
//! public UDP port for it and forwards every datagram that arrives there down the
//! stream, tagged with a synthetic address standing in for the sender; datagrams the
//! node writes up the stream, tagged with that synthetic address, go back out of the
//! port to the real sender. The node's QUIC endpoint runs over [`TunnelSocket`], so
//! dialers complete an ordinary end-to-end QUIC handshake with the node's key and the
//! relay sees only ciphertext.
//!
//! Frame on the stream: `u16` length, 16-byte IPv6-mapped address, `u16` port, payload.

use quinn::udp::{RecvMeta, Transmit};
use quinn::{AsyncUdpSocket, UdpPoller};
use std::io::{self, IoSliceMut};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex};

/// One datagram plus the peer address attached to it.
#[derive(Debug)]
pub struct Packet {
    pub addr: SocketAddr,
    pub data: Vec<u8>,
}

const ADDR_LEN: usize = 18;

pub async fn write_packet<W: AsyncWrite + Unpin>(w: &mut W, p: &Packet) -> io::Result<()> {
    if p.data.len() + ADDR_LEN > u16::MAX as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "packet too large"));
    }
    let mut frame = Vec::with_capacity(2 + ADDR_LEN + p.data.len());
    frame.extend_from_slice(&((ADDR_LEN + p.data.len()) as u16).to_be_bytes());
    let ip6 = match p.addr.ip() {
        IpAddr::V4(v4) => v4.to_ipv6_mapped(),
        IpAddr::V6(v6) => v6,
    };
    frame.extend_from_slice(&ip6.octets());
    frame.extend_from_slice(&p.addr.port().to_be_bytes());
    frame.extend_from_slice(&p.data);
    w.write_all(&frame).await
}

pub async fn read_packet<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Packet> {
    let mut len = [0u8; 2];
    r.read_exact(&mut len).await?;
    let len = u16::from_be_bytes(len) as usize;
    if len < ADDR_LEN {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short packet frame"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).await?;
    let mut ip = [0u8; 16];
    ip.copy_from_slice(&body[..16]);
    let ip6 = Ipv6Addr::from(ip);
    let ip = match ip6.to_ipv4_mapped() {
        Some(v4) => IpAddr::V4(v4),
        None => IpAddr::V6(ip6),
    };
    let port = u16::from_be_bytes([body[16], body[17]]);
    Ok(Packet {
        addr: SocketAddr::new(ip, port),
        data: body[ADDR_LEN..].to_vec(),
    })
}

/// The node side of a tunnel: what quinn sends goes to `outbound`, what arrives on
/// `inbound` is what quinn receives.
#[derive(Debug)]
pub struct TunnelSocket {
    outbound: mpsc::UnboundedSender<Packet>,
    inbound: Mutex<mpsc::UnboundedReceiver<Packet>>,
    local: SocketAddr,
}

/// Build a tunnel socket and the two channel ends a pump drives.
pub fn tunnel_socket() -> (
    Arc<TunnelSocket>,
    mpsc::UnboundedSender<Packet>,
    mpsc::UnboundedReceiver<Packet>,
) {
    let (in_tx, in_rx) = mpsc::unbounded_channel();
    let (out_tx, out_rx) = mpsc::unbounded_channel();
    let sock = Arc::new(TunnelSocket {
        outbound: out_tx,
        inbound: Mutex::new(in_rx),
        local: SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
    });
    (sock, in_tx, out_rx)
}

#[derive(Debug)]
struct AlwaysWritable;

impl UdpPoller for AlwaysWritable {
    fn poll_writable(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

impl AsyncUdpSocket for TunnelSocket {
    fn create_io_poller(self: Arc<Self>) -> Pin<Box<dyn UdpPoller>> {
        Box::pin(AlwaysWritable)
    }

    fn try_send(&self, transmit: &Transmit) -> io::Result<()> {
        let p = Packet {
            addr: transmit.destination,
            data: transmit.contents.to_vec(),
        };
        self.outbound
            .send(p)
            .map_err(|_| io::Error::new(io::ErrorKind::NotConnected, "tunnel closed"))
    }

    fn poll_recv(
        &self,
        cx: &mut Context,
        bufs: &mut [IoSliceMut<'_>],
        meta: &mut [RecvMeta],
    ) -> Poll<io::Result<usize>> {
        let mut rx = match self.inbound.try_lock() {
            Ok(g) => g,
            Err(_) => {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        };
        match rx.poll_recv(cx) {
            Poll::Ready(Some(p)) => {
                let Some(buf) = bufs.first_mut() else {
                    return Poll::Ready(Ok(0));
                };
                let n = p.data.len().min(buf.len());
                buf[..n].copy_from_slice(&p.data[..n]);
                meta[0] = RecvMeta {
                    addr: p.addr,
                    len: n,
                    stride: n,
                    ecn: None,
                    dst_ip: None,
                };
                Poll::Ready(Ok(1))
            }
            Poll::Ready(None) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "tunnel closed",
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }

    fn may_fragment(&self) -> bool {
        false
    }
}

/// Drive a tunnel: datagrams read from `recv` feed `inbound`; datagrams drained from
/// `outbound` are written to `send`. Returns when either direction fails.
pub async fn pump<S, R>(
    mut send: S,
    mut recv: R,
    inbound: mpsc::UnboundedSender<Packet>,
    mut outbound: mpsc::UnboundedReceiver<Packet>,
) -> io::Error
where
    S: AsyncWrite + Unpin + Send + 'static,
    R: AsyncRead + Unpin + Send + 'static,
{
    let down = async move {
        loop {
            match read_packet(&mut recv).await {
                Ok(p) => {
                    if inbound.send(p).is_err() {
                        return io::Error::new(io::ErrorKind::NotConnected, "endpoint gone");
                    }
                }
                Err(e) => return e,
            }
        }
    };
    let up = async move {
        while let Some(p) = outbound.recv().await {
            if let Err(e) = write_packet(&mut send, &p).await {
                return e;
            }
        }
        io::Error::new(io::ErrorKind::NotConnected, "endpoint gone")
    };
    tokio::select! {
        e = down => e,
        e = up => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn packet_frame_roundtrip() {
        let p = Packet {
            addr: "100.64.1.2:4567".parse().unwrap(),
            data: vec![1, 2, 3, 4],
        };
        let mut buf = Vec::new();
        write_packet(&mut buf, &p).await.unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let back = read_packet(&mut cur).await.unwrap();
        assert_eq!(back.addr, p.addr);
        assert_eq!(back.data, p.data);
        let p6 = Packet {
            addr: "[2001:db8::1]:9".parse().unwrap(),
            data: vec![],
        };
        let mut buf = Vec::new();
        write_packet(&mut buf, &p6).await.unwrap();
        let back = read_packet(&mut std::io::Cursor::new(buf)).await.unwrap();
        assert_eq!(back.addr, p6.addr);
    }
}
