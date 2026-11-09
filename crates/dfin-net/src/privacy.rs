//! Strict network policy: no target DNS and no direct fallback.
//!
//! The local Tor daemon is part of the trusted computing base. SOCKS success cannot
//! prove that a proxy actually used Tor; operators must run a trusted Tor instance
//! with IsolateSOCKSAuth enabled. dfin never sends a target to the OS resolver.

use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tor_socksproto::{
    Buffer, Handshake, NextStep, SocksAddr, SocksAuth, SocksClientHandshake, SocksCmd,
    SocksRequest, SocksStatus, SocksVersion,
};

/// Validate a v3 onion destination without DNS or any other I/O.
/// Tor validates the service checksum and authenticates its onion identity.
pub fn onion_target(addr: &str) -> anyhow::Result<(&str, u16)> {
    let (host, port) = addr
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("strict anonymity requires a v3 onion address and port"))?;
    let name = host
        .strip_suffix(".onion")
        .ok_or_else(|| anyhow::anyhow!("strict anonymity refuses non-onion destinations"))?;
    anyhow::ensure!(
        name.len() == 56
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b)),
        "strict anonymity requires a canonical v3 onion hostname"
    );
    let port: u16 = port.parse()?;
    anyhow::ensure!(port != 0, "destination port must be nonzero");
    Ok((host, port))
}

pub fn local_proxy(proxy: SocketAddr) -> anyhow::Result<()> {
    anyhow::ensure!(
        proxy.ip().is_loopback() && proxy.port() != 0,
        "Tor SOCKS proxy must be a numeric loopback address with a nonzero port"
    );
    Ok(())
}

/// Connect through Tor's SOCKS interface with a fresh stream-isolation credential.
/// The caller bounds the complete connect + SOCKS + TLS operation with a timeout.
pub(crate) async fn connect(proxy: SocketAddr, addr: &str) -> anyhow::Result<TcpStream> {
    let (host, port) = onion_target(addr)?;
    local_proxy(proxy)?;
    let mut socket = TcpStream::connect(proxy).await?;
    socket.set_nodelay(true)?;
    let isolation = hex::encode(rand::random::<[u8; 32]>());
    let request = SocksRequest::new(
        SocksVersion::V5,
        SocksCmd::CONNECT,
        SocksAddr::Hostname(host.to_string().try_into()?),
        port,
        SocksAuth::Username(isolation.into_bytes(), b"dfin".to_vec()),
    )?;
    let mut handshake = SocksClientHandshake::new(request);
    let mut buffer = Buffer::new_precise();
    let mut greeting = Vec::with_capacity(2);
    loop {
        match handshake.step(&mut buffer)? {
            NextStep::Send(bytes) => socket.write_all(&bytes).await?,
            NextStep::Recv(mut recv) => {
                let n = socket.read(recv.buf()).await?;
                anyhow::ensure!(n != 0, "Tor proxy closed during SOCKS handshake");
                // The library permits no-auth proxies. Our policy requires SOCKS
                // auth so Tor receives the isolation credential; reject downgrades.
                if greeting.len() < 2 {
                    let take = n.min(2 - greeting.len());
                    greeting.extend_from_slice(&recv.buf()[..take]);
                    if greeting.len() == 2 {
                        anyhow::ensure!(
                            greeting == [5, 2],
                            "Tor proxy refused authenticated stream isolation"
                        );
                    }
                }
                recv.note_received(n)?;
            }
            NextStep::Finished(done) => {
                let reply = done.into_output()?;
                anyhow::ensure!(
                    reply.status() == SocksStatus::SUCCEEDED,
                    "Tor proxy could not reach onion service ({})",
                    reply.status()
                );
                return Ok(socket);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_never_allow_ip_dns_subdomains_or_zero_ports() {
        let good = format!("{}.onion:7001", "a".repeat(56));
        assert!(onion_target(&good).is_ok());
        for bad in [
            "127.0.0.1:7001",
            "[::1]:7001",
            "example.org:443",
            "localhost:9050",
            "oldshort.onion:80",
            "foo.onion.evil:80",
            "user@host.onion:80",
            "https://example.onion:80",
        ] {
            assert!(onion_target(bad).is_err(), "{bad}");
        }
        for bad in [
            good.replace(":7001", ":0"),
            good.to_uppercase(),
            format!("sub.{good}"),
            good.replace(":7001", ":65536"),
        ] {
            assert!(onion_target(&bad).is_err(), "{bad}");
        }
        assert!(local_proxy("0.0.0.0:9050".parse().unwrap()).is_err());
        assert!(local_proxy("192.0.2.1:9050".parse().unwrap()).is_err());
    }

    #[tokio::test]
    async fn proxy_auth_downgrade_is_rejected() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut hello = [0; 4];
            stream.read_exact(&mut hello).await.unwrap();
            assert_eq!(hello, [5, 2, 2, 0]);
            stream.write_all(&[5, 0]).await.unwrap();
            let mut rest = [0; 1];
            assert_eq!(stream.read(&mut rest).await.unwrap(), 0);
        });
        let err = connect(proxy, &format!("{}.onion:80", "a".repeat(56)))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("isolation"));
        server.await.unwrap();
    }
}
