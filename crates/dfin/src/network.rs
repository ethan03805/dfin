//! One network policy shared by every command. Default is strict Tor; direct is explicit.
use clap::{Args, ValueEnum};
use dfin_net::{Endpoint, NodeKey};
use std::net::SocketAddr;

/// Shared by every strict-mode HTTP listener: loopback binding alone does not
/// stop hostile webpages or DNS rebinding. Native callers can omit Origin.
pub(crate) async fn local_request_only(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !local_headers(req.headers()) {
        return (
            axum::http::StatusCode::FORBIDDEN,
            "strict mode API requires a local same-origin request",
        )
            .into_response();
    }
    next.run(req).await
}

fn local_headers(headers: &axum::http::HeaderMap) -> bool {
    let Some(host) = headers.get("host").and_then(|h| h.to_str().ok()) else {
        return false;
    };
    let Ok(addr) = host.parse::<SocketAddr>() else {
        return false;
    };
    if !addr.ip().is_loopback() {
        return false;
    }
    match headers.get("origin") {
        None => true,
        Some(origin) => origin.to_str().is_ok_and(|o| o == format!("http://{host}")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    Tor,
    Direct,
}

#[derive(Clone, Copy, Debug, Args)]
pub struct Network {
    /// Tor (default): onion destinations only, no direct fallback. Direct: LAN/development, no anonymity.
    #[arg(long, value_enum, default_value = "tor", global = true)]
    pub network: Mode,
    /// Trusted local Tor SOCKS endpoint, configured with IsolateSOCKSAuth.
    #[arg(long, default_value = "127.0.0.1:9050", global = true)]
    pub tor_proxy: SocketAddr,
}

impl Network {
    pub fn private(&self) -> bool {
        self.network == Mode::Tor
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.private() {
            dfin_net::privacy::local_proxy(self.tor_proxy)?;
        }
        Ok(())
    }
    pub fn require_direct(&self, command: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.private(),
            "{command} is a direct-network development tool; explicitly use --network direct"
        );
        Ok(())
    }
    pub fn target(&self, address: &str) -> anyhow::Result<()> {
        if self.private() {
            dfin_net::privacy::onion_target(address)?;
        }
        Ok(())
    }
    pub fn listener(&self, address: &str) -> anyhow::Result<()> {
        if self.private() {
            let parsed: SocketAddr = address.parse().map_err(|_| {
                anyhow::anyhow!("strict anonymity listeners require a numeric loopback address")
            })?;
            anyhow::ensure!(
                parsed.ip().is_loopback(),
                "strict anonymity listeners must use loopback (127.0.0.1 or ::1)"
            );
        }
        Ok(())
    }
    pub fn client(&self, key: NodeKey) -> anyhow::Result<Endpoint> {
        match self.network {
            Mode::Tor => Endpoint::tor_client(key, self.tor_proxy),
            Mode::Direct => Endpoint::direct_client(key),
        }
    }
    pub async fn server(&self, key: NodeKey, address: &str) -> anyhow::Result<Endpoint> {
        self.listener(address)?;
        if self.private() {
            Endpoint::tor_server(key, address.parse()?, self.tor_proxy).await
        } else {
            let listen = tokio::net::lookup_host(address)
                .await?
                .next()
                .ok_or_else(|| anyhow::anyhow!("listener address did not resolve"))?;
            Endpoint::direct_server(key, listen)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn local_api_rejects_cross_origin_and_rebinding_hosts() {
        let mut h = axum::http::HeaderMap::new();
        assert!(!local_headers(&h));
        h.insert("host", "127.0.0.1:8080".parse().unwrap());
        assert!(local_headers(&h));
        h.insert("origin", "http://127.0.0.1:8080".parse().unwrap());
        assert!(local_headers(&h));
        for origin in ["null", "https://evil.example", "http://127.0.0.1:8081"] {
            h.insert("origin", origin.parse().unwrap());
            assert!(!local_headers(&h));
        }
        h.remove("origin");
        for host in [
            "evil.example:8080",
            "localhost:8080",
            "0.0.0.0:8080",
            "127.0.0.1.evil:8080",
        ] {
            h.insert("host", host.parse().unwrap());
            assert!(!local_headers(&h));
        }
        h.insert("host", "[::1]:8080".parse().unwrap());
        assert!(local_headers(&h));
    }
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        network: Network,
    }
    #[test]
    fn defaults_are_strict_and_reject_public_listeners() {
        let n = Cli::parse_from(["dfin"]).network;
        assert!(n.private());
        for addr in ["0.0.0.0:1", "[::]:1", "localhost:1", "example.org:1"] {
            assert!(n.listener(addr).is_err());
        }
        assert!(n.listener("127.0.0.1:0").is_ok());
        assert!(n.target("127.0.0.1:7001").is_err());
        assert!(n.require_direct("pod").is_err());
        assert!(!Cli::parse_from(["dfin", "--network", "direct"])
            .network
            .private());
    }
}
