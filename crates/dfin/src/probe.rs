//! `dfin probe`: a vantage point. On a hub's request it pings a node and answers with
//! the round trip and the region label it was started with. A hub with probes in
//! several places publishes their measurements, and a client can require that every
//! probe in a region reach a node within a bound, which puts the node's network
//! interface inside a radius that light in fibre allows: about 100 km per millisecond
//! of round trip. Protocol: `docs/phase1-protocol.md` section 14.

use crate::config::parse_node_id;
use anyhow::Context as _;
use clap::Args;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey};
use dfin_proto::{read_frame_max, ErrorCode, Request, Response};
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Address to listen on (UDP, QUIC).
    #[arg(long, default_value = "0.0.0.0:7300")]
    pub listen: String,
    /// Node key file (64 hex characters). Created if missing.
    #[arg(long, default_value = "dfin-probe.key")]
    pub identity: PathBuf,
    /// The region label this probe stands in, e.g. `us-east`. Published with each result.
    #[arg(long)]
    pub region: String,
    /// Node ids allowed to ask (hubs). Repeatable. With none, anyone may ask, which turns
    /// the probe into a scanner for whoever finds it; a warning says so.
    #[arg(long = "allow", value_parser = parse_node_id)]
    pub allow: Vec<NodeId>,
    /// Probes answered per minute at most.
    #[arg(long, default_value_t = 600)]
    pub max_per_minute: u32,
    /// Added to every reported round trip, in ms. A test aid to stand in for distance.
    #[arg(long, default_value_t = 0.0, hide = true)]
    pub add_delay_ms: f32,
}

struct Probe {
    endpoint: Endpoint,
    region: String,
    allow: Vec<NodeId>,
    max_per_minute: u32,
    add_delay_ms: f32,
    /// Timestamps of recent answers, for the rate limit.
    recent: Mutex<Vec<Instant>>,
}

pub async fn run(args: ProbeArgs) -> anyhow::Result<()> {
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
        "probe ready on {} node {} region {:?}",
        endpoint.local_addr()?,
        endpoint.node_id(),
        args.region
    );
    if args.allow.is_empty() {
        warn!("no --allow given: any node may ask this probe to ping any address");
    }
    let probe = Arc::new(Probe {
        endpoint: endpoint.clone(),
        region: args.region.clone(),
        allow: args.allow.clone(),
        max_per_minute: args.max_per_minute.max(1),
        add_delay_ms: args.add_delay_ms,
        recent: Mutex::new(Vec::new()),
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
            let probe = probe.clone();
            tokio::spawn(async move { probe.serve(conn).await });
        }
    };
    tokio::select! {
        _ = accept_loop => Ok(()),
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: probe exiting");
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

impl Probe {
    async fn serve(self: &Arc<Self>, conn: Connection) {
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let probe = self.clone();
            let peer = conn.peer_id();
            tokio::spawn(async move {
                let req: Request = match tokio::time::timeout(
                    CONTROL_TIMEOUT,
                    read_frame_max(&mut recv, MAX_CONTROL_FRAME),
                )
                .await
                {
                    Ok(Ok(r)) => r,
                    _ => return,
                };
                let resp = match req {
                    Request::Ping => Response::Pong,
                    Request::Probe { addr, node_id } => probe.handle(peer, &addr, NodeId(node_id)).await,
                    _ => error(ErrorCode::BadRequest, "this node is a probe"),
                };
                if let Err(e) = reply(&mut send, &resp).await {
                    debug!("reply: {e}");
                }
            });
        }
    }

    fn allowed_now(&self) -> bool {
        let now = Instant::now();
        let mut recent = self.recent.lock();
        recent.retain(|t| now.duration_since(*t) < Duration::from_secs(60));
        if recent.len() as u32 >= self.max_per_minute {
            return false;
        }
        recent.push(now);
        true
    }

    async fn handle(&self, peer: NodeId, addr: &str, id: NodeId) -> Response {
        if !self.allow.is_empty() && !self.allow.contains(&peer) {
            return error(ErrorCode::BadRequest, "this probe does not answer you");
        }
        if addr.len() > 128 {
            return error(ErrorCode::BadRequest, "address too long");
        }
        if !self.allowed_now() {
            return error(ErrorCode::RateLimited, "probe is busy; try later");
        }
        match self.measure(addr, id).await {
            Some(ms) => {
                debug!(%addr, node = %id.short(), rtt_ms = format!("{ms:.2}"), "probed");
                Response::Probed {
                    rtt_ms: ms + self.add_delay_ms,
                    region: self.region.clone(),
                }
            }
            None => error(ErrorCode::Unreachable, format!("{addr} did not answer")),
        }
    }

    /// Best of three pings to `addr`, pinned to `id`.
    async fn measure(&self, addr: &str, id: NodeId) -> Option<f32> {
        let fut = async {
            let conn = self.endpoint.connect(addr, id).await.ok()?;
            let mut best = f32::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                match conn
                    .call::<_, Response>(&Request::Ping, MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
                    .await
                {
                    Ok(Response::Pong) => best = best.min(t.elapsed().as_secs_f32() * 1e3),
                    _ => return None,
                }
            }
            conn.close("probe");
            Some(best)
        };
        tokio::time::timeout(Duration::from_secs(8), fut)
            .await
            .ok()
            .flatten()
    }
}
