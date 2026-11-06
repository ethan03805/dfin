//! `dfin hub`: the directory.
//!
//! Nodes announce a signed record of what they serve; the hub verifies the signature,
//! dials the announced address to check it answers, and lists the record until it
//! expires. Anyone can fetch the list, signed by the hub. The hub never assigns
//! routes: clients build their own from the list, so a hub cannot steer a user onto
//! chosen nodes. Protocol: `docs/phase1-protocol.md` section 10.

use anyhow::Context as _;
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use clap::Args;
use dfin_net::directory::{sign_directory, verify_record};
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey};
use dfin_proto::{
    read_frame_max, Directory, DirectoryEntry, ErrorCode, NodeRecord, Request, Response,
    SignedRecord, MAX_CLOCK_SKEW_SECS, MAX_RECORD_TTL_SECS, PROTOCOL_VERSION,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

#[derive(Args, Debug)]
pub struct HubArgs {
    /// Address to listen on (UDP, QUIC).
    #[arg(long, default_value = "0.0.0.0:7100")]
    pub listen: String,
    /// Node key file (64 hex characters). Created if missing. Clients pin its public key.
    #[arg(long, default_value = "dfin-hub.key")]
    pub identity: PathBuf,
    /// Also serve the directory as JSON over HTTP at this address (`GET /directory`).
    #[arg(long)]
    pub http: Option<String>,
    /// Records kept at most.
    #[arg(long, default_value_t = 10_000)]
    pub max_records: usize,
    /// Seconds allowed for the reachability probe of an announcing node.
    #[arg(long, default_value_t = 5)]
    pub probe_timeout: u64,
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct Entry {
    record: NodeRecord,
    signed: SignedRecord,
    hub_rtt_ms: f32,
    seen_at: u64,
}

pub struct Hub {
    endpoint: Endpoint,
    entries: Mutex<HashMap<NodeId, Entry>>,
    max_records: usize,
    probe_timeout: Duration,
}

pub async fn run(args: HubArgs) -> anyhow::Result<()> {
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
        "hub ready on {} node {}",
        endpoint.local_addr()?,
        endpoint.node_id()
    );
    let hub = Arc::new(Hub {
        endpoint: endpoint.clone(),
        entries: Mutex::new(HashMap::new()),
        max_records: args.max_records.max(1),
        probe_timeout: Duration::from_secs(args.probe_timeout.max(1)),
    });

    if let Some(http) = &args.http {
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
            .route("/directory", get(http_directory))
            .with_state(hub.clone());
        let listener = tokio::net::TcpListener::bind(http)
            .await
            .with_context(|| format!("binding {http}"))?;
        info!("directory as JSON at http://{}/directory", listener.local_addr()?);
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                warn!("http: {e}");
            }
        });
    }

    let accept_loop = async {
        while let Some(incoming) = endpoint.accept().await {
            let conn = match incoming {
                Ok(c) => c,
                Err(e) => {
                    debug!("handshake failed: {e}");
                    continue;
                }
            };
            let hub = hub.clone();
            tokio::spawn(async move { hub.serve(conn).await });
        }
    };
    tokio::select! {
        _ = accept_loop => Ok(()),
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: hub exiting");
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

async fn http_directory(State(hub): State<Arc<Hub>>) -> Json<Value> {
    hub.sweep();
    let now = unix_now();
    let mut rows: Vec<Value> = hub
        .entries
        .lock()
        .values()
        .map(|e| {
            let r = &e.record;
            json!({
                "node_id": NodeId(r.node_id).to_string(),
                "addrs": r.addrs,
                "model": r.model_name,
                "model_fingerprint": r.model_fingerprint,
                "layers": format!("{}:{}", r.layer_begin, r.layer_end),
                "n_layer": r.n_layer,
                "n_embd": r.n_embd,
                "n_ctx_seq": r.n_ctx_seq,
                "n_seq_max": r.n_seq_max,
                "backend": r.backend,
                "region": r.region,
                "attested": r.attested,
                "cloaked": r.cloaked,
                "hub_rtt_ms": e.hub_rtt_ms,
                "seen_secs_ago": now.saturating_sub(e.seen_at),
                "expires_in_secs": r.expires_at().saturating_sub(now),
            })
        })
        .collect();
    rows.sort_by(|a, b| a["layers"].as_str().cmp(&b["layers"].as_str()));
    Json(json!({ "hub": hub.endpoint.node_id().to_string(), "issued_at": now, "nodes": rows }))
}

impl Hub {
    async fn serve(self: &Arc<Self>, conn: Connection) {
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let hub = self.clone();
            tokio::spawn(async move {
                let req: Request = match tokio::time::timeout(
                    CONTROL_TIMEOUT,
                    read_frame_max(&mut recv, MAX_CONTROL_FRAME),
                )
                .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => {
                        debug!("request: {e}");
                        return;
                    }
                    Err(_) => return,
                };
                let resp = match req {
                    Request::Ping => Response::Pong,
                    Request::Announce { record } => hub.announce(record).await,
                    Request::DirectoryGet => hub.directory(),
                    _ => error(ErrorCode::BadRequest, "this node is a hub"),
                };
                if let Response::Error { code, message } = &resp {
                    debug!(?code, "{message}");
                }
                if let Err(e) = reply(&mut send, &resp).await {
                    debug!("reply: {e}");
                }
            });
        }
    }

    fn sweep(&self) {
        let now = unix_now();
        self.entries
            .lock()
            .retain(|_, e| e.record.expires_at() > now);
    }

    fn check(record: &NodeRecord, now: u64) -> Result<(), Response> {
        let bad = |m: &str| Err(error(ErrorCode::BadRequest, m));
        if record.protocol != PROTOCOL_VERSION {
            return Err(error(
                ErrorCode::VersionMismatch,
                format!("record speaks protocol {}, hub {}", record.protocol, PROTOCOL_VERSION),
            ));
        }
        if record.addrs.is_empty() || record.addrs.len() > 8 {
            return bad("a record names between one and eight addresses");
        }
        if record.addrs.iter().any(|a| a.is_empty() || a.len() > 128) {
            return bad("address too long");
        }
        if record.model_fingerprint.len() > 128
            || record.model_name.len() > 256
            || record.backend.len() > 256
            || record.region.len() > 64
        {
            return bad("text field too long");
        }
        if record.layer_begin <= 0
            || record.layer_end <= record.layer_begin
            || record.layer_end > record.n_layer
        {
            return bad("layer range must be a middle piece inside the model");
        }
        if record.ttl_secs == 0 || record.ttl_secs > MAX_RECORD_TTL_SECS {
            return Err(error(ErrorCode::BadTimestamp, "ttl out of range"));
        }
        if record.issued_at.abs_diff(now) > MAX_CLOCK_SKEW_SECS {
            return Err(error(
                ErrorCode::BadTimestamp,
                "record time stamp is too far from the hub's clock",
            ));
        }
        Ok(())
    }

    async fn announce(&self, signed: SignedRecord) -> Response {
        let record = match verify_record(&signed) {
            Ok(r) => r,
            Err(e) => return error(ErrorCode::BadSignature, e.to_string()),
        };
        let now = unix_now();
        if let Err(e) = Self::check(&record, now) {
            return e;
        }
        let id = NodeId(record.node_id);
        self.sweep();
        {
            let entries = self.entries.lock();
            if !entries.contains_key(&id) && entries.len() >= self.max_records {
                return error(ErrorCode::RouteLimit, "directory is full");
            }
        }
        // The address must answer, with the key the record names.
        let rtt = match self.probe(&record.addrs[0], id).await {
            Some(ms) => ms,
            None => {
                return error(
                    ErrorCode::Unreachable,
                    format!("{} did not answer a ping from the hub", record.addrs[0]),
                )
            }
        };
        info!(
            node = %id.short(),
            addr = %record.addrs[0],
            layers = format!("{}:{}", record.layer_begin, record.layer_end),
            model = %record.model_name,
            rtt_ms = format!("{rtt:.2}"),
            "announce"
        );
        self.entries.lock().insert(
            id,
            Entry {
                record,
                signed,
                hub_rtt_ms: rtt,
                seen_at: now,
            },
        );
        Response::Ack
    }

    /// Best of three pings to `addr`, pinned to `id`; `None` if it does not answer.
    async fn probe(&self, addr: &str, id: NodeId) -> Option<f32> {
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
        tokio::time::timeout(self.probe_timeout, fut)
            .await
            .ok()
            .flatten()
    }

    fn directory(&self) -> Response {
        self.sweep();
        let now = unix_now();
        let mut entries: Vec<(NodeId, DirectoryEntry)> = self
            .entries
            .lock()
            .iter()
            .map(|(id, e)| {
                (
                    *id,
                    DirectoryEntry {
                        record: e.signed.clone(),
                        hub_rtt_ms: e.hub_rtt_ms,
                        seen_at: e.seen_at,
                    },
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let dir = Directory {
            hub_id: *self.endpoint.node_id().as_bytes(),
            issued_at: now,
            entries: entries.into_iter().map(|(_, e)| e).collect(),
        };
        match sign_directory(self.endpoint.key(), &dir) {
            Ok(signed) => Response::Directory { directory: signed },
            Err(e) => error(ErrorCode::Internal, e.to_string()),
        }
    }
}
