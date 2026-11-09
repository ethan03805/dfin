//! `dfin hub`: the directory.
//!
//! Nodes announce a signed record of what they serve; the hub verifies the signature,
//! dials the announced address to check it answers, and lists the record until it
//! expires. Anyone can fetch the list, signed by the hub. The hub never assigns
//! routes: clients build their own from the list. A hub can still bias or personalize
//! that list, so independent route choice is not protection against a malicious hub.
//! Protocol: `docs/protocol.md` section 8.

use anyhow::Context as _;
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use clap::Args;
use dfin_net::directory::{sign_directory, validate_record, verify_record};
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Connection, Endpoint, NodeId, NodeKey};
use dfin_proto::{
    read_frame_max, Directory, DirectoryEntry, ErrorCode, NodeRecord, Request, Response,
    SignedRecord, MAX_CLOCK_SKEW_SECS,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

#[cfg(test)]
mod privacy_tests;

#[derive(Args, Debug)]
pub struct HubArgs {
    /// Local listener (TCP in Tor mode, UDP in direct mode).
    #[arg(long, default_value = "127.0.0.1:7100")]
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
}

/// Time allowed to dial an announcing node and ping it three times. Onion circuits
/// take far longer to build than a direct connection.
const REACH_TIMEOUT_DIRECT: Duration = Duration::from_secs(5);
const REACH_TIMEOUT_TOR: Duration = Duration::from_secs(60);

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

#[derive(Default)]
struct DirectoryState {
    entries: HashMap<NodeId, Entry>,
    pending: HashSet<NodeId>,
}

struct AnnouncementLease<'a> {
    hub: &'a Hub,
    id: NodeId,
}

impl Drop for AnnouncementLease<'_> {
    fn drop(&mut self) {
        self.hub.state.lock().pending.remove(&self.id);
    }
}

/// Closes the reachability check's connection however the check ends.
struct CheckConnection(Connection);

impl Drop for CheckConnection {
    fn drop(&mut self) {
        self.0.close("reachability check ended");
    }
}

pub struct Hub {
    endpoint: Endpoint,
    state: Mutex<DirectoryState>,
    max_records: usize,
    reach_timeout: Duration,
}

pub async fn run(args: HubArgs, network: crate::network::Network) -> anyhow::Result<()> {
    network.listener(&args.listen)?;
    if let Some(http) = &args.http {
        network.listener(http)?;
    }
    let key = NodeKey::load_or_create(&args.identity)
        .with_context(|| format!("node key {}", args.identity.display()))?;
    let endpoint = network.server(key, &args.listen).await?;
    info!(
        "hub ready on {} node {}",
        endpoint.local_addr()?,
        endpoint.node_id()
    );
    let hub = Arc::new(Hub {
        endpoint: endpoint.clone(),
        state: Mutex::new(DirectoryState::default()),
        max_records: args.max_records.max(1),
        reach_timeout: if network.private() {
            REACH_TIMEOUT_TOR
        } else {
            REACH_TIMEOUT_DIRECT
        },
    });

    if let Some(http) = &args.http {
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
            .route("/directory", get(http_directory))
            .with_state(hub.clone());
        let app = if network.private() {
            app.layer(axum::middleware::from_fn(
                crate::network::local_request_only,
            ))
        } else {
            app
        };
        let listener = tokio::net::TcpListener::bind(http)
            .await
            .with_context(|| format!("binding {http}"))?;
        info!(
            "directory as JSON at http://{}/directory",
            listener.local_addr()?
        );
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
        .state
        .lock()
        .entries
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
                "attested": r.attested,
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
        let mut requests = tokio::task::JoinSet::new();
        loop {
            let accepted = tokio::select! {
                _ = conn.closed() => break,
                _ = requests.join_next(), if !requests.is_empty() => continue,
                r = conn.accept_bi(), if requests.len() < 32 => r,
            };
            let (mut send, mut recv) = match accepted {
                Ok(x) => x,
                Err(_) => break,
            };
            let hub = self.clone();
            requests.spawn(async move {
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
        conn.close("hub connection ended");
        requests.shutdown().await;
    }

    fn sweep(&self) {
        let now = unix_now();
        self.state
            .lock()
            .entries
            .retain(|_, e| e.record.expires_at() > now);
    }

    fn check(record: &NodeRecord, now: u64) -> Result<(), Response> {
        validate_record(record, now).map_err(|(code, message)| error(code, message))?;
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
        if self.endpoint.is_private()
            && (record.attested
                || !record.backend.is_empty()
                || record
                    .addrs
                    .iter()
                    .any(|a| dfin_net::privacy::onion_target(a).is_err()))
        {
            return error(
                ErrorCode::BadRequest,
                "Tor directory requires onion-only records without hardware or unverified attestation claims",
            );
        }
        let id = NodeId(record.node_id);
        self.sweep();
        {
            let mut state = self.state.lock();
            if state.pending.contains(&id) {
                return error(ErrorCode::BadRequest, "announcement already in progress");
            }
            // A large directory must not imply unbounded concurrent outbound work.
            if state.pending.len() >= 32 {
                return error(ErrorCode::RateLimited, "hub reachability checks are busy");
            }
            let reserved = state.entries.len()
                + state
                    .pending
                    .iter()
                    .filter(|id| !state.entries.contains_key(id))
                    .count();
            if !state.entries.contains_key(&id) && reserved >= self.max_records {
                return error(ErrorCode::RouteLimit, "directory is full");
            }
            state.pending.insert(id);
        }
        let _lease = AnnouncementLease { hub: self, id };
        // The address must answer, with the key the record names.
        let rtt = match self.reach(&record.addrs[0], id).await {
            Some(ms) => ms,
            None => {
                return error(
                    ErrorCode::Unreachable,
                    format!("{} did not answer a ping from the hub", record.addrs[0]),
                )
            }
        };
        // A slow check may have consumed the remaining lifetime.
        if let Err((code, message)) = validate_record(&record, unix_now()) {
            return error(code, message);
        }
        info!(
            node = %id.short(),
            addr = %record.addrs[0],
            layers = format!("{}:{}", record.layer_begin, record.layer_end),
            model = %record.model_name,
            rtt_ms = format!("{rtt:.2}"),
            "announce"
        );
        self.state.lock().entries.insert(
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
    async fn reach(&self, addr: &str, id: NodeId) -> Option<f32> {
        let fut = async {
            let conn = self.endpoint.connect(addr, id).await.ok()?;
            let _connection = CheckConnection(conn.clone());
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
            conn.close("reachability check");
            Some(best)
        };
        tokio::time::timeout(self.reach_timeout, fut)
            .await
            .ok()
            .flatten()
    }

    fn directory(&self) -> Response {
        self.sweep();
        let now = unix_now();
        let mut entries: Vec<(NodeId, DirectoryEntry)> = self
            .state
            .lock()
            .entries
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

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_proto::{MAX_RECORD_TTL_SECS, PROTOCOL_VERSION};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Generated records: the check never panics, and whatever it accepts has the bounds
    /// the hub relies on afterwards: a first address to dial, a middle layer range inside
    /// the model, a lifetime it can expire, text it can store.
    #[test]
    fn generated_records_are_checked_without_panics() {
        let mut rng = StdRng::seed_from_u64(0x4b);
        let base = 1_800_000_000u64;
        let mut accepted = 0;
        for _ in 0..5000 {
            // Start from a record the hub accepts and push a few fields to edge values,
            // so both outcomes happen often and every bound gets tested on its own.
            let now = [0u64, base, u64::MAX][rng.random_range(0..3)];
            let mut record = NodeRecord {
                node_id: rng.random(),
                addrs: vec!["10.0.0.5:7001".into()],
                model_fingerprint: "fp".into(),
                model_name: "tiny".into(),
                layer_begin: 2,
                layer_end: 6,
                n_layer: 8,
                n_embd: 64,
                n_ctx_seq: 4096,
                n_seq_max: 4,
                act_dtype: dfin_proto::ActDtype::F16,
                backend: "CPU".into(),
                protocol: PROTOCOL_VERSION,
                attested: false,
                issued_at: now,
                ttl_secs: 180,
            };
            let ints = [i32::MIN, -1, 0, 1, 2, 5, 6, 8, 9, i32::MAX];
            let lens = [0usize, 1, 64, 65, 128, 129, 256, 257, 4096];
            let mut p = |rng: &mut StdRng| rng.random_bool(0.25);
            if p(&mut rng) {
                let n = [0usize, 1, 2, 8, 9, 40][rng.random_range(0..6)];
                record.addrs = (0..n)
                    .map(|_| "a".repeat(lens[rng.random_range(0..lens.len())]))
                    .collect();
            }
            if p(&mut rng) {
                record.model_fingerprint = "f".repeat(lens[rng.random_range(0..lens.len())]);
            }
            if p(&mut rng) {
                record.model_name = "m".repeat(lens[rng.random_range(0..lens.len())]);
            }
            if p(&mut rng) {
                record.backend = "b".repeat(lens[rng.random_range(0..lens.len())]);
            }
            if p(&mut rng) {
                record.layer_begin = ints[rng.random_range(0..ints.len())];
            }
            if p(&mut rng) {
                record.layer_end = ints[rng.random_range(0..ints.len())];
            }
            if p(&mut rng) {
                record.n_layer = ints[rng.random_range(0..ints.len())];
            }
            if p(&mut rng) {
                record.protocol = [0, u32::MAX][rng.random_range(0..2)];
            }
            if p(&mut rng) {
                record.issued_at = [
                    0,
                    now.saturating_sub(601),
                    now.saturating_sub(600),
                    now.saturating_add(600),
                    now.saturating_add(601),
                    u64::MAX,
                ][rng.random_range(0..6)];
            }
            if p(&mut rng) {
                record.ttl_secs = [0, 1, 3600, 3601, u32::MAX][rng.random_range(0..5)];
            }
            record.attested = rng.random_bool(0.5);
            if Hub::check(&record, now).is_ok() {
                accepted += 1;
                assert!((1..=8).contains(&record.addrs.len()));
                assert!(record.addrs.iter().all(|a| (1..=128).contains(&a.len())));
                assert!(record.layer_begin > 0 && record.layer_end > record.layer_begin);
                assert!(record.layer_end <= record.n_layer);
                assert!((1..=MAX_RECORD_TTL_SECS).contains(&record.ttl_secs));
                assert!(record.issued_at.abs_diff(now) <= MAX_CLOCK_SKEW_SECS);
                assert!(record.expires_at() > record.issued_at);
                assert_eq!(record.protocol, PROTOCOL_VERSION);
            }
        }
        assert!(
            accepted > 0,
            "the generator never produced an acceptable record"
        );
    }
}
