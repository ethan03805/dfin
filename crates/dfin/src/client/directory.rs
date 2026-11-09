//! Fetching the signed directory from a hub and choosing a route from it.
//!
//! The hub publishes; the client decides. A route is a chain of stages whose layer
//! ranges tile `[begin, end)` exactly. Among chains, fewer hops win, then a lower sum
//! of hub-measured round trips, then chance, so that two clients with the same
//! directory do not all pile onto one node.

use crate::config::{LayerRange, StageEndpoint};
use dfin_net::directory::{validate_record, verify_directory, verify_record};
use dfin_net::transport::CONTROL_TIMEOUT;
use dfin_net::{Endpoint, NodeId};
use dfin_proto::{
    Directory, NextHop, NodeRecord, Request, Response, SignedDirectory, MAX_DIRECTORY_BYTES,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokio::runtime::Handle;

/// What a resolved route came from, for `/status`.
#[derive(Clone, Debug, Default)]
pub struct DirectoryInfo {
    pub hub: String,
    pub issued_at: u64,
    pub from_cache: bool,
    pub nodes_listed: usize,
    pub nodes_usable: usize,
}

/// Fetch the hub's directory (pinned to the hub's key) and verify it.
pub fn fetch(
    rt: &Handle,
    endpoint: &Endpoint,
    hub: &NextHop,
) -> Result<(SignedDirectory, Directory), String> {
    rt.block_on(fetch_async(endpoint, hub))
}

/// The same, from async code.
pub async fn fetch_async(
    endpoint: &Endpoint,
    hub: &NextHop,
) -> Result<(SignedDirectory, Directory), String> {
    let hub_id = NodeId(hub.node_id);
    // A hub must not learn the client identity used for inference or shard fetching.
    let isolated = if endpoint.is_private() {
        Some(endpoint.isolated_client().map_err(|e| e.to_string())?)
    } else {
        None
    };
    let endpoint = isolated.as_ref().unwrap_or(endpoint);
    let conn = endpoint
        .connect(&hub.addr, hub_id)
        .await
        .map_err(|e| format!("hub {}: {e}", hub.addr))?;
    let r: Response = conn
        .call(&Request::DirectoryGet, MAX_DIRECTORY_BYTES, CONTROL_TIMEOUT)
        .await
        .map_err(|e| format!("hub {}: {e}", hub.addr))?;
    conn.close("done");
    let signed = match r {
        Response::Directory { directory } => directory,
        Response::Error { code, message } => {
            return Err(format!("hub {} ({code:?}): {message}", hub.addr))
        }
        other => {
            return Err(format!(
                "hub {}: unexpected reply {}",
                hub.addr,
                other.kind()
            ))
        }
    };
    let dir = verify_directory(&signed, &hub_id)
        .map_err(|e| format!("hub {}: directory does not verify: {e}", hub.addr))?;
    Ok((signed, dir))
}

pub fn load_cache(path: &Path, hub: &NextHop) -> Option<Directory> {
    let bytes = std::fs::read(path).ok()?;
    let signed: SignedDirectory = postcard::from_bytes(&bytes).ok()?;
    verify_directory(&signed, &NodeId(hub.node_id)).ok()
}

pub fn save_cache(path: &Path, signed: &SignedDirectory) {
    if let Ok(bytes) = postcard::to_stdvec(signed) {
        if let Err(e) = std::fs::write(path, bytes) {
            tracing::warn!("cannot write directory cache {}: {e}", path.display());
        }
    }
}

/// Filter even signed/cache-sourced records: a signature does not make a direct target private.
pub fn onion_only(dir: &mut Directory) {
    dir.entries.retain(|entry| {
        verify_record(&entry.record).is_ok_and(|r| {
            !r.attested
                && r.backend.is_empty()
                && !r.addrs.is_empty()
                && r.addrs
                    .iter()
                    .all(|a| dfin_net::privacy::onion_target(a).is_ok())
        })
    });
}

/// A verified, unexpired record that serves the right model.
struct Candidate {
    record: NodeRecord,
    rtt_ms: f32,
}

fn candidates(
    dir: &Directory,
    now: u64,
    fingerprint: &str,
    n_embd: u32,
) -> (usize, Vec<Candidate>) {
    let mut out = Vec::new();
    for e in &dir.entries {
        let Ok(record) = verify_record(&e.record) else {
            continue;
        };
        if validate_record(&record, now).is_err()
            || record.model_fingerprint != fingerprint
            || record.n_embd != n_embd
        {
            continue;
        }
        out.push(Candidate {
            record,
            rtt_ms: e.hub_rtt_ms,
        });
    }
    (dir.entries.len(), out)
}

#[derive(Clone)]
struct Path_ {
    hops: Vec<usize>,
    rtt_sum: f32,
}

/// What `select_route` found.
pub struct Selection {
    pub route: Option<Vec<StageEndpoint>>,
    pub listed: usize,
    pub usable: usize,
}

/// Choose a chain tiling `[begin, end)`; nodes in `exclude` are avoided when any other
/// chain exists.
pub fn select_route(
    dir: &Directory,
    now: u64,
    fingerprint: &str,
    n_embd: u32,
    begin: i32,
    end: i32,
    exclude: &HashSet<NodeId>,
) -> Selection {
    let (listed, cands) = candidates(dir, now, fingerprint, n_embd);
    let usable = cands.len();
    let pick = |excluded: &HashSet<NodeId>| -> Option<Vec<StageEndpoint>> {
        // Dynamic programme over the cursor: best known chain ending at each layer.
        let mut best: HashMap<i32, Path_> = HashMap::new();
        best.insert(
            begin,
            Path_ {
                hops: Vec::new(),
                rtt_sum: 0.0,
            },
        );
        let mut cursors: Vec<i32> = vec![begin];
        let mut i = 0;
        while i < cursors.len() {
            let cur = cursors[i];
            i += 1;
            let Some(here) = best.get(&cur).cloned() else {
                continue;
            };
            for (ci, c) in cands.iter().enumerate() {
                let r = &c.record;
                if r.layer_begin != cur || r.layer_end > end {
                    continue;
                }
                if excluded.contains(&NodeId(r.node_id)) {
                    continue;
                }
                let jitter = rand::random::<f32>() * 0.01;
                let next = Path_ {
                    hops: here.hops.iter().copied().chain([ci]).collect(),
                    rtt_sum: here.rtt_sum + c.rtt_ms.max(0.0) + jitter,
                };
                let better = match best.get(&r.layer_end) {
                    None => true,
                    Some(b) => (next.hops.len(), next.rtt_sum) < (b.hops.len(), b.rtt_sum),
                };
                if better {
                    best.insert(r.layer_end, next);
                    if !cursors.contains(&r.layer_end) {
                        cursors.push(r.layer_end);
                    }
                }
            }
        }
        let path = best.get(&end)?;
        Some(
            path.hops
                .iter()
                .map(|&ci| {
                    let r = &cands[ci].record;
                    StageEndpoint {
                        addr: r.addrs[0].clone(),
                        node_id: NodeId(r.node_id),
                        layers: LayerRange::new(r.layer_begin, r.layer_end),
                    }
                })
                .collect(),
        )
    };
    let route = pick(exclude).or_else(|| {
        if exclude.is_empty() {
            None
        } else {
            pick(&HashSet::new())
        }
    });
    Selection {
        route,
        listed,
        usable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_net::directory::sign_record;
    use dfin_net::NodeKey;
    use dfin_proto::{ActDtype, DirectoryEntry, PROTOCOL_VERSION};

    fn entry(key: &NodeKey, begin: i32, end: i32, rtt: f32, fp: &str) -> DirectoryEntry {
        let record = NodeRecord {
            node_id: *key.id().as_bytes(),
            addrs: vec![format!("10.0.0.{}:7001", begin)],
            model_fingerprint: fp.into(),
            model_name: "m".into(),
            layer_begin: begin,
            layer_end: end,
            n_layer: 8,
            n_embd: 64,
            n_ctx_seq: 512,
            n_seq_max: 2,
            act_dtype: ActDtype::F16,
            backend: "CPU".into(),
            protocol: PROTOCOL_VERSION,
            attested: false,
            issued_at: 1000,
            ttl_secs: 180,
        };
        DirectoryEntry {
            record: sign_record(key, &record).unwrap(),
            hub_rtt_ms: rtt,
            seen_at: 1000,
        }
    }

    fn sel(
        d: &Directory,
        begin: i32,
        end: i32,
        exclude: &HashSet<NodeId>,
    ) -> Option<Vec<StageEndpoint>> {
        select_route(d, 1010, "fp", 64, begin, end, exclude).route
    }

    fn dir(entries: Vec<DirectoryEntry>) -> Directory {
        Directory {
            hub_id: [0; 32],
            issued_at: 1000,
            entries,
        }
    }

    #[test]
    fn private_selection_rejects_signed_direct_and_mixed_records() {
        let key = NodeKey::generate();
        let direct = entry(&key, 2, 6, 1.0, "fp");
        let mut record = verify_record(&direct.record).unwrap();
        record.addrs = vec![format!("{}.onion:7001", "a".repeat(56))];
        record.backend.clear();
        let mut onion = direct.clone();
        onion.record = sign_record(&key, &record).unwrap();
        record.addrs.push("127.0.0.1:7001".into());
        let mut mixed = direct.clone();
        mixed.record = sign_record(&key, &record).unwrap();
        let mut directory = dir(vec![direct, onion, mixed]);
        onion_only(&mut directory);
        assert_eq!(directory.entries.len(), 1);
        assert!(sel(&directory, 2, 6, &HashSet::new()).unwrap()[0]
            .addr
            .ends_with(".onion:7001"));
    }

    #[test]
    fn signed_cached_records_must_have_bounded_lifetimes_and_valid_fields() {
        use dfin_net::directory::sign_directory;
        use dfin_proto::{MAX_CLOCK_SKEW_SECS, MAX_RECORD_TTL_SECS};

        let node = NodeKey::generate();
        let hub = NodeKey::generate();
        let original = entry(&node, 2, 6, 1.0, "fp");
        let mutations: &[(&str, fn(&mut NodeRecord))] = &[
            ("future", |r| r.issued_at = 1010 + MAX_CLOCK_SKEW_SECS + 1),
            ("overflow", |r| r.issued_at = u64::MAX),
            ("excessive lifetime", |r| {
                r.ttl_secs = MAX_RECORD_TTL_SECS + 1
            }),
            ("zero lifetime", |r| {
                r.issued_at = 1011;
                r.ttl_secs = 0;
            }),
            ("expired", |r| {
                r.issued_at = 1000;
                r.ttl_secs = 10;
            }),
            ("range outside model", |r| r.n_layer = 5),
            ("too many addresses", |r| {
                r.addrs = vec!["10.0.0.1:7001".into(); 9]
            }),
            ("oversized address", |r| r.addrs = vec!["a".repeat(129)]),
            ("oversized description", |r| r.model_name = "m".repeat(257)),
        ];
        let path =
            std::env::temp_dir().join(format!("dfin-cache-test-{:016x}", rand::random::<u64>()));
        let hub_hop = NextHop {
            addr: "unused".into(),
            node_id: hub.id().0,
        };
        for (case, mutate) in mutations {
            let mut record = verify_record(&original.record).unwrap();
            mutate(&mut record);
            let mut entry = original.clone();
            entry.record = sign_record(&node, &record).unwrap();
            let mut directory = dir(vec![entry]);
            directory.hub_id = hub.id().0;
            save_cache(&path, &sign_directory(&hub, &directory).unwrap());
            let cached = load_cache(&path, &hub_hop).expect("authentic hub signature");
            std::fs::remove_file(&path).unwrap();
            let selected = select_route(&cached, 1010, "fp", 64, 2, 6, &HashSet::new());
            assert!(
                selected.route.is_none(),
                "accepted signed cached record: {case}"
            );
            assert_eq!(selected.usable, 0, "{case}");
        }

        // Consumers can use records older than the announcement clock-skew limit
        // while their bounded TTL remains valid, including during a hub outage.
        let mut valid = original.clone();
        let mut record = verify_record(&valid.record).unwrap();
        record.ttl_secs = MAX_RECORD_TTL_SECS;
        valid.record = sign_record(&node, &record).unwrap();
        assert!(
            select_route(&dir(vec![valid]), 3010, "fp", 64, 2, 6, &HashSet::new())
                .route
                .is_some()
        );
    }

    #[test]
    fn fewest_hops_then_rtt_then_exclusion() {
        let a = NodeKey::generate(); // 2:4
        let b = NodeKey::generate(); // 4:6
        let c = NodeKey::generate(); // 2:6 in one hop
        let d = NodeKey::generate(); // 2:6 too, slower
        let other = NodeKey::generate(); // other model
        let stale = NodeKey::generate();
        let mut stale_e = entry(&stale, 2, 6, 0.1, "fp");
        stale_e.record = sign_record(
            &stale,
            &NodeRecord {
                issued_at: 1,
                ttl_secs: 1,
                ..verify_record(&stale_e.record).unwrap()
            },
        )
        .unwrap();
        let d_ = dir(vec![
            entry(&a, 2, 4, 1.0, "fp"),
            entry(&b, 4, 6, 1.0, "fp"),
            entry(&c, 2, 6, 5.0, "fp"),
            entry(&d, 2, 6, 9.0, "fp"),
            entry(&other, 2, 6, 0.1, "zz"),
            stale_e,
        ]);
        let s = select_route(&d_, 1010, "fp", 64, 2, 6, &HashSet::new());
        assert_eq!((s.listed, s.usable), (6, 4));
        let route = s.route.unwrap();
        assert_eq!(route.len(), 1);
        assert_eq!(
            route[0].node_id,
            c.id(),
            "one hop beats two, lower rtt beats higher"
        );

        let exclude: HashSet<NodeId> = [c.id()].into_iter().collect();
        assert_eq!(sel(&d_, 2, 6, &exclude).unwrap()[0].node_id, d.id());

        let exclude: HashSet<NodeId> = [c.id(), d.id()].into_iter().collect();
        let route = sel(&d_, 2, 6, &exclude).unwrap();
        assert_eq!(route.len(), 2);
        assert_eq!((route[0].node_id, route[1].node_id), (a.id(), b.id()));

        // Nothing covers 2:7; and excluding everything falls back to the best chain.
        assert!(sel(&d_, 2, 7, &HashSet::new()).is_none());
        let all: HashSet<NodeId> = [a.id(), b.id(), c.id(), d.id()].into_iter().collect();
        assert_eq!(sel(&d_, 2, 6, &all).unwrap()[0].node_id, c.id());
    }

    /// Generated directories: whatever a hub lists (bad signatures, empty address lists,
    /// absurd ranges, NaN round trips, expired or far-future records), route choice must
    /// not panic, and a route it returns must tile the requested layers exactly with
    /// nodes that have an address.
    #[test]
    fn generated_directories_never_break_route_choice() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xd1f);
        let keys: Vec<NodeKey> = (0..6).map(|_| NodeKey::generate()).collect();
        for round in 0..600 {
            let n = rng.random_range(0..8);
            let mut entries = Vec::new();
            for _ in 0..n {
                let key = &keys[rng.random_range(0..keys.len())];
                let fp = if rng.random_bool(0.8) { "fp" } else { "other" };
                let (a, b) = (rng.random_range(-2..10), rng.random_range(-2..10));
                let rtt = [f32::NAN, -1.0, 0.0, 0.5, 3.0, f32::INFINITY][rng.random_range(0..6)];
                let mut e = entry(key, a, b, rtt, fp);
                let mut rec = verify_record(&e.record).unwrap();
                if rng.random_bool(0.2) {
                    rec.n_embd = 65;
                }
                if rng.random_bool(0.2) {
                    rec.addrs.clear();
                }
                if rng.random_bool(0.2) {
                    rec.addrs.push("x".repeat(200));
                }
                if rng.random_bool(0.1) {
                    rec.protocol = 0;
                }
                if rng.random_bool(0.1) {
                    rec.issued_at = 0;
                }
                if rng.random_bool(0.1) {
                    rec.issued_at = u64::MAX;
                    rec.ttl_secs = u32::MAX;
                }
                let signer = if rng.random_bool(0.9) { key } else { &keys[0] };
                e.record = sign_record(signer, &rec).unwrap();
                if rng.random_bool(0.3) {
                    e.record.signature[0] ^= 1;
                }
                e.seen_at = rng.random();
                entries.push(e);
            }
            let d_ = dir(entries);
            let (begin, end) = (rng.random_range(-2..10), rng.random_range(-2..10));
            let exclude: HashSet<NodeId> = keys
                .iter()
                .filter(|_| rng.random_bool(0.3))
                .map(|k| k.id())
                .collect();
            let now = [0u64, 1010, u64::MAX][rng.random_range(0..3)];
            let s = select_route(&d_, now, "fp", 64, begin, end, &exclude);
            assert_eq!(s.listed, d_.entries.len());
            assert!(s.usable <= s.listed);
            let Some(route) = s.route else { continue };
            assert!(
                begin == end || !route.is_empty(),
                "round {round}: empty route"
            );
            let mut cur = begin;
            for hop in &route {
                assert_eq!(hop.layers.begin, cur, "round {round}: gap in the chain");
                assert!(hop.layers.end > hop.layers.begin, "round {round}");
                assert!(
                    !hop.addr.is_empty(),
                    "round {round}: hop without an address"
                );
                cur = hop.layers.end;
            }
            assert_eq!(cur, end, "round {round}: chain does not reach the end");
        }
    }
}
