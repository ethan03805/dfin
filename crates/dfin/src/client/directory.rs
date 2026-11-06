//! Fetching the signed directory from a hub and choosing a route from it.
//!
//! The hub publishes; the client decides. A route is a chain of stages whose layer
//! ranges tile `[begin, end)` exactly. Among chains, fewer hops win, then a lower sum
//! of hub-measured round trips, then chance, so that two clients with the same
//! directory do not all pile onto one node.

use crate::config::{LayerRange, StageEndpoint};
use dfin_net::directory::{verify_directory, verify_record};
use dfin_net::transport::CONTROL_TIMEOUT;
use dfin_net::{Endpoint, NodeId};
use dfin_proto::{
    Directory, NextHop, NodeRecord, Request, Response, SignedDirectory, MAX_DIRECTORY_BYTES,
    PROTOCOL_VERSION,
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

/// Fetch the hub's directory over QUIC (pinned to the hub's key) and verify it.
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
        other => return Err(format!("hub {}: unexpected reply {other:?}", hub.addr)),
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
        if record.protocol != PROTOCOL_VERSION
            || record.model_fingerprint != fingerprint
            || record.n_embd != n_embd
            || record.expires_at() <= now
            || record.addrs.is_empty()
            || record.layer_end <= record.layer_begin
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

/// Choose a chain tiling `[begin, end)`; nodes in `exclude` are avoided when any other
/// chain exists. Returns the route and how many listed records were usable.
pub fn select_route(
    dir: &Directory,
    now: u64,
    fingerprint: &str,
    n_embd: u32,
    begin: i32,
    end: i32,
    exclude: &HashSet<NodeId>,
) -> (Option<Vec<StageEndpoint>>, usize, usize) {
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
                    Some(b) => {
                        (next.hops.len(), next.rtt_sum) < (b.hops.len(), b.rtt_sum)
                    }
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
    (route, listed, usable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_net::directory::sign_record;
    use dfin_net::NodeKey;
    use dfin_proto::{ActDtype, DirectoryEntry};

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
            region: String::new(),
            attested: false,
            cloaked: false,
            issued_at: 1000,
            ttl_secs: 180,
        };
        DirectoryEntry {
            record: sign_record(key, &record).unwrap(),
            hub_rtt_ms: rtt,
            seen_at: 1000,
        }
    }

    fn dir(entries: Vec<DirectoryEntry>) -> Directory {
        Directory {
            hub_id: [0; 32],
            issued_at: 1000,
            entries,
        }
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
        let (route, listed, usable) =
            select_route(&d_, 1010, "fp", 64, 2, 6, &HashSet::new());
        assert_eq!((listed, usable), (6, 4));
        let route = route.unwrap();
        assert_eq!(route.len(), 1);
        assert_eq!(route[0].node_id, c.id(), "one hop beats two, lower rtt beats higher");

        let exclude: HashSet<NodeId> = [c.id()].into_iter().collect();
        let (route, _, _) = select_route(&d_, 1010, "fp", 64, 2, 6, &exclude);
        assert_eq!(route.unwrap()[0].node_id, d.id());

        let exclude: HashSet<NodeId> = [c.id(), d.id()].into_iter().collect();
        let (route, _, _) = select_route(&d_, 1010, "fp", 64, 2, 6, &exclude);
        let route = route.unwrap();
        assert_eq!(route.len(), 2);
        assert_eq!((route[0].node_id, route[1].node_id), (a.id(), b.id()));

        // Nothing covers 2:7; and excluding everything falls back to the best chain.
        let (route, _, _) = select_route(&d_, 1010, "fp", 64, 2, 7, &HashSet::new());
        assert!(route.is_none());
        let all: HashSet<NodeId> = [a.id(), b.id(), c.id(), d.id()].into_iter().collect();
        let (route, _, _) = select_route(&d_, 1010, "fp", 64, 2, 6, &all);
        assert_eq!(route.unwrap()[0].node_id, c.id());
    }
}
