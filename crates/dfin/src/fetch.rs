//! Fetching the shards a piece needs: from nodes in the directory that hold them, or
//! from HTTP seeds named in the manifest or on the command line. Every downloaded
//! shard is checked against the manifest's SHA-256 before it is used.

use crate::client::directory;
use crate::config::{parse_node_addr, LayerRange};
use crate::shards::{have_split, sha256_file, Manifest, Split};
use anyhow::{bail, Context as _};
use clap::Args;
use dfin_net::directory::{validate_record, verify_record};
use dfin_net::{Endpoint, NodeId, NodeKey};
use dfin_proto::{NextHop, Request, Response, MAX_SHARD_CHUNK};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct FetchArgs {
    /// The shard manifest (`*.manifest.json`).
    #[arg(long)]
    pub manifest: PathBuf,
    /// Where the shards go (default: the manifest's directory).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Layer range the shards are for, `a:b` (default: the whole model).
    #[arg(long)]
    pub layers: Option<LayerRange>,
    /// A hub whose directory lists nodes that hold shards of this model.
    #[arg(long, value_parser = parse_node_addr)]
    pub hub: Option<NextHop>,
    /// An HTTP URL prefix serving the shard files. Repeatable; the manifest's seeds are
    /// tried too.
    #[arg(long = "seed")]
    pub seeds: Vec<String>,
    /// Seconds to keep trying while no source has a shard.
    #[arg(long, default_value_t = 120)]
    pub wait: u64,
}

pub async fn run(args: FetchArgs, network: crate::network::Network) -> anyhow::Result<()> {
    if let Some(hub) = &args.hub {
        network.target(&hub.addr)?;
    }
    anyhow::ensure!(
        !network.private() || args.seeds.is_empty(),
        "HTTP seeds are disabled in strict anonymity mode; use onion peers or offline files"
    );
    let manifest = Manifest::load(&args.manifest)?;
    let dir = args
        .out
        .clone()
        .or_else(|| args.manifest.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;
    let (begin, end) = args.layers.map(|r| (r.begin, r.end)).unwrap_or((0, -1));
    let endpoint = network.client(NodeKey::generate())?;
    let n = fetch_missing(
        &endpoint,
        &manifest,
        &dir,
        begin,
        end,
        args.hub.as_ref(),
        &args.seeds,
        Duration::from_secs(args.wait),
    )
    .await?;
    println!(
        "{} shard(s) fetched into {}; model id {}",
        n,
        dir.display(),
        manifest.model_id
    );
    Ok(())
}

/// Make every shard that `[begin, end)` needs present in `dir`. Returns how many were
/// downloaded. Keeps retrying sources until `wait` elapses.
#[allow(clippy::too_many_arguments)]
pub async fn fetch_missing(
    endpoint: &Endpoint,
    manifest: &Manifest,
    dir: &Path,
    begin: i32,
    end: i32,
    hub: Option<&NextHop>,
    seeds: &[String],
    wait: Duration,
) -> anyhow::Result<usize> {
    let needed: Vec<Split> = manifest.needed(begin, end).into_iter().cloned().collect();
    let mut missing: Vec<Split> = needed
        .iter()
        .filter(|s| !have_split(dir, s, false))
        .cloned()
        .collect();
    if missing.is_empty() {
        info!(
            "all {} shards for layers {begin}:{end} are present in {}",
            needed.len(),
            dir.display()
        );
        return Ok(0);
    }
    info!(
        "{} of {} shards for layers {begin}:{end} missing from {}",
        missing.len(),
        needed.len(),
        dir.display()
    );
    let (all_seeds, http) = if endpoint.is_private() {
        anyhow::ensure!(
            seeds.is_empty(),
            "HTTP seeds are disabled in strict anonymity mode"
        );
        anyhow::ensure!(
            hub.is_some(),
            "strict shard fetching requires an onion hub; manifest HTTP seeds are disabled"
        );
        if !manifest.seeds.is_empty() {
            info!("ignoring manifest HTTP seeds in strict anonymity mode");
        }
        (Vec::new(), None)
    } else {
        let mut all_seeds = manifest.seeds.clone();
        all_seeds.extend(seeds.iter().cloned());
        (
            all_seeds,
            Some(
                reqwest::Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()?,
            ),
        )
    };
    let deadline = Instant::now() + wait;
    let mut fetched = 0usize;
    loop {
        let peers = match hub {
            Some(h) => match directory::fetch_async(endpoint, h).await {
                Ok((_, d)) => peers_for(&d, &manifest.model_id),
                Err(e) => {
                    warn!("{e}");
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        let mut still = Vec::new();
        for split in missing {
            if fetch_one(
                endpoint,
                http.as_ref(),
                manifest,
                dir,
                &split,
                &peers,
                &all_seeds,
            )
            .await
            {
                fetched += 1;
            } else {
                still.push(split);
            }
        }
        missing = still;
        if missing.is_empty() {
            return Ok(fetched);
        }
        if Instant::now() > deadline {
            bail!(
                "could not fetch {} shard(s): {}",
                missing.len(),
                missing
                    .iter()
                    .map(|s| s.file.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        info!("{} shard(s) still missing; retrying in 5 s", missing.len());
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

/// A node that serves this model: address, id and layer range.
struct Peer {
    addr: String,
    id: NodeId,
    begin: i32,
    end: i32,
}

fn peers_for(dir: &dfin_proto::Directory, model_id: &str) -> Vec<Peer> {
    let now = crate::hub::unix_now();
    dir.entries
        .iter()
        .filter_map(|e| verify_record(&e.record).ok())
        .filter(|r| r.model_fingerprint == model_id && validate_record(r, now).is_ok())
        .map(|r| Peer {
            addr: r.addrs[0].clone(),
            id: NodeId(r.node_id),
            begin: r.layer_begin,
            end: r.layer_end,
        })
        .collect()
}

/// Try the peers that should hold `split` first, then every other peer, then the seeds.
async fn fetch_one(
    endpoint: &Endpoint,
    http: Option<&reqwest::Client>,
    manifest: &Manifest,
    dir: &Path,
    split: &Split,
    peers: &[Peer],
    seeds: &[String],
) -> bool {
    let holds = |p: &Peer| match (split.kind.as_str(), split.layer) {
        ("layer", Some(l)) => l >= p.begin && l < p.end,
        ("meta", _) => true,
        _ => false,
    };
    let ordered: Vec<&Peer> = peers
        .iter()
        .filter(|p| holds(p))
        .chain(peers.iter().filter(|p| !holds(p)))
        .collect();
    for p in ordered {
        match from_peer(endpoint, manifest, dir, split, p).await {
            Ok(()) => {
                info!(shard = %split.file, from = %p.addr, "fetched from a node");
                return true;
            }
            Err(e) => tracing::debug!(shard = %split.file, from = %p.addr, "{e:#}"),
        }
    }
    for seed in seeds {
        let Some(http) = http else {
            return false;
        };
        match from_seed(http, dir, split, seed).await {
            Ok(()) => {
                info!(shard = %split.file, from = %seed, "fetched from a seed");
                return true;
            }
            Err(e) => warn!(shard = %split.file, from = %seed, "{e:#}"),
        }
    }
    false
}

/// Download to `<file>.part`, check the hash, then rename into place.
async fn finish(dir: &Path, split: &Split, part: &Path) -> anyhow::Result<()> {
    let got = tokio::task::spawn_blocking({
        let part = part.to_path_buf();
        move || sha256_file(&part)
    })
    .await??;
    if got != split.sha256 {
        let _ = tokio::fs::remove_file(part).await;
        bail!("hash mismatch for {}: got {got}", split.file);
    }
    tokio::fs::rename(part, dir.join(&split.file)).await?;
    Ok(())
}

async fn from_peer(
    endpoint: &Endpoint,
    manifest: &Manifest,
    dir: &Path,
    split: &Split,
    peer: &Peer,
) -> anyhow::Result<()> {
    let conn = endpoint.connect(&peer.addr, peer.id).await?;
    let part = dir.join(format!("{}.part", split.file));
    let mut file = tokio::fs::File::create(&part).await?;
    let mut offset = 0u64;
    let limit = MAX_SHARD_CHUNK + 65_536;
    let result: anyhow::Result<()> = async {
        while offset < split.size {
            let len = (split.size - offset).min(MAX_SHARD_CHUNK as u64) as u32;
            let r: Response = conn
                .call(
                    &Request::ShardGet {
                        model_id: manifest.model_id.clone(),
                        split: split.no,
                        offset,
                        len,
                    },
                    limit,
                    Duration::from_secs(120),
                )
                .await?;
            match r {
                Response::Shard { total, data } => {
                    if total != split.size {
                        bail!("peer reports {total} bytes, manifest says {}", split.size);
                    }
                    if data.is_empty() {
                        bail!("peer sent no data at offset {offset}");
                    }
                    file.write_all(&data).await?;
                    offset += data.len() as u64;
                }
                Response::Error { code, message } => bail!("{code:?}: {message}"),
                other => bail!("unexpected reply {other:?}"),
            }
        }
        file.flush().await?;
        Ok(())
    }
    .await;
    conn.close("done");
    drop(file);
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e);
    }
    finish(dir, split, &part).await
}

async fn from_seed(
    http: &reqwest::Client,
    dir: &Path,
    split: &Split,
    seed: &str,
) -> anyhow::Result<()> {
    let url = format!("{}/{}", seed.trim_end_matches('/'), split.file);
    let mut resp = http.get(&url).send().await?.error_for_status()?;
    let part = dir.join(format!("{}.part", split.file));
    let mut file = tokio::fs::File::create(&part).await?;
    let result: anyhow::Result<()> = async {
        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        Ok(())
    }
    .await;
    drop(file);
    if let Err(e) = result {
        let _ = tokio::fs::remove_file(&part).await;
        return Err(e).with_context(|| format!("downloading {url}"));
    }
    finish(dir, split, &part).await
}

/// Serve a shard range from `dir` for a `ShardGet`, or explain why not.
pub async fn serve_shard(
    manifest: &Manifest,
    dir: &Path,
    model_id: &str,
    split_no: u32,
    offset: u64,
    len: u32,
) -> Result<(u64, Vec<u8>), String> {
    if model_id != manifest.model_id {
        return Err("this node holds another model".into());
    }
    let split = manifest
        .split(split_no)
        .ok_or_else(|| format!("no split {split_no} in this model"))?;
    if !have_split(dir, split, false) {
        return Err(format!("shard {} is not on this node", split.file));
    }
    if offset > split.size {
        return Err("offset beyond the shard".into());
    }
    let len = len.min(MAX_SHARD_CHUNK) as u64;
    let len = len.min(split.size - offset) as usize;
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut f = tokio::fs::File::open(dir.join(&split.file))
        .await
        .map_err(|e| e.to_string())?;
    f.seek(std::io::SeekFrom::Start(offset))
        .await
        .map_err(|e| e.to_string())?;
    let mut data = vec![0u8; len];
    f.read_exact(&mut data).await.map_err(|e| e.to_string())?;
    Ok((split.size, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_net::{directory::sign_record, NodeKey};
    use dfin_proto::{ActDtype, Directory, DirectoryEntry, NodeRecord, PROTOCOL_VERSION};

    #[test]
    fn shard_discovery_rejects_signed_records_with_unbounded_lifetimes() {
        let key = NodeKey::generate();
        let now = crate::hub::unix_now();
        let mut record = NodeRecord {
            node_id: key.id().0,
            addrs: vec!["127.0.0.1:7001".into()],
            model_fingerprint: "fp".into(),
            model_name: "tiny".into(),
            layer_begin: 2,
            layer_end: 6,
            n_layer: 8,
            n_embd: 64,
            n_ctx_seq: 512,
            n_seq_max: 2,
            act_dtype: ActDtype::F16,
            backend: String::new(),
            region: String::new(),
            protocol: PROTOCOL_VERSION,
            attested: false,
            cloaked: false,
            issued_at: now,
            ttl_secs: 180,
        };
        let directory = |record: &NodeRecord| Directory {
            hub_id: key.id().0,
            issued_at: now,
            entries: vec![DirectoryEntry {
                record: sign_record(&key, record).unwrap(),
                hub_rtt_ms: 1.0,
                seen_at: now,
                probes: vec![],
            }],
        };
        assert_eq!(peers_for(&directory(&record), "fp").len(), 1);
        record.issued_at = u64::MAX;
        assert!(peers_for(&directory(&record), "fp").is_empty());
        record.issued_at = now;
        record.ttl_secs = u32::MAX;
        assert!(peers_for(&directory(&record), "fp").is_empty());
    }
}
