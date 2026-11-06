//! Sharded models: a manifest listing per-layer GGUF split files with their hashes,
//! the model id derived from those hashes, and which shards a piece needs.
//!
//! `tools/shard_gguf.py` writes the shard set and the manifest. A node holding only
//! some shards still agrees with everyone else on the model id, because the id is a
//! hash over the manifest's shard hashes rather than over the bytes it holds.

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "dfin-shards-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Split {
    pub no: u32,
    pub file: String,
    pub size: u64,
    pub sha256: String,
    /// `meta`, `embd`, `layer` or `output`.
    pub kind: String,
    pub layer: Option<i32>,
    #[serde(default)]
    pub tensors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub model_name: String,
    pub arch: String,
    pub n_layer: i32,
    /// File name of the first split (the one llama.cpp is pointed at).
    pub first: String,
    pub splits: Vec<Split>,
    #[serde(default)]
    pub seeds: Vec<String>,
    #[serde(default)]
    pub model_id: String,
}

impl Manifest {
    pub fn load(path: &Path) -> anyhow::Result<Manifest> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading manifest {}", path.display()))?;
        let m: Manifest =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if m.format != FORMAT {
            bail!("{}: format {:?}, expected {FORMAT}", path.display(), m.format);
        }
        if m.splits.is_empty() {
            bail!("{}: no splits", path.display());
        }
        let computed = m.compute_id();
        if !m.model_id.is_empty() && m.model_id != computed {
            bail!(
                "{}: model_id {} does not match its splits ({computed})",
                path.display(),
                m.model_id
            );
        }
        Ok(Manifest {
            model_id: computed,
            ..m
        })
    }

    /// SHA-256 over the format tag and every split's number and hash, matching the tool.
    pub fn compute_id(&self) -> String {
        let mut h = Sha256::new();
        h.update(format!("{FORMAT}\n").as_bytes());
        for s in &self.splits {
            h.update(format!("{}:{}\n", s.no, s.sha256).as_bytes());
        }
        hex::encode(h.finalize())
    }

    /// Splits a piece needs: the first always, the embedding for a head, each layer in
    /// `[begin, end)` (`end == -1` means through the last layer), the output for a tail.
    pub fn needed(&self, begin: i32, end: i32) -> Vec<&Split> {
        let end = if end < 0 { self.n_layer } else { end };
        let head = begin == 0;
        let tail = end >= self.n_layer;
        self.splits
            .iter()
            .filter(|s| match s.kind.as_str() {
                "meta" => true,
                "embd" => head,
                "output" => tail,
                "layer" => s.layer.is_some_and(|l| l >= begin && l < end),
                _ => false,
            })
            .collect()
    }

    pub fn split(&self, no: u32) -> Option<&Split> {
        self.splits.iter().find(|s| s.no == no)
    }
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Is `split` present in `dir` with the right size (cheap) and, if asked, the right hash?
pub fn have_split(dir: &Path, split: &Split, verify_hash: bool) -> bool {
    let p = dir.join(&split.file);
    let Ok(md) = std::fs::metadata(&p) else {
        return false;
    };
    if md.len() != split.size {
        return false;
    }
    if verify_hash {
        return sha256_file(&p).map(|h| h == split.sha256).unwrap_or(false);
    }
    true
}

/// A model as given on the command line: a single GGUF, a first split, or a manifest.
#[derive(Clone, Debug)]
pub struct ResolvedModel {
    /// The file to hand to llama.cpp.
    pub gguf: PathBuf,
    /// What `Hello` and directory records carry: the manifest's model id for a shard
    /// set, else the whole-file fingerprint.
    pub fingerprint: String,
    pub manifest: Option<Manifest>,
    /// Directory holding the shards (with a manifest).
    pub dir: PathBuf,
}

/// Resolve `path`: a `*.manifest.json` names a shard set; a `.gguf` next to a manifest
/// that lists it as the first split is treated the same; any other GGUF is a single
/// file identified by `model_fingerprint`.
pub fn resolve_model(path: &Path) -> anyhow::Result<ResolvedModel> {
    let dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let manifest_path = if name.ends_with(".manifest.json") {
        Some(path.to_path_buf())
    } else {
        // A first split: look for a manifest in the same directory naming this file.
        let mut found = None;
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.ends_with(".manifest.json") {
                    if let Ok(m) = Manifest::load(&e.path()) {
                        if m.first == name {
                            found = Some(e.path());
                            break;
                        }
                    }
                }
            }
        }
        found
    };
    match manifest_path {
        Some(mp) => {
            let m = Manifest::load(&mp)?;
            Ok(ResolvedModel {
                gguf: dir.join(&m.first),
                fingerprint: m.model_id.clone(),
                manifest: Some(m),
                dir,
            })
        }
        None => {
            let fingerprint = dfin_proto::model_fingerprint(path)
                .with_context(|| format!("fingerprinting {}", path.display()))?;
            Ok(ResolvedModel {
                gguf: path.to_path_buf(),
                fingerprint,
                manifest: None,
                dir,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let mk = |no: u32, kind: &str, layer: Option<i32>| Split {
            no,
            file: format!("m-{:05}-of-00005.gguf", no + 1),
            size: 10,
            sha256: format!("{no:064x}"),
            kind: kind.into(),
            layer,
            tensors: vec![],
        };
        Manifest {
            format: FORMAT.into(),
            model_name: "m".into(),
            arch: "llama".into(),
            n_layer: 2,
            first: "m-00001-of-00005.gguf".into(),
            splits: vec![
                mk(0, "meta", None),
                mk(1, "embd", None),
                mk(2, "layer", Some(0)),
                mk(3, "layer", Some(1)),
                mk(4, "output", None),
            ],
            seeds: vec![],
            model_id: String::new(),
        }
    }

    #[test]
    fn needed_splits_per_piece() {
        let m = manifest();
        let nos = |b, e| m.needed(b, e).iter().map(|s| s.no).collect::<Vec<_>>();
        assert_eq!(nos(0, 1), vec![0, 1, 2]);
        assert_eq!(nos(1, 2), vec![0, 3, 4]);
        assert_eq!(nos(1, -1), vec![0, 3, 4]);
        assert_eq!(nos(0, -1), vec![0, 1, 2, 3, 4]);
        assert_eq!(nos(1, 1), vec![0]);
    }

    #[test]
    fn model_id_is_stable_and_matches_the_tool() {
        let m = manifest();
        let id = m.compute_id();
        assert_eq!(id.len(), 64);
        let mut other = manifest();
        other.splits[2].sha256 = "ff".repeat(32);
        assert_ne!(other.compute_id(), id);
        // The same formula as tools/shard_gguf.py: sha256("dfin-shards-v1\n" + "no:hash\n"...).
        let mut h = Sha256::new();
        h.update(b"dfin-shards-v1\n");
        for s in &m.splits {
            h.update(format!("{}:{}\n", s.no, s.sha256).as_bytes());
        }
        assert_eq!(id, hex::encode(h.finalize()));
    }
}
