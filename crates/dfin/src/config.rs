//! Shared configuration types: layer ranges and the route a client daemon drives.

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// A half-open range of transformer layers `[begin, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerRange {
    pub begin: i32,
    pub end: i32,
}

impl LayerRange {
    pub fn new(begin: i32, end: i32) -> Self {
        LayerRange { begin, end }
    }

    pub fn len(&self) -> i32 {
        self.end - self.begin
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.begin
    }
}

impl fmt::Display for LayerRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.begin, self.end)
    }
}

impl FromStr for LayerRange {
    type Err = anyhow::Error;

    /// Parses `"a:b"` (half-open). `b` may be `-1` or `end` for "through the last layer".
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (a, b) = s
            .split_once(':')
            .with_context(|| format!("layer range {s:?} must look like `a:b`"))?;
        let begin: i32 = a
            .trim()
            .parse()
            .with_context(|| format!("bad layer range start {a:?}"))?;
        let end: i32 = match b.trim() {
            "end" | "" => -1,
            v => v
                .parse()
                .with_context(|| format!("bad layer range end {v:?}"))?,
        };
        if begin < 0 {
            bail!("layer range start must be >= 0");
        }
        if end != -1 && end <= begin {
            bail!("layer range {s:?} is empty");
        }
        Ok(LayerRange { begin, end })
    }
}

/// One remote stage in a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageEndpoint {
    /// `host:port` of a `dfin stage` server.
    pub addr: String,
    pub layers: LayerRange,
}

/// The route a client daemon drives: local head, remote stages in order, local tail.
///
/// TOML form:
/// ```toml
/// model = "models/tiny-llama-f32.gguf"
/// head = "0:2"
/// tail = "6:8"
/// [[stages]]
/// addr = "127.0.0.1:7001"
/// layers = "2:4"
/// [[stages]]
/// addr = "127.0.0.1:7002"
/// layers = "4:6"
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteConfig {
    pub model: PathBuf,
    #[serde(with = "layer_range_str")]
    pub head: LayerRange,
    #[serde(with = "layer_range_str")]
    pub tail: LayerRange,
    #[serde(default)]
    pub stages: Vec<StageEndpointToml>,
}

/// TOML-friendly stage entry (layer range as a string).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageEndpointToml {
    pub addr: String,
    #[serde(with = "layer_range_str")]
    pub layers: LayerRange,
}

impl RouteConfig {
    pub fn from_toml_file(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: RouteConfig =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        // Resolve the model path relative to the config file.
        if cfg.model.is_relative() {
            if let Some(dir) = path.parent() {
                cfg.model = dir.join(&cfg.model);
            }
        }
        Ok(cfg)
    }

    pub fn stages(&self) -> Vec<StageEndpoint> {
        self.stages
            .iter()
            .map(|s| StageEndpoint {
                addr: s.addr.clone(),
                layers: s.layers,
            })
            .collect()
    }

    /// Check that head, stages and tail tile `[0, n_layer)` exactly, in order.
    pub fn validate(&self, n_layer: i32) -> anyhow::Result<()> {
        let mut tail = self.tail;
        if tail.end == -1 {
            tail.end = n_layer;
        }
        if self.head.begin != 0 {
            bail!("head must start at layer 0, got {}", self.head);
        }
        if tail.end != n_layer {
            bail!("tail must end at n_layer={n_layer}, got {}", tail);
        }
        let mut cursor = self.head.end;
        for (i, s) in self.stages.iter().enumerate() {
            if s.layers.begin != cursor {
                bail!(
                    "stage {i} ({}) starts at layer {} but the previous piece ends at {cursor}",
                    s.addr,
                    s.layers.begin
                );
            }
            if s.layers.is_empty() {
                bail!("stage {i} ({}) has an empty layer range", s.addr);
            }
            cursor = s.layers.end;
        }
        if tail.begin != cursor {
            bail!(
                "tail starts at layer {} but the previous piece ends at {cursor}",
                tail.begin
            );
        }
        Ok(())
    }
}

/// Serde adapter: `LayerRange` as the string `"a:b"`.
pub mod layer_range_str {
    use super::LayerRange;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(r: &LayerRange, s: S) -> Result<S::Ok, S::Error> {
        r.to_string().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<LayerRange, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ranges() {
        assert_eq!("0:3".parse::<LayerRange>().unwrap(), LayerRange::new(0, 3));
        assert_eq!(
            "5:end".parse::<LayerRange>().unwrap(),
            LayerRange::new(5, -1)
        );
        assert!("3:3".parse::<LayerRange>().is_err());
        assert!("x".parse::<LayerRange>().is_err());
    }

    #[test]
    fn validate_route() {
        let cfg: RouteConfig = toml::from_str(
            r#"
model = "m.gguf"
head = "0:2"
tail = "6:8"
[[stages]]
addr = "a:1"
layers = "2:4"
[[stages]]
addr = "b:1"
layers = "4:6"
"#,
        )
        .unwrap();
        cfg.validate(8).unwrap();
        assert!(cfg.validate(9).is_err());
    }
}
