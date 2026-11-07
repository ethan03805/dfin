//! Wire protocol between the client daemon and pipeline stage nodes.
//!
//! Version 2 (phase 1): frames travel on QUIC streams (see `dfin-net`); routes are
//! chains of stages set up with sealed per-hop records; results return along the
//! chain or straight to the client. Each frame is a little-endian `u32` byte length
//! followed by a [`postcard`]-encoded [`Request`], [`Response`] or [`ReturnFrame`].
//! The contract is `docs/phase1-protocol.md`.
//!
//! Hidden states travel as [`Activations`]: a dense `[n_tokens, n_embd]` matrix in
//! row-major order (token-major), as f32 or f16. FP8 is a later phase.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io;
use std::path::Path;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Bump whenever the message layout changes incompatibly.
pub const PROTOCOL_VERSION: u32 = 2;

/// Largest frame either side will accept. A 4k-token prefill chunk of a
/// 6144-wide model is about 100 MB in f32, so leave generous headroom.
pub const MAX_FRAME_BYTES: u32 = 1 << 30;

#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    #[error("activation buffer has {got} bytes, expected {expected}")]
    BadActivationLength { got: usize, expected: usize },
}

/// Element type of an activation buffer on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActDtype {
    F32,
    F16,
}

impl ActDtype {
    pub fn bytes_per_elem(self) -> usize {
        match self {
            ActDtype::F32 => 4,
            ActDtype::F16 => 2,
        }
    }
}

impl std::str::FromStr for ActDtype {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Ok(ActDtype::F32),
            "f16" => Ok(ActDtype::F16),
            other => Err(format!("unknown activation dtype {other:?} (expected f32 or f16)")),
        }
    }
}

/// A `[n_tokens, n_embd]` block of residual-stream hidden states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Activations {
    pub dtype: ActDtype,
    pub n_tokens: u32,
    pub n_embd: u32,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl Activations {
    /// Pack an f32 matrix (token-major, `n_tokens * n_embd` values) for the wire.
    pub fn from_f32(values: &[f32], n_tokens: usize, n_embd: usize, dtype: ActDtype) -> Self {
        assert_eq!(values.len(), n_tokens * n_embd, "activation shape mismatch");
        let data = match dtype {
            ActDtype::F32 => {
                let mut out = Vec::with_capacity(values.len() * 4);
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            ActDtype::F16 => {
                let mut out = Vec::with_capacity(values.len() * 2);
                for v in values {
                    out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                }
                out
            }
        };
        Activations {
            dtype,
            n_tokens: n_tokens as u32,
            n_embd: n_embd as u32,
            data,
        }
    }

    /// Unpack to f32 (token-major).
    pub fn to_f32(&self) -> Result<Vec<f32>, ProtoError> {
        // Checked: a peer can claim any shape, and an overflowed product must not slip
        // past the length check by wrapping to the right number.
        let expected = (self.n_tokens as usize)
            .checked_mul(self.n_embd as usize)
            .and_then(|n| n.checked_mul(self.dtype.bytes_per_elem()))
            .unwrap_or(usize::MAX);
        if self.data.len() != expected {
            return Err(ProtoError::BadActivationLength {
                got: self.data.len(),
                expected,
            });
        }
        let n = self.data.len() / self.dtype.bytes_per_elem();
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            ActDtype::F32 => {
                for c in self.data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            ActDtype::F16 => {
                for c in self.data.chunks_exact(2) {
                    out.push(half::f16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            }
        }
        Ok(out)
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

/// Why a stage refused or failed a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    VersionMismatch,
    ModelMismatch,
    LayerRangeMismatch,
    UnknownSession,
    SessionLimit,
    ContextFull,
    Decode,
    BadRequest,
    Internal,
    /// The route id is not known on this connection.
    UnknownRoute,
    /// Too many routes on one connection.
    RouteLimit,
    /// The sealed hop record did not open with this node's key.
    BadRecord,
    /// A later hop failed; the message names it.
    Downstream,
    /// The last hop has no connection to deliver results on.
    NoReturnPath,
    /// A signature did not verify.
    BadSignature,
    /// The hub could not reach the announced address.
    Unreachable,
    /// A record's time stamp or lifetime is out of bounds.
    BadTimestamp,
    /// The node does not hold the requested shard.
    NoSuchShard,
    /// The node answers no more requests of this kind for now.
    RateLimited,
}

/// Address and identity of a node to dial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NextHop {
    pub addr: String,
    pub node_id: [u8; 32],
}

/// Where the last hop sends results. Only the last hop's record is consulted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnMode {
    /// Reply on the stream the forward arrived on; every hop relays it upstream.
    Reverse,
    /// Write results on the connection the client (identified by its key) holds to the last hop.
    Direct { client_id: [u8; 32] },
    /// Attach to a relay with `token` and write results there.
    Relay { relay: NextHop, token: [u8; 16] },
}

/// The plaintext of one hop's sealed instructions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopRecord {
    pub model_fingerprint: String,
    pub layer_begin: i32,
    pub layer_end: i32,
    /// The following hop, or `None` for the last.
    pub next: Option<NextHop>,
    pub ret: ReturnMode,
    /// The next hop's sealed record (empty for the last hop).
    #[serde(with = "serde_bytes")]
    pub inner: Vec<u8>,
}

/// One hop's share of a forward, appended as the forward travels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HopStat {
    pub layer_begin: i32,
    pub layer_end: i32,
    pub compute_ms: f32,
}

/// Client daemon or previous hop to a stage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Request {
    /// First message on every connection. The stage checks the protocol version,
    /// that both sides loaded the same GGUF, and that it serves the requested range.
    Hello {
        version: u32,
        model_fingerprint: String,
        layer_begin: i32,
        layer_end: i32,
    },
    /// Establish a route. `hop` is this node's record, sealed to its key.
    RouteSetup {
        route: u64,
        #[serde(with = "serde_bytes")]
        hop: Vec<u8>,
    },
    /// Run this stage's layers over `act` for `session`, at positions
    /// `pos0 .. pos0 + n_tokens`, extending the session's KV cache, then hand the
    /// output to the next hop. `seq` is a client-chosen monotonically increasing
    /// number per session, echoed back so the client can match results.
    Forward {
        route: u64,
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
        hops: Vec<HopStat>,
    },
    /// Drop KV entries for positions `p0 <= pos < p1` (`p1 = -1` means to the end)
    /// here and on every later hop. Used by prefix caching to roll a session back.
    SeqRm {
        route: u64,
        session: u64,
        p0: i32,
        p1: i32,
    },
    /// Free all state for a session along the route.
    SessionClose { route: u64, session: u64 },
    /// Tear the route down along the chain.
    RouteClose { route: u64 },
    Ping,
    /// Client to relay: pipe my streams to `first_hop` under `token`.
    RelayOpen { token: [u8; 16], first_hop: NextHop },
    /// Last hop to relay: pipe my one-way streams to the client holding `token`.
    RelayAttach { token: [u8; 16] },
    /// Cloaked node to relay: bind a public port for me and tunnel its datagrams on this
    /// stream. The reply is `RelayListening`; the stream then carries packet frames.
    RelayListen,
    /// Node to hub: list me. The hub verifies the signature and dials the first address.
    Announce { record: SignedRecord },
    /// Anyone to hub: the current signed directory.
    DirectoryGet,
    /// Hub to probe: ping `addr` (pinned to `node_id`) and report the round trip.
    Probe { addr: String, node_id: [u8; 32] },
    /// Fetch `len` bytes at `offset` of shard `split` of the model with id `model_id`
    /// from a node that holds it. Replies `Shard`.
    ShardGet {
        model_id: String,
        split: u32,
        offset: u64,
        len: u32,
    },
}

/// Largest shard chunk a node sends in one reply.
pub const MAX_SHARD_CHUNK: u32 = 4 << 20;

impl Request {
    /// Postcard variant index of `Forward`, the one request allowed to be large.
    pub const FORWARD_TAG: u8 = 2;
}

/// Stage to client daemon or previous hop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Response {
    HelloOk {
        layer_begin: i32,
        layer_end: i32,
        n_embd: u32,
        /// Width of the rows this stage takes and produces: `n_embd` times the number of
        /// state vectors at its boundaries (more than one for hyper-connection streams or
        /// residual banks).
        n_embd_inp: u32,
        n_embd_out: u32,
        n_ctx: u32,
        n_seq_max: u32,
        /// Human-readable backend description, for `/status`.
        backend: String,
        node_id: [u8; 32],
    },
    /// The route is set up; `hops` nodes acknowledged.
    RouteOk { hops: u32 },
    Ack,
    ForwardOk {
        session: u64,
        seq: u64,
        act: Activations,
        hops: Vec<HopStat>,
    },
    Pong,
    /// The public `host:port` the relay forwards to the cloaked node.
    RelayListening { addr: String },
    /// A probe's answer: best-of-three round trip and the probe's region label.
    Probed { rtt_ms: f32, region: String },
    Directory { directory: SignedDirectory },
    /// A piece of a shard file; `total` is the file's full length.
    Shard {
        total: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    Error { code: ErrorCode, message: String },
}

/// A result delivered to a return target (direct or relay mode) on a one-way stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReturnFrame {
    pub route: u64,
    pub session: u64,
    pub seq: u64,
    /// `ForwardOk` or `Error`.
    pub body: Response,
}

// ------------------------------------------------------------------------------------------
// Directory: what a node says about itself, signed by its key, and what a hub publishes.

/// A node's self-description. Signed by the node (see `dfin-net`'s `directory` module).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeRecord {
    pub node_id: [u8; 32],
    /// Addresses (`host:port`) the node can be dialled at, best first.
    pub addrs: Vec<String>,
    pub model_fingerprint: String,
    /// Human-readable model description (file stem or architecture summary).
    pub model_name: String,
    pub layer_begin: i32,
    pub layer_end: i32,
    pub n_layer: i32,
    pub n_embd: u32,
    /// KV capacity per session and concurrent sessions.
    pub n_ctx_seq: u32,
    pub n_seq_max: u32,
    pub act_dtype: ActDtype,
    pub backend: String,
    pub protocol: u32,
    /// The operator's own claim; verified by nobody in phase 1.
    pub region: String,
    /// Hardware attestation; always false until phase 2.
    pub attested: bool,
    /// The addresses belong to relays; the node itself accepts no inbound connections
    /// and cannot be placed by timing.
    pub cloaked: bool,
    /// Unix seconds when the record was made, and how long it is valid.
    pub issued_at: u64,
    pub ttl_secs: u32,
}

impl NodeRecord {
    pub fn expires_at(&self) -> u64 {
        self.issued_at.saturating_add(self.ttl_secs as u64)
    }
}

/// A postcard-encoded `NodeRecord` and the node's Ed25519 signature over those bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignedRecord {
    #[serde(with = "serde_bytes")]
    pub body: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

/// One vantage point's measurement of a node: best-of-three ping round trip from a
/// probe standing in `region`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeRtt {
    pub probe_id: [u8; 32],
    pub region: String,
    pub rtt_ms: f32,
}

/// One directory line: a node's signed record plus what the hub observed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub record: SignedRecord,
    /// Best of a few pings from the hub to the node's first address, in ms.
    pub hub_rtt_ms: f32,
    /// Unix seconds when the hub last accepted an announcement from this node.
    pub seen_at: u64,
    /// Round trips from the hub's vantage points (empty for cloaked nodes, whose
    /// addresses are relays).
    pub probes: Vec<ProbeRtt>,
}

/// The hub's list. Signed by the hub (`SignedDirectory`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Directory {
    pub hub_id: [u8; 32],
    pub issued_at: u64,
    pub entries: Vec<DirectoryEntry>,
}

/// A postcard-encoded `Directory` and the hub's signature over those bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignedDirectory {
    #[serde(with = "serde_bytes")]
    pub body: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

/// Largest directory frame a client will accept.
pub const MAX_DIRECTORY_BYTES: u32 = 16 << 20;
/// Longest a record may claim to be valid.
pub const MAX_RECORD_TTL_SECS: u32 = 3600;
/// How far a record's `issued_at` may be from the hub's clock.
pub const MAX_CLOCK_SKEW_SECS: u64 = 600;

/// Bytes a `Forward` or `ForwardOk` frame may carry for `n_tokens * n_embd` values,
/// with headroom for the envelope and hop stats.
pub fn forward_frame_limit(n_tokens: u32, n_embd: u32) -> u32 {
    // Saturating: a peer's advertised width can be anything, and the cap applies anyway.
    let payload = (n_tokens as u64)
        .saturating_mul(n_embd as u64)
        .saturating_mul(4);
    payload
        .saturating_add(65536)
        .min(MAX_FRAME_BYTES as u64) as u32
}

/// Write one length-prefixed postcard frame.
pub async fn write_frame<W, T>(w: &mut W, msg: &T) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let body = postcard::to_stdvec(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if body.len() > MAX_FRAME_BYTES as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let len = (body.len() as u32).to_le_bytes();
    w.write_all(&len).await?;
    w.write_all(&body).await?;
    w.flush().await
}

/// Read one length-prefixed postcard frame of at most [`MAX_FRAME_BYTES`].
pub async fn read_frame<R, T>(r: &mut R) -> io::Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    read_frame_max(r, MAX_FRAME_BYTES).await
}

/// Read one length-prefixed postcard frame of at most `max` bytes. Receivers pick
/// `max` by what message they expect, so a peer cannot make them allocate more.
pub async fn read_frame_max<R, T>(r: &mut R, max: u32) -> io::Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len);
    if len > max.min(MAX_FRAME_BYTES) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame of {len} bytes exceeds the {max} byte limit"),
        ));
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body).await?;
    postcard::from_bytes(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Cheap identity for a GGUF file so the client and every stage can confirm they
/// loaded the same weights: SHA-256 over the file length, the first 4 MiB and the
/// last 4 MiB. Not a security measure; content-addressed shards replace this later.
pub fn model_fingerprint(path: &Path) -> io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    const WINDOW: u64 = 4 << 20;
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut h = Sha256::new();
    h.update(len.to_le_bytes());
    let mut buf = vec![0u8; WINDOW.min(len) as usize];
    f.read_exact(&mut buf)?;
    h.update(&buf);
    if len > WINDOW {
        let tail = WINDOW.min(len - WINDOW);
        f.seek(SeekFrom::End(-(tail as i64)))?;
        let mut buf = vec![0u8; tail as usize];
        f.read_exact(&mut buf)?;
        h.update(&buf);
    }
    Ok(hex::encode(h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_roundtrip_f32() {
        let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        let a = Activations::from_f32(&v, 3, 4, ActDtype::F32);
        assert_eq!(a.to_f32().unwrap(), v);
    }

    #[test]
    fn activations_roundtrip_f16_is_close() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32).sin() * 3.0).collect();
        let a = Activations::from_f32(&v, 8, 8, ActDtype::F16);
        let back = a.to_f32().unwrap();
        for (x, y) in v.iter().zip(&back) {
            assert!((x - y).abs() < 4e-3, "{x} vs {y}");
        }
    }

    #[tokio::test]
    async fn frame_roundtrip() {
        let req = Request::Forward {
            route: 1,
            session: 7,
            seq: 3,
            pos0: 10,
            act: Activations::from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2, ActDtype::F16),
            hops: Vec::new(),
        };
        let mut buf = Vec::new();
        write_frame(&mut buf, &req).await.unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let back: Request = read_frame(&mut cur).await.unwrap();
        match back {
            Request::Forward {
                session, seq, pos0, act, ..
            } => {
                assert_eq!((session, seq, pos0), (7, 3, 10));
                assert_eq!(act.n_tokens, 2);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn forward_tag_matches_variant_index() {
        let req = Request::Forward {
            route: 0,
            session: 0,
            seq: 0,
            pos0: 0,
            act: Activations::from_f32(&[0.0], 1, 1, ActDtype::F32),
            hops: Vec::new(),
        };
        let bytes = postcard::to_stdvec(&req).unwrap();
        assert_eq!(bytes[0], Request::FORWARD_TAG);
        let ping = postcard::to_stdvec(&Request::Ping).unwrap();
        assert_ne!(ping[0], Request::FORWARD_TAG);
    }
}
