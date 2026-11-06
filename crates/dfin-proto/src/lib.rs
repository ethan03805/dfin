//! Wire protocol between the client daemon and pipeline stage nodes.
//!
//! Phase 0 transport: a stream (TCP today, QUIC later) carrying length-prefixed
//! frames. Each frame is a little-endian `u32` byte length followed by a
//! [`postcard`]-encoded [`Request`] or [`Response`].
//!
//! Hidden states travel as [`Activations`]: a dense `[n_tokens, n_embd]` matrix in
//! row-major order (token-major), as f32 or f16. FP8 is a later phase.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io;
use std::path::Path;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Bump whenever the message layout changes incompatibly.
pub const PROTOCOL_VERSION: u32 = 1;

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
        let n = self.n_tokens as usize * self.n_embd as usize;
        let expected = n * self.dtype.bytes_per_elem();
        if self.data.len() != expected {
            return Err(ProtoError::BadActivationLength {
                got: self.data.len(),
                expected,
            });
        }
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
}

/// Client daemon to stage.
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
    /// Run this stage's layers over `act` for `session`, at positions
    /// `pos0 .. pos0 + n_tokens`, extending the session's KV cache.
    /// `seq` is a client-chosen monotonically increasing number per session,
    /// echoed back so the client can match replies.
    Forward {
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
    },
    /// Drop KV entries for positions `p0 <= pos < p1` (`p1 = -1` means to the end).
    /// Used by prefix caching to roll a session back to a common prefix.
    SeqRm { session: u64, p0: i32, p1: i32 },
    /// Free all state for a session.
    SessionClose { session: u64 },
    Ping,
}

/// Stage to client daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Response {
    HelloOk {
        layer_begin: i32,
        layer_end: i32,
        n_embd: u32,
        n_ctx: u32,
        n_seq_max: u32,
        /// Human-readable backend description, for `/status`.
        backend: String,
    },
    ForwardOk {
        session: u64,
        seq: u64,
        act: Activations,
        /// Wall time the stage spent in llama_decode, for latency accounting.
        compute_ms: f32,
    },
    Ok,
    Pong,
    Error { code: ErrorCode, message: String },
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

/// Read one length-prefixed postcard frame.
pub async fn read_frame<R, T>(r: &mut R) -> io::Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len);
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
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
            session: 7,
            seq: 3,
            pos0: 10,
            act: Activations::from_f32(&[1.0, 2.0, 3.0, 4.0], 2, 2, ActDtype::F16),
        };
        let mut buf = Vec::new();
        write_frame(&mut buf, &req).await.unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let back: Request = read_frame(&mut cur).await.unwrap();
        match back {
            Request::Forward { session, seq, pos0, act } => {
                assert_eq!((session, seq, pos0), (7, 3, 10));
                assert_eq!(act.n_tokens, 2);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
