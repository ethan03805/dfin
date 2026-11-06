//! Blocking connection to one `dfin stage` server over the dfin-proto framing.

use crate::config::LayerRange;
use dfin_proto::{Request, Response, MAX_FRAME_BYTES, PROTOCOL_VERSION};
use serde::{de::DeserializeOwned, Serialize};
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// Blocking twin of `dfin_proto::write_frame`. Returns the bytes written.
pub fn write_frame_blocking<W: Write, T: Serialize>(w: &mut W, msg: &T) -> io::Result<usize> {
    let body =
        postcard::to_stdvec(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if body.len() > MAX_FRAME_BYTES as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut frame = Vec::with_capacity(body.len() + 4);
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(&body);
    w.write_all(&frame)?;
    w.flush()?;
    Ok(frame.len())
}

/// Blocking twin of `dfin_proto::read_frame`. Returns the message and bytes read.
pub fn read_frame_blocking<R: Read, T: DeserializeOwned>(r: &mut R) -> io::Result<(T, usize)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len);
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body)?;
    let msg =
        postcard::from_bytes(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((msg, len as usize + 4))
}

#[derive(Debug, thiserror::Error)]
pub enum ConnError {
    /// Transport failure; the connection has been dropped and must be re-established.
    #[error("stage {addr}: connection error: {err}")]
    Io { addr: String, err: io::Error },
    /// The stage answered with an error or something unexpected.
    #[error("stage {addr}: {message}")]
    Protocol { addr: String, message: String },
}

/// Per-stage counters, published in `/status`.
#[derive(Clone, Debug, Default)]
pub struct StageCounters {
    pub forwards: u64,
    pub rtt_ms_sum: f64,
    pub compute_ms_sum: f64,
    /// Single-token forwards only (decode steps), so prefill chunks do not skew the averages.
    pub decode_forwards: u64,
    pub decode_rtt_ms_sum: f64,
    pub decode_compute_ms_sum: f64,
    /// Best of five `Ping` round trips measured right after the handshake, in ms.
    pub ping_ms: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

pub struct StageConn {
    pub addr: String,
    pub layers: LayerRange,
    pub backend: String,
    stream: Option<TcpStream>,
    pub counters: StageCounters,
}

impl StageConn {
    pub fn new(addr: String, layers: LayerRange) -> Self {
        StageConn {
            addr,
            layers,
            backend: String::new(),
            stream: None,
            counters: StageCounters::default(),
        }
    }

    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    fn io_err(&self, err: io::Error) -> ConnError {
        ConnError::Io {
            addr: self.addr.clone(),
            err,
        }
    }

    fn proto_err(&self, message: String) -> ConnError {
        ConnError::Protocol {
            addr: self.addr.clone(),
            message,
        }
    }

    /// Open the TCP connection and perform the `Hello` handshake.
    pub fn connect(&mut self, fingerprint: &str, n_embd: u32) -> Result<(), ConnError> {
        self.stream = None;
        let addrs: Vec<_> = self
            .addr
            .to_socket_addrs()
            .map_err(|e| self.io_err(e))?
            .collect();
        let mut last = io::Error::new(io::ErrorKind::NotFound, "address resolved to nothing");
        let mut stream = None;
        for a in addrs {
            match TcpStream::connect_timeout(&a, Duration::from_secs(5)) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => last = e,
            }
        }
        let stream = stream.ok_or_else(|| self.io_err(last))?;
        stream.set_nodelay(true).map_err(|e| self.io_err(e))?;
        // Generous: a big prefill chunk on a slow stage can take a while.
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .map_err(|e| self.io_err(e))?;
        self.stream = Some(stream);

        let resp = self.call(&Request::Hello {
            version: PROTOCOL_VERSION,
            model_fingerprint: fingerprint.to_string(),
            layer_begin: self.layers.begin,
            layer_end: self.layers.end,
        })?;
        let result = match resp {
            Response::HelloOk {
                layer_begin,
                layer_end,
                n_embd: stage_embd,
                n_ctx,
                n_seq_max,
                backend,
            } => {
                if stage_embd != n_embd {
                    Err(self.proto_err(format!(
                        "n_embd mismatch: stage {stage_embd}, client {n_embd}"
                    )))
                } else if layer_begin != self.layers.begin || layer_end != self.layers.end {
                    Err(self.proto_err(format!(
                        "serves layers {layer_begin}:{layer_end}, route expects {}",
                        self.layers
                    )))
                } else {
                    tracing::info!(
                        addr = %self.addr,
                        layers = %self.layers,
                        n_ctx,
                        n_seq_max,
                        backend = %backend,
                        "connected to stage"
                    );
                    self.backend = backend;
                    Ok(())
                }
            }
            Response::Error { code, message } => {
                Err(self.proto_err(format!("hello rejected ({code:?}): {message}")))
            }
            other => Err(self.proto_err(format!("unexpected hello reply {other:?}"))),
        };
        if result.is_err() {
            self.stream = None;
        } else {
            self.measure_ping();
        }
        result
    }

    /// Round-trip time of the smallest request (`Ping`), best of five, in ms. This is the
    /// floor for a decode hop: framing, TCP and the stage's request loop, without compute.
    fn measure_ping(&mut self) {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            match self.call(&Request::Ping) {
                Ok(Response::Pong) => best = best.min(t.elapsed().as_secs_f64() * 1e3),
                _ => return,
            }
        }
        if best.is_finite() {
            self.counters.ping_ms = best;
            tracing::info!(addr = %self.addr, ping_ms = format!("{best:.2}"), "stage ping");
        }
    }

    /// Send one request and read its reply. Any I/O error drops the connection.
    pub fn call(&mut self, req: &Request) -> Result<Response, ConnError> {
        let Some(stream) = self.stream.as_mut() else {
            return Err(self.io_err(io::Error::new(io::ErrorKind::NotConnected, "not connected")));
        };
        let result = write_frame_blocking(stream, req).and_then(|sent| {
            let (resp, recv) = read_frame_blocking::<_, Response>(stream)?;
            Ok((resp, sent, recv))
        });
        match result {
            Ok((resp, sent, recv)) => {
                self.counters.bytes_sent += sent as u64;
                self.counters.bytes_received += recv as u64;
                Ok(resp)
            }
            Err(err) => {
                self.stream = None;
                Err(self.io_err(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_proto::{ActDtype, Activations};

    #[test]
    fn blocking_frames_roundtrip() {
        let req = Request::Forward {
            session: 5,
            seq: 2,
            pos0: 3,
            act: Activations::from_f32(&[1.0, 2.0], 1, 2, ActDtype::F32),
        };
        let mut buf = Vec::new();
        let n = write_frame_blocking(&mut buf, &req).unwrap();
        assert_eq!(n, buf.len());
        let mut cur = io::Cursor::new(buf);
        let (back, m): (Request, usize) = read_frame_blocking(&mut cur).unwrap();
        assert_eq!(m, n);
        assert!(matches!(
            back,
            Request::Forward {
                session: 5,
                seq: 2,
                pos0: 3,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn blocking_writer_matches_async_reader() {
        let mut buf = Vec::new();
        write_frame_blocking(&mut buf, &Request::Ping).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let back: Request = dfin_proto::read_frame(&mut cur).await.unwrap();
        assert!(matches!(back, Request::Ping));
    }
}
