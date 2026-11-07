//! The route: a chain of stages between the local head and tail, driven from the
//! engine thread. The engine is synchronous; every network call here runs on the
//! tokio runtime through `Handle::block_on`.
//!
//! Protocol: `docs/phase1-protocol.md`.

use crate::config::StageEndpoint;
use dfin_net::transport::{CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{seal, Connection, Endpoint, NetError, NodeId};
use dfin_proto::{
    forward_frame_limit, read_frame_max, write_frame, Activations, ErrorCode, HopRecord, HopStat,
    NextHop, Request, Response, ReturnFrame, ReturnMode, PROTOCOL_VERSION,
};
use std::str::FromStr;
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tokio::sync::mpsc;

/// Time allowed for one forward to cross the whole chain and come back.
pub const FORWARD_TIMEOUT: Duration = Duration::from_secs(600);
const SETUP_TIMEOUT: Duration = Duration::from_secs(90);

/// How results come back from the last hop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReturnPolicy {
    /// Back along the chain. Only the first hop learns the client's address.
    Reverse,
    /// Straight from the last hop on a connection the client holds to it.
    Direct,
    /// Through a relay: the client reaches the first hop through it and the last hop
    /// delivers results to it. No stage learns the client's address.
    Relay(NextHop),
}

impl ReturnPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReturnPolicy::Reverse => "reverse",
            ReturnPolicy::Direct => "direct",
            ReturnPolicy::Relay(_) => "relay",
        }
    }
}

impl FromStr for ReturnPolicy {
    type Err = String;
    /// `reverse` or `direct`; relay mode is chosen by naming a relay instead.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "reverse" => Ok(ReturnPolicy::Reverse),
            "direct" => Ok(ReturnPolicy::Direct),
            other => Err(format!("unknown return mode {other:?} (reverse, direct)")),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    /// Transport failure; the route has been dropped and must be rebuilt.
    #[error("route: {0}")]
    Net(#[from] NetError),
    /// A stage refused something; not retried.
    #[error("{0}")]
    Protocol(String),
    /// A stage answered with an error.
    #[error("stage {addr} ({code:?}): {message}")]
    Stage {
        addr: String,
        code: ErrorCode,
        message: String,
    },
}

/// Route-level counters, published in `/status`.
#[derive(Clone, Debug, Default)]
pub struct RouteCounters {
    pub forwards: u64,
    pub rtt_ms_sum: f64,
    /// Single-token forwards only (decode steps), so prefill chunks do not skew the averages.
    pub decode_forwards: u64,
    pub decode_rtt_ms_sum: f64,
    /// Best of five `Ping` round trips to the first hop, in ms.
    pub ping_ms: f64,
    /// Wall time of the last route setup, in ms.
    pub setup_ms: f64,
    /// Times the route was rebuilt after a failure.
    pub rebuilds: u64,
}

/// Per-hop counters from the hop stats that travel with each forward.
#[derive(Clone, Debug, Default)]
pub struct HopCounters {
    pub forwards: u64,
    pub compute_ms_sum: f64,
    pub decode_forwards: u64,
    pub decode_compute_ms_sum: f64,
}

pub struct Route {
    pub hops: Vec<StageEndpoint>,
    pub policy: ReturnPolicy,
    endpoint: Endpoint,
    rt: Handle,
    fingerprint: String,
    n_embd: u32,
    /// Row width the first hop must take (the head's output) and the last hop must
    /// produce (the tail's input).
    width_in: u32,
    width_out: u32,
    /// Largest forward (tokens) this client sends; bounds result frames.
    max_tokens: u32,
    route_id: u64,
    /// Relay session token (relay mode).
    token: [u8; 16],
    first: Option<Connection>,
    last: Option<Connection>,
    returns: Option<mpsc::Receiver<ReturnFrame>>,
    reader: Option<tokio::task::JoinHandle<()>>,
    pub counters: RouteCounters,
    pub hop_counters: Vec<HopCounters>,
    /// Backend description per hop; known only for hops this client spoke `Hello` with.
    pub backends: Vec<String>,
}

impl From<std::io::Error> for RouteError {
    fn from(e: std::io::Error) -> Self {
        RouteError::Net(NetError::Io(e))
    }
}

fn stage_err(addr: &str, code: ErrorCode, message: String) -> RouteError {
    RouteError::Stage {
        addr: addr.to_string(),
        code,
        message,
    }
}

impl Route {
    pub fn new(
        hops: Vec<StageEndpoint>,
        policy: ReturnPolicy,
        endpoint: Endpoint,
        rt: Handle,
        fingerprint: String,
        n_embd: u32,
        width_in: u32,
        width_out: u32,
        max_tokens: u32,
    ) -> Route {
        let n = hops.len();
        Route {
            hops,
            policy,
            endpoint,
            rt,
            fingerprint,
            n_embd,
            width_in,
            width_out,
            max_tokens,
            route_id: 0,
            token: [0; 16],
            first: None,
            last: None,
            returns: None,
            reader: None,
            counters: RouteCounters::default(),
            hop_counters: vec![HopCounters::default(); n],
            backends: vec![String::new(); n],
        }
    }

    pub fn is_connected(&self) -> bool {
        let live = |c: &Option<Connection>| c.as_ref().is_some_and(|c| !c.is_closed());
        live(&self.first) && (matches!(self.policy, ReturnPolicy::Reverse) || live(&self.last))
    }

    pub fn first_addr(&self) -> &str {
        &self.hops[0].addr
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// `Hello` with one hop; `want` is the row width this client expects it to take (first
    /// hop) or to produce (last hop), checked against what it reports.
    fn hello(
        &self,
        conn: &Connection,
        hop: &StageEndpoint,
        want: Option<(bool, u32)>,
    ) -> Result<String, RouteError> {
        let req = Request::Hello {
            version: PROTOCOL_VERSION,
            model_fingerprint: self.fingerprint.clone(),
            layer_begin: hop.layers.begin,
            layer_end: hop.layers.end,
        };
        let resp: Response =
            self.rt
                .block_on(conn.call(&req, MAX_CONTROL_FRAME, CONTROL_TIMEOUT))?;
        match resp {
            Response::HelloOk {
                layer_begin,
                layer_end,
                n_embd,
                n_embd_inp,
                n_embd_out,
                node_id,
                backend,
                ..
            } => {
                match want {
                    Some((true, w)) if n_embd_inp != w => {
                        return Err(RouteError::Protocol(format!(
                            "stage {} takes rows of {n_embd_inp}, the head produces {w}",
                            hop.addr
                        )))
                    }
                    Some((false, w)) if n_embd_out != w => {
                        return Err(RouteError::Protocol(format!(
                            "stage {} produces rows of {n_embd_out}, the tail takes {w}",
                            hop.addr
                        )))
                    }
                    _ => {}
                }
                if n_embd != self.n_embd {
                    return Err(RouteError::Protocol(format!(
                        "stage {}: n_embd {n_embd}, client {}",
                        hop.addr, self.n_embd
                    )));
                }
                if (layer_begin, layer_end) != (hop.layers.begin, hop.layers.end) {
                    return Err(RouteError::Protocol(format!(
                        "stage {} serves layers {layer_begin}:{layer_end}, route expects {}",
                        hop.addr, hop.layers
                    )));
                }
                if NodeId(node_id) != hop.node_id {
                    return Err(RouteError::Protocol(format!(
                        "stage {} reports node id {} but the connection authenticated {}",
                        hop.addr,
                        NodeId(node_id),
                        hop.node_id
                    )));
                }
                Ok(backend)
            }
            Response::Error { code, message } => Err(stage_err(
                &hop.addr,
                code,
                format!("hello rejected: {message}"),
            )),
            other => Err(RouteError::Protocol(format!(
                "stage {}: unexpected hello reply {}",
                hop.addr,
                other.kind()
            ))),
        }
    }

    /// Seal the per-hop records from the last hop backwards.
    fn sealed_records(&self, client_id: NodeId) -> Result<Vec<u8>, RouteError> {
        let n = self.hops.len();
        let mut inner: Vec<u8> = Vec::new();
        for i in (0..n).rev() {
            let hop = &self.hops[i];
            let last = i + 1 == n;
            let rec = HopRecord {
                model_fingerprint: self.fingerprint.clone(),
                layer_begin: hop.layers.begin,
                layer_end: hop.layers.end,
                next: if last {
                    None
                } else {
                    Some(NextHop {
                        addr: self.hops[i + 1].addr.clone(),
                        node_id: *self.hops[i + 1].node_id.as_bytes(),
                    })
                },
                ret: if last {
                    match &self.policy {
                        ReturnPolicy::Reverse => ReturnMode::Reverse,
                        ReturnPolicy::Direct => ReturnMode::Direct {
                            client_id: *client_id.as_bytes(),
                        },
                        ReturnPolicy::Relay(relay) => ReturnMode::Relay {
                            relay: relay.clone(),
                            token: self.token,
                        },
                    }
                } else {
                    ReturnMode::Reverse
                },
                inner,
            };
            let plain = postcard::to_stdvec(&rec)
                .map_err(|e| RouteError::Protocol(format!("encoding hop record: {e}")))?;
            inner = seal(&hop.node_id, &plain)
                .map_err(|e| RouteError::Protocol(format!("sealing to {}: {e}", hop.addr)))?;
        }
        Ok(inner)
    }

    /// Connect to the ends, set the route up along the chain, start the return reader.
    pub fn connect(&mut self) -> Result<(), RouteError> {
        self.disconnect();
        let t = Instant::now();
        let n = self.hops.len();
        self.token = rand::random();
        let first = match self.policy.clone() {
            ReturnPolicy::Relay(relay) => {
                // Reach the first hop through the relay; results come back the same way.
                let c = self
                    .rt
                    .block_on(self.endpoint.connect(&relay.addr, NodeId(relay.node_id)))?;
                let open = Request::RelayOpen {
                    token: self.token,
                    first_hop: NextHop {
                        addr: self.hops[0].addr.clone(),
                        node_id: *self.hops[0].node_id.as_bytes(),
                    },
                };
                let r: Response =
                    self.rt
                        .block_on(c.call(&open, MAX_CONTROL_FRAME, SETUP_TIMEOUT))?;
                match r {
                    Response::Ack => {}
                    Response::Error { code, message } => {
                        return Err(stage_err(&relay.addr, code, format!("relay: {message}")))
                    }
                    other => {
                        return Err(RouteError::Protocol(format!(
                            "relay {}: unexpected reply {}",
                            relay.addr,
                            other.kind()
                        )))
                    }
                }
                c
            }
            _ => self.rt.block_on(
                self.endpoint
                    .connect(&self.hops[0].addr, self.hops[0].node_id),
            )?,
        };
        self.backends[0] = self.hello(&first, &self.hops[0], Some((true, self.width_in)))?;
        let last = match &self.policy {
            ReturnPolicy::Reverse => None,
            ReturnPolicy::Relay(_) => Some(first.clone()),
            ReturnPolicy::Direct if n == 1 => Some(first.clone()),
            ReturnPolicy::Direct => {
                let hop = &self.hops[n - 1];
                let c = self
                    .rt
                    .block_on(self.endpoint.connect(&hop.addr, hop.node_id))?;
                self.backends[n - 1] = self.hello(&c, hop, Some((false, self.width_out)))?;
                Some(c)
            }
        };

        self.route_id = rand::random();
        let sealed = self.sealed_records(self.endpoint.node_id())?;
        let resp: Response = self.rt.block_on(first.call(
            &Request::RouteSetup {
                route: self.route_id,
                hop: sealed,
            },
            MAX_CONTROL_FRAME,
            SETUP_TIMEOUT,
        ))?;
        match resp {
            Response::RouteOk { hops } if hops as usize == n => {}
            Response::RouteOk { hops } => {
                return Err(RouteError::Protocol(format!(
                    "route setup acknowledged by {hops} hops, expected {n}"
                )))
            }
            Response::Error { code, message } => {
                return Err(stage_err(&self.hops[0].addr, code, message))
            }
            other => {
                return Err(RouteError::Protocol(format!(
                    "unexpected route setup reply {}",
                    other.kind()
                )))
            }
        }

        if let Some(last_conn) = &last {
            let (tx, rx) = mpsc::channel::<ReturnFrame>(16);
            let conn = last_conn.clone();
            let limit = forward_frame_limit(self.max_tokens, self.width_in.max(self.width_out));
            self.reader = Some(self.rt.spawn(async move {
                loop {
                    let mut s = match conn.accept_uni().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::debug!("return path closed: {e}");
                            break;
                        }
                    };
                    match read_frame_max::<_, ReturnFrame>(&mut s, limit).await {
                        Ok(frame) => {
                            if tx.send(frame).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("bad return frame: {e}");
                        }
                    }
                }
            }));
            self.returns = Some(rx);
        }

        self.first = Some(first);
        self.last = last;
        self.counters.setup_ms = t.elapsed().as_secs_f64() * 1e3;
        self.measure_ping();
        tracing::info!(
            route = self.route_id,
            hops = n,
            mode = self.policy.as_str(),
            setup_ms = format!("{:.1}", self.counters.setup_ms),
            "route ready"
        );
        Ok(())
    }

    /// Round-trip time of the smallest request (`Ping`) to the first hop, best of five.
    fn measure_ping(&mut self) {
        let Some(first) = self.first.clone() else {
            return;
        };
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let r: Result<Response, NetError> =
                self.rt
                    .block_on(first.call(&Request::Ping, MAX_CONTROL_FRAME, CONTROL_TIMEOUT));
            match r {
                Ok(Response::Pong) => best = best.min(t.elapsed().as_secs_f64() * 1e3),
                _ => return,
            }
        }
        if best.is_finite() {
            self.counters.ping_ms = best;
            tracing::info!(addr = %self.hops[0].addr, ping_ms = format!("{best:.2}"), "first hop ping");
        }
    }

    /// Begin a new identity context; no existing connection survives the change.
    pub fn replace_endpoint(&mut self, endpoint: Endpoint) {
        self.disconnect();
        self.endpoint = endpoint;
    }

    /// Tear the route down, best effort, and forget the connections.
    pub fn disconnect(&mut self) {
        if let Some(first) = self.first.take() {
            if !first.is_closed() {
                let req = Request::RouteClose {
                    route: self.route_id,
                };
                let _ = self.rt.block_on(first.call::<_, Response>(
                    &req,
                    MAX_CONTROL_FRAME,
                    Duration::from_secs(2),
                ));
            }
            first.close("route closed");
        }
        if let Some(last) = self.last.take() {
            last.close("route closed");
        }
        if let Some(r) = self.reader.take() {
            r.abort();
        }
        self.returns = None;
    }

    fn drop_on_net_error<T>(&mut self, r: Result<T, RouteError>) -> Result<T, RouteError> {
        if let Err(RouteError::Net(_)) = &r {
            self.first = None;
            self.last = None;
        }
        r
    }

    /// Send one forward through the chain and wait for the last hop's output.
    pub fn forward(
        &mut self,
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
    ) -> Result<(Activations, Vec<HopStat>), RouteError> {
        let n_tokens = act.n_tokens;
        let t = Instant::now();
        let r = self.forward_inner(session, seq, pos0, act);
        let r = self.drop_on_net_error(r);
        let (act, hops) = r?;
        let rtt_ms = t.elapsed().as_secs_f64() * 1e3;
        self.counters.forwards += 1;
        self.counters.rtt_ms_sum += rtt_ms;
        let decode = n_tokens == 1;
        if decode {
            self.counters.decode_forwards += 1;
            self.counters.decode_rtt_ms_sum += rtt_ms;
        }
        for (i, h) in hops.iter().enumerate().take(self.hop_counters.len()) {
            let c = &mut self.hop_counters[i];
            c.forwards += 1;
            c.compute_ms_sum += h.compute_ms as f64;
            if decode {
                c.decode_forwards += 1;
                c.decode_compute_ms_sum += h.compute_ms as f64;
            }
        }
        Ok((act, hops))
    }

    fn forward_inner(
        &mut self,
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
    ) -> Result<(Activations, Vec<HopStat>), RouteError> {
        let first = self
            .first
            .clone()
            .ok_or_else(|| RouteError::Net(NetError::Closed("not connected".into())))?;
        let route = self.route_id;
        let limit = forward_frame_limit(act.n_tokens, act.n_embd);
        let req = Request::Forward {
            route,
            session,
            seq,
            pos0,
            act,
            hops: Vec::new(),
        };
        let first_addr = self.hops[0].addr.clone();
        let body = match self.policy {
            ReturnPolicy::Reverse => {
                self.rt
                    .block_on(first.call::<_, Response>(&req, limit, FORWARD_TIMEOUT))?
            }
            ReturnPolicy::Direct | ReturnPolicy::Relay(_) => {
                let returns = self
                    .returns
                    .as_mut()
                    .ok_or_else(|| RouteError::Net(NetError::Closed("no return path".into())))?;
                let fut = async {
                    let (mut send, mut recv) = first.open_bi().await?;
                    write_frame(&mut send, &req).await?;
                    tokio::io::AsyncWriteExt::shutdown(&mut send)
                        .await
                        .map_err(NetError::from)?;
                    let mut ack_pending = true;
                    loop {
                        tokio::select! {
                            r = read_frame_max::<_, Response>(&mut recv, MAX_CONTROL_FRAME), if ack_pending => {
                                ack_pending = false;
                                match r {
                                    Ok(Response::Ack) => {}
                                    Ok(other) => return Ok(other),
                                    Err(e) => return Err(RouteError::Net(e.into())),
                                }
                            }
                            f = returns.recv() => {
                                match f {
                                    Some(frame) if frame.route == route && frame.session == session && frame.seq == seq => {
                                        return Ok(frame.body);
                                    }
                                    Some(frame) => {
                                        tracing::debug!(route = frame.route, session = frame.session, seq = frame.seq, "stale return frame");
                                    }
                                    None => return Err(RouteError::Net(NetError::Closed("return path closed".into()))),
                                }
                            }
                        }
                    }
                };
                match self
                    .rt
                    .block_on(async { tokio::time::timeout(FORWARD_TIMEOUT, fut).await })
                {
                    Ok(r) => r?,
                    Err(_) => return Err(RouteError::Net(NetError::Timeout(FORWARD_TIMEOUT))),
                }
            }
        };
        match body {
            Response::ForwardOk {
                session: rs,
                seq: rseq,
                act,
                hops,
            } => {
                if rs != session || rseq != seq {
                    return Err(RouteError::Protocol(format!(
                        "result for session {rs} seq {rseq}, expected {session} seq {seq}"
                    )));
                }
                Ok((act, hops))
            }
            Response::Error { code, message } => Err(stage_err(&first_addr, code, message)),
            other => Err(RouteError::Protocol(format!(
                "unexpected forward reply {}",
                other.kind()
            ))),
        }
    }

    fn control(&mut self, req: Request) -> Result<(), RouteError> {
        let first = self
            .first
            .clone()
            .ok_or_else(|| RouteError::Net(NetError::Closed("not connected".into())))?;
        let r: Result<Response, RouteError> = self
            .rt
            .block_on(first.call(&req, MAX_CONTROL_FRAME, CONTROL_TIMEOUT))
            .map_err(RouteError::from);
        let r = self.drop_on_net_error(r);
        match r? {
            Response::Ack => Ok(()),
            Response::Error { code, message } => Err(stage_err(&self.hops[0].addr, code, message)),
            other => Err(RouteError::Protocol(format!(
                "unexpected reply {}",
                other.kind()
            ))),
        }
    }

    pub fn seq_rm(&mut self, session: u64, p0: i32, p1: i32) -> Result<(), RouteError> {
        let route = self.route_id;
        self.control(Request::SeqRm {
            route,
            session,
            p0,
            p1,
        })
    }

    pub fn close_session(&mut self, session: u64) -> Result<(), RouteError> {
        let route = self.route_id;
        self.control(Request::SessionClose { route, session })
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        if let Some(r) = self.reader.take() {
            r.abort();
        }
    }
}
