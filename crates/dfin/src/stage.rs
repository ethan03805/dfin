//! `dfin stage`: serve one contiguous middle layer range `[a, b)` of a GGUF as one hop
//! of a route chain, over authenticated Tor/TLS or explicit direct QUIC.
//!
//! Protocol: `docs/phase1-protocol.md`. The first stream on a connection must carry
//! `Hello`. A `RouteSetup` opens this node's sealed record, dials the next hop if there
//! is one and forwards the rest of the records. A `Forward` runs this stage's layers
//! and hands the output down the chain; what happens at the last hop depends on the
//! route's return mode. `SeqRm`, `SessionClose` and `RouteClose` run here, then on
//! every later hop, and are acknowledged once the whole chain has done them.

mod engine;
#[cfg(test)]
mod privacy_tests;

use crate::config::{parse_node_addr, LayerRange};
use crate::shards::{resolve_model, Manifest};
use anyhow::{bail, Context as _};
use clap::{Args, ValueEnum};
use dfin_llama::{Context, ContextParams, FlashAttn, KvType, Model, ModelParams};
use dfin_net::directory::sign_record;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::tunnel::{pump, tunnel_socket};
use dfin_net::RecvStream;
use dfin_net::{unseal, Connection, Endpoint, NetError, NodeId, NodeKey};
use dfin_proto::{
    forward_frame_limit, read_frame_max, write_frame, ActDtype, Activations, ErrorCode, HopRecord,
    HopStat, NextHop, NodeRecord, Request, Response, ReturnFrame, ReturnMode, PROTOCOL_VERSION,
};
use engine::{Engine, Job, Reply};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// Time allowed for a forward to travel the rest of the chain and come back.
pub const FORWARD_TIMEOUT: Duration = Duration::from_secs(600);
/// Time allowed to receive the body of a large forward frame.
const FORWARD_READ_TIMEOUT: Duration = Duration::from_secs(120);
/// Time allowed for route setup down the chain.
const SETUP_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FlashAttnArg {
    Auto,
    On,
    Off,
}

impl From<FlashAttnArg> for FlashAttn {
    fn from(v: FlashAttnArg) -> Self {
        match v {
            FlashAttnArg::Auto => FlashAttn::Auto,
            FlashAttnArg::On => FlashAttn::On,
            FlashAttnArg::Off => FlashAttn::Off,
        }
    }
}

#[derive(Args, Debug)]
pub struct StageArgs {
    /// The model: a GGUF file, the first file of a shard set, or a `*.manifest.json`.
    /// Only this stage's layers are loaded either way.
    #[arg(long)]
    pub model: PathBuf,
    /// With a manifest: download the shards this stage's layers need before loading,
    /// from nodes in the hub's directory or from `--seed` URLs.
    #[arg(long)]
    pub fetch: bool,
    /// HTTP URL prefix serving the shard files (repeatable; the manifest's seeds are
    /// tried too).
    #[arg(long = "seed")]
    pub seeds: Vec<String>,
    /// Seconds `--fetch` keeps looking for a source.
    #[arg(long, default_value_t = 120)]
    pub fetch_wait: u64,
    /// Half-open layer range `a:b` this stage serves. Must not include layer 0's
    /// embedding or the LM head (those run on the client).
    #[arg(long)]
    pub layers: LayerRange,
    /// Local listener (TCP in Tor mode, UDP in direct mode). Defaults to
    /// `127.0.0.1:7001`, or no listener with direct-mode `--cloak`.
    #[arg(long)]
    pub listen: Option<String>,
    /// Direct mode only: be reachable through this relay (`host:port/<node id>`). The stage dials
    /// the relay, which allocates a public port and tunnels its traffic here. No port
    /// is opened on this machine and no dialer's address is ever seen. Repeatable.
    #[arg(long = "cloak", value_parser = parse_node_addr)]
    pub cloaks: Vec<NextHop>,
    /// Node key file (64 hex characters). Created if missing. Its public key is this
    /// node's id, which clients pin.
    #[arg(long, default_value = "dfin-stage.key")]
    pub identity: PathBuf,
    /// Total KV capacity in tokens, shared by all session slots.
    #[arg(long, default_value_t = 8192)]
    pub n_ctx: u32,
    /// Concurrent session slots; each gets `n_ctx / n_seq_max` tokens.
    #[arg(long, default_value_t = 8)]
    pub n_seq_max: u32,
    /// Routes one connection may hold open.
    #[arg(long, default_value_t = 64)]
    pub max_routes: usize,
    /// Layers to offload to the GPU (`-1` = all of this stage's layers).
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    pub n_gpu_layers: i32,
    /// Threads for single-token decode.
    #[arg(long)]
    pub threads: Option<i32>,
    /// Threads for batch (prefill) decode.
    #[arg(long)]
    pub threads_batch: Option<i32>,
    /// Element type of the activations this stage sends on.
    #[arg(long, default_value = "f16")]
    pub act_dtype: ActDtype,
    /// KV cache element type (K and V).
    #[arg(long, default_value = "f16")]
    pub kv_type: KvType,
    #[arg(long, value_enum, default_value_t = FlashAttnArg::Auto)]
    pub flash_attn: FlashAttnArg,
    /// Regex of tensor names to keep in system RAM (e.g. `exps`). Repeatable.
    #[arg(long = "cpu-tensors")]
    pub cpu_tensors: Vec<String>,
    /// Read the weights into memory at load instead of memory-mapping the file.
    /// Only safe when the machine has memory to spare beyond the piece.
    #[arg(long)]
    pub no_mmap: bool,
    /// Lock the mapped weights in RAM at load so the first request is not slowed by page
    /// faults. Needs free memory for the whole piece. Recommended on a serving node.
    #[arg(long)]
    pub mlock: bool,
    /// Max tokens per forward request.
    #[arg(long, default_value_t = 2048)]
    pub n_batch: u32,
    /// Physical micro-batch size.
    #[arg(long, default_value_t = 512)]
    pub n_ubatch: u32,
    /// A hub (`host:port/<node id>`) to announce this stage to. Repeatable.
    #[arg(long = "hub", value_parser = parse_node_addr)]
    pub hubs: Vec<NextHop>,
    /// The address other nodes should dial (`host:port`). Required with `--hub` when
    /// `--listen` is an unspecified address such as `0.0.0.0`.
    #[arg(long)]
    pub advertise: Option<String>,
    /// Region claim published in the record (free text, verified by nobody).
    #[arg(long, default_value = "")]
    pub region: String,
    /// Seconds between announcements; records expire after three times this.
    #[arg(long, default_value_t = 60)]
    pub announce_every: u64,
}

/// What every connection needs to know about this stage.
struct StageInfo {
    fingerprint: String,
    layer_begin: i32,
    layer_end: i32,
    n_embd: u32,
    n_embd_inp: u32,
    n_embd_out: u32,
    n_ctx_seq: u32,
    n_seq_max: u32,
    n_batch: u32,
    backend: String,
    node_id: NodeId,
}

/// Summarise the devices this stage computes on, e.g. `"Vulkan0 (AMD Radeon RX 7800 XT), CPU"`.
pub fn backend_summary(n_gpu_layers: i32) -> String {
    let mut parts = Vec::new();
    for d in dfin_llama::devices() {
        if d.is_gpu {
            if n_gpu_layers != 0 {
                parts.push(format!("{} ({})", d.name, d.description));
            }
        } else {
            parts.push(d.name.clone());
        }
    }
    if parts.is_empty() {
        "CPU".into()
    } else {
        parts.join(", ")
    }
}

pub fn log_devices() {
    for d in dfin_llama::devices() {
        info!(
            "device {}: {} ({}, {} / {} MiB free)",
            d.name,
            d.description,
            if d.is_gpu { "gpu" } else { "cpu/accel" },
            d.memory_free >> 20,
            d.memory_total >> 20
        );
    }
}

pub async fn run(args: StageArgs, network: crate::network::Network) -> anyhow::Result<()> {
    if let Some(listen) = &args.listen {
        network.listener(listen)?;
    }
    for hub in &args.hubs {
        network.target(&hub.addr)?;
    }
    if network.private() {
        anyhow::ensure!(
            args.cloaks.is_empty(),
            "UDP relay cloaking is unavailable in Tor mode; publish an onion service instead"
        );
        anyhow::ensure!(
            args.region.is_empty(),
            "region claims are disabled in strict anonymity mode"
        );
        anyhow::ensure!(
            args.seeds.is_empty(),
            "HTTP seeds are disabled in strict anonymity mode"
        );
        if !args.hubs.is_empty() {
            network.target(
                args.advertise
                    .as_deref()
                    .context("Tor announcements require --advertise <v3-onion>:<port>")?,
            )?;
        } else if let Some(advertise) = &args.advertise {
            network.target(advertise)?;
        }
    }
    dfin_llama::init();
    if args.layers.begin == 0 || args.layers.end == -1 {
        bail!(
            "stage range {} is not a middle piece: stages never hold the embedding or the LM head; the client runs those",
            args.layers
        );
    }
    if args.n_seq_max == 0 {
        bail!("--n-seq-max must be at least 1");
    }
    let key = NodeKey::load_or_create(&args.identity)
        .with_context(|| format!("node key {}", args.identity.display()))?;
    info!(
        "node id {} (key file {})",
        key.id(),
        args.identity.display()
    );

    let resolved = {
        let p = args.model.clone();
        tokio::task::spawn_blocking(move || resolve_model(&p)).await??
    };
    if args.fetch {
        let Some(manifest) = &resolved.manifest else {
            bail!("--fetch needs a sharded model (a *.manifest.json or its first split)");
        };
        if args.hubs.is_empty() && args.seeds.is_empty() && manifest.seeds.is_empty() {
            bail!("--fetch needs a --hub or a --seed to fetch from");
        }
        let ep = network
            .client(NodeKey::generate())
            .context("creating fetch endpoint")?;
        let n = crate::fetch::fetch_missing(
            &ep,
            manifest,
            &resolved.dir,
            args.layers.begin,
            args.layers.end,
            args.hubs.first(),
            &args.seeds,
            Duration::from_secs(args.fetch_wait),
        )
        .await
        .context("fetching shards")?;
        info!("fetched {n} shard(s)");
    }
    let model_path = resolved.gguf.clone();
    let fingerprint = resolved.fingerprint.clone();

    let mparams = ModelParams {
        n_gpu_layers: args.n_gpu_layers,
        layer_begin: args.layers.begin,
        layer_end: args.layers.end,
        cpu_tensor_patterns: args.cpu_tensors.clone(),
        use_mmap: !args.no_mmap,
        use_mlock: args.mlock,
        ..ModelParams::default()
    };
    log_devices();
    info!("loading layers {} of {}", args.layers, model_path.display());
    let model = {
        let p = model_path.clone();
        tokio::task::spawn_blocking(move || Model::load(&p, &mparams))
            .await?
            .with_context(|| format!("loading {}", model_path.display()))?
    };
    let model = Arc::new(model);
    if model.has_head() || model.has_tail() {
        bail!(
            "resolved range {:?} of {} layers includes the embedding or the LM head: stages never hold the embedding or the LM head; the client runs those",
            model.layer_range(),
            model.n_layer()
        );
    }
    let (layer_begin, layer_end) = model.layer_range();
    info!(
        "model: {} | {:.1} MiB resident | layers {}:{} of {} | n_embd {} | state in {} out {}",
        model.desc(),
        model.size_bytes() as f64 / (1024.0 * 1024.0),
        layer_begin,
        layer_end,
        model.n_layer(),
        model.n_embd(),
        model.n_embd_inp(),
        model.n_embd_out()
    );

    let mut cparams = ContextParams {
        n_ctx: args.n_ctx,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_seq_max: args.n_seq_max,
        embeddings: true,
        kv_unified: false,
        flash_attn: args.flash_attn.into(),
        type_k: args.kv_type,
        type_v: args.kv_type,
        ..ContextParams::default()
    };
    if let Some(t) = args.threads {
        cparams.n_threads = t;
        cparams.n_threads_batch = t;
    }
    if let Some(t) = args.threads_batch {
        cparams.n_threads_batch = t;
    }
    let ctx = Context::new(model.clone(), &cparams).context("creating llama context")?;
    let n_seq_max = ctx.n_seq_max().max(1);
    let n_ctx_seq = ctx.n_ctx() / n_seq_max;
    info!(
        "context: n_ctx {} total, {} slots, {} tokens per session, threads {}/{}",
        ctx.n_ctx(),
        n_seq_max,
        n_ctx_seq,
        cparams.n_threads,
        cparams.n_threads_batch
    );

    let info = Arc::new(StageInfo {
        fingerprint,
        layer_begin,
        layer_end,
        n_embd: model.n_embd() as u32,
        n_embd_inp: model.n_embd_inp() as u32,
        n_embd_out: model.n_embd_out() as u32,
        n_ctx_seq,
        n_seq_max,
        n_batch: args.n_batch,
        backend: if network.private() {
            String::new()
        } else {
            backend_summary(args.n_gpu_layers)
        },
        node_id: key.id(),
    });

    let (jobs, _engine_thread) =
        Engine::new(ctx, args.act_dtype, args.n_batch as usize).spawn(256)?;

    let listen_text = match (&args.listen, args.cloaks.is_empty()) {
        (Some(l), _) => Some(l.clone()),
        (None, true) => Some("127.0.0.1:7001".to_string()),
        (None, false) => None,
    };
    let (endpoint, listening) = match &listen_text {
        Some(l) => {
            let ep = network
                .server(key, l)
                .await
                .with_context(|| format!("listening on {l}"))?;
            (ep, true)
        }
        None => (network.client(key).context("creating endpoint")?, false),
    };
    let addr = endpoint.local_addr()?;
    if listening {
        info!(
            "stage ready: layers {}:{} on {} ({}) node {}",
            layer_begin, layer_end, addr, info.backend, info.node_id
        );
    } else {
        info!(
            "stage ready: layers {}:{} ({}) node {}, no public listener; reachable through {} relay(s)",
            layer_begin, layer_end, info.backend, info.node_id, args.cloaks.len()
        );
    }

    let stage = Arc::new(Stage {
        info,
        jobs,
        endpoint: endpoint.clone(),
        clients: Mutex::new(HashMap::new()),
        next_handle: AtomicU64::new(1),
        max_routes: args.max_routes.max(1),
        cloak_addrs: Mutex::new(Vec::new()),
        shards: resolved.manifest.clone().map(|m| (m, resolved.dir.clone())),
    });

    for relay in &args.cloaks {
        tokio::spawn(cloak_loop(stage.clone(), relay.clone()));
    }

    if !args.hubs.is_empty() {
        let advertise = match (&args.advertise, listening) {
            (Some(a), _) => Some(a.clone()),
            (None, true) if !addr.ip().is_unspecified() => Some(addr.to_string()),
            (None, true) if args.cloaks.is_empty() => bail!(
                "--hub needs --advertise host:port when --listen is {} (other nodes cannot dial an unspecified address)",
                listen_text.as_deref().unwrap_or("unspecified")
            ),
            _ => None,
        };
        let template = NodeRecord {
            node_id: *stage.info.node_id.as_bytes(),
            addrs: Vec::new(),
            model_fingerprint: stage.info.fingerprint.clone(),
            model_name: match &resolved.manifest {
                Some(m) => m.model_name.clone(),
                None => args
                    .model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            },
            layer_begin,
            layer_end,
            n_layer: model.n_layer(),
            n_embd: stage.info.n_embd,
            n_ctx_seq: stage.info.n_ctx_seq,
            n_seq_max: stage.info.n_seq_max,
            act_dtype: args.act_dtype,
            backend: stage.info.backend.clone(),
            protocol: PROTOCOL_VERSION,
            region: args.region.clone(),
            attested: false,
            cloaked: network.private() || !args.cloaks.is_empty(),
            issued_at: 0,
            ttl_secs: (args.announce_every.max(5) * 3).min(dfin_proto::MAX_RECORD_TTL_SECS as u64)
                as u32,
        };
        tokio::spawn(announce_loop(
            stage.clone(),
            args.hubs.clone(),
            template,
            advertise,
            Duration::from_secs(args.announce_every.max(5)),
        ));
    }

    let public = async {
        if listening {
            accept_loop(stage.clone(), endpoint.clone()).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = public => Ok(()),
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: stage exiting");
            endpoint.close();
            Ok(())
        }
    }
}

/// Serve every connection that arrives on `endpoint`, until it closes.
async fn accept_loop(stage: Arc<Stage>, endpoint: Endpoint) {
    while let Some(incoming) = endpoint.accept().await {
        let conn = match incoming {
            Ok(c) => c,
            Err(e) => {
                debug!("handshake failed: {e}");
                continue;
            }
        };
        let stage = stage.clone();
        tokio::spawn(async move {
            let peer = conn.remote_address();
            let id = conn.peer_id();
            debug!(?peer, peer_id = %id.short(), "connection opened");
            stage.serve_conn(conn).await;
            debug!(?peer, peer_id = %id.short(), "connection closed");
        });
    }
}

/// Stay reachable through `relay`: register, run a listener over the tunnel, and
/// re-register after any failure.
async fn cloak_loop(stage: Arc<Stage>, relay: NextHop) {
    loop {
        match cloak_once(&stage, &relay).await {
            Ok(()) => info!(relay = %relay.addr, "relay session ended; reconnecting"),
            Err(e) => warn!(relay = %relay.addr, "cloak: {e:#}"),
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn cloak_once(stage: &Arc<Stage>, relay: &NextHop) -> anyhow::Result<()> {
    let conn = stage
        .endpoint
        .connect(&relay.addr, NodeId(relay.node_id))
        .await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    write_frame(&mut send, &Request::RelayListen).await?;
    let resp: Response = tokio::time::timeout(
        CONTROL_TIMEOUT,
        read_frame_max(&mut recv, MAX_CONTROL_FRAME),
    )
    .await
    .context("waiting for the relay")??;
    let public = match resp {
        Response::RelayListening { addr } => addr,
        Response::Error { code, message } => bail!("relay refused ({code:?}): {message}"),
        other => bail!("unexpected relay reply {}", other.kind()),
    };
    let (sock, inbound, outbound) = tunnel_socket();
    let ep = Endpoint::over_tunnel(stage.endpoint.key().clone(), sock)
        .context("endpoint over the tunnel")?;
    stage.cloak_addrs.lock().push(public.clone());
    info!(relay = %relay.addr, public = %public, "reachable through the relay");
    let result = tokio::select! {
        e = pump(send, recv, inbound, outbound) => Err(anyhow::anyhow!("tunnel: {e}")),
        _ = accept_loop(stage.clone(), ep.clone()) => Ok(()),
    };
    ep.close();
    stage.cloak_addrs.lock().retain(|a| a != &public);
    conn.close("tunnel ended");
    result
}

fn error(code: ErrorCode, message: impl Into<String>) -> Response {
    Response::Error {
        code,
        message: message.into(),
    }
}

/// Announce a fresh signed record to every hub, forever. The addresses are the
/// advertised one plus whatever relays currently host this stage.
async fn announce_loop(
    stage: Arc<Stage>,
    hubs: Vec<NextHop>,
    template: NodeRecord,
    advertise: Option<String>,
    every: Duration,
) {
    let endpoint = stage.endpoint.clone();
    let mut conns: HashMap<NodeId, Connection> = HashMap::new();
    let mut announced: HashMap<NodeId, bool> = HashMap::new();
    loop {
        let mut record = template.clone();
        record.issued_at = crate::hub::unix_now();
        record.addrs = advertise.iter().cloned().collect();
        record
            .addrs
            .extend(stage.cloak_addrs.lock().iter().cloned());
        if record.addrs.is_empty() {
            debug!("no address to announce yet");
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }
        let signed = match sign_record(endpoint.key(), &record) {
            Ok(s) => s,
            Err(e) => {
                warn!("cannot sign node record: {e}");
                return;
            }
        };
        for hub in &hubs {
            let id = NodeId(hub.node_id);
            let conn = match conns.get(&id).filter(|c| !c.is_closed()) {
                Some(c) => c.clone(),
                None => match endpoint.connect(&hub.addr, id).await {
                    Ok(c) => {
                        conns.insert(id, c.clone());
                        c
                    }
                    Err(e) => {
                        warn!(hub = %hub.addr, "cannot reach hub: {e}");
                        continue;
                    }
                },
            };
            let r: Result<Response, NetError> = conn
                .call(
                    &Request::Announce {
                        record: signed.clone(),
                    },
                    MAX_CONTROL_FRAME,
                    Duration::from_secs(if endpoint.is_private() { 90 } else { 20 }),
                )
                .await;
            match r {
                Ok(Response::Ack) => {
                    if !announced.get(&id).copied().unwrap_or(false) {
                        info!(hub = %hub.addr, addr = %record.addrs[0], "listed in the directory");
                        announced.insert(id, true);
                    } else {
                        debug!(hub = %hub.addr, "announcement refreshed");
                    }
                }
                Ok(Response::Error { code, message }) => {
                    warn!(hub = %hub.addr, ?code, "announcement refused: {message}");
                    announced.insert(id, false);
                }
                Ok(other) => {
                    warn!(hub = %hub.addr, reply = other.kind(), "unexpected announce reply")
                }
                Err(e) => {
                    warn!(hub = %hub.addr, "announce failed: {e}");
                    conns.remove(&id);
                    announced.insert(id, false);
                }
            }
        }
        tokio::time::sleep(every).await;
    }
}

/// Check a `Hello` against this stage. `Ok` carries `HelloOk`, `Err` the refusal.
/// A range of `-1:-1` means the dialer is a previous hop that does not know our
/// range; it will be checked against the sealed route record instead.
fn check_hello(req: &Request, info: &StageInfo) -> Result<Response, Response> {
    let Request::Hello {
        version,
        model_fingerprint,
        layer_begin,
        layer_end,
    } = req
    else {
        return Err(error(ErrorCode::BadRequest, "first frame must be Hello"));
    };
    if *version != PROTOCOL_VERSION {
        return Err(error(
            ErrorCode::VersionMismatch,
            format!("protocol version {version}, stage speaks {PROTOCOL_VERSION}"),
        ));
    }
    if *model_fingerprint != info.fingerprint {
        return Err(error(
            ErrorCode::ModelMismatch,
            format!(
                "model fingerprint {model_fingerprint} does not match this stage's {}",
                info.fingerprint
            ),
        ));
    }
    let any = (*layer_begin, *layer_end) == (-1, -1);
    if !any && (*layer_begin, *layer_end) != (info.layer_begin, info.layer_end) {
        return Err(error(
            ErrorCode::LayerRangeMismatch,
            format!(
                "requested layers {layer_begin}:{layer_end}, this stage serves {}:{}",
                info.layer_begin, info.layer_end
            ),
        ));
    }
    Ok(Response::HelloOk {
        layer_begin: info.layer_begin,
        layer_end: info.layer_end,
        n_embd: info.n_embd,
        n_embd_inp: info.n_embd_inp,
        n_embd_out: info.n_embd_out,
        n_ctx: info.n_ctx_seq,
        n_seq_max: info.n_seq_max,
        backend: info.backend.clone(),
        node_id: *info.node_id.as_bytes(),
    })
}

/// One route as seen from this hop.
struct RouteState {
    /// Engine key for this route's sessions.
    handle: u64,
    next: Option<(NextHop, RouteConnection)>,
    ret: ReturnMode,
    /// Where results go in `Direct` and `Relay` modes.
    return_conn: Option<ReturnConnection>,
    cancelled: CancellationToken,
    /// Closing waits for cancelled handlers to stop queuing engine jobs.
    operations: tokio::sync::RwLock<()>,
}

impl RouteState {
    fn close_connections(&self) {
        if let Some((_, conn)) = &self.next {
            conn.close("route ended");
        }
        if let Some(ReturnConnection::Relay(conn)) = &self.return_conn {
            conn.close("route ended");
        }
    }
}

/// A downstream connection belongs to one route, including while it is being
/// established. Dropping a failed/cancelled setup closes it immediately.
struct RouteConnection(Connection);

impl std::ops::Deref for RouteConnection {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        &self.0
    }
}

impl Drop for RouteConnection {
    fn drop(&mut self) {
        self.0.close("route ended");
    }
}

enum ReturnConnection {
    // The client's pre-existing connection can carry other routes too.
    Direct(Connection),
    Relay(RouteConnection),
}

impl std::ops::Deref for ReturnConnection {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        match self {
            Self::Direct(c) => c,
            Self::Relay(c) => c,
        }
    }
}

struct ConnState {
    conn: Connection,
    hello: AtomicBool,
    /// None reserves the route number and capacity while setup is pending.
    routes: Mutex<HashMap<u64, Option<Arc<RouteState>>>>,
}

struct RouteReservation {
    state: Arc<ConnState>,
    route: u64,
    committed: bool,
}

impl Drop for RouteReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.state.routes.lock().remove(&self.route);
        }
    }
}

struct Stage {
    info: Arc<StageInfo>,
    jobs: mpsc::Sender<Job>,
    endpoint: Endpoint,
    /// Connections that completed `Hello`, by peer id, for direct returns.
    clients: Mutex<HashMap<NodeId, Connection>>,
    next_handle: AtomicU64,
    max_routes: usize,
    /// Public addresses relays currently forward to this stage.
    cloak_addrs: Mutex<Vec<String>>,
    /// The shard set this stage was loaded from, so it can serve shards it holds.
    shards: Option<(Manifest, PathBuf)>,
}

/// How a request frame failed to arrive.
enum ReadErr {
    Eof,
    Bad(String),
    Net(NetError),
}

impl Stage {
    fn max_forward_frame(&self) -> u32 {
        forward_frame_limit(
            self.info.n_batch,
            self.info.n_embd_inp.max(self.info.n_embd_out),
        )
    }

    async fn serve_conn(self: &Arc<Self>, conn: Connection) {
        let st = Arc::new(ConnState {
            conn: conn.clone(),
            hello: AtomicBool::new(false),
            routes: Mutex::new(HashMap::new()),
        });
        let mut requests = tokio::task::JoinSet::new();
        loop {
            let accepted = tokio::select! {
                _ = conn.closed() => break,
                _ = requests.join_next(), if !requests.is_empty() => continue,
                r = conn.accept_bi(), if requests.len() < 32 => r,
            };
            let (mut send, recv) = match accepted {
                Ok(x) => x,
                Err(e) => {
                    debug!("{e}");
                    break;
                }
            };
            let stage = self.clone();
            let st = st.clone();
            requests.spawn(async move {
                let resp = match stage.read_request(recv).await {
                    Ok(req) => stage.handle(&st, req).await,
                    Err(ReadErr::Eof) => return,
                    Err(ReadErr::Bad(m)) => error(ErrorCode::BadRequest, m),
                    Err(ReadErr::Net(e)) => {
                        debug!("request stream: {e}");
                        return;
                    }
                };
                if let Response::Error { code, message } = &resp {
                    debug!(?code, "{message}");
                }
                if let Err(e) = reply(&mut send, &resp).await {
                    debug!("reply: {e}");
                }
            });
        }
        conn.close("stage connection ended");
        // No handler may install a route or enqueue engine work after teardown.
        requests.shutdown().await;
        self.teardown(&st).await;
    }

    /// Read one request, allowing a large frame only for `Forward`.
    async fn read_request(&self, mut recv: RecvStream) -> Result<Request, ReadErr> {
        let head = async {
            let mut len = [0u8; 4];
            AsyncReadExt::read_exact(&mut recv, &mut len).await?;
            let mut tag = [0u8; 1];
            AsyncReadExt::read_exact(&mut recv, &mut tag).await?;
            Ok::<_, std::io::Error>((u32::from_le_bytes(len), tag[0]))
        };
        let (len, tag) = match tokio::time::timeout(CONTROL_TIMEOUT, head).await {
            Ok(Ok(x)) => x,
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(ReadErr::Eof)
            }
            Ok(Err(e)) => return Err(ReadErr::Net(e.into())),
            Err(_) => return Err(ReadErr::Bad("timed out reading the request header".into())),
        };
        // The tag is a postcard varint. A canonical one for this enum is a single byte
        // below 0x80; a continuation bit means a non-minimal encoding, which postcard
        // would accept but which makes the tag ambiguous. Refuse it outright.
        if tag & 0x80 != 0 {
            return Err(ReadErr::Bad("non-canonical request tag".into()));
        }
        let (max, budget) = if tag == Request::FORWARD_TAG {
            (self.max_forward_frame(), FORWARD_READ_TIMEOUT)
        } else {
            (MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
        };
        if len == 0 || len > max {
            return Err(ReadErr::Bad(format!(
                "frame of {len} bytes exceeds the {max} byte limit for this request"
            )));
        }
        let mut body = vec![0u8; len as usize];
        body[0] = tag;
        match tokio::time::timeout(budget, AsyncReadExt::read_exact(&mut recv, &mut body[1..]))
            .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(ReadErr::Net(e.into())),
            Err(_) => return Err(ReadErr::Bad("timed out reading the request body".into())),
        }
        postcard::from_bytes(&body).map_err(|e| ReadErr::Bad(format!("bad frame: {e}")))
    }

    async fn handle(&self, st: &Arc<ConnState>, req: Request) -> Response {
        match req {
            Request::Hello { .. } => {
                if st.hello.swap(true, Ordering::SeqCst) {
                    return error(ErrorCode::BadRequest, "duplicate Hello");
                }
                match check_hello(&req, &self.info) {
                    Ok(ok) => {
                        self.clients
                            .lock()
                            .insert(st.conn.peer_id(), st.conn.clone());
                        ok
                    }
                    Err(refusal) => {
                        st.hello.store(false, Ordering::SeqCst);
                        if let Response::Error { code, message } = &refusal {
                            warn!(?code, "refusing connection: {message}");
                        }
                        refusal
                    }
                }
            }
            Request::Ping => Response::Pong,
            Request::ShardGet {
                model_id,
                split,
                offset,
                len,
            } => match &self.shards {
                None => error(ErrorCode::NoSuchShard, "this node holds no shard set"),
                Some((manifest, dir)) => {
                    match crate::fetch::serve_shard(manifest, dir, &model_id, split, offset, len)
                        .await
                    {
                        Ok((total, data)) => Response::Shard { total, data },
                        Err(m) => error(ErrorCode::NoSuchShard, m),
                    }
                }
            },
            _ if !st.hello.load(Ordering::SeqCst) => {
                error(ErrorCode::BadRequest, "Hello must come first")
            }
            Request::RouteSetup { route, hop } => self.route_setup(st, route, hop).await,
            Request::Forward {
                route,
                session,
                seq,
                pos0,
                act,
                hops,
            } => self.forward(st, route, session, seq, pos0, act, hops).await,
            Request::SeqRm {
                route,
                session,
                p0,
                p1,
            } => {
                let Some(rs) = st.routes.lock().get(&route).and_then(Clone::clone) else {
                    return error(ErrorCode::UnknownRoute, format!("unknown route {route}"));
                };
                self.with_route(&rs, route, async {
                    let (tx, rx) = oneshot::channel();
                    let job = Job::SeqRm {
                        conn: rs.handle,
                        session,
                        p0,
                        p1,
                        reply: tx,
                    };
                    match self.run_job(job, rx).await {
                        Reply::Ok => {}
                        Reply::Error { code, message } => return error(code, message),
                        Reply::Forward { .. } => {
                            return error(ErrorCode::Internal, "engine replied oddly")
                        }
                    }
                    self.downstream_ack(
                        &rs,
                        &Request::SeqRm {
                            route,
                            session,
                            p0,
                            p1,
                        },
                    )
                    .await
                })
                .await
            }
            Request::SessionClose { route, session } => {
                let Some(rs) = st.routes.lock().get(&route).and_then(Clone::clone) else {
                    return error(ErrorCode::UnknownRoute, format!("unknown route {route}"));
                };
                self.with_route(&rs, route, async {
                    let (tx, rx) = oneshot::channel();
                    let job = Job::Close {
                        conn: rs.handle,
                        session,
                        reply: tx,
                    };
                    match self.run_job(job, rx).await {
                        Reply::Ok => {}
                        Reply::Error { code, message } => return error(code, message),
                        Reply::Forward { .. } => {
                            return error(ErrorCode::Internal, "engine replied oddly")
                        }
                    }
                    self.downstream_ack(&rs, &Request::SessionClose { route, session })
                        .await
                })
                .await
            }
            Request::RouteClose { route } => {
                let Some(rs) = st.routes.lock().get(&route).and_then(Clone::clone) else {
                    return error(
                        ErrorCode::UnknownRoute,
                        format!("unknown or pending route {route}"),
                    );
                };
                // Keep it in the table until cleanup completes: disconnecting
                // during this await must still leave teardown able to find it.
                let response = self.close_route(&rs, route).await;
                let mut routes = st.routes.lock();
                if routes
                    .get(&route)
                    .and_then(Option::as_ref)
                    .is_some_and(|current| Arc::ptr_eq(current, &rs))
                {
                    routes.remove(&route);
                }
                response
            }
            Request::RelayOpen { .. }
            | Request::RelayAttach { .. }
            | Request::RelayListen
            | Request::Announce { .. }
            | Request::DirectoryGet
            | Request::Probe { .. } => error(
                ErrorCode::BadRequest,
                "this node is a stage, not a relay or hub",
            ),
        }
    }

    async fn run_job(&self, job: Job, rx: oneshot::Receiver<Reply>) -> Reply {
        let stopped = || Reply::Error {
            code: ErrorCode::Internal,
            message: "engine stopped".into(),
        };
        if self.jobs.send(job).await.is_err() {
            return stopped();
        }
        rx.await.unwrap_or_else(|_| stopped())
    }

    async fn with_route(
        &self,
        rs: &RouteState,
        route: u64,
        operation: impl std::future::Future<Output = Response>,
    ) -> Response {
        let _guard = rs.operations.read().await;
        tokio::select! {
            biased;
            _ = rs.cancelled.cancelled() => error(ErrorCode::UnknownRoute, format!("route {route} closed")),
            response = operation => response,
        }
    }

    /// Dial a route-owned next hop and complete `Hello` with it. Sharing an
    /// outbound connection would merge independent clients' route namespaces.
    async fn outbound(&self, nh: &NextHop) -> Result<RouteConnection, Response> {
        let id = NodeId(nh.node_id);
        let conn = RouteConnection(
            self.endpoint
                .connect(&nh.addr, id)
                .await
                .map_err(|_| error(ErrorCode::Downstream, "next hop connection failed"))?,
        );
        let hello = Request::Hello {
            version: PROTOCOL_VERSION,
            model_fingerprint: self.info.fingerprint.clone(),
            layer_begin: -1,
            layer_end: -1,
        };
        let r: Response = conn
            .call(&hello, MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
            .await
            .map_err(|_| error(ErrorCode::Downstream, "next hop hello failed"))?;
        match r {
            Response::HelloOk {
                n_embd, n_embd_inp, ..
            } if n_embd == self.info.n_embd && n_embd_inp == self.info.n_embd_out => {}
            Response::HelloOk {
                n_embd, n_embd_inp, ..
            } => {
                return Err(error(
                    ErrorCode::Downstream,
                    format!(
                    "next hop takes rows of {n_embd_inp} (n_embd {n_embd}), this stage produces {}",
                    self.info.n_embd_out
                ),
                ))
            }
            Response::Error { code, .. } => {
                return Err(error(
                    ErrorCode::Downstream,
                    format!("next hop refused hello ({code:?})"),
                ))
            }
            other => {
                return Err(error(
                    ErrorCode::Downstream,
                    format!("next hop returned unexpected hello reply {}", other.kind()),
                ))
            }
        }
        Ok(conn)
    }

    async fn route_setup(&self, st: &Arc<ConnState>, route: u64, hop: Vec<u8>) -> Response {
        {
            let mut routes = st.routes.lock();
            if routes.contains_key(&route) {
                return error(
                    ErrorCode::BadRequest,
                    format!("route {route} already exists"),
                );
            }
            if routes.len() >= self.max_routes {
                return error(
                    ErrorCode::RouteLimit,
                    format!("{} routes already open on this connection", routes.len()),
                );
            }
            routes.insert(route, None);
        }
        let mut reservation = RouteReservation {
            state: st.clone(),
            route,
            committed: false,
        };
        let plain = match unseal(self.endpoint.key(), &hop) {
            Ok(p) => p,
            Err(e) => return error(ErrorCode::BadRecord, e.to_string()),
        };
        let rec: HopRecord = match postcard::from_bytes(&plain) {
            Ok(r) => r,
            Err(e) => return error(ErrorCode::BadRecord, format!("record does not parse: {e}")),
        };
        if rec.model_fingerprint != self.info.fingerprint {
            return error(ErrorCode::ModelMismatch, "route record names another model");
        }
        if (rec.layer_begin, rec.layer_end) != (self.info.layer_begin, self.info.layer_end) {
            return error(
                ErrorCode::LayerRangeMismatch,
                format!(
                    "route wants layers {}:{} from this hop, it serves {}:{}",
                    rec.layer_begin, rec.layer_end, self.info.layer_begin, self.info.layer_end
                ),
            );
        }
        let mut next = None;
        let mut hops = 0u32;
        if let Some(nh) = rec.next.clone() {
            let conn = match self.outbound(&nh).await {
                Ok(c) => c,
                Err(e) => return e,
            };
            let r: Result<Response, NetError> = conn
                .call(
                    &Request::RouteSetup {
                        route,
                        hop: rec.inner,
                    },
                    MAX_CONTROL_FRAME,
                    SETUP_TIMEOUT,
                )
                .await;
            match r {
                Ok(Response::RouteOk { hops: h }) => hops = h,
                Ok(Response::Error { code, .. }) => {
                    return error(
                        ErrorCode::Downstream,
                        format!("next hop refused setup ({code:?})"),
                    )
                }
                Ok(other) => {
                    return error(
                        ErrorCode::Downstream,
                        format!("next hop returned unexpected setup reply {}", other.kind()),
                    )
                }
                Err(_) => return error(ErrorCode::Downstream, "next hop setup failed"),
            }
            next = Some((nh, conn));
        }
        let mut return_conn = None;
        if next.is_none() {
            match &rec.ret {
                ReturnMode::Reverse => {}
                ReturnMode::Direct { client_id } => {
                    let id = NodeId(*client_id);
                    match self.clients.lock().get(&id) {
                        Some(c) if !c.is_closed() => {
                            return_conn = Some(ReturnConnection::Direct(c.clone()))
                        }
                        _ => {
                            return error(
                                ErrorCode::NoReturnPath,
                                "no live direct return connection",
                            )
                        }
                    }
                }
                ReturnMode::Relay { relay, token } => {
                    let conn = match self
                        .endpoint
                        .connect(&relay.addr, NodeId(relay.node_id))
                        .await
                    {
                        Ok(c) => RouteConnection(c),
                        Err(_) => {
                            return error(ErrorCode::NoReturnPath, "return relay connection failed")
                        }
                    };
                    let r: Result<Response, NetError> = conn
                        .call(
                            &Request::RelayAttach { token: *token },
                            MAX_CONTROL_FRAME,
                            CONTROL_TIMEOUT,
                        )
                        .await;
                    match r {
                        Ok(Response::Ack) => return_conn = Some(ReturnConnection::Relay(conn)),
                        Ok(Response::Error { code, .. }) => {
                            return error(
                                ErrorCode::NoReturnPath,
                                format!("return relay refused attachment ({code:?})"),
                            )
                        }
                        Ok(other) => {
                            return error(
                                ErrorCode::NoReturnPath,
                                format!("return relay sent unexpected reply {}", other.kind()),
                            )
                        }
                        Err(_) => {
                            return error(ErrorCode::NoReturnPath, "return relay attachment failed")
                        }
                    }
                }
            }
        }
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        let Some(hops) = hops.checked_add(1) else {
            return error(ErrorCode::Downstream, "invalid downstream hop count");
        };
        let rs = Arc::new(RouteState {
            handle,
            next,
            ret: rec.ret,
            return_conn,
            cancelled: CancellationToken::new(),
            operations: tokio::sync::RwLock::new(()),
        });
        let mut routes = st.routes.lock();
        routes.insert(route, Some(rs));
        reservation.committed = true;
        debug!(route, handle, hops, "route set up");
        Response::RouteOk { hops }
    }

    #[allow(clippy::too_many_arguments)]
    async fn forward(
        &self,
        st: &Arc<ConnState>,
        route: u64,
        session: u64,
        seq: u64,
        pos0: i32,
        act: Activations,
        mut hops: Vec<HopStat>,
    ) -> Response {
        let Some(rs) = st.routes.lock().get(&route).and_then(Clone::clone) else {
            return error(ErrorCode::UnknownRoute, format!("unknown route {route}"));
        };
        self.with_route(&rs, route, async {
            if hops.len() > 256 {
                return error(ErrorCode::BadRequest, "too many hop stats");
            }
            let (tx, rx) = oneshot::channel();
            let job = Job::Forward {
                conn: rs.handle,
                session,
                seq,
                pos0,
                act,
                reply: tx,
            };
            let (out, compute_ms) = match self.run_job(job, rx).await {
                Reply::Forward { act, compute_ms } => (act, compute_ms),
                Reply::Error { code, message } => {
                    // The client may be waiting on the return path rather than on this stream.
                    self.deliver_error(&rs, route, session, seq, code, &message)
                        .await;
                    return error(code, message);
                }
                Reply::Ok => return error(ErrorCode::Internal, "engine replied oddly"),
            };
            hops.push(HopStat {
                layer_begin: self.info.layer_begin,
                layer_end: self.info.layer_end,
                compute_ms,
            });
            match &rs.next {
                Some((_, conn)) => {
                    let limit = forward_frame_limit(out.n_tokens, out.n_embd);
                    let r: Result<Response, NetError> = conn
                        .call(
                            &Request::Forward {
                                route,
                                session,
                                seq,
                                pos0,
                                act: out,
                                hops,
                            },
                            limit,
                            FORWARD_TIMEOUT,
                        )
                        .await;
                    match r {
                        Ok(resp @ (Response::Ack | Response::ForwardOk { .. })) => resp,
                        Ok(Response::Error { code, .. }) => {
                            error(code, "downstream forward refused")
                        }
                        Ok(other) => error(
                            ErrorCode::Downstream,
                            format!("unexpected forward reply {}", other.kind()),
                        ),
                        Err(_) => error(ErrorCode::Downstream, "next hop forward failed"),
                    }
                }
                None => match &rs.ret {
                    ReturnMode::Reverse => Response::ForwardOk {
                        session,
                        seq,
                        act: out,
                        hops,
                    },
                    ReturnMode::Direct { .. } | ReturnMode::Relay { .. } => {
                        let frame = ReturnFrame {
                            route,
                            session,
                            seq,
                            body: Response::ForwardOk {
                                session,
                                seq,
                                act: out,
                                hops,
                            },
                        };
                        match &rs.return_conn {
                            Some(c) => match c.send_uni(&frame).await {
                                Ok(()) => Response::Ack,
                                Err(_) => error(ErrorCode::NoReturnPath, "return path send failed"),
                            },
                            None => error(ErrorCode::NoReturnPath, "no return connection"),
                        }
                    }
                },
            }
        })
        .await
    }

    /// In direct and relay modes the client listens on the return path; tell it there.
    async fn deliver_error(
        &self,
        rs: &RouteState,
        route: u64,
        session: u64,
        seq: u64,
        code: ErrorCode,
        message: &str,
    ) {
        if rs.next.is_some() || matches!(rs.ret, ReturnMode::Reverse) {
            return;
        }
        if let Some(c) = &rs.return_conn {
            let frame = ReturnFrame {
                route,
                session,
                seq,
                body: error(code, message),
            };
            let _ = c.send_uni(&frame).await;
        }
    }

    /// Forward a control request down the chain and wait for its `Ack`.
    async fn downstream_ack(&self, rs: &RouteState, req: &Request) -> Response {
        let Some((_, conn)) = &rs.next else {
            return Response::Ack;
        };
        let r: Result<Response, NetError> =
            conn.call(req, MAX_CONTROL_FRAME, CONTROL_TIMEOUT).await;
        match r {
            Ok(Response::Ack) => Response::Ack,
            Ok(Response::Error { code, .. }) => error(
                ErrorCode::Downstream,
                format!("downstream control request refused ({code:?})"),
            ),
            Ok(other) => error(
                ErrorCode::Downstream,
                format!("unexpected downstream control reply {}", other.kind()),
            ),
            Err(_) => error(ErrorCode::Downstream, "downstream control request failed"),
        }
    }

    async fn close_route(&self, rs: &RouteState, route: u64) -> Response {
        rs.cancelled.cancel();
        let _exclusive = rs.operations.write().await;
        let _ = self.jobs.send(Job::CloseConn { conn: rs.handle }).await;
        let response = self
            .downstream_ack(rs, &Request::RouteClose { route })
            .await;
        rs.close_connections();
        response
    }

    async fn teardown(&self, st: &Arc<ConnState>) {
        let routes: Vec<(u64, Arc<RouteState>)> = st
            .routes
            .lock()
            .drain()
            .filter_map(|(route, state)| state.map(|state| (route, state)))
            .collect();
        // All handlers were joined before this point. Closing route-owned
        // downstream connections cascades cancellation without network waits.
        for (_, rs) in &routes {
            rs.cancelled.cancel();
            rs.close_connections();
        }
        for (_, rs) in routes {
            let _ = self.jobs.send(Job::CloseConn { conn: rs.handle }).await;
        }
        let mut clients = self.clients.lock();
        if let Some(c) = clients.get(&st.conn.peer_id()) {
            if c.stable_id() == st.conn.stable_id() {
                clients.remove(&st.conn.peer_id());
            }
        }
    }
}
