//! `dfin client`: the client daemon.
//!
//! Runs the head (`0:a`) and tail (`b:end`) of the model locally, drives the
//! remote stages in between over dfin-proto, and serves an OpenAI-compatible API.
//!
//! Layout:
//! * [`engine`]: the dedicated engine thread (llama contexts, stage connections,
//!   sessions with prefix caching, the generation loop).
//! * [`http`]: axum handlers that submit jobs to the engine and relay its events.
//! * [`route`]: the chain of stages over Tor/TLS or explicit direct QUIC.
//! * [`sampling`], [`text`]: sampler, UTF-8 accumulation and stop strings.

pub mod directory;
mod engine;
mod http;
pub mod route;
mod sampling;
mod text;

use crate::config::{LayerRange, RouteConfig, StageEndpoint};
use anyhow::Context as _;
use clap::Args;
use dfin_llama::{FlashAttn, KvType};
use dfin_proto::ActDtype;
use parking_lot::Mutex;
pub use route::ReturnPolicy;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct ClientArgs {
    /// The model: a GGUF file, the first file of a shard set, or a `*.manifest.json`.
    /// Optional with `--route` (which names the model itself).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Layers run locally before the first stage, e.g. `0:2`.
    #[arg(long, value_parser = parse_range)]
    pub head: Option<LayerRange>,
    /// Layers run locally after the last stage, e.g. `6:end`.
    #[arg(long, value_parser = parse_range)]
    pub tail: Option<LayerRange>,
    /// A remote stage `host:port/<node id>=a:b`; repeat in pipeline order.
    #[arg(long = "stage", value_parser = parse_stage)]
    pub stages: Vec<StageEndpoint>,
    /// How results come back from the last stage: `direct` (the last stage sends to
    /// this client, fastest) or `reverse` (back along the chain; only the first stage
    /// learns this client's address).
    #[arg(long = "return", default_value = "direct", value_parser = parse_return)]
    pub return_mode: ReturnPolicy,
    /// Reach the first stage through this relay (`host:port/<node id>`) and receive
    /// results through it, so no stage learns this client's address. Overrides `--return`.
    #[arg(long, value_parser = parse_relay)]
    pub relay: Option<dfin_proto::NextHop>,
    /// A hub (`host:port/<node id>`) whose signed directory supplies the stages between
    /// `--head` and `--tail` when no `--stage` is given. The route is rebuilt from the
    /// directory when a stage drops.
    #[arg(long, value_parser = parse_hub)]
    pub hub: Option<dfin_proto::NextHop>,
    /// Where the last verified directory is kept, so routing survives a hub outage.
    #[arg(long, default_value = "dfin-directory.cache")]
    pub directory_cache: PathBuf,
    /// Route TOML (model, head, tail, stages) instead of the flags above.
    #[arg(long)]
    pub route: Option<PathBuf>,
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub listen: String,
    /// Total KV capacity of the head and tail contexts (split across sessions).
    #[arg(long, default_value_t = 8192)]
    pub n_ctx: u32,
    /// Layers of the head and tail to offload to the GPU (`-1` = all).
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub n_gpu_layers: i32,
    /// Regex of tensor names to keep in CPU memory when offloading, e.g. `exps`
    /// keeps MoE routed experts in RAM (repeatable; the `-ot exps=CPU` pattern).
    #[arg(long = "cpu-tensors")]
    pub cpu_tensors: Vec<String>,
    #[arg(long)]
    pub threads: Option<i32>,
    #[arg(long)]
    pub threads_batch: Option<i32>,
    /// Activation dtype on the wire.
    #[arg(long, default_value = "f16", value_parser = parse_act_dtype)]
    pub act_dtype: ActDtype,
    /// Concurrent cached sessions; keep at or below the stages' `--n-seq-max`.
    #[arg(long, default_value_t = 4)]
    pub max_sessions: u32,
    /// Prefill chunk size in tokens.
    #[arg(long, default_value_t = 512)]
    pub prefill_chunk: usize,
    /// Model id served by `/v1/models` (default: model file stem).
    #[arg(long)]
    pub model_name: Option<String>,
    /// Chat template name (`chatml`, `llama3`, ...) or a file with template source.
    #[arg(long)]
    pub chat_template: Option<String>,
    /// KV cache type for head and tail.
    #[arg(long, default_value = "f16", value_parser = parse_kv_type)]
    pub kv_type: KvType,
    #[arg(long, default_value = "auto", value_parser = parse_flash_attn)]
    pub flash_attn: FlashAttn,
    /// Jobs that may wait in the queue before requests get 503.
    #[arg(long, default_value_t = 64)]
    pub queue: usize,
}

fn parse_range(s: &str) -> Result<LayerRange, String> {
    s.parse::<LayerRange>().map_err(|e| format!("{e:#}"))
}

fn parse_stage(s: &str) -> Result<StageEndpoint, String> {
    s.parse::<StageEndpoint>().map_err(|e| format!("{e:#}"))
}

fn parse_return(s: &str) -> Result<ReturnPolicy, String> {
    s.parse()
}

fn parse_relay(s: &str) -> Result<dfin_proto::NextHop, String> {
    crate::config::parse_node_addr(s).map_err(|e| format!("relay {e}"))
}

fn parse_hub(s: &str) -> Result<dfin_proto::NextHop, String> {
    crate::config::parse_node_addr(s).map_err(|e| format!("hub {e}"))
}

fn parse_act_dtype(s: &str) -> Result<ActDtype, String> {
    s.parse()
}

fn parse_kv_type(s: &str) -> Result<KvType, String> {
    s.parse()
}

fn parse_flash_attn(s: &str) -> Result<FlashAttn, String> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Ok(FlashAttn::Auto),
        "on" | "true" | "1" => Ok(FlashAttn::On),
        "off" | "false" | "0" => Ok(FlashAttn::Off),
        other => Err(format!("unknown flash-attn mode {other:?} (auto, on, off)")),
    }
}

pub async fn run(args: ClientArgs, network: crate::network::Network) -> anyhow::Result<()> {
    network.listener(&args.listen)?;
    for target in args.hub.iter().chain(args.relay.iter()) {
        network.target(&target.addr)?;
    }
    let (model_path, head, tail, stages) = match &args.route {
        Some(path) => {
            let cfg = RouteConfig::from_toml_file(path)?;
            let stages = cfg.stages();
            (
                args.model.clone().unwrap_or(cfg.model),
                cfg.head,
                cfg.tail,
                stages,
            )
        }
        None => (
            args.model
                .clone()
                .context("--model is required (or use --route)")?,
            args.head.context("--head is required (or use --route)")?,
            args.tail.context("--tail is required (or use --route)")?,
            args.stages.clone(),
        ),
    };
    for stage in &stages {
        network.target(&stage.addr)?;
    }
    let resolved = crate::shards::resolve_model(&model_path)?;
    let model_name = args
        .model_name
        .clone()
        .unwrap_or_else(|| match &resolved.manifest {
            Some(m) => m.model_name.clone(),
            None => model_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "dfin".into()),
        });
    let model_path = resolved.gguf.clone();

    let cfg = engine::EngineConfig {
        network,
        model_path,
        fingerprint: resolved.fingerprint.clone(),
        head,
        tail,
        stages,
        return_policy: match &args.relay {
            Some(relay) => ReturnPolicy::Relay(relay.clone()),
            None => args.return_mode.clone(),
        },
        hub: args.hub.clone(),
        directory_cache: Some(args.directory_cache.clone()),
        n_ctx: args.n_ctx,
        n_gpu_layers: args.n_gpu_layers,
        cpu_tensors: args.cpu_tensors.clone(),
        threads: args.threads,
        threads_batch: args.threads_batch,
        act_dtype: args.act_dtype,
        max_sessions: args.max_sessions,
        prefill_chunk: args.prefill_chunk,
        chat_template: args.chat_template.clone(),
        kv_type: args.kv_type,
        flash_attn: args.flash_attn,
    };
    let stats = Arc::new(Mutex::new(engine::Stats::default()));
    let (jobs, info) = engine::spawn(cfg, stats.clone(), args.queue).await?;
    if info.return_mode == "relay" && info.route.iter().any(|r| r.kind == "stage") {
        tracing::warn!("the selected dfin relay can inspect model states; Tor hides network addresses, not those states from the relay");
    }
    tracing::info!(
        model = %model_name,
        route = %info.route.iter().map(|r| r.layers.clone()).collect::<Vec<_>>().join(" -> "),
        "engine ready"
    );

    let state = Arc::new(http::Shared {
        jobs,
        stats,
        info,
        model_name,
        created: http::unix_now(),
    });
    let app = http::router(state);
    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    let addr = listener.local_addr()?;
    eprintln!("listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stage_flag() {
        let id = dfin_net::NodeKey::generate().id();
        let s = parse_stage(&format!("127.0.0.1:7001/{id}=2:4")).unwrap();
        assert_eq!(s.addr, "127.0.0.1:7001");
        assert_eq!(s.node_id, id);
        assert_eq!(s.layers, LayerRange::new(2, 4));
        assert!(parse_stage("nope").is_err());
        assert!(parse_stage("127.0.0.1:7001=2:4").is_err());
    }

    #[test]
    fn client_args_parse() {
        use clap::Parser;
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            a: ClientArgs,
        }
        let id = dfin_net::NodeKey::generate().id();
        let t = T::try_parse_from([
            "t",
            "--model",
            "m.gguf",
            "--head",
            "0:2",
            "--tail",
            "6:end",
            "--stage",
            &format!("127.0.0.1:1/{id}=2:4"),
            "--stage",
            &format!("127.0.0.1:2/{id}=4:6"),
            "--return",
            "reverse",
            "--n-gpu-layers",
            "-1",
            "--act-dtype",
            "f32",
            "--flash-attn",
            "off",
        ])
        .unwrap();
        assert_eq!(t.a.stages.len(), 2);
        assert_eq!(t.a.return_mode, ReturnPolicy::Reverse);
        assert_eq!(t.a.n_gpu_layers, -1);
        assert_eq!(t.a.tail, Some(LayerRange::new(6, -1)));
    }
}
