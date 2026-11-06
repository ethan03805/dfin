//! `dfin client`: the client daemon.
//!
//! Runs the head (`0:a`) and tail (`b:end`) of the model locally, drives the
//! remote stages in between over dfin-proto, and serves an OpenAI-compatible API.
//!
//! Layout:
//! * [`engine`]: the dedicated engine thread (llama contexts, stage connections,
//!   sessions with prefix caching, the generation loop).
//! * [`http`]: axum handlers that submit jobs to the engine and relay its events.
//! * [`stage_conn`]: blocking framing and the per-stage connection.
//! * [`sampling`], [`text`]: sampler, UTF-8 accumulation and stop strings.

mod engine;
mod http;
mod sampling;
mod stage_conn;
mod text;

use crate::config::{LayerRange, RouteConfig, StageEndpoint};
use anyhow::Context as _;
use clap::Args;
use dfin_llama::{FlashAttn, KvType};
use dfin_proto::ActDtype;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct ClientArgs {
    /// GGUF model file. Optional with `--route` (which names the model itself).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Layers run locally before the first stage, e.g. `0:2`.
    #[arg(long, value_parser = parse_range)]
    pub head: Option<LayerRange>,
    /// Layers run locally after the last stage, e.g. `6:end`.
    #[arg(long, value_parser = parse_range)]
    pub tail: Option<LayerRange>,
    /// A remote stage `host:port=a:b`; repeat in pipeline order.
    #[arg(long = "stage", value_parser = parse_stage)]
    pub stages: Vec<StageEndpoint>,
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
    let (addr, layers) = s
        .rsplit_once('=')
        .ok_or_else(|| format!("stage {s:?} must look like host:port=a:b"))?;
    let layers = parse_range(layers)?;
    if addr.is_empty() {
        return Err(format!("stage {s:?} has an empty address"));
    }
    Ok(StageEndpoint {
        addr: addr.to_string(),
        layers,
    })
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

pub async fn run(args: ClientArgs) -> anyhow::Result<()> {
    let (model_path, head, tail, stages) = match &args.route {
        Some(path) => {
            let cfg = RouteConfig::from_toml_file(path)?;
            let stages = cfg.stages();
            (args.model.clone().unwrap_or(cfg.model), cfg.head, cfg.tail, stages)
        }
        None => (
            args.model.clone().context("--model is required (or use --route)")?,
            args.head.context("--head is required (or use --route)")?,
            args.tail.context("--tail is required (or use --route)")?,
            args.stages.clone(),
        ),
    };
    let model_name = args.model_name.clone().unwrap_or_else(|| {
        model_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dfin".into())
    });

    let cfg = engine::EngineConfig {
        model_path,
        head,
        tail,
        stages,
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
        let s = parse_stage("127.0.0.1:7001=2:4").unwrap();
        assert_eq!(s.addr, "127.0.0.1:7001");
        assert_eq!(s.layers, LayerRange::new(2, 4));
        assert!(parse_stage("nope").is_err());
    }

    #[test]
    fn client_args_parse() {
        use clap::Parser;
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            a: ClientArgs,
        }
        let t = T::try_parse_from([
            "t",
            "--model",
            "m.gguf",
            "--head",
            "0:2",
            "--tail",
            "6:end",
            "--stage",
            "127.0.0.1:1=2:4",
            "--stage",
            "127.0.0.1:2=4:6",
            "--n-gpu-layers",
            "-1",
            "--act-dtype",
            "f32",
            "--flash-attn",
            "off",
        ])
        .unwrap();
        assert_eq!(t.a.stages.len(), 2);
        assert_eq!(t.a.n_gpu_layers, -1);
        assert_eq!(t.a.tail, Some(LayerRange::new(6, -1)));
    }
}
