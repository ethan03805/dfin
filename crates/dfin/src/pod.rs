//! `dfin pod`: launch a whole pipeline on this machine for testing.
//!
//! `--split 2,5` on an 8-layer model runs one `dfin stage` child for layers `2:5`
//! and a `dfin client` child with head `0:2` and tail `5:end`. `--split 2,4,6` runs
//! stages `2:4` and `4:6`. Stages listen on `127.0.0.1:(base_port + i)`.

use crate::config::LayerRange;
use anyhow::{bail, Context as _};
use clap::Args;
use dfin_llama::{Model, ModelParams};
use dfin_proto::ActDtype;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct PodArgs {
    /// GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Comma-separated layer boundaries, e.g. `2,5` (head 0:2, stage 2:5, tail 5:end).
    #[arg(long, value_delimiter = ',', required = true)]
    pub split: Vec<i32>,
    /// Address of the client's OpenAI-compatible API.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub listen: SocketAddr,
    /// Port of the first stage; stage i listens on `base_port + i`.
    #[arg(long, default_value_t = 7001)]
    pub base_port: u16,
    /// Total KV capacity per piece, shared by all sessions.
    #[arg(long, default_value_t = 8192)]
    pub n_ctx: u32,
    /// Concurrent sessions (stage slots and client `--max-sessions`).
    #[arg(long, default_value_t = 4)]
    pub n_seq_max: u32,
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    pub n_gpu_layers: i32,
    /// Regex of tensor names kept in CPU memory on every piece, e.g. `exps` (repeatable).
    #[arg(long = "cpu-tensors")]
    pub cpu_tensors: Vec<String>,
    #[arg(long)]
    pub threads: Option<i32>,
    #[arg(long, default_value = "f16")]
    pub act_dtype: ActDtype,
}

/// Everything `spawn_pod` needs.
#[derive(Clone, Debug)]
pub struct PodOptions {
    pub model: PathBuf,
    pub split: Vec<i32>,
    pub listen: SocketAddr,
    pub base_port: u16,
    pub n_ctx: u32,
    pub n_seq_max: u32,
    pub n_gpu_layers: i32,
    pub cpu_tensors: Vec<String>,
    pub threads: Option<i32>,
    pub act_dtype: ActDtype,
    /// The `dfin` binary to launch; defaults to `std::env::current_exe()`.
    pub exe: Option<PathBuf>,
    /// `--log` filter passed to the children (otherwise they inherit `DFIN_LOG`).
    pub log: Option<String>,
    /// How long to wait for each stage port and for the client's `/health`.
    pub ready_timeout: Duration,
}

impl PodOptions {
    pub fn from_args(a: &PodArgs) -> Self {
        PodOptions {
            model: a.model.clone(),
            split: a.split.clone(),
            listen: a.listen,
            base_port: a.base_port,
            n_ctx: a.n_ctx,
            n_seq_max: a.n_seq_max,
            n_gpu_layers: a.n_gpu_layers,
            cpu_tensors: a.cpu_tensors.clone(),
            threads: a.threads,
            act_dtype: a.act_dtype,
            exe: None,
            log: None,
            ready_timeout: Duration::from_secs(120),
        }
    }
}

/// The layer pieces of a split: local head, remote middles, local tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitPlan {
    pub head: LayerRange,
    pub stages: Vec<LayerRange>,
    /// `end == n_layer` (concrete, not `-1`).
    pub tail: LayerRange,
}

/// Validate split points (strictly increasing, each in `1..=n_layer-1`, at least two
/// so there is at least one remote stage) and turn them into pieces.
pub fn plan_split(split: &[i32], n_layer: i32) -> anyhow::Result<SplitPlan> {
    if split.len() < 2 {
        bail!("--split needs at least two boundaries (e.g. `2,5`) so that there is at least one stage");
    }
    for w in split.windows(2) {
        if w[1] <= w[0] {
            bail!("--split boundaries must be strictly increasing, got {split:?}");
        }
    }
    for &s in split {
        if s < 1 || s > n_layer - 1 {
            bail!(
                "--split boundary {s} is outside 1..={} (model has {n_layer} layers)",
                n_layer - 1
            );
        }
    }
    Ok(SplitPlan {
        head: LayerRange::new(0, split[0]),
        stages: split
            .windows(2)
            .map(|w| LayerRange::new(w[0], w[1]))
            .collect(),
        tail: LayerRange::new(split[split.len() - 1], n_layer),
    })
}

/// Read `n_layer` from the GGUF without loading weights.
pub fn model_n_layer(path: &Path) -> anyhow::Result<i32> {
    let m = Model::load(
        path,
        &ModelParams {
            vocab_only: true,
            ..ModelParams::default()
        },
    )
    .with_context(|| format!("reading hparams of {}", path.display()))?;
    if m.n_layer() > 0 {
        return Ok(m.n_layer());
    }
    // A vocab-only load leaves `n_layer` at 0; read the GGUF key instead.
    let arch = m
        .meta_str("general.architecture")
        .context("GGUF has no general.architecture")?;
    let n = m
        .meta_str(&format!("{arch}.block_count"))
        .with_context(|| format!("GGUF has no {arch}.block_count"))?;
    n.trim()
        .parse()
        .with_context(|| format!("bad {arch}.block_count {n:?}"))
}

pub fn act_dtype_str(d: ActDtype) -> &'static str {
    match d {
        ActDtype::F32 => "f32",
        ActDtype::F16 => "f16",
    }
}

pub struct PodChild {
    pub name: String,
    pub child: Child,
}

/// A running pod. Dropping it kills every child process.
pub struct PodHandle {
    /// Base URL of the client's API, e.g. `http://127.0.0.1:8080`.
    pub api_url: String,
    /// `host:port` of each stage, in pipeline order.
    pub stage_addrs: Vec<String>,
    pub plan: SplitPlan,
    pub children: Vec<PodChild>,
}

impl PodHandle {
    /// Wait until any child exits; returns its name and status.
    pub async fn wait_any(&mut self) -> (String, std::io::Result<ExitStatus>) {
        let futs: Vec<_> = self
            .children
            .iter_mut()
            .map(|c| {
                Box::pin(async move {
                    let st = c.child.wait().await;
                    (c.name.clone(), st)
                })
            })
            .collect();
        if futs.is_empty() {
            return ("<none>".into(), Ok(ExitStatus::default()));
        }
        futures::future::select_all(futs).await.0
    }

    /// Kill every child and wait for them to exit.
    pub async fn shutdown(mut self) {
        for c in &mut self.children {
            let _ = c.child.kill().await;
        }
    }
}

impl Drop for PodHandle {
    fn drop(&mut self) {
        for c in &mut self.children {
            let _ = c.child.start_kill();
        }
    }
}

fn base_command(exe: &Path, log: &Option<String>) -> Command {
    let mut cmd = Command::new(exe);
    if let Some(l) = log {
        cmd.arg("--log").arg(l);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    cmd
}

fn ensure_port_free(addr: SocketAddr) -> anyhow::Result<()> {
    std::net::TcpListener::bind(addr)
        .map(drop)
        .with_context(|| format!("port {addr} is already in use"))
}

/// Poll until `addr` accepts a TCP connection, failing early if `child` exits.
async fn wait_for_port(
    addr: SocketAddr,
    child: &mut PodChild,
    timeout: Duration,
) -> anyhow::Result<()> {
    let start = Instant::now();
    loop {
        if let Some(st) = child.child.try_wait()? {
            bail!("{} exited before listening on {addr} ({st})", child.name);
        }
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return Ok(());
        }
        if start.elapsed() > timeout {
            bail!(
                "{} did not listen on {addr} within {:?}",
                child.name,
                timeout
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_health(url: &str, child: &mut PodChild, timeout: Duration) -> anyhow::Result<()> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let start = Instant::now();
    loop {
        if let Some(st) = child.child.try_wait()? {
            bail!("{} exited before becoming healthy ({st})", child.name);
        }
        if let Ok(r) = http.get(url).send().await {
            if r.status().is_success() {
                return Ok(());
            }
        }
        if start.elapsed() > timeout {
            bail!("{} did not answer {url} within {:?}", child.name, timeout);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Launch the stages, wait for them, launch the client and wait for `/health`.
pub async fn spawn_pod(opts: PodOptions) -> anyhow::Result<PodHandle> {
    let exe = match &opts.exe {
        Some(p) => p.clone(),
        None => std::env::current_exe().context("locating the dfin binary")?,
    };
    let n_layer = {
        let p = opts.model.clone();
        tokio::task::spawn_blocking(move || model_n_layer(&p)).await??
    };
    let plan = plan_split(&opts.split, n_layer)?;
    let dtype = act_dtype_str(opts.act_dtype);

    let mut stage_addrs = Vec::new();
    for i in 0..plan.stages.len() {
        let port = opts
            .base_port
            .checked_add(i as u16)
            .context("stage port overflows u16")?;
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        ensure_port_free(addr)?;
        stage_addrs.push(addr);
    }
    ensure_port_free(opts.listen)?;

    let mut handle = PodHandle {
        api_url: String::new(),
        stage_addrs: stage_addrs.iter().map(|a| a.to_string()).collect(),
        plan: plan.clone(),
        children: Vec::new(),
    };

    // All stages load in parallel.
    for (range, addr) in plan.stages.iter().zip(&stage_addrs) {
        let mut cmd = base_command(&exe, &opts.log);
        cmd.arg("stage")
            .arg("--model")
            .arg(&opts.model)
            .arg("--layers")
            .arg(range.to_string())
            .arg("--listen")
            .arg(addr.to_string())
            .arg("--n-ctx")
            .arg(opts.n_ctx.to_string())
            .arg("--n-seq-max")
            .arg(opts.n_seq_max.to_string())
            .arg("--n-gpu-layers")
            .arg(opts.n_gpu_layers.to_string())
            .arg("--act-dtype")
            .arg(dtype);
        for p in &opts.cpu_tensors {
            cmd.arg("--cpu-tensors").arg(p);
        }
        if let Some(t) = opts.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        let name = format!("stage {range} @ {addr}");
        info!("starting {name}");
        let child = cmd
            .spawn()
            .with_context(|| format!("spawning {}", exe.display()))?;
        handle.children.push(PodChild { name, child });
    }
    for (i, addr) in stage_addrs.iter().enumerate() {
        wait_for_port(*addr, &mut handle.children[i], opts.ready_timeout).await?;
        info!("{} is up", handle.children[i].name);
    }

    let mut cmd = base_command(&exe, &opts.log);
    cmd.arg("client")
        .arg("--model")
        .arg(&opts.model)
        .arg("--head")
        .arg(plan.head.to_string())
        .arg("--tail")
        .arg(format!("{}:end", plan.tail.begin));
    for (range, addr) in plan.stages.iter().zip(&stage_addrs) {
        cmd.arg("--stage").arg(format!("{addr}={range}"));
    }
    cmd.arg("--listen")
        .arg(opts.listen.to_string())
        .arg("--n-ctx")
        .arg(opts.n_ctx.to_string())
        .arg("--n-gpu-layers")
        .arg(opts.n_gpu_layers.to_string())
        .arg("--act-dtype")
        .arg(dtype)
        .arg("--max-sessions")
        .arg(opts.n_seq_max.to_string());
    for p in &opts.cpu_tensors {
        cmd.arg("--cpu-tensors").arg(p);
    }
    if let Some(t) = opts.threads {
        cmd.arg("--threads").arg(t.to_string());
    }
    info!("starting client on {}", opts.listen);
    let child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", exe.display()))?;
    handle.children.push(PodChild {
        name: format!("client @ {}", opts.listen),
        child,
    });

    let host = if opts.listen.ip().is_unspecified() {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        opts.listen.ip()
    };
    let api_url = format!("http://{}", SocketAddr::new(host, opts.listen.port()));
    let last = handle.children.len() - 1;
    wait_for_health(
        &format!("{api_url}/health"),
        &mut handle.children[last],
        opts.ready_timeout,
    )
    .await?;
    handle.api_url = api_url;
    Ok(handle)
}

pub async fn run(args: PodArgs) -> anyhow::Result<()> {
    let opts = PodOptions::from_args(&args);
    let mut pod = spawn_pod(opts).await?;
    println!(
        "dfin pod ready: OpenAI-compatible API at {}/v1",
        pod.api_url
    );
    println!(
        "  head {} | stages {} | tail {}:end",
        pod.plan.head,
        pod.plan
            .stages
            .iter()
            .zip(&pod.stage_addrs)
            .map(|(r, a)| format!("{r}@{a}"))
            .collect::<Vec<_>>()
            .join(", "),
        pod.plan.tail.begin
    );
    tokio::select! {
        r = tokio::signal::ctrl_c() => {
            r?;
            info!("ctrl-c: stopping pod");
        }
        (name, st) = pod.wait_any() => {
            warn!("{name} exited ({st:?}); stopping pod");
        }
    }
    pod.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_split_basic() {
        let p = plan_split(&[2, 5], 8).unwrap();
        assert_eq!(p.head, LayerRange::new(0, 2));
        assert_eq!(p.stages, vec![LayerRange::new(2, 5)]);
        assert_eq!(p.tail, LayerRange::new(5, 8));
        let p = plan_split(&[2, 4, 6], 8).unwrap();
        assert_eq!(p.stages, vec![LayerRange::new(2, 4), LayerRange::new(4, 6)]);
        assert!(plan_split(&[3], 8).is_err());
        assert!(plan_split(&[5, 2], 8).is_err());
        assert!(plan_split(&[0, 4], 8).is_err());
        assert!(plan_split(&[2, 8], 8).is_err());
        assert!(plan_split(&[2, 7], 8).is_ok());
    }
}
