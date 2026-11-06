//! The engine thread: owns the head and tail contexts and the route to the stages,
//! and runs generation jobs one at a time in FIFO order.
//!
//! Nothing in here touches the tokio runtime except through channels: jobs arrive
//! on a `tokio::sync::mpsc` channel (`blocking_recv`) and events go back on a
//! per-job channel (`blocking_send`).

use super::sampling::{Sampler, SamplingParams};
use super::route::{HopCounters, ReturnPolicy, Route, RouteCounters, RouteError};
use super::text::{StopMatcher, Utf8Accumulator};
use crate::config::{LayerRange, RouteConfig, StageEndpoint, StageEndpointToml};
use anyhow::Context as _;
use dfin_llama::{
    ChatMessage, Context, ContextParams, FlashAttn, KvType, Model, ModelParams, OutputMode, SeqId,
    Token,
};
use dfin_net::{Endpoint, NodeKey};
use dfin_proto::{ActDtype, Activations};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// How many recent decode-step times are kept for the median.
const TOKEN_WINDOW: usize = 1024;

// ----------------------------------------------------------------------------------------------
// Public job/event types

#[derive(Clone, Debug)]
pub enum PromptInput {
    /// `/v1/completions`: raw text, no special-token parsing.
    Text(String),
    /// `/v1/chat/completions`: rendered with the chat template.
    Chat(Vec<ChatMessage>),
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: u32,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
}

pub struct Job {
    pub prompt: PromptInput,
    pub params: GenParams,
    pub events: mpsc::Sender<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        ApiError {
            status: 400,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }
    pub fn overloaded(message: impl Into<String>) -> Self {
        ApiError {
            status: 503,
            kind: "overloaded",
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        ApiError {
            status: 500,
            kind: "engine_error",
            message: message.into(),
        }
    }
}

#[derive(Debug)]
pub enum Event {
    /// The request was accepted and prefill is starting.
    #[allow(dead_code)]
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    /// Newly decoded text (never empty).
    #[allow(dead_code)]
    Token {
        token: Token,
        text: String,
    },
    Done {
        finish_reason: FinishReason,
        usage: Usage,
    },
    Error(ApiError),
}

// ----------------------------------------------------------------------------------------------
// Configuration, static info, stats

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub head: LayerRange,
    pub tail: LayerRange,
    pub stages: Vec<StageEndpoint>,
    pub return_policy: ReturnPolicy,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    /// Tensor-name regexes kept on the CPU buffer type (MoE experts in RAM).
    pub cpu_tensors: Vec<String>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub act_dtype: ActDtype,
    pub max_sessions: u32,
    pub prefill_chunk: usize,
    pub chat_template: Option<String>,
    pub kv_type: KvType,
    pub flash_attn: FlashAttn,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteEntry {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub addr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub layers: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

/// Facts fixed at startup, for `/status`.
#[derive(Clone, Debug)]
pub struct StaticInfo {
    pub model_path: PathBuf,
    pub n_layer: i32,
    pub n_embd: i32,
    pub act_dtype: ActDtype,
    pub route: Vec<RouteEntry>,
    pub return_mode: &'static str,
    pub max_sessions: u32,
    pub per_session_ctx: usize,
}

/// Running counters the engine publishes for the HTTP handlers.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub requests: u64,
    pub tokens_generated: u64,
    pub decode_steps: u64,
    pub decode_ms_sum: f64,
    /// Prefill passes (one per request), their wall time and the tokens they processed.
    pub prefill_count: u64,
    pub prefill_ms_sum: f64,
    pub prefill_tokens_sum: u64,
    pub last_prefill_ms: f64,
    pub last_prefill_tokens: u64,
    pub head_ms_sum: f64,
    pub tail_ms_sum: f64,
    pub recent_token_ms: VecDeque<f64>,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_tokens_reused: u64,
    pub sessions_live: usize,
    pub sessions_busy: usize,
    pub hops: Vec<HopCounters>,
    pub chain: RouteCounters,
}

impl Stats {
    pub fn p50_token_ms(&self) -> f64 {
        if self.recent_token_ms.is_empty() {
            return 0.0;
        }
        let mut v: Vec<f64> = self.recent_token_ms.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    }
}

// ----------------------------------------------------------------------------------------------
// Engine

struct Session {
    id: u64,
    slot: SeqId,
    /// Tokens whose KV is resident in every piece of the pipeline, in order.
    tokens: Vec<Token>,
    last_used: Instant,
    busy: bool,
}

enum EngineErr {
    Route(RouteError),
    Other(String),
    /// The HTTP side went away; stop quietly.
    Cancelled,
}

impl From<RouteError> for EngineErr {
    fn from(e: RouteError) -> Self {
        EngineErr::Route(e)
    }
}

impl EngineErr {
    fn into_api(self) -> Option<ApiError> {
        match self {
            EngineErr::Route(e) => Some(ApiError::internal(e.to_string())),
            EngineErr::Other(m) => Some(ApiError::internal(m)),
            EngineErr::Cancelled => None,
        }
    }
}

fn llama_err(what: &str, e: dfin_llama::Error) -> EngineErr {
    EngineErr::Other(format!("{what}: {e}"))
}

struct Engine {
    head_model: Arc<Model>,
    head: Context,
    tail: Context,
    /// `None` when the route is head and tail only.
    route: Option<Route>,
    sessions: Vec<Session>,
    act_dtype: ActDtype,
    prefill_chunk: usize,
    n_embd: usize,
    per_session_ctx: usize,
    template: String,
    add_bos: bool,
    seq: u64,
    last_head_ms: f64,
    last_tail_ms: f64,
    stats: Arc<Mutex<Stats>>,
}

/// Start the engine thread. Resolves once models are loaded and stages are
/// connected, returning the job queue and the static facts for `/status`.
pub async fn spawn(
    cfg: EngineConfig,
    stats: Arc<Mutex<Stats>>,
    queue: usize,
) -> anyhow::Result<(mpsc::Sender<Job>, StaticInfo)> {
    let (job_tx, job_rx) = mpsc::channel::<Job>(queue.max(1));
    let (ready_tx, ready_rx) = oneshot::channel::<anyhow::Result<StaticInfo>>();
    // One ephemeral client key per daemon start; stages see this instead of an account.
    let endpoint = Endpoint::client(NodeKey::generate()).context("creating QUIC endpoint")?;
    let rt = tokio::runtime::Handle::current();
    std::thread::Builder::new()
        .name("dfin-engine".into())
        .spawn(move || match Engine::new(cfg, stats, endpoint, rt) {
            Ok((engine, info)) => {
                let _ = ready_tx.send(Ok(info));
                engine.run(job_rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .context("spawning engine thread")?;
    let info = ready_rx
        .await
        .context("engine thread exited during startup")??;
    Ok((job_tx, info))
}

fn lcp(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Engine {
    fn new(
        cfg: EngineConfig,
        stats: Arc<Mutex<Stats>>,
        endpoint: Endpoint,
        rt: tokio::runtime::Handle,
    ) -> anyhow::Result<(Engine, StaticInfo)> {
        dfin_llama::init();
        if cfg.max_sessions == 0 {
            anyhow::bail!("--max-sessions must be at least 1");
        }
        if cfg.head.begin != 0 || cfg.head.end <= 0 {
            anyhow::bail!("head must be a range 0:a with a > 0, got {}", cfg.head);
        }

        let head_params = ModelParams {
            n_gpu_layers: cfg.n_gpu_layers,
            cpu_tensor_patterns: cfg.cpu_tensors.clone(),
            ..ModelParams::range(0, cfg.head.end)
        };
        tracing::info!(model = %cfg.model_path.display(), layers = %cfg.head, "loading head");
        let head_model =
            Arc::new(Model::load(&cfg.model_path, &head_params).with_context(|| {
                format!(
                    "loading head layers {} of {}",
                    cfg.head,
                    cfg.model_path.display()
                )
            })?);
        let n_layer = head_model.n_layer();
        let n_embd = head_model.n_embd();

        let route = RouteConfig {
            model: cfg.model_path.clone(),
            head: cfg.head,
            tail: cfg.tail,
            stages: cfg
                .stages
                .iter()
                .map(|s| StageEndpointToml {
                    addr: s.addr.clone(),
                    node_id: s.node_id,
                    layers: s.layers,
                })
                .collect(),
        };
        route.validate(n_layer).context("invalid route")?;
        let tail_range = LayerRange::new(
            cfg.tail.begin,
            if cfg.tail.end == -1 {
                n_layer
            } else {
                cfg.tail.end
            },
        );

        let tail_params = ModelParams {
            n_gpu_layers: cfg.n_gpu_layers,
            cpu_tensor_patterns: cfg.cpu_tensors.clone(),
            ..ModelParams::range(cfg.tail.begin, -1)
        };
        tracing::info!(layers = %tail_range, "loading tail");
        let tail_model =
            Arc::new(Model::load(&cfg.model_path, &tail_params).with_context(|| {
                format!(
                    "loading tail layers {tail_range} of {}",
                    cfg.model_path.display()
                )
            })?);
        if !head_model.has_head() {
            anyhow::bail!("head piece does not own the token embedding");
        }
        if !tail_model.has_tail() {
            anyhow::bail!("tail piece does not own the LM head");
        }

        let prefill_chunk = cfg.prefill_chunk.max(1);
        let mut cp = ContextParams {
            n_ctx: cfg.n_ctx,
            n_batch: prefill_chunk as u32,
            n_ubatch: (prefill_chunk as u32).min(512),
            n_seq_max: cfg.max_sessions,
            flash_attn: cfg.flash_attn,
            type_k: cfg.kv_type,
            type_v: cfg.kv_type,
            kv_unified: false,
            ..ContextParams::default()
        };
        if let Some(t) = cfg.threads {
            cp.n_threads = t;
            cp.n_threads_batch = t;
        }
        if let Some(t) = cfg.threads_batch {
            cp.n_threads_batch = t;
        }
        let head = Context::new(
            head_model.clone(),
            &ContextParams {
                embeddings: true,
                ..cp.clone()
            },
        )
        .context("creating head context")?;
        let tail = Context::new(
            tail_model,
            &ContextParams {
                embeddings: false,
                ..cp
            },
        )
        .context("creating tail context")?;
        let per_session_ctx = (head.n_ctx().min(tail.n_ctx()) / cfg.max_sessions) as usize;
        tracing::info!(
            n_ctx = head.n_ctx(),
            max_sessions = cfg.max_sessions,
            per_session_ctx,
            "KV capacity per session is n_ctx / max_sessions"
        );

        let template = match &cfg.chat_template {
            Some(t) if std::path::Path::new(t).is_file() => {
                std::fs::read_to_string(t).with_context(|| format!("reading chat template {t}"))?
            }
            Some(t) => t.clone(),
            None => head_model
                .chat_template()
                .unwrap_or_else(|| "chatml".to_string()),
        };
        let add_bos = head_model.vocab().add_bos();

        let route = if cfg.stages.is_empty() {
            None
        } else {
            let fingerprint =
                dfin_proto::model_fingerprint(&cfg.model_path).context("fingerprinting model")?;
            let mut route = Route::new(
                cfg.stages.clone(),
                cfg.return_policy.clone(),
                endpoint,
                rt,
                fingerprint,
                n_embd as u32,
                prefill_chunk as u32,
            );
            connect_with_backoff(&mut route, Duration::from_secs(60))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Some(route)
        };

        let sessions = (0..cfg.max_sessions)
            .map(|slot| Session {
                id: rand::random(),
                slot: slot as SeqId,
                tokens: Vec::new(),
                last_used: Instant::now(),
                busy: false,
            })
            .collect();

        let mut route_info = vec![RouteEntry {
            kind: "head",
            addr: None,
            node_id: None,
            layers: cfg.head.to_string(),
            backend: None,
        }];
        if let Some(r) = &route {
            for (i, s) in r.hops.iter().enumerate() {
                route_info.push(RouteEntry {
                    kind: "stage",
                    addr: Some(s.addr.clone()),
                    node_id: Some(s.node_id.to_string()),
                    layers: s.layers.to_string(),
                    backend: Some(r.backends[i].clone()).filter(|b| !b.is_empty()),
                });
            }
        }
        route_info.push(RouteEntry {
            kind: "tail",
            addr: None,
            node_id: None,
            layers: tail_range.to_string(),
            backend: None,
        });

        let info = StaticInfo {
            model_path: cfg.model_path.clone(),
            n_layer,
            n_embd,
            act_dtype: cfg.act_dtype,
            route: route_info,
            return_mode: cfg.return_policy.as_str(),
            max_sessions: cfg.max_sessions,
            per_session_ctx,
        };
        stats.lock().hops = vec![HopCounters::default(); cfg.stages.len()];

        Ok((
            Engine {
                head_model,
                head,
                tail,
                route,
                sessions,
                act_dtype: cfg.act_dtype,
                prefill_chunk,
                n_embd: n_embd as usize,
                per_session_ctx,
                template,
                add_bos,
                seq: 0,
                last_head_ms: 0.0,
                last_tail_ms: 0.0,
                stats,
            },
            info,
        ))
    }

    fn run(mut self, mut jobs: mpsc::Receiver<Job>) {
        while let Some(job) = jobs.blocking_recv() {
            let events = job.events.clone();
            self.stats.lock().requests += 1;
            if let Err(e) = self.process(job) {
                if let Some(api) = e.into_api() {
                    tracing::warn!(status = api.status, "request failed: {}", api.message);
                    let _ = events.blocking_send(Event::Error(api));
                }
            }
            self.publish_stats();
        }
        tracing::info!("job queue closed; engine thread exiting");
    }

    fn publish_stats(&self) {
        let mut st = self.stats.lock();
        st.sessions_live = self
            .sessions
            .iter()
            .filter(|s| !s.tokens.is_empty())
            .count();
        st.sessions_busy = self.sessions.iter().filter(|s| s.busy).count();
        if let Some(r) = &self.route {
            st.hops = r.hop_counters.clone();
            st.chain = r.counters.clone();
        }
    }

    /// Rebuild the route if it dropped. The stages' session state is then unknown,
    /// so every cached session is dropped.
    fn ensure_connected(&mut self) -> Result<(), EngineErr> {
        let Some(route) = &mut self.route else {
            return Ok(());
        };
        if route.is_connected() {
            return Ok(());
        }
        tracing::warn!(first = %route.first_addr(), "route dropped; rebuilding");
        connect_with_backoff(route, Duration::from_secs(5))?;
        route.counters.rebuilds += 1;
        for i in 0..self.sessions.len() {
            if !self.sessions[i].tokens.is_empty() {
                self.reset_session(i);
            }
        }
        Ok(())
    }

    /// Drop a session everywhere, best effort, and give the slot a fresh id.
    fn reset_session(&mut self, idx: usize) {
        let id = self.sessions[idx].id;
        let slot = self.sessions[idx].slot;
        if let Some(r) = &mut self.route {
            if r.is_connected() {
                let _ = r.close_session(id);
            }
        }
        self.head.seq_rm(slot, -1, -1);
        self.tail.seq_rm(slot, -1, -1);
        let sess = &mut self.sessions[idx];
        sess.tokens.clear();
        sess.id = rand::random();
    }

    /// Evict a session to reuse its slot for an unrelated prompt.
    fn evict(&mut self, idx: usize) -> Result<(), EngineErr> {
        let id = self.sessions[idx].id;
        let slot = self.sessions[idx].slot;
        if let Some(r) = &mut self.route {
            match r.close_session(id) {
                Ok(()) => {}
                Err(RouteError::Stage { code, message, .. }) => {
                    tracing::debug!(?code, "SessionClose: {message}");
                }
                Err(e) => return Err(e.into()),
            }
        }
        self.head.seq_rm(slot, -1, -1);
        self.tail.seq_rm(slot, -1, -1);
        let sess = &mut self.sessions[idx];
        sess.tokens.clear();
        sess.id = rand::random();
        Ok(())
    }

    /// Roll a session back so that only its first `keep` tokens remain cached.
    fn rollback(&mut self, idx: usize, keep: usize) -> Result<(), EngineErr> {
        if self.sessions[idx].tokens.len() <= keep {
            return Ok(());
        }
        let id = self.sessions[idx].id;
        let slot = self.sessions[idx].slot;
        let p0 = keep as i32;
        if !self.head.seq_rm(slot, p0, -1) || !self.tail.seq_rm(slot, p0, -1) {
            return Err(EngineErr::Other(format!(
                "KV rollback of slot {slot} to {keep} failed"
            )));
        }
        if let Some(r) = &mut self.route {
            r.seq_rm(id, p0, -1)?;
        }
        self.sessions[idx].tokens.truncate(keep);
        Ok(())
    }

    /// Choose the session for a prompt and roll it back to the reusable prefix.
    /// Returns `(session index, cached prefix length)`, or `None` if every session is busy.
    fn acquire_session(&mut self, prompt: &[Token]) -> Result<Option<(usize, usize)>, EngineErr> {
        // A lone BOS is shared by every prompt; reusing a session for it would
        // throw away that session's cache for no gain.
        let min_hit = if self.add_bos { 2 } else { 1 };
        let best = self
            .sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy)
            .map(|(i, s)| (i, lcp(&s.tokens, prompt)))
            .max_by_key(|&(i, l)| (l, std::cmp::Reverse(i)));
        if let Some((idx, l)) = best {
            if l >= min_hit {
                let common = if l == prompt.len() { l - 1 } else { l };
                if common > 0 {
                    self.rollback(idx, common)?;
                    let mut st = self.stats.lock();
                    st.prefix_hits += 1;
                    st.prefix_tokens_reused += common as u64;
                    return Ok(Some((idx, common)));
                }
            }
        }
        // Miss: a free slot, else the least recently used idle session.
        let free = self
            .sessions
            .iter()
            .position(|s| !s.busy && s.tokens.is_empty());
        let lru = || {
            self.sessions
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.busy)
                .min_by_key(|(_, s)| s.last_used)
                .map(|(i, _)| i)
        };
        let Some(idx) = free.or_else(lru) else {
            return Ok(None);
        };
        if !self.sessions[idx].tokens.is_empty() {
            tracing::debug!(slot = self.sessions[idx].slot, "evicting LRU session");
            self.evict(idx)?;
        }
        self.stats.lock().prefix_misses += 1;
        Ok(Some((idx, 0)))
    }

    fn tokenize(&self, prompt: &PromptInput) -> Result<Vec<Token>, ApiError> {
        let vocab = self.head_model.vocab();
        match prompt {
            PromptInput::Text(s) => vocab
                .tokenize(s, self.add_bos, false)
                .map_err(|e| ApiError::bad_request(format!("tokenization failed: {e}"))),
            PromptInput::Chat(msgs) => {
                let rendered = dfin_llama::chat_apply_template(&self.template, msgs, true)
                    .map_err(|e| ApiError::bad_request(format!("applying chat template: {e}")))?;
                vocab
                    .tokenize(&rendered, self.add_bos, true)
                    .map_err(|e| ApiError::bad_request(format!("tokenization failed: {e}")))
            }
        }
    }

    fn process(&mut self, job: Job) -> Result<(), EngineErr> {
        let prompt = match self.tokenize(&job.prompt) {
            Ok(t) => t,
            Err(e) => {
                let _ = job.events.blocking_send(Event::Error(e));
                return Ok(());
            }
        };
        let reject = |e: ApiError| {
            let _ = job.events.blocking_send(Event::Error(e));
            Ok(())
        };
        if prompt.is_empty() {
            return reject(ApiError::bad_request("prompt is empty"));
        }
        if prompt.len() > self.per_session_ctx {
            return reject(ApiError::bad_request(format!(
                "prompt has {} tokens but the per-session context is {} (n_ctx / max_sessions)",
                prompt.len(),
                self.per_session_ctx
            )));
        }
        if self.sessions.iter().all(|s| s.busy) {
            return reject(ApiError::overloaded("all sessions busy"));
        }
        self.ensure_connected()?;
        let max_tokens = (job.params.max_tokens as usize).min(self.per_session_ctx - prompt.len());

        let Some((idx, common)) = self.acquire_session(&prompt)? else {
            return reject(ApiError::overloaded("all sessions busy"));
        };
        self.sessions[idx].busy = true;
        self.publish_stats();
        let res = self.generate(idx, &prompt, common, max_tokens, &job);
        let sess = &mut self.sessions[idx];
        sess.busy = false;
        sess.last_used = Instant::now();
        match &res {
            Ok(()) | Err(EngineErr::Cancelled) => {}
            Err(_) => self.reset_session(idx),
        }
        res
    }

    fn send(job: &Job, ev: Event) -> Result<(), EngineErr> {
        job.events
            .blocking_send(ev)
            .map_err(|_| EngineErr::Cancelled)
    }

    fn generate(
        &mut self,
        idx: usize,
        prompt: &[Token],
        common: usize,
        max_tokens: usize,
        job: &Job,
    ) -> Result<(), EngineErr> {
        Self::send(
            job,
            Event::Started {
                prompt_tokens: prompt.len(),
                cached_tokens: common,
            },
        )?;
        let (id, slot) = (self.sessions[idx].id, self.sessions[idx].slot);
        debug_assert_eq!(self.sessions[idx].tokens.len(), common);

        // Prefill the uncached suffix in chunks; only the final chunk needs logits.
        let suffix = &prompt[common..];
        let n_chunks = suffix.len().div_ceil(self.prefill_chunk);
        let mut pos = common;
        let mut logits = None;
        let t_prefill = Instant::now();
        for (ci, chunk) in suffix.chunks(self.prefill_chunk).enumerate() {
            let last = ci + 1 == n_chunks;
            let out = self.forward(id, slot, pos as i32, chunk, last)?;
            self.sessions[idx].tokens.extend_from_slice(chunk);
            pos += chunk.len();
            if last {
                logits = out;
            }
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        tracing::debug!(
            prompt = prompt.len(),
            cached = common,
            ms = prefill_ms,
            "prefill done"
        );
        {
            let mut st = self.stats.lock();
            st.prefill_count += 1;
            st.prefill_ms_sum += prefill_ms;
            st.prefill_tokens_sum += suffix.len() as u64;
            st.last_prefill_ms = prefill_ms;
            st.last_prefill_tokens = suffix.len() as u64;
        }
        let mut logits =
            logits.ok_or_else(|| EngineErr::Other("prefill produced no logits".into()))?;

        let model = self.head_model.clone();
        let vocab = model.vocab();
        let mut sampler = Sampler::new(job.params.sampling.clone());
        let mut utf8 = Utf8Accumulator::default();
        let mut stop = StopMatcher::new(job.params.stop.clone());
        let mut completion = 0usize;
        let mut finish = FinishReason::Length;
        let mut stopped_by_string = false;

        while completion < max_tokens {
            let t_step = Instant::now();
            let tok = sampler.sample(&logits) as Token;
            if vocab.is_eog(tok) {
                finish = FinishReason::Stop;
                break;
            }
            completion += 1;
            let piece = vocab.token_to_piece(tok, false);
            let text = utf8.push(&piece);
            let r = stop.push(&text);
            if !r.emit.is_empty() {
                Self::send(
                    job,
                    Event::Token {
                        token: tok,
                        text: r.emit,
                    },
                )?;
            }
            if r.stopped {
                finish = FinishReason::Stop;
                stopped_by_string = true;
                break;
            }
            if completion >= max_tokens {
                finish = FinishReason::Length;
                break;
            }
            let pos0 = self.sessions[idx].tokens.len() as i32;
            let out = self.forward(id, slot, pos0, &[tok], true)?;
            self.sessions[idx].tokens.push(tok);
            logits = out.ok_or_else(|| EngineErr::Other("decode produced no logits".into()))?;
            let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
            let mut st = self.stats.lock();
            st.decode_steps += 1;
            st.decode_ms_sum += step_ms;
            st.head_ms_sum += self.last_head_ms;
            st.tail_ms_sum += self.last_tail_ms;
            if st.recent_token_ms.len() == TOKEN_WINDOW {
                st.recent_token_ms.pop_front();
            }
            st.recent_token_ms.push_back(step_ms);
        }

        if !stopped_by_string {
            let rest = utf8.flush();
            let r = stop.push(&rest);
            let mut tail_text = r.emit;
            if r.stopped {
                finish = FinishReason::Stop;
            } else {
                tail_text.push_str(&stop.flush());
            }
            if !tail_text.is_empty() {
                Self::send(
                    job,
                    Event::Token {
                        token: -1,
                        text: tail_text,
                    },
                )?;
            }
        }
        self.stats.lock().tokens_generated += completion as u64;
        Self::send(
            job,
            Event::Done {
                finish_reason: finish,
                usage: Usage {
                    prompt_tokens: prompt.len(),
                    completion_tokens: completion,
                    total_tokens: prompt.len() + completion,
                },
            },
        )
    }

    /// One pipeline step: head over `tokens` at `pos0`, the route, tail.
    /// Returns the last row's logits when `want_logits`.
    fn forward(
        &mut self,
        session: u64,
        slot: SeqId,
        pos0: i32,
        tokens: &[Token],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, EngineErr> {
        let n = tokens.len();
        let n_embd = self.n_embd;

        let t = Instant::now();
        self.head
            .decode_tokens(tokens, pos0, slot, OutputMode::All)
            .map_err(|e| llama_err("head decode", e))?;
        let emb = self.head.embeddings();
        if emb.len() != n * n_embd {
            return Err(EngineErr::Other(format!(
                "head produced {} hidden values, expected {}",
                emb.len(),
                n * n_embd
            )));
        }
        let mut hidden = emb.to_vec();
        self.last_head_ms = t.elapsed().as_secs_f64() * 1e3;

        if let Some(route) = &mut self.route {
            let act = Activations::from_f32(&hidden, n, n_embd, self.act_dtype);
            self.seq += 1;
            let (out, _hops) = route.forward(session, self.seq, pos0, act)?;
            if out.n_tokens as usize != n || out.n_embd as usize != n_embd {
                return Err(EngineErr::Other(format!(
                    "route returned [{}, {}] activations, expected [{n}, {n_embd}]",
                    out.n_tokens, out.n_embd
                )));
            }
            hidden = out
                .to_f32()
                .map_err(|e| EngineErr::Other(format!("route result: {e}")))?;
        }

        let t = Instant::now();
        let mode = if want_logits {
            OutputMode::Last
        } else {
            OutputMode::None
        };
        self.tail
            .decode_embd(&hidden, pos0, slot, mode)
            .map_err(|e| llama_err("tail decode", e))?;
        let out = if want_logits {
            let l = self.tail.logits_ith(-1);
            if l.is_empty() {
                return Err(EngineErr::Other("tail produced no logits".into()));
            }
            Some(l.to_vec())
        } else {
            None
        };
        self.last_tail_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }
}

/// Set the route up, retrying transport failures with exponential backoff until
/// `budget` elapses. Protocol-level rejections (wrong model, wrong layers) are not retried.
fn connect_with_backoff(route: &mut Route, budget: Duration) -> Result<(), RouteError> {
    let deadline = Instant::now() + budget;
    let mut delay = Duration::from_millis(250);
    loop {
        match route.connect() {
            Ok(()) => return Ok(()),
            Err(e @ RouteError::Protocol(_)) => return Err(e),
            Err(e @ RouteError::Stage { .. }) => return Err(e),
            Err(e) => {
                if Instant::now() + delay > deadline {
                    return Err(e);
                }
                tracing::debug!("{e}; retrying in {delay:?}");
                std::thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_secs(4));
            }
        }
    }
}
