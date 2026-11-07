//! OpenAI-compatible HTTP API (axum). Handlers never call into llama; they
//! submit jobs to the engine thread and relay its events.

use super::engine::{
    ApiError, Event, FinishReason, GenParams, Job, PromptInput, StaticInfo, Stats, Usage,
};
use super::sampling::SamplingParams;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dfin_llama::ChatMessage;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

pub struct Shared {
    pub jobs: mpsc::Sender<Job>,
    pub stats: Arc<Mutex<Stats>>,
    pub info: StaticInfo,
    pub model_name: String,
    pub created: u64,
}

pub type AppState = Arc<Shared>;

pub fn router(state: AppState) -> Router {
    let private = state.info.private_network;
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/status", get(status))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    if private {
        router.layer(axum::middleware::from_fn(
            crate::network::local_request_only,
        ))
    } else {
        router.layer(tower_http::cors::CorsLayer::permissive())
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(error_body(&self))).into_response()
    }
}

fn error_body(e: &ApiError) -> Value {
    json!({ "error": { "message": e.message, "type": e.kind, "code": null } })
}

// ----------------------------------------------------------------------------------------------
// Simple endpoints

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn models(State(st): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [ { "id": st.model_name, "object": "model", "created": st.created, "owned_by": "dfin" } ]
    }))
}

async fn status(State(st): State<AppState>) -> Json<Value> {
    let s = st.stats.lock().clone();
    let steps = s.decode_steps.max(1) as f64;
    let route_now = if s.current_route.is_empty() {
        st.info.route.clone()
    } else {
        s.current_route.clone()
    };
    let stage_entries: Vec<_> = route_now.iter().filter(|r| r.kind == "stage").collect();
    let directory = s.directory.as_ref().map(|d| {
        json!({
            "hub": d.hub,
            "issued_at": d.issued_at,
            "from_cache": d.from_cache,
            "nodes_listed": d.nodes_listed,
            "nodes_usable": d.nodes_usable,
            "fence": d.fence,
            "nodes_fenced_out": d.nodes_fenced_out,
        })
    });
    let stages: Vec<Value> = stage_entries
        .iter()
        .zip(s.hops.iter())
        .map(|(r, c)| {
            let n = c.forwards.max(1) as f64;
            json!({
                "addr": r.addr,
                "node_id": r.node_id,
                "layers": r.layers,
                "backend": r.backend,
                "forwards": c.forwards,
                "avg_compute_ms": c.compute_ms_sum / n,
                "decode_forwards": c.decode_forwards,
                "decode_avg_compute_ms": c.decode_compute_ms_sum / c.decode_forwards.max(1) as f64,
            })
        })
        .collect();
    let act_dtype = match st.info.act_dtype {
        dfin_proto::ActDtype::F32 => "f32",
        dfin_proto::ActDtype::F16 => "f16",
    };
    Json(json!({
        "model": st.model_name,
        "privacy": {
            "network": if st.info.private_network { "tor" } else { "direct" },
            "direct_fallback": false,
            "request_identity_rotation": st.info.private_network,
            "cross_request_prefix_cache": !st.info.private_network,
            "content_confidential_from_nodes": false,
            "relay_can_read_model_states": st.info.return_mode == "relay" && !stage_entries.is_empty(),
            "attestation_verified": false,
        },
        "model_path": st.info.model_path.display().to_string(),
        "n_layer": st.info.n_layer,
        "n_embd": st.info.n_embd,
        "act_dtype": act_dtype,
        "route": route_now,
        "directory": directory,
        "sessions": {
            "max": st.info.max_sessions,
            "live": s.sessions_live,
            "busy": s.sessions_busy,
            "ctx_per_session": st.info.per_session_ctx,
        },
        "prefix_cache": {
            "hits": s.prefix_hits,
            "misses": s.prefix_misses,
            "tokens_reused": s.prefix_tokens_reused,
        },
        "stats": {
            "requests": s.requests,
            "tokens_generated": s.tokens_generated,
            "avg_token_ms": s.decode_ms_sum / steps,
            "prefill_count": s.prefill_count,
            "avg_prefill_ms": s.prefill_ms_sum / s.prefill_count.max(1) as f64,
            "avg_prefill_tokens": s.prefill_tokens_sum as f64 / s.prefill_count.max(1) as f64,
            "last_prefill_ms": s.last_prefill_ms,
            "last_prefill_tokens": s.last_prefill_tokens,
            "p50_token_ms": s.p50_token_ms(),
            "head_ms": s.head_ms_sum / steps,
            "tail_ms": s.tail_ms_sum / steps,
            "chain": {
                "return_mode": st.info.return_mode,
                "forwards": s.chain.forwards,
                "avg_rtt_ms": s.chain.rtt_ms_sum / s.chain.forwards.max(1) as f64,
                "decode_forwards": s.chain.decode_forwards,
                "decode_avg_rtt_ms": s.chain.decode_rtt_ms_sum / s.chain.decode_forwards.max(1) as f64,
                "ping_ms": s.chain.ping_ms,
                "setup_ms": s.chain.setup_ms,
                "rebuilds": s.chain.rebuilds,
            },
            "stages": stages,
        }
    }))
}

// ----------------------------------------------------------------------------------------------
// Request parsing

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// Fields shared by both generation endpoints.
#[derive(Deserialize)]
struct CommonFields {
    #[serde(default)]
    model: Option<String>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    #[serde(default)]
    stop: Value,
    seed: Option<u64>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    prompt: Value,
    #[serde(flatten)]
    common: CommonFields,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Deserialize)]
struct RawMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<RawMessage>,
    #[serde(flatten)]
    common: CommonFields,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

const KNOWN_COMMON: &[&str] = &[
    "model",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "stop",
    "seed",
    "stream",
    "stream_options",
];

const UNSUPPORTED_COMMON: &[&str] = &[
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "user",
];

/// Return only static labels: arbitrary field names may carry private metadata,
/// control characters, or unlimited distinct strings. Do not retain or log them.
fn ignored_field_label(field: &str) -> Option<&'static str> {
    if KNOWN_COMMON.contains(&field) || field == "prompt" || field == "messages" {
        return None;
    }
    Some(
        UNSUPPORTED_COMMON
            .iter()
            .copied()
            .find(|known| *known == field)
            .unwrap_or("unrecognized (name omitted)"),
    )
}

/// Warn once for each recognized unsupported option and once for all other names.
fn warn_ignored(extra: &Map<String, Value>) {
    static WARNED: OnceLock<std::sync::Mutex<HashSet<&'static str>>> = OnceLock::new();
    let set = WARNED.get_or_init(Default::default);
    let mut set = set.lock().unwrap_or_else(|e| e.into_inner());
    for k in extra.keys() {
        let Some(label) = ignored_field_label(k) else {
            continue;
        };
        if set.insert(label) {
            tracing::warn!(
                field = label,
                "ignoring unsupported request field (logged once)"
            );
        }
    }
}

fn parse_body<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("invalid request body: {e}")))
}

fn parse_stop(v: &Value) -> Result<Vec<String>, ApiError> {
    match v {
        Value::Null => Ok(Vec::new()),
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) => a
            .iter()
            .map(|x| {
                x.as_str().map(str::to_string).ok_or_else(|| {
                    ApiError::bad_request("`stop` must be a string or an array of strings")
                })
            })
            .collect(),
        _ => Err(ApiError::bad_request(
            "`stop` must be a string or an array of strings",
        )),
    }
}

fn gen_params(c: &CommonFields) -> Result<GenParams, ApiError> {
    let d = SamplingParams::default();
    Ok(GenParams {
        max_tokens: c.max_tokens.or(c.max_completion_tokens).unwrap_or(128),
        sampling: SamplingParams {
            temperature: c.temperature.unwrap_or(d.temperature),
            top_k: c.top_k.unwrap_or(d.top_k),
            top_p: c.top_p.unwrap_or(d.top_p),
            min_p: c.min_p.unwrap_or(d.min_p),
            seed: c.seed,
        },
        stop: parse_stop(&c.stop)?,
    })
}

fn message_content(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    if p.get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|ty| ty == "text")
                    {
                        out.push_str(t);
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(
            "message `content` must be a string or an array of parts",
        )),
    }
}

// ----------------------------------------------------------------------------------------------
// Generation endpoints

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Completion,
    Chat,
}

async fn completions(State(st): State<AppState>, body: Bytes) -> Response {
    let req: CompletionRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    warn_ignored(&req.extra);
    let prompt = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(a) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap_or_default().to_string()
        }
        Value::Array(_) => {
            return ApiError::bad_request("only a single string prompt is supported")
                .into_response();
        }
        _ => return ApiError::bad_request("`prompt` must be a string").into_response(),
    };
    run_generation(st, Kind::Completion, PromptInput::Text(prompt), &req.common).await
}

async fn chat_completions(State(st): State<AppState>, body: Bytes) -> Response {
    let req: ChatRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    warn_ignored(&req.extra);
    if req.messages.is_empty() {
        return ApiError::bad_request("`messages` must not be empty").into_response();
    }
    let mut msgs = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        match message_content(&m.content) {
            Ok(content) => msgs.push(ChatMessage {
                role: m.role.clone(),
                content,
            }),
            Err(e) => return e.into_response(),
        }
    }
    run_generation(st, Kind::Chat, PromptInput::Chat(msgs), &req.common).await
}

async fn run_generation(
    st: AppState,
    kind: Kind,
    prompt: PromptInput,
    common: &CommonFields,
) -> Response {
    if let Some(model) = &common.model {
        if model != &st.model_name {
            return ApiError::bad_request(format!(
                "only model {:?} is served; attested or other trust tiers are not implemented",
                st.model_name
            ))
            .into_response();
        }
    }
    let params = match gen_params(common) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let stream = common.stream.unwrap_or(false);
    let include_usage = common
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);

    let (tx, mut rx) = mpsc::channel::<Event>(256);
    let job = Job {
        prompt,
        params,
        events: tx,
    };
    if let Err(e) = st.jobs.try_send(job) {
        return match e {
            mpsc::error::TrySendError::Full(_) => {
                ApiError::overloaded("request queue full").into_response()
            }
            mpsc::error::TrySendError::Closed(_) => {
                ApiError::internal("engine stopped").into_response()
            }
        };
    }
    // Wait until the engine accepts (or rejects) the job, so errors keep their status code.
    match rx.recv().await {
        Some(Event::Started { .. }) => {}
        Some(Event::Error(e)) => return e.into_response(),
        Some(other) => {
            return ApiError::internal(format!("unexpected first engine event {other:?}"))
                .into_response();
        }
        None => return ApiError::internal("engine dropped the request").into_response(),
    }

    let (id, object) = match kind {
        Kind::Completion => (
            format!("cmpl-{:016x}", rand::random::<u64>()),
            "text_completion",
        ),
        Kind::Chat => (
            format!("chatcmpl-{:016x}", rand::random::<u64>()),
            "chat.completion",
        ),
    };
    let created = unix_now();
    let model = st.model_name.clone();

    if !stream {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(Event::Token { text: t, .. }) => text.push_str(&t),
                Some(Event::Done {
                    finish_reason,
                    usage,
                }) => {
                    return Json(final_body(
                        kind,
                        &id,
                        object,
                        created,
                        &model,
                        text,
                        finish_reason,
                        usage,
                    ))
                    .into_response();
                }
                Some(Event::Error(e)) => return e.into_response(),
                Some(Event::Started { .. }) => {}
                None => return ApiError::internal("engine dropped the request").into_response(),
            }
        }
    }

    let chunk_object = match kind {
        Kind::Completion => "text_completion",
        Kind::Chat => "chat.completion.chunk",
    };
    let s = async_stream::stream! {
        let chunk = |choice: Value, usage: Option<Usage>| -> Result<SseEvent, Infallible> {
            let mut v = json!({
                "id": id, "object": chunk_object, "created": created, "model": model,
                "choices": [choice],
            });
            if let Some(u) = usage {
                v["usage"] = json!(u);
            }
            Ok(SseEvent::default().data(v.to_string()))
        };
        if kind == Kind::Chat {
            yield chunk(json!({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}), None);
        }
        loop {
            match rx.recv().await {
                Some(Event::Token { text, .. }) => {
                    let choice = match kind {
                        Kind::Completion => json!({"index": 0, "text": text, "finish_reason": null}),
                        Kind::Chat => json!({"index": 0, "delta": {"content": text}, "finish_reason": null}),
                    };
                    yield chunk(choice, None);
                }
                Some(Event::Done { finish_reason, usage }) => {
                    let choice = match kind {
                        Kind::Completion => json!({"index": 0, "text": "", "finish_reason": finish_reason.as_str()}),
                        Kind::Chat => json!({"index": 0, "delta": {}, "finish_reason": finish_reason.as_str()}),
                    };
                    let usage = include_usage.then_some(usage);
                    yield chunk(choice, usage);
                    break;
                }
                Some(Event::Error(e)) => {
                    yield Ok(SseEvent::default().data(error_body(&e).to_string()));
                    break;
                }
                Some(Event::Started { .. }) => {}
                None => {
                    let e = ApiError::internal("engine dropped the request");
                    yield Ok(SseEvent::default().data(error_body(&e).to_string()));
                    break;
                }
            }
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

#[allow(clippy::too_many_arguments)]
fn final_body(
    kind: Kind,
    id: &str,
    object: &str,
    created: u64,
    model: &str,
    text: String,
    finish: FinishReason,
    usage: Usage,
) -> Value {
    let choice = match kind {
        Kind::Completion => json!({ "index": 0, "text": text, "finish_reason": finish.as_str() }),
        Kind::Chat => json!({
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish.as_str(),
        }),
    };
    json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignored_request_field_names_do_not_enter_logs() {
        #[derive(Clone)]
        struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || Capture(writer.clone()))
            .finish();
        let mut extra = Map::new();
        extra.insert("private-field-canary\n\u{1b}[31m".into(), Value::Null);
        tracing::subscriber::with_default(subscriber, || warn_ignored(&extra));
        let log = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
        assert!(
            !log.contains("private-field-canary"),
            "request-controlled field name entered logs: {log:?}"
        );
        assert!(
            !log.contains('\u{1b}'),
            "request-controlled terminal escape entered logs"
        );
    }

    #[test]
    fn arbitrary_ignored_fields_share_one_static_warning_label() {
        let labels: HashSet<_> = (0..10_000)
            .filter_map(|i| ignored_field_label(&format!("private-field-{i}")))
            .collect();
        assert_eq!(labels, HashSet::from(["unrecognized (name omitted)"]));
        assert_eq!(ignored_field_label("tools"), Some("tools"));
        for supported in ["model", "messages", "prompt", "temperature"] {
            assert!(ignored_field_label(supported).is_none());
        }
    }

    #[test]
    fn content_parts_concatenate_text() {
        let v = json!([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]);
        assert_eq!(message_content(&v).unwrap(), "ab");
        assert_eq!(message_content(&json!("x")).unwrap(), "x");
    }

    #[test]
    fn stop_forms() {
        assert_eq!(parse_stop(&json!("a")).unwrap(), vec!["a"]);
        assert_eq!(parse_stop(&json!(["a", "b"])).unwrap(), vec!["a", "b"]);
        assert!(parse_stop(&json!(3)).is_err());
        assert!(parse_stop(&Value::Null).unwrap().is_empty());
    }

    #[test]
    fn chat_request_parses_with_extras() {
        let r: ChatRequest = serde_json::from_value(json!({
            "model": "x", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "max_tokens": 4, "stream_options": {"include_usage": true}
        }))
        .unwrap();
        assert_eq!(r.common.max_tokens, Some(4));
        assert!(r.extra.contains_key("tools"));
        assert!(r.common.stream_options.unwrap().include_usage);
    }
}
