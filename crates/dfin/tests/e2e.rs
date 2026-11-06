//! End-to-end: spawn `dfin stage` servers and a `dfin client`, then check the
//! OpenAI-compatible API against an in-process monolithic greedy generation.
//!
//! Requires the layer-range patch in the vendored llama.cpp (skips otherwise) and
//! the test model at `models/tiny-llama-f32.gguf`.

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use dfin_net::directory::verify_directory;
use dfin_net::transport::{CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Endpoint, NodeId, NodeKey};
use dfin_proto::{Request, Response, MAX_DIRECTORY_BYTES};
use serde_json::{json, Value};
use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PROMPT: &str = "hello world the model";
const MAX_TOKENS: usize = 8;
const MODEL_NAME: &str = "tiny-llama-f32";

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-llama-f32.gguf")
}

fn should_skip() -> bool {
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: vendored libllama lacks the layer-range patch");
        return true;
    }
    if !model_path().is_file() {
        eprintln!("SKIP: test model {} not found", model_path().display());
        return true;
    }
    false
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn free_udp_port() -> u16 {
    let s = UdpSocket::bind("127.0.0.1:0").unwrap();
    s.local_addr().unwrap().port()
}

/// A node key saved to a temp file, for a stage child to use.
struct StageKey {
    path: PathBuf,
    id: NodeId,
}

impl StageKey {
    fn new(name: &str) -> StageKey {
        let key = NodeKey::generate();
        let path = std::env::temp_dir().join(format!(
            "dfin-e2e-{name}-{}-{}.key",
            std::process::id(),
            rand_suffix()
        ));
        key.save(&path).unwrap();
        StageKey { path, id: key.id() }
    }
}

impl Drop for StageKey {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// How many records the hub at `addr` lists (0 if it cannot be reached or verified).
async fn directory_count(probe: &Endpoint, addr: &str, hub_id: NodeId) -> usize {
    let Ok(conn) = probe.connect(addr, hub_id).await else {
        return 0;
    };
    let r: Result<Response, _> = conn
        .call(&Request::DirectoryGet, MAX_DIRECTORY_BYTES, CONTROL_TIMEOUT)
        .await;
    conn.close("probe");
    match r {
        Ok(Response::Directory { directory }) => verify_directory(&directory, &hub_id)
            .map(|d| d.entries.len())
            .unwrap_or(0),
        _ => 0,
    }
}

async fn wait_listed(probe: &Endpoint, addr: &str, hub_id: NodeId, n: usize, procs: &Procs) {
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if directory_count(probe, addr, hub_id).await >= n {
            return;
        }
        if Instant::now() > deadline {
            panic!("hub never listed {n} stages\n{}", procs.logs());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn stage_answers(probe: &Endpoint, addr: &str, id: NodeId) -> bool {
    let Ok(conn) = probe.connect(addr, id).await else {
        return false;
    };
    let r: Result<Response, _> = conn
        .call(&Request::Ping, MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
        .await;
    conn.close("probe");
    matches!(r, Ok(Response::Pong))
}

/// A spawned child process plus the log file its stderr goes to.
struct Proc {
    name: String,
    child: Child,
    log: PathBuf,
}

/// Kills every child on drop, so a failing assertion never leaks processes.
#[derive(Default)]
struct Procs(Vec<Proc>);

impl Drop for Procs {
    fn drop(&mut self) {
        for p in &mut self.0 {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}

impl Procs {
    fn spawn(&mut self, name: &str, args: &[String]) -> usize {
        let log =
            std::env::temp_dir().join(format!("dfin-e2e-{}-{}.log", name, std::process::id()));
        let file = std::fs::File::create(&log).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_dfin"))
            .args(args)
            .env("DFIN_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(file))
            .spawn()
            .unwrap_or_else(|e| panic!("spawning {name}: {e}"));
        self.0.push(Proc {
            name: name.to_string(),
            child,
            log,
        });
        self.0.len() - 1
    }

    fn logs(&self) -> String {
        let mut out = String::new();
        for p in &self.0 {
            let text = std::fs::read_to_string(&p.log).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let start = lines.len().saturating_sub(40);
            out.push_str(&format!("----- {} ({}) -----\n", p.name, p.log.display()));
            out.push_str(&lines[start..].join("\n"));
            out.push('\n');
        }
        out
    }

    fn kill(&mut self, i: usize) {
        let _ = self.0[i].child.kill();
        let _ = self.0[i].child.wait();
    }

    fn check_alive(&mut self, i: usize) {
        if let Ok(Some(st)) = self.0[i].child.try_wait() {
            panic!("{} exited early ({st})\n{}", self.0[i].name, self.logs());
        }
    }

    async fn wait_stage(&mut self, i: usize, port: u16, id: NodeId, timeout: Duration) {
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let probe = Endpoint::client(NodeKey::generate()).unwrap();
        let deadline = Instant::now() + timeout;
        loop {
            if stage_answers(&probe, &addr.to_string(), id).await {
                return;
            }
            self.check_alive(i);
            if Instant::now() > deadline {
                panic!(
                    "{} did not answer on {addr}\n{}",
                    self.0[i].name,
                    self.logs()
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn wait_health(&mut self, i: usize, base: &str, timeout: Duration) {
        let http = reqwest::Client::new();
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(r) = http.get(format!("{base}/health")).send().await {
                if r.status().is_success() {
                    return;
                }
            }
            self.check_alive(i);
            if Instant::now() > deadline {
                panic!("{} never became healthy\n{}", self.0[i].name, self.logs());
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

/// Monolithic greedy reference: full model on CPU, same tokenization as the client.
fn reference_text() -> String {
    dfin_llama::init();
    let model = Arc::new(
        Model::load(
            &model_path(),
            &ModelParams {
                n_gpu_layers: 0,
                ..ModelParams::full()
            },
        )
        .unwrap(),
    );
    let vocab = model.vocab();
    let prompt = vocab.tokenize(PROMPT, vocab.add_bos(), false).unwrap();
    // Same KV geometry as the client under test (--n-ctx 512 --max-sessions 2, slot 0).
    let mut ctx = Context::new(
        model.clone(),
        &ContextParams {
            n_ctx: 512,
            n_batch: 512,
            n_ubatch: 512,
            n_seq_max: 2,
            ..ContextParams::default()
        },
    )
    .unwrap();
    ctx.decode_tokens(&prompt, 0, 0, OutputMode::Last).unwrap();
    let mut bytes = Vec::new();
    let mut pos = prompt.len() as i32;
    for step in 0..MAX_TOKENS {
        let logits = ctx.logits_ith(-1);
        let tok = argmax(logits) as dfin_llama::Token;
        if vocab.is_eog(tok) {
            break;
        }
        bytes.extend_from_slice(&vocab.token_to_piece(tok, false));
        if step + 1 == MAX_TOKENS {
            break;
        }
        ctx.decode_tokens(&[tok], pos, 0, OutputMode::Last).unwrap();
        pos += 1;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

async fn post(http: &reqwest::Client, url: String, body: Value) -> (u16, String) {
    let r = http.post(url).json(&body).send().await.unwrap();
    let status = r.status().as_u16();
    (status, r.text().await.unwrap())
}

/// Steps (1) to (5) of the client contract.
async fn exercise(procs: &Procs, base: &str, expected: &str) {
    let http = reqwest::Client::new();
    let body = json!({ "model": MODEL_NAME, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0 });

    // (1) Non-streamed completion equals the monolithic greedy text.
    let (st, text) = post(&http, format!("{base}/v1/completions"), body.clone()).await;
    assert_eq!(st, 200, "completion failed: {text}\n{}", procs.logs());
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
    let got = v["choices"][0]["text"].as_str().unwrap().to_string();
    assert_eq!(
        got, expected,
        "pipelined text differs from monolithic reference"
    );
    let fr = v["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(fr == "length" || fr == "stop", "finish_reason {fr}");
    assert_eq!(
        v["usage"]["prompt_tokens"].as_u64().unwrap()
            + v["usage"]["completion_tokens"].as_u64().unwrap(),
        v["usage"]["total_tokens"].as_u64().unwrap()
    );

    // (2) The same request again hits the prefix cache.
    let (st, text) = post(&http, format!("{base}/v1/completions"), body.clone()).await;
    assert_eq!(st, 200, "{text}");
    let status: Value = http
        .get(format!("{base}/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["prefix_cache"]["hits"], 1, "status: {status:#}");
    assert_eq!(status["model"], MODEL_NAME);
    assert_eq!(status["n_layer"], 8);

    // (3) Streaming yields the same text.
    let mut sbody = body.clone();
    sbody["stream"] = json!(true);
    let r = http
        .post(format!("{base}/v1/completions"))
        .json(&sbody)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert!(r
        .headers()
        .get("content-type")
        .map(|h| h.to_str().unwrap_or("").starts_with("text/event-stream"))
        .unwrap_or(false));
    let raw = r.text().await.unwrap();
    let mut streamed = String::new();
    let mut done = false;
    let mut finish = None;
    for line in raw.lines() {
        let Some(data) = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
        else {
            continue;
        };
        if data.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let c: Value =
            serde_json::from_str(data).unwrap_or_else(|e| panic!("bad chunk {data:?}: {e}"));
        assert_eq!(c["object"], "text_completion");
        streamed.push_str(c["choices"][0]["text"].as_str().unwrap_or(""));
        if let Some(f) = c["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_string());
        }
    }
    assert!(done, "stream did not end with [DONE]: {raw}");
    assert!(finish.is_some(), "no finish_reason chunk: {raw}");
    assert_eq!(streamed, expected, "streamed text differs");

    // (4) Chat completion shape.
    let (st, text) = post(
        &http,
        format!("{base}/v1/chat/completions"),
        json!({ "model": MODEL_NAME, "messages": [ { "role": "user", "content": "hi there" } ], "max_tokens": 4, "temperature": 0 }),
    )
    .await;
    assert_eq!(st, 200, "chat failed: {text}\n{}", procs.logs());
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["message"]["content"].is_string());
    assert!(v["choices"][0]["finish_reason"].is_string());
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() <= 4);

    // (5) /v1/models lists the model.
    let models: Value = http
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["object"], "list");
    assert!(
        models["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["id"] == MODEL_NAME),
        "{models}"
    );
}

/// Two stages and a client in the given return mode (`direct`, `reverse` or `relay`).
async fn pipeline_with_two_stages(return_mode: &str) {
    if should_skip() {
        return;
    }
    let expected = tokio::task::spawn_blocking(reference_text).await.unwrap();
    eprintln!("reference text: {expected:?}");
    let model = model_path().display().to_string();
    let (p1, p2, pc) = (free_udp_port(), free_udp_port(), free_port());
    let (k1, k2) = (StageKey::new("stage1"), StageKey::new("stage2"));
    let mut procs = Procs::default();
    let mut stage_idx = Vec::new();
    for (name, port, layers, key) in [("stage1", p1, "2:4", &k1), ("stage2", p2, "4:6", &k2)] {
        let args: Vec<String> = [
            "stage",
            "--model",
            &model,
            "--layers",
            layers,
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--identity",
            &key.path.display().to_string(),
            "--n-gpu-layers",
            "0",
            "--act-dtype",
            "f32",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        stage_idx.push((procs.spawn(name, &args), port, key.id));
    }
    let relay = if return_mode == "relay" {
        let port = free_udp_port();
        let key = StageKey::new("relay");
        let args: Vec<String> = [
            "relay",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--identity",
            &key.path.display().to_string(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let i = procs.spawn("relay", &args);
        Some((i, port, key))
    } else {
        None
    };
    for (i, port, id) in stage_idx {
        procs.wait_stage(i, port, id, Duration::from_secs(120)).await;
    }
    if let Some((i, port, key)) = &relay {
        procs.wait_stage(*i, *port, key.id, Duration::from_secs(60)).await;
    }
    let mut args: Vec<String> = [
        "client",
        "--model",
        &model,
        "--head",
        "0:2",
        "--tail",
        "6:end",
        "--stage",
        &format!("127.0.0.1:{p1}/{}=2:4", k1.id),
        "--stage",
        &format!("127.0.0.1:{p2}/{}=4:6", k2.id),
        "--n-gpu-layers",
        "0",
        "--act-dtype",
        "f32",
        "--max-sessions",
        "2",
        "--n-ctx",
        "512",
        "--listen",
        &format!("127.0.0.1:{pc}"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    match &relay {
        Some((_, port, key)) => {
            args.push("--relay".into());
            args.push(format!("127.0.0.1:{port}/{}", key.id));
        }
        None => {
            args.push("--return".into());
            args.push(return_mode.into());
        }
    }
    let ci = procs.spawn("client", &args);
    let base = format!("http://127.0.0.1:{pc}");
    procs.wait_health(ci, &base, Duration::from_secs(120)).await;

    exercise(&procs, &base, &expected).await;

    let status: Value = reqwest::get(format!("{base}/status"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let route = status["route"].as_array().unwrap();
    assert_eq!(route.len(), 4, "{status:#}");
    let stages = status["stats"]["stages"].as_array().unwrap();
    assert_eq!(stages.len(), 2);
    assert!(
        stages.iter().all(|s| s["forwards"].as_u64().unwrap() > 0),
        "{status:#}"
    );
    assert_eq!(status["stats"]["chain"]["return_mode"], return_mode, "{status:#}");
    assert!(
        status["stats"]["chain"]["forwards"].as_u64().unwrap() > 0,
        "{status:#}"
    );
    assert_eq!(status["route"][1]["node_id"], k1.id.to_string());
}

#[tokio::test(flavor = "multi_thread")]
async fn two_stages_direct_return() {
    pipeline_with_two_stages("direct").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_stages_reverse_return() {
    pipeline_with_two_stages("reverse").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_stages_through_relay() {
    pipeline_with_two_stages("relay").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn head_tail_only() {
    if should_skip() {
        return;
    }
    let expected = tokio::task::spawn_blocking(reference_text).await.unwrap();
    let model = model_path().display().to_string();
    let pc = free_port();
    let mut procs = Procs::default();
    let args: Vec<String> = [
        "client",
        "--model",
        &model,
        "--head",
        "0:4",
        "--tail",
        "4:end",
        "--n-gpu-layers",
        "0",
        "--act-dtype",
        "f32",
        "--max-sessions",
        "2",
        "--n-ctx",
        "512",
        "--listen",
        &format!("127.0.0.1:{pc}"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let ci = procs.spawn("client-ht", &args);
    let base = format!("http://127.0.0.1:{pc}");
    procs.wait_health(ci, &base, Duration::from_secs(120)).await;

    exercise(&procs, &base, &expected).await;

    let status: Value = reqwest::get(format!("{base}/status"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["route"].as_array().unwrap().len(), 2, "{status:#}");
    assert_eq!(status["route"][1]["layers"], "4:8");
}

/// Stages announce to a hub; the client is given only the hub and builds its own route.
/// Then the first stage is killed, a replacement for its range announces, and the next
/// requests must succeed over the new route.
#[tokio::test(flavor = "multi_thread")]
async fn route_from_hub_and_reroute_after_stage_death() {
    if should_skip() {
        return;
    }
    let expected = tokio::task::spawn_blocking(reference_text).await.unwrap();
    let model = model_path().display().to_string();
    let probe = Endpoint::client(NodeKey::generate()).unwrap();
    let mut procs = Procs::default();

    let hub_port = free_udp_port();
    let hub_key = StageKey::new("hub");
    let hub_addr = format!("127.0.0.1:{hub_port}");
    let hub_args: Vec<String> = ["hub", "--listen", &hub_addr, "--identity", &hub_key.path.display().to_string()]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let hi = procs.spawn("hub", &hub_args);
    procs.wait_stage(hi, hub_port, hub_key.id, Duration::from_secs(60)).await;
    let hub_flag = format!("{hub_addr}/{}", hub_key.id);

    let spawn_stage = |procs: &mut Procs, name: &str, layers: &str, key: &StageKey, port: u16| -> usize {
        let args: Vec<String> = [
            "stage",
            "--model",
            &model,
            "--layers",
            layers,
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--identity",
            &key.path.display().to_string(),
            "--hub",
            &hub_flag,
            "--announce-every",
            "5",
            "--n-gpu-layers",
            "0",
            "--act-dtype",
            "f32",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        procs.spawn(name, &args)
    };
    let (k1, k2, k3) = (StageKey::new("s1"), StageKey::new("s2"), StageKey::new("s3"));
    let (p1, p2, p3) = (free_udp_port(), free_udp_port(), free_udp_port());
    let s1 = spawn_stage(&mut procs, "stage1", "2:4", &k1, p1);
    let s2 = spawn_stage(&mut procs, "stage2", "4:6", &k2, p2);
    procs.wait_stage(s1, p1, k1.id, Duration::from_secs(120)).await;
    procs.wait_stage(s2, p2, k2.id, Duration::from_secs(120)).await;
    wait_listed(&probe, &hub_addr, hub_key.id, 2, &procs).await;

    let pc = free_port();
    let cache = std::env::temp_dir().join(format!("dfin-e2e-dir-{}.cache", std::process::id()));
    let _ = std::fs::remove_file(&cache);
    let args: Vec<String> = [
        "client",
        "--model",
        &model,
        "--head",
        "0:2",
        "--tail",
        "6:end",
        "--hub",
        &hub_flag,
        "--directory-cache",
        &cache.display().to_string(),
        "--n-gpu-layers",
        "0",
        "--act-dtype",
        "f32",
        "--max-sessions",
        "2",
        "--n-ctx",
        "512",
        "--listen",
        &format!("127.0.0.1:{pc}"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let ci = procs.spawn("client", &args);
    let base = format!("http://127.0.0.1:{pc}");
    procs.wait_health(ci, &base, Duration::from_secs(120)).await;
    exercise(&procs, &base, &expected).await;

    let status: Value = reqwest::get(format!("{base}/status")).await.unwrap().json().await.unwrap();
    assert_eq!(status["route"].as_array().unwrap().len(), 4, "{status:#}");
    assert_eq!(status["directory"]["nodes_usable"], 2, "{status:#}");
    assert_eq!(status["directory"]["from_cache"], false, "{status:#}");
    assert!(cache.exists(), "directory cache was not written");
    let first_hop = status["route"][1]["node_id"].as_str().unwrap().to_string();
    assert!(first_hop == k1.id.to_string(), "{status:#}");

    // A replacement for layers 2:4 joins, then the first stage dies.
    let s3 = spawn_stage(&mut procs, "stage3", "2:4", &k3, p3);
    procs.wait_stage(s3, p3, k3.id, Duration::from_secs(120)).await;
    wait_listed(&probe, &hub_addr, hub_key.id, 3, &procs).await;
    procs.kill(s1);

    let http = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut rerouted = false;
    while Instant::now() < deadline {
        let (code, body) = post(
            &http,
            format!("{base}/v1/completions"),
            json!({ "prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0 }),
        )
        .await;
        if code == 200 {
            let v: Value = serde_json::from_str(&body).unwrap();
            let text = v["choices"][0]["text"].as_str().unwrap_or("");
            assert_eq!(text, expected, "rerouted output differs");
            let status: Value = http.get(format!("{base}/status")).send().await.unwrap().json().await.unwrap();
            let now_first = status["route"][1]["node_id"].as_str().unwrap_or("").to_string();
            if now_first == k3.id.to_string() {
                assert!(status["stats"]["chain"]["rebuilds"].as_u64().unwrap() >= 1, "{status:#}");
                rerouted = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    assert!(rerouted, "client did not reroute to the replacement stage\n{}", procs.logs());
    let _ = std::fs::remove_file(&cache);
}
