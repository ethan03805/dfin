//! `dfin bench`: what does splitting a model cost on this machine?
//!
//! 1. monolithic: one full model, greedy decode;
//! 2. in-process split: head, middles and tail as separate pieces in this process,
//!    hidden states round-tripped through the wire encoding;
//! 3. local pod: real `dfin stage` processes plus a `dfin client` over TCP/HTTP.

use crate::pod::{self, PodOptions, SplitPlan};
use anyhow::Context as _;
use clap::Args;
use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode, Token};
use dfin_proto::{ActDtype, Activations};
use serde_json::Value;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Layer boundaries for the split modes, e.g. `2,5`.
    #[arg(long, value_delimiter = ',', default_values_t = [2, 5])]
    pub split: Vec<i32>,
    #[arg(long, default_value_t = 32)]
    pub prompt_tokens: usize,
    #[arg(long, default_value_t = 64)]
    pub gen_tokens: usize,
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub n_gpu_layers: i32,
    /// Regex of tensor names kept in CPU memory, e.g. `exps` for MoE experts (repeatable).
    #[arg(long = "cpu-tensors")]
    pub cpu_tensors: Vec<String>,
    #[arg(long)]
    pub threads: Option<i32>,
    #[arg(long, default_value = "f16")]
    pub act_dtype: ActDtype,
    /// First stage port for the pod mode.
    #[arg(long, default_value_t = 7101)]
    pub base_port: u16,
    /// Skip the local-pod mode.
    #[arg(long)]
    pub skip_pod: bool,
    /// Benchmark an already-running pipeline at this API base URL (e.g. `http://127.0.0.1:8080`)
    /// instead of spawning a local pod.
    #[arg(long)]
    pub api_url: Option<String>,
}

const SENTENCE: &str =
    "The quick brown fox jumps over the lazy dog while the patient owl watches from the old oak tree. ";

#[derive(Default)]
struct Row {
    mode: String,
    prefill_ms: Option<f64>,
    mean_ms: Option<f64>,
    p50_ms: Option<f64>,
    tok_s: Option<f64>,
    notes: String,
}

fn fmt_opt(v: Option<f64>, prec: usize) -> String {
    v.map(|x| format!("{x:.prec$}"))
        .unwrap_or_else(|| "-".into())
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn p50(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(s[s.len() / 2])
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn argmax(v: &[f32]) -> Token {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as Token
}

fn ctx_params(a: &BenchArgs, embeddings: bool) -> ContextParams {
    let need = a.prompt_tokens + a.gen_tokens + 16;
    let n_ctx = need.div_ceil(256).max(1) * 256;
    let mut p = ContextParams {
        n_ctx: n_ctx as u32,
        n_batch: a.prompt_tokens.max(512) as u32,
        n_ubatch: 512,
        n_seq_max: 1,
        embeddings,
        ..ContextParams::default()
    };
    if let Some(t) = a.threads {
        p.n_threads = t;
        p.n_threads_batch = t;
    }
    p
}

/// Repeat the sentence until it tokenizes to at least `n` tokens, then truncate.
/// Returns the tokens (with BOS if the model wants one) and the matching text.
fn build_prompt(model: &Model, n: usize) -> anyhow::Result<(Vec<Token>, String)> {
    let vocab = model.vocab();
    let mut text = String::new();
    let mut toks;
    loop {
        text.push_str(SENTENCE);
        toks = vocab.tokenize(&text, true, false)?;
        if toks.len() >= n {
            break;
        }
    }
    toks.truncate(n);
    let body: &[Token] = if vocab.add_bos() && toks.first() == Some(&vocab.bos()) {
        &toks[1..]
    } else {
        &toks
    };
    let prompt_text = vocab.detokenize(body, false, false);
    Ok((toks, prompt_text))
}

fn load(a: &BenchArgs, begin: i32, end: i32) -> anyhow::Result<Arc<Model>> {
    let p = ModelParams {
        n_gpu_layers: a.n_gpu_layers,
        cpu_tensor_patterns: a.cpu_tensors.clone(),
        ..ModelParams::range(begin, end)
    };
    Ok(Arc::new(Model::load(&a.model, &p).with_context(|| {
        format!("loading layers {begin}:{end}")
    })?))
}

/// Mode 1. Returns the row and the generated tokens.
fn bench_mono(a: &BenchArgs, prompt: &[Token]) -> anyhow::Result<(Row, Vec<Token>)> {
    let model = load(a, 0, -1)?;
    let mut ctx = Context::new(model, &ctx_params(a, false))?;
    // Untimed warm-up (thread pools, weight repacking, shader compilation).
    ctx.decode_tokens(prompt, 0, 0, OutputMode::Last)?;
    ctx.clear();
    let t = Instant::now();
    ctx.decode_tokens(prompt, 0, 0, OutputMode::Last)?;
    let prefill = ms(t);
    let mut out = vec![argmax(ctx.logits_ith(-1))];
    let mut steps = Vec::new();
    let mut pos = prompt.len() as i32;
    while out.len() < a.gen_tokens {
        let last = *out.last().unwrap();
        let t = Instant::now();
        ctx.decode_tokens(&[last], pos, 0, OutputMode::Last)?;
        steps.push(ms(t));
        out.push(argmax(ctx.logits_ith(-1)));
        pos += 1;
    }
    let m = mean(&steps);
    Ok((
        Row {
            mode: "1 monolithic".into(),
            prefill_ms: Some(prefill),
            mean_ms: m,
            p50_ms: p50(&steps),
            tok_s: m.map(|x| 1000.0 / x),
            notes: format!("{} prompt tok, {} generated", prompt.len(), out.len()),
        },
        out,
    ))
}

struct Piece {
    name: String,
    ctx: Context,
    /// Per-step decode times (generation steps only).
    steps: Vec<f64>,
}

/// One pass through every piece; returns the next token. `times` collects
/// per-piece decode ms and the summed conversion ms.
fn split_step(
    pieces: &mut [Piece],
    tokens: &[Token],
    pos: i32,
    dtype: ActDtype,
    record: bool,
    conv_ms: &mut f64,
) -> anyhow::Result<Token> {
    let n = tokens.len();
    let last = pieces.len() - 1;
    let mut hidden: Vec<f32> = Vec::new();
    for (i, p) in pieces.iter_mut().enumerate() {
        let out_mode = if i == last {
            OutputMode::Last
        } else {
            OutputMode::All
        };
        let t = Instant::now();
        if i == 0 {
            p.ctx.decode_tokens(tokens, pos, 0, out_mode)?;
        } else {
            p.ctx.decode_embd(&hidden, pos, 0, out_mode)?;
        }
        let dt = ms(t);
        if record {
            p.steps.push(dt);
        }
        if i == last {
            return Ok(argmax(p.ctx.logits_ith(-1)));
        }
        // Hop to the next piece through the wire encoding.
        let t = Instant::now();
        let n_embd = p.ctx.n_embd();
        let act = Activations::from_f32(p.ctx.embeddings(), n, n_embd, dtype);
        hidden = act.to_f32()?;
        *conv_ms += ms(t);
    }
    unreachable!("split has a tail piece")
}

/// Mode 2.
fn bench_split(
    a: &BenchArgs,
    plan: &SplitPlan,
    prompt: &[Token],
    reference: &[Token],
) -> anyhow::Result<Row> {
    let mut row = Row {
        mode: "2 in-process split".into(),
        ..Row::default()
    };
    if !dfin_llama::has_layer_range() {
        row.notes = "skipped: libllama lacks the layer-range patch".into();
        return Ok(row);
    }
    let mut pieces = Vec::new();
    let head = load(a, plan.head.begin, plan.head.end)?;
    pieces.push(Piece {
        name: format!("head {}", plan.head),
        ctx: Context::new(head, &ctx_params(a, true))?,
        steps: Vec::new(),
    });
    for r in &plan.stages {
        let m = load(a, r.begin, r.end)?;
        pieces.push(Piece {
            name: format!("mid {r}"),
            ctx: Context::new(m, &ctx_params(a, true))?,
            steps: Vec::new(),
        });
    }
    let tail = load(a, plan.tail.begin, -1)?;
    pieces.push(Piece {
        name: format!("tail {}:end", plan.tail.begin),
        ctx: Context::new(tail, &ctx_params(a, false))?,
        steps: Vec::new(),
    });

    let mut conv = 0.0;
    split_step(&mut pieces, prompt, 0, a.act_dtype, false, &mut conv)?; // untimed warm-up
    for p in &mut pieces {
        p.ctx.clear();
    }
    let t = Instant::now();
    let mut out = vec![split_step(
        &mut pieces,
        prompt,
        0,
        a.act_dtype,
        false,
        &mut conv,
    )?];
    let prefill = ms(t);
    let mut conv = 0.0;
    let mut steps = Vec::new();
    let mut pos = prompt.len() as i32;
    while out.len() < a.gen_tokens {
        let last = *out.last().unwrap();
        let t = Instant::now();
        let next = split_step(&mut pieces, &[last], pos, a.act_dtype, true, &mut conv)?;
        steps.push(ms(t));
        out.push(next);
        pos += 1;
    }
    let m = mean(&steps);
    let n_steps = steps.len().max(1) as f64;
    let breakdown: Vec<String> = pieces
        .iter()
        .map(|p| format!("{} {}", p.name, fmt_opt(mean(&p.steps), 3)))
        .chain(std::iter::once(format!("conv {:.3}", conv / n_steps)))
        .collect();
    let differ = out.iter().zip(reference).filter(|(x, y)| x != y).count()
        + out.len().abs_diff(reference.len());
    let verdict = if differ == 0 {
        "tokens == mono".to_string()
    } else {
        format!("{differ}/{} tokens differ from mono", reference.len())
    };
    row.prefill_ms = Some(prefill);
    row.mean_ms = m;
    row.p50_ms = p50(&steps);
    row.tok_s = m.map(|x| 1000.0 / x);
    row.notes = format!("{verdict}; ms/tok: {}", breakdown.join(", "));
    Ok(row)
}

fn free_local_port() -> anyhow::Result<u16> {
    let l = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    Ok(l.local_addr()?.port())
}

/// Mode 3. Never fails: problems end up in the notes column.
async fn bench_pod(a: &BenchArgs, prompt_text: &str, mono_text: &str) -> Row {
    let mut row = Row {
        mode: if a.api_url.is_some() {
            "3 remote pipeline".into()
        } else {
            "3 local pod (TCP)".into()
        },
        ..Row::default()
    };
    match bench_pod_inner(a, prompt_text, mono_text, &mut row).await {
        Ok(()) => {}
        Err(e) => {
            let msg = format!("{e:#}");
            row.notes = if row.notes.is_empty() {
                format!("error: {msg}")
            } else {
                format!("{}; error: {msg}", row.notes)
            };
        }
    }
    row
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

async fn bench_pod_inner(
    a: &BenchArgs,
    prompt_text: &str,
    mono_text: &str,
    row: &mut Row,
) -> anyhow::Result<()> {
    let (_pod, api_url) = if let Some(url) = &a.api_url {
        (None, url.trim_end_matches('/').to_string())
    } else {
        let need = a.prompt_tokens + a.gen_tokens + 64;
        let n_ctx = (need.div_ceil(256) * 256).max(1024) as u32;
        let opts = PodOptions {
            model: a.model.clone(),
            split: a.split.clone(),
            listen: SocketAddr::new(Ipv4Addr::LOCALHOST.into(), free_local_port()?),
            base_port: a.base_port,
            n_ctx,
            n_seq_max: 1,
            n_gpu_layers: a.n_gpu_layers,
            cpu_tensors: a.cpu_tensors.clone(),
            threads: a.threads,
            act_dtype: a.act_dtype,
            exe: None,
            log: Some("warn".into()),
            ready_timeout: Duration::from_secs(120),
        };
        let pod = pod::spawn_pod(opts).await.context("starting pod")?;
        let url = pod.api_url.clone();
        (Some(pod), url)
    };
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let body = serde_json::json!({
        "model": "bench",
        "prompt": prompt_text,
        "max_tokens": a.gen_tokens,
        "temperature": 0,
        "stream": false,
    });
    let t = Instant::now();
    let resp = http
        .post(format!("{}/v1/completions", api_url))
        .json(&body)
        .send()
        .await
        .context("POST /v1/completions")?;
    let total = ms(t);
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!(
            "HTTP {status}: {}",
            text.chars().take(300).collect::<String>()
        );
    }
    let v: Value = serde_json::from_str(&text).context("parsing completion response")?;
    let n = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let out_text = v["choices"][0]["text"].as_str().unwrap_or("");
    let mut notes = vec![format!("request {total:.1} ms, {n} completion tok")];
    if !out_text.is_empty() && mono_text.starts_with(out_text) {
        notes.push("text == mono prefix".into());
    }

    match http.get(format!("{}/status", api_url)).send().await {
        Ok(r) if r.status().is_success() => {
            let s: Value = r.json().await.unwrap_or(Value::Null);
            let stats = &s["stats"];
            let avg_tok = get_f64(stats, "avg_token_ms");
            row.mean_ms = avg_tok.or_else(|| (n > 0).then(|| total / n as f64));
            row.prefill_ms = get_f64(stats, "prefill_ms")
                .or_else(|| get_f64(stats, "last_prefill_ms"))
                .or_else(|| get_f64(stats, "avg_prefill_ms"));
            if let Some(x) = get_f64(stats, "head_ms") {
                notes.push(format!("head {x:.3}"));
            }
            for (i, st) in stats["stages"].as_array().into_iter().flatten().enumerate() {
                let rtt = get_f64(st, "avg_rtt_ms");
                let comp = get_f64(st, "avg_compute_ms");
                if rtt.is_some() || comp.is_some() {
                    notes.push(format!(
                        "stage{i} rtt {} compute {}",
                        fmt_opt(rtt, 3),
                        fmt_opt(comp, 3)
                    ));
                }
                let drtt = get_f64(st, "decode_avg_rtt_ms");
                if drtt.is_some() {
                    notes.push(format!(
                        "stage{i} decode-only rtt {} compute {} ping {}",
                        fmt_opt(drtt, 3),
                        fmt_opt(get_f64(st, "decode_avg_compute_ms"), 3),
                        fmt_opt(get_f64(st, "ping_ms"), 2)
                    ));
                }
            }
            if let Some(x) = get_f64(stats, "tail_ms") {
                notes.push(format!("tail {x:.3}"));
            }
            if avg_tok.is_none() {
                notes.push("no stats.avg_token_ms; ms/tok = request/n".into());
            }
        }
        Ok(r) => {
            row.mean_ms = (n > 0).then(|| total / n as f64);
            notes.push(format!("/status HTTP {}", r.status()));
        }
        Err(e) => {
            row.mean_ms = (n > 0).then(|| total / n as f64);
            notes.push(format!("/status failed: {e}"));
        }
    }
    row.tok_s = row.mean_ms.map(|x| 1000.0 / x);
    row.notes = notes.join("; ");
    if let Some(pod) = _pod {
        pod.shutdown().await;
    }
    Ok(())
}

fn print_table(rows: &[Row]) {
    let header = [
        "mode",
        "prefill ms",
        "ms/tok mean",
        "ms/tok p50",
        "tok/s",
        "notes",
    ];
    let cells: Vec<[String; 6]> = rows
        .iter()
        .map(|r| {
            [
                r.mode.clone(),
                fmt_opt(r.prefill_ms, 2),
                fmt_opt(r.mean_ms, 3),
                fmt_opt(r.p50_ms, 3),
                fmt_opt(r.tok_s, 1),
                r.notes.clone(),
            ]
        })
        .collect();
    let mut w = header.map(str::len);
    for c in &cells {
        for i in 0..5 {
            w[i] = w[i].max(c[i].len());
        }
    }
    let line = |c: &[String; 6]| {
        println!(
            "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} | {}",
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            w0 = w[0],
            w1 = w[1],
            w2 = w[2],
            w3 = w[3],
            w4 = w[4]
        )
    };
    line(&header.map(String::from));
    println!(
        "|{}|{}|{}|{}|{}|---",
        "-".repeat(w[0] + 2),
        "-".repeat(w[1] + 2),
        "-".repeat(w[2] + 2),
        "-".repeat(w[3] + 2),
        "-".repeat(w[4] + 2)
    );
    for c in &cells {
        line(c);
    }
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    if args.prompt_tokens == 0 || args.gen_tokens == 0 {
        anyhow::bail!("--prompt-tokens and --gen-tokens must be at least 1");
    }
    dfin_llama::init();
    println!("system: {}", dfin_llama::system_info().trim());
    for d in dfin_llama::devices() {
        println!(
            "device: {} | {} | {} | {} / {} MiB free",
            d.name,
            d.description,
            if d.is_gpu { "gpu" } else { "cpu" },
            d.memory_free >> 20,
            d.memory_total >> 20
        );
    }
    println!(
        "layer-range patch: {} | split {:?} | act dtype {} | n_gpu_layers {}",
        dfin_llama::has_layer_range(),
        args.split,
        pod::act_dtype_str(args.act_dtype),
        args.n_gpu_layers
    );

    let a = args.clone();
    let (rows, prompt_text, mono_text) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let n_layer = pod::model_n_layer(&a.model)?;
            let plan = pod::plan_split(&a.split, n_layer)?;
            let vocab_model = Model::load(
                &a.model,
                &ModelParams {
                    vocab_only: true,
                    ..ModelParams::default()
                },
            )?;
            let (prompt, prompt_text) = build_prompt(&vocab_model, a.prompt_tokens)?;
            println!(
                "model: {} layers | plan: head {} | stages {:?} | tail {}:end | prompt {} tok",
                n_layer,
                plan.head,
                plan.stages
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>(),
                plan.tail.begin,
                prompt.len()
            );
            let (mono, mono_tokens) = bench_mono(&a, &prompt)?;
            let mono_text = vocab_model.vocab().detokenize(&mono_tokens, false, false);
            let split = bench_split(&a, &plan, &prompt, &mono_tokens).unwrap_or_else(|e| Row {
                mode: "2 in-process split".into(),
                notes: format!("error: {e:#}"),
                ..Row::default()
            });
            Ok((vec![mono, split], prompt_text, mono_text))
        })
        .await??;
    let mut rows = rows;

    if args.skip_pod {
        rows.push(Row {
            mode: "3 local pod (TCP)".into(),
            notes: "skipped (--skip-pod)".into(),
            ..Row::default()
        });
    } else {
        rows.push(bench_pod(&args, &prompt_text, &mono_text).await);
    }

    println!();
    print_table(&rows);
    Ok(())
}
