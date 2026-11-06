//! Exercises the wrapper on the tiny random-weight Llama GGUF: full-model decode,
//! and a head / middle / tail split whose logits must match the full model.
//!
//! Model path: `DFIN_TEST_MODEL` or `<workspace>/models/tiny-llama-f32.gguf`.

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use std::path::PathBuf;
use std::sync::Arc;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DFIN_TEST_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tiny-llama-f32.gguf")
}

fn cpu_params(begin: i32, end: i32) -> ModelParams {
    ModelParams {
        n_gpu_layers: 0,
        ..ModelParams::range(begin, end)
    }
}

fn ctx_params(embeddings: bool) -> ContextParams {
    ContextParams {
        n_ctx: 256,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 2,
        embeddings,
        ..ContextParams::default()
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn full_model_decodes() {
    dfin_llama::init();
    let model = Arc::new(Model::load(&model_path(), &cpu_params(0, -1)).expect("load full model"));
    assert_eq!(model.n_layer(), 8);
    assert_eq!(model.n_embd(), 64);
    assert!(model.has_head() && model.has_tail() && !model.is_partial());
    let vocab = model.vocab();
    assert_eq!(vocab.n_tokens(), 768);

    let tokens = vocab.tokenize("hello world the model network node", true, false).unwrap();
    assert!(tokens.len() >= 4, "tokens: {tokens:?}");
    assert_eq!(tokens[0], vocab.bos());

    let mut ctx = Context::new(model.clone(), &ctx_params(false)).unwrap();
    let n_out = ctx.decode_tokens(&tokens, 0, 0, OutputMode::Last).unwrap();
    assert_eq!(n_out, 1);
    let logits = ctx.logits_ith(-1).to_vec();
    assert_eq!(logits.len(), 768);
    assert!(logits.iter().all(|x| x.is_finite()));

    // Deterministic: a second context gives identical logits.
    let mut ctx2 = Context::new(model.clone(), &ctx_params(false)).unwrap();
    ctx2.decode_tokens(&tokens, 0, 0, OutputMode::Last).unwrap();
    assert_eq!(max_abs_diff(&logits, ctx2.logits_ith(-1)), 0.0);

    // Incremental decode and KV rollback.
    let next = argmax(&logits) as i32;
    ctx.decode_tokens(&[next], tokens.len() as i32, 0, OutputMode::Last).unwrap();
    let step_logits = ctx.logits_ith(-1).to_vec();
    assert!(ctx.seq_rm(0, tokens.len() as i32, -1));
    assert_eq!(ctx.seq_pos_max(0), tokens.len() as i32 - 1);
    ctx.decode_tokens(&[next], tokens.len() as i32, 0, OutputMode::Last).unwrap();
    assert_eq!(max_abs_diff(&step_logits, ctx.logits_ith(-1)), 0.0);

    // Chat template fallback works even though the GGUF has none.
    assert!(model.chat_template().is_none());
    let rendered = dfin_llama::chat_apply_template(
        "chatml",
        &[dfin_llama::ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }],
        true,
    )
    .unwrap();
    assert!(rendered.contains("<|im_start|>user"), "{rendered}");
}

#[test]
fn split_pipeline_matches_full_model() {
    dfin_llama::init();
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: vendored libllama lacks the layer-range patch");
        return;
    }
    let path = model_path();
    let full = Arc::new(Model::load(&path, &cpu_params(0, -1)).unwrap());
    let head = Arc::new(Model::load(&path, &cpu_params(0, 3)).unwrap());
    let mid = Arc::new(Model::load(&path, &cpu_params(3, 6)).unwrap());
    let tail = Arc::new(Model::load(&path, &cpu_params(6, -1)).unwrap());

    assert_eq!(head.layer_range(), (0, 3));
    assert_eq!(mid.layer_range(), (3, 6));
    assert_eq!(tail.layer_range(), (6, 8));
    assert!(head.has_head() && !head.has_tail());
    assert!(!mid.has_head() && !mid.has_tail());
    assert!(!tail.has_head() && tail.has_tail());
    assert!(head.size_bytes() < full.size_bytes());
    assert!(mid.size_bytes() < full.size_bytes());
    assert!(tail.size_bytes() < full.size_bytes());
    eprintln!(
        "sizes: full {} head {} mid {} tail {}",
        full.size_bytes(),
        head.size_bytes(),
        mid.size_bytes(),
        tail.size_bytes()
    );

    let vocab = full.vocab();
    let tokens = vocab.tokenize("hello world the model network node", true, false).unwrap();
    let n = tokens.len();
    let n_embd = full.n_embd() as usize;

    let mut full_ctx = Context::new(full.clone(), &ctx_params(false)).unwrap();
    let mut head_ctx = Context::new(head.clone(), &ctx_params(true)).unwrap();
    let mut mid_ctx = Context::new(mid.clone(), &ctx_params(true)).unwrap();
    let mut tail_ctx = Context::new(tail.clone(), &ctx_params(false)).unwrap();

    // Prefill: every piece outputs all rows except the tail.
    full_ctx.decode_tokens(&tokens, 0, 0, OutputMode::All).unwrap();
    let full_logits = full_ctx.logits().to_vec();
    assert_eq!(full_logits.len(), n * 768);

    let n_out = head_ctx.decode_tokens(&tokens, 0, 0, OutputMode::All).unwrap();
    assert_eq!(n_out, n);
    let h1 = head_ctx.embeddings().to_vec();
    assert_eq!(h1.len(), n * n_embd, "head must emit one hidden row per token");
    assert!(h1.iter().all(|x| x.is_finite()));

    mid_ctx.decode_embd(&h1, 0, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();
    assert_eq!(h2.len(), n * n_embd);
    assert_ne!(h1, h2);

    tail_ctx.decode_embd(&h2, 0, 0, OutputMode::All).unwrap();
    let pipe_logits = tail_ctx.logits().to_vec();
    assert_eq!(pipe_logits.len(), n * 768);

    let diff = max_abs_diff(&full_logits, &pipe_logits);
    eprintln!("prefill max abs logit diff: {diff}");
    assert!(diff < 1e-3, "pipelined logits diverge from full model: {diff}");
    for t in 0..n {
        assert_eq!(
            argmax(&full_logits[t * 768..(t + 1) * 768]),
            argmax(&pipe_logits[t * 768..(t + 1) * 768]),
            "argmax differs at position {t}"
        );
    }

    // Decode one more token through both paths; this exercises the partial KV caches.
    let next = argmax(&full_logits[(n - 1) * 768..]) as i32;
    full_ctx.decode_tokens(&[next], n as i32, 0, OutputMode::Last).unwrap();
    let full_step = full_ctx.logits_ith(-1).to_vec();

    head_ctx.decode_tokens(&[next], n as i32, 0, OutputMode::All).unwrap();
    let h1 = head_ctx.embeddings().to_vec();
    assert_eq!(h1.len(), n_embd);
    mid_ctx.decode_embd(&h1, n as i32, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();
    tail_ctx.decode_embd(&h2, n as i32, 0, OutputMode::Last).unwrap();
    let pipe_step = tail_ctx.logits_ith(-1).to_vec();
    let diff = max_abs_diff(&full_step, &pipe_step);
    eprintln!("decode-step max abs logit diff: {diff}");
    assert!(diff < 1e-3, "pipelined decode step diverges: {diff}");

    // Roll the pipeline back one token and redo the step: same answer.
    assert!(head_ctx.seq_rm(0, n as i32, -1));
    assert!(mid_ctx.seq_rm(0, n as i32, -1));
    assert!(tail_ctx.seq_rm(0, n as i32, -1));
    assert_eq!(mid_ctx.seq_pos_max(0), n as i32 - 1);
    head_ctx.decode_tokens(&[next], n as i32, 0, OutputMode::All).unwrap();
    let h1 = head_ctx.embeddings().to_vec();
    mid_ctx.decode_embd(&h1, n as i32, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();
    tail_ctx.decode_embd(&h2, n as i32, 0, OutputMode::Last).unwrap();
    assert_eq!(max_abs_diff(&pipe_step, tail_ctx.logits_ith(-1)), 0.0);
}
