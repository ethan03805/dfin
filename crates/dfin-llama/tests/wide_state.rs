//! Architectures whose state between layers is wider than one residual vector: a
//! three-piece pipeline must reproduce the whole model's logits exactly, and the pieces
//! must report the widths the architecture implies.
//!
//! Models: `tools/make_tiny_kimi_k3.py models/tiny-kimi-k3-res.gguf --attn-res-block-size 2`
//! and `tools/make_tiny_glm5_next.py models/tiny-glm5-next.gguf` (skips when absent).

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
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
        n_seq_max: 1,
        embeddings,
        ..ContextParams::default()
    }
}

fn ready(path: &Path) -> bool {
    if !path.exists() {
        eprintln!("SKIP: {} not found", path.display());
        return false;
    }
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: libllama lacks the layer-range patch");
        return false;
    }
    dfin_llama::init();
    true
}

/// Run the model whole and as head `0:a`, middle `a:b`, tail `b:end`; compare logits.
/// `widths` are the expected (head out, mid in, mid out, tail in) row widths.
fn check(path: &Path, a: i32, b: i32, widths: (usize, usize, usize, usize)) {
    let full = Arc::new(Model::load(path, &cpu_params(0, -1)).unwrap());
    let n_embd = full.n_embd() as usize;
    assert_eq!(full.n_embd_inp() as usize, n_embd);
    assert_eq!(full.n_embd_out() as usize, n_embd);
    let tokens = full
        .vocab()
        .tokenize("hello world the model network node", true, false)
        .unwrap();
    let n = tokens.len();
    let mut full_ctx = Context::new(full.clone(), &ctx_params(false)).unwrap();
    full_ctx
        .decode_tokens(&tokens, 0, 0, OutputMode::All)
        .unwrap();
    let full_logits = full_ctx.logits().to_vec();
    let n_vocab = full_ctx.n_vocab();

    let head = Arc::new(Model::load(path, &cpu_params(0, a)).unwrap());
    let mid = Arc::new(Model::load(path, &cpu_params(a, b)).unwrap());
    let tail = Arc::new(Model::load(path, &cpu_params(b, -1)).unwrap());
    let got = (
        head.n_embd_out() as usize,
        mid.n_embd_inp() as usize,
        mid.n_embd_out() as usize,
        tail.n_embd_inp() as usize,
    );
    eprintln!("widths (head out, mid in, mid out, tail in) = {got:?}, n_embd {n_embd}");
    assert_eq!(got, widths, "state widths differ from what the architecture implies");

    let mut head_ctx = Context::new(head.clone(), &ctx_params(true)).unwrap();
    head_ctx
        .decode_tokens(&tokens, 0, 0, OutputMode::All)
        .unwrap();
    let h1 = head_ctx.embeddings().to_vec();
    assert_eq!(h1.len(), n * widths.0);

    let mut mid_ctx = Context::new(mid.clone(), &ctx_params(true)).unwrap();
    mid_ctx.decode_embd(&h1, 0, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();
    assert_eq!(h2.len(), n * widths.2);

    let mut tail_ctx = Context::new(tail.clone(), &ctx_params(false)).unwrap();
    tail_ctx.decode_embd(&h2, 0, 0, OutputMode::All).unwrap();
    let pipe_logits = tail_ctx.logits().to_vec();
    assert_eq!(pipe_logits.len(), full_logits.len());

    let diff = pipe_logits
        .iter()
        .zip(&full_logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| over {n} positions: {diff:e}");
    // Same tolerance as the Phase 0 tests: assembling the wider state in pieces reorders
    // float additions, so a few parts in ten thousand is reassociation, not error.
    assert!(diff < 1e-3, "pipeline differs from the full model: {diff}");
    for i in 0..n {
        let am = pipe_logits[i * n_vocab..(i + 1) * n_vocab]
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0;
        let bm = full_logits[i * n_vocab..(i + 1) * n_vocab]
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0;
        assert_eq!(am, bm, "argmax differs at {i}");
    }

    // A second decode step (one token at the next position) must also match.
    let next = tokens[n - 1];
    full_ctx
        .decode_tokens(&[next], n as i32, 0, OutputMode::Last)
        .unwrap();
    let full_last = full_ctx.logits_ith(-1).to_vec();
    head_ctx
        .decode_tokens(&[next], n as i32, 0, OutputMode::All)
        .unwrap();
    let h1 = head_ctx.embeddings().to_vec();
    mid_ctx.decode_embd(&h1, n as i32, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();
    tail_ctx
        .decode_embd(&h2, n as i32, 0, OutputMode::Last)
        .unwrap();
    let pipe_last = tail_ctx.logits_ith(-1).to_vec();
    let diff = pipe_last
        .iter()
        .zip(&full_last)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("next-token max |logit diff|: {diff:e}");
    assert!(diff < 1e-3, "second step differs: {diff}");
}

#[test]
fn kimi_k3_with_attention_residuals_splits_exactly() {
    let path = models_dir().join("tiny-kimi-k3-res.gguf");
    if !ready(&path) {
        return;
    }
    // 8 layers, checkpoints at layers 0, 2, 4, 6. Before layer 2 the bank holds one
    // vector (plus the prefix sum: 2 x n_embd); before layer 5 it holds three (4 x n_embd).
    let n_embd = 64;
    check(&path, 2, 5, (2 * n_embd, 2 * n_embd, 4 * n_embd, 4 * n_embd));
    // A boundary right after a checkpoint layer: before layer 3 the bank holds two.
    check(&path, 3, 6, (3 * n_embd, 3 * n_embd, 4 * n_embd, 4 * n_embd));
}

#[test]
fn glm5_next_hyper_connections_split_exactly() {
    let path = models_dir().join("tiny-glm5-next.gguf");
    if !ready(&path) {
        return;
    }
    let n_embd = 64;
    check(&path, 2, 5, (4 * n_embd, 4 * n_embd, 4 * n_embd, 4 * n_embd));
}
