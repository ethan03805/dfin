//! Loading pieces of a model from partial shard directories: a directory holding only
//! the first shard and the shards for one layer range must load that range exactly,
//! and a directory missing a needed shard must fail with a clear error.
//!
//! Needs `models/tiny-shards/` from `tools/shard_gguf.py models/tiny-llama-f32.gguf
//! --out models/tiny-shards` (skips otherwise).

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

const N_SPLIT: usize = 11;
const PREFIX: &str = "tiny-llama-f32";

fn shard_name(no: usize) -> String {
    format!("{PREFIX}-{:05}-of-{:05}.gguf", no + 1, N_SPLIT)
}

/// A temp dir holding only the named shards (by split number), with the first always.
fn partial(tag: &str, splits: &[usize]) -> PathBuf {
    let src = models_dir().join("tiny-shards");
    let dir = std::env::temp_dir().join(format!("dfin-shards-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for &no in std::iter::once(&0).chain(splits) {
        let name = shard_name(no);
        std::fs::copy(src.join(&name), dir.join(&name)).unwrap();
    }
    dir
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

fn first(dir: &Path) -> PathBuf {
    dir.join(shard_name(0))
}

fn ready() -> bool {
    let shards = models_dir().join("tiny-shards").join(shard_name(0));
    if !shards.exists() {
        eprintln!("SKIP: {} not found", shards.display());
        return false;
    }
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: libllama lacks the layer-range patch");
        return false;
    }
    dfin_llama::init();
    true
}

#[test]
fn pieces_from_partial_shard_dirs_match_the_full_model() {
    if !ready() {
        return;
    }
    let full_path = models_dir().join("tiny-llama-f32.gguf");
    let full = Arc::new(Model::load(&full_path, &cpu_params(0, -1)).unwrap());
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

    // Head 0:3 from meta + embedding + layers 0..3 (splits 1, 2, 3, 4).
    let head_dir = partial("head", &[1, 2, 3, 4]);
    let head = Arc::new(Model::load(&first(&head_dir), &cpu_params(0, 3)).unwrap());
    assert!(head.has_head() && !head.has_tail());
    let mut head_ctx = Context::new(head.clone(), &ctx_params(true)).unwrap();
    head_ctx
        .decode_tokens(&tokens, 0, 0, OutputMode::All)
        .unwrap();
    let h1 = head_ctx.embeddings().to_vec();

    // Middle 3:6 from meta + layers 3, 4, 5 only (splits 5, 6, 7): no embedding, no output.
    let mid_dir = partial("mid", &[5, 6, 7]);
    let mid = Arc::new(Model::load(&first(&mid_dir), &cpu_params(3, 6)).unwrap());
    assert!(!mid.has_head() && !mid.has_tail());
    assert_eq!(mid.layer_range(), (3, 6));
    let mut mid_ctx = Context::new(mid.clone(), &ctx_params(true)).unwrap();
    mid_ctx.decode_embd(&h1, 0, 0, OutputMode::All).unwrap();
    let h2 = mid_ctx.embeddings().to_vec();

    // Tail 6:end from meta + layers 6, 7 + output (splits 8, 9, 10).
    let tail_dir = partial("tail", &[8, 9, 10]);
    let tail = Arc::new(Model::load(&first(&tail_dir), &cpu_params(6, -1)).unwrap());
    assert!(!tail.has_head() && tail.has_tail());
    let mut tail_ctx = Context::new(tail.clone(), &ctx_params(false)).unwrap();
    tail_ctx.decode_embd(&h2, 0, 0, OutputMode::All).unwrap();
    let pipe_logits = tail_ctx.logits().to_vec();

    assert_eq!(pipe_logits.len(), full_logits.len());
    let diff = pipe_logits
        .iter()
        .zip(&full_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| from shards over {n} positions: {diff:e}");
    assert!(diff < 1e-4, "shard pipeline differs from the full model: {diff}");
    for i in 0..n {
        let a = &pipe_logits[i * n_vocab..(i + 1) * n_vocab];
        let b = &full_logits[i * n_vocab..(i + 1) * n_vocab];
        let am = a.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        let bm = b.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
        assert_eq!(am, bm, "argmax differs at {i}");
    }
    for d in [head_dir, mid_dir, tail_dir] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn a_missing_needed_shard_is_a_clear_error() {
    if !ready() {
        return;
    }
    // Layers 3:6 need splits 5, 6, 7; split 6 (layer 4) is absent.
    let dir = partial("missing", &[5, 7]);
    let err = match Model::load(&first(&dir), &cpu_params(3, 6)) {
        Ok(_) => panic!("loading with a missing shard must fail"),
        Err(e) => e.to_string(),
    };
    eprintln!("error: {err}");
    assert!(
        err.contains("shard") || err.contains("not found"),
        "unexpected error text: {err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
