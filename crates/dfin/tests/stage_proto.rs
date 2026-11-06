//! End-to-end check of `dfin stage`: a head and tail in this process, the middle
//! layers in a spawned stage server, compared against a full in-process model.

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use dfin_proto::{read_frame, write_frame, ActDtype, Activations, ErrorCode, Request, Response, PROTOCOL_VERSION};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;

const MID: (i32, i32) = (3, 6);

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DFIN_TEST_MODEL") {
        return p.into();
    }
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/tiny-llama-f32.gguf"))
}

/// Kills the stage process even if the test panics.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    l.local_addr().unwrap().port()
}

async fn wait_for_port(addr: SocketAddr, child: &mut ChildGuard) {
    let start = Instant::now();
    loop {
        if let Some(st) = child.0.try_wait().unwrap() {
            panic!("stage exited early: {st}");
        }
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        assert!(start.elapsed() < Duration::from_secs(60), "stage did not start listening");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn call(s: &mut TcpStream, req: &Request) -> Response {
    write_frame(s, req).await.unwrap();
    read_frame(s).await.unwrap()
}

async fn hello(addr: SocketAddr, fingerprint: &str, range: (i32, i32)) -> (TcpStream, Response) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.set_nodelay(true).unwrap();
    let r = call(
        &mut s,
        &Request::Hello {
            version: PROTOCOL_VERSION,
            model_fingerprint: fingerprint.into(),
            layer_begin: range.0,
            layer_end: range.1,
        },
    )
    .await;
    (s, r)
}

fn cpu_piece(path: &std::path::Path, begin: i32, end: i32, embeddings: bool) -> Context {
    let m = Model::load(
        path,
        &ModelParams {
            n_gpu_layers: 0,
            ..ModelParams::range(begin, end)
        },
    )
    .unwrap();
    Context::new(
        Arc::new(m),
        &ContextParams {
            n_ctx: 512,
            embeddings,
            ..ContextParams::default()
        },
    )
    .unwrap()
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[b] {
            b = i;
        }
    }
    b
}

fn forward_ok(r: Response) -> (u64, u64, Activations, f32) {
    match r {
        Response::ForwardOk {
            session,
            seq,
            act,
            compute_ms,
        } => (session, seq, act, compute_ms),
        other => panic!("expected ForwardOk, got {other:?}"),
    }
}

#[tokio::test]
async fn stage_matches_full_model() {
    let path = model_path();
    if !path.exists() {
        eprintln!("SKIP: test model {} not found", path.display());
        return;
    }
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: libllama lacks the layer-range patch");
        return;
    }
    dfin_llama::init();

    let port = free_port();
    let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
    let child = Command::new(env!("CARGO_BIN_EXE_dfin"))
        .args(["--log", "warn", "stage", "--model"])
        .arg(&path)
        .args(["--layers", &format!("{}:{}", MID.0, MID.1)])
        .args(["--listen", &addr.to_string()])
        .args(["--n-gpu-layers", "0", "--act-dtype", "f32", "--n-ctx", "1024", "--n-seq-max", "2"])
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn dfin stage");
    let mut child = ChildGuard(child);
    wait_for_port(addr, &mut child).await;

    let fp = dfin_proto::model_fingerprint(&path).unwrap();
    let (mut s, r) = hello(addr, &fp, MID).await;
    let n_embd = match r {
        Response::HelloOk {
            layer_begin,
            layer_end,
            n_embd,
            n_ctx,
            n_seq_max,
            backend,
        } => {
            eprintln!("HelloOk: layers {layer_begin}:{layer_end} n_embd {n_embd} n_ctx {n_ctx} n_seq_max {n_seq_max} backend {backend:?}");
            assert_eq!((layer_begin, layer_end), MID);
            assert_eq!(n_ctx, 512);
            assert_eq!(n_seq_max, 2);
            n_embd as usize
        }
        other => panic!("expected HelloOk, got {other:?}"),
    };

    // Reference: full model logits for every prompt position.
    let mut full = cpu_piece(&path, 0, -1, false);
    let tokens = full
        .model()
        .vocab()
        .tokenize("The quick brown fox jumps over the lazy dog.", true, false)
        .unwrap();
    let n = tokens.len();
    assert!(n >= 4, "prompt too short: {n}");
    full.decode_tokens(&tokens, 0, 0, OutputMode::All).unwrap();
    let ref_logits = full.logits().to_vec();
    let n_vocab = full.n_vocab();
    assert_eq!(ref_logits.len(), n * n_vocab);

    // Head locally.
    let mut head = cpu_piece(&path, 0, MID.0, true);
    head.decode_tokens(&tokens, 0, 0, OutputMode::All).unwrap();
    let h = head.embeddings().to_vec();
    assert_eq!(h.len(), n * n_embd);

    // Middle on the stage.
    let session = 42u64;
    let r = call(
        &mut s,
        &Request::Forward {
            session,
            seq: 1,
            pos0: 0,
            act: Activations::from_f32(&h, n, n_embd, ActDtype::F32),
        },
    )
    .await;
    let (rs, rseq, act, compute_ms) = forward_ok(r);
    assert_eq!((rs, rseq), (session, 1));
    assert_eq!(act.n_tokens as usize, n);
    eprintln!("prefill forward: {n} tokens, {} bytes, {compute_ms:.3} ms", act.byte_len());
    let mid_out = act.to_f32().unwrap();

    // Tail locally.
    let mut tail = cpu_piece(&path, MID.1, -1, false);
    tail.decode_embd(&mid_out, 0, 0, OutputMode::All).unwrap();
    let logits = tail.logits().to_vec();
    assert_eq!(logits.len(), ref_logits.len());
    let max_diff = logits
        .iter()
        .zip(&ref_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| over {n} positions: {max_diff:e}");
    assert!(max_diff < 1e-3, "max abs logit diff {max_diff}");
    for i in 0..n {
        let a = argmax(&logits[i * n_vocab..(i + 1) * n_vocab]);
        let b = argmax(&ref_logits[i * n_vocab..(i + 1) * n_vocab]);
        assert_eq!(a, b, "argmax differs at position {i}");
    }

    // Rollback: one token at pos n, then the same again (stage must roll back itself).
    let one = Activations::from_f32(&h[(n - 1) * n_embd..], 1, n_embd, ActDtype::F32);
    let r1 = forward_ok(
        call(
            &mut s,
            &Request::Forward {
                session,
                seq: 2,
                pos0: n as i32,
                act: one.clone(),
            },
        )
        .await,
    );
    let r2 = forward_ok(
        call(
            &mut s,
            &Request::Forward {
                session,
                seq: 3,
                pos0: n as i32,
                act: one.clone(),
            },
        )
        .await,
    );
    assert_eq!(r2.1, 3);
    assert_eq!(r1.2.data, r2.2.data, "rolled-back forward must reproduce the same output");

    // Position gap is refused.
    match call(
        &mut s,
        &Request::Forward {
            session,
            seq: 4,
            pos0: n as i32 + 5,
            act: one.clone(),
        },
    )
    .await
    {
        Response::Error { code, message } => {
            assert_eq!(code, ErrorCode::BadRequest);
            assert!(message.contains("gap"), "{message}");
        }
        other => panic!("expected position-gap error, got {other:?}"),
    }

    // Wrong activation width is refused.
    let bad = Activations::from_f32(&vec![0.0; n_embd + 1], 1, n_embd + 1, ActDtype::F32);
    match call(&mut s, &Request::Forward { session, seq: 5, pos0: 0, act: bad }).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::BadRequest),
        other => panic!("expected BadRequest, got {other:?}"),
    }

    // Ping, SeqRm, SessionClose.
    assert!(matches!(call(&mut s, &Request::Ping).await, Response::Pong));
    let r = call(&mut s, &Request::SeqRm { session, p0: n as i32, p1: -1 }).await;
    assert!(matches!(r, Response::Ok), "SeqRm: {r:?}");
    match call(&mut s, &Request::SeqRm { session: 999, p0: 0, p1: -1 }).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::UnknownSession),
        other => panic!("expected UnknownSession, got {other:?}"),
    }
    let r = call(&mut s, &Request::SessionClose { session }).await;
    assert!(matches!(r, Response::Ok), "SessionClose: {r:?}");

    // Session limit: two slots, a third session is refused.
    for (i, sess) in [100u64, 101, 102].into_iter().enumerate() {
        let r = call(
            &mut s,
            &Request::Forward {
                session: sess,
                seq: 1,
                pos0: 0,
                act: one.clone(),
            },
        )
        .await;
        if i < 2 {
            forward_ok(r);
        } else {
            match r {
                Response::Error { code, .. } => assert_eq!(code, ErrorCode::SessionLimit),
                other => panic!("expected SessionLimit, got {other:?}"),
            }
        }
    }
    // Closing the connection frees its slots for the next one.
    drop(s);
    let (mut s2, r) = hello(addr, &fp, MID).await;
    assert!(matches!(r, Response::HelloOk { .. }));
    let mut freed = false;
    for _ in 0..50 {
        let r = call(
            &mut s2,
            &Request::Forward {
                session: 7,
                seq: 1,
                pos0: 0,
                act: one.clone(),
            },
        )
        .await;
        if matches!(r, Response::ForwardOk { .. }) {
            freed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(freed, "slots of a closed connection were not freed");

    // Refusals.
    let (_s, r) = hello(addr, "0000", MID).await;
    match r {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::ModelMismatch),
        other => panic!("expected ModelMismatch, got {other:?}"),
    }
    let (_s, r) = hello(addr, &fp, (2, 6)).await;
    match r {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::LayerRangeMismatch),
        other => panic!("expected LayerRangeMismatch, got {other:?}"),
    }
    let mut s3 = TcpStream::connect(addr).await.unwrap();
    match call(&mut s3, &Request::Ping).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::BadRequest),
        other => panic!("expected BadRequest for missing Hello, got {other:?}"),
    }
    drop(child);
}
