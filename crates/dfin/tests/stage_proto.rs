//! Protocol-level checks of `dfin stage` over the QUIC transport: a head and tail in
//! this process, the middle layers in spawned stage servers, compared against a full
//! in-process model. Covers route setup with sealed records, both return modes, a
//! two-stage chain, rollback, and every refusal.

use dfin_llama::{Context, ContextParams, Model, ModelParams, OutputMode};
use dfin_net::transport::{CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{seal, Connection, Endpoint, NodeId, NodeKey};
use dfin_proto::{
    read_frame_max, ActDtype, Activations, ErrorCode, HopRecord, HopStat, NextHop, Request,
    Response, ReturnFrame, ReturnMode, PROTOCOL_VERSION,
};
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MID: (i32, i32) = (3, 6);
const BIG: u32 = 64 << 20;
const LONG: Duration = Duration::from_secs(120);

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DFIN_TEST_MODEL") {
        return p.into();
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/tiny-llama-f32.gguf"
    ))
}

/// Kills the stage process even if the test panics, and removes its key file.
struct StageProc {
    child: Child,
    key_path: PathBuf,
    pub id: NodeId,
    pub addr: SocketAddr,
}

impl Drop for StageProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.key_path);
    }
}

fn free_udp_port() -> u16 {
    let s = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    s.local_addr().unwrap().port()
}

fn spawn_stage(path: &Path, name: &str, range: (i32, i32), n_seq_max: u32) -> StageProc {
    let port = free_udp_port();
    let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
    let key = NodeKey::generate();
    let key_path = std::env::temp_dir().join(format!(
        "dfin-proto-{name}-{}-{}.key",
        std::process::id(),
        port
    ));
    key.save(&key_path).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_dfin"))
        .args(["--log", "warn", "stage", "--model"])
        .arg(path)
        .args(["--layers", &format!("{}:{}", range.0, range.1)])
        .args(["--listen", &addr.to_string()])
        .arg("--identity")
        .arg(&key_path)
        .args([
            "--n-gpu-layers",
            "0",
            "--act-dtype",
            "f32",
            "--n-ctx",
            "1024",
            "--n-seq-max",
            &n_seq_max.to_string(),
        ])
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn dfin stage");
    StageProc {
        child,
        key_path,
        id: key.id(),
        addr,
    }
}

async fn call(conn: &Connection, req: &Request) -> Response {
    conn.call(req, BIG, LONG).await.expect("call")
}

async fn wait_for_stage(ep: &Endpoint, st: &mut StageProc) {
    let start = Instant::now();
    loop {
        if let Some(status) = st.child.try_wait().unwrap() {
            panic!("stage exited early: {status}");
        }
        if let Ok(conn) = ep.connect(&st.addr.to_string(), st.id).await {
            if let Ok(Response::Pong) = conn
                .call::<_, Response>(&Request::Ping, MAX_CONTROL_FRAME, CONTROL_TIMEOUT)
                .await
            {
                conn.close("probe");
                return;
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "stage did not start listening"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn hello(ep: &Endpoint, st: &StageProc, fingerprint: &str, range: (i32, i32)) -> (Connection, Response) {
    let conn = ep.connect(&st.addr.to_string(), st.id).await.unwrap();
    let r = call(
        &conn,
        &Request::Hello {
            version: PROTOCOL_VERSION,
            model_fingerprint: fingerprint.into(),
            layer_begin: range.0,
            layer_end: range.1,
        },
    )
    .await;
    (conn, r)
}

/// Build the sealed records for `hops` (addr, id, range), last hop returning as `ret`.
fn sealed_route(fingerprint: &str, hops: &[(SocketAddr, NodeId, (i32, i32))], ret: ReturnMode) -> Vec<u8> {
    let mut inner = Vec::new();
    for i in (0..hops.len()).rev() {
        let last = i + 1 == hops.len();
        let rec = HopRecord {
            model_fingerprint: fingerprint.to_string(),
            layer_begin: hops[i].2 .0,
            layer_end: hops[i].2 .1,
            next: (!last).then(|| NextHop {
                addr: hops[i + 1].0.to_string(),
                node_id: *hops[i + 1].1.as_bytes(),
            }),
            ret: if last { ret.clone() } else { ReturnMode::Reverse },
            inner,
        };
        inner = seal(&hops[i].1, &postcard::to_stdvec(&rec).unwrap()).unwrap();
    }
    inner
}

async fn setup(conn: &Connection, route: u64, sealed: Vec<u8>) -> Response {
    call(conn, &Request::RouteSetup { route, hop: sealed }).await
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

fn forward_ok(r: Response) -> (u64, u64, Activations, Vec<HopStat>) {
    match r {
        Response::ForwardOk {
            session,
            seq,
            act,
            hops,
        } => (session, seq, act, hops),
        other => panic!("expected ForwardOk, got {other:?}"),
    }
}

fn forward(route: u64, session: u64, seq: u64, pos0: i32, act: Activations) -> Request {
    Request::Forward {
        route,
        session,
        seq,
        pos0,
        act,
        hops: Vec::new(),
    }
}

struct Reference {
    tokens: Vec<i32>,
    logits: Vec<f32>,
    n_vocab: usize,
}

fn reference(path: &Path) -> Reference {
    let mut full = cpu_piece(path, 0, -1, false);
    let tokens = full
        .model()
        .vocab()
        .tokenize("The quick brown fox jumps over the lazy dog.", true, false)
        .unwrap();
    full.decode_tokens(&tokens, 0, 0, OutputMode::All).unwrap();
    let n_vocab = full.n_vocab();
    Reference {
        tokens,
        logits: full.logits().to_vec(),
        n_vocab,
    }
}

fn check_tail(path: &Path, tail_begin: i32, mid_out: &[f32], r: &Reference) {
    let n = r.tokens.len();
    let mut tail = cpu_piece(path, tail_begin, -1, false);
    tail.decode_embd(mid_out, 0, 0, OutputMode::All).unwrap();
    let logits = tail.logits().to_vec();
    assert_eq!(logits.len(), r.logits.len());
    let max_diff = logits
        .iter()
        .zip(&r.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| over {n} positions: {max_diff:e}");
    assert!(max_diff < 1e-3, "max abs logit diff {max_diff}");
    for i in 0..n {
        let a = argmax(&logits[i * r.n_vocab..(i + 1) * r.n_vocab]);
        let b = argmax(&r.logits[i * r.n_vocab..(i + 1) * r.n_vocab]);
        assert_eq!(a, b, "argmax differs at position {i}");
    }
}

fn ready() -> Option<PathBuf> {
    let path = model_path();
    if !path.exists() {
        eprintln!("SKIP: test model {} not found", path.display());
        return None;
    }
    if !dfin_llama::has_layer_range() {
        eprintln!("SKIP: libllama lacks the layer-range patch");
        return None;
    }
    dfin_llama::init();
    Some(path)
}

#[tokio::test]
async fn stage_matches_full_model() {
    let Some(path) = ready() else { return };
    let ep = Endpoint::client(NodeKey::generate()).unwrap();
    let mut st = spawn_stage(&path, "single", MID, 2);
    wait_for_stage(&ep, &mut st).await;

    let fp = dfin_proto::model_fingerprint(&path).unwrap();
    let (s, r) = hello(&ep, &st, &fp, MID).await;
    let n_embd = match r {
        Response::HelloOk {
            layer_begin,
            layer_end,
            n_embd,
            n_ctx,
            n_seq_max,
            backend,
            node_id,
        } => {
            eprintln!("HelloOk: layers {layer_begin}:{layer_end} n_embd {n_embd} n_ctx {n_ctx} n_seq_max {n_seq_max} backend {backend:?}");
            assert_eq!((layer_begin, layer_end), MID);
            assert_eq!(n_ctx, 512);
            assert_eq!(n_seq_max, 2);
            assert_eq!(NodeId(node_id), st.id);
            n_embd as usize
        }
        other => panic!("expected HelloOk, got {other:?}"),
    };

    let route = 11u64;
    let sealed = sealed_route(&fp, &[(st.addr, st.id, MID)], ReturnMode::Reverse);
    match setup(&s, route, sealed).await {
        Response::RouteOk { hops } => assert_eq!(hops, 1),
        other => panic!("expected RouteOk, got {other:?}"),
    }

    let r = reference(&path);
    let n = r.tokens.len();
    assert!(n >= 4, "prompt too short: {n}");

    // Head locally.
    let mut head = cpu_piece(&path, 0, MID.0, true);
    head.decode_tokens(&r.tokens, 0, 0, OutputMode::All).unwrap();
    let h = head.embeddings().to_vec();
    assert_eq!(h.len(), n * n_embd);

    // Middle on the stage.
    let session = 42u64;
    let resp = call(
        &s,
        &forward(route, session, 1, 0, Activations::from_f32(&h, n, n_embd, ActDtype::F32)),
    )
    .await;
    let (rs, rseq, act, hops) = forward_ok(resp);
    assert_eq!((rs, rseq), (session, 1));
    assert_eq!(act.n_tokens as usize, n);
    assert_eq!(hops.len(), 1);
    assert_eq!((hops[0].layer_begin, hops[0].layer_end), MID);
    eprintln!(
        "prefill forward: {n} tokens, {} bytes, {:.3} ms",
        act.byte_len(),
        hops[0].compute_ms
    );
    check_tail(&path, MID.1, &act.to_f32().unwrap(), &r);

    // Rollback: one token at pos n, then the same again (stage must roll back itself).
    let one = Activations::from_f32(&h[(n - 1) * n_embd..], 1, n_embd, ActDtype::F32);
    let r1 = forward_ok(call(&s, &forward(route, session, 2, n as i32, one.clone())).await);
    let r2 = forward_ok(call(&s, &forward(route, session, 3, n as i32, one.clone())).await);
    assert_eq!(r2.1, 3);
    assert_eq!(
        r1.2.data, r2.2.data,
        "rolled-back forward must reproduce the same output"
    );

    // Position gap is refused.
    match call(&s, &forward(route, session, 4, n as i32 + 5, one.clone())).await {
        Response::Error { code, message } => {
            assert_eq!(code, ErrorCode::BadRequest);
            assert!(message.contains("gap"), "{message}");
        }
        other => panic!("expected position-gap error, got {other:?}"),
    }

    // Wrong activation width is refused.
    let bad = Activations::from_f32(&vec![0.0; n_embd + 1], 1, n_embd + 1, ActDtype::F32);
    match call(&s, &forward(route, session, 5, 0, bad)).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::BadRequest),
        other => panic!("expected BadRequest, got {other:?}"),
    }

    // Unknown route.
    match call(&s, &forward(route + 1, session, 6, 0, one.clone())).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::UnknownRoute),
        other => panic!("expected UnknownRoute, got {other:?}"),
    }

    // Ping, SeqRm, SessionClose.
    assert!(matches!(call(&s, &Request::Ping).await, Response::Pong));
    let resp = call(
        &s,
        &Request::SeqRm {
            route,
            session,
            p0: n as i32,
            p1: -1,
        },
    )
    .await;
    assert!(matches!(resp, Response::Ack), "SeqRm: {resp:?}");
    match call(
        &s,
        &Request::SeqRm {
            route,
            session: 999,
            p0: 0,
            p1: -1,
        },
    )
    .await
    {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::UnknownSession),
        other => panic!("expected UnknownSession, got {other:?}"),
    }
    let resp = call(&s, &Request::SessionClose { route, session }).await;
    assert!(matches!(resp, Response::Ack), "SessionClose: {resp:?}");

    // Session limit: two slots, a third session is refused.
    for (i, sess) in [100u64, 101, 102].into_iter().enumerate() {
        let resp = call(&s, &forward(route, sess, 1, 0, one.clone())).await;
        if i < 2 {
            forward_ok(resp);
        } else {
            match resp {
                Response::Error { code, .. } => assert_eq!(code, ErrorCode::SessionLimit),
                other => panic!("expected SessionLimit, got {other:?}"),
            }
        }
    }
    // Closing the connection frees its slots and routes for the next one.
    s.close("done");
    drop(s);
    let (s2, resp) = hello(&ep, &st, &fp, MID).await;
    assert!(matches!(resp, Response::HelloOk { .. }));
    let sealed = sealed_route(&fp, &[(st.addr, st.id, MID)], ReturnMode::Reverse);
    assert!(matches!(setup(&s2, 12, sealed).await, Response::RouteOk { hops: 1 }));
    let mut freed = false;
    for _ in 0..50 {
        let resp = call(&s2, &forward(12, 7, 1, 0, one.clone())).await;
        if matches!(resp, Response::ForwardOk { .. }) {
            freed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(freed, "slots of a closed connection were not freed");

    // Route record refusals: wrong layers, sealed to another key, model mismatch.
    let sealed = sealed_route(&fp, &[(st.addr, st.id, (2, 6))], ReturnMode::Reverse);
    match setup(&s2, 13, sealed).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::LayerRangeMismatch),
        other => panic!("expected LayerRangeMismatch, got {other:?}"),
    }
    let other_key = NodeKey::generate();
    let sealed = sealed_route(&fp, &[(st.addr, other_key.id(), MID)], ReturnMode::Reverse);
    match setup(&s2, 14, sealed).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::BadRecord),
        other => panic!("expected BadRecord, got {other:?}"),
    }
    let sealed = sealed_route("0000", &[(st.addr, st.id, MID)], ReturnMode::Reverse);
    match setup(&s2, 15, sealed).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::ModelMismatch),
        other => panic!("expected ModelMismatch, got {other:?}"),
    }
    // A direct return needs a live client connection; our own counts.
    let sealed = sealed_route(
        &fp,
        &[(st.addr, st.id, MID)],
        ReturnMode::Direct {
            client_id: *ep.node_id().as_bytes(),
        },
    );
    assert!(matches!(setup(&s2, 16, sealed).await, Response::RouteOk { hops: 1 }));
    let sealed = sealed_route(
        &fp,
        &[(st.addr, st.id, MID)],
        ReturnMode::Direct {
            client_id: *NodeKey::generate().id().as_bytes(),
        },
    );
    match setup(&s2, 17, sealed).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::NoReturnPath),
        other => panic!("expected NoReturnPath, got {other:?}"),
    }

    // Hello refusals.
    let (_s, resp) = hello(&ep, &st, "0000", MID).await;
    match resp {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::ModelMismatch),
        other => panic!("expected ModelMismatch, got {other:?}"),
    }
    let (_s, resp) = hello(&ep, &st, &fp, (2, 6)).await;
    match resp {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::LayerRangeMismatch),
        other => panic!("expected LayerRangeMismatch, got {other:?}"),
    }
    let s3 = ep.connect(&st.addr.to_string(), st.id).await.unwrap();
    match call(&s3, &forward(1, 1, 1, 0, one.clone())).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::BadRequest),
        other => panic!("expected BadRequest for missing Hello, got {other:?}"),
    }
    // The wrong node id is refused at the handshake.
    assert!(ep
        .connect(&st.addr.to_string(), NodeKey::generate().id())
        .await
        .is_err());
    drop(st);
}

#[tokio::test]
async fn two_stage_chain_both_return_modes() {
    let Some(path) = ready() else { return };
    let ep = Endpoint::client(NodeKey::generate()).unwrap();
    let a_range = (MID.0, 5);
    let b_range = (5, MID.1);
    let mut a = spawn_stage(&path, "chain-a", a_range, 2);
    let mut b = spawn_stage(&path, "chain-b", b_range, 2);
    wait_for_stage(&ep, &mut a).await;
    wait_for_stage(&ep, &mut b).await;
    let fp = dfin_proto::model_fingerprint(&path).unwrap();

    let r = reference(&path);
    let n = r.tokens.len();
    let (ca, hr) = hello(&ep, &a, &fp, a_range).await;
    let n_embd = match hr {
        Response::HelloOk { n_embd, .. } => n_embd as usize,
        other => panic!("{other:?}"),
    };
    let mut head = cpu_piece(&path, 0, MID.0, true);
    head.decode_tokens(&r.tokens, 0, 0, OutputMode::All).unwrap();
    let h = head.embeddings().to_vec();
    let hops = [(a.addr, a.id, a_range), (b.addr, b.id, b_range)];

    // Reverse: the result comes back on the request stream through A.
    let sealed = sealed_route(&fp, &hops, ReturnMode::Reverse);
    match setup(&ca, 1, sealed).await {
        Response::RouteOk { hops } => assert_eq!(hops, 2),
        other => panic!("expected RouteOk, got {other:?}"),
    }
    let resp = call(
        &ca,
        &forward(1, 5, 1, 0, Activations::from_f32(&h, n, n_embd, ActDtype::F32)),
    )
    .await;
    let (_, _, act, stats) = forward_ok(resp);
    assert_eq!(stats.len(), 2, "{stats:?}");
    assert_eq!((stats[0].layer_begin, stats[0].layer_end), a_range);
    assert_eq!((stats[1].layer_begin, stats[1].layer_end), b_range);
    check_tail(&path, MID.1, &act.to_f32().unwrap(), &r);
    let resp = call(
        &ca,
        &Request::SeqRm {
            route: 1,
            session: 5,
            p0: 0,
            p1: -1,
        },
    )
    .await;
    assert!(matches!(resp, Response::Ack), "chain SeqRm: {resp:?}");

    // Direct: B sends the result straight to us on our connection to B.
    let (cb, hr) = hello(&ep, &b, &fp, b_range).await;
    assert!(matches!(hr, Response::HelloOk { .. }));
    let sealed = sealed_route(
        &fp,
        &hops,
        ReturnMode::Direct {
            client_id: *ep.node_id().as_bytes(),
        },
    );
    match setup(&ca, 2, sealed).await {
        Response::RouteOk { hops } => assert_eq!(hops, 2),
        other => panic!("expected RouteOk, got {other:?}"),
    }
    let ack = call(
        &ca,
        &forward(2, 6, 1, 0, Activations::from_f32(&h, n, n_embd, ActDtype::F32)),
    )
    .await;
    assert!(matches!(ack, Response::Ack), "direct forward reply: {ack:?}");
    let mut uni = tokio::time::timeout(LONG, cb.accept_uni()).await.unwrap().unwrap();
    let frame: ReturnFrame = read_frame_max(&mut uni, BIG).await.unwrap();
    assert_eq!((frame.route, frame.session, frame.seq), (2, 6, 1));
    let (_, _, act, stats) = forward_ok(frame.body);
    assert_eq!(stats.len(), 2);
    check_tail(&path, MID.1, &act.to_f32().unwrap(), &r);

    // Closing the route along the chain.
    let resp = call(&ca, &Request::RouteClose { route: 2 }).await;
    assert!(matches!(resp, Response::Ack), "RouteClose: {resp:?}");
    match call(&ca, &forward(2, 6, 2, 0, Activations::from_f32(&h[..n_embd], 1, n_embd, ActDtype::F32))).await {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::UnknownRoute),
        other => panic!("expected UnknownRoute after RouteClose, got {other:?}"),
    }
    drop((a, b));
}
