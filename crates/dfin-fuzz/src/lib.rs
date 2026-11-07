//! Fuzz targets for everything dfin decodes from a peer.
//!
//! Each function takes arbitrary bytes and must return, never panic, whatever the bytes
//! are. When the bytes do decode, the function also does what the real code does with
//! the value: re-encodes it, converts activations, verifies signatures, opens inner
//! sealed records, parses addresses. The same functions run three ways: under
//! cargo-fuzz with libFuzzer (`fuzz/` at the repository root; nightly, Linux CI), as
//! ordinary tests over random and mutated inputs (`tests/smoke.rs`; every platform),
//! and from `dfin-fuzz-seeds`, which writes valid encodings for libFuzzer to start
//! from. The return value says whether the input decoded, so a harness can check that
//! its seeds are right and that mutations reach both outcomes.
//!
//! What is deliberately not here: the QUIC and TLS layers (quinn and rustls fuzz their
//! own decoders), and the value-level rules of the binary (chain tiling, the hub's
//! record checks), which `crates/dfin` tests with generated values in its own tests.

use dfin_net::directory::{sign_directory, sign_record, verify_directory, verify_record};
use dfin_net::transport::MAX_CONTROL_FRAME;
use dfin_net::tunnel::{read_packet, write_packet};
use dfin_net::{seal, unseal, NodeId, NodeKey};
use dfin_proto::{
    forward_frame_limit, read_frame_max, Activations, Directory, HopRecord, NodeRecord, Request,
    Response, ReturnFrame, ReturnMode, SignedDirectory, SignedRecord, MAX_DIRECTORY_BYTES,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::OnceLock;

/// A target: arbitrary bytes in, "did it decode" out, never a panic.
pub type Target = fn(&[u8]) -> bool;

/// Every target by name, the names the cargo-fuzz binaries use.
pub const TARGETS: &[(&str, Target)] = &[
    ("request", request),
    ("response", response),
    ("return_frame", return_frame),
    ("hop_record", hop_record),
    ("sealed", sealed),
    ("signed_record", signed_record),
    ("signed_directory", signed_directory),
    ("tunnel_packet", tunnel_packet),
    ("frame", frame),
    ("node_id_text", node_id_text),
    ("activations", activations),
];

/// A fixed key, so sealed and signed inputs a fuzzer evolves stay meaningful across runs.
pub fn fixed_key() -> &'static NodeKey {
    static KEY: OnceLock<NodeKey> = OnceLock::new();
    KEY.get_or_init(|| NodeKey::from_bytes([7u8; 32]))
}

fn rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a current-thread runtime")
    })
}

/// A decoded value must encode again, and that encoding must decode to the same bytes:
/// the codec is total and stable on everything it accepts.
fn reencodes<T: Serialize + DeserializeOwned>(v: &T) -> Vec<u8> {
    let bytes = postcard::to_stdvec(v).expect("a decoded value encodes");
    let again: T = postcard::from_bytes(&bytes).expect("an encoding decodes");
    let bytes2 = postcard::to_stdvec(&again).expect("encodes again");
    assert_eq!(bytes, bytes2, "encoding is not stable");
    bytes
}

fn exercise_activations(act: &Activations) {
    let _ = forward_frame_limit(act.n_tokens, act.n_embd);
    let _ = act.byte_len();
    if let Ok(values) = act.to_f32() {
        // A batch that converts has a consistent shape, so it converts back.
        let back = Activations::from_f32(&values, act.n_tokens as usize, act.n_embd as usize, act.dtype);
        assert_eq!((back.n_tokens, back.n_embd), (act.n_tokens, act.n_embd));
        assert_eq!(back.to_f32().expect("converts back").len(), values.len());
    }
}

fn exercise_record(rec: &NodeRecord) {
    let _ = rec.expires_at();
    let _ = NodeId(rec.node_id).short();
    let _ = NodeId(rec.node_id).verifying_key();
    for a in &rec.addrs {
        let _ = a.parse::<SocketAddr>();
    }
}

fn exercise_directory(dir: &Directory) {
    for e in &dir.entries {
        if let Ok(r) = verify_record(&e.record) {
            exercise_record(&r);
        }
        let _ = e.hub_rtt_ms.max(0.0);
        for p in &e.probes {
            let _ = p.rtt_ms <= 50.0;
            let _ = NodeId(p.probe_id).short();
        }
    }
}

fn exercise_hop(rec: &HopRecord, depth: usize) {
    if let Some(next) = &rec.next {
        let _ = next.addr.parse::<SocketAddr>();
        let _ = NodeId(next.node_id).verifying_key();
    }
    match &rec.ret {
        ReturnMode::Reverse => {}
        ReturnMode::Direct { client_id } => {
            let _ = NodeId(*client_id).verifying_key();
        }
        ReturnMode::Relay { relay, .. } => {
            let _ = relay.addr.parse::<SocketAddr>();
        }
    }
    // The inner record is sealed to the next hop. With the fixed key a fuzzer can
    // build a chain of them; a stage only ever opens one, so the depth is bounded here.
    if depth < 8 {
        if let Ok(plain) = unseal(fixed_key(), &rec.inner) {
            if let Ok(inner) = postcard::from_bytes::<HopRecord>(&plain) {
                exercise_hop(&inner, depth + 1);
            }
        }
    }
}

fn exercise_response(resp: &Response) {
    match resp {
        Response::ForwardOk { act, hops, .. } => {
            exercise_activations(act);
            let _ = hops.len();
        }
        Response::Directory { directory } => {
            let _ = verify_directory(directory, &fixed_key().id());
        }
        Response::Shard { total, data } => {
            let _ = total.checked_add(data.len() as u64);
        }
        Response::RelayListening { addr } => {
            let _ = addr.parse::<SocketAddr>();
        }
        Response::HelloOk { node_id, .. } => {
            let _ = NodeId(*node_id).verifying_key();
        }
        _ => {}
    }
}

/// A `Request` as a stage, relay, hub or probe reads it.
pub fn request(data: &[u8]) -> bool {
    let Ok(req) = postcard::from_bytes::<Request>(data) else {
        return false;
    };
    match &req {
        Request::Forward { act, hops, .. } => {
            exercise_activations(act);
            let _ = hops.len();
        }
        Request::RouteSetup { hop, .. } => {
            if let Ok(plain) = unseal(fixed_key(), hop) {
                if let Ok(rec) = postcard::from_bytes::<HopRecord>(&plain) {
                    exercise_hop(&rec, 0);
                }
            }
        }
        Request::Announce { record } => {
            if let Ok(rec) = verify_record(record) {
                exercise_record(&rec);
            }
        }
        Request::Probe { addr, node_id } => {
            let _ = addr.parse::<SocketAddr>();
            let _ = NodeId(*node_id).verifying_key();
        }
        Request::RelayOpen { first_hop, .. } => {
            let _ = first_hop.addr.parse::<SocketAddr>();
        }
        _ => {}
    }
    reencodes(&req);
    true
}

/// A `Response` as a client or a previous hop reads it.
pub fn response(data: &[u8]) -> bool {
    let Ok(resp) = postcard::from_bytes::<Response>(data) else {
        return false;
    };
    exercise_response(&resp);
    reencodes(&resp);
    true
}

/// A `ReturnFrame` as a client's return listener reads it from anyone who dials it.
pub fn return_frame(data: &[u8]) -> bool {
    let Ok(f) = postcard::from_bytes::<ReturnFrame>(data) else {
        return false;
    };
    exercise_response(&f.body);
    reencodes(&f);
    true
}

/// A `HopRecord` as a stage reads it after unsealing.
pub fn hop_record(data: &[u8]) -> bool {
    let Ok(rec) = postcard::from_bytes::<HopRecord>(data) else {
        return false;
    };
    exercise_hop(&rec, 0);
    reencodes(&rec);
    true
}

/// A sealed box as a stage opens it, plus the seal round trip on the same bytes.
pub fn sealed(data: &[u8]) -> bool {
    let key = fixed_key();
    let opened = match unseal(key, data) {
        Ok(plain) => {
            hop_record(&plain);
            true
        }
        Err(_) => false,
    };
    let boxed = seal(&key.id(), data).expect("anything seals to a valid key");
    assert_eq!(unseal(key, &boxed).expect("opens"), data);
    opened
}

/// A `SignedRecord` as a hub or client verifies it, and the bare record under the
/// fixed key: whatever decodes must sign and verify.
pub fn signed_record(data: &[u8]) -> bool {
    let key = fixed_key();
    let mut decoded = false;
    if let Ok(signed) = postcard::from_bytes::<SignedRecord>(data) {
        decoded = true;
        if let Ok(rec) = verify_record(&signed) {
            exercise_record(&rec);
        }
        reencodes(&signed);
    }
    if let Ok(rec) = postcard::from_bytes::<NodeRecord>(data) {
        decoded = true;
        exercise_record(&rec);
        let mine = NodeRecord {
            node_id: *key.id().as_bytes(),
            ..rec
        };
        let signed = sign_record(key, &mine).expect("signs");
        let back = verify_record(&signed).expect("verifies");
        assert_eq!(postcard::to_stdvec(&back).unwrap(), postcard::to_stdvec(&mine).unwrap());
        reencodes(&mine);
    }
    decoded
}

/// A `SignedDirectory` as a client verifies it against the hub it pinned, and the bare
/// directory under the fixed key.
pub fn signed_directory(data: &[u8]) -> bool {
    let key = fixed_key();
    let mut decoded = false;
    if let Ok(signed) = postcard::from_bytes::<SignedDirectory>(data) {
        decoded = true;
        if let Ok(dir) = verify_directory(&signed, &key.id()) {
            exercise_directory(&dir);
        }
        reencodes(&signed);
    }
    if let Ok(dir) = postcard::from_bytes::<Directory>(data) {
        decoded = true;
        exercise_directory(&dir);
        let mine = Directory {
            hub_id: *key.id().as_bytes(),
            ..dir
        };
        let signed = sign_directory(key, &mine).expect("signs");
        let back = verify_directory(&signed, &key.id()).expect("verifies");
        assert_eq!(postcard::to_stdvec(&back).unwrap(), postcard::to_stdvec(&mine).unwrap());
        reencodes(&mine);
    }
    decoded
}

/// A tunnel packet frame as a relay or cloaked stage reads it from the tunnel stream.
pub fn tunnel_packet(data: &[u8]) -> bool {
    let mut cursor = data;
    let Ok(p) = rt().block_on(read_packet(&mut cursor)) else {
        return false;
    };
    let mut out = Vec::new();
    rt().block_on(write_packet(&mut out, &p))
        .expect("a packet that was read writes");
    let mut again = &out[..];
    let p2 = rt()
        .block_on(read_packet(&mut again))
        .expect("a packet that was written reads");
    assert_eq!((p.addr, &p.data), (p2.addr, &p2.data));
    true
}

/// A length-prefixed frame as `read_frame_max` reads it. The first byte picks the
/// limit the reader was given; the rest is the stream.
pub fn frame(data: &[u8]) -> bool {
    let Some((&pick, stream)) = data.split_first() else {
        return false;
    };
    let max = match pick % 4 {
        0 => 1,
        1 => 64,
        2 => MAX_CONTROL_FRAME,
        _ => MAX_DIRECTORY_BYTES,
    };
    // A stage refuses a tag byte with the varint continuation bit set: postcard would
    // accept the non-minimal encoding, but then the first byte no longer names the
    // variant. Mirror that rule, so the tag check below is exact.
    if stream.len() > 4 && stream[4] & 0x80 != 0 {
        return false;
    }
    let mut cursor = stream;
    match rt().block_on(read_frame_max::<_, Request>(&mut cursor, max)) {
        Ok(req) => {
            // A stage peeks at the first body byte to size its read, so a Forward must
            // announce itself there and nothing else may.
            let is_forward = matches!(req, Request::Forward { .. });
            assert_eq!(stream[4] == Request::FORWARD_TAG, is_forward);
            let used = stream.len() - cursor.len();
            assert!(used <= max as usize + 4, "read past the limit");
            true
        }
        Err(_) => false,
    }
}

/// A node id as typed on a command line or read from a route file.
pub fn node_id_text(data: &[u8]) -> bool {
    let text = String::from_utf8_lossy(data);
    match text.parse::<NodeId>() {
        Ok(id) => {
            assert_eq!(id.to_string().parse::<NodeId>().expect("prints and parses"), id);
            true
        }
        Err(_) => false,
    }
}

/// An `Activations` batch on its own, as it sits inside forwards and results.
pub fn activations(data: &[u8]) -> bool {
    let Ok(act) = postcard::from_bytes::<Activations>(data) else {
        return false;
    };
    exercise_activations(&act);
    reencodes(&act);
    true
}

/// Valid encodings for each target: what a fuzzer mutates from, and what the smoke
/// test truncates and flips.
pub mod seeds {
    use super::*;
    use dfin_net::tunnel::Packet;
    use dfin_proto::{
        write_frame, ActDtype, DirectoryEntry, ErrorCode, HopStat, NextHop, ProbeRtt,
        PROTOCOL_VERSION,
    };

    fn enc<T: Serialize>(v: &T) -> Vec<u8> {
        postcard::to_stdvec(v).unwrap()
    }

    fn act(dtype: ActDtype) -> Activations {
        let values: Vec<f32> = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect();
        Activations::from_f32(&values, 2, 4, dtype)
    }

    fn record() -> NodeRecord {
        NodeRecord {
            node_id: *fixed_key().id().as_bytes(),
            addrs: vec!["10.0.0.5:7001".into(), "[2001:db8::5]:7001".into()],
            model_fingerprint: "fp".into(),
            model_name: "tiny".into(),
            layer_begin: 2,
            layer_end: 6,
            n_layer: 8,
            n_embd: 64,
            n_ctx_seq: 4096,
            n_seq_max: 4,
            act_dtype: ActDtype::F16,
            backend: "CPU".into(),
            protocol: PROTOCOL_VERSION,
            region: "us".into(),
            attested: false,
            cloaked: false,
            issued_at: 1_800_000_000,
            ttl_secs: 180,
        }
    }

    fn signed_record_() -> SignedRecord {
        sign_record(fixed_key(), &record()).unwrap()
    }

    fn directory_() -> Directory {
        Directory {
            hub_id: *fixed_key().id().as_bytes(),
            issued_at: 1_800_000_010,
            entries: vec![
                DirectoryEntry {
                    record: signed_record_(),
                    hub_rtt_ms: 1.5,
                    seen_at: 1_800_000_005,
                    probes: vec![ProbeRtt {
                        probe_id: [9; 32],
                        region: "us".into(),
                        rtt_ms: 12.0,
                    }],
                },
                DirectoryEntry {
                    record: sign_record(
                        fixed_key(),
                        &NodeRecord {
                            cloaked: true,
                            ..record()
                        },
                    )
                    .unwrap(),
                    hub_rtt_ms: 0.4,
                    seen_at: 1_800_000_009,
                    probes: vec![],
                },
            ],
        }
    }

    fn last_hop() -> HopRecord {
        HopRecord {
            model_fingerprint: "fp".into(),
            layer_begin: 4,
            layer_end: 6,
            next: None,
            ret: ReturnMode::Direct {
                client_id: *fixed_key().id().as_bytes(),
            },
            inner: vec![],
        }
    }

    fn middle_hop() -> HopRecord {
        HopRecord {
            model_fingerprint: "fp".into(),
            layer_begin: 2,
            layer_end: 4,
            next: Some(NextHop {
                addr: "10.0.0.6:7001".into(),
                node_id: *fixed_key().id().as_bytes(),
            }),
            ret: ReturnMode::Relay {
                relay: NextHop {
                    addr: "203.0.113.5:7000".into(),
                    node_id: *fixed_key().id().as_bytes(),
                },
                token: [3; 16],
            },
            inner: seal(&fixed_key().id(), &enc(&last_hop())).unwrap(),
        }
    }

    fn stats() -> Vec<HopStat> {
        vec![HopStat {
            layer_begin: 2,
            layer_end: 4,
            compute_ms: 3.25,
        }]
    }

    fn framed(pick: u8, req: &Request) -> Vec<u8> {
        let mut out = vec![pick];
        rt().block_on(write_frame(&mut out, req)).unwrap();
        out
    }

    pub fn requests() -> Vec<Request> {
        vec![
            Request::Hello {
                version: PROTOCOL_VERSION,
                model_fingerprint: "fp".into(),
                layer_begin: 2,
                layer_end: 4,
            },
            Request::Ping,
            Request::RouteSetup {
                route: 7,
                hop: seal(&fixed_key().id(), &enc(&middle_hop())).unwrap(),
            },
            Request::Forward {
                route: 7,
                session: 1,
                seq: 2,
                pos0: 0,
                act: act(ActDtype::F16),
                hops: stats(),
            },
            Request::SeqRm {
                route: 7,
                session: 1,
                p0: 3,
                p1: -1,
            },
            Request::SessionClose { route: 7, session: 1 },
            Request::RouteClose { route: 7 },
            Request::RelayOpen {
                token: [3; 16],
                first_hop: NextHop {
                    addr: "10.0.0.5:7001".into(),
                    node_id: *fixed_key().id().as_bytes(),
                },
            },
            Request::RelayAttach { token: [3; 16] },
            Request::RelayListen,
            Request::Announce {
                record: signed_record_(),
            },
            Request::DirectoryGet,
            Request::Probe {
                addr: "10.0.0.5:7001".into(),
                node_id: *fixed_key().id().as_bytes(),
            },
            Request::ShardGet {
                model_id: "ab".repeat(32),
                split: 3,
                offset: 4 << 20,
                len: 4 << 20,
            },
        ]
    }

    pub fn responses() -> Vec<Response> {
        vec![
            Response::HelloOk {
                layer_begin: 2,
                layer_end: 4,
                n_embd: 64,
                n_embd_inp: 64,
                n_embd_out: 256,
                n_ctx: 4096,
                n_seq_max: 4,
                backend: "CPU".into(),
                node_id: *fixed_key().id().as_bytes(),
            },
            Response::RouteOk { hops: 2 },
            Response::Ack,
            Response::ForwardOk {
                session: 1,
                seq: 2,
                act: act(ActDtype::F32),
                hops: stats(),
            },
            Response::Pong,
            Response::RelayListening {
                addr: "203.0.113.5:7210".into(),
            },
            Response::Probed {
                rtt_ms: 12.5,
                region: "us".into(),
            },
            Response::Directory {
                directory: sign_directory(fixed_key(), &directory_()).unwrap(),
            },
            Response::Shard {
                total: 12345,
                data: vec![1, 2, 3, 4],
            },
            Response::Error {
                code: ErrorCode::BadRequest,
                message: "no".into(),
            },
        ]
    }

    /// Seeds for the named target; empty for an unknown name.
    pub fn for_target(name: &str) -> Vec<Vec<u8>> {
        match name {
            "request" => requests().iter().map(enc).collect(),
            "response" => responses().iter().map(enc).collect(),
            "return_frame" => responses()
                .into_iter()
                .map(|body| {
                    enc(&ReturnFrame {
                        route: 7,
                        session: 1,
                        seq: 2,
                        body,
                    })
                })
                .collect(),
            "hop_record" => vec![enc(&last_hop()), enc(&middle_hop())],
            "sealed" => vec![
                seal(&fixed_key().id(), &enc(&middle_hop())).unwrap(),
                seal(&fixed_key().id(), b"").unwrap(),
            ],
            "signed_record" => vec![enc(&signed_record_()), enc(&record())],
            "signed_directory" => vec![
                enc(&sign_directory(fixed_key(), &directory_()).unwrap()),
                enc(&directory_()),
            ],
            "tunnel_packet" => [
                Packet {
                    addr: "192.0.2.1:4433".parse().unwrap(),
                    data: vec![0xc0; 40],
                },
                Packet {
                    addr: "[2001:db8::1]:4433".parse().unwrap(),
                    data: vec![],
                },
            ]
            .iter()
            .map(|p| {
                let mut out = Vec::new();
                rt().block_on(write_packet(&mut out, p)).unwrap();
                out
            })
            .collect(),
            "frame" => {
                let reqs = requests();
                vec![
                    framed(2, &reqs[0]),
                    framed(2, &reqs[3]),
                    framed(3, &reqs[10]),
                    framed(1, &reqs[1]),
                ]
            }
            "node_id_text" => vec![fixed_key().id().to_string().into_bytes()],
            "activations" => vec![enc(&act(ActDtype::F16)), enc(&act(ActDtype::F32))],
            _ => vec![],
        }
    }
}
