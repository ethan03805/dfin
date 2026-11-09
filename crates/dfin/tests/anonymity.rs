//! Full inference through a controlled SOCKS fixture. This is NOT a real Tor anonymity test.
#[path = "../../dfin-net/tests/support/mod.rs"]
mod support;
use dfin_net::{Endpoint, NodeKey};
use dfin_proto::{Request, Response, MAX_DIRECTORY_BYTES};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Processes {
    children: Vec<Child>,
    dir: PathBuf,
}
impl Processes {
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("dfin-anonymity-{:016x}", rand::random::<u64>()));
        std::fs::create_dir(&dir).unwrap();
        Self {
            children: vec![],
            dir,
        }
    }
    fn spawn(&mut self, name: &str, proxy: SocketAddr, args: &[&str]) -> usize {
        let log = std::fs::File::create(self.dir.join(format!("{name}.log"))).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_dfin"))
            .current_dir(&self.dir)
            .args(["--tor-proxy", &proxy.to_string()])
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        self.children.push(child);
        self.children.len() - 1
    }
    fn logs(&self) -> String {
        std::fs::read_dir(&self.dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|e| e == "log"))
            .map(|e| {
                format!(
                    "{}\n{}",
                    e.path().display(),
                    std::fs::read_to_string(e.path()).unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
    fn check(&mut self) {
        for child in &mut self.children {
            assert!(
                child.try_wait().unwrap().is_none(),
                "process exited; logs in {}",
                self.dir.display()
            );
        }
    }
}
impl Drop for Processes {
    fn drop(&mut self) {
        for c in &mut self.children {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}
fn local() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}
fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_hub_http_rejects_rebinding_and_foreign_origins() {
    let mut procs = Processes::new();
    let api = local();
    let base = format!("http://{api}");
    // No proxy connection is needed to serve the local inspection API.
    procs.spawn(
        "http-hub",
        "127.0.0.1:9050".parse().unwrap(),
        &[
            "hub",
            "--listen",
            &local().to_string(),
            "--http",
            &api.to_string(),
        ],
    );
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    wait_health(&http, &base, &mut procs).await;
    let directory: Value = http
        .get(format!("{base}/directory"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(directory["hub"].as_str().unwrap().len(), 64);
    for route in ["health", "directory"] {
        for host in [
            "attacker.example:7101",
            "localhost:7101",
            "127.0.0.1.attacker.example:7101",
        ] {
            let response = http
                .get(format!("{base}/{route}"))
                .header("Host", host)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                403,
                "rebinding Host was accepted: {host}"
            );
        }
        for origin in [
            "null".to_string(),
            "https://attacker.example".into(),
            format!("http://127.0.0.1:{}", api.port().saturating_sub(1)),
        ] {
            let response = http
                .get(format!("{base}/{route}"))
                .header("Origin", &origin)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                403,
                "foreign Origin was accepted: {origin}"
            );
        }
        let same_origin = http
            .get(format!("{base}/{route}"))
            .header("Origin", &base)
            .send()
            .await
            .unwrap();
        assert_eq!(same_origin.status(), 200);
    }
}

async fn wait_health(http: &reqwest::Client, base: &str, procs: &mut Processes) {
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if http
            .get(format!("{base}/health"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            return;
        }
        procs.check();
        assert!(
            Instant::now() < deadline,
            "health timeout\n{}",
            procs.logs()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strict_discovery_inference_isolation_and_fail_closed() {
    run_scenario(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires scripts/test-tor-local.py and real Tor binaries"]
async fn real_tor_discovery_and_inference() {
    run_scenario(true).await;
}

async fn run_scenario(real_tor: bool) {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-llama-f32.gguf");
    if !dfin_llama::has_layer_range() || !model.is_file() {
        eprintln!("SKIP: patched llama and tiny fixture required");
        return;
    }
    let model = model.display().to_string();
    let mut procs = Processes::new();
    let hub_key = NodeKey::generate();
    let stage_keys = [NodeKey::generate(), NodeKey::generate()];
    let relay_key = NodeKey::generate();
    for (name, key) in [
        ("hub", &hub_key),
        ("s1", &stage_keys[0]),
        ("s2", &stage_keys[1]),
        ("relay", &relay_key),
    ] {
        key.save(&procs.dir.join(format!("{name}.key"))).unwrap();
    }
    let config: Value = if real_tor {
        serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DFIN_TOR_TEST_CONFIG").expect("use scripts/test-tor-local.py"),
            )
            .unwrap(),
        )
        .unwrap()
    } else {
        Value::Null
    };
    let address = |name: &str| -> SocketAddr {
        if real_tor {
            config[name]["local"].as_str().unwrap().parse().unwrap()
        } else {
            local()
        }
    };
    let onion = |name: &str, letter| -> String {
        if real_tor {
            config[name]["onion"].as_str().unwrap().to_string()
        } else {
            support::onion(letter)
        }
    };
    let hub_local = address("hub");
    let stage_local = [address("s1"), address("s2")];
    let relay_local = address("relay");
    let hub_onion = onion("hub", 'h');
    let stage_onion = [onion("s1", 'a'), onion("s2", 'b')];
    let relay_onion = onion("relay", 'r');
    let mut proxy = if real_tor {
        None
    } else {
        Some(
            support::Proxy::start(HashMap::from([
                (hub_onion.clone(), hub_local),
                (stage_onion[0].clone(), stage_local[0]),
                (stage_onion[1].clone(), stage_local[1]),
                (relay_onion.clone(), relay_local),
            ]))
            .await,
        )
    };
    let proxy_addr = proxy
        .as_ref()
        .map(|p| p.addr)
        .unwrap_or_else(|| config["proxy"].as_str().unwrap().parse().unwrap());
    let hub_flag = format!("{hub_onion}/{}", hub_key.id());
    procs.spawn(
        "hub",
        proxy_addr,
        &[
            "hub",
            "--listen",
            &hub_local.to_string(),
            "--identity",
            &procs.dir.join("hub.key").display().to_string(),
        ],
    );
    for (i, layers) in ["2:4", "4:6"].iter().enumerate() {
        let name = format!("s{}", i + 1);
        procs.spawn(
            &name,
            proxy_addr,
            &[
                "stage",
                "--model",
                &model,
                "--layers",
                layers,
                "--listen",
                &stage_local[i].to_string(),
                "--identity",
                &procs.dir.join(format!("{name}.key")).display().to_string(),
                "--hub",
                &hub_flag,
                "--advertise",
                &stage_onion[i],
                "--announce-every",
                "5",
                "--n-gpu-layers",
                "0",
                "--act-dtype",
                "f32",
            ],
        );
    }
    procs.spawn(
        "relay",
        proxy_addr,
        &[
            "relay",
            "--listen",
            &relay_local.to_string(),
            "--identity",
            &procs.dir.join("relay.key").display().to_string(),
        ],
    );
    let probe = Endpoint::tor_client(NodeKey::generate(), proxy_addr).unwrap();
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let listed = async {
            let conn = probe.connect(&hub_onion, hub_key.id()).await.ok()?;
            let reply: Response = conn
                .call(
                    &Request::DirectoryGet,
                    MAX_DIRECTORY_BYTES,
                    Duration::from_secs(3),
                )
                .await
                .ok()?;
            conn.close("check");
            let Response::Directory { directory } = reply else {
                return None;
            };
            let directory =
                dfin_net::directory::verify_directory(&directory, &hub_key.id()).ok()?;
            for entry in &directory.entries {
                let record = dfin_net::directory::verify_record(&entry.record).unwrap();
                assert!(record.cloaked && record.backend.is_empty() && record.region.is_empty());
                assert!(record
                    .addrs
                    .iter()
                    .all(|a| dfin_net::privacy::onion_target(a).is_ok()));
            }
            Some(directory.entries.len())
        }
        .await;
        if listed == Some(2) {
            break;
        }
        procs.check();
        assert!(
            Instant::now() < deadline,
            "directory timeout\n{}",
            procs.logs()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap();
    let mut expected = None;
    // Local full coverage establishes deterministic reference text, then all three return paths.
    for mode in ["local", "direct", "reverse", "relay"] {
        let api = local();
        let base = format!("http://{api}");
        let cache = procs
            .dir
            .join(format!("{mode}.cache"))
            .display()
            .to_string();
        let mut args = strings(&[
            "client",
            "--model",
            &model,
            "--head",
            "0:2",
            "--tail",
            if mode == "local" { "2:end" } else { "6:end" },
            "--listen",
            &api.to_string(),
            "--model-name",
            "tiny",
            "--max-sessions",
            "2",
            "--n-ctx",
            "512",
            "--act-dtype",
            "f32",
            "--n-gpu-layers",
            "0",
        ]);
        if mode != "local" {
            args.extend(strings(&["--hub", &hub_flag, "--directory-cache", &cache]));
        }
        if mode == "relay" {
            args.extend(strings(&[
                "--relay",
                &format!("{relay_onion}/{}", relay_key.id()),
            ]));
        } else if mode != "local" {
            args.extend(strings(&["--return", mode]));
        }
        let child = procs.spawn(
            mode,
            proxy_addr,
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        wait_health(&http, &base, &mut procs).await;
        let denied = http
            .get(format!("{base}/status"))
            .header("Origin", "https://attacker.example")
            .send()
            .await
            .unwrap();
        assert_eq!(denied.status(), 403);
        let denied = http
            .get(format!("{base}/status"))
            .header("Host", "rebind.example:8080")
            .send()
            .await
            .unwrap();
        assert_eq!(denied.status(), 403);
        let wrong = http
            .post(format!("{base}/v1/completions"))
            .json(&json!({"model":"tiny:attested","prompt":"hi"}))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), 400);
        for _ in 0..2 {
            let response = http.post(format!("{base}/v1/completions")).json(&json!({
                "model":"tiny", "prompt":"hello world the model", "max_tokens":8, "temperature":0
            })).send().await.unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            assert!(status.is_success(), "{mode}: {body}\n{}", procs.logs());
            let value: Value = serde_json::from_str(&body).unwrap();
            let text = value["choices"][0]["text"].as_str().unwrap().to_string();
            if let Some(expected) = &expected {
                assert_eq!(&text, expected, "{mode}");
            } else {
                expected = Some(text);
            }
        }
        let status: Value = http
            .get(format!("{base}/status"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["privacy"]["network"], "tor");
        assert_eq!(status["privacy"]["request_identity_rotation"], true);
        assert_eq!(status["prefix_cache"]["tokens_reused"], 0);
        assert_eq!(status["privacy"]["content_confidential_from_nodes"], false);
        assert_eq!(
            status["privacy"]["relay_can_read_model_states"],
            mode == "relay"
        );
        // Let cleanup finish before terminating, to exercise the post-request path.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = procs.children[child].kill();
        let _ = procs.children[child].wait();
        // check() should only inspect live children in later iterations.
        procs.children.remove(child);
    }
    let mut credentials = std::collections::HashSet::new();
    if let Some(proxy) = &mut proxy {
        assert!(proxy.connections.load(std::sync::atomic::Ordering::Relaxed) >= 20);
        while let Ok((target, credential)) = proxy.observed.try_recv() {
            assert!(dfin_net::privacy::onion_target(&target).is_ok());
            assert!(
                credentials.insert(credential),
                "SOCKS isolation credentials were reused"
            );
        }
        assert!(
            credentials.len() >= 20,
            "expected discovery, routes, probes, and rotated requests"
        );
    }
    // Malicious requests cannot turn a Tor relay into a direct UDP forwarder.
    let relay = probe.connect(&relay_onion, relay_key.id()).await.unwrap();
    let r: Response = relay
        .call(&Request::RelayListen, 65536, Duration::from_secs(3))
        .await
        .unwrap();
    assert!(matches!(r, Response::Error { .. }));
    let relay = probe.connect(&relay_onion, relay_key.id()).await.unwrap();
    let r: Response = relay
        .call(
            &Request::RelayOpen {
                token: [0; 16],
                first_hop: dfin_proto::NextHop {
                    addr: "127.0.0.1:1".into(),
                    node_id: *stage_keys[0].id().as_bytes(),
                },
            },
            65536,
            Duration::from_secs(3),
        )
        .await
        .unwrap();
    assert!(matches!(r, Response::Error { .. }));
}

#[test]
fn strict_cli_rejects_leaks_before_loading_models_or_binding_services() {
    for args in [
        vec!["client", "--listen", "0.0.0.0:8080"],
        vec!["hub", "--listen", "localhost:7100"],
        vec!["relay", "--public-host", "example.org"],
        vec!["probe", "--region", "us"],
        vec!["pod", "--model", "missing.gguf", "--split", "2,6"],
        vec![
            "fetch",
            "--manifest",
            "missing.json",
            "--seed",
            "https://example.org/",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dfin"))
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?}");
        let err = String::from_utf8_lossy(&output.stderr);
        assert!(
            err.contains("strict") || err.contains("Tor") || err.contains("--network direct"),
            "{args:?}: {err}"
        );
    }
}

#[test]
fn manifest_seeds_cannot_trigger_a_direct_http_download() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tiny-shards/tiny-llama-f32.manifest.json");
    if !source.exists() {
        eprintln!("SKIP: generated shard manifest required");
        return;
    }
    let trap = TcpListener::bind("127.0.0.1:0").unwrap();
    trap.set_nonblocking(true).unwrap();
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
    manifest["seeds"] = json!([format!("http://{}", trap.local_addr().unwrap())]);
    let state = Processes::new();
    let path = state.dir.join("model.manifest.json");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dfin"))
        .args([
            "fetch",
            "--manifest",
            &path.display().to_string(),
            "--wait",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("manifest HTTP seeds are disabled"));
    assert_eq!(
        trap.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

/// A hostile stage observes on-wire identifiers and fails the first forward. Even
/// repeated errors must not carry identity, session, route, or sequence continuity.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failing_requests_rotate_every_explicit_client_identifier() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-llama-f32.gguf");
    if !model.is_file() || !dfin_llama::has_layer_range() {
        eprintln!("SKIP: tiny patched model required");
        return;
    }
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(
        key.clone(),
        "127.0.0.1:0".parse().unwrap(),
        "127.0.0.1:9050".parse().unwrap(),
    )
    .await
    .unwrap();
    let target = support::onion('a');
    let proxy = support::Proxy::start(HashMap::from([(
        target.clone(),
        server.local_addr().unwrap(),
    )]))
    .await;
    let (tx, mut observed) = tokio::sync::mpsc::channel(4);
    let accepting = server.clone();
    let id = key.id();
    let task = tokio::spawn(async move {
        while let Some(Ok(conn)) = accepting.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                    let req: Request = dfin_proto::read_frame_max(&mut recv, 4 << 20)
                        .await
                        .unwrap();
                    let response = match req {
                        Request::Hello { .. } => Response::HelloOk {
                            layer_begin: 2,
                            layer_end: 6,
                            n_embd: 64,
                            n_embd_inp: 64,
                            n_embd_out: 64,
                            n_ctx: 512,
                            n_seq_max: 2,
                            backend: String::new(),
                            node_id: *id.as_bytes(),
                        },
                        Request::Ping => Response::Pong,
                        Request::RouteSetup { .. } => Response::RouteOk { hops: 1 },
                        Request::Forward {
                            session,
                            seq,
                            route,
                            ..
                        } => {
                            tx.send((conn.peer_id(), session, route, seq))
                                .await
                                .unwrap();
                            Response::Error {
                                code: dfin_proto::ErrorCode::Decode,
                                message: "test failure".into(),
                            }
                        }
                        _ => Response::Ack,
                    };
                    if dfin_net::transport::reply(&mut send, &response)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
    });
    let mut procs = Processes::new();
    let api = local();
    let base = format!("http://{api}");
    procs.spawn(
        "identity-client",
        proxy.addr,
        &[
            "client",
            "--model",
            &model.display().to_string(),
            "--head",
            "0:2",
            "--tail",
            "6:end",
            "--stage",
            &format!("{target}/{id}=2:6"),
            "--return",
            "reverse",
            "--listen",
            &api.to_string(),
            "--n-gpu-layers",
            "0",
            "--n-ctx",
            "512",
        ],
    );
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    wait_health(&http, &base, &mut procs).await;
    let mut ids = vec![];
    for _ in 0..2 {
        let r = http
            .post(format!("{base}/v1/completions"))
            .json(&json!({"prompt":"same prompt", "max_tokens":1}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 500);
        ids.push(
            tokio::time::timeout(Duration::from_secs(3), observed.recv())
                .await
                .unwrap()
                .unwrap(),
        );
    }
    assert_ne!(ids[0].0, ids[1].0, "TLS client identity was reused");
    assert_ne!(ids[0].1, ids[1].1, "session identifier was reused");
    assert_ne!(ids[0].2, ids[1].2, "route identifier was reused");
    assert_eq!(
        (ids[0].3, ids[1].3),
        (1, 1),
        "sequence numbers leaked daemon continuity"
    );
    server.close();
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_prefill_closes_route_and_discards_cancelled_queue() {
    run_cancellation_case(CancelAt::Prefill).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_route_setup_closes_connection_and_discards_cancelled_queue() {
    run_cancellation_case(CancelAt::RouteSetup).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_discovery_closes_isolated_connection_without_retrying_cached_route() {
    run_cancellation_case(CancelAt::Directory).await;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CancelAt {
    Prefill,
    RouteSetup,
    Directory,
}

async fn run_cancellation_case(cancel_at: CancelAt) {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::AsyncWriteExt;

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-llama-f32.gguf");
    if !dfin_llama::has_layer_range() || !model.is_file() {
        eprintln!("SKIP: patched llama and tiny fixture required");
        return;
    }
    let key = NodeKey::generate();
    let id = key.id();
    let server = Endpoint::tor_server(key.clone(), local(), "127.0.0.1:9050".parse().unwrap())
        .await
        .unwrap();
    let target = support::onion('a');
    let proxy = support::Proxy::start(HashMap::from([(
        target.clone(),
        server.local_addr().unwrap(),
    )]))
    .await;
    let setups = Arc::new(AtomicUsize::new(0));
    let forwards = Arc::new(AtomicUsize::new(0));
    let lookups = Arc::new(AtomicUsize::new(0));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let record = dfin_proto::NodeRecord {
        node_id: *id.as_bytes(),
        addrs: vec![target.clone()],
        model_fingerprint: dfin_proto::model_fingerprint(&model).unwrap(),
        model_name: "tiny".into(),
        layer_begin: 2,
        layer_end: 6,
        n_layer: 8,
        n_embd: 64,
        n_ctx_seq: 512,
        n_seq_max: 2,
        act_dtype: dfin_proto::ActDtype::F16,
        backend: String::new(),
        protocol: dfin_proto::PROTOCOL_VERSION,
        region: String::new(),
        attested: false,
        cloaked: true,
        issued_at: now,
        ttl_secs: 180,
    };
    let signed = dfin_net::directory::sign_directory(
        &key,
        &dfin_proto::Directory {
            hub_id: *id.as_bytes(),
            issued_at: now,
            entries: vec![dfin_proto::DirectoryEntry {
                record: dfin_net::directory::sign_record(&key, &record).unwrap(),
                hub_rtt_ms: 1.0,
                seen_at: now,
                probes: vec![],
            }],
        },
    )
    .unwrap();
    let (tx, mut observed) = tokio::sync::mpsc::channel(8);
    let accepting = server.clone();
    let setups_seen = setups.clone();
    let lookups_seen = lookups.clone();
    let task = tokio::spawn(async move {
        while let Some(Ok(conn)) = accepting.accept().await {
            let tx = tx.clone();
            let setups = setups_seen.clone();
            let forwards = forwards.clone();
            let lookups = lookups_seen.clone();
            let signed = signed.clone();
            tokio::spawn(async move {
                while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                    let Ok(req) =
                        dfin_proto::read_frame_max::<_, Request>(&mut recv, 4 << 20).await
                    else {
                        break;
                    };
                    let response = match req {
                        Request::DirectoryGet => {
                            if lookups.fetch_add(1, Ordering::SeqCst) == 1 {
                                tx.send(conn.clone()).await.unwrap();
                                conn.closed().await;
                                break;
                            }
                            Response::Directory {
                                directory: signed.clone(),
                            }
                        }
                        Request::Hello { .. } => Response::HelloOk {
                            layer_begin: 2,
                            layer_end: 6,
                            n_embd: 64,
                            n_embd_inp: 64,
                            n_embd_out: 64,
                            n_ctx: 512,
                            n_seq_max: 2,
                            backend: String::new(),
                            node_id: *id.as_bytes(),
                        },
                        Request::Ping => Response::Pong,
                        Request::RouteSetup { .. } => {
                            let n = setups.fetch_add(1, Ordering::SeqCst);
                            if cancel_at == CancelAt::RouteSetup && n == 1 {
                                tx.send(conn.clone()).await.unwrap();
                                conn.closed().await;
                                break;
                            }
                            Response::RouteOk { hops: 1 }
                        }
                        Request::Forward { .. } => {
                            tx.send(conn.clone()).await.unwrap();
                            if cancel_at == CancelAt::Prefill
                                && forwards.fetch_add(1, Ordering::SeqCst) == 0
                            {
                                // An adversary withholds the first reply. Cancellation
                                // must close the transport without waiting for it.
                                conn.closed().await;
                                break;
                            }
                            Response::Error {
                                code: dfin_proto::ErrorCode::Decode,
                                message: "test failure".into(),
                            }
                        }
                        _ => Response::Ack,
                    };
                    if dfin_net::transport::reply(&mut send, &response)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
    });
    let mut procs = Processes::new();
    let api = local();
    let base = format!("http://{api}");
    let (source_flag, source_value) = if cancel_at == CancelAt::Directory {
        ("--hub", format!("{target}/{id}"))
    } else {
        ("--stage", format!("{target}/{id}=2:6"))
    };
    procs.spawn(
        "cancel-client",
        proxy.addr,
        &[
            "client",
            "--model",
            &model.display().to_string(),
            "--head",
            "0:2",
            "--tail",
            "6:end",
            source_flag,
            &source_value,
            "--return",
            "reverse",
            "--listen",
            &api.to_string(),
            "--n-gpu-layers",
            "0",
            "--n-ctx",
            "512",
            "--prefill-chunk",
            "2",
            "--max-sessions",
            "1",
        ],
    );
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    wait_health(&http, &base, &mut procs).await;
    assert_eq!(setups.load(Ordering::SeqCst), 1, "startup route");
    let body = json!({"prompt": "a long prompt requiring several chunks", "max_tokens": 1, "stream": true}).to_string();
    let request = format!("POST /v1/completions HTTP/1.1\r\nHost: {api}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    let mut active = tokio::net::TcpStream::connect(api).await.unwrap();
    active.write_all(request.as_bytes()).await.unwrap();
    let stalled = tokio::time::timeout(Duration::from_secs(5), observed.recv())
        .await
        .unwrap()
        .unwrap();
    let first_id = stalled.peer_id();
    let setups_before_cancel = if cancel_at == CancelAt::Directory {
        1
    } else {
        2
    };
    assert_eq!(setups.load(Ordering::SeqCst), setups_before_cancel);

    // Queue another request and cancel it while the engine is held by the first.
    let mut queued = tokio::net::TcpStream::connect(api).await.unwrap();
    queued.write_all(request.as_bytes()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(queued);
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(active);
    tokio::time::timeout(Duration::from_secs(2), stalled.closed())
        .await
        .expect("cancelled caller left its remote route alive");

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let status: Value = http
            .get(format!("{base}/status"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if status["sessions"]["busy"] == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "cancelled job kept the engine busy"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let response = http
        .post(format!("{base}/v1/completions"))
        .json(&json!({"prompt": "next request", "max_tokens": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 500);
    let next = tokio::time::timeout(Duration::from_secs(2), observed.recv())
        .await
        .unwrap()
        .unwrap();
    assert_ne!(first_id, next.peer_id(), "cancelled identity was reused");
    assert_eq!(
        setups.load(Ordering::SeqCst),
        setups_before_cancel + 1,
        "cancelled queued work still contacted the stage"
    );
    if cancel_at == CancelAt::Directory {
        assert_eq!(
            lookups.load(Ordering::SeqCst),
            3,
            "cancelled queued work still queried the hub"
        );
    }
    assert!(
        observed.try_recv().is_err(),
        "cancelled prompt continued forwarding"
    );
    server.close();
    task.abort();
}
