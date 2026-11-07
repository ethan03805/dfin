//! Real hub handlers with controlled loopback peers; no public network or anonymity claim.
use super::*;
use dfin_net::{directory::sign_record, SendStream};
use dfin_proto::{ActDtype, PROTOCOL_VERSION};
use tokio::task::JoinHandle;

const WAIT: Duration = Duration::from_secs(3);

struct Fixture {
    hub: Arc<Hub>,
    task: JoinHandle<()>,
}

impl Fixture {
    fn new(max_records: usize) -> Self {
        let endpoint =
            Endpoint::direct_server(NodeKey::generate(), "127.0.0.1:0".parse().unwrap()).unwrap();
        let hub = Arc::new(Hub {
            endpoint: endpoint.clone(),
            state: Mutex::new(DirectoryState::default()),
            max_records,
            probe_timeout: Duration::from_secs(30),
            probes: vec![],
            region: None,
            probe_conns: tokio::sync::Mutex::new(HashMap::new()),
        });
        let copy = hub.clone();
        let task = tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    r = endpoint.accept() => {
                        let Some(Ok(conn)) = r else { break };
                        let hub = copy.clone();
                        tasks.spawn(async move { hub.serve(conn).await });
                    }
                    _ = tasks.join_next(), if !tasks.is_empty() => {}
                }
            }
        });
        Self { hub, task }
    }

    async fn client(&self) -> (Endpoint, Connection) {
        let ep = Endpoint::direct_client(NodeKey::generate()).unwrap();
        let conn = ep
            .connect(
                &self.hub.endpoint.local_addr().unwrap().to_string(),
                self.hub.endpoint.node_id(),
            )
            .await
            .unwrap();
        (ep, conn)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.hub.endpoint.close();
        self.task.abort();
    }
}

struct Node {
    endpoint: Endpoint,
    signed: SignedRecord,
    responders: Vec<JoinHandle<()>>,
}

impl Drop for Node {
    fn drop(&mut self) {
        self.endpoint.close();
        for task in &self.responders {
            task.abort();
        }
    }
}

impl Node {
    fn new() -> Self {
        let key = NodeKey::generate();
        let endpoint = Endpoint::direct_server(key, "127.0.0.1:0".parse().unwrap()).unwrap();
        let record = NodeRecord {
            node_id: endpoint.node_id().0,
            addrs: vec![endpoint.local_addr().unwrap().to_string()],
            model_fingerprint: "fp".into(),
            model_name: "tiny".into(),
            layer_begin: 2,
            layer_end: 6,
            n_layer: 8,
            n_embd: 64,
            n_ctx_seq: 512,
            n_seq_max: 2,
            act_dtype: ActDtype::F32,
            backend: String::new(),
            protocol: PROTOCOL_VERSION,
            region: String::new(),
            attested: false,
            cloaked: false,
            issued_at: unix_now(),
            ttl_secs: 180,
        };
        let signed = sign_record(endpoint.key(), &record).unwrap();
        Self {
            endpoint,
            signed,
            responders: vec![],
        }
    }

    async fn hold_first_ping(&self) -> (Connection, SendStream) {
        tokio::time::timeout(WAIT, async {
            let conn = self.endpoint.accept().await.unwrap().unwrap();
            let (send, mut recv) = conn.accept_bi().await.unwrap();
            assert!(matches!(
                read_frame_max::<_, Request>(&mut recv, MAX_CONTROL_FRAME)
                    .await
                    .unwrap(),
                Request::Ping
            ));
            (conn, send)
        })
        .await
        .unwrap()
    }

    fn answer_later_connections(&mut self) {
        let ep = self.endpoint.clone();
        self.responders.push(tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            while let Some(Ok(conn)) = ep.accept().await {
                tasks.spawn(async move {
                    while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let request: Request =
                            read_frame_max(&mut recv, MAX_CONTROL_FRAME).await.unwrap();
                        assert!(matches!(request, Request::Ping));
                        let _ = reply(&mut send, &Response::Pong).await;
                    }
                });
            }
        }));
    }
}

fn announce(hub: &Arc<Hub>, node: &Node) -> JoinHandle<Response> {
    let hub = hub.clone();
    let signed = node.signed.clone();
    tokio::spawn(async move { hub.announce(signed).await })
}

async fn finish_probe(conn: &Connection, mut first: SendStream) {
    reply(&mut first, &Response::Pong).await.unwrap();
    for _ in 0..2 {
        let (mut send, mut recv) = conn.accept_bi().await.unwrap();
        assert!(matches!(
            read_frame_max::<_, Request>(&mut recv, MAX_CONTROL_FRAME)
                .await
                .unwrap(),
            Request::Ping
        ));
        reply(&mut send, &Response::Pong).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_announcements_cannot_overfill_the_directory() {
    let f = Fixture::new(1);
    let first = Node::new();
    let pending = announce(&f.hub, &first);
    let (conn, send) = first.hold_first_ping().await;
    let mut second = Node::new();
    second.answer_later_connections();
    let response = tokio::time::timeout(WAIT, f.hub.announce(second.signed.clone()))
        .await
        .unwrap();
    finish_probe(&conn, send).await;
    assert!(matches!(pending.await.unwrap(), Response::Ack));
    assert_eq!(
        f.hub.state.lock().entries.len(),
        1,
        "concurrent probes bypassed the record limit"
    );
    assert!(
        matches!(
            response,
            Response::Error {
                code: ErrorCode::RouteLimit,
                ..
            }
        ),
        "{response:?}"
    );
    // A full directory still admits a refresh from the already listed node.
    let mut first = first;
    first.answer_later_connections();
    assert!(matches!(
        tokio::time::timeout(WAIT, f.hub.announce(first.signed.clone()))
            .await
            .unwrap(),
        Response::Ack
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_pending_announcements_do_not_start_another_probe() {
    let f = Fixture::new(2);
    let mut node = Node::new();
    let pending = announce(&f.hub, &node);
    let (conn, send) = node.hold_first_ping().await;
    node.answer_later_connections();
    let response = tokio::time::timeout(WAIT, f.hub.announce(node.signed.clone()))
        .await
        .unwrap();
    finish_probe(&conn, send).await;
    assert!(matches!(pending.await.unwrap(), Response::Ack));
    assert!(
        matches!(
            response,
            Response::Error {
                code: ErrorCode::BadRequest,
                ..
            }
        ),
        "{response:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_cancels_announcement_probe_and_releases_capacity() {
    let f = Fixture::new(1);
    let node = Node::new();
    let (_ep, client) = f.client().await;
    let copy = client.clone();
    let signed = node.signed.clone();
    let pending = tokio::spawn(async move {
        copy.call::<_, Response>(
            &Request::Announce { record: signed },
            MAX_CONTROL_FRAME,
            WAIT,
        )
        .await
    });
    let (conn, _send) = node.hold_first_ping().await;
    client.close("cancel announcement");
    pending.abort();
    assert!(
        tokio::time::timeout(WAIT, conn.closed()).await.is_ok(),
        "probe outlived its caller"
    );
    let mut next = Node::new();
    next.answer_later_connections();
    assert!(matches!(
        tokio::time::timeout(WAIT, f.hub.announce(next.signed.clone()))
            .await
            .unwrap(),
        Response::Ack
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_probe_closes_its_connection_and_releases_the_reservation() {
    let f = Fixture::new(1);
    let mut node = Node::new();
    let pending = announce(&f.hub, &node);
    let (conn, mut send) = node.hold_first_ping().await;
    reply(&mut send, &Response::Ack).await.unwrap();
    assert!(matches!(
        pending.await.unwrap(),
        Response::Error {
            code: ErrorCode::Unreachable,
            ..
        }
    ));
    assert!(tokio::time::timeout(WAIT, conn.closed()).await.is_ok());
    node.answer_later_connections();
    assert!(matches!(
        tokio::time::timeout(WAIT, f.hub.announce(node.signed.clone()))
            .await
            .unwrap(),
        Response::Ack
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_large_directory_still_bounds_simultaneous_probes() {
    let f = Fixture::new(1000);
    let mut held = Vec::new();
    for _ in 0..32 {
        let node = Node::new();
        let task = announce(&f.hub, &node);
        let (conn, send) = node.hold_first_ping().await;
        held.push((node, conn, send, task));
    }
    let mut extra = Node::new();
    extra.answer_later_connections();
    let response = tokio::time::timeout(WAIT, f.hub.announce(extra.signed.clone()))
        .await
        .unwrap();
    for (_, _, _, task) in &held {
        task.abort();
    }
    assert!(
        matches!(
            response,
            Response::Error {
                code: ErrorCode::RateLimited,
                ..
            }
        ),
        "{response:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_cancelled_announcements_release_all_reservations() {
    let f = Fixture::new(1);
    for _ in 0..40 {
        let node = Node::new();
        let (endpoint, client) = f.client().await;
        let copy = client.clone();
        let signed = node.signed.clone();
        let request = tokio::spawn(async move {
            copy.call::<_, Response>(
                &Request::Announce { record: signed },
                MAX_CONTROL_FRAME,
                WAIT,
            )
            .await
        });
        let (probe, _send) = node.hold_first_ping().await;
        client.close("cancelled iteration");
        request.abort();
        let _ = request.await;
        assert!(tokio::time::timeout(WAIT, probe.closed()).await.is_ok());
        tokio::time::timeout(WAIT, async {
            while !f.hub.state.lock().pending.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled announcement retained a reservation");
        assert!(f.hub.state.lock().entries.is_empty());
        endpoint.close();
    }
    let mut final_node = Node::new();
    final_node.answer_later_connections();
    assert!(matches!(
        tokio::time::timeout(WAIT, f.hub.announce(final_node.signed.clone()))
            .await
            .unwrap(),
        Response::Ack
    ));
}
