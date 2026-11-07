//! Adversarial connection/route tests with the real stage handler and a fake engine.
//! Direct loopback transport here provides no anonymity; inference is covered by e2e.
use super::*;
use dfin_net::{seal, SendStream};
use tokio::task::JoinHandle;

const WAIT: Duration = Duration::from_secs(3);

struct Fixture {
    stage: Arc<Stage>,
    _jobs: mpsc::Receiver<Job>,
    task: JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stage.endpoint.close();
        self.task.abort();
    }
}

impl Fixture {
    fn new(begin: i32, end: i32, max_routes: usize) -> Self {
        let endpoint =
            Endpoint::direct_server(NodeKey::generate(), "127.0.0.1:0".parse().unwrap()).unwrap();
        let (jobs, rx) = mpsc::channel(32);
        let stage = Arc::new(Stage {
            info: Arc::new(StageInfo {
                fingerprint: "test-model".into(),
                layer_begin: begin,
                layer_end: end,
                n_embd: 64,
                n_embd_inp: 64,
                n_embd_out: 64,
                n_ctx_seq: 128,
                n_seq_max: 4,
                n_batch: 16,
                backend: "test".into(),
                node_id: endpoint.node_id(),
            }),
            jobs,
            endpoint: endpoint.clone(),
            clients: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
            max_routes,
            cloak_addrs: Mutex::new(vec![]),
            shards: None,
        });
        // Own connection tasks, including when a failed assertion drops the fixture.
        let copy = stage.clone();
        let task = tokio::spawn(async move {
            let mut tasks = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    r = endpoint.accept() => {
                        let Some(Ok(conn)) = r else { break };
                        let stage = copy.clone();
                        tasks.spawn(async move { stage.serve_conn(conn).await });
                    }
                    _ = tasks.join_next(), if !tasks.is_empty() => {}
                }
            }
        });
        Self {
            stage,
            _jobs: rx,
            task,
        }
    }

    fn next(&self) -> NextHop {
        NextHop {
            addr: self.stage.endpoint.local_addr().unwrap().to_string(),
            node_id: self.stage.endpoint.node_id().0,
        }
    }

    fn hop(&self, next: Option<NextHop>, inner: Vec<u8>) -> Vec<u8> {
        seal(
            &self.stage.endpoint.node_id(),
            &postcard::to_stdvec(&HopRecord {
                model_fingerprint: "test-model".into(),
                layer_begin: self.stage.info.layer_begin,
                layer_end: self.stage.info.layer_end,
                next,
                inner,
                ret: ReturnMode::Reverse,
            })
            .unwrap(),
        )
        .unwrap()
    }

    async fn client(&self) -> (Endpoint, Connection) {
        let ep = Endpoint::direct_client(NodeKey::generate()).unwrap();
        let conn = ep
            .connect(&self.next().addr, self.stage.endpoint.node_id())
            .await
            .unwrap();
        assert!(matches!(
            call(
                &conn,
                Request::Hello {
                    version: PROTOCOL_VERSION,
                    model_fingerprint: "test-model".into(),
                    layer_begin: -1,
                    layer_end: -1,
                }
            )
            .await,
            Response::HelloOk { .. }
        ));
        (ep, conn)
    }
}

async fn call(conn: &Connection, request: Request) -> Response {
    conn.call(&request, MAX_CONTROL_FRAME, WAIT).await.unwrap()
}

// Hold the downstream RouteSetup reply so a concurrent request deterministically
// arrives while the first route is pending.
async fn pending(
    f: &Fixture,
    client: &Connection,
) -> (Endpoint, Connection, SendStream, JoinHandle<Response>) {
    let peer =
        Endpoint::direct_server(NodeKey::generate(), "127.0.0.1:0".parse().unwrap()).unwrap();
    let nh = NextHop {
        addr: peer.local_addr().unwrap().to_string(),
        node_id: peer.node_id().0,
    };
    let hop = f.hop(Some(nh), vec![]);
    let client = client.clone();
    let task =
        tokio::spawn(async move { call(&client, Request::RouteSetup { route: 1, hop }).await });
    let conn = tokio::time::timeout(WAIT, peer.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let (mut send, mut recv) = conn.accept_bi().await.unwrap();
    assert!(matches!(
        read_frame_max::<_, Request>(&mut recv, MAX_CONTROL_FRAME)
            .await
            .unwrap(),
        Request::Hello { .. }
    ));
    reply(
        &mut send,
        &check_hello(
            &Request::Hello {
                version: PROTOCOL_VERSION,
                model_fingerprint: "test-model".into(),
                layer_begin: -1,
                layer_end: -1,
            },
            &f.stage.info,
        )
        .unwrap(),
    )
    .await
    .unwrap();
    let (send, mut recv) = conn.accept_bi().await.unwrap();
    assert!(matches!(
        read_frame_max::<_, Request>(&mut recv, MAX_CONTROL_FRAME)
            .await
            .unwrap(),
        Request::RouteSetup { route: 1, .. }
    ));
    (peer, conn, send, task)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pending_setup_reserves_the_route_number() {
    let f = Fixture::new(2, 4, 2);
    let (_ep, client) = f.client().await;
    let (_peer, _conn, _send, task) = pending(&f, &client).await;
    let response = call(
        &client,
        Request::RouteSetup {
            route: 1,
            hop: f.hop(None, vec![]),
        },
    )
    .await;
    task.abort();
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
async fn pending_setup_counts_toward_route_capacity() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, _conn, _send, task) = pending(&f, &client).await;
    let response = call(
        &client,
        Request::RouteSetup {
            route: 2,
            hop: f.hop(None, vec![]),
        },
    )
    .await;
    task.abort();
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
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_cancels_pending_downstream_setup() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, conn, _send, task) = pending(&f, &client).await;
    client.close("caller cancelled");
    task.abort();
    assert!(
        tokio::time::timeout(WAIT, conn.closed()).await.is_ok(),
        "downstream setup outlived the client"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejected_setup_does_not_keep_a_downstream_connection() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, conn, mut send, task) = pending(&f, &client).await;
    reply(&mut send, &error(ErrorCode::BadRecord, "refused"))
        .await
        .unwrap();
    assert!(matches!(
        task.await.unwrap(),
        Response::Error {
            code: ErrorCode::Downstream,
            ..
        }
    ));
    assert!(
        tokio::time::timeout(WAIT, conn.closed()).await.is_ok(),
        "failed setup retained its downstream connection"
    );
    assert!(
        matches!(
            call(
                &client,
                Request::RouteSetup {
                    route: 1,
                    hop: f.hop(None, vec![]),
                }
            )
            .await,
            Response::RouteOk { hops: 1 }
        ),
        "failed setup leaked its reservation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn different_clients_can_use_the_same_route_number_downstream() {
    let first = Fixture::new(2, 4, 2);
    let second = Fixture::new(4, 6, 2);
    let (_ep_a, a) = first.client().await;
    let (_ep_b, b) = first.client().await;
    let hop = first.hop(Some(second.next()), second.hop(None, vec![]));
    let response_a = call(
        &a,
        Request::RouteSetup {
            route: 1,
            hop: hop.clone(),
        },
    )
    .await;
    assert!(
        matches!(response_a, Response::RouteOk { hops: 2 }),
        "{response_a:?}"
    );
    let response_b = call(&b, Request::RouteSetup { route: 1, hop }).await;
    assert!(
        matches!(response_b, Response::RouteOk { hops: 2 }),
        "independent client collided: {response_b:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closing_a_route_cancels_its_pending_engine_reply() {
    let mut f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    assert!(matches!(
        call(
            &client,
            Request::RouteSetup {
                route: 1,
                hop: f.hop(None, vec![]),
            }
        )
        .await,
        Response::RouteOk { hops: 1 }
    ));
    let forward = {
        let client = client.clone();
        tokio::spawn(async move {
            call(
                &client,
                Request::Forward {
                    route: 1,
                    session: 1,
                    seq: 1,
                    pos0: 0,
                    act: Activations::from_f32(&[0.0; 64], 1, 64, ActDtype::F32),
                    hops: vec![],
                },
            )
            .await
        })
    };
    let Some(Job::Forward {
        conn: handle,
        reply,
        ..
    }) = f._jobs.recv().await
    else {
        panic!("expected forward")
    };
    assert!(matches!(
        call(&client, Request::RouteClose { route: 1 }).await,
        Response::Ack
    ));
    assert!(
        reply.is_closed(),
        "closed route still waits for an engine reply"
    );
    assert!(matches!(f._jobs.recv().await, Some(Job::CloseConn { conn }) if conn == handle));
    assert!(matches!(
        forward.await.unwrap(),
        Response::Error {
            code: ErrorCode::UnknownRoute,
            ..
        }
    ));
    // Reusing the route number after completed cleanup creates a distinct handle.
    assert!(matches!(
        call(
            &client,
            Request::RouteSetup {
                route: 1,
                hop: f.hop(None, vec![]),
            }
        )
        .await,
        Response::RouteOk { hops: 1 }
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overflowing_downstream_hop_count_is_rejected_and_released() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, conn, mut send, task) = pending(&f, &client).await;
    reply(&mut send, &Response::RouteOk { hops: u32::MAX })
        .await
        .unwrap();
    assert!(matches!(
        task.await.unwrap(),
        Response::Error {
            code: ErrorCode::Downstream,
            ..
        }
    ));
    assert!(tokio::time::timeout(WAIT, conn.closed()).await.is_ok());
    assert!(matches!(
        call(
            &client,
            Request::RouteSetup {
                route: 1,
                hop: f.hop(None, vec![]),
            }
        )
        .await,
        Response::RouteOk { hops: 1 }
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn downstream_setup_error_does_not_reveal_hidden_hop_or_peer_text() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (peer, _conn, mut send, task) = pending(&f, &client).await;
    reply(
        &mut send,
        &error(ErrorCode::BadRecord, "private-downstream-canary"),
    )
    .await
    .unwrap();
    let Response::Error { code, message } = task.await.unwrap() else {
        panic!("expected error")
    };
    assert_eq!(code, ErrorCode::Downstream);
    assert!(
        !message.contains(&peer.local_addr().unwrap().to_string()),
        "disclosed downstream address: {message}"
    );
    assert!(
        !message.contains("private-downstream-canary"),
        "reflected peer-controlled error text"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unexpected_setup_response_does_not_reflect_its_payload() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, _conn, mut send, task) = pending(&f, &client).await;
    reply(
        &mut send,
        &Response::RelayListening {
            addr: "private-response-canary".into(),
        },
    )
    .await
    .unwrap();
    let Response::Error { message, .. } = task.await.unwrap() else {
        panic!("expected error")
    };
    assert!(
        !message.contains("private-response-canary"),
        "reflected an unexpected response payload: {message}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn downstream_forward_error_does_not_reflect_peer_text() {
    let mut f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (_peer, conn, mut send, setup) = pending(&f, &client).await;
    reply(&mut send, &Response::RouteOk { hops: 1 })
        .await
        .unwrap();
    assert!(matches!(
        setup.await.unwrap(),
        Response::RouteOk { hops: 2 }
    ));
    let copy = client.clone();
    let forward = tokio::spawn(async move {
        call(
            &copy,
            Request::Forward {
                route: 1,
                session: 1,
                seq: 1,
                pos0: 0,
                act: Activations::from_f32(&[0.0; 64], 1, 64, ActDtype::F32),
                hops: vec![],
            },
        )
        .await
    });
    let Some(Job::Forward {
        act, reply: result, ..
    }) = f._jobs.recv().await
    else {
        panic!("expected engine work")
    };
    result
        .send(Reply::Forward {
            act,
            compute_ms: 1.0,
        })
        .unwrap();
    let (mut send, mut recv) = conn.accept_bi().await.unwrap();
    let request: Request = read_frame_max(&mut recv, MAX_CONTROL_FRAME).await.unwrap();
    assert!(matches!(request, Request::Forward { .. }));
    reply(
        &mut send,
        &error(ErrorCode::SessionLimit, "private-forward-canary"),
    )
    .await
    .unwrap();
    let Response::Error { code, message } = forward.await.unwrap() else {
        panic!("expected error")
    };
    assert_eq!(code, ErrorCode::SessionLimit);
    assert!(
        !message.contains("private-forward-canary"),
        "reflected a downstream forward error: {message}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn downstream_control_error_does_not_reflect_hidden_details() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let (peer, conn, mut send, setup) = pending(&f, &client).await;
    reply(&mut send, &Response::RouteOk { hops: 1 })
        .await
        .unwrap();
    assert!(matches!(
        setup.await.unwrap(),
        Response::RouteOk { hops: 2 }
    ));
    let copy = client.clone();
    let closing = tokio::spawn(async move { call(&copy, Request::RouteClose { route: 1 }).await });
    let (mut send, mut recv) = conn.accept_bi().await.unwrap();
    let request: Request = read_frame_max(&mut recv, MAX_CONTROL_FRAME).await.unwrap();
    assert!(matches!(request, Request::RouteClose { .. }));
    reply(
        &mut send,
        &error(ErrorCode::BadRequest, "private-control-canary"),
    )
    .await
    .unwrap();
    let Response::Error { code, message } = closing.await.unwrap() else {
        panic!("expected error")
    };
    assert_eq!(code, ErrorCode::Downstream);
    assert!(!message.contains("private-control-canary"));
    assert!(!message.contains(&peer.local_addr().unwrap().to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn absent_direct_return_does_not_reveal_the_last_hops_client_id() {
    let f = Fixture::new(2, 4, 1);
    let (_ep, client) = f.client().await;
    let return_id = NodeKey::generate().id();
    let hop = seal(
        &f.stage.endpoint.node_id(),
        &postcard::to_stdvec(&HopRecord {
            model_fingerprint: "test-model".into(),
            layer_begin: 2,
            layer_end: 4,
            next: None,
            inner: vec![],
            ret: ReturnMode::Direct {
                client_id: return_id.0,
            },
        })
        .unwrap(),
    )
    .unwrap();
    let Response::Error { code, message } =
        call(&client, Request::RouteSetup { route: 1, hop }).await
    else {
        panic!("expected error")
    };
    assert_eq!(code, ErrorCode::NoReturnPath);
    assert!(
        !message.contains(&return_id.short()),
        "exposed a sealed return identity: {message}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_disconnects_release_client_identities_and_engine_handles() {
    let mut f = Fixture::new(2, 4, 1);
    let mut handles = std::collections::HashSet::new();
    for _ in 0..100 {
        let (endpoint, client) = f.client().await;
        assert!(matches!(
            call(
                &client,
                Request::RouteSetup {
                    route: 1,
                    hop: f.hop(None, vec![]),
                }
            )
            .await,
            Response::RouteOk { hops: 1 }
        ));
        client.close("iteration finished");
        let Some(Job::CloseConn { conn }) =
            tokio::time::timeout(WAIT, f._jobs.recv()).await.unwrap()
        else {
            panic!("route did not enqueue engine cleanup");
        };
        assert!(handles.insert(conn), "reused an old engine handle");
        endpoint.close();
    }
    tokio::time::timeout(WAIT, async {
        while !f.stage.clients.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("disconnected client identities remained in stage state");
    assert_eq!(handles.len(), 100);
}
