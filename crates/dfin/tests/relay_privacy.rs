//! Hostile clients and stalled destinations against the real relay process.
//! The local SOCKS fixture provides no anonymity; it observes transport behavior.
#[path = "../../dfin-net/tests/support/mod.rs"]
#[allow(dead_code)]
mod support;

use dfin_net::{Connection, Endpoint, NodeId, NodeKey};
use dfin_proto::{ErrorCode, NextHop, Request, Response};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(5);

fn local() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

struct RelayFixture {
    child: Child,
    dir: PathBuf,
    endpoint: Endpoint,
    _proxy: support::Proxy,
    onion: String,
    id: NodeId,
    first_hop: NextHop,
}

impl Drop for RelayFixture {
    fn drop(&mut self) {
        self.endpoint.close();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl RelayFixture {
    async fn start(peer: SocketAddr, peer_id: NodeId, max_sessions: usize) -> Self {
        let dir =
            std::env::temp_dir().join(format!("dfin-relay-privacy-{:016x}", rand::random::<u64>()));
        std::fs::create_dir(&dir).unwrap();
        let key = NodeKey::generate();
        key.save(&dir.join("relay.key")).unwrap();
        let relay_local = local();
        let onion = support::onion('r');
        let first_hop = NextHop {
            addr: support::onion('a'),
            node_id: *peer_id.as_bytes(),
        };
        let proxy = support::Proxy::start(HashMap::from([
            (onion.clone(), relay_local),
            (first_hop.addr.clone(), peer),
        ]))
        .await;
        let log = std::fs::File::create(dir.join("relay.log")).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_dfin"))
            .current_dir(&dir)
            .args([
                "--tor-proxy",
                &proxy.addr.to_string(),
                "relay",
                "--listen",
                &relay_local.to_string(),
                "--identity",
                "relay.key",
                "--max-sessions",
                &max_sessions.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let endpoint = Endpoint::tor_client(NodeKey::generate(), proxy.addr).unwrap();
        let mut fixture = Self {
            child,
            dir,
            endpoint,
            _proxy: proxy,
            onion,
            id: key.id(),
            first_hop,
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            assert!(
                fixture.child.try_wait().unwrap().is_none(),
                "relay exited: {}",
                std::fs::read_to_string(fixture.dir.join("relay.log")).unwrap_or_default()
            );
            if let Ok(Ok(conn)) =
                tokio::time::timeout(WAIT, fixture.endpoint.connect(&fixture.onion, fixture.id))
                    .await
            {
                if matches!(
                    conn.call::<_, Response>(&Request::Ping, 64 << 10, WAIT)
                        .await,
                    Ok(Response::Pong)
                ) {
                    conn.close("ready");
                    break;
                }
            }
            assert!(Instant::now() < deadline, "relay startup timeout");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        fixture
    }

    async fn connect(&self) -> Connection {
        tokio::time::timeout(WAIT, self.endpoint.connect(&self.onion, self.id))
            .await
            .unwrap()
            .unwrap()
    }

    async fn open(&self, conn: &Connection, token: [u8; 16]) -> Response {
        conn.call(
            &Request::RelayOpen {
                token,
                first_hop: self.first_hop.clone(),
            },
            64 << 10,
            WAIT,
        )
        .await
        .unwrap()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_relay_token_cannot_replace_a_live_session() {
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(key.clone(), local(), "127.0.0.1:9050".parse().unwrap())
        .await
        .unwrap();
    let fixture = RelayFixture::start(server.local_addr().unwrap(), key.id(), 2).await;
    let first = fixture.connect().await;
    let token = [7; 16];
    assert!(matches!(fixture.open(&first, token).await, Response::Ack));
    let hop = tokio::time::timeout(WAIT, server.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let second = fixture.connect().await;
    assert!(
        matches!(
            fixture.open(&second, token).await,
            Response::Error {
                code: ErrorCode::BadRequest,
                ..
            }
        ),
        "duplicate token replaced the original session"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(200), server.accept())
            .await
            .is_err(),
        "duplicate token dialled a new destination"
    );
    assert!(!first.is_closed());
    assert!(!hop.is_closed());
    let other = fixture.connect().await;
    assert!(
        matches!(fixture.open(&other, [8; 16]).await, Response::Ack),
        "duplicate request consumed a session slot"
    );
    first.close("done");
    other.close("done");
    server.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pending_relay_dials_count_towards_the_session_limit() {
    // Accept TCP but never finish TLS, keeping the relay's outbound dial pending.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let fixture =
        RelayFixture::start(listener.local_addr().unwrap(), NodeKey::generate().id(), 1).await;
    let first = fixture.connect().await;
    let dial = Request::RelayOpen {
        token: [1; 16],
        first_hop: fixture.first_hop.clone(),
    };
    let calling = first.clone();
    let pending = tokio::spawn(async move {
        calling
            .call::<_, Response>(&dial, 64 << 10, Duration::from_secs(20))
            .await
    });
    let (stalled, _) = tokio::time::timeout(WAIT, listener.accept())
        .await
        .unwrap()
        .unwrap();
    let second = fixture.connect().await;
    let reply = fixture.open(&second, [2; 16]).await;
    assert!(
        matches!(
            reply,
            Response::Error {
                code: ErrorCode::SessionLimit,
                ..
            }
        ),
        "pending dial did not reserve capacity"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(200), listener.accept())
            .await
            .is_err()
    );
    first.close("cancel pending session");
    pending.await.unwrap().unwrap_err();
    // Closing the caller must also stop the half-open outbound connection.
    let mut stalled = stalled;
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), stalled.read_to_end(&mut bytes))
        .await
        .expect("cancelled relay dial remained open")
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn closing_relay_client_also_closes_its_idle_return_attachment() {
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(key.clone(), local(), "127.0.0.1:9050".parse().unwrap())
        .await
        .unwrap();
    let fixture = RelayFixture::start(server.local_addr().unwrap(), key.id(), 2).await;
    let client = fixture.connect().await;
    let token = [3; 16];
    assert!(matches!(fixture.open(&client, token).await, Response::Ack));
    let hop = tokio::time::timeout(WAIT, server.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let attached = fixture.connect().await;
    assert!(matches!(
        attached
            .call::<_, Response>(&Request::RelayAttach { token }, 64 << 10, WAIT)
            .await
            .unwrap(),
        Response::Ack
    ));
    client.close("caller cancelled");
    tokio::time::timeout(Duration::from_secs(2), hop.closed())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), attached.closed())
        .await
        .expect("return attachment survived its client session");
    server.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn relay_allows_one_return_path_and_invalidates_it_on_disconnect() {
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(key.clone(), local(), "127.0.0.1:9050".parse().unwrap())
        .await
        .unwrap();
    let fixture = RelayFixture::start(server.local_addr().unwrap(), key.id(), 2).await;
    let client = fixture.connect().await;
    let token = [9; 16];
    assert!(matches!(fixture.open(&client, token).await, Response::Ack));
    let hop = tokio::time::timeout(WAIT, server.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let attached = fixture.connect().await;
    assert!(matches!(
        attached
            .call::<_, Response>(&Request::RelayAttach { token }, 64 << 10, WAIT)
            .await
            .unwrap(),
        Response::Ack
    ));
    let duplicate = fixture.connect().await;
    assert!(matches!(
        duplicate
            .call::<_, Response>(&Request::RelayAttach { token }, 64 << 10, WAIT)
            .await
            .unwrap(),
        Response::Error {
            code: ErrorCode::BadRequest,
            ..
        }
    ));
    duplicate.close("rejected duplicate");
    assert!(
        !client.is_closed(),
        "rejecting an extra attachment killed the owner"
    );
    attached.close("return path lost");
    tokio::time::timeout(Duration::from_secs(2), client.closed())
        .await
        .expect("client survived its return path");
    tokio::time::timeout(Duration::from_secs(2), hop.closed())
        .await
        .unwrap();

    // Reusing a token after its owner closes is allowed. Late cleanup of the old
    // session must not remove the new owner or prevent its return attachment.
    let replacement = fixture.connect().await;
    assert!(matches!(
        fixture.open(&replacement, token).await,
        Response::Ack
    ));
    let _new_hop = tokio::time::timeout(WAIT, server.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let new_return = fixture.connect().await;
    assert!(matches!(
        new_return
            .call::<_, Response>(&Request::RelayAttach { token }, 64 << 10, WAIT)
            .await
            .unwrap(),
        Response::Ack
    ));
    replacement.close("done");
    server.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn relay_requires_a_first_stream_within_the_control_deadline() {
    let fixture = RelayFixture::start(local(), NodeKey::generate().id(), 1).await;
    let idle = fixture.connect().await;
    tokio::time::timeout(
        dfin_net::transport::CONTROL_TIMEOUT + Duration::from_secs(5),
        idle.closed(),
    )
    .await
    .expect("authenticated idle client held a relay listener slot indefinitely");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "waits for the real two-minute relay deadline; run explicitly in CI"]
async fn relay_expires_an_unattached_session() {
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(key.clone(), local(), "127.0.0.1:9050".parse().unwrap())
        .await
        .unwrap();
    let fixture = RelayFixture::start(server.local_addr().unwrap(), key.id(), 1).await;
    let client = fixture.connect().await;
    assert!(matches!(
        fixture.open(&client, [4; 16]).await,
        Response::Ack
    ));
    let hop = tokio::time::timeout(WAIT, server.accept())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(125), client.closed())
        .await
        .expect("unattached session survived its advertised two-minute deadline");
    tokio::time::timeout(Duration::from_secs(2), hop.closed())
        .await
        .unwrap();
    server.close();
}
