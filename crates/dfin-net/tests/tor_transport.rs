mod support;
use dfin_net::transport::{reply, CONTROL_TIMEOUT, MAX_CONTROL_FRAME};
use dfin_net::{Endpoint, NodeId, NodeKey};
use dfin_proto::read_frame_max;
use std::{collections::HashMap, time::Duration};
use support::{onion, Proxy};

#[tokio::test]
async fn tor_tls_multiplexes_large_requests_and_both_return_directions() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let key = NodeKey::generate();
        let server = Endpoint::tor_server(
            key.clone(),
            "127.0.0.1:0".parse().unwrap(),
            "127.0.0.1:9050".parse().unwrap(),
        )
        .await
        .unwrap();
        let mut proxy =
            Proxy::start(HashMap::from([(onion('a'), server.local_addr().unwrap())])).await;
        let client_key = NodeKey::generate();
        let client = Endpoint::tor_client(client_key.clone(), proxy.addr).unwrap();
        let target = onion('a');
        let (outgoing, incoming) = tokio::join!(client.connect(&target, key.id()), server.accept());
        let outgoing = outgoing.unwrap();
        let incoming = incoming.unwrap().unwrap();
        assert_eq!(incoming.peer_id(), client_key.id());
        assert_eq!(outgoing.peer_id(), key.id());
        let serve = tokio::spawn({
            let incoming = incoming.clone();
            async move {
                let mut jobs = tokio::task::JoinSet::new();
                for _ in 0..6 {
                    let (mut send, mut recv) = incoming.accept_bi().await.unwrap();
                    jobs.spawn(async move {
                        let data: Vec<u8> = read_frame_max(&mut recv, 8 << 20).await.unwrap();
                        reply(&mut send, &data).await.unwrap();
                    });
                }
                while let Some(result) = jobs.join_next().await {
                    result.unwrap();
                }
            }
        });
        let mut calls = tokio::task::JoinSet::new();
        for i in 0..6 {
            let conn = outgoing.clone();
            calls.spawn(async move {
                let data = vec![i; 5 << 20];
                let got: Vec<u8> = conn.call(&data, 8 << 20, CONTROL_TIMEOUT).await.unwrap();
                assert_eq!(got, data);
            });
        }
        while let Some(result) = calls.join_next().await {
            result.unwrap();
        }
        serve.await.unwrap();
        incoming.send_uni(&"return".to_string()).await.unwrap();
        let mut recv = outgoing.accept_uni().await.unwrap();
        assert_eq!(
            read_frame_max::<_, String>(&mut recv, MAX_CONTROL_FRAME)
                .await
                .unwrap(),
            "return"
        );
        outgoing.send_uni(&"reverse".to_string()).await.unwrap();
        let mut recv = incoming.accept_uni().await.unwrap();
        assert_eq!(
            read_frame_max::<_, String>(&mut recv, MAX_CONTROL_FRAME)
                .await
                .unwrap(),
            "reverse"
        );
        let first = proxy.observed.recv().await.unwrap();
        assert_eq!(first.0, onion('a'));
        let fresh = client.isolated_client().unwrap();
        assert_ne!(fresh.node_id(), client.node_id());
        let (second, accepted) = tokio::join!(fresh.connect(&target, key.id()), server.accept());
        let _second = second.unwrap();
        let _accepted = accepted.unwrap().unwrap();
        assert_ne!(
            first.1,
            proxy.observed.recv().await.unwrap().1,
            "SOCKS credentials must rotate"
        );
        outgoing.close("test complete");
        incoming.closed().await;
        assert!(incoming.accept_bi().await.is_err());
        server.close();
        assert!(server.accept().await.is_none());
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn wrong_tls_identity_and_unreachable_onion_fail_closed() {
    let key = NodeKey::generate();
    let server = Endpoint::tor_server(
        key,
        "127.0.0.1:0".parse().unwrap(),
        "127.0.0.1:9050".parse().unwrap(),
    )
    .await
    .unwrap();
    let proxy = Proxy::start(HashMap::from([(onion('a'), server.local_addr().unwrap())])).await;
    let client = Endpoint::tor_client(NodeKey::generate(), proxy.addr).unwrap();
    let error = client
        .connect(&onion('a'), NodeKey::generate().id())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("mismatch"), "{error}");
    assert!(client
        .connect(&onion('b'), NodeKey::generate().id())
        .await
        .is_err());
}

#[tokio::test]
async fn forbidden_addresses_never_even_contact_the_proxy() {
    let mut proxy = Proxy::start(HashMap::new()).await;
    let client = Endpoint::tor_client(NodeKey::generate(), proxy.addr).unwrap();
    for target in [
        "127.0.0.1:7001",
        "[::1]:7001",
        "example.com:443",
        "localhost:7001",
    ] {
        assert!(client
            .connect(target, NodeKey::generate().id())
            .await
            .is_err());
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(50), proxy.observed.recv())
            .await
            .is_err()
    );
    assert_eq!(
        proxy.connections.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert!(Endpoint::tor_server(
        NodeKey::generate(),
        "0.0.0.0:0".parse().unwrap(),
        proxy.addr
    )
    .await
    .is_err());
    client.close();
    assert!(client
        .connect(&onion('a'), NodeKey::generate().id())
        .await
        .is_err());
}

#[tokio::test]
async fn weak_raw_destination_id_never_contacts_the_proxy() {
    let proxy = Proxy::start(HashMap::new()).await;
    let endpoint = Endpoint::tor_client(NodeKey::generate(), proxy.addr).unwrap();
    let mut identity = [0u8; 32];
    identity[0] = 1;
    for id in [NodeId(identity), NodeId([0; 32])] {
        assert!(id.verifying_key().is_err());
        assert!(endpoint.connect(&onion('a'), id).await.is_err());
    }
    assert_eq!(
        proxy.connections.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "invalid protocol-supplied identities must be rejected before dialing"
    );
}
