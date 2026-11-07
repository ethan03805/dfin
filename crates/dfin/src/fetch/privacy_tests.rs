//! Hostile download sources and interrupted writes, using only temporary test files.
use super::*;
use dfin_net::{
    transport::{reply, MAX_CONTROL_FRAME},
    Connection,
};
use dfin_proto::read_frame_max;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinHandle;

struct Folder(PathBuf);
impl Folder {
    fn new() -> Self {
        let p =
            std::env::temp_dir().join(format!("dfin-download-test-{:016x}", rand::random::<u64>()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        // This fixture creates only files in its own new directory; do not recurse.
        if let Ok(entries) = std::fs::read_dir(&self.0) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn split(size: u64, bytes: &[u8]) -> Split {
    Split {
        no: 0,
        file: "test.gguf".into(),
        size,
        sha256: hex::encode(Sha256::digest(bytes)),
        kind: "meta".into(),
        layer: None,
        tensors: vec![],
    }
}

fn manifest(split: &Split) -> Manifest {
    Manifest {
        format: crate::shards::FORMAT.into(),
        model_name: "test".into(),
        arch: "llama".into(),
        n_layer: 1,
        first: split.file.clone(),
        splits: vec![split.clone()],
        seeds: vec![],
        model_id: "test-id".into(),
    }
}

async fn seed(response: &'static [u8]) -> (String, JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0u8; 4096];
        let _ = socket.read(&mut request).await.unwrap();
        let _ = socket.write_all(response).await;
    });
    (url, task)
}

#[tokio::test]
async fn completed_download_preserves_an_existing_partial_file() {
    let folder = Folder::new();
    let old = folder.0.join("test.gguf.part");
    std::fs::write(&old, b"existing unfinished work").unwrap();
    let (url, server) =
        seed(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nGOOD").await;
    from_seed(&reqwest::Client::new(), &folder.0, &split(4, b"GOOD"), &url)
        .await
        .unwrap();
    server.await.unwrap();
    assert_eq!(std::fs::read(folder.0.join("test.gguf")).unwrap(), b"GOOD");
    assert_eq!(
        std::fs::read(&old).ok().as_deref(),
        Some(b"existing unfinished work".as_slice()),
        "overwrote or removed a pre-existing partial file"
    );
}

#[tokio::test]
async fn chunked_seed_cannot_exceed_manifest_size_even_with_a_matching_hash() {
    let folder = Folder::new();
    let (url, server) = seed(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n7\r\nTOOLONG\r\n0\r\n\r\n").await;
    let result = from_seed(
        &reqwest::Client::new(),
        &folder.0,
        &split(1, b"TOOLONG"),
        &url,
    )
    .await;
    server.await.unwrap();
    assert!(
        result.is_err(),
        "accepted more bytes than the manifest declares"
    );
    assert!(std::fs::read_dir(&folder.0).unwrap().next().is_none());
}

async fn answer_shard(conn: &Connection, total: u64, data: Vec<u8>) {
    let (mut send, mut recv) = conn.accept_bi().await.unwrap();
    let request: Request = read_frame_max(&mut recv, MAX_CONTROL_FRAME).await.unwrap();
    assert!(matches!(request, Request::ShardGet { .. }));
    reply(&mut send, &Response::Shard { total, data })
        .await
        .unwrap();
}

#[tokio::test]
async fn peer_cannot_exceed_the_requested_range_even_with_a_matching_hash() {
    let folder = Folder::new();
    let endpoint = Endpoint::direct_client(NodeKey::generate()).unwrap();
    let server =
        Endpoint::direct_server(NodeKey::generate(), "127.0.0.1:0".parse().unwrap()).unwrap();
    let peer = Peer {
        addr: server.local_addr().unwrap().to_string(),
        id: server.node_id(),
        begin: 0,
        end: 1,
    };
    let shard = split(1, b"TOOLONG");
    let copy = server.clone();
    let task = tokio::spawn(async move {
        let conn = copy.accept().await.unwrap().unwrap();
        answer_shard(&conn, 1, b"TOOLONG".to_vec()).await;
        let _ = conn.closed().await;
    });
    let result = from_peer(&endpoint, &manifest(&shard), &folder.0, &shard, &peer).await;
    server.close();
    task.abort();
    endpoint.close();
    assert!(
        result.is_err(),
        "accepted a response larger than the requested shard range"
    );
    assert!(std::fs::read_dir(&folder.0).unwrap().next().is_none());
}

#[tokio::test]
async fn cancelling_a_download_removes_its_partial_file() {
    let folder = Folder::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let dir = folder.0.clone();
    let task = tokio::spawn(async move {
        from_seed(&reqwest::Client::new(), &dir, &split(4, b"GOOD"), &url).await
    });
    let (mut socket, _) = listener.accept().await.unwrap();
    let mut request = [0u8; 4096];
    socket.read(&mut request).await.unwrap();
    socket
        .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nGO\r\n")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if std::fs::read_dir(&folder.0)
                .unwrap()
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "part"))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    let _ = task.await;
    drop(socket);
    // Creation runs on a blocking worker; allow its owned result to be dropped
    // even when cancellation happens just before it returns to the caller.
    tokio::time::timeout(Duration::from_secs(3), async {
        while std::fs::read_dir(&folder.0).unwrap().next().is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancelled download left its partial file behind");
}

#[tokio::test]
async fn short_or_bad_hash_seed_does_not_install_or_keep_a_partial_file() {
    for response in [
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nGO\r\n0\r\n\r\n".as_slice(),
        b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nEVIL".as_slice(),
    ] {
        let folder = Folder::new();
        let (url, server) = seed(response).await;
        assert!(from_seed(&reqwest::Client::new(), &folder.0, &split(4, b"GOOD"), &url).await.is_err());
        server.await.unwrap();
        assert!(std::fs::read_dir(&folder.0).unwrap().next().is_none());
    }
}
