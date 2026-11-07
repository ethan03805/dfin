//! Test-only SOCKS fixture. This forwards locally and provides NO anonymity.
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    task::{JoinHandle, JoinSet},
};

pub struct Proxy {
    pub addr: SocketAddr,
    pub connections: Arc<AtomicUsize>,
    pub observed: mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    task: JoinHandle<()>,
}
impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Proxy {
    pub async fn start(targets: HashMap<String, SocketAddr>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let targets = Arc::new(targets);
        let (tx, observed) = mpsc::unbounded_channel();
        let connections = Arc::new(AtomicUsize::new(0));
        let counter = connections.clone();
        let task = tokio::spawn(async move {
            let mut handlers = JoinSet::new();
            loop {
                tokio::select! {
                    r = listener.accept() => {
                        let Ok((mut s, _)) = r else { break };
                        counter.fetch_add(1, Ordering::Relaxed);
                        let targets = targets.clone();
                        let tx = tx.clone();
                        handlers.spawn(async move {
                            let result = async {
                                let mut greeting = [0; 4];
                                s.read_exact(&mut greeting).await?;
                                assert_eq!(greeting, [5, 2, 2, 0]);
                                s.write_all(&[5, 2]).await?;
                                assert_eq!(s.read_u8().await?, 1);
                                let n = s.read_u8().await? as usize;
                                let mut username = vec![0; n];
                                s.read_exact(&mut username).await?;
                                let n = s.read_u8().await? as usize;
                                let mut password = vec![0; n];
                                s.read_exact(&mut password).await?;
                                assert_eq!(password, b"dfin");
                                s.write_all(&[1, 0]).await?;
                                let mut request = [0; 4];
                                s.read_exact(&mut request).await?;
                                assert_eq!(request, [5, 1, 0, 3], "target must remain a hostname, not a resolved IP");
                                let n = s.read_u8().await? as usize;
                                let mut host = vec![0; n];
                                s.read_exact(&mut host).await?;
                                let port = s.read_u16().await?;
                                let target = format!("{}:{port}", String::from_utf8(host).unwrap());
                                let _ = tx.send((target.clone(), username));
                                let Some(addr) = targets.get(&target) else {
                                    s.write_all(&[5, 4, 0, 1, 0, 0, 0, 0, 0, 0]).await?;
                                    return Ok::<_, std::io::Error>(());
                                };
                                let mut upstream = TcpStream::connect(addr).await?;
                                s.write_all(&[5, 0, 0, 1, 127, 0, 0, 1, 0, 0]).await?;
                                tokio::io::copy_bidirectional(&mut s, &mut upstream).await?;
                                Ok(())
                            }.await;
                            let _ = result; // Disconnects and deliberately bad TLS are expected.
                        });
                    },
                    Some(r) = handlers.join_next() => { r.unwrap(); },
                }
            }
        });
        Self {
            addr,
            connections,
            observed,
            task,
        }
    }
}
pub fn onion(letter: char) -> String {
    format!("{}.onion:7001", letter.to_string().repeat(56))
}
