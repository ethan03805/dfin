//! Node identities and authenticated Tor/TLS and explicit direct QUIC transports.
//!
//! Every participant has an Ed25519 key. Its public key is the *node id*. Connections
//! use TLS 1.3; each side presents a self-signed certificate carrying its
//! Ed25519 key, the dialer pins the node id it expects, and the listener accepts any
//! client certificate and remembers its key as the peer id. There are no certificate
//! authorities anywhere.
//!
//! The same key, converted to X25519, receives sealed route records ([`seal`] and
//! [`unseal`]), so one key per node is all that is ever distributed.
//!
//! Tor carries multiplexed TCP streams and enforces onion-only targets; direct QUIC
//! must be selected explicitly. See `docs/anonymity.md` and `docs/protocol.md`.

pub mod directory;
pub mod identity;
pub mod privacy;
mod quic;
pub mod sealed;
mod stream;
mod tls;
pub mod transport;

pub use identity::{NodeId, NodeKey};
pub use sealed::{seal, unseal};
pub use transport::{Connection, Endpoint, NetError, RecvStream, SendStream, ALPN};
