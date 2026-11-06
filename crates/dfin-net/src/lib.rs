//! Node identities and the authenticated QUIC transport.
//!
//! Every participant has an Ed25519 key. Its public key is the *node id*. Connections
//! are QUIC with TLS 1.3; each side presents a self-signed certificate carrying its
//! Ed25519 key, the dialer pins the node id it expects, and the listener accepts any
//! client certificate and remembers its key as the peer id. There are no certificate
//! authorities anywhere.
//!
//! The same key, converted to X25519, receives sealed route records ([`seal`] and
//! [`unseal`]), so one key per node is all that is ever distributed.
//!
//! See `docs/phase1-protocol.md`.

pub mod identity;
pub mod sealed;
mod tls;
pub mod transport;

pub use identity::{NodeId, NodeKey};
pub use sealed::{seal, unseal};
pub use transport::{Connection, Endpoint, NetError, ALPN};
pub use quinn::{RecvStream, SendStream};
