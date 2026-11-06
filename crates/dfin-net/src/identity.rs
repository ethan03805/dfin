//! Ed25519 node keys and ids.

use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io;
use std::path::Path;
use std::str::FromStr;

/// A node's public key: 32 bytes, shown as 64 hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn verifying_key(&self) -> Result<VerifyingKey, IdentityError> {
        VerifyingKey::from_bytes(&self.0).map_err(|_| IdentityError::InvalidKey)
    }

    /// The X25519 public key that receives sealed records for this node.
    pub fn x25519(&self) -> Result<x25519_dalek::PublicKey, IdentityError> {
        let vk = self.verifying_key()?;
        Ok(x25519_dalek::PublicKey::from(vk.to_montgomery().to_bytes()))
    }

    /// First 8 hex characters, for logs.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..4])
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.short())
    }
}

impl FromStr for NodeId {
    type Err = IdentityError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s.trim()).map_err(|_| IdentityError::BadIdText)?;
        let arr: [u8; 32] = bytes.try_into().map_err(|_| IdentityError::BadIdText)?;
        // Reject ids that are not valid curve points now, so routes fail at parse time.
        NodeId(arr).verifying_key()?;
        Ok(NodeId(arr))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("node id must be 64 hex characters")]
    BadIdText,
    #[error("node id is not a valid Ed25519 public key")]
    InvalidKey,
    #[error("key file {path}: {err}")]
    File { path: String, err: io::Error },
    #[error("key file {path} does not hold 32 hex-encoded bytes")]
    BadKeyFile { path: String },
}

/// A node's private key. Cloneable; the secret is zeroised only when every clone drops.
#[derive(Clone)]
pub struct NodeKey {
    signing: SigningKey,
}

impl fmt::Debug for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeKey({})", self.id().short())
    }
}

impl NodeKey {
    pub fn generate() -> Self {
        let bytes: [u8; 32] = rand::random();
        NodeKey::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        NodeKey {
            signing: SigningKey::from_bytes(&bytes),
        }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    pub fn id(&self) -> NodeId {
        NodeId(self.signing.verifying_key().to_bytes())
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing
    }

    /// The X25519 secret matching [`NodeId::x25519`]: the clamped, unreduced Ed25519
    /// scalar (the first half of SHA-512 of the seed), which is what the public key
    /// was multiplied by. A reduced scalar would clamp to a different value.
    pub fn x25519(&self) -> x25519_dalek::StaticSecret {
        use sha2::{Digest, Sha512};
        let h = Sha512::digest(self.signing.to_bytes());
        let mut s = [0u8; 32];
        s.copy_from_slice(&h[..32]);
        s[0] &= 248;
        s[31] &= 127;
        s[31] |= 64;
        x25519_dalek::StaticSecret::from(s)
    }

    /// Load the key at `path` (64 hex characters), or create and save a new one.
    pub fn load_or_create(path: &Path) -> Result<Self, IdentityError> {
        let p = path.display().to_string();
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let bytes = hex::decode(text.trim())
                    .map_err(|_| IdentityError::BadKeyFile { path: p.clone() })?;
                let arr: [u8; 32] = bytes
                    .try_into()
                    .map_err(|_| IdentityError::BadKeyFile { path: p.clone() })?;
                Ok(NodeKey::from_bytes(arr))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let key = NodeKey::generate();
                key.save(path)?;
                Ok(key)
            }
            Err(err) => Err(IdentityError::File { path: p, err }),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), IdentityError> {
        let p = path.display().to_string();
        let wrap = |err| IdentityError::File { path: p.clone(), err };
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(wrap)?;
            }
        }
        let text = hex::encode(self.to_bytes()) + "\n";
        std::fs::write(path, text).map_err(wrap)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_roundtrip_and_validation() {
        let k = NodeKey::generate();
        let id = k.id();
        let text = id.to_string();
        assert_eq!(text.len(), 64);
        assert_eq!(text.parse::<NodeId>().unwrap(), id);
        assert!("zz".parse::<NodeId>().is_err());
        assert!(hex::encode([0u8; 31]).parse::<NodeId>().is_err());
    }

    #[test]
    fn x25519_keys_agree() {
        let a = NodeKey::generate();
        let b = NodeKey::generate();
        let ab = a.x25519().diffie_hellman(&b.id().x25519().unwrap());
        let ba = b.x25519().diffie_hellman(&a.id().x25519().unwrap());
        assert_eq!(ab.as_bytes(), ba.as_bytes());
    }

    #[test]
    fn key_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dfin-net-key-{}", rand::random::<u64>()));
        let path = dir.join("node.key");
        let k1 = NodeKey::load_or_create(&path).unwrap();
        let k2 = NodeKey::load_or_create(&path).unwrap();
        assert_eq!(k1.id(), k2.id());
        let _ = std::fs::remove_dir_all(dir);
    }
}
