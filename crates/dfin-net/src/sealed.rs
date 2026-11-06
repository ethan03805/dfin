//! Sealed boxes to a node id: anyone can seal, only the node can open.
//!
//! `ephemeral X25519 || ChaCha20-Poly1305(plaintext)`, keyed by
//! `HKDF-SHA256(ikm = shared secret, salt = ephemeral_pub || recipient_pub,
//! info = "dfin-hop-seal-v1")` with a zero nonce, which is safe because every
//! ephemeral key is used once.

use crate::identity::{IdentityError, NodeId, NodeKey};
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;

const INFO: &[u8] = b"dfin-hop-seal-v1";

#[derive(Debug, thiserror::Error)]
pub enum SealError {
    #[error("{0}")]
    Identity(#[from] IdentityError),
    #[error("sealed data is too short")]
    TooShort,
    #[error("sealed data does not open with this key")]
    Open,
    #[error("sealing failed")]
    Seal,
    #[error("degenerate key exchange")]
    Degenerate,
}

fn derive(shared: &[u8], eph_pub: &[u8; 32], recipient: &[u8; 32]) -> [u8; 32] {
    let mut salt = Vec::with_capacity(64);
    salt.extend_from_slice(eph_pub);
    salt.extend_from_slice(recipient);
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut key = [0u8; 32];
    hk.expand(INFO, &mut key).expect("32 bytes is a valid HKDF output length");
    key
}

pub fn seal(to: &NodeId, plaintext: &[u8]) -> Result<Vec<u8>, SealError> {
    let recipient = to.x25519()?;
    let eph = x25519_dalek::StaticSecret::from(rand::random::<[u8; 32]>());
    let eph_pub = x25519_dalek::PublicKey::from(&eph);
    let shared = eph.diffie_hellman(&recipient);
    if !shared.was_contributory() {
        return Err(SealError::Degenerate);
    }
    let key = derive(shared.as_bytes(), eph_pub.as_bytes(), recipient.as_bytes());
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));
    let ct = cipher
        .encrypt(Nonce::from_slice(&[0u8; 12]), plaintext)
        .map_err(|_| SealError::Seal)?;
    let mut out = Vec::with_capacity(32 + ct.len());
    out.extend_from_slice(eph_pub.as_bytes());
    out.extend_from_slice(&ct);
    Ok(out)
}

pub fn unseal(key: &NodeKey, sealed: &[u8]) -> Result<Vec<u8>, SealError> {
    if sealed.len() < 32 + 16 {
        return Err(SealError::TooShort);
    }
    let mut eph_bytes = [0u8; 32];
    eph_bytes.copy_from_slice(&sealed[..32]);
    let eph_pub = x25519_dalek::PublicKey::from(eph_bytes);
    let me = key.x25519();
    let my_pub = x25519_dalek::PublicKey::from(&me);
    let shared = me.diffie_hellman(&eph_pub);
    if !shared.was_contributory() {
        return Err(SealError::Degenerate);
    }
    let k = derive(shared.as_bytes(), eph_pub.as_bytes(), my_pub.as_bytes());
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&k));
    cipher
        .decrypt(Nonce::from_slice(&[0u8; 12]), &sealed[32..])
        .map_err(|_| SealError::Open)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_roundtrip() {
        let k = NodeKey::generate();
        let msg = b"layers 3:7, next 10.0.0.5:7001";
        let sealed = seal(&k.id(), msg).unwrap();
        assert_eq!(unseal(&k, &sealed).unwrap(), msg);
        assert_ne!(seal(&k.id(), msg).unwrap(), sealed, "ephemeral keys differ");
    }

    #[test]
    fn wrong_key_fails() {
        let k = NodeKey::generate();
        let other = NodeKey::generate();
        let sealed = seal(&k.id(), b"secret").unwrap();
        assert!(unseal(&other, &sealed).is_err());
        let mut tampered = sealed.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(unseal(&k, &tampered).is_err());
    }
}
