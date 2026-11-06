//! Signing and verifying node records and directories.
//!
//! A record is signed by the node it describes; a directory is signed by the hub that
//! published it. Both signatures are Ed25519 over the postcard bytes of the body, with
//! a domain separator so a record can never be mistaken for a directory.

use crate::identity::{NodeId, NodeKey};
use dfin_proto::{Directory, NodeRecord, SignedDirectory, SignedRecord};
use ed25519_dalek::{Signature, Signer, Verifier};

const RECORD_DOMAIN: &[u8] = b"dfin-node-record-v1\0";
const DIRECTORY_DOMAIN: &[u8] = b"dfin-directory-v1\0";

#[derive(Debug, thiserror::Error)]
pub enum SignError {
    #[error("record does not encode: {0}")]
    Encode(String),
    #[error("body does not decode: {0}")]
    Decode(String),
    #[error("signature does not verify")]
    Signature,
    #[error("signer id in the body is not a valid key")]
    BadSigner,
}

fn message(domain: &[u8], body: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(domain.len() + body.len());
    m.extend_from_slice(domain);
    m.extend_from_slice(body);
    m
}

pub fn sign_record(key: &NodeKey, record: &NodeRecord) -> Result<SignedRecord, SignError> {
    let body = postcard::to_stdvec(record).map_err(|e| SignError::Encode(e.to_string()))?;
    let sig = key.signing_key().sign(&message(RECORD_DOMAIN, &body));
    Ok(SignedRecord {
        body,
        signature: sig.to_bytes().to_vec(),
    })
}

/// Decode the record and check that the node named inside signed it.
pub fn verify_record(signed: &SignedRecord) -> Result<NodeRecord, SignError> {
    let record: NodeRecord =
        postcard::from_bytes(&signed.body).map_err(|e| SignError::Decode(e.to_string()))?;
    verify(
        &NodeId(record.node_id),
        RECORD_DOMAIN,
        &signed.body,
        &signed.signature,
    )?;
    Ok(record)
}

pub fn sign_directory(key: &NodeKey, dir: &Directory) -> Result<SignedDirectory, SignError> {
    let body = postcard::to_stdvec(dir).map_err(|e| SignError::Encode(e.to_string()))?;
    let sig = key.signing_key().sign(&message(DIRECTORY_DOMAIN, &body));
    Ok(SignedDirectory {
        body,
        signature: sig.to_bytes().to_vec(),
    })
}

/// Decode the directory and check that `hub` signed it and is the hub named inside.
pub fn verify_directory(signed: &SignedDirectory, hub: &NodeId) -> Result<Directory, SignError> {
    let dir: Directory =
        postcard::from_bytes(&signed.body).map_err(|e| SignError::Decode(e.to_string()))?;
    if dir.hub_id != hub.0 {
        return Err(SignError::Signature);
    }
    verify(hub, DIRECTORY_DOMAIN, &signed.body, &signed.signature)?;
    Ok(dir)
}

fn verify(id: &NodeId, domain: &[u8], body: &[u8], sig: &[u8]) -> Result<(), SignError> {
    let vk = id.verifying_key().map_err(|_| SignError::BadSigner)?;
    let sig = Signature::from_slice(sig).map_err(|_| SignError::Signature)?;
    vk.verify(&message(domain, body), &sig)
        .map_err(|_| SignError::Signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfin_proto::{ActDtype, DirectoryEntry, PROTOCOL_VERSION};

    fn record(key: &NodeKey) -> NodeRecord {
        NodeRecord {
            node_id: *key.id().as_bytes(),
            addrs: vec!["10.0.0.5:7001".into()],
            model_fingerprint: "abc".into(),
            model_name: "tiny".into(),
            layer_begin: 2,
            layer_end: 5,
            n_layer: 8,
            n_embd: 64,
            n_ctx_seq: 1024,
            n_seq_max: 4,
            act_dtype: ActDtype::F16,
            backend: "CPU".into(),
            protocol: PROTOCOL_VERSION,
            region: "us".into(),
            attested: false,
            cloaked: false,
            issued_at: 1_800_000_000,
            ttl_secs: 180,
        }
    }

    #[test]
    fn record_roundtrip_and_tamper() {
        let key = NodeKey::generate();
        let r = record(&key);
        let signed = sign_record(&key, &r).unwrap();
        assert_eq!(verify_record(&signed).unwrap(), r);
        let mut bad = signed.clone();
        bad.body[0] ^= 1;
        assert!(verify_record(&bad).is_err());
        // Signed by another key but naming this node: refused.
        let other = NodeKey::generate();
        let forged = sign_record(&other, &r).unwrap();
        assert!(verify_record(&forged).is_err());
    }

    #[test]
    fn directory_roundtrip_and_wrong_hub() {
        let node = NodeKey::generate();
        let hub = NodeKey::generate();
        let dir = Directory {
            hub_id: *hub.id().as_bytes(),
            issued_at: 1_800_000_000,
            entries: vec![DirectoryEntry {
                record: sign_record(&node, &record(&node)).unwrap(),
                hub_rtt_ms: 1.5,
                seen_at: 1_800_000_000,
            }],
        };
        let signed = sign_directory(&hub, &dir).unwrap();
        assert_eq!(verify_directory(&signed, &hub.id()).unwrap(), dir);
        assert!(verify_directory(&signed, &NodeKey::generate().id()).is_err());
        // A record cannot pose as a directory even with the same key.
        let rec = sign_record(&hub, &record(&hub)).unwrap();
        let posed = SignedDirectory {
            body: rec.body,
            signature: rec.signature,
        };
        assert!(verify_directory(&posed, &hub.id()).is_err());
    }
}
