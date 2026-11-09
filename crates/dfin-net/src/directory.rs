//! Signing and verifying node records and directories.
//!
//! A record is signed by the node it describes; a directory is signed by the hub that
//! published it. Both signatures are Ed25519 over the postcard bytes of the body, with
//! a domain separator so a record can never be mistaken for a directory.

use crate::identity::{NodeId, NodeKey};
use dfin_proto::{
    Directory, ErrorCode, NodeRecord, SignedDirectory, SignedRecord, MAX_CLOCK_SKEW_SECS,
    MAX_RECORD_TTL_SECS, PROTOCOL_VERSION,
};
use ed25519_dalek::{Signature, Signer};

// The version in each domain is that of the signed layout. v2 dropped the record's
// `region` and `cloaked` fields and the entry's per-probe round trips.
const RECORD_DOMAIN: &[u8] = b"dfin-node-record-v2\0";
const DIRECTORY_DOMAIN: &[u8] = b"dfin-directory-v2\0";

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

/// Check the bounds every consumer needs, even when both signatures are valid.
/// This validates claims, not their truth or the directory's completeness.
/// Old records remain usable until their bounded TTL expires; hubs separately
/// impose a tighter age limit on newly received announcements.
pub fn validate_record(record: &NodeRecord, now: u64) -> Result<(), (ErrorCode, &'static str)> {
    let bad = |message| Err((ErrorCode::BadRequest, message));
    if record.protocol != PROTOCOL_VERSION {
        return Err((ErrorCode::VersionMismatch, "record protocol does not match"));
    }
    if record.addrs.is_empty() || record.addrs.len() > 8 {
        return bad("a record names between one and eight addresses");
    }
    if record.addrs.iter().any(|a| a.is_empty() || a.len() > 128) {
        return bad("address empty or too long");
    }
    if record.model_fingerprint.len() > 128
        || record.model_name.len() > 256
        || record.backend.len() > 256
    {
        return bad("text field too long");
    }
    if record.layer_begin <= 0
        || record.layer_end <= record.layer_begin
        || record.layer_end > record.n_layer
    {
        return bad("layer range must be a middle piece inside the model");
    }
    if record.ttl_secs == 0 || record.ttl_secs > MAX_RECORD_TTL_SECS {
        return Err((ErrorCode::BadTimestamp, "ttl out of range"));
    }
    let Some(expires) = record.issued_at.checked_add(record.ttl_secs as u64) else {
        return Err((ErrorCode::BadTimestamp, "record lifetime overflows"));
    };
    if expires <= now {
        return Err((ErrorCode::BadTimestamp, "record expired"));
    }
    if record.issued_at.saturating_sub(now) > MAX_CLOCK_SKEW_SECS {
        return Err((
            ErrorCode::BadTimestamp,
            "record timestamp is too far in the future",
        ));
    }
    Ok(())
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
    // The expected signer is already known: authenticate bytes before decoding.
    verify(hub, DIRECTORY_DOMAIN, &signed.body, &signed.signature)?;
    let dir: Directory =
        postcard::from_bytes(&signed.body).map_err(|e| SignError::Decode(e.to_string()))?;
    if dir.hub_id != hub.0 {
        return Err(SignError::Signature);
    }
    Ok(dir)
}

fn verify(id: &NodeId, domain: &[u8], body: &[u8], sig: &[u8]) -> Result<(), SignError> {
    let vk = id.verifying_key().map_err(|_| SignError::BadSigner)?;
    let sig = Signature::from_slice(sig).map_err(|_| SignError::Signature)?;
    vk.verify_strict(&message(domain, body), &sig)
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
            attested: false,
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

    #[test]
    fn weak_identity_cannot_forge_records_or_directories_without_a_secret() {
        // The Edwards identity has order one. With A=R=identity and S=0,
        // ordinary Ed25519 verification accepts any message without a secret.
        let mut identity = [0u8; 32];
        identity[0] = 1;
        let mut signature = vec![0u8; 64];
        signature[0] = 1;
        let mut r = record(&NodeKey::generate());
        r.node_id = identity;
        let forged_record = SignedRecord {
            body: postcard::to_stdvec(&r).unwrap(),
            signature: signature.clone(),
        };
        assert!(
            verify_record(&forged_record).is_err(),
            "accepted a record without a signing secret"
        );
        let dir = Directory {
            hub_id: identity,
            issued_at: r.issued_at,
            entries: vec![],
        };
        let forged_directory = SignedDirectory {
            body: postcard::to_stdvec(&dir).unwrap(),
            signature,
        };
        assert!(verify_directory(&forged_directory, &NodeId(identity)).is_err());
    }
}
