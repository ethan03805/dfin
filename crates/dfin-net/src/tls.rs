//! Self-signed certificates bound to node keys, and the verifiers that pin them.

use crate::identity::{NodeId, NodeKey};
use ed25519_dalek::pkcs8::EncodePrivateKey;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, Error as TlsError, SignatureScheme};
use std::sync::Arc;

/// Name presented in the certificate and used as SNI; never checked.
pub const DUMMY_NAME: &str = "dfin";

pub fn provider() -> Arc<CryptoProvider> {
    static PROVIDER: std::sync::OnceLock<Arc<CryptoProvider>> = std::sync::OnceLock::new();
    PROVIDER
        .get_or_init(|| Arc::new(rustls::crypto::ring::default_provider()))
        .clone()
}

/// A certificate and private key for `key`, both as DER.
pub fn self_signed(key: &NodeKey) -> anyhow::Result<(CertificateDer<'static>, PrivateKeyDer<'static>)> {
    let pkcs8 = key.signing_key().to_pkcs8_der()?;
    let pkcs8_der = PrivatePkcs8KeyDer::from(pkcs8.as_bytes().to_vec());
    let kp = rcgen::KeyPair::from_pkcs8_der_and_sign_algo(&pkcs8_der, &rcgen::PKCS_ED25519)?;
    let mut params = rcgen::CertificateParams::new(vec![DUMMY_NAME.to_string()])?;
    params.distinguished_name = rcgen::DistinguishedName::new();
    let cert = params.self_signed(&kp)?;
    Ok((cert.der().clone(), PrivateKeyDer::Pkcs8(pkcs8_der)))
}

/// The Ed25519 key in a certificate's subject public key info.
pub fn node_id_from_cert(cert: &CertificateDer<'_>) -> Result<NodeId, TlsError> {
    let (_, parsed) = x509_parser::parse_x509_certificate(cert.as_ref())
        .map_err(|e| TlsError::General(format!("certificate does not parse: {e}")))?;
    let spki = parsed.public_key();
    if spki.algorithm.algorithm != x509_parser::oid_registry::OID_SIG_ED25519 {
        return Err(TlsError::General("certificate key is not Ed25519".into()));
    }
    let bytes: [u8; 32] = spki
        .subject_public_key
        .data
        .as_ref()
        .try_into()
        .map_err(|_| TlsError::General("Ed25519 key is not 32 bytes".into()))?;
    let id = NodeId(bytes);
    id.verifying_key()
        .map_err(|_| TlsError::General("certificate key is not a valid Ed25519 point".into()))?;
    Ok(id)
}

/// Dialer side: the server must present exactly the pinned node id.
#[derive(Debug)]
pub struct PinnedServer {
    pub expect: NodeId,
    pub provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for PinnedServer {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let got = node_id_from_cert(end_entity)?;
        if got != self.expect {
            return Err(TlsError::General(format!(
                "node id mismatch: expected {}, peer presented {}",
                self.expect, got
            )));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Err(TlsError::PeerIncompatible(
            rustls::PeerIncompatible::Tls12NotOffered,
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }
}

/// Listener side: any client certificate is fine; its key becomes the peer id.
#[derive(Debug)]
pub struct AnyClient {
    pub provider: Arc<CryptoProvider>,
}

impl ClientCertVerifier for AnyClient {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, TlsError> {
        node_id_from_cert(end_entity)?;
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Err(TlsError::PeerIncompatible(
            rustls::PeerIncompatible::Tls12NotOffered,
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_carries_the_node_id() {
        let key = NodeKey::generate();
        let (cert, _) = self_signed(&key).unwrap();
        assert_eq!(node_id_from_cert(&cert).unwrap(), key.id());
    }
}
