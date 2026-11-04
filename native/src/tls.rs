//! TLS 1.3 with mandatory client certificates; operator material is loaded once at startup.
use std::fs::File;
use std::io::BufReader;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;

pub fn configure(cert: &Path, key: &Path, ca: &Path) -> Result<(Arc<rustls::ServerConfig>, String), String> {
    fn reader(path: &Path, mode: u32) -> Result<BufReader<File>, String> {
        let meta = std::fs::metadata(path).map_err(|_| "cannot read TLS material")?;
        if !meta.is_file() || meta.permissions().mode() & 0o777 != mode {
            return Err(format!("TLS material requires mode {:04o}", mode));
        }
        File::open(path).map(BufReader::new).map_err(|_| "cannot open TLS material".into())
    }
    let certs = rustls_pemfile::certs(&mut reader(cert, 0o644)?).collect::<Result<Vec<_>, _>>().map_err(|_| "invalid TLS certificates")?;
    let leaf = certs.first().ok_or("empty TLS certificate chain")?;
    let fingerprint = crate::sha256::hex_digest(leaf.as_ref());
    let key = rustls_pemfile::private_key(&mut reader(key, 0o600)?).map_err(|_| "invalid TLS private key")?.ok_or("missing TLS private key")?;
    let mut roots = rustls::RootCertStore::empty();
    for cert in rustls_pemfile::certs(&mut reader(ca, 0o644)?) {
        roots.add(cert.map_err(|_| "invalid client CA")?).map_err(|_| "invalid client CA")?;
    }
    let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots)).build().map_err(|_| "invalid client CA roots")?;
    let config = rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .with_client_cert_verifier(verifier)
        .with_single_cert(certs, key).map_err(|_| "TLS certificate and private key do not match")?;
    Ok((Arc::new(config), fingerprint))
}
