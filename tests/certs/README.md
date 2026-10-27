# Test certificates for network mode

Fixture material for `tests/test_network.py` and for supervisors' network-mode integration tests
(Media Runtime launches `tests/serve_fixture.py` with these files). They are **test-only**: the
CA private key was discarded after signing, nothing outside tests trusts this CA, and the private
keys below are public by being committed. Never use them for a real worker.

| File                                 | What it is                                                                 |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `ca.pem`                             | Test CA (`CN=Media Engine test CA`); pass as `--tls-client-ca`, and as the client's trust root |
| `server.pem`, `server.key`           | Server certificate signed by the CA; SAN `IP:127.0.0.1`, `DNS:localhost`; `serverAuth` |
| `client.pem`, `client.key`           | Client certificate signed by the CA; `clientAuth`                          |
| `untrusted.pem`, `untrusted.key`     | Self-signed client certificate the CA did not issue (must be refused)      |
| `extensions.cnf`                     | X.509 extensions used by the commands below                                |

All keys are EC P-256; every certificate is valid for 10 years (3650 days, from 2026-10-27).
They are generated once and committed; tests never generate certificates.

Git does not record file modes, so a checkout has the keys at your umask (usually 0644). The
worker refuses a `--tls-key` readable by group or others, so a test (or you) copies the key to a
private file with mode 0600 before launching; certificates must not be group/other writable.

## Regenerating

From this directory, with OpenSSL 3:

```sh
set -e
key() { openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$1.key"; }
key ca
openssl req -new -x509 -key ca.key -subj "/CN=Media Engine test CA" -days 3650 -sha256 \
  -extensions ca -config extensions.cnf -out ca.pem
for name in server client; do
  key $name
  openssl req -new -key $name.key -subj "/CN=media-engine-test-$name" -out $name.csr
  openssl x509 -req -in $name.csr -CA ca.pem -CAkey ca.key -set_serial 0x$(openssl rand -hex 16) \
    -days 3650 -sha256 -extfile extensions.cnf -extensions $name -out $name.pem
  rm $name.csr
done
key untrusted
openssl req -new -x509 -key untrusted.key -subj "/CN=media-engine-test-untrusted" -days 3650 \
  -sha256 -extensions client -config extensions.cnf -out untrusted.pem
rm ca.key
openssl verify -CAfile ca.pem server.pem client.pem
```

Regenerating changes `certificate_sha256`; nothing in the tests pins a literal fingerprint (they
compare against the SHA-256 of `server.pem`'s DER).
