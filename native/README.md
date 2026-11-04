# media-engine-native

The owned native engine of [Media Engine](../README.md): a separately built Rust executable that
implements `media-worker/1.0`–`1.3` directly, with `backend_id: native-sdxl` and
`ownership: engine`. It serves the `sdxl-base` family (text-to-image with `count` 1–4, and
image-to-image), LoRA adapters, `auto` precision and authorized fallbacks, device identity,
retention, resumable SSE and cancellation, with the same bundle registrations, digests and
family rules as the Python backends. The Python package is not needed on a host that runs only
this binary.

## Build

```sh
cargo build --release --locked --manifest-path native/Cargo.toml          # CPU
cargo build --release --locked --manifest-path native/Cargo.toml --features cuda   # CUDA (needs the CUDA toolkit)
```

The binary is `native/target/release/media-engine-native`. Its third-party code is
`candle-core` (tensor execution; CUDA through its `cuda` feature), `png` (deflate for PNG),
and `rustls`/`rustls-pemfile` (TLS 1.3 and operator PEM material);
everything else, the HTTP/1.1 and SSE server, JSON, SHA-256, the CLIP BPE tokenizer,
torch-compatible seeded noise, the SDXL UNet, VAE and CLIP encoders, the Euler scheduler, LoRA
merging and all validation, is code in this directory. See the native ledger in
[../docs/decisions.md](../docs/decisions.md).

## Launch

The launch contract is the Python worker's:

```sh
MEDIA_ENGINE_TOKEN=… media-engine-native --catalog /abs/catalog.json --directory /abs/work \
  --device cuda:0 --port 0
```

`--probe` prints the protocol/version record; the first stdout line after binding is the startup
record (`protocol: media-worker/1.1`, `highest_protocol: media-worker/1.3`, `device_identity`,
transport fields); `MEDIA_ENGINE_SUPERVISED=1` stops the worker when its inherited stdin closes
unless `--no-parent-watch` is given; SIGTERM/SIGINT stop admission and cancel the running
execution. Other flags: `--memory-budget-mb`, `--max-retained`, `--max-storage-mb`,
`--max-input-mb`, `--bind`, `--tls-cert`, `--tls-key`, `--tls-client-ca`.

From 1.1.0, a non-loopback bind requires all three TLS files. TLS uses TLS 1.3 only and
requires a client certificate issued by the configured client CA, plus the bearer token.
The private key must be mode 0600; certificate and CA files must be mode 0644. Material
is read once at launch, and the leaf certificate SHA-256 is included in startup and info.
`--no-parent-watch` is useful under a service manager. Loopback HTTP remains the default.

Authenticated `GET /metrics` reports bounded process-lifetime counters without model,
execution, prompt or credential labels. The same names are served by both Python backends.
JSON bodies are limited to 4 MiB and 64 containers of nesting; PNG upload limits remain
configurable. Browser origins, hostile loopback Host headers, duplicate HTTP headers and
transfer encodings are refused. Catalogs passed with `--catalog` are re-read before
validation/load, so Runtime can atomically add approved paths without restarting the worker.
Catalog LoRA handles may name an individual `.safetensors` file: only that file is hashed and
loaded, regardless of its neighbors. Directory handles remain exact-file-set registrations.
Hugging Face snapshot directories are supported when the root is
`models--…/snapshots/<40 lowercase hex revision>` and every file symlink resolves to a regular
file strictly within that repository's `blobs` directory. Other symlinks, directory links,
cross-repository links and changed bytes are refused. These catalog resolution rules add no
new wire fields; native SDXL continues to negotiate protocol 1.3.

## Not advertised
- Families other than `sdxl-base` (`esrgan`, `svd`, `stable-audio-open`): `unsupported`, field
  `family`, before any file is read; the tasks are not advertised.
- Optimizations: none advertised; `bfloat16` is listed as `no_benchmark`.
- A build with `--features cuda` has not been compiled or run in this workspace.

## Tests

```sh
cargo test --manifest-path native/Cargo.toml
MEDIA_ENGINE_NATIVE=native/target/release/media-engine-native \
  PYTHONPATH=src python -m unittest discover -s tests -p test_native.py -v
```

`tests/test_native.py` runs the conformance suite against the binary and, with the inference
extra installed, compares it with the Python owned pipeline on the tiny random-weight bundle:
pixels within one 8-bit level for the same request, identical event shapes, cancellation, and
LoRA adapters changing the output the same way.

`tests/test_serving.py` checks the HTTP trust boundary, metrics authentication and mutual TLS
against the native binary when `MEDIA_ENGINE_NATIVE` is set.
