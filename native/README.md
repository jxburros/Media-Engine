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
`candle-core` (tensor execution; CUDA through its `cuda` feature) and `png` (deflate for PNG);
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
`--max-input-mb`, `--bind` (loopback only).

## Not in 1.0.0

- Network mode: there is no TLS in this binary, so a non-loopback `--bind` and the `--tls-*`
  flags are refused at launch. Remote supervisors use the Python worker's network mode.
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
