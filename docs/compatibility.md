# Compatibility record

The record [engine-spec.md](../engine-spec.md#compatibility-and-integration) requires: which
Engine release speaks which protocol minors, what it was tested against, on which platform, with
which families and extras, and which conformance suite version it passed. One row per release;
rows are never edited after release except to add later test evidence.

## Releases

| Engine | Protocol minors served | Startup record `protocol` | Runtime versions tested         | Platform tested                       | Families           | Extras                         | Conformance suite                |
| ------ | ---------------------- | ------------------------- | ------------------------------- | ------------------------------------- | ------------------ | ------------------------------ | -------------------------------- |
| 0.6.0  | 1.0, 1.1, 1.2, 1.3 (1.3: image-to-video only) | `media-worker/1.1` (`highest_protocol` 1.3) | 0.5.0 (8/8 integration tests) and 0.6.0 (11/11) (Go, `go test ./...`, all packages pass; fixture over loopback HTTP, default and 1.1-pinned); no 1.3 consumer yet (Runtime 0.9.0) | Linux x64, CPU only, Python 3.12.3 (venv, CPU torch 2.6.0: inference, conformance) / 3.13 (protocol tests) | sdxl-base, esrgan, svd (svd on `diffusers-sdxl` only) | `inference`, `inference-owned` (unchanged pins; svd needs nothing beyond `inference`) | `media-worker-conformance/1.1` |
| 0.5.0  | 1.0, 1.1, 1.2 (network mode fields within 1.2) | `media-worker/1.1` | 0.4.0 and 0.5.0 (Go, `go test ./...`, all packages pass; the fixture over loopback HTTP, default and 1.1-pinned); no network-mode consumer yet (Runtime 0.7.0) | Linux x64, CPU only, Python 3.13 (protocol and network tests); TLS on loopback only | sdxl-base, esrgan | `inference`, `inference-owned` (unchanged; not re-run, extras not installed) | `media-worker-conformance/1.0`, over loopback HTTP and mutual TLS |
| 0.4.0  | 1.0, 1.1, 1.2          | `media-worker/1.1`        | 0.3.0 (14/14, fixture default and fixture pinned to 1.1); 0.4.0 and 0.5.0 (Go, `go test ./...`, default and 1.1-pinned fixture) | Linux x64, CPU only, Python 3.12.3 (venv) / 3.13 (protocol tests) | sdxl-base, esrgan | `inference`, `inference-owned` | `media-worker-conformance/1.0`   |
| 0.3.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.3.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | — (suite introduced in 0.4.0)    |
| 0.2.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.2.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |
| 0.1.0  | 1.0                    | `media-worker/1.0`        | 0.1.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |

The 0.5.0 row names Runtime 0.7.0 as the network-mode consumer; the Runtime plan schedules
network mode as Runtime 0.8.0 (R4) and image-to-video as 0.9.0 (R5), which the release map in
[implementation-plan.md](implementation-plan.md) now matches.

## Conformance results for 0.6.0

`media-worker-conformance/1.1` adds the `image_to_video` section (and image-to-video checks in
discovery, negotiation and validation) to the 1.0 sections; a worker that does not advertise
image-to-video passes it as not applicable.

| Backend                        | Device / transport           | Bundle                                                    | Result     |
| ------------------------------ | ---------------------------- | --------------------------------------------------------- | ---------- |
| test fixture (1.3 default)     | none, loopback HTTP          | fixture bundles from `serve_fixture.py` (incl. `video`)   | conformant |
| test fixture (1.3 default)     | none, mutual TLS on 127.0.0.1 | fixture bundles                                          | conformant, same section results as loopback HTTP |
| test fixture (pinned to 1.2)   | none, loopback HTTP          | fixture bundles                                           | conformant (image_to_video not applicable) |
| test fixture (pinned to 1.1)   | none, loopback HTTP          | fixture bundles                                           | conformant |
| `diffusers-sdxl`               | CPU float32, loopback HTTP   | tiny random-weight SDXL + tiny RRDBNet + tiny svd (64×64) | conformant |
| `owned-sdxl`                   | CPU float32, loopback HTTP   | tiny random-weight SDXL + tiny RRDBNet                    | conformant (image_to_video not applicable: svd not served) |
| `diffusers-sdxl`               | RTX 4090 (reference)         | SDXL base 1.0, Real-ESRGAN, SVD img2vid-xt                | not run: no CUDA device or real weights in this workspace |

The fixture rows run in CI (`tests/test_conformance_fixture.py`, `tests/test_network.py`). The CPU
rows were run in a Python 3.12 virtual environment with the pinned `inference` set and CPU torch
2.6.0 (`torch==2.6.0+cpu`, not the locked CUDA wheel) with `python -m tests.conformance.run
--registration … --launch …/python -m media_engine.server --backend … --device cpu --port 0
--max-retained 10 --catalog …`, using the tiny bundles `tests/test_inference.py` builds.

## Image-to-video checklist (operator, reference RTX 4090)

The svd path of 0.6.0 is implemented and exercised on CPU with a tiny random-weight bundle only.
On the reference host:

1. Approve a local `stable-video-diffusion-img2vid-xt` Diffusers bundle (safetensors only, one
   weights file per component folder); validate it and record the estimate's `latents_mb`.
2. Run the conformance suite against `diffusers-sdxl` with that bundle as `video` at
   `--device cuda:0`; record the result here.
3. 1024×576, 25 frames, 25 steps, float16, `decode_chunk` 8, 4 and 1: record wall time, peak
   allocated memory against the estimate and that the archive holds 25 frames at 1024×576.
4. 576×1024 once; `frames: 64` once (or record the OOM and the restart it requires); cancel
   once during sampling and once during decoding and confirm nothing is published.
5. Encode the archive with an external tool at `fps` and compare the poster with frame 1.

## Conformance results for 0.5.0

| Backend                      | Transport                                   | Result     |
| ---------------------------- | ------------------------------------------- | ---------- |
| test fixture (1.2 default)   | loopback HTTP                               | conformant |
| test fixture (1.2 default)   | mutual TLS 1.3 on 127.0.0.1 (`tests/certs`) | conformant, same section results as loopback HTTP |
| test fixture (pinned to 1.1) | loopback HTTP                               | conformant |
| `diffusers-sdxl`, `owned-sdxl` | CPU float32, loopback                     | not re-run: inference extras not installed here; no inference code changed since 0.4.0 |
| any backend                  | mutual TLS across hosts                     | not run: single host, no real network |

The fixture rows run in CI (`tests/test_conformance_fixture.py`, `tests/test_network.py`). The
mutual-TLS row also passes through the runner:
`python -m tests.conformance.run --registration ROOT/conformance.json --tls-ca tests/certs/ca.pem
--tls-cert tests/certs/client.pem --tls-key tests/certs/client.key --launch python
tests/serve_fixture.py ROOT --tls-cert … --tls-key <0600 copy> --tls-client-ca …`.

## Network-mode checklist (operator, two hosts)

Network mode in 0.5.0 is tested on loopback only. On a worker host and a separate supervisor host:

1. Issue operator certificates (README "Network mode"); launch with `--bind <LAN address>` (and
   once with `0.0.0.0`), the three TLS files and `--no-parent-watch` under a service manager.
2. From the supervisor host run the conformance suite with `--url https://<host>:<port>` and the
   client certificate; record the result here.
3. From the supervisor host confirm plain HTTP, no client certificate and a certificate from
   another CA all fail before any HTTP response, and a wrong bearer token gets `401`.
4. Compare `certificate_sha256` with `openssl x509 -outform DER | sha256sum` of the served
   certificate; drop the supervisor's connection mid-execution and confirm the worker keeps
   running with `ownership_available: false`.
5. Run the 0.4.0 hardware checklist below over the TLS connection on the reference RTX 4090.

## Conformance results for 0.4.0

`media-worker-conformance/1.0` (sections: discovery, negotiation, validation, lifecycle,
single_flight, idempotency, events, cancellation, outputs, multi_output, upscale, inputs,
retention, privacy). "Conformant" means every section passed and none was skipped.

| Backend                      | Device                     | Bundle                                   | Result     |
| ---------------------------- | -------------------------- | ---------------------------------------- | ---------- |
| test fixture (1.2 default)   | none                       | fixture bundles from `serve_fixture.py`  | conformant |
| test fixture (pinned to 1.1) | none                       | fixture bundles                          | conformant |
| `diffusers-sdxl`             | CPU float32                | tiny random-weight SDXL + tiny RRDBNet   | conformant |
| `owned-sdxl`                 | CPU float32                | tiny random-weight SDXL + tiny RRDBNet   | conformant |
| `diffusers-sdxl`, `owned-sdxl` | RTX 4090 (reference)     | SDXL base 1.0, Real-ESRGAN x2/x4         | not run: no CUDA device or real weights in this workspace |

The two fixture rows run in CI (`tests/test_conformance_fixture.py`). The CPU rows were run with
`python -m tests.conformance.run --registration … --launch .venv/bin/python -m media_engine.server
--backend … --device cpu --port 0 --max-retained 10`.

## Hardware checklist (operator, reference RTX 4090)

The GPU path of 0.4.0 is implemented but not hardware-validated. On the reference host:

1. `uv sync --frozen --extra inference`; `media-engine --probe`.
2. Run the conformance suite against both backends with an SDXL base bundle and a converted
   Real-ESRGAN bundle at `--device cuda:0`; record the result here.
3. `count: 4` at 1024×1024 float16: record peak allocated memory against the estimate.
4. Upscale a 1024×1024 PNG with `tile: 512` at x4 (4096×4096): record time, peak memory and
   compare with `tile: 0`.
5. `precision: "auto"` with and without `--memory-budget-mb`; `authorized_fallbacks: ["cpu"]`
   with the device hidden (`CUDA_VISIBLE_DEVICES=`) must load on CPU and report it.
6. `media-engine convert` on `sd_xl_base_1.0.safetensors` against the Diffusers reference bundle:
   `unmapped_keys` empty; the converted bundle's text-to-image output matches the reference
   bundle's within one 8-bit level.
