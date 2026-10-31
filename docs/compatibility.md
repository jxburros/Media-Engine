# Compatibility record

The record [engine-spec.md](../engine-spec.md#compatibility-and-integration) requires: which
Engine release speaks which protocol minors, what it was tested against, on which platform, with
which families and extras, and which conformance suite version it passed. One row per release;
rows are never edited after release except to add later test evidence.

## Releases

| Engine | Protocol minors served | Startup record `protocol` | Runtime versions tested         | Platform tested                       | Families           | Extras                         | Conformance suite                |
| ------ | ---------------------- | ------------------------- | ------------------------------- | ------------------------------------- | ------------------ | ------------------------------ | -------------------------------- |
| 0.8.0  | 1.0, 1.1, 1.2, 1.3 (1.3: image-to-video, text-to-audio, optimization options) | `media-worker/1.1` (`highest_protocol` 1.3) | 0.7.0 (15/15 integration tests) and 0.8.0 (18/18) (Go, `go test ./...`, all packages pass; fixture over loopback HTTP and mutual TLS, default and pinned); Runtime 0.9.0 and 0.10.0: see below | Linux x64, CPU only (4-core x86_64, AVX-512), Python 3.12.13 (venv, CPU torch 2.6.0 + torchao 0.9.0: inference, conformance, benchmark tool) / 3.13 (protocol tests) | sdxl-base, esrgan, svd, stable-audio-open (unchanged) | `inference`, `inference-owned`, `inference-audio` (unchanged pins), `inference-optimize` (new: `torchao==0.9.0`) | `media-worker-conformance/1.3` |
| 0.7.0  | 1.0, 1.1, 1.2, 1.3 (1.3: image-to-video and text-to-audio) | `media-worker/1.1` (`highest_protocol` 1.3) | 0.6.0 (11/11 integration tests) and 0.7.0 (15/15) (Go, `go test ./...`, all packages pass; fixture over loopback HTTP, default and 1.1-pinned); no 1.3 consumer yet (Runtime 0.9.0 video, 0.10.0 audio) | Linux x64, CPU only, Python 3.12.13 (venv, CPU torch 2.6.0: inference, conformance) / 3.13 (protocol tests) | sdxl-base, esrgan, svd, stable-audio-open (svd and stable-audio-open on `diffusers-sdxl` only) | `inference`, `inference-owned` (unchanged pins), `inference-audio` (new: `torchsde==0.2.6`) | `media-worker-conformance/1.2` |
| 0.6.0  | 1.0, 1.1, 1.2, 1.3 (1.3: image-to-video only) | `media-worker/1.1` (`highest_protocol` 1.3) | 0.5.0 (8/8 integration tests) and 0.6.0 (11/11) (Go, `go test ./...`, all packages pass; fixture over loopback HTTP, default and 1.1-pinned); no 1.3 consumer yet (Runtime 0.9.0) | Linux x64, CPU only, Python 3.12.3 (venv, CPU torch 2.6.0: inference, conformance) / 3.13 (protocol tests) | sdxl-base, esrgan, svd (svd on `diffusers-sdxl` only) | `inference`, `inference-owned` (unchanged pins; svd needs nothing beyond `inference`) | `media-worker-conformance/1.1` |
| 0.5.0  | 1.0, 1.1, 1.2 (network mode fields within 1.2) | `media-worker/1.1` | 0.4.0 and 0.5.0 (Go, `go test ./...`, all packages pass; the fixture over loopback HTTP, default and 1.1-pinned); no network-mode consumer yet (Runtime 0.7.0) | Linux x64, CPU only, Python 3.13 (protocol and network tests); TLS on loopback only | sdxl-base, esrgan | `inference`, `inference-owned` (unchanged; not re-run, extras not installed) | `media-worker-conformance/1.0`, over loopback HTTP and mutual TLS |
| 0.4.0  | 1.0, 1.1, 1.2          | `media-worker/1.1`        | 0.3.0 (14/14, fixture default and fixture pinned to 1.1); 0.4.0 and 0.5.0 (Go, `go test ./...`, default and 1.1-pinned fixture) | Linux x64, CPU only, Python 3.12.3 (venv) / 3.13 (protocol tests) | sdxl-base, esrgan | `inference`, `inference-owned` | `media-worker-conformance/1.0`   |
| 0.3.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.3.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | — (suite introduced in 0.4.0)    |
| 0.2.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.2.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |
| 0.1.0  | 1.0                    | `media-worker/1.0`        | 0.1.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |

The 0.5.0 row names Runtime 0.7.0 as the network-mode consumer; the Runtime plan schedules
network mode as Runtime 0.8.0 (R4) and image-to-video as 0.9.0 (R5), which the release map in
[implementation-plan.md](implementation-plan.md) now matches.

## Optimizations in 0.8.0

The benchmark index (`src/media_engine/optimizations.json`) ships empty, so no backend advertises
an optimization on any device. Benchmark files:

| File                                              | Backend      | Device class            | Cited by the index | Purpose |
| ------------------------------------------------- | ------------ | ----------------------- | ------------------ | ------- |
| `docs/benchmarks/2026-10-31-cpu-x86_64-tool-validation.json` | `owned-sdxl` | `{kind: cpu, architecture: x86_64}` | no | Validates `tools/benchmark.py` end to end on the tiny random-weight bundle (64×64, 3 steps); says nothing about real SDXL |

## Conformance results for 0.8.0

`media-worker-conformance/1.3` adds the `optimizations` section to the 1.2 sections; a worker
below 1.3 passes it as not applicable, and a worker that advertises nothing passes it by refusing
every candidate.

| Backend                        | Device / transport            | Bundle                                                                  | Result     |
| ------------------------------ | ----------------------------- | ----------------------------------------------------------------------- | ---------- |
| test fixture (1.3 default)     | none, loopback HTTP           | fixture bundles from `serve_fixture.py`; advertises bfloat16, int8-weight-only, inductor (no benchmark: protocol surface only) | conformant |
| test fixture (1.3 default)     | none, mutual TLS on 127.0.0.1 | fixture bundles                                                         | conformant, same section results as loopback HTTP |
| test fixture (pinned to 1.2)   | none, loopback HTTP           | fixture bundles                                                         | conformant (image_to_video, text_to_audio, optimizations not applicable) |
| test fixture (pinned to 1.1)   | none, loopback HTTP           | fixture bundles                                                         | conformant |
| `diffusers-sdxl`               | CPU float32, loopback HTTP    | tiny random-weight SDXL + tiny RRDBNet + tiny svd + tiny stable-audio-open | conformant (optimizations: none advertised, all refused) |
| `owned-sdxl`                   | CPU float32, loopback HTTP    | tiny random-weight SDXL + tiny RRDBNet                                  | conformant (image_to_video, text_to_audio not applicable: not served; optimizations: none advertised) |
| `diffusers-sdxl`, `owned-sdxl` | RTX 4090 (reference)          | SDXL base 1.0 and the other reference bundles                           | not run: no CUDA device or real weights in this workspace |

Run as for 0.7.0, in a Python 3.12.13 virtual environment with the pinned `inference`,
`inference-audio` and `inference-optimize` sets on CPU torch 2.6.0. Every CPU-capable candidate
(bfloat16, int8-weight-only, inductor) was also run through both real backends by
`tests/test_inference.py` with a test-only index naming this CPU.

## Optimization checklist (operator, reference RTX 4090)

Nothing is advertised until this is done and a release cites the result. On the reference host:

1. `uv sync --frozen --extra inference --extra inference-optimize`; confirm
   `capabilities.unadvertised_optimizations` lists all four candidates with `no_benchmark`
   (flash: `device_unsupported` means the build lacks the kernel or the GPU is below sm80).
2. For each backend, `python tools/benchmark.py --backend diffusers-sdxl --device cuda:0
   --catalog … --registration sdxl.json --option precision=bfloat16 --option attention=flash
   --option quantization=int8-weight-only --option compile=inductor --precision float16` at
   1024×1024, 30 steps, `--runs 5`; repeat with `owned-sdxl`. Repeat for the other families
   where a candidate applies (esrgan: bfloat16, inductor; svd: flash, int8, inductor;
   stable-audio-open: all four).
3. Rerun each qualifying option set once more and confirm the output digests are identical to
   the first run (determinism on one device and build); look at the images and listen to the clips
   for each, and write the quality note from the recorded differences and that inspection.
4. Add the accepted entries to `src/media_engine/optimizations.json` with the file's path and
   SHA-256, commit the benchmark files under `docs/benchmarks/`, add a row to the table above
   and run the conformance suite against each backend on `cuda:0`: its `optimizations` section
   then loads and executes every advertised entry.

## Conformance results for 0.7.0

`media-worker-conformance/1.2` adds the `text_to_audio` section (and text-to-audio checks in
discovery, negotiation and validation) to the 1.1 sections; a worker that does not advertise
text-to-audio passes it as not applicable.

| Backend                        | Device / transport            | Bundle                                                                  | Result     |
| ------------------------------ | ----------------------------- | ----------------------------------------------------------------------- | ---------- |
| test fixture (1.3 default)     | none, loopback HTTP           | fixture bundles from `serve_fixture.py` (incl. `video`, `audio`)        | conformant |
| test fixture (1.3 default)     | none, mutual TLS on 127.0.0.1 | fixture bundles                                                         | conformant, same section results as loopback HTTP |
| test fixture (pinned to 1.2)   | none, loopback HTTP           | fixture bundles                                                         | conformant (image_to_video, text_to_audio not applicable) |
| test fixture (pinned to 1.1)   | none, loopback HTTP           | fixture bundles                                                         | conformant |
| `diffusers-sdxl`               | CPU float32, loopback HTTP    | tiny random-weight SDXL + tiny RRDBNet + tiny svd + tiny stable-audio-open (1 kHz stereo) | conformant |
| `owned-sdxl`                   | CPU float32, loopback HTTP    | tiny random-weight SDXL + tiny RRDBNet                                  | conformant (image_to_video, text_to_audio not applicable: not served) |
| `diffusers-sdxl`               | RTX 4090 (reference)          | SDXL base 1.0, Real-ESRGAN, SVD img2vid-xt, Stable Audio Open 1.0        | not run: no CUDA device or real weights in this workspace |

The fixture rows run in CI (`tests/test_conformance_fixture.py`, `tests/test_network.py`). The CPU
rows were run in a Python 3.12.13 virtual environment with the pinned `inference` and
`inference-audio` sets and CPU torch 2.6.0 (`torch==2.6.0+cpu`, not the locked CUDA wheel) with
`python -m tests.conformance.run --registration … --launch …/python -m media_engine.server
--backend … --device cpu --port 0 --max-retained 10 --catalog …`, using the tiny bundles
`tests/test_inference.py` builds. Runtime 0.6.0 and 0.7.0 were run from clean checkouts of their
release commits against this tree's fixture; neither consumes 1.3.

## Text-to-audio checklist (operator, reference RTX 4090)

The stable-audio-open path of 0.7.0 is implemented and exercised on CPU with a tiny random-weight
bundle only. On the reference host:

1. `uv sync --frozen --extra inference --extra inference-audio`; approve a local
   `stable-audio-open-1.0` Diffusers bundle (safetensors only, `tokenizer/tokenizer.json`
   present, no top-level single-file checkpoint); validate it and confirm `sample_rate` 44100,
   `channels` 2, `hop_length` 2048 and `duration_range` [1, 47]; record the estimate.
2. Run the conformance suite against `diffusers-sdxl` with that bundle as `audio` at
   `--device cuda:0`; record the result here.
3. 10 s and 47 s at 100 and 200 steps, float16 and float32: record wall time and peak allocated
   memory against the estimate, and that the WAVs are 10.031 s and 47.044 s (216 and 1013 latent frames) at 44.1 kHz stereo.
4. Listen to one clip per precision; record `measurements.clipped_samples`; cancel once during
   sampling and confirm nothing is published.
5. Compare one 10 s clip's samples with the library pipeline at `audio_end_in_s` 10.031 (same
   seed, steps and guidance) on the same device.

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
