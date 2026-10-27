# Compatibility record

The record [engine-spec.md](../engine-spec.md#compatibility-and-integration) requires: which
Engine release speaks which protocol minors, what it was tested against, on which platform, with
which families and extras, and which conformance suite version it passed. One row per release;
rows are never edited after release except to add later test evidence.

## Releases

| Engine | Protocol minors served | Startup record `protocol` | Runtime versions tested         | Platform tested                       | Families           | Extras                         | Conformance suite                |
| ------ | ---------------------- | ------------------------- | ------------------------------- | ------------------------------------- | ------------------ | ------------------------------ | -------------------------------- |
| 0.4.0  | 1.0, 1.1, 1.2          | `media-worker/1.1`        | 0.3.0 (14/14, fixture default and fixture pinned to 1.1); 0.4.0 and 0.5.0 (Go, `go test ./...`, default and 1.1-pinned fixture) | Linux x64, CPU only, Python 3.12.3 (venv) / 3.13 (protocol tests) | sdxl-base, esrgan | `inference`, `inference-owned` | `media-worker-conformance/1.0`   |
| 0.3.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.3.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | — (suite introduced in 0.4.0)    |
| 0.2.0  | 1.0, 1.1               | `media-worker/1.1`        | 0.2.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |
| 0.1.0  | 1.0                    | `media-worker/1.0`        | 0.1.0                           | Linux x64, CPU only                   | sdxl-base          | `inference`                    | —                                |

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
