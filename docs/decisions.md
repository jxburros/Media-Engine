# Implementation decisions and dependency ledger

Runtime-wide decisions for 0.1.0 through 0.3.0 are recorded in Media Runtime's
`docs/decisions.md`. This file holds the dependency ledger the dependency policy in
[engine-spec.md](../engine-spec.md#dependency-policy) requires and an entry per release from
0.4.0.

## 0.4.0 — protocol 1.2

- **Startup record `protocol` stays `media-worker/1.1`.** Runtime 0.3.0 refuses a startup record
  whose `protocol` it does not know, so reporting 1.2 there would break every 1.1 supervisor that
  launches the worker. The record's shape only gained fields, so `protocol` names the
  launch-record minor; the new `highest_protocol` and `supported_protocols` carry negotiation,
  exactly as `/v1/info` does (which reports 1.2).
- **`count` defaults live in `task_parameters.text-to-image.defaults`, not `defaults`.** 1.1
  supervisors merge `defaults` into every request; putting `count` there would make them send a
  1.2 field. `task_defaults.upscale` is safe because 1.1 supervisors never name that task.
- **The upscaler is engine code on both backends.** There is no Diffusers ESRGAN pipeline worth
  depending on; `upscale.py` owns the RRDBNet with Real-ESRGAN key names so converted releases load
  unchanged, and validation checks the weights header against the configured architecture without
  importing torch. `loaders[].ownership` is `owned` for `esrgan` on every backend.
- **Upscale output size limit is `invalid_request`, field `output_side`.** With today's input
  limits (≤ 2048) and scales (≤ 4) the 8192 limit cannot be exceeded; the check is still made at
  validation and submission so a later larger input limit cannot bypass it, and the fixture can
  lower the limit to exercise it.
- **Fallback records are objects** `{fallback, reason, device, precision}` so a supervisor can
  record what changed without parsing reasons. Unauthorized loads are untouched: no estimate check
  is introduced on the 0.3.0 path.
- **Owned weight loading builds model classes on the meta device and assigns safetensors state
  strictly**; CLIP's non-persistent `position_ids` are recomputed, and any other unmaterialized
  buffer fails the load rather than running with garbage.
- **PNG files from the owned path differ in bytes from Pillow's** (filter 0, one IDAT chunk); the
  pixels are identical. The plan's "byte-identical PNG" acceptance is therefore met for decoded
  pixels, and owned/reference parity is asserted on pixels.
- **`conversion.json` and `source.conversion`.** `reference_bundle_digest` is the canonical digest
  of the reference directory's `{file: sha256}` map (equal to the digest of its registration's
  `configuration.files`), because `convert` receives a catalog handle, not a registration. A 1.2
  registration may carry the provenance as `source.conversion`; the worker checks it against the
  file so a registration cannot claim provenance the directory does not have.
- **Conformance suite** `media-worker-conformance/1.0` lives in `tests/conformance`; a section
  about a minor or task the worker does not advertise passes as not applicable, and any skipped
  section (missing registration or startup record) prevents a "conformant" report.

## Dependency policy

Standard library first. A third-party dependency is accepted only when it does something unsafe
or impractical to own, is pinned by checksum, and is isolated in one module so it can be replaced.
Every dependency is listed below with the backend that needs it and the reason it is not owned
code. A dependency without an entry is a defect. No external service is used at any stage.

## Ledger — Python worker (0.4.0)

| Package       | Extra(s)                    | Imported by                                   | Why it is not owned code                                        | Next change                                      |
| ------------- | --------------------------- | --------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------ |
| torch         | inference, inference-owned  | reference, owned, upscaler, convert           | Tensor execution and CUDA kernels                               | Remains until the native engine                  |
| diffusers     | inference, inference-owned  | reference (pipelines); owned and convert (model classes, scheduler, single-file key mapping) | Reference pipelines; SDXL model classes; checkpoint key mapping | Owned path keeps model classes only              |
| transformers  | inference, inference-owned  | reference, owned, convert                     | CLIP model classes and tokenizers                               | Owned path may replace CLIP tokenization later   |
| safetensors   | inference, inference-owned  | reference, owned (direct weight loading), upscaler, convert | Tensor loading/saving (headers are parsed by owned code)        | Keep                                             |
| numpy         | inference, inference-owned  | diffusers internals (e.g. the Euler scheduler) on every path; no engine module imports it | Required by diffusers itself; pinned so the owned extra resolves identically | Drops out of the pin set with diffusers         |
| accelerate    | inference                   | reference (Diffusers loading helpers)         | Diffusers' low-memory loading                                   | Not installed with inference-owned               |
| Pillow        | inference (transitively also inference-owned via diffusers) | reference (PNG decode/encode); not imported by owned code or the upscaler | Image I/O inside the Diffusers pipelines   | Owned path writes/reads PNG with `png.py`        |
| peft          | inference                   | reference (LoRA through library layers)       | PEFT layer injection                                            | Not installed with inference-owned               |
| setuptools    | build                       | packaging                                     | Packaging                                                       | Keep                                             |

`capabilities.dependencies` reports, per backend, the versions of the packages that backend
imports: the reference backend lists all seven, the owned backend lists torch, diffusers,
transformers and safetensors. The CPU test suite runs the owned backend, the upscaler and the
converter in an environment built from `inference-owned` alone (no accelerate, no peft).

Standard-library-only by design: the HTTP/SSE server (`http.server`), JSON, hashing, PNG reading
and writing (`png.py`: `zlib`, `struct`), the conformance suite (`unittest`, `urllib`, `socket`),
and later TLS in network mode (`ssl`), WAV (`wave`) and ZIP (`zipfile`). Test-only tooling not in
any extra: `jsonschema` was used ad hoc to check the schema against real bodies and is not a
dependency.

Optional and not yet accepted: one pinned MP4 encoder wheel for the `video` output format; the
lossless frame archive is the baseline and this entry is added only on evidence of need.

## Ledger — native engine (planned, phase E6)

| Crate   | Why it is not owned code                                   |
| ------- | ---------------------------------------------------------- |
| candle  | Tensor execution; replaced piecewise by owned kernels later |
| cudarc  | Launching owned CUDA kernels                               |
| rustls  | TLS for network mode                                       |
| png     | Deflate-backed PNG encoding                                |

Owned instead of depended on: the HTTP/1.1 and SSE server, the CLIP BPE tokenizer, WAV and ZIP
writers, validation, bundle verification. A `tokenizers` crate is justified only for the T5
tokenizer of the audio family and is recorded when that family reaches the native engine.
