# Changelog

## 0.4.0 — 2026-10-27

Protocol `media-worker/1.2` (1.0 and 1.1 still served unchanged; every new behavior is gated on
the request's minor).

- **Conformance suite** `tests/conformance` (`media-worker-conformance/1.0`) runs against any
  worker URL and credential; `python -m tests.conformance.run` reports per-section results and
  never reports "conformant" with a skipped section. CI runs it against the fixture at 1.2 and
  pinned to 1.1; it also passes against both real backends on CPU with tiny bundles.
- **Multiple outputs**: `count` 1–4 on 1.2 text-to-image produces `image, image_2 …` with seeds
  `seed + k − 1`; per-output download at `/v1/executions/{id}/outputs/{output_id}`; outputs are
  stored per execution directory; a missing output fails the execution by name.
- **Families and loaders**: a `FAMILIES` table; `capabilities.loaders`; optional `bundle.format`;
  the `(format, family)` pair is checked before any file is read; `loaded_bundle.tasks`; an
  execution for a task the loaded family does not serve is `unsupported`, field `task`.
- **Upscale** task and `esrgan` family: an engine-owned RRDBNet (`upscale.py`, Real-ESRGAN key
  layout) with explicit overlapping tiles, `upscaling` progress in tiles, cancellation between
  tiles, header-checked weights and an output-size limit decided before any work.
- **`media-engine convert`**: offline single-file SDXL checkpoint conversion into a Diffusers
  bundle with `conversion.json` provenance; unmapped keys fail by name. 1.2 registrations may
  carry the provenance as `source.conversion`, checked against the file.
- **Automatic settings and authorized fallbacks**: `precision: "auto"`, `authorized_fallbacks`
  (`cpu`, `lower_precision`) with `resolved_options`, `resolutions` and `applied_fallbacks`;
  unauthorized loads behave exactly as 0.3.0.
- **Device identity** in `/v1/info` and the startup record (hashed host ID, device, memory, pid).
- **Owned-pipeline dependency trim**: standard-library `png.py`; the owned backend loads
  safetensors straight into the model classes (no accelerate) and never imports Pillow, numpy or
  peft; new `inference-owned` extra; `capabilities.dependencies` is per backend.
- The startup record keeps `protocol: "media-worker/1.1"` (Runtime 0.3.0 refuses unknown values)
  and adds `highest_protocol`; `/v1/info` and `/v1/capabilities` report 1.2.
- `tests/serve_fixture.py` gains 1.2 fixtures (`upscaler`, `converted`, `checkpoint` handles),
  a `--protocol media-worker/1.1` pin and device/fallback/limit controls for supervisor tests.
- Schema `$id` 1.2; new `docs/compatibility.md`; `docs/decisions.md` 0.4.0 entry and ledger.

Runtime 0.3.0's integration suite passes 14/14 against this release (fixture default and pinned
to 1.1). The GPU paths (float16, CUDA fallbacks, device names, x4 upscaling of real weights,
conversion of the real SDXL checkpoint) are implemented but not hardware-validated; see the
checklist in `docs/compatibility.md`.

## Specification revision 0.2.0

Specification revision 0.2.0 builds out every capability revision 0.1.0 deferred: multiple
outputs and `count`, the `upscale` task with the `esrgan` family, loader capabilities and the
offline `convert` subcommand, automatic settings with authorized fallbacks, device identity,
network mode over mutual TLS, image-to-video, text-to-audio, targeted optimization gates and the
owned native engine behind a protocol conformance suite. `docs/implementation-plan.md` orders them
into releases 0.4.0 through 1.0.0 with protocol minors 1.2 and 1.3. No code changed.

## 0.3.0 — 2026-10-26

LoRA adapters as a separately advertised capability (`capabilities.adapters`). A load request may
name up to four registered adapters with scales; they are validated like bundles (catalog handle,
per-file digests, one PEFT-layout safetensors file) and become part of the loaded identity. The
reference backend applies them through the library's PEFT layers; the owned pipeline folds
scale × B·A into the target weights itself. Kohya and legacy layouts are refused with
`unsupported` until converted. Memory estimates include adapter parameters. `peft` is pinned in
the inference extra. The CPU inference test checks that adapters change the output and that both
backends agree.

## 0.2.0 — 2026-10-26

Protocol `media-worker/1.1` with minor negotiation (1.0 requests still served). Added the
`image-to-image` task behind an opaque `POST /v1/inputs` PNG upload, `strength`, effective step
reporting and strict size matching. Added the `owned-sdxl` backend: the project-owned component
assembly, prompt conditioning, latent initialization, Euler/CFG sampling loop, image noising and
VAE lifecycle, selected explicitly with `--backend`. Load validation now returns a safetensors-header
memory estimate with its basis and exclusions. Added a CPU inference test on a tiny random-weight
SDXL-shaped bundle that runs both backends when the inference extra is installed.

## 0.1.0 — 2026-10-26

Implemented the media-worker/1.0 reference worker: strict offline artifact bundles, pinned isolated
Diffusers/PyTorch environment, asynchronous load/unload, warm pipeline reuse, single-flight execution,
idempotent attempts, resumable SSE, cooperative cancellation, bounded retained outputs and safe
errors. Added machine-readable request schemas, operator documentation and CPU-only protocol fixtures.
Real SDXL inference is implemented but not exercised on GPU hardware in this workspace.
