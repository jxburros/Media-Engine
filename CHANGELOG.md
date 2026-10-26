# Changelog

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
