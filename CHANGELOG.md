# Changelog

## 0.1.0 — 2026-10-26

Implemented the media-worker/1.0 reference worker: strict offline artifact bundles, pinned isolated
Diffusers/PyTorch environment, asynchronous load/unload, warm pipeline reuse, single-flight execution,
idempotent attempts, resumable SSE, cooperative cancellation, bounded retained outputs and safe
errors. Added machine-readable request schemas, operator documentation and CPU-only protocol fixtures.
Real SDXL inference is implemented but not exercised on GPU hardware in this workspace.
