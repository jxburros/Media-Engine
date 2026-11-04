# Release and hardware evidence

`release.json` is authoritative. Version changes update the Python package, worker identity,
Cargo packages and lockfiles together. Every main change advances the release version and
has a real `CHANGELOG.md` section. The shared toolkit is pinned to reviewed commit
`d7f42f9910c075ef34915fa2b95cf28dd2be534a` in both workflow references and checkout inputs.

After CI passes and the PR is merged, manually run **Release** on `main`, first with
`dry_run: true`. This builds and verifies all required Linux x64 assets together: native CPU
and RTX 4090 CUDA archives, Python wheel and `uv.lock`, and a non-root CUDA Python image
archive. A missing or failed leg prevents publication. The image archive can be loaded with
`gunzip -c media-engine-python-vVERSION-linux-x64-cuda-image.tar.gz | docker load`.
Native CUDA binaries need the NVIDIA driver and compatible CUDA 12.6 runtime libraries.

Inspect the dry-run artifacts, then manually dispatch with dry-run disabled to publish one
complete GitHub release and `checksums.txt`. The workflow does not publish from tags or PRs,
replace releases, or push images to a registry. Configure real reviewer/branch restrictions
for the `release` environment; its name alone does not establish protection. The standard
profile verifies SHA-256 and GitHub upload metadata; it is not a signed updater protocol.
No release was published by implementing these workflows.

## GPU jobs

Provision a trusted runner with labels `self-hosted, linux, x64, media-gpu`, an RTX 4090,
CUDA toolkit 12.6, the pinned Python extras (`inference`, `inference-audio`,
`inference-optimize`, `inference-video`) and approved local model weights. Restrict the
`gpu-validation` environment to `main` and authorized operators. The job is manual-only,
checks `github.ref == refs/heads/main`, and never checks out a PR revision.

Set repository/environment variables `MEDIA_GPU_PYTHON` to that environment's Python and
`MEDIA_GPU_CONFIG` to a runner-local copy of `packaging/gpu-validation.example.json`.
The catalog maps opaque handles to existing approved bundles. `conformance.json` is the
conformance runner's `{bundle, upscaler, video, audio, options?}` registration; each bundle
contains its verified file digests. `sdxl-bundle.json` is its bare SDXL `bundle` object for
`tools/benchmark.py`. No command downloads weights. An operator must review the model licenses.

Dispatch **GPU conformance and benchmarks**. It builds the native CUDA worker, runs conformance
against all three backends, records failures, and uploads `gpu-results`. It also compares
explicit `none`, `model` and `sequential` offload on the reference backend for SDXL 1024×1024,
SVD 1024×576, and 10/47-second audio, retaining measured peak allocated/reserved GPU memory,
process RSS, timing, output digests and baseline differences. Missing bundles and failed modes
are recorded as failures; no estimate is presented as a measurement. Enabling `benchmark`
runs the reference and owned Python backends twice with identical settings and all four
optimization candidates. It records hardware/software identity, wall time, memory, output
digests and quality differences. The job never edits `optimizations.json` or claims a
candidate qualifies without review. After examining evidence, cite the exact artifact hashes
and workflow URL in `docs/compatibility.md`; inspect generated media for quality before
approving any optimization. The job definitions are prepared; no GPU result is implied.

## Build inputs

Build-only tools are Python 3.12.14, Rust 1.99.1, `build==1.2.2.post1` (PEP 517 wheel
frontend), and `uv==0.9.5` (frozen lockfile export with hashes for the image). Native CUDA
compilation uses `nvidia/cuda:12.6.3-devel-ubuntu24.04` with compute capability 89; the Python
image uses `python:3.12.14-slim-bookworm` plus the CUDA libraries pinned by `uv.lock`.
These tools are isolated to packaging; serving imports no build frontend. Native archives
include the Cargo lockfile and collected dependency copyright/license notices. CPU native
release binaries are built on Ubuntu 24.04; use a compatible glibc host. Go Runtime's binary
is independently static.
