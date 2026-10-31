# Media Engine action items from the 2026-10-31 cross-repo review

**Date:** 2026-10-31 · **Source:** [cross-repo review](2026-10-31-cross-repo-review.md) (section references below are to that document) · **Agent:** Claude Code (Claude)

This is the to-do list for **Media Engine**: what to build, the decisions only the maintainer can make,
and the inconsistencies the review found. It is a dated working list. Tick items in this file or in
the tracking issue. When an item becomes real work, the change itself updates `docs/protocol.md`,
`schemas/worker.schema.json`, `docs/decisions.md` and `docs/compatibility.md` as usual. Any new
protocol surface needs a protocol minor and conformance-suite coverage.

Priority means: **P0** blocks any real use. **P1** is needed before an application can rely on the stack
as a ComfyUI replacement. **P2** is important. **P3** is later. Items marked **(Runtime)** have a
matching item in
[Media-Runtime's action items](https://github.com/jxburros/Media-Runtime/blob/main/docs/reviews/2026-10-31-action-items.md).
Items marked **(decision)** wait on a decision in section D.

## A. P0: prove it works

- [ ] **A1. Continuous integration.** Add `.github/workflows/ci.yml` running:
  - `PYTHONPATH=src python -m unittest discover -s tests -v` without the inference extra
  - the same with `uv sync --frozen --extra inference --extra inference-audio` on CPU torch, so the
    tiny random-weight bundles run
  - the conformance suite against `tests/serve_fixture.py`
  - `cargo test --workspace --manifest-path native/Cargo.toml`
  - a compile-only `cargo build --features cuda` in a CUDA toolkit container, so the CUDA path
    cannot rot

  Optionally, also run Runtime's `go test ./...` against this commit. (§5.8, §6.3)
- [ ] **A2. GPU checklists.** Run the operator checklists in `docs/compatibility.md` on the reference
  RTX 4090 with real weights. That covers each family on `diffusers-sdxl` and `owned-sdxl`, the
  native-engine checklist (a CUDA build plus conformance), and the optimization checklist. Record
  every row, including failures. (§1, §9.3)
- [ ] **A3. First benchmark.** Run `tools/benchmark.py` on the 4090 and cite the qualifying entries in
  `src/media_engine/optimizations.json`. The index is empty today, so no optimization is
  advertised on any device.
- [ ] **A4. Releases.** None exist yet. Through the shared `JX-Holdings-LLC/release-infrastructure`
  workflows, publish (§6.1):
  - `media-engine-native` archives per platform with contracted names
    (`media-engine-native-v<ver>-<platform>-<arch>[-cuda].tar.gz`), `checksums.txt`, and Onyx's
    ldd/otool check that fails a leg on unexpected shared libraries
  - the Python wheel together with `uv.lock`
  - a **CUDA container image** for the Python backends (non-root, multi-target, as in Onyx's
    `Dockerfile`)

  Runtime's managed install depends on these. **(Runtime E1)**

## B. P1: ComfyUI-capable generation

New families and tasks land on the **reference backend first** (§6.4). Each one needs a family entry
in `capabilities.loaders`, its limits and defaults, a tiny random-weight test bundle, a
fixture-backend equivalent for Runtime's integration tests, and conformance coverage.

- [ ] **B1. Samplers and schedulers.** Advertise a small, explicit set: Euler, Euler ancestral, DPM++ 2M
  (with Karras sigmas), DDIM and UniPC. Today `contracts.py:977` refuses anything but `euler`. The
  owned pipeline and the native engine advertise only the ones they implement. **(Runtime C1)**
- [ ] **B2. Utility image operations.** `resize` (named filter), `crop`, `pad` and `composite` as an
  explicit task in owned, deterministic code, never applied implicitly. **(Runtime C2)**
  **(decision D2)**
- [ ] **B3. Inpainting.** An `inpaint` task taking `image` and `mask` inputs (same size, validated) on
  `sdxl-base`, then on the new families. **(Runtime C3)**
- [ ] **B4. Live previews.** Optional `preview` events every N sampling steps: a low-resolution latent→RGB
  or tiny-VAE decode, exposed as a session-scoped handle. Off by default and advertised in
  capabilities. **(Runtime C4)**
- [ ] **B5. New image families.** One Flux-class DiT family, then SD3.5. Record the reference weights and
  their licenses in `docs/decisions.md` before starting. **(Runtime C5)**
- [ ] **B6. ControlNet.** A load-identity component, like LoRA adapters: `controlnets: [{handle, digest,
  scale}]` on the load body, plus a `control_image` input per execution. Start with SDXL.
  **(Runtime C6)**
- [ ] **B7. MP4 encoder extra.** An optional pinned `inference-video` extra producing `video/mp4`
  (H.264, yuv420p, constant frame rate), advertised in `output_formats` only when installed. It is
  never searched for on `PATH`. **(Runtime C7)** **(decision D1)**
- [ ] **B8. File-level catalog handles.** Let a handle name one file, not only a directory, so a
  ComfyUI `models/checkpoints/` or `models/loras/` folder can be catalogued in place without "hash
  every file in the directory" pulling in neighbours. **(Runtime B4)**

## C. P1–P2: serving and operations

- [ ] **C1 (P1). Prometheus `GET /metrics`.** Execution counts by task and state, phase durations, load
  time, peak allocated memory and busy refusals, served by the Python and native servers alike.
  Gate it on the bearer credential like everything else. **(Runtime B6)** (§6.2)
- [ ] **C2 (P1). Hardening review of both servers.** Onyx-style checks with named constants: refuse
  requests carrying an `Origin` header, reject a non-loopback `Host` on a loopback bind, cap JSON
  nesting depth, and cap body size (beyond the 32 MiB per-input limit). Check `server.py` and
  `native/src/http.rs`. (§6.2)
- [ ] **C3 (P2). Converters in `media-engine convert`.** Add `--kind esrgan-pth` (load with
  `weights_only=True` in an isolated step) and `--kind lora-kohya` → PEFT. Today both are manual
  operator steps in the README.
- [ ] **C4 (P2). More conditioning.** IP-Adapter, textual-inversion embeddings and VAE swap as
  load-identity components, and CLIP skip as an advertised parameter.
- [ ] **C5 (P2). Offload modes.** `offload: model|sequential` as an explicit, advertised load option.
  Never chosen by `auto` unless advertised, and caller-authorized like `authorized_fallbacks`.
- [ ] **C6 (P2). ControlNet preprocessors** (depth, pose, canny) as tasks, only those needed by B6.
- [ ] **C7 (P2). Native engine network mode.** The binary refuses `--tls-*` today. Add it only if remote
  native workers are wanted.
- [ ] **C8 (P3). Text-encoding cache.** Cache text-encoder outputs keyed by tokenizer, encoder digest and
  prompt, within one session. **(Runtime E12)**
- [ ] **C9 (P3). More families.** Further video and audio families after the B items.

## D. Decisions for the maintainer

- [ ] **D1. MP4 encoder extra.** Ship it, optional and pinned, as recommended? This is the open decision
  in `engine-spec.md` "Decisions and next steps".
- [ ] **D2. Utility operations.** Confirm the reading "nothing is resized *silently*", and decide whether
  pixel operations live in Engine (B2) or in Runtime. This is decided jointly with Runtime D2.
- [ ] **D3. Ownership pacing.** Freeze new owned and native stages (native video and audio loops, custom
  kernels) until the B items and one GPU benchmark (A3) exist? (§6.4)
- [ ] **D4. Rename `diffusers-sdxl`.** The reference backend also serves `svd` and `stable-audio-open`, so
  the name misleads. Rename it (for example to `diffusers`) with the old ID kept as an alias for at
  least one protocol minor, or leave it?
- [ ] **D5. First DiT family.** Which Flux-class model goes first, given each candidate's weight license
  and how Diffusers supports it at the pinned version?
- [ ] **D6. Governance.** Adopt Spec-Driven Docs (`AGENTS.md`, `CLAUDE.md`, `development-docs/`), or add
  only an `AGENTS.md` with the hard constraints? (§5.8)
- [ ] **D7. Version and changelog policy.** Bump on every change to main, like JX and Onyx? Does this
  docs-only change need a version bump?
- [ ] **D8. GPU CI runner.** Provide a self-hosted GPU runner for weekly real-weight runs, like JX's
  `real-backend.yml`, or keep GPU validation manual?

## E. Inconsistencies found

- [ ] **E1.** `README.md:15-23` says "This release delivers two of its backend stages" and "Engine 0.8.0
  advertises no optimization yet", but the package is 1.0.0 and the next paragraph adds a third
  stage.
- [ ] **E2.** `README.md:247` says "**0.8.0 ships no benchmark**". That is still true for 1.0.0, but
  it is worded against the older release.
- [ ] **E3.** `engine-spec.md:22` titles the delivery table "Status in Engine 0.7.0", although its rows
  include 0.8.0 and 1.0.0 deliveries.
- [ ] **E4.** `engine-spec.md:45-46` (product sheet) lists only text-to-image and image-to-image as
  delivered, and upscaling, video and audio as planned.
- [ ] **E5.** `engine-spec.md:341` still says "The next implementation step is protocol 1.2 in the Python
  worker…".
- [ ] **E6.** `engine-spec.md` front matter: `status: draft_proposal`, `spec_revision: 0.2.0`.
- [ ] **E7.** `.python-version` pins 3.12.14, but the 1.0.0 changelog's validation ran in a Python
  3.12.13 virtual environment. Record whether that is acceptable, or validate on the pin.
- [ ] **E8.** The backend ID `diffusers-sdxl` serves non-SDXL families. See D4.
