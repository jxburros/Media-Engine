# Changelog

## 0.6.0 — 2026-10-29

Protocol `media-worker/1.3` with image-to-video (phase E3). 1.0, 1.1 and 1.2 are still served
unchanged; every new behavior is gated on the request's minor, and a 1.2 caller that names
`image-to-video` or an `svd` bundle gets `unsupported` (field `task` or `family`).

- **`svd` family and `image-to-video` task**: a Diffusers `StableVideoDiffusionPipeline` bundle
  (unet, vae, image_encoder → safetensors; feature_extractor, scheduler → json) with the
  approved classes checked from `model_index.json`; sizes from an optional `video.json`
  (default 1024×576 and 576×1024) and the default frame count from the unet's `num_frames`,
  both reported as `video_sizes`/`default_frames` on validation, load and `loaded_bundle`, and
  `capabilities.limits.video_sizes` follows the loaded bundle.
- **Request**: `inputs.image` only; `seed`, `steps` (1–50) required; `frames` (1–64), `fps`
  (1–30, default 7), `motion` (1–255, default 127), `noise_augmentation` (0–1, default 0.02) and
  `decode_chunk` (1–16, default 8) optional; width/height must be an advertised size and equal the
  input (never resized). The fixed guidance ramp is reported as `min_guidance`/`max_guidance`.
- **Outputs**: `frames`, an `application/zip` of stored lossless PNGs `000001.png` … plus
  `frames.json` (`fps`, `width`, `height`, `frames`), reproducible byte for byte; and `poster`,
  frame 1's PNG. The worker validates the archive and reads `frames`, `fps`, `duration_seconds`,
  `width` and `height` back from it into the output record. Storage is reserved per frame before
  sampling. New standard-library module `video.py` (writer and reader).
- **Reference backend only**: `diffusers-sdxl` runs the library pipeline from the verified
  components with stages `encoding`, `sampling` (step progress), `decoding` (per chunk, `unit:
  "frames"`), `packaging`, `saving`, and checks cancellation at every UNet call and step, before
  every decode chunk and before packaging. `owned-sdxl` does not serve `svd` (`unsupported`, field
  `family`); `capabilities.loaders` lists `svd` only for `diffusers-sdxl`.
- **Memory estimate**: `svd` adds `latents_mb` and `per_frame_bytes` (sampling latents of 64
  frames at the largest size) with its basis; decoder and attention activations are excluded and
  stated.
- **Conformance suite `media-worker-conformance/1.1`**: an `image_to_video` section plus video
  checks in discovery, negotiation and validation; the registration document gains `video`.
- `tests/serve_fixture.py` serves a `video` handle (svd, two frames, sizes 64×64, 1024×576,
  576×1024; tiny random stand-in weights) through `TestBackend`, which writes a real frame archive
  through `video.py`; `--protocol media-worker/1.2` now reproduces the 0.5.0 fixture surface and
  `--protocol media-worker/1.1` still reproduces 0.3.0.
- Schema `$id` 1.3 (`svd` bundles, the video execution body, `frame_manifest`,
  `video_configuration`; 1.0–1.2 shapes still valid); `docs/protocol.md` "What changed in 1.3";
  README "Image-to-video" with the reference bundle and its license; compatibility row and video
  checklist; `docs/decisions.md` 0.6.0 entry. The plan's Runtime-consumer column now matches the
  Runtime plan (network mode 0.8.0, video 0.9.0, audio 0.10.0). No new dependency; the optional
  MP4 encoder extra is not part of this release.

Validation, stated plainly: there is no GPU and there are no real Stable Video Diffusion weights
in this workspace, so the svd path ran only on CPU float32 with a tiny random-weight bundle
(64×64, two frames, two steps) in a Python 3.12.3 virtual environment holding the pinned
`inference` set with CPU torch 2.6.0 (`+cpu` wheel, not the locked CUDA build). There the frames
match the library pipeline's own decode within one 8-bit level and the whole suite passes
(64 tests, one IPv6 skip); the conformance suite reports `diffusers-sdxl` and `owned-sdxl`
conformant on CPU with tiny bundles. float16, CUDA memory, the reference 1024×576 × 25-frame
workload and the estimate's accuracy are implemented but not hardware-validated; see the
image-to-video checklist in `docs/compatibility.md`. The `inference-owned`-only environment was
not rebuilt for this release (no owned-path code changed). Runtime 0.5.0 (8/8 integration tests)
and 0.6.0 (11/11) pass unchanged against the 0.6.0 fixture; no Runtime consumes 1.3 yet
(Runtime 0.9.0, phase R5).

## 0.5.0 — 2026-10-27

Network mode over mutual TLS (phase E2). Protocol minor stays `media-worker/1.2`: no request,
event or error changed, and the new fields are response additions 1.2 permits.

- **Launch flags** `--bind` (IP address, default `127.0.0.1`), `--tls-cert`, `--tls-key`,
  `--tls-client-ca` and `--no-parent-watch`. A non-loopback bind without all three TLS files is
  refused at launch, partial TLS material is refused everywhere, and `create_server` itself
  refuses to serve plain HTTP off loopback. TLS on loopback is allowed.
- **Mutual TLS** with the standard-library `ssl` module: TLS 1.3 minimum, client certificate
  required and verified against the operator's CA, handshake in the connection's thread with the
  existing 15 s timeout. Plain HTTP, certificate-less, untrusted and TLS 1.2 clients get no HTTP
  response. The bearer credential is still required on everything but `/health`.
- **Launch file checks**: the key must not be accessible by group or others (0600), certificate
  and CA must not be group/other writable (0644); encrypted keys and mismatched pairs are refused
  with a message naming the flag. Files are read once at start and never reloaded.
- **Transport fields** in `/v1/info` and the startup record: `scheme`, `bind`,
  `certificate_sha256` (SHA-256 of the server certificate DER, `null` without TLS) and
  `ownership_available` (`false` with `--no-parent-watch`). The inherited-stdin watch is skipped
  with `--no-parent-watch`; SIGTERM handling is unchanged. A 1.1-pinned fixture omits the fields.
- `tests/serve_fixture.py` takes the same flags (and `MEDIA_ENGINE_FIXTURE_BIND`, `_TLS_CERT`,
  `_TLS_KEY`, `_TLS_CLIENT_CA`, `_NO_PARENT_WATCH`) so supervisors can run integration tests in
  network mode; its defaults and existing options are unchanged.
- `tests/certs`: committed test CA, server (SAN `127.0.0.1`, `localhost`), client and untrusted
  client certificates (EC P-256, 10 years) with the regeneration commands; never generated at
  test time. They ship in the sdist.
- The conformance suite (still `media-worker-conformance/1.0`; no assertion changed) connects over
  `https` with `MEDIA_ENGINE_TLS_CA`/`_CERT`/`_KEY` or `run.py --tls-ca/--tls-cert/--tls-key`.
  `tests/test_network.py` runs it over mutual TLS and asserts the same section results as loopback
  HTTP, plus the refusals, file modes, transport fields and parent watch.
- Schema `$defs.transport`; README "Network mode"; `docs/protocol.md` Transport section;
  compatibility row; `docs/decisions.md` 0.5.0 entry. No new dependency.

Validation, stated plainly: TLS was exercised on loopback only (`127.0.0.1`, and `0.0.0.0`
reached through loopback) on one Linux x64 host; no traffic crossed a real network, and no
remote supervisor exists yet (Runtime 0.7.0 is the consumer). Runtime 0.4.0 and 0.5.0 (Go)
integration suites pass unchanged against this release over loopback HTTP. The inference extras
are not installed in this workspace, so the CPU inference, conversion and real-backend
conformance runs were not repeated for 0.5.0 (no inference code changed). There is no GPU here;
the 0.4.0 GPU checklist is still open and network mode adds its own checklist in
`docs/compatibility.md`.

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
