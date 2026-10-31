# Implementation decisions and dependency ledger

Runtime-wide decisions for 0.1.0 through 0.3.0 are recorded in Media Runtime's
`docs/decisions.md`. This file holds the dependency ledger the dependency policy in
[engine-spec.md](../engine-spec.md#dependency-policy) requires and an entry per release from
0.4.0.

## 0.7.0 — text-to-audio (protocol 1.3)

- **One new dependency, in its own extra.** The `stable-audio-open` family's scheduler,
  `CosineDPMSolverMultistepScheduler`, draws its noise from a Brownian tree through
  `torchsde`; diffusers registers it only as a stand-in that fails on use without it. So the
  plan's `inference-audio` extra is not empty: it pins `torchsde==0.2.6`, which brings `scipy`
  and `trampoline` (recorded in `uv.lock`; scipy resolves per Python version). No `torchaudio`:
  the Oobleck VAE is a diffusers model class and the WAV is written by the standard library. The
  extra is installed alongside `inference`; without it the task is not advertised and
  `capabilities.missing_task_dependencies` says why (below). `uv.lock` is refreshed for this
  extra only; no existing pin moved.
- **Reference weights:** `stabilityai/stable-audio-open-1.0` (44.1 kHz stereo, up to 47.55 s) in
  Diffusers format with safetensors weights. Stability AI licenses it under the Stability AI
  Community License, separately from this project (the operator checks the model card of the
  revision installed; the repository is gated). Its T5 encoder derives from `google-t5/t5-base`
  (Apache-2.0). Operators supply and approve the weights; nothing here downloads them. No real
  weights were available in this workspace.
- **Reference backend only.** As for `svd`: `DiffusersBackend.families` adds `stable-audio-open`,
  the owned backend refuses it (`unsupported`, field `family`) before reading any file and never
  lists it. An owned audio loop is an optimization-stage item after the reference path is
  validated on hardware.
- **Approved classes as the library writes them.** `model_index.json` names the projection model
  `["stable_audio", "StableAudioProjectionModel"]` (the pipeline module, not `diffusers`), and the
  scheduler check, previously Euler for every Diffusers family, is now the family's
  `scheduler_class`. `tokenizer/tokenizer.json` is required so `T5TokenizerFast` loads without
  sentencepiece, which is not pinned.
- **Facts come from the configs.** Sample rate and channels from the VAE config, the latent
  granularity (`hop_length`) as the product of its `downsampling_ratios`, the longest clip from
  the transformer's `sample_size`; library defaults (the reference bundle's values) where a
  config is silent. The plan's "duration within the bundle's advertised range (reference 1–47)"
  is an optional `audio.json` range, defaulting to 1 s to the whole seconds the latent length
  allows, mirroring `video.json`. They are reported on validation, load and `loaded_bundle`, and
  `limits.duration_seconds` follows the loaded bundle as `video_sizes` does.
- **Duration rounding is up, to whole latent frames, and reported.** The library always samples
  the whole latent length and cuts the decoded waveform at `int(end × sample_rate)`. The worker
  cuts at the smallest whole number of latent frames covering the request, so the clip is never
  shorter than asked and ends on a latent boundary, and conditions the model on that effective
  duration so the conditioning describes the clip returned. Arithmetic is decimal on the number as
  written. `effective_settings` and the output record report the effective duration; a request
  past the range is `invalid_request`, never trimmed. The CPU test checks the samples against the
  library pipeline run with the same effective duration (within one 16-bit step).
- **Parameters.** `seed`, `steps` (1–200, the plan's range) and `duration_seconds` are required;
  `guidance` (0–30) defaults to the library's 7 and is listed in `task_defaults`, not `defaults`,
  so supervisors that merge `defaults` never send it to an image task. Image fields, the
  sampler/scheduler fields and `count` are refused by name rather than ignored; the family has
  one scheduler.
- **WAV format.** Canonical 44-byte RIFF header, PCM 16-bit, interleaved little-endian; samples
  clamped to [-1, 1], scaled by 32767 and rounded to nearest (`measurements.clipped_samples`
  counts what was clamped). The server reads the header back and checks the file holds exactly
  the frames it declares before publishing; the exact file size is reserved at submission.
- **`missing_task_dependencies`.** The spec asks for the missing dependency per task in
  `capabilities.missing_dependencies`, but that list already decides `ready` and whether a load
  is accepted: adding `torchsde` there would make a worker without the audio extra refuse SDXL.
  The per-task names are a separate response addition instead; `missing_dependencies` keeps its
  meaning. `capabilities.dependencies` on the reference backend adds `torchsde`.
- **Memory estimate.** Weights plus the sampling latents and decoded waveform of the whole latent
  length (constant per bundle, because the library samples and decodes all of it whatever the
  duration); attention, T5 and decoder activations excluded and stated.
- **Same minor.** 0.6.0 already advertised `media-worker/1.3`; text-to-audio is added within it
  and gated the same way: a 1.2 caller naming the task or the family gets `unsupported`. The
  schema `$id` stays 1.3 and gains the audio shapes. The fixture pinned to 1.2 still reproduces
  0.5.0 exactly. Conformance suite `media-worker-conformance/1.2` adds the `text_to_audio`
  section; no 1.1 assertion changed.

## 0.6.0 — image-to-video (protocol 1.3)

- **No new dependency.** The `svd` family uses model and pipeline classes already in the pinned
  `diffusers` and `transformers`, Pillow for the reference pipeline's input handling, the
  standard-library `png.py` for frames and `zipfile` for the archive. `pyproject.toml` pins,
  the extras and `uv.lock` are unchanged except the package's own version line. The optional MP4
  encoder extra is not built: the lossless archive is the baseline, and nothing yet shows a
  caller needs a container from the worker.
- **Reference weights:** `stabilityai/stable-video-diffusion-img2vid-xt` (25 frames, 1024×576),
  with `stable-video-diffusion-img2vid` (14 frames) as the smaller alternative, in Diffusers
  format with safetensors weights. They are licensed by Stability AI separately from this project
  (the Stability AI Community License for current revisions; the operator checks the model card
  of the revision installed). Operators supply and approve them; nothing here downloads them.
  No real weights were available in this workspace.
- **Reference backend only.** `DiffusersBackend.families` adds `svd`; the owned backend keeps
  `sdxl-base` and `esrgan`, so a load naming `svd` there is `unsupported`, field `family`, before
  any file is read, and its `loaders` never list the family. An owned video loop is an
  optimization-stage item after the reference path is validated on hardware.
- **Sizes come from the bundle.** The plan's "`video_sizes` advertised from the bundle" is an
  optional top-level `video.json` (`{"sizes": [...]}`, part of the registration's file digests)
  defaulting to the reference 1024×576 and 576×1024. The library's unet config does not encode
  the training size (`sample_size` is 96 on the reference bundle), so a declared file is the only
  way a smaller bundle, such as the test bundles at 64×64, can be served without resizing. The
  worker never resizes or crops the input; a size the bundle does not list is `invalid_request`,
  field `video_size`.
- **`frames` defaults to the bundle's unet `num_frames`.** That default is a bundle fact, so it
  is reported as `loaded_bundle.default_frames` and listed in `task_parameters.<task>.bundle_defaults`
  rather than in `defaults`; `effective_settings.frames` is resolved once the bundle is loaded.
  `capabilities.defaults` is unchanged, so 1.1/1.2 supervisors that merge it never send a video
  field.
- **No guidance parameter.** The reference pipeline ramps guidance from 1.0 (first frame) to 3.0
  (last frame); 1.3 does not expose it, and `effective_settings` reports `min_guidance` and
  `max_guidance` so the value is explicit, not hidden.
- **Stages come from the library pipeline.** `encoding` is the pipeline's image conditioning;
  `sampling` is emitted from a forward pre-hook on the UNet at its first call (the pipeline has no
  hook between conditioning and the loop), and the hook is also a cancellation checkpoint.
  Decoding is the pipeline's own `decode_latents` loop reimplemented so each chunk reports
  progress and checks cancellation; the CPU inference test checks the frames against the
  library's own output within one 8-bit level.
- **Archive format.** Members are stored (PNG is already deflated), `frames.json` comes first,
  names are `000001.png` …, and every member has the DOS-epoch timestamp and mode 0644 so the
  same frames always give the same archive bytes and digest. The server validates the archive
  and reads `frames`, `fps` and the size back from `frames.json` before publishing, as it reads
  image sizes back from PNG headers. `duration_seconds` is `frames / fps`, unrounded.
- **Storage is reserved per frame** (one image-sized slot per frame plus the poster) at
  submission, so a result that cannot fit is `storage_full` before sampling.
- **The memory estimate adds only the sampling latents** that grow with frames
  (`per_frame_bytes`, `latents_mb` at the largest size and `limits.frames`); UNet attention and
  VAE decoder activations are excluded and stated so. The hardware checklist measures the peak.
- **`capabilities.output_formats` keeps its 1.0 value (`["png"]`).** Changing a field's value is
  not an addition; the per-task `output_formats` name the archive's MIME type.
- **The fixture pinned to 1.2 reproduces 0.5.0 exactly**, including limits and task defaults, so
  supervisors can test that a 1.2 worker never offers video. `make_svd_bundle()` writes a few
  seeded random floats per weight role (not a network; the fixture never reads them).
- **Conformance suite `media-worker-conformance/1.1`.** Assertions were added (the
  `image_to_video` section and video checks in discovery, negotiation and validation), so the
  version changed; no 1.0 assertion changed, and a worker without image-to-video sees the 1.0
  results.
- **Release map.** The plan's Runtime-consumer column now follows the Runtime plan: network mode
  (E2) is consumed by Runtime 0.8.0 (R4), image-to-video (E3) by 0.9.0 (R5) and text-to-audio (E4)
  by 0.10.0.

## 0.5.0 — network mode (protocol 1.2)

- **No new dependency.** TLS is the standard-library `ssl` module over the OpenSSL that CPython
  links; nothing is added to `pyproject.toml`, the extras or `uv.lock` (only the package's own
  version line changed). The ledger below is unchanged.
- **The minor stays 1.2.** `scheme`, `bind`, `certificate_sha256` and `ownership_available` are
  response additions, which 1.2 permits; no request, event or error changed. They are reported only
  by a worker that speaks 1.2, like `device_identity`, so the 1.1-pinned fixture still reproduces
  the 0.3.0 surface. The startup record's `protocol` stays `media-worker/1.1`.
- **`--bind` takes an IP address literal, never a host name.** Whether a host name is loopback
  depends on resolution at launch; refusing names keeps "non-loopback needs TLS" unambiguous.
  `0.0.0.0`, `::` and every non-`127.0.0.0/8`, non-`::1` address count as non-loopback.
- **TLS on loopback is accepted without a warning.** The plan calls it allowed but pointless; tests
  and a future local TLS supervisor need it, and refusing would only add a flag. Partial TLS
  material (one or two of the three files) is refused on every address.
- **File modes are checked as permission bits, not exact values.** The key must have no group or
  other bits (0600, 0400 …); certificate and CA must not be group/other writable (0644, 0444 …).
  Ownership is not checked. Encrypted keys are refused instead of prompting on a service launch.
  Python's `load_cert_chain` takes paths, so the certificate and key are opened again by OpenSSL
  immediately after the mode check; both happen once, at launch, and nothing reloads them.
- **The handshake runs in the connection's thread** (`do_handshake_on_connect=False`, then an
  explicit handshake under the existing 15 s socket timeout), so a stalled peer cannot block the
  accept loop. A failed handshake closes the connection without logging.
- **`certificate_sha256` is the lowercase hex SHA-256 of the leaf certificate's DER** (the first
  certificate in `--tls-cert`), `null` without TLS. It is what a supervisor pins and what
  `getpeercert(binary_form=True)` returns.
- **`ownership_available` is `false` exactly when `--no-parent-watch` is given**, even on
  loopback. Without the flag it is `true`: the local launcher's SIGTERM (and, when
  `MEDIA_ENGINE_SUPERVISED=1`, stdin close) still stop the worker; a remote supervisor must not
  read `true` as owning the process itself.
- **No revocation, OCSP or reload.** Rotation and revocation are restarts with new files; the
  README states the operator's responsibilities.
- **Test certificates are committed** under `tests/certs` with their regeneration commands; the CA
  key was discarded, so nothing can be issued from that CA. Git drops file modes, so tests copy
  the key to a 0600 file before launching.
- **The conformance suite stays `media-worker-conformance/1.0`.** Only how it connects changed
  (`MEDIA_ENGINE_TLS_*`, `run.py --tls-*`); no assertion was added or changed, so results over
  TLS and loopback HTTP are directly comparable. Network-field assertions live in
  `tests/test_network.py`.

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

## Ledger — Python worker (0.7.0; `inference-audio` added, other pins unchanged since 0.4.0)

| Package       | Extra(s)                    | Imported by                                   | Why it is not owned code                                        | Next change                                      |
| ------------- | --------------------------- | --------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------ |
| torch         | inference, inference-owned  | reference (incl. svd), owned, upscaler, convert | Tensor execution and CUDA kernels                               | Remains until the native engine                  |
| diffusers     | inference, inference-owned  | reference (pipelines, incl. `StableVideoDiffusionPipeline` and `StableAudioPipeline`); owned and convert (model classes, scheduler, single-file key mapping) | Reference pipelines; SDXL, svd and Stable Audio model classes; checkpoint key mapping | Owned path keeps model classes only              |
| transformers  | inference, inference-owned  | reference, owned, convert                     | CLIP model classes and tokenizers; the svd CLIP image encoder and processor; the T5 encoder and fast tokenizer (through its `tokenizers` dependency) | Owned path may replace CLIP tokenization later   |
| torchsde      | inference-audio             | reference, stable-audio-open only (imported by diffusers' `CosineDPMSolverMultistepScheduler`; no engine module imports it) | The scheduler's Brownian-tree noise sampler; brings `scipy` and `trampoline` | Owned audio loop could own the noise sampler     |
| safetensors   | inference, inference-owned  | reference, owned (direct weight loading), upscaler, convert | Tensor loading/saving (headers are parsed by owned code)        | Keep                                             |
| numpy         | inference, inference-owned  | diffusers internals (e.g. the Euler scheduler) on every path; no engine module imports it | Required by diffusers itself; pinned so the owned extra resolves identically | Drops out of the pin set with diffusers         |
| accelerate    | inference                   | reference (Diffusers loading helpers)         | Diffusers' low-memory loading                                   | Not installed with inference-owned               |
| Pillow        | inference (transitively also inference-owned via diffusers) | reference (PNG decode/encode, svd input image); not imported by owned code, the upscaler or the frame writer | Image I/O inside the Diffusers pipelines   | Owned path writes/reads PNG with `png.py`        |
| peft          | inference                   | reference (LoRA through library layers)       | PEFT layer injection                                            | Not installed with inference-owned               |
| setuptools    | build                       | packaging                                     | Packaging                                                       | Keep                                             |

`capabilities.dependencies` reports, per backend, the versions of the packages that backend
imports: the reference backend lists all seven plus `torchsde` (`null` without the audio extra),
the owned backend lists torch, diffusers, transformers and safetensors. The CPU test suite runs the owned backend, the upscaler and the
converter in an environment built from `inference-owned` alone (no accelerate, no peft).

Standard-library-only by design: the HTTP/SSE server (`http.server`), JSON, hashing, PNG reading
and writing (`png.py`: `zlib`, `struct`), the conformance suite (`unittest`, `urllib`, `socket`),
TLS in network mode (`ssl`, from 0.5.0), frame archives (`video.py`: `zipfile`, `json`, from
0.6.0), and WAV (`audio.py`: `wave`, `struct`, from 0.7.0). Test-only tooling not in
any extra: `jsonschema` was used ad hoc to check the schema against real bodies and is not a
dependency.

Optional and not yet accepted: one pinned MP4 encoder wheel for the `video` output format; the
lossless frame archive is the baseline (shipped in 0.6.0) and this entry is added only on
evidence of need.

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
