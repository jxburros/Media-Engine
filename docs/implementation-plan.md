# Media Engine implementation plan — deferred capabilities

This plan turns [engine-spec.md](../engine-spec.md) revision 0.2.0 into ordered, releasable
increments of the Python worker. It covers every capability the specification marks "Specified
below": multiple outputs, upscaling, loader capabilities and single-file conversion, automatic
settings and authorized fallbacks, device identity, network mode, image-to-video, text-to-audio,
targeted optimization and the owned native engine, plus the protocol conformance suite that gates
backends. Out-of-scope items (graph import, executable recipes, plugin marketplaces, in-process
multi-device, model downloading) have no phase here.

Each phase names its Engine release, protocol minor, the companion Runtime release that consumes
it, the files it touches, the tests that prove it and the acceptance rule that closes it. Phases
are ordered by dependency and by value: protocol 1.2 unlocks the most Runtime work, network mode
unlocks distributed execution, and the media families follow once multi-output and the conformance
suite exist. Dates are deliberately absent; versions are the schedule.

## Baseline (Engine 0.3.0)

| Area                    | Where                                                        | State                                                            |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| Protocol constants      | `src/media_engine/contracts.py` (`PROTOCOL`, `TASKS`, `ROLES`, `LIMITS`, `APPROVED_CLASSES`) | `media-worker/1.1`, two tasks, one family |
| Bundle/adapter identity | `contracts.py` (`bundle_path`, `adapter_path`, `lora_layout`, `memory_estimate`) | SDXL roles only                                       |
| HTTP lifecycle          | `src/media_engine/server.py` (`Worker`, `create_server`, `main`) | Loopback HTTP, bearer token, one `image` output, `/outputs/image` route |
| Backends                | `src/media_engine/backend.py` (`TorchBackend`, `DiffusersBackend`), `src/media_engine/pipeline.py` (`OwnedPipelineBackend`) | `generate()` returns one PNG at `destination` |
| Schemas                 | `schemas/worker.schema.json` (`bundle`, `execution`, `load`, `unload`, `input_record`, `memory_estimate`, `adapter`, `validate_adapters`) | 1.1 shapes |
| Tests                   | `tests/test_worker.py` (protocol, CPU-only), `tests/test_inference.py` (real CPU inference on a tiny bundle), `tests/fixture.py` (`TestBackend`, `make_bundle`, `make_adapter`, `make_png`), `tests/serve_fixture.py` (used by Runtime's integration suite) | Both backends covered |
| Contract document       | `docs/protocol.md`                                           | Describes 1.1                                                    |

Conventions every phase keeps: unknown fields are errors (`strict()`), every new option is
advertised before it is accepted, errors never carry prompts/paths/exceptions, `/health` stays
coarse, new behavior is gated on the request's `protocol` minor, and the fixture backend in
`tests/` must implement every new backend method so Runtime's cross-process tests keep running
without GPU hardware.

## Release map

| Engine | Protocol | Phase                                                   | Runtime consumer |
| ------ | -------- | ------------------------------------------------------- | ---------------- |
| 0.4.0  | 1.2      | E1 Conformance suite, multi-output, loaders, upscale, conversion, auto/fallbacks, device identity | 0.4.0, 0.5.0, 0.6.0 |
| 0.5.0  | 1.2      | E2 Network mode (mutual TLS)                            | 0.7.0            |
| 0.6.0  | 1.3      | E3 Image-to-video (`svd`)                               | 0.8.0            |
| 0.7.0  | 1.3      | E4 Text-to-audio (`stable-audio-open`)                  | 0.9.0            |
| 0.8.0  | 1.3      | E5 Targeted optimization options                        | 0.9.x            |
| 1.0.0  | 1.3      | E6 Owned native engine                                  | 1.0.0            |

Minor 1.2 ships complete in 0.4.0; 0.5.0 adds fields only (`scheme`, `bind`, certificate
fingerprint, `ownership_available`) which 1.2 permits as response additions. Minor 1.3 ships in
0.6.0 and 0.7.0 together: 0.6.0 advertises 1.3 with video only and 0.7.0 adds audio; the
`tasks` list, not the minor, tells a caller what exists.

## Phase E1 — Engine 0.4.0, protocol 1.2

### E1.1 Protocol conformance suite

Build the suite first so every later change, and every later backend, is measured by it.

- Add `tests/conformance/__init__.py` and `tests/conformance/suite.py`: a `unittest` suite
  parameterized by `MEDIA_ENGINE_URL`, `MEDIA_ENGINE_TOKEN` and a bundle registration document
  path. Sections: discovery (`/health`, `/v1/info`, `/v1/capabilities`, startup record shape when
  launched by the suite), negotiation of each listed minor, `/v1/validate` for every advertised
  task, load/unload operations and their terminal records, single-flight busy `409`, idempotent
  and conflicting attempts, SSE replay with `Last-Event-ID` and `resume_required`, cancellation at
  encoding/sampling/decoding checkpoints, output finalization and SHA-256 match, retention limits
  (`storage_full`, `max_retained`), input lifecycle, and error privacy (no path, prompt or
  exception text in any error body).
- Add `tests/conformance/run.py` as a CLI (`python -m tests.conformance.run --url … --bundle …`)
  that reports per-section pass/fail and refuses to report "conformant" with any skipped section.
- Add `tests/test_conformance_fixture.py` that launches `tests/serve_fixture.py` and runs the
  suite against it, so the suite itself is tested in CI without hardware.
- Document in `docs/protocol.md` that a backend is advertised only when the suite passes; record
  the suite version in the compatibility record (E1.8).

Acceptance: the suite passes against the fixture backend and, where hardware exists, against
`diffusers-sdxl` and `owned-sdxl`; each later phase adds its sections here before touching the
server.

### E1.2 Multiple outputs and `count`

- `contracts.py`: add `count` to `validate_parameters` (integer, `LIMITS["count"] = [1, 4]`,
  allowed only for `text-to-image` and only when the request protocol is 1.2+; otherwise
  `unsupported`, field `count`). Add `OUTPUTS = {"text-to-image": ["image"], "image-to-image":
  ["image"]}` and a helper `output_ids(task, parameters)` that expands `image, image_2 … image_N`.
- `backend.py` / `pipeline.py`: change `generate()` to return a list of output records
  `{output_id, path, mime_type, width, height}` plus measurements; the reference backend passes
  `num_images_per_prompt=count` with per-image generators seeded `seed + k − 1`; the owned
  pipeline initializes one latent per image with the same seeds so both backends keep the
  "reproduce one image alone" rule. Batch size affects the memory estimate: `estimate()` gains a
  `count` term for activations stated in `basis`.
- `server.py`: replace the `/outputs/image` route with `/outputs/(?P<output_id>[a-z][a-z0-9_]{0,31})`;
  store per-output files under the execution directory; emit one `output` event per finalized
  file; the snapshot's `outputs` lists every record; success requires every ID from
  `output_ids()`; DELETE discards all. Storage accounting sums all outputs.
- `schemas/worker.schema.json`: `execution.parameters.count`; `output_record` definition with the
  type-specific fields from the spec (`frames`, `fps`, `duration_seconds`, `sample_rate`,
  `channels` optional now, used by E3/E4).
- `tests/fixture.py` `TestBackend.generate` returns `count` PNGs; `tests/test_worker.py` adds
  multi-output retention, per-output download and 1.1-caller refusal of `count`;
  `tests/test_inference.py` checks that `count: 2` output 2 equals a `count: 1` run at `seed + 1`
  on both backends.

Acceptance: a 1.1 request still produces exactly one `image`; a 1.2 request with `count: 3`
yields three finalized outputs with distinct digests, each reproducible alone.

### E1.3 Families and loaders

- `contracts.py`: replace the flat `ROLES`/`WEIGHT_ROLES`/`APPROVED_CLASSES` with a `FAMILIES`
  table keyed by family: `{format, roles, weight_roles, approved_classes, tasks, config_file}`.
  `sdxl-base` keeps today's values. `bundle_path()` takes the family from the bundle, checks the
  `(format, family)` pair against the table before reading any weights and names the family in
  `unsupported`.
- `backend.py` `capabilities()`: add `loaders` (one entry per family the installed extras can
  serve) and keep `model_families` as the flat list for 1.1 callers; `/v1/info` adds
  `loaded_bundle.tasks`.
- `server.py` `submit()`: reject a task the loaded family does not serve (`unsupported`, field
  `task`) before the busy check.
- Schema: `bundle.family` becomes an enum of known families; `bundle.format` is added (default
  `diffusers-safetensors` for 1.1 bodies).
- Tests: a bundle with an unknown family fails validation without opening weight files (assert
  via a directory with no weights); a loaded `sdxl-base` bundle rejects `upscale`.

### E1.4 Upscale task and the `esrgan` family

- New `src/media_engine/upscale.py`: an engine-owned RRDBNet implementation (residual-in-residual
  dense blocks, pixel-shuffle or nearest-conv upsampling as configured) that loads a single
  safetensors state dict whose keys are validated against the configured architecture before any
  tensor is moved to the device; a `tile()` helper that splits the input into overlapping tiles,
  runs each, and stitches, emitting `progress {stage:"upscaling", unit:"tiles"}` and checking the
  cancellation event between tiles.
- `contracts.py`: family `esrgan` with roles `network` (safetensors) and `configuration`
  (`config.json`: `architecture:"rrdbnet"`, `scale` ∈ {2, 4}, `in_channels`, `out_channels`,
  `num_feat`, `num_block`, `num_grow_ch`), format `single-file-safetensors`, tasks `["upscale"]`.
  Task parameters: `tile` (0 or one of `LIMITS["tile_sizes"] = [256, 512, 1024]`),
  `tile_overlap` (0–64, default 16); width/height must equal the input; no seed;
  `LIMITS["output_side"] = 8192`. `OUTPUTS["upscale"] = ["image"]`. `memory_estimate()` gains the
  per-tile activation term for this family with its basis text.
- `backend.py`: `TorchBackend.load()` dispatches by family; for `esrgan` both `DiffusersBackend`
  and `OwnedPipelineBackend` use the shared `upscale.py` network (the family has no Diffusers
  pipeline, so the ownership stage reported for this family is `owned` on both backends and
  `capabilities.loaders` says so). `generate()` for `upscale` runs the tiled network and writes a
  PNG of exactly `input × scale`.
- `server.py`: `upscale` requires `inputs.image`, forbids `prompt`, and checks `input × scale ≤
  output_side` at validation and again at submit.
- Schema: `esrgan` bundle shape, `upscale` execution body.
- Tests: `tests/fixture.py` gains `make_esrgan_bundle()` with tiny random weights and
  `make_png()`; `tests/test_inference.py` runs a 64×64 → 128×128 (scale 2) upscale through the
  real network on CPU and checks tiled and untiled outputs agree within one 8-bit level;
  `tests/test_worker.py` covers validation, progress units and cancellation between tiles.
- `examples/catalog.json` adds an upscaler handle; README documents converting a Real-ESRGAN
  `.pth` to safetensors as an operator action (the `convert` subcommand below handles SDXL
  checkpoints, not upscaler releases; a short documented `safetensors.torch.save_file` recipe
  suffices and is explicitly not run by the worker).

Acceptance: a registered `esrgan` bundle loads on both backends, upscales a PNG with correct
output dimensions and progress, refuses oversized results before sampling, and the conformance
suite's task sections pass for `upscale`.

### E1.5 `media-engine convert`

- New `src/media_engine/convert.py` and a `convert` subcommand in `server.py main()` (argparse
  subparsers; the bare invocation keeps today's serve behavior so existing launch commands work).
  Arguments: `--input FILE`, `--reference-handle HANDLE`, `--catalog FILE`, `--output DIR`,
  optional `--precision float16|float32`.
- Behavior: refuse non-safetensors input by signature; read the header; map keys to `unet`,
  `vae`, `text_encoder`, `text_encoder_2` with the Diffusers single-file conversion helpers
  (`diffusers.loaders.single_file_utils`) called offline with the reference bundle's local
  configuration; copy `tokenizer*`, `scheduler` and `model_index.json` from the reference bundle;
  write safetensors components; write `conversion.json` (`source_sha256`, `reference_bundle_digest`,
  `converter: {media_engine, diffusers, torch, safetensors}`, `mapped_keys` per role,
  `unmapped_keys` must be empty or the command fails and deletes the partial output).
- Tests: `tests/test_convert.py` builds a tiny single-file checkpoint from the fixture bundle's
  state dicts (inverse mapping on the fixture shapes), converts it, and asserts the result
  validates as an `sdxl-base` bundle, loads through `TestBackend`, and that a checkpoint with an
  extra key fails with that key named. No network, no real weights.
- Docs: `docs/protocol.md` gets a "Conversion provenance" paragraph; README gets the workflow.

Acceptance: a converted directory registers and serves identically to a native bundle, and
`conversion.json` is the only difference a supervisor can observe.

### E1.6 Automatic settings and authorized fallbacks

- `contracts.py`: `AUTO_OPTIONS = ["precision"]`, `FALLBACKS = ["cpu", "lower_precision"]`;
  `validate_load()` accepts `precision: "auto"` and `authorized_fallbacks: [...]` on 1.2 bodies
  only.
- `backend.py`: `TorchBackend.resolve_options(root, options, adapters)` returns
  `(resolved, resolutions)`: `auto` precision picks float16 on CUDA when the float16 estimate fits
  `memory_budget_mb` (or device memory when no budget), else float32 on CPU-only devices, with a
  reason string naming the estimate and budget. `load()` applies fallbacks in order: a CUDA
  device that is unavailable or whose estimate exceeds the budget falls to `cpu`/float32 only when
  `cpu` is authorized; an estimate over budget at float32 falls to float16 only when
  `lower_precision` is authorized. Record `applied_fallbacks` on the backend; include it in
  `capabilities()`/`info()` as `loaded_bundle.applied_fallbacks` and in every snapshot.
- `server.py`: `/v1/validate` and the load operation result carry `resolved_options`,
  `resolutions`, `applied_fallbacks`; `capabilities` advertises `auto_options` and `fallbacks`.
- Tests: fixture backend exposes a settable "device unavailable" and "estimate" so `test_worker.py`
  can show the unauthorized path fails with today's codes and the authorized path records the
  fallback; a 1.1 body with `auto` is `unsupported`.

Acceptance: identical requests resolve identically on the same worker; nothing changes a
generation parameter; the unauthorized behavior is byte-for-byte the 0.3.0 behavior.

### E1.7 Device identity

- `backend.py`: `device_identity()` returns `{device, index, name, total_memory_mb, host_id, pid}`
  from `torch.cuda.get_device_properties` when CUDA is selected and available, `name: "cpu"`
  otherwise; `host_id` is SHA-256 of `/etc/machine-id` when readable, else of the hostname, marked
  with its `source`.
- `server.py`: include it in `/v1/info` and the startup record (no private paths are involved).
- Tests: shape and stability within a process; the fixture backend returns a deterministic value.

### E1.8 Documentation, schema and release

- `docs/protocol.md`: add a "What changed in 1.2" table and sections for multi-output, loaders,
  upscale, conversion provenance, auto/fallbacks and device identity, mirroring the 1.1 section.
- New `docs/compatibility.md`: the compatibility record the spec requires — Engine version,
  protocol minors, Runtime versions tested, platform, families, conformance suite version.
- `schemas/worker.schema.json` `$id` bumps to 1.2; keep the 1.1 shapes valid.
- `pyproject.toml` version 0.4.0; `CHANGELOG.md`; README sections for every new feature;
  `contracts.py` `WORKER_VERSION`, `PROTOCOL`, `SUPPORTED_PROTOCOLS`.
- Run `PYTHONPATH=src python -m unittest discover -s tests -v` and, with the inference extra, the
  CPU inference tests; run Runtime 0.3.0's `npm test` against this Engine to prove 1.1 callers see
  no change before Runtime 0.4.0 exists.

## Phase E2 — Engine 0.5.0, network mode

- `server.py main()`: add `--bind` (default `127.0.0.1`), `--tls-cert`, `--tls-key`,
  `--tls-client-ca`, `--no-parent-watch`. Refuse a non-loopback bind unless all three TLS
  arguments are present; refuse TLS arguments on loopback without `--bind` only with a clear
  message (loopback TLS is allowed but pointless). Build an `ssl.SSLContext` with
  `PROTOCOL_TLS_SERVER`, minimum TLS 1.3, `verify_mode=CERT_REQUIRED`, the client CA loaded, and
  wrap the `ThreadingHTTPServer` socket. Key and certificate files are read once at start and
  must be mode 0600/0644 respectively; the worker never reloads them.
- Startup record and `/v1/info`: `scheme`, `bind`, `certificate_sha256` and
  `ownership_available` (`false` when `--no-parent-watch` is set).
- The inherited-stdin watch in `main()` becomes conditional; SIGTERM handling is unchanged.
- Tests: `tests/certs/` holds a fixture CA, server and client certificate pair generated once
  with a documented command and a 10-year validity (regenerated by `tests/certs/README.md`
  instructions, never at test time). `tests/test_network.py` launches the fixture worker on
  `127.0.0.1` with TLS and asserts: plain HTTP is refused, a client without a certificate is
  refused at the handshake, a client with the fixture certificate and a wrong bearer token gets
  `401`, and the conformance suite passes over TLS. A separate assertion covers the refusal to
  start on `0.0.0.0` without TLS material.
- Docs: README "Network mode" section with the operator's certificate responsibilities;
  `docs/protocol.md` transport section; `docs/compatibility.md` row.

Acceptance: the conformance suite passes over mutual TLS with the same results as over loopback
HTTP, and no code path serves plain HTTP on a non-loopback address.

## Phase E3 — Engine 0.6.0, protocol 1.3, image-to-video

- `pyproject.toml`: extra `inference-video = ["av==<pinned>"]` (PyAV wheels bundle FFmpeg; no
  PATH lookup). `backend.py` `missing_dependencies()` reports `av` per task so `image-to-video`
  is simply absent from `tasks` when the extra is missing.
- `contracts.py`: family `svd` (roles `unet` → `UNetSpatioTemporalConditionModel`, `vae` →
  `AutoencoderKLTemporalDecoder`, `image_encoder` → `CLIPVisionModelWithProjection`,
  `feature_extractor` → `CLIPImageProcessor`, `scheduler` → `EulerDiscreteScheduler`), format
  `diffusers-safetensors`, tasks `["image-to-video"]`. Parameters: `seed`, `steps` (1–50),
  `frames` (bundle default from `unet` config, 1–64), `fps` (1–30, default 7), `motion` (1–255,
  default 127), `noise_augmentation` (0–1, default 0.02), `decode_chunk` (1–16, default 8);
  width/height equal the input and must be in `LIMITS["video_sizes"]` advertised from the bundle
  (reference 1024×576 and 576×1024). `OUTPUTS["image-to-video"] = ["video", "poster"]`.
- `backend.py` `DiffusersBackend`: load `StableVideoDiffusionPipeline` from the verified
  components; `generate()` emits `encoding` (image conditioning), `sampling` with step callback,
  `decoding` with per-chunk progress (`unit:"frames"`), `encoding_container` while PyAV writes
  H.264/yuv420p at `fps` with constant frame rate to a temporary file, then `saving`; the poster
  PNG is frame 0. The output record carries `frames`, `fps`, `duration_seconds`, `width`,
  `height`. Memory estimate adds the per-frame latent term with its basis.
- New `src/media_engine/video.py`: the MP4 writer and a signature check used by tests
  (`ftyp` box). Cancellation is checked per step, per decode chunk and before container encoding.
- `pipeline.py`: the owned pipeline does not serve `svd` in 0.6.0; `capabilities.loaders` lists
  `svd` only for `diffusers-sdxl`, and a request to the owned backend is `unsupported`. Owned
  `svd` is a later optimization-stage item once the reference path is validated on hardware.
- Tests: `tests/fixture.py` gains `make_svd_bundle()` with tiny random weights and the fixture
  backend produces a two-frame MP4 through `video.py`; `tests/test_inference.py` runs a real CPU
  `image-to-video` at 64×64, 2 frames, 2 steps and validates the container, frame count and
  poster; `test_worker.py` covers validation and 1.2-caller refusal.
- Docs: protocol 1.3 section, README (video extra, reference bundle, license), compatibility row,
  decisions entry recording the reference weights.

Acceptance: a video execution produces a playable MP4 and poster with correct metadata; a worker
without the extra does not advertise the task; 1.2 callers are unaffected.

## Phase E4 — Engine 0.7.0, protocol 1.3, text-to-audio

- `pyproject.toml`: extra `inference-audio` (any additional pins the family needs beyond torch,
  such as `torchaudio` if the VAE requires it; otherwise the base extra suffices and the extra is
  empty but documented).
- `contracts.py`: family `stable-audio-open` (roles `transformer` → `StableAudioDiTModel`, `vae`
  → `AutoencoderOobleck`, `text_encoder` → `T5EncoderModel`, `tokenizer` → `T5TokenizerFast`,
  `projection_model` → `StableAudioProjectionModel`, `scheduler` → the family's scheduler class),
  tasks `["text-to-audio"]`. Parameters: `seed`, `steps` (1–200), `guidance` (0–30),
  `duration_seconds` within the bundle's advertised range (reference 1–47); sample rate and
  channels from the VAE config. `OUTPUTS["text-to-audio"] = ["audio"]`.
- `backend.py`: load `StableAudioPipeline`; `generate()` emits `encoding`, `sampling`,
  `decoding`, `saving`; new `src/media_engine/audio.py` writes 16-bit PCM WAV with the stdlib
  `wave` module, reports `sample_rate`, `channels`, `duration_seconds` (actual, after latent
  granularity) in the output record.
- Tests: tiny random-weight audio fixture, real CPU run of one second at a low sample rate,
  WAV header check, 1.2-caller refusal, duration reporting.
- Docs and compatibility as in E3; decisions entry for the reference weights and license.

Acceptance: a WAV with the reported sample rate, channels and duration; actual duration is
reported, never trimmed silently.

## Phase E5 — Engine 0.8.0, targeted optimization

- New `tools/benchmark.py`: loads a bundle, runs a fixed request set per option value, records
  wall time, peak device memory, output digest and (for images) a per-pixel difference against
  the baseline; writes `docs/benchmarks/<date>-<device>.json`. An optimization is advertised only
  when a benchmark file for the reference hardware exists and is cited in `docs/compatibility.md`.
- `contracts.py` / `backend.py`: load options gain `precision: bfloat16`, `attention: flash`
  (advertised only when the pinned kernel package imports), `quantization:
  none|int8-weight-only` (pinned `torchao` in an `inference-optimize` extra), `compile:
  none|inductor`. Each is part of the load identity. `capabilities.optimizations` lists
  `{option, value, benchmark, determinism, quality_note}` per advertised entry; `auto_options`
  does not include any of them until a benchmark justifies it for a device class.
- The owned pipeline applies the same options where the semantics are identical; where it cannot
  (for example a kernel the owned loop does not call), the capability is backend-specific and
  says so.
- Tests: option validation, load-identity change on switch, and the inference test asserting
  that `bfloat16` on CPU (where available) produces an image within the stated tolerance.

Acceptance: every advertised optimization has a cited benchmark; none is applied implicitly.

## Phase E6 — Engine 1.0.0, owned native engine

- Decision record: choose the language (C++ or Rust) with the reasons, the tensor/GPU libraries
  retained, and the build/release identity scheme; record it in `docs/decisions.md` (add that
  file to this repository, mirroring Runtime's).
- Repository layout: `native/` with its own build, producing `media-engine-native`, sharing
  `schemas/worker.schema.json` and the catalog/bundle formats. It implements the launch contract
  (`--probe`, startup record, `--catalog`, `--directory`, inherited credential, `--port 0`,
  `--device`), `backend_id: native-sdxl`, `ownership: engine`, and the full 1.3 surface for
  `sdxl-base` first; other families follow as separate increments after passing the suite.
- Oracle tests: extend `tests/test_inference.py` to run the same tiny-bundle requests against
  the native binary when `MEDIA_ENGINE_NATIVE` points at it, asserting the semantic equalities
  already asserted between the two Python backends (same seed → within tolerance, LoRA changes
  output, cancellation and progress shapes identical).
- Conformance: the suite from E1.1 runs against the native binary in CI; `docs/compatibility.md`
  records the suite version it passed.
- Packaging: the native binary ships as its own versioned artifact; the Python package keeps the
  reference and owned backends and is not required on a host that runs only the native engine.

Acceptance: conformance and oracle suites pass; Runtime's integration suite passes against the
native worker unchanged.

## Cross-cutting work

- **Fixture backend parity.** Runtime's integration tests launch `tests/serve_fixture.py`. Every
  phase extends `TestBackend` (and the fixture bundle builders) in the same change so Runtime can
  test against the new protocol surface without hardware. Fixture outputs stay identified in
  provenance and are never an inference fallback.
- **Schema discipline.** One `schemas/worker.schema.json` with `$defs` per minor; a `protocol`
  field selects the allowed shape. A test asserts every request in `docs/protocol.md` examples
  validates.
- **Compatibility record.** `docs/compatibility.md` is updated in every release commit with
  Engine version, protocol minors, Runtime versions tested, platform, families, extras,
  conformance suite version and benchmark files.
- **Hardware validation.** This workspace has no CUDA device or real weights. Each phase's
  acceptance includes a hardware checklist in the release notes that an operator runs on the
  reference RTX 4090 target; the CHANGELOG states explicitly when a release's GPU path is
  implemented but not hardware-validated, as 0.1.0–0.3.0 do.
- **Release checklist.** Version bump in `pyproject.toml` and `contracts.py`, CHANGELOG entry,
  README update, protocol document, schema `$id`, compatibility record, `uv.lock` refresh when
  pins change, `unittest` and inference tests green, Runtime integration suite green against the
  new worker for both the current and the previous Runtime release.

## Risks and open decisions

| Item                                   | Risk or decision                                                                                      | Plan                                                             |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Single-file conversion coverage        | Diffusers' key mapping covers SDXL base variants but not every fine-tune layout                       | Fail on unmapped keys with names; no partial bundles             |
| Upscaler weights                       | Real-ESRGAN releases are `.pth`; operators must convert                                               | Document; worker refuses anything but safetensors                |
| PyAV dependency size                   | Adds a large wheel to the environment                                                                 | Separate `inference-video` extra; task absent without it         |
| Audio reference weights and license    | Candidate weights carry a community license                                                           | Record in decisions; operators supply weights                    |
| TLS operator burden                    | Certificate generation and rotation are outside the worker                                            | Document minimal `openssl` commands; refuse ambiguous launches   |
| Native engine scope                    | A full native SDXL implementation is large                                                            | Gate on the conformance suite; ship `sdxl-base` text-to-image first |
| Memory estimates for new families      | Activation terms are family-specific and unmeasured here                                              | State basis and exclusions; hardware checklist measures peaks    |
