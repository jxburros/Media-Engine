---
document_type: product_specification
status: draft_proposal
spec_revision: 0.2.0
created_at: 2026-10-26
updated_at: 2026-10-26
working_name: Media Engine
---

# Media Engine Specification

Media Engine is a separately supervisable inference worker for local media generation. It loads an explicit model bundle, executes a supported pipeline, reports genuine progress and returns outputs through a versioned public protocol. It can serve Media Runtime or another local supervisor.

The first implementation is a reference worker built on Diffusers and PyTorch. A project-owned pipeline and later native engine can implement the same protocol. Owning the engine does not require replacing every tensor library or GPU kernel.

This specification describes proposed behavior. `MUST` defines a required behavior within its stated delivery stage. [Media Runtime](runtime-spec.md) owns persistent jobs, scheduling, recipes and application integration; this specification is authoritative for the shared worker protocol.

## Delivery status

Revision 0.1.0 of this specification described the reference worker. Engine 0.3.0 delivers it, the owned SDXL pipeline backend, `media-worker/1.1`, image-to-image and LoRA adapters; [docs/protocol.md](docs/protocol.md) is the implementation contract for what ships. Revision 0.2.0 specifies the capabilities the first revision deferred so that each has an explicit stage, protocol version and acceptance rule before implementation. The [implementation plan](docs/implementation-plan.md) orders them.

| Capability                                            | Stage               | Protocol | Status in Engine 0.3.0 |
| ----------------------------------------------------- | ------------------- | -------- | ---------------------- |
| Reference worker, text-to-image, cancellation, SSE    | Reference worker    | 1.0      | Delivered              |
| Image-to-image, owned pipeline, memory estimates      | Owned pipeline      | 1.1      | Delivered              |
| LoRA adapters as load identity                        | Owned pipeline      | 1.1      | Delivered              |
| Multiple outputs per execution, batch count           | Additional tasks    | 1.2      | Specified below        |
| Upscaling task and the `esrgan` family                | Additional tasks    | 1.2      | Specified below        |
| Loader capabilities and single-file conversion        | Additional tasks    | 1.2      | Specified below        |
| Automatic settings and authorized fallbacks           | Additional tasks    | 1.2      | Specified below        |
| Device identity for supervisor worker pools           | Additional tasks    | 1.2      | Specified below        |
| Network mode for remote supervisors                   | Remote execution    | 1.2      | Specified below        |
| Image-to-video task and the `svd` family              | Additional media    | 1.3      | Specified below        |
| Text-to-audio task and the `stable-audio-open` family | Additional media    | 1.3      | Specified below        |
| Targeted optimization options                         | Optimization        | 1.3      | Specified below        |
| Owned native engine and protocol conformance suite    | Owned engine        | 1.x      | Specified below        |

Arbitrary graph import, executable recipe code, plugin marketplaces, in-process multi-device execution and model downloading remain outside this specification. Nothing in a later stage weakens a rule stated for an earlier one.

## Product sheet

| Property               | Proposed specification                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| Primary caller         | A local runtime or process supervisor; a remote supervisor over mutual TLS in network mode   |
| Delivered operations   | Text-to-image and image-to-image on one supported SDXL base pipeline, with LoRA adapters      |
| Planned operations     | Upscaling, image-to-video and text-to-audio, each a separately advertised task and family    |
| Initial implementation | Python worker with pinned Diffusers, PyTorch and required local components                   |
| Future implementation  | Owned pipeline/server, potentially native code, using selected compute libraries             |
| Process scope          | One device and one loaded pipeline bundle, including its component models                    |
| Concurrency            | One active execution; no pending-job queue; several processes serve several devices          |
| Protocol               | Versioned loopback HTTP/JSON, SSE events, opaque input/output handles; TLS in network mode   |
| Readiness              | Service readiness, model readiness and device readiness reported separately                  |
| Lifetime               | Warm worker survives executions; explicit unload releases the bundle                         |
| Privacy                | No telemetry, automatic downloads, arbitrary URL fetching or remote model code                |
| Distribution           | Independently versioned worker packages/binaries with dependency and build identity          |

## Responsibilities

The engine owns component loading, tokenization and text encoding for the selected pipeline, latent initialization, sampling, decoding, device allocation, actual inference progress and temporary output bytes. It reports effective settings, supported limits, memory estimates and measured resource data where available.

It MUST NOT choose a different model, compose application prompts, route tasks, own Studio users/projects, maintain a durable waiting queue, install model weights, or execute arbitrary recipe code. A second concurrent execution is rejected as busy. The supervisor owns persistent job identity and scheduling.

A standalone caller may operate the public engine API directly, but it then owns durable job recording and output collection. The worker provides execution handles, not a permanent media library.

## Model bundles

The initial loader accepts a complete local Diffusers-format bundle for the supported pipeline. A file ending in `.safetensors` is not sufficient evidence of compatibility. Other architectures, quantizations and single-file conversions require explicit loader capabilities.

The supervisor resolves a bundle manifest containing:

| Field                                          | Meaning                                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| `schema_version`, `bundle_id`, `bundle_digest` | Manifest format and immutable component/configuration identity               |
| `family`, `pipeline`                           | Explicit architecture family and requested task                              |
| `components`                                   | Roles, registered local artifact handles, content digests and tensor formats |
| `configuration`                                | Local architecture, scheduler and tokenizer artifacts with identities        |
| `source`                                       | Available upstream repository/revision and license metadata                  |

Required roles for the reference bundle include the denoiser, both SDXL text encoder/tokenizer pairs, VAE and pipeline configuration. Roles are family-specific; the protocol does not force every future model to have that structure.

At launch, an operator supplies a read-only artifact catalog mapping opaque handles to approved local files/directories and a separate private work directory. Requests use those handles, never arbitrary client paths or URLs. The worker validates paths remain within approved roots, including symlink resolution, and verifies expected component identities before load. Digest verification may be cached only against trustworthy unchanged artifact identity.

Loading runs offline with remote custom-code execution disabled. A missing, changed or unsupported component produces an error naming its role. The engine MUST NOT substitute a component or download a replacement.

### Families and loaders

From protocol 1.2 the worker advertises `capabilities.loaders`, a list of `{format, families, roles}` entries. A format names how artifacts are laid out on disk; a family names an architecture whose roles, configuration and approved component classes the worker knows. Loading a bundle whose `(format, family)` pair is not advertised fails with `unsupported` and field `family` before any file is read for weights.

| Family              | Format                   | Roles                                                                        | Task(s)          | Stage            |
| ------------------- | ------------------------ | ---------------------------------------------------------------------------- | ---------------- | ---------------- |
| `sdxl-base`         | `diffusers-safetensors`  | unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, scheduler   | text/image→image | Delivered        |
| `esrgan`            | `single-file-safetensors` | network, configuration                                                       | upscale          | Additional tasks |
| `svd`               | `diffusers-safetensors`  | unet, vae, image_encoder, feature_extractor, scheduler                       | image-to-video   | Additional media |
| `stable-audio-open` | `diffusers-safetensors`  | transformer, vae, text_encoder, tokenizer, projection_model, scheduler       | text-to-audio    | Additional media |

Every family keeps the existing rules: safetensors weights only, no Python or pickle weights, no symlinks, per-file SHA-256 digests, role-prefixed component digests and a canonical bundle digest. The `single-file-safetensors` format is one weights file plus one configuration file under one catalog handle; it is a layout, not permission to load arbitrary checkpoints. A single-file SDXL checkpoint is not loadable in that format; it is converted first (see [Single-file checkpoint conversion](#single-file-checkpoint-conversion)).

A bundle's `pipeline` names its primary task. From 1.2 the worker reports `loaded_bundle.tasks`, the tasks the loaded family can serve, and an execution whose task the loaded bundle cannot serve fails with `unsupported` and field `task`.

### Single-file checkpoint conversion

Conversion is an explicit, offline operator action performed by the `media-engine convert` subcommand, never by a load request. It reads one single-file SDXL safetensors checkpoint, takes architecture, scheduler and tokenizer configuration from an approved reference bundle handle in the launch catalog, and writes a complete Diffusers-format bundle directory containing safetensors components only.

- Conversion MUST refuse pickle, Python or unknown-key checkpoints and MUST NOT download configuration, tokenizers or missing components. Keys it cannot map fail the conversion with the key names.
- The output directory contains `conversion.json`: source file digest, reference bundle digest, converter package versions and the mapped key counts per role. The supervisor's catalog inspection surfaces this provenance.
- The converted directory is then approved, catalogued and registered like any other bundle; the worker treats it identically and cannot tell it apart except through `conversion.json`.
- Precision is preserved unless the operator passes an explicit precision; the recorded output digests make the result identity-stable.

## Loading and GPU memory

The worker reports `starting`, `ready`, `loading`, `busy`, `unloading` or `error`, plus whether a bundle is loaded. `ready` means the service can accept supported control operations; generation additionally requires a compatible loaded bundle and device. A bound port alone does not establish generation readiness.

Loading and unloading are asynchronous control operations with their own operation IDs and observable status. They cannot overlap inference in the initial worker. Load failure returns the worker to an explicit usable empty state or an error state requiring restart. Unload is complete only when component references and applicable device allocations are released; retained framework overhead is reported where measurable.

The initial configuration selects one GPU explicitly. Precision, offload and attention settings are backend capabilities and must be validated before loading. Automatic execution settings, when supported, return resolved values and reasons. CPU fallback or a precision change requires caller authorization; it is never an invisible recovery action.

Memory estimates include the basis, device, pipeline settings and uncertainty. Measurements distinguish process memory, device allocation and framework-reserved memory where available. Image resolution and pipeline stages affect working memory; language-model KV-cache calculations are not a substitute.

An out-of-memory error fails the execution with `out_of_memory`. The worker either returns to a demonstrably usable state or requests restart. It MUST NOT silently regenerate at a smaller size. The supervisor decides whether to unload/restart and retry explicitly.

### Automatic settings and authorized fallbacks

From protocol 1.2 a load option may carry the value `auto` where the capability advertises it (`capabilities.auto_options`). The worker resolves `auto` before loading from the selected device, the bundle's memory estimate and its own measured defaults. The validate response and the load operation result carry `resolved_options` and a `resolutions` list of `{option, requested, resolved, reason}`. A supervisor records the resolved values; a later request repeating them resolves identically on the same worker, device and bundle.

A load body may also carry `authorized_fallbacks`, a list drawn from `capabilities.fallbacks`: initially `cpu` (load on the host CPU in float32 when the selected device is unavailable or cannot hold the bundle) and `lower_precision` (choose the next advertised lower-memory precision when the estimate exceeds the device budget). Without the matching authorization the worker fails the load with `out_of_memory` or `unsupported` as today. When a fallback is applied, the operation result, `loaded_bundle.applied_fallbacks` and every subsequent execution snapshot name it. A fallback is never applied during an execution, never changes the family or bundle, and never alters generation parameters.

## Devices

A worker process serves exactly one device. Multi-GPU hosts run one process per device, each with its own port, credential, private work directory and session; a supervisor pool is the only concurrency model. In-process multi-device execution is not planned.

From protocol 1.2 `/v1/info` and the startup record include `device_identity`: the device index, the driver-reported device name, total device memory where measurable, a stable `host_id` derived from the host's machine identity and the process ID. A supervisor uses it to attribute jobs, enforce per-device budgets and detect two workers claiming one device. `--memory-budget-mb` remains per process; the worker does not coordinate with other processes on the same device.

## Worker protocol

The draft wire protocol is `media-worker/1.0`. This protocol version is independent of worker software, recipe and model-manifest versions. Runtime and worker negotiate a supported major and minor before any load or generation request. Unknown required fields/options are rejected; adding response fields does not imply a new capability.

### Protocol minors

Each minor is additive. A worker lists every minor it serves in `supported_protocols`; a request's `protocol` field selects the minor, and a request MUST NOT use a field, task, option or output shape introduced by a higher minor than the one it names.

| Minor | Additions                                                                                                                                                                                      |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0   | Text-to-image, load/unload operations, single-flight executions, SSE resume, cancellation, one `image` output                                                                                   |
| 1.1   | `supported_protocols`, image-to-image, `POST /v1/inputs`, `memory_estimate`, `effective_steps`, `owned-sdxl`, LoRA `adapters` on the load body                                                  |
| 1.2   | Multiple outputs and `count`, `upscale`, `loaders`/`families`, `auto` options with `resolutions`, `authorized_fallbacks`, `device_identity`, network mode fields, `loaded_bundle.tasks`           |
| 1.3   | `image-to-video` with `video`/`poster` outputs, `text-to-audio` with `audio` output, per-task `output_formats`, optimization options advertised under `capabilities.optimizations`               |

A 1.1 caller against a 1.3 worker sees 1.1 behavior exactly: one `image` output, no new tasks, no new options. The supervisor's compatibility record lists which minor each runtime release negotiates.

### Process launch and discovery

The executable exposes `--help`, `--version`, and a machine-readable protocol/version probe. The worker launches on loopback with `--port 0` support, an operator-supplied artifact catalog, a private work directory and a per-launch credential supplied through a protected inherited channel rather than command-line arguments.

After binding, stdout emits one JSON startup record with `protocol`, `worker_version`, random `session_id`, selected port and process ID. It excludes credentials and private paths. Logs go to stderr with bounded supervisor capture. The supervisor verifies discovery and readiness before passing work; it never infers readiness from the startup record alone.

### Network mode

By default the worker binds loopback and speaks plain HTTP. Network mode is an explicit launch configuration for a supervisor on another host; it is part of the remote-execution stage and changes no request or event semantics.

- `--bind ADDRESS` other than a loopback address requires `--tls-cert`, `--tls-key` and `--tls-client-ca`. The worker MUST refuse to start on a non-loopback address without all three, and MUST NOT serve plain HTTP there, including `/health`.
- Transport is TLS 1.3 or later with client certificate verification against the operator-supplied CA. The bearer credential is still required on every endpoint except `/health`; the certificate authenticates the peer, the credential authorizes the session.
- The startup record and `/v1/info` report `scheme`, `bind` and the server certificate fingerprint so a supervisor can pin it. The private work directory, catalog and credentials stay on the worker host.
- A remote supervisor cannot own the process: SIGTERM, stdin-close and forced termination remain host-local. `--no-parent-watch` disables the inherited-stdin shutdown for service-manager launches. The worker advertises `ownership_available: false` in that case so the supervisor records interruption, not confirmed cancellation, when it loses the connection.
- Inputs and outputs already travel as bytes under session-scoped handles, so no file path crosses the network. Timeouts, retries and reconnection remain the supervisor's policy; the worker's retention guarantee is unchanged.

### Endpoints

All endpoints except coarse `/health` require the per-launch bearer credential. Every resource is scoped to this worker session.

| Endpoint                                                | Contract                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `GET /health`                                           | Coarse liveness; `503` during startup or fatal error                                  |
| `GET /v1/info`                                          | Protocol/build versions, session ID, worker/device/model readiness, device identity   |
| `GET /v1/capabilities`                                  | Available tasks, families, loaders, formats, options, limits, fallbacks, optimizations |
| `POST /v1/validate`                                     | Validate resolved bundle/operation settings without loading or executing              |
| `POST /v1/model/load`                                   | Begin bundle loading; `202` with `operation_id`                                       |
| `POST /v1/model/unload`                                 | Begin unloading when idle; `202` with `operation_id`                                  |
| `GET /v1/operations/{operation_id}`                     | Loading/unloading state, resolutions, applied fallbacks and structured error          |
| `POST /v1/inputs`                                       | Upload one input file by declared MIME type; `201` with opaque `input_id`             |
| `GET/DELETE /v1/inputs/{input_id}`                      | Inspect or discard an uploaded input                                                  |
| `POST /v1/executions`                                   | Accept one operation immediately or reject as busy; `202` with execution handle       |
| `GET /v1/executions?attempt_id=`                        | Find an accepted attempt without resubmission                                         |
| `GET /v1/executions/{execution_id}`                     | Current execution snapshot and last event sequence                                    |
| `GET /v1/executions/{execution_id}/events`              | SSE with monotonic sequence IDs and `Last-Event-ID` resume                            |
| `POST /v1/executions/{execution_id}/cancel`             | Request cooperative cancellation or return existing terminal state                    |
| `GET /v1/executions/{execution_id}/outputs/{output_id}` | Stream finalized bytes by opaque handle                                               |
| `DELETE /v1/executions/{execution_id}`                  | Discard a terminal execution and its temporary outputs after collection               |

The worker retains executions/events/outputs for its session until explicit discard, subject to advertised storage/record limits. It rejects new work when it cannot preserve uncollected records; it does not evict them silently. Session shutdown ends this guarantee. Durable retention belongs to the runtime.

Capabilities specify exact accepted options, numeric ranges/alignment, required model components, output formats and defaults. They distinguish features implemented by the worker from features available for the loaded bundle and selected device. Unsupported combinations are rejected before dispatch and checked again against actual execution state.

### Execution request

An execution is one coarse generation operation. Intermediate tensors stay inside the worker process. A representative request is:

```json
{
  "protocol": "media-worker/1.0",
  "attempt_id": "attempt_123",
  "bundle_digest": "sha256:<resolved-bundle-digest>",
  "task": "text-to-image",
  "inputs": { "prompt": "A portrait in soft window light" },
  "parameters": {
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 6
  }
}
```

The digest placeholder represents a verified digest, not a valid literal request. Sampler, scheduler and backend-specific options must be resolved from supported identifiers before execution; the worker returns all effective values. The bundle digest must match the loaded bundle.

`attempt_id` is a supervisor-generated idempotency key within one worker session. The worker atomically records an accepted request before returning its handle. An identical retry returns the same handle; the same ID with a different body returns `409`. This permits recovery from a lost acceptance response while the worker survives. A new session has no deduplication guarantee for previous attempts.

No waiting execution is accepted behind the active one. Busy or conflicting load/unload operations return `409` with a structured reason. Storage exhaustion returns `503` with `storage_full`; unknown tasks/options and invalid settings return `400`.

### Multiple outputs

From protocol 1.2 an execution may finalize more than one output. `capabilities.task_parameters.<task>.outputs` lists the output IDs a task produces and their MIME types, and `limits.count` bounds the per-execution batch. For `text-to-image` the parameter `count` (default 1, at most the advertised limit, 1 on 1.0/1.1 requests) produces outputs `image`, `image_2` … `image_N` in seed order: output `k` uses `seed + k - 1`, so a supervisor can reproduce any single image alone. Count multiplies working memory; the estimate and the busy/OOM rules apply unchanged, and the worker MUST NOT reduce `count` to fit.

Every output record carries `output_id`, `mime_type`, `size`, `sha256` and type-specific facts: `width`/`height` for images, `width`/`height`/`frames`/`fps`/`duration_seconds` for video, `sample_rate`/`channels`/`duration_seconds` for audio. An execution succeeds only when every advertised output of the task is finalized; a partial batch is a failure that names the missing output.

### States and events

Execution states are `running`, `succeeded`, `failed` and `cancelled`; there is no engine `queued` state. Acceptance reserves the single execution slot. Stages include `encoding`, `sampling`, `decoding` and `saving`; `upscaling` is added for the upscale task and `encoding_container` for video. Model loading is observed through its control-operation status before execution.

Events are `state`, `stage`, `progress`, `output` and `error`. Each carries `session_id`, `execution_id`, increasing `sequence` and timestamp. Sampling progress includes actual `completed_steps` and `total_steps`; stage progress is not asserted to be overall completion. Tile, frame and chunk progress use the same `{stage, scope:"stage", completed_steps, total_steps}` shape with the unit named in `unit`. A snapshot includes current state/stage, progress, effective settings, last sequence, outputs and any error.

The supervisor may reconnect with `Last-Event-ID`; the worker replays subsequent retained events and continues live delivery. If the cursor is invalid or no longer retained, it returns an explicit resume error and requires a snapshot. A missing execution in a new session is never reported as successful or automatically restarted.

Final output records include `output_id`, MIME type, byte count, digest and actual dimensions. Only finalized files are downloadable. Success means inference and worker-local output finalization completed; the runtime separately commits durable copies before settling its own job. Run metadata includes actual seed, component identities, effective generation/execution settings and implementation versions. Determinism is qualified by backend, hardware and dependency versions.

### Cancellation and errors

Cancellation is cooperative at supported execution checkpoints, including denoising callbacks in the reference worker. Acceptance of cancellation returns `202` and sets intent; terminal `cancelled` follows only after computation stops and execution resources are settled. A disconnect from the event stream is not cancellation. If success committed first, cancellation returns that result.

The supervisor may terminate its own unresponsive worker after its configured deadline. This interrupts execution and loses session-local records; the runtime records an interrupted job rather than pretending the worker confirmed cancellation. SIGTERM stops admission and attempts bounded cleanup. Forced termination requires a fresh worker session and reload.

Errors contain `code`, readable `message`, optional `field` or component role, `retryable`, and an optional `restart_required` flag. Codes include `invalid_request`, `unsupported`, `missing_component`, `artifact_changed`, `model_not_ready`, `busy`, `out_of_memory`, `storage_full`, `execution_failed` and `protocol_mismatch`. Retry advice never initiates a retry. Auth uses `401/403`, inaccessible handles use `404`, and errors exclude credentials, prompt bodies and uncontrolled paths or stderr.

## Additional tasks

Each task below is advertised separately in `capabilities.tasks`, has its own family, inputs, parameters, limits, stages and outputs, and is rejected with `unsupported` by a worker or protocol minor that does not list it. A task never changes the rules of another task.

### Upscale

`upscale` (protocol 1.2) enlarges one uploaded PNG by the loaded bundle's fixed integer factor. The first family is `esrgan`: a single RRDB-style super-resolution network whose architecture parameters (`scale`, channels, feature width, block count) come from the bundle's configuration file. The network implementation belongs to the engine; the weights belong to the operator. Real-ESRGAN `.pth` releases are converted to safetensors by the operator before approval; the worker refuses anything else.

| Item       | Rule                                                                                                                          |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Inputs     | `image` (an `input_id`); no prompt                                                                                            |
| Parameters | `tile` (0 = whole image, or an advertised tile size), `tile_overlap`; width/height MUST equal the input; no seed             |
| Output     | `image` as PNG at exactly input size × `scale`; `limits.output_side` bounds the result (proposed 8192)                       |
| Stages     | `upscaling` with per-tile progress, then `saving`                                                                             |
| Memory     | Estimate from network parameters plus the largest tile's activations at the load precision; tiling is explicit, never automatic |

Upscaling is deterministic for a given bundle and precision up to backend floating-point qualification; no seed is recorded. A supervisor chaining generation and upscaling runs two executions, loading each bundle explicitly; the worker holds one bundle at a time.

### Image-to-video

`image-to-video` (protocol 1.3) animates one uploaded PNG with the `svd` family (Stable Video Diffusion image-to-video, spatio-temporal UNet, temporal-decoder VAE, CLIP image encoder, Euler scheduler) through the reference backend first, then the owned pipeline.

| Item       | Rule                                                                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inputs     | `image`; width/height MUST equal the input and fall within the family's advertised sizes (the reference bundle trains at 1024×576)                    |
| Parameters | `seed`, `steps`, `frames` (bundle default), `fps`, `motion` (motion bucket 1–255), `noise_augmentation` (0–1), `decode_chunk` (frames per VAE decode) |
| Outputs    | `video` as `video/mp4` (H.264, yuv420p, constant frame rate at `fps`) and `poster` as the first frame in PNG                                           |
| Stages     | `encoding` (image conditioning), `sampling` with step progress, `decoding` with per-chunk progress, `encoding_container`, `saving`                     |
| Encoder    | A pinned PyAV wheel in the `inference-video` extra; the worker never searches `PATH` for an ffmpeg executable                                          |

Frame count and frame rate are explicit parameters recorded in the output record and run metadata. The worker MUST NOT drop frames, re-time or re-encode at a different rate to fit a size limit; an oversized result is `storage_full` or a validation failure, decided before sampling from the advertised per-frame estimate.

### Text-to-audio

`text-to-audio` (protocol 1.3) generates one audio clip from a prompt with the `stable-audio-open` family (diffusion transformer, Oobleck VAE, T5 text encoder and tokenizer, timing projection, scheduler). The reference model choice is recorded as an implementation decision; the family rules apply to any model with those roles.

| Item       | Rule                                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Inputs     | `prompt`, optional `negative_prompt`                                                                                                  |
| Parameters | `seed`, `steps`, `guidance`, `duration_seconds` within the bundle's advertised range; `sample_rate` and channels come from the bundle |
| Output     | `audio` as `audio/wav` (PCM 16-bit, written by the engine without an external encoder)                                                |
| Stages     | `encoding`, `sampling` with step progress, `decoding`, `saving`                                                                       |

The output record carries sample rate, channel count and actual duration. Requested duration is honored exactly up to the model's latent granularity; the effective duration is reported, never silently trimmed to a limit.

## Reference implementation

The first worker wraps the supported Diffusers pipeline in an isolated, pinned Python environment. It owns the HTTP lifecycle and adapts library callbacks into the public execution events. Blocking GPU work must not block health, status or cancellation handling. The first implementation supports one active inference thread/process path and serializes access to its pipeline.

The reference worker exposes only options it can apply. Diffusers defaults are made explicit in resolved metadata. Python exceptions become structured failures; import/dependency failures appear before readiness. A fake worker, if provided for development, identifies itself in discovery and provenance and is never a silent generation fallback.

Pin the Python version, Diffusers/PyTorch builds, CUDA requirements and supporting libraries per supported platform. Package environments independently from Studio and the runtime. Source revision, package checksums and dependency identity travel with releases. Backend license terms and model-weight licenses remain distinct.

Additional media families add optional extras (`inference-video`, `inference-audio`) rather than enlarging the base inference environment. A worker whose extras are absent still serves the protocol and reports the missing dependency per task in `capabilities.missing_dependencies`; the task is then not advertised.

## Progressive inference ownership

| Stage                 | Engine ownership                                                                                        | Libraries retained                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Reference worker      | Public protocol, validation, lifecycle, events, cancellation and output handling                        | Complete Diffusers pipeline and PyTorch                              |
| Owned pipeline        | Component assembly, denoising loop, scheduling integration, caching and memory lifecycle for one family | Selected Diffusers model/scheduler components and PyTorch operations |
| Owned engine          | Dedicated server/executor and supported model implementations; native code where useful                | Selected tensor, GPU, decoding and tokenizer libraries               |
| Targeted optimization | Specific quantization, attention, kernels or device execution paths with demonstrated value             | Existing implementations that remain appropriate                     |

Implement the owned engine as an additional backend while retaining the reference implementation. An optimized library inside the worker need not become a separate service or repository. Extract libraries only when there is a real reuse or release boundary. This mirrors Onyx's independent serving/execution layer over llama.cpp/ggml without assuming that ownership requires a complete compute-stack rewrite.

Preserve request semantics across implementations. Backend-specific features are advertised explicitly, and a caller chooses whether to use them. Backend switching must not silently reinterpret a recipe, scheduler or parameter. Do not promise matching image bytes across implementations.

### Protocol conformance suite

Before any backend beyond the two delivered ones is advertised, the engine repository MUST provide a conformance suite that runs against any worker URL and credential: discovery, negotiation of every listed minor, validation, load/unload, single-flight and idempotent attempts, SSE resume, cancellation at each checkpoint, output finalization and digest, retention limits and error privacy. A backend or native engine is "conformant" when the suite passes with no skipped section; the compatibility record cites the suite version.

### Owned native engine

The owned engine is a separately built executable implementing `media-worker` directly. It MUST pass the conformance suite, expose the same launch contract (`--probe`, startup record, catalog, private directory, inherited credential) and advertise `backend_id: native-sdxl` with `ownership: engine`. It need not reproduce the Python package's internals; it keeps bundle identity, digests and family rules unchanged so the same registration serves both. The implementation language is an open decision recorded when work starts; C++ and Rust are the candidates. The Python reference backend remains supported and is the behavioral oracle for the semantics tests.

### Targeted optimization

Each optimization is an advertised value of an existing load option or a new entry under `capabilities.optimizations`, selected explicitly by the caller: candidate values are `attention: flash` (when the pinned kernel package is installed), `precision: bfloat16`, `quantization: int8-weight-only` and `compile: inductor`. An optimization is advertised only when the release records a benchmark on the reference hardware showing the improvement, its effect on determinism and any quality qualification. It is never applied through `auto` resolution unless `capabilities.auto_options` says so for that device class. Switching an optimization is a load-identity change, like precision.

## Compatibility and integration

Publish worker/API versions independently from Media Runtime. A protocol-major mismatch prevents execution with an actionable compatibility error. Minor additions are negotiated; optional features require capability checks. Keep a maintained compatibility record of runtime version, worker package, protocol, platform and supported model bundle identities.

The supervisor interacts through public startup, discovery and HTTP behavior. It does not import worker internals or depend on private Diffusers classes. The runtime translates engine events into its durable job model; Studio sees the runtime contract and requires no direct worker credentials or tensor access.

## Decisions and next steps

The delivered reference target is one SDXL base bundle on Linux x64/NVIDIA; the recorded pins are in the repository's implementation decisions. Open decisions for the deferred stages, to be recorded when each starts: the exact upscaler, video and audio reference weights and their licenses; whether the MP4 encoder dependency is acceptable for the base extra or stays optional; the native engine language; the first optimization worth advertising; and the certificate tooling expected of operators in network mode.

The next implementation step is protocol 1.2 in the Python worker: multiple outputs, the upscale task with the `esrgan` family and loader capabilities, the conversion subcommand, automatic settings with authorized fallbacks and device identity, together with the conformance suite that every later backend must pass. The [implementation plan](docs/implementation-plan.md) sequences the rest.
