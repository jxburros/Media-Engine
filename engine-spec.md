---
document_type: product_specification
status: draft_proposal
spec_revision: 0.1.0
created_at: 2026-10-26
working_name: Media Engine
---

# Media Engine Specification

Media Engine is a separately supervisable inference worker for local media generation. It loads an explicit model bundle, executes a supported pipeline, reports genuine progress and returns outputs through a versioned public protocol. It can serve Media Runtime or another local supervisor.

The first implementation is a reference worker built on Diffusers and PyTorch. A project-owned pipeline and later native engine can implement the same protocol. Owning the engine does not require replacing every tensor library or GPU kernel.

This specification describes proposed behavior. `MUST` defines a required behavior within its stated delivery stage. [Media Runtime](runtime-spec.md) owns persistent jobs, scheduling, recipes and application integration; this specification is authoritative for the shared worker protocol.

## Product sheet

| Property               | Proposed specification                                                              |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Primary caller         | A local runtime or process supervisor                                               |
| Initial operation      | Text-to-image using one supported SDXL base pipeline                                |
| Initial implementation | Python worker with pinned Diffusers, PyTorch and required local components          |
| Future implementation  | Owned pipeline/server, potentially native code, using selected compute libraries    |
| Process scope          | One device and one loaded pipeline bundle, including its component models           |
| Concurrency            | One active execution; no pending-job queue in the initial worker                    |
| Protocol               | Versioned loopback HTTP/JSON, SSE events, opaque output handles                     |
| Readiness              | Service readiness, model readiness and device readiness reported separately         |
| Lifetime               | Warm worker survives executions; explicit unload releases the bundle                |
| Privacy                | No telemetry, automatic downloads, arbitrary URL fetching or remote model code      |
| Distribution           | Independently versioned worker packages/binaries with dependency and build identity |

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

## Loading and GPU memory

The worker reports `starting`, `ready`, `loading`, `busy`, `unloading` or `error`, plus whether a bundle is loaded. `ready` means the service can accept supported control operations; generation additionally requires a compatible loaded bundle and device. A bound port alone does not establish generation readiness.

Loading and unloading are asynchronous control operations with their own operation IDs and observable status. They cannot overlap inference in the initial worker. Load failure returns the worker to an explicit usable empty state or an error state requiring restart. Unload is complete only when component references and applicable device allocations are released; retained framework overhead is reported where measurable.

The initial configuration selects one GPU explicitly. Precision, offload and attention settings are backend capabilities and must be validated before loading. Automatic execution settings, when supported, return resolved values and reasons. CPU fallback or a precision change requires caller authorization; it is never an invisible recovery action.

Memory estimates include the basis, device, pipeline settings and uncertainty. Measurements distinguish process memory, device allocation and framework-reserved memory where available. Image resolution and pipeline stages affect working memory; language-model KV-cache calculations are not a substitute.

An out-of-memory error fails the execution with `out_of_memory`. The worker either returns to a demonstrably usable state or requests restart. It MUST NOT silently regenerate at a smaller size. The supervisor decides whether to unload/restart and retry explicitly.

## Worker protocol

The draft wire protocol is `media-worker/1.0`. This protocol version is independent of worker software, recipe and model-manifest versions. Runtime and worker negotiate a supported major and minor before any load or generation request. Unknown required fields/options are rejected; adding response fields does not imply a new capability.

### Process launch and discovery

The executable exposes `--help`, `--version`, and a machine-readable protocol/version probe. The worker launches on loopback with `--port 0` support, an operator-supplied artifact catalog, a private work directory and a per-launch credential supplied through a protected inherited channel rather than command-line arguments.

After binding, stdout emits one JSON startup record with `protocol`, `worker_version`, random `session_id`, selected port and process ID. It excludes credentials and private paths. Logs go to stderr with bounded supervisor capture. The supervisor verifies discovery and readiness before passing work; it never infers readiness from the startup record alone.

### Endpoints

All endpoints except coarse `/health` require the per-launch bearer credential. Every resource is scoped to this worker session.

| Endpoint                                                | Contract                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `GET /health`                                           | Coarse liveness; `503` during startup or fatal error                                  |
| `GET /v1/info`                                          | Protocol/build versions, session ID, worker/device/model readiness                    |
| `GET /v1/capabilities`                                  | Available tasks, families, formats, options, limits and cancellation/progress support |
| `POST /v1/validate`                                     | Validate resolved bundle/operation settings without loading or executing              |
| `POST /v1/model/load`                                   | Begin bundle loading; `202` with `operation_id`                                       |
| `POST /v1/model/unload`                                 | Begin unloading when idle; `202` with `operation_id`                                  |
| `GET /v1/operations/{operation_id}`                     | Loading/unloading state and structured error                                          |
| `POST /v1/executions`                                   | Accept one operation immediately or reject as busy; `202` with execution handle       |
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

### States and events

Execution states are `running`, `succeeded`, `failed` and `cancelled`; there is no engine `queued` state. Acceptance reserves the single execution slot. Stages include `encoding`, `sampling`, `decoding` and `saving`. Model loading is observed through its control-operation status before execution.

Events are `state`, `stage`, `progress`, `output` and `error`. Each carries `session_id`, `execution_id`, increasing `sequence` and timestamp. Sampling progress includes actual `completed_steps` and `total_steps`; stage progress is not asserted to be overall completion. A snapshot includes current state/stage, progress, effective settings, last sequence, outputs and any error.

The supervisor may reconnect with `Last-Event-ID`; the worker replays subsequent retained events and continues live delivery. If the cursor is invalid or no longer retained, it returns an explicit resume error and requires a snapshot. A missing execution in a new session is never reported as successful or automatically restarted.

Final output records include `output_id`, MIME type, byte count, digest and actual dimensions. Only finalized files are downloadable. Success means inference and worker-local output finalization completed; the runtime separately commits durable copies before settling its own job. Run metadata includes actual seed, component identities, effective generation/execution settings and implementation versions. Determinism is qualified by backend, hardware and dependency versions.

### Cancellation and errors

Cancellation is cooperative at supported execution checkpoints, including denoising callbacks in the reference worker. Acceptance of cancellation returns `202` and sets intent; terminal `cancelled` follows only after computation stops and execution resources are settled. A disconnect from the event stream is not cancellation. If success committed first, cancellation returns that result.

The supervisor may terminate its own unresponsive worker after its configured deadline. This interrupts execution and loses session-local records; the runtime records an interrupted job rather than pretending the worker confirmed cancellation. SIGTERM stops admission and attempts bounded cleanup. Forced termination requires a fresh worker session and reload.

Errors contain `code`, readable `message`, optional `field` or component role, `retryable`, and an optional `restart_required` flag. Codes include `invalid_request`, `unsupported`, `missing_component`, `artifact_changed`, `model_not_ready`, `busy`, `out_of_memory`, `storage_full`, `execution_failed` and `protocol_mismatch`. Retry advice never initiates a retry. Auth uses `401/403`, inaccessible handles use `404`, and errors exclude credentials, prompt bodies and uncontrolled paths or stderr.

## Reference implementation

The first worker wraps the supported Diffusers pipeline in an isolated, pinned Python environment. It owns the HTTP lifecycle and adapts library callbacks into the public execution events. Blocking GPU work must not block health, status or cancellation handling. The first implementation supports one active inference thread/process path and serializes access to its pipeline.

The reference worker exposes only options it can apply. Diffusers defaults are made explicit in resolved metadata. Python exceptions become structured failures; import/dependency failures appear before readiness. A fake worker, if provided for development, identifies itself in discovery and provenance and is never a silent generation fallback.

Pin the Python version, Diffusers/PyTorch builds, CUDA requirements and supporting libraries per supported platform. Package environments independently from Studio and the runtime. Source revision, package checksums and dependency identity travel with releases. Backend license terms and model-weight licenses remain distinct.

## Progressive inference ownership

| Stage                 | Engine ownership                                                                                        | Libraries retained                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Reference worker      | Public protocol, validation, lifecycle, events, cancellation and output handling                        | Complete Diffusers pipeline and PyTorch                              |
| Owned pipeline        | Component assembly, denoising loop, scheduling integration, caching and memory lifecycle for one family | Selected Diffusers model/scheduler components and PyTorch operations |
| Owned engine          | Dedicated server/executor and supported model implementations; native code where useful                 | Selected tensor, GPU, decoding and tokenizer libraries               |
| Targeted optimization | Specific quantization, attention, kernels or device execution paths with demonstrated value             | Existing implementations that remain appropriate                     |

Implement the owned engine as an additional backend while retaining the reference implementation. An optimized library inside the worker need not become a separate service or repository. Extract libraries only when there is a real reuse or release boundary. This mirrors Onyx's independent serving/execution layer over llama.cpp/ggml without assuming that ownership requires a complete compute-stack rewrite.

Preserve request semantics across implementations. Backend-specific features are advertised explicitly, and a caller chooses whether to use them. Backend switching must not silently reinterpret a recipe, scheduler or parameter. Do not promise matching image bytes across implementations.

## Compatibility and integration

Publish worker/API versions independently from Media Runtime. A protocol-major mismatch prevents execution with an actionable compatibility error. Minor additions are negotiated; optional features require capability checks. Keep a maintained compatibility record of runtime version, worker package, protocol, platform and supported model bundle identities.

The supervisor interacts through public startup, discovery and HTTP behavior. It does not import worker internals or depend on private Diffusers classes. The runtime translates engine events into its durable job model; Studio sees the runtime contract and requires no direct worker credentials or tensor access.

## Initial decisions and next step

The proposed reference target is one SDXL base bundle on Linux x64/NVIDIA. Select and record the exact model artifacts, GPU, Python/dependency pins, precision and compatible sampler/scheduler identifiers when implementation starts. Final product/repository names, release license and the eventual native implementation language remain open. Python is sufficient for the first worker; choosing C++ or another native language is not a prerequisite.

The next implementation step is machine-readable definitions of this worker protocol and a reference worker that loads the selected bundle, executes one text-to-image request, supports real cancellation and returns finalized bytes plus provenance. The first replacement target is the pipeline loop, before custom tensor libraries or GPU kernels.
