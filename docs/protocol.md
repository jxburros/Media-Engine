# media-worker/1.1 implementation contract

This document concretizes [engine-spec.md](../engine-spec.md). Engine 0.2.0 speaks
`media-worker/1.1` and still accepts `media-worker/1.0` requests; `/v1/info`, `/v1/capabilities`
and the startup record list `supported_protocols`. A supervisor picks the highest minor both sides
support and sends that value in every request's `protocol` field. A different major is rejected
with `protocol_mismatch`. Request schemas are in `schemas/worker.schema.json`; unknown fields and
options are errors. Authentication is a bearer credential supplied through the launch environment,
and all sockets bind loopback.

`GET /v1/info` reports service state independently of device/dependency readiness and loaded model
identity, plus the backend's `ownership` stage. `GET /v1/capabilities` advertises implemented tasks,
per-task inputs/parameters, ranges, exact generation/load defaults, precision, progress/cancellation,
retention limits, the memory estimate for the validated or loaded bundle, and actual dependency
versions. Coarse `/health` contains no paths or model information. Startup discovery is not
generation readiness.

## What changed in 1.1

| Addition                                   | Meaning                                                                          |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| `supported_protocols`                      | Minor negotiation; 1.0 callers see unchanged behavior                            |
| task `image-to-image`                      | Separately advertised in `tasks`; needs `inputs.image` and `parameters.strength` |
| `POST /v1/inputs`                          | Upload one PNG (`Content-Type: image/png`) and receive an opaque `input_id`      |
| `GET/DELETE /v1/inputs/{input_id}`         | Inspect or discard an uploaded input; busy while an execution references it      |
| `memory_estimate`                          | Parameter bytes per role from safetensors headers at the load precision          |
| `effective_steps`                          | Actual sampling steps; for image-to-image it is `floor(steps × strength)`        |
| `backend_id` `owned-sdxl`                  | The project-owned pipeline stage, selected explicitly at launch                  |

A 1.0 request cannot name `image-to-image`; the worker returns `unsupported` with field `task`.

## Backends

The executable is launched with `--backend diffusers-sdxl` (default, the reference stage: the
complete Diffusers pipelines execute) or `--backend owned-sdxl` (the owned-pipeline stage: this
project assembles the components, encodes prompts, initializes latents, runs the Euler loop with
classifier-free guidance, noises image inputs and decodes). Both honor identical requests, ranges,
events and cancellation checkpoints. Component classes stay the installed Diffusers/Transformers
model classes. The supervisor selects the worker and therefore the backend; a request's
`backend_id` must match the launched backend. Matching image bytes between backends are not promised.

## Bundles and control

The operator launch catalog maps opaque handles to absolute approved directories. A resolved bundle
contains `schema_version:1`, `bundle_id`, `family:sdxl-base`, `pipeline:text-to-image`, `components`,
`configuration`, `source`, and `bundle_digest`. The same bundle serves both tasks.

`configuration` is `{artifact_handle, files}` where `files` maps every relative file name to raw
lowercase SHA-256 hex. Each component has `role`, the same `artifact_handle`, `tensor_format`
(`safetensors` for unet/vae/text_encoder/text_encoder_2, otherwise `json`), and `digest`.
A component digest is `sha256:` plus the hash of its role-prefixed subset of `files`.
The bundle digest hashes the whole bundle excluding `bundle_digest`.
Canonical JSON uses recursively sorted object keys, UTF-8, no insignificant whitespace, and integers
as integers. Source metadata keys are repository, revision and license. No executable fields exist.

`POST /v1/validate` accepts either the load body below or a complete execution body. Validation never
loads weights onto the device. Bundle validation verifies content and standard installed component
class identities and returns `memory_estimate`: parameter counts read from safetensors headers,
converted to bytes at the requested precision, per role and in total, with `source`, `basis`,
`device` and `uncertainty`. It excludes activations, the VAE float32 upcast, allocator reserve and
framework overhead; it is a lower bound, not a fit guarantee. Execution validation returns the
task and `effective_steps`. The operator must keep catalog roots immutable; validation is not a
filesystem sandbox.

`POST /v1/model/load` accepts `{protocol, bundle, options}`. Options are exactly
`{precision:"float16"|"float32", attention:"sdpa", offload:"none"}`. CPU only supports float32.
`POST /v1/model/unload` accepts `{protocol}`. Both return 202 and an `operation_id` immediately.
`GET /v1/operations/{operation_id}` reports running/succeeded/failed; a succeeded load carries the
estimate. A failed load leaves an empty worker; no substitute model is loaded. Extension: DELETE
on a terminal operation discards its record.

## Inputs

`POST /v1/inputs` takes raw PNG bytes with `Content-Type: image/png`, at most `max_input_mb`
(default 32 MiB, `--max-input-mb`). The worker checks the signature, IHDR and IEND, requires 8-bit
depth and a greyscale/RGB(A) color type, and requires dimensions within the advertised limits and
multiples of eight. It stores the bytes under its private session directory and returns
`{input_id, session_id, mime_type, size, sha256, width, height}` with 201. Inputs count toward the
retained-record and storage limits and are never evicted; discard them with DELETE after the
executions that used them are collected. A handle from another session is `input_not_found`.

## Execution

`POST /v1/executions` takes exactly `{protocol,attempt_id,bundle_digest,task,inputs,parameters}`.
Tasks are `text-to-image` and `image-to-image`. Inputs are prompt (required), negative_prompt
(optional), max 10,000 characters each, and for image-to-image `image` (an `input_id`). Parameters
are resolved seed (uint32), width/height (64–2048, multiples of eight), steps (1–150), guidance
(0–30), sampler `euler`, scheduler `euler`, and for image-to-image `strength` (0 < s ≤ 1, with
`floor(steps × strength) ≥ 1`). For image-to-image, width and height must equal the uploaded
image's real dimensions; the worker never resizes (`invalid_request`, field `width`). Loaded
precision/attention/offload are part of the load identity; generation cannot silently change them.
One PNG is produced at the requested size.

Acceptance returns 202 with an opaque `execution_id`, `session_id`, `attempt_id`, `task`, the
request `protocol`, and the snapshot. Attempt IDs match `attempt_[A-Za-z0-9_-]{1,80}`. Identical
attempts deduplicate within the session until discard; conflicts return 409. No engine queue
exists. Extension: `GET /v1/executions?attempt_id=…` finds an accepted attempt without
resubmission, so supervisors can reconcile a lost response.

Snapshots include state/stage, effective settings (with `effective_steps`), implementation
versions, last_sequence, all retained events, outputs, measurements and structured error. Polling
is a supported recovery mechanism alongside SSE.
`GET /v1/executions/{execution_id}/events` replays after Last-Event-ID and follows live events. Invalid
cursors return 409 resume_required. IDs increase from one, scoped to execution/session. Events are
state, stage, progress, output and error; each carries timestamp, session_id, execution_id, sequence,
type and data. Progress is `{stage:"sampling",scope:"stage",completed_steps,total_steps}` where
`total_steps` is the effective count; never total progress.

`POST /v1/executions/{execution_id}/cancel` acknowledges intent (202), or returns the immutable
terminal snapshot (200). Cooperative checkpoints stop encoding/sampling/decoding transitions and
every sampling step. A supervisor may kill only a process it owns; that is interruption, not
confirmed cancellation. Terminal states are succeeded/failed/cancelled.

Outputs include output_id=image, image/png MIME, byte size, SHA-256 and actual width/height.
`GET /v1/executions/{execution_id}/outputs/image` streams finalized bytes only. DELETE on a terminal
execution discards records/events/output after collection. Missing output bytes return 410.
Outputs survive client disconnects within the worker session, not process death.

Errors include code, readable message, retryable and optional field/restart_required. They never
contain raw exceptions, prompts, credentials or private paths. Busy returns 409; storage_full 503;
invalid requests 400; oversized uploads 413; missing handles 404; auth 401/403. OOM fails the
execution and marks the worker error/restart-required. A fresh session never implies an old
attempt succeeded.
