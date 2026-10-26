# media-worker/1.0 implementation contract

This document concretizes [engine-spec.md](../engine-spec.md). Runtime 0.1.0 and Engine 0.1.0
negotiate the exact `media-worker/1.0` minor; a different major/minor is rejected. Request schemas
are in `schemas/worker.schema.json`; unknown fields/options are errors. Authentication is a bearer
credential supplied through the launch environment, and all sockets bind loopback.

`GET /v1/info` reports service state independently of device/dependency readiness and loaded model
identity. `GET /v1/capabilities` advertises implemented tasks, ranges, exact generation/load defaults,
precision, progress/cancellation, retention limits and actual dependency versions. Coarse `/health`
contains no paths or model information. Startup discovery is not generation readiness.

## Bundles and control

The operator launch catalog maps opaque handles to absolute approved directories. A resolved bundle
contains `schema_version:1`, `bundle_id`, `family:sdxl-base`, `pipeline:text-to-image`, `components`,
`configuration`, `source`, and `bundle_digest`.

`configuration` is `{artifact_handle, files}` where `files` maps every relative file name to raw
lowercase SHA-256 hex. Each component has `role`, the same `artifact_handle`, `tensor_format`
(`safetensors` for unet/vae/text_encoder/text_encoder_2, otherwise `json`), and `digest`.
A component digest is `sha256:` plus the hash of its role-prefixed subset of `files`.
The bundle digest hashes the whole bundle excluding `bundle_digest`.
Canonical JSON uses recursively sorted object keys, UTF-8, no insignificant whitespace, and integers
as integers. Source metadata keys are repository, revision and license. No executable fields exist.

`POST /v1/validate` accepts either the load body below or a complete execution body. Validation never
loads weights onto the device. Bundle validation verifies content and standard installed component
class identities. The operator must keep catalog roots immutable; validation is not a filesystem sandbox.

`POST /v1/model/load` accepts `{protocol, bundle, options}`. Options are exactly
`{precision:"float16"|"float32", attention:"sdpa", offload:"none"}`. CPU only supports float32.
`POST /v1/model/unload` accepts `{protocol}`. Both return 202 and an `operation_id` immediately.
`GET /v1/operations/{operation_id}` reports running/succeeded/failed. A failed load leaves an empty
worker; no substitute model is loaded. Extension: DELETE on a terminal operation discards its record.

## Execution

`POST /v1/executions` takes exactly `{protocol,attempt_id,bundle_digest,task,inputs,parameters}`.
Task is `text-to-image`; inputs are prompt (required) and negative_prompt (optional), max 10,000
characters each. Parameters are resolved seed (uint32), width/height (64–2048, multiples of eight),
steps (1–150), guidance (0–30), sampler `euler` and scheduler `euler`. Loaded precision/attention/offload
are part of the load identity; generation cannot silently change them. One PNG is produced.

Acceptance returns 202 with an opaque `execution_id`, `session_id`, `attempt_id` and snapshot.
Attempt IDs match `attempt_[A-Za-z0-9_-]{1,80}`. Identical attempts deduplicate within the session
until discard; conflicts return 409. No engine queue exists. Extension: `GET /v1/executions?attempt_id=…`
finds an accepted attempt without resubmission, so supervisors can reconcile a lost response.

Snapshots include state/stage, effective settings, implementation versions, last_sequence, all retained
events, outputs and structured error. Polling is a supported recovery mechanism alongside SSE.
`GET /v1/executions/{execution_id}/events` replays after Last-Event-ID and follows live events. Invalid
cursors return 409 resume_required. IDs increase from one, scoped to execution/session. Events are
state, stage, progress, output and error; each carries timestamp, session_id, execution_id, sequence,
type and data. Progress is `{stage:"sampling",scope:"stage",completed_steps,total_steps}`; never total progress.

`POST /v1/executions/{execution_id}/cancel` acknowledges intent (202), or returns the immutable
terminal snapshot (200). Cooperative checkpoints stop encoding/sampling/decoding transitions and
sampling callbacks. A supervisor may kill only a process it owns; that is interruption, not confirmed
cancellation. Terminal states are succeeded/failed/cancelled.

Outputs include output_id=image, image/png MIME, byte size, SHA-256 and actual width/height.
`GET /v1/executions/{execution_id}/outputs/image` streams finalized bytes only. DELETE on a terminal
execution discards records/events/output after collection. Missing output bytes return 410.
Outputs survive client disconnects within the worker session, not process death.

Errors include code, readable message, retryable and optional field/restart_required. They never
contain raw exceptions, prompts, credentials or private paths. Busy returns 409; storage_full 503;
invalid requests 400; missing handles 404; auth 401/403. OOM fails the execution and marks the worker
error/restart-required. A fresh session never implies an old attempt succeeded.
