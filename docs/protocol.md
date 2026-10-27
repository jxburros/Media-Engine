# media-worker/1.2 implementation contract

This document concretizes [engine-spec.md](../engine-spec.md). Engine 0.5.0 speaks
`media-worker/1.2` and still accepts `media-worker/1.0` and `media-worker/1.1` requests;
`/v1/info`, `/v1/capabilities` and the startup record list `supported_protocols`. Every 1.2
behavior is gated on the request's `protocol` minor: a 1.0/1.1 request sees exactly the 0.3.0
responses (response objects may gain fields, never lose or change them). A supervisor picks the highest minor both sides
support and sends that value in every request's `protocol` field. A different major is rejected
with `protocol_mismatch`. Request schemas are in `schemas/worker.schema.json`; unknown fields and
options are errors. Authentication is a bearer credential supplied through the launch environment.
Sockets bind loopback unless the operator launches network mode over mutual TLS (see
[Transport](#transport)); Engine 0.5.0 added that without a new minor.

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
| `adapters` (0.3.0)                         | LoRA adapters on the load body; advertised in `capabilities.adapters`            |

A 1.0 request cannot name `image-to-image`; the worker returns `unsupported` with field `task`.

## What changed in 1.2

| Addition                                         | Meaning                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `parameters.count` (text-to-image)               | 1–4 outputs `image, image_2 … image_N`; output k uses `seed + k − 1`                           |
| `GET /v1/executions/{id}/outputs/{output_id}`    | Per-output download; `outputs/image` keeps working                                            |
| `task_parameters.<task>.outputs`                 | Output IDs a task produces (plus `output_formats`, `defaults`, `families`)                    |
| `capabilities.loaders`, `bundle.format`          | `(format, family)` pairs the worker can load; `esrgan` uses `single-file-safetensors`          |
| task `upscale`, family `esrgan`                  | Engine-owned RRDBNet; `inputs.image`, `tile`, `tile_overlap`; output exactly input × `scale`  |
| `loaded_bundle.family`, `.tasks`, `.scale`       | What the loaded bundle serves; an execution for another task is `unsupported`, field `task`   |
| `precision: "auto"`, `capabilities.auto_options` | Resolved before loading; `resolved_options` and `resolutions` on validate and the operation    |
| `authorized_fallbacks: ["cpu","lower_precision"]` | Applied only when authorized; `applied_fallbacks` on operation, `loaded_bundle`, snapshots    |
| `device_identity`                                | `/v1/info` and the startup record: device, index, name, memory, hashed `host_id`, pid         |
| `bundle.source.conversion`                       | Provenance of a `media-engine convert` output, checked against its `conversion.json`          |
| `highest_protocol` (startup record)              | The highest minor served; the record's `protocol` stays `media-worker/1.1` (see Discovery)    |
| `scheme`, `bind`, `certificate_sha256`, `ownership_available` (0.5.0) | `/v1/info` and the startup record: the transport served (see [Transport](#transport)) |

A 1.1 caller sees none of this: `count` is `unsupported` (field `count`), `upscale` is
`unsupported` (field `task`), `precision: "auto"` is `unsupported` (field `options`, as any
unadvertised precision was in 0.3.0), `authorized_fallbacks` is `unsupported` (field
`authorized_fallbacks`), and `format`, an `esrgan` family or `source.conversion` in a 1.1
bundle is refused (`invalid_request` for the unknown field, `unsupported`/`family` for the
family). 1.1 snapshots carry no `count` in `effective_settings` and no `applied_fallbacks` unless
a 1.2 load applied one. A backend is advertised only after the conformance suite
(`tests/conformance`, version `media-worker-conformance/1.0`) passes against it with no skipped
section; [compatibility.md](compatibility.md) records the result.

### Discovery in 1.2

The startup record is `{protocol, supported_protocols, highest_protocol, worker_version,
backend_id, session_id, port, pid, device_identity}`, plus the transport fields from Engine 0.5.0. Its `protocol` field names the launch-record
minor and stays `media-worker/1.1`: the record's shape only gained fields in 1.2, and Runtime 0.3.0
refuses a startup record whose `protocol` it does not know. Supervisors negotiate from
`supported_protocols` (identical in the record, `/v1/info` and `/v1/capabilities`); `/v1/info`
and `/v1/capabilities` report `protocol: "media-worker/1.2"`. `--probe` reports 1.2.

`/v1/info` adds `device_identity`. `/v1/capabilities` adds (example from the fixture):

```json
{
  "tasks": ["text-to-image", "image-to-image", "upscale"],
  "model_families": ["sdxl-base", "esrgan"],
  "formats": ["diffusers-safetensors", "single-file-safetensors"],
  "loaders": [
    {"format": "diffusers-safetensors", "families": ["sdxl-base"],
     "roles": ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"],
     "tasks": ["text-to-image", "image-to-image"], "ownership": "reference"},
    {"format": "single-file-safetensors", "families": ["esrgan"], "roles": ["network", "configuration"],
     "tasks": ["upscale"], "ownership": "owned"}
  ],
  "task_parameters": {
    "text-to-image": {"inputs": ["prompt", "negative_prompt"], "parameters": ["count"],
      "defaults": {"count": 1}, "outputs": ["image"], "output_formats": {"image": "image/png"},
      "repeated_outputs": ["image"], "families": ["sdxl-base"]},
    "image-to-image": {"inputs": ["prompt", "negative_prompt", "image"], "parameters": ["strength"],
      "input_formats": ["image/png"], "outputs": ["image"], "output_formats": {"image": "image/png"},
      "families": ["sdxl-base"]},
    "upscale": {"inputs": ["image"], "parameters": ["tile", "tile_overlap"],
      "defaults": {"tile": 0, "tile_overlap": 16}, "input_formats": ["image/png"],
      "outputs": ["image"], "output_formats": {"image": "image/png"}, "families": ["esrgan"],
      "stages": ["upscaling", "saving"], "progress_unit": "tiles"}
  },
  "task_defaults": {"image-to-image": {"strength": 0.75}, "upscale": {"tile": 0, "tile_overlap": 16}},
  "limits": {"count": [1, 4], "tile_sizes": [256, 512, 1024], "tile_overlap": [0, 64], "output_side": 8192},
  "auto_options": ["precision"],
  "fallbacks": ["cpu", "lower_precision"],
  "adapters": {"families": ["sdxl-base"]}
}
```

(`adapters` and `limits` keep their 1.1 keys; only the additions are shown.) `defaults` is
unchanged: 1.2 parameter defaults live in `task_parameters.<task>.defaults`, so a 1.1 supervisor
that merges `defaults` into every request never sends `count`. `loaders[].ownership` is the
backend's stage for `sdxl-base` and `owned` for `esrgan` on every backend, because the upscaler
network is engine code. `dependencies` lists what each backend actually imports (the owned backend
reports torch, diffusers, transformers and safetensors only).

```json
"device_identity": {"device": "cuda:0", "index": 0, "name": "NVIDIA GeForce RTX 4090",
  "total_memory_mb": 24076, "host_id": "<64 hex>", "host_id_source": "machine-id", "pid": 4242}
```

`host_id` is SHA-256 of `/etc/machine-id` (`host_id_source: "machine-id"`), or of the hostname
(`"hostname"`) when the machine ID is unreadable; neither raw value is exposed. On `--device cpu`
it reports `name: "cpu"`, `index: null` and physical memory. Identity is stable for the process.

### Multiple outputs

A 1.2 `text-to-image` request may carry `parameters.count` (integer within `limits.count`,
default 1). The execution must finalize `image`, `image_2` … `image_N`; output k uses
`seed + k − 1`, so any output can be reproduced alone with `count: 1` at that seed (byte-identical
on the fixture and on CPU float32; within one 8-bit level otherwise). `seed + count − 1` must be a
valid uint32. For each ID listed in `task_parameters.<task>.repeated_outputs`, an execution with
`count: N` produces `id, id_2 … id_N`; every other ID in `outputs` appears once. `/v1/validate`
on a 1.2 execution body returns `outputs` (the exact IDs) and `effective_settings`.

Each finalized file gets one `output` event, all after generation and before the terminal
`state` event; the snapshot's `outputs` lists `{output_id, mime_type, size, sha256, width,
height}` in ID order (width/height are read back from the PNG header). An execution succeeds only
when every expected ID is finalized: a missing one fails the execution with `execution_failed`
and `field` naming it, and nothing is published. Storage accounting reserves and sums every
output; `DELETE` discards all of them. `GET /v1/executions/{id}/outputs/{output_id}` (ID pattern
`[a-z][a-z0-9_]{0,31}`) streams one finalized file with its `mime_type`; an ID the execution does
not have is `404 output_not_found`, a vanished file `410 output_unavailable`. 1.2 snapshots carry
`effective_settings.count` and `applied_fallbacks` (empty when none).

### Families and loaders

| Family      | `format`                  | `pipeline`      | Components (`role` → `tensor_format`)                                              | Tasks                          |
| ----------- | ------------------------- | --------------- | ---------------------------------------------------------------------------------- | ------------------------------ |
| `sdxl-base` | `diffusers-safetensors`   | `text-to-image` | unet, vae, text_encoder, text_encoder_2 → `safetensors`; tokenizer, tokenizer_2, scheduler → `json` | text-to-image, image-to-image |
| `esrgan`    | `single-file-safetensors` | `upscale`       | network → `safetensors`; configuration → `json`                                    | upscale                        |

`bundle.format` is optional for `sdxl-base` (absent means `diffusers-safetensors`) and required
for `esrgan`; it is a 1.2 field. The `(format, family)` pair and `pipeline` are checked first,
before any file is read: an unknown family, a 1.2 family on a 1.1 request, a format that does not
match or a wrong pipeline is `unsupported` with field `family`.

An `esrgan` registration (exactly what a supervisor builds from the directory):

```json
{
  "schema_version": 1, "bundle_id": "realesrgan-x2", "format": "single-file-safetensors",
  "family": "esrgan", "pipeline": "upscale",
  "components": [
    {"role": "network", "artifact_handle": "upscaler", "tensor_format": "safetensors",
     "digest": "sha256:<digest of {\"network.safetensors\": \"<hex>\"}>"},
    {"role": "configuration", "artifact_handle": "upscaler", "tensor_format": "json",
     "digest": "sha256:<digest of {\"config.json\": \"<hex>\"}>"}
  ],
  "configuration": {"artifact_handle": "upscaler",
                    "files": {"config.json": "<hex>", "network.safetensors": "<hex>"}},
  "source": {"license": "BSD-3-Clause"},
  "bundle_digest": "sha256:<digest of everything above>"
}
```

The directory holds exactly two top-level files: one `*.safetensors` weights file (any name; the
`network` component digest covers that one entry) and `config.json` (the `configuration`
component covers it). `config.json` is
`{architecture: "rrdbnet", scale: 2|4, in_channels, out_channels, num_feat, num_block,
num_grow_ch}` plus optional `upsampler: "nearest-conv"` (default, Real-ESRGAN layout; scale 2
runs the body at half resolution after pixel-unshuffle) or `"pixel-shuffle"`. Validation reads the
weights header and requires its tensor names and shapes to equal the configured architecture
exactly (`unsupported`, field `network`); a configuration that is not a supported RRDBNet is
`unsupported`, field `configuration`. Changed bytes are `artifact_changed` with field `network` or
`configuration`. Adapters apply to `sdxl-base` only (`unsupported`, field `adapters`).

A 1.2 bundle validation and load report `family`, `tasks` and family facts (`scale` for
`esrgan`); `loaded_bundle` gains `family`, `tasks`, `resolved_options`, `applied_fallbacks` and,
for `esrgan`, `scale`. When an execution names the loaded bundle's digest but a task its family
does not serve, it is refused with `unsupported`, field `task`, before the busy check. (A request
naming a different digest is `model_not_ready`, as in 1.1.)

### Upscale

```json
{"protocol": "media-worker/1.2", "attempt_id": "attempt_up1", "bundle_digest": "sha256:<esrgan>",
 "task": "upscale", "inputs": {"image": "input_<32 hex>"},
 "parameters": {"width": 512, "height": 512, "tile": 256, "tile_overlap": 16}}
```

`inputs` is exactly `{image}`; a `prompt` or `negative_prompt` is `invalid_request` with field
`prompt`. `parameters` is `width`, `height` (must equal the uploaded image, `invalid_request`
field `width` otherwise) and optional `tile` (0 = one whole-image pass, or one of
`limits.tile_sizes`; field `tile`) and `tile_overlap` (`limits.tile_overlap`, default 16; field
`tile_overlap`). There is no seed, steps or guidance; any other field is `invalid_request`. With
the bundle loaded, `max(width, height) × scale` must not exceed `limits.output_side`; otherwise
`/v1/validate` and submission both return `invalid_request` with field `output_side` before
anything runs. Tiling is explicit, never chosen by the worker.

Stages are `upscaling` then `saving`; progress is
`{stage: "upscaling", scope: "stage", unit: "tiles", completed_steps, total_steps}` with one step
per tile, and cancellation is checked before every tile. The output `image` is a PNG of exactly
`width × scale` by `height × scale`, written by the engine's standard-library PNG writer.
`effective_settings` is `{width, height, tile, tile_overlap, scale, output_width,
output_height}`; `measurements.tiles` counts the tiles. The memory estimate adds the activations
of one 1024-pixel tile with maximum overlap (`activations_mb`) and states that `tile: 0` is not
covered. Upscaling is deterministic for a bundle and precision; both backends run the same
network and produce identical bytes on CPU.

### Conversion provenance

`media-engine convert --input FILE --reference-handle HANDLE --catalog FILE --output DIR
[--precision float16|float32]` is an offline operator command, never a request. It refuses
anything that is not a safetensors container by its signature, maps the checkpoint's
`model.diffusion_model.*`, `first_stage_model.*`, `conditioner.embedders.0.transformer.*` and
`conditioner.embedders.1.model.*` keys with the Diffusers single-file helpers using the reference
bundle's local configuration, verifies every converted tensor name and shape against the reference
model classes, copies `tokenizer`, `tokenizer_2`, `scheduler`, the four component `config.json`
files and `model_index.json` from the reference, and writes `conversion.json`:

```json
{"schema_version": 1, "source_sha256": "<64 hex of the checkpoint file>",
 "reference_bundle_digest": "sha256:<canonical digest of the reference directory's {file: sha256} map>",
 "converter": {"media_engine": "0.4.0", "diffusers": "0.32.2", "torch": "2.6.0", "safetensors": "0.5.3"},
 "precision": "preserved",
 "mapped_keys": {"unet": 1680, "vae": 248, "text_encoder": 196, "text_encoder_2": 517},
 "unmapped_keys": [], "ignored_keys": []}
```

Any key outside those prefixes or not consumed by the mapping fails with `unmapped_keys` and the
names; shape or name mismatches fail with `incompatible_keys`; the partial output is deleted
(errors print `{"error": code, "keys": [...]}` on stderr, never paths or exceptions, exit 2).
`ignored_keys` lists stored derived buffers (`position_ids`) the classes recompute. The result
is an ordinary `sdxl-base` directory whose `configuration.files` includes the top-level
`conversion.json`; it registers, loads and serves exactly like a native bundle.

A 1.2 registration may carry the provenance as `bundle.source.conversion`, an object with the
required `source_sha256`, `reference_bundle_digest`, `converter` (exactly the four keys) and
`mapped_keys` (exactly the four weight roles), and optionally `schema_version` (1), `precision`,
`unmapped_keys` (must be empty) and `ignored_keys`. The worker validates its shape strictly
(`invalid_request`, field `conversion`) and requires every field given to equal the directory's
`conversion.json` (`artifact_changed`, field `conversion`, also when the directory has none or the
family is not `sdxl-base`). It is part of the bundle digest like the rest of `source`. A 1.0/1.1
bundle with `source.conversion` is `invalid_request`.

### Automatic settings and authorized fallbacks

A 1.2 load body (and `/v1/validate`) may use `options.precision: "auto"` (listed in
`capabilities.auto_options`) and `authorized_fallbacks`, a list of distinct values from
`capabilities.fallbacks` (`invalid_request`, field `authorized_fallbacks` otherwise). Before
loading the worker resolves:

- `auto`: `float16` on a CUDA device (reason names the float16 estimate against
  `--memory-budget-mb`, or the device's total memory when no budget is set), `float32` on a
  float32-only device.
- `lower_precision` (only if authorized): on an available CUDA device whose estimate at the
  resolved precision exceeds the budget, the next advertised lower-memory precision that fits.
- `cpu` (only if authorized): when the selected device is unavailable, or the estimate still
  exceeds the budget, load on the host CPU in float32.

Without an authorization nothing is checked or changed: the load behaves exactly as 0.3.0 (an
unavailable device is `503 backend_unavailable`; an over-budget load is attempted). Resolution is
deterministic for the same worker, device, bundle and body. Fallbacks never change the family,
bundle, adapters or any generation parameter and are never applied during an execution. A 1.2
validate response and load operation carry:

```json
{"resolved_options": {"precision": "float32", "attention": "sdpa", "offload": "none"},
 "resolutions": [{"option": "precision", "requested": "auto", "resolved": "float16",
                  "reason": "the selected device is unavailable; float16 is its lowest-memory advertised precision"}],
 "applied_fallbacks": [{"fallback": "cpu", "reason": "the selected device is unavailable",
                        "device": "cpu", "precision": "float32"}]}
```

`resolutions` lists only `auto` options (the precision that `auto` chose, before fallbacks);
`resolved_options` is what was loaded; the memory estimate is computed for it. `loaded_bundle`
keeps the requested `options` (so a supervisor's load-identity comparison is unchanged) and adds
`resolved_options` and `applied_fallbacks`; every subsequent 1.2 snapshot names
`applied_fallbacks`. After a CPU fallback `capabilities.ready` is true while `device_available`
stays false.


## Transport

Engine 0.5.0 adds network mode as response fields within minor 1.2; no request, event or error
changes, and a supervisor that ignores the new fields sees 0.4.0 behavior. The launch flags are:

| Flag                 | Meaning                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `--bind ADDRESS`     | IP address literal to listen on (default `127.0.0.1`); a host name is refused                 |
| `--tls-cert FILE`    | PEM server certificate chain, leaf first; not writable by group or others                     |
| `--tls-key FILE`     | PEM unencrypted private key; not readable or writable by group or others (mode 0600)           |
| `--tls-client-ca FILE` | PEM CA certificates client certificates must chain to; not writable by group or others      |
| `--no-parent-watch`  | Do not stop when inherited stdin closes; advertises `ownership_available: false`              |

Rules:

- A non-loopback `--bind` without all three TLS files is refused at launch (exit 2, a message on
  stderr naming the missing flags); partial TLS material is refused on any address. Nothing on a
  non-loopback address is ever served as plain HTTP, `/health` included. TLS on loopback is
  allowed.
- TLS 1.3 is the minimum. The client certificate is required and verified against
  `--tls-client-ca`; a client without one, with an untrusted one, or speaking plain HTTP or an
  older TLS version fails the handshake and receives no HTTP response. The bearer credential is
  still required on every endpoint except `/health`: the certificate authenticates the peer, the
  credential authorizes the session (`401 unauthorized` otherwise, exactly as on loopback).
- The certificate, key and CA files are read once at launch and never reloaded; rotation is a
  restart. Mode violations, unreadable files, a key that does not match the certificate or an
  encrypted key are refused at launch with a message naming the flag.
- SIGTERM handling is unchanged. The inherited-stdin watch (`MEDIA_ENGINE_SUPERVISED=1`) is
  skipped with `--no-parent-watch`. A remote supervisor cannot own the process; when
  `ownership_available` is `false` it must record a lost connection as interruption, not
  confirmed cancellation.

`/v1/info` and the startup record of a 1.2 worker carry:

```json
{"scheme": "https", "bind": "0.0.0.0",
 "certificate_sha256": "<64 hex: SHA-256 of the server certificate's DER>",
 "ownership_available": false}
```

On loopback HTTP they are `{"scheme": "http", "bind": "127.0.0.1", "certificate_sha256": null,
"ownership_available": true}`. `certificate_sha256` hashes the leaf certificate the handshake
presents, so a supervisor can pin it. The private work directory, catalog and credential stay on
the worker host. A worker pinned to 1.1 (the test fixture's `--protocol media-worker/1.1`) omits
these fields, as it omits `device_identity`. The conformance suite connects over `https` with
`MEDIA_ENGINE_TLS_CA`, `MEDIA_ENGINE_TLS_CERT` and `MEDIA_ENGINE_TLS_KEY` and asserts the same
sections either way.

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
`configuration`, `source`, and `bundle_digest`. The same bundle serves both tasks. The `esrgan`
family and the optional `format` field are 1.2 additions described above.

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
1.2 bodies may also use `precision:"auto"` and `authorized_fallbacks` (see above).
`POST /v1/model/unload` accepts `{protocol}`. Both return 202 and an `operation_id` immediately.
`GET /v1/operations/{operation_id}` reports running/succeeded/failed; a succeeded load carries the
estimate. A failed load leaves an empty worker; no substitute model is loaded. Extension: DELETE
on a terminal operation discards its record.

## Adapters

`capabilities.adapters` advertises `{formats:["peft-lora"], targets:[unet, text_encoder,
text_encoder_2], max_adapters:4, scale:[0,2], identity:"load"}`. A load body (and `/v1/validate`)
may carry `adapters: [{adapter, scale}]`, 1.1 only. An adapter record is
`{schema_version:1, adapter_id, artifact_handle, files, digest, source}` built like a bundle: the
handle names an approved directory in the launch catalog, `files` maps every relative file to its
SHA-256, exactly one top-level `.safetensors` file exists, and `digest` hashes the record without
itself. Validation reads the safetensors header and accepts only the PEFT layout
(`unet.<module>.lora_A.weight` / `lora_B.weight`, optional `alpha`, with `text_encoder.` and
`text_encoder_2.` prefixes); Kohya `lora_unet_*` and legacy processor layouts return `unsupported`
with field `adapter`. The response and the successful load operation report each adapter's layout
(targets, module count, ranks, parameters) and the memory estimate includes adapter parameters.

Adapters are load identity: `loaded_bundle.adapters` lists `{adapter_id, digest, scale, layout}`,
and a supervisor that wants a different set, scale or none at all issues a new load. Executions do
not name adapters. `/v1/validate` with `{protocol, adapters}` and no bundle validates adapters alone.

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
Tasks are `text-to-image`, `image-to-image` and (1.2) `upscale`, described above. Inputs are prompt (required), negative_prompt
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
`total_steps` is the effective count; never total progress. Upscale progress adds `unit:"tiles"`.

`POST /v1/executions/{execution_id}/cancel` acknowledges intent (202), or returns the immutable
terminal snapshot (200). Cooperative checkpoints stop encoding/sampling/decoding transitions and
every sampling step. A supervisor may kill only a process it owns; that is interruption, not
confirmed cancellation. Terminal states are succeeded/failed/cancelled.

Outputs include output_id (`image`, plus `image_2` … for a 1.2 `count`), image/png MIME, byte
size, SHA-256 and actual width/height. `GET /v1/executions/{execution_id}/outputs/{output_id}`
streams finalized bytes only. DELETE on a terminal
execution discards records/events/output after collection. Missing output bytes return 410.
Outputs survive client disconnects within the worker session, not process death.

Errors include code, readable message, retryable and optional field/restart_required. They never
contain raw exceptions, prompts, credentials or private paths. Busy returns 409; storage_full 503;
invalid requests 400; oversized uploads 413; missing handles 404; auth 401/403. OOM fails the
execution and marks the worker error/restart-required. A fresh session never implies an old
attempt succeeded.
