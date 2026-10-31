# media-worker/1.3 implementation contract

This document concretizes [engine-spec.md](../engine-spec.md). Engine 0.8.0 speaks
`media-worker/1.3` and still accepts `media-worker/1.0`, `media-worker/1.1` and
`media-worker/1.2` requests; `/v1/info`, `/v1/capabilities` and the startup record list
`supported_protocols`. Every 1.2 and 1.3 behavior is gated on the request's `protocol` minor: a
1.0/1.1 request sees exactly the 0.3.0 responses and a 1.2 request exactly the 0.5.0 responses
(response objects may gain fields, never lose or change them). A supervisor picks the highest minor both sides
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
(`tests/conformance`; `media-worker-conformance/1.0` in 0.4.0–0.5.0, `1.1` with the
image-to-video section from 0.6.0, `1.2` with the text-to-audio section from 0.7.0) passes
against it with no skipped section;
[compatibility.md](compatibility.md) records the result.

### Discovery in 1.2

The startup record is `{protocol, supported_protocols, highest_protocol, worker_version,
backend_id, session_id, port, pid, device_identity}`, plus the transport fields from Engine 0.5.0. Its `protocol` field names the launch-record
minor and stays `media-worker/1.1`: the record's shape only gained fields in 1.2, and Runtime 0.3.0
refuses a startup record whose `protocol` it does not know. Supervisors negotiate from
`supported_protocols` (identical in the record, `/v1/info` and `/v1/capabilities`); `/v1/info`
and `/v1/capabilities` report the highest minor served (`media-worker/1.3` from Engine 0.6.0),
as does `--probe`.

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

## What changed in 1.3

| Addition                                          | Meaning                                                                                         |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| task `image-to-video`, family `svd`               | Animates one uploaded PNG; reference backend only (`diffusers-sdxl`) in Engine 0.6.0            |
| `parameters.frames`, `fps`, `motion`, `noise_augmentation`, `decode_chunk` | Video parameters; `seed` and `steps` (1–50) are required, the rest have defaults |
| outputs `frames`, `poster`                        | `frames` is `application/zip` (lossless PNG per frame plus `frames.json`); `poster` is frame 1  |
| output record `frames`, `fps`, `duration_seconds` | On the `frames` output, read back from `frames.json`; `duration_seconds = frames / fps`         |
| stage `packaging`; progress `unit: "frames"`      | Frames are zipped after decoding; decoding progress counts frames                               |
| `limits.video_steps`, `.frames`, `.fps`, `.motion`, `.noise_augmentation`, `.decode_chunk`, `.video_sizes` | Ranges for the video parameters; `video_sizes` follows the loaded svd bundle (reference sizes otherwise) |
| `loaded_bundle.video_sizes`, `.default_frames`    | The loaded svd bundle's sizes (`video.json`, optional) and its unet's `num_frames`             |
| `task_parameters.image-to-video`                  | Inputs, parameters, defaults, `bundle_defaults`, outputs, `output_formats`, stages, archive shape |
| `memory_estimate.latents_mb`, `.per_frame_bytes`  | svd: the sampling latents of the most frames at the largest size                                |
| task `text-to-audio`, family `stable-audio-open` (0.7.0) | One clip from a prompt; reference backend only (`diffusers-sdxl`) in Engine 0.7.0          |
| `parameters.duration_seconds`, `guidance` (0.7.0) | Audio parameters; `seed`, `steps` (1–200) and `duration_seconds` are required, `guidance` defaults to 7 |
| output `audio` (0.7.0)                            | `audio/wav`, canonical 16-bit PCM; the record adds `sample_rate`, `channels`, `samples`, `duration_seconds` read back from the file |
| `limits.audio_steps`, `.duration_seconds`, `.sample_rate`, `.audio_channels` (0.7.0) | Ranges for the audio parameters and bundle facts; `duration_seconds` follows the loaded bundle |
| `loaded_bundle.sample_rate`, `.channels`, `.hop_length`, `.duration_range` (0.7.0) | The loaded stable-audio-open bundle's facts, read from its configs                   |
| `task_parameters.text-to-audio` (0.7.0)           | Inputs, parameters, `required_parameters`, defaults, outputs, `output_formats`, stages, `bundle_facts`, audio shape |
| `capabilities.missing_task_dependencies` (0.7.0)  | Per task, the missing distributions that keep it unadvertised (`{"text-to-audio": ["torchsde"]}` without the `inference-audio` extra); readiness still follows `missing_dependencies` |
| `memory_estimate.latents_mb`, `.waveform_mb` (0.7.0) | stable-audio-open: the sampling latents and the decoded waveform of the whole latent length |
| load options `precision: bfloat16`, `attention: flash`, `quantization`, `compile` (0.8.0) | Optimization values, accepted only when `capabilities.optimizations` lists them; `quantization` and `compile` (baseline `none`) are new optional fields of 1.3 load bodies only |
| `capabilities.optimizations`, `.unadvertised_optimizations` (0.8.0) | Each advertised optimization with its cited benchmark, and why every other candidate is not advertised |
| `memory_estimate.optimizations` (0.8.0)           | Per selected optimization, what `value_mb` does and does not account for                         |

Nothing else changed: no 1.0–1.2 request, event, error or output shape. A 1.2 (or older) caller
that names `image-to-video` or `text-to-audio` gets `unsupported`, field `task`; an `svd` or
`stable-audio-open` bundle on a 1.2 load or validation is `unsupported`, field `family` (a
1.0/1.1 body still fails on its fields first, as for `esrgan`). `tasks`, `model_families` and
`loaders` list what the worker serves, whatever the caller's minor, so a 1.2 supervisor may see
`image-to-video` or `text-to-audio` listed and must not request them; `defaults` is unchanged, so
a supervisor that merges it into every request never sends a video or audio field. Minor 1.3 was
first advertised by Engine 0.6.0, which did not serve text-to-audio; Engine 0.7.0 adds it without
a new minor, and Engine 0.8.0 adds the optimization options the same way. The `tasks` list, not
the minor, tells a caller what exists; `capabilities.optimizations`, not the minor, tells it which
optimizations it may select.

### Discovery in 1.3

The startup record still reports `protocol: "media-worker/1.1"` (see Discovery in 1.2) and
`highest_protocol: "media-worker/1.3"`. `/v1/capabilities` adds, on a reference backend with the
inference extra installed:

```json
{
  "tasks": ["text-to-image", "image-to-image", "upscale", "image-to-video"],
  "model_families": ["sdxl-base", "esrgan", "svd"],
  "loaders": [
    {"format": "diffusers-safetensors", "families": ["svd"],
     "roles": ["unet", "vae", "image_encoder", "feature_extractor", "scheduler"],
     "tasks": ["image-to-video"], "ownership": "reference"}
  ],
  "task_parameters": {
    "image-to-video": {"inputs": ["image"],
      "parameters": ["frames", "fps", "motion", "noise_augmentation", "decode_chunk"],
      "defaults": {"fps": 7, "motion": 127, "noise_augmentation": 0.02, "decode_chunk": 8},
      "bundle_defaults": ["frames"], "input_formats": ["image/png"],
      "outputs": ["frames", "poster"],
      "output_formats": {"frames": "application/zip", "poster": "image/png"},
      "families": ["svd"], "stages": ["encoding", "sampling", "decoding", "packaging", "saving"],
      "progress_units": {"decoding": "frames"},
      "archive": {"frame_format": "image/png", "frame_names": "000001.png",
                  "manifest": "frames.json", "compression": "stored"}}
  },
  "task_defaults": {"image-to-video": {"fps": 7, "motion": 127, "noise_augmentation": 0.02, "decode_chunk": 8}},
  "limits": {"video_steps": [1, 50], "frames": [1, 64], "fps": [1, 30], "motion": [1, 255],
             "noise_augmentation": [0, 1], "decode_chunk": [1, 16],
             "video_sizes": [[1024, 576], [576, 1024]]}
}
```

(Only the additions are shown; the text-to-audio additions from 0.7.0 are under
[Text-to-audio](#text-to-audio).) The owned backend (`owned-sdxl`) does not serve `svd`: it never
lists the family, its loader or the task, and a load or validation naming an `svd` bundle is
`unsupported`, field `family`, before any file is read. `capabilities.output_formats` keeps its
1.0 value; the per-task `output_formats` are authoritative. A worker pinned to 1.2 (the test
fixture's `--protocol media-worker/1.2`) reports exactly the 0.5.0 surface.

### The `svd` family

| Family | `format`                | `pipeline`       | Components (`role` → `tensor_format`)                                         | Tasks          |
| ------ | ----------------------- | ---------------- | ----------------------------------------------------------------------------- | -------------- |
| `svd`  | `diffusers-safetensors` | `image-to-video` | unet, vae, image_encoder → `safetensors`; feature_extractor, scheduler → `json` | image-to-video |

The directory is a Diffusers `StableVideoDiffusionPipeline` bundle: `model_index.json` names
exactly `unet` → `UNetSpatioTemporalConditionModel`, `vae` → `AutoencoderKLTemporalDecoder` (both
`diffusers`), `image_encoder` → `CLIPVisionModelWithProjection`, `feature_extractor` →
`CLIPImageProcessor` (both `transformers`) and `scheduler` → `EulerDiscreteScheduler`; anything
else is `unsupported` with the role or `pipeline` as field. Component digests cover each role's
folder, as for `sdxl-base`. An optional top-level `video.json`, `{"sizes": [[width, height], …]}`
(1–8 sizes, each side 64–2048 and a multiple of eight), declares the sizes the bundle serves;
without it the bundle serves the reference sizes, 1024×576 and 576×1024. The default frame count
is the unet configuration's `num_frames` (25 when absent, the library default) and must lie within
`limits.frames`. A `video.json` or default outside those rules is `unsupported`, field
`configuration`, at validation. `source.conversion` and adapters are refused as for `esrgan`.

A 1.3 validation and load report `family: "svd"`, `tasks: ["image-to-video"]`, `video_sizes` and
`default_frames`; `loaded_bundle` carries the same facts. While an `svd` bundle is loaded,
`capabilities.limits.video_sizes` is that bundle's list; otherwise it is the reference sizes (the
test fixture advertises its video bundle's sizes, 64×64, 1024×576 and 576×1024).

### Image-to-video

```json
{"protocol": "media-worker/1.3", "attempt_id": "attempt_v1", "bundle_digest": "sha256:<svd>",
 "task": "image-to-video", "inputs": {"image": "input_<32 hex>"},
 "parameters": {"width": 1024, "height": 576, "seed": 42, "steps": 25,
                "frames": 25, "fps": 7, "motion": 127, "noise_augmentation": 0.02, "decode_chunk": 8}}
```

`inputs` is exactly `{image}`; a `prompt` or `negative_prompt` is `invalid_request`, field
`prompt`. `parameters` requires `width`, `height`, `seed` (uint32) and `steps`
(`limits.video_steps`); `frames`, `fps`, `motion` (the motion bucket), `noise_augmentation` and
`decode_chunk` (frames per VAE decode) are optional with the defaults above and `frames` from the
bundle. Each is checked against its limit (`invalid_request` with the parameter as field; `fps`,
`motion`, `frames`, `decode_chunk` and `steps` are integers). There is no guidance parameter: the
reference pipeline's per-frame guidance ramp from 1.0 to 3.0 is fixed and reported as
`min_guidance`/`max_guidance` in `effective_settings`. Any other field is `invalid_request`.

With the bundle loaded, `[width, height]` must be one of `loaded_bundle.video_sizes`
(`invalid_request`, field `video_size`), and at submission it must equal the uploaded image
(`invalid_request`, field `width`); the input is never resized or cropped. The storage
reservation is one image-sized slot per frame plus one for the poster, so a result that cannot fit
is `503 storage_full` before sampling; frames are never dropped or re-timed to fit.
`effective_settings` is `{width, height, seed, steps, frames, fps, motion, noise_augmentation,
decode_chunk, min_guidance, max_guidance, effective_steps}` (`frames` resolved from the bundle
when omitted; on `/v1/validate` before the bundle is loaded it is absent).

Stages are `encoding` (the image conditioning: CLIP image embedding and VAE encoding of the
noised input), `sampling` (from the first UNet call) with `{stage: "sampling", scope: "stage",
completed_steps, total_steps}` per step, `decoding` with `{stage: "decoding", scope: "stage",
unit: "frames", completed_steps, total_steps: frames}` after each chunk of `decode_chunk` frames,
`packaging` while the archive is written, then `saving`. Cancellation is checked at every UNet
call and every step, before every decode chunk and before packaging; a cancelled execution
publishes nothing. `measurements` adds `frames` and `decode_chunks`.

The execution finalizes two outputs:

| Output   | MIME type         | Content                                                                                       |
| -------- | ----------------- | --------------------------------------------------------------------------------------------- |
| `frames` | `application/zip` | `frames.json` then `000001.png` … one 8-bit RGB PNG per frame, every member stored (not deflated), fixed timestamps |
| `poster` | `image/png`       | Frame 1, byte for byte equal to `000001.png`                                                  |

`frames.json` is `{"fps": 7, "width": 1024, "height": 576, "frames": 25}`. The worker reads it
back (and checks every member's name, storage and PNG header against it) before publishing, and
the `frames` output record carries `{output_id, mime_type, size, sha256, width, height, frames,
fps, duration_seconds}` from it; `poster` has the image fields. The same seed, settings, bundle
and software stack give the same archive bytes. Encoding a container (MP4 or otherwise) is the
caller's business: the worker ships no encoder and never looks for one on `PATH`.

The memory estimate of an `svd` bundle adds `latents_mb`: the sampling latents of
`limits.frames` frames at the bundle's largest size (`per_frame_bytes` each: the guided UNet
input, the guided image latents repeated per frame and the denoised latent), with a `frames`
scaling note. It excludes UNet attention and VAE decoder activations, which also grow with frames
and `decode_chunk`, and says so in `uncertainty`.

### Text-to-audio

Engine 0.7.0 adds text-to-audio within minor 1.3. `/v1/capabilities` on a reference backend with
the `inference` and `inference-audio` extras installed adds:

```json
{
  "tasks": ["text-to-image", "image-to-image", "upscale", "image-to-video", "text-to-audio"],
  "model_families": ["sdxl-base", "esrgan", "svd", "stable-audio-open"],
  "loaders": [
    {"format": "diffusers-safetensors", "families": ["stable-audio-open"],
     "roles": ["transformer", "vae", "text_encoder", "tokenizer", "projection_model", "scheduler"],
     "tasks": ["text-to-audio"], "ownership": "reference"}
  ],
  "task_parameters": {
    "text-to-audio": {"inputs": ["prompt", "negative_prompt"],
      "parameters": ["guidance", "duration_seconds"],
      "required_parameters": ["seed", "steps", "duration_seconds"],
      "defaults": {"guidance": 7}, "outputs": ["audio"],
      "output_formats": {"audio": "audio/wav"}, "families": ["stable-audio-open"],
      "stages": ["encoding", "sampling", "decoding", "saving"],
      "bundle_facts": ["sample_rate", "channels", "hop_length", "duration_range"],
      "audio": {"encoding": "pcm_s16le", "container": "wav",
                "duration": "rounded up to whole latent frames (hop_length samples) and reported"}}
  },
  "task_defaults": {"text-to-audio": {"guidance": 7}},
  "limits": {"audio_steps": [1, 200], "duration_seconds": [1, 47],
             "sample_rate": [1000, 192000], "audio_channels": [1, 2]},
  "missing_task_dependencies": {},
  "dependencies": {"torchsde": "0.2.6"}
}
```

(Only the additions are shown.) Without `torchsde` the backend stays ready for its other families
but lists neither the family, its loader nor the task, and `missing_task_dependencies` is
`{"text-to-audio": ["torchsde"]}`; `missing_dependencies` (which decides `ready` and whether a
load is accepted) is unchanged. The owned backend does not serve `stable-audio-open`, exactly as
for `svd`: no family, loader or task, and a load or validation naming such a bundle is
`unsupported`, field `family`, before any file is read.

#### The `stable-audio-open` family

| Family              | `format`                | `pipeline`      | Components (`role` → `tensor_format`)                                                       | Tasks         |
| ------------------- | ----------------------- | --------------- | ------------------------------------------------------------------------------------------- | ------------- |
| `stable-audio-open` | `diffusers-safetensors` | `text-to-audio` | transformer, vae, text_encoder, projection_model → `safetensors`; tokenizer, scheduler → `json` | text-to-audio |

The directory is a Diffusers `StableAudioPipeline` bundle: `model_index.json` names exactly
`transformer` → `StableAudioDiTModel`, `vae` → `AutoencoderOobleck`, `scheduler` →
`CosineDPMSolverMultistepScheduler` (all `diffusers`), `text_encoder` → `T5EncoderModel`,
`tokenizer` → `T5TokenizerFast` (both `transformers`) and `projection_model` →
`StableAudioProjectionModel` under the library entry `stable_audio` (as the library itself
writes it); the scheduler configuration must name the same scheduler. Anything else is
`unsupported` with the role or `pipeline` as field. The tokenizer folder must contain
`tokenizer.json` (`unsupported`, field `tokenizer`, otherwise), so the fast tokenizer loads without
sentencepiece. Component digests cover each role's folder, as for `sdxl-base`.

The bundle's facts come from its configurations, without loading weights: `sample_rate` is the
VAE's `sampling_rate` (1000–192000 Hz), `channels` its `audio_channels` (1–2), `hop_length` (the
latent granularity, samples per latent frame) the product of its `downsampling_ratios`, and the
longest clip it can produce is the transformer's `sample_size` × `hop_length` / `sample_rate`
seconds (the library defaults, which are the reference bundle's values, where a configuration is
silent: 44100 Hz stereo, 2048 samples per frame, 1024 frames, 47.55 s). An optional top-level
`audio.json`, `{"duration_seconds": [low, high]}`, declares the range the bundle serves, with `0 <
low ≤ high` and `high` within that longest clip; without it the range is 1 s to the whole seconds
the bundle allows (`[1, 47]` for the reference bundle). Anything outside these rules is
`unsupported`, field `configuration`, at validation. `source.conversion` and adapters are
refused as for `esrgan`.

A 1.3 validation and load report `family: "stable-audio-open"`, `tasks: ["text-to-audio"]`,
`sample_rate`, `channels`, `hop_length` and `duration_range`; `loaded_bundle` carries the same
facts. While such a bundle is loaded, `capabilities.limits.duration_seconds` is its range;
otherwise it is the reference range, 1–47 s.

#### The request

```json
{"protocol": "media-worker/1.3", "attempt_id": "attempt_a1", "bundle_digest": "sha256:<audio>",
 "task": "text-to-audio", "inputs": {"prompt": "rain on a tin roof", "negative_prompt": "low quality"},
 "parameters": {"seed": 42, "steps": 100, "guidance": 7, "duration_seconds": 10}}
```

`inputs` is `{prompt}` with an optional `negative_prompt` (10,000 characters each; an `image` is
`invalid_request`, field `image`). `parameters` requires `seed` (uint32), `steps`
(`limits.audio_steps`, integer) and `duration_seconds` (a number, `0 < d ≤ 600` before a bundle
is loaded); `guidance` (0–30) defaults to 7. `width`, `height`, `sampler`, `scheduler` and `count`
are `invalid_request` with that field (the family has one scheduler and no image size); any other
field is `invalid_request`. With the bundle loaded, `duration_seconds` must lie within
`loaded_bundle.duration_range` (`invalid_request`, field `duration_seconds`): a longer request is
refused, never trimmed to the limit.

The requested duration is honored up to the model's latent granularity: the clip is the smallest
whole number of latent frames that covers it, `latent_frames = ceil(duration_seconds ×
sample_rate / hop_length)` (computed on the number as written, so 1.0 is not a frame longer
through binary rounding), and the effective duration is `latent_frames × hop_length /
sample_rate`, never shorter than the request. The model is conditioned on that effective
duration. With the reference bundle a 10 s request is 216 frames, 442,368 samples, 10.031 s.
`effective_settings` is `{seed, steps, guidance, duration_seconds, effective_steps}` plus, once
the bundle is loaded, `{sample_rate, channels, latent_frames, effective_duration_seconds}`. The
storage reservation is the WAV's exact size (44 header bytes plus 2 bytes per sample per channel),
so a clip that cannot fit is `503 storage_full` before sampling.

Stages are `encoding` (T5 prompt and timing conditioning), `sampling` (from the first
transformer call) with `{stage: "sampling", scope: "stage", completed_steps, total_steps}` per
step, `decoding` (the VAE decodes the transformer's whole latent length, as the library does, and
the waveform is cut at the latent boundary that covers the request) and `saving`. Cancellation is
checked at every transformer call and every step, after sampling and before saving; a cancelled
execution publishes nothing. `measurements` adds `latent_frames`, `samples` and `clipped_samples`
(samples beyond full scale before quantization, which are clamped).

The execution finalizes one output:

| Output  | MIME type   | Content                                                                                                     |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| `audio` | `audio/wav` | A 44-byte RIFF/WAVE header (`fmt ` PCM, 16 bits) then one `data` chunk of interleaved little-endian samples |

Samples are the decoder's output clamped to [-1, 1], scaled by 32767 and rounded to nearest. The
worker reads the header back (and checks that the file holds exactly the frames it declares)
before publishing, and the output record carries `{output_id, mime_type, size, sha256,
sample_rate, channels, samples, duration_seconds}` from it. The same seed, settings, bundle and
software stack give the same bytes. The WAV is written by the standard library; there is no
encoder, resampler or external tool.

The memory estimate of a `stable-audio-open` bundle adds `latents_mb` (per latent frame the guided
transformer input and output and the latent, over the whole latent length) and `waveform_mb` (the
decoded waveform before it is cut). Both are constant per bundle because the library samples and
decodes the whole latent length whatever the duration; transformer attention, T5 encoder and VAE
decoder activations are excluded and stated in `uncertainty`.


### Optimizations

Engine 0.8.0 adds four optimization candidates as explicit load-option values. None is ever
applied implicitly: a caller selects one in the load body, it becomes part of the load identity
(`loaded_bundle.options`, and the operation's `resolved_options`), and switching it is a fresh
load, exactly like a precision change.

| Option         | Value              | Families                                | Effect |
| -------------- | ------------------ | --------------------------------------- | ------ |
| `precision`    | `bfloat16`         | sdxl-base, esrgan, stable-audio-open    | Every component held and run in bfloat16; the SDXL VAE is not upcast |
| `attention`    | `flash`            | sdxl-base, svd, stable-audio-open       | SDPA restricted to the FlashAttention kernel for each execution; needs `precision` float16 or bfloat16 |
| `quantization` | `int8-weight-only` | sdxl-base, svd, stable-audio-open       | The denoiser's linear weights stored as int8 with per-channel scales; not combined with adapters |
| `compile`      | `inductor`         | all                                     | The denoiser's forward compiled with `torch.compile(backend="inductor")`; the first execution of each shape compiles |

**Advertising.** A candidate is listed in `capabilities.optimizations` only when the code that
applies it is installed, the selected device can run it, and the release's benchmark index cites
a benchmark for this backend and device class. Each entry carries the citation:

```json
{"option": "quantization", "value": "int8-weight-only", "families": ["sdxl-base"],
 "device_class": {"kind": "cuda", "name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9"},
 "benchmark": "docs/benchmarks/<date>-<device>.json", "benchmark_sha256": "<64 hex>",
 "improvement": "…", "determinism": "…", "quality_note": "…",
 "effect": "…", "adapters": false}
```

(`precisions` is added when the entry requires one, as flash does.) Every other candidate is
listed in `capabilities.unadvertised_optimizations` as `{option, value, reason}` with `reason`
one of `missing_dependency` (plus `missing`, the distribution names), `device_unsupported` or
`no_benchmark`. **Engine 0.8.0 ships no benchmark entry**, so both backends advertise
`optimizations: []` on every device; see the compatibility record for the checklist that adds
the first one.

**Requests.** A 1.3 load body's `options` may hold `quantization` and `compile` (both default to
`none`, which is always accepted) and the values above. Refusals, checked before any file is read:

| Body                                                              | Error                                    |
| ----------------------------------------------------------------- | ---------------------------------------- |
| a value not in `capabilities.optimizations`, or not for the bundle's family | `unsupported`, field `options`  |
| `attention: flash` with a precision it does not list (including `auto`) | `unsupported`, field `options`     |
| `quantization: int8-weight-only` with adapters                     | `unsupported`, field `adapters`          |
| any optimization with `authorized_fallbacks` containing `cpu`      | `unsupported`, field `authorized_fallbacks` (a benchmark is valid for one device class) |
| a 1.0–1.2 body with `quantization` or `compile`                    | `invalid_request` (unknown field, as in 0.7.0) |
| a 1.0–1.2 body with `bfloat16` or `flash`                          | `unsupported`, field `options` (as in 0.7.0) |

`auto` never resolves to an optimized value (`auto_options` is unchanged and `precisions` does not
list bfloat16), and `lower_precision` never falls back to one. With an optimization selected,
validation and load results add `memory_estimate.optimizations`: `[{option, value, effect}]`
saying what `value_mb` does not count (int8 weights are still counted unquantized, compiled-graph
memory is excluded). Events, progress, cancellation checkpoints and outputs are unchanged; the
compiled forward leaves the cancellation hooks eager.

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
Tasks are `text-to-image`, `image-to-image`, (1.2) `upscale` and (1.3) `image-to-video` and `text-to-audio`, described above. Inputs are prompt (required), negative_prompt
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
`total_steps` is the effective count; never total progress. Upscale progress adds `unit:"tiles"`;
image-to-video decoding progress has `stage:"decoding"` and `unit:"frames"`.

`POST /v1/executions/{execution_id}/cancel` acknowledges intent (202), or returns the immutable
terminal snapshot (200). Cooperative checkpoints stop encoding/sampling/decoding transitions and
every sampling step. A supervisor may kill only a process it owns; that is interruption, not
confirmed cancellation. Terminal states are succeeded/failed/cancelled.

Outputs include output_id (`image`, plus `image_2` … for a 1.2 `count`; `frames` and `poster`
for 1.3 image-to-video; `audio` for 1.3 text-to-audio), MIME type (image/png, application/zip for
`frames`, audio/wav for `audio`), byte size, SHA-256 and actual width/height (plus `frames`, `fps`
and `duration_seconds` on `frames`; `sample_rate`, `channels`, `samples` and `duration_seconds`
instead on `audio`). `GET /v1/executions/{execution_id}/outputs/{output_id}`
streams finalized bytes only. DELETE on a terminal
execution discards records/events/output after collection. Missing output bytes return 410.
Outputs survive client disconnects within the worker session, not process death.

Errors include code, readable message, retryable and optional field/restart_required. They never
contain raw exceptions, prompts, credentials or private paths. Busy returns 409; storage_full 503;
invalid requests 400; oversized uploads 413; missing handles 404; auth 401/403. OOM fails the
execution and marks the worker error/restart-required. A fresh session never implies an old
attempt succeeded.
