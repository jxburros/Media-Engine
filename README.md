# Media Engine

An independently installed, offline media inference worker for [Media Runtime](https://github.com/jxburros/Media-Runtime)
or another supervisor. Engine 1.1.0's Python workers serve `media-worker/1.0`–`1.4`; the
standalone native worker serves 1.0–1.3. Requests use authenticated loopback HTTP or mutual TLS
in explicit network mode. Every process has one device, one loaded identity and one execution
slot, with genuine progress, cooperative cancellation, resumable SSE and temporary verified outputs.

Three backend IDs remain stable:

- **Reference** (`diffusers-sdxl`): complete Diffusers pipelines, including SDXL, ESRGAN, SVD,
  Stable Audio Open, FLUX.1-schnell and SD3.5 Medium when their dependencies are available.
- **Owned Python** (`owned-sdxl`): owned SDXL denoising loop and ESRGAN network; its supported
  tasks, samplers and conditioning are advertised independently from the reference backend.
- **Owned native** (`native-sdxl`, `ownership: engine`): a separate Rust/candle executable in
  [native/](native/README.md), with SDXL text-to-image/image-to-image, batches and LoRA, no Python
  dependency, and optional mutual TLS. Additional native families are not advertised.

Protocol 1.4 adds capability-gated sampler selection, inpainting, explicit pixel operations,
conditioning inputs, nonfinal SDXL previews, optional pinned MP4 encoding and explicit reference
CPU-offload modes. The lossless video frame archive and PCM WAV remain baseline outputs.
Nothing silently resizes an image, substitutes a model or downloads weights during serving.
[engine-spec.md](engine-spec.md) describes the current scope; [the protocol](docs/protocol.md)
defines request shapes and compatibility gates.

CPU random-weight tests and native conformance establish executable behavior, not real-weight
quality or GPU performance. RTX 4090 acceptance jobs are prepared. The benchmark index remains
empty, so benchmark-gated optimization candidates are not advertised as measured improvements.

## Install

Reference environment: Linux x64, Python **3.12.14**, NVIDIA/CUDA, PyTorch **2.6.0**,
Diffusers **0.32.2**. `uv.lock` pins transitive artifacts/checksums separately from Runtime/Studio.
Installation is an explicit network action; serving never downloads models or executes remote code.

```sh
uv sync --frozen --extra inference          # both backends (reference set)
uv sync --frozen --extra inference-owned    # owned-sdxl and the upscaler only: no accelerate, no peft
uv sync --frozen --extra inference --extra inference-optimize   # adds torchao for int8 weight-only
uv sync --frozen --extra inference --extra inference-audio   # plus text-to-audio (torchsde)
.venv/bin/media-engine --version
.venv/bin/media-engine --probe
```

The `inference-owned` extra pins torch, diffusers (model classes only), transformers,
safetensors and numpy; the owned backend never imports Pillow, numpy, accelerate or peft itself
(PNG is written by the standard-library `png.py`). `capabilities.dependencies` reports what the
launched backend imports.

The source package without the inference extra can run the protocol server and report missing
dependencies; it cannot generate. It never substitutes simulated output. Test backends live outside
the installed package in `tests/` and identify themselves in snapshots/provenance.

Supply a complete local `stabilityai/stable-diffusion-xl-base-1.0` Diffusers-format bundle,
including UNet, VAE, both CLIP encoders and tokenizers, pipeline config and Euler scheduler config.
Use safetensors weights only; Python, pickle weights and unsafe symlinks are rejected. Immutable
Hugging Face snapshot-to-blob links are accepted only inside their own repository cache tree. The service does
not fetch weights, convert checkpoints during a request (conversion is the separate offline
`convert` command below) or select a model for callers. Each local deployment is
identified by SHA-256 of every registered file; optional upstream revision/license metadata belongs
to the registration. Keep approved files immutable/read-only while the worker serves them.

Copy `examples/catalog.json`, replace the absolute model paths and keep the same handles (`sdxl-base`, `upscaler`, `video`, `audio`)
in Runtime's artifact catalog. The **operator** supplies these paths at launch. Load requests carry
only approved opaque artifact handles, component digests and relative file identities.

Set `MEDIA_ENGINE_TOKEN` to a separately generated random credential of at least 32 characters,
through a protected service environment file (never a command-line argument). Then:

```sh
.venv/bin/media-engine --catalog /absolute/catalog.json --directory /absolute/worker-data --device cuda:0 --port 7811
```

Add `--backend owned-sdxl` to run the owned pipeline instead of the reference Diffusers pipelines.
The backend is part of the worker's identity; a supervisor's request must name the launched
backend explicitly, and nothing reinterprets a request when the backend changes. Both Python backends accept the shared SDXL bundle contract; their capabilities explicitly
differ for tasks, samplers, conditioning and load options. They do not promise identical image bytes.

`--port 0` chooses a free loopback port. Stdout emits one JSON startup record with protocol,
version, random session ID, port and PID; the supervisor must still check `/v1/info` and
`/v1/capabilities`. All endpoints except coarse `/health` require the per-launch bearer credential.
Browser origins are rejected. `--device cpu` explicitly selects CPU/float32; it is never a fallback.

LoRA adapters are registered like bundles: an approved catalog directory holding one PEFT-layout
safetensors file, hashed and named by an opaque handle. Up to four adapters with scales travel on
the load request and form part of the loaded identity. The reference backend applies them through
PEFT; the owned pipeline merges the low-rank deltas into the weights itself. Kohya-format files must
be converted first; the worker refuses layouts it cannot verify rather than guessing.

A 1.2 text-to-image request may set `count` (1–4): the execution finalizes `image`, `image_2` …
with seeds `seed`, `seed + 1` …, so any image can be reproduced alone. Each output is downloaded
from `/v1/executions/{id}/outputs/{output_id}`; the execution succeeds only when all of them exist.

Image-to-image uploads a PNG to `POST /v1/inputs` first and references the returned handle; the
worker never reads caller paths or URLs. The request must state the image's real size, and
`strength` sets the effective step count (`floor(steps × strength)`, at least one).

The default GPU options are float16, SDPA attention and no offloading. The worker only advertises
options it applies. The reference hardware target is an RTX 4090 with 24 GiB, Linux x64 and a driver
compatible with the locked CUDA PyTorch distribution. This target has not been hardware-certified
by the automated CPU tests. Real-weight/CUDA acceptance requires the prepared hardware jobs.

`--memory-budget-mb` sets PyTorch's per-process allocator budget for the selected GPU. This does
not coordinate or terminate other GPU services. Validating or loading a bundle reports a memory
estimate read from the safetensors headers (parameter bytes at the load precision, per role); it
states its basis and that activations, the VAE upcast and framework overhead are excluded.
Execution metadata reports actual peak allocated bytes where measurable. An OOM requires a fresh
worker session; no smaller image or alternate precision is retried automatically.

## Upscaling

An `esrgan` bundle is one approved directory holding exactly one safetensors weights file and a
`config.json` such as `{"architecture": "rrdbnet", "scale": 4, "in_channels": 3, "out_channels": 3,
"num_feat": 64, "num_block": 23, "num_grow_ch": 32}` (Real-ESRGAN x4plus; x2plus is `scale: 2`).
It is registered with `format: "single-file-safetensors"`, `family: "esrgan"`, `pipeline:
"upscale"` and two components, `network` and `configuration` (see [the protocol](docs/protocol.md#families-and-loaders)).
The weights header must match the configured architecture exactly. An `upscale` execution names an
uploaded PNG and its real size, optionally `tile` (256/512/1024, 0 = whole image) and
`tile_overlap`; the output is exactly input × scale, progress counts tiles, and a result over
`limits.output_side` is refused before any work. Tiling is never chosen for you.

Real-ESRGAN releases are `.pth` pickles, which serving refuses. Use the explicit isolated
`media-engine convert --kind esrgan-pth` workflow described under
[offline weight migration](#offline-weight-migration) before cataloguing the safetensors output.

## Image-to-video

A 1.3 `image-to-video` execution animates one uploaded PNG with an `svd` bundle (Stable Video
Diffusion: spatio-temporal UNet, temporal-decoder VAE, CLIP image encoder, Euler scheduler) on
the reference backend (`diffusers-sdxl`; the owned backend refuses the family). Nothing beyond
the `inference` extra is needed: frames are written by the standard-library PNG writer and packed
with `zipfile`.

Supply a complete local Diffusers-format `stabilityai/stable-video-diffusion-img2vid-xt` (25
frames) or `stable-video-diffusion-img2vid` (14 frames) bundle with safetensors weights:
`model_index.json`, `unet/`, `vae/`, `image_encoder/`, `feature_extractor/` and `scheduler/`.
Keep only the weights you serve in each folder (the worker hashes every file and the library
loads the non-variant `*.safetensors`). Register it with `family: "svd"`, `pipeline:
"image-to-video"`, `format: "diffusers-safetensors"` and those five roles. The bundle serves the
reference sizes 1024×576 and 576×1024; an operator may add a top-level `video.json`,
`{"sizes": [[1024, 576], [576, 1024]]}`, to declare other sizes the weights were trained for.
While it is loaded, `capabilities.limits.video_sizes` and `loaded_bundle.video_sizes` list them.

The request names the uploaded image and its real size (one of the bundle's sizes; nothing is
resized), `seed`, `steps` (1–50) and optionally `frames` (1–64, default the bundle's
`num_frames`), `fps` (1–30, default 7), `motion` (motion bucket 1–255, default 127),
`noise_augmentation` (0–1, default 0.02) and `decode_chunk` (frames per VAE decode, 1–16, default
8; lower it to save memory). Two outputs are finalized: `frames`, an `application/zip` with
`frames.json` (`fps`, `width`, `height`, `frames`) and `000001.png` … stored losslessly, and
`poster`, frame 1 as a PNG. The `frames` record carries `frames`, `fps` and `duration_seconds`.
Progress counts sampling steps, then decoded frames; cancellation is honored at every step and
decode chunk. Encode a container yourself from the archive, for example
`ffmpeg -framerate 7 -i %06d.png -c:v libx264 -pix_fmt yuv420p out.mp4` after unzipping; the
worker ships no encoder and never runs one.

The Stable Video Diffusion weights are licensed by Stability AI under their own terms (the
Stability AI Community License for the current releases; check the model card of the revision
you install), separately from this MIT-licensed worker. The operator obtains and approves them;
the worker never downloads them.

## Text-to-audio

A 1.3 `text-to-audio` execution (Engine 0.7.0) generates one clip from a prompt with a
`stable-audio-open` bundle (diffusion transformer, Oobleck VAE, T5 text encoder and tokenizer,
timing projection, cosine DPM-Solver scheduler) on the reference backend (`diffusers-sdxl`; the
owned backend refuses the family). It needs the `inference-audio` extra besides `inference`: the
family's scheduler imports `torchsde`. Without it the worker still serves everything else and
reports `missing_task_dependencies: {"text-to-audio": ["torchsde"]}`. The WAV is written by the
standard-library `wave` module; there is no audio encoder or resampler.

Supply a complete local Diffusers-format `stabilityai/stable-audio-open-1.0` bundle with
safetensors weights: `model_index.json`, `transformer/`, `vae/`, `text_encoder/`, `tokenizer/`,
`projection_model/` and `scheduler/`. If your copy also holds the single-file checkpoint and
`model_config.json` of another loader at the top level, remove them: the worker hashes every file
in the directory and the bundle is the Diffusers layout only. Keep only the weights you serve in
each folder. `tokenizer/` must contain `tokenizer.json`
so the fast T5 tokenizer loads without sentencepiece; if your copy lacks it, write it once offline
with `T5TokenizerFast.from_pretrained(...).save_pretrained(...)` in an environment that has
sentencepiece. Register it with `family: "stable-audio-open"`, `pipeline: "text-to-audio"`,
`format: "diffusers-safetensors"` and those six roles. Sample rate (44.1 kHz) and channels
(stereo) come from the VAE config; the longest clip is the transformer's latent length times the
VAE's 2048-sample latent frame, 47.55 s, so the bundle serves 1–47 s. An operator may add a
top-level `audio.json`, `{"duration_seconds": [1, 30]}`, to serve a narrower range. While it is
loaded, `capabilities.limits.duration_seconds` and `loaded_bundle.duration_range` report it.

The request carries `prompt`, optionally `negative_prompt`, and `seed`, `steps` (1–200),
`duration_seconds` (within the range) and optionally `guidance` (0–30, default 7). A duration
past the range is refused, never trimmed. The clip is whole latent frames: a request is rounded
up to the next multiple of 2048 samples (46.4 ms at 44.1 kHz; 10 s becomes 10.031 s) and the
model is conditioned on that effective duration, which `effective_settings` and the output record
report. One output, `audio`, is finalized as `audio/wav`, 16-bit PCM; its record carries
`sample_rate`, `channels`, `samples` and `duration_seconds` read back from the file. Progress
counts sampling steps; cancellation is honored at every step.

The Stable Audio Open weights are licensed by Stability AI under the Stability AI Community
License (check the model card of the revision you install; the repository is gated); its T5
text encoder derives from `google-t5/t5-base` (Apache-2.0). Both are separate from this
MIT-licensed worker. The operator obtains and approves them; the worker never downloads them.

## Converting single-file checkpoints

A single-file SDXL `.safetensors` checkpoint is not loadable as a bundle; convert it once, offline:

```sh
.venv/bin/media-engine convert --input /models/my-sdxl.safetensors --reference-handle sdxl-base \
  --catalog /absolute/catalog.json --output /models/my-sdxl-diffusers [--precision float16]
```

The converter refuses anything that is not a safetensors container, takes the architecture,
tokenizers and scheduler from the reference bundle's handle (nothing is downloaded), maps the keys
with the Diffusers single-file helpers, checks every tensor against the reference model classes,
and writes safetensors components plus `conversion.json` (source SHA-256, reference digest,
converter versions, mapped key counts). Any key it cannot map fails the conversion by name and the
partial output is removed. Approve and catalog the new directory like any other bundle; a 1.2
registration may repeat the provenance as `source.conversion`, which the worker checks against
`conversion.json`.

## Automatic settings, fallbacks and device identity

On 1.2 loads `precision` may be `auto` (float16 on CUDA, float32 on CPU; the reason names the
estimate and budget) and `authorized_fallbacks` may list `cpu` (load on the host CPU in float32
when the device is unavailable or over budget) and `lower_precision` (float16 when float32 is over
`--memory-budget-mb`). Nothing happens without that authorization: an unavailable device is still
refused and no precision changes. Applied fallbacks are reported on the load operation,
`loaded_bundle` and every later snapshot, and never alter generation parameters. `/v1/info` and the
startup record carry `device_identity` (device, index, name, total memory, a SHA-256 host ID from
`/etc/machine-id` or the hostname, and pid) so a supervisor pool can tell workers apart.

## Optimization options

A 1.3 load may select `precision: bfloat16`, `attention: flash`, `quantization:
int8-weight-only` or `compile: inductor`, but only values listed in
`capabilities.optimizations`, each of which cites the benchmark that justified it (improvement,
determinism, quality note). The rest are listed in `capabilities.unadvertised_optimizations` with
the reason (`missing_dependency`, `device_unsupported`, `no_benchmark`). An optimization is part
of the load identity, is never chosen by `auto` and cannot be combined with the `cpu` fallback.
int8 weight-only needs the `inference-optimize` extra (`torchao`); the others need only torch (and
a C++ compiler on CPU, or triton on CUDA, for inductor). Details: [protocol](docs/protocol.md#optimizations).

**The current benchmark index has no entries**, so these candidates are not advertised on any device. To produce the evidence:

```sh
uv sync --frozen --extra inference --extra inference-optimize
python tools/benchmark.py --backend diffusers-sdxl --device cuda:0 --catalog catalog.json \
  --registration sdxl.json --precision float16 --option precision=bfloat16 \
  --option attention=flash --option quantization=int8-weight-only --option compile=inductor
```

The tool loads each option set in-process, records load, first-execution (compilation) and
steady-state times, peak device memory, output digests and per-pixel or per-sample differences
from the baseline in `docs/benchmarks/<date>-<device class>.json`, and prints candidate index
entries for the sets that qualify. A release cites an entry by adding it, with the file's SHA-256,
to `src/media_engine/optimizations.json` (see the checklist in
[docs/compatibility.md](docs/compatibility.md)).

## Network mode

By default the worker listens on `127.0.0.1` over plain HTTP. For a supervisor on another host,
bind a non-loopback address and give all three TLS files:

```sh
.venv/bin/media-engine --catalog /absolute/catalog.json --directory /absolute/worker-data \
  --bind 0.0.0.0 --port 7811 --no-parent-watch \
  --tls-cert /etc/media-engine/server.pem --tls-key /etc/media-engine/server.key \
  --tls-client-ca /etc/media-engine/clients-ca.pem
```

The worker refuses to start on a non-loopback `--bind` (an IP address, not a host name) unless
`--tls-cert`, `--tls-key` and `--tls-client-ca` are all present, and it never serves plain HTTP
there, `/health` included. Transport is TLS 1.3 only; every client must present a certificate that
chains to `--tls-client-ca`, or the handshake fails before any HTTP is read. The bearer credential
is still required on every endpoint except `/health`: the certificate authenticates the peer, the
credential authorizes the session. TLS on loopback is accepted (useful for tests) but adds nothing
over loopback HTTP. The startup record and `/v1/info` report `scheme` (`http`/`https`), `bind`,
`certificate_sha256` (SHA-256 of the server certificate's DER, for pinning; `null` without TLS)
and `ownership_available`.

A remote supervisor cannot own the process: SIGTERM, stdin-close and forced termination remain
local to the worker host. `--no-parent-watch` disables the inherited-stdin shutdown for
service-manager launches (systemd and similar) and makes the worker advertise
`ownership_available: false`, so the supervisor records a lost connection as interruption, never
as confirmed cancellation. SIGTERM handling is unchanged.

The operator owns the certificates; the worker never generates, fetches, rotates or reloads them.

- Issue a server certificate whose subject alternative names cover every address or name remote
  supervisors connect to, and a client certificate per supervisor from a CA you control (the
  client CA need not be the server's CA). Keep CA private keys off the worker host.
- `--tls-key` must be an unencrypted PEM key that is not readable by group or others
  (`chmod 600`); `--tls-cert` and `--tls-client-ca` must not be writable by group or others
  (`chmod 644`). The worker refuses other modes with a message naming the flag.
- The files are read once at start. To rotate, replace them and restart the worker; supervisors
  pinning `certificate_sha256` must learn the new fingerprint.
- The catalog, work directory and credential stay on the worker host; deliver the per-launch
  credential to the remote supervisor over a channel you already trust.
- Revocation lists are not consulted: remove a client by issuing from a new client CA and
  restarting.

Minimal OpenSSL 3 commands (EC P-256, one year; adjust names, addresses and lifetimes):

```sh
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 365 \
  -subj "/CN=media clients CA" -keyout clients-ca.key -out clients-ca.pem
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 365 \
  -subj "/CN=gpu-host-1" -addext "subjectAltName=DNS:gpu-host-1.example,IP:192.0.2.10" \
  -keyout server.key -out server.pem
openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
  -subj "/CN=runtime-1" -keyout client.key -out client.csr
openssl x509 -req -in client.csr -CA clients-ca.pem -CAkey clients-ca.key -days 365 \
  -extfile <(printf "extendedKeyUsage=clientAuth") -out client.pem
chmod 600 server.key client.key clients-ca.key
openssl x509 -in server.pem -outform DER | sha256sum   # equals certificate_sha256
```

The supervisor then trusts `server.pem` (self-signed here, or your server CA) and presents
`client.pem`/`client.key`. `tests/certs` holds test-only material of this shape.

## Lifetime and collection

The worker has no waiting queue. Loading, unloading and inference cannot overlap. An identical
attempt retry returns its existing execution; conflicting reuse fails. A busy worker refuses
admission. The pipeline remains loaded until explicit unload or process shutdown.

Every execution, event, uploaded input and output is retained until explicit discard or session
shutdown. Limits (default 128 records, 1024 MiB temporary storage, 32 MiB per input) reject new
work rather than evicting uncollected results. Successful output files are finalized before publication. Runtime must copy
and verify them, commit its own durable manifest, then discard the execution. Control operation
records have an explicit discard extension too. See [the wire contract](docs/protocol.md).

SIGTERM stops admission and requests cooperative cancellation with bounded cleanup. A supervised
worker (`MEDIA_ENGINE_SUPERVISED=1`) also stops when its inherited stdin closes, unless launched
with `--no-parent-watch`. Cancellation acknowledgement is intent only;
`cancelled` is terminal only after computation exits. If completion wins the race, it stays successful.
Standalone callers own their persistent jobs and storage; this service is not a permanent library.

## Development

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m tests.conformance.run --registration ROOT/conformance.json \
  --launch python tests/serve_fixture.py ROOT
```

The conformance suite (`tests/conformance`, `media-worker-conformance/1.3`) runs against any
worker: `--url URL` with the credential in `MEDIA_ENGINE_TOKEN`, or `--launch CMD…` to start it
and check its startup record too. It reports each section and says "conformant" only when none
failed or was skipped; a backend is advertised only after it passes. Results are kept in
[docs/compatibility.md](docs/compatibility.md). `tests/serve_fixture.py ROOT` (options in its
docstring) is the hardware-free worker supervisors' integration tests launch; it takes the same
network-mode flags as the worker. Against a network-mode worker pass `--tls-ca`, `--tls-cert` and
`--tls-key` (or `MEDIA_ENGINE_TLS_CA`, `_CERT`, `_KEY`) for mutual TLS; `tests/test_network.py`
runs the suite over TLS with the committed test certificates in `tests/certs`.

`tests/test_worker.py` exercises protocol/lifecycle behavior without GPU downloads.
`tests/test_inference.py` runs real CPU inference through both backends on a tiny random-weight
SDXL-shaped bundle and a tiny RRDBNet, and image-to-video through the reference backend on a tiny
random-weight svd bundle (64×64, two frames, two steps), and text-to-audio through the reference
backend on a tiny random-weight stable-audio-open bundle (1 kHz stereo, one second, two steps;
skipped without `torchsde`), and every CPU-capable optimization through both backends with a
test-only benchmark index (int8 weight-only skipped without `torchao`, inductor without a C++
compiler); `tests/test_convert.py` converts a tiny single-file
checkpoint built from it. They are skipped unless the inference extra is installed (`uv sync --frozen --extra
inference` then `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`). It proves the
owned loop and the reference pipeline execute the same contract and, for the same seed and settings,
produce pixel-identical text-to-image output and image-to-image output within one 8-bit level on that
CPU stack, and that LoRA adapters change the output identically on both backends; it says nothing
about image quality. Run the Runtime
repository's `go test ./...` with both repositories beside each other for cross-process durable-job tests.
The native engine has its own tests: `cargo test --workspace --manifest-path native/Cargo.toml`, and
`MEDIA_ENGINE_NATIVE=native/target/release/media-engine-native` makes `tests/test_native.py` run the
conformance suite against the binary and compare it with `owned-sdxl` on the tiny bundle.
Machine-readable request definitions live in [schemas/worker.schema.json](schemas/worker.schema.json).
Engine 1.1.0 Python workers speak `media-worker/1.4` and retain the 1.0–1.3 surfaces for older callers.
The native worker's highest minor is 1.3. Runtime 1.1.0 negotiates either;
its startup record keeps `protocol: "media-worker/1.1"` for those launchers and reports `highest_protocol`.
`tests/serve_fixture.py --protocol media-worker/1.2` (or `1.1`) pins the fixture to an older surface.

To smoke-test this worker together with Media Runtime on your own local models (any backend,
including `media-engine-native`), use Runtime's
[testbench](https://github.com/jxburros/Media-Runtime/tree/main/testbench): one config file listing
your model directories, one command, and an HTML report of every feature with its outputs.

Source is MIT licensed. Dependency licenses and model-weight license terms are separate;
SDXL base weights carry CreativeML Open RAIL++-M terms, and Stable Video Diffusion and Stable Audio
Open weights carry Stability AI's terms; all are supplied by the operator.

## Offline weight migration

The serving process continues to accept safe tensor artifacts only. Explicit operator conversion
can migrate Real-ESRGAN `.pth` weights and Kohya SDXL LoRA files before registration:

```sh
media-engine convert --kind esrgan-pth --input RealESRGAN.pth \
  --config rrdbnet.json --output converted-upscaler
media-engine convert --kind lora-kohya --input style.safetensors \
  --reference-handle sdxl-base --catalog catalog.json --output converted-style
```

`rrdbnet.json` explicitly declares the architecture, scale, channels and block dimensions used by
Engine's ESRGAN loader. The `.pth` reader runs in a separate isolated Python process with
`torch.load(weights_only=True)`, a timeout, CPU placement and no arbitrary pickle allow-list.
It accepts the published `params_ema`/`params` wrapper or a bare tensor state dict and refuses
unknown keys or shapes. This step is never invoked by model loading or a generation request.

Kohya conversion supports linear SDXL UNet and dual-CLIP LoRA targets named by the reference
bundle's local configuration. Down/up matrices become PEFT A/B matrices; alpha is preserved.
Unsupported convolutional/LyCORIS layouts and unknown keys are errors, never silently discarded.
The result contains safetensors and conversion provenance; existing output directories are refused.

## Explicit CPU offload

On a ready CUDA reference worker, protocol 1.4 advertises `offload_modes` (`none`, `model`,
`sequential`) for the `sdxl-base`, `svd` and `stable-audio-open` families. Select a mode in load
options; `auto` never selects one. `model` transfers whole components as needed; `sequential`
uses Accelerate's per-submodule hooks and trades transfer overhead for lower GPU residency.
Owned/native workers and CPU-only workers offer `none`. Offload is not combined with compilation
or weight quantization, and a CPU fallback cannot silently reinterpret an offload request.
Memory estimates remain conservative full-residency estimates; no measured savings are claimed.

## Generation extensions in protocol 1.4

The reference SDXL backend advertises exact sampler/scheduler pairs: `euler/euler`,
`euler-ancestral/euler`, `dpmpp-2m/karras`, `ddim/ddim` and `unipc/unipc`. Owned/native SDXL
remain Euler-only. A requested pair is never substituted. Reference `inpaint` requires an image,
a same-size mask and prompt; white mask pixels identify the region to regenerate.

`image-transform` explicitly resizes (named nearest/bilinear filter), crops, pads, composites or
computes Canny edges (explicit ordered thresholds). Python workers execute these pixel operations
against a registered SDXL identity; the operation does not run inference or silently alter a
following generation request. Runtime recipes compose these operations into hires-fix or outpaint
flows, with every geometry change named and recorded.

FLUX.1-schnell (`flux-schnell`) and SD3.5 Medium (`sd35`) are reference families using the explicit
`flow-euler/flow-match` pair. They support their advertised text/image generation and inpaint paths.
Schnell requires guidance zero and an empty negative prompt; it is not a substitute for FLUX.1-dev.
Approve the exact model revision and its license before local registration. Native equivalents
remain deferred.

SDXL previews are opt-in through `preview_interval` on a worker that advertises them. They are
bounded nonfinal artifacts; they do not replace completed outputs or certify final image quality.
For optional MP4, install `uv sync --frozen --extra inference --extra inference-video`
(`av==14.2.0`) and explicitly select `encode_video: true` on a worker advertising the format.
The worker keeps the original frame archive/poster and never discovers an encoder on `PATH`.

Reference SDXL conditioning in 1.4 supports a verified local VAE override, textual-inversion
embeddings for either CLIP encoder, one IP-Adapter with a required uploaded `reference_image`,
and one ControlNet with a required uploaded `control_image` on text-to-image. Conditioning is
part of load identity; IP-Adapter and ControlNet cannot be combined. Optional `clip_skip` is 1–2
with Diffusers semantics (1 selects the third-from-last layer; omission uses the penultimate).
The conditioning path requires SDPA, no offload and no quantization/compilation. Exact limits,
files and mutually exclusive combinations are declared in the protocol; no remote model code
or implicit component replacement is allowed.
