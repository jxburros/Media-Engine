# Media Engine

An independently installed, offline media inference worker for [Media Runtime](https://github.com/jxburros/Media-Runtime)
or another local supervisor. It implements `media-worker/1.3` (and still serves 1.0, 1.1 and 1.2
callers unchanged) over authenticated loopback HTTP (or, in network mode, mutual TLS for a
supervisor on another host), with asynchronous load/unload, a warm pipeline, one
execution slot, SDXL text-to-image (up to four images per execution) and image-to-image, ESRGAN
upscaling, Stable Video Diffusion image-to-video as a lossless frame archive, real
sampling/tile/frame callbacks, cooperative cancellation, resumable SSE, opaque
input/output handles, header-based memory estimates, automatic precision with caller-authorized
fallbacks, device identity and an offline single-file checkpoint converter.

The authoritative product scope is [engine-spec.md](engine-spec.md); the deferred capabilities it
specifies are sequenced in [docs/implementation-plan.md](docs/implementation-plan.md). This release delivers two of
its backend stages behind one protocol: the **reference worker** (`diffusers-sdxl`, the complete
Diffusers pipelines execute) and the **owned pipeline** (`owned-sdxl`, this project assembles the
components and runs the denoising loop itself). The `esrgan` upscaler network is engine code on
both. The `svd` image-to-video family runs on the reference backend only. Native engines, the
audio family, an owned video loop and custom GPU kernels are future stages; they are not
advertised as capabilities.

## Install

Reference environment: Linux x64, Python **3.12.14**, NVIDIA/CUDA, PyTorch **2.6.0**,
Diffusers **0.32.2**. `uv.lock` pins transitive artifacts/checksums separately from Runtime/Studio.
Installation is an explicit network action; serving never downloads models or executes remote code.

```sh
uv sync --frozen --extra inference          # both backends (reference set)
uv sync --frozen --extra inference-owned    # owned-sdxl and the upscaler only: no accelerate, no peft
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
Use safetensors weights only; Python, pickle weights and symlinks are rejected. The service does
not fetch weights, convert checkpoints during a request (conversion is the separate offline
`convert` command below) or select a model for callers. Each local deployment is
identified by SHA-256 of every registered file; optional upstream revision/license metadata belongs
to the registration. Keep approved files immutable/read-only while the worker serves them.

Copy `examples/catalog.json`, replace the absolute model paths and keep the same handles (`sdxl-base`, `upscaler`, `video`)
in Runtime's artifact catalog. The **operator** supplies these paths at launch. Load requests carry
only approved opaque artifact handles, component digests and relative file identities.

Set `MEDIA_ENGINE_TOKEN` to a separately generated random credential of at least 32 characters,
through a protected service environment file (never a command-line argument). Then:

```sh
.venv/bin/media-engine --catalog /absolute/catalog.json --directory /absolute/worker-data --device cuda:0 --port 7811
```

Add `--backend owned-sdxl` to run the owned pipeline instead of the reference Diffusers pipelines.
The backend is part of the worker's identity; a supervisor's request must name the launched
backend explicitly, and nothing reinterprets a request when the backend changes. Both backends
accept the same bundle, options, ranges and events. They do not promise identical image bytes.

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
in this workspace. No model weights or CUDA device are present here.

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

Real-ESRGAN releases are `.pth` pickles, which the worker refuses. Converting one is an operator
action outside the worker, in a separate throwaway environment, on a file you trust:

```python
import torch
from safetensors.torch import save_file
state = torch.load("RealESRGAN_x4plus.pth", map_location="cpu", weights_only=True)
state = state.get("params_ema", state.get("params", state))
save_file({k: v.contiguous() for k, v in state.items()}, "network.safetensors")
```

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

The conformance suite (`tests/conformance`, `media-worker-conformance/1.1`) runs against any
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
random-weight svd bundle (64×64, two frames, two steps); `tests/test_convert.py` converts a tiny single-file
checkpoint built from it. They are skipped unless the inference extra is installed (`uv sync --frozen --extra
inference` then `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`). It proves the
owned loop and the reference pipeline execute the same contract and, for the same seed and settings,
produce pixel-identical text-to-image output and image-to-image output within one 8-bit level on that
CPU stack, and that LoRA adapters change the output identically on both backends; it says nothing
about image quality. Run the Runtime
repository's `go test ./...` with both repositories beside each other for cross-process durable-job tests.
Machine-readable request definitions live in [schemas/worker.schema.json](schemas/worker.schema.json).
Engine 0.6.0 speaks `media-worker/1.3` and serves `media-worker/1.0`–`1.2` callers such as Runtime 0.1.0–0.6.0;
its startup record keeps `protocol: "media-worker/1.1"` for those launchers and reports `highest_protocol`.
`tests/serve_fixture.py --protocol media-worker/1.2` (or `1.1`) pins the fixture to an older surface.

Source is MIT licensed. Dependency licenses and model-weight license terms are separate;
SDXL base weights carry CreativeML Open RAIL++-M terms and Stable Video Diffusion weights carry
Stability AI's terms; both are supplied by the operator.
