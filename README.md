# Media Engine

An independently installed, offline SDXL inference worker for [Media Runtime](https://github.com/jxburros/Media-Runtime)
or another local supervisor. It implements `media-worker/1.1` (and still serves 1.0 callers) over
authenticated loopback HTTP, with asynchronous load/unload, a warm pipeline, one execution slot,
text-to-image and image-to-image, real sampling callbacks, cooperative cancellation, resumable SSE,
opaque input/output handles and header-based memory estimates.

The authoritative product scope is [engine-spec.md](engine-spec.md); the deferred capabilities it
specifies are sequenced in [docs/implementation-plan.md](docs/implementation-plan.md). This release delivers two of
its backend stages behind one protocol: the **reference worker** (`diffusers-sdxl`, the complete
Diffusers pipelines execute) and the **owned pipeline** (`owned-sdxl`, this project assembles the
components and runs the denoising loop itself). Native engines, LoRA, other families and custom GPU
kernels are future stages; they are not advertised as capabilities.

## Install

Reference environment: Linux x64, Python **3.12.14**, NVIDIA/CUDA, PyTorch **2.6.0**,
Diffusers **0.32.2**. `uv.lock` pins transitive artifacts/checksums separately from Runtime/Studio.
Installation is an explicit network action; serving never downloads models or executes remote code.

```sh
uv sync --frozen --extra inference
.venv/bin/media-engine --version
.venv/bin/media-engine --probe
```

The source package without the inference extra can run the protocol server and report missing
dependencies; it cannot generate. It never substitutes simulated output. Test backends live outside
the installed package in `tests/` and identify themselves in snapshots/provenance.

Supply a complete local `stabilityai/stable-diffusion-xl-base-1.0` Diffusers-format bundle,
including UNet, VAE, both CLIP encoders and tokenizers, pipeline config and Euler scheduler config.
Use safetensors weights only; Python, pickle weights and symlinks are rejected. The service does
not fetch weights, convert checkpoints or select a model for callers. Each local deployment is
identified by SHA-256 of every registered file; optional upstream revision/license metadata belongs
to the registration. Keep approved files immutable/read-only while the worker serves them.

Copy `examples/catalog.json`, replace the absolute model path and keep the same handle (`sdxl-base`)
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
worker also stops when its inherited stdin closes. Cancellation acknowledgement is intent only;
`cancelled` is terminal only after computation exits. If completion wins the race, it stays successful.
Standalone callers own their persistent jobs and storage; this service is not a permanent library.

## Development

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

`tests/test_worker.py` exercises protocol/lifecycle behavior without GPU downloads.
`tests/test_inference.py` runs real CPU inference through both backends on a tiny random-weight
SDXL-shaped bundle; it is skipped unless the inference extra is installed (`uv sync --frozen --extra
inference` then `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`). It proves the
owned loop and the reference pipeline execute the same contract and, for the same seed and settings,
produce byte-identical text-to-image output and image-to-image output within one 8-bit level on that
CPU stack, and that LoRA adapters change the output identically on both backends; it says nothing
about image quality. Run the Runtime
repository's `npm test` with both repositories beside each other for cross-process durable-job tests.
Machine-readable request definitions live in [schemas/worker.schema.json](schemas/worker.schema.json).
Engine 0.3.0 speaks `media-worker/1.1` and serves `media-worker/1.0` callers such as Runtime 0.1.0.

Source is MIT licensed. Dependency licenses and model-weight license terms are separate;
SDXL base weights carry CreativeML Open RAIL++-M terms and are supplied by the operator.
