# Media Engine

An independently installed, offline SDXL reference worker for [Media Runtime](https://github.com/jxburros/Media-Runtime)
or another local supervisor. It implements `media-worker/1.0` over authenticated loopback HTTP,
with asynchronous load/unload, a warm pipeline, one execution slot, real sampling callbacks,
cooperative cancellation, resumable SSE and temporary output handles.

The authoritative product scope is [engine-spec.md](engine-spec.md). This release is its **reference
worker** stage, using Diffusers and PyTorch. Owned pipelines, native engines, LoRA, other families
and custom GPU kernels are future stages; they are not advertised as capabilities.

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

`--port 0` chooses a free loopback port. Stdout emits one JSON startup record with protocol,
version, random session ID, port and PID; the supervisor must still check `/v1/info` and
`/v1/capabilities`. All endpoints except coarse `/health` require the per-launch bearer credential.
Browser origins are rejected. `--device cpu` explicitly selects CPU/float32; it is never a fallback.

The default GPU options are float16, SDPA attention and no offloading. The worker only advertises
options it applies. The reference hardware target is an RTX 4090 with 24 GiB, Linux x64 and a driver
compatible with the locked CUDA PyTorch distribution. This target has not been hardware-certified
in this workspace. No model weights or CUDA device are present here.

`--memory-budget-mb` sets PyTorch's per-process allocator budget for the selected GPU. This does
not coordinate or terminate other GPU services. Estimates report unknown values honestly;
execution metadata reports actual peak allocated bytes where measurable. An OOM requires a fresh
worker session; no smaller image or alternate precision is retried automatically.

## Lifetime and collection

The worker has no waiting queue. Loading, unloading and inference cannot overlap. An identical
attempt retry returns its existing execution; conflicting reuse fails. A busy worker refuses
admission. The pipeline remains loaded until explicit unload or process shutdown.

Every execution, event and output is retained until explicit discard or session shutdown. Limits
(default 128 records and 1024 MiB temporary output storage) reject new work rather than evicting
uncollected results. Successful output files are finalized before publication. Runtime must copy
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

These tests exercise protocol/lifecycle behavior without GPU downloads. Run the Runtime repository's
`npm test` with both repositories beside each other for cross-process durable-job tests.
Machine-readable request definitions live in [schemas/worker.schema.json](schemas/worker.schema.json).
Both apps at version 0.1.0 speak `media-worker/1.0`.

Source is MIT licensed. Dependency licenses and model-weight license terms are separate;
SDXL base weights carry CreativeML Open RAIL++-M terms and are supplied by the operator.
