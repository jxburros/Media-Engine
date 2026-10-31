# Cross-repo review: Media Runtime, Media Engine, JX Runtime, Onyx Engine

**Date:** 2026-10-31 · **Type:** dated analysis. It is not maintained, and the code wins where they
disagree. · **Agent:** Claude Code (Claude)

The same file is in `Media-Runtime/docs/reviews/` and `Media-Engine/docs/reviews/`.

## 0. Purpose, scope and method

**The goal.** Media Runtime and Media Engine should become a **headless backend that replaces ComfyUI** in my own
applications, the way JX Runtime and Onyx Engine replace Ollama. The focus is different: durable,
verifiable and secure media jobs, rather than a node-graph playground.

**The question.** What can the four repositories learn from each other, and what can each copy from the
others, to get there?

**Repositories and snapshots.** All were read on branch `claude/lucid-feynman-vean0r`, which equals `main` at the time of reading.

| Repo | Commit | Version |
| --- | --- | --- |
| `jxburros/Media-Runtime` | `1048054` | 0.10.0 |
| `jxburros/Media-Engine` | `81eba18` | 1.0.0 (Python 0.8.0 surface + native) |
| `jxburros/JX-Runtime` | `a6cd934` | 1.5.10 |
| `JX-Holdings-LLC/Onyx-Engine` | `d5f4b71` | 0.7.0 |

**Method.**
- I read the READMEs, specs, implementation plans, decision ledgers, compatibility records,
  CI workflows and key source files.
- I spot-checked the important claims against the code.
- **Nothing was built, run or benchmarked for this review.** Test counts and pass/fail
  statements are what each repository records about itself.
- Where a claim rests only on a repository's own docs, I say so.

---

## 1. Executive summary

1. **The analogy holds structurally but is inverted in maturity.**
   - JX Runtime and Onyx are mature on *operability*: installers, release pipelines, CI,
     auth/pairing, metrics, model downloads, a UI and a tray. They are thin on *contract*: the engine
     has no protocol version, there is no conformance suite, and JX tests Onyx only against a fake.
   - The Media pair is the opposite. Its *contract* is exceptional: a versioned, negotiated worker
     protocol (`media-worker/1.0`–`1.3`), a conformance suite, durable jobs, verified outputs,
     provenance manifests, and a dependency ledger. It has almost **no operability**: no CI, no
     tags or releases, no installers, no model acquisition, no metrics, no key management.
   - Each family has what the other lacks. That is the core finding of this review.

2. **Nothing in the Media stack has run on a GPU or on real weights.** Both compatibility records say
   so plainly (`Media-Runtime/docs/compatibility.md` "Hardware validation: none of these rows ran on a
   CUDA device or real weights"; `Media-Engine/docs/compatibility.md`).
   - The stack covers SDXL, ESRGAN, SVD, Stable Audio, worker pools, mutual TLS and a Rust native
     engine. All of it was built in about five days of mostly agent-authored commits (git log
     2026-10-26 → 10-31).
   - JX Runtime's own `CLAUDE.md` warns against "overconfident claims without validation"
     (issue #59).
   - **The first priority is not another feature. It is one real RTX 4090 testbench run and CI.**

3. **As a ComfyUI replacement, the biggest gap is image-generation depth, not infrastructure.**
   - Today the stack serves *one* image family (`sdxl-base`) with *one* sampler (`euler`, enforced at
     `Media-Engine/src/media_engine/contracts.py:977`).
   - It has no inpainting or masks, no ControlNet, no IP-Adapter, no live previews, no
     resize/crop, and no newer DiT image families.
   - Applications that use ComfyUI as a backend depend on exactly those things. Breadth of
     operations should now take precedence over more ownership stages.

4. **Media Runtime should copy JX Runtime's operator and security layer almost wholesale.** That means:
   - scoped API keys minted through the API
   - rate limits and an audit log
   - opt-in, verified model installs from Hugging Face
   - *watched model folders that read an existing ComfyUI `models/` tree in place* (JX does the same for an Ollama store)
   - a managed engine install pinned by SHA-256
   - Prometheus metrics
   - per-user service registration and installers
   - the shared JX Holdings release pipeline

5. **JX Runtime and Onyx should copy the Media pair's contract discipline.** That means:
   - an explicit engine protocol version and capability endpoint
   - a conformance suite run against the *real* Onyx binary in JX's CI
   - a JSON startup record on stdout
   - graceful SIGTERM drain (Onyx has no signal handler)
   - model digests verified on every load
   - memory estimates owned by the engine
   - "advertise an optimization only with a cited benchmark", which answers JX #109 and Onyx's missing benchmarks
   - worker pools across GPUs and remote engines over mutual TLS

6. **The family needs one cross-cutting agreement: a shared GPU budget between JX Runtime and Media Runtime**
   on the same machine. Both specs explicitly leave this unowned (`runtime-spec.md` "Global GPU
   coordination requires a separate shared agreement"). The user owns both sides, so this can now be designed
   instead of deferred.

---

## 2. The four repositories at a glance

| | **JX Runtime** | **Onyx Engine** | **Media Runtime** | **Media Engine** |
| --- | --- | --- | --- | --- |
| Role | Supervisor / control plane for LLMs | LLM inference server | Supervisor / control plane for media | Media inference worker |
| Replaces | Ollama | llama-server (llama.cpp) | ComfyUI (backend use) | ComfyUI's execution core |
| Language | Node ≥18 (1 prod dep: express) | C++17 over pinned llama.cpp | Go 1.24, static, 1 dep (pure-Go SQLite) | Python 3.12 (Diffusers/PyTorch) + Rust/candle native |
| Size | ~112 src files, 115 test files | ~6.9k own LOC | ~15k LOC incl. tests | ~21k LOC incl. tests |
| Public API | OpenAI-compatible `/v1` + `/runtime` management (~45 routes), OpenAPI committed | OpenAI-compatible `/v1`, `/props`, `/metrics` | `/v1` jobs/models/recipes/inputs/workers, SSE | `media-worker/1.3` (loopback HTTP or mTLS) |
| Engine contract | Inferred: `--help` probing, `/props`, fake-engine fixtures | None versioned; stability page + `--help` rule | Negotiates protocol minor, checks capabilities | Versioned protocol, JSON schema, conformance suite |
| Jobs | Request/response, bounded FIFO, 2 lanes | Slots, continuous batching, `--max-queue` | Durable SQLite FIFO, idempotency, reconciliation | Single-flight executions, no queue |
| Model acquisition | HF/URL/import/watched folders, resumable, pinned catalogue | Loads a path | **None**: operator hand-writes catalog | **None**: offline `convert` only |
| Auth | Hashed scoped keys, perms, pairing, users, rate limits, audit | Optional `--api-key`, Origin/Host guards | Env-var credentials, namespaces, operator separation | Per-launch bearer + optional mTLS |
| Observability | Prometheus + Grafana, diagnostics export, doctor + repair plans | Prometheus `/metrics` | `doctor`, TUI | Snapshots/events only |
| CI | Yes (self-hosted Windows; others paused) | Yes (Linux/macOS/Windows) | **None** | **None** |
| Releases | Shared release infra, installers, checksums | 9 archives, Docker CPU image, checksums | **No tags** | **No tags** |
| GPU-validated | Partially (QA reports; perf unmeasured, #109) | CPU-only CI; GPU paths "not exercised" | **No** | **No** |
| Governance | Spec-Driven Docs (AGENTS.md, CLAUDE.md, development-docs) | CONTRIBUTING, stability page, no AGENTS.md | Spec + plan + decisions + compatibility, no AGENTS.md | Same as Runtime |

Sources: each repo's README, `docs/compatibility.md` and `.github/workflows/`. For JX:
`src/binaries.js`, `src/auth.js` and `docs/api.md`. For Onyx: `src/server.cpp`,
`docs/reference/stability.md` and `.release/config.json`.

---

## 3. How the analogy maps, and where it breaks

```
   applications ──►  JX Runtime  ──►  Onyx Engine (llama.cpp)          LLM family
   applications ──►  Media Runtime ──► Media Engine (Diffusers | owned | native)   media family
```

**Where it holds:**
- Both supervisors own scheduling, model registry, auth and the public API.
- Both engines are single-model and replaceable.
- `engine-spec.md` §"Progressive inference ownership" explicitly cites Onyx's model, "an
  independent serving/execution layer over llama.cpp/ggml", as the template for owning the engine
  without rewriting the compute stack.

**Where it breaks. These differences are intentional and should stay:**
- **Public API compatibility.**
  - JX wins adoption by speaking the de-facto standard (the OpenAI wire format), even though it
    does *not* speak Ollama's `/api/*` (`JX-Runtime/docs/ollama-compatibility.md` §Scope).
  - There is no de-facto standard media generation API. ComfyUI's `/prompt` takes a graph of
    *ComfyUI node class names*, which Media Runtime cannot honor without becoming ComfyUI.
  - Media Runtime's own job API *is* the product. The right parallel to JX's "OpenAI-compatible, not
    Ollama wire-compatible" is "**ComfyUI-capable, not ComfyUI wire-compatible**". See §4.4.
- **Request duration.**
  - LLM calls are seconds long and streamed, so JX is request/response.
  - Media jobs are seconds to many minutes and produce files, so Media Runtime is durable
    jobs plus a file store.
  - Neither should copy the other's core execution model.
- **Engine relationship.**
  - Onyx is a general server that clients may call directly, and JX treats it as llama-server-compatible.
  - Media Engine is a *worker* that refuses to be a library ("this service is not a
    permanent library") and has no queue.
  - Both postures fit their domains.

---

## 4. Gap analysis: what a ComfyUI replacement needs

ComfyUI as an application backend means, in practice:
- **API:** `POST /prompt` with an API-format workflow, `/ws` progress with binary preview frames,
  `/history`, `/view`, `/upload/image`, `/interrupt`, `/queue`, `/object_info` and `/system_stats`.
- **Models:** a `models/` tree split by type (`checkpoints`, `loras`, `vae`, `controlnet`,
  `upscale_models`, `embeddings`, `clip`/`text_encoders`, `diffusion_models`), with single-file
  checkpoints loaded directly.
- **Breadth:** many families and conditioning methods, available within days of a model's release.
- **Runtime behaviour:** smart memory management and offloading, and caching of unchanged node
  outputs between runs.
- **Extensibility:** a custom-node ecosystem, which is also its main supply-chain risk.

ComfyUI is GPL-3.0. Its code cannot be vendored into these MIT repositories, so everything below
is reimplemented or comes from Apache-2.0 Diffusers.

### 4.1 Capability table

**Priority** means: **P0** = blocks any real use; **P1** = table stakes for replacing ComfyUI in
an application; **P2** = important; **P3** = later.

| Capability | ComfyUI | Media stack today | Recommendation | Priority |
| --- | --- | --- | --- | --- |
| Proven to run on a real GPU | Yes | **No** (CPU fixtures and random weights only) | Run the testbench on the RTX 4090 with real SDXL, ESRGAN, SVD and Stable Audio; record it in both compatibility records | **P0** |
| CI | n/a | **None** in either repo | See §5.8 | **P0** |
| Image families | SD1.x, SDXL, SD3.x, Flux-class DiTs, many more | `sdxl-base` only | Add families on the **reference (Diffusers) backend first**. That backend exists to be the fast path to breadth. Start with one Flux-class DiT family and SD3.5, then consider SD1.5 for its LoRA ecosystem | **P1** |
| Samplers/schedulers | Dozens | `euler` only (`contracts.py:977`) | Advertise a small, explicit set (Euler, Euler a, DPM++ 2M with Karras sigmas, DDIM, UniPC) as `capabilities` values. Keep "never silently changed" | **P1** |
| Inpainting / outpainting (mask input) | Yes | No | New task `inpaint` with a `mask` input (PNG, same size, validated). Outpaint = pad + inpaint, using the utility ops below | **P1** |
| Structural conditioning (ControlNet, T2I-Adapter) | Yes | No | A new load-identity component, like LoRA adapters today: `controlnets: [{handle, digest, scale}]`, plus a `control_image` input per execution | **P1** |
| Image-prompt conditioning (IP-Adapter / reference) | Yes | No | Same pattern as ControlNet | **P2** |
| Live previews during sampling | Binary frames over `/ws` | Step counts only | Optional `preview` events: low-resolution latent→RGB or a tiny VAE decode every N steps, offered as an output handle (not inline base64). Off by default and advertised in capabilities | **P1** |
| Resize, crop, pad, composite | Core nodes | Refused by design ("nothing is resized") | Add **explicit** deterministic utility operations (`resize`, `crop`, `pad`, `composite`) as owned code. They are recipe operations the caller names, which keeps the no-silent-change rule. They unlock hires-fix (`generate → upscale → resize → image-to-image`) and outpainting as linear recipes | **P1** |
| MP4/WebM output | Via nodes (ffmpeg) | Frame ZIP only; MP4 path never exercised | Resolve the open decision in favour of an **optional pinned encoder extra** in Engine (spec already allows `inference-video`). Applications need a playable file | **P1** |
| Single-file checkpoints | Loaded directly | Offline `convert` to Diffusers layout | Keep conversion, but let Runtime drive it. `models import FILE` runs Engine's converter as an explicit operator action and catalogs the result (see §5.2) | **P2** |
| `.pth` upscalers, Kohya LoRAs | Loaded directly | Refused; operator converts by hand | Ship the converters in Engine's `convert` subcommand (`--kind esrgan-pth`, `--kind lora-kohya`) with `weights_only=True` loading in an isolated step | **P2** |
| Textual-inversion embeddings, CLIP skip, VAE swap | Yes | No | Embeddings and VAE swaps become load-identity components. CLIP skip is an advertised parameter | **P2** |
| Graph execution (branching, fan-out) | Arbitrary DAG | Linear recipes | Extend recipe schema to a declarative **DAG** (several consumers per output, `foreach` over a batch). Still no code, no graph import | **P2** |
| Execution caching | Unchanged nodes reused | Warm pipeline only | Cache text-encoder outputs and intermediate durable outputs keyed by digest. Recipes already identify inputs by digest, which makes this safe | **P3** |
| Low-VRAM modes / offload | Automatic | "no offloading" default, OOM fails | Advertise `offload: model|sequential` as an explicit load option. Selection stays caller-authorized, like `authorized_fallbacks` | **P2** |
| Video families beyond SVD; audio beyond Stable Audio | Many | SVD, Stable Audio Open | After the image P1s. Same per-family pattern | **P3** |
| Captioning, background removal, face restore, depth/pose preprocessors | Custom nodes | None | Only the preprocessors needed by ControlNet (depth, pose, canny) belong in Engine as tasks. The rest is P3 | **P3** |
| Completion webhooks | No (clients poll / ws) | No (SSE only) | `callback` on submission: a signed POST on terminal state, to an operator-allow-listed URL. Useful for server-side applications | **P2** |
| Custom nodes / plugins | Yes | Out of scope by spec | **Keep out of scope.** This is the deliberate difference in focus (no executable recipe code, no supply-chain surface) | — |

### 4.2 What Media already does better than ComfyUI

Keep these and lead with them. They are the reason to use this stack instead of ComfyUI:
- **Durable jobs.** Jobs survive restarts and client disconnects, retries are idempotent, and work
  is reconciled by worker session and attempt (`runtime-spec.md` §Durable jobs).
- **Verified outputs and provenance.** Every output is checked by size, SHA-256 and format
  signature before success, and every job gets a provenance manifest (model digests, seed, settings,
  worker and device identity).
- **No silent changes.** Model, size, precision and scheduler are never changed to make a job fit.
  Fallbacks need caller authorization.
- **No code execution.** There are no custom nodes, no remote code, no downloads during inference,
  and pickle weights are refused.
- **Security.** Namespaces, authentication on every route and mutual TLS to remote workers.
- **Multi-GPU and remote execution.** A worker pool and remote workers as first-class
  configuration.

### 4.3 Product positioning to write into `runtime-spec.md`

> Media Runtime is a backend for applications that need ComfyUI-class image, video and audio
> generation with durable jobs, verified outputs and no executable plugins. It is not
> ComfyUI-compatible on the wire; it is ComfyUI-*capable* through declarative recipes.

This mirrors JX's "The runtime executes. The application decides." (`JX-Runtime/development-docs/coreIdentity.md`).

### 4.4 About ComfyUI workflow import

The specs exclude "arbitrary ComfyUI graph import" (`runtime-spec.md`, `engine-spec.md`). I agree
for the runtime. If migration from existing ComfyUI workflows matters, the JX precedent applies:
- **JX's approach:** JX does not implement Ollama's API, but it *reads an Ollama model store* and
  documents every deliberate difference (`docs/ollama-compatibility.md`).
- **The media equivalent:**
  - an **offline, out-of-process translator** that converts a *subset* of API-format workflows
    (checkpoint → KSampler → VAE decode → save, with LoRA, upscale and inpaint) into schema-2 or
    DAG recipes, and refuses everything else by node name;
  - a `docs/comfyui-migration.md` modelled on JX's Ollama migration doc.
- **Decision needed:** this touches an explicit out-of-scope line, so it needs your decision (§11).

---

## 5. What Media Runtime should take from JX Runtime

These are ordered by value for "a backend I can use in my own applications".

### 5.1 Access control for applications (JX `src/auth.js`, `src/pairing.js`, `src/ratelimit.js`)

Media Runtime reads static credentials from environment variables named in `runtime.config.json`.
Applications need keys they can mint, scope and revoke without a restart.

- **Hashed API keys.** Keys are created with `POST /runtime/tokens`-style routes and shown once. Each
  key carries:
  - permissions: `jobs.submit`, `jobs.read`, `models.read`, `models.register`, `admin`
  - a model allow-list
  - an expiry
  - CIDR restrictions
  - concurrency and queue-share limits
  - JX hot-reloads key changes within about one second.
- **Namespaces map naturally onto keys.** Media's `allowed_namespaces` becomes a key property.
- **Rate limits.** Per-IP limiting before key verification (so a flood of bad keys gets 429), per-key
  sliding windows, and a per-key share of the queue (today `max_queued_jobs` is global).
- **Audit log.** Rotating, attacker-triggerable events capped. Record registrations, key changes,
  cancellations and fallbacks.
- **Pairing codes.** For applications on another device: a single-use code exchanged for a scoped key
  plus the TLS fingerprint. Media's API network mode already reports `certificate_sha256`.
- **Constraint.** These JX modules are flagged for extra scrutiny (`JX-Runtime/AGENTS.md` §15).
  Port the *design* to Go and review it as new security code; do not treat JX's tests as evidence
  for the port.

### 5.2 Model acquisition (JX `src/modelManager.js`, `src/downloader.js`, `src/modelFolders.js`, `src/catalog.js`, `src/safeFetch.js`)

This is the largest usability gap. Today the operator must:
- download a bundle by hand
- delete extra files
- write `tokenizer.json` offline
- convert `.pth` files in a throwaway environment
- write absolute paths into *two* catalogs (Runtime's `artifact_catalog` and Engine's `catalog.json`)

Copy JX's design, keeping Media's "installation is an explicit operator action, never part of a
job" rule:
- **Watched folders, read in place.** Point Media Runtime at an existing **ComfyUI `models/`
  directory**, an HF cache or a plain folder. Classify files by layout, never copy or delete them,
  and tell an unplugged drive from a deletion (JX does this by device ID).
  - This is the media counterpart of JX reading an Ollama store, and the cheapest migration path
    from ComfyUI.
  - The constraint "Engine hashes every file in the directory" means a ComfyUI checkpoint folder
    needs per-file handles, not per-directory ones. Engine's catalog would gain file-level handles
    for single-file formats.
- **Opt-in Hugging Face installs** pinned to a revision with per-file SHA-256. Use a curated starter
  catalogue like JX's `src/catalog.js`, with pins kept fresh by a script like
  `scripts/update-catalog-pins.js`. Include `allow_patterns` so only the safetensors variant you
  serve lands on disk; that removes today's "keep only the weights you serve" chore.
- **Resumable downloads, a free-space check and SSRF-safe fetching.** Port JX `safeFetch`'s rules:
  public unicast only, redirects re-checked at each hop, `Authorization` dropped across origins.
- **One catalog, not two.** Runtime should generate and hand Engine its catalog for owned workers
  (it already owns the launch arguments), so handles cannot drift between the two.

### 5.3 Managed engine installation (JX `src/binaries.js`, `src/processGuard.js`)

- JX downloads Onyx release archives pinned by SHA-256 per platform, chooses a CPU variant from
  `/proc/cpuinfo`, auto-upgrades stale builds, and reaps orphaned engines left by `kill -9`.
- Media Runtime can do the same once Engine has releases (§6.1):
  - **Native engine.** Pin `media-engine-native` archives the same way. This is the strategic
    payoff of the Rust engine: a single-binary, no-Python install, exactly like Onyx.
  - **Python reference/owned backends.** Manage a private `uv`-synced virtual environment from a
    pinned Engine wheel plus `uv.lock`, or use a pinned CUDA container image (§6.1).
- Add JX's PID-file orphan reaper. Media Runtime already refuses to kill external workers; an
  orphaned *owned* worker after a Runtime crash is the gap.

### 5.4 Operator surface (JX `src/doctor.js`, `src/repair.js`, `src/telemetry.js`, `src/service.js`, `grafana/`)

- **Prometheus metrics.** Media Runtime has none. Expose:
  - `media_jobs_total{recipe,state}`
  - `media_job_phase_seconds{phase=queued|loading|sampling|decoding|commit}`
  - `media_worker_busy`
  - `media_outputs_bytes_total`
  - `media_queue_depth`
  - per-worker peak memory as reported by the worker

  Ship a starter Grafana dashboard like `JX-Runtime/grafana/`.
- **Guided-repair errors.** JX's `error.details.repair` gives pre-filled fix requests ("install the
  backend", "use a smaller context") that an application can render as buttons. Media's refusals
  are already precise (`error.field`), so adding a `repair` plan is cheap and very useful to
  application UIs. Examples: "register the converted bundle", "lower `count` to the advertised
  limit", "authorize the `cpu` fallback".
- **Diagnostics export.** One JSON file, no prompts, for bug reports.
- **Service registration.** JX's per-user systemd unit, launchd agent or Windows logon task, with
  named instances. Media Runtime is Linux-only today, so systemd first.
- **Managed mode handshake.** JX's `serve --managed --port 0` emits NDJSON `ready/stopping/stopped`
  on stdout for a parent process. Media Runtime embedded inside an application's process tree needs
  the same thing. Media *Engine* already does this toward Runtime, so Runtime should offer it upward.
- **Discovery.** JX's `/.well-known/jx-runtime` descriptor plus local announce file lets apps find
  the runtime. Use a matching `/.well-known/media-runtime` so an application can find both
  runtimes the same way.
- **Headless means no web UI.** JX's UI is not needed; the TUI is enough. Keep JX's "no build
  step, no external requests" rule if a status page is ever added.

### 5.5 Residency and memory policy (JX `src/estimate.js`, `src/scheduler.js`)

- JX estimates weights + KV + compute buffers before load, picks a GPU-first plan, explains each
  automatic choice (`autoDecisions`), evicts idle models least-recently-used, and honours
  `keep_alive` / `idleUnloadMinutes`.
- Media keeps a pipeline loaded "until explicit unload or process shutdown" and has no idle unload.
  That matters when the same GPU also serves JX.
- **Add:**
  - `idle_unload_minutes` per worker
  - a per-job `keep_loaded` hint
  - affinity-aware LRU in the pool, which already prefers a worker with the model loaded
- **Pair it with the cross-family GPU agreement** in §9.1.

### 5.6 Integration documentation (JX `docs/compatibility.md`, `docs/version-compatibility.md`, `docs/openapi.json`, `tests/docs.test.js`)

- **Commit an OpenAPI document** and add JX's drift test: a served route without docs, or a
  documented route that isn't served, fails CI. Media's API is large and evolving; this is the
  cheapest way to keep `runtime-spec.md`, the README and `gateway-client` honest.
- **Versioned compatibility doc for integrators**, like JX's AI Server Studio rules. It should state
  what a client may rely on across Runtime minors.
- **Typed clients.** Media is ahead of JX here (see §7.5). Add a small Python client alongside the
  TypeScript one, since many media applications are Python.

### 5.7 Packaging and distribution (JX `scripts/build-binary.js`, `scripts/installers/`, `.release/`)

- Media Runtime is already one static Go binary, which is easier than JX's Node SEA.
- **Missing:** release archives, `checksums.txt`, a `curl | sh` installer that verifies checksums and
  never uses sudo, a `.deb`, and an update check (check only, as JX does).
- **Use the shared `JX-Holdings-LLC/release-infrastructure` workflows** that JX and Onyx already
  pin: manual dispatch, dry run first, all-or-nothing matrix, contracted asset names.

### 5.8 Engineering process (JX `AGENTS.md`, `CLAUDE.md`, `development-docs/`, `.github/workflows/`)

- **CI first.**
  - The Go suite, plus `-race`, against Engine's fixture: clone Media-Engine at a pinned commit and
    use its `serve_fixture.py`.
  - The Python suite and the conformance suite in Engine.
  - `cargo test` for `native/`.
  - Everything the compatibility records claim was run by hand should run on every PR.
- **A weekly real-backend job**, like JX's `real-backend.yml` and `engine-ground-truth.yml`: the
  testbench on a self-hosted GPU runner, when one exists.
- **Governance docs.**
  - **Recommended:** adopt Spec-Driven Docs (`AGENTS.md`, `CLAUDE.md`, `development-docs/coreIdentity.md`
    with a Never list). These repositories are agent-built at high speed. The existing specs and
    decision ledgers are good raw material, and the Never rules already exist as prose ("MUST NOT
    silently change…").
  - **Alternative:** if the full system is too heavy, at least add `AGENTS.md` with the hard
    constraints and validation rules.
- **Version bump policy and changelog convention.** JX and Onyx bump on every main change. Decide
  whether Media follows suit.

### 5.9 Do not copy from JX

- **The 289 KB changelog.** Media's release-scoped changelog is easier to read.
- **A 1,474-line config validator in one file.** Media's `internal/runtime/runtime.go` is already
  2,507 lines; split it before porting JX features into it.
- **CI that depends on a single self-hosted Windows runner** (JX's hosted legs are "paused").

---

## 6. What Media Engine should take from Onyx Engine

### 6.1 Releases and distribution (Onyx `.github/workflows/release.yml`, `Dockerfile`, `packaging/`)

- **Native engine archives.** Release them per platform, with `checksums.txt` and contracted names
  (`media-engine-native-v<ver>-<platform>-<arch>[-cuda].tar.gz`).
  - Copy Onyx's ldd/otool check that fails a leg if the binary links unexpected shared libraries.
  - Copy the all-or-nothing matrix.
- **A CUDA container image for the Python backends.** This is more useful for Media than the CPU
  image is for Onyx, because "pinned Python + CUDA PyTorch + Diffusers" is the hardest part of
  installing Engine.
  - Copy Onyx's non-root uid and multi-target Dockerfile.
  - Onyx ships only a CPU image. Media should ship CUDA first.
- **Wheels plus the lock.** Publish the Python package as a wheel together with `uv.lock`, so Runtime
  can manage a virtual environment (§5.3).
- **Pin vendored dependencies by commit with a stamp file.** Copy Onyx's `packaging/vendor/*.conf` and
  `.vendor-source-commit` for candle and for any future kernels. `Cargo.lock` is committed, which
  covers crates.io.

### 6.2 Serving behaviour

- **Prometheus `/metrics`.** Onyx exposes llama-server-compatible names. Engine should expose
  execution counts, phase durations, peak allocated memory, load time and rejected-busy counts.
  - Runtime would scrape and re-export them, so operators see one surface.
- **Pre-routing hardening constants.** Onyx refuses any request with an `Origin` header, checks
  `Host` against DNS rebinding on loopback, caps JSON nesting at 64 and caps body size, with all
  limits as named constants (`src/server.cpp` 28–46).
  - Engine rejects browser origins, but check the Host-header and nesting rules in both the
    Python and Rust servers.
- **Bind first, then answer 503 "loading".** Engine already separates service and model readiness,
  so no change is needed. Note this as shared good practice.

### 6.3 Testing breadth (Onyx `.github/workflows/ci.yml`, `scripts/smoke-test.sh`)

- Onyx runs its tiny-model smoke tests on Linux, macOS and Windows on every PR (123 checks).
- Engine already has tiny random-weight bundles (`tests/test_inference.py`) and a hardware-free
  fixture. The only missing piece is CI. This is the same item as §5.8.
- The native engine is loopback-only, `sdxl-base`-only, and has no compiled CUDA build. Its CI
  should at least build `--features cuda` without running it, so the CUDA path cannot rot.

### 6.4 Ownership pacing

- Onyx deliberately *keeps* llama.cpp's `common/` until each piece has a measured reason to
  replace it, with exit tests such as "bit-identical at temp 0" (`reports/Replacing llama cpp
  common code.md`).
- Media Engine went further, faster:
  - a from-scratch Rust SDXL (tokenizer, UNet, VAE, scheduler) before any GPU run
  - about **four times slower than the Python owned pipeline on CPU**, because of a candle-core
    0.9.2 convolution workaround (Engine CHANGELOG 1.0.0)
- **Recommendation:** freeze further ownership stages (native video and audio loops, custom kernels)
  until the reference backend has the P1 breadth from §4.1 and one GPU benchmark exists.
  - New families should land on `diffusers-sdxl` first. Consider renaming it to `diffusers`,
    because it already serves SVD and audio, so the name misleads.
  - The native engine's job for now is to be the easy-to-install, signed binary for SDXL-class
    image generation.

### 6.5 Do not copy from Onyx

- **Onyx's lack of a protocol version.** Media's explicit version is better.
- **SIGTERM dropping in-flight requests.**
- **No model hash verification.**
- **Plain HTTP only.** Onyx leaves TLS to a proxy; Media's mutual TLS for remote workers is the
  better design for a multi-host media farm.

---

## 7. What JX Runtime should take from Media Runtime

### 7.1 An explicit engine contract and a conformance suite against the real Onyx

- **How JX checks compatibility today:**
  - JX treats Onyx as llama-server-compatible by subclassing `LlamaCppAdapter`.
  - It discovers features by probing `--help` and reading `/props`.
  - It tests Onyx only against `tests/fixtures/fake-onyx-engine.js`.
  - It pins Onyx `v0.6.0` (`src/binaries.js:302`) while Onyx is at 0.7.0.
  - `auto` backend selection never picks Onyx, and every Onyx build is marked `accelerated: false`.
- **What Media does.** Media Runtime negotiates a protocol minor, checks every feature against
  advertised capabilities, and refuses with the field named.
- **Adopt:**
  1. A capability document from Onyx, either `GET /v1/capabilities` or an extension of `/props`,
     with a contract version string. Onyx's stability page already defines the compatibility rules;
     this gives them a machine-readable form.
  2. A conformance suite in the Onyx repo, like Media Engine's `tests/conformance`, runnable against
     any URL. JX's CI runs it against the *real pinned Onyx binary* on a tiny model, not only
     against the fake.
  3. A compatibility record like Media's `docs/compatibility.md`, listing JX version × Onyx
     version × suite result.

### 7.2 Worker pools and remote engines

- JX is one backend per process, has no multi-GPU support, and fits each model to "the single GPU
  with most free VRAM" (`src/estimate.js` ~L174).
- Media Runtime's pool design gives a pattern JX could adopt for multi-GPU hosts and small
  multi-host labs:
  - one engine process per device, with device identity
  - duplicate-device refusal
  - affinity scheduling
  - pinning
  - per-worker reconciliation
  - remote workers over mutual TLS with fingerprint pinning
- **Keep it explicit configuration, never automatic** (JX's coreIdentity "not smart" scheduling rule
  still applies).

### 7.3 Provenance and verification

- **Media's practice:** every load verifies component digests, and every job records model digests,
  backend version and device identity.
- **JX's practice:** JX records sha256 at install and re-hashes only on `doctor
  --check-provenance`. It already tells clients which model served a request through
  `X-JX-Embeddings-Model`.
- **Cheap wins:**
  - verify the digest at load, with caching keyed by size, mtime and inode, as Media allows
  - an `X-JX-Served-By` header or `jx` response extension carrying model digest and engine version
    on every `/v1` response
  - this helps applications that must prove which model produced an output

### 7.4 Durable batch jobs, optional

- Chat completions should stay request/response.
- Long batch work such as bulk embeddings or batch completions would benefit from Media's job
  model: idempotency keys, durable FIFO, resumable SSE events and cancellation intent.
- If JX ever adds a batch API, take Media Runtime's semantics as written. Don't invent new ones.

### 7.5 Typed client and real-model testbench

- **Gateway client.** Media ships `gateway-client/`, a dependency-free TypeScript client versioned
  with the service. JX ships ambient types only (`types/jx-runtime.d.ts`). A small typed client for
  `/runtime` would help AI Server Studio and other applications.
- **Testbench.** Media's `testbench/testbench.py` is one stdlib Python file that starts both
  services on *your* models, runs one job per feature and writes an HTML report. JX's
  `perf-smoke.yml` "has never run" (roadmap) and #108/#109 need real-hardware runs. A JX
  testbench in the same shape would turn those issues into a repeatable command.

### 7.6 Dependency ledger and "not validated" sections

- Media keeps a dependency ledger with one justification per dependency (`docs/decisions.md`), and
  every changelog entry says what was **not** validated.
- JX already requires discussion for new production dependencies, and Onyx's changelog has a
  Validation section. A ledger file makes the JX rule auditable.

---

## 8. What Onyx Engine should take from Media Engine

| Media Engine practice | Onyx today | Adopt |
| --- | --- | --- |
| Startup record: one JSON line on stdout (protocol, version, session ID, port, PID) | Two parseable stderr lines | Add an opt-in `--startup-json` stdout record. Keep the stderr lines for compatibility |
| Graceful SIGTERM: stop admission, cooperative cancel, bounded cleanup; stdin-close watch when supervised | No signal handler; SIGTERM drops in-flight requests | Handle SIGTERM/SIGINT with a drain deadline. Add a supervised-stdin watch so an orphan dies with JX |
| Versioned protocol + `/v1/capabilities` | `/props`, `--help` rule | See §7.1 |
| Component digests verified on load | None ("load models you trust") | Optional `--model-sha256` that refuses a mismatch. JX already knows each model's sha256 |
| Memory estimate from file headers, with its basis stated | None; `--fit` on roadmap | Estimate from GGUF tensor metadata + KV formula, exposed on `/props`. JX's `src/estimate.js` could then defer to the engine |
| Optimizations advertised only with a cited benchmark (`optimizations.json`) | No benchmarks at all | Apply the rule to speculative decoding, KV quantization and `--tick-tokens` defaults. It answers Onyx "Evidence" item 9 and JX #109 |
| mutual TLS network mode with fingerprint in `/v1/info` | Plain HTTP, proxy for TLS | Only if JX adopts remote engines (§7.2). Otherwise keep the proxy |
| Conformance suite that runs against any worker | Smoke script only | See §7.1 |
| `AGENTS.md`-free but spec, plan, decisions and compatibility docs | CONTRIBUTING + stability page, no AGENTS.md | Onyx is agent-maintained (changelog signed "Agent: Claude Code"). Adopt `AGENTS.md`/Spec-Driven Docs as JX did |

---

## 9. Family-level recommendations (all four)

### 9.1 A shared GPU agreement between JX Runtime and Media Runtime

Running JX and Media on one GPU is the expected setup for "my own applications". Today:
- JX evicts by its own memory estimate.
- Media has `--memory-budget-mb` per worker and no idle unload.
- Neither knows about the other, and both specs defer this.

**Proposal.** A small, explicit **GPU lease file or socket** per device, owned by neither runtime and
documented in both specs:
- Each runtime declares a reservation (MiB) and its current usage.
- Each may request that the other unload idle residents.
- Each can refuse a request that would exceed the device.

Keep it advisory and auditable. No runtime kills or reconfigures the other, which matches both
specs' "never kill an externally managed service" rule.

### 9.2 Shared conventions worth unifying now (cheap while the Media API is pre-1.0)

- **Error envelope.**
  - JX uses `{error:{code,message,type,correlationId,details.repair}}`.
  - Media uses `{error:{code,message,field},request_id}`.
  - Converge on one shape: Media's `field` plus JX's `repair` and `type`, keyed by `request_id`.
- **Request ID header.** Both should honour and echo `X-Request-Id`. JX and Onyx already do.
- **Discovery.** Both expose `/.well-known/<runtime>` with the same `schemaVersion`/`surfaces` shape.
- **Key format and pairing flow.** For example, `jxr_…` and `mdr_…` keys with the same permission
  model, so one application settings screen can pair with both runtimes.
- **Release infrastructure, version policy and changelog convention.** One pipeline (§5.7).
- **Governance.** One `AGENTS.md`/Spec-Driven Docs system across all four repositories.

### 9.3 Honesty about validation

All four repositories are largely agent-authored and have open "not validated on real hardware"
items:
- JX #108/#109
- Onyx's GPU paths "not exercised"
- the Media stack's entire GPU surface

A shared rule would help: **no feature counts as delivered in a README until a real-hardware row
exists in the compatibility record**. Until then it is "delivered (fixture-tested)", as Media
Engine's spec table already says.

---

## 10. Suggested sequencing for the Media pair

1. **Prove it (P0).**
   - CI in both repositories.
   - One RTX 4090 testbench run with real weights, recorded in both compatibility records.
   - First tagged releases through the shared release infrastructure.
   - Runtime R7, so it accepts `native-sdxl`.
2. **Make it usable by an application (P1, operator side).**
   - API keys with permissions, rate limits and an audit log.
   - Watched model folders that read a ComfyUI `models/` tree in place.
   - Prometheus metrics.
   - Guided-repair errors.
   - Managed-mode handshake.
   - systemd unit.
   - OpenAPI document plus drift test.
3. **Make it ComfyUI-capable (P1, generation side).** On the reference backend, in this order:
   1. the sampler/scheduler set
   2. resize/crop/pad/composite utility operations
   3. inpainting
   4. live previews
   5. one Flux-class DiT family and SD3.5
   6. ControlNet
   7. MP4 encoder extra
4. **Make it easy to install (P2).**
   - Managed Engine install: native archives pinned by SHA-256, a CUDA container image, a managed venv.
   - Opt-in pinned Hugging Face installs.
   - Converters for `.pth` and Kohya files.
   - Completion webhooks.
5. **Grow (P2–P3).**
   - DAG recipes.
   - IP-Adapter, embeddings and VAE swap.
   - Offload modes.
   - Execution caching.
   - The JX↔Media GPU agreement.
   - More video and audio families.
   - Only after that, further native ownership.

---

## 11. Decisions this review needs from you

1. **ComfyUI workflow import.** Keep it fully out of scope (current spec), or allow an *offline,
   out-of-process* subset translator plus a migration guide (§4.4)? This touches an explicit
   out-of-scope line in both specs.
2. **Utility image operations** (resize, crop, pad, composite). The current rule is "nothing is
   resized". The proposal keeps "nothing is resized *silently*" by making resizing an explicit
   recipe operation. Confirm this reading.
3. **MP4 encoder extra.** The open decision in `engine-spec.md` "Decisions and next steps". The
   recommendation is yes, optional and pinned.
4. **Ownership pacing.** Freeze new native/owned stages until P1 breadth and a GPU benchmark exist (§6.4)?
5. **Governance.** Adopt Spec-Driven Docs (`AGENTS.md`/`CLAUDE.md`) in the Media repositories and in
   Onyx Engine?
6. **GPU agreement.** Design the JX↔Media GPU lease now, or keep operator-set budgets (§9.1)?

---

## 12. Evidence and limitations

- **Read, not run.** No build, test, benchmark or GPU run was performed for this review. Test counts
  come from each repository's own changelog or QA report. Those counts disagree for JX: 2,912 passed
  in `docs/human-qa/QA_Report_2026-10-28.md` versus 2,618 in `.release/README.md`.
- **Unverified release state.** I could not see GitHub Releases or GHCR, so I could not confirm
  whether Onyx 0.7.0 or JX 1.5.10 artifacts are published.
- **ComfyUI from general knowledge.** Its API, folder layout and behaviour are described from
  general knowledge of the project, not from a checkout. Verify specifics, especially the license,
  against the upstream repository before relying on them.
- **Incidental drift noticed in JX Runtime.** Not fixed here:
  - `development-docs/productRoadmap.md` §3 still says version 1.5.1 and "not a double-click
    installer".
  - `docs/cli.md` omits `setup`, `update` and `folders`.
  - `docs/compatibility.md` names pin `b10612`, while `docs/engines.md` names `ollama-v0.32.9` as
    the managed default.
- **Incidental drift noticed in Onyx Engine:**
  - `.ai/qa.md` says the smoke suite skips on Windows, but CI runs it there.
  - `research_notes/serving_techniques.md` cites a removed `docs/architecture.md`.
