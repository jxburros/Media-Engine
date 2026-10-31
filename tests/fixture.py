"""Explicit test-only backend. Not packaged and never selected by the production executable."""

import hashlib
import json
import os
import random
import struct
import time
import zlib
from pathlib import Path
from media_engine.backend import Cancelled, plan_load, task_parameters
from media_engine.contracts import (
    AUDIO_LIMITS,
    AUTO_OPTIONS,
    DEFAULTS,
    FALLBACKS,
    FAMILIES,
    FORMATS,
    LIMITS,
    PRECISION_BYTES,
    SUPPORTED_PROTOCOLS,
    TASK_DEFAULTS,
    TASKS,
    VIDEO_LIMITS,
    audio_facts,
    audio_length,
    digest,
    effective_steps,
    memory_estimate,
    minor,
    output_ids,
    video_facts,
)
from media_engine import audio, optimize
from media_engine import png as png_module
from media_engine import video
from media_engine.upscale import architecture, tile_count, validate_config

FIXTURE_VERSION = "1.0.0-test-fixture"
NOTICE = "TEST FIXTURE ONLY - NOT MODEL WEIGHTS"
# What the fixture advertises under capabilities.optimizations (1.3): every candidate but flash,
# which it lists as device_unsupported so callers see both shapes. No benchmark exists for a
# fixture; the citation says so and nothing about speed or quality is claimed.
FIXTURE_OPTIMIZATIONS = (("precision", "bfloat16"), ("quantization", "int8-weight-only"), ("compile", "inductor"))


def fixture_optimizations(pairs=FIXTURE_OPTIMIZATIONS):
    advertised, unadvertised = [], []
    for c in optimize.CANDIDATES:
        key = {"option": c["option"], "value": c["value"]}
        if (c["option"], c["value"]) not in pairs:
            unadvertised.append({**key, "reason": "device_unsupported"})
            continue
        record = {
            **key,
            "families": list(c["families"]),
            "device_class": {"kind": "test-fixture"},
            "benchmark": "test-fixture (no benchmark; protocol surface only)",
            "benchmark_sha256": "0" * 64,
            "improvement": "none claimed: the fixture does not run inference",
            "determinism": "fixture outputs depend on the seed only",
            "quality_note": "fixture outputs are flat colors, not inference",
            "effect": c["effect"],
            "adapters": c["adapters"],
        }
        if "precisions" in c:
            record["precisions"] = list(c["precisions"])
        advertised.append(record)
    return advertised, unadvertised


def zero_safetensors(shapes):
    """A valid safetensors container of zero-filled F32 tensors; never model weights."""
    header, offset = {"__metadata__": {"fixture": NOTICE}}, 0
    for name, shape in shapes.items():
        size = 4
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + bytes(offset)


def random_safetensors(shapes, seed):
    """A valid safetensors container of small seeded pseudo-random F32 values; never model weights."""
    generator = random.Random(seed)
    header, offset, data = {"__metadata__": {"fixture": NOTICE}}, 0, []
    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        data.append(struct.pack(f"<{count}f", *(generator.uniform(-0.1, 0.1) for _ in range(count))))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + 4 * count]}
        offset += 4 * count
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(data)


def fixture_safetensors(role):
    """A valid safetensors container holding a few zero bytes; never model weights."""
    return zero_safetensors({role + ".weight": [4, 2]})


def make_png(width, height, color=(0x80, 0x60, 0x40)):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    return (
        png
        + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * width) * height))
        + chunk(b"IEND", b"")
    )


def seed_color(seed):
    """Distinct, reproducible fixture pixels per seed (so output k equals a run at seed + k − 1)."""
    value = hashlib.sha256(str(seed).encode()).digest()
    return value[0], value[1], value[2]


def file_digests(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def describe_bundle(root, handle, bundle_id, family="sdxl-base", source=None):
    """The registration document for an existing bundle directory, as a supervisor builds it."""
    from media_engine.contracts import role_files

    root = Path(root)
    files = file_digests(root)
    spec = FAMILIES[family]
    bundle = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "family": family,
        "pipeline": spec["pipeline"],
        "components": [
            {
                "role": r,
                "artifact_handle": handle,
                "digest": digest(role_files(family, r, files)),
                "tensor_format": "safetensors" if r in spec["weight_roles"] else "json",
            }
            for r in spec["roles"]
        ],
        "configuration": {"artifact_handle": handle, "files": files},
        "source": source or {"license": "test-only"},
    }
    if family != "sdxl-base":
        bundle = {**bundle, "format": spec["format"]}
    return {**bundle, "bundle_digest": digest(bundle)}


def make_adapter(root, adapter_id="fixture-lora", handle="fixture-lora"):
    """A PEFT-layout LoRA safetensors file with tiny zero tensors; never trained weights."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tensors = {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": [2, 4],
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight": [4, 2],
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": [2, 4],
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": [4, 2],
    }
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 4 * shape[0] * shape[1]
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    (root / "adapter.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(offset)
    )
    record = {
        "schema_version": 1,
        "adapter_id": adapter_id,
        "artifact_handle": handle,
        "files": file_digests(root),
        "source": {"license": "test-only"},
    }
    return {**record, "digest": digest(record)}


def make_bundle(root, handle="fixture", bundle_id="fixture", conversion=None, source=None):
    """An sdxl-base directory of zero-byte stand-in weights; optionally with conversion.json."""
    root = Path(root)
    roles = {r: list(c) for r, c in FAMILIES["sdxl-base"]["approved_classes"].items()}
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline", **roles})
    )
    for i, role in enumerate(roles):
        folder = root / role
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text("{}")
        if i < 4:
            (folder / "test.safetensors").write_bytes(fixture_safetensors(role))
    (root / "scheduler/scheduler_config.json").write_text(
        '{"_class_name":"EulerDiscreteScheduler"}'
    )
    if conversion is not None:
        (root / "conversion.json").write_text(json.dumps(conversion, indent=2, sort_keys=True))
    return describe_bundle(root, handle, bundle_id, source=source)


ESRGAN_CONFIG = {
    "architecture": "rrdbnet",
    "scale": 2,
    "in_channels": 3,
    "out_channels": 3,
    "num_feat": 4,
    "num_block": 1,
    "num_grow_ch": 2,
}


def make_esrgan_bundle(root, handle="upscaler", bundle_id="fixture-upscaler", **config):
    """A single-file-safetensors esrgan bundle: zero weights shaped for the configured RRDBNet."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = {**ESRGAN_CONFIG, **config}
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    shapes = architecture(validate_config(config))
    (root / "network.safetensors").write_bytes(zero_safetensors(shapes))
    return describe_bundle(root, handle, bundle_id, "esrgan")


# 64×64 lets supervisors' tests stay small; the reference sizes are listed too.
SVD_SIZES = [[64, 64], [1024, 576], [576, 1024]]


def make_svd_bundle(root, handle="video", bundle_id="fixture-video", frames=2, sizes=SVD_SIZES):
    """An svd directory with tiny random stand-in weights, `num_frames` 2 and declared sizes.

    The weights are a few seeded random floats per role, not a network: the fixture backend never
    reads them, and the real svd bundle for inference tests is built in test_inference.py.
    """
    root = Path(root)
    roles = {r: list(c) for r, c in FAMILIES["svd"]["approved_classes"].items()}
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableVideoDiffusionPipeline", **roles}, indent=2, sort_keys=True)
    )
    configs = {
        "unet": ("config.json", {"_class_name": "UNetSpatioTemporalConditionModel", "in_channels": 8, "num_frames": frames}),
        "vae": ("config.json", {"_class_name": "AutoencoderKLTemporalDecoder", "block_out_channels": [32, 64]}),
        "image_encoder": ("config.json", {"model_type": "clip_vision_model"}),
        "feature_extractor": ("preprocessor_config.json", {"image_processor_type": "CLIPImageProcessor"}),
        "scheduler": ("scheduler_config.json", {"_class_name": "EulerDiscreteScheduler"}),
    }
    weights = {"unet": "diffusion_pytorch_model", "vae": "diffusion_pytorch_model", "image_encoder": "model"}
    for index, (role, (name, config)) in enumerate(configs.items()):
        folder = root / role
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(json.dumps(config, indent=2, sort_keys=True))
        if role in weights:
            (folder / (weights[role] + ".safetensors")).write_bytes(
                random_safetensors({role + ".weight": [4, 2]}, index)
            )
    if sizes is not None:
        (root / "video.json").write_text(json.dumps({"sizes": sizes}))
    return describe_bundle(root, handle, bundle_id, "svd")


# 8 kHz stereo with a 256-sample latent frame keeps fixture clips small; 1488 latent frames give
# 47.616 s, so the default range is the reference bundle's 1–47 s.
AUDIO_CONFIG = {"sampling_rate": 8000, "audio_channels": 2, "downsampling_ratios": [2, 4, 4, 8], "sample_size": 1488}


def make_audio_bundle(root, handle="audio", bundle_id="fixture-audio", duration=None, **config):
    """A stable-audio-open directory with tiny random stand-in weights and the audio facts above.

    As with make_svd_bundle the weights are a few seeded random floats per role, not a network;
    the real bundle for inference tests is built in test_inference.py. `duration` writes an
    audio.json range; otherwise the range follows from the configs.
    """
    root = Path(root)
    config = {**AUDIO_CONFIG, **config}
    roles = {r: list(c) for r, c in FAMILIES["stable-audio-open"]["approved_classes"].items()}
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableAudioPipeline", **roles}, indent=2, sort_keys=True)
    )
    configs = {
        "transformer": ("config.json", {"_class_name": "StableAudioDiTModel", "in_channels": 3, "sample_size": config["sample_size"]}),
        "vae": ("config.json", {"_class_name": "AutoencoderOobleck", "sampling_rate": config["sampling_rate"], "audio_channels": config["audio_channels"], "downsampling_ratios": config["downsampling_ratios"]}),
        "text_encoder": ("config.json", {"model_type": "t5"}),
        "tokenizer": ("tokenizer.json", {"model": {"type": "WordLevel"}}),
        "projection_model": ("config.json", {"_class_name": "StableAudioProjectionModel"}),
        "scheduler": ("scheduler_config.json", {"_class_name": "CosineDPMSolverMultistepScheduler"}),
    }
    weights = {"transformer": "diffusion_pytorch_model", "vae": "diffusion_pytorch_model", "text_encoder": "model", "projection_model": "diffusion_pytorch_model"}
    for index, (role, (name, settings)) in enumerate(configs.items()):
        folder = root / role
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(json.dumps(settings, indent=2, sort_keys=True))
        if role in weights:
            (folder / (weights[role] + ".safetensors")).write_bytes(
                random_safetensors({role + ".weight": [4, 2]}, 10 + index)
            )
    if duration is not None:
        (root / "audio.json").write_text(json.dumps({"duration_seconds": duration}))
    return describe_bundle(root, handle, bundle_id, "stable-audio-open")


def fixture_tone(seed, samples, rate, channels):
    """A seeded sine tone per channel (not inference): reproducible from the seed, never silent."""
    import math

    frequency = 110 + seed % 440
    return audio.pcm16(
        0.25 * math.sin(2 * math.pi * frequency * (k / rate) + c)
        for k in range(samples)
        for c in range(channels)
    )


def make_checkpoint(root):
    """A bare single-file SDXL-style checkpoint: one safetensors file, no model_index/config."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint.safetensors").write_bytes(
        zero_safetensors(
            {
                "model.diffusion_model.input_blocks.0.0.weight": [4, 4, 3, 3],
                "first_stage_model.encoder.conv_in.weight": [4, 3, 3, 3],
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": [8, 4],
                "conditioner.embedders.1.model.token_embedding.weight": [8, 4],
            }
        )
    )
    return root / "checkpoint.safetensors"


def make_converted(
    root, handle="converted", bundle_id="fixture-converted", reference=None, provenance_in_source=True
):
    """An sdxl-base bundle carrying conversion provenance, as `media-engine convert` writes it."""
    checkpoint = zero_safetensors({"model.diffusion_model.input_blocks.0.0.weight": [4, 2]})
    record = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "reference_bundle_digest": digest(file_digests(reference)) if reference else "sha256:" + "0" * 64,
        "converter": {
            "media_engine": "0.4.0",
            "diffusers": "test-fixture",
            "torch": "test-fixture",
            "safetensors": "test-fixture",
        },
        "precision": "preserved",
        "mapped_keys": {"unet": 1, "vae": 1, "text_encoder": 1, "text_encoder_2": 1},
        "unmapped_keys": [],
        "ignored_keys": [],
    }
    # 1.2 registrations carry the provenance as source.conversion (Runtime R1.4).
    provenance = {k: record[k] for k in ("source_sha256", "reference_bundle_digest", "converter", "mapped_keys")}
    source = {"license": "test-only", **({"conversion": provenance} if provenance_in_source else {})}
    return make_bundle(root, handle, bundle_id, conversion=record, source=source)


class TestBackend:
    backend_id = (
        "diffusers-sdxl"  # Exercises reference protocol semantics, not inference.
    )
    ownership = "test-fixture"

    def __init__(
        self,
        protocols=SUPPORTED_PROTOCOLS,
        device_unavailable=False,
        precisions=("float32",),
        memory_budget_mb=None,
        estimate_mb=None,
        output_side=None,
        device_index=0,
        host_id="fixture-host",
        optimizations=FIXTURE_OPTIMIZATIONS,
    ):
        self.generations = 0
        self.loads = 0
        self.loaded_estimate = None
        self.protocols = tuple(protocols)
        self.device_unavailable = device_unavailable
        self.precision_list = list(precisions)
        self.memory_budget_mb = memory_budget_mb
        self.estimate_mb = estimate_mb
        self.output_side = output_side
        self.device_index = device_index
        self.host_id = host_id
        self.device = "test-only"
        self.run_device = self.device
        self.applied_fallbacks = []
        self.family = None
        self.scale = None
        self.default_frames = None
        self.audio_facts = None
        self.loaded_options = None
        self.optimizations = tuple(optimizations)

    @property
    def level(self):
        """The highest minor served: 1 reproduces the 0.3.0 surface, 2 the 0.5.0 surface."""
        return minor(self.protocols[-1])

    @property
    def current(self):
        return self.level >= 2

    def capabilities(self):
        caps = {
            "protocol": self.protocols[-1],
            "supported_protocols": list(self.protocols),
            "worker_version": FIXTURE_VERSION,
            "backend_id": self.backend_id,
            "ownership": self.ownership,
            "test_fixture": True,
            "ready": not self.device_unavailable or self.run_device == "cpu",
            "missing_dependencies": [],
            "device": self.device,
            "device_available": not self.device_unavailable,
            "memory_budget_mb": self.memory_budget_mb,
            "memory_estimate": self.loaded_estimate
            or {"value_mb": None, "source": "unavailable", "uncertainty": "unknown"},
            "defaults": DEFAULTS,
            "task_defaults": {"image-to-image": TASK_DEFAULTS["image-to-image"]},
            "load_defaults": {
                "precision": self.precision_list[0],
                "attention": "sdpa",
                "offload": "none",
            },
            "limits": {k: LIMITS[k] for k in ("width", "height", "steps", "guidance", "strength", "adapter_scale", "size_multiple")},
            "precisions": self.precision_list,
            "tasks": ["text-to-image", "image-to-image"],
            "task_parameters": {
                "text-to-image": {"inputs": ["prompt", "negative_prompt"]},
                "image-to-image": {
                    "inputs": ["prompt", "negative_prompt", "image"],
                    "parameters": ["strength"],
                    "input_formats": ["image/png"],
                },
            },
            "model_families": ["sdxl-base"],
            "adapters": {
                "formats": ["peft-lora"],
                "targets": ["unet", "text_encoder", "text_encoder_2"],
                "max_adapters": 4,
                "scale": [0, 2],
                "identity": "load",
            },
        }
        if not self.current:
            return caps  # Pinned to 1.1: exactly the 0.3.0 fixture's surface.
        # Pinned to 1.2: exactly the 0.5.0 fixture's surface (no svd or stable-audio-open, no
        # video or audio limits or tasks).
        families = {n: f for n, f in FAMILIES.items() if f["since"] <= self.level}
        tasks = [t for t in TASKS if any(t in f["tasks"] for f in families.values())]
        newer = VIDEO_LIMITS + AUDIO_LIMITS
        limits = {k: v for k, v in LIMITS.items() if self.level >= 3 or k not in newer}
        if self.level >= 3:
            limits["video_sizes"] = SVD_SIZES  # what the fixture's video bundle serves
        if self.output_side:
            limits["output_side"] = self.output_side
        caps.update(
            task_defaults={k: v for k, v in TASK_DEFAULTS.items() if k in tasks},
            limits=limits,
            tasks=tasks,
            task_parameters=task_parameters(tasks),
            model_families=list(families),
            loaders=[
                {
                    "format": spec["format"],
                    "families": [name],
                    "roles": list(spec["roles"]),
                    "tasks": list(spec["tasks"]),
                    "ownership": "owned" if name == "esrgan" else self.ownership,
                }
                for name, spec in families.items()
            ],
            formats=list(FORMATS),
            auto_options=AUTO_OPTIONS,
            fallbacks=FALLBACKS,
        )
        caps["adapters"] = {**caps["adapters"], "families": ["sdxl-base"]}
        if self.level >= 3:
            caps["missing_task_dependencies"] = {}
            advertised, unadvertised = fixture_optimizations(self.optimizations)
            caps["optimizations"] = advertised
            caps["unadvertised_optimizations"] = unadvertised
        return caps

    def device_identity(self):
        return {
            "device": self.device,
            "index": self.device_index,
            "name": "test-fixture-device",
            "total_memory_mb": self.memory_budget_mb or 24576,
            "host_id": hashlib.sha256(self.host_id.encode()).hexdigest(),
            "host_id_source": "fixture",
            "pid": os.getpid(),
        }

    def compute_estimate(self, root, options, adapters=(), family="sdxl-base", device=None):
        estimate = memory_estimate(root, options["precision"], device or "test-only", family)
        if self.estimate_mb is not None:
            # A settable estimate lets protocol tests exercise budget-driven fallbacks.
            estimate["value_mb"] = self.estimate_mb * PRECISION_BYTES[options["precision"]] // 4
            estimate["source"] = "test_fixture_override"
        estimate["adapters"] = [
            {"adapter_id": r["adapter_id"], "parameters": layout["parameters"]}
            for r, _a, _w, _s, layout in adapters
        ]
        return estimate

    def estimate(self, root, options, adapters=(), family="sdxl-base", device=None):
        self.loaded_estimate = self.compute_estimate(root, options, adapters, family, device)
        return self.loaded_estimate

    def plan(self, root, options, adapters=(), family="sdxl-base", authorized=()):
        if options["precision"] != "auto" and not authorized:
            return {"options": dict(options), "resolutions": [], "applied_fallbacks": [], "device": self.device}
        return plan_load(
            options,
            authorized,
            self.device,
            not self.device_unavailable,
            self.precision_list,
            self.memory_budget_mb,
            lambda p: self.compute_estimate(root, {**options, "precision": p}, adapters, family)[
                "value_mb"
            ],
        )

    def load(self, root, options, adapters=(), family="sdxl-base", device=None):
        self.loads += 1
        self.adapters = [(r["adapter_id"], scale) for r, _a, _w, scale, _l in adapters]
        self.family, self.run_device, self.loaded_options = family, device or self.device, options
        self.scale = (
            json.loads((Path(root) / "config.json").read_text())["scale"]
            if family == "esrgan"
            else None
        )
        self.default_frames = video_facts(root)["default_frames"] if family == "svd" else None
        self.audio_facts = audio_facts(root) if family == "stable-audio-open" else None
        time.sleep(0.02)

    def unload(self):
        self.family, self.scale, self.run_device, self.applied_fallbacks = None, None, self.device, []
        self.default_frames = self.audio_facts = None
        return {"allocated_mb": 0, "reserved_mb": 0}

    def generate(self, request, directory, emit, cancelled, image_path=None):
        self.generations += 1
        if request["task"] == "upscale":
            return self.upscale(request, directory, emit, cancelled, image_path)
        if request["task"] == "image-to-video":
            return self.generate_video(request, directory, emit, cancelled, image_path)
        if request["task"] == "text-to-audio":
            return self.generate_audio(request, directory, emit, cancelled)
        if request["task"] == "image-to-image":
            # The fixture proves the handle resolves to the uploaded bytes; it does not sample.
            assert Path(image_path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        params = request["parameters"]
        total = effective_steps(params, request["task"])
        if request["inputs"]["prompt"] == "uncancellable":
            time.sleep(3)
        emit("stage", {"stage": "encoding"})
        if request["inputs"]["prompt"] == "fail":
            raise RuntimeError("secret prompt and private path must never be exposed")
        emit("stage", {"stage": "sampling"})
        for step in range(total):
            time.sleep(0.03)
            if cancelled.is_set():
                raise Cancelled()
            emit(
                "progress",
                {
                    "stage": "sampling",
                    "scope": "stage",
                    "completed_steps": step + 1,
                    "total_steps": total,
                },
            )
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "decoding"})
        time.sleep(0.02)
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "saving"})
        records = []
        ids = output_ids(request["task"], params)
        if request["inputs"]["prompt"] == "partial batch":
            ids = ids[:-1]  # Lets tests prove a missing output fails the execution by name.
        for k, output_id in enumerate(ids):
            path = Path(directory) / (output_id + ".png")
            path.write_bytes(make_png(params["width"], params["height"], seed_color(params["seed"] + k)))
            records.append(
                {
                    "output_id": output_id,
                    "path": path,
                    "mime_type": "image/png",
                    "width": params["width"],
                    "height": params["height"],
                }
            )
        return records, {"test_fixture": True, "effective_steps": total, "peak_allocated_mb": 0}

    def upscale(self, request, directory, emit, cancelled, image_path):
        params = {**TASK_DEFAULTS["upscale"], **request["parameters"]}
        width, height = png_module.header(Path(image_path).read_bytes())[:2]
        assert (width, height) == (params["width"], params["height"])
        total = tile_count(width, height, params["tile"])
        emit("stage", {"stage": "upscaling"})
        for index in range(total):
            if cancelled.is_set():
                raise Cancelled()
            time.sleep(0.03)
            emit(
                "progress",
                {
                    "stage": "upscaling",
                    "scope": "stage",
                    "unit": "tiles",
                    "completed_steps": index + 1,
                    "total_steps": total,
                },
            )
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "saving"})
        path = Path(directory) / "image.png"
        path.write_bytes(make_png(width * self.scale, height * self.scale, (0x40, 0x60, 0x80)))
        record = {
            "output_id": "image",
            "path": path,
            "mime_type": "image/png",
            "width": width * self.scale,
            "height": height * self.scale,
        }
        return [record], {"test_fixture": True, "tiles": total, "peak_allocated_mb": 0}

    def generate_video(self, request, directory, emit, cancelled, image_path):
        """Real stages, progress, checkpoints and archive; frames are flat colors, not inference."""
        params = {**TASK_DEFAULTS["image-to-video"], **request["parameters"]}
        width, height = png_module.header(Path(image_path).read_bytes())[:2]
        assert (width, height) == (params["width"], params["height"])
        frames = params.get("frames", self.default_frames)
        emit("stage", {"stage": "encoding"})
        time.sleep(0.02)
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "sampling"})
        for step in range(params["steps"]):
            time.sleep(0.03)
            if cancelled.is_set():
                raise Cancelled()
            emit(
                "progress",
                {"stage": "sampling", "scope": "stage", "completed_steps": step + 1, "total_steps": params["steps"]},
            )
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "decoding"})
        pngs = []
        for start in range(0, frames, params["decode_chunk"]):
            if cancelled.is_set():
                raise Cancelled()
            time.sleep(0.02)
            for k in range(start, min(start + params["decode_chunk"], frames)):
                # Frame k is reproducible from seed + k, like output k of a text-to-image batch.
                pngs.append(png_module.write(width, height, bytes(seed_color(params["seed"] + k)) * (width * height)))
            emit(
                "progress",
                {"stage": "decoding", "scope": "stage", "unit": "frames", "completed_steps": len(pngs), "total_steps": frames},
            )
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "packaging"})
        archive = Path(directory) / "frames.zip"
        video.write_archive(archive, pngs, params["fps"], width, height)
        emit("stage", {"stage": "saving"})
        poster = Path(directory) / "poster.png"
        poster.write_bytes(pngs[0])
        records = [
            {"output_id": "frames", "path": archive, "mime_type": video.MIME_TYPE, "width": width, "height": height, "frames": frames, "fps": params["fps"]},
            {"output_id": "poster", "path": poster, "mime_type": "image/png", "width": width, "height": height},
        ]
        return records, {"test_fixture": True, "effective_steps": params["steps"], "frames": frames, "peak_allocated_mb": 0}

    def generate_audio(self, request, directory, emit, cancelled):
        """Real stages, progress, checkpoints and WAV; the samples are a seeded tone, not inference."""
        params = {**TASK_DEFAULTS["text-to-audio"], **request["parameters"]}
        facts = self.audio_facts
        length = audio_length(params["duration_seconds"], facts)
        emit("stage", {"stage": "encoding"})
        time.sleep(0.02)
        if request["inputs"]["prompt"] == "fail":
            raise RuntimeError("secret prompt and private path must never be exposed")
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "sampling"})
        for step in range(params["steps"]):
            time.sleep(0.03)
            if cancelled.is_set():
                raise Cancelled()
            emit(
                "progress",
                {"stage": "sampling", "scope": "stage", "completed_steps": step + 1, "total_steps": params["steps"]},
            )
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "decoding"})
        time.sleep(0.02)
        pcm = fixture_tone(params["seed"], length["samples"], facts["sample_rate"], facts["channels"])
        if cancelled.is_set():
            raise Cancelled()
        emit("stage", {"stage": "saving"})
        path = Path(directory) / "audio.wav"
        audio.write_wav(path, pcm, facts["sample_rate"], facts["channels"])
        record = {
            "output_id": "audio",
            "path": path,
            "mime_type": audio.MIME_TYPE,
            "sample_rate": facts["sample_rate"],
            "channels": facts["channels"],
            "duration_seconds": length["effective_duration_seconds"],
        }
        return [record], {
            "test_fixture": True,
            "effective_steps": params["steps"],
            "latent_frames": length["latent_frames"],
            "samples": length["samples"],
            "peak_allocated_mb": 0,
        }
