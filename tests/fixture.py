"""Explicit test-only backend. Not packaged and never selected by the production executable."""

import json
import struct
import time
import zlib
from pathlib import Path
from media_engine.backend import Cancelled
from media_engine.contracts import (
    DEFAULTS,
    LIMITS,
    PROTOCOL,
    SUPPORTED_PROTOCOLS,
    TASK_DEFAULTS,
    TASKS,
    digest,
    effective_steps,
    memory_estimate,
)


def fixture_safetensors(role):
    """A valid safetensors container holding a few zero bytes; never model weights."""
    header = json.dumps(
        {
            "__metadata__": {"fixture": "TEST FIXTURE ONLY - NOT MODEL WEIGHTS"},
            role + ".weight": {
                "dtype": "F32",
                "shape": [4, 2],
                "data_offsets": [0, 32],
            },
        }
    ).encode()
    return struct.pack("<Q", len(header)) + header + bytes(32)


def make_png(width, height):
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
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\x80\x60\x40" * width) * height))
        + chunk(b"IEND", b"")
    )


def make_bundle(root):
    root = Path(root)
    roles = {
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    }
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
    import hashlib

    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    bundle = {
        "schema_version": 1,
        "bundle_id": "fixture",
        "family": "sdxl-base",
        "pipeline": "text-to-image",
        "components": [
            {
                "role": r,
                "artifact_handle": "fixture",
                "digest": digest(
                    {k: v for k, v in files.items() if k.startswith(r + "/")}
                ),
                "tensor_format": "safetensors" if i < 4 else "json",
            }
            for i, r in enumerate(roles)
        ],
        "configuration": {"artifact_handle": "fixture", "files": files},
        "source": {"license": "test-only"},
    }
    return {**bundle, "bundle_digest": digest(bundle)}


class TestBackend:
    backend_id = (
        "diffusers-sdxl"  # Exercises reference protocol semantics, not inference.
    )
    ownership = "test-fixture"

    def __init__(self):
        self.generations = 0
        self.loads = 0
        self.loaded_estimate = None

    def capabilities(self):
        return {
            "protocol": PROTOCOL,
            "supported_protocols": list(SUPPORTED_PROTOCOLS),
            "worker_version": "0.2.0-test-fixture",
            "backend_id": self.backend_id,
            "ownership": self.ownership,
            "test_fixture": True,
            "ready": True,
            "device": "test-only",
            "device_available": True,
            "memory_budget_mb": None,
            "memory_estimate": self.loaded_estimate
            or {"value_mb": None, "source": "unavailable", "uncertainty": "unknown"},
            "defaults": DEFAULTS,
            "task_defaults": TASK_DEFAULTS,
            "load_defaults": {
                "precision": "float32",
                "attention": "sdpa",
                "offload": "none",
            },
            "limits": LIMITS,
            "precisions": ["float32"],
            "tasks": list(TASKS),
            "task_parameters": {
                "text-to-image": {"inputs": ["prompt", "negative_prompt"]},
                "image-to-image": {
                    "inputs": ["prompt", "negative_prompt", "image"],
                    "parameters": ["strength"],
                    "input_formats": ["image/png"],
                },
            },
            "model_families": ["sdxl-base"],
        }

    def estimate(self, root, options):
        self.loaded_estimate = memory_estimate(root, options["precision"], "test-only")
        return self.loaded_estimate

    def load(self, root, options):
        self.loads += 1
        time.sleep(0.02)

    def unload(self):
        return {"allocated_mb": 0, "reserved_mb": 0}

    def generate(self, request, destination, emit, cancelled, image_path=None):
        self.generations += 1
        if request["task"] == "image-to-image":
            # The fixture proves the handle resolves to the uploaded bytes; it does not sample.
            assert Path(image_path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        total = effective_steps(request["parameters"], request["task"])
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
        emit("stage", {"stage": "decoding"})
        png = make_png(request["parameters"]["width"], request["parameters"]["height"])
        emit("stage", {"stage": "saving"})
        Path(destination).write_bytes(png)
        return {"test_fixture": True, "effective_steps": total, "peak_allocated_mb": 0}
