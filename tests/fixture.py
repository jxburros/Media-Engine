"""Explicit test-only backend. Not packaged and never selected by the production executable."""

import json
import struct
import time
import zlib
from pathlib import Path
from media_engine.backend import Cancelled
from media_engine.contracts import DEFAULTS, digest


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
            (folder / "test.safetensors").write_bytes(
                b"TEST FIXTURE ONLY - NOT MODEL WEIGHTS"
            )
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

    def __init__(self):
        self.generations = 0
        self.loads = 0

    def capabilities(self):
        return {
            "protocol": "media-worker/1.0",
            "worker_version": "0.1.0-test-fixture",
            "backend_id": self.backend_id,
            "test_fixture": True,
            "ready": True,
            "device": "test-only",
            "device_available": True,
            "memory_budget_mb": None,
            "defaults": DEFAULTS,
            "load_defaults": {
                "precision": "float32",
                "attention": "sdpa",
                "offload": "none",
            },
            "precisions": ["float32"],
            "tasks": ["text-to-image"],
            "model_families": ["sdxl-base"],
        }

    def load(self, root, options):
        self.loads += 1
        time.sleep(0.02)

    def unload(self):
        return {"allocated_mb": 0, "reserved_mb": 0}

    def generate(self, request, destination, emit, cancelled):
        self.generations += 1
        if request["inputs"]["prompt"] == "uncancellable":
            time.sleep(3)
        emit("stage", {"stage": "encoding"})
        if request["inputs"]["prompt"] == "fail":
            raise RuntimeError("secret prompt and private path must never be exposed")
        emit("stage", {"stage": "sampling"})
        for step in range(request["parameters"]["steps"]):
            time.sleep(0.03)
            if cancelled.is_set():
                raise Cancelled()
            emit(
                "progress",
                {
                    "stage": "sampling",
                    "scope": "stage",
                    "completed_steps": step + 1,
                    "total_steps": request["parameters"]["steps"],
                },
            )
        emit("stage", {"stage": "decoding"})
        width, height = request["parameters"]["width"], request["parameters"]["height"]

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
        png += chunk(
            b"IDAT", zlib.compress((b"\0" + b"\x80\x60\x40" * width) * height)
        ) + chunk(b"IEND", b"")
        emit("stage", {"stage": "saving"})
        Path(destination).write_bytes(png)
        return {"test_fixture": True, "peak_allocated_mb": 0}
