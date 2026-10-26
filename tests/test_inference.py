"""Real CPU inference through both backends on a tiny random-weight SDXL-shaped bundle.

Skipped unless the inference extra is installed. The bundle is architecturally SDXL base
(dual CLIP conditioning, time-id embedding, Euler scheduler, KL VAE) with tiny dimensions and
random weights, so it proves the owned pipeline and the reference pipeline execute the same
protocol end to end; it says nothing about image quality or hardware compatibility.
"""

import hashlib
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

from fixture import make_png
from media_engine.contracts import DEFAULTS, PROTOCOL, digest

HAVE_INFERENCE = all(
    importlib.util.find_spec(m) is not None
    for m in ("torch", "diffusers", "transformers", "safetensors", "PIL")
)


def tiny_sdxl_bundle(root):
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from diffusers import StableDiffusionXLPipeline
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    torch.manual_seed(0)
    root = Path(root)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=1,
        projection_class_embeddings_input_dim=80,
        cross_attention_dim=64,
        norm_num_groups=32,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
        sample_size=64,
        norm_num_groups=32,
    )
    config = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=3,
        pad_token_id=1,
        vocab_size=64,
        hidden_act="gelu",
        projection_dim=32,
        max_position_embeddings=77,
    )
    vocabulary = ["<|startoftext|>", "<|endoftext|>"] + [
        token + "</w>" for token in ("a", "test", "portrait", "window", "light")
    ]
    vocabulary += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    vocabulary += [chr(c) + "</w>" for c in range(ord("a"), ord("z") + 1)]
    vocabulary = vocabulary[:64]
    tokenizer_dir = root / "tok"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "vocab.json").write_text(
        json.dumps({t: i for i, t in enumerate(vocabulary)})
    )
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\nt e\nte s\ntes t</w>\n")
    tokenizer = CLIPTokenizer(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.txt"),
        model_max_length=77,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    pipe = StableDiffusionXLPipeline(
        unet=unet,
        vae=vae,
        text_encoder=CLIPTextModel(config),
        text_encoder_2=CLIPTextModelWithProjection(config),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer,
        scheduler=EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
            steps_offset=1,
        ),
        add_watermarker=False,
    )
    bundle_dir = root / "model"
    pipe.save_pretrained(str(bundle_dir), safe_serialization=True)
    # The tokenizer vocabulary is a test artifact; the engine only reads JSON/text here.
    files = {
        str(p.relative_to(bundle_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bundle_dir.rglob("*")
        if p.is_file()
    }
    roles = [
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    ]
    bundle = {
        "schema_version": 1,
        "bundle_id": "tiny-sdxl",
        "family": "sdxl-base",
        "pipeline": "text-to-image",
        "components": [
            {
                "role": r,
                "artifact_handle": "tiny",
                "digest": digest({k: v for k, v in files.items() if k.startswith(r + "/")}),
                "tensor_format": "safetensors" if i < 4 else "json",
            }
            for i, r in enumerate(roles)
        ],
        "configuration": {"artifact_handle": "tiny", "files": files},
        "source": {"license": "test-only random weights"},
    }
    return bundle_dir, {**bundle, "bundle_digest": digest(bundle)}


@unittest.skipUnless(HAVE_INFERENCE, "inference extra is not installed")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_dir, cls.bundle = tiny_sdxl_bundle(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def worker(self, backend_id):
        from media_engine.backend import backend_class
        from media_engine.server import Worker

        backend = backend_class(backend_id)("cpu")
        return Worker(
            self.root / "work" / backend_id, backend, {"tiny": str(self.model_dir)}
        )

    def finish(self, worker, operation=None, execution=None):
        while True:
            if operation and worker.operations[operation]["state"] != "running":
                return worker.operations[operation]
            if execution and worker.snapshot(execution)["state"] != "running":
                return worker.snapshot(execution)
            time.sleep(0.02)

    def request(self, attempt, task="text-to-image", image=None, guidance=6):
        body = {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": task,
            "inputs": {"prompt": "a test portrait", "negative_prompt": "window"},
            "parameters": {
                **DEFAULTS,
                "seed": 7,
                "width": 64,
                "height": 64,
                "steps": 3,
                "guidance": guidance,
            },
        }
        if task == "image-to-image":
            body["inputs"]["image"] = image
            body["parameters"]["strength"] = 0.7
        return body

    def exercise(self, backend_id):
        from PIL import Image

        worker = self.worker(backend_id)
        caps = worker.capabilities()
        self.assertTrue(caps["ready"], caps)
        self.assertEqual(caps["backend_id"], backend_id)
        op = worker.control(
            "load",
            {"protocol": PROTOCOL, "bundle": self.bundle, "options": caps["load_defaults"]},
        )
        result = self.finish(worker, operation=op["operation_id"])
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual(result["memory_estimate"]["precision"], "float32")
        self.assertGreater(result["memory_estimate"]["value_mb"], 0)
        snap = self.finish(
            worker, execution=worker.submit(self.request("attempt_t2i"))["execution_id"]
        )
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        self.assertEqual(snap["progress"]["total_steps"], 3)
        self.assertEqual(
            [e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"],
            ["encoding", "sampling", "decoding", "saving"],
        )
        png = worker.directory / (snap["execution_id"] + ".png")
        with Image.open(png) as image:
            self.assertEqual(image.size, (64, 64))
        self.assertEqual(
            hashlib.sha256(png.read_bytes()).hexdigest(), snap["outputs"][0]["sha256"]
        )
        # Same seed and settings reproduce bytes on the same CPU/software stack.
        again = self.finish(
            worker, execution=worker.submit(self.request("attempt_again"))["execution_id"]
        )
        self.assertEqual(again["outputs"][0]["sha256"], snap["outputs"][0]["sha256"])
        # Guidance ≤ 1 disables classifier-free guidance without changing the contract.
        plain = self.finish(
            worker,
            execution=worker.submit(self.request("attempt_plain", guidance=1))[
                "execution_id"
            ],
        )
        self.assertEqual(plain["state"], "succeeded", plain.get("error"))
        record = worker.store_input(make_png(64, 64))
        edited = self.finish(
            worker,
            execution=worker.submit(
                self.request("attempt_i2i", "image-to-image", record["input_id"])
            )["execution_id"],
        )
        self.assertEqual(edited["state"], "succeeded", edited.get("error"))
        self.assertEqual(edited["effective_settings"]["effective_steps"], 2)
        self.assertEqual(edited["measurements"]["effective_steps"], 2)
        self.assertEqual(edited["progress"]["total_steps"], 2)
        cancelled = worker.submit(self.request("attempt_cancel"))
        worker.cancel(cancelled["execution_id"])
        self.assertEqual(
            self.finish(worker, execution=cancelled["execution_id"])["state"], "cancelled"
        )
        unload = worker.control("unload", {"protocol": PROTOCOL})
        self.assertEqual(
            self.finish(worker, operation=unload["operation_id"])["state"], "succeeded"
        )
        self.assertIsNone(worker.loaded)
        return {
            "text-to-image": (worker.directory / (snap["execution_id"] + ".png")).read_bytes(),
            "image-to-image": (
                worker.directory / (edited["execution_id"] + ".png")
            ).read_bytes(),
        }

    def test_reference_backend(self):
        self.exercise("diffusers-sdxl")

    def test_owned_pipeline_backend(self):
        self.exercise("owned-sdxl")

    def test_owned_pipeline_preserves_reference_semantics(self):
        """Same seed and settings: the owned loop reproduces the reference pipeline's pixels.

        Text-to-image is byte-identical on this CPU float32 stack; image-to-image agrees to within
        one 8-bit level after PNG rounding. This is evidence of matching request semantics, not a
        cross-hardware guarantee.
        """
        import io

        import numpy
        from PIL import Image

        reference = self.exercise("diffusers-sdxl")
        owned = self.exercise("owned-sdxl")
        self.assertEqual(reference["text-to-image"], owned["text-to-image"])
        difference = numpy.abs(
            numpy.asarray(Image.open(io.BytesIO(reference["image-to-image"])), dtype=float)
            - numpy.asarray(Image.open(io.BytesIO(owned["image-to-image"])), dtype=float)
        )
        self.assertLessEqual(difference.max(), 2)


if __name__ == "__main__":
    unittest.main()
