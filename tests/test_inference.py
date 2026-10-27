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

from fixture import describe_bundle, make_png
from media_engine import png
from media_engine.contracts import DEFAULTS, PROTOCOL, digest

# The owned pipeline needs only the inference-owned extra; the reference backend needs the full set.
HAVE_OWNED = all(
    importlib.util.find_spec(m) is not None
    for m in ("torch", "diffusers", "transformers", "safetensors")
)
HAVE_INFERENCE = HAVE_OWNED and all(
    importlib.util.find_spec(m) is not None for m in ("PIL", "accelerate", "peft", "numpy")
)


def pixels(data):
    """Decoded RGB bytes: file bytes differ between PNG writers, pixels must not."""
    return png.read_rgb(data)


def difference(a, b):
    a, b = pixels(a), pixels(b)
    assert a[:2] == b[:2]
    return max(abs(x - y) for x, y in zip(a[2], b[2]))


def mean_difference(a, b):
    a, b = pixels(a)[2], pixels(b)[2]
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


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
    return bundle_dir, describe_bundle(bundle_dir, "tiny", "tiny-sdxl", source={"license": "test-only random weights"})


def tiny_lora(root, unet_dir, rank=4):
    """A random PEFT-layout LoRA touching two UNet attention projections and one CLIP projection."""
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file

    torch.manual_seed(1)
    unet = UNet2DConditionModel.from_pretrained(str(unet_dir), subfolder="unet")
    tensors = {}
    chosen = [n for n, m in unet.named_modules() if n.endswith("attn1.to_q") or n.endswith("attn2.to_v")][:2]
    for name in chosen:
        layer = unet.get_submodule(name)
        tensors[f"unet.{name}.lora_A.weight"] = torch.randn(rank, layer.in_features) * 0.5
        tensors[f"unet.{name}.lora_B.weight"] = torch.randn(layer.out_features, rank) * 0.5
    tensors["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"] = torch.randn(rank, 32) * 0.5
    tensors["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight"] = torch.randn(32, rank) * 0.5
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "adapter.safetensors"))
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    record = {
        "schema_version": 1,
        "adapter_id": "tiny-lora",
        "artifact_handle": "tiny-lora",
        "files": files,
        "source": {"license": "test-only random weights"},
    }
    return {**record, "digest": digest(record)}


def tiny_esrgan_bundle(root, scale=2):
    """A random-weight RRDBNet in the esrgan single-file layout (Real-ESRGAN key names)."""
    import torch
    from safetensors.torch import save_file

    from fixture import ESRGAN_CONFIG
    from media_engine import upscale

    torch.manual_seed(3)
    config = {**ESRGAN_CONFIG, "scale": scale, "num_feat": 8, "num_grow_ch": 4}
    network = upscale.build(upscale.validate_config(config))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in network.state_dict().items()}, str(root / "network.safetensors"))
    (root / "config.json").write_text(json.dumps(config))
    return describe_bundle(root, "tiny-upscaler", "tiny-upscaler", "esrgan")


class Harness:
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_dir, cls.bundle = tiny_sdxl_bundle(cls.root)
        cls.adapter = tiny_lora(cls.root / "lora", cls.model_dir)
        cls.upscaler = tiny_esrgan_bundle(cls.root / "upscaler")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def worker(self, backend_id):
        from media_engine.backend import backend_class
        from media_engine.server import Worker

        backend = backend_class(backend_id)("cpu")
        return Worker(
            self.root / "work" / backend_id,
            backend,
            {
                "tiny": str(self.model_dir),
                "tiny-lora": str(self.root / "lora"),
                "tiny-upscaler": str(self.root / "upscaler"),
            },
        )

    def finish(self, worker, operation=None, execution=None):
        while True:
            if operation and worker.operations[operation]["state"] != "running":
                return worker.operations[operation]
            if execution and worker.snapshot(execution)["state"] != "running":
                return worker.snapshot(execution)
            time.sleep(0.02)

    def output(self, worker, snap, output_id="image"):
        return (worker.directory / snap["execution_id"] / (output_id + ".png")).read_bytes()

    def load(self, worker, bundle, adapters=(), options=None):
        op = worker.control(
            "load",
            {
                "protocol": PROTOCOL,
                "bundle": bundle,
                "options": options or worker.capabilities()["load_defaults"],
                "adapters": list(adapters),
            },
        )
        result = self.finish(worker, operation=op["operation_id"])
        self.assertEqual(result["state"], "succeeded", result)
        return result

    def run_request(self, worker, body):
        snap = self.finish(worker, execution=worker.submit(body)["execution_id"])
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        return snap

    def request(self, attempt, task="text-to-image", image=None, guidance=6, seed=7, count=None):
        body = {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": task,
            "inputs": {"prompt": "a test portrait", "negative_prompt": "window"},
            "parameters": {
                **DEFAULTS,
                "seed": seed,
                "width": 64,
                "height": 64,
                "steps": 3,
                "guidance": guidance,
            },
        }
        if count is not None:
            body["parameters"]["count"] = count
        if task == "image-to-image":
            body["inputs"]["image"] = image
            body["parameters"]["strength"] = 0.7
        return body

    def exercise(self, backend_id, adapters=()):
        worker = self.worker(backend_id)
        caps = worker.capabilities()
        self.assertTrue(caps["ready"], caps)
        self.assertEqual(caps["backend_id"], backend_id)
        result = self.load(worker, self.bundle, adapters)
        self.assertEqual(result["memory_estimate"]["precision"], "float32")
        self.assertGreater(result["memory_estimate"]["value_mb"], 0)
        snap = self.run_request(worker, self.request("attempt_t2i"))
        self.assertEqual(snap["progress"]["total_steps"], 3)
        self.assertEqual(
            [e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"],
            ["encoding", "sampling", "decoding", "saving"],
        )
        data = self.output(worker, snap)
        self.assertEqual(png.header(data)[:2], (64, 64))
        self.assertEqual(hashlib.sha256(data).hexdigest(), snap["outputs"][0]["sha256"])
        # Same seed and settings reproduce bytes on the same CPU/software stack.
        again = self.run_request(worker, self.request("attempt_again"))
        self.assertEqual(again["outputs"][0]["sha256"], snap["outputs"][0]["sha256"])
        # Guidance ≤ 1 disables classifier-free guidance without changing the contract.
        self.run_request(worker, self.request("attempt_plain", guidance=1))
        record = worker.store_input(make_png(64, 64))
        edited = self.run_request(
            worker, self.request("attempt_i2i", "image-to-image", record["input_id"])
        )
        self.assertEqual(edited["effective_settings"]["effective_steps"], 2)
        self.assertEqual(edited["measurements"]["effective_steps"], 2)
        self.assertEqual(edited["progress"]["total_steps"], 2)
        cancelled = worker.submit(self.request("attempt_cancel"))
        worker.cancel(cancelled["execution_id"])
        self.assertEqual(
            self.finish(worker, execution=cancelled["execution_id"])["state"], "cancelled"
        )
        outputs = {
            "text-to-image": data,
            "image-to-image": self.output(worker, edited),
        }
        unload = worker.control("unload", {"protocol": PROTOCOL})
        self.assertEqual(
            self.finish(worker, operation=unload["operation_id"])["state"], "succeeded"
        )
        self.assertIsNone(worker.loaded)
        return outputs

    def batch(self, backend_id):
        """count: 2 output 2 reproduces a count: 1 run at seed + 1 (and output 1 at seed)."""
        worker = self.worker(backend_id)
        self.load(worker, self.bundle)
        batch = self.run_request(worker, self.request("attempt_batch", count=2))
        self.assertEqual([o["output_id"] for o in batch["outputs"]], ["image", "image_2"])
        self.assertNotEqual(batch["outputs"][0]["sha256"], batch["outputs"][1]["sha256"])
        self.assertEqual(batch["effective_settings"]["count"], 2)
        first = self.run_request(worker, self.request("attempt_first"))
        second = self.run_request(worker, self.request("attempt_second", seed=8))
        self.assertLessEqual(difference(self.output(worker, batch), self.output(worker, first)), 1)
        self.assertLessEqual(
            difference(self.output(worker, batch, "image_2"), self.output(worker, second)), 1
        )
        return self.output(worker, batch, "image_2"), self.output(worker, second)

    def upscale_run(self, backend_id):
        worker = self.worker(backend_id)
        loaded = self.load(worker, self.upscaler)
        self.assertGreater(loaded["memory_estimate"]["activations_mb"], 0)
        self.assertEqual(worker.info()["loaded_bundle"]["tasks"], ["upscale"])
        record = worker.store_input(make_png(64, 64))
        body = {
            "protocol": PROTOCOL,
            "attempt_id": "attempt_upscale",
            "bundle_digest": self.upscaler["bundle_digest"],
            "task": "upscale",
            "inputs": {"image": record["input_id"]},
            "parameters": {"width": 64, "height": 64},
        }
        snap = self.run_request(worker, body)
        self.assertEqual((snap["outputs"][0]["width"], snap["outputs"][0]["height"]), (128, 128))
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        self.assertEqual(progress[-1]["unit"], "tiles")
        self.assertEqual(progress[-1]["stage"], "upscaling")
        data = self.output(worker, snap)
        self.assertEqual(png.header(data)[:2], (128, 128))
        tiled = self.run_request(
            worker,
            {**body, "attempt_id": "attempt_tiled", "parameters": {"width": 64, "height": 64, "tile": 256}},
        )
        self.assertEqual(tiled["outputs"][0]["sha256"], snap["outputs"][0]["sha256"])
        return data


@unittest.skipUnless(HAVE_OWNED, "inference-owned extra is not installed")
class OwnedInferenceTests(Harness, unittest.TestCase):
    def test_owned_pipeline_backend(self):
        self.exercise("owned-sdxl")

    def test_owned_batch_reproduces_single_seeds(self):
        self.batch("owned-sdxl")

    def test_owned_upscale(self):
        self.upscale_run("owned-sdxl")

    def test_tiled_upscale_agrees_with_whole_image(self):
        """Overlapping tiles stitched by `tile()` agree with one whole-image pass within one level."""
        import threading

        import torch

        from media_engine import upscale

        config = upscale.read_config(self.root / "upscaler")
        network = upscale.load(
            self.root / "upscaler", self.root / "upscaler/network.safetensors", config, "cpu", torch.float32
        )
        image = png.png_to_tensor(make_png(64, 64)).unsqueeze(0)
        image[:, :, 16:40, 8:56] = torch.linspace(0, 1, 48)
        events = []
        never = threading.Event()
        with torch.inference_mode():
            whole = upscale.tile(network, image, 2, 0, 0, lambda *e: events.append(e), never, RuntimeError)
            tiled = upscale.tile(network, image, 2, 32, 16, lambda *e: events.append(e), never, RuntimeError)
        self.assertEqual(tiled.shape, (1, 3, 128, 128))
        self.assertEqual(events[-1][1]["total_steps"], 4)
        a = png.tensor_to_png(whole[0].clamp(0, 1))
        b = png.tensor_to_png(tiled[0].clamp(0, 1))
        self.assertLessEqual(difference(a, b), 1)

    def test_owned_load_uses_no_library_loading_helpers(self):
        """The owned backend's declared dependencies exclude accelerate, peft, Pillow and numpy."""
        from media_engine.pipeline import OwnedPipelineBackend

        names = {name for name, _ in OwnedPipelineBackend.dependencies}
        self.assertEqual(names, {"torch", "diffusers", "transformers", "safetensors"})
        caps = self.worker("owned-sdxl").capabilities()
        self.assertEqual(set(caps["dependencies"]), names)


@unittest.skipUnless(HAVE_INFERENCE, "inference extra is not installed")
class InferenceTests(Harness, unittest.TestCase):
    def test_reference_backend(self):
        self.exercise("diffusers-sdxl")

    def test_reference_batch_reproduces_single_seeds(self):
        self.batch("diffusers-sdxl")

    def test_upscale_agrees_across_backends(self):
        """The esrgan network is engine code on both backends, so the outputs are identical."""
        self.assertEqual(self.upscale_run("diffusers-sdxl"), self.upscale_run("owned-sdxl"))

    def test_png_writer_matches_pillow_pixels(self):
        """The stdlib writer is lossless: decoding it gives the bytes Pillow writes for the tensor."""
        import io

        import numpy
        import torch
        from PIL import Image

        torch.manual_seed(5)
        tensor = torch.rand(3, 24, 40)
        ours = png.tensor_to_png(tensor)
        array = (tensor.permute(1, 2, 0).float().numpy() * 255).round().astype(numpy.uint8)
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        self.assertEqual(pixels(ours), pixels(stream.getvalue()))
        with Image.open(io.BytesIO(ours)) as decoded:
            self.assertEqual(decoded.tobytes(), array.tobytes())

    def test_owned_pipeline_preserves_reference_semantics(self):
        """Same seed and settings: the owned loop reproduces the reference pipeline's pixels.

        Text-to-image is pixel-identical on this CPU float32 stack (the PNG files differ only in
        the writer's filter choices); image-to-image agrees to within two 8-bit levels. This is
        evidence of matching request semantics, not a cross-hardware guarantee.
        """
        reference = self.exercise("diffusers-sdxl")
        owned = self.exercise("owned-sdxl")
        self.assertEqual(pixels(reference["text-to-image"]), pixels(owned["text-to-image"]))
        self.assertLessEqual(difference(reference["image-to-image"], owned["image-to-image"]), 2)

    def test_batches_agree_across_backends(self):
        reference, _ = self.batch("diffusers-sdxl")
        owned, _ = self.batch("owned-sdxl")
        self.assertLessEqual(difference(reference, owned), 2)

    def test_lora_adapters_change_output_and_agree_across_backends(self):
        """LoRA is a load-time identity: the library's PEFT layers (reference) and the owned
        in-place merge produce the same pictures, and both differ from the base model."""
        adapters = [{"adapter": self.adapter, "scale": 1.0}]
        base = self.exercise("owned-sdxl")["text-to-image"]
        reference = self.exercise("diffusers-sdxl", adapters)["text-to-image"]
        owned = self.exercise("owned-sdxl", adapters)["text-to-image"]
        self.assertGreater(mean_difference(base, owned), 1)
        self.assertLessEqual(difference(reference, owned), 2)


if __name__ == "__main__":
    unittest.main()
