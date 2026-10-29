"""Real CPU inference through both backends on a tiny random-weight SDXL-shaped bundle.

Skipped unless the inference extra is installed. The bundle is architecturally SDXL base
(dual CLIP conditioning, time-id embedding, Euler scheduler, KL VAE) with tiny dimensions and
random weights, so it proves the owned pipeline and the reference pipeline execute the same
protocol end to end; it says nothing about image quality or hardware compatibility. The svd
bundle (spatio-temporal UNet, temporal-decoder VAE, CLIP image encoder, Euler scheduler) is tiny
and random in the same way and runs image-to-video through the reference backend only; so is the
stable-audio-open bundle (diffusion transformer, Oobleck VAE at 1 kHz stereo, T5 encoder with a
word-level fast tokenizer built offline, timing projection, cosine DPM-Solver scheduler), which
runs text-to-audio through the reference backend only.
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
# image-to-video needs the svd model classes and Pillow, nothing beyond the base inference set.
HAVE_VIDEO = HAVE_OWNED and importlib.util.find_spec("PIL") is not None
# text-to-audio needs the family's model classes plus torchsde (the inference-audio extra).
HAVE_AUDIO = HAVE_OWNED and importlib.util.find_spec("torchsde") is not None


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


def tiny_svd_bundle(root, frames=2):
    """A random-weight Stable Video Diffusion bundle at 64×64 (VAE factor 2, two frames)."""
    import torch
    from diffusers import (
        AutoencoderKLTemporalDecoder,
        EulerDiscreteScheduler,
        StableVideoDiffusionPipeline,
        UNetSpatioTemporalConditionModel,
    )
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

    torch.manual_seed(4)
    unet = UNetSpatioTemporalConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=8,
        out_channels=4,
        down_block_types=("CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal"),
        up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
        cross_attention_dim=32,
        num_attention_heads=(2, 4),
        # fps, motion bucket and noise augmentation, each embedded in 8 dimensions.
        projection_class_embeddings_input_dim=24,
        addition_time_embed_dim=8,
        num_frames=frames,
    )
    vae = AutoencoderKLTemporalDecoder(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        latent_channels=4,
        layers_per_block=1,
    )
    # The pipeline resizes the conditioning image to 224 for CLIP; 32-pixel patches keep it tiny.
    encoder = CLIPVisionModelWithProjection(
        CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            intermediate_size=37,
            patch_size=32,
        )
    )
    scheduler = EulerDiscreteScheduler(
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        interpolation_type="linear",
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        sigma_max=700.0,
        sigma_min=0.002,
        steps_offset=1,
        timestep_spacing="leading",
        timestep_type="continuous",
        use_karras_sigmas=True,
    )
    pipe = StableVideoDiffusionPipeline(
        vae=vae,
        image_encoder=encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=CLIPImageProcessor(crop_size=224, size=224),
    )
    root = Path(root)
    pipe.save_pretrained(str(root), safe_serialization=True)
    (root / "video.json").write_text(json.dumps({"sizes": [[64, 64]]}))
    return describe_bundle(root, "tiny-video", "tiny-svd", "svd", source={"license": "test-only random weights"})


def tiny_audio_bundle(root):
    """A random-weight Stable Audio Open bundle: 1 kHz stereo, 8 samples per latent frame, 2.048 s.

    The T5 tokenizer is a word-level `tokenizers` model wrapped as T5TokenizerFast and saved as
    tokenizer.json, so neither the bundle nor the test needs sentencepiece or a download.
    """
    import torch
    from diffusers import (
        AutoencoderOobleck,
        CosineDPMSolverMultistepScheduler,
        StableAudioDiTModel,
        StableAudioPipeline,
    )
    from diffusers.pipelines.stable_audio import StableAudioProjectionModel
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast

    torch.manual_seed(5)
    words = ["<pad>", "</s>", "<unk>"] + "a the soft loud rain drum bird wood hammer test tone".split()
    model = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.Whitespace()
    model.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = T5TokenizerFast(
        tokenizer_object=model, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", extra_ids=0, model_max_length=16
    )
    encoder = T5EncoderModel(
        T5Config(vocab_size=len(words), d_model=32, d_ff=37, d_kv=8, num_layers=2, num_heads=4, relative_attention_num_buckets=8)
    )
    vae = AutoencoderOobleck(
        encoder_hidden_size=6,
        downsampling_ratios=[2, 4],
        decoder_channels=3,
        decoder_input_channels=3,
        audio_channels=2,
        channel_multiples=[2, 4],
        sampling_rate=1000,
    )
    transformer = StableAudioDiTModel(
        sample_size=256,
        in_channels=3,
        num_layers=2,
        attention_head_dim=4,
        num_attention_heads=2,
        num_key_value_attention_heads=2,
        out_channels=3,
        cross_attention_dim=4,
        time_proj_dim=8,
        global_states_input_dim=8,
        cross_attention_input_dim=4,
    )
    projection = StableAudioProjectionModel(text_encoder_dim=32, conditioning_dim=4, min_value=0, max_value=32)
    scheduler = CosineDPMSolverMultistepScheduler(
        solver_order=2, prediction_type="v_prediction", sigma_data=1.0, sigma_schedule="exponential"
    )
    pipe = StableAudioPipeline(
        vae=vae,
        text_encoder=encoder,
        projection_model=projection,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    root = Path(root)
    pipe.save_pretrained(str(root), safe_serialization=True)
    return describe_bundle(root, "tiny-audio", "tiny-audio", "stable-audio-open", source={"license": "test-only random weights"})


class Harness:
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_dir, cls.bundle = tiny_sdxl_bundle(cls.root)
        cls.adapter = tiny_lora(cls.root / "lora", cls.model_dir)
        cls.upscaler = tiny_esrgan_bundle(cls.root / "upscaler")
        cls.video = tiny_svd_bundle(cls.root / "video") if HAVE_VIDEO else None
        cls.audio = tiny_audio_bundle(cls.root / "audio") if HAVE_AUDIO else None

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
                "tiny-video": str(self.root / "video"),
                "tiny-audio": str(self.root / "audio"),
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


@unittest.skipUnless(HAVE_VIDEO, "inference extra (svd model classes and Pillow) is not installed")
class VideoInferenceTests(Harness, unittest.TestCase):
    def video_request(self, attempt, input_id, **params):
        return {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.video["bundle_digest"],
            "task": "image-to-video",
            "inputs": {"image": input_id},
            "parameters": {"width": 64, "height": 64, "seed": 3, "steps": 2, **params},
        }

    def test_reference_image_to_video(self):
        """64×64, two frames, two steps through StableVideoDiffusionPipeline on CPU float32."""
        import zipfile

        from media_engine import video

        worker = self.worker("diffusers-sdxl")
        caps = worker.capabilities()
        self.assertIn("image-to-video", caps["tasks"])
        self.assertIn(["svd"], [l["families"] for l in caps["loaders"]])
        loaded = self.load(worker, self.video)
        self.assertGreater(loaded["memory_estimate"]["latents_mb"] + loaded["memory_estimate"]["value_mb"], 0)
        self.assertEqual(worker.info()["loaded_bundle"]["default_frames"], 2)
        record = worker.store_input(make_png(64, 64, (0xC0, 0x40, 0x20)))
        snap = self.run_request(worker, self.video_request("attempt_video", record["input_id"], decode_chunk=1, fps=6))
        self.assertEqual(
            [e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"],
            ["encoding", "sampling", "decoding", "packaging", "saving"],
        )
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        self.assertEqual([p["completed_steps"] for p in progress if p["stage"] == "sampling"], [1, 2])
        self.assertEqual([(p["unit"], p["completed_steps"]) for p in progress if p["stage"] == "decoding"], [("frames", 1), ("frames", 2)])
        frames, poster = snap["outputs"]
        self.assertEqual((frames["frames"], frames["fps"], frames["duration_seconds"]), (2, 6, 2 / 6))
        folder = worker.directory / snap["execution_id"]
        archive = folder / "frames.zip"
        manifest, pngs = video.read_archive(archive)
        self.assertEqual(manifest, {"fps": 6, "width": 64, "height": 64, "frames": 2})
        with zipfile.ZipFile(archive) as opened:
            self.assertEqual(sorted(opened.namelist()), ["000001.png", "000002.png", "frames.json"])
        self.assertEqual([png.header(f)[:2] for f in pngs], [(64, 64), (64, 64)])
        self.assertEqual((folder / "poster.png").read_bytes(), pngs[0])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), frames["sha256"])
        self.assertEqual(snap["measurements"]["decode_chunks"], 2)
        # Same seed and settings reproduce the archive bytes on this CPU stack; another seed does
        # not. (decode_chunk changes what the temporal decoder sees together, so only like is
        # compared with like.)
        again = self.run_request(worker, self.video_request("attempt_again", record["input_id"], decode_chunk=1, fps=6))
        self.assertEqual(again["outputs"][0]["sha256"], frames["sha256"])
        other = self.run_request(worker, self.video_request("attempt_other", record["input_id"], decode_chunk=1, fps=6, seed=4))
        self.assertNotEqual(other["outputs"][0]["sha256"], frames["sha256"])
        whole = self.run_request(worker, self.video_request("attempt_whole", record["input_id"]))
        self.assertEqual(whole["measurements"]["decode_chunks"], 1)
        self.assertEqual(whole["outputs"][0]["frames"], 2)
        cancelled = worker.submit(self.video_request("attempt_cancel", record["input_id"], steps=50))
        worker.cancel(cancelled["execution_id"])
        self.assertEqual(self.finish(worker, execution=cancelled["execution_id"])["state"], "cancelled")

    def test_frames_match_the_library_pipeline(self):
        """Our per-chunk decode and PNG writer give the pixels StableVideoDiffusionPipeline returns."""
        import torch
        from diffusers import StableVideoDiffusionPipeline
        from PIL import Image

        from media_engine import video

        worker = self.worker("diffusers-sdxl")
        self.load(worker, self.video)
        image = make_png(64, 64, (0x30, 0x80, 0xA0))
        record = worker.store_input(image)
        snap = self.run_request(worker, self.video_request("attempt_match", record["input_id"], decode_chunk=2))
        _, ours = video.read_archive(worker.directory / snap["execution_id"] / "frames.zip")
        pipe = StableVideoDiffusionPipeline.from_pretrained(str(self.root / "video"), local_files_only=True, use_safetensors=True)
        pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            frames = pipe(
                Image.new("RGB", (64, 64), (0x30, 0x80, 0xA0)),
                height=64,
                width=64,
                num_frames=2,
                num_inference_steps=2,
                fps=7,
                motion_bucket_id=127,
                noise_aug_strength=0.02,
                decode_chunk_size=2,
                generator=torch.Generator("cpu").manual_seed(3),
                output_type="pt",
            ).frames[0]
        for ours_frame, theirs in zip(ours, frames):
            self.assertLessEqual(difference(ours_frame, png.tensor_to_png(theirs)), 1)

    def test_owned_backend_refuses_svd(self):
        worker = self.worker("owned-sdxl")
        self.assertNotIn("image-to-video", worker.capabilities()["tasks"])
        from media_engine.contracts import WorkerError

        with self.assertRaises(WorkerError) as caught:
            worker.validate_load({"protocol": PROTOCOL, "bundle": self.video, "options": worker.capabilities()["load_defaults"]})
        self.assertEqual((caught.exception.code, caught.exception.field), ("unsupported", "family"))


@unittest.skipUnless(HAVE_AUDIO, "inference-audio extra (torchsde) or the model classes are not installed")
class AudioInferenceTests(Harness, unittest.TestCase):
    def audio_request(self, attempt, **params):
        return {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.audio["bundle_digest"],
            "task": "text-to-audio",
            "inputs": {"prompt": "soft rain", "negative_prompt": "loud drum"},
            "parameters": {"seed": 3, "steps": 2, "duration_seconds": 1, **params},
        }

    def test_reference_text_to_audio(self):
        """One second at 1 kHz stereo, two steps, through StableAudioPipeline on CPU float32."""
        from media_engine import audio

        worker = self.worker("diffusers-sdxl")
        caps = worker.capabilities()
        self.assertIn("text-to-audio", caps["tasks"])
        self.assertIn(["stable-audio-open"], [l["families"] for l in caps["loaders"]])
        self.assertEqual(caps["missing_task_dependencies"], {})
        loaded = self.load(worker, self.audio)
        roles = loaded["memory_estimate"]["roles"]
        self.assertEqual(set(roles), {"transformer", "vae", "text_encoder", "projection_model"})
        self.assertTrue(all(r["parameters"] > 0 for r in roles.values()))
        facts = worker.info()["loaded_bundle"]
        self.assertEqual(
            {k: facts[k] for k in ("sample_rate", "channels", "hop_length", "duration_range")},
            {"sample_rate": 1000, "channels": 2, "hop_length": 8, "duration_range": [1, 2]},
        )
        snap = self.run_request(worker, self.audio_request("attempt_audio"))
        self.assertEqual(
            [e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"],
            ["encoding", "sampling", "decoding", "saving"],
        )
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        self.assertEqual([p["completed_steps"] for p in progress if p["stage"] == "sampling"], [1, 2])
        (output,) = snap["outputs"]
        # One second is 125 whole latent frames of 8 samples: exact, nothing added or removed.
        self.assertEqual(
            {k: output[k] for k in ("mime_type", "sample_rate", "channels", "samples", "duration_seconds")},
            {"mime_type": "audio/wav", "sample_rate": 1000, "channels": 2, "samples": 1000, "duration_seconds": 1.0},
        )
        path = worker.directory / snap["execution_id"] / "audio.wav"
        data = path.read_bytes()
        self.assertEqual(data[:4] + data[8:16] + data[36:40], b"RIFFWAVEfmt data")
        self.assertEqual(len(data), 44 + 1000 * 2 * 2)
        self.assertEqual(hashlib.sha256(data).hexdigest(), output["sha256"])
        header, pcm = audio.read_wav(path)
        self.assertEqual((header["sample_rate"], header["channels"], header["samples"]), (1000, 2, 1000))
        self.assertGreater(len(set(pcm)), 2)  # not silence
        self.assertEqual(snap["measurements"]["latent_frames"], 125)
        # 1.005 s is 125.6 frames: rounded up to 126 frames, 1008 samples, and reported as 1.008 s.
        rounded = self.run_request(worker, self.audio_request("attempt_rounded", duration_seconds=1.005))
        self.assertEqual((rounded["outputs"][0]["samples"], rounded["outputs"][0]["duration_seconds"]), (1008, 1.008))
        self.assertEqual(rounded["effective_settings"]["effective_duration_seconds"], 1.008)
        self.assertEqual(rounded["effective_settings"]["duration_seconds"], 1.005)
        # Same seed and settings reproduce the bytes on this CPU stack; another seed does not.
        again = self.run_request(worker, self.audio_request("attempt_again"))
        self.assertEqual(again["outputs"][0]["sha256"], output["sha256"])
        other = self.run_request(worker, self.audio_request("attempt_other", seed=4))
        self.assertNotEqual(other["outputs"][0]["sha256"], output["sha256"])
        from media_engine.contracts import WorkerError

        with self.assertRaises(WorkerError) as caught:
            worker.submit(self.audio_request("attempt_long", duration_seconds=2.5))
        self.assertEqual((caught.exception.code, caught.exception.field), ("invalid_request", "duration_seconds"))
        with self.assertRaises(WorkerError) as caught:
            worker.submit({**self.audio_request("attempt_old"), "protocol": "media-worker/1.2"})
        self.assertEqual((caught.exception.code, caught.exception.field), ("unsupported", "task"))
        cancelled = worker.submit(self.audio_request("attempt_cancel", steps=200))
        worker.cancel(cancelled["execution_id"])
        self.assertEqual(self.finish(worker, execution=cancelled["execution_id"])["state"], "cancelled")

    def test_waveform_matches_the_library_pipeline(self):
        """Our decode, cut and 16-bit writer give the samples StableAudioPipeline returns."""
        import struct

        import torch
        from diffusers import StableAudioPipeline

        from media_engine import audio

        worker = self.worker("diffusers-sdxl")
        self.load(worker, self.audio)
        snap = self.run_request(worker, self.audio_request("attempt_match", duration_seconds=1.005, guidance=4))
        _, ours = audio.read_wav(worker.directory / snap["execution_id"] / "audio.wav")
        pipe = StableAudioPipeline.from_pretrained(str(self.root / "audio"), local_files_only=True, use_safetensors=True)
        pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            # The worker conditions on, and returns, the effective duration (1.008 s).
            theirs = pipe(
                "soft rain",
                negative_prompt="loud drum",
                audio_end_in_s=1.008,
                num_inference_steps=2,
                guidance_scale=4,
                generator=torch.Generator("cpu").manual_seed(3),
                output_type="pt",
            ).audios[0]
        self.assertEqual(theirs.shape, (2, 1008))
        expected = audio.pcm16(theirs.T.flatten().tolist())
        ours_values = struct.unpack(f"<{len(ours) // 2}h", ours)
        theirs_values = struct.unpack(f"<{len(expected) // 2}h", expected)
        self.assertEqual(len(ours_values), len(theirs_values))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(ours_values, theirs_values)), 1)

    def test_owned_backend_refuses_audio(self):
        worker = self.worker("owned-sdxl")
        self.assertNotIn("text-to-audio", worker.capabilities()["tasks"])
        from media_engine.contracts import WorkerError

        with self.assertRaises(WorkerError) as caught:
            worker.validate_load({"protocol": PROTOCOL, "bundle": self.audio, "options": worker.capabilities()["load_defaults"]})
        self.assertEqual((caught.exception.code, caught.exception.field), ("unsupported", "family"))


if __name__ == "__main__":
    unittest.main()
