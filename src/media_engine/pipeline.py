"""Owned SDXL pipeline: this project assembles the components and runs the denoising loop.

This is the "owned pipeline" stage of engine-spec.md. The worker owns component assembly,
tokenization and text conditioning, latent initialization, the Euler sampling loop, classifier-free
guidance, image-to-image noising, VAE encode/decode and the memory lifecycle. It retains the
Diffusers model/scheduler component classes and PyTorch operations; it does not use the Diffusers
pipeline classes. Request semantics match the reference backend; matching bytes are not promised.
Component weights are read with `safetensors` and assigned to the model classes directly, and PNG
input/output goes through the standard-library `png` module, so this path needs neither Pillow,
numpy, accelerate nor peft.
"""

import json
import re
from pathlib import Path

from .backend import Cancelled, TorchBackend, offline_environment
from .contracts import effective_steps, output_ids


class OwnedPipelineBackend(TorchBackend):
    backend_id = "owned-sdxl"
    ownership = "owned-pipeline"
    # No Pillow, numpy, accelerate or peft: PNG, weight loading and LoRA merging are owned here.
    dependencies = (
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("safetensors", "safetensors"),
    )
    # svd is reference-only for now: an owned video loop waits until the reference path is
    # validated on hardware. An svd bundle is `unsupported`, field `family`, on this backend.
    families = ("sdxl-base", "esrgan")

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        super().__init__(device, memory_budget_mb)
        self.components = None
        self.merged_adapters = []

    # ----- adapters --------------------------------------------------------------------------
    @staticmethod
    def merge_lora(models, weights, scale):
        """Fold scale × B·A (times alpha/rank when present) into the target weights in place.

        Merging keeps inference cost identical to the base model; the adapters therefore belong
        to the load identity, and a different adapter set means a fresh load.
        """
        import torch
        from safetensors.torch import load_file

        state = load_file(str(weights))
        groups = {}
        for name, tensor in state.items():
            target, module, part = re.fullmatch(
                r"(unet|text_encoder|text_encoder_2)\.(.+)\.(lora_A\.weight|lora_B\.weight|alpha)",
                name,
            ).groups()
            groups.setdefault((target, module), {})[part] = tensor
        merged = 0
        with torch.no_grad():
            for (target, module), parts in groups.items():
                layer = models[target].get_submodule(module)
                down, up = parts["lora_A.weight"].float(), parts["lora_B.weight"].float()
                rank = down.shape[0]
                alpha = float(parts["alpha"]) if "alpha" in parts else float(rank)
                delta = (up.flatten(1) @ down.flatten(1)).view(layer.weight.shape)
                layer.weight.add_((scale * alpha / rank * delta).to(layer.weight.dtype))
                merged += 1
        return merged

    # ----- lifecycle -------------------------------------------------------------------------
    @staticmethod
    def weight_files(folder):
        """The component's safetensors files: an index's shards, else its one (non-variant) file."""
        indexes = sorted(folder.glob("*.safetensors.index.json"))
        if len(indexes) == 1:
            shards = sorted(set(json.loads(indexes[0].read_text())["weight_map"].values()))
            return [folder / name for name in shards]
        files = sorted(folder.glob("*.safetensors"))
        if len(files) == 1:
            return files
        standard = [f for f in files if f.name in ("diffusion_pytorch_model.safetensors", "model.safetensors")]
        if len(standard) == 1:
            return standard
        raise RuntimeError("component weights are ambiguous")

    def build(self, model_class, config, folder, dtype):
        """Construct a model class on the meta device and assign its safetensors state directly.

        No library loading helper runs: the weights are read with `safetensors`, cast to the load
        precision, validated strictly against the class's parameters and assigned in place.
        """
        import torch
        from safetensors.torch import load_file

        with torch.device("meta"):
            model = (
                model_class.from_config(config)
                if hasattr(model_class, "from_config")
                else model_class(config)
            )
        state = {}
        for path in self.weight_files(folder):
            state.update(load_file(str(path), device="cpu"))
        state = {
            k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()
        }
        model.load_state_dict(state, strict=True, assign=True)
        for name, buffer in list(model.named_buffers()):
            if not buffer.is_meta:
                continue
            if not name.endswith("position_ids"):
                raise RuntimeError("component has an unmaterialized buffer")
            # CLIP's non-persistent position ids are derived, never stored.
            owner = model.get_submodule(name.rsplit(".", 1)[0])
            owner.register_buffer(
                "position_ids",
                torch.arange(buffer.shape[-1]).expand((1, -1)),
                persistent=False,
            )
        return model.eval()

    def load_pipeline(self, root, options, adapters=()):
        offline_environment()
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor2_0
        from transformers import (
            CLIPTextConfig,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.apply_budget()
        dtype = getattr(torch, options["precision"])
        common = {"local_files_only": True}
        root = Path(root)
        index = json.loads((root / "model_index.json").read_text())
        unet = self.build(
            UNet2DConditionModel,
            UNet2DConditionModel.load_config(str(root), subfolder="unet", **common),
            root / "unet",
            dtype,
        )
        vae = self.build(
            AutoencoderKL,
            AutoencoderKL.load_config(str(root), subfolder="vae", **common),
            root / "vae",
            dtype,
        )
        # SDXL's VAE overflows in float16; keep it float32 for the whole session so that the
        # memory lifecycle is explicit instead of an upcast hidden inside every execution.
        vae_dtype = (
            torch.float32
            if dtype == torch.float16 and vae.config.get("force_upcast", True)
            else dtype
        )
        vae.to(vae_dtype)
        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())
        text_encoder = self.build(
            CLIPTextModel,
            CLIPTextConfig.from_pretrained(str(root), subfolder="text_encoder", **common),
            root / "text_encoder",
            dtype,
        )
        text_encoder_2 = self.build(
            CLIPTextModelWithProjection,
            CLIPTextConfig.from_pretrained(str(root), subfolder="text_encoder_2", **common),
            root / "text_encoder_2",
            dtype,
        )
        targets = {
            "unet": unet,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }
        self.merged_adapters = [
            {
                "adapter_id": record["adapter_id"],
                "scale": scale,
                "modules": self.merge_lora(targets, adapter_weights, scale),
            }
            for record, _root, adapter_weights, scale, _layout in adapters
        ]
        device = self.run_device
        self.components = {
            "unet": unet.to(device).eval(),
            "vae": vae.to(device).eval(),
            "text_encoder": text_encoder.to(device).eval(),
            "text_encoder_2": text_encoder_2.to(device).eval(),
            "tokenizer": CLIPTokenizer.from_pretrained(
                str(root), subfolder="tokenizer", **common
            ),
            "tokenizer_2": CLIPTokenizer.from_pretrained(
                str(root), subfolder="tokenizer_2", **common
            ),
            "scheduler_config": EulerDiscreteScheduler.load_config(
                str(root), subfolder="scheduler", **common
            ),
            "force_zeros_for_empty_prompt": bool(
                index.get("force_zeros_for_empty_prompt", True)
            ),
            "dtype": dtype,
            "vae_dtype": vae_dtype,
            "vae_scale_factor": 2 ** (len(vae.config.block_out_channels) - 1),
        }

    def unload_pipeline(self):
        self.components = None

    # ----- conditioning ----------------------------------------------------------------------
    def encode_text(self, text):
        import torch

        c = self.components
        hidden, pooled = [], None
        for tokenizer, encoder in (
            (c["tokenizer"], c["text_encoder"]),
            (c["tokenizer_2"], c["text_encoder_2"]),
        ):
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.run_device)
            output = encoder(tokens, output_hidden_states=True)
            # SDXL conditions on the penultimate layer; the pooled projection comes from encoder 2.
            pooled = output[0]
            hidden.append(output.hidden_states[-2])
        return torch.cat(hidden, dim=-1), pooled

    def conditioning(self, inputs, guidance, width, height):
        import torch

        c = self.components
        embeds, pooled = self.encode_text(inputs["prompt"])
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]], device=self.run_device, dtype=c["dtype"]
        )
        expected = c["unet"].config.get("projection_class_embeddings_input_dim")
        if expected is not None and expected != time_ids.shape[1] * c[
            "unet"
        ].config.addition_time_embed_dim + pooled.shape[-1]:
            raise RuntimeError("bundle conditioning dimensions are not SDXL base")
        if guidance <= 1:
            return embeds, pooled, time_ids, False
        negative = inputs.get("negative_prompt")
        if negative is None and c["force_zeros_for_empty_prompt"]:
            negative_embeds, negative_pooled = (
                torch.zeros_like(embeds),
                torch.zeros_like(pooled),
            )
        else:
            negative_embeds, negative_pooled = self.encode_text(negative or "")
        return (
            torch.cat([negative_embeds, embeds]),
            torch.cat([negative_pooled, pooled]),
            torch.cat([time_ids, time_ids]),
            True,
        )

    # ----- latents ---------------------------------------------------------------------------
    def encode_image(self, path, width, height, generator):
        """Encode the approved PNG input; a size other than the request is an error, not a resize."""
        from .png import png_to_tensor

        c = self.components
        image = png_to_tensor(Path(path).read_bytes())
        if tuple(image.shape[1:]) != (height, width):
            raise ValueError("input image size does not match the request")
        tensor = (image * 2 - 1).unsqueeze(0).to(self.run_device, c["vae_dtype"])
        posterior = c["vae"].encode(tensor).latent_dist
        return (posterior.sample(generator) * c["vae"].config.scaling_factor).to(
            c["dtype"]
        )

    def decode(self, latents):
        """Decode one latent to PNG bytes through the standard-library writer."""
        from .png import tensor_to_png

        c = self.components
        latents = latents.to(c["vae_dtype"]) / c["vae"].config.scaling_factor
        image = c["vae"].decode(latents, return_dict=False)[0]
        return tensor_to_png((image / 2 + 0.5).clamp(0, 1)[0])

    # ----- execution -------------------------------------------------------------------------
    def generate_images(self, request, directory, emit, cancelled, image_path=None):
        import torch
        from diffusers import EulerDiscreteScheduler

        c = self.components
        if c is None:
            raise RuntimeError("Model not loaded")
        params = request["parameters"]
        task = request["task"]
        width, height = params["width"], params["height"]
        total = effective_steps(params, task)
        device = self.run_device
        self.begin(cancelled)
        try:
            with torch.inference_mode():
                emit("stage", {"stage": "encoding"})
                embeds, pooled, time_ids, cfg = self.conditioning(
                    request["inputs"], params["guidance"], width, height
                )
                # One generator per image, seeded seed + k − 1: each latent is drawn exactly as a
                # count-1 run at that seed would draw it.
                generators = [
                    torch.Generator(device=device).manual_seed(s) for s in self.seeds(params)
                ]
                count = len(generators)
                scheduler = EulerDiscreteScheduler.from_config(c["scheduler_config"])
                scheduler.set_timesteps(params["steps"], device=device)
                shape = (
                    1,
                    c["unet"].config.in_channels,
                    height // c["vae_scale_factor"],
                    width // c["vae_scale_factor"],
                )
                if task == "image-to-image":
                    start = (params["steps"] - total) * scheduler.order
                    timesteps = scheduler.timesteps[start:]
                    if hasattr(scheduler, "set_begin_index"):
                        scheduler.set_begin_index(start)
                    # The posterior sample consumes the generator before the noise draw, so
                    # a seed means the same thing here as in the reference pipeline.
                    initial = self.encode_image(image_path, width, height, generators[0])
                    noise = torch.randn(
                        shape, generator=generators[0], device=device, dtype=c["dtype"]
                    )
                    latents = scheduler.add_noise(initial, noise, timesteps[:1])
                else:
                    timesteps = scheduler.timesteps
                    noise = torch.cat(
                        [
                            torch.randn(shape, generator=g, device=device, dtype=c["dtype"])
                            for g in generators
                        ]
                    )
                    latents = noise * scheduler.init_noise_sigma
                if count > 1:
                    # Conditioning rows follow the latent batch: [negative × N, positive × N].
                    def batch(t):
                        parts = t.chunk(2) if cfg else (t,)
                        return torch.cat([part.repeat_interleave(count, dim=0) for part in parts])

                    embeds, pooled, time_ids = batch(embeds), batch(pooled), batch(time_ids)
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "sampling"})
                added = {"text_embeds": pooled, "time_ids": time_ids}
                for index, t in enumerate(timesteps):
                    if cancelled.is_set():
                        raise Cancelled()
                    model_input = torch.cat([latents] * 2) if cfg else latents
                    model_input = scheduler.scale_model_input(model_input, t)
                    prediction = c["unet"](
                        model_input,
                        t,
                        encoder_hidden_states=embeds,
                        added_cond_kwargs=added,
                        return_dict=False,
                    )[0]
                    if cfg:
                        unconditional, conditional = prediction.chunk(2)
                        prediction = unconditional + params["guidance"] * (
                            conditional - unconditional
                        )
                    latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
                    emit(
                        "progress",
                        {
                            "stage": "sampling",
                            "completed_steps": index + 1,
                            "total_steps": len(timesteps),
                            "scope": "stage",
                        },
                    )
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "decoding"})
                images = []
                for latent in latents:
                    if cancelled.is_set():
                        raise Cancelled()
                    images.append(self.decode(latent.unsqueeze(0)))
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "saving"})
                records = []
                for output_id, data in zip(output_ids(task, params), images):
                    path = Path(directory) / (output_id + ".png")
                    path.write_bytes(data)
                    records.append(
                        {
                            "output_id": output_id,
                            "path": path,
                            "mime_type": "image/png",
                            "width": width,
                            "height": height,
                        }
                    )
                return records, {"effective_steps": len(timesteps), **self.measurements()}
        finally:
            self.finish()
