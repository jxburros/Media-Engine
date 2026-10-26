"""Offline Diffusers adapter. No inference dependencies are imported by the API server."""

import importlib.util
import importlib.metadata
import os
import resource

from .contracts import DEFAULTS


class Cancelled(Exception):
    pass


class DiffusersBackend:
    backend_id = "diffusers-sdxl"

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        self.pipe = None
        self.device = device
        self.memory_budget_mb = memory_budget_mb

    def capabilities(self):
        missing = [
            m
            for m in (
                "torch",
                "diffusers",
                "transformers",
                "accelerate",
                "safetensors",
                "PIL",
            )
            if importlib.util.find_spec(m) is None
        ]
        ready = not missing
        device_available = False
        if ready:
            import torch

            device_available = self.device == "cpu" or (
                torch.cuda.is_available()
                and int(self.device.split(":")[1]) < torch.cuda.device_count()
            )
        return {
            "protocol": "media-worker/1.0",
            "worker_version": "0.1.0",
            "backend_id": self.backend_id,
            "ready": ready and device_available,
            "missing_dependencies": missing,
            "device": self.device,
            "device_available": device_available,
            "memory_budget_mb": self.memory_budget_mb,
            "memory_estimate": {
                "value_mb": None,
                "source": "unavailable",
                "uncertainty": "unknown",
            },
            "dependencies": {
                name: importlib.metadata.version(name)
                if importlib.util.find_spec(module) is not None
                else None
                for name, module in (
                    ("torch", "torch"),
                    ("diffusers", "diffusers"),
                    ("transformers", "transformers"),
                    ("accelerate", "accelerate"),
                    ("safetensors", "safetensors"),
                    ("Pillow", "PIL"),
                )
            },
            "max_concurrency": 1,
            "model_families": ["sdxl-base"],
            "tasks": ["text-to-image"],
            "defaults": DEFAULTS,
            "load_defaults": {
                "precision": "float32" if self.device == "cpu" else "float16",
                "attention": "sdpa",
                "offload": "none",
            },
            "limits": {
                "width": [64, 2048],
                "height": [64, 2048],
                "steps": [1, 150],
                "guidance": [0, 30],
                "size_multiple": 8,
            },
            "precisions": ["float32"]
            if self.device == "cpu"
            else ["float16", "float32"],
        }

    def load(self, root, options):
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
        )
        import torch
        from diffusers import StableDiffusionXLPipeline
        from diffusers.models.attention_processor import AttnProcessor2_0

        if self.device.startswith("cuda") and self.memory_budget_mb:
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, self.memory_budget_mb * 1024**2 / total), self.device
            )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(root),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=getattr(torch, options["precision"]),
            add_watermarker=False,
        )
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        pipe.vae.set_attn_processor(AttnProcessor2_0())
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def unload(self):
        import gc

        self.pipe = None
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                return {
                    "allocated_mb": torch.cuda.memory_allocated(self.device) // 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved(self.device) // 1024**2,
                }
        return {"allocated_mb": None, "reserved_mb": None}

    def generate(self, request, destination, emit, cancelled):
        import torch

        params = request["parameters"]
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Model not loaded")
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        if cancelled.is_set():
            raise Cancelled()
        try:
            pipe.set_progress_bar_config(disable=True)
            pipe.to(self.device)
            if cancelled.is_set():
                raise Cancelled()
            emit("stage", {"stage": "encoding"})
            with torch.inference_mode():
                embeddings = pipe.encode_prompt(
                    prompt=request["inputs"]["prompt"],
                    device=self.device,
                    negative_prompt=request["inputs"].get("negative_prompt"),
                    do_classifier_free_guidance=params["guidance"] > 1,
                )
                emit("stage", {"stage": "sampling"})

                def step(pipeline, index, timestep, values):
                    if cancelled.is_set():
                        raise Cancelled()
                    emit(
                        "progress",
                        {
                            "stage": "sampling",
                            "completed_steps": index + 1,
                            "total_steps": params["steps"],
                            "scope": "stage",
                        },
                    )
                    return values

                latent = pipe(
                    prompt_embeds=embeddings[0],
                    negative_prompt_embeds=embeddings[1],
                    pooled_prompt_embeds=embeddings[2],
                    negative_pooled_prompt_embeds=embeddings[3],
                    width=params["width"],
                    height=params["height"],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    generator=torch.Generator(device=self.device).manual_seed(
                        params["seed"]
                    ),
                    callback_on_step_end=step,
                    output_type="latent",
                ).images
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "decoding"})
                if pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast:
                    pipe.upcast_vae()
                latent = latent.to(
                    next(iter(pipe.vae.post_quant_conv.parameters())).dtype
                )
                decoded = pipe.vae.decode(
                    latent / pipe.vae.config.scaling_factor, return_dict=False
                )[0]
                output = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "saving"})
                output.save(destination, format="PNG")
                measured = (
                    torch.cuda.max_memory_allocated(self.device) // 1024**2
                    if self.device.startswith("cuda")
                    else None
                )
                return {
                    "peak_allocated_mb": measured,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device)
                    // 1024**2
                    if self.device.startswith("cuda")
                    else None,
                    "process_peak_rss_mb": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    / 1024,
                }
        finally:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
