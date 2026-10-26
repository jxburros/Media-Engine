"""Offline Diffusers adapters. No inference dependencies are imported by the API server."""

import importlib.util
import importlib.metadata
import os
import resource

from .contracts import (
    DEFAULTS,
    LIMITS,
    PROTOCOL,
    SUPPORTED_PROTOCOLS,
    TASK_DEFAULTS,
    TASKS,
    WORKER_VERSION,
    effective_steps,
    memory_estimate,
)

DEPENDENCIES = (
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("Pillow", "PIL"),
)


class Cancelled(Exception):
    pass


def offline_environment():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
    )


class TorchBackend:
    """Shared readiness, device and measurement behavior for both SDXL backends."""

    backend_id = "torch"
    ownership = "reference"

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        self.device = device
        self.memory_budget_mb = memory_budget_mb
        self.loaded_estimate = None

    def missing_dependencies(self):
        return [
            name
            for name, module in DEPENDENCIES
            if importlib.util.find_spec(module) is None
        ]

    def device_available(self):
        if self.missing_dependencies():
            return False
        import torch

        return self.device == "cpu" or (
            torch.cuda.is_available()
            and int(self.device.split(":")[1]) < torch.cuda.device_count()
        )

    def capabilities(self):
        missing = self.missing_dependencies()
        device_available = self.device_available()
        return {
            "protocol": PROTOCOL,
            "supported_protocols": list(SUPPORTED_PROTOCOLS),
            "worker_version": WORKER_VERSION,
            "backend_id": self.backend_id,
            "ownership": self.ownership,
            "ready": not missing and device_available,
            "missing_dependencies": missing,
            "device": self.device,
            "device_available": device_available,
            "memory_budget_mb": self.memory_budget_mb,
            "memory_estimate": self.loaded_estimate
            or {
                "value_mb": None,
                "source": "unavailable",
                "uncertainty": "no bundle validated or loaded in this session",
            },
            "dependencies": {
                name: importlib.metadata.version(name)
                if importlib.util.find_spec(module) is not None
                else None
                for name, module in DEPENDENCIES
            },
            "max_concurrency": 1,
            "model_families": ["sdxl-base"],
            "tasks": list(TASKS),
            "task_parameters": {
                "text-to-image": {"inputs": ["prompt", "negative_prompt"]},
                "image-to-image": {
                    "inputs": ["prompt", "negative_prompt", "image"],
                    "parameters": ["strength"],
                    "input_formats": ["image/png"],
                },
            },
            "defaults": DEFAULTS,
            "task_defaults": TASK_DEFAULTS,
            "load_defaults": {
                "precision": "float32" if self.device == "cpu" else "float16",
                "attention": "sdpa",
                "offload": "none",
            },
            "limits": LIMITS,
            "precisions": ["float32"]
            if self.device == "cpu"
            else ["float16", "float32"],
        }

    def estimate(self, root, options):
        self.loaded_estimate = memory_estimate(root, options["precision"], self.device)
        return self.loaded_estimate

    def apply_budget(self):
        import torch

        if self.device.startswith("cuda") and self.memory_budget_mb:
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, self.memory_budget_mb * 1024**2 / total), self.device
            )

    def release_device(self):
        import gc

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

    def measurements(self):
        import torch

        cuda = self.device.startswith("cuda")
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) // 1024**2
            if cuda
            else None,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) // 1024**2
            if cuda
            else None,
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        }

    def begin(self, cancelled):
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        if cancelled.is_set():
            raise Cancelled()

    def finish(self):
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def load_image(self, path, width, height):
        """Decode the approved PNG input; a size other than the request is an error, not a resize."""
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError("input image size does not match the request")
            return image.convert("RGB")


class DiffusersBackend(TorchBackend):
    """Reference worker stage: the complete Diffusers pipelines execute inference."""

    backend_id = "diffusers-sdxl"
    ownership = "reference"

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        super().__init__(device, memory_budget_mb)
        self.pipe = None
        self.img2img = None

    def load(self, root, options):
        offline_environment()
        import torch
        from diffusers import StableDiffusionXLPipeline
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.apply_budget()
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
        self.img2img = None

    def unload(self):
        self.pipe = None
        self.img2img = None
        return self.release_device()

    def generate(self, request, destination, emit, cancelled, image_path=None):
        import torch

        params = request["parameters"]
        task = request["task"]
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Model not loaded")
        self.begin(cancelled)
        try:
            pipe.set_progress_bar_config(disable=True)
            total = effective_steps(params, task)
            emit("stage", {"stage": "encoding"})
            with torch.inference_mode():
                embeddings = pipe.encode_prompt(
                    prompt=request["inputs"]["prompt"],
                    device=self.device,
                    negative_prompt=request["inputs"].get("negative_prompt"),
                    do_classifier_free_guidance=params["guidance"] > 1,
                )
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "sampling"})

                def step(pipeline, index, timestep, values):
                    if cancelled.is_set():
                        raise Cancelled()
                    emit(
                        "progress",
                        {
                            "stage": "sampling",
                            "completed_steps": index + 1,
                            "total_steps": total,
                            "scope": "stage",
                        },
                    )
                    return values

                common = dict(
                    prompt_embeds=embeddings[0],
                    negative_prompt_embeds=embeddings[1],
                    pooled_prompt_embeds=embeddings[2],
                    negative_pooled_prompt_embeds=embeddings[3],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    generator=torch.Generator(device=self.device).manual_seed(
                        params["seed"]
                    ),
                    callback_on_step_end=step,
                    output_type="latent",
                )
                if task == "image-to-image":
                    from diffusers import StableDiffusionXLImg2ImgPipeline

                    if self.img2img is None:
                        self.img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(
                            pipe, requires_aesthetics_score=False
                        )
                        self.img2img.set_progress_bar_config(disable=True)
                    image = self.load_image(image_path, params["width"], params["height"])
                    latent = self.img2img(
                        image=image, strength=params["strength"], **common
                    ).images
                else:
                    latent = pipe(
                        width=params["width"], height=params["height"], **common
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
                return {"effective_steps": total, **self.measurements()}
        finally:
            self.finish()


BACKENDS = {}


def backend_class(backend_id):
    from .pipeline import OwnedPipelineBackend

    BACKENDS.setdefault(DiffusersBackend.backend_id, DiffusersBackend)
    BACKENDS.setdefault(OwnedPipelineBackend.backend_id, OwnedPipelineBackend)
    return BACKENDS[backend_id]
