"""Offline Diffusers adapters. No inference dependencies are imported by the API server."""

import hashlib
import importlib.util
import importlib.metadata
import os
import resource
import socket
from pathlib import Path

from .contracts import (
    ADAPTER_FORMAT,
    ADAPTER_TARGETS,
    AUTO_OPTIONS,
    DEFAULTS,
    FALLBACKS,
    FAMILIES,
    FORMATS,
    LIMITS,
    MAX_ADAPTERS,
    OUTPUTS,
    PRECISION_BYTES,
    PROTOCOL,
    SUPPORTED_PROTOCOLS,
    TASK_DEFAULTS,
    TASKS,
    WORKER_VERSION,
    effective_steps,
    memory_estimate,
    output_ids,
)

DEPENDENCIES = (
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("Pillow", "PIL"),
    ("peft", "peft"),
)
# The esrgan family runs the engine-owned network on every backend.
UPSCALE_DEPENDENCIES = (("torch", "torch"), ("safetensors", "safetensors"))


class Cancelled(Exception):
    pass


def offline_environment():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
    )


def host_identity():
    """A stable host identifier that reveals neither the machine ID nor the hostname."""
    try:
        machine = Path("/etc/machine-id").read_text().strip()
    except OSError:
        machine = ""
    if machine:
        return hashlib.sha256(machine.encode()).hexdigest(), "machine-id"
    return hashlib.sha256(socket.gethostname().encode()).hexdigest(), "hostname"


def task_parameters(tasks):
    """Per-task inputs, parameters, defaults and output IDs (shared with the test fixture)."""
    table = {
        "text-to-image": {
            "inputs": ["prompt", "negative_prompt"],
            "parameters": ["count"],
            "defaults": {"count": 1},
            "outputs": OUTPUTS["text-to-image"],
            "output_formats": {"image": "image/png"},
            "repeated_outputs": OUTPUTS["text-to-image"],
            "families": ["sdxl-base"],
        },
        "image-to-image": {
            "inputs": ["prompt", "negative_prompt", "image"],
            "parameters": ["strength"],
            "input_formats": ["image/png"],
            "outputs": OUTPUTS["image-to-image"],
            "output_formats": {"image": "image/png"},
            "families": ["sdxl-base"],
        },
        "upscale": {
            "inputs": ["image"],
            "parameters": ["tile", "tile_overlap"],
            "defaults": dict(TASK_DEFAULTS["upscale"]),
            "input_formats": ["image/png"],
            "outputs": OUTPUTS["upscale"],
            "output_formats": {"image": "image/png"},
            "families": ["esrgan"],
            "stages": ["upscaling", "saving"],
            "progress_unit": "tiles",
        },
    }
    return {task: table[task] for task in tasks}


def plan_load(options, authorized, device, available, precisions, budget_mb, estimate_mb):
    """Resolve `auto` options and authorized fallbacks before loading; deterministic, no effects.

    Without an authorization nothing here changes the request, so the unauthorized path is the
    0.3.0 behavior exactly. `estimate_mb(precision)` is the header-based estimate.
    """
    resolved, resolutions, applied = dict(options), [], []
    accelerator = device != "cpu"
    budget = f"{budget_mb} MiB budget" if budget_mb is not None else None
    if options["precision"] == "auto":
        if accelerator and "float16" in precisions:
            value = "float16"
            estimate = estimate_mb(value)
            if not available:
                reason = "the selected device is unavailable; float16 is its lowest-memory advertised precision"
            elif budget is None:
                reason = f"float16 is the lowest-memory precision this device advertises; estimate {estimate} MiB, no budget or device memory measurable"
            elif estimate <= budget_mb:
                reason = f"float16 estimate {estimate} MiB fits the {budget}"
            else:
                reason = f"float16 estimate {estimate} MiB exceeds the {budget}; it is the lowest-memory advertised precision"
        else:
            value = "float32"
            reason = "the selected device advertises float32 only"
        resolved["precision"] = value
        resolutions.append(
            {"option": "precision", "requested": "auto", "resolved": value, "reason": reason}
        )
    run_device = device
    if accelerator:
        precision = resolved["precision"]
        over = available and budget_mb is not None and estimate_mb(precision) > budget_mb
        if over and "lower_precision" in authorized:
            lower = [
                p
                for p in precisions
                if PRECISION_BYTES[p] < PRECISION_BYTES[precision]
                and estimate_mb(p) <= budget_mb
            ]
            if lower:
                resolved["precision"] = lower[0]
                applied.append(
                    {
                        "fallback": "lower_precision",
                        "reason": f"{precision} estimate {estimate_mb(precision)} MiB exceeds the {budget}; {lower[0]} needs {estimate_mb(lower[0])} MiB",
                        "device": device,
                        "precision": lower[0],
                    }
                )
                over = False
        if (not available or over) and "cpu" in authorized:
            reason = (
                "the selected device is unavailable"
                if not available
                else f"{resolved['precision']} estimate {estimate_mb(resolved['precision'])} MiB exceeds the {budget}"
            )
            resolved["precision"], run_device = "float32", "cpu"
            applied.append(
                {"fallback": "cpu", "reason": reason, "device": "cpu", "precision": "float32"}
            )
    return {
        "options": resolved,
        "resolutions": resolutions,
        "applied_fallbacks": applied,
        "device": run_device,
    }


class TorchBackend:
    """Shared readiness, device and measurement behavior for both SDXL backends."""

    backend_id = "torch"
    ownership = "reference"
    dependencies = DEPENDENCIES

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        self.device = device
        self.run_device = device
        self.memory_budget_mb = memory_budget_mb
        self.loaded_estimate = None
        self.applied_fallbacks = []
        self.family = None
        self.upscaler = None
        self.upscale_config = None

    def missing_dependencies(self, dependencies=None):
        return [
            name
            for name, module in dependencies or self.dependencies
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

    def served_families(self):
        families = []
        if not self.missing_dependencies():
            families.append("sdxl-base")
        if not self.missing_dependencies(UPSCALE_DEPENDENCIES):
            families.append("esrgan")
        return families

    def loaders(self, families):
        return [
            {
                "format": FAMILIES[f]["format"],
                "families": [f],
                "roles": list(FAMILIES[f]["roles"]),
                "tasks": list(FAMILIES[f]["tasks"]),
                # The upscaler has no library pipeline: its network is engine code on every backend.
                "ownership": "owned" if f == "esrgan" else self.ownership,
            }
            for f in families
        ]

    def capabilities(self):
        missing = self.missing_dependencies()
        device_available = self.device_available()
        families = self.served_families()
        tasks = [t for t in TASKS if any(t in FAMILIES[f]["tasks"] for f in families)] or [
            "text-to-image",
            "image-to-image",
        ]
        return {
            "protocol": PROTOCOL,
            "supported_protocols": list(SUPPORTED_PROTOCOLS),
            "worker_version": WORKER_VERSION,
            "backend_id": self.backend_id,
            "ownership": self.ownership,
            # After an authorized CPU fallback the loaded bundle runs on the host CPU.
            "ready": not missing and (device_available or self.run_device == "cpu"),
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
                for name, module in self.dependencies
            },
            "max_concurrency": 1,
            "model_families": families or ["sdxl-base"],
            "loaders": self.loaders(families),
            "tasks": tasks,
            "task_parameters": task_parameters(tasks),
            "adapters": {
                "formats": [ADAPTER_FORMAT],
                "targets": list(ADAPTER_TARGETS),
                "max_adapters": MAX_ADAPTERS,
                "scale": LIMITS["adapter_scale"],
                "identity": "load",
                "families": ["sdxl-base"],
            },
            "defaults": DEFAULTS,
            "task_defaults": TASK_DEFAULTS,
            "load_defaults": {
                "precision": "float32" if self.device == "cpu" else "float16",
                "attention": "sdpa",
                "offload": "none",
            },
            "limits": LIMITS,
            "precisions": self.precisions(),
            "auto_options": AUTO_OPTIONS,
            "fallbacks": FALLBACKS,
            "formats": list(FORMATS),
        }

    def precisions(self):
        return ["float32"] if self.device == "cpu" else ["float16", "float32"]

    def device_identity(self):
        host_id, source = host_identity()
        identity = {
            "device": self.device,
            "index": None if self.device == "cpu" else int(self.device.split(":")[1]),
            "name": "cpu" if self.device == "cpu" else None,
            "total_memory_mb": None,
            "host_id": host_id,
            "host_id_source": source,
            "pid": os.getpid(),
        }
        if self.device == "cpu":
            try:
                identity["total_memory_mb"] = (
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024**2
                )
            except (ValueError, OSError, AttributeError):
                pass
        elif self.device_available():
            import torch

            properties = torch.cuda.get_device_properties(self.device)
            identity["name"] = properties.name
            identity["total_memory_mb"] = properties.total_memory // 1024**2
        return identity

    def compute_estimate(self, root, options, adapters=(), family="sdxl-base", device=None):
        estimate = memory_estimate(root, options["precision"], device or self.device, family)
        bytes_per_element = PRECISION_BYTES[options["precision"]]
        estimate["adapters"] = [
            {
                "adapter_id": record["adapter_id"],
                "parameters": layout["parameters"],
                "loaded_mb": layout["parameters"] * bytes_per_element // 1024**2,
            }
            for record, _root, _weights, _scale, layout in adapters
        ]
        estimate["value_mb"] += sum(a["loaded_mb"] for a in estimate["adapters"])
        return estimate

    def estimate(self, root, options, adapters=(), family="sdxl-base", device=None):
        self.loaded_estimate = self.compute_estimate(root, options, adapters, family, device)
        return self.loaded_estimate

    def budget_mb(self):
        if self.memory_budget_mb or self.device == "cpu" or not self.device_available():
            return self.memory_budget_mb
        import torch

        return torch.cuda.get_device_properties(self.device).total_memory // 1024**2

    def plan(self, root, options, adapters=(), family="sdxl-base", authorized=()):
        if options["precision"] != "auto" and not authorized:
            # 1.0/1.1 bodies and 1.2 bodies without auto or fallbacks: nothing to resolve.
            return {"options": dict(options), "resolutions": [], "applied_fallbacks": [], "device": self.device}
        available = self.device_available()
        return plan_load(
            options,
            authorized,
            self.device,
            available,
            self.precisions(),
            self.budget_mb() if available else None,
            lambda p: self.compute_estimate(root, {**options, "precision": p}, adapters, family)[
                "value_mb"
            ],
        )

    def apply_budget(self):
        import torch

        if self.run_device.startswith("cuda") and self.memory_budget_mb:
            total = torch.cuda.get_device_properties(self.run_device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, self.memory_budget_mb * 1024**2 / total), self.run_device
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

        cuda = self.run_device.startswith("cuda")
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.run_device) // 1024**2
            if cuda
            else None,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.run_device) // 1024**2
            if cuda
            else None,
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        }

    def begin(self, cancelled):
        import torch

        if self.run_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.run_device)
        if cancelled.is_set():
            raise Cancelled()

    def finish(self):
        import torch

        if self.run_device.startswith("cuda"):
            torch.cuda.synchronize(self.run_device)
            torch.cuda.empty_cache()

    def load_image(self, path, width, height):
        """Decode the approved PNG input; a size other than the request is an error, not a resize."""
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError("input image size does not match the request")
            return image.convert("RGB")

    # ----- lifecycle shared by every family ---------------------------------------------------
    def load(self, root, options, adapters=(), family="sdxl-base", device=None):
        self.run_device = device or self.device
        self.family = family
        if family == "esrgan":
            return self.load_upscaler(root, options)
        return self.load_pipeline(root, options, adapters)

    def unload(self):
        self.upscaler = self.upscale_config = None
        self.unload_pipeline()
        measurements = self.release_device()
        self.run_device, self.family, self.applied_fallbacks = self.device, None, []
        return measurements

    def load_upscaler(self, root, options):
        offline_environment()
        import torch

        from . import upscale

        self.apply_budget()
        config = upscale.read_config(root)
        weights = next(Path(root).glob("*.safetensors"))
        self.upscaler = upscale.load(
            root, weights, config, self.run_device, getattr(torch, options["precision"])
        )
        self.upscale_config = config

    def generate(self, request, directory, emit, cancelled, image_path=None):
        """Write every output for the request into `directory`; return (records, measurements)."""
        if request["task"] == "upscale":
            return self.upscale(request, directory, emit, cancelled, image_path)
        return self.generate_images(request, directory, emit, cancelled, image_path)

    def upscale(self, request, directory, emit, cancelled, image_path):
        import torch

        from . import png, upscale

        if self.upscaler is None:
            raise RuntimeError("Model not loaded")
        params = {**TASK_DEFAULTS["upscale"], **request["parameters"]}
        scale = self.upscale_config["scale"]
        self.begin(cancelled)
        try:
            with torch.inference_mode():
                emit("stage", {"stage": "upscaling"})
                image = png.png_to_tensor(Path(image_path).read_bytes())
                if tuple(image.shape[1:]) != (params["height"], params["width"]):
                    raise ValueError("input image size does not match the request")
                dtype = next(self.upscaler.parameters()).dtype
                result = upscale.tile(
                    self.upscaler,
                    image.unsqueeze(0).to(self.run_device, dtype),
                    scale,
                    params["tile"],
                    params["tile_overlap"],
                    emit,
                    cancelled,
                    Cancelled,
                )
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "saving"})
                path = Path(directory) / "image.png"
                path.write_bytes(png.tensor_to_png(result[0].clamp(0, 1)))
                record = {
                    "output_id": "image",
                    "path": path,
                    "mime_type": "image/png",
                    "width": params["width"] * scale,
                    "height": params["height"] * scale,
                }
                return [record], {
                    "tiles": upscale.tile_count(params["width"], params["height"], params["tile"]),
                    **self.measurements(),
                }
        finally:
            self.finish()

    @staticmethod
    def seeds(params):
        """Output k of a batch uses seed + k − 1 so any single image can be reproduced alone."""
        return [params["seed"] + k for k in range(params.get("count", 1))]


class DiffusersBackend(TorchBackend):
    """Reference worker stage: the complete Diffusers pipelines execute inference."""

    backend_id = "diffusers-sdxl"
    ownership = "reference"

    def __init__(self, device="cuda:0", memory_budget_mb=None):
        super().__init__(device, memory_budget_mb)
        self.pipe = None
        self.img2img = None

    def load_pipeline(self, root, options, adapters=()):
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
        if adapters:
            # The library injects PEFT layers; adapters are part of this load's identity.
            for record, adapter_root, weights, _scale, _layout in adapters:
                pipe.load_lora_weights(
                    str(adapter_root),
                    weight_name=weights.name,
                    adapter_name=record["adapter_id"],
                    local_files_only=True,
                )
            pipe.set_adapters(
                [record["adapter_id"] for record, *_ in adapters],
                [float(scale) for _r, _a, _w, scale, _l in adapters],
            )
        pipe.to(self.run_device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.img2img = None

    def unload_pipeline(self):
        self.pipe = None
        self.img2img = None

    def generate_images(self, request, directory, emit, cancelled, image_path=None):
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
            seeds = self.seeds(params)
            emit("stage", {"stage": "encoding"})
            with torch.inference_mode():
                embeddings = pipe.encode_prompt(
                    prompt=request["inputs"]["prompt"],
                    device=self.run_device,
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

                generators = [
                    torch.Generator(device=self.run_device).manual_seed(s) for s in seeds
                ]
                common = dict(
                    prompt_embeds=embeddings[0],
                    negative_prompt_embeds=embeddings[1],
                    pooled_prompt_embeds=embeddings[2],
                    negative_pooled_prompt_embeds=embeddings[3],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    # One generator per image: each latent is drawn exactly as a count-1 run would.
                    generator=generators[0] if len(seeds) == 1 else generators,
                    num_images_per_prompt=len(seeds),
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
                    latents = self.img2img(
                        image=image, strength=params["strength"], **common
                    ).images
                else:
                    latents = pipe(
                        width=params["width"], height=params["height"], **common
                    ).images
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "decoding"})
                if pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast:
                    pipe.upcast_vae()
                latents = latents.to(
                    next(iter(pipe.vae.post_quant_conv.parameters())).dtype
                )
                images = []
                for latent in latents:
                    if cancelled.is_set():
                        raise Cancelled()
                    decoded = pipe.vae.decode(
                        latent.unsqueeze(0) / pipe.vae.config.scaling_factor, return_dict=False
                    )[0]
                    images.append(pipe.image_processor.postprocess(decoded, output_type="pil")[0])
                if cancelled.is_set():
                    raise Cancelled()
                emit("stage", {"stage": "saving"})
                records = []
                for output_id, output in zip(output_ids(task, params), images):
                    path = Path(directory) / (output_id + ".png")
                    output.save(path, format="PNG")
                    records.append(
                        {
                            "output_id": output_id,
                            "path": path,
                            "mime_type": "image/png",
                            "width": output.size[0],
                            "height": output.size[1],
                        }
                    )
                return records, {"effective_steps": total, **self.measurements()}
        finally:
            self.finish()


BACKENDS = {}


def backend_class(backend_id):
    from .pipeline import OwnedPipelineBackend

    BACKENDS.setdefault(DiffusersBackend.backend_id, DiffusersBackend)
    BACKENDS.setdefault(OwnedPipelineBackend.backend_id, OwnedPipelineBackend)
    return BACKENDS[backend_id]
