"""Explicit reference-pipeline CPU offload; never an automatic or implicit fallback."""
import importlib.util

from .contracts import WorkerError

MODES = ("none", "model", "sequential")
FAMILIES = ("sdxl-base", "svd", "stable-audio-open")


def capabilities(backend_id, device, ready):
    enabled = (backend_id == "diffusers-sdxl" and device.startswith("cuda:") and ready
               and importlib.util.find_spec("accelerate") is not None)
    return {"offload_modes": list(MODES) if enabled else ["none"],
            "offload_families": list(FAMILIES) if enabled else []}


def validate(options, minor, caps, family):
    mode = options.get("offload")
    if mode == "none":
        return
    if (minor < 4 or mode not in caps.get("offload_modes", ["none"])
            or family not in caps.get("offload_families", [])
            or options.get("quantization", "none") != "none"
            or options.get("compile", "none") != "none"):
        raise WorkerError(400, "unsupported", "options.offload")


def place(pipe, mode, device):
    """Attach hooks before any whole-pipeline CUDA transfer (the point of offload)."""
    if mode == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif mode == "none":
        pipe.to(device)
    else:
        raise WorkerError(400, "unsupported", "options.offload")


def release(pipe, mode):
    # The Engine decodes latents itself after the Diffusers sampling call. Return that
    # last VAE to CPU too, including on cancellation and on image-to-image reuse.
    if pipe is not None and mode == "model":
        for hook in getattr(pipe, "_all_hooks", ()):
            hook.offload()
