"""Explicit low-resolution latent RGB previews; never final decoded generation outputs."""
from . import png

MAX_BYTES = 1024 * 1024
MAX_RETAINED = 4


def emit_latents(latents, emit, interval, step):
    if not interval or step % interval:
        return
    # A bounded, deterministic approximation, not a VAE decode. The explicit preview request
    # never changes sampling or consumes RNG. Select the first image of a batch.
    image = latents[0, :3].detach().float()
    height, width = image.shape[-2:]
    image = image[:, ::max(1, (height + 127) // 128), ::max(1, (width + 127) // 128)]
    image = ((image.tanh() + 1) * 127.5).round().clamp(0, 255).to(device="cpu", dtype=__import__("torch").uint8)
    height, width = image.shape[-2:]
    data = png.write(width, height, image.permute(1, 2, 0).contiguous().numpy().tobytes())
    emit("preview", {"png": data, "width": width, "height": height, "step": step, "kind": "latent_rgb"})
