"""Engine-owned RRDBNet super-resolution network for the `esrgan` family.

The architecture (residual-in-residual dense blocks with nearest-neighbour + convolution
upsampling, or sub-pixel convolution) is this project's code; only the weights are the
operator's. The key layout matches the published Real-ESRGAN RRDBNet so converted x2/x4
releases load unchanged. `architecture()` is pure Python so bundle validation can check a
weights header against the configuration without importing torch.
"""

import json
import math
from pathlib import Path

CONFIG_KEYS = {"architecture", "scale", "in_channels", "out_channels", "num_feat", "num_block", "num_grow_ch"}
UPSAMPLERS = ("nearest-conv", "pixel-shuffle")
SCALES = (2, 4)


def validate_config(config):
    """Return the normalized configuration or None when it is not a supported RRDBNet."""
    if not isinstance(config, dict) or not CONFIG_KEYS <= config.keys():
        return None
    if config.keys() - CONFIG_KEYS - {"upsampler"}:
        return None
    if config["architecture"] != "rrdbnet" or config.get("upsampler", "nearest-conv") not in UPSAMPLERS:
        return None
    if type(config["scale"]) is not int or config["scale"] not in SCALES:
        return None
    for key in ("in_channels", "out_channels", "num_feat", "num_block", "num_grow_ch"):
        if type(config[key]) is not int or not 1 <= config[key] <= 1024:
            return None
    return {"upsampler": "nearest-conv", **config}


def unshuffle(config):
    """Real-ESRGAN's x2 network runs its body at half resolution (pixel-unshuffle by 2)."""
    return 2 if config["upsampler"] == "nearest-conv" and config["scale"] == 2 else 1


def architecture(config):
    """Expected parameter names and shapes for a validated configuration."""
    nf, gc, blocks = config["num_feat"], config["num_grow_ch"], config["num_block"]
    shapes = {}

    def conv(name, out_ch, in_ch):
        shapes[name + ".weight"] = [out_ch, in_ch, 3, 3]
        shapes[name + ".bias"] = [out_ch]

    conv("conv_first", nf, config["in_channels"] * unshuffle(config) ** 2)
    for b in range(blocks):
        for r in (1, 2, 3):
            for c in range(1, 5):
                conv(f"body.{b}.rdb{r}.conv{c}", gc, nf + (c - 1) * gc)
            conv(f"body.{b}.rdb{r}.conv5", nf, nf + 4 * gc)
    conv("conv_body", nf, nf)
    if config["upsampler"] == "nearest-conv":
        conv("conv_up1", nf, nf)
        conv("conv_up2", nf, nf)
    else:
        for i in range(1, int(math.log2(config["scale"])) + 1):
            conv(f"conv_up{i}", nf * 4, nf)
    conv("conv_hr", nf, nf)
    conv("conv_last", config["out_channels"], nf)
    return shapes


def activation_bytes(config, side, bytes_per_element):
    """Largest simultaneous feature maps for one square input region of `side` pixels.

    Two terms dominate: the dense-block concatenation at body resolution (nf + 4·gc channels plus
    the residual input) and the two full-resolution feature maps of the last upsampling stage.
    """
    nf, gc, scale = config["num_feat"], config["num_grow_ch"], config["scale"]
    body = (side // unshuffle(config)) ** 2 * (2 * nf + 4 * gc)
    upsampled = (side * scale) ** 2 * nf * 2 + (side * scale) ** 2 * (config["out_channels"])
    return max(body, upsampled) * bytes_per_element


def read_config(root):
    try:
        return validate_config(json.loads((Path(root) / "config.json").read_text()))
    except (OSError, ValueError):
        return None


def build(config):
    """Construct the torch module for a validated configuration (imports torch lazily)."""
    import torch
    from torch import nn
    from torch.nn import functional

    nf, gc = config["num_feat"], config["num_grow_ch"]

    class DenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            for c in range(1, 5):
                setattr(self, f"conv{c}", nn.Conv2d(nf + (c - 1) * gc, gc, 3, 1, 1))
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

        def forward(self, x):
            features = [x]
            for c in range(1, 5):
                features.append(
                    functional.leaky_relu(getattr(self, f"conv{c}")(torch.cat(features, 1)), 0.2)
                )
            return self.conv5(torch.cat(features, 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = DenseBlock(), DenseBlock(), DenseBlock()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = unshuffle(config)
            self.conv_first = nn.Conv2d(config["in_channels"] * self.factor**2, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(config["num_block"])])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            if config["upsampler"] == "nearest-conv":
                self.stages = 2
                self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
                self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            else:
                self.stages = int(math.log2(config["scale"]))
                for i in range(1, self.stages + 1):
                    setattr(self, f"conv_up{i}", nn.Conv2d(nf, nf * 4, 3, 1, 1))
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, config["out_channels"], 3, 1, 1)

        def forward(self, x):
            if self.factor > 1:
                x = functional.pixel_unshuffle(x, self.factor)
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            for i in range(1, self.stages + 1):
                conv = getattr(self, f"conv_up{i}")
                if config["upsampler"] == "nearest-conv":
                    feat = conv(functional.interpolate(feat, scale_factor=2, mode="nearest"))
                else:
                    feat = functional.pixel_shuffle(conv(feat), 2)
                feat = functional.leaky_relu(feat, 0.2)
            return self.conv_last(functional.leaky_relu(self.conv_hr(feat), 0.2))

    return RRDBNet()


def load(root, weights, config, device, dtype):
    """Validate the state dict against the configured architecture, then move it to the device."""
    import torch
    from safetensors.torch import load_file

    state = load_file(str(weights), device="cpu")
    expected = architecture(config)
    if set(state) != set(expected) or any(list(state[k].shape) != v for k, v in expected.items()):
        raise ValueError("network weights do not match the configured architecture")
    with torch.device("meta"):
        network = build(config)
    network.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
    return network.to(device).eval()


def regions(width, height, tile, overlap, align):
    """Yield (core box, padded box) tiles covering the image; 0 means one whole-image tile."""
    if not tile:
        yield (0, 0, width, height), (0, 0, width, height)
        return
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            core = (x, y, min(x + tile, width), min(y + tile, height))
            left = max(0, (core[0] - overlap) // align * align)
            top = max(0, (core[1] - overlap) // align * align)
            right = min(width, -(-(core[2] + overlap) // align) * align)
            bottom = min(height, -(-(core[3] + overlap) // align) * align)
            yield core, (left, top, right, bottom)


def tile_count(width, height, tile):
    return 1 if not tile else math.ceil(width / tile) * math.ceil(height / tile)


def tile(network, image, scale, tile_size, overlap, emit, cancelled, cancel_error):
    """Run the network over overlapping tiles and stitch the cores; image is (1, C, H, W)."""
    import torch

    _, _, height, width = image.shape
    total = tile_count(width, height, tile_size)
    output = torch.zeros(
        (1, network.conv_last.out_channels, height * scale, width * scale),
        dtype=image.dtype,
        device=image.device,
    )
    align = getattr(network, "factor", 1)
    for index, (core, padded) in enumerate(regions(width, height, tile_size, overlap, align)):
        if cancelled.is_set():
            raise cancel_error()
        left, top, right, bottom = padded
        result = network(image[:, :, top:bottom, left:right])
        x0, y0 = (core[0] - left) * scale, (core[1] - top) * scale
        x1, y1 = x0 + (core[2] - core[0]) * scale, y0 + (core[3] - core[1]) * scale
        output[:, :, core[1] * scale : core[3] * scale, core[0] * scale : core[2] * scale] = result[
            :, :, y0:y1, x0:x1
        ]
        emit(
            "progress",
            {
                "stage": "upscaling",
                "scope": "stage",
                "unit": "tiles",
                "completed_steps": index + 1,
                "total_steps": total,
            },
        )
    return output
