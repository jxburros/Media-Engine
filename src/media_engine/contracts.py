"""Validation shared by every worker endpoint; never accepts executable model code."""

import hashlib
import json
import math
import re
import struct
from fractions import Fraction
from pathlib import Path

PROTOCOL = "media-worker/1.3"
SUPPORTED_PROTOCOLS = (
    "media-worker/1.0",
    "media-worker/1.1",
    "media-worker/1.2",
    "media-worker/1.3",
)
WORKER_VERSION = "1.0.0"
TASKS = ("text-to-image", "image-to-image", "upscale", "image-to-video", "text-to-audio")
# The minor that introduced each task; a lower-minor request naming it is `unsupported`.
TASK_SINCE = {
    "text-to-image": 0,
    "image-to-image": 1,
    "upscale": 2,
    "image-to-video": 3,
    "text-to-audio": 3,
}
SDXL_ROLES = (
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
)
FAMILIES = {
    "sdxl-base": {
        "format": "diffusers-safetensors",
        "pipeline": "text-to-image",
        "roles": SDXL_ROLES,
        "weight_roles": SDXL_ROLES[:4],
        "approved_classes": {
            "unet": ("diffusers", "UNet2DConditionModel"),
            "vae": ("diffusers", "AutoencoderKL"),
            "text_encoder": ("transformers", "CLIPTextModel"),
            "text_encoder_2": ("transformers", "CLIPTextModelWithProjection"),
            "tokenizer": ("transformers", "CLIPTokenizer"),
            "tokenizer_2": ("transformers", "CLIPTokenizer"),
            "scheduler": ("diffusers", "EulerDiscreteScheduler"),
        },
        "tasks": ("text-to-image", "image-to-image"),
        "config_file": "model_index.json",
        "pipeline_class": "StableDiffusionXLPipeline",
        "since": 0,
    },
    "esrgan": {
        "format": "single-file-safetensors",
        "pipeline": "upscale",
        "roles": ("network", "configuration"),
        "weight_roles": ("network",),
        "approved_classes": {},
        "tasks": ("upscale",),
        "config_file": "config.json",
        "since": 2,
    },
    "svd": {
        "format": "diffusers-safetensors",
        "pipeline": "image-to-video",
        "roles": ("unet", "vae", "image_encoder", "feature_extractor", "scheduler"),
        "weight_roles": ("unet", "vae", "image_encoder"),
        "approved_classes": {
            "unet": ("diffusers", "UNetSpatioTemporalConditionModel"),
            "vae": ("diffusers", "AutoencoderKLTemporalDecoder"),
            "image_encoder": ("transformers", "CLIPVisionModelWithProjection"),
            "feature_extractor": ("transformers", "CLIPImageProcessor"),
            "scheduler": ("diffusers", "EulerDiscreteScheduler"),
        },
        "tasks": ("image-to-video",),
        "config_file": "model_index.json",
        "pipeline_class": "StableVideoDiffusionPipeline",
        "since": 3,
    },
    "stable-audio-open": {
        "format": "diffusers-safetensors",
        "pipeline": "text-to-audio",
        "roles": ("transformer", "vae", "text_encoder", "tokenizer", "projection_model", "scheduler"),
        "weight_roles": ("transformer", "vae", "text_encoder", "projection_model"),
        "approved_classes": {
            "transformer": ("diffusers", "StableAudioDiTModel"),
            "vae": ("diffusers", "AutoencoderOobleck"),
            "text_encoder": ("transformers", "T5EncoderModel"),
            "tokenizer": ("transformers", "T5TokenizerFast"),
            # The library registers the projection under its pipeline module, not `diffusers`.
            "projection_model": ("stable_audio", "StableAudioProjectionModel"),
            "scheduler": ("diffusers", "CosineDPMSolverMultistepScheduler"),
        },
        "tasks": ("text-to-audio",),
        "config_file": "model_index.json",
        "pipeline_class": "StableAudioPipeline",
        "scheduler_class": "CosineDPMSolverMultistepScheduler",
        "since": 3,
    },
}
# 1.0/1.1 names, kept for importers; they describe the sdxl-base family.
ROLES = FAMILIES["sdxl-base"]["roles"]
WEIGHT_ROLES = FAMILIES["sdxl-base"]["weight_roles"]
APPROVED_CLASSES = FAMILIES["sdxl-base"]["approved_classes"]
FORMATS = ("diffusers-safetensors", "single-file-safetensors")
OUTPUTS = {
    "text-to-image": ["image"],
    "image-to-image": ["image"],
    "upscale": ["image"],
    "image-to-video": ["frames", "poster"],
    "text-to-audio": ["audio"],
}
# Output MIME types and the file suffix each is stored under in the execution directory.
OUTPUT_SUFFIXES = {"image/png": ".png", "application/zip": ".zip", "audio/wav": ".wav"}
DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 6,
    "sampler": "euler",
    "scheduler": "euler",
}
TASK_DEFAULTS = {
    "image-to-image": {"strength": 0.75},
    "upscale": {"tile": 0, "tile_overlap": 16},
    # `frames` defaults to the bundle's unet `num_frames`; it is resolved once the bundle loads.
    "image-to-video": {"fps": 7, "motion": 127, "noise_augmentation": 0.02, "decode_chunk": 8},
    # The library pipeline's default guidance; seed, steps and duration_seconds are required.
    "text-to-audio": {"guidance": 7},
}
# The reference pipeline's per-frame guidance ramp; not a request parameter in 1.3, so it is
# reported in effective_settings rather than left implicit.
VIDEO_GUIDANCE = {"min_guidance": 1.0, "max_guidance": 3.0}
MAX_ADAPTERS = 4
ADAPTER_TARGETS = ("unet", "text_encoder", "text_encoder_2")
ADAPTER_FORMAT = "peft-lora"
LIMITS = {
    "width": [64, 2048],
    "height": [64, 2048],
    "steps": [1, 150],
    "guidance": [0, 30],
    "strength": [0, 1],
    "adapter_scale": [0, 2],
    "size_multiple": 8,
    "count": [1, 4],
    "tile_sizes": [256, 512, 1024],
    "tile_overlap": [0, 64],
    "output_side": 8192,
    "video_steps": [1, 50],
    "frames": [1, 64],
    "fps": [1, 30],
    "motion": [1, 255],
    "noise_augmentation": [0, 1],
    "decode_chunk": [1, 16],
    # The reference svd bundle's sizes; a bundle may declare its own in video.json.
    "video_sizes": [[1024, 576], [576, 1024]],
    "audio_steps": [1, 200],
    # The reference stable-audio-open bundle's range; a loaded bundle advertises its own.
    "duration_seconds": [1, 47],
    "sample_rate": [1000, 192000],
    "audio_channels": [1, 2],
}
# The 1.3 limit keys, so a worker pinned to 1.2 can advertise exactly the 0.5.0 limits.
VIDEO_LIMITS = ("video_steps", "frames", "fps", "motion", "noise_augmentation", "decode_chunk", "video_sizes")
AUDIO_LIMITS = ("audio_steps", "duration_seconds", "sample_rate", "audio_channels")
MAX_VIDEO_SIZES = 8
# No bundle may declare a clip longer than this, whatever its latent length allows; it bounds
# request validation before a bundle is loaded and the storage reserved for one clip.
MAX_AUDIO_SECONDS = 600
AUTO_OPTIONS = ["precision"]
FALLBACKS = ["cpu", "lower_precision"]
PRECISION_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}
SAFETENSORS_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


class WorkerError(Exception):
    def __init__(self, status, code, field=None):
        self.status, self.code, self.field = status, code, field


def strict(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise WorkerError(400, "invalid_request")


def protocol(body):
    """Return the negotiated minor the caller speaks; a different major/minor is rejected."""
    if body.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise WorkerError(400, "protocol_mismatch", "protocol")
    return body["protocol"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def minor(version):
    """The minor number of a supported protocol string (`media-worker/1.2` → 2)."""
    return int(version.rsplit(".", 1)[1])


def role_files(family, role, files):
    """The subset of `files` a component digest covers for this family's layout."""
    if FAMILIES[family]["format"] == "diffusers-safetensors":
        return {k: v for k, v in files.items() if k.startswith(role + "/")}
    if role == "network":
        return {k: v for k, v in files.items() if k.endswith(".safetensors")}
    return {k: v for k, v in files.items() if k == "config.json"}


def bundle_path(bundle, catalog, verify=True, version=PROTOCOL):
    level = minor(version)
    strict(
        bundle,
        (
            "schema_version",
            "bundle_id",
            "bundle_digest",
            "family",
            "pipeline",
            "components",
            "configuration",
            "source",
        ),
        ("format",) if level >= 2 else (),
    )
    # The (format, family) pair is checked before anything on disk is read.
    name = bundle["family"] if isinstance(bundle["family"], str) else None
    family = FAMILIES.get(name)
    if (
        bundle["schema_version"] != 1
        or family is None
        or family["since"] > level
        or bundle.get("format", "diffusers-safetensors") != family["format"]
        or bundle["pipeline"] != family["pipeline"]
    ):
        raise WorkerError(400, "unsupported", "family")
    if (
        digest({k: v for k, v in bundle.items() if k != "bundle_digest"})
        != bundle["bundle_digest"]
    ):
        raise WorkerError(400, "artifact_changed", "bundle_digest")
    if not isinstance(bundle["bundle_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", bundle["bundle_id"]
    ):
        raise WorkerError(400, "invalid_request", "bundle_id")
    strict(
        bundle["source"],
        (),
        ("repository", "revision", "license") + (("conversion",) if level >= 2 else ()),
    )
    if any(
        not isinstance(v, str) for k, v in bundle["source"].items() if k != "conversion"
    ):
        raise WorkerError(400, "invalid_request", "source")
    conversion = bundle["source"].get("conversion")
    if conversion is not None:
        check_conversion(conversion)
    configuration = bundle["configuration"]
    strict(configuration, ("artifact_handle", "files"))
    handle = configuration["artifact_handle"]
    if not isinstance(handle, str) or handle not in catalog:
        raise WorkerError(400, "missing_component", "artifact_handle")
    root = Path(catalog[handle]).resolve()
    files = configuration["files"]
    if not isinstance(files, dict) or not files:
        raise WorkerError(400, "missing_component", "configuration")
    if family["format"] == "single-file-safetensors" and (
        len(files) != 2
        or "config.json" not in files
        or any("/" in k for k in files)
        or len(role_files(name, "network", files)) != 1
    ):
        # One weights file plus one configuration file; nothing else is part of the layout.
        raise WorkerError(400, "unsupported", "configuration")
    roles_expected = family["roles"]
    if not isinstance(bundle["components"], list) or len(bundle["components"]) != len(
        roles_expected
    ):
        raise WorkerError(400, "missing_component", "components")
    roles = set()
    for component in bundle["components"]:
        strict(component, ("role", "artifact_handle", "digest", "tensor_format"))
        role = component["role"]
        if (
            role not in roles_expected
            or role in roles
            or component["artifact_handle"] != handle
        ):
            raise WorkerError(400, "unsupported", "components")
        roles.add(role)
        entries = role_files(name, role, files)
        if not entries or digest(entries) != component["digest"]:
            raise WorkerError(400, "artifact_changed", role)
        expected_format = "safetensors" if role in family["weight_roles"] else "json"
        if component["tensor_format"] != expected_format:
            raise WorkerError(400, "unsupported", role)
        if expected_format == "safetensors" and not any(
            k.endswith(".safetensors") for k in entries
        ):
            raise WorkerError(400, "missing_component", role)
    if not root.is_dir():
        raise WorkerError(400, "missing_component", "configuration")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WorkerError(400, "unsupported", "symlink")
        if path.is_file():
            relative = str(path.relative_to(root))
            if path.suffix in {".py", ".pkl", ".pickle", ".bin", ".pt", ".pth"}:
                raise WorkerError(400, "unsupported", "tensor_format")
            actual[relative] = path
    if set(actual) != set(files):
        raise WorkerError(400, "artifact_changed", "configuration")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise WorkerError(400, "invalid_request", "digest")
        if verify:
            with actual[relative].open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    if family["format"] == "single-file-safetensors":
                        raise WorkerError(
                            400,
                            "artifact_changed",
                            "configuration" if relative == "config.json" else "network",
                        )
                    raise WorkerError(400, "artifact_changed", relative.split("/")[0])
    if conversion is not None:
        # Provenance named in the registration must be the provenance the directory carries.
        if name != "sdxl-base" or "conversion.json" not in actual:
            raise WorkerError(400, "artifact_changed", "conversion")
        try:
            recorded = json.loads(actual["conversion.json"].read_text())
        except (OSError, ValueError):
            raise WorkerError(400, "artifact_changed", "conversion") from None
        if not isinstance(recorded, dict) or any(
            recorded.get(k) != v for k, v in conversion.items()
        ):
            raise WorkerError(400, "artifact_changed", "conversion")
    if name == "esrgan":
        return check_esrgan(root, actual)
    try:
        index = json.loads((root / "model_index.json").read_text())
        scheduler = json.loads((root / "scheduler/scheduler_config.json").read_text())
    except (OSError, ValueError):
        raise WorkerError(400, "missing_component", "configuration") from None
    if (
        index.get("_class_name") != family["pipeline_class"]
        or scheduler.get("_class_name") != family.get("scheduler_class", "EulerDiscreteScheduler")
    ):
        raise WorkerError(400, "unsupported", "pipeline")
    # Only standard installed library classes; reject custom component imports.
    approved = family["approved_classes"]
    for key, value in index.items():
        if key.startswith("_") or key in approved:
            continue
        if key == "force_zeros_for_empty_prompt" and type(value) is bool:
            continue
        if key in {"feature_extractor", "image_encoder"} and value == [None, None]:
            continue
        raise WorkerError(400, "unsupported", "pipeline")
    for role, value in approved.items():
        if index.get(role) != list(value):
            raise WorkerError(400, "unsupported", role)
    if name == "svd":
        video_facts(root)  # sizes and the default frame count must be usable before loading
    if name == "stable-audio-open":
        audio_facts(root)  # sample rate, channels and duration range must be usable before loading
    return root


CONVERSION_FIELDS = ("source_sha256", "reference_bundle_digest", "converter", "mapped_keys")
CONVERSION_OPTIONAL = ("schema_version", "precision", "unmapped_keys", "ignored_keys")


def check_conversion(conversion):
    """`source.conversion` (1.2): the converted directory's conversion.json provenance."""
    strict(conversion, CONVERSION_FIELDS, CONVERSION_OPTIONAL)
    invalid = WorkerError(400, "invalid_request", "conversion")
    if not isinstance(conversion["source_sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", conversion["source_sha256"]
    ):
        raise invalid
    if not isinstance(conversion["reference_bundle_digest"], str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", conversion["reference_bundle_digest"]
    ):
        raise invalid
    converter = conversion["converter"]
    if (
        not isinstance(converter, dict)
        or set(converter) != {"media_engine", "diffusers", "torch", "safetensors"}
        or any(v is not None and not isinstance(v, str) for v in converter.values())
    ):
        raise invalid
    mapped = conversion["mapped_keys"]
    if (
        not isinstance(mapped, dict)
        or set(mapped) != set(FAMILIES["sdxl-base"]["weight_roles"])
        or any(type(v) is not int or v < 0 for v in mapped.values())
    ):
        raise invalid
    if conversion.get("schema_version", 1) != 1 or conversion.get("unmapped_keys", []) != []:
        raise invalid
    if "precision" in conversion and conversion["precision"] not in ("preserved", "float16", "float32"):
        raise invalid
    ignored = conversion.get("ignored_keys", [])
    if not isinstance(ignored, list) or any(not isinstance(k, str) for k in ignored):
        raise invalid


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if not 0 < length <= 64 * 1024 * 1024:
            raise WorkerError(400, "unsupported", "tensor_format")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise WorkerError(400, "unsupported", "tensor_format")
    return header


def check_esrgan(root, files):
    """The configuration must describe an RRDBNet and the network header must match it exactly."""
    from .upscale import architecture, read_config

    config = read_config(root)
    if config is None:
        raise WorkerError(400, "unsupported", "configuration")
    weights = next(p for k, p in files.items() if k.endswith(".safetensors"))
    try:
        header = safetensors_header(weights)
        shapes = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        raise WorkerError(400, "unsupported", "network") from None
    if shapes != architecture(config):
        raise WorkerError(400, "unsupported", "network")
    return root


def video_facts(root):
    """The svd bundle's advertised sizes and default frame count, read without loading weights.

    Sizes come from an optional top-level `video.json` (`{"sizes": [[width, height], …]}`) and
    default to the reference bundle's 1024×576 and 576×1024; the default frame count is the
    unet's `num_frames` (the library's default, 25, when the config omits it). A bundle whose
    sizes or default frame count fall outside the advertised limits is `unsupported`, field
    `configuration`, so nothing is decided later from an unusable bundle.
    """
    root = Path(root)
    unsupported = WorkerError(400, "unsupported", "configuration")
    sizes = LIMITS["video_sizes"]
    try:
        if (root / "video.json").is_file():
            declared = json.loads((root / "video.json").read_text())
            if not isinstance(declared, dict) or set(declared) != {"sizes"}:
                raise unsupported
            sizes = declared["sizes"]
        unet = json.loads((root / "unet/config.json").read_text())
    except (OSError, ValueError):
        raise unsupported from None
    if (
        not isinstance(sizes, list)
        or not 1 <= len(sizes) <= MAX_VIDEO_SIZES
        or not isinstance(unet, dict)
    ):
        raise unsupported
    for size in sizes:
        if not isinstance(size, list) or len(size) != 2:
            raise unsupported
        for value, field in zip(size, ("width", "height")):
            low, high = LIMITS[field]
            if type(value) is not int or not low <= value <= high or value % LIMITS["size_multiple"]:
                raise unsupported
    frames = unet.get("num_frames", 25)
    low, high = LIMITS["frames"]
    if type(frames) is not int or not low <= frames <= high:
        raise unsupported
    return {"video_sizes": [list(s) for s in sizes], "default_frames": frames}


def audio_facts(root):
    """The stable-audio-open bundle's sample rate, channels, granularity and duration range.

    Read from the configs without loading weights: `sampling_rate` and `audio_channels` from the
    VAE config, the latent granularity (`hop_length`, samples per latent frame) as the product of
    its `downsampling_ratios`, and the latent length from the transformer's `sample_size` (the
    library's defaults where a config omits them, which are the reference bundle's values). The
    longest clip the bundle can produce is `sample_size × hop_length / sampling_rate` seconds; the
    advertised range is an optional top-level `audio.json` (`{"duration_seconds": [low, high]}`)
    within that, else 1 s to the whole seconds it allows (1–47 for the reference bundle). The
    tokenizer must ship `tokenizer.json`, so the fast tokenizer loads without sentencepiece. An
    unusable bundle is `unsupported`, field `configuration` (`tokenizer` for the tokenizer).
    """
    root = Path(root)
    unsupported = WorkerError(400, "unsupported", "configuration")
    if not (root / "tokenizer/tokenizer.json").is_file():
        raise WorkerError(400, "unsupported", "tokenizer")
    try:
        vae = json.loads((root / "vae/config.json").read_text())
        transformer = json.loads((root / "transformer/config.json").read_text())
        declared = None
        if (root / "audio.json").is_file():
            declared = json.loads((root / "audio.json").read_text())
    except (OSError, ValueError):
        raise unsupported from None
    if not isinstance(vae, dict) or not isinstance(transformer, dict):
        raise unsupported
    rate = vae.get("sampling_rate", 44100)
    channels = vae.get("audio_channels", 2)
    ratios = vae.get("downsampling_ratios", [2, 4, 4, 8, 8])
    length = transformer.get("sample_size", 1024)
    if (
        type(rate) is not int
        or not LIMITS["sample_rate"][0] <= rate <= LIMITS["sample_rate"][1]
        or type(channels) is not int
        or not LIMITS["audio_channels"][0] <= channels <= LIMITS["audio_channels"][1]
        or not isinstance(ratios, list)
        or not ratios
        or any(type(r) is not int or r < 1 for r in ratios)
        or type(length) is not int
        or length < 1
    ):
        raise unsupported
    hop = math.prod(ratios)
    capacity = Fraction(length * hop, rate)
    if declared is None:
        low, high = 1, min(math.floor(capacity), MAX_AUDIO_SECONDS)
    else:
        if not isinstance(declared, dict) or set(declared) != {"duration_seconds"}:
            raise unsupported
        span = declared["duration_seconds"]
        if not isinstance(span, list) or len(span) != 2 or any(not is_number(v) for v in span):
            raise unsupported
        low, high = span
    if not 0 < low <= high or high > capacity or high > MAX_AUDIO_SECONDS:
        raise unsupported
    return {
        "sample_rate": rate,
        "channels": channels,
        "hop_length": hop,
        "duration_range": [low, high],
    }


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def audio_length(duration, facts):
    """The clip a requested duration produces: whole latent frames covering it, never fewer.

    The request is honored up to the latent granularity (`hop_length` samples): the clip is the
    smallest whole number of latent frames that covers `duration`, so the effective duration is
    the request rounded up to that granularity and reported as such. Decimal arithmetic on the
    number as written keeps 1.0 s from becoming one frame more through binary rounding.
    """
    rate, hop = facts["sample_rate"], facts["hop_length"]
    exact = Fraction(repr(duration)) if type(duration) is float else Fraction(duration)
    latent_frames = math.ceil(exact * rate / hop)
    samples = latent_frames * hop
    return {
        "latent_frames": latent_frames,
        "samples": samples,
        "effective_duration_seconds": samples / rate,
    }


def family_facts(root, family):
    """Configuration facts a supervisor needs from the loaded family (scale, video sizes)."""
    if family == "esrgan":
        from .upscale import read_config

        return {"scale": read_config(root)["scale"]}
    if family == "svd":
        return video_facts(root)
    if family == "stable-audio-open":
        return audio_facts(root)
    return {}


def safetensors_parameters(path):
    """Read tensor shapes/dtypes from a safetensors header without loading any weights."""
    totals = {}
    for name, tensor in safetensors_header(path).items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        totals[tensor["dtype"]] = totals.get(tensor["dtype"], 0) + count
    return totals


def weight_files(root, family, role):
    if FAMILIES[family]["format"] == "single-file-safetensors":
        return sorted(Path(root).glob("*.safetensors"))
    return sorted((Path(root) / role).glob("*.safetensors"))


def memory_estimate(root, precision, device, family="sdxl-base"):
    """Parameter bytes per role at the load precision; activations only where stated."""
    bytes_per_element = PRECISION_BYTES[precision]
    roles = {}
    for role in FAMILIES[family]["weight_roles"]:
        parameters = 0
        stored = 0
        for path in weight_files(root, family, role):
            for dtype, count in safetensors_parameters(path).items():
                parameters += count
                stored += count * SAFETENSORS_BYTES.get(dtype, 4)
        roles[role] = {
            "parameters": parameters,
            "stored_mb": stored // 1024**2,
            "loaded_mb": parameters * bytes_per_element // 1024**2,
        }
    total = sum(r["loaded_mb"] for r in roles.values())
    if family == "esrgan":
        from .upscale import activation_bytes, read_config

        side = LIMITS["tile_sizes"][-1] + 2 * LIMITS["tile_overlap"][1]
        activations = activation_bytes(read_config(root), side, bytes_per_element) // 1024**2
        return {
            "value_mb": total + activations,
            "source": "safetensors_headers",
            "basis": f"network parameters at the requested load precision plus the activations of one {side}-pixel tile (the largest advertised tile with maximum overlap)",
            "precision": precision,
            "device": device,
            "roles": roles,
            "activations_mb": activations,
            "uncertainty": "tile 0 (whole image) needs activations proportional to the full input and is not covered; excludes allocator reserve and framework overhead",
        }
    if family == "svd":
        return video_estimate(root, total, bytes_per_element, precision, device, roles)
    if family == "stable-audio-open":
        return audio_estimate(root, total, bytes_per_element, precision, device, roles)
    return {
        "value_mb": total,
        "source": "safetensors_headers",
        "basis": "parameter count at the requested load precision",
        "precision": precision,
        "device": device,
        "roles": roles,
        "uncertainty": "excludes activations, VAE float32 upcast, allocator reserve and framework overhead",
    }


def video_estimate(root, total, bytes_per_element, precision, device, roles):
    """svd: parameters plus the sampling latents of the most frames at the largest size.

    Per frame and per latent pixel (the VAE's spatial factor comes from its config) the sampling
    loop holds the guided UNet input (2 × `in_channels`), the conditioning image latents repeated
    for that frame (also guided, `in_channels`) and the denoised latent (`in_channels` / 2). That is
    what grows with `frames` in every step; UNet and VAE activations also grow with frames and
    `decode_chunk` and are not modeled here.
    """
    root = Path(root)
    try:
        unet = json.loads((root / "unet/config.json").read_text())
        vae = json.loads((root / "vae/config.json").read_text())
    except (OSError, ValueError):
        unet, vae = {}, {}
    channels = unet.get("in_channels", 8)
    factor = 2 ** (len(vae.get("block_out_channels", (128, 256, 512, 512))) - 1)
    facts = video_facts(root)
    width, height = max(facts["video_sizes"], key=lambda s: s[0] * s[1])
    pixels = (width // factor) * (height // factor)
    per_frame = (2 * channels + channels + channels // 2) * pixels * bytes_per_element
    frames = LIMITS["frames"][1]
    latents = per_frame * frames // 1024**2
    return {
        "value_mb": total + latents,
        "source": "safetensors_headers",
        "basis": f"component parameters at the requested load precision plus the sampling latents of {frames} frames at {width}×{height} ({per_frame} bytes per frame: the guided UNet input, the repeated image latents and the denoised latent)",
        "precision": precision,
        "device": device,
        "roles": roles,
        "latents_mb": latents,
        "per_frame_bytes": per_frame,
        "frames": {"limit": LIMITS["frames"], "scaling": "latent memory grows linearly with frames; per_frame_bytes × frames"},
        "uncertainty": "excludes UNet attention and VAE decoder activations (which grow with frames and decode_chunk), the image encoder's activations, allocator reserve and framework overhead",
    }


def audio_estimate(root, total, bytes_per_element, precision, device, roles):
    """stable-audio-open: parameters plus the sampling latents and the decoded waveform.

    The library samples the transformer's whole latent length whatever the requested duration
    and decodes all of it before the clip is cut, so the terms are fixed per bundle: per latent
    frame the loop holds the guided transformer input and its guided output (2 × `in_channels`
    each) and the latent (`in_channels`); the VAE then returns `sample_size × hop_length` samples
    per channel. Transformer attention, the T5 encoder's and the VAE decoder's activations are
    not modeled here.
    """
    root = Path(root)
    try:
        transformer = json.loads((root / "transformer/config.json").read_text())
    except (OSError, ValueError):
        transformer = {}
    facts = audio_facts(root)
    channels = transformer.get("in_channels", 64)
    length = transformer.get("sample_size", 1024)
    latents = 5 * channels * length * bytes_per_element
    waveform = length * facts["hop_length"] * facts["channels"] * bytes_per_element
    working = (latents + waveform) // 1024**2
    return {
        "value_mb": total + working,
        "source": "safetensors_headers",
        "basis": f"component parameters at the requested load precision plus the sampling latents of the whole latent length ({length} frames × {channels} channels: the guided transformer input and output and the latent, {latents} bytes) and the decoded waveform before it is cut ({length * facts['hop_length']} samples × {facts['channels']} channels, {waveform} bytes)",
        "precision": precision,
        "device": device,
        "roles": roles,
        "latents_mb": latents // 1024**2,
        "waveform_mb": waveform // 1024**2,
        "duration": {"limit": facts["duration_range"], "scaling": "constant: the whole latent length is sampled and decoded for every duration"},
        "uncertainty": "excludes transformer attention, T5 encoder and VAE decoder activations, the 16-bit PCM copy written to disk, allocator reserve and framework overhead",
    }


def png_dimensions(data):
    """Validate a PNG signature/IHDR and return (width, height) without decoding pixels."""
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise WorkerError(400, "invalid_request", "image")
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    if depth != 8 or color not in (0, 2, 4, 6) or data[-8:-4] != b"IEND":
        raise WorkerError(400, "unsupported", "image")
    for value in (width, height):
        if not LIMITS["width"][0] <= value <= LIMITS["width"][1] or value % 8:
            raise WorkerError(400, "invalid_request", "image")
    return width, height


def effective_steps(parameters, task):
    if task == "upscale":
        return None
    if task in ("image-to-video", "text-to-audio"):
        return parameters["steps"]
    if task != "image-to-image":
        return parameters["steps"]
    return min(int(parameters["steps"] * parameters["strength"]), parameters["steps"])


def output_ids(task, parameters):
    """Every output ID an execution must finalize: `image, image_2 … image_N` for `count: N`."""
    count = parameters.get("count", 1) if task == "text-to-image" else 1
    return [
        name if k == 1 else f"{name}_{k}" for name in OUTPUTS[task] for k in range(1, count + 1)
    ]


def effective_settings(parameters, task, version):
    if task == "upscale":
        return {**TASK_DEFAULTS["upscale"], **parameters}
    if task == "image-to-video":
        # `frames` joins from the loaded bundle's facts when the request leaves it out.
        return {
            **TASK_DEFAULTS["image-to-video"],
            **parameters,
            **VIDEO_GUIDANCE,
            "effective_steps": parameters["steps"],
        }
    if task == "text-to-audio":
        # sample_rate, channels and the effective duration join from the loaded bundle's facts.
        return {
            **TASK_DEFAULTS["text-to-audio"],
            **parameters,
            "effective_steps": parameters["steps"],
        }
    settings = {**parameters, "effective_steps": effective_steps(parameters, task)}
    if task == "text-to-image" and minor(version) >= 2:
        settings = {"count": 1, **settings}
    return settings


def validate_upscale(body):
    inputs, params = body["inputs"], body["parameters"]
    if isinstance(inputs, dict) and ("prompt" in inputs or "negative_prompt" in inputs):
        # Upscaling is not conditioned on text; a prompt would be silently ignored otherwise.
        raise WorkerError(400, "invalid_request", "prompt")
    strict(inputs, ("image",))
    if not isinstance(inputs["image"], str) or not re.fullmatch(
        r"input_[a-f0-9]{32}", inputs["image"]
    ):
        raise WorkerError(400, "invalid_request", "image")
    strict(params, ("width", "height"), ("tile", "tile_overlap"))
    for field in ("width", "height"):
        value = params[field]
        low, high = LIMITS[field]
        if type(value) is not int or not low <= value <= high or value % LIMITS["size_multiple"]:
            raise WorkerError(400, "invalid_request", field)
    tile = params.get("tile", TASK_DEFAULTS["upscale"]["tile"])
    if type(tile) is not int or (tile and tile not in LIMITS["tile_sizes"]):
        raise WorkerError(400, "invalid_request", "tile")
    overlap = params.get("tile_overlap", TASK_DEFAULTS["upscale"]["tile_overlap"])
    low, high = LIMITS["tile_overlap"]
    if type(overlap) is not int or not low <= overlap <= high:
        raise WorkerError(400, "invalid_request", "tile_overlap")
    return "upscale"


def validate_video(body):
    inputs, params = body["inputs"], body["parameters"]
    if isinstance(inputs, dict) and ("prompt" in inputs or "negative_prompt" in inputs):
        # The svd family is conditioned on the image alone; a prompt would be silently ignored.
        raise WorkerError(400, "invalid_request", "prompt")
    strict(inputs, ("image",))
    if not isinstance(inputs["image"], str) or not re.fullmatch(
        r"input_[a-f0-9]{32}", inputs["image"]
    ):
        raise WorkerError(400, "invalid_request", "image")
    strict(
        params,
        ("width", "height", "seed", "steps"),
        ("frames", "fps", "motion", "noise_augmentation", "decode_chunk"),
    )
    for field in ("width", "height"):
        value = params[field]
        low, high = LIMITS[field]
        if type(value) is not int or not low <= value <= high or value % LIMITS["size_multiple"]:
            raise WorkerError(400, "invalid_request", field)
    for field, limit in (
        ("seed", [0, 4294967295]),
        ("steps", LIMITS["video_steps"]),
        ("frames", LIMITS["frames"]),
        ("fps", LIMITS["fps"]),
        ("motion", LIMITS["motion"]),
        ("decode_chunk", LIMITS["decode_chunk"]),
    ):
        if field in params:
            value = params[field]
            if type(value) is not int or not limit[0] <= value <= limit[1]:
                raise WorkerError(400, "invalid_request", field)
    if "noise_augmentation" in params:
        value = params["noise_augmentation"]
        low, high = LIMITS["noise_augmentation"]
        if type(value) not in (int, float) or not low <= value <= high:
            raise WorkerError(400, "invalid_request", "noise_augmentation")
    return "image-to-video"


def validate_audio(body):
    inputs, params = body["inputs"], body["parameters"]
    if isinstance(inputs, dict) and "image" in inputs:
        # The family is conditioned on text and timing alone; an image would be silently ignored.
        raise WorkerError(400, "invalid_request", "image")
    strict(inputs, ("prompt",), ("negative_prompt",))
    for key, value in inputs.items():
        if (
            not isinstance(value, str)
            or len(value) > 10000
            or (key == "prompt" and not value.strip())
        ):
            raise WorkerError(400, "invalid_request", key)
    if isinstance(params, dict):
        for field in ("width", "height", "sampler", "scheduler", "count"):
            if field in params:
                # Image parameters and scheduler choice do not exist for this task.
                raise WorkerError(400, "invalid_request", field)
    strict(params, ("seed", "steps", "duration_seconds"), ("guidance",))
    for field, limit in (("seed", [0, 4294967295]), ("steps", LIMITS["audio_steps"])):
        value = params[field]
        if type(value) is not int or not limit[0] <= value <= limit[1]:
            raise WorkerError(400, "invalid_request", field)
    guidance = params.get("guidance", TASK_DEFAULTS["text-to-audio"]["guidance"])
    if not is_number(guidance) or not LIMITS["guidance"][0] <= guidance <= LIMITS["guidance"][1]:
        raise WorkerError(400, "invalid_request", "guidance")
    duration = params["duration_seconds"]
    # The loaded bundle's range is checked once it is known (Worker.loaded_checks).
    if not is_number(duration) or not 0 < duration <= MAX_AUDIO_SECONDS:
        raise WorkerError(400, "invalid_request", "duration_seconds")
    return "text-to-audio"


def validate_parameters(body):
    strict(
        body,
        ("protocol", "attempt_id", "bundle_digest", "task", "inputs", "parameters"),
    )
    version = protocol(body)
    level = minor(version)
    if not isinstance(body["bundle_digest"], str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", body["bundle_digest"]
    ):
        raise WorkerError(400, "invalid_request", "bundle_digest")
    task = body["task"]
    if not isinstance(task, str) or task not in TASKS or TASK_SINCE[task] > level:
        raise WorkerError(400, "unsupported", "task")
    if not isinstance(body["attempt_id"], str) or not re.fullmatch(
        r"attempt_[A-Za-z0-9_-]{1,80}", body["attempt_id"]
    ):
        raise WorkerError(400, "invalid_request", "attempt_id")
    if task == "upscale":
        return validate_upscale(body)
    if task == "image-to-video":
        return validate_video(body)
    if task == "text-to-audio":
        return validate_audio(body)
    image_required = ("image",) if task == "image-to-image" else ()
    strict(body["inputs"], ("prompt",) + image_required, ("negative_prompt",))
    for key, value in body["inputs"].items():
        if key == "image":
            if not isinstance(value, str) or not re.fullmatch(r"input_[a-f0-9]{32}", value):
                raise WorkerError(400, "invalid_request", key)
            continue
        if (
            not isinstance(value, str)
            or len(value) > 10000
            or (key == "prompt" and not value.strip())
        ):
            raise WorkerError(400, "invalid_request", key)
    params = body["parameters"]
    if isinstance(params, dict) and "count" in params:
        # count is a 1.2 text-to-image parameter; older minors and other tasks never see it.
        if level < 2 or task != "text-to-image":
            raise WorkerError(400, "unsupported", "count")
        low, high = LIMITS["count"]
        if type(params["count"]) is not int or not low <= params["count"] <= high:
            raise WorkerError(400, "invalid_request", "count")
    strength_required = ("strength",) if task == "image-to-image" else ()
    strict(
        params,
        ("width", "height", "steps", "guidance", "seed", "sampler", "scheduler")
        + strength_required,
        ("count",) if task == "text-to-image" and level >= 2 else (),
    )
    for field, low, high, multiple in (
        ("width", 64, 2048, 8),
        ("height", 64, 2048, 8),
        ("steps", 1, 150, 1),
        ("seed", 0, 4294967295, 1),
    ):
        value = params[field]
        if type(value) is not int or not low <= value <= high or value % multiple:
            raise WorkerError(400, "invalid_request", field)
    if params.get("count", 1) > 1 and params["seed"] + params["count"] - 1 > 4294967295:
        # Output k uses seed + k − 1; every seed in the batch must be a valid uint32.
        raise WorkerError(400, "invalid_request", "seed")
    if (
        type(params["guidance"]) not in (int, float)
        or not 0 <= params["guidance"] <= 30
    ):
        raise WorkerError(400, "invalid_request", "guidance")
    if task == "image-to-image":
        strength = params["strength"]
        if type(strength) not in (int, float) or not 0 < strength <= 1:
            raise WorkerError(400, "invalid_request", "strength")
        if effective_steps(params, task) < 1:
            # strength × steps would denoise for zero steps; that is not generation.
            raise WorkerError(400, "invalid_request", "strength")
    if params["sampler"] != "euler" or params["scheduler"] != "euler":
        raise WorkerError(400, "unsupported", "scheduler")
    return task


def adapter_path(adapter, catalog, verify=True):
    """Resolve a registered LoRA adapter to its approved directory and single weight file."""
    strict(
        adapter,
        ("schema_version", "adapter_id", "artifact_handle", "files", "digest", "source"),
    )
    if adapter["schema_version"] != 1:
        raise WorkerError(400, "unsupported", "adapter")
    if not isinstance(adapter["adapter_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", adapter["adapter_id"]
    ):
        raise WorkerError(400, "invalid_request", "adapter_id")
    if digest({k: v for k, v in adapter.items() if k != "digest"}) != adapter["digest"]:
        raise WorkerError(400, "artifact_changed", "adapter")
    strict(adapter["source"], (), ("repository", "revision", "license"))
    if any(not isinstance(v, str) for v in adapter["source"].values()):
        raise WorkerError(400, "invalid_request", "source")
    handle = adapter["artifact_handle"]
    if not isinstance(handle, str) or handle not in catalog:
        raise WorkerError(400, "missing_component", "adapter")
    root = Path(catalog[handle]).resolve()
    files = adapter["files"]
    if not isinstance(files, dict) or not files or not root.is_dir():
        raise WorkerError(400, "missing_component", "adapter")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WorkerError(400, "unsupported", "symlink")
        if path.is_file():
            if path.suffix in {".py", ".pkl", ".pickle", ".bin", ".pt", ".pth"}:
                raise WorkerError(400, "unsupported", "tensor_format")
            actual[str(path.relative_to(root))] = path
    if set(actual) != set(files):
        raise WorkerError(400, "artifact_changed", "adapter")
    weights = [k for k in files if k.endswith(".safetensors")]
    if len(weights) != 1 or "/" in weights[0]:
        raise WorkerError(400, "unsupported", "adapter")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise WorkerError(400, "invalid_request", "digest")
        if verify:
            with actual[relative].open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise WorkerError(400, "artifact_changed", "adapter")
    return root, actual[weights[0]]


def lora_layout(path):
    """Classify a LoRA safetensors file from its header; only the PEFT layout is supported."""
    with Path(path).open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if not 0 < length <= 64 * 1024 * 1024:
            raise WorkerError(400, "unsupported", "adapter")
        header = json.loads(stream.read(length))
    modules = {}
    parameters = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        match = re.fullmatch(
            r"(unet|text_encoder|text_encoder_2)\.(.+)\.(lora_A\.weight|lora_B\.weight|alpha)",
            name,
        )
        if not match:
            # Kohya/"lora_unet_" and legacy processor layouts need conversion first.
            raise WorkerError(400, "unsupported", "adapter")
        target, module, part = match.groups()
        entry = modules.setdefault((target, module), {})
        entry[part] = tensor["shape"]
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        parameters += count
    if not modules:
        raise WorkerError(400, "unsupported", "adapter")
    ranks = set()
    for (target, module), parts in modules.items():
        if "lora_A.weight" not in parts or "lora_B.weight" not in parts:
            raise WorkerError(400, "unsupported", "adapter")
        if parts["lora_A.weight"][0] != parts["lora_B.weight"][1]:
            raise WorkerError(400, "unsupported", "adapter")
        ranks.add(parts["lora_A.weight"][0])
    return {
        "format": ADAPTER_FORMAT,
        "targets": sorted({t for t, _ in modules}),
        "modules": len(modules),
        "ranks": sorted(ranks),
        "parameters": parameters,
    }


def validate_adapters(adapters, catalog, version, verify=True):
    """Validate the load body's adapters list; returns resolved (record, root, file, scale) tuples."""
    if not isinstance(adapters, list) or len(adapters) > MAX_ADAPTERS:
        raise WorkerError(400, "invalid_request", "adapters")
    if adapters and version == "media-worker/1.0":
        raise WorkerError(400, "invalid_request", "adapters")
    resolved = []
    seen = set()
    for entry in adapters:
        strict(entry, ("adapter", "scale"))
        scale = entry["scale"]
        low, high = LIMITS["adapter_scale"]
        if type(scale) not in (int, float) or not low <= scale <= high:
            raise WorkerError(400, "invalid_request", "scale")
        root, weights = adapter_path(entry["adapter"], catalog, verify)
        if entry["adapter"]["adapter_id"] in seen:
            raise WorkerError(400, "invalid_request", "adapter_id")
        seen.add(entry["adapter"]["adapter_id"])
        resolved.append((entry["adapter"], root, weights, scale, lora_layout(weights)))
    return resolved
