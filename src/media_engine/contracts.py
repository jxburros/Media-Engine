"""Validation shared by every worker endpoint; never accepts executable model code."""

import hashlib
import json
import re
import struct
from pathlib import Path

PROTOCOL = "media-worker/1.2"
SUPPORTED_PROTOCOLS = ("media-worker/1.0", "media-worker/1.1", "media-worker/1.2")
WORKER_VERSION = "0.4.0"
TASKS = ("text-to-image", "image-to-image", "upscale")
# The minor that introduced each task; a lower-minor request naming it is `unsupported`.
TASK_SINCE = {"text-to-image": 0, "image-to-image": 1, "upscale": 2}
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
}
# 1.0/1.1 names, kept for importers; they describe the sdxl-base family.
ROLES = FAMILIES["sdxl-base"]["roles"]
WEIGHT_ROLES = FAMILIES["sdxl-base"]["weight_roles"]
APPROVED_CLASSES = FAMILIES["sdxl-base"]["approved_classes"]
FORMATS = ("diffusers-safetensors", "single-file-safetensors")
OUTPUTS = {"text-to-image": ["image"], "image-to-image": ["image"], "upscale": ["image"]}
DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 6,
    "sampler": "euler",
    "scheduler": "euler",
}
TASK_DEFAULTS = {"image-to-image": {"strength": 0.75}, "upscale": {"tile": 0, "tile_overlap": 16}}
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
}
AUTO_OPTIONS = ["precision"]
FALLBACKS = ["cpu", "lower_precision"]
PRECISION_BYTES = {"float16": 2, "float32": 4}
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
        index.get("_class_name") != "StableDiffusionXLPipeline"
        or scheduler.get("_class_name") != "EulerDiscreteScheduler"
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


def family_facts(root, family):
    """Configuration facts a supervisor needs from the loaded family (the upscale factor)."""
    if family == "esrgan":
        from .upscale import read_config

        return {"scale": read_config(root)["scale"]}
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
    return {
        "value_mb": total,
        "source": "safetensors_headers",
        "basis": "parameter count at the requested load precision",
        "precision": precision,
        "device": device,
        "roles": roles,
        "uncertainty": "excludes activations, VAE float32 upcast, allocator reserve and framework overhead",
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
