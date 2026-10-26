"""Validation shared by every worker endpoint; never accepts executable model code."""

import hashlib
import json
import re
import struct
from pathlib import Path

PROTOCOL = "media-worker/1.1"
SUPPORTED_PROTOCOLS = ("media-worker/1.0", "media-worker/1.1")
WORKER_VERSION = "0.2.0"
TASKS = ("text-to-image", "image-to-image")
ROLES = (
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
)
WEIGHT_ROLES = ROLES[:4]
DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 6,
    "sampler": "euler",
    "scheduler": "euler",
}
TASK_DEFAULTS = {"image-to-image": {"strength": 0.75}}
LIMITS = {
    "width": [64, 2048],
    "height": [64, 2048],
    "steps": [1, 150],
    "guidance": [0, 30],
    "strength": [0, 1],
    "size_multiple": 8,
}
APPROVED_CLASSES = {
    "unet": ("diffusers", "UNet2DConditionModel"),
    "vae": ("diffusers", "AutoencoderKL"),
    "text_encoder": ("transformers", "CLIPTextModel"),
    "text_encoder_2": ("transformers", "CLIPTextModelWithProjection"),
    "tokenizer": ("transformers", "CLIPTokenizer"),
    "tokenizer_2": ("transformers", "CLIPTokenizer"),
    "scheduler": ("diffusers", "EulerDiscreteScheduler"),
}
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


def bundle_path(bundle, catalog, verify=True):
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
    )
    if (
        bundle["schema_version"] != 1
        or bundle["family"] != "sdxl-base"
        or bundle["pipeline"] != "text-to-image"
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
    strict(bundle["source"], (), ("repository", "revision", "license"))
    if any(not isinstance(v, str) for v in bundle["source"].values()):
        raise WorkerError(400, "invalid_request", "source")
    configuration = bundle["configuration"]
    strict(configuration, ("artifact_handle", "files"))
    handle = configuration["artifact_handle"]
    if not isinstance(handle, str) or handle not in catalog:
        raise WorkerError(400, "missing_component", "artifact_handle")
    root = Path(catalog[handle]).resolve()
    files = configuration["files"]
    if not isinstance(files, dict) or not files:
        raise WorkerError(400, "missing_component", "configuration")
    if not isinstance(bundle["components"], list) or len(bundle["components"]) != len(
        ROLES
    ):
        raise WorkerError(400, "missing_component", "components")
    roles = set()
    for component in bundle["components"]:
        strict(component, ("role", "artifact_handle", "digest", "tensor_format"))
        role = component["role"]
        if role not in ROLES or role in roles or component["artifact_handle"] != handle:
            raise WorkerError(400, "unsupported", "components")
        roles.add(role)
        entries = {k: v for k, v in files.items() if k.startswith(role + "/")}
        if not entries or digest(entries) != component["digest"]:
            raise WorkerError(400, "artifact_changed", role)
        expected_format = "safetensors" if role in WEIGHT_ROLES else "json"
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
                    raise WorkerError(400, "artifact_changed", relative.split("/")[0])
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
    for key, value in index.items():
        if key.startswith("_") or key in APPROVED_CLASSES:
            continue
        if key == "force_zeros_for_empty_prompt" and type(value) is bool:
            continue
        if key in {"feature_extractor", "image_encoder"} and value == [None, None]:
            continue
        raise WorkerError(400, "unsupported", "pipeline")
    for role, value in APPROVED_CLASSES.items():
        if index.get(role) != list(value):
            raise WorkerError(400, "unsupported", role)
    return root


def safetensors_parameters(path):
    """Read tensor shapes/dtypes from a safetensors header without loading any weights."""
    with Path(path).open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if not 0 < length <= 64 * 1024 * 1024:
            raise WorkerError(400, "unsupported", "tensor_format")
        header = json.loads(stream.read(length))
    totals = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        totals[tensor["dtype"]] = totals.get(tensor["dtype"], 0) + count
    return totals


def memory_estimate(root, precision, device):
    """Parameter bytes per role at the load precision. Activations are not estimated."""
    bytes_per_element = {"float16": 2, "float32": 4}[precision]
    roles = {}
    for role in WEIGHT_ROLES:
        parameters = 0
        stored = 0
        for path in sorted((Path(root) / role).glob("*.safetensors")):
            for dtype, count in safetensors_parameters(path).items():
                parameters += count
                stored += count * SAFETENSORS_BYTES.get(dtype, 4)
        roles[role] = {
            "parameters": parameters,
            "stored_mb": stored // 1024**2,
            "loaded_mb": parameters * bytes_per_element // 1024**2,
        }
    total = sum(r["loaded_mb"] for r in roles.values())
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
    if task != "image-to-image":
        return parameters["steps"]
    return min(int(parameters["steps"] * parameters["strength"]), parameters["steps"])


def validate_parameters(body):
    strict(
        body,
        ("protocol", "attempt_id", "bundle_digest", "task", "inputs", "parameters"),
    )
    version = protocol(body)
    if not isinstance(body["bundle_digest"], str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", body["bundle_digest"]
    ):
        raise WorkerError(400, "invalid_request", "bundle_digest")
    task = body["task"]
    if task not in TASKS or (task != "text-to-image" and version == "media-worker/1.0"):
        raise WorkerError(400, "unsupported", "task")
    if not isinstance(body["attempt_id"], str) or not re.fullmatch(
        r"attempt_[A-Za-z0-9_-]{1,80}", body["attempt_id"]
    ):
        raise WorkerError(400, "invalid_request", "attempt_id")
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
    strength_required = ("strength",) if task == "image-to-image" else ()
    strict(
        params,
        ("width", "height", "steps", "guidance", "seed", "sampler", "scheduler")
        + strength_required,
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
