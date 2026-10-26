"""Validation shared by every worker endpoint; never accepts executable model code."""

import hashlib
import json
import re
from pathlib import Path

PROTOCOL = "media-worker/1.0"
ROLES = (
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
)
DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 6,
    "sampler": "euler",
    "scheduler": "euler",
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
    if body.get("protocol") != PROTOCOL:
        raise WorkerError(400, "protocol_mismatch", "protocol")


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
        expected_format = "safetensors" if role in ROLES[:4] else "json"
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
    approved = {
        "unet": ("diffusers", "UNet2DConditionModel"),
        "vae": ("diffusers", "AutoencoderKL"),
        "text_encoder": ("transformers", "CLIPTextModel"),
        "text_encoder_2": ("transformers", "CLIPTextModelWithProjection"),
        "tokenizer": ("transformers", "CLIPTokenizer"),
        "tokenizer_2": ("transformers", "CLIPTokenizer"),
        "scheduler": ("diffusers", "EulerDiscreteScheduler"),
    }
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


def validate_parameters(body):
    strict(
        body,
        ("protocol", "attempt_id", "bundle_digest", "task", "inputs", "parameters"),
    )
    protocol(body)
    if not isinstance(body["bundle_digest"], str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", body["bundle_digest"]
    ):
        raise WorkerError(400, "invalid_request", "bundle_digest")
    if body["task"] != "text-to-image":
        raise WorkerError(400, "unsupported", "task")
    if not isinstance(body["attempt_id"], str) or not re.fullmatch(
        r"attempt_[A-Za-z0-9_-]{1,80}", body["attempt_id"]
    ):
        raise WorkerError(400, "invalid_request", "attempt_id")
    strict(body["inputs"], ("prompt",), ("negative_prompt",))
    for key, value in body["inputs"].items():
        if (
            not isinstance(value, str)
            or len(value) > 10000
            or (key == "prompt" and not value.strip())
        ):
            raise WorkerError(400, "invalid_request", key)
    params = body["parameters"]
    strict(
        params, ("width", "height", "steps", "guidance", "seed", "sampler", "scheduler")
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
    if params["sampler"] != "euler" or params["scheduler"] != "euler":
        raise WorkerError(400, "unsupported", "scheduler")
