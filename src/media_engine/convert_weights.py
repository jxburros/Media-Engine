"""Offline migration of RRDBNet .pth and Kohya SDXL LoRA into safe Engine artifacts.

Pickle-bearing input is opened only by an isolated, short-lived weights_only subprocess.
The serving workers never call this module. Architecture and key matching fail closed.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .convert import ConversionError, file_map, reference_bundle, reference_models, safetensors_header, versions
from .contracts import digest

_EXTRACT = r'''
import sys
import torch
from safetensors.torch import save_file
value = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if isinstance(value, dict) and ("params_ema" in value or "params" in value):
    if set(value) - {"params", "params_ema"}:
        raise ValueError("unknown checkpoint fields")
    value = value.get("params_ema", value.get("params"))
if not isinstance(value, dict) or not value or not all(
    isinstance(k, str) and isinstance(v, torch.Tensor) and v.layout == torch.strided
    and v.is_floating_point() for k, v in value.items()
):
    raise ValueError("expected floating-point tensor state dict")
save_file({k: v.detach().contiguous().clone() for k, v in value.items()}, sys.argv[2])
'''


def _source(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ConversionError("invalid_source")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(output, source_hash, reference_files, tensors, filename, mapped, config=None):
    from safetensors.torch import save_file

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ConversionError("output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        save_file(tensors, str(partial / filename))
        if config is not None:
            (partial / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
        record = {
            "schema_version": 1,
            "source_sha256": source_hash,
            "reference_bundle_digest": digest(reference_files),
            "converter": versions(),
            "precision": "preserved",
            "mapped_keys": mapped,
            "unmapped_keys": [],
            "ignored_keys": [],
        }
        (partial / "conversion.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        # Refuse replacing even an empty pre-existing output directory.
        output.mkdir(mode=0o755)
        try:
            for item in partial.iterdir():
                item.rename(output / item.name)
        except BaseException:
            shutil.rmtree(output)
            raise
        partial.rmdir()
        return record
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def esrgan_pth(source, config_path, output):
    """Convert a published Real-ESRGAN params_ema/params or bare RRDBNet state dict."""
    from safetensors.torch import load_file
    from .upscale import architecture, validate_config

    source = Path(source).absolute()
    source_hash = _source(source)
    try:
        config_bytes = Path(config_path).read_bytes()
        config = validate_config(json.loads(config_bytes))
    except (OSError, ValueError):
        config = None
    if config is None:
        raise ConversionError("unsupported_configuration")
    if Path(output).exists() or Path(output).is_symlink():
        raise ConversionError("output_exists")
    with tempfile.TemporaryDirectory(prefix="media-engine-convert-") as temp:
        extracted = Path(temp) / "weights.safetensors"
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _EXTRACT, str(source), str(extracted)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=180, check=False,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
        except subprocess.TimeoutExpired:
            raise ConversionError("conversion_timeout") from None
        if result.returncode != 0:
            raise ConversionError("unsafe_or_invalid_checkpoint")
        tensors = load_file(str(extracted), device="cpu")
    expected = architecture(config)
    unknown = sorted(tensors.keys() - expected.keys())
    if unknown:
        raise ConversionError("unmapped_keys", unknown)
    bad = sorted(expected.keys() - tensors.keys()) + sorted(
        k for k in expected.keys() & tensors.keys() if list(tensors[k].shape) != expected[k]
    )
    if bad:
        raise ConversionError("incompatible_keys", bad)
    # A changing source is never attributed to the digest measured before extraction.
    if _source(source) != source_hash:
        raise ConversionError("source_changed")
    return _publish(output, source_hash, {"config.json": hashlib.sha256(config_bytes).hexdigest()},
                    tensors, "network.safetensors", {"network": len(tensors)}, config)


def lora_kohya(source, catalog, handle, output):
    """Map Kohya linear SDXL LoRA keys using exact targets from the local reference configs."""
    import torch
    from safetensors.torch import load_file

    source_hash = _source(source)
    safetensors_header(source)
    root, reference_files = reference_bundle(catalog, handle)
    _, models = reference_models(root)
    lookup = {}
    for role, prefix in (("unet", "lora_unet_"), ("text_encoder", "lora_te1_"), ("text_encoder_2", "lora_te2_")):
        for name, layer in models[role].named_modules():
            if isinstance(layer, torch.nn.Linear):
                encoded = prefix + name.replace(".", "_")
                if encoded in lookup:
                    raise ConversionError("ambiguous_reference", [encoded])
                lookup[encoded] = (role + "." + name, layer.in_features, layer.out_features)
    state = load_file(str(source), device="cpu")
    groups, unknown = {}, []
    suffixes = {".lora_down.weight": "lora_A.weight", ".lora_up.weight": "lora_B.weight", ".alpha": "alpha"}
    for key, tensor in state.items():
        suffix = next((s for s in suffixes if key.endswith(s)), None)
        target = key[:-len(suffix)] if suffix else None
        if target not in lookup:
            unknown.append(key)
            continue
        groups.setdefault(target, {})[suffixes[suffix]] = tensor
    if unknown:
        raise ConversionError("unmapped_keys", sorted(unknown))
    if not groups:
        raise ConversionError("incompatible_keys")
    mapped, tensors, bad = {}, {}, []
    for encoded, parts in groups.items():
        name, inputs, outputs = lookup[encoded]
        a, b, alpha = parts.get("lora_A.weight"), parts.get("lora_B.weight"), parts.get("alpha")
        if (a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1
                or a.shape[0] != b.shape[1] or a.shape[1] != inputs or b.shape[0] != outputs
                or any(not v.is_floating_point() for v in (a, b))
                or (alpha is not None and (alpha.numel() != 1 or not math.isfinite(float(alpha))))):
            bad.append(encoded)
            continue
        for part, value in parts.items():
            tensors[name + "." + part] = value.contiguous().clone()
        role = name.split(".")[0]
        mapped[role] = mapped.get(role, 0) + len(parts)
    if bad:
        raise ConversionError("incompatible_keys", sorted(bad))
    if _source(source) != source_hash:
        raise ConversionError("source_changed")
    return _publish(output, source_hash, reference_files, tensors, "adapter.safetensors", mapped)
