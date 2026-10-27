"""`media-engine convert`: an explicit, offline operator action, never run by a load request.

It reads one single-file SDXL safetensors checkpoint, takes architecture, scheduler and tokenizer
configuration from an approved reference bundle named by a catalog handle, maps the checkpoint's
keys onto the Diffusers component layout with the library's single-file conversion helpers, and
writes a complete Diffusers-format bundle of safetensors components plus `conversion.json`.
Nothing is downloaded; any key that cannot be mapped fails the conversion by name and the
partial output is deleted.
"""

import hashlib
import importlib.metadata
import json
import shutil
import struct
import sys
from pathlib import Path

from .contracts import WORKER_VERSION, digest

PREFIXES = {
    "unet": "model.diffusion_model.",
    "vae": "first_stage_model.",
    "text_encoder": "conditioner.embedders.0.transformer.",
    "text_encoder_2": "conditioner.embedders.1.model.",
}
WEIGHT_NAMES = {
    "unet": "diffusion_pytorch_model.safetensors",
    "vae": "diffusion_pytorch_model.safetensors",
    "text_encoder": "model.safetensors",
    "text_encoder_2": "model.safetensors",
}
COPIED = ("tokenizer", "tokenizer_2", "scheduler")
# Derived buffers some checkpoints store; the model classes recompute them.
DERIVED = ("position_ids",)
# The OpenCLIP helper rebuilds these by splitting or transposing, so their storage is new.
REBUILT = (".in_proj_weight", ".in_proj_bias", "text_projection")
UNSAFE_SUFFIXES = {".py", ".pkl", ".pickle", ".bin", ".pt", ".pth", ".ckpt"}


class ConversionError(Exception):
    def __init__(self, code, keys=None):
        self.code, self.keys = code, keys


def add_arguments(parser):
    parser.add_argument("--input", required=True, help="single-file SDXL .safetensors checkpoint")
    parser.add_argument("--reference-handle", required=True, help="catalog handle of an approved sdxl-base bundle")
    parser.add_argument("--catalog", required=True, help="operator catalog mapping handles to absolute directories")
    parser.add_argument("--output", required=True, help="new directory for the converted bundle")
    parser.add_argument("--precision", choices=["float16", "float32"], help="cast weights; default preserves them")


def safetensors_header(path):
    """Refuse anything that is not a safetensors container by its signature, then read the header."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(9)
        if len(prefix) < 9 or prefix[:2] == b"PK" or prefix[:1] == b"\x80":
            raise ConversionError("not_safetensors")
        (length,) = struct.unpack("<Q", prefix[:8])
        if not 2 <= length <= 64 * 1024 * 1024 or prefix[8:9] != b"{":
            raise ConversionError("not_safetensors")
        try:
            header = json.loads(prefix[8:] + stream.read(length - 1))
        except ValueError:
            raise ConversionError("not_safetensors") from None
    if not isinstance(header, dict):
        raise ConversionError("not_safetensors")
    return header


def file_map(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.suffix in UNSAFE_SUFFIXES:
            raise ConversionError("unsupported_reference")
        if path.is_file():
            with path.open("rb") as stream:
                files[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return files


def reference_bundle(catalog, handle):
    if handle not in catalog or not Path(catalog[handle]).is_absolute():
        raise ConversionError("unknown_reference")
    root = Path(catalog[handle]).resolve()
    try:
        index = json.loads((root / "model_index.json").read_text())
    except (OSError, ValueError):
        raise ConversionError("unsupported_reference") from None
    if index.get("_class_name") != "StableDiffusionXLPipeline":
        raise ConversionError("unsupported_reference")
    for role in (*PREFIXES, *COPIED):
        if not (root / role).is_dir():
            raise ConversionError("unsupported_reference", [role])
    return root, file_map(root)


def versions():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "media_engine": WORKER_VERSION,
        "diffusers": version("diffusers"),
        "torch": version("torch"),
        "safetensors": version("safetensors"),
    }


def reference_models(root):
    """The four weight-role model classes built on the meta device from the reference configs."""
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    common = {"local_files_only": True}
    configs = {
        "unet": UNet2DConditionModel.load_config(str(root), subfolder="unet", **common),
        "vae": AutoencoderKL.load_config(str(root), subfolder="vae", **common),
        "text_encoder": CLIPTextConfig.from_pretrained(str(root), subfolder="text_encoder", **common),
        "text_encoder_2": CLIPTextConfig.from_pretrained(str(root), subfolder="text_encoder_2", **common),
    }
    with torch.device("meta"):
        models = {
            "unet": UNet2DConditionModel.from_config(configs["unet"]),
            "vae": AutoencoderKL.from_config(configs["vae"]),
            "text_encoder": CLIPTextModel(configs["text_encoder"]),
            "text_encoder_2": CLIPTextModelWithProjection(configs["text_encoder_2"]),
        }
    return configs, models


def map_role(role, source, configs, models):
    from diffusers.loaders import single_file_utils as helpers

    if role == "unet":
        return helpers.convert_ldm_unet_checkpoint(source, configs["unet"])
    if role == "vae":
        return helpers.convert_ldm_vae_checkpoint(source, configs["vae"])
    if role == "text_encoder":
        return helpers.convert_ldm_clip_checkpoint(source)
    return helpers.convert_open_clip_checkpoint(models[role], source, prefix=PREFIXES[role])


def convert_state(state, root, precision=None):
    """Map a single-file state dict to per-role Diffusers state dicts; fail on any unmapped key."""
    import torch

    configs, models = reference_models(root)
    unmapped = [k for k in state if not any(k.startswith(p) for p in PREFIXES.values())]
    ignored, mapped, converted, problems = [], {}, {}, []
    for role, prefix in PREFIXES.items():
        source = {k: v for k, v in state.items() if k.startswith(prefix)}
        expected = {k: list(v.shape) for k, v in models[role].state_dict().items()}
        result = map_role(role, source, configs, models)
        for key in [k for k in result if k not in expected and k.endswith(DERIVED)]:
            del result[key]
        problems += [f"{role}:{k}" for k in sorted(expected.keys() - result.keys())]
        problems += [f"{role}:{k}" for k in sorted(result.keys() - expected.keys())]
        problems += [
            f"{role}:{k}" for k in sorted(expected.keys() & result.keys()) if list(result[k].shape) != expected[k]
        ]
        used = {result[k].untyped_storage().data_ptr() for k in result if k in expected}
        count = 0
        for key, tensor in source.items():
            if tensor.untyped_storage().data_ptr() in used or (
                role == "text_encoder_2" and key.endswith(REBUILT)
            ):
                count += 1
            elif key.endswith(DERIVED):
                ignored.append(key)
            else:
                unmapped.append(key)
        mapped[role] = count
        seen, tensors = set(), {}
        for key in sorted(k for k in result if k in expected):
            tensor = result[key]
            if precision and tensor.is_floating_point():
                tensor = tensor.to(getattr(torch, precision))
            pointer = tensor.untyped_storage().data_ptr()
            # safetensors refuses shared storage; views of one source tensor are copied apart.
            tensors[key] = tensor.contiguous().clone() if pointer in seen else tensor.contiguous()
            seen.add(pointer)
        converted[role] = tensors
    if unmapped:
        raise ConversionError("unmapped_keys", sorted(unmapped))
    if problems:
        raise ConversionError("incompatible_keys", problems)
    return converted, mapped, sorted(ignored)


def convert(source, catalog, handle, output, precision=None):
    from .backend import offline_environment

    offline_environment()
    source, output = Path(source), Path(output)
    if source.suffix in UNSAFE_SUFFIXES:
        raise ConversionError("not_safetensors")
    safetensors_header(source)
    root, reference_files = reference_bundle(catalog, handle)
    if output.exists():
        raise ConversionError("output_exists")
    from safetensors.torch import load_file, save_file

    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    converted, mapped, ignored = convert_state(load_file(str(source), device="cpu"), root, precision)
    partial = output.with_name("." + output.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        partial.mkdir(parents=True, mode=0o755)
        for role, tensors in converted.items():
            (partial / role).mkdir()
            save_file(tensors, str(partial / role / WEIGHT_NAMES[role]))
            shutil.copyfile(root / role / "config.json", partial / role / "config.json")
        for role in COPIED:
            shutil.copytree(root / role, partial / role)
        shutil.copyfile(root / "model_index.json", partial / "model_index.json")
        record = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "reference_bundle_digest": digest(reference_files),
            "converter": versions(),
            "precision": precision or "preserved",
            "mapped_keys": mapped,
            "unmapped_keys": [],
            "ignored_keys": ignored,
        }
        (partial / "conversion.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        partial.rename(output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return record


def run(args):
    try:
        catalog = json.loads(Path(args.catalog).read_text())
        if not isinstance(catalog, dict):
            raise ValueError
    except (OSError, ValueError):
        print(json.dumps({"error": "invalid_catalog"}), file=sys.stderr)
        return 2
    try:
        record = convert(args.input, catalog, args.reference_handle, args.output, args.precision)
    except ConversionError as exc:
        # Key names are checkpoint structure, not private data; paths and exceptions never print.
        print(json.dumps({"error": exc.code, **({"keys": exc.keys} if exc.keys else {})}), file=sys.stderr)
        return 2
    except ImportError:
        print(json.dumps({"error": "missing_dependencies"}), file=sys.stderr)
        return 2
    except OSError:
        print(json.dumps({"error": "io_error"}), file=sys.stderr)
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0
