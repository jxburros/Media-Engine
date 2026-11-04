"""Verified local SDXL load components. No hub identifiers or pickle files are accepted."""
import hashlib
import json
import re
from pathlib import Path

from .contracts import WorkerError, approved_snapshot_link, digest, minor, safetensors_header, strict

FILES = {
    "vae": {"config.json", "diffusion_pytorch_model.safetensors"},
    "textual-inversion": {"embedding.safetensors"},
    "ip-adapter": {"ip_adapter.safetensors", "image_encoder/config.json", "image_encoder/model.safetensors", "preprocessor_config.json"},
}
CAPABILITIES = {"max_components": 4, "kinds": list(FILES), "families": ["sdxl-base"],
                "identity": "load", "ip_adapter_tasks": ["text-to-image"]}


def summaries(entries):
    return [{**{k: e["component"][k] for k in ("conditioning_id", "kind", "digest")},
             **{k: e[k] for k in ("scale", "token", "encoder") if k in e}} for e in entries]


def validate(entries, catalog, version, verify=True):
    if not isinstance(entries, list) or len(entries) > 4:
        raise WorkerError(400, "invalid_request", "conditioning")
    if entries and minor(version) < 4:
        raise WorkerError(400, "unsupported", "conditioning")
    result, seen, identities = [], set(), set()
    for entry in entries:
        strict(entry, ("component",), ("scale", "token", "encoder"))
        record = entry["component"]
        strict(record, ("schema_version", "conditioning_id", "kind", "artifact_handle", "files", "source", "digest"))
        kind = record["kind"]
        if type(record["schema_version"]) is not int or record["schema_version"] != 1 or not isinstance(kind,str) or kind not in FILES:
            raise WorkerError(400, "unsupported", "conditioning.kind")
        identity = record["conditioning_id"]
        if not isinstance(identity,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity) or identity in identities:
            raise WorkerError(400, "invalid_request", "conditioning_id")
        identities.add(identity)
        if digest({k:v for k,v in record.items() if k != "digest"}) != record["digest"]:
            raise WorkerError(400, "artifact_changed", "conditioning.digest")
        options = {"vae": (), "textual-inversion": ("token", "encoder"), "ip-adapter": ("scale",)}[kind]
        strict(entry, ("component",) + options)
        slot = kind
        if kind == "textual-inversion":
            if entry["encoder"] not in ("text_encoder", "text_encoder_2"):
                raise WorkerError(400, "invalid_request", "conditioning.encoder")
            token = entry["token"]
            if not isinstance(token,str) or not re.fullmatch(r"[^\s\x00-\x1f\x7f]{1,80}",token):
                raise WorkerError(400, "invalid_request", "conditioning.token")
            slot += ":" + entry["encoder"]
        if kind == "ip-adapter" and (type(entry["scale"]) not in (int,float) or not 0 <= entry["scale"] <= 2):
            raise WorkerError(400, "invalid_request", "conditioning.scale")
        if slot in seen:
            raise WorkerError(400, "invalid_request", "conditioning")
        seen.add(slot)
        strict(record["source"], (), ("repository", "revision", "license"))
        if any(not isinstance(v,str) for v in record["source"].values()):
            raise WorkerError(400, "invalid_request", "source")
        files = record["files"]
        if not isinstance(files,dict) or set(files) != FILES[kind] or any(not isinstance(v,str) or not re.fullmatch(r"[a-f0-9]{64}",v) for v in files.values()):
            raise WorkerError(400, "unsupported", "conditioning.files")
        handle = record["artifact_handle"]
        if not isinstance(handle,str) or handle not in catalog:
            raise WorkerError(400, "missing_component", "artifact_handle")
        root = Path(catalog[handle]).resolve()
        if not root.is_dir():
            raise WorkerError(400, "missing_component", "conditioning")
        paths = {}
        for path in root.rglob("*"):
            if path.is_symlink() and not approved_snapshot_link(path,root):
                raise WorkerError(400, "unsupported", "conditioning.files")
            if path.is_dir():
                continue
            if not path.is_file():
                raise WorkerError(400, "unsupported", "conditioning.files")
            paths[path.relative_to(root).as_posix()] = path
        if set(paths) != set(files):
            raise WorkerError(400, "artifact_changed", "conditioning.files")
        for name,path in paths.items():
            if verify:
                sha = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        sha.update(block)
                if sha.hexdigest() != files[name]:
                    raise WorkerError(400, "artifact_changed", "conditioning.files")
        try:
            for name,path in paths.items():
                if name.endswith(".json"):
                    if path.stat().st_size > 1024*1024:
                        raise ValueError()
                    config = json.loads(path.read_text())
                    if not isinstance(config,dict) or "auto_map" in config:
                        raise ValueError()
                    if kind == "vae" and config.get("_class_name") != "AutoencoderKL":
                        raise ValueError()
                    if name == "image_encoder/config.json" and config.get("model_type") != "clip_vision_model":
                        raise ValueError()
                else:
                    header = safetensors_header(path)
                    keys = set(header) - {"__metadata__"}
                    if not keys or any(v.get("dtype") not in ("F16", "BF16", "F32") for k,v in header.items() if k != "__metadata__"):
                        raise ValueError()
                    if kind == "textual-inversion" and (keys != {"embedding"} or len(header["embedding"]["shape"]) not in (1,2)):
                        raise ValueError()
                    if name == "ip_adapter.safetensors" and (not any(k.startswith("image_proj.") for k in keys) or not any(k.startswith("ip_adapter.") for k in keys) or any(not k.startswith(("image_proj.","ip_adapter.")) for k in keys)):
                        raise ValueError()
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            raise WorkerError(400, "unsupported", "conditioning.layout") from None
        result.append((entry, root))
    return result


def attach(backend, resolved):
    import torch
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    pipe = backend.pipe
    dtype = getattr(torch, backend.load_options["precision"])
    backend.has_ip_adapter = False
    for entry,root in resolved:
        kind = entry["component"]["kind"]
        if kind == "vae":
            vae = AutoencoderKL.from_pretrained(str(root), local_files_only=True, use_safetensors=True, torch_dtype=dtype)
            if vae.config.latent_channels != pipe.vae.config.latent_channels or len(vae.config.block_out_channels) != len(pipe.vae.config.block_out_channels):
                raise WorkerError(400, "unsupported", "conditioning.vae")
            pipe.register_modules(vae=vae.to(backend.run_device))
        elif kind == "textual-inversion":
            encoder_name = entry["encoder"]
            encoder = getattr(pipe,encoder_name)
            tokenizer = getattr(pipe,"tokenizer_2" if encoder_name.endswith("_2") else "tokenizer")
            embedding = load_file(str(root/"embedding.safetensors"))["embedding"]
            if embedding.shape[-1] != encoder.config.hidden_size or embedding.ndim == 2 and not 1 <= embedding.shape[0] <= 16:
                raise WorkerError(400, "unsupported", "conditioning.embedding")
            if not torch.isfinite(embedding).all() or entry["token"] in tokenizer.get_vocab():
                raise WorkerError(400, "invalid_request", "conditioning.token")
            pipe.load_textual_inversion(embedding, token=entry["token"], tokenizer=tokenizer, text_encoder=encoder)
        else:
            encoder = CLIPVisionModelWithProjection.from_pretrained(str(root/"image_encoder"), local_files_only=True, use_safetensors=True, torch_dtype=dtype)
            extractor = CLIPImageProcessor.from_pretrained(str(root), local_files_only=True)
            pipe.register_modules(image_encoder=encoder.to(backend.run_device), feature_extractor=extractor)
            pipe.load_ip_adapter(str(root), subfolder="", weight_name="ip_adapter.safetensors", image_encoder_folder=None, local_files_only=True)
            pipe.set_ip_adapter_scale(entry["scale"])
            backend.has_ip_adapter = True
    backend.text_cache.clear()
