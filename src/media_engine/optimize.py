"""Targeted optimization options (phase E5): candidates, the benchmark index and advertising.

An optimization is an explicit value of a load option that the caller selects; it is never
applied implicitly and never chosen by `auto` resolution. A candidate is advertised under
`capabilities.optimizations` only when (a) the code that applies it is importable, (b) the
selected device can run it and (c) the packaged benchmark index (`optimizations.json`) holds an
entry for this backend and device class, citing a benchmark file recorded with
`tools/benchmark.py` on that class of hardware. Everything else is listed under
`capabilities.unadvertised_optimizations` with a coarse reason, so an operator can see why.

Switching an optimization is a load-identity change, like precision: the options are part of the
load body and of `loaded_bundle.options`.

Nothing here imports torch at module import; the API server stays dependency-free.
"""

import contextlib
import functools
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

from .contracts import FAMILIES, WorkerError, minor

ALL_FAMILIES = tuple(FAMILIES)
# Families with attention (esrgan is convolutional) and with large linear layers (int8
# weight-only quantizes nn.Linear weights only, so it does nothing useful for esrgan).
ATTENTION_FAMILIES = ("sdxl-base", "svd", "stable-audio-open")

# The load options added by minor 1.3 for optimizations, with the value that means "off".
NEW_OPTIONS = {"quantization": "none", "compile": "none"}
# The baseline value of each optimizable option (what an unoptimized load sends).
BASELINE = {"precision": None, "attention": "sdpa", "quantization": "none", "compile": "none"}
SINCE = 3  # Optimizations exist for media-worker/1.3 callers only.

CANDIDATES = (
    {
        "option": "precision",
        "value": "bfloat16",
        # Not svd: the pinned library pipeline upcasts its VAE input only for float16, so a
        # bfloat16 VAE receives float32 frames and fails (decisions.md, 0.8.0).
        "families": ("sdxl-base", "esrgan", "stable-audio-open"),
        "requires": (),
        "device": "bfloat16",
        "adapters": True,
        "effect": "every component is held and run in bfloat16 (2 bytes per parameter); the SDXL VAE is not upcast because bfloat16 has the float32 exponent range",
    },
    {
        "option": "attention",
        "value": "flash",
        "families": ATTENTION_FAMILIES,
        # PyTorch's own FlashAttention-2 kernel behind scaled_dot_product_attention: no package
        # beyond the pinned torch (decisions.md, 0.8.0).
        "requires": (),
        "device": "flash",
        "precisions": ("float16", "bfloat16"),
        "adapters": True,
        "effect": "scaled-dot-product attention is restricted to the FlashAttention kernel for every execution of this load; a shape the kernel cannot serve fails instead of falling back",
    },
    {
        "option": "quantization",
        "value": "int8-weight-only",
        "families": ATTENTION_FAMILIES,
        "requires": (("torchao", "torchao"),),
        "device": None,
        "adapters": False,
        "effect": "the denoiser's linear-layer weights are stored as int8 with per-channel scales and dequantized in the matmul; activations stay at the load precision",
    },
    {
        "option": "compile",
        "value": "inductor",
        "families": ALL_FAMILIES,
        "requires": (),
        "device": "inductor",
        "adapters": True,
        "effect": "the denoiser's (esrgan: the network's) forward is compiled with torch.compile(backend='inductor'); the first execution of each shape includes compilation time",
    },
)
CANDIDATE_KEYS = {(c["option"], c["value"]) for c in CANDIDATES}
INDEX_PATH = Path(__file__).with_name("optimizations.json")
INDEX_SCHEMA = 1


def candidate(option, value):
    return next(c for c in CANDIDATES if (c["option"], c["value"]) == (option, value))


def read_index(path=INDEX_PATH):
    """The packaged benchmark index; a malformed index advertises nothing rather than guessing."""
    try:
        index = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(index, dict) or index.get("schema_version") != INDEX_SCHEMA:
        return []
    entries = index.get("entries")
    return [e for e in entries if valid_entry(e)] if isinstance(entries, list) else []


ENTRY_FIELDS = (
    "option",
    "value",
    "backend_ids",
    "device_class",
    "families",
    "benchmark",
    "benchmark_sha256",
    "determinism",
    "quality_note",
    "improvement",
)


def valid_entry(entry):
    return (
        isinstance(entry, dict)
        and set(ENTRY_FIELDS) <= entry.keys()
        and (entry["option"], entry["value"]) in CANDIDATE_KEYS
        and isinstance(entry["backend_ids"], list)
        and isinstance(entry["device_class"], dict)
        and isinstance(entry["families"], list)
        and all(isinstance(entry[k], str) and entry[k] for k in ("benchmark", "benchmark_sha256", "determinism", "quality_note", "improvement"))
    )


def class_matches(entry_class, device_class):
    """An index entry names the device class it was measured on; every field it names must match."""
    return device_class is not None and all(
        device_class.get(key) == value for key, value in entry_class.items()
    )


@functools.lru_cache(maxsize=None)
def compiler_available():
    """Inductor's CPU code generation needs a C++ compiler ($CXX, else g++)."""
    return shutil.which(os.environ.get("CXX") or "g++") is not None


def device_class(device):
    """The class of hardware a benchmark is valid for; None when the device cannot be inspected."""
    if device == "cpu":
        return {"kind": "cpu", "architecture": platform.machine()}
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if not torch.cuda.is_available() or int(device.split(":")[1]) >= torch.cuda.device_count():
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "kind": "cuda",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def device_supports(kind, device, klass):
    """Whether the selected device can run a candidate at all (independent of any benchmark)."""
    if kind is None:
        return True
    if klass is None:
        return False
    if klass["kind"] != "cuda":
        # PyTorch runs bfloat16 on every CPU (fast only with native support, which the
        # benchmark decides); FlashAttention is a CUDA kernel; inductor needs a C++ compiler.
        return {"bfloat16": True, "flash": False, "inductor": compiler_available()}[kind]
    import torch

    if kind == "bfloat16":
        return torch.cuda.is_bf16_supported()
    if kind == "flash":
        major = int(klass["compute_capability"].split(".")[0])
        return major >= 8 and torch.backends.cuda.is_flash_attention_available()
    if kind == "inductor":
        return importlib.util.find_spec("triton") is not None
    raise ValueError(kind)


def survey(backend_id, device, families, index=None, missing=None):
    """(advertised, unadvertised) for a backend on a device serving `families`.

    `missing(dependencies)` returns the absent distribution names; `index` is the list of
    benchmark entries (the packaged index by default).
    """
    index = read_index() if index is None else index
    klass = device_class(device)
    advertised, unadvertised = [], []
    for c in CANDIDATES:
        served = [f for f in c["families"] if f in families]
        if not served:
            continue
        key = {"option": c["option"], "value": c["value"]}
        absent = missing(c["requires"]) if c["requires"] else []
        if absent:
            unadvertised.append({**key, "reason": "missing_dependency", "missing": absent})
            continue
        if not device_supports(c["device"], device, klass):
            unadvertised.append({**key, "reason": "device_unsupported"})
            continue
        entry = next(
            (
                e
                for e in index
                if (e["option"], e["value"]) == (c["option"], c["value"])
                and backend_id in e["backend_ids"]
                and class_matches(e["device_class"], klass)
            ),
            None,
        )
        if entry is None:
            unadvertised.append({**key, "reason": "no_benchmark"})
            continue
        families_listed = [f for f in served if f in entry["families"]]
        if not families_listed:
            unadvertised.append({**key, "reason": "no_benchmark"})
            continue
        record = {
            **key,
            "families": families_listed,
            "device_class": entry["device_class"],
            "benchmark": entry["benchmark"],
            "benchmark_sha256": entry["benchmark_sha256"],
            "improvement": entry["improvement"],
            "determinism": entry["determinism"],
            "quality_note": entry["quality_note"],
            "effect": c["effect"],
            "adapters": c["adapters"],
        }
        if "precisions" in c:
            record["precisions"] = list(c["precisions"])
        advertised.append(record)
    return advertised, unadvertised


def requested(options):
    """The non-baseline optimization values a load body selects, as (option, value) pairs."""
    pairs = []
    for option, value in options.items():
        if (option, value) in CANDIDATE_KEYS:
            pairs.append((option, value))
    return pairs


def check_load(options, version, caps, family, adapters, authorized):
    """Refuse an optimization the worker does not advertise, or a combination it cannot honor.

    Called after the strict field check; `options` may hold `quantization`/`compile` only on 1.3.
    """
    pairs = requested(options)
    if not pairs:
        return
    if minor(version) < SINCE:
        raise WorkerError(400, "unsupported", "options")
    offered = {(o["option"], o["value"]): o for o in caps.get("optimizations", [])}
    for pair in pairs:
        record = offered.get(pair)
        if record is None or family not in record["families"]:
            raise WorkerError(400, "unsupported", "options")
        if "precisions" in record and options["precision"] not in record["precisions"]:
            # flash needs an explicit half precision; `auto` may resolve to float32 on a CPU.
            raise WorkerError(400, "unsupported", "options")
        if adapters and not record["adapters"]:
            raise WorkerError(400, "unsupported", "adapters")
    if "cpu" in authorized:
        # A benchmark is valid for one device class; a CPU fallback would silently leave it.
        raise WorkerError(400, "unsupported", "authorized_fallbacks")


def estimate_note(options):
    """What the header-based estimate does and does not account for, per selected optimization."""
    notes = []
    for option, value in requested(options):
        if option == "precision":
            continue  # bfloat16 is counted at 2 bytes per parameter like float16.
        if option == "quantization":
            effect = "value_mb counts the unquantized weights; int8 weight-only stores the denoiser's linear weights at one byte per element after load, so resident memory is lower than stated"
        elif option == "compile":
            effect = "excludes compiled-graph and code-cache memory, which the benchmark measures"
        else:
            effect = "FlashAttention does not materialize the attention matrix; activations are excluded as before"
        notes.append({"option": option, "value": value, "effect": effect})
    return notes


# ----- applying options (torch imported lazily, only by backends) -------------------------
def denoiser_optimizations(module, options):
    """Quantize, then compile, the module that dominates the sampling loop. Returns the module."""
    if options.get("quantization") == "int8-weight-only":
        from torchao.quantization import int8_weight_only, quantize_

        quantize_(module, int8_weight_only())
    if options.get("compile") == "inductor":
        import torch

        # Compile forward only: the module's hooks (cancellation checks) stay eager.
        module.forward = torch.compile(module.forward, backend="inductor", dynamic=False)
    return module


def attention_scope(options):
    """A context that restricts SDPA to the FlashAttention kernel for `attention: flash`."""
    if options and options.get("attention") == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    return contextlib.nullcontext()


def release(options):
    """Drop compiled graphs with the model they were compiled for."""
    if options and options.get("compile") == "inductor":
        import torch

        torch._dynamo.reset()
