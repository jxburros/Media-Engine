"""Benchmark load options on one bundle and device: the evidence an optimization needs (E5).

Usage (from the repository root, with the inference extras installed):

    python tools/benchmark.py --backend diffusers-sdxl --device cuda:0 \\
        --catalog catalog.json --registration sdxl.json \\
        --option precision=bfloat16 --option attention=flash \\
        --option quantization=int8-weight-only --option compile=inductor

For each option set (the baseline first, then each `--option` alone, then any `--combine`d set)
the tool loads the bundle in-process through the real worker, runs a fixed request set
`--runs` times after one warm-up execution, and records load time, first-execution time
(which includes compilation for `compile`), steady-state wall time, peak device memory, every
output digest and, against the baseline, the per-pixel (PNG) or per-sample (WAV) difference.
It writes `docs/benchmarks/<date>-<device class>.json` and prints, for every option set that
qualifies (at least `--min-speedup` faster or `--max-memory-ratio` lighter than the baseline),
the index entry a release would add to `src/media_engine/optimizations.json`. It never edits
the index: citing a benchmark is a release decision, recorded with the file's SHA-256.

Every candidate is loadable here whether or not it is advertised, because the tool runs the
backend with a benchmark index naming every candidate for this device class.
"""

import argparse
import hashlib
import io
import json
import statistics
import struct
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from media_engine import optimize, png  # noqa: E402
from media_engine.contracts import DEFAULTS, PROTOCOL, WORKER_VERSION  # noqa: E402

SCHEMA_VERSION = 1


def pending_index(backend_id, klass):
    """An index naming every candidate for this class, so the run can load each one."""
    return [
        {
            "option": c["option"],
            "value": c["value"],
            "backend_ids": [backend_id],
            "device_class": klass or {},
            "families": list(c["families"]),
            "benchmark": "pending: this benchmark run",
            "benchmark_sha256": "0" * 64,
            "improvement": "pending",
            "determinism": "pending",
            "quality_note": "pending",
        }
        for c in optimize.CANDIDATES
    ]


def request_set(family, size, steps, duration):
    """The fixed requests per family: three seeds, so outputs and differences are comparable."""
    width, height = size
    if family == "sdxl-base":
        return [
            {
                "task": "text-to-image",
                "inputs": {"prompt": "a lighthouse on a cliff at dusk, oil painting", "negative_prompt": "blurry"},
                "parameters": {**DEFAULTS, "seed": seed, "width": width, "height": height, "steps": steps, "guidance": 6},
            }
            for seed in (1, 2, 3)
        ]
    if family == "esrgan":
        return [{"task": "upscale", "inputs": {"image": "@input"}, "parameters": {"width": width, "height": height}}]
    if family == "svd":
        return [
            {
                "task": "image-to-video",
                "inputs": {"image": "@input"},
                "parameters": {"seed": seed, "steps": steps, "width": width, "height": height},
            }
            for seed in (1, 2)
        ]
    if family == "stable-audio-open":
        return [
            {
                "task": "text-to-audio",
                "inputs": {"prompt": "rain on a tin roof"},
                "parameters": {"seed": seed, "steps": steps, "duration_seconds": duration},
            }
            for seed in (1, 2)
        ]
    raise ValueError(family)


def wait(worker, operation=None, execution=None):
    while True:
        if operation and worker.operations[operation]["state"] != "running":
            return worker.operations[operation]
        if execution and worker.snapshot(execution)["state"] != "running":
            return worker.snapshot(execution)
        time.sleep(0.01)


def samples(data):
    """Interleaved signed 16-bit samples of a canonical PCM WAV."""
    with wave.open(io.BytesIO(data), "rb") as stream:
        frames = stream.readframes(stream.getnframes())
    return struct.unpack(f"<{len(frames) // 2}h", frames)


def difference(baseline, candidate, mime_type):
    """Max and mean absolute difference: 8-bit levels for PNG, 16-bit steps for WAV."""
    if mime_type == "image/png":
        a, b = png.read_rgb(baseline), png.read_rgb(candidate)
        if a[:2] != b[:2]:
            return {"comparable": False}
        values = [abs(x - y) for x, y in zip(a[2], b[2])]
        unit = "8-bit levels"
    elif mime_type == "audio/wav":
        a, b = samples(baseline), samples(candidate)
        if len(a) != len(b):
            return {"comparable": False}
        values = [abs(x - y) for x, y in zip(a, b)]
        unit = "16-bit steps"
    else:
        same = hashlib.sha256(baseline).digest() == hashlib.sha256(candidate).digest()
        return {"comparable": True, "identical": same, "unit": "bytes"}
    return {"comparable": True, "max": max(values), "mean": round(sum(values) / len(values), 4), "unit": unit}


def measure(worker, registration, options, requests, runs, input_png):
    """Load once with `options`, warm up, run the request set `runs` times; return the record."""
    body = {"protocol": PROTOCOL, "bundle": registration, "options": options}
    started = time.perf_counter()
    operation = wait(worker, operation=worker.control("load", body)["operation_id"])
    load_seconds = time.perf_counter() - started
    if operation["state"] != "succeeded":
        return {"options": options, "error": operation["error"]}
    digest = registration["bundle_digest"]
    input_id = worker.store_input(input_png)["input_id"] if input_png else None
    executions, peaks, outputs, first = [], [], [], None
    for run in range(runs + 1):
        elapsed = 0.0
        for k, request in enumerate(requests):
            inputs = {
                name: (input_id if value == "@input" else value)
                for name, value in request["inputs"].items()
            }
            submitted = {
                **request,
                "inputs": inputs,
                "protocol": PROTOCOL,
                "attempt_id": f"attempt_bench_{time.monotonic_ns()}_{run}_{k}",
                "bundle_digest": digest,
            }
            t = time.perf_counter()
            snap = wait(worker, execution=worker.submit(submitted)["execution_id"])
            elapsed += time.perf_counter() - t
            if snap["state"] != "succeeded":
                return {"options": options, "error": snap["error"]}
            peaks.append(snap.get("measurements") or {})
            if run == 0:
                outputs.append(
                    [
                        {
                            "output_id": o["output_id"],
                            "mime_type": o["mime_type"],
                            "sha256": o["sha256"],
                            "bytes": worker.output_path(snap["execution_id"], o["output_id"])[0].read_bytes(),
                        }
                        for o in snap["outputs"]
                    ]
                )
            worker.release(snap["execution_id"])
        if run == 0:
            first = elapsed  # the warm-up: includes compilation and first-touch allocation
        else:
            executions.append(elapsed)

    def peak(key):
        values = [m[key] for m in peaks if isinstance(m.get(key), (int, float))]
        return max(values) if values else None

    return {
        "options": options,
        "load_seconds": round(load_seconds, 3),
        "first_execution_seconds": round(first, 3),
        "execution_seconds": [round(e, 3) for e in executions],
        "median_seconds": round(statistics.median(executions), 3) if executions else None,
        "peak_allocated_mb": peak("peak_allocated_mb"),
        "peak_reserved_mb": peak("peak_reserved_mb"),
        "process_peak_rss_mb": peak("process_peak_rss_mb"),
        "memory_estimate_mb": operation["memory_estimate"]["value_mb"],
        "outputs": outputs,
    }


def summarize(baseline, result, min_speedup, max_memory_ratio):
    """Speed and memory relative to the baseline, output differences and the qualification."""
    summary = {}
    if baseline.get("median_seconds") and result.get("median_seconds"):
        summary["speedup"] = round(baseline["median_seconds"] / result["median_seconds"], 3)
    # Device peaks are reset per execution. Process RSS is a lifetime high-water mark, so it is
    # recorded but never compared between option sets measured in one process.
    if baseline.get("peak_allocated_mb") and result.get("peak_allocated_mb"):
        summary["memory_ratio"] = round(result["peak_allocated_mb"] / baseline["peak_allocated_mb"], 3)
        summary["memory_measure"] = "peak_allocated_mb"
    summary["differences"] = [
        [
            {"output_id": b["output_id"], **difference(b["bytes"], c["bytes"], b["mime_type"])}
            for b, c in zip(base_outputs, outputs)
        ]
        for base_outputs, outputs in zip(baseline["outputs"], result["outputs"])
    ]
    summary["qualifies"] = bool(
        summary.get("speedup", 0) >= min_speedup or summary.get("memory_ratio", 2) <= max_memory_ratio
    )
    return summary


def benchmark(worker, registration, family, option_sets, runs, requests, input_png=None, min_speedup=1.05, max_memory_ratio=0.95, note=None):
    """Run every option set (the first is the baseline) and return the benchmark document."""
    results = []
    for options in option_sets:
        results.append(measure(worker, registration, options, requests, runs, input_png))
    baseline = results[0]
    for result in results[1:]:
        if "error" not in result and "error" not in baseline:
            result["summary"] = summarize(baseline, result, min_speedup, max_memory_ratio)
    for result in results:
        for outputs in result.get("outputs", []):
            for output in outputs:
                output.pop("bytes")
    caps = worker.capabilities()
    identity = worker.backend.device_identity()
    identity.pop("pid", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/benchmark.py",
        "media_engine": WORKER_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend_id": caps["backend_id"],
        "device_class": optimize.device_class(worker.backend.device) if worker.backend.device != "test-only" else {"kind": "test-fixture"},
        "device_identity": identity,
        "dependencies": caps.get("dependencies", {}),
        "bundle_digest": registration["bundle_digest"],
        "family": family,
        "runs": runs,
        "requests": [{k: v for k, v in r.items() if k != "inputs"} for r in requests],
        "thresholds": {"min_speedup": min_speedup, "max_memory_ratio": max_memory_ratio},
        "note": note,
        "results": results,
    }


def index_entries(document, path, families):
    """The index entries a release would add for the qualifying option sets (single options only)."""
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    baseline = document["results"][0]["options"]
    entries = []
    for result in document["results"][1:]:
        changed = {k: v for k, v in result["options"].items() if baseline.get(k) != v}
        if len(changed) != 1 or not result.get("summary", {}).get("qualifies"):
            continue
        (option, value), = changed.items()
        summary = result["summary"]
        worst = max(
            (d.get("max", 0) for outputs in summary["differences"] for d in outputs if d.get("comparable")),
            default=None,
        )
        entries.append(
            {
                "option": option,
                "value": value,
                "backend_ids": [document["backend_id"]],
                "device_class": document["device_class"],
                "families": families,
                "benchmark": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
                "benchmark_sha256": sha,
                "improvement": f"speedup {summary.get('speedup')}x"
                + (f", peak allocated memory ratio {summary['memory_ratio']}" if "memory_ratio" in summary else ""),
                "determinism": "REVIEW: reproducible on this device and build? (rerun and compare digests)",
                "quality_note": f"REVIEW: max difference from the baseline {worst} on the benchmark requests",
            }
        )
    return entries


def parse_options(text):
    option, _, value = text.partition("=")
    if (option, value) not in optimize.CANDIDATE_KEYS:
        raise argparse.ArgumentTypeError(f"not an optimization candidate: {text}")
    return option, value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=("diffusers-sdxl", "owned-sdxl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path, help="the bundle registration document (JSON)")
    parser.add_argument("--precision", default=None, help="baseline precision (default: the backend's load default)")
    parser.add_argument("--option", action="append", type=parse_options, default=[], help="option=value to measure alone")
    parser.add_argument("--combine", action="append", default=[], help="comma-separated option=value set measured together")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=10, help="text-to-audio seconds")
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--max-memory-ratio", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmarks")
    parser.add_argument("--label", default=None, help="file name suffix (default: the device class)")
    parser.add_argument("--note", default=None, help="free text recorded in the document (bundle, purpose)")
    args = parser.parse_args(argv)

    from media_engine.backend import backend_class
    from media_engine.server import Worker

    registration = json.loads(args.registration.read_text())
    catalog = json.loads(args.catalog.read_text())
    family = registration["family"]
    klass = optimize.device_class(args.device)
    backend = backend_class(args.backend)(args.device, None, pending_index(args.backend, klass))
    defaults = backend.capabilities()["load_defaults"]
    base = {**defaults, "precision": args.precision or defaults["precision"]}
    option_sets = [base] + [{**base, option: value} for option, value in args.option]
    for combined in args.combine:
        option_sets.append({**base, **dict(parse_options(item) for item in combined.split(","))})
    size = (args.width, args.height)
    input_png = None
    if family in ("esrgan", "svd"):
        input_png = png.write(args.width, args.height, bytes((0x80, 0x60, 0x40)) * (args.width * args.height))
    with tempfile.TemporaryDirectory() as directory:
        worker = Worker(directory, backend, catalog, max_storage_mb=1 << 20)
        document = benchmark(
            worker,
            registration,
            family,
            option_sets,
            args.runs,
            request_set(family, size, args.steps, args.duration),
            input_png,
            args.min_speedup,
            args.max_memory_ratio,
            args.note,
        )
    label = args.label or (
        "-".join(str(v) for v in klass.values()).replace(" ", "_").lower() if klass else args.device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{datetime.now(timezone.utc).date().isoformat()}-{label}.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {path}")
    entries = index_entries(document, path, [family])
    if entries:
        print("qualifying index entries (review determinism and quality_note before citing):")
        print(json.dumps(entries, indent=2))
    else:
        print("no option set qualified; nothing to cite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
