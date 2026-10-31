"""Oracle test: the native SDXL pipeline against the Python owned pipeline on the tiny bundle.

Builds the tiny random-weight SDXL-shaped bundle and LoRA from tests/test_inference.py in a
temporary directory, runs `OwnedPipelineBackend("cpu")` and the Rust `oracle` example on identical
requests, and asserts the decoded pixels differ by at most one 8-bit level, that the event
sequences and effective steps match, that `count: 2` image 2 equals a seed + 1 single run, and
that the LoRA changes the output while still matching Python. Cancellation at every checkpoint
must stop both implementations after the same events.

A second, "wide" bundle exercises what the tiny one does not and the real SDXL base bundle
does: three resolution levels, two resnets per block, per-block transformer depths and head
counts, a quick_gelu encoder beside a gelu encoder of a different width, a tokenizer_2 whose pad
token is "!", the non-legacy eos pooling, sharded `*.safetensors.index.json` weights, a
three-level VAE (two asymmetric-pad downsamples) and a latent size that is not a multiple of
the UNet's overall downsampling (the `upsample_size` path).

    PYTHONPATH=src:tests /tmp/venv/bin/python native/sdxl/tests/oracle.py

`ORACLE_BIN` may point at a prebuilt example binary; otherwise it is built with cargo.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[2]
REPO = NATIVE.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

from fixture import make_png  # noqa: E402
from media_engine import png  # noqa: E402
from test_inference import tiny_lora, tiny_sdxl_bundle  # noqa: E402

TOLERANCE = 1


def wide_sdxl_bundle(root):
    """SDXL-base-shaped in structure (3 levels, depths [1, 2, 3], heads [2, 4, 8]), tiny in width."""
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    torch.manual_seed(11)
    root = Path(root)
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    words = [w + "</w>" for w in ("a", "test", "portrait", "window", "light")]
    tail = letters + [c + "</w>" for c in letters] + [",", ",</w>", "!</w>"]
    vocabularies = {
        "tok": ["<|startoftext|>", "<|endoftext|>", "!"] + words + tail,
        # tokenizer_2 as in SDXL: "!" is id 0 and the pad token; eos is not id 2.
        "tok2": ["!", "<|startoftext|>", "<|endoftext|>"] + words + tail,
    }
    tokenizers = {}
    for name, vocabulary in vocabularies.items():
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocabulary)}))
        (folder / "merges.txt").write_text("#version: 0.2\nt e\nte s\ntes t</w>\np o\npo r\n")
        tokenizers[name] = CLIPTokenizer(
            str(folder / "vocab.json"),
            str(folder / "merges.txt"),
            model_max_length=77,
            pad_token="!" if name == "tok2" else "<|endoftext|>",
        )
    hidden, hidden_2, projection = 32, 48, 24
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64, 64),
        layers_per_block=2,
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4, 8),
        transformer_layers_per_block=(1, 2, 3),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=6 * 8 + projection,
        cross_attention_dim=hidden + hidden_2,
        norm_num_groups=32,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 3,
        up_block_types=["UpDecoderBlock2D"] * 3,
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=32,
        scaling_factor=0.13025,
    )
    common = dict(layer_norm_eps=1e-05, max_position_embeddings=77, pad_token_id=1, bos_token_id=0)
    encoder = CLIPTextModel(
        CLIPTextConfig(hidden_size=hidden, intermediate_size=37, num_attention_heads=4, num_hidden_layers=3,
                       hidden_act="quick_gelu", eos_token_id=2, vocab_size=len(vocabularies["tok"]),
                       projection_dim=32, **common)
    )
    encoder_2 = CLIPTextModelWithProjection(
        CLIPTextConfig(hidden_size=hidden_2, intermediate_size=64, num_attention_heads=4, num_hidden_layers=4,
                       hidden_act="gelu", vocab_size=len(vocabularies["tok2"]), projection_dim=projection,
                       # Non-legacy pooling: the first position of tokenizer_2's eos id (2 is legacy).
                       eos_token_id=tokenizers["tok2"].eos_token_id, **common)
    )
    pipe = StableDiffusionXLPipeline(
        unet=unet,
        vae=vae,
        text_encoder=encoder,
        text_encoder_2=encoder_2,
        tokenizer=tokenizers["tok"],
        tokenizer_2=tokenizers["tok2"],
        scheduler=EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", timestep_spacing="leading", steps_offset=1
        ),
        add_watermarker=False,
    )
    bundle = root / "model"
    pipe.save_pretrained(str(bundle), safe_serialization=True)
    # Sharded layouts, as large real components ship.
    for path in (bundle / "unet").glob("*.safetensors"):
        path.unlink()
    unet.save_pretrained(str(bundle / "unet"), safe_serialization=True, max_shard_size="300KB")
    for path in (bundle / "text_encoder_2").glob("*.safetensors"):
        path.unlink()
    encoder_2.save_pretrained(str(bundle / "text_encoder_2"), safe_serialization=True, max_shard_size="40KB")
    assert list((bundle / "unet").glob("*.safetensors.index.json")), "unet was not sharded"
    assert list((bundle / "text_encoder_2").glob("*.safetensors.index.json")), "text_encoder_2 was not sharded"
    return bundle


class CancelAfter:
    """A cancellation flag that lets `n` checkpoints pass and is set from the next one on."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.n


def oracle_binary():
    if os.environ.get("ORACLE_BIN"):
        return Path(os.environ["ORACLE_BIN"])
    subprocess.run(
        ["cargo", "build", "--quiet", "-p", "media-engine-sdxl", "--example", "oracle"],
        cwd=NATIVE,
        check=True,
    )
    return NATIVE / "target" / "debug" / "examples" / "oracle"


def pixels(data):
    return png.read_rgb(data)


def difference(a, b):
    a, b = pixels(a), pixels(b)
    assert a[:2] == b[:2], (a[:2], b[:2])
    return max(abs(x - y) for x, y in zip(a[2], b[2]))


def mean_difference(a, b):
    a, b = pixels(a)[2], pixels(b)[2]
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class Python:
    """The Python owned pipeline, driven directly (one backend per adapter set)."""

    def __init__(self, model_dir, adapters=()):
        from media_engine.backend import backend_class

        self.backend = backend_class("owned-sdxl")("cpu")
        records = [({"adapter_id": f"adapter{i}"}, None, path, scale, "peft") for i, (path, scale) in enumerate(adapters)]
        self.backend.load(
            model_dir, {"precision": "float32", "attention": "sdpa", "offload": "none"}, records, "sdxl-base", "cpu"
        )
        self.merged = [a["modules"] for a in self.backend.merged_adapters]

    def run(self, case, image, directory, cancelled=None):
        params = {k: case[k] for k in ("seed", "steps", "guidance", "width", "height")}
        for key in ("count", "strength"):
            if key in case:
                params[key] = case[key]
        inputs = {"prompt": case["prompt"]}
        if "negative" in case:
            inputs["negative_prompt"] = case["negative"]
        events = []

        def emit(kind, data):
            if kind == "stage":
                events.append(f"stage {data['stage']}")
            elif kind == "progress":
                events.append(f"progress {data['completed_steps']}/{data['total_steps']}")

        request = {"task": case.get("task", "text-to-image"), "inputs": inputs, "parameters": params}
        started = time.perf_counter()
        from media_engine.backend import Cancelled

        try:
            records, measurements = self.backend.generate(request, directory, emit, cancelled or threading.Event(), image)
        except Cancelled:
            return None, events, None, time.perf_counter() - started
        seconds = time.perf_counter() - started
        # The server emits "saving" itself; the native crate stops at decoding.
        assert events[-1] == "stage saving", events
        outputs = [Path(r["path"]).read_bytes() for r in records]
        return outputs, events[:-1], measurements["effective_steps"], seconds


def run_native(binary, model_dir, case, image, directory, adapters=(), cancel_after=None):
    args = [str(binary), str(model_dir), str(directory), "--prompt", case["prompt"]]
    if cancel_after is not None:
        args += ["--cancel-after", str(cancel_after)]
    for key in ("seed", "steps", "guidance", "width", "height", "count", "strength"):
        if key in case:
            args += [f"--{key}", str(case[key])]
    if "negative" in case:
        args += ["--negative", case["negative"]]
    if case.get("task") == "image-to-image":
        args += ["--task", "image-to-image", "--image", str(image)]
    for path, scale in adapters:
        args += ["--lora", f"{path}:{scale}"]
    started = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True)
    seconds = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"native oracle failed: {result.stdout}\n{result.stderr}")
    lines = result.stdout.splitlines()
    events = [line for line in lines if line.startswith(("stage ", "progress "))]
    if any(line.startswith("cancelled") for line in lines):
        return None, events, None, None, seconds, None
    merged = json.loads(next(line for line in lines if line.startswith("merged "))[7:])
    steps = int(next(line for line in lines if line.startswith("effective_steps "))[16:])
    timing = next(line for line in lines if line.startswith("timing "))
    count = case.get("count", 1)
    names = ["image.png"] + [f"image_{k}.png" for k in range(2, count + 1)]
    outputs = [(Path(directory) / name).read_bytes() for name in names]
    return outputs, events, steps, merged, seconds, timing


def main():
    binary = oracle_binary()
    base = {"prompt": "a test portrait", "negative": "window", "seed": 7, "steps": 3, "guidance": 6, "width": 64, "height": 64}
    cases = {
        "text-to-image": dict(base),
        "guidance 1 (no CFG)": {**base, "guidance": 1},
        "negative prompt absent": {k: v for k, v in base.items() if k != "negative"},
        "negative prompt empty": {**base, "negative": ""},
        "count 2": {**base, "count": 2},
        "seed 8": {**base, "seed": 8},
        "image-to-image": {**base, "task": "image-to-image", "strength": 0.7},
    }
    failures = []
    report = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model_dir, _bundle = tiny_sdxl_bundle(root)
        tiny_lora(root / "lora", model_dir)
        adapter = [(root / "lora" / "adapter.safetensors", 0.8)]
        image = root / "flat.png"
        image.write_bytes(make_png(64, 64))
        python = Python(model_dir)
        results = {}
        current = {"model_dir": model_dir}

        def compare(name, case, py, adapters=()):
            directory = root / "out" / name.replace(" ", "_").replace("(", "").replace(")", "")
            (directory / "py").mkdir(parents=True)
            py_out, py_events, py_steps, py_seconds = py.run(case, image, directory / "py")
            rs_out, rs_events, rs_steps, rs_merged, rs_seconds, timing = run_native(
                binary, current["model_dir"], case, image, directory / "rs", adapters
            )
            diffs = [difference(a, b) for a, b in zip(py_out, rs_out)]
            ok = len(py_out) == len(rs_out) and all(d <= TOLERANCE for d in diffs)
            ok = ok and py_events == rs_events and py_steps == rs_steps and rs_merged == py.merged
            report.append(
                f"{name:24s} max|Δ|={diffs} effective_steps={rs_steps} events_equal={py_events == rs_events} "
                f"python={py_seconds:.2f}s native_process={rs_seconds:.2f}s ({timing[7:]})"
            )
            if not ok:
                failures.append(f"{name}: diffs {diffs}, events {py_events} vs {rs_events}, steps {py_steps}/{rs_steps}")
            results[name] = (py_out, rs_out)

        for name, case in cases.items():
            compare(name, case, python)
        lora_python = Python(model_dir, adapter)
        compare("LoRA scale 0.8", dict(base), lora_python, adapter)

        # count: 2 image 2 is the seed + 1 single run (on each implementation, and across them).
        batch_py, batch_rs = results["count 2"]
        single_py, single_rs = results["seed 8"]
        first_py, first_rs = results["text-to-image"]
        reproduce = {
            "native image_2 vs native seed 8": difference(batch_rs[1], single_rs[0]),
            "native image vs native seed 7": difference(batch_rs[0], first_rs[0]),
            "native image_2 vs python seed 8": difference(batch_rs[1], single_py[0]),
            "python image_2 vs python seed 8": difference(batch_py[1], single_py[0]),
        }
        for label, value in reproduce.items():
            report.append(f"{label:36s} max|Δ|={value}")
            if value > TOLERANCE:
                failures.append(f"{label}: {value}")
        lora_mean = mean_difference(results["LoRA scale 0.8"][1][0], first_rs[0])
        report.append(f"LoRA vs no adapter (native)          mean|Δ|={lora_mean:.2f}")
        if lora_mean <= 1:
            failures.append(f"LoRA did not change the native output (mean {lora_mean})")

        # Cancellation at each checkpoint: the same events, then cancelled (or, past the last
        # checkpoint, success) on both implementations.
        for n in range(0, 10):
            case = dict(base)
            directory = root / "out" / f"cancel{n}"
            (directory / "py").mkdir(parents=True)
            py_out, py_events, _, _ = python.run(case, image, directory / "py", CancelAfter(n))
            rs_out, rs_events, *_ = run_native(binary, model_dir, case, image, directory / "rs", cancel_after=n)
            same = py_events == rs_events and (py_out is None) == (rs_out is None)
            report.append(
                f"cancel after {n} checkpoints      python={'cancelled' if py_out is None else 'finished'} "
                f"native={'cancelled' if rs_out is None else 'finished'} events_equal={py_events == rs_events}"
            )
            if not same:
                failures.append(f"cancel after {n}: {py_events} / {rs_events}")

        wide_dir = wide_sdxl_bundle(root / "wide")
        wide = Python(wide_dir)
        wide_base = {
            "prompt": "a test portrait, window light!",
            "negative": "window",
            "seed": 3,
            "steps": 4,
            "guidance": 5,
            "width": 64,
            "height": 64,
        }
        current["model_dir"] = wide_dir
        compare("wide text-to-image", wide_base, wide)
        compare("wide 72x64 upsample_size", {**wide_base, "width": 72, "seed": 4}, wide)
        compare("wide 64x96 count 2", {**wide_base, "height": 96, "count": 2}, wide)
        compare("wide image-to-image", {**wide_base, "task": "image-to-image", "strength": 0.5}, wide)
        current["model_dir"] = model_dir
    print("\n".join(report))
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"oracle: all cases within {TOLERANCE} level(s)")


if __name__ == "__main__":
    main()
