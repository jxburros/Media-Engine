"""Build offline, random-weight SDXL/LoRA artifacts for Runtime's real-backend tests.

Unlike serve_fixture.py, these artifacts are executed by the real inference engines.
Requires the inference extra. No weights or tokenizer files are downloaded.
"""

import argparse
import json
from pathlib import Path

from test_inference import tiny_lora, tiny_sdxl_bundle
from fixture import describe_bundle


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model, _ = tiny_sdxl_bundle(root)
    tiny_lora(root / "lora", model)
    catalog = {"fixture": str(model), "fixture-lora": str(root / "lora")}
    (root / "catalog.json").write_text(json.dumps(catalog) + "\n")
    # Conformance uses the same verified bytes under its own opaque handles.
    bundle = describe_bundle(model, "fixture", "tiny-sdxl", source={"license": "test-only random weights"})
    (root / "registration.json").write_text(json.dumps({"bundle": bundle}) + "\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    prepare(args.directory)
