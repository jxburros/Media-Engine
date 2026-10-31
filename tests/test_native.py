"""The owned native engine against its oracles (Engine 1.0.0, phase E6).

Skipped unless MEDIA_ENGINE_NATIVE names a built `media-engine-native` binary
(`cargo build --release --manifest-path native/Cargo.toml`). Two oracles:

- The Python owned pipeline (`owned-sdxl`) on the tiny random-weight SDXL-shaped bundle from
  test_inference.py: for the same request the native engine must reproduce its pixels within
  the stated tolerance (same seed → same image), LoRA adapters must change the output the same
  way, and the stage/progress event shapes and cancellation behavior must be identical. These
  need the inference extra; without it only the conformance test runs.
- The conformance suite, which must report the native engine conformant with no skipped section.
"""

import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from fixture import make_png
from test_inference import HAVE_OWNED, difference, tiny_lora, tiny_sdxl_bundle, tiny_esrgan_bundle
from media_engine.contracts import DEFAULTS, PROTOCOL

NATIVE = os.environ.get("MEDIA_ENGINE_NATIVE")
HERE = Path(__file__).resolve().parent
# Measured, not chosen: on the tiny bundle the native engine and owned-sdxl agree to within one
# 8-bit level (float32 CPU; different matmul orders, same arithmetic).
TOLERANCE = 1


class Native:
    """A launched native worker and a minimal HTTP client for it."""

    def __init__(self, catalog, directory, *extra):
        self.token = secrets.token_hex(32)
        self.process = subprocess.Popen(
            [NATIVE, "--catalog", str(catalog), "--directory", str(directory), "--device", "cpu", "--port", "0", *extra],
            stdout=subprocess.PIPE,
            env={**os.environ, "MEDIA_ENGINE_TOKEN": self.token},
            text=True,
        )
        self.record = json.loads(self.process.stdout.readline())
        self.url = f"http://127.0.0.1:{self.record['port']}"

    def call(self, method, path, body=None, data=None, content_type="application/json"):
        if body is not None:
            data = json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method)
        request.add_header("Authorization", "Bearer " + self.token)
        if data is not None:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
                status = response.status
                kind = response.headers.get("Content-Type")
        except urllib.error.HTTPError as error:
            payload, status, kind = error.read(), error.code, error.headers.get("Content-Type")
        return status, (json.loads(payload) if kind == "application/json" else payload)

    def json(self, method, path, body=None, expect=200):
        status, value = self.call(method, path, body)
        assert status == expect, (method, path, status, value)
        return value

    def wait(self, path, key="state"):
        while True:
            value = self.json("GET", path)
            if value[key] != "running":
                return value
            time.sleep(0.02)

    def stop(self):
        self.process.terminate()
        self.process.wait(20)
        self.process.stdout.close()


@unittest.skipUnless(NATIVE, "MEDIA_ENGINE_NATIVE is not set")
class NativeConformanceTests(unittest.TestCase):
    def test_native_engine_is_conformant(self):
        from fixture import make_bundle
        from tests.conformance.run import run_suite

        if not HAVE_OWNED:
            self.skipTest("the tiny bundle needs torch/diffusers to build")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir, bundle = tiny_sdxl_bundle(root)
            upscaler = tiny_esrgan_bundle(root / "upscaler")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tiny": str(model_dir), "tiny-upscaler": str(root / "upscaler")}))
            registration = root / "registration.json"
            registration.write_text(json.dumps({"bundle": bundle, "upscaler": upscaler}))
            native = Native(catalog, root / "work", "--max-retained", "10")
            saved = dict(os.environ)
            try:
                self.assertEqual(native.record["backend_id"], "native-sdxl")
                os.environ.update(
                    MEDIA_ENGINE_URL=native.url,
                    MEDIA_ENGINE_TOKEN=native.token,
                    MEDIA_ENGINE_STARTUP_RECORD=json.dumps(native.record),
                    MEDIA_ENGINE_REGISTRATION=str(registration),
                )
                stream = io.StringIO()
                conformant, sections = run_suite(stream=stream)
                self.assertTrue(conformant, (sections, stream.getvalue()[-3000:]))
                self.assertNotIn("skip", sections.values())
            finally:
                os.environ.clear()
                os.environ.update(saved)
                native.stop()


@unittest.skipUnless(NATIVE and HAVE_OWNED, "MEDIA_ENGINE_NATIVE and the inference-owned extra are needed")
class NativeOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_dir, cls.bundle = tiny_sdxl_bundle(cls.root)
        cls.adapter = tiny_lora(cls.root / "lora", cls.model_dir)
        cls.catalog = {"tiny": str(cls.model_dir), "tiny-lora": str(cls.root / "lora")}
        path = cls.root / "catalog.json"
        path.write_text(json.dumps(cls.catalog))
        cls.native = Native(path, cls.root / "native")
        from media_engine.backend import backend_class
        from media_engine.server import Worker

        cls.owned = Worker(cls.root / "owned", backend_class("owned-sdxl")("cpu"), cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls.native.stop()
        cls.temp.cleanup()

    def options(self):
        return {"precision": "float32", "attention": "sdpa", "offload": "none"}

    def load_both(self, adapters=()):
        body = {"protocol": PROTOCOL, "bundle": self.bundle, "options": self.options(), "adapters": list(adapters)}
        op = self.native.json("POST", "/v1/model/load", body, expect=202)
        result = self.native.wait("/v1/operations/" + op["operation_id"])
        self.assertEqual(result["state"], "succeeded", result)
        op = self.owned.control("load", body)
        while self.owned.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.02)
        self.assertEqual(self.owned.operations[op["operation_id"]]["state"], "succeeded")

    def request(self, attempt, task="text-to-image", image=None, **params):
        inputs = {"prompt": "a test portrait", "negative_prompt": "window"}
        if params.pop("no_negative", False):
            del inputs["negative_prompt"]
        body = {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": task,
            "inputs": inputs,
            "parameters": {**DEFAULTS, "seed": 7, "width": 64, "height": 64, "steps": 3, "guidance": 6, **params},
        }
        if task == "image-to-image":
            body["inputs"]["image"] = image
            body["parameters"].setdefault("strength", 0.7)
        return body

    def run_native(self, body):
        snap = self.native.wait("/v1/executions/" + self.native.json("POST", "/v1/executions", body, expect=202)["execution_id"])
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        images = []
        for output in snap["outputs"]:
            status, data = self.native.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/{output['output_id']}")
            self.assertEqual(status, 200)
            images.append(data)
        return snap, images

    def run_owned(self, body):
        execution = self.owned.submit(body)["execution_id"]
        while self.owned.snapshot(execution)["state"] == "running":
            time.sleep(0.02)
        snap = self.owned.snapshot(execution)
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        folder = self.owned.directory / execution
        return snap, [(folder / (o["output_id"] + ".png")).read_bytes() for o in snap["outputs"]]

    def shapes(self, snap):
        """Event types, stages and progress fields: what a supervisor observes."""
        return [
            (e["type"], e["data"].get("stage"), tuple(sorted(e["data"])) if e["type"] == "progress" else None)
            for e in snap["events"]
            if e["type"] in ("stage", "progress", "state")
        ]

    def test_identity_and_requests_agree_with_the_owned_pipeline(self):
        info = self.native.json("GET", "/v1/info")
        self.assertEqual((info["backend_id"], info["ownership"]), ("native-sdxl", "engine"))
        self.load_both()
        cases = {
            "t2i": self.request("attempt_t2i"),
            "plain": self.request("attempt_plain", guidance=1),
            "no_negative": self.request("attempt_noneg", no_negative=True),
            "batch": self.request("attempt_batch", count=2),
            "seed8": self.request("attempt_seed8", seed=8),
        }
        status, record = self.native.call("POST", "/v1/inputs", data=make_png(64, 64), content_type="image/png")
        self.assertEqual(status, 201)
        owned_input = self.owned.store_input(make_png(64, 64))
        results = {}
        for name, body in cases.items():
            native_snap, native_images = self.run_native(body)
            owned_snap, owned_images = self.run_owned(body)
            self.assertEqual(len(native_images), len(owned_images))
            for a, b in zip(native_images, owned_images):
                self.assertLessEqual(difference(a, b), TOLERANCE, name)
            self.assertEqual(self.shapes(native_snap), self.shapes(owned_snap), name)
            self.assertEqual(native_snap["effective_settings"], owned_snap["effective_settings"], name)
            results[name] = native_images
        # Output 2 of a batch reproduces a count-1 run at seed + 1.
        self.assertLessEqual(difference(results["batch"][1], results["seed8"][0]), 1)
        native_snap, native_images = self.run_native(self.request("attempt_i2i", "image-to-image", record["input_id"]))
        owned_snap, owned_images = self.run_owned(self.request("attempt_i2i", "image-to-image", owned_input["input_id"]))
        self.assertLessEqual(difference(native_images[0], owned_images[0]), TOLERANCE)
        self.assertEqual(native_snap["progress"]["total_steps"], owned_snap["progress"]["total_steps"])
        self.assertEqual(self.shapes(native_snap), self.shapes(owned_snap))
        # Cancellation: requested at once, terminal as cancelled with nothing published.
        execution = self.native.json("POST", "/v1/executions", self.request("attempt_cancel", steps=50), expect=202)["execution_id"]
        self.native.call("POST", f"/v1/executions/{execution}/cancel")
        snap = self.native.wait("/v1/executions/" + execution)
        self.assertEqual((snap["state"], snap["outputs"], snap["cancel_requested"]), ("cancelled", [], True))
        self.native.json("DELETE", "/v1/executions/" + execution)

        # LoRA adapters change the output, and identically on both engines.
        self.load_both([{"adapter": self.adapter, "scale": 0.8}])
        native_snap, adapted = self.run_native(self.request("attempt_lora"))
        _owned_snap, owned_adapted = self.run_owned(self.request("attempt_lora"))
        self.assertGreater(difference(adapted[0], results["t2i"][0]), 1)
        self.assertLessEqual(difference(adapted[0], owned_adapted[0]), TOLERANCE)
        self.assertEqual(self.native.json("GET", "/v1/info")["loaded_bundle"]["adapters"][0]["adapter_id"], "tiny-lora")


if __name__ == "__main__":
    unittest.main()
