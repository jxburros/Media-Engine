import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from fixture import TestBackend, make_adapter, make_bundle, make_png
from media_engine.server import Worker, create_server
from media_engine.contracts import PROTOCOL, WorkerError, DEFAULTS, bundle_path


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.backend = TestBackend()
        self.adapter = make_adapter(self.root / "lora")
        self.worker = Worker(
            self.root / "outputs",
            self.backend,
            {"fixture": str(self.root / "model"), "fixture-lora": str(self.root / "lora")},
        )

    def tearDown(self):
        while self.worker.active or self.worker.state in {"loading", "unloading"}:
            time.sleep(0.01)
        self.temp.cleanup()

    def load(self):
        op = self.worker.control(
            "load",
            {
                "protocol": PROTOCOL,
                "bundle": self.bundle,
                "options": self.backend.capabilities()["load_defaults"],
            },
        )
        while self.worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.01)
        self.assertEqual(
            self.worker.operations[op["operation_id"]]["state"], "succeeded"
        )

    def request(self, attempt="attempt_a", steps=3):
        return {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": "text-to-image",
            "inputs": {"prompt": "test"},
            "parameters": {
                **DEFAULTS,
                "seed": 42,
                "width": 64,
                "height": 64,
                "steps": steps,
            },
        }

    def wait(self, execution):
        while self.worker.snapshot(execution)["state"] == "running":
            time.sleep(0.01)
        return self.worker.snapshot(execution)

    def test_single_flight_idempotency_and_warm_reuse(self):
        self.load()
        first = self.worker.submit(self.request())
        self.assertEqual(
            first["execution_id"], self.worker.submit(self.request())["execution_id"]
        )
        with self.assertRaises(WorkerError) as caught:
            self.worker.submit(self.request("attempt_b"))
        self.assertEqual(caught.exception.code, "busy")
        altered = self.request()
        altered["parameters"]["seed"] = 43
        with self.assertRaises(WorkerError):
            self.worker.submit(altered)
        snap = self.wait(first["execution_id"])
        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(snap["outputs"][0]["width"], 64)
        self.assertEqual(
            [e["sequence"] for e in snap["events"]],
            list(range(1, snap["last_sequence"] + 1)),
        )
        second = self.worker.submit(self.request("attempt_b"))
        self.wait(second["execution_id"])
        self.assertEqual(self.backend.loads, 1)
        self.assertEqual(self.backend.generations, 2)
        self.worker.release(first["execution_id"])
        with self.assertRaises(WorkerError):
            self.worker.snapshot(first["execution_id"])

    def test_cancellation_and_private_errors(self):
        self.load()
        run = self.worker.submit(self.request(steps=50))
        self.worker.cancel(run["execution_id"])
        self.assertEqual(self.wait(run["execution_id"])["state"], "cancelled")
        req = self.request("attempt_failure")
        req["inputs"]["prompt"] = "fail"
        failed = self.wait(self.worker.submit(req)["execution_id"])
        self.assertEqual(failed["error"]["code"], "execution_failed")
        self.assertNotIn("secret", json.dumps(failed["error"]))

    def test_catalog_identity_and_unknown_options(self):
        bundle_path(self.bundle, self.worker.catalog)
        (self.root / "model/unet/config.json").write_text('{"changed":true}')
        with self.assertRaises(WorkerError) as caught:
            bundle_path(self.bundle, self.worker.catalog)
        self.assertEqual(caught.exception.code, "artifact_changed")
        with self.assertRaises(WorkerError):
            self.worker.submit({**self.request(), "extra": True})

    def test_readiness_storage_and_http_event_resume(self):
        self.load()
        token = "x" * 40
        server = create_server(self.worker, token, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def get(path, headers=None):
            return urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}" + path,
                    headers=headers or {},
                )
            )

        try:
            self.assertEqual(get("/health").status, 200)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                get("/v1/info")
            self.assertEqual(caught.exception.code, 401)
            run = self.worker.submit(self.request())
            self.wait(run["execution_id"])
            headers = {"Authorization": "Bearer " + token, "Last-Event-ID": "2"}
            # image-to-image through the opaque input handle; 1.0 callers cannot use it.
            upload = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/inputs",
                data=make_png(64, 64),
                method="POST",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "image/png",
                },
            )
            record = json.load(urllib.request.urlopen(upload))
            self.assertEqual((record["width"], record["height"]), (64, 64))
            img2img = self.request("attempt_img2img", steps=4)
            img2img.update(task="image-to-image")
            img2img["inputs"]["image"] = record["input_id"]
            img2img["parameters"]["strength"] = 0.5
            with self.assertRaises(WorkerError) as caught:
                self.worker.submit({**img2img, "protocol": "media-worker/1.0"})
            self.assertEqual(caught.exception.field, "task")
            wrong_size = json.loads(json.dumps(img2img))
            wrong_size["parameters"]["width"] = 128
            with self.assertRaises(WorkerError) as caught:
                self.worker.submit(wrong_size)
            self.assertEqual(caught.exception.field, "width")
            edited = self.wait(self.worker.submit(img2img)["execution_id"])
            self.assertEqual(edited["state"], "succeeded")
            self.assertEqual(edited["task"], "image-to-image")
            self.assertEqual(edited["effective_settings"]["effective_steps"], 2)
            self.assertEqual(edited["progress"]["total_steps"], 2)
            self.assertNotIn("image_path", edited)
            self.worker.release_input(record["input_id"])
            with self.assertRaises(WorkerError):
                self.worker.input_record(record["input_id"])
            estimate = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/v1/validate",
                        data=json.dumps(
                            {
                                "protocol": "media-worker/1.0",
                                "bundle": self.bundle,
                                "options": self.backend.capabilities()["load_defaults"],
                            }
                        ).encode(),
                        method="POST",
                        headers={"Authorization": "Bearer " + token},
                    )
                )
            )["memory_estimate"]
            self.assertEqual(estimate["source"], "safetensors_headers")
            self.assertEqual(estimate["roles"]["unet"]["parameters"], 8)
            self.assertIn("activations", estimate["uncertainty"])
            events = (
                get("/v1/executions/" + run["execution_id"] + "/events", headers)
                .read()
                .decode()
            )
            self.assertIn("id: 3\n", events)
            self.assertNotIn("id: 2\n", events)
            self.worker.max_retained = 1
            with self.assertRaises(WorkerError) as caught:
                self.worker.submit(self.request("attempt_full"))
            self.assertEqual(caught.exception.code, "storage_full")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()


class AdapterTests(WorkerTests):
    def test_adapters_are_load_identity(self):
        body = {
            "protocol": PROTOCOL,
            "bundle": self.bundle,
            "options": self.backend.capabilities()["load_defaults"],
            "adapters": [{"adapter": self.adapter, "scale": 0.8}],
        }
        with self.assertRaises(WorkerError) as caught:
            self.worker.validate_load({**body, "protocol": "media-worker/1.0"})
        self.assertEqual(caught.exception.field, "adapters")
        bad_scale = json.loads(json.dumps(body))
        bad_scale["adapters"][0]["scale"] = 3
        with self.assertRaises(WorkerError) as caught:
            self.worker.validate_load(bad_scale)
        self.assertEqual(caught.exception.field, "scale")
        tampered = json.loads(json.dumps(body))
        tampered["adapters"][0]["adapter"]["source"] = {"license": "other"}
        with self.assertRaises(WorkerError) as caught:
            self.worker.validate_load(tampered)
        self.assertEqual(caught.exception.code, "artifact_changed")
        op = self.worker.control("load", body)
        while self.worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.01)
        record = self.worker.operations[op["operation_id"]]
        self.assertEqual(record["state"], "succeeded", record["error"])
        self.assertEqual(self.backend.adapters, [("fixture-lora", 0.8)])
        loaded = self.worker.info()["loaded_bundle"]["adapters"][0]
        self.assertEqual(loaded["adapter_id"], "fixture-lora")
        self.assertEqual(loaded["layout"]["targets"], ["text_encoder", "unet"])
        self.assertEqual(loaded["layout"]["ranks"], [2])
        self.assertEqual(record["memory_estimate"]["adapters"][0]["parameters"], 32)
        # Generation is unchanged by adapters: they travel with the loaded identity.
        snap = self.wait(self.worker.submit(self.request())["execution_id"])
        self.assertEqual(snap["state"], "succeeded")

    def test_unsupported_layouts_are_refused(self):
        import struct

        kohya = self.root / "kohya"
        kohya.mkdir()
        header = json.dumps(
            {"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]}}
        ).encode()
        (kohya / "adapter.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(32))
        import hashlib
        from media_engine.contracts import digest

        files = {"adapter.safetensors": hashlib.sha256((kohya / "adapter.safetensors").read_bytes()).hexdigest()}
        record = {"schema_version": 1, "adapter_id": "kohya", "artifact_handle": "kohya", "files": files, "source": {}}
        self.worker.catalog["kohya"] = str(kohya)
        with self.assertRaises(WorkerError) as caught:
            self.worker.validate_load(
                {
                    "protocol": PROTOCOL,
                    "bundle": self.bundle,
                    "options": self.backend.capabilities()["load_defaults"],
                    "adapters": [{"adapter": {**record, "digest": digest(record)}, "scale": 1}],
                }
            )
        self.assertEqual((caught.exception.code, caught.exception.field), ("unsupported", "adapter"))

    # Inherited protocol tests run again with an adapter catalog present; keep them once.
    for name in list(vars(WorkerTests)):
        if name.startswith("test_"):
            locals()[name] = None
