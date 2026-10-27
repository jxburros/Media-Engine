import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from fixture import (
    TestBackend,
    describe_bundle,
    make_adapter,
    make_bundle,
    make_checkpoint,
    make_converted,
    make_esrgan_bundle,
    make_png,
)
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


class ProtocolTwelveTests(unittest.TestCase):
    """media-worker/1.2: multi-output, families, upscale, auto/fallbacks, device identity."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.upscaler = make_esrgan_bundle(self.root / "upscaler")
        self.catalog = {
            "fixture": str(self.root / "model"),
            "upscaler": str(self.root / "upscaler"),
        }
        self.backend = TestBackend()
        self.worker = Worker(self.root / "work", self.backend, self.catalog)

    def tearDown(self):
        while self.worker.active or self.worker.state in {"loading", "unloading"}:
            time.sleep(0.01)
        self.temp.cleanup()

    def load(self, bundle=None, protocol=PROTOCOL, expect="succeeded", **extra):
        body = {
            "protocol": protocol,
            "bundle": bundle or self.bundle,
            "options": self.backend.capabilities()["load_defaults"],
            **extra,
        }
        op = self.worker.control("load", body)
        while self.worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.01)
        record = self.worker.operations[op["operation_id"]]
        self.assertEqual(record["state"], expect, record["error"])
        return record

    def request(self, attempt="attempt_a", protocol=PROTOCOL, steps=2, **params):
        return {
            "protocol": protocol,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": "text-to-image",
            "inputs": {"prompt": "test"},
            "parameters": {**DEFAULTS, "seed": 42, "width": 64, "height": 64, "steps": steps, **params},
        }

    def upscale_request(self, input_id, attempt="attempt_up", **params):
        return {
            "protocol": PROTOCOL,
            "attempt_id": attempt,
            "bundle_digest": self.upscaler["bundle_digest"],
            "task": "upscale",
            "inputs": {"image": input_id},
            "parameters": {"width": 64, "height": 64, **params},
        }

    def wait(self, execution):
        while self.worker.snapshot(execution)["state"] == "running":
            time.sleep(0.01)
        return self.worker.snapshot(execution)

    def code(self, call, *args):
        with self.assertRaises(WorkerError) as caught:
            call(*args)
        return caught.exception.code, caught.exception.field

    def test_count_produces_distinct_reproducible_outputs(self):
        self.load()
        with self.assertRaises(WorkerError) as caught:
            self.worker.submit(self.request(protocol="media-worker/1.1", count=2))
        self.assertEqual((caught.exception.code, caught.exception.field), ("unsupported", "count"))
        self.assertEqual(self.code(self.worker.submit, self.request(count=5)), ("invalid_request", "count"))
        single = self.wait(self.worker.submit(self.request("attempt_one", protocol="media-worker/1.1"))["execution_id"])
        self.assertEqual([o["output_id"] for o in single["outputs"]], ["image"])
        self.assertNotIn("count", single["effective_settings"])
        self.assertNotIn("applied_fallbacks", single)
        batch = self.wait(self.worker.submit(self.request(count=3))["execution_id"])
        self.assertEqual(batch["state"], "succeeded")
        self.assertEqual([o["output_id"] for o in batch["outputs"]], ["image", "image_2", "image_3"])
        self.assertEqual(len({o["sha256"] for o in batch["outputs"]}), 3)
        self.assertEqual(len([e for e in batch["events"] if e["type"] == "output"]), 3)
        self.assertEqual(batch["effective_settings"]["count"], 3)
        self.assertEqual(batch["applied_fallbacks"], [])
        alone = self.wait(self.worker.submit(self.request("attempt_alone", seed=43))["execution_id"])
        self.assertEqual(alone["outputs"][0]["sha256"], batch["outputs"][1]["sha256"])
        self.assertEqual(alone["effective_settings"]["count"], 1)
        files = sorted(p.name for p in (self.worker.directory / batch["execution_id"]).iterdir())
        self.assertEqual(files, ["image.png", "image_2.png", "image_3.png"])
        self.assertGreater(self.worker.storage_used(), 3 * batch["outputs"][0]["size"] - 1)
        self.worker.release(batch["execution_id"])
        self.assertFalse((self.worker.directory / batch["execution_id"]).exists())
        validated = self.worker.validate_execution(self.request(count=2))
        self.assertEqual(validated["outputs"], ["image", "image_2"])
        self.assertNotIn("outputs", self.worker.validate_execution(self.request(protocol="media-worker/1.1")))

    def test_partial_batch_fails_naming_missing_output(self):
        self.load()
        body = self.request(count=2)
        body["inputs"]["prompt"] = "partial batch"
        snap = self.wait(self.worker.submit(body)["execution_id"])
        self.assertEqual(snap["state"], "failed")
        self.assertEqual((snap["error"]["code"], snap["error"]["field"]), ("execution_failed", "image_2"))
        self.assertEqual(snap["outputs"], [])
        self.assertFalse((self.worker.directory / snap["execution_id"]).exists())

    def test_family_table_and_task_routing(self):
        empty = self.root / "no-weights"
        empty.mkdir()
        self.catalog["empty"] = str(empty)
        unknown = describe_bundle(self.root / "model", "empty", "x")
        unknown = {k: v for k, v in unknown.items() if k != "bundle_digest"}
        unknown["family"] = "flux-dev"
        from media_engine.contracts import digest

        unknown["bundle_digest"] = digest(unknown)
        self.assertEqual(self.code(bundle_path, unknown, self.catalog), ("unsupported", "family"))
        mismatched = dict(self.upscaler, format="diffusers-safetensors")
        self.assertEqual(self.code(bundle_path, mismatched, self.catalog), ("unsupported", "family"))
        # A 1.1 caller cannot name a 1.2 family or the format field.
        self.assertEqual(
            self.code(bundle_path, self.upscaler, self.catalog, True, "media-worker/1.1"),
            ("invalid_request", None),
        )
        self.assertEqual(bundle_path(self.upscaler, self.catalog), (self.root / "upscaler").resolve())
        caps = self.worker.capabilities()
        self.assertIn("upscale", caps["tasks"])
        self.assertEqual(
            {l["families"][0]: l["format"] for l in caps["loaders"]},
            {"sdxl-base": "diffusers-safetensors", "esrgan": "single-file-safetensors"},
        )
        self.assertEqual(caps["task_parameters"]["text-to-image"]["outputs"], ["image"])
        self.assertEqual(caps["limits"]["count"], [1, 4])
        self.load()
        self.assertEqual(self.worker.info()["loaded_bundle"]["tasks"], ["text-to-image", "image-to-image"])
        record = self.worker.store_input(make_png(64, 64))
        wrong = self.upscale_request(record["input_id"])
        wrong["bundle_digest"] = self.bundle["bundle_digest"]
        self.assertEqual(self.code(self.worker.submit, wrong), ("unsupported", "task"))
        self.assertEqual(self.code(self.worker.submit, self.upscale_request(record["input_id"])), ("model_not_ready", None))
        old = dict(wrong, protocol="media-worker/1.1")
        self.assertEqual(self.code(self.worker.submit, old), ("unsupported", "task"))
        # Adapters belong to the sdxl-base family only.
        adapter = make_adapter(self.root / "lora")
        self.catalog["fixture-lora"] = str(self.root / "lora")
        body = {
            "protocol": PROTOCOL,
            "bundle": self.upscaler,
            "options": caps["load_defaults"],
            "adapters": [{"adapter": adapter, "scale": 1}],
        }
        self.assertEqual(self.code(self.worker.validate_load, body), ("unsupported", "adapters"))

    def test_esrgan_bundle_shape_is_checked_from_headers(self):
        config = json.loads((self.root / "upscaler/config.json").read_text())
        (self.root / "bad").mkdir()
        bad = make_esrgan_bundle(self.root / "bad", handle="bad", num_block=2)
        (self.root / "bad/config.json").write_text(json.dumps(config))
        bad = describe_bundle(self.root / "bad", "bad", "bad", "esrgan")
        self.catalog["bad"] = str(self.root / "bad")
        self.assertEqual(self.code(bundle_path, bad, self.catalog), ("unsupported", "network"))
        (self.root / "bad/config.json").write_text(json.dumps({**config, "scale": 3}))
        bad = describe_bundle(self.root / "bad", "bad", "bad", "esrgan")
        self.assertEqual(self.code(bundle_path, bad, self.catalog), ("unsupported", "configuration"))

    def test_upscale_validation_progress_and_cancellation(self):
        self.load(self.upscaler)
        loaded = self.worker.info()["loaded_bundle"]
        self.assertEqual((loaded["family"], loaded["scale"], loaded["tasks"]), ("esrgan", 2, ["upscale"]))
        record = self.worker.store_input(make_png(64, 64))
        body = self.upscale_request(record["input_id"])
        prompt = json.loads(json.dumps(body))
        prompt["inputs"]["prompt"] = "x"
        self.assertEqual(self.code(self.worker.submit, prompt), ("invalid_request", "prompt"))
        seeded = json.loads(json.dumps(body))
        seeded["parameters"]["seed"] = 1
        self.assertEqual(self.code(self.worker.submit, seeded)[0], "invalid_request")
        self.assertEqual(self.code(self.worker.submit, self.upscale_request(record["input_id"], tile=300)), ("invalid_request", "tile"))
        self.assertEqual(self.code(self.worker.submit, self.upscale_request(record["input_id"], tile_overlap=65)), ("invalid_request", "tile_overlap"))
        self.assertEqual(self.code(self.worker.submit, dict(body, protocol="media-worker/1.1")), ("unsupported", "task"))
        self.assertEqual(self.code(self.worker.submit, self.upscale_request(record["input_id"], width=128)), ("invalid_request", "width"))
        sdxl = self.request()
        sdxl["bundle_digest"] = self.upscaler["bundle_digest"]
        self.assertEqual(self.code(self.worker.submit, sdxl), ("unsupported", "task"))
        validated = self.worker.validate_execution(body)
        self.assertEqual(validated["effective_settings"]["output_width"], 128)
        self.assertNotIn("effective_steps", validated)
        snap = self.wait(self.worker.submit(body)["execution_id"])
        self.assertEqual(snap["state"], "succeeded", snap["error"])
        self.assertEqual((snap["outputs"][0]["width"], snap["outputs"][0]["height"]), (128, 128))
        self.assertEqual(snap["progress"], {"stage": "upscaling", "scope": "stage", "unit": "tiles", "completed_steps": 1, "total_steps": 1})
        self.assertEqual([e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"], ["upscaling", "saving"])
        self.assertEqual(snap["effective_settings"], {"width": 64, "height": 64, "tile": 0, "tile_overlap": 16, "scale": 2, "output_width": 128, "output_height": 128})
        big = self.worker.store_input(make_png(2048, 2048))
        many = self.upscale_request(big["input_id"], "attempt_tiles", width=2048, height=2048, tile=256)
        run = self.worker.submit(many)
        while not self.worker.snapshot(run["execution_id"])["progress"]:
            time.sleep(0.01)
        self.worker.cancel(run["execution_id"])
        cancelled = self.wait(run["execution_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertLess(cancelled["progress"]["completed_steps"], 64)
        self.assertEqual(cancelled["progress"]["total_steps"], 64)
        # The result size is bounded before any computation; nothing is shrunk to fit.
        self.backend.output_side = 2048
        self.assertEqual(self.code(self.worker.submit, dict(many, attempt_id="attempt_big")), ("invalid_request", "output_side"))
        self.assertEqual(self.code(self.worker.validate_execution, many), ("invalid_request", "output_side"))

    def test_auto_precision_and_authorized_fallbacks(self):
        options = {"precision": "auto", "attention": "sdpa", "offload": "none"}
        body = {"protocol": PROTOCOL, "bundle": self.bundle, "options": options}
        self.assertEqual(
            self.code(self.worker.validate_load, dict(body, protocol="media-worker/1.1")),
            ("unsupported", "options"),
        )
        self.assertEqual(
            self.code(self.worker.validate_load, {**body, "protocol": "media-worker/1.1", "options": self.backend.capabilities()["load_defaults"], "authorized_fallbacks": ["cpu"]}),
            ("unsupported", "authorized_fallbacks"),
        )
        self.assertEqual(self.code(self.worker.validate_load, dict(body, authorized_fallbacks=["gpu"])), ("invalid_request", "authorized_fallbacks"))
        first, second = self.worker.validation(body), self.worker.validation(body)
        self.assertEqual(first["resolved_options"]["precision"], "float32")
        self.assertEqual(first["resolutions"][0]["requested"], "auto")
        self.assertEqual(first["resolutions"], second["resolutions"])
        op = self.load(options=options)
        self.assertEqual(op["resolved_options"]["precision"], "float32")
        self.assertEqual(self.worker.info()["loaded_bundle"]["options"], options)
        # Unauthorized: an unavailable device is refused exactly as in 0.3.0.
        self.backend.device_unavailable = True
        self.assertEqual(self.code(self.worker.control, "load", dict(body, options=self.backend.capabilities()["load_defaults"])), ("backend_unavailable", None))
        authorized = dict(body, authorized_fallbacks=["cpu"])
        predicted = self.worker.validation(authorized)
        self.assertEqual(predicted["applied_fallbacks"][0]["fallback"], "cpu")
        op = self.load(options=options, authorized_fallbacks=["cpu"])
        self.assertEqual(op["applied_fallbacks"], predicted["applied_fallbacks"])
        self.assertEqual(self.backend.run_device, "cpu")
        self.assertTrue(self.worker.capabilities()["ready"])
        self.assertEqual(self.worker.info()["loaded_bundle"]["applied_fallbacks"][0]["device"], "cpu")
        snap = self.wait(self.worker.submit(self.request(protocol="media-worker/1.1"))["execution_id"])
        self.assertEqual(snap["applied_fallbacks"][0]["fallback"], "cpu")
        self.assertEqual(snap["effective_settings"]["seed"], 42)

    def test_lower_precision_fallback_follows_the_budget(self):
        self.backend = TestBackend(precisions=("float16", "float32"), memory_budget_mb=100, estimate_mb=150)
        self.worker = Worker(self.root / "work2", self.backend, self.catalog)
        fp32 = {"precision": "float32", "attention": "sdpa", "offload": "none"}
        plain = self.load(options=fp32)
        self.assertEqual(plain["applied_fallbacks"], [])  # unauthorized: 0.3.0 behavior
        lower = self.load(options=fp32, authorized_fallbacks=["lower_precision", "cpu"])
        self.assertEqual([f["fallback"] for f in lower["applied_fallbacks"]], ["lower_precision"])
        self.assertEqual(lower["resolved_options"]["precision"], "float16")
        self.assertEqual(lower["memory_estimate"]["value_mb"], 75)
        auto = self.worker.validation({"protocol": PROTOCOL, "bundle": self.bundle, "options": dict(fp32, precision="auto")})
        self.assertEqual(auto["resolved_options"]["precision"], "float16")
        self.assertIn("fits", auto["resolutions"][0]["reason"])
        self.backend.estimate_mb = 500
        cpu = self.worker.validation({"protocol": PROTOCOL, "bundle": self.bundle, "options": fp32, "authorized_fallbacks": ["lower_precision", "cpu"]})
        self.assertEqual([f["fallback"] for f in cpu["applied_fallbacks"]], ["cpu"])
        self.assertEqual(cpu["resolved_options"]["precision"], "float32")

    def test_device_identity_and_pinned_protocol(self):
        identity = self.worker.info()["device_identity"]
        self.assertEqual(set(identity), {"device", "index", "name", "total_memory_mb", "host_id", "host_id_source", "pid"})
        self.assertEqual(identity, self.worker.info()["device_identity"])
        other = Worker(self.root / "other", TestBackend(device_index=1, host_id="second"), self.catalog)
        self.assertNotEqual(other.info()["device_identity"]["host_id"], identity["host_id"])
        from media_engine.backend import TorchBackend

        real = TorchBackend("cpu").device_identity()
        self.assertEqual((real["device"], real["name"], real["index"]), ("cpu", "cpu", None))
        self.assertRegex(real["host_id"], "^[a-f0-9]{64}$")
        pinned_protocols = ("media-worker/1.0", "media-worker/1.1")
        pinned = Worker(self.root / "pinned", TestBackend(protocols=pinned_protocols), self.catalog, protocols=pinned_protocols)
        info = pinned.info()
        self.assertEqual((info["protocol"], info["supported_protocols"]), ("media-worker/1.1", list(pinned_protocols)))
        self.assertNotIn("device_identity", info)
        caps = pinned.capabilities()
        self.assertEqual(caps["tasks"], ["text-to-image", "image-to-image"])
        self.assertNotIn("loaders", caps)
        self.assertEqual(self.code(pinned.submit, self.request()), ("protocol_mismatch", "protocol"))

    def test_converted_bundle_and_bare_checkpoint(self):
        converted = make_converted(self.root / "converted", reference=self.root / "model")
        checkpoint = make_checkpoint(self.root / "checkpoint")
        self.catalog.update(converted=str(self.root / "converted"), checkpoint=str(checkpoint.parent))
        self.assertIn("conversion.json", converted["configuration"]["files"])
        bundle_path(converted, self.catalog)
        self.load(converted)
        self.assertEqual(self.worker.info()["loaded_bundle"]["bundle_digest"], converted["bundle_digest"])
        conversion = converted["source"]["conversion"]
        self.assertEqual(set(conversion), {"source_sha256", "reference_bundle_digest", "converter", "mapped_keys"})
        # 1.0/1.1 bodies keep rejecting source fields they do not know.
        self.assertEqual(self.code(bundle_path, converted, self.catalog, True, "media-worker/1.1"), ("invalid_request", None))
        from media_engine.contracts import digest

        def resign(bundle, **source):
            body = {k: v for k, v in bundle.items() if k != "bundle_digest"}
            body["source"] = {**bundle["source"], **source}
            return {**body, "bundle_digest": digest(body)}

        lying = resign(converted, conversion={**conversion, "source_sha256": "a" * 64})
        self.assertEqual(self.code(bundle_path, lying, self.catalog), ("artifact_changed", "conversion"))
        malformed = resign(converted, conversion={**conversion, "extra": 1})
        self.assertEqual(self.code(bundle_path, malformed, self.catalog), ("invalid_request", None))
        bad_digest = resign(converted, conversion={**conversion, "reference_bundle_digest": "nope"})
        self.assertEqual(self.code(bundle_path, bad_digest, self.catalog), ("invalid_request", "conversion"))
        plain = make_converted(self.root / "converted2", "converted2", reference=self.root / "model", provenance_in_source=False)
        self.catalog["converted2"] = str(self.root / "converted2")
        bundle_path(plain, self.catalog, True, "media-worker/1.1")
        not_converted = resign(self.bundle, conversion=conversion)
        self.assertEqual(self.code(bundle_path, not_converted, self.catalog), ("artifact_changed", "conversion"))
        bare = describe_bundle(checkpoint.parent, "checkpoint", "bare")
        self.assertEqual(self.code(bundle_path, bare, self.catalog)[0], "artifact_changed")

    def test_http_output_routes(self):
        self.load()
        token = "y" * 40
        server = create_server(self.worker, token, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        headers = {"Authorization": "Bearer " + token}

        def get(path):
            return urllib.request.urlopen(urllib.request.Request(base + path, headers=headers))

        try:
            snap = self.wait(self.worker.submit(self.request(count=2))["execution_id"])
            import hashlib

            for output in snap["outputs"]:
                response = get(f"/v1/executions/{snap['execution_id']}/outputs/{output['output_id']}")
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertEqual(hashlib.sha256(response.read()).hexdigest(), output["sha256"])
            with self.assertRaises(urllib.error.HTTPError) as caught:
                get(f"/v1/executions/{snap['execution_id']}/outputs/image_3")
            self.assertEqual(caught.exception.code, 404)
            self.assertEqual(json.load(caught.exception)["error"]["code"], "output_not_found")
            info = json.load(get("/v1/info"))
            self.assertEqual(info["protocol"], "media-worker/1.2")
            self.assertIn("device_identity", info)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
