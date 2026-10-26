import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from fixture import TestBackend, make_bundle
from media_engine.server import Worker, create_server
from media_engine.contracts import PROTOCOL, WorkerError, DEFAULTS, bundle_path


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.backend = TestBackend()
        self.worker = Worker(
            self.root / "outputs", self.backend, {"fixture": str(self.root / "model")}
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
