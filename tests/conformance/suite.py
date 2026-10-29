"""The media-worker conformance suite: a `unittest` suite, one TestCase class per section.

Configuration comes from the environment so the same suite measures any backend:

  MEDIA_ENGINE_URL             worker base URL, e.g. http://127.0.0.1:7811
  MEDIA_ENGINE_TOKEN           the per-launch bearer credential
  MEDIA_ENGINE_REGISTRATION    path to a JSON document {"bundle": <sdxl-base registration>,
                               "upscaler": <esrgan registration, needed when upscale is
                               advertised>, "video": <svd registration, needed when
                               image-to-video is advertised>, "options": <optional load options>}
  MEDIA_ENGINE_STARTUP_RECORD  the worker's startup record (JSON text) when the suite launched it
  MEDIA_ENGINE_TLS_CA          for an https:// URL (network mode): PEM CA that issued the
                               worker's certificate (default: the system trust store)
  MEDIA_ENGINE_TLS_CERT        for an https:// URL: PEM client certificate presented to the
  MEDIA_ENGINE_TLS_KEY         worker, and its private key (mutual TLS)

Transport settings change how the suite connects, never what it asserts: a worker passes the same
sections over mutual TLS as over loopback HTTP.

A section that cannot run for lack of configuration is skipped, and a skipped section means the
worker is not reported conformant (see run.py). A section about a minor or task the worker does
not advertise passes as not applicable: conformance is to what is advertised.
"""

import hashlib
import json
import os
import re
import socket
import ssl
import struct
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

SUITE_VERSION = "media-worker-conformance/1.1"
SECRET = "conformance-secret-prompt-7f3a"
TERMINAL = {"succeeded", "failed", "cancelled"}


def make_png(width, height, color=(0x20, 0x90, 0x50)):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = (b"\0" + bytes(color) * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def tls_context():
    """The client side of network mode: verify the worker and present a client certificate."""
    context = ssl.create_default_context(cafile=os.environ.get("MEDIA_ENGINE_TLS_CA") or None)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    cert = os.environ.get("MEDIA_ENGINE_TLS_CERT")
    if cert:
        context.load_cert_chain(cert, os.environ.get("MEDIA_ENGINE_TLS_KEY") or None)
    return context


class Client:
    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token
        self.errors = []  # every error body seen, for the privacy section
        self.tls = tls_context() if urlsplit(self.url).scheme == "https" else None

    def call(self, method, path, body=None, data=None, headers=None, auth=True):
        headers = dict(headers or {})
        if auth:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60, context=self.tls) as response:
                payload = response.read()
                status, response_headers = response.status, response.headers
        except urllib.error.HTTPError as exc:
            payload, status, response_headers = exc.read(), exc.code, exc.headers
        if response_headers.get("Content-Type", "").startswith("application/json"):
            value = json.loads(payload)
            if status >= 400:
                self.errors.append(payload.decode())
            return status, value, response_headers
        return status, payload, response_headers

    def json(self, method, path, body=None, expect=200):
        status, value, _ = self.call(method, path, body)
        if status != expect:
            raise AssertionError(f"{method} {path}: expected {expect}, got {status}: {value}")
        return value

    def events(self, execution_id, last_event_id=None, on_event=None):
        """Read the SSE stream to its end; returns (status, [event dicts])."""
        parts = urlsplit(self.url)
        connection = socket.create_connection((parts.hostname, parts.port), timeout=60)
        if self.tls:
            connection = self.tls.wrap_socket(connection, server_hostname=parts.hostname)
        headers = f"Authorization: Bearer {self.token}\r\nHost: {parts.hostname}\r\n"
        if last_event_id is not None:
            headers += f"Last-Event-ID: {last_event_id}\r\n"
        connection.sendall(
            f"GET /v1/executions/{execution_id}/events HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode()
        )
        stream = connection.makefile("rb")
        status = int(stream.readline().split()[1])
        while stream.readline() not in (b"\r\n", b""):
            pass
        events, block = [], []
        if status != 200:
            connection.close()
            return status, events
        for line in stream:
            line = line.decode().rstrip("\n")
            if line:
                block.append(line)
                continue
            data = [l[6:] for l in block if l.startswith("data: ")]
            block = []
            if data:
                event = json.loads(data[0])
                events.append(event)
                if on_event:
                    on_event(event)
        connection.close()
        return status, events


def configuration():
    url, token = os.environ.get("MEDIA_ENGINE_URL"), os.environ.get("MEDIA_ENGINE_TOKEN")
    path = os.environ.get("MEDIA_ENGINE_REGISTRATION")
    registration = json.loads(Path(path).read_text()) if path else None
    startup = os.environ.get("MEDIA_ENGINE_STARTUP_RECORD")
    return url, token, registration, json.loads(startup) if startup else None


class Section(unittest.TestCase):
    section = "base"

    @classmethod
    def setUpClass(cls):
        cls.url, cls.token, cls.registration, cls.startup = configuration()
        if not cls.url or not cls.token:
            raise unittest.SkipTest("MEDIA_ENGINE_URL and MEDIA_ENGINE_TOKEN are required")
        cls.client = Client(cls.url, cls.token)
        cls.caps = cls.client.json("GET", "/v1/capabilities")
        cls.info = cls.client.json("GET", "/v1/info")
        cls.protocols = cls.caps.get("supported_protocols") or [cls.caps["protocol"]]
        cls.version = cls.protocols[-1]

    def minor(self, version=None):
        return int((version or self.version).rsplit(".", 1)[1])

    @property
    def bundle(self):
        if not self.registration or "bundle" not in self.registration:
            self.skipTest("MEDIA_ENGINE_REGISTRATION with a bundle is required")
        return self.registration["bundle"]

    def options(self):
        return (self.registration or {}).get("options") or self.caps["load_defaults"]

    def operation(self, action, body, expect="succeeded"):
        accepted = self.client.json("POST", f"/v1/model/{action}", body, expect=202)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            record = self.client.json("GET", "/v1/operations/" + accepted["operation_id"])
            if record["state"] != "running":
                self.assertEqual(record["state"], expect, record.get("error"))
                self.client.call("DELETE", "/v1/operations/" + accepted["operation_id"])
                return record
            time.sleep(0.05)
        self.fail("operation did not finish")

    def ensure_loaded(self, bundle=None):
        bundle = bundle or self.bundle
        info = self.client.json("GET", "/v1/info")
        loaded = info.get("loaded_bundle") or {}
        if loaded.get("bundle_digest") != bundle["bundle_digest"] or loaded.get("adapters"):
            self.operation(
                "load", {"protocol": self.version, "bundle": bundle, "options": self.options()}
            )

    def request(self, attempt, steps=2, version=None, **params):
        return {
            "protocol": version or self.version,
            "attempt_id": attempt,
            "bundle_digest": self.bundle["bundle_digest"],
            "task": "text-to-image",
            "inputs": {"prompt": "a conformance test"},
            "parameters": {
                **self.caps["defaults"],
                "seed": 11,
                "width": 64,
                "height": 64,
                "steps": steps,
                **params,
            },
        }

    def attempt(self, name):
        return f"attempt_{name}_{os.getpid()}_{time.monotonic_ns()}"

    def submit(self, body, expect=202):
        return self.client.json("POST", "/v1/executions", body, expect=expect)

    def wait(self, execution_id, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = self.client.json("GET", "/v1/executions/" + execution_id)
            if snap["state"] in TERMINAL:
                return snap
            time.sleep(0.05)
        self.fail("execution did not finish")

    def discard(self, execution_id):
        self.client.json("DELETE", "/v1/executions/" + execution_id)

    def run_to_end(self, body):
        snap = self.wait(self.submit(body)["execution_id"])
        return snap

    def error(self, status, value, code, field=None):
        self.assertIsInstance(value, dict)
        self.assertEqual(value["error"]["code"], code, value)
        if field is not None:
            self.assertEqual(value["error"].get("field"), field, value)
        self.assertEqual(set(value["error"]) - {"code", "message", "retryable", "field", "restart_required"}, set())


class Discovery(Section):
    section = "discovery"

    def test_health_is_coarse_and_unauthenticated(self):
        status, value, _ = self.client.call("GET", "/health", auth=False)
        self.assertEqual((status, value), (200, {"status": "ok"}))

    def test_credential_is_required(self):
        status, value, _ = self.client.call("GET", "/v1/info", auth=False)
        self.assertEqual(status, 401)
        self.error(status, value, "unauthorized")
        status, value, _ = self.client.call("GET", "/v1/info", headers={"Origin": "http://x"})
        self.assertEqual(status, 403)

    def test_info_and_capabilities(self):
        for key in ("protocol", "worker_version", "backend_id", "session_id", "state", "service_ready", "device_ready", "model_ready", "loaded_bundle", "device", "build"):
            self.assertIn(key, self.info)
        for key in ("tasks", "task_parameters", "defaults", "load_defaults", "limits", "precisions", "max_retained_executions", "max_storage_mb", "max_input_mb", "memory_estimate", "backend_id", "ready"):
            self.assertIn(key, self.caps)
        self.assertEqual(self.info["session_id"], self.caps["session_id"])
        self.assertIn(self.info["protocol"], self.protocols)
        for task in self.caps["tasks"]:
            self.assertIn(task, self.caps["task_parameters"])
        if self.minor() >= 2:
            for key in ("loaders", "auto_options", "fallbacks"):
                self.assertIn(key, self.caps)
            for task in self.caps["tasks"]:
                self.assertTrue(self.caps["task_parameters"][task]["outputs"])
            identity = self.info["device_identity"]
            for key in ("device", "index", "name", "total_memory_mb", "host_id", "host_id_source", "pid"):
                self.assertIn(key, identity)
        if "image-to-video" in self.caps["tasks"]:
            self.assertGreaterEqual(self.minor(), 3)
            video = self.caps["task_parameters"]["image-to-video"]
            self.assertEqual(video["outputs"][:2], ["frames", "poster"])
            self.assertEqual(video["output_formats"]["frames"], "application/zip")
            self.assertEqual(video["output_formats"]["poster"], "image/png")

    def test_startup_record(self):
        if not self.startup:
            self.skipTest("the suite did not launch this worker (no startup record)")
        for key in ("protocol", "worker_version", "session_id", "port", "pid"):
            self.assertIn(key, self.startup)
        self.assertEqual(self.startup["session_id"], self.info["session_id"])
        self.assertEqual(self.startup["port"], urlsplit(self.url).port)
        text = json.dumps(self.startup)
        self.assertNotIn(self.token, text)
        self.assertNotRegex(text, r'"/(home|tmp|root|var|Users)/')
        if self.minor() >= 2:
            self.assertEqual(self.startup["device_identity"]["host_id"], self.info["device_identity"]["host_id"])


class Negotiation(Section):
    section = "negotiation"

    def test_every_listed_minor_validates(self):
        for version in self.protocols:
            value = self.client.json("POST", "/v1/validate", self.request("attempt_n", version=version))
            self.assertEqual(value["task"], "text-to-image")

    def test_other_versions_are_refused(self):
        for version in ("media-worker/2.0", "media-worker/1.99", "other/1.0"):
            status, value, _ = self.client.call("POST", "/v1/validate", self.request("attempt_n", version=version))
            self.assertEqual(status, 400)
            self.error(status, value, "protocol_mismatch", "protocol")

    def test_newer_fields_are_refused_on_older_minors(self):
        if "media-worker/1.0" in self.protocols and "image-to-image" in self.caps["tasks"]:
            body = self.request("attempt_n", version="media-worker/1.0")
            body.update(task="image-to-image")
            body["inputs"]["image"] = "input_" + "0" * 32
            body["parameters"]["strength"] = 0.5
            status, value, _ = self.client.call("POST", "/v1/validate", body)
            self.error(status, value, "unsupported", "task")
        older = [v for v in self.protocols if self.minor(v) < 2]
        if self.minor() >= 2 and older:
            status, value, _ = self.client.call("POST", "/v1/validate", self.request("attempt_n", version=older[-1], count=2))
            self.error(status, value, "unsupported", "count")
            body = {"protocol": older[-1], "bundle": self.bundle, "options": dict(self.options(), precision="auto")}
            status, value, _ = self.client.call("POST", "/v1/validate", body)
            self.error(status, value, "unsupported", "options")
        older = [v for v in self.protocols if self.minor(v) < 3]
        if "image-to-video" in self.caps["tasks"] and older:
            # A 1.2 (or older) caller sees no video: the task and the family are unsupported.
            body = video_request(older[-1], "attempt_n", "sha256:" + "0" * 64, "input_" + "0" * 32)
            status, value, _ = self.client.call("POST", "/v1/validate", body)
            self.error(status, value, "unsupported", "task")
            video = (self.registration or {}).get("video")
            if video and self.minor(older[-1]) >= 2:
                body = {"protocol": older[-1], "bundle": video, "options": self.options()}
                status, value, _ = self.client.call("POST", "/v1/validate", body)
                self.error(status, value, "unsupported", "family")


class Validation(Section):
    section = "validation"

    def test_every_advertised_task_validates(self):
        for task in self.caps["tasks"]:
            if task == "upscale":
                if not (self.registration or {}).get("upscaler"):
                    self.skipTest("upscale is advertised but no upscaler registration was supplied")
                body = {
                    "protocol": self.version,
                    "attempt_id": "attempt_v",
                    "bundle_digest": self.registration["upscaler"]["bundle_digest"],
                    "task": task,
                    "inputs": {"image": "input_" + "0" * 32},
                    "parameters": {"width": 64, "height": 64},
                }
            elif task == "image-to-video":
                if not (self.registration or {}).get("video"):
                    self.skipTest("image-to-video is advertised but no video registration was supplied")
                body = video_request(self.version, "attempt_v", self.registration["video"]["bundle_digest"], "input_" + "0" * 32)
            else:
                body = self.request("attempt_v")
                body["task"] = task
                if task == "image-to-image":
                    body["inputs"]["image"] = "input_" + "0" * 32
                    body["parameters"]["strength"] = 0.5
            value = self.client.json("POST", "/v1/validate", body)
            self.assertEqual(value["task"], task)
            if self.minor() >= 2:
                self.assertEqual(value["outputs"][0], self.caps["task_parameters"][task]["outputs"][0])

    def test_load_validation_reports_an_estimate(self):
        value = self.client.json("POST", "/v1/validate", {"protocol": self.version, "bundle": self.bundle, "options": self.options()})
        self.assertTrue(value["valid"])
        for key in ("value_mb", "source", "uncertainty"):
            self.assertIn(key, value["memory_estimate"])
        if self.minor() >= 2:
            self.assertIn("resolved_options", value)

    def test_unknown_fields_and_bad_values_are_refused(self):
        status, value, _ = self.client.call("POST", "/v1/validate", {**self.request("attempt_v"), "extra": 1})
        self.error(status, value, "invalid_request")
        status, value, _ = self.client.call("POST", "/v1/validate", self.request("attempt_v", width=65))
        self.error(status, value, "invalid_request", "width")
        status, value, _ = self.client.call("POST", "/v1/validate", self.request("attempt_v", sampler="ddim"))
        self.error(status, value, "unsupported", "scheduler")


class Lifecycle(Section):
    section = "lifecycle"

    def test_unload_then_load(self):
        record = self.operation("unload", {"protocol": self.version})
        self.assertEqual(record["action"], "unload")
        info = self.client.json("GET", "/v1/info")
        self.assertFalse(info["model_ready"])
        status, value, _ = self.client.call("POST", "/v1/executions", self.request(self.attempt("nl")))
        self.assertEqual(status, 409)
        self.error(status, value, "model_not_ready")
        record = self.operation("load", {"protocol": self.version, "bundle": self.bundle, "options": self.options()})
        self.assertIn("memory_estimate", record)
        info = self.client.json("GET", "/v1/info")
        self.assertTrue(info["model_ready"])
        self.assertEqual(info["loaded_bundle"]["bundle_digest"], self.bundle["bundle_digest"])

    def test_unknown_operation(self):
        status, value, _ = self.client.call("GET", "/v1/operations/op_" + "0" * 32)
        self.assertEqual(status, 404)
        self.error(status, value, "operation_not_found")


class SingleFlight(Section):
    section = "single_flight"

    def test_second_execution_is_busy(self):
        self.ensure_loaded()
        first = self.submit(self.request(self.attempt("long"), steps=40))
        status, value, _ = self.client.call("POST", "/v1/executions", self.request(self.attempt("second")))
        self.assertEqual(status, 409)
        self.error(status, value, "busy")
        status, value, _ = self.client.call("POST", "/v1/model/unload", {"protocol": self.version})
        self.assertEqual(status, 409)
        self.client.call("POST", f"/v1/executions/{first['execution_id']}/cancel")
        self.wait(first["execution_id"])
        self.discard(first["execution_id"])


class Idempotency(Section):
    section = "idempotency"

    def test_identical_and_conflicting_attempts(self):
        self.ensure_loaded()
        body = self.request(self.attempt("idem"))
        first = self.submit(body)
        again = self.submit(body)
        self.assertEqual(first["execution_id"], again["execution_id"])
        status, value, _ = self.client.call("POST", "/v1/executions", {**body, "parameters": {**body["parameters"], "seed": 12}})
        self.assertEqual(status, 409)
        self.error(status, value, "attempt_conflict")
        self.wait(first["execution_id"])
        found = self.client.json("GET", "/v1/executions?attempt_id=" + body["attempt_id"])
        self.assertEqual(found["execution_id"], first["execution_id"])
        self.discard(first["execution_id"])
        status, value, _ = self.client.call("GET", "/v1/executions?attempt_id=" + body["attempt_id"])
        self.assertEqual(status, 404)


class Events(Section):
    section = "events"

    def test_replay_and_resume(self):
        self.ensure_loaded()
        snap = self.run_to_end(self.request(self.attempt("events"), steps=3))
        self.assertEqual(snap["state"], "succeeded")
        sequences = [e["sequence"] for e in snap["events"]]
        self.assertEqual(sequences, list(range(1, snap["last_sequence"] + 1)))
        status, events = self.client.events(snap["execution_id"])
        self.assertEqual(status, 200)
        self.assertEqual([e["sequence"] for e in events], sequences)
        for event in events:
            for key in ("session_id", "execution_id", "sequence", "timestamp", "type", "data"):
                self.assertIn(key, event)
        status, events = self.client.events(snap["execution_id"], last_event_id=2)
        self.assertEqual([e["sequence"] for e in events], sequences[2:])
        status, _ = self.client.events(snap["execution_id"], last_event_id=snap["last_sequence"] + 5)
        self.assertEqual(status, 409)
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        self.assertTrue(progress)
        for data in progress:
            self.assertEqual(data["scope"], "stage")
            self.assertLessEqual(data["completed_steps"], data["total_steps"])
        self.discard(snap["execution_id"])


class Cancellation(Section):
    section = "cancellation"

    def cancel_at(self, stage, steps=30):
        self.ensure_loaded()
        accepted = self.submit(self.request(self.attempt("cancel_" + stage), steps=steps))
        execution_id = accepted["execution_id"]
        sent = threading.Event()

        def on_event(event):
            if not sent.is_set() and event["type"] == "stage" and event["data"]["stage"] == stage:
                sent.set()
                self.client.call("POST", f"/v1/executions/{execution_id}/cancel")

        self.client.events(execution_id, on_event=on_event)
        snap = self.wait(execution_id)
        self.assertTrue(sent.is_set(), f"stage {stage} was never reported")
        # A cancel that arrives after success returns the terminal snapshot and records nothing.
        self.assertEqual(snap["cancel_requested"], snap["state"] != "succeeded")
        self.discard(execution_id)
        return snap

    def test_cancel_during_encoding(self):
        snap = self.cancel_at("encoding")
        self.assertEqual(snap["state"], "cancelled")
        self.assertEqual(snap["outputs"], [])

    def test_cancel_during_sampling(self):
        snap = self.cancel_at("sampling")
        self.assertEqual(snap["state"], "cancelled")
        self.assertLess(snap["progress"]["completed_steps"] if snap["progress"] else 0, 30)

    def test_cancel_during_decoding(self):
        snap = self.cancel_at("decoding", steps=2)
        # Completion may win the race after decoding; then the result is complete, never partial.
        self.assertIn(snap["state"], ("cancelled", "succeeded"))
        self.assertEqual(bool(snap["outputs"]), snap["state"] == "succeeded")

    def test_cancel_of_terminal_returns_it(self):
        self.ensure_loaded()
        snap = self.run_to_end(self.request(self.attempt("terminal")))
        status, value, _ = self.client.call("POST", f"/v1/executions/{snap['execution_id']}/cancel")
        self.assertEqual((status, value["state"]), (200, "succeeded"))
        self.discard(snap["execution_id"])


class Outputs(Section):
    section = "outputs"

    def test_finalized_outputs_match_their_digests(self):
        self.ensure_loaded()
        snap = self.run_to_end(self.request(self.attempt("outputs")))
        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual([o["output_id"] for o in snap["outputs"]], ["image"])
        for output in snap["outputs"]:
            status, data, headers = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/{output['output_id']}")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], output["mime_type"])
            self.assertEqual(len(data), output["size"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), output["sha256"])
            self.assertEqual(struct.unpack(">II", data[16:24]), (output["width"], output["height"]))
        status, value, _ = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/nothing")
        self.assertEqual(status, 404)
        self.discard(snap["execution_id"])
        status, value, _ = self.client.call("GET", f"/v1/executions/{snap['execution_id']}")
        self.assertEqual(status, 404)
        self.error(status, value, "execution_not_found")


class MultiOutput(Section):
    section = "multi_output"

    def test_count_outputs(self):
        if self.minor() < 2 or "count" not in self.caps["task_parameters"]["text-to-image"].get("parameters", []):
            return  # not applicable: the worker does not advertise count
        self.ensure_loaded()
        snap = self.run_to_end(self.request(self.attempt("count"), count=2))
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        self.assertEqual([o["output_id"] for o in snap["outputs"]], ["image", "image_2"])
        self.assertEqual(len([e for e in snap["events"] if e["type"] == "output"]), 2)
        for output in snap["outputs"]:
            status, data, _ = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/{output['output_id']}")
            self.assertEqual(hashlib.sha256(data).hexdigest(), output["sha256"])
        alone = self.run_to_end(self.request(self.attempt("alone"), seed=12))
        # Output k is seed + k − 1; byte equality is backend/hardware qualified, so check identity
        # only where the backend reproduces bytes (the fixture and CPU float32 do).
        self.assertEqual(alone["outputs"][0]["width"], snap["outputs"][1]["width"])
        self.discard(snap["execution_id"])
        self.discard(alone["execution_id"])


class Retention(Section):
    section = "retention"

    def test_records_are_never_evicted(self):
        self.ensure_loaded()
        limit = self.caps["max_retained_executions"]
        accepted, refused = [], None
        for index in range(limit + 1):
            status, value, _ = self.client.call("POST", "/v1/executions", self.request(self.attempt(f"fill{index}"), steps=1))
            if status == 503:
                refused = value
                break
            self.assertEqual(status, 202, value)
            accepted.append(value["execution_id"])
            self.wait(value["execution_id"])
        self.assertIsNotNone(refused, "the worker accepted more executions than max_retained")
        self.error(503, refused, "storage_full")
        self.assertLessEqual(len(accepted), limit)
        for execution_id in accepted:
            snap = self.client.json("GET", "/v1/executions/" + execution_id)
            self.assertEqual(snap["state"], "succeeded")
        for execution_id in accepted:
            self.discard(execution_id)
        snap = self.run_to_end(self.request(self.attempt("after")))
        self.assertEqual(snap["state"], "succeeded")
        self.discard(snap["execution_id"])


class Inputs(Section):
    section = "inputs"

    def test_input_lifecycle(self):
        status, value, _ = self.client.call("POST", "/v1/inputs", data=make_png(64, 64), headers={"Content-Type": "image/png"})
        self.assertEqual(status, 201, value)
        for key in ("input_id", "session_id", "mime_type", "size", "sha256", "width", "height"):
            self.assertIn(key, value)
        self.assertEqual(value["sha256"], hashlib.sha256(make_png(64, 64)).hexdigest())
        self.assertEqual(self.client.json("GET", "/v1/inputs/" + value["input_id"]), value)
        status, error, _ = self.client.call("POST", "/v1/inputs", data=b"not a png at all, padded" * 4, headers={"Content-Type": "image/png"})
        self.assertEqual(status, 400)
        status, error, _ = self.client.call("POST", "/v1/inputs", data=make_png(64, 64), headers={"Content-Type": "image/jpeg"})
        self.assertEqual(status, 400)
        if "image-to-image" in self.caps["tasks"]:
            self.ensure_loaded()
            body = self.request(self.attempt("i2i"), steps=20)
            body["task"] = "image-to-image"
            body["inputs"]["image"] = value["input_id"]
            body["parameters"]["strength"] = 1
            run = self.submit(body)
            status, error, _ = self.client.call("DELETE", "/v1/inputs/" + value["input_id"])
            if self.client.json("GET", "/v1/executions/" + run["execution_id"])["state"] == "running":
                self.assertEqual(status, 409)
            self.wait(run["execution_id"])
            self.discard(run["execution_id"])
        self.client.call("DELETE", "/v1/inputs/" + value["input_id"])
        status, error, _ = self.client.call("GET", "/v1/inputs/" + value["input_id"])
        self.assertEqual(status, 404)
        self.error(status, error, "input_not_found")


class Upscale(Section):
    section = "upscale"

    def test_upscale(self):
        if "upscale" not in self.caps["tasks"]:
            return  # not applicable
        upscaler = (self.registration or {}).get("upscaler")
        if not upscaler:
            self.skipTest("upscale is advertised but no upscaler registration was supplied")
        status, record, _ = self.client.call("POST", "/v1/inputs", data=make_png(64, 64), headers={"Content-Type": "image/png"})
        self.assertEqual(status, 201)
        self.operation("load", {"protocol": self.version, "bundle": upscaler, "options": self.options()})
        info = self.client.json("GET", "/v1/info")
        scale = info["loaded_bundle"]["scale"]
        self.assertEqual(info["loaded_bundle"]["tasks"], ["upscale"])
        body = {
            "protocol": self.version,
            "attempt_id": self.attempt("upscale"),
            "bundle_digest": upscaler["bundle_digest"],
            "task": "upscale",
            "inputs": {"image": record["input_id"]},
            "parameters": {"width": 64, "height": 64},
        }
        bad = json.loads(json.dumps(body))
        bad["inputs"]["prompt"] = SECRET
        status, value, _ = self.client.call("POST", "/v1/executions", bad)
        self.error(status, value, "invalid_request", "prompt")
        status, value, _ = self.client.call("POST", "/v1/executions", self.request(self.attempt("wrongtask")) | {"bundle_digest": upscaler["bundle_digest"]})
        self.error(status, value, "unsupported", "task")
        snap = self.run_to_end(body)
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        output = snap["outputs"][0]
        self.assertEqual((output["width"], output["height"]), (64 * scale, 64 * scale))
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        self.assertTrue(progress)
        self.assertTrue(all(p["stage"] == "upscaling" and p["unit"] == "tiles" for p in progress))
        status, data, _ = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/image")
        self.assertEqual(struct.unpack(">II", data[16:24]), (64 * scale, 64 * scale))
        self.discard(snap["execution_id"])
        self.client.call("DELETE", "/v1/inputs/" + record["input_id"])
        self.ensure_loaded()


def video_request(version, attempt, bundle_digest, input_id, width=64, height=64, **params):
    return {
        "protocol": version,
        "attempt_id": attempt,
        "bundle_digest": bundle_digest,
        "task": "image-to-video",
        "inputs": {"image": input_id},
        "parameters": {"width": width, "height": height, "seed": 5, "steps": 2, **params},
    }


class ImageToVideo(Section):
    section = "image_to_video"

    def test_image_to_video(self):
        if "image-to-video" not in self.caps["tasks"]:
            return  # not applicable
        video = (self.registration or {}).get("video")
        if not video:
            self.skipTest("image-to-video is advertised but no video registration was supplied")
        self.operation("load", {"protocol": self.version, "bundle": video, "options": self.options()})
        loaded = self.client.json("GET", "/v1/info")["loaded_bundle"]
        self.assertEqual(loaded["tasks"], ["image-to-video"])
        width, height = loaded["video_sizes"][0]
        status, record, _ = self.client.call("POST", "/v1/inputs", data=make_png(width, height), headers={"Content-Type": "image/png"})
        self.assertEqual(status, 201, record)

        def body(name, **params):
            return video_request(self.version, self.attempt(name), video["bundle_digest"], record["input_id"], width, height, **params)

        bad = body("prompt")
        bad["inputs"]["prompt"] = SECRET
        status, value, _ = self.client.call("POST", "/v1/executions", bad)
        self.error(status, value, "invalid_request", "prompt")
        status, value, _ = self.client.call("POST", "/v1/executions", body("frames", frames=65))
        self.error(status, value, "invalid_request", "frames")
        unadvertised = body("size")
        unadvertised["parameters"].update(width=width + 8)
        status, value, _ = self.client.call("POST", "/v1/executions", unadvertised)
        self.error(status, value, "invalid_request", "video_size")
        status, value, _ = self.client.call("POST", "/v1/executions", self.request(self.attempt("wrongtask")) | {"bundle_digest": video["bundle_digest"]})
        self.error(status, value, "unsupported", "task")

        snap = self.run_to_end(body("video", frames=2, fps=8, decode_chunk=1))
        self.assertEqual(snap["state"], "succeeded", snap.get("error"))
        self.assertEqual([o["output_id"] for o in snap["outputs"]], ["frames", "poster"])
        stages = [e["data"]["stage"] for e in snap["events"] if e["type"] == "stage"]
        self.assertEqual(stages, ["encoding", "sampling", "decoding", "packaging", "saving"])
        progress = [e["data"] for e in snap["events"] if e["type"] == "progress"]
        decoding = [p for p in progress if p["stage"] == "decoding"]
        self.assertTrue([p for p in progress if p["stage"] == "sampling"])
        self.assertEqual([(p["unit"], p["completed_steps"], p["total_steps"]) for p in decoding], [("frames", 1, 2), ("frames", 2, 2)])
        self.assertEqual(snap["effective_settings"]["frames"], 2)
        frames, poster = snap["outputs"]
        self.assertEqual(frames["mime_type"], "application/zip")
        self.assertEqual(poster["mime_type"], "image/png")
        self.assertEqual((frames["width"], frames["height"], frames["frames"], frames["fps"]), (width, height, 2, 8))
        self.assertAlmostEqual(frames["duration_seconds"], 0.25)
        status, data, headers = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/frames")
        self.assertEqual((status, headers["Content-Type"]), (200, "application/zip"))
        self.assertEqual(hashlib.sha256(data).hexdigest(), frames["sha256"])
        with zipfile.ZipFile(BytesIO(data)) as archive:
            self.assertEqual(sorted(archive.namelist()), ["000001.png", "000002.png", "frames.json"])
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()))
            manifest = json.loads(archive.read("frames.json"))
            self.assertEqual(manifest, {"fps": 8, "width": width, "height": height, "frames": 2})
            first = archive.read("000001.png")
            for name in ("000001.png", "000002.png"):
                self.assertEqual(struct.unpack(">II", archive.read(name)[16:24]), (width, height))
        status, data, _ = self.client.call("GET", f"/v1/executions/{snap['execution_id']}/outputs/poster")
        self.assertEqual(hashlib.sha256(data).hexdigest(), poster["sha256"])
        self.assertEqual(data, first)  # the poster is frame 1, byte for byte
        self.discard(snap["execution_id"])

        accepted = self.submit(body("cancel", steps=50))
        sent = threading.Event()

        def on_event(event):
            if not sent.is_set() and event["type"] == "stage" and event["data"]["stage"] == "sampling":
                sent.set()
                self.client.call("POST", f"/v1/executions/{accepted['execution_id']}/cancel")

        self.client.events(accepted["execution_id"], on_event=on_event)
        cancelled = self.wait(accepted["execution_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(cancelled["outputs"], [])
        self.discard(accepted["execution_id"])
        self.client.call("DELETE", "/v1/inputs/" + record["input_id"])
        self.ensure_loaded()


class Privacy(Section):
    section = "privacy"

    def test_errors_never_carry_prompts_paths_or_exceptions(self):
        self.ensure_loaded()
        body = self.request(self.attempt("privacy"))
        body["inputs"]["prompt"] = SECRET
        for broken in (
            {**body, "parameters": {**body["parameters"], "width": 63}},
            {**body, "inputs": {"prompt": SECRET, "unexpected": SECRET}},
            {**body, "bundle_digest": "sha256:" + "1" * 64},
            {**body, "task": SECRET},
        ):
            self.client.call("POST", "/v1/executions", broken)
            self.client.call("POST", "/v1/validate", broken)
        self.client.call("POST", "/v1/validate", {"protocol": self.version, "bundle": {**self.bundle, "configuration": {"artifact_handle": SECRET, "files": {}}}, "options": self.options()})
        self.client.call("POST", "/v1/executions", data=b"{" + SECRET.encode(), headers={"Content-Type": "application/json"})
        self.client.call("GET", "/v1/executions/exec_" + "f" * 32)
        self.assertTrue(self.client.errors)
        for text in self.client.errors:
            self.assertNotIn(SECRET, text)
            self.assertNotIn("Traceback", text)
            self.assertNotIn("Error:", text)
            self.assertIsNone(re.search(r"/(home|tmp|root|var|Users|etc)/", text), text)
            self.assertEqual(set(json.loads(text)), {"error"})


SECTIONS = [Discovery, Negotiation, Validation, Lifecycle, SingleFlight, Idempotency, Events, Cancellation, Outputs, MultiOutput, Upscale, ImageToVideo, Inputs, Retention, Privacy]


def load_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite(loader.loadTestsFromTestCase(section) for section in SECTIONS)
