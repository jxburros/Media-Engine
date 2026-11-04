"""HTTP trust boundary and bounded, prompt-free metrics on actual listeners."""
import http.client
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fixture import TestBackend, make_bundle
from media_engine.contracts import DEFAULTS, PROTOCOL
from media_engine.server import Worker, create_server

TOKEN = "test-serving-credential-at-least-32-characters"


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.worker = Worker(self.root / "worker", TestBackend(), {"fixture": str(self.root / "model")})
        self.server = create_server(self.worker, TOKEN, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temp.cleanup()

    def call(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if isinstance(data, dict):
            data = json.dumps(data)
        conn.request(method, path, data, {"Authorization": "Bearer " + TOKEN, **(headers or {})})
        response = conn.getresponse()
        value = response.read()
        status = response.status
        conn.close()
        return status, value

    def test_loopback_host_origin_lengths_and_depth(self):
        for host in ("attacker.example", "127.0.0.1.evil", "127.0.0.1@evil", "localhost:bad", "localhost/evil"):
            with self.subTest(host=host):
                self.assertEqual(self.call("GET", "/health", headers={"Host": host})[0], 403)
        self.assertEqual(self.call("GET", "/health", headers={"Origin": ""})[0], 403)
        self.assertEqual(self.call("POST", "/v1/validate", "[]", {"Content-Length": "4194305"})[0], 413)
        for data in (b"[" * 65 + b"0" + b"]" * 65, b"{}"): 
            headers = {"Transfer-Encoding": "chunked"} if data == b"{}" else {}
            self.assertEqual(self.call("POST", "/v1/validate", data, headers)[0], 400)
        self.assertEqual(self.call("GET", "/health", headers={"Host": "localhost"})[0], 200)

    def test_duplicate_length_is_refused(self):
        with socket.create_connection(("127.0.0.1", self.port), 5) as sock:
            sock.sendall(b"POST /v1/validate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n{}")
            self.assertIn(b" 400 ", sock.recv(4096))

    def test_metrics_are_authenticated_and_survive_record_release(self):
        self.assertEqual(self.call("GET", "/metrics", headers={"Authorization": "Bearer wrong"})[0], 401)
        status, data = self.call("POST", "/v1/model/load", {"protocol": PROTOCOL, "bundle": self.bundle, "options": self.worker.backend.capabilities()["load_defaults"]})
        self.assertEqual(status, 202)
        op = json.loads(data)["operation_id"]
        deadline = time.monotonic() + 5
        while self.worker.operations[op]["state"] == "running" and time.monotonic() < deadline:
            time.sleep(.01)
        request = {"protocol": PROTOCOL, "attempt_id": "attempt_metrics", "bundle_digest": self.bundle["bundle_digest"], "task": "text-to-image", "inputs": {"prompt": "never-export-this-prompt"}, "parameters": {**DEFAULTS, "seed": 1, "width": 64, "height": 64, "steps": 10}}
        status, data = self.call("POST", "/v1/executions", request)
        self.assertEqual(status, 202, data)
        execution = json.loads(data)["execution_id"]
        request["attempt_id"] = "attempt_second"
        self.assertEqual(self.call("POST", "/v1/executions", request)[0], 409)
        deadline = time.monotonic() + 5
        while self.worker.snapshot(execution)["state"] == "running" and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.call("DELETE", "/v1/executions/" + execution)[0], 200)
        status, data = self.call("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b'media_engine_executions_total{task="text-to-image",state="succeeded"} 1', data)
        self.assertIn(b"media_engine_busy_refusals_total 1", data)
        self.assertIn(b'media_engine_phase_seconds_total{task="text-to-image",phase="sampling"}', data)
        self.assertNotIn(b"never-export", data)
        self.assertNotIn(TOKEN.encode(), data)
        self.assertNotIn(execution.encode(), data)


@unittest.skipUnless(os.environ.get("MEDIA_ENGINE_NATIVE"), "MEDIA_ENGINE_NATIVE is not set")
class NativeServingTests(ServingTests):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        catalog = self.root / "catalog.json"
        catalog.write_text("{}")
        self.process = subprocess.Popen([os.environ["MEDIA_ENGINE_NATIVE"], "--catalog", str(catalog), "--directory", str(self.root / "native"), "--device", "cpu", "--port", "0"], stdout=subprocess.PIPE, text=True, env={**os.environ, "MEDIA_ENGINE_TOKEN": TOKEN})
        self.port = json.loads(self.process.stdout.readline())["port"]

    def tearDown(self):
        self.process.terminate()
        self.process.wait(10)
        self.process.stdout.close()
        self.temp.cleanup()

    def test_metrics_are_authenticated_and_survive_record_release(self):
        # Lifecycle counters are exercised by the native oracle; this listener has no weights.
        self.assertEqual(self.call("GET", "/metrics", headers={"Authorization": "Bearer wrong"})[0], 401)
        status, data = self.call("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b"media_engine_busy 0", data)

    def test_duplicate_length_is_refused(self):
        with socket.create_connection(("127.0.0.1", self.port), 5) as sock:
            sock.sendall(b"POST /v1/validate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n{}")
            self.assertIn(b" 400 ", sock.recv(4096))


@unittest.skipUnless(os.environ.get("MEDIA_ENGINE_NATIVE"), "MEDIA_ENGINE_NATIVE is not set")
class NativeTLS(unittest.TestCase):
    def test_mutual_tls_transport_and_launch_refusals(self):
        import hashlib
        import shutil
        import ssl
        native = os.environ["MEDIA_ENGINE_NATIVE"]
        certs = Path(__file__).parent / "certs"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "catalog.json").write_text("{}")
            for name in ("server.pem", "server.key", "ca.pem"):
                shutil.copyfile(certs / name, root / name)
                (root / name).chmod(0o600 if name.endswith("key") else 0o644)
            base = [native, "--catalog", str(root / "catalog.json"), "--directory", str(root / "work"), "--port", "0", "--device", "cpu"]
            env = {**os.environ, "MEDIA_ENGINE_TOKEN": TOKEN}
            refused = subprocess.run([*base, "--bind", "0.0.0.0"], capture_output=True, env=env, timeout=10)
            self.assertEqual(refused.returncode, 2)
            proc = subprocess.Popen([*base, "--bind", "127.0.0.1", "--no-parent-watch", "--tls-cert", str(root / "server.pem"), "--tls-key", str(root / "server.key"), "--tls-client-ca", str(root / "ca.pem")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            try:
                line = proc.stdout.readline()
                self.assertTrue(line, proc.stderr.read() if proc.poll() is not None else "missing startup")
                record = json.loads(line)
                self.assertEqual(record["scheme"], "https")
                self.assertEqual(record["ownership_available"], False)
                self.assertEqual(record["certificate_sha256"], hashlib.sha256(ssl.PEM_cert_to_DER_cert((root / "server.pem").read_text())).hexdigest())
                context = ssl.create_default_context(cafile=str(root / "ca.pem"))
                conn = http.client.HTTPSConnection("127.0.0.1", record["port"], context=context, timeout=5)
                with self.assertRaises((ssl.SSLError, OSError, http.client.RemoteDisconnected)):
                    conn.request("GET", "/v1/info", headers={"Authorization": "Bearer " + TOKEN})
                    conn.getresponse().read()
                conn.close()
                context.load_cert_chain(str(certs / "client.pem"), str(certs / "client.key"))
                conn = http.client.HTTPSConnection("127.0.0.1", record["port"], context=context, timeout=5)
                conn.request("GET", "/v1/info", headers={"Authorization": "Bearer " + TOKEN})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["certificate_sha256"], record["certificate_sha256"])
                conn.close()
                conn = http.client.HTTPSConnection("127.0.0.1", record["port"], context=context, timeout=5)
                conn.request("GET", "/metrics", headers={"Authorization": "Bearer wrong"})
                self.assertEqual(conn.getresponse().status, 401)
                conn.close()
                context.maximum_version = ssl.TLSVersion.TLSv1_2
                conn = http.client.HTTPSConnection("127.0.0.1", record["port"], context=context, timeout=5)
                with self.assertRaises(ssl.SSLError):
                    conn.request("GET", "/health")
                    conn.getresponse()
                conn.close()
            finally:
                proc.terminate()
                proc.wait(10)
                proc.stdout.close()
                proc.stderr.close()
