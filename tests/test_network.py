"""Network mode: mutual TLS on the fixture worker, launch refusals and the transport fields.

Everything runs on loopback with the committed certificates in tests/certs (never generated
here); a real multi-host network is outside what this workspace can test.
"""

import hashlib
import io
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERTS = HERE / "certs"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))

from tests.conformance.run import run_suite  # noqa: E402
from media_engine.server import NetworkError, Worker, create_server  # noqa: E402
from tests.fixture import TestBackend  # noqa: E402

SERVER_SHA256 = hashlib.sha256(
    ssl.PEM_cert_to_DER_cert((CERTS / "server.pem").read_text())
).hexdigest()
TRANSPORT = ("scheme", "bind", "certificate_sha256", "ownership_available")


def client_context(cert="client", maximum=None):
    context = ssl.create_default_context(cafile=str(CERTS / "ca.pem"))
    if cert:
        context.load_cert_chain(str(CERTS / f"{cert}.pem"), str(CERTS / f"{cert}.key"))
    if maximum:
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = maximum
    return context


class NetworkModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.processes = []
        self.saved = dict(os.environ)
        # Git does not keep modes; the worker refuses a key readable by group or others.
        self.key = self.dir / "server.key"
        shutil.copyfile(CERTS / "server.key", self.key)
        self.key.chmod(0o600)
        self.cert = self.dir / "server.pem"
        shutil.copyfile(CERTS / "server.pem", self.cert)
        self.cert.chmod(0o644)
        self.ca = self.dir / "ca.pem"
        shutil.copyfile(CERTS / "ca.pem", self.ca)
        self.ca.chmod(0o644)

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.wait(10)
            process.stdout.close()
            if process.stderr:
                process.stderr.close()
        os.environ.clear()
        os.environ.update(self.saved)
        self.temp.cleanup()

    def tls_args(self, key=None, cert=None):
        return [
            "--tls-cert", str(cert or self.cert),
            "--tls-key", str(key or self.key),
            "--tls-client-ca", str(self.ca),
        ]

    def start(self, *extra, command=None, env=None):
        root = self.dir / f"root{len(self.processes)}"
        token = secrets.token_hex(32)
        environment = {
            **os.environ,
            "PYTHONPATH": str(HERE.parent / "src"),
            "MEDIA_ENGINE_TOKEN": token,
            "MEDIA_ENGINE_FIXTURE_MAX_RETAINED": "10",
            **(env or {}),
        }
        process = subprocess.Popen(
            command or [sys.executable, str(HERE / "serve_fixture.py"), str(root), *extra],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            text=True,
        )
        self.processes.append(process)
        return root, token, process

    def launch(self, *extra, env=None):
        root, token, process = self.start(*extra, env=env)
        line = process.stdout.readline()
        if not line:
            self.fail("worker did not start: " + process.stderr.read())
        return root, token, json.loads(line)

    def refused(self, *extra, command=None, env=None):
        _root, _token, process = self.start(*extra, command=command, env=env)
        _out, err = process.communicate(timeout=60)
        self.assertEqual(process.returncode, 2, err)
        return err

    def https(self, port, path, token=None, context=None):
        headers = {"Authorization": "Bearer " + token} if token else {}
        request = urllib.request.Request(f"https://127.0.0.1:{port}{path}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30, context=context or client_context()) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def assert_no_http_response(self, port, context=None):
        """The worker must answer nothing at the HTTP layer: the connection fails or closes."""
        connection = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            if context:
                connection = context.wrap_socket(connection, server_hostname="127.0.0.1")
            connection.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            data = b""
            while chunk := connection.recv(4096):
                data += chunk
        except (ssl.SSLError, ConnectionResetError, BrokenPipeError):
            data = b""
        finally:
            connection.close()
        self.assertNotIn(b"HTTP/", data)
        self.assertNotIn(b"status", data)

    def test_startup_record_and_info_report_the_transport(self):
        _root, token, record = self.launch(*self.tls_args())
        self.assertEqual(
            {key: record[key] for key in TRANSPORT},
            {"scheme": "https", "bind": "127.0.0.1", "certificate_sha256": SERVER_SHA256,
             "ownership_available": True},
        )
        status, info = self.https(record["port"], "/v1/info", token)
        self.assertEqual(status, 200)
        self.assertEqual({key: info[key] for key in TRANSPORT}, {key: record[key] for key in TRANSPORT})
        # The fingerprint is what a supervisor pins: the DER the handshake actually presents.
        with socket.create_connection(("127.0.0.1", record["port"]), timeout=10) as raw:
            with client_context().wrap_socket(raw, server_hostname="localhost") as tls:
                self.assertEqual(tls.version(), "TLSv1.3")
                presented = hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()
        self.assertEqual(presented, record["certificate_sha256"])
        text = json.dumps(record)
        self.assertNotIn(token, text)
        self.assertNotIn(str(self.dir), text)

    def test_plain_http_is_refused(self):
        _root, _token, record = self.launch(*self.tls_args())
        self.assert_no_http_response(record["port"])
        with self.assertRaises((urllib.error.URLError, ConnectionError, OSError)):
            urllib.request.urlopen(f"http://127.0.0.1:{record['port']}/health", timeout=10)

    def test_clients_without_a_trusted_certificate_are_refused_at_the_handshake(self):
        _root, token, record = self.launch(*self.tls_args())
        self.assert_no_http_response(record["port"], client_context(cert=None))
        self.assert_no_http_response(record["port"], client_context(cert="untrusted"))
        with self.assertRaises((ssl.SSLError, ConnectionError, OSError)):
            with socket.create_connection(("127.0.0.1", record["port"]), timeout=10) as raw:
                client_context(maximum=ssl.TLSVersion.TLSv1_2).wrap_socket(raw, server_hostname="localhost")
        # The worker is unaffected by the refused connections.
        self.assertEqual(self.https(record["port"], "/health"), (200, {"status": "ok"}))

    def test_certificate_authenticates_the_peer_and_the_bearer_authorizes_it(self):
        _root, token, record = self.launch(*self.tls_args())
        self.assertEqual(self.https(record["port"], "/health"), (200, {"status": "ok"}))
        status, value = self.https(record["port"], "/v1/info", "0" * 64)
        self.assertEqual(status, 401)
        self.assertEqual(value["error"]["code"], "unauthorized")
        status, value = self.https(record["port"], "/v1/info")
        self.assertEqual(status, 401)
        self.assertEqual(self.https(record["port"], "/v1/info", token)[0], 200)

    def conformance(self, root, token, record, tls):
        os.environ["MEDIA_ENGINE_URL"] = f"{record.get('scheme', 'http')}://127.0.0.1:{record['port']}"
        os.environ["MEDIA_ENGINE_TOKEN"] = token
        os.environ["MEDIA_ENGINE_STARTUP_RECORD"] = json.dumps(record)
        os.environ["MEDIA_ENGINE_REGISTRATION"] = str(root / "conformance.json")
        for name, value in (("CA", "ca.pem"), ("CERT", "client.pem"), ("KEY", "client.key")):
            if tls:
                os.environ["MEDIA_ENGINE_TLS_" + name] = str(CERTS / value)
            else:
                os.environ.pop("MEDIA_ENGINE_TLS_" + name, None)
        return run_suite(stream=io.StringIO())

    def test_conformance_over_mutual_tls_matches_loopback_http(self):
        root, token, record = self.launch(*self.tls_args())
        over_tls = self.conformance(root, token, record, tls=True)
        self.assertTrue(over_tls[0], over_tls[1])
        root, token, record = self.launch()
        self.assertEqual(record["scheme"], "http")
        over_http = self.conformance(root, token, record, tls=False)
        self.assertEqual(over_tls, over_http)

    def test_conformance_runner_launches_a_network_mode_worker(self):
        root = self.dir / "launched"
        command = [
            sys.executable, "-m", "tests.conformance.run", "--registration", str(root / "conformance.json"),
            "--tls-ca", str(CERTS / "ca.pem"), "--tls-cert", str(CERTS / "client.pem"),
            "--tls-key", str(CERTS / "client.key"),
            "--launch", sys.executable, str(HERE / "serve_fixture.py"), str(root), *self.tls_args(),
        ]
        environment = {**os.environ, "PYTHONPATH": str(HERE.parent / "src"),
                       "MEDIA_ENGINE_FIXTURE_MAX_RETAINED": "10"}
        result = subprocess.run(command, cwd=HERE.parent, env=environment, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("conformant: media-worker-conformance/1.0", result.stdout)

    def test_non_loopback_bind_requires_all_tls_material(self):
        err = self.refused("--bind", "0.0.0.0")
        self.assertIn("refusing to listen on non-loopback address 0.0.0.0", err)
        self.assertIn("never serves plain HTTP", err)
        err = self.refused("--bind", "0.0.0.0", "--tls-cert", str(self.cert), "--tls-key", str(self.key))
        self.assertIn("must be given together", err)
        self.refused(env={"MEDIA_ENGINE_FIXTURE_BIND": "::"})
        err = self.refused("--bind", "localhost")
        self.assertIn("IP address", err)
        # The real executable applies the same rule before reading anything else.
        catalog = self.dir / "catalog.json"
        catalog.write_text("{}")
        command = [sys.executable, "-m", "media_engine.server", "--catalog", str(catalog),
                   "--device", "cpu", "--port", "0", "--bind", "0.0.0.0"]
        err = self.refused(command=command)
        self.assertIn("refusing to listen on non-loopback address 0.0.0.0", err)
        self.assertIn("never serves plain HTTP", self.refused(command=command[:-1] + ["10.1.2.3"]))

    def test_any_address_with_tls_serves_only_tls(self):
        _root, token, record = self.launch("--bind", "0.0.0.0", *self.tls_args())
        self.assertEqual((record["scheme"], record["bind"]), ("https", "0.0.0.0"))
        self.assertEqual(self.https(record["port"], "/v1/info", token)[0], 200)
        self.assert_no_http_response(record["port"])

    def test_key_and_certificate_modes_are_enforced(self):
        loose = self.dir / "loose.key"
        shutil.copyfile(self.key, loose)
        for mode in (0o644, 0o640, 0o604):
            loose.chmod(mode)
            err = self.refused(*self.tls_args(key=loose))
            self.assertIn("--tls-key is accessible by group or others", err)
            self.assertIn("chmod 600", err)
        writable = self.dir / "writable.pem"
        shutil.copyfile(self.cert, writable)
        writable.chmod(0o664)
        self.assertIn("--tls-cert is writable by group or others", self.refused(*self.tls_args(cert=writable)))
        self.ca.chmod(0o666)
        self.assertIn("--tls-client-ca is writable", self.refused(*self.tls_args()))
        self.ca.chmod(0o644)
        other = self.dir / "client.key"
        shutil.copyfile(CERTS / "client.key", other)
        other.chmod(0o600)
        self.assertIn("not a matching certificate and key", self.refused(*self.tls_args(key=other)))
        self.assertIn("could not be read", self.refused(*self.tls_args(key=self.dir / "missing.key")))
        # Read once at start: replacing the files later changes nothing the worker serves.
        _root, token, record = self.launch(*self.tls_args())
        self.key.write_text("replaced")
        self.cert.write_text("replaced")
        self.assertEqual(self.https(record["port"], "/v1/info", token)[1]["certificate_sha256"], SERVER_SHA256)

    def test_loopback_defaults_and_no_parent_watch(self):
        _root, token, record = self.launch()
        self.assertEqual(
            {key: record[key] for key in TRANSPORT},
            {"scheme": "http", "bind": "127.0.0.1", "certificate_sha256": None, "ownership_available": True},
        )
        _root, token, record = self.launch("--no-parent-watch", *self.tls_args())
        self.assertIs(record["ownership_available"], False)
        _root, token, record = self.launch(env={"MEDIA_ENGINE_FIXTURE_NO_PARENT_WATCH": "1"})
        self.assertIs(record["ownership_available"], False)
        _root, token, record = self.launch("--protocol", "media-worker/1.1")
        for key in TRANSPORT:
            self.assertNotIn(key, record)

    def test_create_server_never_serves_plain_http_off_loopback(self):
        worker = Worker(self.dir / "work", TestBackend(), {})
        with self.assertRaises(NetworkError):
            create_server(worker, "x" * 32, 0, "0.0.0.0")
        server = create_server(worker, "x" * 32, 0)
        self.assertEqual(worker.info()["scheme"], "http")
        server.server_close()
        try:
            server = create_server(worker, "x" * 32, 0, "::1")
        except OSError:
            self.skipTest("IPv6 loopback is unavailable here")
        try:
            self.assertEqual(worker.transport["bind"], "::1")
            self.assertEqual(worker.info()["scheme"], "http")
        finally:
            server.server_close()


class ParentWatchTests(unittest.TestCase):
    """The inherited-stdin watch of the real executable is conditional on --no-parent-watch."""

    def run_worker(self, *extra):
        try:
            import media_engine.backend as backend  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("backend module unavailable")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        catalog = Path(temp.name) / "catalog.json"
        catalog.write_text("{}")
        environment = {**os.environ, "PYTHONPATH": str(HERE.parent / "src"),
                       "MEDIA_ENGINE_TOKEN": secrets.token_hex(32), "MEDIA_ENGINE_SUPERVISED": "1"}
        process = subprocess.Popen(
            [sys.executable, "-m", "media_engine.server", "--catalog", str(catalog), "--device", "cpu",
             "--port", "0", "--directory", str(Path(temp.name) / "engine"), *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, text=True,
        )
        self.addCleanup(lambda: (process.poll() is None and process.kill(), process.wait(10),
                                 process.stdout.close(), process.stderr.close()))
        line = process.stdout.readline()
        if not line:
            self.skipTest("the real worker could not start here: " + process.stderr.read()[-200:])
        record = json.loads(line)
        process.stdin.close()
        try:
            process.wait(3 if "--no-parent-watch" in extra else 15)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        return record, exited

    def test_stdin_close_stops_a_supervised_worker(self):
        record, exited = self.run_worker()
        self.assertIs(record["ownership_available"], True)
        self.assertTrue(exited)

    def test_no_parent_watch_ignores_stdin_close(self):
        record, exited = self.run_worker("--no-parent-watch")
        self.assertIs(record["ownership_available"], False)
        self.assertFalse(exited)


if __name__ == "__main__":
    unittest.main()
