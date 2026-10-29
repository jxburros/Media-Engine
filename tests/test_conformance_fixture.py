"""Run the conformance suite against tests/serve_fixture.py, so the suite itself is tested in CI."""

import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tests.conformance.run import run_suite  # noqa: E402


class ConformanceFixtureTests(unittest.TestCase):
    def launch(self, *extra, env=None):
        root = Path(self.temp.name) / f"root{len(self.processes)}"
        token = secrets.token_hex(32)
        environment = {
            **os.environ,
            "PYTHONPATH": str(HERE.parent / "src"),
            "MEDIA_ENGINE_TOKEN": token,
            "MEDIA_ENGINE_FIXTURE_MAX_RETAINED": "10",
            **(env or {}),
        }
        process = subprocess.Popen(
            [sys.executable, str(HERE / "serve_fixture.py"), str(root), *extra],
            stdout=subprocess.PIPE,
            env=environment,
            text=True,
        )
        self.processes.append(process)
        record = json.loads(process.stdout.readline())
        return root, token, record

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.processes = []
        self.saved = dict(os.environ)

    def tearDown(self):
        for process in self.processes:
            process.terminate()
            process.wait(10)
            process.stdout.close()
        os.environ.clear()
        os.environ.update(self.saved)
        self.temp.cleanup()

    def suite(self, root, token, record, startup=True, registration=True):
        os.environ["MEDIA_ENGINE_URL"] = f"http://127.0.0.1:{record['port']}"
        os.environ["MEDIA_ENGINE_TOKEN"] = token
        for key, value in (
            ("MEDIA_ENGINE_STARTUP_RECORD", json.dumps(record) if startup else None),
            ("MEDIA_ENGINE_REGISTRATION", str(root / "conformance.json") if registration else None),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return run_suite(stream=io.StringIO())

    def test_fixture_is_conformant_at_1_3(self):
        root, token, record = self.launch()
        self.assertEqual(record["highest_protocol"], "media-worker/1.3")
        self.assertEqual(record["protocol"], "media-worker/1.1")
        conformant, sections = self.suite(root, token, record)
        self.assertTrue(conformant, sections)

    def test_fixture_is_conformant_pinned_to_1_2(self):
        root, token, record = self.launch("--protocol", "media-worker/1.2")
        self.assertEqual(record["supported_protocols"][-1], "media-worker/1.2")
        self.assertEqual(record["highest_protocol"], "media-worker/1.2")
        self.assertIn("device_identity", record)
        conformant, sections = self.suite(root, token, record)
        self.assertTrue(conformant, sections)

    def test_fixture_is_conformant_pinned_to_1_1(self):
        root, token, record = self.launch("--protocol", "media-worker/1.1")
        self.assertEqual(record["supported_protocols"], ["media-worker/1.0", "media-worker/1.1"])
        self.assertNotIn("device_identity", record)
        conformant, sections = self.suite(root, token, record)
        self.assertTrue(conformant, sections)

    def test_a_skipped_section_is_never_conformant(self):
        root, token, record = self.launch()
        conformant, sections = self.suite(root, token, record, startup=False)
        self.assertFalse(conformant)
        self.assertEqual(sections["discovery"], "skip")
        conformant, sections = self.suite(root, token, record, registration=False)
        self.assertFalse(conformant)

    def test_fixture_options(self):
        root, token, record = self.launch(
            env={
                "MEDIA_ENGINE_FIXTURE_DEVICE_UNAVAILABLE": "1",
                "MEDIA_ENGINE_FIXTURE_DEVICE_INDEX": "3",
                "MEDIA_ENGINE_FIXTURE_HOST_ID": "host-b",
            }
        )
        identity = record["device_identity"]
        self.assertEqual(identity["index"], 3)
        catalog = json.loads((root / "catalog.json").read_text())
        self.assertEqual(set(catalog), {"fixture", "fixture-lora", "upscaler", "converted", "checkpoint", "video", "audio"})
        self.assertEqual(
            [p.name for p in (root / "checkpoint").iterdir()], ["checkpoint.safetensors"]
        )
        registrations = json.loads((root / "registrations.json").read_text())
        self.assertEqual(registrations["upscaler"]["format"], "single-file-safetensors")
        self.assertIn("conversion", registrations["converted"]["source"])
        self.assertEqual(registrations["video"]["family"], "svd")
        self.assertEqual(registrations["video"]["pipeline"], "image-to-video")
        self.assertEqual(registrations["audio"]["family"], "stable-audio-open")
        self.assertEqual(registrations["audio"]["pipeline"], "text-to-audio")
        _root, _token, other = self.launch()
        self.assertNotEqual(other["device_identity"]["host_id"], identity["host_id"])


if __name__ == "__main__":
    unittest.main()
