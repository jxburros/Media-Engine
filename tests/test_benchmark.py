"""tools/benchmark.py on the fixture worker: the document shape, differences and qualification."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from fixture import TestBackend, make_audio_bundle, make_bundle
from media_engine.server import Worker

TOOL = Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_tool", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class BenchmarkToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.audio = make_audio_bundle(self.root / "audio")
        self.worker = Worker(
            self.root / "work",
            TestBackend(),
            {"fixture": str(self.root / "model"), "audio": str(self.root / "audio")},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_document_records_every_option_set_against_the_baseline(self):
        base = {"precision": "float32", "attention": "sdpa", "offload": "none"}
        sets = [base, {**base, "precision": "bfloat16"}, {**base, "compile": "inductor"}]
        requests = tool.request_set("sdxl-base", (64, 64), 2, 1)
        document = tool.benchmark(self.worker, self.bundle, "sdxl-base", sets, 1, requests)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual([r["options"] for r in document["results"]], sets)
        baseline, bf16, compiled = document["results"]
        self.assertEqual(len(baseline["outputs"]), 3)  # three seeds
        self.assertEqual(len(baseline["execution_seconds"]), 1)
        self.assertNotIn("summary", baseline)
        for result in (bf16, compiled):
            # The fixture's pixels depend on the seed only: identical to the baseline.
            for outputs in result["summary"]["differences"]:
                self.assertEqual(outputs[0]["max"], 0)
                self.assertEqual(outputs[0]["unit"], "8-bit levels")
            self.assertIn("qualifies", result["summary"])
        self.assertTrue(all("bytes" not in o for r in document["results"] for outs in r["outputs"] for o in outs))
        json.dumps(document)  # serializable as written

    def test_audio_differences_and_index_entries(self):
        base = {"precision": "float32", "attention": "sdpa", "offload": "none"}
        sets = [base, {**base, "quantization": "int8-weight-only"}]
        requests = tool.request_set("stable-audio-open", None or (0, 0), 2, 1)
        document = tool.benchmark(self.worker, self.audio, "stable-audio-open", sets, 1, requests, min_speedup=0, max_memory_ratio=0)
        difference = document["results"][1]["summary"]["differences"][0][0]
        self.assertEqual((difference["unit"], difference["max"]), ("16-bit steps", 0))
        path = self.root / "bench.json"
        path.write_text(json.dumps(document))
        entries = tool.index_entries(document, path, ["stable-audio-open"])
        self.assertEqual([(e["option"], e["value"]) for e in entries], [("quantization", "int8-weight-only")])
        self.assertEqual(len(entries[0]["benchmark_sha256"]), 64)
        # A release still reviews determinism and quality before citing; the tool says so.
        self.assertTrue(entries[0]["determinism"].startswith("REVIEW"))

    def test_unknown_option_is_refused(self):
        with self.assertRaises(Exception):
            tool.parse_options("precision=float8")


if __name__ == "__main__":
    unittest.main()
