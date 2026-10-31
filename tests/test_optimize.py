"""Targeted optimization options (Engine 0.8.0, phase E5): advertising, validation, identity."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from fixture import TestBackend, make_adapter, make_bundle, make_esrgan_bundle
from media_engine import optimize
from media_engine.contracts import PROTOCOL, WorkerError
from media_engine.server import Worker

CPU = {"kind": "cpu", "architecture": "x86_64"}


def entry(option, value, backend="diffusers-sdxl", klass=CPU, families=("sdxl-base",)):
    return {
        "option": option,
        "value": value,
        "backend_ids": [backend],
        "device_class": dict(klass),
        "families": list(families),
        "benchmark": "docs/benchmarks/example.json",
        "benchmark_sha256": "a" * 64,
        "determinism": "deterministic on one device and build",
        "quality_note": "within one 8-bit level of float32",
        "improvement": "1.3x faster sampling",
    }


def nothing_missing(_dependencies):
    return []


class SurveyTests(unittest.TestCase):
    """The advertising rule: importable, runnable on the device, and benchmarked for its class."""

    def setUp(self):
        self.klass = optimize.device_class("cpu")

    def survey(self, index, families=("sdxl-base", "esrgan"), missing=nothing_missing, backend="diffusers-sdxl"):
        return optimize.survey(backend, "cpu", list(families), index, missing)

    def reasons(self, unadvertised):
        return {(u["option"], u["value"]): u["reason"] for u in unadvertised}

    def test_nothing_is_advertised_without_a_benchmark(self):
        advertised, unadvertised = self.survey([])
        self.assertEqual(advertised, [])
        reasons = self.reasons(unadvertised)
        self.assertEqual(reasons[("precision", "bfloat16")], "no_benchmark")
        self.assertEqual(reasons[("attention", "flash")], "device_unsupported")
        self.assertEqual(reasons[("quantization", "int8-weight-only")], "no_benchmark")
        self.assertIn(reasons[("compile", "inductor")], ("no_benchmark", "device_unsupported"))

    def test_a_matching_benchmark_advertises_with_its_citation(self):
        index = [entry("precision", "bfloat16", klass=self.klass)]
        advertised, _ = self.survey(index)
        self.assertEqual(len(advertised), 1)
        record = advertised[0]
        self.assertEqual((record["option"], record["value"]), ("precision", "bfloat16"))
        self.assertEqual(record["families"], ["sdxl-base"])  # only the families benchmarked
        for field in ("benchmark", "benchmark_sha256", "determinism", "quality_note", "improvement", "effect"):
            self.assertTrue(record[field], field)

    def test_benchmarks_are_per_backend_and_device_class(self):
        other_backend = [entry("precision", "bfloat16", backend="owned-sdxl", klass=self.klass)]
        self.assertEqual(self.survey(other_backend)[0], [])
        gpu = [entry("precision", "bfloat16", klass={"kind": "cuda", "compute_capability": "8.9"})]
        self.assertEqual(self.survey(gpu)[0], [])
        arm = [entry("precision", "bfloat16", klass={"kind": "cpu", "architecture": "not-this-one"})]
        self.assertEqual(self.survey(arm)[0], [])

    def test_missing_dependency_and_family_scope(self):
        index = [entry("quantization", "int8-weight-only", klass=self.klass)]
        _, unadvertised = self.survey(index, missing=lambda deps: [name for name, _ in deps])
        failed = [u for u in unadvertised if u["option"] == "quantization"][0]
        self.assertEqual((failed["reason"], failed["missing"]), ("missing_dependency", ["torchao"]))
        # esrgan is convolutional: neither attention nor int8 weight-only applies to it.
        advertised, unadvertised = self.survey(index, families=("esrgan",))
        self.assertEqual(advertised, [])
        self.assertNotIn("quantization", [u["option"] for u in unadvertised])
        self.assertNotIn("attention", [u["option"] for u in unadvertised])

    def test_index_must_be_well_formed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.json"
            path.write_text("{not json")
            self.assertEqual(optimize.read_index(path), [])
            path.write_text(json.dumps({"schema_version": 2, "entries": [entry("precision", "bfloat16")]}))
            self.assertEqual(optimize.read_index(path), [])
            bad = entry("precision", "bfloat16")
            del bad["benchmark"]
            path.write_text(json.dumps({"schema_version": 1, "entries": [bad, entry("precision", "float8")]}))
            self.assertEqual(optimize.read_index(path), [])
            path.write_text(json.dumps({"schema_version": 1, "entries": [entry("precision", "bfloat16")]}))
            self.assertEqual(len(optimize.read_index(path)), 1)

    def test_the_packaged_index_advertises_nothing_in_0_8_0(self):
        # No benchmark has been recorded on the reference hardware; see docs/compatibility.md.
        index = json.loads(optimize.INDEX_PATH.read_text())
        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(index["entries"], [])
        self.assertEqual(index["reference_hardware"]["compute_capability"], "8.9")


class OptimizationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = make_bundle(self.root / "model")
        self.upscaler = make_esrgan_bundle(self.root / "upscaler")
        self.adapter = make_adapter(self.root / "lora")
        self.catalog = {
            "fixture": str(self.root / "model"),
            "upscaler": str(self.root / "upscaler"),
            "fixture-lora": str(self.root / "lora"),
        }
        self.backend = TestBackend(precisions=("float16", "float32"))
        self.worker = Worker(self.root / "work", self.backend, self.catalog)

    def tearDown(self):
        while self.worker.active or self.worker.state in {"loading", "unloading"}:
            time.sleep(0.01)
        self.temp.cleanup()

    def body(self, protocol=PROTOCOL, bundle=None, **options):
        return {
            "protocol": protocol,
            "bundle": bundle or self.bundle,
            "options": {"precision": "float16", "attention": "sdpa", "offload": "none", **options},
        }

    def code(self, call, *args):
        with self.assertRaises(WorkerError) as caught:
            call(*args)
        return caught.exception.code, caught.exception.field

    def load(self, body):
        op = self.worker.control("load", body)
        while self.worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.01)
        record = self.worker.operations[op["operation_id"]]
        self.assertEqual(record["state"], "succeeded", record["error"])
        return record

    def test_capabilities_advertise_each_with_citation_and_reasons(self):
        caps = self.worker.capabilities()
        pairs = [(o["option"], o["value"]) for o in caps["optimizations"]]
        self.assertEqual(
            pairs,
            [("precision", "bfloat16"), ("quantization", "int8-weight-only"), ("compile", "inductor")],
        )
        for record in caps["optimizations"]:
            for field in ("benchmark", "determinism", "quality_note", "families", "effect"):
                self.assertIn(field, record)
        self.assertEqual(
            caps["unadvertised_optimizations"],
            [{"option": "attention", "value": "flash", "reason": "device_unsupported"}],
        )
        # Never an automatic option, and never an advertised precision for auto to pick.
        self.assertEqual(caps["auto_options"], ["precision"])
        self.assertNotIn("bfloat16", caps["precisions"])
        pinned = TestBackend(protocols=("media-worker/1.0", "media-worker/1.1", "media-worker/1.2"))
        self.assertNotIn("optimizations", pinned.capabilities())

    def test_selected_optimizations_are_load_identity(self):
        options = {"precision": "bfloat16", "quantization": "int8-weight-only", "compile": "inductor"}
        validation = self.worker.validation(self.body(**options))
        self.assertEqual(validation["resolved_options"]["quantization"], "int8-weight-only")
        notes = {n["option"] for n in validation["memory_estimate"]["optimizations"]}
        self.assertEqual(notes, {"quantization", "compile"})
        self.assertEqual(validation["memory_estimate"]["precision"], "bfloat16")
        op = self.load(self.body(**options))
        self.assertEqual(op["resolved_options"]["compile"], "inductor")
        loaded = self.worker.info()["loaded_bundle"]
        self.assertEqual(loaded["options"]["quantization"], "int8-weight-only")
        self.assertEqual(self.backend.loaded_options["precision"], "bfloat16")
        # A baseline load is a different identity: a fresh load, recorded as such.
        self.load(self.body())
        self.assertEqual(self.backend.loads, 2)
        self.assertNotIn("quantization", self.worker.info()["loaded_bundle"]["options"])
        # Explicit "none" is the baseline value and needs nothing advertised.
        self.worker.validation(self.body(quantization="none", compile="none"))

    def test_older_callers_and_unadvertised_values_are_refused(self):
        older = "media-worker/1.2"
        # A 1.2 body keeps the three-field shape: the new fields are unknown fields.
        self.assertEqual(self.code(self.worker.validate_load, self.body(older, quantization="int8-weight-only")), ("invalid_request", None))
        self.assertEqual(self.code(self.worker.validate_load, self.body(older, quantization="none")), ("invalid_request", None))
        # A new value of an existing option is refused by name, as before.
        self.assertEqual(self.code(self.worker.validate_load, self.body(older, precision="bfloat16")), ("unsupported", "options"))
        self.assertEqual(self.code(self.worker.validate_load, self.body(attention="flash")), ("unsupported", "options"))
        self.assertEqual(self.code(self.worker.validate_load, self.body(quantization="int4")), ("unsupported", "options"))
        self.assertEqual(self.code(self.worker.validate_load, self.body(compile="eager")), ("unsupported", "options"))
        none = TestBackend(optimizations=())
        worker = Worker(self.root / "none", none, self.catalog)
        self.assertEqual(worker.capabilities()["optimizations"], [])
        self.assertEqual(self.code(worker.validate_load, self.body(precision="bfloat16", **{})), ("unsupported", "options"))

    def test_combinations_the_worker_cannot_honor(self):
        # int8 weight-only applies to attention families only, not the convolutional upscaler.
        upscaler = self.body(bundle=self.upscaler, quantization="int8-weight-only")
        self.assertEqual(self.code(self.worker.validate_load, upscaler), ("unsupported", "options"))
        self.worker.validation(self.body(bundle=self.upscaler, compile="inductor"))
        # Adapters are not combined with int8 weight-only in 0.8.0.
        adapters = dict(self.body(quantization="int8-weight-only"), adapters=[{"adapter": self.adapter, "scale": 1}])
        self.assertEqual(self.code(self.worker.validate_load, adapters), ("unsupported", "adapters"))
        adapters["options"] = self.body(compile="inductor")["options"]
        self.worker.validation(adapters)
        # A benchmark is valid for one device class: no CPU fallback with an optimization.
        fallback = dict(self.body(compile="inductor"), authorized_fallbacks=["cpu"])
        self.assertEqual(self.code(self.worker.validate_load, fallback), ("unsupported", "authorized_fallbacks"))
        self.worker.validation(dict(self.body(compile="inductor"), authorized_fallbacks=["lower_precision"]))

    def test_auto_never_selects_an_optimization(self):
        result = self.worker.validation(self.body(precision="auto", compile="inductor"))
        self.assertIn(result["resolved_options"]["precision"], ("float16", "float32"))
        self.assertEqual(result["resolved_options"]["compile"], "inductor")


if __name__ == "__main__":
    unittest.main()
