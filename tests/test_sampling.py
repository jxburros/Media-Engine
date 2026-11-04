"""Protocol refusals and real reference inference for explicit sampling choices."""
import tempfile
import threading
import unittest
from pathlib import Path

from media_engine.contracts import DEFAULTS, PROTOCOL, WorkerError, validate_parameters, validate_sampling
from media_engine.sampling import SAMPLING, advertised, create
from test_inference import HAVE_INFERENCE, tiny_sdxl_bundle


def request(pair, protocol=PROTOCOL):
    return {"protocol": protocol, "attempt_id": "attempt_sampling", "bundle_digest": "sha256:" + "a" * 64,
            "task": "text-to-image", "inputs": {"prompt": "test"},
            "parameters": {**DEFAULTS, "width": 64, "height": 64, "seed": 7, "steps": 2,
                           "sampler": pair[0], "scheduler": pair[1]}}


class SamplingContractTests(unittest.TestCase):
    def test_pairs_require_new_protocol_and_advertisement(self):
        for pair in SAMPLING:
            body = request(pair)
            self.assertEqual(validate_parameters(body), "text-to-image")
            validate_sampling(body, {"sampling": advertised(True)})
            if pair != SAMPLING[0]:
                with self.assertRaises(WorkerError):
                    validate_parameters(request(pair, "media-worker/1.3"))
                with self.assertRaises(WorkerError):
                    validate_sampling(body, {"sampling": advertised(False)})
        with self.assertRaises(WorkerError):
            validate_parameters(request(("euler", "karras")))


@unittest.skipUnless(HAVE_INFERENCE, "inference extra is required")
class SamplingInferenceTests(unittest.TestCase):
    def test_reference_executes_every_pair_without_state_leakage(self):
        from media_engine.backend import DiffusersBackend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _ = tiny_sdxl_bundle(root)
            backend = DiffusersBackend("cpu")
            backend.load(model, {"precision": "float32", "attention": "sdpa", "offload": "none"})
            images = []
            for i, pair in enumerate((*SAMPLING, SAMPLING[0])):
                output = root / str(i); output.mkdir()
                records, _ = backend.generate(request(pair), output, lambda *_: None, threading.Event())
                self.assertEqual(len(records), 1)
                images.append(records[0]["path"].read_bytes())
            self.assertEqual(images[0], images[-1], "switching samplers must not mutate bundle scheduler settings")
            self.assertGreater(len(set(images)), 1)
            backend.unload()
