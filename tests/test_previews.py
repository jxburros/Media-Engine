import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fixture import TestBackend, make_bundle
from media_engine import png
from media_engine.contracts import DEFAULTS, PROTOCOL, WorkerError
from media_engine.server import Worker


class PreviewTests(unittest.TestCase):
    def test_preview_handles_are_bounded_private_and_expire(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = make_bundle(root / "model")
            worker = Worker(root / "worker", TestBackend(), {"fixture": str(root / "model")})
            op = worker.control("load", {"protocol": PROTOCOL, "bundle": bundle, "options": worker.capabilities()["load_defaults"]})
            deadline = time.monotonic() + 5
            while worker.operations[op["operation_id"]]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(.01)
            body = {"protocol": PROTOCOL, "attempt_id": "attempt_previews", "bundle_digest": bundle["bundle_digest"], "task": "text-to-image", "inputs": {"prompt": "test"}, "parameters": {**DEFAULTS, "seed": 1, "width": 64, "height": 64, "steps": 30, "preview_interval": 1}}
            execution = worker.submit(body)["execution_id"]
            seen = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                snapshot = worker.snapshot(execution)
                events = [e for e in snapshot["events"] if e["type"] == "preview"]
                if len(events) >= 6 and snapshot["state"] == "running":
                    seen = events
                    self.assertNotIn("png", events[-1]["data"])
                    self.assertLessEqual(len(worker.previews), 4)
                    data = worker.preview(events[-1]["data"]["preview_id"])
                    self.assertEqual(png.header(data)[:2], (16,16))
                    with self.assertRaises(WorkerError) as caught:
                        worker.preview(events[0]["data"]["preview_id"])
                    self.assertEqual(caught.exception.status,410)
                    worker.cancel(execution)
                    break
                time.sleep(.01)
            self.assertTrue(seen)
            while worker.snapshot(execution)["state"] == "running" and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(worker.previews,{})
            self.assertEqual(worker.snapshot(execution)["outputs"],[])


@unittest.skipUnless(importlib.util.find_spec("torch"), "inference extra is not installed")
class EncodingCacheTests(unittest.TestCase):
    def test_cache_clones_results_bounds_memory_and_clears_load_identity(self):
        import torch
        from media_engine.conditioning_cache import TextCache
        cache = TextCache(max_bytes=32,max_entries=2)
        calls=[]
        def compute():
            calls.append(1)
            return (torch.arange(4,dtype=torch.float32),None)
        result=cache.get(["prompt"],compute)
        result[0].zero_()
        cached=cache.get(["prompt"],compute)
        self.assertEqual(cached[0].tolist(),[0,1,2,3])
        self.assertEqual(len(calls),1)
        cache.get(["second"],compute)
        cache.get(["third"],compute)
        self.assertLessEqual(cache.bytes,32)
        self.assertEqual(len(cache.entries),2)
        cache.get(["prompt"],compute)
        self.assertEqual(len(calls),4)
        cache.clear()
        self.assertEqual(cache.bytes,0)
        cache.get(["prompt"],compute)
        self.assertEqual(len(calls),5)

    def test_latent_preview_does_not_change_tensor_or_rng(self):
        import torch
        from media_engine.previews import emit_latents
        torch.manual_seed(3)
        tensor=torch.randn(1,4,256,256)
        before=tensor.clone()
        state=torch.get_rng_state().clone()
        events=[]
        emit_latents(tensor,lambda kind,data:events.append((kind,data)),2,1)
        self.assertEqual(events,[])
        emit_latents(tensor,lambda kind,data:events.append((kind,data)),2,2)
        self.assertTrue(torch.equal(tensor,before))
        self.assertTrue(torch.equal(torch.get_rng_state(),state))
        self.assertEqual(png.header(events[0][1]["png"])[:2],(128,128))
