import tempfile
import threading
import unittest
from pathlib import Path
from media_engine import mp4, png
from media_engine.contracts import PROTOCOL, WorkerError, output_ids, validate_parameters


@unittest.skipUnless(mp4.available(), "inference-video extra required")
class MP4Tests(unittest.TestCase):
    def test_encode_decode_metadata_and_cancellation(self):
        from media_engine.backend import Cancelled
        frames = [png.write(64, 64, bytes(color) * 4096) for color in ((255, 0, 0), (0, 0, 255))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.mp4"
            mp4.write(path, frames, 7, threading.Event())
            self.assertEqual(mp4.read_metadata(path), {"width": 64, "height": 64, "frames": 2, "fps": 7, "duration_seconds": 2/7})
            cancelled = threading.Event(); cancelled.set()
            with self.assertRaises(Cancelled): mp4.write(path, frames, 7, cancelled)
            self.assertFalse(path.exists())

    def test_protocol_gate(self):
        body = {"protocol": PROTOCOL, "task": "image-to-video", "attempt_id": "attempt_video",
                "bundle_digest": "sha256:" + "a" * 64, "inputs": {"image": "input_" + "a" * 32},
                "parameters": {"width": 64, "height": 64, "seed": 1, "steps": 2, "encode_video": True}}
        validate_parameters(body)
        self.assertEqual(output_ids(body["task"], body["parameters"]), ["frames", "poster", "video"])
        body["protocol"] = "media-worker/1.3"
        with self.assertRaises(WorkerError): validate_parameters(body)
