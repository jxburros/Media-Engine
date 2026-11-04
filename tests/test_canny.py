import threading
import unittest
from media_engine import png
from media_engine.image_ops import transform
from media_engine.contracts import WorkerError
from media_engine.backend import Cancelled

class CannyTests(unittest.TestCase):
    def test_constant_and_step_edges_and_strict_thresholds(self):
        p=dict(width=64,height=64,operation='canny',threshold_low=.05,threshold_high=.15)
        flat=png.write(64,64,bytes([100])*64*64*3)
        self.assertFalse(any(png.read_rgb(transform(flat,p))[2]))
        step=png.write(64,64,bytes(v for y in range(64) for x in range(64) for v in ([0]*3 if x<32 else [255]*3)))
        a=transform(step,p);self.assertEqual(a,transform(step,p))
        edges=png.read_rgb(a)[2];self.assertGreater(sum(v==255 for v in edges),100)
        self.assertTrue(all(edges[(y*64+x)*3]==0 for y in range(64) for x in range(20)))
        with self.assertRaises(WorkerError):transform(step,{**p,'threshold_low':.5,'threshold_high':.1})
        with self.assertRaises(WorkerError):transform(step,{**p,'width':128})
        cancelled=threading.Event();cancelled.set()
        with self.assertRaises(Cancelled):transform(step,p,cancelled=cancelled)
