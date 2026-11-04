import threading
import unittest
from media_engine import image_ops,png
from media_engine.contracts import WorkerError

class ImageOperationTests(unittest.TestCase):
    def test_explicit_geometry_and_pixels(self):
        red=png.write(64,64,bytes((255,0,0))*4096)
        padded=image_ops.transform(red,{'operation':'pad','width':128,'height':128,'x':32,'y':32,'background':[0,0,255]})
        w,h,data=png.read_rgb(padded)
        self.assertEqual((w,h),(128,128));self.assertEqual(data[:3],bytes((0,0,255)))
        cropped=image_ops.transform(padded,{'operation':'crop','width':64,'height':64,'x':32,'y':32})
        self.assertEqual(png.read_rgb(cropped),png.read_rgb(red))
        blue=png.write(64,64,bytes((0,0,255))*4096)
        composite=image_ops.transform(red,{'operation':'composite','width':64,'height':64,'opacity':0.5},blue)
        self.assertEqual(png.read_rgb(composite)[2][:3],bytes((128,0,128)))
        for filter in ('nearest','bilinear'):
            resized=image_ops.transform(red,{'operation':'resize','width':128,'height':128,'filter':filter})
            self.assertEqual(png.read_rgb(resized)[2],bytes((255,0,0))*128*128)
    def test_rejects_implicit_filter_clipping_and_cancel(self):
        from media_engine.backend import Cancelled
        image=png.write(64,64,bytes(64*64*3))
        for params in ({'operation':'resize','width':128,'height':128},{'operation':'crop','width':128,'height':128},{'operation':'pad','width':64,'height':64,'x':1}):
            with self.assertRaises(WorkerError):image_ops.transform(image,params)
        cancel=threading.Event();cancel.set()
        with self.assertRaises(Cancelled):image_ops.transform(image,{'operation':'resize','width':64,'height':64,'filter':'nearest'},cancelled=cancel)
