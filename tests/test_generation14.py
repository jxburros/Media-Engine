import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from fixture import make_adapter, make_esrgan_bundle, make_png
from media_engine.contracts import PROTOCOL, WorkerError, adapter_path, bundle_path, digest, validate_parameters, DEFAULTS
from test_inference import HAVE_INFERENCE, tiny_sdxl_bundle


class FileCatalogTests(unittest.TestCase):
    def test_adapter_file_does_not_scan_neighbors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = make_adapter(root)
            file = next(root.glob('*.safetensors'))
            # A directory registration includes every file. A file registration includes just its file.
            adapter['files'] = {file.name: adapter['files'][file.name]}
            adapter['digest'] = digest({k:v for k,v in adapter.items() if k!='digest'})
            (root/'neighbor.pth').write_bytes(b'not part of this handle')
            self.assertEqual(adapter_path(adapter, {'fixture-lora':str(file)})[1], file)
            with self.assertRaises(WorkerError): adapter_path(adapter, {'fixture-lora':str(root)})

    def test_esrgan_file_uses_only_its_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = make_esrgan_bundle(root)
            file = next(root.glob('*.safetensors'))
            shutil.copyfile(root/'config.json', str(file)+'.json')
            (root/'neighbor.pth').write_bytes(b'not part of this handle')
            self.assertEqual(bundle_path(bundle, {'upscaler':str(file)}), file)
            with self.assertRaises(WorkerError): bundle_path(bundle, {'upscaler':str(file)}, version='media-worker/1.3')

    def test_selected_snapshot_adapter_retains_logical_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)/'models--approved--adapter'
            snapshot=repo/'snapshots'/('a'*40);snapshot.mkdir(parents=True)
            blobs=repo/'blobs';blobs.mkdir()
            record=make_adapter(snapshot)
            selected=next(snapshot.glob('*.safetensors'))
            record['files']={selected.name:record['files'][selected.name]}
            record['digest']=digest({k:v for k,v in record.items() if k!='digest'})
            blob=blobs/'opaquehash';selected.replace(blob);selected.symlink_to(blob)
            self.assertEqual(adapter_path(record,{'fixture-lora':str(selected)})[1],selected)

    def test_hugging_face_snapshot_links_are_confined_to_its_blob_store(self):
        from media_engine.contracts import approved_snapshot_link
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)/'models--approved--model'
            snapshot=repo/'snapshots'/('a'*40);snapshot.mkdir(parents=True)
            blobs=repo/'blobs';blobs.mkdir()
            blob=blobs/'hash';blob.write_bytes(b'weights')
            link=snapshot/'model.safetensors';link.symlink_to(blob)
            self.assertTrue(approved_snapshot_link(link,snapshot))
            outside=Path(temp)/'other';outside.write_bytes(b'outside')
            link.unlink();link.symlink_to(outside)
            self.assertFalse(approved_snapshot_link(link,snapshot))
            link.unlink();link.symlink_to(blobs,target_is_directory=True)
            self.assertFalse(approved_snapshot_link(link,snapshot))
            link.unlink();blob.unlink();blobs.rmdir()
            external=Path(temp)/'external-blobs';external.mkdir();(external/'hash').write_bytes(b'weights')
            blobs.symlink_to(external,target_is_directory=True);link.symlink_to(blobs/'hash')
            self.assertFalse(approved_snapshot_link(link,snapshot))


@unittest.skipUnless(HAVE_INFERENCE, 'inference extra required')
class InpaintInferenceTests(unittest.TestCase):
    def test_real_inpaint_with_durable_mask(self):
        from media_engine.backend import DiffusersBackend
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); model,_=tiny_sdxl_bundle(root)
            backend=DiffusersBackend('cpu')
            backend.load(model,{'precision':'float32','attention':'sdpa','offload':'none'})
            image=root/'image.png'; mask=root/'mask.png'
            image.write_bytes(make_png(64,64,(20,40,60)))
            mask.write_bytes(make_png(64,64,(255,255,255)))
            body={'protocol':PROTOCOL,'attempt_id':'attempt_inpaint','bundle_digest':'sha256:'+'a'*64,
                  'task':'inpaint','inputs':{'prompt':'test','image':'input_'+'a'*32,'mask':'input_'+'b'*32},
                  'parameters':{**DEFAULTS,'width':64,'height':64,'steps':2,'seed':7,'strength':1.0}}
            validate_parameters(body)
            output=root/'output';output.mkdir()
            records,_=backend.generate(body,output,lambda *_:None,threading.Event(),{'image':image,'mask':mask})
            self.assertEqual(records[0]['mime_type'],'image/png')
            self.assertNotEqual(records[0]['path'].read_bytes(),image.read_bytes())
            body['protocol']='media-worker/1.3'
            with self.assertRaises(WorkerError):validate_parameters(body)
            backend.unload()
