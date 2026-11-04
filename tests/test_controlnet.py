import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from test_inference import HAVE_INFERENCE,tiny_sdxl_bundle
from fixture import make_png

@unittest.skipUnless(HAVE_INFERENCE,'inference extra needed')
class ControlNetTests(unittest.TestCase):
    def test_verified_component_changes_real_inference(self):
        import torch
        from diffusers import ControlNetModel,UNet2DConditionModel
        from media_engine.backend import DiffusersBackend
        from media_engine import controlnet
        from media_engine.contracts import DEFAULTS,PROTOCOL,WorkerError,digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,bundle=tiny_sdxl_bundle(root)
            unet=UNet2DConditionModel.from_pretrained(model/'unet',local_files_only=True)
            net=ControlNetModel.from_unet(unet,conditioning_embedding_out_channels=(16,32))
            # from_unet intentionally zeroes its residual projections; make the tiny test signal nonzero.
            torch.manual_seed(4)
            for block in [*net.controlnet_down_blocks,net.controlnet_mid_block,net.controlnet_cond_embedding.conv_out]:
                torch.nn.init.normal_(block.weight,std=0.02)
            path=root/'controlnet';net.save_pretrained(path,safe_serialization=True)
            record=dict(schema_version=1,controlnet_id='tiny-control',artifact_handle='control',family='sdxl-base',source={'license':'test-only random weights'},
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})
            record['digest']=digest(record)
            image=root/'control.png';image.write_bytes(make_png(64,64,(120,60,30)))
            backend=DiffusersBackend('cpu');backend.load(model,{'precision':'float32','attention':'sdpa','offload':'none'})
            request=dict(protocol=PROTOCOL,attempt_id='attempt_control',bundle_digest='sha256:'+'a'*64,task='text-to-image',inputs={'prompt':'test','control_image':'input_'+'b'*32},
                parameters={**DEFAULTS,'width':64,'height':64,'steps':2,'seed':3})
            images=[]
            for scale in (0,1):
                resolved=controlnet.validate([{'controlnet':record,'scale':scale}],{'control':str(path)},PROTOCOL)
                controlnet.attach(backend,resolved)
                out=root/str(scale);out.mkdir()
                records,_=backend.generate(request,out,lambda *_:None,threading.Event(),{'control_image':image})
                images.append(records[0]['path'].read_bytes())
            self.assertNotEqual(*images)
            from media_engine.server import Worker
            worker=Worker(root/'worker',backend,{'tiny':str(model),'control':str(path)})
            load={'protocol':PROTOCOL,'bundle':bundle,'options':{'precision':'float32','attention':'sdpa','offload':'none'},'controlnets':[{'controlnet':record,'scale':1}]}
            operation=worker.control('load',load)
            deadline=time.monotonic()+10
            while worker.operations[operation['operation_id']]['state']=='running' and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(worker.operations[operation['operation_id']]['state'],'succeeded')
            input_record=worker.store_input(image.read_bytes())
            request['inputs']['control_image']=input_record['input_id'];request['bundle_digest']=bundle['bundle_digest']
            self.assertTrue(worker.validate_execution(request)['valid'])
            execution=worker.submit(request)['execution_id']
            while worker.snapshot(execution)['state']=='running' and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(worker.snapshot(execution)['state'],'succeeded')
            self.assertEqual(worker.loaded['controlnets'][0]['digest'],record['digest'])
            del request['inputs']['control_image']
            with self.assertRaises(WorkerError):worker.validate_execution(request)
            (path/'config.json').write_text('{}')
            with self.assertRaises(WorkerError):controlnet.validate([{'controlnet':record,'scale':1}],{'control':str(path)},PROTOCOL)
            backend.unload();self.assertIsNone(backend.controlnet)


class ControlNetArtifactTests(unittest.TestCase):
    def test_malformed_local_artifacts_fail_with_client_errors(self):
        from fixture import make_controlnet,file_digests
        from media_engine.controlnet import validate
        from media_engine.contracts import digest,PROTOCOL,WorkerError
        for name,data in [('config.json',b'[]'),('config.json',b'null'),('diffusion_pytorch_model.safetensors',b'bad')]:
            with self.subTest(name=name,data=data),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);record=make_controlnet(root)
                (root/name).write_bytes(data);record['files']=file_digests(root)
                record['digest']=digest({k:v for k,v in record.items() if k!='digest'})
                with self.assertRaises(WorkerError) as caught:validate([{'controlnet':record,'scale':1}],{'fixture-controlnet':str(root)},PROTOCOL)
                self.assertEqual(caught.exception.status,400)
