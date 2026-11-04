"""Local artifact refusal plus actual tiny SDXL conditioning; no network or model downloads."""
import copy
import hashlib
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from fixture import make_png
from media_engine import conditioning
from media_engine.contracts import DEFAULTS, PROTOCOL, WorkerError, digest, validate_parameters, validate_sampling
from test_inference import HAVE_INFERENCE, tiny_sdxl_bundle


def descriptor(root, kind, identity=None):
    record = dict(schema_version=1, conditioning_id=identity or kind, kind=kind,
        artifact_handle=identity or kind, source={"license":"test-only random weights"},
        files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()})
    record['digest']=digest(record)
    return record


def request(**parameters):
    return dict(protocol=PROTOCOL,attempt_id='attempt_conditioning',bundle_digest='sha256:'+'a'*64,
        task='text-to-image',inputs={'prompt':'test'},parameters={**DEFAULTS,'width':64,'height':64,'steps':2,'seed':4,**parameters})


class ConditioningContractTests(unittest.TestCase):
    def test_manifest_identity_old_minor_hash_and_duplicate_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            header=json.dumps({'embedding':{'dtype':'F32','shape':[32],'data_offsets':[0,128]}}).encode()
            (root/'embedding.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*128)
            record=descriptor(root,'textual-inversion')
            use={'component':record,'token':'<test>','encoder':'text_encoder'}
            catalog={'textual-inversion':str(root)}
            self.assertEqual(conditioning.validate([use],catalog,PROTOCOL)[0][1],root)
            for uses,version in (([use],'media-worker/1.3'),([use,use],PROTOCOL)):
                with self.assertRaises(WorkerError):conditioning.validate(uses,catalog,version)
            bad=copy.deepcopy(use);bad['component']['source']['license']='changed'
            with self.assertRaises(WorkerError):conditioning.validate([bad],catalog,PROTOCOL)
            (root/'embedding.safetensors').write_bytes(b'changed')
            with self.assertRaises(WorkerError):conditioning.validate([use],catalog,PROTOCOL)

    def test_unlisted_files_symlinks_and_pickle_never_enter_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'embedding.safetensors').symlink_to('/etc/hosts')
            record=descriptor(root,'textual-inversion');use={'component':record,'token':'<test>','encoder':'text_encoder'}
            with self.assertRaises(WorkerError):conditioning.validate([use],{'textual-inversion':str(root)},PROTOCOL)
            (root/'embedding.safetensors').unlink();(root/'embedding.safetensors').write_bytes(b'bad')
            (root/'weights.bin').write_bytes(b'pickle')
            with self.assertRaises(WorkerError):conditioning.validate([use],{'textual-inversion':str(root)},PROTOCOL)

    def test_clip_skip_is_strict_and_backend_gated(self):
        body=request(clip_skip=1);validate_parameters(body)
        caps={'tasks':['text-to-image'],'clip_skip':[1,2]}
        validate_sampling(body,caps,'sdxl-base')
        for value in (True,0,3,1.5):
            body['parameters']['clip_skip']=value
            with self.assertRaises(WorkerError):validate_parameters(body)
        body=request(clip_skip=1)
        with self.assertRaises(WorkerError):validate_sampling(body,{'tasks':['text-to-image']},'sdxl-base')
        body['protocol']='media-worker/1.3'
        with self.assertRaises(WorkerError):validate_parameters(body)

    def test_component_weights_participate_in_authorized_fallback_planning(self):
        from fixture import TestBackend, make_bundle, make_conditioning, make_controlnet
        from media_engine.server import Worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=make_bundle(root/'model')
            component=make_conditioning(root/'ip','ip-adapter','ip')
            control=make_controlnet(root/'control')
            catalog={'fixture':str(root/'model'),'ip':str(root/'ip'),control['artifact_handle']:str(root/'control')}
            backend=TestBackend(estimate_mb=100,memory_budget_mb=100,precisions=('float16','float32'))
            worker=Worker(root/'worker',backend,catalog)
            options={'precision':'float32','attention':'sdpa','offload':'none'}
            for extra in ({'conditioning':[{'component':component,'scale':1}]},{'controlnets':[{'controlnet':control,'scale':1}]}):
                body={'protocol':PROTOCOL,'bundle':bundle,'options':options,**extra}
                with self.assertRaises(WorkerError) as error:worker.validation(body)
                self.assertEqual((error.exception.status,error.exception.code),(503,'out_of_memory'))
                self.assertEqual(backend.loads,0)
                cpu=worker.validation({**body,'authorized_fallbacks':['cpu']})
                self.assertEqual(cpu['memory_estimate']['device'],'cpu')
                self.assertEqual(cpu['memory_estimate']['value_mb'],101)
                self.assertEqual(cpu['applied_fallbacks'][0]['fallback'],'cpu')
                lower=worker.validation({**body,'authorized_fallbacks':['lower_precision','cpu']})
                self.assertEqual(lower['resolved_options']['precision'],'float16')
                self.assertEqual(lower['memory_estimate']['value_mb'],51)
                self.assertEqual(lower['applied_fallbacks'][0]['fallback'],'lower_precision')
            # The component-aware adjustment removes only the old VAE's resident weights.
            from media_engine.component_memory import include
            base={'value_mb':100,'roles':{'vae':{'loaded_mb':20}}}
            replacement={'parameters':5*1024*1024,'replaces_vae':True}
            self.assertEqual(include(base,replacement,'float32')['value_mb'],100)
            self.assertEqual(include(base,replacement,'float16')['value_mb'],90)
            self.assertEqual(include(base,replacement,'float16')['load_peak_mb'],100)

    def test_http_registration_load_identity_reference_ownership_and_unload(self):
        import http.client
        import time
        from fixture import TestBackend, make_bundle, make_conditioning
        from media_engine.server import Worker, create_server
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=make_bundle(root/'model')
            component=make_conditioning(root/'ip','ip-adapter','conditioning-ip')
            worker=Worker(root/'worker',TestBackend(),{'fixture':str(root/'model'),'conditioning-ip':str(root/'ip')})
            server=create_server(worker,'test-token-for-conditioning-http-32-characters',0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            def call(path,body):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                conn.request('POST',path,json.dumps(body),{'Authorization':'Bearer test-token-for-conditioning-http-32-characters'})
                response=conn.getresponse();status=response.status;data=json.loads(response.read());conn.close()
                return status,data
            def wait(operation):
                deadline=time.monotonic()+5
                while worker.operations[operation]['state']=='running' and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(worker.operations[operation]['state'],'succeeded',worker.operations[operation])
            try:
                use={'component':component,'scale':.5}
                status,data=call('/v1/validate',{'protocol':PROTOCOL,'conditioning':[use]})
                self.assertEqual(status,200,data)
                self.assertEqual(data['conditioning'],conditioning.summaries([use]))
                status,data=call('/v1/model/load',{'protocol':PROTOCOL,'bundle':bundle,'options':worker.capabilities()['load_defaults'],'conditioning':[use]})
                self.assertEqual(status,202,data);wait(data['operation_id'])
                self.assertEqual(worker.loaded['conditioning'],conditioning.summaries([use]))
                self.assertGreater(worker.loaded['memory_estimate']['additional_components_mb'],0)
                body=request();body['bundle_digest']=bundle['bundle_digest']
                self.assertEqual(call('/v1/executions',body)[0],400)
                body['inputs']['reference_image']='input_'+'a'*32
                self.assertEqual(call('/v1/executions',body)[0],404)
                image=worker.store_input(make_png(64,64))
                body['inputs']['reference_image']=image['input_id']
                status,data=call('/v1/executions',body);self.assertEqual(status,202,data)
                with self.assertRaises(WorkerError):worker.release_input(image['input_id'])
                deadline=time.monotonic()+5
                while worker.snapshot(data['execution_id'])['state']=='running' and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(worker.snapshot(data['execution_id'])['state'],'succeeded')
                worker.release_input(image['input_id'])
                status,data=call('/v1/model/unload',{'protocol':PROTOCOL});self.assertEqual(status,202,data);wait(data['operation_id'])
                self.assertIsNone(worker.loaded);self.assertEqual(worker.backend.conditioning,[])
            finally:
                server.shutdown();server.server_close();thread.join(5)


@unittest.skipUnless(HAVE_INFERENCE,'inference extra required')
class ConditioningInferenceTests(unittest.TestCase):
    def test_textual_inversion_clip_skip_vae_and_unload(self):
        import torch
        from safetensors.torch import save_file
        from media_engine.backend import DiffusersBackend
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,_=tiny_sdxl_bundle(root)
            backend=DiffusersBackend('cpu');options={'precision':'float32','attention':'sdpa','offload':'none'}
            backend.load(model,options)
            def run(name,body):
                out=root/name;out.mkdir()
                records,_=backend.generate(body,out,lambda *_:None,threading.Event())
                return records[0]['path'].read_bytes()
            original=run('original',request())
            self.assertNotEqual(original,run('clip',request(clip_skip=1)))
            embeddings=root/'embedding';embeddings.mkdir()
            save_file({'embedding':torch.arange(32,dtype=torch.float32)/16},str(embeddings/'embedding.safetensors'))
            component=descriptor(embeddings,'textual-inversion')
            use={'component':component,'token':'<newtoken>','encoder':'text_encoder'}
            before=len(backend.pipe.tokenizer)
            body=request();body['inputs']['prompt']='<newtoken>'
            without_embedding=run('without-embedding',body)
            conditioning.attach(backend,conditioning.validate([use],{'textual-inversion':str(embeddings)},PROTOCOL))
            self.assertEqual(backend.pipe.text_encoder.get_input_embeddings().weight.shape[0],before+1)
            body=request();body['inputs']['prompt']='<newtoken>'
            self.assertNotEqual(without_embedding,run('embedding-result',body))
            replacement=root/'replacement';backend.pipe.vae.save_pretrained(replacement,safe_serialization=True)
            from diffusers import AutoencoderKL
            vae=AutoencoderKL.from_pretrained(replacement,local_files_only=True)
            with torch.no_grad():vae.decoder.conv_out.bias.add_(0.75)
            vae.save_pretrained(replacement,safe_serialization=True)
            use={'component':descriptor(replacement,'vae')}
            conditioning.attach(backend,conditioning.validate([use],{'vae':str(replacement)},PROTOCOL))
            self.assertNotEqual(original,run('vae-result',request()))
            backend.unload();self.assertFalse(backend.text_cache.entries)
            backend.load(model,options)
            self.assertNotIn('<newtoken>',backend.pipe.tokenizer.get_vocab())
            self.assertEqual(original,run('reloaded',request()))
            backend.unload()

    def test_ip_adapter_scale_changes_real_pixels(self):
        import torch
        from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection,CLIPImageProcessor
        from safetensors.torch import save_file
        from media_engine.backend import DiffusersBackend
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,_=tiny_sdxl_bundle(root)
            backend=DiffusersBackend('cpu');backend.load(model,{'precision':'float32','attention':'sdpa','offload':'none'})
            weights=root/'ip';weights.mkdir()
            config=CLIPVisionConfig(hidden_size=32,intermediate_size=37,projection_dim=32,num_hidden_layers=2,num_attention_heads=4,image_size=32,patch_size=4)
            CLIPVisionModelWithProjection(config).save_pretrained(weights/'image_encoder',safe_serialization=True)
            CLIPImageProcessor(size={'shortest_edge':32},crop_size={'height':32,'width':32}).save_pretrained(weights)
            tensors={'image_proj.proj.weight':torch.randn(256,32)*.1,'image_proj.proj.bias':torch.randn(256)*.1,
                'image_proj.norm.weight':torch.ones(64),'image_proj.norm.bias':torch.zeros(64)}
            index=1
            for name in backend.pipe.unet.attn_processors:
                if name.endswith('attn1.processor'):continue
                if name.startswith('mid_block'):hidden=64
                elif name.startswith('up_blocks'):hidden=[64,32][int(name.split('.')[1])]
                else:hidden=[32,64][int(name.split('.')[1])]
                for projection in ('to_k_ip','to_v_ip'):
                    tensors[f'ip_adapter.{index}.{projection}.weight']=torch.randn(hidden,64)*.1
                index+=2
            save_file(tensors,str(weights/'ip_adapter.safetensors'))
            use={'component':descriptor(weights,'ip-adapter'),'scale':1}
            conditioning.attach(backend,conditioning.validate([use],{'ip-adapter':str(weights)},PROTOCOL))
            image=root/'reference.png';image.write_bytes(make_png(64,64,(120,60,30)))
            body=request();body['inputs']['reference_image']='input_'+'b'*32
            results=[]
            for scale in (0,1):
                backend.pipe.set_ip_adapter_scale(scale)
                out=root/f'out-{scale}';out.mkdir()
                records,_=backend.generate(body,out,lambda *_:None,threading.Event(),{'reference_image':image})
                results.append(records[0]['path'].read_bytes())
            self.assertNotEqual(*results)
            backend.unload();self.assertFalse(backend.has_ip_adapter)
