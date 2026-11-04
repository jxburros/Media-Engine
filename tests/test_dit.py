"""Real tiny local Flux and SD3 reference inference, not production quality evidence."""
import tempfile
import threading
import unittest
from pathlib import Path
from test_inference import HAVE_INFERENCE, tiny_sdxl_bundle
from fixture import describe_bundle,make_png


def tiny_dit(root, family):
    import torch
    from diffusers import (StableDiffusionXLPipeline, FluxPipeline, StableDiffusion3Pipeline,
        FluxTransformer2DModel, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler)
    from transformers import T5Config,T5EncoderModel,T5TokenizerFast,CLIPTextModelWithProjection
    from tokenizers import Tokenizer,models,pre_tokenizers,processors
    model,_=tiny_sdxl_bundle(root/'base')
    base=StableDiffusionXLPipeline.from_pretrained(model,local_files_only=True)
    words=['<pad>','</s>','<unk>','a','test']
    tokenizer=Tokenizer(models.WordLevel({w:i for i,w in enumerate(words)},unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer.post_processor=processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
    tok=T5TokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',eos_token='</s>',unk_token='<unk>',extra_ids=0,model_max_length=256)
    torch.manual_seed(4)
    t5=T5EncoderModel(T5Config(vocab_size=len(words),d_model=64,d_ff=64,d_kv=8,num_layers=1,num_heads=4))
    base.vae.register_to_config(shift_factor=0.0)
    common=dict(vae=base.vae,scheduler=FlowMatchEulerDiscreteScheduler())
    if family=='flux-schnell':
        transformer=FluxTransformer2DModel(in_channels=16,num_layers=1,num_single_layers=1,
            attention_head_dim=16,num_attention_heads=2,joint_attention_dim=64,pooled_projection_dim=32,axes_dims_rope=(4,6,6))
        pipe=FluxPipeline(**common,transformer=transformer,text_encoder=base.text_encoder,tokenizer=base.tokenizer,text_encoder_2=t5,tokenizer_2=tok)
    else:
        from diffusers import AutoencoderKL
        common["vae"]=AutoencoderKL.from_config(base.vae.config,latent_channels=16)
        transformer=SD3Transformer2DModel(sample_size=32,patch_size=2,in_channels=16,out_channels=16,
            num_layers=1,attention_head_dim=16,num_attention_heads=2,joint_attention_dim=64,caption_projection_dim=32,pooled_projection_dim=64,pos_embed_max_size=32)
        pipe=StableDiffusion3Pipeline(**common,transformer=transformer,text_encoder=CLIPTextModelWithProjection(base.text_encoder.config),
            text_encoder_2=base.text_encoder_2,tokenizer=base.tokenizer,tokenizer_2=base.tokenizer_2,text_encoder_3=t5,tokenizer_3=tok)
    path=root/'dit';pipe.save_pretrained(path,safe_serialization=True)
    return path,describe_bundle(path,'dit','tiny-dit',family,{'license':'test-only random weights'})


@unittest.skipUnless(HAVE_INFERENCE,'inference extra needed')
class DiTTests(unittest.TestCase):
    def test_both_families_execute_deterministically(self):
        from media_engine.backend import DiffusersBackend
        from media_engine.contracts import DEFAULTS,PROTOCOL,bundle_path,validate_parameters,validate_sampling,WorkerError
        for family in ('flux-schnell','sd35'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path,bundle=tiny_dit(root,family)
                self.assertEqual(bundle_path(bundle,{'dit':str(path)}),path)
                backend=DiffusersBackend('cpu');backend.load(path,{'precision':'float32','attention':'sdpa','offload':'none'},family=family)
                request={'protocol':PROTOCOL,'attempt_id':'attempt_dit','bundle_digest':bundle['bundle_digest'],'task':'text-to-image','inputs':{'prompt':'a test'},
                    'parameters':{**DEFAULTS,'width':64,'height':64,'steps':2,'seed':3,'guidance':0 if family=='flux-schnell' else 3,'sampler':'flow-euler','scheduler':'flow-match'}}
                validate_parameters(request);validate_sampling(request,backend.capabilities(),family)
                data=[]
                for index in range(2):
                    out=root/str(index);out.mkdir()
                    records,_=backend.generate(request,out,lambda *_:None,threading.Event())
                    data.append(records[0]['path'].read_bytes())
                self.assertEqual(data[0],data[1]);self.assertGreater(len(data[0]),100)
                image=root/'source.png';image.write_bytes(make_png(64,64,(20,40,60)))
                mask=root/'mask.png';mask.write_bytes(make_png(64,64,(255,255,255)))
                for task in ('image-to-image','inpaint'):
                    request['task']=task;request['parameters']['strength']=1.0
                    request['inputs']['image']='input_'+'a'*32
                    if task=='inpaint':request['inputs']['mask']='input_'+'b'*32
                    validate_parameters(request);validate_sampling(request,backend.capabilities(),family)
                    out=root/task;out.mkdir()
                    records,_=backend.generate(request,out,lambda *_:None,threading.Event(),{'image':image,'mask':mask})
                    self.assertNotEqual(records[0]['path'].read_bytes(),image.read_bytes())
                request['parameters']['sampler']='euler';request['parameters']['scheduler']='euler'
                with self.assertRaises(WorkerError):validate_sampling(request,backend.capabilities(),family)
                backend.unload()


class DiTContractTests(unittest.TestCase):
    def test_family_specific_settings_never_silently_disappear(self):
        from media_engine.contracts import DEFAULTS,PROTOCOL,WorkerError,validate_sampling
        from media_engine.sampling import advertised
        request={'protocol':PROTOCOL,'task':'text-to-image','inputs':{'prompt':'test'},
            'parameters':{**DEFAULTS,'seed':1,'guidance':0,'sampler':'flow-euler','scheduler':'flow-match'}}
        caps={'tasks':['text-to-image'],'sampling':advertised(True)}
        validate_sampling(request,caps,'flux-schnell')
        for field,value in [('clip_skip',1),('preview_interval',2),('guidance',6)]:
            prior=request['parameters'].get(field)
            request['parameters'][field]=value
            with self.assertRaises(WorkerError):validate_sampling(request,caps,'flux-schnell')
            if prior is None:del request['parameters'][field]
            else:request['parameters'][field]=prior
        with self.assertRaises(WorkerError):validate_sampling(request,caps,'sdxl-base')


class DiTGeometryTests(unittest.TestCase):
    def test_bundle_geometry_is_checked_before_execution(self):
        from media_engine.contracts import DEFAULTS,PROTOCOL,WorkerError,family_facts
        from media_engine.dit import check_dimensions
        from media_engine.server import Worker
        from fixture import TestBackend,make_dit_bundle
        for family in ('flux-schnell','sd35'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);bundle=make_dit_bundle(root/'model',family)
                facts=family_facts(root/'model',family)
                self.assertEqual(facts['size_multiple'],16)
                worker=Worker(root/'worker',TestBackend(),{family:str(root/'model')})
                worker.loaded={'bundle_digest':bundle['bundle_digest']};worker.loaded_family=family;worker.loaded_facts=facts
                body={'protocol':PROTOCOL,'attempt_id':'attempt_dimensions','bundle_digest':bundle['bundle_digest'],'task':'text-to-image','inputs':{'prompt':'test'},
                    'parameters':{**DEFAULTS,'seed':1,'guidance':0,'sampler':'flow-euler','scheduler':'flow-match','width':72,'height':72}}
                with self.assertRaises(WorkerError) as caught:worker.validate_execution(body)
                self.assertEqual(caught.exception.status,400)
                with self.assertRaises(WorkerError):worker.submit(body)
                self.assertIsNone(worker.active)
                check_dimensions({'width':64,'height':64},16)
                with self.assertRaises(WorkerError):check_dimensions({'width':72,'height':64},16)
