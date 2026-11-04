"""Published JSON Schema agrees with strict additive protocol-minor gates."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('jsonschema'),'jsonschema validator is optional for source checks')
class SchemaTests(unittest.TestCase):
    def test_new_generation_fields_and_legacy_refusals(self):
        import jsonschema
        from media_engine.contracts import DEFAULTS
        schema=json.loads((Path(__file__).parents[1]/'schemas/worker.schema.json').read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        validator=jsonschema.Draft202012Validator(schema)
        base={'protocol':'media-worker/1.4','attempt_id':'attempt_schema','bundle_digest':'sha256:'+'a'*64,'task':'text-to-image','inputs':{'prompt':'test'},'parameters':{**DEFAULTS,'seed':1}}
        for sampler,scheduler in [('euler','euler'),('euler-ancestral','euler'),('dpmpp-2m','karras'),('ddim','ddim'),('unipc','unipc'),('flow-euler','flow-match')]:
            req=copy.deepcopy(base);req['parameters'].update(sampler=sampler,scheduler=scheduler)
            self.assertTrue(validator.is_valid(req),list(validator.iter_errors(req)))
            req['protocol']='media-worker/1.3'
            self.assertEqual(validator.is_valid(req),sampler=='euler')
        req=copy.deepcopy(base);req['task']='inpaint';req['parameters']['strength']=.8
        req['inputs'].update(image='input_'+'a'*32,mask='input_'+'b'*32)
        self.assertTrue(validator.is_valid(req));req['protocol']='media-worker/1.3';self.assertFalse(validator.is_valid(req))
        req=copy.deepcopy(base);req.update(task='image-transform',inputs={'image':'input_'+'a'*32},parameters={'width':64,'height':64,'operation':'canny','threshold_low':.1,'threshold_high':.3})
        self.assertTrue(validator.is_valid(req));req['parameters']['prompt']='wrong';self.assertFalse(validator.is_valid(req))
        req=copy.deepcopy(base);req['parameters']['preview_interval']=3
        self.assertTrue(validator.is_valid(req));req['protocol']='media-worker/1.3';self.assertFalse(validator.is_valid(req))
