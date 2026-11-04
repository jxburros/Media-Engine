"""Real offline migrations, including refusal of executable pickle payloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from media_engine.convert import ConversionError
from media_engine.convert_weights import esrgan_pth, lora_kohya

HAVE = all(importlib.util.find_spec(x) for x in ("torch", "safetensors", "diffusers", "transformers"))


@unittest.skipUnless(HAVE, "inference extra required")
class WeightConversionTests(unittest.TestCase):
    def test_rrdbnet_round_trip_and_refusals(self):
        import torch
        from safetensors.torch import load_file
        from fixture import ESRGAN_CONFIG
        from media_engine.upscale import build, validate_config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {**ESRGAN_CONFIG, "num_feat": 8, "num_grow_ch": 4}
            state = build(validate_config(config)).state_dict()
            source, cfg = root / "network.pth", root / "config.json"
            cfg.write_text(json.dumps(config))
            torch.save({"params_ema": state}, source)
            record = esrgan_pth(source, cfg, root / "converted")
            self.assertEqual(record["unmapped_keys"], [])
            actual = load_file(str(root / "converted/network.safetensors"))
            self.assertEqual(set(actual), set(state))
            self.assertTrue(all(torch.equal(state[k], actual[k]) for k in state))
            with self.assertRaisesRegex(ConversionError, "output_exists"):
                esrgan_pth(source, cfg, root / "converted")
            torch.save({**state, "unknown": torch.zeros(1)}, source)
            with self.assertRaises(ConversionError) as caught:
                esrgan_pth(source, cfg, root / "bad")
            self.assertEqual(caught.exception.keys, ["unknown"])
            self.assertFalse((root / "bad").exists())

    def test_weights_only_subprocess_rejects_executable_pickle(self):
        import torch
        from fixture import ESRGAN_CONFIG
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "executed"
            class Payload:
                def __reduce__(self):
                    return (eval, ("__import__('pathlib').Path(" + repr(str(marker)) + ").touch()",))
            source, cfg = root / "unsafe.pth", root / "config.json"
            torch.save(Payload(), source)
            cfg.write_text(json.dumps(ESRGAN_CONFIG))
            with self.assertRaisesRegex(ConversionError, "unsafe_or_invalid_checkpoint"):
                esrgan_pth(source, cfg, root / "out")
            self.assertFalse(marker.exists())
            self.assertFalse((root / "out").exists())

    def test_kohya_keys_preserve_lora_and_alpha(self):
        import torch
        from safetensors.torch import load_file, save_file
        from test_inference import tiny_sdxl_bundle, tiny_lora
        from media_engine.contracts import lora_layout
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, _ = tiny_sdxl_bundle(root)
            tiny_lora(root / "adapter", model)
            expected = load_file(str(root / "adapter/adapter.safetensors"))
            kohya = {}
            for key, value in expected.items():
                role, rest = key.split(".", 1)
                module, part = rest.rsplit(".lora_", 1)
                prefix = {"unet": "lora_unet_", "text_encoder": "lora_te1_", "text_encoder_2": "lora_te2_"}[role]
                encoded = prefix + module.replace(".", "_")
                kohya[encoded + (".lora_down.weight" if part == "A.weight" else ".lora_up.weight")] = value
            first = next(iter(kohya)).rsplit(".lora_", 1)[0]
            kohya[first + ".alpha"] = torch.tensor(2.0)
            source = root / "kohya.safetensors"
            save_file(kohya, str(source))
            lora_kohya(source, {"ref": str(model)}, "ref", root / "converted")
            actual = load_file(str(root / "converted/adapter.safetensors"))
            self.assertTrue(all(torch.equal(v, actual[k]) for k, v in expected.items()))
            self.assertEqual(sum(k.endswith(".alpha") for k in actual), 1)
            self.assertEqual(lora_layout(root / "converted/adapter.safetensors")["format"], "peft-lora")
            kohya["lora_unet_unknown.lora_down.weight"] = torch.zeros(2,2)
            save_file(kohya, str(source))
            with self.assertRaises(ConversionError) as caught:
                lora_kohya(source, {"ref": str(model)}, "ref", root / "bad")
            self.assertIn("lora_unet_unknown.lora_down.weight", caught.exception.keys)
            self.assertFalse((root / "bad").exists())
