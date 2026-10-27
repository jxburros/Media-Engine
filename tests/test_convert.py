"""`media-engine convert` on a tiny single-file checkpoint built from the tiny SDXL bundle.

The checkpoint is produced by the inverse of the Diffusers single-file mapping (written here for
the fixture's shapes), so converting it must reproduce the original component tensors exactly.
No network and no real weights are involved.
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from fixture import TestBackend, describe_bundle
from media_engine.contracts import PROTOCOL, bundle_path
from media_engine.convert import ConversionError, convert, safetensors_header

HAVE_TORCH = all(
    importlib.util.find_spec(m) is not None for m in ("torch", "diffusers", "transformers", "safetensors")
)
RESNET = (
    ("norm1", "in_layers.0"),
    ("conv1", "in_layers.2"),
    ("norm2", "out_layers.0"),
    ("conv2", "out_layers.3"),
    ("time_emb_proj", "emb_layers.1"),
    ("conv_shortcut", "skip_connection"),
)


def unet_to_ldm(state, config):
    """Diffusers UNet keys → LDM `model.diffusion_model.*` keys (SDXL layout)."""
    per = config["layers_per_block"] + 1
    blocks = len(config["block_out_channels"])
    prefixes = {
        "time_embedding.linear_1.": "time_embed.0.",
        "time_embedding.linear_2.": "time_embed.2.",
        "add_embedding.linear_1.": "label_emb.0.0.",
        "add_embedding.linear_2.": "label_emb.0.2.",
        "conv_in.": "input_blocks.0.0.",
        "conv_norm_out.": "out.0.",
        "conv_out.": "out.2.",
        "mid_block.attentions.0.": "middle_block.1.",
        "mid_block.resnets.0.": "middle_block.0.",
        "mid_block.resnets.1.": "middle_block.2.",
    }
    for i in range(blocks):
        for j in range(config["layers_per_block"]):
            prefixes[f"down_blocks.{i}.resnets.{j}."] = f"input_blocks.{per * i + j + 1}.0."
            prefixes[f"down_blocks.{i}.attentions.{j}."] = f"input_blocks.{per * i + j + 1}.1."
        if i < blocks - 1:
            prefixes[f"down_blocks.{i}.downsamplers.0.conv."] = f"input_blocks.{per * (i + 1)}.0.op."
        for j in range(per):
            prefixes[f"up_blocks.{i}.resnets.{j}."] = f"output_blocks.{per * i + j}.0."
            prefixes[f"up_blocks.{i}.attentions.{j}."] = f"output_blocks.{per * i + j}.1."
    out = {}
    for key, value in state.items():
        prefix = max((p for p in prefixes if key.startswith(p)), key=len, default=None)
        if prefix is None and ".upsamplers.0.conv." in key:
            i = int(key.split(".")[1])
            has_attention = any(k.startswith(f"up_blocks.{i}.attentions.") for k in state)
            target = f"output_blocks.{per * i + per - 1}.{2 if has_attention else 1}.conv."
            new = target + key.split(".upsamplers.0.conv.")[1]
        else:
            new = prefixes[prefix] + key[len(prefix):]
            if ".resnets." in prefix or prefix.startswith("mid_block.resnets"):
                for diffusers, ldm in RESNET:
                    new = new.replace("." + diffusers + ".", "." + ldm + ".")
        out["model.diffusion_model." + new] = value
    return out


def vae_to_ldm(state, config):
    blocks = len(config["block_out_channels"])
    out = {}
    for key, value in state.items():
        new = key
        for i in range(blocks):
            new = new.replace(f"encoder.down_blocks.{i}.resnets.", f"encoder.down.{i}.block.")
            new = new.replace(f"encoder.down_blocks.{i}.downsamplers.0.", f"encoder.down.{i}.downsample.")
            new = new.replace(f"decoder.up_blocks.{i}.resnets.", f"decoder.up.{blocks - 1 - i}.block.")
            new = new.replace(f"decoder.up_blocks.{i}.upsamplers.0.", f"decoder.up.{blocks - 1 - i}.upsample.")
        for i in range(2):
            new = new.replace(f"mid_block.resnets.{i}.", f"mid.block_{i + 1}.")
        new = new.replace("mid_block.attentions.0.", "mid.attn_1.").replace("conv_shortcut", "nin_shortcut")
        new = new.replace("conv_norm_out", "norm_out")
        if ".mid.attn_1." in new:
            for diffusers, ldm in (("group_norm", "norm"), ("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "proj_out")):
                new = new.replace(f".{diffusers}.", f".{ldm}.")
            if new.endswith(("q.weight", "k.weight", "v.weight", "proj_out.weight")):
                value = value[:, :, None, None]
        out["first_stage_model." + new] = value
    return out


def openclip_to_ldm(state, prefix="conditioner.embedders.1.model."):
    import torch

    out, layers = {}, {}
    for key, value in state.items():
        if key == "text_projection.weight":
            out[prefix + "text_projection"] = value.T.contiguous()
        elif key == "text_model.embeddings.position_embedding.weight":
            out[prefix + "positional_embedding"] = value
        elif key == "text_model.embeddings.token_embedding.weight":
            out[prefix + "token_embedding.weight"] = value
        elif key.startswith("text_model.final_layer_norm."):
            out[prefix + "ln_final." + key.rsplit(".", 1)[1]] = value
        elif ".self_attn." in key and key.split(".")[-2] in ("q_proj", "k_proj", "v_proj"):
            index = int(key.split(".")[3])
            layers.setdefault((index, key.rsplit(".", 1)[1]), {})[key.split(".")[-2]] = value
        else:
            new = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            for diffusers, ldm in (("layer_norm1", "ln_1"), ("layer_norm2", "ln_2"), (".mlp.fc1.", ".mlp.c_fc."), (".mlp.fc2.", ".mlp.c_proj."), (".self_attn.", ".attn.")):
                new = new.replace(diffusers, ldm)
            out[prefix + new] = value
    for (index, kind), parts in layers.items():
        out[f"{prefix}transformer.resblocks.{index}.attn.in_proj_{kind}"] = torch.cat(
            [parts["q_proj"], parts["k_proj"], parts["v_proj"]]
        )
    return out


def single_file(bundle_dir):
    from safetensors.torch import load_file

    root = Path(bundle_dir)
    load = lambda role, name: load_file(str(root / role / name))
    unet_config = json.loads((root / "unet/config.json").read_text())
    vae_config = json.loads((root / "vae/config.json").read_text())
    state = {}
    state.update(unet_to_ldm(load("unet", "diffusion_pytorch_model.safetensors"), unet_config))
    state.update(vae_to_ldm(load("vae", "diffusion_pytorch_model.safetensors"), vae_config))
    state.update(
        {"conditioner.embedders.0.transformer." + k: v for k, v in load("text_encoder", "model.safetensors").items()}
    )
    state.update(openclip_to_ldm(load("text_encoder_2", "model.safetensors")))
    return state


class SignatureTests(unittest.TestCase):
    """Refusals that happen before any inference dependency is imported."""

    def test_non_safetensors_inputs_are_refused_by_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, data in (
                ("model.safetensors", b"\x80\x02}q\x00."),  # pickle protocol 2
                ("model.safetensors", b"PK\x03\x04" + bytes(32)),  # torch zip archive
                ("model.safetensors", b"\x10\x00\x00\x00\x00\x00\x00\x00not json here!!!"),
                ("model.ckpt", b"\x10\x00\x00\x00\x00\x00\x00\x00{}"),
            ):
                path = Path(temp) / name
                path.write_bytes(data)
                with self.assertRaises(ConversionError) as caught:
                    convert(path, {}, "ref", Path(temp) / "out")
                self.assertEqual(caught.exception.code, "not_safetensors")

    def test_cli_reports_codes_without_paths(self):
        from media_engine.server import main

        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "in.safetensors").write_bytes(b"\x80\x02")
            (Path(temp) / "catalog.json").write_text("{}")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main(["convert", "--input", str(Path(temp) / "in.safetensors"), "--reference-handle", "ref", "--catalog", str(Path(temp) / "catalog.json"), "--output", str(Path(temp) / "out")])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(json.loads(stderr.getvalue()), {"error": "not_safetensors"})
            self.assertNotIn(temp, stderr.getvalue())


@unittest.skipUnless(HAVE_TORCH, "inference-owned extra is not installed")
class ConvertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import save_file
        from test_inference import tiny_sdxl_bundle

        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.model_dir, cls.bundle = tiny_sdxl_bundle(cls.root)
        cls.catalog = {"tiny": str(cls.model_dir)}
        cls.state = single_file(cls.model_dir)
        cls.checkpoint = cls.root / "tiny-sdxl.safetensors"
        save_file(cls.state, str(cls.checkpoint))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_conversion_reproduces_the_bundle(self):
        from safetensors.torch import load_file

        output = self.root / "converted"
        record = convert(self.checkpoint, self.catalog, "tiny", output)
        self.assertEqual(record["unmapped_keys"], [])
        self.assertEqual(sum(record["mapped_keys"].values()), len(self.state))
        self.assertEqual(set(record["converter"]), {"media_engine", "diffusers", "torch", "safetensors"})
        self.assertEqual(json.loads((output / "conversion.json").read_text()), record)
        for role, name in (("unet", "diffusion_pytorch_model.safetensors"), ("vae", "diffusion_pytorch_model.safetensors"), ("text_encoder", "model.safetensors"), ("text_encoder_2", "model.safetensors")):
            original, converted = load_file(str(self.model_dir / role / name)), load_file(str(output / role / name))
            self.assertEqual(set(original), set(converted), role)
            for key in original:
                self.assertTrue((original[key] == converted[key].reshape(original[key].shape)).all(), key)
        # It registers and loads like a native bundle; conversion.json is the only difference.
        bundle = describe_bundle(output, "converted", "converted")
        catalog = {**self.catalog, "converted": str(output)}
        self.assertEqual(bundle_path(bundle, catalog), output.resolve())
        from media_engine.server import Worker

        worker = Worker(self.root / "work", TestBackend(), catalog)
        op = worker.control("load", {"protocol": PROTOCOL, "bundle": bundle, "options": {"precision": "float32", "attention": "sdpa", "offload": "none"}})
        import time

        while worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.01)
        self.assertEqual(worker.operations[op["operation_id"]]["state"], "succeeded")
        native = set(describe_bundle(self.model_dir, "tiny", "x")["configuration"]["files"])
        self.assertEqual(set(bundle["configuration"]["files"]) - native, {"conversion.json"})

    def test_converted_bundle_runs_on_the_owned_pipeline(self):
        from media_engine.backend import backend_class
        from media_engine.server import Worker
        import time

        output = self.root / "converted-run"
        convert(self.checkpoint, self.catalog, "tiny", output, precision="float32")
        bundle = describe_bundle(output, "run", "run")
        worker = Worker(self.root / "work-run", backend_class("owned-sdxl")("cpu"), {"run": str(output)})
        op = worker.control("load", {"protocol": PROTOCOL, "bundle": bundle, "options": {"precision": "float32", "attention": "sdpa", "offload": "none"}})
        while worker.operations[op["operation_id"]]["state"] == "running":
            time.sleep(0.02)
        self.assertEqual(worker.operations[op["operation_id"]]["state"], "succeeded", worker.operations[op["operation_id"]]["error"])

    def test_extra_key_fails_by_name_and_leaves_nothing(self):
        import torch
        from safetensors.torch import save_file

        state = dict(self.state)
        state["model.diffusion_model.mystery.weight"] = torch.zeros(2)
        state["cond_stage_model.other"] = torch.zeros(1)
        path = self.root / "extra.safetensors"
        save_file(state, str(path))
        output = self.root / "extra-out"
        with self.assertRaises(ConversionError) as caught:
            convert(path, self.catalog, "tiny", output)
        self.assertEqual(caught.exception.code, "unmapped_keys")
        self.assertEqual(caught.exception.keys, ["cond_stage_model.other", "model.diffusion_model.mystery.weight"])
        self.assertFalse(output.exists())
        self.assertFalse(any(p.name.startswith(".extra-out") for p in self.root.iterdir()))

    def test_header_reader(self):
        self.assertIn("first_stage_model.encoder.conv_in.weight", safetensors_header(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
