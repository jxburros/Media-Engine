"""Explicit offload negotiation and the real Accelerate hooks on CPU oracle components."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from media_engine import offload
from media_engine.contracts import WorkerError


class OffloadValidationTests(unittest.TestCase):
    def test_only_explicit_advertised_modes_on_new_protocol(self):
        caps = {"offload_modes": list(offload.MODES), "offload_families": list(offload.FAMILIES)}
        options = {"precision": "float32", "attention": "sdpa", "offload": "model"}
        offload.validate(options, 4, caps, "sdxl-base")
        for minor, advertised, family, modified in (
            (3, caps, "sdxl-base", options),
            (4, {"offload_modes": ["none"]}, "sdxl-base", options),
            (4, caps, "esrgan", options),
            (4, caps, "sdxl-base", {**options, "compile": "inductor"}),
            (4, caps, "sdxl-base", {**options, "offload": "auto"}),
        ):
            with self.assertRaises(WorkerError):
                offload.validate(modified, minor, advertised, family)
        self.assertEqual(offload.capabilities("diffusers-sdxl", "cpu", True)["offload_modes"], ["none"])
        self.assertEqual(offload.capabilities("owned-sdxl", "cuda:0", True)["offload_modes"], ["none"])


HAVE = all(importlib.util.find_spec(x) for x in ("torch", "accelerate", "diffusers", "transformers"))


@unittest.skipUnless(HAVE, "inference extra required")
class OffloadHookTests(unittest.TestCase):
    def test_real_accelerate_hooks_preserve_cpu_pipeline_pixels(self):
        import torch
        from diffusers import StableDiffusionXLPipeline
        from test_inference import tiny_sdxl_bundle
        with tempfile.TemporaryDirectory() as temp:
            model, _ = tiny_sdxl_bundle(Path(temp))
            expected = None
            for mode in offload.MODES:
                with self.subTest(mode=mode):
                    pipe = StableDiffusionXLPipeline.from_pretrained(str(model), local_files_only=True,
                                use_safetensors=True, torch_dtype=torch.float32, add_watermarker=False)
                    pipe.set_progress_bar_config(disable=True)
                    # Exercise real hooks on CPU without advertising CPU as a product mode.
                    # Real CUDA memory savings remain a separate hardware validation gate.
                    if mode == "sequential":
                        # Diffusers always appends an accelerator index (cpu:0 on this
                        # oracle), which Accelerate 1.4 cannot round-trip as an nn.Parameter.
                        # Exercise the same real hooks with an unindexed CPU device.
                        from accelerate import cpu_offload
                        for component in pipe.components.values():
                            if isinstance(component, torch.nn.Module):
                                cpu_offload(component, execution_device=torch.device("cpu"))
                    else:
                        offload.place(pipe, mode, "cpu")
                    image = pipe("test", height=64, width=64, num_inference_steps=2,
                                 generator=torch.Generator().manual_seed(7)).images[0]
                    if expected is None:
                        expected = image.tobytes()
                    self.assertEqual(image.tobytes(), expected)
                    offload.release(pipe, mode)
                    del pipe
