"""Explicit, backend-specific scheduler selection; no automatic substitution."""

SAMPLING = (
    ("euler", "euler"),
    ("euler-ancestral", "euler"),
    ("dpmpp-2m", "karras"),
    ("ddim", "ddim"),
    ("unipc", "unipc"),
    ("flow-euler", "flow-match"),
)


def advertised(reference=False):
    return [{"sampler": a, "scheduler": b} for a, b in (SAMPLING if reference else SAMPLING[:1])]


def create(config, sampler, scheduler):
    from diffusers import (
        EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler, DDIMScheduler, UniPCMultistepScheduler,
    )
    classes = {
        ("euler", "euler"): (EulerDiscreteScheduler, {}),
        ("euler-ancestral", "euler"): (EulerAncestralDiscreteScheduler, {}),
        ("dpmpp-2m", "karras"): (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
        ("ddim", "ddim"): (DDIMScheduler, {}),
        ("unipc", "unipc"): (UniPCMultistepScheduler, {"solver_order": 2}),
    }
    cls, options = classes[(sampler, scheduler)]
    return cls.from_config(config, **options)
