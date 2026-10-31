"""Record diffusers 0.32.2 EulerDiscreteScheduler schedules for the Rust unit tests.

Writes tests/data/euler_schedules.json: for each scheduler config and step count, the
timesteps, sigmas and init_noise_sigma `set_timesteps` produces. Run with the reference
interpreter:  python native/sdxl/tests/make_scheduler_fixture.py
"""

import json
from pathlib import Path

from diffusers import EulerDiscreteScheduler

OUT = Path(__file__).resolve().parent / "data" / "euler_schedules.json"
SDXL = {"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "timestep_spacing": "leading", "steps_offset": 1}
CONFIGS = [
    (SDXL, [1, 2, 3, 7, 30, 50]),
    ({**SDXL, "timestep_spacing": "linspace", "steps_offset": 0}, [1, 4, 25]),
    ({**SDXL, "timestep_spacing": "trailing", "steps_offset": 0}, [3, 7, 30]),
    ({"beta_schedule": "linear", "timestep_spacing": "leading", "steps_offset": 1}, [5, 20]),
    ({"beta_schedule": "squaredcos_cap_v2", "timestep_spacing": "trailing"}, [6]),
    ({**SDXL, "final_sigmas_type": "sigma_min"}, [10]),
]


def main():
    rows = []
    for config, steps in CONFIGS:
        for n in steps:
            scheduler = EulerDiscreteScheduler(**config)
            scheduler.set_timesteps(n)
            rows.append(
                {
                    "config": config,
                    "steps": n,
                    "timesteps": scheduler.timesteps.tolist(),
                    "sigmas": scheduler.sigmas.tolist(),
                    "init_noise_sigma": float(scheduler.init_noise_sigma),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")


main()
