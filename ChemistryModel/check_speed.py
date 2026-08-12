import time

import numpy as np
import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation
import build_box


def timed(label, steps=200):
    boxes = []
    generator = np.random.default_rng(0)
    for _ in range(8):
        symbols, positions = build_box.build(
            {"CH4": 6, "NH3": 4, "H2O": 6, "H2": 8}, 12.0, generator
        )
        boxes.append((symbols, positions))

    sim = BatchedReactiveSimulation(
        boxes=boxes, box_size=12.0, random_seed=0, relax_on_start=False
    )

    sim.step(20)                      # warm up, ignore
    started = time.monotonic()
    sim.step(steps)
    elapsed = time.monotonic() - started

    print(f"  {label:<28} {elapsed:6.2f}s for {steps} steps "
          f"({steps / elapsed:6.1f} steps/s)")


print("8 boxes of Miller-Urey, 80 atoms each\n")

original = R.ENVIRONMENT_SOFTENING
timed(f"softening {original}")

R.ENVIRONMENT_SOFTENING = 0.0
timed("softening 0.0")

R.ENVIRONMENT_SOFTENING = original
print(f"\nover-coordination weight is {R.OVER_COORDINATION_DEPTH_WEIGHT}")