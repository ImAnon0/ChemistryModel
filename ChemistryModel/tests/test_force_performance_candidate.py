"""Equivalence gate for experimental force-path implementations."""

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from performance_benchmark import boxes_for


def test_index_select_gather_matches_reference():
    boxes, box = boxes_for("representative", 4, 28200)
    reference = BatchedReactiveSimulation(
        boxes=boxes, box_size=box, time_step=0.25,
        target_temperature=800.0, friction=0.0, device="cuda",
        random_seed=28200,
    )
    candidate = BatchedReactiveSimulation(
        boxes=boxes, box_size=box, time_step=0.25,
        target_temperature=800.0, friction=0.0, device="cuda",
        random_seed=28200,
    )
    candidate.experimental_index_select_gather = True
    candidate.positions.copy_(reference.positions)
    candidate.velocities.copy_(reference.velocities)
    candidate.build_neighbours()
    reference.build_neighbours()
    force_ref, energy_ref = reference.compute_forces()
    force_new, energy_new = candidate.compute_forces()
    assert torch.equal(energy_ref, energy_new)
    force_error = torch.max(torch.abs(force_ref - force_new)).item()
    assert force_error <= 2e-6, force_error

    for _ in range(5):
        reference.step(1)
        candidate.step(1)
    position_error = torch.max(torch.abs(
        reference.positions - candidate.positions
    )).item()
    velocity_error = torch.max(torch.abs(
        reference.velocities - candidate.velocities
    )).item()
    final_force_error = torch.max(torch.abs(
        reference.forces - candidate.forces
    )).item()
    assert position_error <= 2e-6, position_error
    assert velocity_error <= 2e-6, velocity_error
    assert final_force_error <= 2e-5, final_force_error
    assert np.allclose(reference.potential_per_box,
                       candidate.potential_per_box, rtol=0, atol=2e-4)


if __name__ == "__main__":
    test_index_select_gather_matches_reference()
    print("PASS  force performance candidate")
