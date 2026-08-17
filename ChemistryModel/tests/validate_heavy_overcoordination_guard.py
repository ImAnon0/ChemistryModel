"""
Regression guard for heavy-atom over-coordination.

The heavy valence-state layer decides chemical topology. It must not erase
the base model's radial crowding penalty for contacts that are not selected
as bonds.

This geometry gives one carbon five C-C contacts. Angles are disabled in both
models so the only heavy-topology correction that could distinguish them is
an accidental replacement of the radial over-coordination term.

Before the fix this case differed by the full over-coordination energy and
~19 eV/A in force. After the fix base and optimised-valence are identical.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from valence_state_optimised_torch import (
    OptimisedValenceStateBatchedSimulation,
)


DTYPE = torch.float64
ENERGY_TOL = 1e-10
FORCE_TOL = 1e-9


def crowded_carbon_geometry():
    centre = np.array([10.0, 10.0, 10.0], dtype=float)

    # Four full C-C contacts at 1.7 A, plus a fifth contact at 2.0 A in the
    # smooth cutoff region. The fifth therefore produces both a non-zero
    # radial over-coordination energy and a non-zero force from that term.
    radius = 1.7
    vectors = [
        np.array([0.0, 0.0, radius]),
        np.array([0.0, 0.0, -radius]),
        np.array([radius, 0.0, 0.0]),
        np.array([
            radius * np.cos(2.0 * np.pi / 3.0),
            radius * np.sin(2.0 * np.pi / 3.0),
            0.0,
        ]),
        np.array([
            2.0 * np.cos(4.0 * np.pi / 3.0),
            2.0 * np.sin(4.0 * np.pi / 3.0),
            0.0,
        ]),
    ]

    positions = np.asarray(
        [centre] + [centre + vector for vector in vectors],
        dtype=float,
    )

    return ["C"] * len(positions), positions


def build(simulation_class):
    symbols, positions = crowded_carbon_geometry()

    simulation = simulation_class(
        boxes=[(symbols, positions)],
        box_size=30.0,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )

    # Isolate the over-coordination invariant. Heavy-valence is allowed to
    # replace heavy-centred angle topology in production, so angles are zero
    # in both models for this test.
    simulation.angle_stiffness.zero_()

    forces, energy = simulation.compute_forces()

    return simulation, forces, energy


def main():
    base, base_forces, base_energy = build(
        BatchedReactiveSimulation
    )

    candidate, candidate_forces, candidate_energy = build(
        OptimisedValenceStateBatchedSimulation
    )

    base_over = float(base._energy_parts["over"].sum())
    energy_difference = abs(
        float(candidate_energy - base_energy)
    )
    force_difference = float(
        torch.max(
            torch.abs(candidate_forces - base_forces)
        )
    )

    print("HEAVY OVER-COORDINATION REGRESSION GUARD")
    print()
    print(f"base radial over term : {base_over:.12f} eV")
    print(f"|dE| base/valence    : {energy_difference:.12e} eV")
    print(f"|dF|max base/valence : {force_difference:.12e} eV/A")

    if base_over <= 1.0:
        raise AssertionError(
            "test geometry did not create a meaningful heavy "
            "over-coordination penalty"
        )

    if energy_difference > ENERGY_TOL:
        raise AssertionError(
            "optimised valence removed or altered the base heavy "
            "radial over-coordination energy"
        )

    if force_difference > FORCE_TOL:
        raise AssertionError(
            "optimised valence removed or altered the base heavy "
            "radial over-coordination force"
        )

    print()
    print(
        "FINAL PASS - heavy valence topology leaves radial "
        "over-coordination protection intact."
    )


if __name__ == "__main__":
    main()
