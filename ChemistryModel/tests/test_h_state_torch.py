import numpy as np
import torch

import reactive as R

from batched_torch import BatchedReactiveSimulation
from h_state_reference import hydrogen_state_energy
from h_state_torch import HStateReferenceBatchedSimulation


DTYPE = torch.float64


def build_base(symbols, positions):
    return BatchedReactiveSimulation(
        boxes=[(symbols, positions)],
        box_size=20.0,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=DTYPE,
    )


def build_state(symbols, positions):
    return HStateReferenceBatchedSimulation(
        boxes=[(symbols, positions)],
        box_size=20.0,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=DTYPE,
    )


def test_isolated_h2_matches_base_energy_and_forces():
    distance = R.BOND_TABLE[("H", "H")][0]

    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0 + distance, 5.0, 5.0],
    ])

    symbols = ["H", "H"]

    base = build_base(symbols, positions)
    state = build_state(symbols, positions)

    assert np.isclose(
        base.potential_per_box[0],
        state.potential_per_box[0],
        atol=1e-10,
    )

    assert np.allclose(
        base.forces.detach().cpu().numpy(),
        state.forces.detach().cpu().numpy(),
        atol=1e-9,
    )


def test_isolated_ch_matches_base_energy_and_forces():
    distance = R.BOND_TABLE[("C", "H")][0]

    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0 + distance, 5.0, 5.0],
    ])

    symbols = ["C", "H"]

    base = build_base(symbols, positions)
    state = build_state(symbols, positions)

    assert np.isclose(
        base.potential_per_box[0],
        state.potential_per_box[0],
        atol=1e-10,
    )

    assert np.allclose(
        base.forces.detach().cpu().numpy(),
        state.forces.detach().cpu().numpy(),
        atol=1e-9,
    )


def test_symmetric_h3_matches_numpy_reference():
    distance = 0.9406

    positions = np.array([
        [5.0 - distance, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0 + distance, 5.0, 5.0],
    ])

    symbols = ["H", "H", "H"]

    torch_model = build_state(
        symbols,
        positions,
    )

    numpy_result = hydrogen_state_energy(
        positions,
        symbols,
        box_size=20.0,
        match_torch_environment=True,
    )

    assert np.isclose(
        torch_model.potential_per_box[0],
        numpy_result.energy,
        atol=1e-8,
    )


def test_h3_forces_are_finite_and_symmetric():
    distance = 0.9406

    positions = np.array([
        [5.0 - distance, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0 + distance, 5.0, 5.0],
    ])

    model = build_state(
        ["H", "H", "H"],
        positions,
    )

    forces = (
        model.forces
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.isfinite(forces))

    assert np.allclose(
        forces[0],
        -forces[2],
        atol=1e-8,
    )

    assert np.allclose(
        forces[1],
        np.zeros(3),
        atol=1e-8,
    )