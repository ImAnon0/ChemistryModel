"""
Focused tests for the experimental SAPT orbital bridge.

Run:
    py -m pytest test_sapt_orbital_bridge_torch.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import reactive as R
import h_state_torch as hs
import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)

from sapt_orbital_bridge_torch import (
    ORBITAL_BRIDGE_MODEL_REVISION,
    OrbitalBridgeSaptHStateBatchedSimulation,
    _bridge_gate_maps,
)


DTYPE = torch.float64


def make_simulation(
    model,
    symbols,
    positions,
    *,
    strength=0.0,
    mixing=SAPT_H_STATE_MIXING,
):
    kwargs = dict(
        boxes=[
            (
                symbols,
                np.asarray(
                    positions,
                    dtype=float,
                )
                + scan.CENTRE,
            )
        ],
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=DTYPE,
        h_state_mixing=mixing,
    )

    if (
        model
        is OrbitalBridgeSaptHStateBatchedSimulation
    ):
        kwargs[
            "orbital_bridge_strength"
        ] = strength

    return model(
        **kwargs
    )


def h3_geometry(
    radius=0.94,
):
    return (
        [
            "H",
            "H",
            "H",
        ],
        np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    radius,
                    0.0,
                    0.0,
                ],
                [
                    2.0 * radius,
                    0.0,
                    0.0,
                ],
            ],
            dtype=float,
        ),
    )


def h2_geometry():
    h = int(
        R.ELEMENT_INDEX[
            "H"
        ]
    )

    re = float(
        R.BOND_LENGTH[
            h,
            h,
        ]
    )

    return (
        [
            "H",
            "H",
        ],
        np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    re,
                    0.0,
                    0.0,
                ],
            ],
            dtype=float,
        ),
    )


def energy(
    simulation,
):
    return float(
        simulation.potential_per_box[
            0
        ]
    )


def test_two_state_bridge_gate_is_exactly_existing_contact_overlap():
    states = (
        (
            0,
        ),
        (
            1,
        ),
    )

    edge_atoms = (
        (
            0,
            1,
        ),
        (
            1,
            2,
        ),
    )

    edge_tapers = (
        torch.tensor(
            0.80,
            dtype=DTYPE,
        ),
        torch.tensor(
            0.60,
            dtype=DTYPE,
        ),
    )

    types = R.types_from_symbols(
        [
            "H",
            "H",
            "H",
        ]
    )

    like = torch.stack(
        edge_tapers
    )

    gates, transitions = (
        _bridge_gate_maps(
            states,
            edge_atoms,
            edge_tapers,
            types,
            like=like,
        )
    )

    expected = (
        hs._contact_overlap(
            edge_tapers[
                0
            ],
            edge_tapers[
                1
            ],
        )
    )

    assert (
        0,
        1,
    ) in transitions

    assert gates[
        0
    ][
        1
    ].item() == pytest.approx(
        expected.item()
    )

    assert gates[
        1
    ][
        0
    ].item() == pytest.approx(
        expected.item()
    )


def test_zero_strength_executes_exact_current_sapt_adapter():
    symbols, positions = (
        h3_geometry()
    )

    current = make_simulation(
        SaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    experimental = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.0,
    )

    assert energy(
        experimental
    ) == pytest.approx(
        energy(
            current
        ),
        abs=1.0e-12,
        rel=0.0,
    )


def test_isolated_h2_has_no_bridge_even_at_nonzero_strength():
    symbols, positions = (
        h2_geometry()
    )

    zero = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.0,
    )

    bridged = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.50,
    )

    assert energy(
        bridged
    ) == pytest.approx(
        energy(
            zero
        ),
        abs=1.0e-12,
        rel=0.0,
    )


def test_bridge_lowers_active_h3_transfer_geometry_at_fixed_mixing():
    symbols, positions = (
        h3_geometry()
    )

    zero = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.0,
    )

    bridged = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.10,
    )

    assert energy(
        bridged
    ) < energy(
        zero
    )


def test_bridge_energy_remains_autograd_finite():
    symbols, positions = (
        h3_geometry()
    )

    sim = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.10,
    )

    moved = (
        sim.positions
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    total = torch.sum(
        sim.energy_per_atom(
            moved
        )
    )

    gradient = torch.autograd.grad(
        total,
        moved,
    )[
        0
    ]

    assert torch.isfinite(
        total
    )

    assert torch.all(
        torch.isfinite(
            gradient
        )
    )


def test_model_revision_and_strength_are_explicit():
    symbols, positions = (
        h2_geometry()
    )

    sim = make_simulation(
        OrbitalBridgeSaptHStateBatchedSimulation,
        symbols,
        positions,
        strength=0.125,
    )

    assert sim.physics_model_revision == (
        ORBITAL_BRIDGE_MODEL_REVISION
    )

    assert sim.orbital_bridge_strength == pytest.approx(
        0.125
    )


def test_negative_bridge_strength_is_rejected():
    symbols, positions = (
        h2_geometry()
    )

    with pytest.raises(
        ValueError
    ):
        make_simulation(
            OrbitalBridgeSaptHStateBatchedSimulation,
            symbols,
            positions,
            strength=-0.01,
        )
