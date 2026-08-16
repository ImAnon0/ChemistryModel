"""
Focused tests for the state-aware SAPT valence experiment.

Run:
    py -m pytest test_sapt_state_aware_valence_torch.py -q

Important:
The state-aware model is NOT expected to reproduce the current H3 energy
bit-for-bit. That was an incorrect assumption in the first test version.

Why:
The whole point of this experiment is to propagate H-state occupancy back into
the base coordination / bond-order / environment machinery. H3 also contains
an unoccupied competing H-H edge, so its base valence bookkeeping legitimately
changes slightly. What should remain true is:
  - isolated H2 is unchanged;
  - H3 remains symmetric;
  - the H3 shift is small enough that the barrier can be explicitly checked
    and, if needed, re-anchored after the architectural test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import reactive as R
import hf_surface_scan as scan

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)

from sapt_state_aware_valence_torch import (
    StateAwareValenceSaptHStateBatchedSimulation,
    _state_covalent_taper,
    _bond_order_from_taper,
)


DTYPE = torch.float64


def build(
    model,
    symbols,
    positions,
):
    return model(
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
        h_state_mixing=SAPT_H_STATE_MIXING,
    )


def energy(sim):
    return float(
        sim.potential_per_box[
            0
        ]
    )


def test_state_taper_removes_only_unoccupied_h_edge():
    taper = torch.tensor(
        [
            [
                0.0,
                0.8,
                0.6,
            ],
            [
                0.8,
                0.0,
                0.7,
            ],
            [
                0.6,
                0.7,
                0.0,
            ],
        ],
        dtype=DTYPE,
    )

    neighbours = torch.tensor(
        [
            [
                0,
                1,
                2,
            ],
            [
                0,
                1,
                2,
            ],
            [
                0,
                1,
                2,
            ],
        ],
        dtype=torch.long,
    )

    filtered = _state_covalent_taper(
        taper=taper,
        neighbours=neighbours,
        edge_atoms=(
            (
                0,
                1,
            ),
            (
                1,
                2,
            ),
        ),
        state=(
            0,
        ),
    )

    assert filtered[
        0,
        1
    ].item() == pytest.approx(
        0.8
    )

    assert filtered[
        1,
        0
    ].item() == pytest.approx(
        0.8
    )

    assert filtered[
        1,
        2
    ].item() == pytest.approx(
        0.0
    )

    assert filtered[
        2,
        1
    ].item() == pytest.approx(
        0.0
    )

    assert filtered[
        0,
        2
    ].item() == pytest.approx(
        0.6
    )


def test_h2_is_unchanged():
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

    symbols = [
        "H",
        "H",
    ]

    positions = np.array(
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
    )

    current = build(
        SaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    state_aware = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    assert energy(
        state_aware
    ) == pytest.approx(
        energy(
            current
        ),
        abs=1.0e-10,
        rel=0.0,
    )


def test_symmetric_h3_changes_only_slightly_at_current_anchor_geometry():
    radius = 0.935

    symbols = [
        "H",
        "H",
        "H",
    ]

    positions = np.array(
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
    )

    current = build(
        SaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    state_aware = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    shift = (
        energy(
            state_aware
        )
        - energy(
            current
        )
    )

    # The first implementation showed ~+0.00171 eV here. This is an intended
    # consequence of making H-state occupancy feed back into the base valence
    # bookkeeping, not a regression to suppress. Keep a tight "small shift"
    # guard so a later accidental large H3 change is still caught.
    assert abs(
        shift
    ) < 0.005


def test_symmetric_h3_is_mirror_invariant():
    radius = 0.935

    symbols = [
        "H",
        "H",
        "H",
    ]

    forward = np.array(
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
    )

    mirrored = forward[
        ::-1
    ].copy()

    first = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        forward,
    )

    second = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        mirrored,
    )

    assert energy(
        first
    ) == pytest.approx(
        energy(
            second
        ),
        abs=1.0e-10,
        rel=0.0,
    )


def test_unoccupied_methane_transfer_edge_reduces_carbon_coordination():
    scan.apply_system(
        "methane"
    )

    spectators = np.asarray(
        scan.SYSTEMS[
            "methane"
        ][
            "frozen"
        ],
        dtype=float,
    )

    symbols, positions = scan.SYSTEMS[
        "methane"
    ][
        "geometry"
    ](
        1.44,
        0.85,
        spectators,
    )

    sim = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    from batched_torch import BatchedReactiveSimulation

    probe_positions = (
        sim.positions
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    BatchedReactiveSimulation.energy_per_atom(
        sim,
        probe_positions,
    )

    values = sim._reactive_intermediates[
        1
    ]

    neighbours_numpy = (
        values[
            "neighbours"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    active_numpy = (
        (
            values[
                "taper"
            ]
            .detach()
            .cpu()
            .numpy()
            > 1.0e-12
        )
        & sim.neighbour_mask
        .detach()
        .cpu()
        .numpy()
    )

    (
        edge_atoms,
        _,
        _,
    ) = sim._active_edges_for_box(
        0,
        values,
        neighbours_numpy,
        active_numpy,
    )

    breaking = None

    for edge_index, pair in enumerate(
        edge_atoms
    ):
        if set(
            pair
        ) == {
            0,
            1,
        }:
            breaking = edge_index
            break

    assert breaking is not None

    state = tuple(
        edge_index
        for edge_index, pair
        in enumerate(
            edge_atoms
        )
        if edge_index != breaking
        and 1 not in pair
    )

    state_taper = _state_covalent_taper(
        taper=values[
            "taper"
        ],
        neighbours=values[
            "neighbours"
        ],
        edge_atoms=edge_atoms,
        state=state,
    )

    (
        _,
        coordination,
        _,
    ) = _bond_order_from_taper(
        sim,
        state_taper,
        values[
            "neighbours"
        ],
    )

    assert coordination[
        0
    ].item() < values[
        "coordination"
    ][
        0
    ].item()


def test_autograd_is_finite():
    scan.apply_system(
        "formaldehyde"
    )

    spectators = np.asarray(
        scan.SYSTEMS[
            "formaldehyde"
        ][
            "frozen"
        ],
        dtype=float,
    )

    symbols, positions = scan.SYSTEMS[
        "formaldehyde"
    ][
        "geometry"
    ](
        1.28,
        1.01,
        spectators,
    )

    sim = build(
        StateAwareValenceSaptHStateBatchedSimulation,
        symbols,
        positions,
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
