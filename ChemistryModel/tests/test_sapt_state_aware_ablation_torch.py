"""
Focused tests for state-aware component ablations.

Run:
    py -m pytest test_sapt_state_aware_ablation_torch.py -q
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

from sapt_state_aware_ablation_torch import (
    ABLATION_MODES,
    StateAwareAblationSaptHStateBatchedSimulation,
)


DTYPE = torch.float64


def build(
    model,
    symbols,
    positions,
    *,
    mode=None,
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
        h_state_mixing=SAPT_H_STATE_MIXING,
    )

    if mode is not None:
        kwargs[
            "ablation_mode"
        ] = mode

    return model(
        **kwargs
    )


def energy(
    simulation,
):
    return float(
        simulation.potential_per_box[
            0
        ]
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


@pytest.mark.parametrize(
    "mode",
    ABLATION_MODES,
)
def test_h2_unchanged_for_every_ablation(
    mode,
):
    symbols, positions = h2_geometry()

    current = build(
        SaptHStateBatchedSimulation,
        symbols,
        positions,
    )

    changed = build(
        StateAwareAblationSaptHStateBatchedSimulation,
        symbols,
        positions,
        mode=mode,
    )

    assert energy(
        changed
    ) == pytest.approx(
        energy(
            current
        ),
        abs=1.0e-10,
        rel=0.0,
    )


def test_invalid_mode_rejected():
    symbols, positions = h2_geometry()

    with pytest.raises(
        ValueError
    ):
        build(
            StateAwareAblationSaptHStateBatchedSimulation,
            symbols,
            positions,
            mode="not-a-mode",
        )


@pytest.mark.parametrize(
    "mode",
    ABLATION_MODES,
)
def test_formaldehyde_saddle_autograd_finite(
    mode,
):
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
        StateAwareAblationSaptHStateBatchedSimulation,
        symbols,
        positions,
        mode=mode,
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
