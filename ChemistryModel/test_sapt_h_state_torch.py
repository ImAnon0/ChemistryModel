import math

import numpy as np
import torch
import pytest

import reactive as R
import nonbonded_continuous_torch as nb

from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    _descriptor_weights_for_state,
)


DTYPE = torch.float64


def synthetic_formaldehyde_descriptor(
    heavy_contact=1.0,
):
    """
    C=O plus two occupied C-H edges and one unoccupied incoming-H edge.

    Only the heavy-heavy C=O contact is supplied through the reactive
    neighbour table. H covalency is supplied by the selected H state.
    """

    symbols = [
        "C",
        "O",
        "H",
        "H",
        "H",
    ]

    types = R.types_from_symbols(
        symbols
    )

    atom_count = len(
        symbols
    )

    # Padded directed neighbour table. Only C<->O is needed here because
    # H-containing contacts are deliberately supplied through edge_atoms.
    neighbours = torch.zeros(
        (
            atom_count,
            2,
        ),
        dtype=torch.long,
    )

    neighbour_mask = torch.zeros(
        (
            atom_count,
            2,
        ),
        dtype=torch.bool,
    )

    neighbours[0, 0] = 1
    neighbour_mask[0, 0] = True

    neighbours[1, 0] = 0
    neighbour_mask[1, 0] = True

    heavy_value = torch.as_tensor(
        heavy_contact,
        dtype=DTYPE,
    )

    taper = torch.zeros(
        (
            atom_count,
            2,
        ),
        dtype=DTYPE,
    )

    taper = (
        taper
        + heavy_value
        * torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=DTYPE,
        )
    )

    # C-H donor, C-H spectator, H-H transfer alternative.
    edge_atoms = (
        (0, 2),
        (0, 3),
        (2, 4),
    )

    edge_tapers = (
        torch.tensor(
            1.0,
            dtype=DTYPE,
        ),
        torch.tensor(
            1.0,
            dtype=DTYPE,
        ),
        torch.tensor(
            0.80,
            dtype=DTYPE,
        ),
    )

    # Reactant diabatic state: both real C-H bonds occupied; the incoming
    # H-H alternative is unoccupied.
    state = (
        0,
        1,
    )

    weights = (
        _descriptor_weights_for_state(
            box=0,
            per_box=atom_count,
            types_numpy=types,
            neighbours=neighbours,
            neighbour_mask=(
                neighbour_mask
            ),
            taper=taper,
            edge_atoms=edge_atoms,
            edge_tapers=edge_tapers,
            state=state,
        )
    )

    return (
        symbols,
        weights,
    )


def test_descriptor_includes_heavy_heavy_and_only_occupied_h_edges():
    _, weights = (
        synthetic_formaldehyde_descriptor()
    )

    # Persistent C=O environment.
    assert weights[
        0,
        1,
    ].item() == 1.0

    assert weights[
        1,
        0,
    ].item() == 1.0

    # Occupied C-H state edges.
    assert weights[
        0,
        2,
    ].item() == 1.0

    assert weights[
        0,
        3,
    ].item() == 1.0

    # The competing H-H edge is a candidate, not a covalent neighbour in
    # this diabatic state.
    assert weights[
        2,
        4,
    ].item() == 0.0

    assert torch.allclose(
        weights,
        weights.T,
        atol=0.0,
        rtol=0.0,
    )


def test_switching_h_state_keeps_carbonyl_environment():
    symbols = [
        "C",
        "O",
        "H",
        "H",
        "H",
    ]

    types = R.types_from_symbols(
        symbols
    )

    neighbours = torch.zeros(
        (5, 2),
        dtype=torch.long,
    )

    mask = torch.zeros(
        (5, 2),
        dtype=torch.bool,
    )

    neighbours[0, 0] = 1
    neighbours[1, 0] = 0
    mask[0, 0] = True
    mask[1, 0] = True

    taper = torch.zeros(
        (5, 2),
        dtype=DTYPE,
    )

    taper[0, 0] = 1.0
    taper[1, 0] = 1.0

    edge_atoms = (
        (0, 2),
        (0, 3),
        (2, 4),
    )

    edge_tapers = tuple(
        torch.tensor(
            value,
            dtype=DTYPE,
        )
        for value
        in (
            1.0,
            1.0,
            0.8,
        )
    )

    reactant = (
        _descriptor_weights_for_state(
            box=0,
            per_box=5,
            types_numpy=types,
            neighbours=neighbours,
            neighbour_mask=mask,
            taper=taper,
            edge_atoms=edge_atoms,
            edge_tapers=edge_tapers,
            state=(
                0,
                1,
            ),
        )
    )

    product = (
        _descriptor_weights_for_state(
            box=0,
            per_box=5,
            types_numpy=types,
            neighbours=neighbours,
            neighbour_mask=mask,
            taper=taper,
            edge_atoms=edge_atoms,
            edge_tapers=edge_tapers,
            state=(
                1,
                2,
            ),
        )
    )

    # Heavy covalent environment is state-independent.
    assert reactant[
        0,
        1,
    ].item() == 1.0

    assert product[
        0,
        1,
    ].item() == 1.0

    # H occupancy changes with the diabatic state.
    assert reactant[
        0,
        2,
    ].item() == 1.0

    assert reactant[
        2,
        4,
    ].item() == 0.0

    assert product[
        0,
        2,
    ].item() == 0.0

    assert product[
        2,
        4,
    ].item() == pytest.approx(
        0.8
    )


def test_carbonyl_polar_pi_descriptor_survives_adapter_weights():
    symbols, weights = (
        synthetic_formaldehyde_descriptor()
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],      # C
            [0.0, 1.20, 0.0],     # O, compressed C=O
            [0.95, -0.55, 0.0],   # donor H
            [-0.95, -0.55, 0.0],  # spectator H
            [0.0, -2.0, 1.5],     # incoming H, not covalent in this state
        ],
        dtype=DTYPE,
    )

    fragment = (
        nb.ContinuousTorchFragment(
            symbols=symbols,
            positions=positions,
            bond_weights=weights,
        )
    )

    out_of_plane = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=DTYPE,
    )

    strength = (
        nb.polar_multiple_bond_strength(
            fragment,
            0,
        )
    )

    projector = (
        nb.perpendicular_projector_value(
            fragment,
            0,
            out_of_plane,
        )
    )

    correction = (
        nb.polar_pi_correction(
            fragment,
            0,
            out_of_plane,
        )
    )

    assert strength.item() > 0.0
    assert projector.item() > 0.9
    assert correction.item() > 0.0

    # Demonstrate the specific failure the v1 adapter had: if C=O is absent
    # from the covalent environment, the carbonyl polar-pi strength vanishes.
    h_only = weights.clone()
    h_only[0, 1] = 0.0
    h_only[1, 0] = 0.0

    fragment_without_co = (
        nb.ContinuousTorchFragment(
            symbols=symbols,
            positions=positions,
            bond_weights=h_only,
        )
    )

    missing = (
        nb.polar_multiple_bond_strength(
            fragment_without_co,
            0,
        )
    )

    assert abs(
        missing.item()
    ) < 1.0e-12


def test_heavy_contact_weight_remains_autograd_connected():
    heavy_contact = torch.tensor(
        0.63,
        dtype=DTYPE,
        requires_grad=True,
    )

    _, weights = (
        synthetic_formaldehyde_descriptor(
            heavy_contact
        )
    )

    value = weights[
        0,
        1,
    ]

    gradient = torch.autograd.grad(
        value,
        heavy_contact,
    )[0]

    assert gradient.item() == 1.0


def test_real_distance_taper_separates_carbonyl_from_water_oo_contact():
    c = R.ELEMENT_INDEX[
        "C"
    ]

    o = R.ELEMENT_INDEX[
        "O"
    ]

    co_taper = R.smooth_cutoff(
        np.asarray(
            [1.20]
        ),
        np.asarray(
            [
                R.CUTOFF_INNER[
                    c,
                    o,
                ]
            ]
        ),
        np.asarray(
            [
                R.CUTOFF_OUTER[
                    c,
                    o,
                ]
            ]
        ),
    )[0]

    oo_taper = R.smooth_cutoff(
        np.asarray(
            [2.365]
        ),
        np.asarray(
            [
                R.CUTOFF_INNER[
                    o,
                    o,
                ]
            ]
        ),
        np.asarray(
            [
                R.CUTOFF_OUTER[
                    o,
                    o,
                ]
            ]
        ),
    )[0]

    assert co_taper == 1.0
    assert oo_taper == 0.0


def test_sapt_h_state_mixing_stays_frozen():
    assert SAPT_H_STATE_MIXING == pytest.approx(
        0.534590721,
        abs=0.0,
    )
