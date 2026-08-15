import math

import numpy as np

import reactive as R
from h_state_reference import (
    H_STATE_MIXING,
    hydrogen_state_energy,
)


def equilibrium_pair(first, second):
    types = R.types_from_symbols([first, second])

    distance = float(
        R.BOND_LENGTH[types[0], types[1]]
    )

    positions = np.array([
        [0.0, 0.0, 0.0],
        [distance, 0.0, 0.0],
    ])

    return positions


def test_single_h_bonds_recover_existing_wells_exactly():
    for first, second in (
        ("H", "H"),
        ("C", "H"),
        ("N", "H"),
        ("O", "H"),
    ):
        positions = equilibrium_pair(first, second)

        result = hydrogen_state_energy(
            positions,
            [first, second],
        )

        types = R.types_from_symbols(
            [first, second]
        )

        depth = float(
            R.BOND_DEPTH[types[0], types[1]]
        )

        assert np.isclose(
            result.energy,
            -depth,
            atol=1e-10,
        )

        assert len(result.states) == 1
        assert len(result.edges) == 1

        edge = result.edges[0]

        assert np.isclose(
            result.occupations[edge],
            1.0,
            atol=1e-12,
        )


def test_symmetric_h3_has_half_bonds_and_042_ev_barrier():
    # This geometry is the symmetric H3 crossing for the current H-H
    # parameters at the reference mixing value.
    distance = 0.9406

    positions = np.array([
        [-distance, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [+distance, 0.0, 0.0],
    ])

    result = hydrogen_state_energy(
        positions,
        ["H", "H", "H"],
    )

    assert len(result.states) == 2

    assert np.isclose(
        result.occupations[(0, 1)],
        0.5,
        atol=1e-8,
    )

    assert np.isclose(
        result.occupations[(1, 2)],
        0.5,
        atol=1e-8,
    )

    h = R.ELEMENT_INDEX["H"]
    isolated_h2 = -float(
        R.BOND_DEPTH[h, h]
    )

    barrier = result.energy - isolated_h2

    assert np.isclose(
        barrier,
        0.4200,
        atol=5e-4,
    )


def test_three_equal_competitors_do_not_gain_extra_resonance():
    # One H surrounded by three equivalent carbons.
    #
    # The diagonal states are equivalent. Without crowding normalisation,
    # three all-to-all states would gain twice the two-state resonance
    # lowering. The reference must keep the maximum lowering at eta * D.
    radius = float(
        R.BOND_LENGTH[
            R.ELEMENT_INDEX["C"],
            R.ELEMENT_INDEX["H"],
        ]
    )

    positions = [[0.0, 0.0, 0.0]]

    for index in range(3):
        angle = 2.0 * math.pi * index / 3.0

        positions.append([
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.0,
        ])

    result = hydrogen_state_energy(
        np.asarray(positions),
        ["H", "C", "C", "C"],
    )

    assert len(result.states) == 3

    assert np.allclose(
        result.diagonal_energies,
        result.diagonal_energies[0],
        atol=1e-10,
    )

    c = R.ELEMENT_INDEX["C"]
    h = R.ELEMENT_INDEX["H"]

    depth = float(R.BOND_DEPTH[c, h])

    lowering = (
        result.diagonal_energies[0]
        - result.energy
    )

    expected = H_STATE_MIXING * depth

    assert np.isclose(
        lowering,
        expected,
        atol=1e-8,
    )


def test_equilibrium_water_selects_both_oh_bonds():
    angle = math.radians(104.5)
    distance = 0.96

    positions = np.array([
        [0.0, 0.0, 0.0],
        [distance, 0.0, 0.0],
        [
            distance * math.cos(angle),
            distance * math.sin(angle),
            0.0,
        ],
    ])

    result = hydrogen_state_energy(
        positions,
        ["O", "H", "H"],
    )

    # Current short H-H cutoff means only the two real O-H contacts are
    # candidates in this first architecture test.
    assert result.edges == (
        (0, 1),
        (0, 2),
    )

    assert result.states == (
        ((0, 1), (0, 2)),
    )

    assert np.isclose(
        result.occupations[(0, 1)],
        1.0,
    )

    assert np.isclose(
        result.occupations[(0, 2)],
        1.0,
    )

    oxygen = R.ELEMENT_INDEX["O"]
    hydrogen = R.ELEMENT_INDEX["H"]

    expected = -2.0 * float(
        R.BOND_DEPTH[oxygen, hydrogen]
    )

    assert np.isclose(
        result.energy,
        expected,
        atol=1e-10,
    )


def test_atom_permutation_does_not_change_energy():
    positions = np.array([
        [0.0, 0.0, 0.0],     # H
        [1.10, 0.0, 0.0],    # C
        [-1.10, 0.0, 0.0],   # N
        [0.0, 1.15, 0.0],    # O
    ])

    symbols = ["H", "C", "N", "O"]

    original = hydrogen_state_energy(
        positions,
        symbols,
    ).energy

    permutation = np.array([2, 0, 3, 1])

    moved = hydrogen_state_energy(
        positions[permutation],
        [symbols[index] for index in permutation],
    ).energy

    assert np.isclose(
        original,
        moved,
        atol=1e-10,
    )


def test_energy_fades_smoothly_at_existing_hh_outer_cutoff():
    h = R.ELEMENT_INDEX["H"]

    outer = float(
        R.CUTOFF_OUTER[h, h]
    )

    def energy(distance):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [distance, 0.0, 0.0],
        ])

        return hydrogen_state_energy(
            positions,
            ["H", "H"],
        ).energy

    # Outside is identically zero.
    assert energy(outer + 1e-4) == 0.0

    # The cosine taper is C1, so approaching the cutoff should also make
    # both energy and force tend toward zero.
    step = 1e-5

    numerical_force = -(
        energy(outer + step)
        - energy(outer - step)
    ) / (2.0 * step)

    assert abs(energy(outer - step)) < 1e-6
    assert abs(numerical_force) < 1e-3