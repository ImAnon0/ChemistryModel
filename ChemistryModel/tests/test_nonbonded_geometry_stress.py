import math

import numpy as np
import pytest

import nonbonded_continuous as con


# ============================================================
# HELPERS
# ============================================================


def make_probe(z=2.0):
    symbols = ["H", "H"]

    positions = np.asarray([
        [0.0, 0.0, z],
        [0.0, 0.0, z + 0.74144],
    ])

    weights = np.asarray([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def carbonyl_fragment(
    co_distance,
    *,
    second_h_angle_deg=120.0,
    second_h_weight=1.0,
):
    """
    Idealized carbonyl-like C bonded to O,H,H.

    C=O lies along +x.
    First H is at +120 deg.
    Second H is at -second_h_angle_deg.

    The C-O bond weight stays 1.0; multiple-bond character comes
    continuously from C-O compression.
    """

    ch = 1.101

    angle1 = math.radians(120.0)
    angle2 = math.radians(-second_h_angle_deg)

    symbols = ["C", "O", "H", "H"]

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [co_distance, 0.0, 0.0],
        [
            ch * math.cos(angle1),
            ch * math.sin(angle1),
            0.0,
        ],
        [
            ch * math.cos(angle2),
            ch * math.sin(angle2),
            0.0,
        ],
    ])

    weights = np.zeros(
        (4, 4),
        dtype=float,
    )

    weights[0, 1] = 1.0
    weights[1, 0] = 1.0

    weights[0, 2] = 1.0
    weights[2, 0] = 1.0

    weights[0, 3] = second_h_weight
    weights[3, 0] = second_h_weight

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def near_linear_fragment(angle_rad):
    """
    Central carbon with a compressed polar C-O bond and one C-H
    support bond separated by angle_rad.

    The geometry is intentionally driven toward collinearity so the
    local plane tensor collapses.
    """

    co = 1.208
    ch = 1.086

    symbols = ["C", "O", "H"]

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [co, 0.0, 0.0],
        [
            ch * math.cos(angle_rad),
            ch * math.sin(angle_rad),
            0.0,
        ],
    ])

    weights = np.zeros(
        (3, 3),
        dtype=float,
    )

    weights[0, 1] = 1.0
    weights[1, 0] = 1.0

    weights[0, 2] = 1.0
    weights[2, 0] = 1.0

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def rotate_fragment(fragment, rotation):
    return con.ContinuousFragment(
        symbols=list(fragment.symbols),
        positions=fragment.positions @ rotation.T,
        bond_weights=np.array(
            fragment.bond_weights,
            copy=True,
        ),
    )


def permute_fragment(fragment, permutation):
    permutation = list(permutation)

    positions = fragment.positions[
        permutation
    ]

    symbols = [
        fragment.symbols[i]
        for i in permutation
    ]

    weights = fragment.bond_weights[
        np.ix_(
            permutation,
            permutation,
        )
    ]

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def random_rotation(seed=7123):
    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        size=(3, 3)
    )

    q, _ = np.linalg.qr(matrix)

    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0

    return q


# ============================================================
# MULTIPLE-BOND RADIAL SMOOTHNESS
# ============================================================


def test_multiple_character_is_smooth_at_single_bond_boundary():
    re = con.SINGLE_BOND_RE[
        ("C", "O")
    ]

    h = 1.0e-5

    distances = np.asarray([
        re - 2*h,
        re - h,
        re,
        re + h,
        re + 2*h,
    ])

    values = np.asarray([
        con.multiple_character(
            carbonyl_fragment(distance),
            0,
            1,
        )
        for distance in distances
    ])

    first = np.gradient(
        values,
        distances,
    )

    assert np.all(
        np.isfinite(first)
    )

    # smootherstep is C2 and should approach zero slope as the
    # multiple-bond character switches off.
    assert abs(first[2]) < 1.0e-5


def test_multiple_character_is_smooth_at_saturation_boundary():
    re = con.SINGLE_BOND_RE[
        ("C", "O")
    ]

    saturation = (
        re
        * (
            1.0
            - con.MULTIPLE_COMPRESSION_FRACTION
        )
    )

    h = 1.0e-5

    distances = np.asarray([
        saturation - 2*h,
        saturation - h,
        saturation,
        saturation + h,
        saturation + 2*h,
    ])

    values = np.asarray([
        con.multiple_character(
            carbonyl_fragment(distance),
            0,
            1,
        )
        for distance in distances
    ])

    first = np.gradient(
        values,
        distances,
    )

    assert np.all(
        np.isfinite(first)
    )

    # Saturated side is flat and the smootherstep slope should
    # meet it smoothly.
    assert abs(first[2]) < 1.0e-5


def test_carbonyl_energy_has_no_radial_cusp():
    probe = make_probe()

    re = con.SINGLE_BOND_RE[
        ("C", "O")
    ]

    distances = np.linspace(
        re * 0.80,
        re * 1.05,
        2001,
    )

    energies = np.asarray([
        con.fragment_repulsion_energy(
            carbonyl_fragment(distance),
            probe,
        )
        for distance in distances
    ])

    first = np.gradient(
        energies,
        distances,
    )

    second = np.gradient(
        first,
        distances,
    )

    assert np.all(
        np.isfinite(energies)
    )

    assert np.all(
        np.isfinite(first)
    )

    assert np.all(
        np.isfinite(second)
    )

    # Broad spike detector rather than an accuracy target.
    median_curvature = np.median(
        np.abs(second)
    ) + 1.0e-12

    assert (
        np.max(np.abs(second))
        / median_curvature
        < 200.0
    )


# ============================================================
# FRACTIONAL-WEIGHT INVARIANCE
# ============================================================


def test_fractional_weight_rotation_invariance():
    target = carbonyl_fragment(
        1.250,
        second_h_weight=0.37,
    )

    probe = make_probe()

    original = con.fragment_repulsion_energy(
        target,
        probe,
    )

    rotation = random_rotation()

    rotated_target = rotate_fragment(
        target,
        rotation,
    )

    rotated_probe = rotate_fragment(
        probe,
        rotation,
    )

    rotated = con.fragment_repulsion_energy(
        rotated_target,
        rotated_probe,
    )

    assert rotated == pytest.approx(
        original,
        rel=1.0e-11,
        abs=1.0e-11,
    )


def test_fractional_weight_permutation_invariance():
    target = carbonyl_fragment(
        1.250,
        second_h_weight=0.37,
    )

    probe = make_probe()

    original = con.fragment_repulsion_energy(
        target,
        probe,
    )

    permuted_target = permute_fragment(
        target,
        [3, 1, 0, 2],
    )

    permuted_probe = permute_fragment(
        probe,
        [1, 0],
    )

    permuted = con.fragment_repulsion_energy(
        permuted_target,
        permuted_probe,
    )

    assert permuted == pytest.approx(
        original,
        rel=1.0e-11,
        abs=1.0e-11,
    )


# ============================================================
# LOCAL-PLANE COLLAPSE
# ============================================================


def test_plane_projector_has_no_near_linear_spike():
    """
    This is the deliberately nasty geometry test.

    A normalized plane tensor can become numerically dangerous when
    all local bond axes become almost collinear.  We require the
    projector to switch off without an ultra-narrow O(1) jump.
    """

    direction = np.asarray([
        0.0,
        0.0,
        1.0,
    ])

    angles = np.linspace(
        0.0,
        0.01,
        1001,
    )

    values = np.asarray([
        con.perpendicular_projector_value(
            near_linear_fragment(angle),
            0,
            direction,
        )
        for angle in angles
    ])

    assert np.all(
        np.isfinite(values)
    )

    jumps = np.abs(
        np.diff(values)
    )

    # A 1e-5-radian geometry change must not turn an essentially
    # absent plane into an almost fully active plane.
    assert np.max(jumps) < 0.05


def test_polar_pi_energy_has_no_near_linear_force_spike():
    probe = make_probe()

    angles = np.linspace(
        0.0,
        0.01,
        1001,
    )

    energies = np.asarray([
        con.fragment_repulsion_energy(
            near_linear_fragment(angle),
            probe,
        )
        for angle in angles
    ])

    derivative = np.gradient(
        energies,
        angles,
    )

    assert np.all(
        np.isfinite(energies)
    )

    assert np.all(
        np.isfinite(derivative)
    )

    # This is a stress threshold, not a fitted physical target.
    # We only reject extremely localized numerical stiffness.
    median = np.median(
        np.abs(derivative)
    ) + 1.0e-12

    assert (
        np.max(np.abs(derivative))
        / median
        < 100.0
    )
