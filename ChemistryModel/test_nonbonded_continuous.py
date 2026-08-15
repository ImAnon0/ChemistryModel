import csv
import json
from pathlib import Path

import numpy as np
import pytest

import nonbonded_reference as ref
import nonbonded_continuous as con


DATA = (
    Path(__file__).parent
    / "research_data"
    / "sapt"
)


def parse_atoms(text):
    raw = json.loads(text)

    symbols = [atom["element"] for atom in raw]
    positions = np.asarray(
        [atom["xyz"] for atom in raw],
        dtype=float,
    )

    return symbols, positions


def load_rows(filename):
    path = DATA / filename

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("status", "ok") == "ok"
        ]


def static_bonds_for_row(row, mode="single"):
    symbols, _ = parse_atoms(
        row["target_atoms_json"]
    )

    explicit = row.get(
        "target_bond_orders_json",
        "",
    )

    if explicit:
        return [
            ref.Bond(
                int(a),
                int(b),
                float(order),
            )
            for a, b, order in json.loads(explicit)
        ]

    bonds = []

    for a, b in json.loads(
        row["target_bonds_json"]
    ):
        a = int(a)
        b = int(b)
        order = 1.0

        if mode == "formaldehyde":
            if {symbols[a], symbols[b]} == {"C", "O"}:
                order = 2.0

        elif mode == "ethylene":
            if symbols[a] == "C" and symbols[b] == "C":
                order = 2.0

        bonds.append(
            ref.Bond(
                a,
                b,
                order,
            )
        )

    return bonds


def fragments_for_row(row, mode="single"):
    symbols, positions = parse_atoms(
        row["target_atoms_json"]
    )

    bonds = static_bonds_for_row(
        row,
        mode,
    )

    static_target = ref.Fragment(
        symbols=symbols,
        positions=positions,
        bonds=bonds,
    )

    weights = con.weights_from_bonds(
        len(symbols),
        bonds,
    )

    continuous_target = con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )

    probe_symbols, probe_positions = parse_atoms(
        row["probe_atoms_json"]
    )

    static_probe = ref.Fragment(
        symbols=probe_symbols,
        positions=probe_positions,
        bonds=[
            ref.Bond(
                0,
                1,
                1.0,
            )
        ],
    )

    probe_weights = con.weights_from_bonds(
        2,
        [(0, 1)],
    )

    continuous_probe = con.ContinuousFragment(
        symbols=probe_symbols,
        positions=probe_positions,
        bond_weights=probe_weights,
    )

    return (
        static_target,
        static_probe,
        continuous_target,
        continuous_probe,
    )


def compare_row(row, mode="single"):
    (
        static_target,
        static_probe,
        continuous_target,
        continuous_probe,
    ) = fragments_for_row(
        row,
        mode,
    )

    static_energy = ref.fragment_repulsion_energy(
        static_target,
        static_probe,
    )

    continuous_energy = con.fragment_repulsion_energy(
        continuous_target,
        continuous_probe,
    )

    return (
        static_energy,
        continuous_energy,
    )


@pytest.mark.parametrize(
    "filename,system,mode",
    [
        (
            "sapt_mixed_environment.csv",
            "CH3OH",
            "single",
        ),
        (
            "sapt_mixed_environment.csv",
            "CH3NH2",
            "single",
        ),
        (
            "sapt_ethylene.csv",
            "C2H4",
            "ethylene",
        ),
    ],
)
def test_continuous_matches_static_on_nonpolar_pi_systems(
    filename,
    system,
    mode,
):
    rows = [
        row
        for row in load_rows(filename)
        if row["system"] == system
    ]

    relative_errors = []

    for row in rows:
        static, continuous = compare_row(
            row,
            mode,
        )

        relative_errors.append(
            abs(continuous - static)
            / static
        )

    assert max(relative_errors) < 1e-9


def test_continuous_matches_static_on_formaldehyde():
    rows = [
        row
        for row in load_rows(
            "sapt_molecular_holdout.csv"
        )
        if row["system"] == "CH2O"
    ]

    relative_errors = []

    for row in rows:
        static, continuous = compare_row(
            row,
            "formaldehyde",
        )

        relative_errors.append(
            abs(continuous - static)
            / static
        )

    assert max(relative_errors) < 1e-9


def test_continuous_matches_static_on_acetaldehyde():
    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    relative_errors = []

    for row in rows:
        static, continuous = compare_row(
            row,
            "single",
        )

        relative_errors.append(
            abs(continuous - static)
            / static
        )

    assert max(relative_errors) < 1e-4


def make_weight_sweep_fragment(second_weight):
    symbols = [
        "O",
        "H",
        "H",
    ]

    angle = np.deg2rad(
        104.47
    )

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.960, 0.0, 0.0],
        [
            0.960*np.cos(angle),
            0.960*np.sin(angle),
            0.0,
        ],
    ])

    weights = np.zeros(
        (3, 3),
        dtype=float,
    )

    weights[0, 1] = 1.0
    weights[1, 0] = 1.0

    weights[0, 2] = second_weight
    weights[2, 0] = second_weight

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def make_probe():
    symbols = [
        "H",
        "H",
    ]

    positions = np.asarray([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.74144],
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


def test_energy_is_continuous_through_bond_weight_sweep():
    probe = make_probe()

    weights = np.linspace(
        0.0,
        1.0,
        1001,
    )

    energies = np.asarray([
        con.fragment_repulsion_energy(
            make_weight_sweep_fragment(weight),
            probe,
        )
        for weight in weights
    ])

    assert np.all(
        np.isfinite(energies)
    )

    jumps = np.abs(
        np.diff(energies)
    )

    assert np.max(jumps) < 0.02


def test_first_derivative_is_continuous_through_weight_sweep():
    probe = make_probe()

    weights = np.linspace(
        0.0,
        1.0,
        2001,
    )

    energies = np.asarray([
        con.fragment_repulsion_energy(
            make_weight_sweep_fragment(weight),
            probe,
        )
        for weight in weights
    ])

    derivative = np.gradient(
        energies,
        weights,
    )

    derivative_change = np.abs(
        np.diff(derivative)
    )

    assert np.all(
        np.isfinite(derivative)
    )

    assert np.max(
        derivative_change
    ) < 0.1


def carbonyl_with_spectator_weight(spectator_weight):
    symbols = [
        "C",
        "O",
        "H",
        "H",
    ]

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [1.208, 0.0, 0.0],
        [-0.580, +0.935, 0.0],
        [-0.580, -0.935, 0.0],
    ])

    weights = np.zeros(
        (4, 4),
        dtype=float,
    )

    weights[0, 1] = 1.0
    weights[1, 0] = 1.0

    weights[0, 2] = 1.0
    weights[2, 0] = 1.0

    weights[0, 3] = spectator_weight
    weights[3, 0] = spectator_weight

    return con.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )


def test_plane_response_is_smooth_when_support_changes():
    direction = np.asarray([
        0.0,
        0.0,
        1.0,
    ])

    weights = np.linspace(
        0.0,
        1.0,
        2001,
    )

    values = np.asarray([
        con.polar_pi_correction(
            carbonyl_with_spectator_weight(weight),
            0,
            direction,
        )
        for weight in weights
    ])

    assert np.all(
        np.isfinite(values)
    )

    changes = np.abs(
        np.diff(values)
    )

    assert np.max(
        changes
    ) < 0.005


def test_multibond_gate_has_correct_endpoints():
    one = make_weight_sweep_fragment(
        0.0
    )

    two = make_weight_sweep_fragment(
        1.0
    )

    assert con.multibond_gate(
        one,
        0,
    ) == pytest.approx(
        0.0,
        abs=1e-14,
    )

    assert con.multibond_gate(
        two,
        0,
    ) == pytest.approx(
        1.0,
        abs=1e-14,
    )


def test_smootherstep_endpoints():
    assert con.smootherstep01(-1.0) == 0.0
    assert con.smootherstep01(0.0) == 0.0
    assert con.smootherstep01(1.0) == 1.0
    assert con.smootherstep01(2.0) == 1.0
    assert con.smootherstep01(
        0.5
    ) == pytest.approx(
        0.5
    )
