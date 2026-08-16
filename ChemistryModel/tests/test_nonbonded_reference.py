import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import nonbonded_reference as nb


DATA = Path(__file__).resolve().parents[1] / "research_data" / "sapt"


# ============================================================
# HELPERS
# ============================================================


def parse_atoms(text):
    raw = json.loads(text)

    symbols = [
        atom["element"]
        for atom in raw
    ]

    positions = np.asarray(
        [
            atom["xyz"]
            for atom in raw
        ],
        dtype=float,
    )

    return symbols, positions


def parse_bonds(text, default_order=1.0):
    return [
        nb.Bond(
            int(a),
            int(b),
            float(default_order),
        )
        for a, b in json.loads(text)
    ]


def fragment_from_row(
    row,
    *,
    bond_order_mode="single",
):
    symbols, positions = parse_atoms(
        row["target_atoms_json"]
    )

    raw_bonds = json.loads(
        row["target_bonds_json"]
    )

    bonds = []

    if (
        "target_bond_orders_json" in row
        and row["target_bond_orders_json"]
    ):
        for a, b, order in json.loads(
            row["target_bond_orders_json"]
        ):
            bonds.append(
                nb.Bond(
                    int(a),
                    int(b),
                    float(order),
                )
            )

    else:
        for a, b in raw_bonds:

            order = 1.0

            if bond_order_mode == "formaldehyde":
                pair = {
                    symbols[int(a)],
                    symbols[int(b)],
                }

                if pair == {"C", "O"}:
                    order = 2.0

            elif bond_order_mode == "ethylene":
                if (
                    symbols[int(a)] == "C"
                    and symbols[int(b)] == "C"
                ):
                    order = 2.0

            bonds.append(
                nb.Bond(
                    int(a),
                    int(b),
                    order,
                )
            )

    return nb.Fragment(
        symbols=symbols,
        positions=positions,
        bonds=bonds,
    )


def probe_from_row(row):
    symbols, positions = parse_atoms(
        row["probe_atoms_json"]
    )

    assert symbols == ["H", "H"]

    return nb.Fragment(
        symbols=symbols,
        positions=positions,
        bonds=[
            nb.Bond(
                0,
                1,
                1.0,
            )
        ],
    )


def load_rows(filename):
    path = DATA / filename

    assert path.exists(), (
        f"Missing SAPT regression data: {path}"
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:

        return [
            row
            for row in csv.DictReader(handle)
            if row.get("status", "ok") == "ok"
        ]


def prediction_for_row(
    row,
    *,
    bond_order_mode="single",
    zeta=nb.ZETA,
):
    target = fragment_from_row(
        row,
        bond_order_mode=bond_order_mode,
    )

    probe = probe_from_row(row)

    return nb.fragment_repulsion_energy(
        target,
        probe,
        zeta=zeta,
    )


def percentage_error(
    prediction,
    target,
):
    return (
        100.0
        * abs(prediction - target)
        / target
    )


def mape_for_rows(
    rows,
    *,
    bond_order_mode="single",
    zeta=nb.ZETA,
):
    errors = []

    for row in rows:

        target = float(
            row["exch10_eV"]
        )

        prediction = prediction_for_row(
            row,
            bond_order_mode=bond_order_mode,
            zeta=zeta,
        )

        errors.append(
            percentage_error(
                prediction,
                target,
            )
        )

    return float(
        np.mean(errors)
    )


def worst_for_rows(
    rows,
    *,
    bond_order_mode="single",
    zeta=nb.ZETA,
):
    errors = []

    for row in rows:

        target = float(
            row["exch10_eV"]
        )

        prediction = prediction_for_row(
            row,
            bond_order_mode=bond_order_mode,
            zeta=zeta,
        )

        errors.append(
            percentage_error(
                prediction,
                target,
            )
        )

    return float(
        np.max(errors)
    )


def rows_for_approach(
    rows,
    approach,
):
    return [
        row
        for row in rows
        if row["approach"] == approach
    ]


def random_rotation(seed=12345):
    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        size=(3, 3)
    )

    q, _ = np.linalg.qr(matrix)

    # Enforce a proper rotation rather than reflection.
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0

    return q


def transform_fragment(
    fragment,
    *,
    rotation=None,
    translation=None,
):
    positions = np.array(
        fragment.positions,
        copy=True,
    )

    if rotation is not None:
        positions = (
            positions
            @ rotation.T
        )

    if translation is not None:
        positions = (
            positions
            + np.asarray(
                translation,
                dtype=float,
            )
        )

    return nb.Fragment(
        symbols=list(
            fragment.symbols
        ),
        positions=positions,
        bonds=list(
            fragment.bonds
        ),
    )


def permute_fragment(
    fragment,
    permutation,
):
    permutation = list(
        permutation
    )

    inverse = {
        old: new
        for new, old
        in enumerate(permutation)
    }

    symbols = [
        fragment.symbols[old]
        for old in permutation
    ]

    positions = np.asarray(
        [
            fragment.positions[old]
            for old in permutation
        ],
        dtype=float,
    )

    bonds = [
        nb.Bond(
            inverse[bond.i],
            inverse[bond.j],
            bond.order,
        )
        for bond in fragment.bonds
    ]

    return nb.Fragment(
        symbols=symbols,
        positions=positions,
        bonds=bonds,
    )


# ============================================================
# PARAMETER PROVENANCE
# ============================================================


def test_frozen_parameter_values():

    assert nb.ELEMENT_PARAMETERS["H"].A == pytest.approx(
        3.95924
    )

    assert nb.ELEMENT_PARAMETERS["H"].B == pytest.approx(
        4.20796
    )

    assert nb.ELEMENT_PARAMETERS["H"].k == pytest.approx(
        -0.27107
    )

    assert nb.ELEMENT_PARAMETERS["C"].A == pytest.approx(
        14.24570
    )

    assert nb.ELEMENT_PARAMETERS["C"].B == pytest.approx(
        3.75862
    )

    assert nb.ELEMENT_PARAMETERS["C"].k == pytest.approx(
        +0.33993
    )

    assert nb.ELEMENT_PARAMETERS["N"].A == pytest.approx(
        18.09959
    )

    assert nb.ELEMENT_PARAMETERS["N"].B == pytest.approx(
        4.24701
    )

    assert nb.ELEMENT_PARAMETERS["N"].k == pytest.approx(
        +0.12138
    )

    assert nb.ELEMENT_PARAMETERS["O"].A == pytest.approx(
        20.93635
    )

    assert nb.ELEMENT_PARAMETERS["O"].B == pytest.approx(
        4.73944
    )

    assert nb.ELEMENT_PARAMETERS["O"].k == pytest.approx(
        +0.03677
    )

    assert nb.LAMBDA_H == pytest.approx(
        0.547542
    )

    assert nb.G2 == pytest.approx(
        +0.230153
    )

    assert nb.G3 == pytest.approx(
        -0.126666
    )

    assert nb.H1 == pytest.approx(
        -0.19440
    )

    assert nb.ZETA == pytest.approx(
        +0.271733629
    )


# ============================================================
# BASIC MATHEMATICAL BEHAVIOUR
# ============================================================


def test_repulsion_is_positive():

    fragment_a = nb.Fragment(
        symbols=["H", "H"],
        positions=np.array([
            [-0.37072, 0.0, 0.0],
            [+0.37072, 0.0, 0.0],
        ]),
        bonds=[
            nb.Bond(0, 1)
        ],
    )

    fragment_b = nb.Fragment(
        symbols=["H", "H"],
        positions=np.array([
            [1.5, -0.37072, 0.0],
            [1.5, +0.37072, 0.0],
        ]),
        bonds=[
            nb.Bond(0, 1)
        ],
    )

    energy = nb.fragment_repulsion_energy(
        fragment_a,
        fragment_b,
    )

    assert math.isfinite(energy)
    assert energy > 0.0


def test_simple_repulsion_decreases_with_separation():

    fragment_a = nb.Fragment(
        symbols=["H"],
        positions=np.array([
            [0.0, 0.0, 0.0]
        ]),
        bonds=[],
    )

    energies = []

    for distance in np.linspace(
        1.0,
        4.0,
        40,
    ):

        fragment_b = nb.Fragment(
            symbols=["H"],
            positions=np.array([
                [distance, 0.0, 0.0]
            ]),
            bonds=[],
        )

        energies.append(
            nb.fragment_repulsion_energy(
                fragment_a,
                fragment_b,
            )
        )

    differences = np.diff(
        energies
    )

    assert np.all(
        differences < 0.0
    )


# ============================================================
# INVARIANCE TESTS
# ============================================================


def test_translation_invariance():

    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    row = rows_for_approach(
        rows,
        "out_of_plane",
    )[2]

    target = fragment_from_row(
        row
    )

    probe = probe_from_row(
        row
    )

    original = nb.fragment_repulsion_energy(
        target,
        probe,
    )

    shift = np.array([
        3.217,
        -1.902,
        4.111,
    ])

    moved_target = transform_fragment(
        target,
        translation=shift,
    )

    moved_probe = transform_fragment(
        probe,
        translation=shift,
    )

    moved = nb.fragment_repulsion_energy(
        moved_target,
        moved_probe,
    )

    assert moved == pytest.approx(
        original,
        rel=1e-12,
        abs=1e-12,
    )


def test_rotation_invariance():

    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    row = rows_for_approach(
        rows,
        "out_of_plane",
    )[1]

    target = fragment_from_row(
        row
    )

    probe = probe_from_row(
        row
    )

    original = nb.fragment_repulsion_energy(
        target,
        probe,
    )

    rotation = random_rotation()

    rotated_target = transform_fragment(
        target,
        rotation=rotation,
    )

    rotated_probe = transform_fragment(
        probe,
        rotation=rotation,
    )

    rotated = nb.fragment_repulsion_energy(
        rotated_target,
        rotated_probe,
    )

    assert rotated == pytest.approx(
        original,
        rel=1e-11,
        abs=1e-11,
    )


def test_atom_permutation_invariance():

    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    row = rows_for_approach(
        rows,
        "out_of_plane",
    )[1]

    target = fragment_from_row(
        row
    )

    probe = probe_from_row(
        row
    )

    original = nb.fragment_repulsion_energy(
        target,
        probe,
    )

    permutation = list(
        reversed(
            range(
                len(target.symbols)
            )
        )
    )

    permuted_target = permute_fragment(
        target,
        permutation,
    )

    permuted_probe = permute_fragment(
        probe,
        [1, 0],
    )

    permuted = nb.fragment_repulsion_energy(
        permuted_target,
        permuted_probe,
    )

    assert permuted == pytest.approx(
        original,
        rel=1e-11,
        abs=1e-11,
    )


# ============================================================
# POLAR-PI STRUCTURAL CONTROL
# ============================================================


def test_ethylene_is_exactly_invariant_to_polar_pi_term():

    rows = load_rows(
        "sapt_ethylene.csv"
    )

    differences = []

    for row in rows:

        without = prediction_for_row(
            row,
            bond_order_mode="ethylene",
            zeta=0.0,
        )

        with_term = prediction_for_row(
            row,
            bond_order_mode="ethylene",
            zeta=nb.ZETA,
        )

        differences.append(
            abs(
                with_term - without
            )
        )

    assert max(differences) < 1e-12


# ============================================================
# ACETALDEHYDE DEVELOPMENT REGRESSION
# ============================================================


def test_acetaldehyde_polar_pi_term_fixes_out_of_plane():

    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    subset = rows_for_approach(
        rows,
        "out_of_plane",
    )

    before = mape_for_rows(
        subset,
        zeta=0.0,
    )

    after = mape_for_rows(
        subset,
        zeta=nb.ZETA,
    )

    # Regression values from the experiment:
    #
    # before ~46.60%
    # after  ~1.08%
    #
    # Leave a little numerical tolerance around them.

    assert 44.0 < before < 49.0
    assert after < 2.0


def test_acetaldehyde_in_plane_directions_are_unchanged():

    rows = load_rows(
        "sapt_acetaldehyde.csv"
    )

    for approach in (
        "oxygen_end",
        "carbonyl_carbon_end",
        "aldehyde_CH",
    ):

        subset = rows_for_approach(
            rows,
            approach,
        )

        for row in subset:

            without = prediction_for_row(
                row,
                zeta=0.0,
            )

            with_term = prediction_for_row(
                row,
                zeta=nb.ZETA,
            )

            assert with_term == pytest.approx(
                without,
                rel=1e-12,
                abs=1e-12,
            )


# ============================================================
# UNSEEN FORMALDEHYDE TRANSFER REGRESSION
# ============================================================


def test_formaldehyde_unseen_polar_pi_transfer():

    rows = [
        row
        for row in load_rows(
            "sapt_molecular_holdout.csv"
        )
        if row["system"] == "CH2O"
    ]

    out_of_plane = rows_for_approach(
        rows,
        "out_of_plane",
    )

    before = mape_for_rows(
        out_of_plane,
        bond_order_mode="formaldehyde",
        zeta=0.0,
    )

    after = mape_for_rows(
        out_of_plane,
        bond_order_mode="formaldehyde",
        zeta=nb.ZETA,
    )

    # Frozen observed behaviour:
    #
    # before ~36.98%
    # after  ~13.12%

    assert 34.0 < before < 40.0
    assert after < 15.0
    assert after < before


def test_formaldehyde_full_unseen_set_regression():

    rows = [
        row
        for row in load_rows(
            "sapt_molecular_holdout.csv"
        )
        if row["system"] == "CH2O"
    ]

    overall = mape_for_rows(
        rows,
        bond_order_mode="formaldehyde",
        zeta=nb.ZETA,
    )

    worst = worst_for_rows(
        rows,
        bond_order_mode="formaldehyde",
        zeta=nb.ZETA,
    )

    # Experimental result:
    #
    # overall MAPE ~9.70%
    # worst         ~19.85%

    assert overall < 11.0
    assert worst < 22.0


# ============================================================
# REFERENCE DATA PRESENCE
# ============================================================


def test_research_dataset_is_present():

    expected = {
        "sapt_training.csv",
        "sapt_reverse_t.csv",
        "sapt_molecular_holdout.csv",
        "sapt_mixed_environment.csv",
        "sapt_ethylene.csv",
        "sapt_acetaldehyde.csv",
    }

    actual = {
        path.name
        for path in DATA.glob(
            "*.csv"
        )
    }

    missing = (
        expected - actual
    )

    assert not missing, (
        f"Missing research datasets: "
        f"{sorted(missing)}"
    )
