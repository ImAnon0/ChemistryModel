import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import nonbonded_continuous as npcon
import nonbonded_continuous_torch as thcon


DATA = (
    Path(__file__).parent
    / "research_data"
    / "sapt"
)


# ============================================================
# DATA HELPERS
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
            if row.get(
                "status",
                "ok",
            ) == "ok"
        ]


def numpy_fragments_from_row(row):
    symbols, positions = parse_atoms(
        row["target_atoms_json"]
    )

    bonds = [
        tuple(pair)
        for pair in json.loads(
            row["target_bonds_json"]
        )
    ]

    weights = npcon.weights_from_bonds(
        len(symbols),
        bonds,
    )

    target = npcon.ContinuousFragment(
        symbols=symbols,
        positions=positions,
        bond_weights=weights,
    )

    probe_symbols, probe_positions = (
        parse_atoms(
            row["probe_atoms_json"]
        )
    )

    probe = npcon.ContinuousFragment(
        symbols=probe_symbols,
        positions=probe_positions,
        bond_weights=np.asarray([
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
    )

    return target, probe


def torch_fragments_from_row(
    row,
    *,
    device="cpu",
):
    symbols, positions = parse_atoms(
        row["target_atoms_json"]
    )

    bonds = [
        tuple(pair)
        for pair in json.loads(
            row["target_bonds_json"]
        )
    ]

    positions_t = torch.tensor(
        positions,
        dtype=torch.float64,
        device=device,
    )

    weights_t = thcon.weights_from_bonds(
        len(symbols),
        bonds,
        dtype=torch.float64,
        device=device,
    )

    target = thcon.ContinuousTorchFragment(
        symbols=symbols,
        positions=positions_t,
        bond_weights=weights_t,
    )

    probe_symbols, probe_positions = (
        parse_atoms(
            row["probe_atoms_json"]
        )
    )

    probe_positions_t = torch.tensor(
        probe_positions,
        dtype=torch.float64,
        device=device,
    )

    probe = thcon.ContinuousTorchFragment(
        symbols=probe_symbols,
        positions=probe_positions_t,
        bond_weights=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        ),
    )

    return target, probe


# ============================================================
# FULL NUMPY <-> TORCH ENERGY AGREEMENT
# ============================================================


@pytest.mark.parametrize(
    "filename",
    [
        "sapt_molecular_holdout.csv",
        "sapt_mixed_environment.csv",
        "sapt_ethylene.csv",
        "sapt_acetaldehyde.csv",
    ],
)
def test_torch_matches_numpy_across_sapt_structures(
    filename,
):
    rows = load_rows(
        filename
    )

    relative_errors = []

    for row in rows:
        np_target, np_probe = (
            numpy_fragments_from_row(
                row
            )
        )

        th_target, th_probe = (
            torch_fragments_from_row(
                row
            )
        )

        np_energy = (
            npcon.fragment_repulsion_energy(
                np_target,
                np_probe,
            )
        )

        th_energy = (
            thcon.fragment_repulsion_energy(
                th_target,
                th_probe,
            ).item()
        )

        relative_errors.append(
            abs(
                th_energy
                - np_energy
            )
            / max(
                abs(np_energy),
                1.0e-12,
            )
        )

    assert max(
        relative_errors
    ) < 1.0e-10


# ============================================================
# CUSTOM DIFFERENTIABLE GEOMETRY
# ============================================================


def make_carbonyl_torch(
    *,
    co_distance=1.250,
    second_h_weight=0.37,
    requires_grad=False,
    device="cpu",
):
    ch = 1.101
    angle = math.radians(
        120.0
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [co_distance, 0.0, 0.0],
            [
                ch*math.cos(angle),
                ch*math.sin(angle),
                0.0,
            ],
            [
                ch*math.cos(-angle),
                ch*math.sin(-angle),
                0.0,
            ],
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=requires_grad,
    )

    weights = torch.zeros(
        (4, 4),
        dtype=torch.float64,
        device=device,
    )

    weights[0, 1] = 1.0
    weights[1, 0] = 1.0

    weights[0, 2] = 1.0
    weights[2, 0] = 1.0

    weights[0, 3] = (
        second_h_weight
    )

    weights[3, 0] = (
        second_h_weight
    )

    return thcon.ContinuousTorchFragment(
        symbols=[
            "C",
            "O",
            "H",
            "H",
        ],
        positions=positions,
        bond_weights=weights,
    )


def make_probe_torch(
    *,
    z=2.0,
    requires_grad=False,
    device="cpu",
):
    positions = torch.tensor(
        [
            [0.0, 0.0, z],
            [
                0.0,
                0.0,
                z + 0.74144,
            ],
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=requires_grad,
    )

    weights = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    return thcon.ContinuousTorchFragment(
        symbols=["H", "H"],
        positions=positions,
        bond_weights=weights,
    )


def energy_from_positions(
    target_positions,
    probe_positions,
    target_weights,
):
    target = thcon.ContinuousTorchFragment(
        symbols=[
            "C",
            "O",
            "H",
            "H",
        ],
        positions=target_positions,
        bond_weights=target_weights,
    )

    probe = thcon.ContinuousTorchFragment(
        symbols=[
            "H",
            "H",
        ],
        positions=probe_positions,
        bond_weights=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=probe_positions.dtype,
            device=probe_positions.device,
        ),
    )

    return thcon.fragment_repulsion_energy(
        target,
        probe,
    )


# ============================================================
# AUTOGRAD FORCE CHECK
# ============================================================


def test_autograd_position_gradient_matches_finite_difference():
    target = make_carbonyl_torch(
        requires_grad=True
    )

    probe = make_probe_torch(
        requires_grad=True
    )

    energy = (
        thcon.fragment_repulsion_energy(
            target,
            probe,
        )
    )

    grad_target, grad_probe = (
        torch.autograd.grad(
            energy,
            (
                target.positions,
                probe.positions,
            ),
        )
    )

    h = 1.0e-6

    target_base = (
        target.positions
        .detach()
        .clone()
    )

    probe_base = (
        probe.positions
        .detach()
        .clone()
    )

    weights = (
        target.bond_weights
        .detach()
        .clone()
    )

    fd_target = torch.zeros_like(
        target_base
    )

    fd_probe = torch.zeros_like(
        probe_base
    )

    for atom in range(
        target_base.shape[0]
    ):
        for axis in range(3):
            plus = target_base.clone()
            minus = target_base.clone()

            plus[atom, axis] += h
            minus[atom, axis] -= h

            e_plus = energy_from_positions(
                plus,
                probe_base,
                weights,
            ).item()

            e_minus = energy_from_positions(
                minus,
                probe_base,
                weights,
            ).item()

            fd_target[
                atom,
                axis,
            ] = (
                e_plus
                - e_minus
            ) / (
                2.0*h
            )

    for atom in range(
        probe_base.shape[0]
    ):
        for axis in range(3):
            plus = probe_base.clone()
            minus = probe_base.clone()

            plus[atom, axis] += h
            minus[atom, axis] -= h

            e_plus = energy_from_positions(
                target_base,
                plus,
                weights,
            ).item()

            e_minus = energy_from_positions(
                target_base,
                minus,
                weights,
            ).item()

            fd_probe[
                atom,
                axis,
            ] = (
                e_plus
                - e_minus
            ) / (
                2.0*h
            )

    assert torch.allclose(
        grad_target,
        fd_target,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    assert torch.allclose(
        grad_probe,
        fd_probe,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_total_autograd_force_obeys_translation_invariance():
    target = make_carbonyl_torch(
        requires_grad=True
    )

    probe = make_probe_torch(
        requires_grad=True
    )

    energy = (
        thcon.fragment_repulsion_energy(
            target,
            probe,
        )
    )

    grad_target, grad_probe = (
        torch.autograd.grad(
            energy,
            (
                target.positions,
                probe.positions,
            ),
        )
    )

    total_gradient = (
        torch.sum(
            grad_target,
            dim=0,
        )
        + torch.sum(
            grad_probe,
            dim=0,
        )
    )

    assert torch.max(
        torch.abs(
            total_gradient
        )
    ).item() < 1.0e-9


# ============================================================
# BOND-WEIGHT AUTOGRAD
# ============================================================


def test_autograd_bond_weight_gradient_matches_finite_difference():
    base_fragment = (
        make_carbonyl_torch()
    )

    positions = (
        base_fragment.positions
        .detach()
    )

    base = torch.zeros(
        (4, 4),
        dtype=torch.float64,
    )

    base[0, 1] = 1.0
    base[1, 0] = 1.0

    base[0, 2] = 1.0
    base[2, 0] = 1.0

    mask = torch.zeros(
        (4, 4),
        dtype=torch.float64,
    )

    mask[0, 3] = 1.0
    mask[3, 0] = 1.0

    weight = torch.tensor(
        0.37,
        dtype=torch.float64,
        requires_grad=True,
    )

    matrix = (
        base
        + weight*mask
    )

    target = (
        thcon.ContinuousTorchFragment(
            symbols=[
                "C",
                "O",
                "H",
                "H",
            ],
            positions=positions,
            bond_weights=matrix,
        )
    )

    energy = (
        thcon.fragment_repulsion_energy(
            target,
            make_probe_torch(),
        )
    )

    gradient = (
        torch.autograd.grad(
            energy,
            weight,
        )[0].item()
    )

    h = 1.0e-6

    def evaluate(value):
        matrix_local = (
            base
            + value*mask
        )

        fragment = (
            thcon.ContinuousTorchFragment(
                symbols=[
                    "C",
                    "O",
                    "H",
                    "H",
                ],
                positions=positions,
                bond_weights=matrix_local,
            )
        )

        return (
            thcon.fragment_repulsion_energy(
                fragment,
                make_probe_torch(),
            ).item()
        )

    finite_difference = (
        evaluate(
            0.37 + h
        )
        - evaluate(
            0.37 - h
        )
    ) / (
        2.0*h
    )

    assert gradient == pytest.approx(
        finite_difference,
        rel=2.0e-5,
        abs=2.0e-5,
    )


# ============================================================
# WEIGHT-DERIVATIVE CONTINUITY
# ============================================================


def test_bond_weight_autograd_stays_smooth_through_zero_to_one():
    angle = math.radians(
        104.47
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.960, 0.0, 0.0],
            [
                0.960*math.cos(
                    angle
                ),
                0.960*math.sin(
                    angle
                ),
                0.0,
            ],
        ],
        dtype=torch.float64,
    )

    base = torch.zeros(
        (3, 3),
        dtype=torch.float64,
    )

    base[0, 1] = 1.0
    base[1, 0] = 1.0

    mask = torch.zeros(
        (3, 3),
        dtype=torch.float64,
    )

    mask[0, 2] = 1.0
    mask[2, 0] = 1.0

    derivatives = []

    for raw_weight in torch.linspace(
        0.0,
        1.0,
        201,
        dtype=torch.float64,
    ):
        weight = (
            raw_weight
            .detach()
            .requires_grad_(True)
        )

        fragment = (
            thcon.ContinuousTorchFragment(
                symbols=[
                    "O",
                    "H",
                    "H",
                ],
                positions=positions,
                bond_weights=(
                    base
                    + weight*mask
                ),
            )
        )

        energy = (
            thcon.fragment_repulsion_energy(
                fragment,
                make_probe_torch(),
            )
        )

        derivative = (
            torch.autograd.grad(
                energy,
                weight,
            )[0]
        )

        derivatives.append(
            derivative.item()
        )

    derivatives = np.asarray(
        derivatives
    )

    assert np.all(
        np.isfinite(
            derivatives
        )
    )

    assert np.max(
        np.abs(
            np.diff(
                derivatives
            )
        )
    ) < 0.1


# ============================================================
# C=O COMPRESSION FORCE CONTINUITY
# ============================================================


def test_carbonyl_compression_autograd_has_no_force_cusp():
    re = npcon.SINGLE_BOND_RE[
        ("C", "O")
    ]

    derivatives = []

    ch = 1.101
    angle = math.radians(
        120.0
    )

    weights = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    for raw_distance in torch.linspace(
        re*0.80,
        re*1.05,
        201,
        dtype=torch.float64,
    ):
        distance = (
            raw_distance
            .detach()
            .requires_grad_(True)
        )

        zero = (
            distance
            * 0.0
        )

        positions = torch.stack([
            torch.stack([
                zero,
                zero,
                zero,
            ]),
            torch.stack([
                distance,
                zero,
                zero,
            ]),
            torch.stack([
                zero
                + ch*math.cos(
                    angle
                ),
                zero
                + ch*math.sin(
                    angle
                ),
                zero,
            ]),
            torch.stack([
                zero
                + ch*math.cos(
                    -angle
                ),
                zero
                + ch*math.sin(
                    -angle
                ),
                zero,
            ]),
        ])

        fragment = (
            thcon.ContinuousTorchFragment(
                symbols=[
                    "C",
                    "O",
                    "H",
                    "H",
                ],
                positions=positions,
                bond_weights=weights,
            )
        )

        energy = (
            thcon.fragment_repulsion_energy(
                fragment,
                make_probe_torch(),
            )
        )

        derivative = (
            torch.autograd.grad(
                energy,
                distance,
            )[0]
        )

        derivatives.append(
            derivative.item()
        )

    derivatives = np.asarray(
        derivatives
    )

    assert np.all(
        np.isfinite(
            derivatives
        )
    )

    assert np.max(
        np.abs(
            np.diff(
                derivatives
            )
        )
    ) < 1.0


# ============================================================
# NEAR-LINEAR PLANE COLLAPSE AUTOGRAD
# ============================================================


def test_near_linear_plane_collapse_has_no_autograd_spike():
    derivatives = []

    co = 1.208
    ch = 1.086

    weights = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    for raw_angle in torch.linspace(
        0.0,
        0.01,
        201,
        dtype=torch.float64,
    ):
        angle = (
            raw_angle
            .detach()
            .requires_grad_(True)
        )

        zero = (
            angle
            * 0.0
        )

        positions = torch.stack([
            torch.stack([
                zero,
                zero,
                zero,
            ]),
            torch.stack([
                zero + co,
                zero,
                zero,
            ]),
            torch.stack([
                ch*torch.cos(
                    angle
                ),
                ch*torch.sin(
                    angle
                ),
                zero,
            ]),
        ])

        fragment = (
            thcon.ContinuousTorchFragment(
                symbols=[
                    "C",
                    "O",
                    "H",
                ],
                positions=positions,
                bond_weights=weights,
            )
        )

        energy = (
            thcon.fragment_repulsion_energy(
                fragment,
                make_probe_torch(),
            )
        )

        derivative = (
            torch.autograd.grad(
                energy,
                angle,
            )[0]
        )

        derivatives.append(
            abs(
                derivative.item()
            )
        )

    derivatives = np.asarray(
        derivatives
    )

    assert np.all(
        np.isfinite(
            derivatives
        )
    )

    median = (
        np.median(
            derivatives
        )
        + 1.0e-12
    )

    assert (
        np.max(
            derivatives
        )
        / median
    ) < 100.0


# ============================================================
# OPTIONAL CUDA CONTROL
# ============================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_cuda_energy_matches_cpu():
    cpu_target = (
        make_carbonyl_torch(
            device="cpu"
        )
    )

    cpu_probe = (
        make_probe_torch(
            device="cpu"
        )
    )

    cpu_energy = (
        thcon.fragment_repulsion_energy(
            cpu_target,
            cpu_probe,
        ).item()
    )

    gpu_target = (
        make_carbonyl_torch(
            device="cuda"
        )
    )

    gpu_probe = (
        make_probe_torch(
            device="cuda"
        )
    )

    gpu_energy = (
        thcon.fragment_repulsion_energy(
            gpu_target,
            gpu_probe,
        ).item()
    )

    assert gpu_energy == pytest.approx(
        cpu_energy,
        rel=1.0e-9,
        abs=1.0e-10,
    )
