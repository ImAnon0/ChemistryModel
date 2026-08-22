import torch

from chemistry_engine.terms.electrostatics import calculate_dipole


def test_neutral_zero_charge_has_zero_dipole():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    charges = torch.tensor([0.0, 0.0], dtype=torch.float64)

    dipole = calculate_dipole(positions, charges)

    assert torch.allclose(
        dipole,
        torch.zeros(3, dtype=torch.float64),
    )
