import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


def test_qeq_relaxation_diagnostic_inputs_exist():
    cases = known_diagnostic_cases()

    assert "H2O" in cases
    assert "CH2O" in cases


def test_qeq_toggle_changes_only_extension_energy():
    positions, atoms = known_diagnostic_cases()["H2O"]

    context = type("Context", (), {})()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
    )
    context.atomic_numbers = atoms

    off = ElectrostaticEnergyTerm(enabled=False).energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    on = ElectrostaticEnergyTerm(enabled=True).energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    assert off.item() == 0.0
    assert torch.isfinite(on)
