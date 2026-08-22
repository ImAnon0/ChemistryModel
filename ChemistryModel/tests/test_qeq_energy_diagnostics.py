import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


def make_context(positions, atoms):
    class Context:
        pass

    context = Context()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
    )
    context.atomic_numbers = atoms
    return context


def test_qeq_energy_is_disabled_by_default():
    positions, atoms = known_diagnostic_cases()["H2O"]
    context = make_context(positions, atoms)

    energy = ElectrostaticEnergyTerm().energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    assert energy.item() == 0.0


def test_qeq_energy_can_be_enabled():
    positions, atoms = known_diagnostic_cases()["H2O"]
    context = make_context(positions, atoms)

    energy = ElectrostaticEnergyTerm(enabled=True).energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    assert torch.isfinite(energy)
