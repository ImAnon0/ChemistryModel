import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


def context_from_case(name):
    positions, atoms = known_diagnostic_cases()[name]

    class Context:
        pass

    context = Context()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
        requires_grad=True,
    )
    context.atomic_numbers = atoms
    return context


def test_qeq_energy_off_has_no_force():
    context = context_from_case("H2O")

    energy = ElectrostaticEnergyTerm(enabled=False).energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    assert energy.item() == 0.0


def test_qeq_energy_on_produces_finite_forces():
    context = context_from_case("H2O")
    term = ElectrostaticEnergyTerm(enabled=True)

    energy = term.energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    forces = -torch.autograd.grad(
        energy,
        context.positions,
    )[0]

    assert torch.isfinite(forces).all()
