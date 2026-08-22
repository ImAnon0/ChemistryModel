from __future__ import annotations

import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


class Context:
    pass


def build_context(positions, atoms):
    context = Context()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
        requires_grad=True,
    )
    context.atomic_numbers = atoms
    return context


def evaluate_case(name, enabled):
    positions, atoms = known_diagnostic_cases()[name]
    context = build_context(positions, atoms)

    term = ElectrostaticEnergyTerm(enabled=enabled)

    energy = term.energy(
        context,
        torch.tensor(0.0, dtype=torch.float64),
    )

    if energy.requires_grad:
        forces = -torch.autograd.grad(
            energy,
            context.positions,
            create_graph=False,
        )[0]
    else:
        forces = torch.zeros_like(context.positions)

    return {
        "energy": energy.detach(),
        "forces": forces.detach(),
        "charges": term.diagnostics()["charges"].detach(),
    }


def run():
    for name in ("H2O", "CH2O"):
        off = evaluate_case(name, False)
        on = evaluate_case(name, True)

        print()
        print("=" * 40)
        print(name)
        print("=" * 40)
        print("off energy:", off["energy"].item())
        print("off force norm:", torch.linalg.norm(off["forces"]).item())
        print("on energy :", on["energy"].item())
        print("on force norm:", torch.linalg.norm(on["forces"]).item())
        print("charges:", on["charges"].tolist())


if __name__ == "__main__":
    run()
