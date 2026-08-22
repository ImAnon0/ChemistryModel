from __future__ import annotations

import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


class DiagnosticContext:
    pass


def build_context(positions, atomic_numbers):
    context = DiagnosticContext()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
    )
    context.atomic_numbers = atomic_numbers
    return context


def run():
    for name, (positions, atoms) in known_diagnostic_cases().items():
        context = build_context(positions, atoms)

        off = ElectrostaticEnergyTerm(enabled=False)
        on = ElectrostaticEnergyTerm(enabled=True)

        off_energy = off.energy(context, torch.tensor(0.0, dtype=torch.float64))
        on_energy = on.energy(context, torch.tensor(0.0, dtype=torch.float64))

        print()
        print("=" * 40)
        print(name)
        print("=" * 40)
        print("charges:")
        for atom, charge in zip(
            atoms,
            on.diagnostics()["charges"],
        ):
            print(f"{atom}: {charge.item(): .6f}")

        print(f"off energy: {off_energy.item(): .12f}")
        print(f"on energy : {on_energy.item(): .12f}")
        print(f"charge sum: {on.diagnostics()['charge_sum'].item(): .12f}")
        print(f"dipole: {on.diagnostics()['dipole'].tolist()}")


if __name__ == "__main__":
    run()
