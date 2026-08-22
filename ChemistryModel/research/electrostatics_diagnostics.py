from __future__ import annotations

import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm


def inspect_qeq_state(positions, atomic_numbers):
    class DiagnosticContext:
        pass

    context = DiagnosticContext()
    context.positions = torch.tensor(
        positions,
        dtype=torch.float64,
    )
    context.atomic_numbers = atomic_numbers

    term = ElectrostaticEnergyTerm()
    charges = term.solve_charges(context)

    return {
        "charges": charges.detach().cpu(),
        "charge_sum": charges.sum().item(),
        "dipole": term.diagnostics()["dipole"].detach().cpu(),
    }


def known_diagnostic_cases():
    return {
        "H2": (
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            (1, 1),
        ),
        "CH4": (
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            (6, 1, 1, 1, 1),
        ),
        "H2O": (
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
            (8, 1, 1),
        ),
        "CH2O": (
            [
                [0.0, 0.0, 0.0],
                [1.21, 0.0, 0.0],
                [-0.7, 0.95, 0.0],
                [-0.7, -0.95, 0.0],
            ],
            (6, 8, 1, 1),
        ),
    }
