"""Standalone research facade for the continuous SQE C0 solve."""

from __future__ import annotations

from dataclasses import dataclass
import json

import torch

from .diagnostics import C0Diagnostics
from .parameters import C0_PARAMETERS, C0ParameterSet
from .solver import solve_c0


@dataclass(frozen=True)
class C0Result:
    energy: torch.Tensor
    charges: torch.Tensor
    forces: torch.Tensor | None
    dipole_e_A: torch.Tensor
    dipole_debye: torch.Tensor
    polarizability_A3: torch.Tensor
    transfer_amplitudes: torch.Tensor
    edge_index: torch.Tensor
    compliance: torch.Tensor
    response_matrix: torch.Tensor
    diagnostics: C0Diagnostics


class C0ContinuousSQE:
    model_id = "research_continuous_sqe_c0_v1"
    research_only = True

    def __init__(self, parameters: C0ParameterSet = C0_PARAMETERS):
        if parameters.independent_parameter_count != 17:
            raise ValueError("C0 must contain exactly 17 independent parameters")
        self.parameters = parameters

    def evaluate(
        self,
        positions,
        atomic_numbers,
        *,
        total_charge=0.0,
        calculate_forces=False,
    ):
        if float(total_charge) != 0.0:
            raise ValueError(
                "C0 v1 is neutral-only; a nonzero charge needs declared "
                "reference charges, not a global charge constraint"
            )
        if not torch.is_floating_point(positions):
            raise TypeError("positions must use a floating-point dtype")
        working_positions = positions
        if calculate_forces and not working_positions.requires_grad:
            working_positions = positions.detach().clone().requires_grad_(True)
        solved = solve_c0(working_positions, tuple(atomic_numbers), self.parameters)
        forces = None
        if calculate_forces:
            forces = -torch.autograd.grad(
                solved.energy,
                working_positions,
                create_graph=False,
                retain_graph=True,
            )[0]
        dipole = torch.sum(solved.charges[:, None] * working_positions, dim=0)
        return C0Result(
            energy=solved.energy,
            charges=solved.charges,
            forces=forces,
            dipole_e_A=dipole,
            dipole_debye=dipole * 4.80320471257,
            polarizability_A3=solved.polarizability_A3,
            transfer_amplitudes=solved.transfer_amplitudes,
            edge_index=solved.edge_index,
            compliance=solved.compliance,
            response_matrix=solved.response_matrix,
            diagnostics=solved.diagnostics,
        )


def main():
    cases = {
        "H2": ([[0.0, 0.0, 0.0], [0.74144, 0.0, 0.0]], (1, 1)),
        "H2O": (
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
            (8, 1, 1),
        ),
        "O_H_10A": ([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], (8, 1)),
    }
    model = C0ContinuousSQE()
    output = {}
    for name, (coordinates, atomic_numbers) in cases.items():
        result = model.evaluate(
            torch.tensor(coordinates, dtype=torch.float64),
            atomic_numbers,
            calculate_forces=True,
        )
        output[name] = {
            "energy_eV": float(result.energy.detach()),
            "charges_e": result.charges.detach().tolist(),
            "dipole_e_A": result.dipole_e_A.detach().tolist(),
            "maximum_force_eV_per_A": float(torch.max(torch.abs(result.forces))),
            "diagnostics": result.diagnostics.as_dict(),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
