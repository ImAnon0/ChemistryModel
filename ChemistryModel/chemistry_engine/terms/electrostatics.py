from __future__ import annotations

import torch

from ..context import InteractionContext


ELECTROSTATIC_PARAMETERS = {
    "H": {"atomic_number": 1, "chi": 4.528, "hardness": 13.89},
    "C": {"atomic_number": 6, "chi": 5.343, "hardness": 10.126},
    "N": {"atomic_number": 7, "chi": 6.899, "hardness": 11.786},
    "O": {"atomic_number": 8, "chi": 8.741, "hardness": 13.364},
}


def calculate_dipole(positions, charges):
    return torch.sum(charges[:, None] * positions, dim=0)


class ElectrostaticEnergyTerm:
    name = "electrostatics"

    def __init__(self, total_charge=0.0, enabled=False):
        self.total_charge = total_charge
        self.enabled = enabled
        self.parameters = ELECTROSTATIC_PARAMETERS
        self.last_state = {}

    def _parameter_arrays(self, atomic_numbers, device, dtype):
        lookup = {
            value["atomic_number"]: value
            for value in ELECTROSTATIC_PARAMETERS.values()
        }

        chi = torch.tensor(
            [lookup[int(z)]["chi"] for z in atomic_numbers],
            device=device,
            dtype=dtype,
        )

        hardness = torch.tensor(
            [lookup[int(z)]["hardness"] for z in atomic_numbers],
            device=device,
            dtype=dtype,
        )

        return chi, hardness

    def _build_qeq_matrix(self, context):
        positions = context.positions
        device = positions.device
        dtype = positions.dtype

        chi, hardness = self._parameter_arrays(
            context.atomic_numbers,
            device,
            dtype,
        )

        distances = torch.cdist(positions, positions)
        count = len(context.atomic_numbers)

        coupling = torch.zeros_like(distances)
        mask = ~torch.eye(count, dtype=torch.bool, device=device)

        coupling[mask] = 1.0 / torch.clamp(
            distances[mask],
            min=1e-8,
        )

        return chi, torch.diag(hardness) + coupling

    def solve_charges(self, context):
        chi, matrix = self._build_qeq_matrix(context)

        count = len(context.atomic_numbers)
        device = context.positions.device
        dtype = context.positions.dtype

        constraint = torch.ones((count, 1), device=device, dtype=dtype)

        augmented = torch.cat(
            [
                torch.cat([matrix, constraint], dim=1),
                torch.cat(
                    [
                        constraint.T,
                        torch.zeros((1, 1), device=device, dtype=dtype),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

        rhs = torch.cat(
            [
                -chi,
                torch.tensor(
                    [self.total_charge],
                    device=device,
                    dtype=dtype,
                ),
            ]
        )

        solution = torch.linalg.solve(augmented, rhs)
        charges = solution[:-1]

        self.last_state = {
            "charges": charges,
            "charge_sum": charges.sum(),
            "dipole": calculate_dipole(context.positions, charges),
            "solver": "qeq",
            "qeq_matrix": matrix,
            "chi": chi,
        }

        return charges

    def diagnostics(self):
        return self.last_state

    def energy(self, context, current_energy):
        charges = self.solve_charges(context)

        if not self.enabled:
            return torch.zeros_like(current_energy)

        matrix = self.last_state["qeq_matrix"]
        chi = self.last_state["chi"]

        contribution = 0.5 * torch.dot(charges, matrix @ charges)
        contribution = contribution + torch.dot(chi, charges)

        return contribution
