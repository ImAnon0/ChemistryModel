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

    def _build_qeq_matrix(self, context, indices=None):
        positions = context.positions
        if indices is not None:
            positions = positions.index_select(0, indices)
        device = positions.device
        dtype = positions.dtype

        atomic_numbers = context.atomic_numbers
        if indices is not None:
            atomic_numbers = tuple(
                atomic_numbers[index]
                for index in indices.detach().cpu().tolist()
            )

        chi, hardness = self._parameter_arrays(
            atomic_numbers,
            device,
            dtype,
        )

        displacement = positions[:, None, :] - positions[None, :, :]
        box_size = float(getattr(context, "box_size", 0.0) or 0.0)
        if box_size > 0.0:
            displacement = displacement - box_size * torch.round(
                displacement / box_size
            )
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        count = len(atomic_numbers)

        coupling = torch.zeros_like(distances)
        mask = ~torch.eye(count, dtype=torch.bool, device=device)

        coupling[mask] = 1.0 / torch.clamp(
            distances[mask],
            min=1e-8,
        )

        return chi, torch.diag(hardness) + coupling

    def _box_indices(self, context):
        count = len(context.atomic_numbers)
        assignments = getattr(context, "batch_assignment", None)
        if assignments is None or len(assignments) != count:
            assignments = (0,) * count

        ordered_boxes = tuple(dict.fromkeys(int(value) for value in assignments))
        device = context.positions.device
        return tuple(
            torch.tensor(
                [index for index, value in enumerate(assignments) if int(value) == box],
                device=device,
                dtype=torch.long,
            )
            for box in ordered_boxes
        )

    def _solve_box(self, context, indices):
        chi, matrix = self._build_qeq_matrix(context, indices)

        count = len(indices)
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
        return solution[:-1], chi, matrix

    def solve_charges(self, context):
        box_indices = self._box_indices(context)
        box_solutions = [
            self._solve_box(context, indices)
            for indices in box_indices
        ]

        charges = torch.zeros(
            len(context.atomic_numbers),
            device=context.positions.device,
            dtype=context.positions.dtype,
        )
        for indices, (box_charges, _, _) in zip(box_indices, box_solutions):
            charges = charges.index_copy(0, indices, box_charges)
        chi = torch.zeros_like(charges)
        for indices, (_, box_chi, _) in zip(box_indices, box_solutions):
            chi = chi.index_copy(0, indices, box_chi)

        box_charge_sums = torch.stack([
            box_charges.sum()
            for box_charges, _, _ in box_solutions
        ])
        box_dipoles = torch.stack([
            calculate_dipole(
                context.positions.index_select(0, indices),
                box_charges,
            )
            for indices, (box_charges, _, _) in zip(box_indices, box_solutions)
        ])

        self.last_state = {
            "charges": charges,
            "charge_sum": charges.sum(),
            "dipole": calculate_dipole(context.positions, charges),
            "box_charge_sums": box_charge_sums,
            "box_dipoles": box_dipoles,
            "solver": "qeq",
            "qeq_matrix": (
                box_solutions[0][2]
                if len(box_solutions) == 1
                else tuple(solution[2] for solution in box_solutions)
            ),
            "chi": chi,
            "box_indices": box_indices,
            "box_solutions": tuple(box_solutions),
        }

        return charges

    def diagnostics(self):
        return self.last_state

    def _release_solve_graph(self):
        def detached(value):
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, tuple):
                return tuple(detached(item) for item in value)
            return value

        self.last_state = {
            name: detached(value)
            for name, value in self.last_state.items()
            if name != "box_solutions"
        }

    def provenance(self):
        return {
            "parameters": self.parameters,
            "total_charge_per_box": self.total_charge,
            "coupling": "minimum_image_unshielded_inverse_distance_v1",
            "distance_unit": "angstrom",
            "energy_unit": "nominal_eV_unverified",
        }

    def energy(self, context, current_energy):
        charges = self.solve_charges(context)

        if not self.enabled:
            contribution = torch.zeros_like(current_energy)
            self._release_solve_graph()
            return contribution

        contribution = charges.sum() * 0.0
        for box_charges, chi, matrix in self.last_state["box_solutions"]:
            contribution = contribution + 0.5 * torch.dot(
                box_charges,
                matrix @ box_charges,
            )
            contribution = contribution + torch.dot(chi, box_charges)

        self._release_solve_graph()
        return contribution
