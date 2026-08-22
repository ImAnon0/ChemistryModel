"""Rejected charge-localising reference candidates (research only).

Nothing in this module is registered with ``chemistry_engine``.  The corrected
Gaussian atom-space QTPIE candidate is retained to make its successful
dissociation limit and failed conditioning/reactive-coordinate gates exactly
reproducible.
"""

from __future__ import annotations

import math

import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm


BOHR_PER_ANGSTROM = 1.8897261254578281
HARTREE_TO_EV = 27.211386245988

GAUSSIAN_QTPIE_PARAMETERS = {
    "H": {"atomic_number": 1, "chi": 4.528, "hardness": 13.890,
          "gaussian_alpha_bohr2": 0.5434},
    "C": {"atomic_number": 6, "chi": 5.343, "hardness": 10.126,
          "gaussian_alpha_bohr2": 0.2069},
    "N": {"atomic_number": 7, "chi": 6.899, "hardness": 11.760,
          "gaussian_alpha_bohr2": 0.2214},
    "O": {"atomic_number": 8, "chi": 8.741, "hardness": 13.364,
          "gaussian_alpha_bohr2": 0.2240},
}


class GaussianQTPIECandidate(ElectrostaticEnergyTerm):
    """Neutral, corrected-Gaussian atom-space QTPIE diagnostic.

    This faithfully preserves the specific convention tested by the audit; it
    is not a production candidate because its constrained hardness matrix is
    indefinite or nearly singular for important reactive geometries.
    """

    def __init__(self, total_charge=0.0, enabled=False):
        if float(total_charge) != 0.0:
            raise ValueError("this QTPIE diagnostic is neutral-only")
        super().__init__(total_charge=total_charge, enabled=enabled)
        self.parameters = GAUSSIAN_QTPIE_PARAMETERS

    def _parameter_arrays(self, atomic_numbers, device, dtype):
        lookup = {value["atomic_number"]: value for value in self.parameters.values()}
        chi = torch.tensor([lookup[int(z)]["chi"] for z in atomic_numbers],
                           device=device, dtype=dtype)
        hardness = torch.tensor(
            [lookup[int(z)]["hardness"] for z in atomic_numbers],
            device=device, dtype=dtype,
        )
        alpha = torch.tensor(
            [lookup[int(z)]["gaussian_alpha_bohr2"] for z in atomic_numbers],
            device=device, dtype=dtype,
        )
        return chi, hardness, alpha

    def _build_qeq_matrix(self, context, indices=None):
        positions = context.positions if indices is None else context.positions.index_select(0, indices)
        atomic_numbers = context.atomic_numbers
        if indices is not None:
            atomic_numbers = tuple(
                atomic_numbers[index] for index in indices.detach().cpu().tolist()
            )
        bare_chi, hardness, alpha = self._parameter_arrays(
            atomic_numbers, positions.device, positions.dtype
        )
        displacement = positions[:, None, :] - positions[None, :, :]
        box_size = float(getattr(context, "box_size", 0.0) or 0.0)
        if box_size > 0.0:
            displacement = displacement - box_size * torch.round(displacement / box_size)

        distance_squared = torch.sum((displacement * BOHR_PER_ANGSTROM).square(), dim=-1)
        distances = torch.sqrt(torch.clamp(distance_squared, min=1e-30))
        alpha_i, alpha_j = alpha[:, None], alpha[None, :]
        beta = alpha_i * alpha_j / (alpha_i + alpha_j)
        overlap = (
            2.0 * torch.sqrt(alpha_i * alpha_j) / (alpha_i + alpha_j)
        ).pow(1.5) * torch.exp(-beta * distance_squared)
        overlap = overlap.clone()
        overlap.fill_diagonal_(1.0)
        effective_chi = torch.sum(
            overlap * (bare_chi[:, None] - bare_chi[None, :]), dim=1
        ) / torch.sum(overlap, dim=1)

        coulomb = HARTREE_TO_EV * torch.erf(torch.sqrt(beta) * distances) / distances
        coincident = HARTREE_TO_EV * 2.0 * torch.sqrt(beta / math.pi)
        coulomb = torch.where(distance_squared > 1e-28, coulomb, coincident)
        count = len(atomic_numbers)
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=positions.device)
        return effective_chi, torch.diag(hardness) + torch.where(
            off_diagonal, coulomb, torch.zeros_like(coulomb)
        )

    def solve_charges(self, context):
        charges = super().solve_charges(context)
        self.last_state["solver"] = "gaussian_qtpie_atom_space_rejected_candidate"
        self.last_state["hardness_matrix"] = self.last_state["qeq_matrix"]
        self.last_state["effective_chi"] = self.last_state["chi"]
        return charges

    def provenance(self):
        return {
            "parameters": self.parameters,
            "formulation": "chen_atom_space_gaussian_qtpie_corrected_v1",
            "status": "rejected_research_candidate",
            "reason": "indefinite_or_near_singular_reactive_response",
        }
