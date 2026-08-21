"""Parameter-free mean-state heavy-valence energy prototype.

This module is deliberately outside production selection. It reuses the
validated Optimised-Valence membership solver and changes only the bookkeeping
of heavy-heavy attraction and heavy-centred overcoordination.
"""

from __future__ import annotations

import torch

import reactive as R
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


HEAVY_VALENCE_ENERGY_MODEL_NAME = (
    "research_mean_state_heavy_valence_energy_v0"
)
HEAVY_VALENCE_ENERGY_MODEL_REVISION = 0


class HeavyValenceStateEnergyPrototype(
    OptimisedValenceStateBatchedSimulation
):
    """Research-only energy-bearing heavy-valence competition model."""

    physics_model_name = HEAVY_VALENCE_ENERGY_MODEL_NAME
    physics_model_revision = HEAVY_VALENCE_ENERGY_MODEL_REVISION
    research_only = True

    def _local_valence_membership(self, values):
        membership = super()._local_valence_membership(values)

        # Retain the live tensor for the energy correction evaluated
        # immediately after the inherited topology correction. Do not detach:
        # membership derivatives are part of the prototype force.
        self._heavy_energy_membership = membership
        return membership

    def _valence_topology_correction(self, positions):
        # This computes the validated heavy-angle replacement and, through the
        # override above, the exact smooth heavy-valence membership.
        angle_correction = super()._valence_topology_correction(positions)

        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError(
                "heavy-valence energy prototype requires live reactive "
                "intermediates"
            )

        values = cached[1]
        membership = getattr(self, "_heavy_energy_membership", None)
        if membership is None:
            raise RuntimeError("heavy-valence membership was not retained")

        hydrogen = int(R.ELEMENT_INDEX["H"])
        centre_is_heavy = self.types != hydrogen
        neighbour_types = self.types[values["neighbours"]]
        neighbour_is_heavy = neighbour_types != hydrogen
        heavy_heavy = (
            centre_is_heavy[:, None]
            & neighbour_is_heavy
            & (values["mask"] > 0.0)
        )

        attractive = values.get("state_attractive")
        if attractive is None:
            attractive = 2.0 * values["pair_depth"] * torch.exp(
                -values["pair_width"] * values["shift"]
            )

        # Base energy counts half of every directed pair. Remove the rejected
        # share of heavy-heavy attraction while preserving repulsion exactly.
        rejected_attraction = torch.where(
            heavy_heavy,
            0.5
            * values["taper"]
            * attractive
            * (1.0 - membership),
            torch.zeros_like(values["taper"]),
        )
        attraction_correction = torch.sum(rejected_attraction, dim=1)

        # H contacts retain raw coordination because their energetic ownership
        # remains with the H-state model. Heavy-heavy contacts use the same
        # membership that capacity-limits their attraction above.
        effective_contact = torch.where(
            heavy_heavy,
            values["taper"] * membership,
            values["taper"],
        )
        effective_coordination = torch.sum(effective_contact, dim=1)
        effective_excess = torch.clamp(
            effective_coordination - values["valence"], min=0.0
        )

        over_scale = self.over_coordination_scale(
            values["taper"],
            values["unsoftened_depth"],
            values["mask"],
            cache_key=positions,
        )
        prototype_over = (
            self.over_penalty * over_scale * effective_excess.square()
        )
        original_over = self._profile_energy_parts["over"]
        over_correction = centre_is_heavy.to(self.dtype) * (
            prototype_over - original_over
        )

        total_correction = attraction_correction + over_correction
        self._heavy_valence_energy_diagnostics = {
            "membership": membership.detach(),
            "heavy_heavy_mask": heavy_heavy.detach(),
            "rejected_attraction_per_atom": attraction_correction.detach(),
            "original_over_per_atom": original_over.detach(),
            "prototype_over_per_atom": prototype_over.detach(),
            "over_correction_per_atom": over_correction.detach(),
            "effective_coordination": effective_coordination.detach(),
            "energy_correction_per_atom": total_correction.detach(),
        }
        return angle_correction + total_correction
