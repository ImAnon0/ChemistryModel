"""Shared research-only energy replacement bookkeeping."""

from __future__ import annotations

import torch

import reactive as R


def heavy_heavy_mask(simulation, values):
    hydrogen = int(R.ELEMENT_INDEX["H"])
    centre_heavy = simulation.types != hydrogen
    neighbour_heavy = simulation.types[values["neighbours"]] != hydrogen
    return centre_heavy[:, None] & neighbour_heavy & (values["mask"] > 0.0)


def attractive_magnitude(values):
    attractive = values.get("state_attractive")
    if attractive is not None:
        return attractive
    return 2.0 * values["pair_depth"] * torch.exp(
        -values["pair_width"] * values["shift"]
    )


def free_energy(energies, temperature):
    """Stable finite-temperature state free energy."""
    tau = torch.as_tensor(
        temperature, device=energies.device, dtype=energies.dtype
    )
    return -tau * torch.logsumexp(-energies / tau, dim=0)


class HeavyAttractionEnergyReplacementMixin:
    """Compose a desired heavy-attraction scalar with inherited angles."""

    formulation_name = "unspecified"

    def _valence_topology_correction(self, positions):
        angle_correction = super()._valence_topology_correction(positions)
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("missing live reactive intermediates")
        values = cached[1]
        membership = getattr(self, "_research_energy_membership", None)
        desired = getattr(self, "_research_desired_heavy_attraction", None)
        if membership is None or desired is None:
            raise RuntimeError("formulation did not provide energy state data")

        heavy_heavy = heavy_heavy_mask(self, values)
        contact_attraction = values["taper"] * attractive_magnitude(values)
        base_attraction = -0.5 * torch.sum(
            torch.where(
                heavy_heavy, contact_attraction,
                torch.zeros_like(contact_attraction),
            ),
            dim=1,
        )
        attraction_correction = desired - base_attraction

        hydrogen = int(R.ELEMENT_INDEX["H"])
        centre_heavy = self.types != hydrogen
        effective_contact = torch.where(
            heavy_heavy,
            values["taper"] * membership,
            values["taper"],
        )
        effective_coordination = torch.sum(effective_contact, dim=1)
        excess = torch.clamp(
            effective_coordination - values["valence"], min=0.0
        )
        scale = self.over_coordination_scale(
            values["taper"], values["unsoftened_depth"], values["mask"],
            cache_key=positions,
        )
        replacement_over = self.over_penalty * scale * excess.square()
        original_over = self._profile_energy_parts["over"]
        over_correction = centre_heavy.to(self.dtype) * (
            replacement_over - original_over
        )

        total = attraction_correction + over_correction
        self._heavy_valence_energy_diagnostics = {
            "formulation": self.formulation_name,
            "membership": membership.detach(),
            "heavy_heavy_mask": heavy_heavy.detach(),
            "base_heavy_attraction_per_atom": base_attraction.detach(),
            "desired_heavy_attraction_per_atom": desired.detach(),
            "attraction_correction_per_atom": attraction_correction.detach(),
            "original_over_per_atom": original_over.detach(),
            "replacement_over_per_atom": replacement_over.detach(),
            "effective_coordination": effective_coordination.detach(),
            "energy_correction_per_atom": total.detach(),
            **getattr(self, "_research_formulation_diagnostics", {}),
        }
        return angle_correction + total
