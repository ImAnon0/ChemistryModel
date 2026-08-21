"""Local finite-temperature free-energy heavy-valence reference."""

from __future__ import annotations

import itertools

import torch

import reactive as R
from research.heavy_valence_state.energy_common import (
    HeavyAttractionEnergyReplacementMixin,
    attractive_magnitude,
    free_energy,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


class LocalFreeEnergyHeavyValencePrototype(
    HeavyAttractionEnergyReplacementMixin,
    OptimisedValenceStateBatchedSimulation,
):
    """Independent heavy centres, but with an explicit scalar free energy."""

    physics_model_name = "research_local_free_energy_heavy_valence_v0"
    physics_model_revision = 0
    research_only = True
    formulation_name = "local_free_energy"

    def _local_valence_membership(self, values):
        taper = values["taper"]
        mask = values["mask"]
        contact_attraction = taper * attractive_magnitude(values)
        hydrogen = int(R.ELEMENT_INDEX["H"])
        rows = []
        desired = []
        state_counts = []

        for atom in range(taper.shape[0]):
            row_zero = taper[atom] * 0.0
            if int(self.types_numpy[atom]) == hydrogen:
                rows.append(row_zero + mask[atom])
                desired.append(row_zero.sum())
                state_counts.append(1)
                continue

            active_tensor = torch.nonzero(
                (mask[atom] > 0.0) & (taper[atom] > 0.0), as_tuple=False
            ).flatten()
            active = tuple(int(value) for value in active_tensor.detach().cpu())
            capacity = max(int(round(float(
                self.valence[self.types[atom]].detach().cpu()
            ))), 0)
            if not active or capacity <= 0:
                rows.append(row_zero)
                desired.append(row_zero.sum())
                state_counts.append(1)
                continue
            if len(active) <= capacity:
                rows.append(row_zero + mask[atom])
                hh_terms = [
                    contact_attraction[atom, slot]
                    for slot in active
                    if int(self.types_numpy[int(values["neighbours"][atom, slot])])
                    != hydrogen
                ]
                desired.append(
                    -0.5 * torch.stack(hh_terms).sum()
                    if hh_terms else row_zero.sum()
                )
                state_counts.append(1)
                continue

            states = tuple(itertools.combinations(range(len(active)), capacity))
            state_mask = torch.zeros(
                (len(states), len(active)), device=self.device, dtype=self.dtype
            )
            for index, state in enumerate(states):
                if state:
                    state_mask[index, list(state)] = 1.0
            active_index = torch.tensor(active, device=self.device, dtype=torch.long)
            all_attraction = contact_attraction[atom, active_index]
            hh_indicator = torch.tensor([
                int(self.types_numpy[int(values["neighbours"][atom, slot])])
                != hydrogen for slot in active
            ], device=self.device, dtype=self.dtype)
            full_energies = -(state_mask @ all_attraction)
            reference_energies = -(
                state_mask @ (all_attraction * (1.0 - hh_indicator))
            )
            probabilities = torch.softmax(
                -full_energies / self.heavy_valence_temperature, dim=0
            )
            active_membership = probabilities @ state_mask
            row = row_zero.scatter(0, active_index, active_membership)
            rows.append(row)
            desired.append(0.5 * (
                free_energy(full_energies, self.heavy_valence_temperature)
                - free_energy(reference_energies, self.heavy_valence_temperature)
            ))
            state_counts.append(len(states))

        membership = torch.stack(rows)
        self._research_energy_membership = membership
        self._research_desired_heavy_attraction = torch.stack(desired)
        self._research_formulation_diagnostics = {
            "local_state_counts": tuple(state_counts),
            "maximum_local_state_count": max(state_counts, default=1),
        }
        return membership
