"""Exact small-system shared-edge heavy-valence free-energy reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import reactive as R
from research.heavy_valence_state.energy_common import (
    HeavyAttractionEnergyReplacementMixin,
    attractive_magnitude,
    free_energy,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


MAX_JOINT_STATES = 100_000


@dataclass(frozen=True)
class _Candidate:
    key: tuple
    heavy_endpoints: tuple[int, ...]
    directed_slots: tuple[tuple[int, int], ...]
    is_heavy_heavy: bool


def _maximal_capacity_states(candidates, capacities):
    """Enumerate maximal subsets under simultaneous endpoint capacities."""
    states = []
    degrees = {atom: 0 for atom in capacities}
    selected = [False] * len(candidates)

    def visit(index):
        if len(states) > MAX_JOINT_STATES:
            raise RuntimeError(
                f"joint heavy-edge reference exceeded {MAX_JOINT_STATES} states"
            )
        if index == len(candidates):
            for candidate_index, candidate in enumerate(candidates):
                if selected[candidate_index]:
                    continue
                if all(
                    degrees[atom] < capacities[atom]
                    for atom in candidate.heavy_endpoints
                ):
                    return
            states.append(tuple(selected))
            return

        candidate = candidates[index]
        can_select = all(
            degrees[atom] < capacities[atom]
            for atom in candidate.heavy_endpoints
        )
        if can_select:
            selected[index] = True
            for atom in candidate.heavy_endpoints:
                degrees[atom] += 1
            visit(index + 1)
            for atom in candidate.heavy_endpoints:
                degrees[atom] -= 1
        selected[index] = False
        visit(index + 1)

    visit(0)
    return tuple(states)


class JointEdgeStateHeavyValencePrototype(
    HeavyAttractionEnergyReplacementMixin,
    OptimisedValenceStateBatchedSimulation,
):
    """One shared heavy-heavy occupancy constrained at both endpoints."""

    physics_model_name = "research_joint_edge_state_heavy_valence_v0"
    physics_model_revision = 0
    research_only = True
    formulation_name = "joint_edge_free_energy"

    def _local_valence_membership(self, values):
        taper = values["taper"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        attraction = taper * attractive_magnitude(values)
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy_atoms = [
            atom for atom, element in enumerate(self.types_numpy)
            if int(element) != hydrogen
        ]
        capacities = {
            atom: max(int(round(float(
                self.valence[self.types[atom]].detach().cpu()
            ))), 0)
            for atom in heavy_atoms
        }

        directed = {}
        for atom in heavy_atoms:
            for slot in range(taper.shape[1]):
                if float(mask[atom, slot].detach().cpu()) <= 0.0:
                    continue
                if float(taper[atom, slot].detach().cpu()) <= 0.0:
                    continue
                other = int(neighbours[atom, slot].detach().cpu())
                if other == atom:
                    continue
                directed[(atom, other)] = slot

        candidates = []
        seen_hh = set()
        for (atom, other), slot in sorted(directed.items()):
            other_is_heavy = int(self.types_numpy[other]) != hydrogen
            if other_is_heavy:
                edge = (min(atom, other), max(atom, other))
                if edge in seen_hh:
                    continue
                seen_hh.add(edge)
                reverse = directed.get((other, atom))
                slots = ((atom, slot),)
                if reverse is not None:
                    slots += ((other, reverse),)
                candidates.append(_Candidate(
                    key=("HH",) + edge,
                    heavy_endpoints=edge,
                    directed_slots=slots,
                    is_heavy_heavy=True,
                ))
            else:
                candidates.append(_Candidate(
                    key=("H", atom, other),
                    heavy_endpoints=(atom,),
                    directed_slots=((atom, slot),),
                    is_heavy_heavy=False,
                ))

        # Factor the exact partition function over heavy-centre components.
        adjacency = {atom: set() for atom in heavy_atoms}
        for candidate in candidates:
            if candidate.is_heavy_heavy:
                first, second = candidate.heavy_endpoints
                adjacency[first].add(second)
                adjacency[second].add(first)
        components = []
        remaining = set(heavy_atoms)
        while remaining:
            root = min(remaining)
            stack = [root]
            component = set()
            while stack:
                atom = stack.pop()
                if atom in component:
                    continue
                component.add(atom)
                stack.extend(adjacency[atom] - component)
            remaining -= component
            components.append(tuple(sorted(component)))

        row_overrides = {atom: {} for atom in heavy_atoms}
        desired = [taper[atom].sum() * 0.0 for atom in range(taper.shape[0])]
        component_state_counts = []
        component_candidate_counts = []

        for component in components:
            component_set = set(component)
            local_candidates = [
                candidate for candidate in candidates
                if candidate.heavy_endpoints[0] in component_set
            ]
            if not local_candidates:
                component_state_counts.append(1)
                component_candidate_counts.append(0)
                continue
            states = _maximal_capacity_states(
                local_candidates,
                {atom: capacities[atom] for atom in component},
            )
            if not states:
                raise RuntimeError("joint heavy-edge component has no maximal state")
            state_mask = torch.tensor(
                states, device=self.device, dtype=self.dtype
            )
            energy_terms = []
            hh_indicator = []
            for candidate in local_candidates:
                samples = [attraction[atom, slot] for atom, slot in candidate.directed_slots]
                term = torch.stack(samples).mean()
                energy_terms.append(term)
                hh_indicator.append(1.0 if candidate.is_heavy_heavy else 0.0)
            energy_terms = torch.stack(energy_terms)
            hh_indicator_tensor = torch.tensor(
                hh_indicator, device=self.device, dtype=self.dtype
            )
            full_energies = -(state_mask @ energy_terms)
            reference_energies = -(
                state_mask @ (energy_terms * (1.0 - hh_indicator_tensor))
            )
            probabilities = torch.softmax(
                -full_energies / self.heavy_valence_temperature, dim=0
            )
            occupancies = probabilities @ state_mask
            for candidate_index, candidate in enumerate(local_candidates):
                occupancy = occupancies[candidate_index]
                for atom, slot in candidate.directed_slots:
                    row_overrides[atom][slot] = occupancy

            component_energy = (
                free_energy(full_energies, self.heavy_valence_temperature)
                - free_energy(reference_energies, self.heavy_valence_temperature)
            )
            share = component_energy / float(len(component))
            for atom in component:
                desired[atom] = desired[atom] + share
            component_state_counts.append(len(states))
            component_candidate_counts.append(len(local_candidates))

        rows = []
        for atom in range(taper.shape[0]):
            row = mask[atom] + taper[atom].sum() * 0.0
            overrides = row_overrides.get(atom, {})
            if overrides:
                slots = torch.tensor(
                    sorted(overrides), device=self.device, dtype=torch.long
                )
                occupancy = torch.stack([overrides[int(slot)] for slot in slots])
                row = row.scatter(0, slots, occupancy)
            rows.append(row)
        membership = torch.stack(rows)
        self._research_energy_membership = membership
        self._research_desired_heavy_attraction = torch.stack(desired)
        self._research_formulation_diagnostics = {
            "joint_component_state_counts": tuple(component_state_counts),
            "joint_component_candidate_counts": tuple(component_candidate_counts),
            "maximum_joint_state_count": max(component_state_counts, default=1),
            "maximum_joint_candidate_count": max(
                component_candidate_counts, default=0
            ),
        }
        return membership
