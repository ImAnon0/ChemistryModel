"""Table-surface shared bond-state free-energy reference.

This deliberately slow float64 research model derives heavy-heavy bond order
from shared capacity states and the existing single/double/triple Morse tables.
It is not registered in any production selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import reactive as R
from research.heavy_valence_continuous_edge.prototype import _Edge
from research.heavy_valence_state.energy_common import attractive_magnitude, free_energy
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


MAX_BOND_STATES = 100_000


@dataclass(frozen=True)
class _Unit:
    kind: str
    heavy_endpoints: tuple[int, ...]
    taper: torch.Tensor
    energy: torch.Tensor
    lower: int | None
    edge_index: int | None
    level: int


def _valid_capacity_states(units, capacities):
    """Enumerate every hierarchical subset within simultaneous capacities."""
    selected = [False] * len(units)
    degree = {atom: 0 for atom in capacities}
    states = []

    def visit(index):
        if len(states) > MAX_BOND_STATES:
            raise RuntimeError(
                f"bond-state reference exceeded {MAX_BOND_STATES} states"
            )
        if index == len(units):
            states.append(tuple(selected))
            return
        unit = units[index]
        selected[index] = False
        visit(index + 1)
        hierarchy_ok = unit.lower is None or selected[unit.lower]
        capacity_ok = all(
            degree[atom] < capacities[atom] for atom in unit.heavy_endpoints
        )
        if hierarchy_ok and capacity_ok:
            selected[index] = True
            for atom in unit.heavy_endpoints:
                degree[atom] += 1
            visit(index + 1)
            for atom in unit.heavy_endpoints:
                degree[atom] -= 1
        selected[index] = False

    visit(0)
    return tuple(states)


class SharedBondStateHamiltonianPrototype(
    OptimisedValenceStateBatchedSimulation
):
    """Shared table-surface bond-order states, research-only v1."""

    physics_model_name = "research_shared_bond_state_hamiltonian_v1"
    physics_model_revision = 1
    research_only = True

    def _heavy_edges(self, values):
        hydrogen = int(R.ELEMENT_INDEX["H"])
        taper = values["taper"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        directed = {}
        for atom in range(taper.shape[0]):
            if int(self.types_numpy[atom]) == hydrogen:
                continue
            for slot in range(taper.shape[1]):
                if float(mask[atom, slot].detach().cpu()) <= 0.0:
                    continue
                if float(taper[atom, slot].detach().cpu()) <= 1e-12:
                    continue
                other = int(neighbours[atom, slot].detach().cpu())
                if other == atom or int(self.types_numpy[other]) == hydrogen:
                    continue
                directed[(atom, other)] = slot
        edges = []
        for first, second in sorted({tuple(sorted(pair)) for pair in directed}):
            slots = []
            if (first, second) in directed:
                slots.append((first, directed[first, second]))
            if (second, first) in directed:
                slots.append((second, directed[second, first]))
            edges.append(_Edge((first, second), tuple(slots)))
        return tuple(edges)

    @staticmethod
    def _components(heavy_atoms, edges):
        adjacency = {atom: set() for atom in heavy_atoms}
        for edge in edges:
            first, second = edge.atoms
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
        return tuple(components)

    def _edge_surface(self, edge, values):
        """Return tapered R1 and telescoping Vn-R1 increments."""
        taper = values["taper"]
        distance = values["distances"]
        first, second = edge.atoms
        first_type = self.types[first]
        second_type = self.types[second]
        t_edge = torch.stack([
            taper[atom, slot] for atom, slot in edge.directed_slots
        ]).mean()
        r_edge = torch.stack([
            distance[atom, slot] for atom, slot in edge.directed_slots
        ]).mean()
        softening = torch.stack([
            values["pair_depth"][atom, slot]
            / torch.clamp(values["unsoftened_depth"][atom, slot], min=1e-30)
            for atom, slot in edge.directed_slots
        ]).mean()
        tables = (
            (self.bond_length, self.bond_depth, self.bond_width),
            (self.double_length, self.double_depth, self.double_width),
            (self.triple_length, self.triple_depth, self.triple_width),
        )
        surfaces = []
        depths = []
        for length_table, depth_table, width_table in tables:
            length = length_table[first_type, second_type]
            depth = depth_table[first_type, second_type] * softening
            width = width_table[first_type, second_type]
            shift = r_edge - length
            repulsive = depth * torch.exp(-2.0 * width * shift)
            attractive = 2.0 * depth * torch.exp(-width * shift)
            surfaces.append(t_edge * (repulsive - attractive))
            depths.append(depth)
        single_shift = r_edge - tables[0][0][first_type, second_type]
        single_repulsive = (
            t_edge
            * depths[0]
            * torch.exp(-2.0 * tables[0][2][first_type, second_type] * single_shift)
        )
        increments = [surfaces[0] - single_repulsive]
        increments.extend(
            surfaces[level] - surfaces[level - 1] for level in (1, 2)
        )
        depth_increments = [depths[0]]
        depth_increments.extend(
            torch.clamp(depths[level] - depths[level - 1], min=0.0)
            for level in (1, 2)
        )
        return t_edge, single_repulsive, tuple(increments), tuple(depth_increments)

    def _local_valence_membership(self, values):
        inherited = super()._local_valence_membership(values)
        taper = values["taper"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy_atoms = tuple(
            atom for atom, element in enumerate(self.types_numpy)
            if int(element) != hydrogen
        )
        capacities = {
            atom: max(int(round(float(
                self.valence[self.types[atom]].detach().cpu()
            ))), 0)
            for atom in heavy_atoms
        }
        edges = self._heavy_edges(values)
        components = self._components(heavy_atoms, edges)
        edge_by_atom = {atom: [] for atom in heavy_atoms}
        for edge_index, edge in enumerate(edges):
            for atom in edge.atoms:
                edge_by_atom[atom].append(edge_index)

        h_contacts = {atom: [] for atom in heavy_atoms}
        active_contact_count = {atom: 0 for atom in heavy_atoms}
        for atom in heavy_atoms:
            for slot in range(taper.shape[1]):
                if float(mask[atom, slot].detach().cpu()) <= 0.0:
                    continue
                if float(taper[atom, slot].detach().cpu()) <= 1e-12:
                    continue
                other = int(neighbours[atom, slot].detach().cpu())
                if other == atom:
                    continue
                active_contact_count[atom] += 1
                if int(self.types_numpy[other]) == hydrogen:
                    h_contacts[atom].append((other, slot))

        rows = [inherited[atom] for atom in range(taper.shape[0])]
        desired = [taper[atom].sum() * 0.0 for atom in range(taper.shape[0])]
        capacity_load = torch.zeros(
            taper.shape[0], device=self.device, dtype=self.dtype
        )
        replace_over = torch.zeros(
            taper.shape[0], device=self.device, dtype=self.dtype
        )
        component_diagnostics = []
        edge_order = [taper.sum() * 0.0 for _ in edges]
        edge_sigma = [taper.sum() * 0.0 for _ in edges]

        current_pair = taper * (values["repulsive"] - attractive_magnitude(values))

        for component in components:
            component_set = set(component)
            component_edges = sorted({
                edge_index for atom in component for edge_index in edge_by_atom[atom]
            })
            if not component_edges:
                component_diagnostics.append({
                    "atoms": component, "mode": "no_heavy_edges", "states": 1,
                })
                continue
            competitive = any(
                active_contact_count[atom] > capacities[atom] for atom in component
            )
            if not competitive:
                for edge_index in component_edges:
                    edge = edges[edge_index]
                    samples = [
                        current_pair[atom, slot] for atom, slot in edge.directed_slots
                    ]
                    energy = torch.stack(samples).mean()
                    for atom in edge.atoms:
                        desired[atom] = desired[atom] + 0.5 * energy
                    edge_order[edge_index] = torch.stack([
                        values["order"][atom, slot]
                        for atom, slot in edge.directed_slots
                    ]).mean()
                    edge_sigma[edge_index] = edge_order[edge_index] * 0.0 + 1.0
                component_diagnostics.append({
                    "atoms": component, "mode": "identity", "states": 1,
                })
                continue

            # A C1, parameter-free gate replaces the discontinuous choice
            # between the identity and state surfaces. One whole excess
            # valence unit is the natural (dimensionless) transition scale.
            local_gates = []
            for atom in component:
                raw_excess = torch.clamp(
                    torch.sum(taper[atom]) - self.valence[self.types[atom]],
                    0.0, 1.0,
                )
                local_gates.append(
                    raw_excess.square() * (3.0 - 2.0 * raw_excess)
                )
            component_gate = 1.0 - torch.prod(
                1.0 - torch.stack(local_gates)
            )
            for atom in component:
                replace_over[atom] = component_gate

            units = []
            for atom in component:
                for _, slot in sorted(h_contacts[atom]):
                    units.append(_Unit(
                        kind="H", heavy_endpoints=(atom,),
                        taper=taper[atom, slot],
                        energy=-taper[atom, slot] * attractive_magnitude(values)[atom, slot],
                        lower=None, edge_index=None, level=1,
                    ))

            common_repulsive = taper.sum() * 0.0
            base_component = taper.sum() * 0.0
            channel_indices = {edge_index: [] for edge_index in component_edges}
            for edge_index in component_edges:
                edge = edges[edge_index]
                t_edge, repulsive, increments, depth_increments = self._edge_surface(
                    edge, values
                )
                common_repulsive = common_repulsive + repulsive
                base_component = base_component + torch.stack([
                    current_pair[atom, slot] for atom, slot in edge.directed_slots
                ]).mean()
                lower = None
                max_order = min(3, *(capacities[atom] for atom in edge.atoms))
                for level in range(max_order):
                    if float(depth_increments[level].detach().cpu()) <= 1e-12:
                        break
                    unit_index = len(units)
                    units.append(_Unit(
                        kind="HH", heavy_endpoints=edge.atoms, taper=t_edge,
                        energy=increments[level], lower=lower,
                        edge_index=edge_index, level=level + 1,
                    ))
                    channel_indices[edge_index].append(unit_index)
                    lower = unit_index

            states = _valid_capacity_states(units, {
                atom: capacities[atom] for atom in component
            })
            state_mask = torch.tensor(states, device=self.device, dtype=self.dtype)
            energies = torch.stack([unit.energy for unit in units])
            is_h = torch.tensor(
                [unit.kind == "H" for unit in units],
                device=self.device, dtype=self.dtype,
            )
            full_diagonal = state_mask @ energies
            reference_diagonal = state_mask @ (energies * is_h)
            probabilities = torch.softmax(
                -full_diagonal / self.heavy_valence_temperature, dim=0
            )
            occupancies = probabilities @ state_mask
            hamiltonian_energy = (
                common_repulsive
                + free_energy(full_diagonal, self.heavy_valence_temperature)
                - free_energy(reference_diagonal, self.heavy_valence_temperature)
            )
            component_energy = (
                base_component
                + component_gate * (hamiltonian_energy - base_component)
            )
            share = component_energy / float(len(component))
            for atom in component:
                desired[atom] = desired[atom] + share

            for edge_index in component_edges:
                edge = edges[edge_index]
                indices = channel_indices[edge_index]
                state_sigma = occupancies[indices[0]] if indices else taper.sum() * 0.0
                state_order = (
                    occupancies[torch.tensor(indices, device=self.device)].sum()
                    if indices else taper.sum() * 0.0
                )
                inherited_sigma = torch.stack([
                    inherited[atom, slot] for atom, slot in edge.directed_slots
                ]).mean()
                inherited_order = torch.stack([
                    values["order"][atom, slot] for atom, slot in edge.directed_slots
                ]).mean()
                sigma = inherited_sigma + component_gate * (
                    state_sigma - inherited_sigma
                )
                order_value = inherited_order + component_gate * (
                    state_order - inherited_order
                )
                edge_sigma[edge_index] = sigma
                edge_order[edge_index] = order_value
                for atom, slot in edge.directed_slots:
                    rows[atom] = rows[atom].scatter(
                        0, torch.tensor([slot], device=self.device), sigma.reshape(1)
                    )
                for atom in edge.atoms:
                    capacity_load[atom] = capacity_load[atom] + order_value
            for unit_index, unit in enumerate(units):
                if unit.kind == "H":
                    atom = unit.heavy_endpoints[0]
                    capacity_load[atom] = capacity_load[atom] + occupancies[unit_index]
            component_diagnostics.append({
                "atoms": component,
                "mode": "table_surface_free_energy",
                "states": len(states),
                "units": len(units),
                "heavy_channels": sum(unit.kind == "HH" for unit in units),
                "h_contacts": sum(unit.kind == "H" for unit in units),
                "gate": float(component_gate.detach().cpu()),
            })

        self._bond_state_data = {
            "edges": tuple(edge.atoms for edge in edges),
            "edge_order": torch.stack(edge_order) if edge_order else taper.sum().reshape(1)[:0],
            "edge_sigma": torch.stack(edge_sigma) if edge_sigma else taper.sum().reshape(1)[:0],
            "capacity_load": capacity_load,
            "replace_over": replace_over,
            "components": tuple(component_diagnostics),
        }
        self._bond_state_desired_pair = torch.stack(desired)
        return torch.stack(rows)

    def _valence_topology_correction(self, positions):
        angle_correction = super()._valence_topology_correction(positions)
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("bond-state reference lost reactive intermediates")
        values = cached[1]
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy = self.types != hydrogen
        neighbour_heavy = self.types[values["neighbours"]] != hydrogen
        hh = heavy[:, None] & neighbour_heavy & (values["mask"] > 0.0)
        current_pair = values["taper"] * (
            values["repulsive"] - attractive_magnitude(values)
        )
        base = 0.5 * torch.sum(
            torch.where(hh, current_pair, torch.zeros_like(current_pair)), dim=1
        )
        original_over = self._profile_energy_parts["over"]
        load = self._bond_state_data["capacity_load"]
        excess = torch.clamp(load - values["valence"], min=0.0)
        scale = self.over_coordination_scale(
            values["taper"], values["unsoftened_depth"], values["mask"],
            cache_key=positions,
        )
        replacement_over = self.over_penalty * scale * excess.square()
        replace_over = self._bond_state_data["replace_over"] * heavy.to(self.dtype)
        over_term = replace_over * (replacement_over - original_over)
        correction = self._bond_state_desired_pair - base + over_term
        self._heavy_valence_energy_diagnostics = {
            "formulation": "shared_table_surface_bond_state_free_energy",
            "base_heavy_pair_per_atom": base.detach(),
            "desired_heavy_pair_per_atom": self._bond_state_desired_pair.detach(),
            "edge_atoms": self._bond_state_data["edges"],
            "edge_order": self._bond_state_data["edge_order"].detach(),
            "edge_sigma": self._bond_state_data["edge_sigma"].detach(),
            "capacity_load": load.detach(),
            "components": self._bond_state_data["components"],
            "original_over_per_atom": original_over.detach(),
            "replacement_over_per_atom": replacement_over.detach(),
            "energy_correction_per_atom": correction.detach(),
        }
        return angle_correction + correction
