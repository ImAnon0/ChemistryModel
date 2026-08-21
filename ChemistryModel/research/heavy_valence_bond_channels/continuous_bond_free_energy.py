"""Continuous table-surface shared bond-order free-energy reference."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

import reactive as R
from research.heavy_valence_bond_channels.bond_state_hamiltonian import (
    SharedBondStateHamiltonianPrototype,
    _Unit,
)
from research.heavy_valence_state.energy_common import attractive_magnitude
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


class ContinuousBondFreeEnergyPrototype(OptimisedValenceStateBatchedSimulation):
    """Mean-field shared bond-order channels derived from table surfaces."""

    physics_model_name = "research_continuous_table_surface_bond_free_energy_v2"
    physics_model_revision = 2
    research_only = True

    _heavy_edges = SharedBondStateHamiltonianPrototype._heavy_edges
    _components = staticmethod(SharedBondStateHamiltonianPrototype._components)
    _edge_surface = SharedBondStateHamiltonianPrototype._edge_surface

    def _component_gate(self, component, component_edges, edges, values):
        taper = values["taper"]
        local_gates = []
        for atom in component:
            overload = torch.clamp(
                torch.sum(taper[atom]) - self.valence[self.types[atom]],
                0.0, 1.0,
            )
            local_gates.append(overload.square() * (3.0 - 2.0 * overload))
        return 1.0 - torch.prod(1.0 - torch.stack(local_gates))

    def _solve_free_energy(self, units, capacities, reference=False):
        count = len(units)
        atoms = tuple(sorted(capacities))
        atom_index = {atom: index for index, atom in enumerate(atoms)}
        q = torch.stack([unit.taper for unit in units])
        energy = torch.stack([unit.energy for unit in units])
        if reference:
            selector = torch.tensor(
                [unit.kind == "H" for unit in units],
                device=self.device, dtype=self.dtype,
            )
            energy = energy * selector
        incidence = torch.zeros(
            (len(atoms), count), device=self.device, dtype=self.dtype
        )
        for column, unit in enumerate(units):
            for atom in unit.heavy_endpoints:
                incidence[atom_index[atom], column] = q[column]
        capacity = torch.tensor(
            [capacities[atom] for atom in atoms],
            device=self.device, dtype=self.dtype,
        )
        hierarchy = [
            (unit.lower, index) for index, unit in enumerate(units)
            if unit.lower is not None
        ]

        q_np = q.detach().cpu().numpy().astype(np.float64, copy=False)
        e_np = energy.detach().cpu().numpy().astype(np.float64, copy=False)
        m_np = incidence.detach().cpu().numpy().astype(np.float64, copy=False)
        cap_np = capacity.detach().cpu().numpy().astype(np.float64, copy=False)
        tau = float(self.heavy_valence_temperature)
        lower_bound = 1e-10
        upper_bound = 1.0 - lower_bound

        def entropy(x):
            return x * np.log(x) + (1.0 - x) * np.log(1.0 - x)

        def objective(x):
            return float(np.sum(e_np * x + tau * q_np * entropy(x)))

        def gradient(x):
            return e_np + tau * q_np * np.log(x / (1.0 - x))

        rows = [-m_np[row] for row in range(len(atoms))]
        bounds = [cap_np[row] for row in range(len(atoms))]
        for lower, upper in hierarchy:
            row = np.zeros(count, dtype=np.float64)
            row[lower] = 1.0
            row[upper] = -1.0
            rows.append(row)
            bounds.append(0.0)
        constraint_matrix = np.asarray(rows)
        constraint_offset = np.asarray(bounds)
        thermal_ratio = np.divide(
            e_np, tau * q_np,
            out=np.zeros_like(e_np), where=q_np > 1e-14,
        )
        start = 1.0 / (1.0 + np.exp(np.clip(thermal_ratio, -40.0, 40.0)))
        start = np.clip(start, lower_bound * 10.0, upper_bound)
        # Project a chemically informed logistic start into endpoint capacity
        # while retaining channel hierarchy. This greatly improves SLSQP's
        # behaviour on large, strongly bonded components.
        for _ in range(100):
            for lower, upper in hierarchy:
                start[upper] = min(start[upper], start[lower])
            usage = m_np @ start
            overloaded = usage > cap_np
            if not np.any(overloaded):
                break
            factors = np.ones_like(cap_np)
            factors[overloaded] = cap_np[overloaded] / usage[overloaded]
            for column, unit in enumerate(units):
                start[column] *= min(
                    factors[atom_index[atom]] for atom in unit.heavy_endpoints
                )
            start = np.clip(start, lower_bound * 10.0, upper_bound)
        constraints = [{
                "type": "ineq",
                "fun": lambda x: constraint_offset + constraint_matrix @ x,
                "jac": lambda x: constraint_matrix,
            }]
        result = minimize(
            objective, start, jac=gradient,
            bounds=[(lower_bound, upper_bound)] * count,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 2000, "disp": False},
        )
        if not result.success:
            result = minimize(
                objective, result.x, jac=gradient,
                bounds=[(lower_bound, upper_bound)] * count,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-9, "maxiter": 2000, "disp": False},
            )
        if not result.success:
            violation = np.maximum(m_np @ result.x - cap_np, 0.0).max(initial=0.0)
            hierarchy_violation = max(
                (result.x[upper] - result.x[lower] for lower, upper in hierarchy),
                default=0.0,
            )
            raise RuntimeError(
                "continuous bond free energy failed "
                f"({'reference' if reference else 'full'}): {result.message}; "
                f"iterations={result.nit}, capacity_violation={violation:.3e}, "
                f"hierarchy_violation={hierarchy_violation:.3e}"
            )

        x = torch.tensor(
            result.x, device=self.device, dtype=self.dtype
        )
        entropy_live = x * torch.log(x) + (1.0 - x) * torch.log(1.0 - x)
        live_value = torch.sum(energy * x + tau * q * entropy_live)

        # Envelope theorem for geometry-dependent taper coefficients in the
        # endpoint constraints. SLSQP reports multipliers for the user
        # inequalities in their supplied order; bound multipliers do not
        # matter because those bounds are geometry independent.
        multipliers = np.asarray(result.multipliers, dtype=np.float64)
        capacity_multiplier = torch.tensor(
            multipliers[:len(atoms)], device=self.device, dtype=self.dtype
        )
        capacity_residual = incidence @ x - capacity
        live_value = live_value + torch.sum(
            capacity_multiplier
            * (capacity_residual - capacity_residual.detach())
        )
        return x, live_value, {
            "iterations": int(result.nit),
            "max_capacity_violation": float(
                np.maximum(m_np @ result.x - cap_np, 0.0).max(initial=0.0)
            ),
            "active_capacity_constraints": int(np.count_nonzero(
                cap_np - m_np @ result.x <= 2e-7
            )),
        }

    def _local_valence_membership(self, values):
        # Keep the established differentiable topology/angle memberships.
        # This experiment changes the scalar radial bond representation only.
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

        current_pair = taper * (values["repulsive"] - attractive_magnitude(values))
        desired = [taper[atom].sum() * 0.0 for atom in range(taper.shape[0])]
        replacement_weight = torch.zeros(
            taper.shape[0], device=self.device, dtype=self.dtype
        )
        diagnostics = []
        edge_orders = [taper.sum() * 0.0 for _ in edges]

        for component in components:
            component_edges = sorted({
                edge_index for atom in component for edge_index in edge_by_atom[atom]
            })
            if not component_edges:
                diagnostics.append({"atoms": component, "mode": "no_heavy_edges"})
                continue
            competitive = any(
                active_contact_count[atom] > capacities[atom] for atom in component
            )
            base_component = sum((
                torch.stack([
                    current_pair[atom, slot]
                    for atom, slot in edges[edge_index].directed_slots
                ]).mean()
                for edge_index in component_edges
            ), taper.sum() * 0.0)
            if not competitive:
                for edge_index in component_edges:
                    edge_orders[edge_index] = torch.stack([
                        values["order"][atom, slot]
                        for atom, slot in edges[edge_index].directed_slots
                    ]).mean()
                share = base_component / float(len(component))
                for atom in component:
                    desired[atom] = desired[atom] + share
                diagnostics.append({"atoms": component, "mode": "identity"})
                continue

            gate = self._component_gate(
                component, component_edges, edges, values
            )
            for atom in component:
                replacement_weight[atom] = gate

            units = []
            for atom in component:
                for _, slot in sorted(h_contacts[atom]):
                    units.append(_Unit(
                        kind="H", heavy_endpoints=(atom,), taper=taper[atom, slot],
                        energy=-taper[atom, slot] * attractive_magnitude(values)[atom, slot],
                        lower=None, edge_index=None, level=1,
                    ))
            common_repulsive = taper.sum() * 0.0
            channel_indices = {edge_index: [] for edge_index in component_edges}
            for edge_index in component_edges:
                edge = edges[edge_index]
                t_edge, repulsive, increments, depth_increments = self._edge_surface(
                    edge, values
                )
                common_repulsive = common_repulsive + repulsive
                lower = None
                max_order = min(3, *(capacities[atom] for atom in edge.atoms))
                for level in range(max_order):
                    if float(depth_increments[level].detach().cpu()) <= 1e-12:
                        break
                    index = len(units)
                    units.append(_Unit(
                        kind="HH", heavy_endpoints=edge.atoms, taper=t_edge,
                        energy=increments[level], lower=lower,
                        edge_index=edge_index, level=level + 1,
                    ))
                    channel_indices[edge_index].append(index)
                    lower = index

            full_x, full_value, full_info = self._solve_free_energy(
                units, {atom: capacities[atom] for atom in component}, False
            )
            _, reference_value, reference_info = self._solve_free_energy(
                units, {atom: capacities[atom] for atom in component}, True
            )
            state_component = common_repulsive + full_value - reference_value
            component_energy = base_component + gate * (
                state_component - base_component
            )
            share = component_energy / float(len(component))
            for atom in component:
                desired[atom] = desired[atom] + share
            for edge_index in component_edges:
                indices = channel_indices[edge_index]
                state_order = full_x[torch.tensor(indices, device=self.device)].sum()
                inherited_order = torch.stack([
                    values["order"][atom, slot]
                    for atom, slot in edges[edge_index].directed_slots
                ]).mean()
                edge_orders[edge_index] = inherited_order + gate * (
                    state_order - inherited_order
                )
            diagnostics.append({
                "atoms": component,
                "mode": "continuous_table_surface_free_energy",
                "units": len(units),
                "gate": float(gate.detach().cpu()),
                "full": full_info,
                "reference": reference_info,
            })

        self._continuous_bond_data = {
            "edges": tuple(edge.atoms for edge in edges),
            "edge_order": torch.stack(edge_orders) if edge_orders else taper.sum().reshape(1)[:0],
            "replacement_weight": replacement_weight,
            "components": tuple(diagnostics),
        }
        self._continuous_bond_desired_pair = torch.stack(desired)
        return inherited

    def _valence_topology_correction(self, positions):
        angle_correction = super()._valence_topology_correction(positions)
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("continuous bond reference lost reactive intermediates")
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
        weight = self._continuous_bond_data["replacement_weight"] * heavy.to(self.dtype)
        correction = (
            self._continuous_bond_desired_pair - base - weight * original_over
        )
        self._heavy_valence_energy_diagnostics = {
            "formulation": "continuous_table_surface_bond_free_energy",
            "base_heavy_pair_per_atom": base.detach(),
            "desired_heavy_pair_per_atom": self._continuous_bond_desired_pair.detach(),
            "edge_atoms": self._continuous_bond_data["edges"],
            "edge_order": self._continuous_bond_data["edge_order"].detach(),
            "replacement_weight": weight.detach(),
            "components": self._continuous_bond_data["components"],
            "original_over_per_atom": original_over.detach(),
            "energy_correction_per_atom": correction.detach(),
        }
        return angle_correction + correction


class OverlapGatedBondFreeEnergyPrototype(ContinuousBondFreeEnergyPrototype):
    """V3: require both overload and genuine heavy-heavy orbital overlap."""

    physics_model_name = "research_overlap_gated_bond_free_energy_v3"
    physics_model_revision = 3

    def _component_gate(self, component, component_edges, edges, values):
        overload_gate = super()._component_gate(
            component, component_edges, edges, values
        )
        taper = values["taper"]
        distances = values["distances"]
        edge_gates = []
        for edge_index in component_edges:
            edge = edges[edge_index]
            first, second = edge.atoms
            first_type = self.types[first]
            second_type = self.types[second]
            t_edge = torch.stack([
                taper[atom, slot] for atom, slot in edge.directed_slots
            ]).mean()
            r_edge = torch.stack([
                distances[atom, slot] for atom, slot in edge.directed_slots
            ]).mean()
            length = self.bond_length[first_type, second_type]
            width = self.bond_width[first_type, second_type]
            overlap = torch.clamp(
                t_edge * torch.exp(-width * (r_edge - length)), 0.0, 1.0
            )
            edge_gates.append(overlap.square() * (3.0 - 2.0 * overlap))
        bond_gate = 1.0 - torch.prod(1.0 - torch.stack(edge_gates))
        return overload_gate * bond_gate
