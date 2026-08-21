"""Shared incremental heavy-heavy bond-order channel reference."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

import reactive as R
from research.heavy_valence_continuous_edge.prototype import (
    ContinuousSharedEdgeHeavyValencePrototype,
    _feasible_start,
    _independent_rows,
)
from research.heavy_valence_state.energy_common import attractive_magnitude


class SharedBondOrderChannelPrototype(
    ContinuousSharedEdgeHeavyValencePrototype
):
    """One shared continuous sigma/second/third channel per HH edge."""

    physics_model_name = "research_shared_bond_order_channels_v0"
    physics_model_revision = 0
    research_only = True

    def _solve_channels(
        self, q, attraction, endpoint_incidence, capacity, hierarchy
    ):
        if q.numel() == 0:
            return q, {
                "solver": "empty", "active_capacity_constraints": 0,
                "active_hierarchy_constraints": 0, "zero_channels": 0,
                "maximum_constraint_violation": 0.0,
            }

        q_np = q.detach().cpu().numpy().astype(np.float64, copy=False)
        a_np = attraction.detach().cpu().numpy().astype(np.float64, copy=False)
        m_np = endpoint_incidence.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        cap_np = capacity.detach().cpu().numpy().astype(np.float64, copy=False)
        usage = m_np @ q_np
        if np.all(usage <= cap_np + 1e-11):
            return q, {
                "solver": "unconstrained", "active_capacity_constraints": 0,
                "active_hierarchy_constraints": len(hierarchy),
                "zero_channels": 0,
                "maximum_constraint_violation": float(
                    np.maximum(usage - cap_np, 0.0).max(initial=0.0)
                ),
            }

        if np.any(a_np <= 0.0):
            raise RuntimeError("active bond-order channel has no energy increment")

        hierarchy_rows = []
        for lower, upper in hierarchy:
            row = np.zeros_like(q_np)
            row[lower] = -1.0 / q_np[lower]
            row[upper] = 1.0 / q_np[upper]
            hierarchy_rows.append(row)
        if hierarchy_rows:
            h_np = np.asarray(hierarchy_rows)
            constraint_np = np.vstack((m_np, h_np))
            bound_np = np.concatenate((cap_np, np.zeros(len(hierarchy))))
        else:
            constraint_np = m_np
            bound_np = cap_np

        coefficient = a_np / np.maximum(np.square(q_np), 1e-30)

        def objective(value):
            delta = value - q_np
            return float(np.sum(coefficient * np.square(delta)))

        def gradient(value):
            return 2.0 * coefficient * (value - q_np)

        start = _feasible_start(q_np, m_np, cap_np)
        result = minimize(
            objective,
            start,
            jac=gradient,
            bounds=[(0.0, float(value)) for value in q_np],
            constraints=[{
                "type": "ineq",
                "fun": lambda value: bound_np - constraint_np @ value,
                "jac": lambda value: -constraint_np,
            }],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"bond-channel QP failed: {result.message}")

        scale = max(float(np.max(q_np)), 1.0)
        zero = np.flatnonzero(result.x <= 2e-7 * scale).tolist()
        free = [index for index in range(q_np.size) if index not in zero]
        slack = bound_np - constraint_np @ result.x
        active = np.flatnonzero(slack <= 2e-7 * scale).tolist()

        # Rebuild the live constraint matrix so gradients include q-dependent
        # normalised hierarchy rows rather than a detached optimiser result.
        live_rows = [endpoint_incidence[index] for index in range(m_np.shape[0])]
        live_bounds = [capacity[index] for index in range(m_np.shape[0])]
        for lower, upper in hierarchy:
            row = torch.zeros_like(q)
            row = row.scatter(
                0,
                torch.tensor([lower, upper], device=q.device),
                torch.stack((-1.0 / q[lower], 1.0 / q[upper])),
            )
            live_rows.append(row)
            live_bounds.append(q.sum() * 0.0)
        constraint = torch.stack(live_rows)
        bound = torch.stack(live_bounds)

        b = torch.zeros_like(q)
        independent: list[int] = []
        if free:
            free_tensor = torch.tensor(free, device=q.device, dtype=torch.long)
            q_free = q[free_tensor]
            a_free = attraction[free_tensor]
            d_free = q_free.square() / torch.clamp(2.0 * a_free, min=1e-30)
            weighted = (
                constraint_np[:, free]
                * np.sqrt(
                    np.square(q_np[free])
                    / np.maximum(2.0 * a_np[free], 1e-30)
                )[None, :]
            )
            independent = _independent_rows(weighted, active)
            if independent:
                row_tensor = torch.tensor(
                    independent, device=q.device, dtype=torch.long
                )
                matrix = constraint[row_tensor][:, free_tensor]
                gram = (matrix * d_free[None, :]) @ matrix.T
                rhs = matrix @ q_free - bound[row_tensor]
                multiplier = torch.linalg.solve(gram, rhs)
                b_free = q_free - d_free * (matrix.T @ multiplier)
            else:
                b_free = q_free
            b = b.scatter(0, free_tensor, b_free)

        b = torch.minimum(torch.clamp(b, min=0.0), q)
        violation = constraint @ b - bound
        capacity_active = sum(index < m_np.shape[0] for index in active)
        return b, {
            "solver": "slsqp_active_kkt_channels",
            "active_capacity_constraints": capacity_active,
            "active_hierarchy_constraints": len(active) - capacity_active,
            "independent_constraints": len(independent),
            "zero_channels": len(zero),
            "maximum_constraint_violation": float(
                torch.clamp(violation, min=0.0).max().detach().cpu()
            ),
            "slsqp_iterations": int(result.nit),
        }

    def _local_valence_membership(self, values):
        # Bypass the scalar parent's membership and retain only the validated
        # production H/heavy topology as the starting point.
        inherited = super(
            ContinuousSharedEdgeHeavyValencePrototype, self
        )._local_valence_membership(values)
        taper = values["taper"]
        order = values["order"]
        neighbours = values["neighbours"]
        attractive = attractive_magnitude(values)
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy_atoms = [
            atom for atom, element in enumerate(self.types_numpy)
            if int(element) != hydrogen
        ]
        heavy_index = {atom: index for index, atom in enumerate(heavy_atoms)}
        edges = self._edge_structure(values)

        q_values = []
        energy_values = []
        channel_records = []
        hierarchy = []
        channel_by_edge = []
        edge_full_attraction = []
        unsupported = []

        for edge_index, edge in enumerate(edges):
            t_edge = torch.stack([
                taper[atom, slot] for atom, slot in edge.directed_slots
            ]).mean()
            o_edge = torch.stack([
                order[atom, slot] for atom, slot in edge.directed_slots
            ]).mean()
            a_edge = torch.stack([
                taper[atom, slot] * attractive[atom, slot]
                for atom, slot in edge.directed_slots
            ]).mean()
            edge_full_attraction.append(a_edge)
            first, second = edge.atoms
            first_type = self.types[first]
            second_type = self.types[second]
            depths = torch.stack((
                self.bond_depth[first_type, second_type],
                self.double_depth[first_type, second_type],
                self.triple_depth[first_type, second_type],
            ))
            increments = torch.stack((
                depths[0], depths[1] - depths[0], depths[2] - depths[1]
            ))
            fractions = torch.stack(tuple(
                torch.clamp(o_edge - float(channel), 0.0, 1.0)
                for channel in range(3)
            ))
            edge_q = t_edge * fractions
            weights = increments * edge_q
            active = [
                channel for channel in range(3)
                if float(edge_q[channel].detach().cpu()) > 1e-12
            ]
            bad = [
                channel for channel in active
                if float(weights[channel].detach().cpu()) <= 1e-14
            ]
            if bad:
                unsupported.append((edge.atoms, tuple(channel + 1 for channel in bad)))
                continue
            normalisation = torch.stack([weights[channel] for channel in active]).sum()
            local_indices = []
            for channel in active:
                variable = len(q_values)
                q_values.append(edge_q[channel])
                energy_values.append(a_edge * weights[channel] / normalisation)
                channel_records.append((edge_index, channel, edge.atoms))
                local_indices.append(variable)
            for lower, upper in zip(local_indices, local_indices[1:]):
                hierarchy.append((lower, upper))
            channel_by_edge.append(tuple(local_indices))

        if unsupported:
            raise RuntimeError(
                "preferred bond-order channel has no matching depth increment: "
                f"{unsupported}"
            )

        q = torch.stack(q_values) if q_values else taper.sum().reshape(1)[:0]
        channel_attraction = (
            torch.stack(energy_values)
            if energy_values else taper.sum().reshape(1)[:0]
        )
        incidence = torch.zeros(
            (len(heavy_atoms), len(q_values)), device=self.device, dtype=self.dtype
        )
        for variable, (_, _, atoms) in enumerate(channel_records):
            for atom in atoms:
                incidence[heavy_index[atom], variable] = 1.0

        h_load = torch.stack([
            torch.sum(
                taper[atom]
                * (self.types[neighbours[atom]] == hydrogen).to(self.dtype)
            )
            for atom in heavy_atoms
        ]) if heavy_atoms else taper.sum().reshape(1)[:0]
        elemental_capacity = torch.stack([
            self.valence[self.types[atom]] for atom in heavy_atoms
        ]) if heavy_atoms else taper.sum().reshape(1)[:0]
        residual = torch.clamp(elemental_capacity - h_load, min=0.0)
        b, solver = self._solve_channels(
            q, channel_attraction, incidence, residual, tuple(hierarchy)
        )
        channel_occupancy = b / torch.clamp(q, min=1e-30)

        rows = [inherited[atom] for atom in range(taper.shape[0])]
        capacity_rows = [taper[atom] * 0.0 for atom in range(taper.shape[0])]
        desired = [taper[atom].sum() * 0.0 for atom in range(taper.shape[0])]
        edge_occupancy = []
        edge_capacity = []
        edge_preferred = []
        for edge_index, edge in enumerate(edges):
            variables = channel_by_edge[edge_index]
            index = torch.tensor(variables, device=self.device, dtype=torch.long)
            total_b = b[index].sum()
            total_q = q[index].sum()
            occupancy = total_b / torch.clamp(total_q, min=1e-30)
            edge_energy = torch.sum(
                -channel_attraction[index]
                * (2.0 * channel_occupancy[index] - channel_occupancy[index].square())
            )
            edge_occupancy.append(occupancy)
            edge_capacity.append(total_b)
            edge_preferred.append(total_q)
            for atom, slot in edge.directed_slots:
                rows[atom] = rows[atom].scatter(
                    0, torch.tensor([slot], device=self.device), occupancy.reshape(1)
                )
                capacity_rows[atom] = capacity_rows[atom].scatter(
                    0, torch.tensor([slot], device=self.device), total_b.reshape(1)
                )
            for atom in edge.atoms:
                desired[atom] = desired[atom] + 0.5 * edge_energy

        self._continuous_edge_state = {
            "edges": tuple(
                (first, second, channel + 1)
                for _, channel, (first, second) in channel_records
            ),
            "preferred_capacity": q,
            "capacity": b,
            "occupancy": channel_occupancy,
            "edge_occupancy": torch.stack(edge_occupancy) if edge_occupancy else q,
            "edge_capacity": torch.stack(edge_capacity) if edge_capacity else q,
            "edge_preferred_capacity": (
                torch.stack(edge_preferred) if edge_preferred else q
            ),
            "channel_attraction": channel_attraction,
            "channels_by_edge": tuple(channel_by_edge),
            "edge_full_attraction": (
                torch.stack(edge_full_attraction) if edge_full_attraction else q
            ),
            "hierarchy": tuple(hierarchy),
            "incidence": incidence,
            "h_capacity_load": h_load,
            "residual_capacity": residual,
            "heavy_atoms": tuple(heavy_atoms),
            "capacity_rows": torch.stack(capacity_rows),
            "solver": solver,
        }
        self._continuous_desired_heavy_attraction = torch.stack(desired)
        return torch.stack(rows)
