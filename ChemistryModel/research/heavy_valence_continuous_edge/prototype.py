"""Continuous shared-edge bond-capacity reference.

This intentionally slow float64/SciPy reference is not wired into any
production selector.  It provides one capacity variable for each undirected
heavy-heavy edge and differentiates the resulting scalar energy with Torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

import reactive as R
from research.heavy_valence_state.energy_common import attractive_magnitude
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


@dataclass(frozen=True)
class _Edge:
    atoms: tuple[int, int]
    directed_slots: tuple[tuple[int, int], ...]


def _independent_rows(matrix: np.ndarray, candidates: list[int]) -> list[int]:
    """Return a deterministic independent subset of candidate row indices."""
    selected: list[int] = []
    rank = 0
    for index in candidates:
        trial = matrix[selected + [index]]
        trial_rank = int(np.linalg.matrix_rank(trial, tol=1e-10))
        if trial_rank > rank:
            selected.append(index)
            rank = trial_rank
    return selected


def _feasible_start(q: np.ndarray, incidence: np.ndarray, capacity: np.ndarray):
    value = q.copy()
    for _ in range(100):
        usage = incidence @ value
        overloaded = usage > capacity + 1e-13
        if not np.any(overloaded):
            break
        factors = np.ones_like(capacity)
        factors[overloaded] = capacity[overloaded] / usage[overloaded]
        edge_factor = np.ones_like(value)
        for edge in range(value.size):
            incident = incidence[:, edge] > 0.0
            if np.any(incident):
                edge_factor[edge] = np.min(factors[incident])
        value *= edge_factor
    return np.clip(value, 0.0, q)


class ContinuousSharedEdgeHeavyValencePrototype(
    OptimisedValenceStateBatchedSimulation
):
    """Research QP with one continuous capacity shared by both endpoints."""

    physics_model_name = "research_continuous_shared_edge_heavy_valence_v0"
    physics_model_revision = 0
    research_only = True

    def _edge_structure(self, values):
        hydrogen = int(R.ELEMENT_INDEX["H"])
        taper = values["taper"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        directed: dict[tuple[int, int], int] = {}
        for atom in range(taper.shape[0]):
            if int(self.types_numpy[atom]) == hydrogen:
                continue
            for slot in range(taper.shape[1]):
                if float(mask[atom, slot].detach().cpu()) <= 0.0:
                    continue
                if float(taper[atom, slot].detach().cpu()) <= 0.0:
                    continue
                other = int(neighbours[atom, slot].detach().cpu())
                if other == atom or int(self.types_numpy[other]) == hydrogen:
                    continue
                directed[(atom, other)] = slot

        edges: list[_Edge] = []
        for first, second in sorted({tuple(sorted(pair)) for pair in directed}):
            slots = []
            if (first, second) in directed:
                slots.append((first, directed[(first, second)]))
            if (second, first) in directed:
                slots.append((second, directed[(second, first)]))
            edges.append(_Edge((first, second), tuple(slots)))
        return tuple(edges)

    def _solve_capacity(self, q, attraction, incidence, capacity):
        """Solve the weighted projection and rebuild its active KKT in Torch."""
        if q.numel() == 0:
            return q, {
                "solver": "empty", "active_capacity_constraints": 0,
                "zero_capacity_edges": 0, "maximum_capacity_violation": 0.0,
            }

        q_np = q.detach().cpu().numpy().astype(np.float64, copy=False)
        a_np = attraction.detach().cpu().numpy().astype(np.float64, copy=False)
        m_np = incidence.detach().cpu().numpy().astype(np.float64, copy=False)
        cap_np = capacity.detach().cpu().numpy().astype(np.float64, copy=False)
        usage_np = m_np @ q_np
        if np.all(usage_np <= cap_np + 1e-11):
            return q, {
                "solver": "unconstrained", "active_capacity_constraints": 0,
                "zero_capacity_edges": 0,
                "maximum_capacity_violation": float(
                    np.maximum(usage_np - cap_np, 0.0).max(initial=0.0)
                ),
            }

        coefficient = a_np / np.maximum(q_np * q_np, 1e-30)

        def objective(value):
            delta = value - q_np
            return float(np.sum(coefficient * delta * delta))

        def gradient(value):
            return 2.0 * coefficient * (value - q_np)

        result = minimize(
            objective,
            _feasible_start(q_np, m_np, cap_np),
            jac=gradient,
            bounds=[(0.0, float(value)) for value in q_np],
            constraints=[{
                "type": "ineq",
                "fun": lambda value: cap_np - m_np @ value,
                "jac": lambda value: -m_np,
            }],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"continuous-edge QP failed: {result.message}")

        scale = max(float(np.max(q_np)), 1.0)
        zero = np.flatnonzero(result.x <= 2e-7 * scale).tolist()
        free = [index for index in range(q_np.size) if index not in zero]
        slack = cap_np - m_np @ result.x
        active = np.flatnonzero(slack <= 2e-7 * scale).tolist()

        b = torch.zeros_like(q)
        if free:
            free_tensor = torch.tensor(free, device=q.device, dtype=torch.long)
            q_free = q[free_tensor]
            a_free = attraction[free_tensor]
            d_free = q_free.square() / torch.clamp(2.0 * a_free, min=1e-30)
            weighted_rows = (
                m_np[:, free] * np.sqrt(
                    q_np[free] ** 2 / np.maximum(2.0 * a_np[free], 1e-30)
                )[None, :]
            )
            independent = _independent_rows(weighted_rows, active)
            if independent:
                row_tensor = torch.tensor(
                    independent, device=q.device, dtype=torch.long
                )
                matrix = incidence[row_tensor][:, free_tensor]
                gram = (matrix * d_free[None, :]) @ matrix.T
                rhs = matrix @ q_free - capacity[row_tensor]
                multiplier = torch.linalg.solve(gram, rhs)
                b_free = q_free - d_free * (matrix.T @ multiplier)
            else:
                b_free = q_free
            b = b.scatter(0, free_tensor, b_free)

        # Active-set changes are the ordinary piecewise-smooth boundaries of
        # a strictly convex projection.  Clamp only roundoff-sized excursions.
        b = torch.minimum(torch.clamp(b, min=0.0), q)
        violation = incidence @ b - capacity
        return b, {
            "solver": "slsqp_active_kkt",
            "active_capacity_constraints": len(active),
            "independent_capacity_constraints": len(independent) if free else 0,
            "zero_capacity_edges": len(zero),
            "maximum_capacity_violation": float(
                torch.clamp(violation, min=0.0).max().detach().cpu()
            ),
            "slsqp_iterations": int(result.nit),
        }

    def _local_valence_membership(self, values):
        # Preserve the validated H/contact topology first.  Only HH slots are
        # replaced below, and H-state energy itself remains untouched.
        inherited = super()._local_valence_membership(values)
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

        q_terms = []
        a_terms = []
        for edge in edges:
            q_samples = [taper[a, s] * order[a, s] for a, s in edge.directed_slots]
            a_samples = [taper[a, s] * attractive[a, s] for a, s in edge.directed_slots]
            q_terms.append(torch.stack(q_samples).mean())
            a_terms.append(torch.stack(a_samples).mean())
        q = torch.stack(q_terms) if q_terms else taper.sum().reshape(1)[:0]
        edge_attraction = (
            torch.stack(a_terms) if a_terms else taper.sum().reshape(1)[:0]
        )

        incidence = torch.zeros(
            (len(heavy_atoms), len(edges)), device=self.device, dtype=self.dtype
        )
        for edge_index, edge in enumerate(edges):
            for atom in edge.atoms:
                incidence[heavy_index[atom], edge_index] = 1.0

        h_load_terms = []
        for atom in heavy_atoms:
            neighbour_types = self.types[neighbours[atom]]
            h_contact = (neighbour_types == hydrogen).to(self.dtype)
            h_load_terms.append(torch.sum(
                taper[atom] * h_contact
            ))
        h_load = (
            torch.stack(h_load_terms)
            if h_load_terms else taper.sum().reshape(1)[:0]
        )
        elemental_capacity = torch.stack([
            self.valence[self.types[atom]] for atom in heavy_atoms
        ]) if heavy_atoms else taper.sum().reshape(1)[:0]
        residual_capacity = torch.clamp(elemental_capacity - h_load, min=0.0)
        b, solver = self._solve_capacity(
            q, edge_attraction, incidence, residual_capacity
        )
        occupancy = b / torch.clamp(q, min=1e-30)

        rows = [inherited[atom] for atom in range(taper.shape[0])]
        capacity_rows = [taper[atom] * 0.0 for atom in range(taper.shape[0])]
        desired = [taper[atom].sum() * 0.0 for atom in range(taper.shape[0])]
        for edge_index, edge in enumerate(edges):
            x = occupancy[edge_index]
            # E=-A(2x-x^2): zero at no bond, exactly the inherited attraction
            # and zero dE/db at the preferred current bond capacity x=1.
            edge_energy = -edge_attraction[edge_index] * (2.0 * x - x.square())
            for atom, slot in edge.directed_slots:
                rows[atom] = rows[atom].scatter(
                    0,
                    torch.tensor([slot], device=self.device),
                    x.reshape(1),
                )
                capacity_rows[atom] = capacity_rows[atom].scatter(
                    0,
                    torch.tensor([slot], device=self.device),
                    b[edge_index].reshape(1),
                )
            for atom in edge.atoms:
                desired[atom] = desired[atom] + 0.5 * edge_energy

        self._continuous_edge_state = {
            "edges": tuple(edge.atoms for edge in edges),
            "preferred_capacity": q,
            "capacity": b,
            "occupancy": occupancy,
            "incidence": incidence,
            "h_capacity_load": h_load,
            "residual_capacity": residual_capacity,
            "heavy_atoms": tuple(heavy_atoms),
            "capacity_rows": torch.stack(capacity_rows),
            "solver": solver,
        }
        self._continuous_desired_heavy_attraction = torch.stack(desired)
        return torch.stack(rows)

    def _valence_topology_correction(self, positions):
        angle_correction = super()._valence_topology_correction(positions)
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("continuous-edge reference lost reactive intermediates")
        values = cached[1]
        state = self._continuous_edge_state
        desired = self._continuous_desired_heavy_attraction
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy = self.types != hydrogen
        neighbour_heavy = self.types[values["neighbours"]] != hydrogen
        hh = heavy[:, None] & neighbour_heavy & (values["mask"] > 0.0)
        contact_attraction = values["taper"] * attractive_magnitude(values)
        base = -0.5 * torch.sum(
            torch.where(hh, contact_attraction, torch.zeros_like(contact_attraction)),
            dim=1,
        )

        capacity_load = torch.zeros_like(values["valence"])
        for local, atom in enumerate(state["heavy_atoms"]):
            capacity_load[atom] = (
                state["h_capacity_load"][local]
                + state["capacity_rows"][atom].sum()
            )
        # Hydrogen rows retain the production overcoordination term; heavy
        # rows use the exact same capacity constrained above.
        effective_excess = torch.clamp(capacity_load - values["valence"], min=0.0)
        scale = self.over_coordination_scale(
            values["taper"], values["unsoftened_depth"], values["mask"],
            cache_key=positions,
        )
        replacement_over = self.over_penalty * scale * effective_excess.square()
        original_over = self._profile_energy_parts["over"]
        correction = desired - base + heavy.to(self.dtype) * (
            replacement_over - original_over
        )
        self._heavy_valence_energy_diagnostics = {
            "formulation": "continuous_shared_edge_capacity",
            "membership": self._continuous_edge_state["occupancy"].detach(),
            "edge_atoms": self._continuous_edge_state["edges"],
            "preferred_capacity": self._continuous_edge_state["preferred_capacity"].detach(),
            "edge_capacity": self._continuous_edge_state["capacity"].detach(),
            "h_capacity_load": self._continuous_edge_state["h_capacity_load"].detach(),
            "residual_capacity": self._continuous_edge_state["residual_capacity"].detach(),
            "base_heavy_attraction_per_atom": base.detach(),
            "desired_heavy_attraction_per_atom": desired.detach(),
            "original_over_per_atom": original_over.detach(),
            "replacement_over_per_atom": replacement_over.detach(),
            "capacity_load": capacity_load.detach(),
            "energy_correction_per_atom": correction.detach(),
            **self._continuous_edge_state["solver"],
        }
        return angle_correction + correction
