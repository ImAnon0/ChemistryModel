"""Dual, continuous, unified H/heavy bond-capacity research prototype.

This module is intentionally not registered by any production selector.  It
reuses the established radial tables and H-transfer couplings, but obtains all
H and heavy-heavy attractive energy from one capacity-constrained scalar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

import reactive as R
from batched_torch import BatchedReactiveSimulation
from h_state_factorised_torch import _all_valid_states
from h_state_torch import (
    _contact_overlap,
    _crowding_normalisation,
    _single_h_transfer,
)
from research.heavy_valence_bond_channels.bond_state_hamiltonian import (
    SharedBondStateHamiltonianPrototype,
)
from research.heavy_valence_state.energy_common import attractive_magnitude
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


@dataclass
class _HFactor:
    atoms: tuple[tuple[int, int], ...]
    rows: tuple[int, ...]
    slots: tuple[int, ...]
    states: tuple[tuple[int, ...], ...]
    hamiltonian: torch.Tensor
    capacity: torch.Tensor


@dataclass
class _HeavyFactor:
    atoms: tuple[int, int]
    directed_slots: tuple[tuple[int, int], ...]
    energies: torch.Tensor
    capacity: torch.Tensor


def _ground_state(
    matrix: torch.Tensor,
    regularisation_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smoothly regularised ground value and basis probabilities."""

    if matrix.shape[0] == 1:
        return matrix[0, 0], torch.ones(1, device=matrix.device, dtype=matrix.dtype)
    values, vectors = torch.linalg.eigh(matrix)
    scaled = -values / float(regularisation_temperature)
    eigen_probabilities = torch.softmax(scaled, dim=0)
    probabilities = vectors.square() @ eigen_probabilities
    value = -float(regularisation_temperature) * torch.logsumexp(scaled, dim=0)
    return value, probabilities


def _discrete_free_energy(
    energies: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable free energy and probabilities for one discrete bond factor."""

    scaled = -energies / float(temperature)
    probabilities = torch.softmax(scaled, dim=0)
    value = -float(temperature) * torch.logsumexp(scaled, dim=0)
    return value, probabilities


class UnifiedBondCapacityEnergyPrototype(OptimisedValenceStateBatchedSimulation):
    """One dual capacity allocation for H transfer and heavy bond order.

    This first variant isolates the radial hypothesis.  It keeps the validated
    production heavy-angle topology so the bond-energy result can be assessed
    independently of an angle-model redesign.  The sibling topology variant
    below feeds the same solved occupancies into heavy-centred angles.
    """

    physics_model_name = "research_unified_bond_capacity_energy_v0"
    physics_model_revision = 0
    research_only = True

    _heavy_edges = SharedBondStateHamiltonianPrototype._heavy_edges
    _edge_surface = SharedBondStateHamiltonianPrototype._edge_surface

    def __init__(
        self,
        *args,
        unified_capacity_temperature=0.01,
        unified_h_regularisation_temperature=1e-4,
        **kwargs,
    ):
        self.unified_capacity_temperature = float(unified_capacity_temperature)
        if self.unified_capacity_temperature <= 0.0:
            raise ValueError("unified_capacity_temperature must be positive")
        self.unified_h_regularisation_temperature = float(
            unified_h_regularisation_temperature
        )
        if self.unified_h_regularisation_temperature <= 0.0:
            raise ValueError(
                "unified_h_regularisation_temperature must be positive"
            )
        self._unified_membership = None
        self._unified_diagnostics = None
        self._unified_lambda_cache = {}
        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        # Evaluate the base exactly once, retaining its live intermediates.
        base = BatchedReactiveSimulation.energy_per_atom(self, positions)
        try:
            capacity_correction = self._unified_capacity_correction(
                positions, base
            )
            topology_correction = self._valence_topology_correction(positions)
        finally:
            self._reactive_intermediates = None
        return base + capacity_correction + topology_correction

    def _h_factor(self, edge_atoms, edge_rows, edge_slots, values, heavy_index):
        taper = values["taper"]
        depth = values["pair_depth"]
        width = values["pair_width"]
        shift = values["shift"]
        zero = taper.sum() * 0.0
        states = _all_valid_states(edge_atoms, self.types_numpy)
        edge_taper = [taper[row, slot] for row, slot in zip(edge_rows, edge_slots)]
        edge_depth = [depth[row, slot] for row, slot in zip(edge_rows, edge_slots)]
        edge_attraction = [
            edge_taper[index]
            * 2.0
            * edge_depth[index]
            * torch.exp(-width[row, slot] * shift[row, slot])
            for index, (row, slot) in enumerate(zip(edge_rows, edge_slots))
        ]

        diagonal = torch.stack([
            -torch.stack([edge_attraction[index] for index in state]).sum()
            if state else zero
            for state in states
        ])
        transitions = {}
        weighted_degree = {}
        for first in range(len(states)):
            for second in range(first + 1, len(states)):
                transfer = _single_h_transfer(
                    states[first], states[second], edge_atoms, self.types_numpy
                )
                if transfer is None:
                    continue
                old, new, hydrogen = transfer
                overlap = _contact_overlap(edge_taper[old], edge_taper[new])
                transitions[first, second] = old, new, hydrogen, overlap
                weight_value = overlap.square()
                weighted_degree[first, hydrogen] = (
                    weighted_degree.get((first, hydrogen), zero) + weight_value
                )
                weighted_degree[second, hydrogen] = (
                    weighted_degree.get((second, hydrogen), zero) + weight_value
                )

        couplings = {}
        for key, (old, new, hydrogen, overlap) in transitions.items():
            first, second = key
            normal_first = _crowding_normalisation(
                weighted_degree[first, hydrogen]
            )
            normal_second = _crowding_normalisation(
                weighted_degree[second, hydrogen]
            )
            couplings[key] = (
                self.h_state_mixing
                * torch.sqrt(torch.clamp(edge_depth[old] * edge_depth[new], min=1e-12))
                * overlap
                / torch.sqrt(torch.clamp(normal_first * normal_second, min=1e-12))
            )

        rows = []
        for first in range(len(states)):
            matrix_row = []
            for second in range(len(states)):
                if first == second:
                    value = diagonal[first]
                else:
                    key = (min(first, second), max(first, second))
                    value = -couplings[key] if key in couplings else zero
                matrix_row.append(value)
            rows.append(torch.stack(matrix_row))
        hamiltonian = torch.stack(rows)

        capacity_rows = []
        hydrogen = int(R.ELEMENT_INDEX["H"])
        for state in states:
            loads = [zero for _ in heavy_index]
            for edge_index in state:
                for atom in edge_atoms[edge_index]:
                    if int(self.types_numpy[atom]) == hydrogen:
                        continue
                    loads[heavy_index[atom]] = (
                        loads[heavy_index[atom]] + edge_taper[edge_index]
                    )
            capacity_rows.append(torch.stack(loads) if loads else zero.reshape(1)[:0])
        capacity = torch.stack(capacity_rows)
        return _HFactor(
            tuple(edge_atoms), tuple(edge_rows), tuple(edge_slots), states,
            hamiltonian, capacity,
        )

    def _heavy_factor(self, edge, values, heavy_index):
        taper, repulsive, increments, depth_increments = self._edge_surface(
            edge, values
        )
        first, second = edge.atoms
        max_order = min(
            3,
            int(round(float(self.valence[self.types[first]].detach().cpu()))),
            int(round(float(self.valence[self.types[second]].detach().cpu()))),
        )
        energies = [repulsive * 0.0]
        total = repulsive * 0.0
        for level in range(max_order):
            if float(depth_increments[level].detach().cpu()) <= 1e-12:
                break
            total = total + increments[level]
            energies.append(total)
        state_energy = torch.stack(energies)
        rows = []
        for order in range(len(energies)):
            load = [repulsive * 0.0 for _ in heavy_index]
            amount = taper * float(order)
            load[heavy_index[first]] = amount
            load[heavy_index[second]] = amount
            rows.append(torch.stack(load))
        return _HeavyFactor(
            edge.atoms, edge.directed_slots, state_energy, torch.stack(rows)
        ), repulsive

    def _solve_dual(self, h_factors, heavy_factors, capacity, initial=None):
        count = int(capacity.numel())
        if count == 0:
            return np.zeros(0, dtype=np.float64), {
                "success": True, "iterations": 0, "maximum_capacity_excess": 0.0
            }
        h_matrices = [factor.hamiltonian.detach().cpu().numpy() for factor in h_factors]
        h_capacity = [factor.capacity.detach().cpu().numpy() for factor in h_factors]
        hh_energy = [factor.energies.detach().cpu().numpy() for factor in heavy_factors]
        hh_capacity = [factor.capacity.detach().cpu().numpy() for factor in heavy_factors]
        capacity_np = capacity.detach().cpu().numpy().astype(np.float64, copy=False)
        tau = self.unified_capacity_temperature
        h_tau = self.unified_h_regularisation_temperature

        def value_gradient(lam):
            value = -float(np.dot(lam, capacity_np))
            usage = np.zeros_like(lam)
            for matrix, loads in zip(h_matrices, h_capacity):
                shifted = matrix + np.diag(loads @ lam)
                eigenvalues, eigenvectors = np.linalg.eigh(shifted)
                scaled = -eigenvalues / h_tau
                maximum = float(np.max(scaled))
                weights = np.exp(scaled - maximum)
                eigen_probability = weights / np.sum(weights)
                probability = np.square(eigenvectors) @ eigen_probability
                value += -h_tau * (maximum + np.log(np.sum(weights)))
                usage += probability @ loads
            for energies, loads in zip(hh_energy, hh_capacity):
                shifted = energies + loads @ lam
                scaled = -shifted / tau
                maximum = float(np.max(scaled))
                weights = np.exp(scaled - maximum)
                probability = weights / np.sum(weights)
                value += -tau * (maximum + np.log(np.sum(weights)))
                usage += probability @ loads
            return value, usage - capacity_np

        def objective(lam):
            value, gradient = value_gradient(lam)
            return -value, -gradient

        start = (
            np.asarray(initial, dtype=np.float64)
            if initial is not None and len(initial) == count
            else np.zeros(count, dtype=np.float64)
        )
        start = np.maximum(start, 0.0)
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * count,
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 1000},
        )
        attempts = [("L-BFGS-B", result)]

        def kkt_error(candidate):
            _, gradient = value_gradient(candidate)
            active = candidate > 1e-7
            active_error = np.max(np.abs(gradient[active]), initial=0.0)
            boundary_error = np.max(np.maximum(gradient[~active], 0.0), initial=0.0)
            return max(float(active_error), float(boundary_error))

        if (not result.success) or kkt_error(result.x) > 2e-7:
            result = minimize(
                objective,
                np.maximum(result.x, 0.0),
                jac=True,
                method="SLSQP",
                bounds=[(0.0, None)] * count,
                options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
            )
            attempts.append(("SLSQP", result))
        if (not result.success) or kkt_error(result.x) > 2e-7:
            result = minimize(
                lambda value: objective(value)[0],
                np.maximum(result.x, 0.0),
                method="Powell",
                bounds=[(0.0, None)] * count,
                options={"xtol": 1e-11, "ftol": 1e-13, "maxiter": 4000},
            )
            attempts.append(("Powell", result))
        final_kkt = kkt_error(result.x)
        if (not result.success) or final_kkt > 1e-5:
            history = "; ".join(
                f"{method}: success={attempt.success}, message={attempt.message}"
                for method, attempt in attempts
            )
            raise RuntimeError(
                f"unified capacity dual failed KKT={final_kkt:.3e}; {history}"
            )
        _, final_gradient = value_gradient(result.x)
        complementarity_excess = np.maximum(final_gradient, 0.0)
        return np.asarray(result.x, dtype=np.float64), {
            "success": True,
            "method": attempts[-1][0],
            "iterations": int(result.nit),
            "kkt_error": final_kkt,
            "maximum_capacity_excess": float(complementarity_excess.max(initial=0.0)),
            "active_capacity_prices": int(np.count_nonzero(result.x > 1e-8)),
            "maximum_capacity_price_eV": float(result.x.max(initial=0.0)),
        }

    def _evaluate_factors(self, h_factors, heavy_factors, lam):
        value = lam.sum() * 0.0
        h_probabilities = []
        hh_probabilities = []
        for factor in h_factors:
            factor_value, probability = _ground_state(
                factor.hamiltonian + torch.diag(factor.capacity @ lam),
                self.unified_h_regularisation_temperature,
            )
            value = value + factor_value
            h_probabilities.append(probability)
        for factor in heavy_factors:
            factor_value, probability = _discrete_free_energy(
                factor.energies + factor.capacity @ lam,
                self.unified_capacity_temperature,
            )
            value = value + factor_value
            hh_probabilities.append(probability)
        return value, h_probabilities, hh_probabilities

    def _unified_capacity_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("unified capacity model requires live intermediates")
        values = cached[1]
        taper = values["taper"]
        mask = values["mask"]
        neighbours = values["neighbours"]
        hydrogen = int(R.ELEMENT_INDEX["H"])
        active_numpy = (
            (taper.detach().cpu().numpy() > 1e-12)
            & self.neighbour_mask.detach().cpu().numpy()
        )
        neighbours_numpy = neighbours.detach().cpu().numpy()
        heavy_edges = self._heavy_edges(values)
        heavy_by_box = [[] for _ in range(self.box_count)]
        for edge in heavy_edges:
            heavy_by_box[edge.atoms[0] // self.per_box].append(edge)

        correction = torch.zeros_like(base_per_atom)
        all_indices = torch.arange(len(base_per_atom), device=self.device)
        membership = mask.clone()
        box_diagnostics = []
        attractive = attractive_magnitude(values)
        current_pair = taper * (values["repulsive"] - attractive)
        original_over = self._profile_energy_parts["over"]

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box
            heavy_atoms = tuple(
                atom for atom in range(start, stop)
                if int(self.types_numpy[atom]) != hydrogen
            )
            heavy_index = {atom: index for index, atom in enumerate(heavy_atoms)}
            capacity = torch.stack([
                self.valence[self.types[atom]] for atom in heavy_atoms
            ]) if heavy_atoms else taper.sum().reshape(1)[:0]

            edge_atoms, edge_rows, edge_slots = self._active_edges_for_box(
                box, values, neighbours_numpy, active_numpy
            )
            components = self._hydrogen_edge_components(edge_atoms) if edge_atoms else ()
            h_factors = []
            for component in components:
                h_factors.append(self._h_factor(
                    tuple(edge_atoms[index] for index in component),
                    tuple(edge_rows[index] for index in component),
                    tuple(edge_slots[index] for index in component),
                    values, heavy_index,
                ))

            heavy_factors = []
            common_repulsive = taper.sum() * 0.0
            for factor in h_factors:
                common_repulsive = common_repulsive + torch.stack([
                    taper[row, slot] * values["repulsive"][row, slot]
                    for row, slot in zip(factor.rows, factor.slots)
                ]).sum()
            for edge in heavy_by_box[box]:
                factor, repulsive = self._heavy_factor(edge, values, heavy_index)
                heavy_factors.append(factor)
                common_repulsive = common_repulsive + repulsive

            lam_np, solve_info = self._solve_dual(
                h_factors,
                heavy_factors,
                capacity,
                self._unified_lambda_cache.get(box),
            )
            self._unified_lambda_cache[box] = lam_np.copy()
            lam = torch.tensor(lam_np, device=self.device, dtype=self.dtype)
            factor_value, h_probability, hh_probability = self._evaluate_factors(
                h_factors, heavy_factors, lam
            )
            unified_pair = common_repulsive + factor_value - torch.sum(lam * capacity)

            # Unique base pair value for this box, used only by the research
            # adapter to replace the old radial representation algebraically.
            pair_terms = [current_pair[row, slot] for factor in h_factors
                          for row, slot in zip(factor.rows, factor.slots)]
            pair_terms.extend(
                torch.stack([current_pair[row, slot] for row, slot in factor.directed_slots]).mean()
                for factor in heavy_factors
            )
            base_pair = torch.stack(pair_terms).sum() if pair_terms else taper.sum() * 0.0
            delta = unified_pair - base_pair - original_over[start:stop].sum()
            anchor = start
            correction = correction + (all_indices == anchor).to(self.dtype) * delta

            # Expected topology is retained for the topology experiment and
            # diagnostics.  It is not used by this radial-control class.
            for factor, probability in zip(h_factors, h_probability):
                edge_occupancy = []
                for edge_index in range(len(factor.atoms)):
                    edge_occupancy.append(torch.stack([
                        probability[state_index]
                        for state_index, state in enumerate(factor.states)
                        if edge_index in state
                    ]).sum() if any(edge_index in state for state in factor.states)
                    else probability.sum() * 0.0)
                for (first, second), occupancy in zip(factor.atoms, edge_occupancy):
                    heavy_atom = first if int(self.types_numpy[first]) != hydrogen else second
                    other = second if heavy_atom == first else first
                    slots = torch.nonzero(
                        (neighbours[heavy_atom] == other) & (mask[heavy_atom] > 0),
                        as_tuple=False,
                    ).flatten()
                    if slots.numel():
                        membership[heavy_atom, slots[0]] = occupancy
            heavy_bond_orders = []
            for factor, probability in zip(heavy_factors, hh_probability):
                occupancy = 1.0 - probability[0]
                expected_order = torch.sum(
                    probability
                    * torch.arange(
                        probability.numel(), device=self.device, dtype=self.dtype
                    )
                )
                for atom, slot in factor.directed_slots:
                    membership[atom, slot] = occupancy
                heavy_bond_orders.append({
                    "atoms": factor.atoms,
                    "expected_order": float(expected_order.detach().cpu()),
                    "bond_probability": float(occupancy.detach().cpu()),
                })

            usage = torch.zeros_like(capacity)
            for factor, probability in zip(h_factors, h_probability):
                usage = usage + probability @ factor.capacity
            for factor, probability in zip(heavy_factors, hh_probability):
                usage = usage + probability @ factor.capacity
            box_diagnostics.append({
                "box": box,
                "h_factors": len(h_factors),
                "heavy_factors": len(heavy_factors),
                "largest_h_states": max((len(f.states) for f in h_factors), default=0),
                "capacity": capacity.detach().cpu().tolist(),
                "usage": usage.detach().cpu().tolist(),
                "lambda_eV": lam.detach().cpu().tolist(),
                "solver": solve_info,
                "unified_pair_eV": float(unified_pair.detach().cpu()),
                "base_pair_eV": float(base_pair.detach().cpu()),
                "removed_over_eV": float(original_over[start:stop].sum().detach().cpu()),
                "heavy_bond_orders": heavy_bond_orders,
            })

        self._unified_membership = membership
        self._unified_diagnostics = {
            "formulation": "dual_factorised_unified_bond_capacity",
            "boxes": tuple(box_diagnostics),
            "topology_source": "established_heavy_valence",
        }
        return correction


class UnifiedBondCapacityTopologyPrototype(UnifiedBondCapacityEnergyPrototype):
    """Experiment feeding unified expected occupancies into angle topology."""

    physics_model_name = "research_unified_bond_capacity_topology_v0"

    def _local_valence_membership(self, values):
        if self._unified_membership is None:
            raise RuntimeError("unified topology requested before capacity solve")
        if self._unified_diagnostics is not None:
            self._unified_diagnostics["topology_source"] = "unified_expected_occupancy"
        return self._unified_membership
