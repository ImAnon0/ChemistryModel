"""Variational bond-capacity and geometry research prototypes.

The implementation is deliberately float64/SciPy reference code.  It is not
registered in production selection and changes no production equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import torch
from scipy.optimize import minimize

import reactive as R
from batched_torch import BatchedReactiveSimulation
from research.unified_bond_capacity import (
    UnifiedBondCapacityEnergyPrototype,
    UnifiedBondCapacityTopologyPrototype,
)


@dataclass
class _GeometryProblem:
    heavy_atoms: tuple[int, ...]
    heavy_index: dict[int, int]
    capacity: torch.Tensor
    h_factors: tuple
    heavy_factors: tuple
    h_slices: tuple[slice, ...]
    heavy_slices: tuple[slice, ...]
    factor_slices: tuple[slice, ...]
    edge_atoms: tuple[tuple[int, int], ...]
    edge_taper: torch.Tensor
    bond_matrix: torch.Tensor
    order_matrix: torch.Tensor
    capacity_matrix: torch.Tensor
    common_repulsive: torch.Tensor
    base_pair: torch.Tensor
    base_over: torch.Tensor
    base_angle: torch.Tensor


@dataclass
class _LocalGeometryFactor:
    atom: int
    incident_edges: tuple[int, ...]
    states: tuple[tuple[int, ...], ...]
    variable_slice: slice


class PostSolvedWeightedGeometryPrototype(UnifiedBondCapacityTopologyPrototype):
    """Control: radial solve first, then weight old angles by occupancy."""

    physics_model_name = "research_post_solved_weighted_geometry_v0"
    research_only = True


class VariationalWeightedGeometryPrototype(UnifiedBondCapacityEnergyPrototype):
    """Jointly minimize factor probabilities and occupancy-weighted geometry."""

    physics_model_name = "research_variational_weighted_geometry_v0"
    physics_model_revision = 0
    research_only = True
    geometry_mode = "continuous_domain"

    def __init__(self, *args, geometry_state_temperature=0.01, **kwargs):
        self.geometry_state_temperature = float(geometry_state_temperature)
        if self.geometry_state_temperature <= 0.0:
            raise ValueError("geometry_state_temperature must be positive")
        self._geometry_solver_cache = {}
        self._geometry_diagnostics = None
        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        base = BatchedReactiveSimulation.energy_per_atom(self, positions)
        try:
            correction = self._variational_geometry_correction(positions, base)
        finally:
            self._reactive_intermediates = None
        return base + correction

    def _build_problem(self, box, positions, values, heavy_edges, active_numpy):
        start = box * self.per_box
        stop = start + self.per_box
        hydrogen = int(R.ELEMENT_INDEX["H"])
        heavy_atoms = tuple(
            atom for atom in range(start, stop)
            if int(self.types_numpy[atom]) != hydrogen
        )
        heavy_index = {atom: index for index, atom in enumerate(heavy_atoms)}
        capacity = torch.stack([
            self.valence[self.types[atom]] for atom in heavy_atoms
        ]) if heavy_atoms else values["taper"].sum().reshape(1)[:0]
        neighbours_numpy = values["neighbours"].detach().cpu().numpy()
        edge_atoms, edge_rows, edge_slots = self._active_edges_for_box(
            box, values, neighbours_numpy, active_numpy
        )
        components = self._hydrogen_edge_components(edge_atoms) if edge_atoms else ()
        h_factors = tuple(self._h_factor(
            tuple(edge_atoms[index] for index in component),
            tuple(edge_rows[index] for index in component),
            tuple(edge_slots[index] for index in component),
            values, heavy_index,
        ) for component in components)
        local_heavy_edges = tuple(
            edge for edge in heavy_edges if start <= edge.atoms[0] < stop
        )
        heavy_factors = []
        common_repulsive = values["taper"].sum() * 0.0
        for factor in h_factors:
            common_repulsive = common_repulsive + torch.stack([
                values["taper"][row, slot] * values["repulsive"][row, slot]
                for row, slot in zip(factor.rows, factor.slots)
            ]).sum()
        for edge in local_heavy_edges:
            factor, repulsive = self._heavy_factor(edge, values, heavy_index)
            heavy_factors.append(factor)
            common_repulsive = common_repulsive + repulsive
        heavy_factors = tuple(heavy_factors)

        offset = 0
        h_slices = []
        heavy_slices = []
        for factor in h_factors:
            h_slices.append(slice(offset, offset + len(factor.states)))
            offset += len(factor.states)
        for factor in heavy_factors:
            heavy_slices.append(slice(offset, offset + factor.energies.numel()))
            offset += factor.energies.numel()
        factor_slices = tuple((*h_slices, *heavy_slices))

        edge_map = {}
        edge_atoms_all = []
        edge_tapers = []
        for factor in h_factors:
            for atoms, row, slot in zip(factor.atoms, factor.rows, factor.slots):
                key = tuple(sorted(atoms))
                if key not in edge_map:
                    edge_map[key] = len(edge_atoms_all)
                    edge_atoms_all.append(key)
                    edge_tapers.append(values["taper"][row, slot])
        for factor in heavy_factors:
            key = tuple(sorted(factor.atoms))
            if key not in edge_map:
                edge_map[key] = len(edge_atoms_all)
                edge_atoms_all.append(key)
                edge_tapers.append(torch.stack([
                    values["taper"][row, slot]
                    for row, slot in factor.directed_slots
                ]).mean())

        edge_count = len(edge_atoms_all)
        bond_matrix = torch.zeros(
            (edge_count, offset), device=self.device, dtype=self.dtype
        )
        order_matrix = torch.zeros_like(bond_matrix)
        for factor, variable_slice in zip(h_factors, h_slices):
            for state_index, state in enumerate(factor.states):
                for local_edge in state:
                    edge = edge_map[tuple(sorted(factor.atoms[local_edge]))]
                    bond_matrix[edge, variable_slice.start + state_index] = 1.0
                    order_matrix[edge, variable_slice.start + state_index] = 1.0
        for factor, variable_slice in zip(heavy_factors, heavy_slices):
            edge = edge_map[tuple(sorted(factor.atoms))]
            for state in range(factor.energies.numel()):
                if state > 0:
                    bond_matrix[edge, variable_slice.start + state] = 1.0
                    order_matrix[edge, variable_slice.start + state] = float(state)

        capacity_blocks = [factor.capacity.T for factor in h_factors]
        capacity_blocks.extend(factor.capacity.T for factor in heavy_factors)
        capacity_matrix = (
            torch.cat(capacity_blocks, dim=1)
            if capacity_blocks else torch.zeros(
                (len(heavy_atoms), 0), device=self.device, dtype=self.dtype
            )
        )
        current_pair = values["taper"] * (
            values["repulsive"]
            - values.get("state_attractive", 2.0 * values["pair_depth"] * torch.exp(
                -values["pair_width"] * values["shift"]
            ))
        )
        pair_terms = [
            current_pair[row, slot]
            for factor in h_factors
            for row, slot in zip(factor.rows, factor.slots)
        ]
        pair_terms.extend(
            torch.stack([
                current_pair[row, slot] for row, slot in factor.directed_slots
            ]).mean()
            for factor in heavy_factors
        )
        zero = values["taper"].sum() * 0.0
        base_pair = torch.stack(pair_terms).sum() if pair_terms else zero
        return _GeometryProblem(
            heavy_atoms=heavy_atoms,
            heavy_index=heavy_index,
            capacity=capacity,
            h_factors=h_factors,
            heavy_factors=heavy_factors,
            h_slices=tuple(h_slices),
            heavy_slices=tuple(heavy_slices),
            factor_slices=factor_slices,
            edge_atoms=tuple(edge_atoms_all),
            edge_taper=torch.stack(edge_tapers) if edge_tapers else zero.reshape(1)[:0],
            bond_matrix=bond_matrix,
            order_matrix=order_matrix,
            capacity_matrix=capacity_matrix,
            common_repulsive=common_repulsive,
            base_pair=base_pair,
            base_over=self._profile_energy_parts["over"][start:stop].sum(),
            base_angle=self._profile_energy_parts["angle"][start:stop].sum(),
        )

    def _angle_for_edges(self, positions, centre, first_edge, second_edge, problem):
        first_atoms = problem.edge_atoms[first_edge]
        second_atoms = problem.edge_atoms[second_edge]
        first = first_atoms[1] if first_atoms[0] == centre else first_atoms[0]
        second = second_atoms[1] if second_atoms[0] == centre else second_atoms[0]
        first_vector = positions[first] - positions[centre]
        second_vector = positions[second] - positions[centre]
        first_vector = first_vector - self.box_size * torch.round(first_vector / self.box_size)
        second_vector = second_vector - self.box_size * torch.round(second_vector / self.box_size)
        cosine = torch.sum(first_vector * second_vector) / torch.clamp(
            torch.linalg.vector_norm(first_vector)
            * torch.linalg.vector_norm(second_vector), min=1e-12
        )
        return torch.acos(torch.clamp(cosine, -1.0 + 1e-9, 1.0 - 1e-9))

    def _continuous_geometry_energy(self, x, problem, positions, detach_inputs):
        bond_matrix = problem.bond_matrix.detach() if detach_inputs else problem.bond_matrix
        order_matrix = problem.order_matrix.detach() if detach_inputs else problem.order_matrix
        edge_taper = problem.edge_taper.detach() if detach_inputs else problem.edge_taper
        live_positions = positions.detach() if detach_inputs else positions
        bond = bond_matrix @ x
        order = order_matrix @ x
        total = x.sum() * 0.0
        hydrogen = int(R.ELEMENT_INDEX["H"])
        for atom in problem.heavy_atoms:
            incident = [
                index for index, atoms in enumerate(problem.edge_atoms)
                if atom in atoms
            ]
            if len(incident) < 2:
                continue
            index = torch.tensor(incident, device=self.device, dtype=torch.long)
            weights = edge_taper[index] * bond[index]
            bonded_order = torch.sum(edge_taper[index] * order[index])
            outer = self.outer_electrons[self.types[atom]]
            lone_pairs = torch.clamp((outer - bonded_order) / 2.0, min=0.0)
            if self.geometry_mode == "continuous_domain":
                steric = torch.clamp(torch.sum(weights) + lone_pairs, 2.0, 4.0)
                base_angle = torch.where(
                    steric < 3.0,
                    180.0 + (120.0 - 180.0) * (steric - 2.0),
                    120.0 + (109.47 - 120.0) * (steric - 3.0),
                )
                rest_angles = torch.deg2rad(
                    base_angle - self.lone_pair_squeeze * lone_pairs
                ).reshape(1)
            else:
                rest_angles = torch.deg2rad(
                    torch.tensor([180.0, 120.0, 109.47], device=self.device, dtype=self.dtype)
                    - self.lone_pair_squeeze * lone_pairs
                )
            domain_energies = []
            for rest in rest_angles:
                domain = x.sum() * 0.0
                for left in range(len(incident)):
                    for right in range(left + 1, len(incident)):
                        first_weight = weights[left]
                        second_weight = weights[right]
                        difference = first_weight - second_weight
                        weaker = 0.5 * (
                            first_weight + second_weight
                            - torch.sqrt(difference.square() + 1e-8) + 1e-4
                        )
                        directionality = torch.clamp(
                            0.5 * lone_pairs, 0.0, 1.0
                        )
                        engagement = weaker + (1.0 - weaker) * directionality
                        pair_weight = first_weight * second_weight * engagement
                        angle = self._angle_for_edges(
                            live_positions, atom, incident[left], incident[right], problem
                        )
                        domain = domain + (
                            0.5 * self.angle_stiffness[self.types[atom]]
                            * pair_weight * (angle - rest).square()
                        )
                domain_energies.append(domain)
            states = torch.stack(domain_energies)
            if self.geometry_mode == "electron_domain_free_energy":
                tau = self.geometry_state_temperature
                total = total + (
                    -tau * torch.logsumexp(-states / tau, dim=0)
                    + tau * np.log(3.0)
                )
            else:
                total = total + states[0]
        return total

    def _factor_geometry_free_energy(self, x, problem, positions, detach_inputs=False):
        total = x.sum() * 0.0
        for factor, variable_slice in zip(problem.h_factors, problem.h_slices):
            probabilities = x[variable_slice]
            matrix = factor.hamiltonian.detach() if detach_inputs else factor.hamiltonian
            amplitude = torch.sqrt(torch.clamp(probabilities, min=1e-30))
            total = total + amplitude @ matrix @ amplitude
        for factor, variable_slice in zip(problem.heavy_factors, problem.heavy_slices):
            probabilities = x[variable_slice]
            energies = factor.energies.detach() if detach_inputs else factor.energies
            entropy = probabilities * torch.log(torch.clamp(probabilities, min=1e-30))
            total = total + torch.sum(
                probabilities * energies
                + self.unified_capacity_temperature * entropy
            )
        return total + self._continuous_geometry_energy(
            x, problem, positions, detach_inputs
        )

    def _solve_geometry(self, box, problem, positions):
        count = problem.capacity_matrix.shape[1]
        if count == 0:
            return np.zeros(0), np.zeros(len(problem.heavy_atoms)), {
                "success": True, "iterations": 0, "capacity_violation": 0.0,
                "simplex_violation": 0.0,
            }
        lam_np, _ = self._solve_dual(
            problem.h_factors, problem.heavy_factors, problem.capacity,
            self._unified_lambda_cache.get(box),
        )
        lam = torch.tensor(lam_np, device=self.device, dtype=self.dtype)
        _, h_probability, heavy_probability = self._evaluate_factors(
            problem.h_factors, problem.heavy_factors, lam
        )
        initial = np.concatenate([
            probability.detach().cpu().numpy()
            for probability in (*h_probability, *heavy_probability)
        ])
        cached = self._geometry_solver_cache.get(box)
        if cached is not None and len(cached) == count:
            initial = cached.copy()
        lower = 1e-12
        initial = np.clip(initial, lower, 1.0)
        for variable_slice in problem.factor_slices:
            initial[variable_slice] /= np.sum(initial[variable_slice])

        equality = np.zeros((len(problem.factor_slices), count), dtype=np.float64)
        for row, variable_slice in enumerate(problem.factor_slices):
            equality[row, variable_slice] = 1.0
        capacity_matrix = problem.capacity_matrix.detach().cpu().numpy()
        capacity = problem.capacity.detach().cpu().numpy()

        def objective(value):
            tensor = torch.tensor(
                value, device=self.device, dtype=self.dtype, requires_grad=True
            )
            energy = self._factor_geometry_free_energy(
                tensor, problem, positions, detach_inputs=True
            )
            gradient = torch.autograd.grad(energy, tensor)[0]
            return float(energy.detach().cpu()), gradient.detach().cpu().numpy()

        result = minimize(
            objective, initial, jac=True, method="SLSQP",
            bounds=[(lower, 1.0)] * count,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda value: equality @ value - 1.0,
                    "jac": lambda value: equality,
                },
                {
                    "type": "ineq",
                    "fun": lambda value: capacity - capacity_matrix @ value,
                    "jac": lambda value: -capacity_matrix,
                },
            ],
            options={"ftol": 1e-11, "maxiter": 2000, "disp": False},
        )
        simplex_violation = float(np.max(np.abs(equality @ result.x - 1.0), initial=0.0))
        capacity_violation = float(np.max(
            np.maximum(capacity_matrix @ result.x - capacity, 0.0), initial=0.0
        ))
        if (
            not result.success
            or simplex_violation > 2e-7
            or capacity_violation > 2e-7
        ):
            raise RuntimeError(
                "variational geometry solve failed: "
                f"{result.message}; simplex={simplex_violation:.3e}, "
                f"capacity={capacity_violation:.3e}"
            )
        self._geometry_solver_cache[box] = result.x.copy()
        multipliers = np.asarray(result.multipliers, dtype=np.float64)
        capacity_multipliers = multipliers[len(problem.factor_slices):]
        return result.x, capacity_multipliers, {
            "success": True,
            "iterations": int(result.nit),
            "simplex_violation": simplex_violation,
            "capacity_violation": capacity_violation,
            "active_capacity_constraints": int(np.count_nonzero(
                capacity - capacity_matrix @ result.x < 2e-7
            )),
        }

    def _variational_geometry_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("variational geometry requires live intermediates")
        values = cached[1]
        active_numpy = (
            (values["taper"].detach().cpu().numpy() > 1e-12)
            & self.neighbour_mask.detach().cpu().numpy()
        )
        heavy_edges = self._heavy_edges(values)
        correction = torch.zeros_like(base_per_atom)
        all_indices = torch.arange(len(base_per_atom), device=self.device)
        diagnostics = []
        for box in range(self.box_count):
            problem = self._build_problem(
                box, positions, values, heavy_edges, active_numpy
            )
            x_np, multipliers_np, solver = self._solve_geometry(
                box, problem, positions
            )
            x = torch.tensor(x_np, device=self.device, dtype=self.dtype)
            factor_geometry = self._factor_geometry_free_energy(
                x, problem, positions, detach_inputs=False
            )
            residual = problem.capacity_matrix @ x - problem.capacity
            multipliers = torch.tensor(
                multipliers_np, device=self.device, dtype=self.dtype
            )
            envelope = torch.sum(
                multipliers * (residual - residual.detach())
            )
            desired = problem.common_repulsive + factor_geometry + envelope
            delta = desired - problem.base_pair - problem.base_over - problem.base_angle
            anchor = box * self.per_box
            correction = correction + (all_indices == anchor).to(self.dtype) * delta
            bond = problem.bond_matrix @ x
            order = problem.order_matrix @ x
            diagnostics.append({
                "box": box,
                "mode": self.geometry_mode,
                "solver": solver,
                "factor_variables": int(x.numel()),
                "h_factors": len(problem.h_factors),
                "heavy_factors": len(problem.heavy_factors),
                "edge_atoms": problem.edge_atoms,
                "bond_participation": bond.detach().cpu().tolist(),
                "bond_order": order.detach().cpu().tolist(),
                "desired_energy_eV": float(desired.detach().cpu()),
            })
        self._geometry_diagnostics = {
            "formulation": "joint_variational_bond_geometry",
            "mode": self.geometry_mode,
            "boxes": tuple(diagnostics),
        }
        return correction


class VariationalElectronDomainGeometryPrototype(VariationalWeightedGeometryPrototype):
    """Joint bond solve with a local 2/3/4-domain geometry free energy."""

    physics_model_name = "research_variational_electron_domain_geometry_v0"
    geometry_mode = "electron_domain_free_energy"


class VariationalJointGeometryStatePrototype(VariationalWeightedGeometryPrototype):
    """Local joint bond-order/domain states with exact marginal consistency."""

    physics_model_name = "research_variational_joint_geometry_state_v0"
    geometry_mode = "joint_local_geometry_state"
    max_local_geometry_states = 20_000

    def __init__(self, *args, local_state_regularisation=1e-4, **kwargs):
        self.local_state_regularisation = float(local_state_regularisation)
        if self.local_state_regularisation <= 0.0:
            raise ValueError("local_state_regularisation must be positive")
        super().__init__(*args, **kwargs)

    def _radial_order_indicators(self, problem):
        radial_count = problem.capacity_matrix.shape[1]
        mutable = []
        for edge in range(len(problem.edge_atoms)):
            maximum = int(round(float(problem.order_matrix[edge].max().detach().cpu())))
            mutable.append([
                torch.zeros(radial_count, device=self.device, dtype=self.dtype)
                for _ in range(maximum + 1)
            ])
        edge_lookup = {
            tuple(sorted(atoms)): index
            for index, atoms in enumerate(problem.edge_atoms)
        }
        for factor, variable_slice in zip(problem.h_factors, problem.h_slices):
            for state_index, state in enumerate(factor.states):
                selected = set(state)
                column = variable_slice.start + state_index
                for local_edge, atoms in enumerate(factor.atoms):
                    edge = edge_lookup[tuple(sorted(atoms))]
                    order = int(local_edge in selected)
                    mutable[edge][order][column] = 1.0
        for factor, variable_slice in zip(
            problem.heavy_factors, problem.heavy_slices
        ):
            edge = edge_lookup[tuple(sorted(factor.atoms))]
            for order in range(factor.energies.numel()):
                mutable[edge][order][variable_slice.start + order] = 1.0
        return tuple(tuple(rows) for rows in mutable)

    def _build_joint_factors(self, problem):
        radial_count = problem.capacity_matrix.shape[1]
        offset = radial_count
        factors = []
        for atom in problem.heavy_atoms:
            incident = tuple(
                index for index, atoms in enumerate(problem.edge_atoms)
                if atom in atoms
            )
            if not incident:
                continue
            maxima = [
                int(round(float(problem.order_matrix[edge].max().detach().cpu())))
                for edge in incident
            ]
            states = tuple(
                tuple(int(value) for value in state)
                for state in itertools.product(*[
                    range(maximum + 1) for maximum in maxima
                ])
            )
            if len(states) > self.max_local_geometry_states:
                raise RuntimeError(
                    f"local geometry factor at atom {atom} has {len(states)} states; "
                    f"limit is {self.max_local_geometry_states}"
                )
            factors.append(_LocalGeometryFactor(
                atom=atom,
                incident_edges=incident,
                states=states,
                variable_slice=slice(offset, offset + len(states)),
            ))
            offset += len(states)
        return tuple(factors), offset

    def _local_state_energy(self, factor, state, problem, positions, detach_inputs):
        edge_taper = problem.edge_taper.detach() if detach_inputs else problem.edge_taper
        live_positions = positions.detach() if detach_inputs else positions
        orders = torch.tensor(state, device=self.device, dtype=self.dtype)
        incident = torch.tensor(
            factor.incident_edges, device=self.device, dtype=torch.long
        )
        participation = (orders > 0.0).to(self.dtype)
        weights = edge_taper[incident] * participation
        bonded_order = torch.sum(edge_taper[incident] * orders)
        outer = self.outer_electrons[self.types[factor.atom]]
        lone_pairs = torch.clamp((outer - bonded_order) / 2.0, min=0.0)
        steric = torch.clamp(torch.sum(weights) + lone_pairs, 2.0, 4.0)
        base_angle = torch.where(
            steric < 3.0,
            180.0 + (120.0 - 180.0) * (steric - 2.0),
            120.0 + (109.47 - 120.0) * (steric - 3.0),
        )
        rest = torch.deg2rad(
            base_angle - self.lone_pair_squeeze * lone_pairs
        )
        energy = weights.sum() * 0.0
        for left in range(len(factor.incident_edges)):
            if state[left] == 0:
                continue
            for right in range(left + 1, len(factor.incident_edges)):
                if state[right] == 0:
                    continue
                first_weight = weights[left]
                second_weight = weights[right]
                difference = first_weight - second_weight
                weaker = 0.5 * (
                    first_weight + second_weight
                    - torch.sqrt(difference.square() + 1e-8) + 1e-4
                )
                directionality = torch.clamp(0.5 * lone_pairs, 0.0, 1.0)
                engagement = weaker + (1.0 - weaker) * directionality
                pair_weight = first_weight * second_weight * engagement
                angle = self._angle_for_edges(
                    live_positions,
                    factor.atom,
                    factor.incident_edges[left],
                    factor.incident_edges[right],
                    problem,
                )
                energy = energy + (
                    0.5 * self.angle_stiffness[self.types[factor.atom]]
                    * pair_weight * (angle - rest).square()
                )
        return energy

    def _joint_free_energy(
        self, x, problem, local_factors, positions, detach_inputs=False
    ):
        radial_count = problem.capacity_matrix.shape[1]
        radial = x[:radial_count]
        # Reuse only the H/heavy radial part from the parent; its geometry
        # method is intentionally bypassed here.
        total = radial.sum() * 0.0
        for factor, variable_slice in zip(problem.h_factors, problem.h_slices):
            probabilities = radial[variable_slice]
            matrix = factor.hamiltonian.detach() if detach_inputs else factor.hamiltonian
            amplitude = torch.sqrt(torch.clamp(probabilities, min=1e-30))
            total = total + amplitude @ matrix @ amplitude
        for factor, variable_slice in zip(problem.heavy_factors, problem.heavy_slices):
            probabilities = radial[variable_slice]
            energies = factor.energies.detach() if detach_inputs else factor.energies
            total = total + torch.sum(
                probabilities * energies
                + self.unified_capacity_temperature
                * probabilities * torch.log(torch.clamp(probabilities, min=1e-30))
            )
        for factor in local_factors:
            probabilities = x[factor.variable_slice]
            state_energies = torch.stack([
                self._local_state_energy(
                    factor, state, problem, positions, detach_inputs
                )
                for state in factor.states
            ])
            total = total + torch.sum(
                probabilities * state_energies
                + self.local_state_regularisation
                * probabilities * torch.log(torch.clamp(probabilities, min=1e-30))
            )
        return total

    def _joint_constraints(self, problem, local_factors, order_indicators, total_count):
        rows = []
        targets = []
        labels = []
        for variable_slice in problem.factor_slices:
            row = np.zeros(total_count)
            row[variable_slice] = 1.0
            rows.append(row)
            targets.append(1.0)
            labels.append("radial_normalisation")
        for factor in local_factors:
            row = np.zeros(total_count)
            row[factor.variable_slice] = 1.0
            rows.append(row)
            targets.append(1.0)
            labels.append("local_normalisation")
            for local_edge, edge in enumerate(factor.incident_edges):
                for order in range(1, len(order_indicators[edge])):
                    row = np.zeros(total_count)
                    for state_index, state in enumerate(factor.states):
                        if state[local_edge] == order:
                            row[factor.variable_slice.start + state_index] = 1.0
                    row[:problem.capacity_matrix.shape[1]] -= (
                        order_indicators[edge][order].detach().cpu().numpy()
                    )
                    rows.append(row)
                    targets.append(0.0)
                    labels.append("marginal_consistency")
        return np.asarray(rows), np.asarray(targets), tuple(labels)

    def _joint_initial(self, problem, local_factors, order_indicators):
        lam_np, _ = self._solve_dual(
            problem.h_factors, problem.heavy_factors, problem.capacity
        )
        lam = torch.tensor(lam_np, device=self.device, dtype=self.dtype)
        _, h_probability, heavy_probability = self._evaluate_factors(
            problem.h_factors, problem.heavy_factors, lam
        )
        radial = np.concatenate([
            value.detach().cpu().numpy()
            for value in (*h_probability, *heavy_probability)
        ])
        radial = np.maximum(radial, 0.0)
        for variable_slice in problem.factor_slices:
            radial[variable_slice] /= np.sum(radial[variable_slice])
        values = [radial]
        for factor in local_factors:
            marginals = []
            for local_edge, edge in enumerate(factor.incident_edges):
                marginal = np.asarray([
                    float(
                        order_indicators[edge][order].detach().cpu().numpy() @ radial
                    )
                    for order in range(len(order_indicators[edge]))
                ])
                marginal = np.maximum(marginal, 0.0)
                marginal /= np.sum(marginal)
                marginals.append(marginal)
            local = np.asarray([
                np.prod([
                    marginals[edge][order]
                    for edge, order in enumerate(state)
                ])
                for state in factor.states
            ])
            local /= np.sum(local)
            values.append(local)
        return np.concatenate(values)

    def _solve_joint(self, box, problem, positions):
        local_factors, total_count = self._build_joint_factors(problem)
        order_indicators = self._radial_order_indicators(problem)
        equality, equality_target, labels = self._joint_constraints(
            problem, local_factors, order_indicators, total_count
        )
        initial = self._joint_initial(problem, local_factors, order_indicators)
        cached = self._geometry_solver_cache.get(box)
        if cached is not None and len(cached) == total_count:
            initial = cached.copy()
        lower = 0.0
        initial = np.clip(initial, lower, 1.0)

        def objective(value):
            tensor = torch.tensor(
                value, device=self.device, dtype=self.dtype, requires_grad=True
            )
            energy = self._joint_free_energy(
                tensor, problem, local_factors, positions, detach_inputs=True
            )
            gradient = torch.autograd.grad(energy, tensor)[0]
            return float(energy.detach().cpu()), gradient.detach().cpu().numpy()

        radial_count = problem.capacity_matrix.shape[1]
        capacity_matrix = np.zeros((len(problem.heavy_atoms), total_count))
        capacity_matrix[:, :radial_count] = (
            problem.capacity_matrix.detach().cpu().numpy()
        )
        capacity = problem.capacity.detach().cpu().numpy()
        result = minimize(
            objective, initial, jac=True, method="SLSQP",
            bounds=[(lower, 1.0)] * total_count,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda value: equality @ value - equality_target,
                    "jac": lambda value: equality,
                },
                {
                    "type": "ineq",
                    "fun": lambda value: capacity - capacity_matrix @ value,
                    "jac": lambda value: -capacity_matrix,
                },
            ],
            options={"ftol": 1e-11, "maxiter": 3000, "disp": False},
        )
        violation = float(np.max(
            np.abs(equality @ result.x - equality_target), initial=0.0
        ))
        capacity_violation = float(np.max(
            np.maximum(capacity_matrix @ result.x - capacity, 0.0), initial=0.0
        ))
        if not result.success or violation > 2e-7 or capacity_violation > 2e-7:
            raise RuntimeError(
                f"joint geometry solve failed: {result.message}; "
                f"equality={violation:.3e}, capacity={capacity_violation:.3e}"
            )
        self._geometry_solver_cache[box] = result.x.copy()
        multipliers = np.asarray(result.multipliers, dtype=np.float64)
        capacity_multipliers = multipliers[len(equality):]
        return result.x, local_factors, capacity_multipliers, {
            "success": True,
            "iterations": int(result.nit),
            "equality_violation": violation,
            "capacity_violation": capacity_violation,
            "local_factors": len(local_factors),
            "local_states": sum(len(factor.states) for factor in local_factors),
            "variables": total_count,
            "constraints": len(labels),
        }

    def _variational_geometry_correction(self, positions, base_per_atom):
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None or cached[0] is not positions:
            raise RuntimeError("joint geometry requires live intermediates")
        values = cached[1]
        active_numpy = (
            (values["taper"].detach().cpu().numpy() > 1e-12)
            & self.neighbour_mask.detach().cpu().numpy()
        )
        heavy_edges = self._heavy_edges(values)
        correction = torch.zeros_like(base_per_atom)
        all_indices = torch.arange(len(base_per_atom), device=self.device)
        diagnostics = []
        for box in range(self.box_count):
            problem = self._build_problem(
                box, positions, values, heavy_edges, active_numpy
            )
            x_np, local_factors, multipliers_np, solver = self._solve_joint(
                box, problem, positions
            )
            x = torch.tensor(x_np, device=self.device, dtype=self.dtype)
            free_energy = self._joint_free_energy(
                x, problem, local_factors, positions, detach_inputs=False
            )
            radial_count = problem.capacity_matrix.shape[1]
            residual = (
                problem.capacity_matrix @ x[:radial_count] - problem.capacity
            )
            multipliers = torch.tensor(
                multipliers_np, device=self.device, dtype=self.dtype
            )
            envelope = torch.sum(
                multipliers * (residual - residual.detach())
            )
            desired = problem.common_repulsive + free_energy + envelope
            delta = desired - problem.base_pair - problem.base_over - problem.base_angle
            anchor = box * self.per_box
            correction = correction + (all_indices == anchor).to(self.dtype) * delta
            diagnostics.append({
                "box": box,
                "mode": self.geometry_mode,
                "solver": solver,
                "edge_atoms": problem.edge_atoms,
                "bond_participation": (
                    problem.bond_matrix @ x[:radial_count]
                ).detach().cpu().tolist(),
                "bond_order": (
                    problem.order_matrix @ x[:radial_count]
                ).detach().cpu().tolist(),
                "desired_energy_eV": float(desired.detach().cpu()),
            })
        self._geometry_diagnostics = {
            "formulation": "joint_local_geometry_state",
            "mode": self.geometry_mode,
            "boxes": tuple(diagnostics),
        }
        return correction
