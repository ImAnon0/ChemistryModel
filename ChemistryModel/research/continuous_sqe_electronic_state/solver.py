"""Float64-friendly reference solver for continuous SQE C0."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .diagnostics import C0Diagnostics, build_diagnostics
from .parameters import C0ParameterSet


@dataclass(frozen=True)
class C0SolveResult:
    energy: torch.Tensor
    charges: torch.Tensor
    transfer_amplitudes: torch.Tensor
    scaled_transfer_variables: torch.Tensor
    edge_index: torch.Tensor
    compliance: torch.Tensor
    incidence: torch.Tensor
    hardness_matrix: torch.Tensor
    response_matrix: torch.Tensor
    electronegativity: torch.Tensor
    polarizability_A3: torch.Tensor
    diagnostics: C0Diagnostics


def _element_arrays(atomic_numbers, parameters, *, device, dtype):
    try:
        values = [parameters.elements[int(number)] for number in atomic_numbers]
    except KeyError as error:
        raise ValueError(f"unsupported atomic number in C0: {error.args[0]}") from error
    return tuple(
        torch.tensor(sequence, device=device, dtype=dtype)
        for sequence in (
            [value.electronegativity_eV for value in values],
            [value.intrinsic_hardness_eV for value in values],
            [value.gaussian_sigma_A for value in values],
            [value.transfer_capacity_e2_per_eV for value in values],
            [value.covalent_radius_A for value in values],
        )
    )


def _edge_index(atom_count, device):
    return torch.triu_indices(atom_count, atom_count, offset=1, device=device)


def _incidence(atom_count, edges, *, device, dtype):
    edge_count = edges.shape[1]
    incidence = torch.zeros((atom_count, edge_count), device=device, dtype=dtype)
    columns = torch.arange(edge_count, device=device)
    incidence[edges[0], columns] = 1.0
    incidence[edges[1], columns] = -1.0
    return incidence


def _smooth_compact_support(distances, inner, outer):
    coordinate = torch.clamp((distances - inner) / (outer - inner), 0.0, 1.0)
    smootherstep = coordinate**3 * (
        10.0 - 15.0 * coordinate + 6.0 * coordinate**2
    )
    return 1.0 - smootherstep


def _gaussian_coulomb_matrix(positions, sigma, coulomb_constant):
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    sigma_squared = sigma[:, None].square() + sigma[None, :].square()
    denominator = torch.sqrt(2.0 * sigma_squared)
    safe_distance = torch.clamp(distance, min=torch.finfo(positions.dtype).tiny)
    finite_distance = coulomb_constant * torch.erf(distance / denominator) / safe_distance
    coincident_limit = (
        coulomb_constant * math.sqrt(2.0 / math.pi) / torch.sqrt(sigma_squared)
    )
    return torch.where(distance > 0.0, finite_distance, coincident_limit)


def solve_c0(positions, atomic_numbers, parameters: C0ParameterSet):
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if len(atomic_numbers) != positions.shape[0]:
        raise ValueError("atomic number count does not match positions")
    if positions.shape[0] < 2:
        raise ValueError("C0 requires at least two atoms")

    device, dtype = positions.device, positions.dtype
    chi, intrinsic_hardness, sigma, elemental_capacity, covalent_radius = (
        _element_arrays(atomic_numbers, parameters, device=device, dtype=dtype)
    )
    edges = _edge_index(len(atomic_numbers), device)
    incidence = _incidence(len(atomic_numbers), edges, device=device, dtype=dtype)
    displacement = positions.index_select(0, edges[0]) - positions.index_select(
        0, edges[1]
    )
    distance = torch.linalg.vector_norm(displacement, dim=1)

    pair_radius = covalent_radius[edges[0]] + covalent_radius[edges[1]]
    capacity_argument = parameters.capacity_steepness * (
        distance / (parameters.capacity_radius_scale * pair_radius) - 1.0
    )
    capacity_shape = 0.5 * torch.erfc(capacity_argument)
    support_amplitude = _smooth_compact_support(
        distance, parameters.support_inner_A, parameters.support_outer_A
    )
    pair_capacity = torch.sqrt(
        elemental_capacity[edges[0]] * elemental_capacity[edges[1]]
    )
    # Work with sqrt(C) directly.  Squaring a C2 compact-support amplitude
    # avoids differentiating sqrt(C) at an exactly vanished channel.
    sqrt_compliance = (
        torch.sqrt(pair_capacity)
        * torch.sqrt(torch.clamp(capacity_shape, min=torch.finfo(dtype).tiny))
        * support_amplitude
    )
    compliance = sqrt_compliance.square()

    coulomb = _gaussian_coulomb_matrix(
        positions, sigma, parameters.coulomb_eV_A_per_e2
    )
    hardness = torch.diag(intrinsic_hardness) + coulomb
    transfer_map = incidence * sqrt_compliance[None, :]
    response = torch.eye(len(distance), device=device, dtype=dtype)
    response = response + transfer_map.T @ hardness @ transfer_map
    rhs = -(transfer_map.T @ chi)
    scaled_transfers = torch.linalg.solve(response, rhs)
    transfers = sqrt_compliance * scaled_transfers
    charges = incidence @ transfers
    energy = (
        0.5 * torch.dot(scaled_transfers, scaled_transfers)
        + torch.dot(chi, charges)
        + 0.5 * torch.dot(charges, hardness @ charges)
    )
    # Uniform-field linear response.  For E_field = -F . mu with
    # mu = R^T q, differentiating the stationary equations gives
    # alpha_raw = R^T X A^-1 X^T R.  Multiplication by the Coulomb constant
    # converts e A^2 / V to A^3.
    field_rhs = transfer_map.T @ positions
    field_response = torch.linalg.solve(response, field_rhs)
    polarizability = (
        positions.T @ transfer_map @ field_response
        * parameters.coulomb_eV_A_per_e2
    )
    diagnostics = build_diagnostics(
        response,
        hardness,
        rhs,
        scaled_transfers,
        transfers,
        charges,
        compliance,
    )
    return C0SolveResult(
        energy=energy,
        charges=charges,
        transfer_amplitudes=transfers,
        scaled_transfer_variables=scaled_transfers,
        edge_index=edges,
        compliance=compliance,
        incidence=incidence,
        hardness_matrix=hardness,
        response_matrix=response,
        electronegativity=chi,
        polarizability_A3=polarizability,
        diagnostics=diagnostics,
    )
