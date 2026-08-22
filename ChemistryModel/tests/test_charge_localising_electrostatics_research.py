"""Acceptance-gate evidence for the rejected Gaussian QTPIE candidate."""

from types import SimpleNamespace

import pytest
import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.charge_localising_electrostatics import GaussianQTPIECandidate
from research.electrostatics_diagnostics import known_diagnostic_cases


def _context(coordinates, atomic_numbers, *, requires_grad=False):
    positions = torch.tensor(coordinates, dtype=torch.float64)
    positions.requires_grad_(requires_grad)
    return SimpleNamespace(
        positions=positions,
        atomic_numbers=tuple(atomic_numbers),
        batch_assignment=(0,) * len(atomic_numbers),
        box_size=0.0,
    )


def _evaluate(term_type, coordinates, atomic_numbers, *, requires_grad=False):
    context = _context(coordinates, atomic_numbers, requires_grad=requires_grad)
    term = term_type(enabled=True)
    energy = term.energy(context, torch.zeros((), dtype=torch.float64))
    return context, term, energy


def _minimum_projected_eigenvalue(matrix):
    count = len(matrix)
    projector = torch.eye(count, dtype=matrix.dtype) - torch.ones_like(matrix) / count
    basis = torch.linalg.svd(projector).U[:, : count - 1]
    return torch.linalg.eigvalsh(basis.T @ matrix @ basis)[0]


def test_gaussian_qtpie_localises_separated_oh_but_qeq_does_not():
    coordinates = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    _, qeq, qeq_energy = _evaluate(ElectrostaticEnergyTerm, coordinates, (8, 1))
    _, qtpie, qtpie_energy = _evaluate(GaussianQTPIECandidate, coordinates, (8, 1))

    assert abs(float(qeq.diagnostics()["charges"][0])) > 0.1
    assert abs(float(qeq_energy)) > 0.1
    assert abs(float(qtpie.diagnostics()["charges"][0])) < 1e-12
    assert abs(float(qtpie_energy)) < 1e-12


def test_gaussian_qtpie_rejected_for_indefinite_h3_response():
    _, term, _ = _evaluate(
        GaussianQTPIECandidate,
        [[-0.74, 0.0, 0.0], [0.0, 0.0, 0.0], [0.92, 0.0, 0.0]],
        (1, 1, 1),
    )
    minimum = _minimum_projected_eigenvalue(term.diagnostics()["qeq_matrix"])
    assert float(minimum) < -0.7


def test_gaussian_qtpie_rejected_for_unphysical_methane_charge_response():
    coordinates, atomic_numbers = known_diagnostic_cases()["CH4"]
    _, term, _ = _evaluate(GaussianQTPIECandidate, coordinates, atomic_numbers)
    assert float(torch.max(torch.abs(term.diagnostics()["charges"]))) > 7.0


def test_gaussian_qtpie_candidate_force_is_still_variational_and_invariant():
    coordinates, atomic_numbers = known_diagnostic_cases()["H2O"]
    context, _, energy = _evaluate(
        GaussianQTPIECandidate, coordinates, atomic_numbers, requires_grad=True
    )
    force = -torch.autograd.grad(energy, context.positions)[0]
    assert torch.isfinite(force).all()
    assert torch.allclose(force.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12)

    step = 1e-6
    plus = [list(row) for row in coordinates]
    minus = [list(row) for row in coordinates]
    plus[1][0] += step
    minus[1][0] -= step
    plus_energy = _evaluate(GaussianQTPIECandidate, plus, atomic_numbers)[2]
    minus_energy = _evaluate(GaussianQTPIECandidate, minus, atomic_numbers)[2]
    finite_difference = -float(plus_energy - minus_energy) / (2.0 * step)
    assert float(force[1, 0]) == pytest.approx(finite_difference, abs=2e-7)
