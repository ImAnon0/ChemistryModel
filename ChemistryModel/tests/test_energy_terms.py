from __future__ import annotations

import torch

from chemistry_engine.terms import NullEnergyTerm
from unified_radial_equivalence import build_simulation, load_fixture
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


def test_null_energy_term_is_exactly_zero():
    case = next(
        case for case in load_fixture()["cases"]
        if case["name"] == "h2_equilibrium"
    )

    simulation = build_simulation(
        UnifiedBondCapacityEnergyPrototype,
        [case],
        device="cpu",
        dtype=torch.float64,
        box_size=40.0,
    )

    term = NullEnergyTerm()

    contribution = term.energy(
        simulation._last_chemistry_result,
        simulation._potential_energy,
    )

    assert torch.equal(
        contribution,
        torch.zeros_like(contribution),
    )
