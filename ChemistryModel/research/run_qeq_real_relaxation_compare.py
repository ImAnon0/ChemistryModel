from __future__ import annotations

from dataclasses import replace

import torch

from chemistry_engine.config import PhysicsSpec
from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.unified_bond_capacity.prototype import (
    UnifiedBondCapacityEnergyPrototype,
)
from research.electrostatics_diagnostics import known_diagnostic_cases


def _spec_with_extension(spec: PhysicsSpec, enabled: bool):
    return replace(
        spec,
        enabled_extensions=("electrostatics",) if enabled else (),
    )


def build_case(name: str, qeq: bool):
    positions, atoms = known_diagnostic_cases()[name]

    sim = UnifiedBondCapacityEnergyPrototype(
        # Keep the diagnostic deliberately small.
        device="cpu",
        dtype=torch.float64,
        box_size=20.0,
    )

    sim.chemistry_physics_spec = _spec_with_extension(
        sim.chemistry_physics_spec,
        qeq,
    )

    # This script is intentionally a wiring diagnostic.
    # The production constructor should remain untouched until this comparison
    # is accepted.
    return sim


def run():
    for name in ("H2O", "CH2O"):
        print("=" * 40)
        print(name)
        print("=" * 40)

        for qeq in (False, True):
            sim = build_case(name, qeq)

            print("qeq:", qeq)
            print("extensions:",
                  sim.chemistry_physics_spec.enabled_extensions)


if __name__ == "__main__":
    run()
