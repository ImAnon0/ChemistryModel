"""Prove that O-O topology exclusion leaves radial physics unchanged."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import reactive as R
from reactive_torch import ReactiveSimulation


DEFAULT_OUTPUT = Path(
    "research_data/benchmark/oo_topology_exclusion_probe.json"
)


SYMBOLS = ["O", "H", "H", "O", "H", "H"]
POSITIONS = np.asarray([
    [0.00, 0.00, 0.00],
    [0.96, 0.00, 0.00],
    [-0.24, 0.93, 0.00],
    [2.35, 0.00, 0.00],
    [3.31, 0.00, 0.00],
    [2.11, 0.93, 0.00],
], dtype=np.float64)


class CaptureTopologySimulation(ReactiveSimulation):
    def __init__(self, *args, **kwargs):
        self._share_reactive_intermediates = True
        super().__init__(*args, **kwargs)


class ExcludeOOTopologySimulation(CaptureTopologySimulation):
    def topology_taper(self, taper, centre_types, other_types, mask):
        current = super().topology_taper(
            taper, centre_types, other_types, mask
        )
        oxygen = int(R.ELEMENT_INDEX["O"])
        oo_contact = (
            (centre_types == oxygen)
            & (other_types == oxygen)
            & (mask > 0.0)
        )
        return torch.where(
            oo_contact, torch.zeros_like(current), current
        )


def evaluate(simulation_class):
    simulation = simulation_class(
        symbols=SYMBOLS,
        positions=POSITIONS,
        box_size=20.0,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )
    values = simulation._reactive_intermediates[1]
    oxygen = int(R.ELEMENT_INDEX["O"])
    oo = (
        (values["centre_types"] == oxygen)
        & (values["other_types"] == oxygen)
        & (values["mask"] > 0.0)
    )
    return {
        "total_energy_eV": float(simulation.potential_energy),
        "radial_energy_eV": float(
            simulation._energy_parts["bond"].sum().cpu()
        ),
        "overcoordination_energy_eV": float(
            simulation._energy_parts["over"].sum().cpu()
        ),
        "angle_energy_eV": float(
            simulation._energy_parts["angle"].sum().cpu()
        ),
        "radial_taper_sum": float(
            values["taper"].sum().detach().cpu()
        ),
        "topology_taper_sum": float(
            values["topology_taper"].sum().detach().cpu()
        ),
        "oo_radial_tapers": (
            values["taper"][oo].detach().cpu().tolist()
        ),
        "oo_topology_tapers": (
            values["topology_taper"][oo].detach().cpu().tolist()
        ),
    }


def run_probe():
    before = evaluate(CaptureTopologySimulation)
    after = evaluate(ExcludeOOTopologySimulation)
    tolerance = 1e-12
    checks = {
        "radial_taper_identical": abs(
            before["radial_taper_sum"] - after["radial_taper_sum"]
        ) <= tolerance,
        "radial_energy_identical": abs(
            before["radial_energy_eV"] - after["radial_energy_eV"]
        ) <= tolerance,
        "oo_topology_changed": (
            before["oo_topology_tapers"] != after["oo_topology_tapers"]
        ),
        "excluded_oo_topology_is_zero": all(
            abs(value) <= tolerance
            for value in after["oo_topology_tapers"]
        ),
        "overcoordination_changed": abs(
            before["overcoordination_energy_eV"]
            - after["overcoordination_energy_eV"]
        ) > tolerance,
        "angle_changed": abs(
            before["angle_energy_eV"] - after["angle_energy_eV"]
        ) > tolerance,
    }
    return {
        "description": (
            "Diagnostic-only O-O topology exclusion through the production "
            "topology_taper hook; no parameter or production default changed."
        ),
        "symbols": SYMBOLS,
        "positions_angstrom": POSITIONS.tolist(),
        "before": before,
        "after": after,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    result = run_probe()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print("O-O topology exclusion probe")
    print()
    print(f"radial taper identical : {result['checks']['radial_taper_identical']}")
    print(f"radial energy identical: {result['checks']['radial_energy_identical']}")
    print(f"topology changed       : {result['checks']['oo_topology_changed']}")
    print(f"overcoord changed      : {result['checks']['overcoordination_changed']}")
    print(f"angle changed          : {result['checks']['angle_changed']}")
    print(f"overall                : {'PASS' if result['passed'] else 'FAIL'}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
