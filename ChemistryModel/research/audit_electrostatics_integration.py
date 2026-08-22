"""Reproducible, non-fitting audit of the opt-in QEq extension.

This script never changes parameters or trajectories.  It compares the
electrostatics-enabled dataset with a same-engine disabled reconstruction,
checks the older ``base_results.csv`` provenance mismatch, and measures QEq
state smoothness along the stored reaction grids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from chemistry_engine.terms.electrostatics import ElectrostaticEnergyTerm
from research.electrostatics_diagnostics import known_diagnostic_cases


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/geometries.json")
DEFAULT_OLD_BASE = Path("research_data/qm_residual/base_results.csv")
DEFAULT_ELECTRO = Path("research_data/qm_residual/electrostatics_results.csv")
DEFAULT_RESIDUAL = Path(
    "research_data/qm_residual/electrostatics_residual_dataset.csv"
)
DEFAULT_OUTPUT = Path("research_data/qm_residual/electrostatics_audit.json")
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}
E_ANGSTROM_TO_DEBYE = 4.80320471257


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metrics(errors):
    values = list(errors)
    return {
        "count": len(values),
        "mae_eV": statistics.fmean(abs(value) for value in values),
        "rmse_eV": math.sqrt(statistics.fmean(value * value for value in values)),
        "signed_mean_eV": statistics.fmean(values),
        "maximum_absolute_eV": max(abs(value) for value in values),
    }


def _group_metrics(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {
        name: {
            "disabled": _metrics(row["disabled_error_eV"] for row in group),
            "enabled": _metrics(row["enabled_error_eV"] for row in group),
            "improved_count": sum(
                abs(row["enabled_error_eV"]) < abs(row["disabled_error_eV"])
                for row in group
            ),
            "worsened_count": sum(
                abs(row["enabled_error_eV"]) > abs(row["disabled_error_eV"])
                for row in group
            ),
        }
        for name, group in sorted(grouped.items())
    }


def _same_engine_residuals(electro_rows, residual_rows):
    electro_by_id = {row["geometry_id"]: row for row in electro_rows}
    by_system = defaultdict(list)
    for row in residual_rows:
        by_system[row["system"]].append(row)

    compared = []
    for system, rows in by_system.items():
        reference = next(
            row for row in rows if row["sample_kind"] == "reactant_reference"
        )
        ref_electro = float(
            electro_by_id[reference["geometry_id"]]["electrostatics_energy_eV"]
        )
        enabled_reference = float(reference["base_reference_eV"])
        disabled_reference = enabled_reference - ref_electro
        qm_reference = float(reference["qm_reference_eV"])

        for row in rows:
            electro = float(
                electro_by_id[row["geometry_id"]]["electrostatics_energy_eV"]
            )
            enabled = float(row["base_energy_eV"])
            disabled = enabled - electro
            qm = float(row["qm_energy_eV"])
            qm_relative = qm - qm_reference
            enabled_error = (enabled - enabled_reference) - qm_relative
            disabled_error = (disabled - disabled_reference) - qm_relative
            compared.append(
                {
                    "geometry_id": row["geometry_id"],
                    "system": system,
                    "region": row["region"],
                    "sample_kind": row["sample_kind"],
                    "electrostatics_eV": electro,
                    "enabled_error_eV": enabled_error,
                    "disabled_error_eV": disabled_error,
                }
            )
    return compared


def _old_baseline_mismatch(old_rows, electro_rows):
    old_by_id = {row["geometry_id"]: row for row in old_rows}
    differences = defaultdict(list)
    for row in electro_rows:
        geometry_id = row["geometry_id"]
        disabled_unified = (
            float(row["base_energy_eV"])
            - float(row["electrostatics_energy_eV"])
        )
        old = float(old_by_id[geometry_id]["base_energy_eV"])
        differences[row["system"]].append(disabled_unified - old)
    return {
        system: _metrics(values)
        for system, values in sorted(differences.items())
    }


def _context(positions, atomic_numbers, box_size=30.0):
    return SimpleNamespace(
        positions=positions,
        atomic_numbers=tuple(atomic_numbers),
        batch_assignment=(0,) * len(atomic_numbers),
        box_size=box_size,
    )


def _qeq_observation(coordinates, atomic_numbers, box_size=30.0):
    positions = torch.tensor(coordinates, dtype=torch.float64)
    context = _context(positions, atomic_numbers, box_size)
    term = ElectrostaticEnergyTerm(enabled=True)
    energy = term.energy(context, torch.zeros((), dtype=torch.float64))
    state = term.diagnostics()
    matrix = state["qeq_matrix"].detach()
    count = len(atomic_numbers)
    projector = torch.eye(count, dtype=torch.float64) - torch.ones(
        (count, count), dtype=torch.float64
    ) / count
    vectors = torch.linalg.svd(projector).U[:, : count - 1]
    projected = vectors.T @ matrix @ vectors
    eigenvalues = torch.linalg.eigvalsh(projected)
    condition = float(eigenvalues[-1] / eigenvalues[0]) if count > 1 else 1.0
    dipole = state["dipole"].detach()
    return {
        "charges": state["charges"].detach().tolist(),
        "charge_sum": float(state["charge_sum"].detach()),
        "energy_eV": float(energy.detach()),
        "dipole_e_angstrom": dipole.tolist(),
        "dipole_debye": float(torch.linalg.vector_norm(dipole)) * E_ANGSTROM_TO_DEBYE,
        "minimum_projected_eigenvalue": float(eigenvalues[0]),
        "projected_condition_number": condition,
    }


def _simple_systems():
    cases = dict(known_diagnostic_cases())
    cases["H3"] = (
        [[-0.74, 0.0, 0.0], [0.0, 0.0, 0.0], [0.92, 0.0, 0.0]],
        (1, 1, 1),
    )
    return {
        name: _qeq_observation(coordinates, atomic_numbers)
        for name, (coordinates, atomic_numbers) in cases.items()
    }


def _dissociation_scan():
    rows = []
    for distance in (1.0, 2.0, 5.0, 10.0, 25.0, 100.0):
        observation = _qeq_observation(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            (8, 1),
            box_size=0.0,
        )
        rows.append(
            {
                "distance_angstrom": distance,
                "oxygen_charge_e": observation["charges"][0],
                "hydrogen_charge_e": observation["charges"][1],
                "energy_eV": observation["energy_eV"],
            }
        )
    return {
        "pair": "neutral O + H under one global neutral constraint",
        "rows": rows,
        "asymptotic_charge_magnitude_e": abs(rows[-1]["oxygen_charge_e"]),
        "charge_localises_on_separation": abs(rows[-1]["oxygen_charge_e"]) < 1e-3,
    }


def _reaction_smoothness(geometry_path):
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    rows = []
    for geometry in payload["geometries"]:
        if geometry["sample_kind"] != "grid":
            continue
        observation = _qeq_observation(
            geometry["coordinates_angstrom"],
            [ATOMIC_NUMBERS[symbol] for symbol in geometry["symbols"]],
        )
        coordinate = geometry["reaction_coordinate"]
        rows.append(
            {
                "geometry_id": geometry["geometry_id"],
                "system": geometry["system"],
                "donor": float(coordinate["donor_distance_angstrom"]),
                "transfer": float(coordinate["transfer_distance_angstrom"]),
                "energy": observation["energy_eV"],
                "charges": observation["charges"],
            }
        )

    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    result = {}
    for system, group in sorted(by_system.items()):
        lookup = {(row["donor"], row["transfer"]): row for row in group}
        pairs = []
        for axis, fixed_axis in (("donor", "transfer"), ("transfer", "donor")):
            fixed_values = sorted({row[fixed_axis] for row in group})
            for fixed in fixed_values:
                line = sorted(
                    (row for row in group if row[fixed_axis] == fixed),
                    key=lambda row: row[axis],
                )
                for left, right in zip(line, line[1:]):
                    step = right[axis] - left[axis]
                    pairs.append(
                        {
                            "energy_delta": abs(right["energy"] - left["energy"]),
                            "energy_slope": abs(right["energy"] - left["energy"]) / step,
                            "charge_delta": max(
                                abs(a - b)
                                for a, b in zip(left["charges"], right["charges"])
                            ),
                        }
                    )
        result[system] = {
            "grid_count": len(group),
            "maximum_adjacent_energy_change_eV": max(
                pair["energy_delta"] for pair in pairs
            ),
            "maximum_adjacent_energy_slope_eV_per_angstrom": max(
                pair["energy_slope"] for pair in pairs
            ),
            "maximum_adjacent_atomic_charge_change_e": max(
                pair["charge_delta"] for pair in pairs
            ),
            "all_values_finite": all(
                math.isfinite(value)
                for row in group
                for value in [row["energy"], *row["charges"]]
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--old-base", type=Path, default=DEFAULT_OLD_BASE)
    parser.add_argument("--electrostatics", type=Path, default=DEFAULT_ELECTRO)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    old_rows = _read_csv(args.old_base)
    electro_rows = _read_csv(args.electrostatics)
    residual_rows = _read_csv(args.residual)
    compared = _same_engine_residuals(electro_rows, residual_rows)

    output = {
        "comparison_note": (
            "Disabled energies are reconstructed from the same unified-radial "
            "evaluation as enabled energies; old base_results.csv is a different model."
        ),
        "old_base_vs_disabled_unified": _old_baseline_mismatch(
            old_rows, electro_rows
        ),
        "qm_residual": {
            "overall": {
                "disabled": _metrics(row["disabled_error_eV"] for row in compared),
                "enabled": _metrics(row["enabled_error_eV"] for row in compared),
            },
            "by_system": _group_metrics(compared, "system"),
            "by_region": _group_metrics(compared, "region"),
        },
        "simple_systems": _simple_systems(),
        "dissociation_scan": _dissociation_scan(),
        "reaction_grid_smoothness": _reaction_smoothness(args.geometries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
