"""Validation gates for the observable-anchored electronic hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_box
import reactive as R
from research.electronic_state_correction import (
    CombinedElectronicStatePrototype,
    LocalElectronicDescriptorPrototype,
    MultipoleDensityPrototype,
    PolarisationResponsePrototype,
)
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _evaluate,
)
from research.heavy_valence_state.compare_qm_microscopes import _metrics
from research.heavy_valence_state.validate_formulation_dynamics import (
    _build as build_nve,
    _cases as nve_cases,
    _run as run_nve,
)
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/directional_electronic_qm.csv"
)
DEFAULT_GRAMBOW = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/electronic_state_validation.json"
)
DEFAULT_CSV = Path(
    "research_data/benchmark/diagnostics/electronic_state_qm.csv"
)

MODELS = {
    "unified_radial": UnifiedBondCapacityEnergyPrototype,
    "local_scalar": LocalElectronicDescriptorPrototype,
    "polarisation_vector": PolarisationResponsePrototype,
    "multipole_tensor": MultipoleDensityPrototype,
    "combined": CombinedElectronicStatePrototype,
}
CANDIDATES = tuple(name for name in MODELS if name != "unified_radial")


def _model_energy(model, geometry):
    coordinates = np.asarray(geometry["coordinates_angstrom"], dtype=float)
    coordinates -= coordinates.mean(axis=0)
    coordinates += 20.0
    simulation, force, energy = _evaluate(model, geometry["symbols"], coordinates)
    if not math.isfinite(float(energy)) or not bool(torch.isfinite(force).all()):
        raise ValueError("non-finite energy or force")
    return simulation, float(energy), force


def _qm_microscopes(geometries_path, baseline_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))[
        "geometries"
    ]
    by_id = {row["geometry_id"]: row for row in geometries}
    with baseline_path.open("r", newline="", encoding="utf-8") as handle:
        baseline = {row["geometry_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for index, geometry in enumerate(geometries):
        old = baseline[geometry["geometry_id"]]
        row = {
            "geometry_id": geometry["geometry_id"],
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "qm_energy_eV": float(old["qm_energy_eV"]),
            "unified_radial_energy_eV": float(old["unified_radial_energy_eV"]),
        }
        for name in CANDIDATES:
            simulation, energy, force = _model_energy(MODELS[name], geometry)
            row[f"{name}_energy_eV"] = energy
            row[f"{name}_force_max_eV_per_angstrom"] = float(
                torch.linalg.vector_norm(force, dim=1).max()
            )
            row[f"{name}_correction_eV"] = simulation._electronic_state_diagnostics[
                "correction_eV"
            ]
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"QM microscopes: {index + 1}/{len(geometries)}", flush=True)

    references = {
        row["system"]: row for row in rows
        if row["sample_kind"] == "reactant_reference"
    }
    residuals = defaultdict(lambda: defaultdict(list))
    for row in rows:
        reference = references[row["system"]]
        qm_relative = row["qm_energy_eV"] - reference["qm_energy_eV"]
        for name in MODELS:
            relative = row[f"{name}_energy_eV"] - reference[f"{name}_energy_eV"]
            row[f"{name}_relative_eV"] = relative
            row[f"{name}_residual_eV"] = qm_relative - relative
            if row["sample_kind"] == "dense_transfer_scan":
                residuals[row["system"]][name].append(
                    row[f"{name}_residual_eV"]
                )
    summary = {"systems": {}, "all_dense": {}}
    for system, values_by_model in sorted(residuals.items()):
        summary["systems"][system] = {
            name: _metrics(values) for name, values in values_by_model.items()
        }
    for name in MODELS:
        values = [
            value for values_by_model in residuals.values()
            for value in values_by_model[name]
        ]
        summary["all_dense"][name] = _metrics(values)
    return rows, summary


def _molecule_preservation():
    names = (
        "h3", "methane", "formaldehyde", "water", "ethane", "methanol",
        "hydroxylamine", "hydrogen_peroxide",
    )
    geometries = dict(zip(names, _accepted_geometries()))
    geometries["h2"] = build_box.BUILDERS["H2"]()
    geometries["ammonia"] = build_box.BUILDERS["NH3"]()
    geometries["n2"] = (["N", "N"], [[0, 0, 0], [1.0976, 0, 0]])
    result = {}
    for name, (symbols, coordinates) in geometries.items():
        positions = np.asarray(coordinates, dtype=float) + 20.0
        _, baseline_force, baseline_energy = _evaluate(
            UnifiedBondCapacityEnergyPrototype, symbols, positions
        )
        result[name] = {}
        for model_name in CANDIDATES:
            _, force, energy = _evaluate(MODELS[model_name], symbols, positions)
            result[name][model_name] = {
                "energy_change_eV": float(energy - baseline_energy),
                "maximum_force_change_eV_per_angstrom": float(
                    torch.max(torch.abs(force - baseline_force))
                ),
            }
    return result


def _water_test_geometry(geometries_path):
    rows = json.loads(geometries_path.read_text(encoding="utf-8"))["geometries"]
    return min(
        (
            row for row in rows
            if row["system"] == "water"
            and row["sample_kind"] == "dense_transfer_scan"
        ),
        key=lambda row: abs(
            float(row["reaction_coordinate"]["transfer_distance_angstrom"])
            - 1.30
        ),
    )


def _force_symmetry(geometries_path):
    geometry = _water_test_geometry(geometries_path)
    symbols = geometry["symbols"]
    positions = np.asarray(geometry["coordinates_angstrom"], dtype=float) + 20.0
    epsilon = 1e-5
    oxygen = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    permutation = np.arange(len(symbols))
    permutation[oxygen[0]], permutation[oxygen[1]] = (
        permutation[oxygen[1]], permutation[oxygen[0]]
    )
    result = {}
    for name in CANDIDATES:
        _, force, energy = _evaluate(MODELS[name], symbols, positions)
        plus, minus = positions.copy(), positions.copy()
        plus[-1, 0] += epsilon
        minus[-1, 0] -= epsilon
        plus_energy = _evaluate(MODELS[name], symbols, plus)[2]
        minus_energy = _evaluate(MODELS[name], symbols, minus)[2]
        numerical = -(float(plus_energy) - float(minus_energy)) / (2 * epsilon)
        _, permuted_force, permuted_energy = _evaluate(
            MODELS[name], [symbols[index] for index in permutation],
            positions[permutation],
        )
        result[name] = {
            "finite_difference_force_error_eV_per_angstrom": float(force[-1, 0]) - numerical,
            "permutation_energy_error_eV": float(permuted_energy - energy),
            "permutation_force_max_error_eV_per_angstrom": float(torch.max(
                torch.abs(permuted_force - force[torch.as_tensor(permutation)])
            )),
        }
    return result


def _cutoff_continuity():
    h, o = R.ELEMENT_INDEX["H"], R.ELEMENT_INDEX["O"]
    cutoff = float(R.CUTOFF_OUTER[h, o])
    delta = 1e-4
    result = {}
    for name in CANDIDATES:
        samples = []
        for distance in (cutoff - delta, cutoff, cutoff + delta):
            symbols = ["O", "H"]
            positions = np.asarray([[0, 0, 0], [distance, 0, 0]], dtype=float) + 20.0
            _, base_force, base_energy = _evaluate(
                UnifiedBondCapacityEnergyPrototype, symbols, positions
            )
            _, force, energy = _evaluate(MODELS[name], symbols, positions)
            samples.append({
                "distance_angstrom": distance,
                "correction_energy_eV": float(energy - base_energy),
                "correction_force_x_eV_per_angstrom": float(
                    force[1, 0] - base_force[1, 0]
                ),
            })
        result[name] = {
            "samples": samples,
            "correction_energy_span_eV": max(x["correction_energy_eV"] for x in samples)
            - min(x["correction_energy_eV"] for x in samples),
            "maximum_correction_force_eV_per_angstrom": max(
                abs(x["correction_force_x_eV_per_angstrom"]) for x in samples
            ),
        }
    return result


def _hydrogen_identity():
    cases = {
        "h2": (["H", "H"], [[0, 0, 0], [0.74144, 0, 0]]),
        "h3_transfer": (["H"] * 3, [[0, 0, 0], [0.84, 0, 0], [1.90, 0, 0]]),
        "h_plus_h2": (["H"] * 3, [[0, 0, 0], [0.74144, 0, 0], [2.20, 0, 0]]),
    }
    result = {}
    for case, (symbols, coordinates) in cases.items():
        positions = np.asarray(coordinates, dtype=float) + 20.0
        _, base_force, base_energy = _evaluate(
            UnifiedBondCapacityEnergyPrototype, symbols, positions
        )
        result[case] = {}
        for name in CANDIDATES:
            _, force, energy = _evaluate(MODELS[name], symbols, positions)
            result[case][name] = {
                "energy_difference_eV": float(energy - base_energy),
                "force_max_difference_eV_per_angstrom": float(
                    torch.max(torch.abs(force - base_force))
                ),
            }
    return result


def _nve(geometries_path, grambow_path):
    cases = nve_cases(geometries_path, grambow_path)
    selected = {
        "water_transfer": cases["water_transfer"],
        "grambow_crowded_reactant": cases["grambow_crowded_reactant"],
    }
    result = {}
    for case_name, case in selected.items():
        reference = build_nve(UnifiedBondCapacityEnergyPrototype, case)
        velocities = reference.velocities.detach().clone()
        result[case_name] = {}
        for name, model in MODELS.items():
            simulation = build_nve(model, case)
            simulation.velocities = velocities.clone()
            try:
                result[case_name][name] = run_nve(simulation, case["steps"])
                result[case_name][name]["status"] = "pass"
            except Exception as exc:
                result[case_name][name] = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        print(f"NVE: {case_name}", flush=True)
    return result


def validate(geometries, baseline, grambow, include_nve):
    rows, microscopes = _qm_microscopes(geometries, baseline)
    result = {
        "scope": "research-only; production and unified radial untouched",
        "qm_microscopes": microscopes,
        "molecule_preservation": _molecule_preservation(),
        "force_and_symmetry": _force_symmetry(geometries),
        "cutoff_continuity": _cutoff_continuity(),
        "hydrogen_identity": _hydrogen_identity(),
    }
    if include_nve:
        result["nve"] = _nve(geometries, grambow)
    gates = {}
    water_baseline = microscopes["systems"]["water"]["unified_radial"]["rmse_eV"]
    for name in CANDIDATES:
        molecule_rows = [values[name] for values in result["molecule_preservation"].values()]
        force = result["force_and_symmetry"][name]
        cutoff = result["cutoff_continuity"][name]
        hydrogen_rows = [values[name] for values in result["hydrogen_identity"].values()]
        candidate = {
            "water_no_worse_than_unified": microscopes["systems"]["water"][name]["rmse_eV"]
            <= water_baseline + 1e-6,
            "finite_difference_force": abs(force["finite_difference_force_error_eV_per_angstrom"]) < 2e-4,
            "permutation_symmetry": abs(force["permutation_energy_error_eV"]) < 1e-9
            and force["permutation_force_max_error_eV_per_angstrom"] < 1e-8,
            "cutoff_continuity": cutoff["correction_energy_span_eV"] < 1e-6
            and cutoff["maximum_correction_force_eV_per_angstrom"] < 1e-4,
            "settled_molecules_preserved": max(abs(x["energy_change_eV"]) for x in molecule_rows) < 1e-6
            and max(x["maximum_force_change_eV_per_angstrom"] for x in molecule_rows) < 1e-5,
            "hydrogen_safety_inherited_exactly": max(abs(x["energy_difference_eV"]) for x in hydrogen_rows) < 1e-10
            and max(x["force_max_difference_eV_per_angstrom"] for x in hydrogen_rows) < 1e-9,
        }
        if include_nve:
            nve_rows = [values[name] for values in result["nve"].values()]
            candidate["nve_finite_zero_caps"] = all(
                row.get("status") == "pass" and row["move_caps"] == 0
                for row in nve_rows
            )
            candidate["nve_drift_below_0_05_eV"] = all(
                row.get("max_absolute_drift_eV", float("inf")) < 0.05
                for row in nve_rows
            )
        candidate["all_gates"] = all(candidate.values())
        gates[name] = candidate
    result["gates"] = gates
    result["grambow_promotion"] = {
        "allowed_candidates": [name for name, values in gates.items() if values["all_gates"]],
        "rule": "Only candidates passing water, force, symmetry, cutoff, molecule, H safety and NVE may open Grambow.",
    }
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--grambow", type=Path, default=DEFAULT_GRAMBOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--skip-nve", action="store_true")
    args = parser.parse_args()
    rows, result = validate(
        args.geometries, args.baseline, args.grambow, not args.skip_nve
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "qm": result["qm_microscopes"],
        "gates": result["gates"],
        "grambow_promotion": result["grambow_promotion"],
        "nve": result.get("nve", "skipped"),
    }, indent=2))


if __name__ == "__main__":
    main()
