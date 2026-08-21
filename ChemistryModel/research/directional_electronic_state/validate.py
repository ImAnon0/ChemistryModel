"""Static, QM, and dynamics gates for the directional P2 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import build_box
import reactive as R
from research.directional_electronic_state import (
    StateConditionedP2CouplingPrototype,
)
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _evaluate,
)
from research.heavy_valence_state.compare_qm_microscopes import _load_qm, _metrics
from research.heavy_valence_state.validate_formulation_dynamics import (
    _build as build_nve,
    _cases as nve_cases,
    _run as run_nve,
)
from research.unified_bond_capacity import UnifiedBondCapacityEnergyPrototype
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_GRAMBOW = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/directional_electronic_validation.json"
)
DEFAULT_CSV = Path(
    "research_data/benchmark/diagnostics/directional_electronic_qm.csv"
)

MODELS = {
    "production": OptimisedValenceStateBatchedSimulation,
    "unified_radial": UnifiedBondCapacityEnergyPrototype,
    "state_conditioned_p2": StateConditionedP2CouplingPrototype,
}


def _model_energy(model, geometry):
    coordinates = np.asarray(geometry["coordinates_angstrom"], dtype=float)
    coordinates -= coordinates.mean(axis=0)
    coordinates += 20.0
    simulation, force, energy = _evaluate(model, geometry["symbols"], coordinates)
    if not math.isfinite(float(energy)) or not bool(torch.isfinite(force).all()):
        raise ValueError("non-finite energy or force")
    return simulation, float(energy), float(
        torch.linalg.vector_norm(force, dim=1).max()
    )


def _qm_microscopes(geometries_path, qm_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))[
        "geometries"
    ]
    qm = _load_qm(qm_path)
    rows = []
    transition_factors = []
    for index, geometry in enumerate(geometries):
        coordinate = geometry.get("reaction_coordinate", {})
        row = {
            "geometry_id": geometry["geometry_id"],
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": coordinate.get(
                "transfer_distance_angstrom", ""
            ),
            "qm_energy_eV": qm[geometry["geometry_id"]],
        }
        for name, model in MODELS.items():
            simulation, energy, force_max = _model_energy(model, geometry)
            row[f"{name}_energy_eV"] = energy
            row[f"{name}_force_max_eV_per_angstrom"] = force_max
            if name == "state_conditioned_p2":
                diagnostics = simulation._directional_electronic_diagnostics
                row["p2_transition_count"] = diagnostics["transition_count"]
                row["p2_factor_min"] = diagnostics["minimum_transition_factor"]
                row["p2_factor_max"] = diagnostics["maximum_transition_factor"]
                transition_factors.extend(
                    item["transition_factor"] for item in diagnostics["transitions"]
                )
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"QM microscopes: {index + 1}/{len(geometries)}", flush=True)

    references = {
        row["system"]: row
        for row in rows
        if row["sample_kind"] == "reactant_reference"
    }
    residuals = defaultdict(lambda: defaultdict(list))
    for row in rows:
        reference = references[row["system"]]
        row["qm_relative_eV"] = row["qm_energy_eV"] - reference["qm_energy_eV"]
        for name in MODELS:
            row[f"{name}_relative_eV"] = (
                row[f"{name}_energy_eV"] - reference[f"{name}_energy_eV"]
            )
            row[f"{name}_residual_eV"] = (
                row["qm_relative_eV"] - row[f"{name}_relative_eV"]
            )
            if row["sample_kind"] == "dense_transfer_scan":
                residuals[row["system"]][name].append(
                    row[f"{name}_residual_eV"]
                )
    summary = {
        "evaluated_geometries": len(rows),
        "systems": {},
        "all_dense": {},
        "directional_factor_range": {
            "minimum": min(transition_factors, default=1.0),
            "maximum": max(transition_factors, default=1.0),
            "count": len(transition_factors),
        },
    }
    for system, values_by_model in sorted(residuals.items()):
        summary["systems"][system] = {
            name: _metrics(values) for name, values in values_by_model.items()
        }
    for name in MODELS:
        values = [
            value
            for values_by_model in residuals.values()
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
    rows = []
    for name, (symbols, coordinates) in geometries.items():
        positions = np.asarray(coordinates, dtype=float) + 20.0
        _, reference_force, reference_energy = _evaluate(
            UnifiedBondCapacityEnergyPrototype, symbols, positions
        )
        _, force, energy = _evaluate(
            StateConditionedP2CouplingPrototype, symbols, positions
        )
        rows.append({
            "molecule": name,
            "energy_change_from_radial_eV": float(energy - reference_energy),
            "maximum_force_change_from_radial_eV_per_angstrom": float(
                torch.max(torch.abs(force - reference_force))
            ),
        })
    return rows


def _water_geometry(geometries_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))[
        "geometries"
    ]
    row = min(
        (
            item for item in geometries
            if item["system"] == "water"
            and item["sample_kind"] == "dense_transfer_scan"
        ),
        key=lambda item: abs(float(
            item["reaction_coordinate"]["transfer_distance_angstrom"]
        ) - 1.16),
    )
    return row["symbols"], np.asarray(row["coordinates_angstrom"], dtype=float) + 20.0


def _force_and_symmetry(geometries_path):
    symbols, positions = _water_geometry(geometries_path)
    _, force, energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, positions
    )
    epsilon = 1e-5
    plus = positions.copy()
    minus = positions.copy()
    plus[-1, 0] += epsilon
    minus[-1, 0] -= epsilon
    plus_energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, plus
    )[2]
    minus_energy = _evaluate(
        StateConditionedP2CouplingPrototype, symbols, minus
    )[2]
    numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
    oxygen = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    permutation = np.arange(len(symbols))
    permutation[oxygen[0]], permutation[oxygen[1]] = (
        permutation[oxygen[1]], permutation[oxygen[0]]
    )
    _, permuted_force, permuted_energy = _evaluate(
        StateConditionedP2CouplingPrototype,
        [symbols[index] for index in permutation], positions[permutation],
    )
    return {
        "autograd_force_eV_per_angstrom": float(force[-1, 0]),
        "finite_difference_force_eV_per_angstrom": numerical,
        "force_difference_eV_per_angstrom": float(force[-1, 0]) - numerical,
        "permuted_energy_difference_eV": float(permuted_energy - energy),
        "permuted_force_max_difference_eV_per_angstrom": float(torch.max(
            torch.abs(permuted_force - force[torch.as_tensor(permutation)])
        )),
    }


def _cutoff_continuity():
    hydrogen = R.ELEMENT_INDEX["H"]
    oxygen = R.ELEMENT_INDEX["O"]
    cutoff = float(R.CUTOFF_OUTER[oxygen, hydrogen])
    delta = 1e-4
    samples = []
    for distance in (cutoff - delta, cutoff, cutoff + delta):
        symbols = ["O", "H", "O", "H", "H"]
        positions = np.asarray([
            [0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
            [0.96 + distance, 0.0, 0.0],
            [0.96 + distance, 0.96, 0.0], [0.96 + distance, 0.0, 0.96],
        ]) + 20.0
        _, force, energy = _evaluate(
            StateConditionedP2CouplingPrototype, symbols, positions
        )
        samples.append({
            "distance_angstrom": distance,
            "energy_eV": float(energy),
            "moving_oxygen_force_x_eV_per_angstrom": float(force[2, 0]),
        })
    return {
        "O_H_outer_cutoff_angstrom": cutoff,
        "samples": samples,
        "energy_span_eV": max(row["energy_eV"] for row in samples)
        - min(row["energy_eV"] for row in samples),
        "maximum_absolute_force_eV_per_angstrom": max(
            abs(row["moving_oxygen_force_x_eV_per_angstrom"]) for row in samples
        ),
    }


def _nve(geometries_path, grambow_path):
    cases = nve_cases(geometries_path, grambow_path)
    result = {}
    for case_name, case in cases.items():
        reference = build_nve(UnifiedBondCapacityEnergyPrototype, case)
        initial_velocities = reference.velocities.detach().clone()
        result[case_name] = {}
        for name, model in (
            ("unified_radial", UnifiedBondCapacityEnergyPrototype),
            ("state_conditioned_p2", StateConditionedP2CouplingPrototype),
        ):
            simulation = build_nve(model, case)
            simulation.velocities = initial_velocities.clone()
            try:
                result[case_name][name] = run_nve(simulation, case["steps"])
                result[case_name][name]["status"] = "pass"
            except Exception as error:
                result[case_name][name] = {
                    "status": "fail",
                    "error": f"{type(error).__name__}: {error}",
                }
        print(f"NVE: {case_name}", flush=True)
    return result


def validate(geometries, qm, grambow, include_nve):
    rows, microscopes = _qm_microscopes(geometries, qm)
    molecules = _molecule_preservation()
    force = _force_and_symmetry(geometries)
    cutoff = _cutoff_continuity()
    result = {
        "scope": "research only; unified radial and production are frozen",
        "qm_microscopes": microscopes,
        "molecule_preservation": molecules,
        "force_and_symmetry": force,
        "cutoff_continuity": cutoff,
    }
    if include_nve:
        result["nve"] = _nve(geometries, grambow)
    candidate_water = microscopes["systems"]["water"]["state_conditioned_p2"]
    result["gates"] = {
        "water_rmse_at_most_radial": candidate_water["rmse_eV"]
        <= microscopes["systems"]["water"]["unified_radial"]["rmse_eV"] + 1e-6,
        "finite_difference_force": abs(
            force["force_difference_eV_per_angstrom"]
        ) < 2e-4,
        "permutation_symmetry": abs(force["permuted_energy_difference_eV"]) < 1e-9
        and force["permuted_force_max_difference_eV_per_angstrom"] < 1e-8,
        "cutoff_continuity": cutoff["energy_span_eV"] < 1e-3
        and cutoff["maximum_absolute_force_eV_per_angstrom"] < 1e-2,
        "molecules_preserved": max(
            abs(row["energy_change_from_radial_eV"]) for row in molecules
        ) < 1e-8 and max(
            row["maximum_force_change_from_radial_eV_per_angstrom"]
            for row in molecules
        ) < 1e-7,
    }
    if include_nve:
        candidate_nve = [
            values["state_conditioned_p2"] for values in result["nve"].values()
        ]
        result["gates"]["nve_finite_zero_caps"] = all(
            row.get("status") == "pass" and row["move_caps"] == 0
            for row in candidate_nve
        )
        result["gates"]["nve_drift_below_0_05_eV"] = all(
            row.get("max_absolute_drift_eV", float("inf")) < 0.05
            for row in candidate_nve
        )
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--grambow", type=Path, default=DEFAULT_GRAMBOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--skip-nve", action="store_true")
    args = parser.parse_args()
    rows, result = validate(
        args.geometries, args.qm, args.grambow, not args.skip_nve
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
        "nve": result.get("nve", "skipped"),
    }, indent=2))


if __name__ == "__main__":
    main()
