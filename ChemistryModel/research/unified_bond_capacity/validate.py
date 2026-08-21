"""Small-molecule, force, symmetry, QM, and NVE validation gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import build_box
import bond_calibration
from research.heavy_valence_continuous_edge.tests.test_continuous_edge import (
    _accepted_geometries,
    _crowded_carbon,
    _evaluate,
)
from research.heavy_valence_state import HeavyValenceStateEnergyPrototype
from research.heavy_valence_state.compare_qm_microscopes import (
    _evaluate as evaluate_qm,
    _load_qm,
    _metrics,
)
from research.heavy_valence_state.validate_formulation_dynamics import (
    _build as build_nve,
    _cases as nve_cases,
    _run as run_nve,
)
from research.unified_bond_capacity import (
    UnifiedBondCapacityEnergyPrototype,
    UnifiedBondCapacityTopologyPrototype,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_GRAMBOW = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/unified_bond_capacity_validation.json"
)

MODELS = {
    "production": OptimisedValenceStateBatchedSimulation,
    "local_v0": HeavyValenceStateEnergyPrototype,
    "unified_radial": UnifiedBondCapacityEnergyPrototype,
    "unified_topology": UnifiedBondCapacityTopologyPrototype,
}


def _curve_simulation(model, symbols, coordinates):
    positions = np.asarray(coordinates, dtype=float)
    positions -= positions.mean(axis=0)
    positions += 15.0
    return model(
        boxes=[(symbols, positions)], box_size=30.0, time_step=0.1,
        target_temperature=0.0, friction=0.0, device="cpu",
        dtype=torch.float64, random_seed=17, relax_on_start=False,
    )


def _h_curves():
    def h2_energy(distance):
        return float(_curve_simulation(
            UnifiedBondCapacityEnergyPrototype,
            ["H", "H"],
            [[-distance / 2.0, 0.0, 0.0], [distance / 2.0, 0.0, 0.0]],
        ).potential_energy)

    def h3_symmetric_energy(distance):
        return float(_curve_simulation(
            UnifiedBondCapacityEnergyPrototype,
            ["H", "H", "H"],
            [[-distance, 0.0, 0.0], [0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        ).potential_energy)

    h2_result = minimize_scalar(
        h2_energy, bounds=(0.55, 1.20), method="bounded",
        options={"xatol": 1e-10},
    )
    h3_result = minimize_scalar(
        h3_symmetric_energy, bounds=(0.55, 1.50), method="bounded",
        options={"xatol": 1e-9},
    )
    h2_simulation = _curve_simulation(
        UnifiedBondCapacityEnergyPrototype,
        ["H", "H"],
        [[-h2_result.x / 2.0, 0.0, 0.0], [h2_result.x / 2.0, 0.0, 0.0]],
    )
    approach = np.linspace(0.45, 4.0, 180)
    approach_energy = [float(_curve_simulation(
        UnifiedBondCapacityEnergyPrototype,
        ["H", "H", "H"],
        [[0.0, 0.0, 0.0], [h2_result.x, 0.0, 0.0],
         [h2_result.x + distance, 0.0, 0.0]],
    ).potential_energy) for distance in approach]
    approach_minimum = min(approach_energy)
    return {
        "h2": {
            "minimum_distance_angstrom": float(h2_result.x),
            "minimum_energy_eV": float(h2_result.fun),
            "target_distance_angstrom": 0.74144,
            "maximum_force_at_minimum_eV_per_angstrom": float(
                torch.max(torch.abs(h2_simulation.forces))
            ),
        },
        "h3_symmetric": {
            "minimum_half_separation_angstrom": float(h3_result.x),
            "minimum_energy_eV": float(h3_result.fun),
            "minimum_relative_to_h2_plus_h_eV": float(h3_result.fun - h2_result.fun),
            "incorrect_stable_h3": bool(h3_result.fun < h2_result.fun - 1e-4),
        },
        "h_plus_h2_approach": {
            "minimum_energy_eV": approach_minimum,
            "minimum_relative_to_h2_plus_h_eV": approach_minimum - float(h2_result.fun),
            "incorrect_bound_minimum": bool(approach_minimum < h2_result.fun - 1e-4),
        },
    }


def _molecule_preservation():
    existing_names = (
        "h3", "methane", "formaldehyde", "water", "ethane", "methanol",
        "hydroxylamine", "hydrogen_peroxide",
    )
    geometries = dict(zip(existing_names, _accepted_geometries()))
    geometries["h2"] = build_box.BUILDERS["H2"]()
    geometries["ammonia"] = build_box.BUILDERS["NH3"]()
    rows = []
    for name, (symbols, coordinates) in geometries.items():
        positions = np.asarray(coordinates, dtype=float) + 20.0
        _, reference_force, reference_energy = _evaluate(
            OptimisedValenceStateBatchedSimulation, symbols, positions
        )
        _, candidate_force, candidate_energy = _evaluate(
            UnifiedBondCapacityEnergyPrototype, symbols, positions
        )
        rows.append({
            "molecule": name,
            "energy_change_eV": float(candidate_energy - reference_energy),
            "maximum_force_change_eV_per_angstrom": float(
                torch.max(torch.abs(candidate_force - reference_force))
            ),
        })
    return rows


def _bond_order_states():
    ethane = bond_calibration.ethane_geometry(1.525)
    cases = {
        "ethane_C_C": (*ethane, 1.0),
        "formaldehyde_C_O": (
            ["C", "O", "H", "H"],
            [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0],
             [-0.55, 0.9526, 0.0], [-0.55, -0.9526, 0.0]],
            2.0,
        ),
        "nitrogen_N_N": (["N", "N"], [[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], 3.0),
    }
    result = {}
    for name, (symbols, coordinates, expected) in cases.items():
        simulation = _curve_simulation(
            UnifiedBondCapacityEnergyPrototype, symbols, coordinates
        )
        diagnostic = simulation._unified_diagnostics["boxes"][0]["heavy_bond_orders"]
        result[name] = {
            "expected_order": expected,
            "model_expected_order": diagnostic[0]["expected_order"],
            "bond_probability": diagnostic[0]["bond_probability"],
        }
    return result


def _force_symmetry():
    symbols, positions = _crowded_carbon(1.95)
    epsilon = 1e-5
    result = {}
    for name, model in (
        ("unified_radial", UnifiedBondCapacityEnergyPrototype),
        ("unified_topology", UnifiedBondCapacityTopologyPrototype),
    ):
        simulation, force, energy = _evaluate(model, symbols, positions)
        plus = positions.copy()
        minus = positions.copy()
        plus[-1, 0] += epsilon
        minus[-1, 0] -= epsilon
        plus_energy = _evaluate(model, symbols, plus)[2]
        minus_energy = _evaluate(model, symbols, minus)[2]
        numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
        permutation = np.asarray([0, 4, 2, 5, 1, 3])
        _, permuted_force, permuted_energy = _evaluate(
            model,
            [symbols[index] for index in permutation],
            positions[permutation],
        )
        result[name] = {
            "energy_eV": float(energy),
            "autograd_force_eV_per_angstrom": float(force[-1, 0]),
            "finite_difference_force_eV_per_angstrom": numerical,
            "force_difference_eV_per_angstrom": float(force[-1, 0]) - numerical,
            "permuted_energy_difference_eV": float(permuted_energy - energy),
            "permuted_force_max_difference_eV_per_angstrom": float(torch.max(
                torch.abs(permuted_force - force[torch.tensor(permutation)])
            )),
            "solver": simulation._unified_diagnostics["boxes"][0]["solver"],
        }
    return result


def _qm_microscopes(geometries_path, qm_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))["geometries"]
    qm = _load_qm(qm_path)
    raw = []
    for index, geometry in enumerate(geometries):
        row = {
            "geometry": geometry,
            "qm": qm[geometry["geometry_id"]],
            "energies": {},
        }
        for name, model in MODELS.items():
            row["energies"][name] = evaluate_qm(model, geometry, 30.0, "cpu")[0]
        raw.append(row)
        if (index + 1) % 25 == 0:
            print(f"QM microscopes: {index + 1}/{len(geometries)}", flush=True)
    references = {
        row["geometry"]["system"]: row for row in raw
        if row["geometry"]["sample_kind"] == "reactant_reference"
    }
    residuals = defaultdict(lambda: defaultdict(list))
    for row in raw:
        geometry = row["geometry"]
        if geometry["sample_kind"] != "dense_transfer_scan":
            continue
        reference = references[geometry["system"]]
        qm_relative = row["qm"] - reference["qm"]
        for name in MODELS:
            model_relative = row["energies"][name] - reference["energies"][name]
            residuals[geometry["system"]][name].append(qm_relative - model_relative)
    summary = {"systems": {}, "all_dense": {}}
    for system, models in sorted(residuals.items()):
        summary["systems"][system] = {
            name: _metrics(values) for name, values in models.items()
        }
    for name in MODELS:
        values = [
            value for models in residuals.values() for value in models[name]
        ]
        summary["all_dense"][name] = _metrics(values)
    return summary


def _nve(geometries_path, grambow_path):
    cases = nve_cases(geometries_path, grambow_path)
    selected = {
        "water_transfer": cases["water_transfer"],
        "grambow_crowded_reactant": cases["grambow_crowded_reactant"],
        "symmetric_preference_exchange": cases["symmetric_preference_exchange"],
    }
    result = {}
    for case_name, case in selected.items():
        result[case_name] = {}
        reference = build_nve(OptimisedValenceStateBatchedSimulation, case)
        initial_velocities = reference.velocities.detach().clone()
        for name, model in (
            ("production", OptimisedValenceStateBatchedSimulation),
            ("unified_radial", UnifiedBondCapacityEnergyPrototype),
        ):
            simulation = build_nve(model, case)
            simulation.velocities = initial_velocities.clone()
            result[case_name][name] = run_nve(simulation, case["steps"])
        print(f"NVE: {case_name}", flush=True)
    return result


def validate(geometries, qm, grambow, include_nve):
    result = {
        "h_curves": _h_curves(),
        "bond_order_states": _bond_order_states(),
        "molecule_preservation": _molecule_preservation(),
        "force_and_symmetry": _force_symmetry(),
        "qm_microscopes": _qm_microscopes(geometries, qm),
    }
    if include_nve:
        result["nve"] = _nve(geometries, grambow)
    water = result["qm_microscopes"]["systems"]["water"]["unified_radial"]
    result["gates"] = {
        "finite_and_force_consistent": abs(
            result["force_and_symmetry"]["unified_radial"]["force_difference_eV_per_angstrom"]
        ) < 1e-4,
        "permutation_symmetric": abs(
            result["force_and_symmetry"]["unified_radial"]["permuted_energy_difference_eV"]
        ) < 1e-8,
        "no_stable_symmetric_h3": not result["h_curves"]["h3_symmetric"]["incorrect_stable_h3"],
        "no_bound_h_plus_h2_complex": not result["h_curves"]["h_plus_h2_approach"]["incorrect_bound_minimum"],
        "molecules_preserved": max(
            abs(row["energy_change_eV"]) for row in result["molecule_preservation"]
        ) < 1e-6,
        "single_double_triple_states": all(
            abs(row["model_expected_order"] - row["expected_order"]) < 1e-6
            for row in result["bond_order_states"].values()
        ),
        "water_rmse_below_0_23_eV": water["rmse_eV"] < 0.23,
    }
    if include_nve:
        candidate_nve = [
            case["unified_radial"] for case in result["nve"].values()
        ]
        result["gates"].update({
            "nve_finite_no_move_caps": all(
                row["move_caps"] == 0
                and math.isfinite(row["max_absolute_drift_eV"])
                for row in candidate_nve
            ),
            "nve_max_drift_below_0_05_eV": max(
                row["max_absolute_drift_eV"] for row in candidate_nve
            ) < 0.05,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--grambow", type=Path, default=DEFAULT_GRAMBOW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-nve", action="store_true")
    args = parser.parse_args()
    result = validate(args.geometries, args.qm, args.grambow, not args.skip_nve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(result["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
