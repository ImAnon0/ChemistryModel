"""Reproducible gates for the research-only unified geometry formulations."""

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
from research.unified_bond_geometry import (
    PostSolvedWeightedGeometryPrototype,
    VariationalElectronDomainGeometryPrototype,
    VariationalJointGeometryStatePrototype,
    VariationalWeightedGeometryPrototype,
)
from valence_state_optimised_torch import OptimisedValenceStateBatchedSimulation


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_QM = Path("research_data/qm_residual/dense_scan_qm.csv")
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/unified_bond_geometry_validation.json"
)
DEFAULT_CSV = Path(
    "research_data/benchmark/diagnostics/unified_bond_geometry_qm.csv"
)

MODELS = {
    "production": OptimisedValenceStateBatchedSimulation,
    "unified_radial": UnifiedBondCapacityEnergyPrototype,
    "post_weighted": PostSolvedWeightedGeometryPrototype,
    "variational_weighted": VariationalWeightedGeometryPrototype,
    "electron_domain": VariationalElectronDomainGeometryPrototype,
    "joint_local_state": VariationalJointGeometryStatePrototype,
}
CANDIDATES = tuple(
    name for name in MODELS if name not in ("production", "unified_radial")
)


def _model_energy(model, geometry, box_size=40.0):
    coordinates = np.asarray(geometry["coordinates_angstrom"], dtype=float)
    coordinates -= coordinates.mean(axis=0)
    coordinates += box_size / 2.0
    _, force, energy = _evaluate(model, geometry["symbols"], coordinates)
    force_max = float(torch.linalg.vector_norm(force, dim=1).max())
    if not math.isfinite(float(energy)) or not math.isfinite(force_max):
        raise ValueError("non-finite energy or force")
    return float(energy), force_max


def _qm_microscopes(geometries_path, qm_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))[
        "geometries"
    ]
    qm = _load_qm(qm_path)
    rows = []
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
            energy, force_max = _model_energy(model, geometry)
            row[f"{name}_energy_eV"] = energy
            row[f"{name}_force_max_eV_per_angstrom"] = force_max
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

    summary = {"evaluated_geometries": len(rows), "systems": {}, "all_dense": {}}
    for system, values_by_model in sorted(residuals.items()):
        summary["systems"][system] = {
            name: _metrics(values) for name, values in values_by_model.items()
        }
    for name in MODELS:
        values = [
            residual
            for values_by_model in residuals.values()
            for residual in values_by_model[name]
        ]
        summary["all_dense"][name] = _metrics(values)
    return rows, summary


def _molecule_invariance():
    names = (
        "h3", "methane", "formaldehyde", "water", "ethane", "methanol",
        "hydroxylamine", "hydrogen_peroxide",
    )
    geometries = dict(zip(names, _accepted_geometries()))
    geometries["h2"] = build_box.BUILDERS["H2"]()
    geometries["ammonia"] = build_box.BUILDERS["NH3"]()
    result = {}
    for molecule, (symbols, coordinates) in geometries.items():
        positions = np.asarray(coordinates, dtype=float) + 20.0
        _, reference_force, reference_energy = _evaluate(
            UnifiedBondCapacityEnergyPrototype, symbols, positions
        )
        result[molecule] = {}
        for name in CANDIDATES:
            _, force, energy = _evaluate(MODELS[name], symbols, positions)
            result[molecule][name] = {
                "energy_change_from_radial_eV": float(energy - reference_energy),
                "maximum_force_change_from_radial_eV_per_angstrom": float(
                    torch.max(torch.abs(force - reference_force))
                ),
            }
    return result


def _force_and_permutation(geometries_path):
    geometries = json.loads(geometries_path.read_text(encoding="utf-8"))[
        "geometries"
    ]
    geometry = min(
        (
            item for item in geometries
            if item["system"] == "water"
            and item["sample_kind"] == "dense_transfer_scan"
        ),
        key=lambda item: abs(
            float(item["reaction_coordinate"]["transfer_distance_angstrom"])
            - 1.16
        ),
    )
    symbols = geometry["symbols"]
    positions = np.asarray(geometry["coordinates_angstrom"], dtype=float) + 15.0
    epsilon = 1e-5
    permutation = np.arange(len(symbols))
    oxygen = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    if len(oxygen) >= 2:
        permutation[oxygen[0]], permutation[oxygen[1]] = (
            permutation[oxygen[1]], permutation[oxygen[0]]
        )
    result = {}
    for name in MODELS:
        _, force, energy = _evaluate(MODELS[name], symbols, positions)
        plus = positions.copy()
        minus = positions.copy()
        plus[-1, 0] += epsilon
        minus[-1, 0] -= epsilon
        plus_energy = _evaluate(MODELS[name], symbols, plus)[2]
        minus_energy = _evaluate(MODELS[name], symbols, minus)[2]
        numerical = -(float(plus_energy) - float(minus_energy)) / (2.0 * epsilon)
        _, permuted_force, permuted_energy = _evaluate(
            MODELS[name],
            [symbols[index] for index in permutation],
            positions[permutation],
        )
        result[name] = {
            "autograd_force_eV_per_angstrom": float(force[-1, 0]),
            "finite_difference_force_eV_per_angstrom": numerical,
            "force_difference_eV_per_angstrom": float(force[-1, 0]) - numerical,
            "permuted_energy_difference_eV": float(permuted_energy - energy),
            "permuted_force_max_difference_eV_per_angstrom": float(
                torch.max(torch.abs(
                    permuted_force - force[torch.as_tensor(permutation)]
                ))
            ),
        }
    return result


def _cutoff_continuity():
    hydrogen = R.ELEMENT_INDEX["H"]
    oxygen = R.ELEMENT_INDEX["O"]
    cutoff = float(R.CUTOFF_OUTER[oxygen, hydrogen])
    delta = 1e-4
    result = {"O_H_outer_cutoff_angstrom": cutoff, "models": {}}
    for name, model in MODELS.items():
        samples = []
        for distance in (cutoff - delta, cutoff, cutoff + delta):
            symbols = ["O", "H", "H"]
            positions = np.asarray(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
                 [0.5 * distance, 0.866025403784 * distance, 0.0]],
                dtype=float,
            ) + 15.0
            _, force, energy = _evaluate(model, symbols, positions)
            samples.append({
                "distance_angstrom": distance,
                "energy_eV": float(energy),
                "moving_atom_radial_force_eV_per_angstrom": float(
                    torch.dot(
                        force[2],
                        torch.as_tensor(
                            [0.5, 0.866025403784, 0.0], dtype=force.dtype
                        ),
                    )
                ),
            })
        result["models"][name] = {
            "samples": samples,
            "energy_span_eV": max(row["energy_eV"] for row in samples)
            - min(row["energy_eV"] for row in samples),
            "maximum_absolute_force_eV_per_angstrom": max(
                abs(row["moving_atom_radial_force_eV_per_angstrom"])
                for row in samples
            ),
        }
    return result


def _matched_water_nve(geometries_path):
    grambow = Path("research_data/benchmark/grambow_endpoints.json")
    case = nve_cases(geometries_path, grambow)["water_transfer"]
    reference = build_nve(UnifiedBondCapacityEnergyPrototype, case)
    initial_velocities = reference.velocities.detach().clone()
    result = {}
    for name in ("unified_radial", *CANDIDATES):
        simulation = build_nve(MODELS[name], case)
        simulation.velocities = initial_velocities.clone()
        try:
            result[name] = run_nve(simulation, case["steps"])
            result[name]["status"] = (
                "pass"
                if result[name]["move_caps"] == 0
                and result[name]["max_absolute_drift_eV"] < 0.05
                else "fail_gate"
            )
        except Exception as error:  # diagnostic must preserve a scientific failure
            result[name] = {
                "status": "fail",
                "error": f"{type(error).__name__}: {error}",
            }
        print(f"NVE water transfer: {name}", flush=True)
    return {
        "case": "matched water transfer",
        "steps": case["steps"],
        "time_step_fs": case["dt"],
        "models": result,
        "crowded_joint_state_case": (
            "not run: local joint state count is exponential and the static "
            "water gate already failed"
        ),
    }


def validate(geometries_path, qm_path, include_nve=True):
    rows, microscopes = _qm_microscopes(geometries_path, qm_path)
    molecules = _molecule_invariance()
    force = _force_and_permutation(geometries_path)
    cutoff = _cutoff_continuity()
    radial_water = microscopes["systems"]["water"]["unified_radial"]["rmse_eV"]
    gates = {}
    for name in CANDIDATES:
        molecule_rows = [entry[name] for entry in molecules.values()]
        gates[name] = {
            "finite_difference_force": abs(
                force[name]["force_difference_eV_per_angstrom"]
            ) < 5e-4,
            "permutation_symmetry": abs(
                force[name]["permuted_energy_difference_eV"]
            ) < 1e-8 and force[name][
                "permuted_force_max_difference_eV_per_angstrom"
            ] < 1e-5,
            "cutoff_continuity": cutoff["models"][name]["energy_span_eV"] < 1e-3
            and cutoff["models"][name][
                "maximum_absolute_force_eV_per_angstrom"
            ] < 1e-2,
            "accepted_molecule_invariance": max(
                abs(row["energy_change_from_radial_eV"])
                for row in molecule_rows
            ) < 1e-6 and max(
                row["maximum_force_change_from_radial_eV_per_angstrom"]
                for row in molecule_rows
            ) < 1e-5,
            "water_not_worse_than_radial": microscopes["systems"]["water"][name][
                "rmse_eV"
            ] <= radial_water + 1e-6,
        }
        gates[name]["passes_static_and_qm_gates"] = all(gates[name].values())
    nve = _matched_water_nve(geometries_path) if include_nve else {
        "status": "skipped by command line"
    }
    if include_nve:
        for name in CANDIDATES:
            gates[name]["water_nve_drift_below_0_05_and_no_caps"] = (
                nve["models"][name]["status"] == "pass"
            )
            gates[name]["passes_all_evaluated_gates"] = (
                gates[name]["passes_static_and_qm_gates"]
                and gates[name]["water_nve_drift_below_0_05_and_no_caps"]
            )
    result = {
        "scope": "research only; no production physics modified",
        "models": list(MODELS),
        "qm_microscopes": microscopes,
        "molecule_invariance": molecules,
        "force_and_permutation": force,
        "cutoff_continuity": cutoff,
        "gates": gates,
        "frozen_grambow_context": {
            "production": {
                "barrier_mae_eV": 4.520,
                "barrier_rmse_eV": 6.483,
                "barrier_sign_agreement_percent": 87.0,
                "reaction_mae_eV": 4.406,
                "reaction_rmse_eV": 7.069,
            },
            "unified_radial": {
                "barrier_mae_eV": 1.475,
                "barrier_rmse_eV": 1.858,
                "barrier_sign_agreement_percent": 99.5,
                "reaction_mae_eV": 2.233,
                "reaction_rmse_eV": 2.872,
            },
            "candidate_metrics": "not evaluated because the water QM gate failed",
        },
        "joint_state_scaling": {
            "formula": "product over incident edges of (maximum_order + 1)",
            "binary_H_contacts": {
                str(degree): 2 ** degree for degree in (2, 3, 4, 5, 6, 7, 8)
            },
            "order_0_to_3_heavy_contacts": {
                str(degree): 4 ** degree for degree in (2, 3, 4, 5, 6, 7, 8)
            },
            "prototype_state_limit": VariationalJointGeometryStatePrototype.max_local_geometry_states,
        },
        "dynamics_validation": nve,
        "downstream_validation": {
            "nve": "matched water-transfer NVE run; no candidate promoted",
            "grambow": "not run: every candidate failed the water QM gate",
        },
    }
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--skip-nve", action="store_true")
    args = parser.parse_args()
    rows, result = validate(args.geometries, args.qm, not args.skip_nve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "qm": result["qm_microscopes"],
        "gates": result["gates"],
        "dynamics": result["dynamics_validation"],
        "downstream_validation": result["downstream_validation"],
    }, indent=2))
    if any(
        row["passes_static_and_qm_gates"] for row in result["gates"].values()
    ):
        raise SystemExit("a candidate passed; run NVE and Grambow before decision")


if __name__ == "__main__":
    main()
