"""Fit observable-source and non-water response scales for hypotheses.

No Grambow quantity is read.  MBIS charges and total molecular dipoles define
the local source representation.  Formaldehyde plus alternating methane dense
points define the energy-response scales; the remaining methane points and all
H3/water points remain independent screens.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.electronic_state_correction.prototype import (
    numpy_electronic_features,
    smooth_contact_numpy,
)


DEBYE_PER_E_ANGSTROM = 4.80320471257
DEFAULT_MANIFEST = Path("research_data/electronic_observables/manifest.json")
DEFAULT_OBSERVABLES = Path("research_data/electronic_observables/observables.json")
DEFAULT_DENSE = Path("research_data/qm_residual/dense_scan_geometries.json")
DEFAULT_BASELINE = Path(
    "research_data/benchmark/diagnostics/directional_electronic_qm.csv"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/electronic_state_parameters.json"
)
DEFAULT_SCREEN = Path(
    "research_data/benchmark/diagnostics/electronic_state_hypothesis_screen.csv"
)


def _charge_design(symbols, coordinates, unknowns):
    positions = np.asarray(coordinates, dtype=float)
    design = np.zeros((len(symbols), len(unknowns)))
    index = {symbol: column for column, symbol in enumerate(unknowns)}
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            distance = np.linalg.norm(positions[second] - positions[first])
            taper = smooth_contact_numpy(
                distance, symbols[first], symbols[second]
            )
            if taper == 0.0:
                continue
            # q_i += taper * (z_j - z_i); q_j receives the opposite.
            for atom, other, sign in (
                (first, second, 1.0), (second, first, 1.0)
            ):
                if symbols[other] in index:
                    design[atom, index[symbols[other]]] += sign * taper
                if symbols[atom] in index:
                    design[atom, index[symbols[atom]]] -= sign * taper
    return design


def fit_source_values(manifest, observables):
    records = {row["geometry_id"]: row for row in observables["records"]}
    fit_rows = [
        row for row in manifest["geometries"] if row["role"] != "final_holdout"
    ]
    present = {symbol for row in fit_rows for symbol in row["symbols"]}
    unknowns = [symbol for symbol in ("C", "N", "O") if symbol in present]
    matrix, target = [], []
    for row in fit_rows:
        result = records[row["geometry_id"]]
        design = _charge_design(
            row["symbols"], row["coordinates_angstrom"], unknowns
        )
        # Partition-derived local supervision.
        matrix.extend((design / 0.10).tolist())
        target.extend((np.asarray(result["mbis_charges_e"]) / 0.10).tolist())
        # Stronger molecular observable; translation is harmless for neutral q.
        dipole_design = (
            np.asarray(row["coordinates_angstrom"]).T @ design
            * DEBYE_PER_E_ANGSTROM
        )
        matrix.extend((dipole_design / 0.20).tolist())
        target.extend((np.asarray(result["dipole_debye"]) / 0.20).tolist())
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    ridge = 1e-6
    solution = np.linalg.solve(
        matrix.T @ matrix + ridge * np.eye(len(unknowns)), matrix.T @ target
    )
    values = {"H": 0.0, "C": 0.0, "N": 0.0, "O": 0.0}
    values.update(dict(zip(unknowns, solution.tolist())))

    diagnostics = {}
    for role in ("characterisation", "validation", "final_holdout"):
        charge_errors, dipole_errors = [], []
        for row in manifest["geometries"]:
            if row["role"] != role:
                continue
            result = records[row["geometry_id"]]
            design = _charge_design(
                row["symbols"], row["coordinates_angstrom"], unknowns
            )
            predicted_q = design @ solution
            charge_errors.extend(
                (predicted_q - np.asarray(result["mbis_charges_e"])).tolist()
            )
            predicted_dipole = (
                np.asarray(row["coordinates_angstrom"]).T @ predicted_q
                * DEBYE_PER_E_ANGSTROM
            )
            dipole_errors.extend(
                (predicted_dipole - np.asarray(result["dipole_debye"])).tolist()
            )
        diagnostics[role] = {
            "charge_rmse_e": float(np.sqrt(np.mean(np.square(charge_errors)))),
            "dipole_component_rmse_debye": float(
                np.sqrt(np.mean(np.square(dipole_errors)))
            ),
        }
    return values, {
        "gauge": "H source fixed to zero",
        "fitted_elements": unknowns,
        "unidentifiable_elements": sorted(set(("C", "N", "O")) - set(unknowns)),
        "fit_roles": ["characterisation", "validation"],
        "targets": ["MBIS atomic charges", "total molecular dipole vector"],
        "ridge": ridge,
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "design_condition": float(np.linalg.cond(matrix)),
        "metrics": diagnostics,
    }


def _metrics(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mae_eV": float(np.abs(values).mean()),
        "rmse_eV": float(np.sqrt(np.mean(values * values))),
        "maximum_absolute_eV": float(np.abs(values).max()),
    }


def fit_energy_hypotheses(dense, baseline_rows, source_values):
    geometry = {row["geometry_id"]: row for row in dense["geometries"]}
    rows = []
    for row in baseline_rows:
        source = geometry[row["geometry_id"]]
        if source["sample_kind"] != "dense_transfer_scan":
            continue
        features = numpy_electronic_features(
            source["symbols"], source["coordinates_angstrom"], source_values
        )
        rows.append({
            "geometry_id": row["geometry_id"],
            "system": row["system"],
            "baseline_residual_eV": float(row["unified_radial_residual_eV"]),
            **features,
        })

    # The energy calibration is deliberately broader than one reaction while
    # retaining independent within-family and whole-family tests: all
    # formaldehyde points plus alternating methane points train; the other
    # methane points and every H3/water point remain unopened holdouts.
    train = [
        row for row in rows
        if row["system"] == "formaldehyde"
        or (
            row["system"] == "methane"
            and int(row["geometry_id"].rsplit("_", 1)[1]) % 2 == 0
        )
    ]
    feature_names = ["local_scalar", "polarisation_vector", "multipole_tensor"]
    hypotheses = {}
    for name, selected in [
        ("local_scalar", ["local_scalar"]),
        ("polarisation_vector", ["polarisation_vector"]),
        ("multipole_tensor", ["multipole_tensor"]),
        ("combined", feature_names),
    ]:
        matrix = np.asarray([[row[key] for key in selected] for row in train])
        target = np.asarray([row["baseline_residual_eV"] for row in train])
        ridge = 1e-8
        coefficients = np.linalg.solve(
            matrix.T @ matrix + ridge * np.eye(len(selected)),
            matrix.T @ target,
        )
        condition = float(np.linalg.cond(matrix)) if len(selected) > 1 else 1.0
        hypotheses[name] = {
            "coefficients_eV": dict(zip(selected, coefficients.tolist())),
            "energy_fit_systems": [
                "formaldehyde", "methane alternating even-index points"
            ],
            "energy_holdout_systems": [
                "h3", "methane alternating odd-index points", "water"
            ],
            "fit_observations": len(train),
            "parameter_count": len(selected),
            "ridge": ridge,
            "design_rank": int(np.linalg.matrix_rank(matrix)),
            "design_condition": condition,
            "metrics": {},
        }
        for system in sorted({row["system"] for row in rows}):
            errors = []
            for row in rows:
                if row["system"] != system:
                    continue
                correction = sum(
                    coefficient * row[key]
                    for key, coefficient in zip(selected, coefficients)
                )
                errors.append(row["baseline_residual_eV"] - correction)
            hypotheses[name]["metrics"][system] = _metrics(errors)

    output_rows = []
    for row in rows:
        output = dict(row)
        for name, result in hypotheses.items():
            correction = sum(
                result["coefficients_eV"].get(key, 0.0) * row[key]
                for key in feature_names
            )
            output[f"{name}_correction_eV"] = correction
            output[f"{name}_residual_eV"] = (
                row["baseline_residual_eV"] - correction
            )
        output_rows.append(output)
    return hypotheses, output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--observables", type=Path, default=DEFAULT_OBSERVABLES)
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    observables = json.loads(args.observables.read_text(encoding="utf-8"))
    dense = json.loads(args.dense.read_text(encoding="utf-8"))
    with args.baseline.open("r", newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    source_values, source_fit = fit_source_values(manifest, observables)
    hypotheses, screen_rows = fit_energy_hypotheses(
        dense, baseline_rows, source_values
    )
    payload = {
        "schema_version": 1,
        "scope": "research-only; no Grambow targets read",
        "source_values_e": source_values,
        "source_fit": source_fit,
        "hypotheses": hypotheses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.screen.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(screen_rows[0]))
        writer.writeheader()
        writer.writerows(screen_rows)
    print(json.dumps(payload, indent=2))
    print(f"screen: {args.screen}")


if __name__ == "__main__":
    main()
