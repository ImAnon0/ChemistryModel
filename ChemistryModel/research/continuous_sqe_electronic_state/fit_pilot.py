"""Fit and audit the frozen molecule-family-blocked C0 screening pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, erfc
import torch

from .parameters import C0_PARAMETERS, C0ParameterSet, ElementC0Parameters
from .prototype import C0ContinuousSQE


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research_data/electronic_observables/c0_pilot"
ELEMENTS = (1, 6, 7, 8)
CHI_ELEMENTS = (6, 7, 8)
SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def _vector(parameters):
    values = [parameters.elements[z].electronegativity_eV for z in CHI_ELEMENTS]
    values += [parameters.elements[z].intrinsic_hardness_eV for z in ELEMENTS]
    values += [parameters.elements[z].gaussian_sigma_A for z in ELEMENTS]
    values += [parameters.elements[z].transfer_capacity_e2_per_eV for z in ELEMENTS]
    values += [parameters.capacity_radius_scale, parameters.capacity_steepness]
    return np.asarray(values, dtype=float)


def _parameters(values, convention="continuous_sqe_c0_qm_pilot_v1"):
    values = np.asarray(values, dtype=float)
    cursor = 0
    chi = {1: 0.0}
    for z in CHI_ELEMENTS:
        chi[z] = float(values[cursor]); cursor += 1
    hardness = {}; sigma = {}; capacity = {}
    for target in (hardness, sigma, capacity):
        for z in ELEMENTS:
            target[z] = float(values[cursor]); cursor += 1
    elements = {
        z: ElementC0Parameters(
            chi[z], hardness[z], sigma[z], capacity[z],
            C0_PARAMETERS.elements[z].covalent_radius_A,
        )
        for z in ELEMENTS
    }
    return C0ParameterSet(
        elements=elements,
        capacity_radius_scale=float(values[cursor]),
        capacity_steepness=float(values[cursor + 1]),
        convention=convention,
    )


def _load():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    observed = json.loads((DATA / "observables.json").read_text(encoding="utf-8"))
    by_id = {row["geometry_id"]: row for row in observed["records"]}
    records = []
    for row in manifest["geometries"]:
        obs = by_id[row["geometry_id"]]
        if obs["status"] != "ok":
            raise RuntimeError(f"incomplete QM record: {row['geometry_id']}")
        records.append({**row, "qm": obs})
    return manifest, records


def _predict(record, values):
    params = _parameters(values)
    model = C0ContinuousSQE(params)
    positions = torch.tensor(record["coordinates_angstrom"], dtype=torch.float64)
    numbers = tuple(SYMBOL_TO_Z[s] for s in record["symbols"])
    result = model.evaluate(positions, numbers)
    return {
        "dipole": result.dipole_debye.detach().numpy(),
        "alpha": result.polarizability_A3.detach().numpy(),
        "charges": result.charges.detach().numpy(),
        "energy": float(result.energy.detach()),
        "minimum_eigenvalue": result.diagnostics.minimum_response_eigenvalue,
        "condition_number": result.diagnostics.response_condition_number,
        "residual": result.diagnostics.relative_solve_residual,
    }


def _predict_fast(record, values):
    """NumPy equivalent used only to remove Torch object overhead in fitting."""
    params = _parameters(values)
    positions = np.asarray(record["coordinates_angstrom"], dtype=float)
    numbers = [SYMBOL_TO_Z[s] for s in record["symbols"]]
    n = len(numbers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    incidence = np.zeros((n, len(pairs)))
    for edge, (i, j) in enumerate(pairs):
        incidence[i, edge] = 1.0; incidence[j, edge] = -1.0
    chi = np.array([params.elements[z].electronegativity_eV for z in numbers])
    intrinsic = np.array([params.elements[z].intrinsic_hardness_eV for z in numbers])
    sigma = np.array([params.elements[z].gaussian_sigma_A for z in numbers])
    elemental_capacity = np.array([params.elements[z].transfer_capacity_e2_per_eV for z in numbers])
    radii = np.array([params.elements[z].covalent_radius_A for z in numbers])
    i = np.array([p[0] for p in pairs], dtype=int); j = np.array([p[1] for p in pairs], dtype=int)
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    argument = params.capacity_steepness * (
        distances / (params.capacity_radius_scale * (radii[i] + radii[j])) - 1.0
    )
    shape = 0.5 * erfc(argument)
    coordinate = np.clip(
        (distances - params.support_inner_A) / (params.support_outer_A - params.support_inner_A),
        0.0, 1.0,
    )
    support = 1.0 - coordinate**3 * (10.0 - 15.0 * coordinate + 6.0 * coordinate**2)
    sqrt_c = np.sqrt(np.sqrt(elemental_capacity[i] * elemental_capacity[j])) * np.sqrt(shape) * support
    displacement = positions[:, None, :] - positions[None, :, :]
    distance_matrix = np.linalg.norm(displacement, axis=-1)
    sigma2 = sigma[:, None] ** 2 + sigma[None, :] ** 2
    denominator = np.sqrt(2.0 * sigma2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb = params.coulomb_eV_A_per_e2 * erf(distance_matrix / denominator) / distance_matrix
    diagonal = params.coulomb_eV_A_per_e2 * np.sqrt(2.0 / np.pi) / np.sqrt(sigma2)
    coulomb[np.diag_indices(n)] = diagonal[np.diag_indices(n)]
    hardness = np.diag(intrinsic) + coulomb
    transfer_map = incidence * sqrt_c[None, :]
    response = np.eye(len(pairs)) + transfer_map.T @ hardness @ transfer_map
    u = np.linalg.solve(response, -(transfer_map.T @ chi))
    charges = incidence @ (sqrt_c * u)
    alpha = positions.T @ transfer_map @ np.linalg.solve(response, transfer_map.T @ positions)
    alpha *= params.coulomb_eV_A_per_e2
    return {
        "dipole": np.sum(charges[:, None] * positions, axis=0) * 4.80320471257,
        "alpha": alpha,
        "charges": charges,
        "esp_au": np.sum(
            charges[None, :] * 0.529177210903
            / np.linalg.norm(
                np.asarray(record["qm"]["external_potential"]["points_angstrom"])[:, None, :]
                - positions[None, :, :], axis=2,
            ), axis=1,
        ),
    }


def _family_counts(records):
    counts = {}
    for record in records:
        counts[record["family"]] = counts.get(record["family"], 0) + 1
    return counts


def _residual(values, records, seed):
    counts = _family_counts(records)
    output = []
    # Observable weights follow the preregistered pilot roles. Localisation and
    # smoothness are enforced as independent gates. MBIS remains deliberately
    # weak, and zero-field QM energy is never fitted.
    for record in records:
        prediction = _predict_fast(record, values)
        qm = record["qm"]
        family_weight = 1.0 / counts[record["family"]]
        dipole = prediction["dipole"] - np.asarray(qm["dipole_debye"])
        alpha = prediction["alpha"] - np.asarray(qm["polarizability"]["tensor_angstrom3"])
        charges = prediction["charges"] - np.asarray(qm["mbis_charges_e"])
        esp = prediction["esp_au"] - np.asarray(qm["external_potential"]["potential_au"])
        output.extend((dipole * np.sqrt(0.25 * family_weight / 3.0) / 1.0).tolist())
        output.extend((alpha * np.sqrt(0.25 * family_weight / 9.0) / 1.0).ravel().tolist())
        output.extend((esp * np.sqrt(0.15 * family_weight / len(esp)) / 0.005).tolist())
        output.extend((charges * np.sqrt(0.10 * family_weight / len(charges)) / 0.25).tolist())
    scale = np.maximum(np.abs(seed), np.array([1,1,1, *([5]*4), *([.4]*4), *([.05]*4), 1, 1]))
    output.extend((np.sqrt(0.05 / len(seed)) * (values - seed) / scale).tolist())
    return np.asarray(output)


def _metrics(records, values):
    rows = []
    for record in records:
        pred = _predict(record, values)
        qm = record["qm"]
        qmd = np.asarray(qm["dipole_debye"])
        qma = np.asarray(qm["polarizability"]["tensor_angstrom3"])
        qmq = np.asarray(qm["mbis_charges_e"])
        esp_points = np.asarray(qm["external_potential"]["points_angstrom"])
        positions = np.asarray(record["coordinates_angstrom"])
        esp_pred = np.sum(
            pred["charges"][None, :] * 0.529177210903
            / np.linalg.norm(esp_points[:, None, :] - positions[None, :, :], axis=2), axis=1,
        )
        esp_qm = np.asarray(qm["external_potential"]["potential_au"])
        rows.append({
            "geometry_id": record["geometry_id"], "family": record["family"],
            "split": record["split"],
            "dipole_vector_sq": float(np.mean((pred["dipole"] - qmd) ** 2)),
            "dipole_magnitude_abs": float(abs(np.linalg.norm(pred["dipole"]) - np.linalg.norm(qmd))),
            "alpha_tensor_sq": float(np.mean((pred["alpha"] - qma) ** 2)),
            "alpha_isotropic_abs": float(abs(np.trace(pred["alpha"] - qma) / 3.0)),
            "mbis_charge_sq": float(np.mean((pred["charges"] - qmq) ** 2)),
            "external_potential_sq": float(np.mean((esp_pred - esp_qm) ** 2)),
            "minimum_eigenvalue": pred["minimum_eigenvalue"],
            "condition_number": pred["condition_number"],
            "solve_residual": pred["residual"],
        })
    return {
        "count": len(rows),
        "dipole_vector_rmse_D": float(np.sqrt(np.mean([r["dipole_vector_sq"] for r in rows]))),
        "dipole_magnitude_mae_D": float(np.mean([r["dipole_magnitude_abs"] for r in rows])),
        "polarizability_tensor_rmse_A3": float(np.sqrt(np.mean([r["alpha_tensor_sq"] for r in rows]))),
        "polarizability_isotropic_mae_A3": float(np.mean([r["alpha_isotropic_abs"] for r in rows])),
        "mbis_charge_rmse_e": float(np.sqrt(np.mean([r["mbis_charge_sq"] for r in rows]))),
        "external_potential_rmse_au": float(np.sqrt(np.mean([r["external_potential_sq"] for r in rows]))),
        "minimum_response_eigenvalue": float(min(r["minimum_eigenvalue"] for r in rows)),
        "maximum_condition_number": float(max(r["condition_number"] for r in rows)),
        "maximum_solve_residual": float(max(r["solve_residual"] for r in rows)),
        "rows": rows,
    }


def _objective_score(records, values, seed):
    residual = _residual(values, records, seed)
    return float(np.mean(residual * residual))


def _parameter_payload(parameters):
    return {
        "convention": parameters.convention,
        "independent_parameter_count": parameters.independent_parameter_count,
        "electronegativity_gauge": "H = 0 eV",
        "elements": {
            str(z): {
                "electronegativity_eV": p.electronegativity_eV,
                "intrinsic_hardness_eV": p.intrinsic_hardness_eV,
                "gaussian_sigma_A": p.gaussian_sigma_A,
                "transfer_capacity_e2_per_eV": p.transfer_capacity_e2_per_eV,
                "covalent_radius_A_fixed": p.covalent_radius_A,
            } for z, p in parameters.elements.items()
        },
        "capacity_radius_scale": parameters.capacity_radius_scale,
        "capacity_steepness": parameters.capacity_steepness,
        "support_inner_A_fixed": parameters.support_inner_A,
        "support_outer_A_fixed": parameters.support_outer_A,
    }


def _smoothness(values):
    model = C0ContinuousSQE(_parameters(values))
    energies, charges = [], []
    for distance in np.linspace(0.75, 5.0, 426):
        result = model.evaluate(
            torch.tensor([[0, 0, 0], [distance, 0, 0]], dtype=torch.float64),
            (8, 1),
        )
        energies.append(float(result.energy)); charges.append(float(result.charges[0]))
    return {
        "oh_grid_points": len(energies),
        "maximum_adjacent_energy_step_eV": float(np.max(np.abs(np.diff(energies)))),
        "maximum_adjacent_charge_step_e": float(np.max(np.abs(np.diff(charges)))),
        "charge_at_4p5_A_e": charges[np.argmin(np.abs(np.linspace(.75,5,426)-4.5))],
        "charge_at_5_A_e": charges[-1],
        "all_finite": bool(np.isfinite(energies).all() and np.isfinite(charges).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=350)
    args = parser.parse_args()
    manifest, records = _load()
    train = [r for r in records if r["split"] == "training"]
    validation = [r for r in records if r["split"] == "validation"]
    holdout = [r for r in records if r["split"] == "locked_holdout"]
    seed = _vector(C0_PARAMETERS)
    lower = np.array([-5,-5,-5, *([0.5]*4), *([0.2]*4), *([0.002]*4), 0.7, 0.2])
    upper = np.array([10,10,10, *([30.0]*4), *([1.5]*4), *([1.0]*4), 2.5, 8.0])
    rng = np.random.default_rng(20260822)
    candidates = []
    starts = [seed]
    for _ in range(args.starts - 1):
        perturb = np.exp(rng.normal(0.0, 0.28, len(seed)))
        trial = np.clip(seed * perturb, lower + 1e-8, upper - 1e-8)
        trial[:3] = np.clip(seed[:3] + rng.normal(0, 0.7, 3), lower[:3], upper[:3])
        starts.append(trial)
    for index, start in enumerate(starts):
        fitted = least_squares(
            _residual, start, bounds=(lower, upper), args=(train, seed),
            max_nfev=args.max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        validation_score = _objective_score(validation, fitted.x, seed)
        candidates.append((validation_score, fitted))
        print(f"start {index + 1}/{len(starts)} train={np.mean(fitted.fun**2):.8g} validation={validation_score:.8g}")
    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1]
    values = selected.x
    singular = np.linalg.svd(selected.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(selected.jac))
    parameter_payload = _parameter_payload(_parameters(values))
    parameter_payload["seed_vector"] = seed.tolist()
    parameter_payload["fitted_vector"] = values.tolist()
    parameter_payload["bounds"] = {"lower": lower.tolist(), "upper": upper.tolist()}
    parameter_payload["selected_multistart_validation_score"] = candidates[0][0]
    (DATA / "fitted_parameters.json").write_text(
        json.dumps(parameter_payload, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "research_only": True,
        "production_integration": False,
        "manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        "locked_holdout_sha256": manifest["locked_holdout_sha256"],
        "objective": {
            "dipole": 0.25, "polarizability": 0.25, "external_potential": 0.15,
            "mbis_proxy": 0.10, "parameter_regularisation": 0.05,
            "fragment_localisation_gate_not_fit": 0.15,
            "reactive_smoothness_gate_not_fit": 0.05,
            "qm_energy_weight": 0.0,
            "normalisation": {
                "dipole_D": 1.0, "polarizability_A3": 1.0,
                "external_potential_au": 0.005, "mbis_charge_e": 0.25,
            },
        },
        "split_families": manifest["splits"],
        "initial": {split: _metrics(group, seed) for split, group in
                    (("training", train), ("validation", validation), ("locked_holdout", holdout))},
        "fitted": {split: _metrics(group, values) for split, group in
                   (("training", train), ("validation", validation), ("locked_holdout", holdout))},
        "identifiability": {
            "jacobian_shape": list(selected.jac.shape), "numerical_rank": rank,
            "largest_singular_value": float(singular[0]),
            "smallest_singular_value": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] else float("inf"),
        },
        "optimisation": {
            "starts": len(starts), "selected_nfev": selected.nfev,
            "selected_training_score": float(np.mean(selected.fun**2)),
            "selected_validation_score": candidates[0][0],
            "all_validation_scores": [float(item[0]) for item in candidates],
        },
        "smoothness_and_localisation": _smoothness(values),
        "known_structural_limit": "Atom-only split charges cannot reproduce transverse/intra-atomic polarizability.",
    }
    (DATA / "pilot_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    concise = {phase: {split: {k: v for k, v in metrics.items() if k != "rows"}
                       for split, metrics in values_by_split.items()}
               for phase, values_by_split in (("initial", report["initial"]), ("fitted", report["fitted"]))}
    print(json.dumps({"metrics": concise, "identifiability": report["identifiability"],
                      "smoothness": report["smoothness_and_localisation"]}, indent=2))


if __name__ == "__main__":
    main()
