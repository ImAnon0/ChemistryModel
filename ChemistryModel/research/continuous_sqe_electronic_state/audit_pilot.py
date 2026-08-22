"""Independent numerical gates for the fitted research-only C0 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .fit_pilot import DATA, _load, _parameters, _predict, SYMBOL_TO_Z
from .prototype import C0ContinuousSQE


def _fragment_case(model, left_symbols, left, right_symbols, right, separation):
    right = np.asarray(right, dtype=float) + np.array([separation, 0.0, 0.0])
    positions = torch.tensor([*left, *right.tolist()], dtype=torch.float64)
    numbers = tuple(SYMBOL_TO_Z[s] for s in [*left_symbols, *right_symbols])
    result = model.evaluate(positions, numbers)
    split = len(left_symbols)
    return {
        "separation_A": separation,
        "left_net_charge_e": float(result.charges[:split].sum()),
        "right_net_charge_e": float(result.charges[split:].sum()),
        "total_charge_e": float(result.charges.sum()),
        "energy_eV": float(result.energy),
    }


def main():
    parameter_data = json.loads((DATA / "fitted_parameters.json").read_text(encoding="utf-8"))
    values = np.asarray(parameter_data["fitted_vector"], dtype=float)
    parameters = _parameters(values)
    model = C0ContinuousSQE(parameters)
    _, records = _load()
    maximum_charge_residual = 0.0
    minimum_eigenvalue = float("inf")
    maximum_condition = 0.0
    for record in records:
        prediction = _predict(record, values)
        maximum_charge_residual = max(
            maximum_charge_residual, abs(float(np.sum(prediction["charges"])))
        )
        minimum_eigenvalue = min(minimum_eigenvalue, prediction["minimum_eigenvalue"])
        maximum_condition = max(maximum_condition, prediction["condition_number"])

    oh = _fragment_case(
        model, ["O", "H"], [[0, 0, 0], [0.97, 0, 0]], ["H"], [[0, 0, 0]], 8.0
    )
    water = [[0, 0, 0], [0.9572, 0, 0], [-0.239, 0.927, 0]]
    ammonia = [[0, 0, 0.116], [0, 0.938, -0.273], [0.812, -0.469, -0.273], [-0.812, -0.469, -0.273]]
    multi = [
        _fragment_case(model, ["O", "H", "H"], water, ["N", "H", "H", "H"], ammonia, distance)
        for distance in (8.0, 20.0, 100.0)
    ]
    h2_record = next(record for record in records if record["geometry_id"] == "h2_equilibrium")
    h2 = _predict(h2_record, values)
    h2_qm = np.asarray(h2_record["qm"]["polarizability"]["tensor_angstrom3"])
    lower = np.asarray(parameter_data["bounds"]["lower"])
    upper = np.asarray(parameter_data["bounds"]["upper"])
    output = {
        "all_qm_records_ok": all(record["qm"]["status"] == "ok" for record in records),
        "configuration_count": len(records),
        "maximum_total_charge_residual_e": maximum_charge_residual,
        "minimum_response_eigenvalue": minimum_eigenvalue,
        "maximum_response_condition_number": maximum_condition,
        "parameters_at_lower_bound": np.flatnonzero(np.isclose(values, lower, atol=1e-7)).tolist(),
        "parameters_at_upper_bound": np.flatnonzero(np.isclose(values, upper, atol=1e-7)).tolist(),
        "separated_OH_plus_H": oh,
        "separated_water_plus_ammonia": multi,
        "h2_polarizability_A3": {
            "qm": h2_qm.tolist(), "c0": h2["alpha"].tolist(),
            "qm_eigenvalues": np.linalg.eigvalsh(h2_qm).tolist(),
            "c0_eigenvalues": np.linalg.eigvalsh(h2["alpha"]).tolist(),
        },
        "decision_gates": {
            "positive_response": minimum_eigenvalue >= 1.0 - 1e-10,
            "condition_below_1e6": maximum_condition < 1e6,
            "charge_conservation_below_1e-12": maximum_charge_residual < 1e-12,
            "no_bound_hits": not np.any(np.isclose(values, lower, atol=1e-7))
                             and not np.any(np.isclose(values, upper, atol=1e-7)),
        },
    }
    (DATA / "stability_audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
