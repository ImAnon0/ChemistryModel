"""Diagnostics for the research-only continuous SQE reference solve."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


@dataclass(frozen=True)
class C0Diagnostics:
    atom_count: int
    edge_count: int
    active_edge_count: int
    minimum_response_eigenvalue: float
    maximum_response_eigenvalue: float
    response_condition_number: float
    minimum_hardness_eigenvalue: float
    hardness_condition_number: float
    relative_solve_residual: float
    total_charge_residual_e: float
    maximum_absolute_charge_e: float
    maximum_absolute_transfer_e: float
    minimum_compliance_e2_per_eV: float
    maximum_compliance_e2_per_eV: float
    all_finite: bool

    def as_dict(self):
        return asdict(self)


def build_diagnostics(
    response_matrix,
    hardness_matrix,
    rhs,
    scaled_transfers,
    transfers,
    charges,
    compliance,
):
    detached_matrix = response_matrix.detach()
    eigenvalues = torch.linalg.eigvalsh(detached_matrix)
    hardness_eigenvalues = torch.linalg.eigvalsh(hardness_matrix.detach())
    minimum = eigenvalues[0]
    maximum = eigenvalues[-1]
    residual = detached_matrix @ scaled_transfers.detach() - rhs.detach()
    residual_scale = max(float(torch.linalg.vector_norm(rhs.detach())), 1.0)
    active = compliance.detach() > 0.0
    finite_values = (
        torch.isfinite(response_matrix).all()
        & torch.isfinite(scaled_transfers).all()
        & torch.isfinite(transfers).all()
        & torch.isfinite(charges).all()
        & torch.isfinite(compliance).all()
    )
    return C0Diagnostics(
        atom_count=int(charges.numel()),
        edge_count=int(compliance.numel()),
        active_edge_count=int(torch.count_nonzero(active)),
        minimum_response_eigenvalue=float(minimum),
        maximum_response_eigenvalue=float(maximum),
        response_condition_number=float(maximum / minimum),
        minimum_hardness_eigenvalue=float(hardness_eigenvalues[0]),
        hardness_condition_number=float(
            hardness_eigenvalues[-1] / hardness_eigenvalues[0]
        ),
        relative_solve_residual=float(torch.linalg.vector_norm(residual))
        / residual_scale,
        total_charge_residual_e=abs(float(torch.sum(charges.detach()))),
        maximum_absolute_charge_e=float(torch.max(torch.abs(charges.detach()))),
        maximum_absolute_transfer_e=float(torch.max(torch.abs(transfers.detach()))),
        minimum_compliance_e2_per_eV=(
            float(torch.min(compliance.detach()[active]))
            if bool(torch.any(active))
            else 0.0
        ),
        maximum_compliance_e2_per_eV=float(torch.max(compliance.detach())),
        all_finite=bool(finite_values),
    )


def _observe(model, coordinates, atomic_numbers):
    positions = torch.tensor(coordinates, dtype=torch.float64)
    result = model.evaluate(positions, atomic_numbers, calculate_forces=True)
    return {
        "energy_eV": float(result.energy.detach()),
        "charges_e": result.charges.detach().tolist(),
        "dipole_e_A": result.dipole_e_A.detach().tolist(),
        "maximum_force_eV_per_A": float(torch.max(torch.abs(result.forces))),
        "diagnostics": result.diagnostics.as_dict(),
    }


def run_reference_diagnostics(geometry_path):
    from research.electrostatics_diagnostics import known_diagnostic_cases

    from .prototype import C0ContinuousSQE

    model = C0ContinuousSQE()
    cases = dict(known_diagnostic_cases())
    cases["H3"] = (
        [[-0.74, 0.0, 0.0], [0.0, 0.0, 0.0], [0.92, 0.0, 0.0]],
        (1, 1, 1),
    )
    simple = {
        name: _observe(model, coordinates, atomic_numbers)
        for name, (coordinates, atomic_numbers) in cases.items()
    }

    separation = []
    for distance in (0.8, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0, 100.0):
        observed = _observe(
            model,
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            (8, 1),
        )
        separation.append(
            {
                "distance_A": distance,
                "oxygen_charge_e": observed["charges_e"][0],
                "energy_eV": observed["energy_eV"],
                "minimum_response_eigenvalue": observed["diagnostics"][
                    "minimum_response_eigenvalue"
                ],
                "condition_number": observed["diagnostics"][
                    "response_condition_number"
                ],
            }
        )

    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}
    geometries = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    grid_rows = []
    for geometry in geometries["geometries"]:
        if geometry["sample_kind"] != "grid":
            continue
        observed = _observe(
            model,
            geometry["coordinates_angstrom"],
            [atomic_numbers[symbol] for symbol in geometry["symbols"]],
        )
        coordinate = geometry["reaction_coordinate"]
        grid_rows.append(
            {
                "system": geometry["system"],
                "donor": float(coordinate["donor_distance_angstrom"]),
                "transfer": float(coordinate["transfer_distance_angstrom"]),
                "energy": observed["energy_eV"],
                "charges": observed["charges_e"],
                "minimum_eigenvalue": observed["diagnostics"][
                    "minimum_response_eigenvalue"
                ],
                "condition": observed["diagnostics"]["response_condition_number"],
            }
        )

    grouped = defaultdict(list)
    for row in grid_rows:
        grouped[row["system"]].append(row)
    smoothness = {}
    for system, rows in sorted(grouped.items()):
        adjacent = []
        for varying, fixed in (("donor", "transfer"), ("transfer", "donor")):
            for fixed_value in sorted({row[fixed] for row in rows}):
                line = sorted(
                    (row for row in rows if row[fixed] == fixed_value),
                    key=lambda row: row[varying],
                )
                for left, right in zip(line, line[1:]):
                    adjacent.append(
                        (
                            abs(right["energy"] - left["energy"]),
                            max(
                                abs(a - b)
                                for a, b in zip(left["charges"], right["charges"])
                            ),
                        )
                    )
        smoothness[system] = {
            "geometry_count": len(rows),
            "maximum_adjacent_energy_change_eV": max(value[0] for value in adjacent),
            "maximum_adjacent_atomic_charge_change_e": max(
                value[1] for value in adjacent
            ),
            "minimum_response_eigenvalue": min(row["minimum_eigenvalue"] for row in rows),
            "maximum_condition_number": max(row["condition"] for row in rows),
            "all_values_finite": all(
                math.isfinite(value)
                for row in rows
                for value in (row["energy"], *row["charges"])
            ),
        }

    all_diagnostics = [case["diagnostics"] for case in simple.values()]
    return {
        "model_id": model.model_id,
        "parameter_convention": model.parameters.convention,
        "parameter_count": model.parameters.independent_parameter_count,
        "simple_systems": simple,
        "oh_separation": separation,
        "reaction_grid_smoothness": smoothness,
        "gate_summary": {
            "all_simple_finite": all(item["all_finite"] for item in all_diagnostics),
            "minimum_simple_response_eigenvalue": min(
                item["minimum_response_eigenvalue"] for item in all_diagnostics
            ),
            "maximum_simple_condition_number": max(
                item["response_condition_number"] for item in all_diagnostics
            ),
            "asymptotic_oh_charge_e": abs(separation[-1]["oxygen_charge_e"]),
            "all_reaction_grids_finite": all(
                item["all_values_finite"] for item in smoothness.values()
            ),
            "minimum_reaction_grid_eigenvalue": min(
                item["minimum_response_eigenvalue"] for item in smoothness.values()
            ),
            "maximum_reaction_grid_condition_number": max(
                item["maximum_condition_number"] for item in smoothness.values()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("research_data/qm_residual/geometries.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/continuous_sqe_electronic_state/c0_diagnostics.json"
        ),
    )
    args = parser.parse_args()
    report = run_reference_diagnostics(args.geometries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
