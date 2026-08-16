"""
Validate the local-component H-state correction.

This is intentionally NOT an equivalence test against the old whole-box
H-state everywhere, because the whole-box formulation has a demonstrated
size-consistency bug.

Checks
------
1. Distant H3 separability:
       E(A+B) == E(A) + E(B)
       forces on A/B unchanged

2. 106-point dense QM microscope:
       compare historical whole-box H-state vs local-component H-state
       after the SAME per-system reactant-reference alignment used by
       compare_hstate_qm_dense.py.

3. Report:
       dense-transfer MAE/RMSE/max residual by system
       maximum old->component relative-energy change by system
       worst adjacent residual step
       product-reference relative energies

No parameter is fitted or changed.

Run:
    py validate_h_state_components.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from h_state_torch import HStateReferenceBatchedSimulation
from h_state_component_torch import HStateComponentBatchedSimulation


GEOMETRIES = Path(
    "research_data/qm_residual/dense_scan_geometries.json"
)

QM_RESULTS = Path(
    "research_data/qm_residual/dense_scan_qm.csv"
)

OUTPUT = Path(
    "research_data/qm_residual/dense_scan_hstate_components.csv"
)

BOX_SIZE = 30.0


def rmse(values):
    return math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def load_qm():
    rows = {}

    with QM_RESULTS.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rows[row["geometry_id"]] = float(
                    row["qm_energy_eV"]
                )

    return rows


def evaluate(
    model_class,
    symbols,
    coordinates,
    box_size=BOX_SIZE,
):
    simulation = model_class(
        boxes=[
            (
                list(symbols),
                np.asarray(
                    coordinates,
                    dtype=float,
                ),
            )
        ],
        box_size=box_size,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    energy = float(
        simulation.potential_per_box[0]
    )

    forces = (
        simulation.forces
        .detach()
        .cpu()
        .numpy()
        .reshape(
            simulation.per_box,
            3,
        )
    )

    diagnostics = getattr(
        simulation,
        "_h_component_diagnostics",
        None,
    )

    return energy, forces, diagnostics


# ----------------------------------------------------------------------
# Size consistency
# ----------------------------------------------------------------------

H3_SPACING_A = 0.90
H3_CLUSTER_SEPARATION_A = 15.0
SEPARABILITY_BOX = 40.0


def h3_geometry(origin):
    origin = np.asarray(
        origin,
        dtype=float,
    )

    return (
        ["H", "H", "H"],
        np.array([
            [0.0, 0.0, 0.0],
            [H3_SPACING_A, 0.0, 0.0],
            [2.0 * H3_SPACING_A, 0.0, 0.0],
        ]) + origin,
    )


def separability_result(model_class):
    single_symbols, single_positions = (
        h3_geometry(
            [8.0, 8.0, 8.0]
        )
    )

    second_symbols, second_positions = (
        h3_geometry(
            [
                8.0,
                8.0 + H3_CLUSTER_SEPARATION_A,
                8.0,
            ]
        )
    )

    double_symbols = (
        single_symbols
        + second_symbols
    )

    double_positions = np.concatenate(
        [
            single_positions,
            second_positions,
        ],
        axis=0,
    )

    single_energy, single_forces, _ = (
        evaluate(
            model_class,
            single_symbols,
            single_positions,
            box_size=SEPARABILITY_BOX,
        )
    )

    double_energy, double_forces, diagnostics = (
        evaluate(
            model_class,
            double_symbols,
            double_positions,
            box_size=SEPARABILITY_BOX,
        )
    )

    return {
        "single_energy": single_energy,
        "double_energy": double_energy,
        "nonadditivity": (
            double_energy
            - 2.0 * single_energy
        ),
        "force_error_a": float(
            np.max(
                np.abs(
                    double_forces[:3]
                    - single_forces
                )
            )
        ),
        "force_error_b": float(
            np.max(
                np.abs(
                    double_forces[3:]
                    - single_forces
                )
            )
        ),
        "diagnostics": diagnostics,
    }


def print_separability(name, result):
    print(name)

    print(
        f"  E(single)                 "
        f"{result['single_energy']:+.12f} eV"
    )

    print(
        f"  E(two distant)            "
        f"{result['double_energy']:+.12f} eV"
    )

    print(
        f"  E(double)-2E(single)      "
        f"{result['nonadditivity']:+.12e} eV"
    )

    print(
        f"  max |dF| cluster A        "
        f"{result['force_error_a']:.12e} eV/A"
    )

    print(
        f"  max |dF| cluster B        "
        f"{result['force_error_b']:.12e} eV/A"
    )

    if result["diagnostics"] is not None:
        print(
            "  component diagnostics     "
            f"{result['diagnostics']}"
        )

    print()


# ----------------------------------------------------------------------
# Dense QM microscope
# ----------------------------------------------------------------------

def dense_comparison():
    payload = json.loads(
        GEOMETRIES.read_text(
            encoding="utf-8"
        )
    )

    geometries = payload["geometries"]
    qm = load_qm()

    missing = [
        geometry["geometry_id"]
        for geometry in geometries
        if geometry["geometry_id"] not in qm
    ]

    if missing:
        raise RuntimeError(
            f"QM missing {len(missing)} rows; first={missing[0]}"
        )

    raw = []

    print("DENSE QM MICROSCOPE")
    print(
        f"geometries : {len(geometries)}"
    )
    print(
        "models     : historical whole-box H-state, "
        "local-component H-state"
    )
    print(
        "device     : cpu / float64"
    )
    print()

    for index, geometry in enumerate(
        geometries,
        start=1,
    ):
        gid = geometry["geometry_id"]
        symbols = geometry["symbols"]
        coordinates = geometry[
            "coordinates_angstrom"
        ]

        old_energy, old_force, _ = evaluate(
            HStateReferenceBatchedSimulation,
            symbols,
            coordinates,
        )

        component_energy, component_force, diagnostics = (
            evaluate(
                HStateComponentBatchedSimulation,
                symbols,
                coordinates,
            )
        )

        rc = geometry.get(
            "reaction_coordinate",
            {},
        )

        raw.append({
            "geometry_id": gid,
            "system": geometry["system"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "transfer_distance_angstrom": rc.get(
                "transfer_distance_angstrom",
                "",
            ),
            "donor_distance_angstrom": rc.get(
                "donor_distance_angstrom",
                "",
            ),
            "qm_energy_eV": qm[gid],
            "old_hstate_energy_eV": old_energy,
            "component_hstate_energy_eV": component_energy,
            "raw_component_minus_old_eV": (
                component_energy
                - old_energy
            ),
            "old_force_max_eV_per_angstrom": float(
                np.max(
                    np.linalg.norm(
                        old_force,
                        axis=1,
                    )
                )
            ),
            "component_force_max_eV_per_angstrom": float(
                np.max(
                    np.linalg.norm(
                        component_force,
                        axis=1,
                    )
                )
            ),
            "component_count": (
                diagnostics[
                    "component_counts_per_box"
                ][0]
                if diagnostics is not None
                else ""
            ),
            "largest_component_edges": (
                diagnostics[
                    "largest_component_edges"
                ]
                if diagnostics is not None
                else ""
            ),
        })

        if (
            index == 1
            or index % 20 == 0
            or index == len(geometries)
        ):
            print(
                f"[{index:3d}/{len(geometries):3d}] "
                f"{gid:<38s} "
                f"old={old_energy:+.6f}  "
                f"local={component_energy:+.6f} eV"
            )

    refs = {}

    for row in raw:
        if row["sample_kind"] != "reactant_reference":
            continue

        refs[row["system"]] = {
            "qm": row["qm_energy_eV"],
            "old": row["old_hstate_energy_eV"],
            "component": row[
                "component_hstate_energy_eV"
            ],
        }

    final = []

    for row in raw:
        ref = refs[row["system"]]

        qm_relative = (
            row["qm_energy_eV"]
            - ref["qm"]
        )

        old_relative = (
            row["old_hstate_energy_eV"]
            - ref["old"]
        )

        component_relative = (
            row["component_hstate_energy_eV"]
            - ref["component"]
        )

        merged = dict(row)

        merged.update({
            "qm_relative_eV": qm_relative,
            "old_hstate_relative_eV": old_relative,
            "component_hstate_relative_eV": component_relative,
            "old_hstate_residual_eV": (
                qm_relative
                - old_relative
            ),
            "component_hstate_residual_eV": (
                qm_relative
                - component_relative
            ),
            "relative_component_minus_old_eV": (
                component_relative
                - old_relative
            ),
        })

        final.append(merged)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                final[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(final)

    return final


def print_dense_summary(final):
    dense = defaultdict(list)

    for row in final:
        if row["sample_kind"] == "dense_transfer_scan":
            dense[row["system"]].append(row)

    print()
    print(
        f"wrote      : {OUTPUT}"
    )
    print()

    print("DENSE TRANSFER RESIDUALS VS QM")
    print(
        f"{'system':14s} "
        f"{'old MAE':>9s} {'old RMSE':>9s} {'old max':>9s} "
        f"{'local MAE':>10s} {'local RMSE':>10s} {'local max':>10s}"
    )

    for system in sorted(dense):
        rows = dense[system]

        old = [
            row["old_hstate_residual_eV"]
            for row in rows
        ]

        local = [
            row["component_hstate_residual_eV"]
            for row in rows
        ]

        print(
            f"{system:14s} "
            f"{statistics.fmean(abs(v) for v in old):9.4f} "
            f"{rmse(old):9.4f} "
            f"{max(abs(v) for v in old):9.4f} "
            f"{statistics.fmean(abs(v) for v in local):10.4f} "
            f"{rmse(local):10.4f} "
            f"{max(abs(v) for v in local):10.4f}"
        )

    print()
    print("MODEL CHANGE AFTER REACTANT ALIGNMENT")

    for system in sorted(dense):
        values = [
            row["relative_component_minus_old_eV"]
            for row in dense[system]
        ]

        print(
            f"{system:14s} "
            f"mean |local-old|="
            f"{statistics.fmean(abs(v) for v in values):.6f} eV  "
            f"max={max(abs(v) for v in values):.6f} eV"
        )

    print()
    print("WORST ADJACENT RESIDUAL STEP")

    for system in sorted(dense):
        rows = sorted(
            dense[system],
            key=lambda row: float(
                row["transfer_distance_angstrom"]
            ),
        )

        for column, label in (
            (
                "old_hstate_residual_eV",
                "old",
            ),
            (
                "component_hstate_residual_eV",
                "local",
            ),
        ):
            pairs = []

            for left, right in zip(
                rows,
                rows[1:],
            ):
                jump = (
                    right[column]
                    - left[column]
                )

                pairs.append(
                    (
                        abs(jump),
                        jump,
                        left,
                        right,
                    )
                )

            if not pairs:
                continue

            _, jump, left, right = max(
                pairs,
                key=lambda item: item[0],
            )

            print(
                f"{system:14s} "
                f"{label:6s} "
                f"{float(left['transfer_distance_angstrom']):.3f}->"
                f"{float(right['transfer_distance_angstrom']):.3f} A  "
                f"dResidual={jump:+.6f} eV"
            )

    print()
    print("PRODUCT REFERENCE RELATIVE ENERGIES")
    print(
        f"{'system':14s} "
        f"{'QM':>9s} "
        f"{'old H':>9s} "
        f"{'local H':>9s}"
    )

    systems = sorted({
        row["system"]
        for row in final
    })

    for system in systems:
        products = [
            row
            for row in final
            if (
                row["system"] == system
                and row["sample_kind"]
                == "product_reference"
            )
        ]

        if len(products) != 1:
            continue

        row = products[0]

        print(
            f"{system:14s} "
            f"{row['qm_relative_eV']:+9.4f} "
            f"{row['old_hstate_relative_eV']:+9.4f} "
            f"{row['component_hstate_relative_eV']:+9.4f}"
        )


def main():
    print("LOCAL-COMPONENT H-STATE VALIDATION")
    print()

    historical = separability_result(
        HStateReferenceBatchedSimulation
    )

    local = separability_result(
        HStateComponentBatchedSimulation
    )

    print_separability(
        "HISTORICAL WHOLE-BOX H-STATE",
        historical,
    )

    print_separability(
        "LOCAL-COMPONENT H-STATE",
        local,
    )

    separability_pass = (
        abs(
            local["nonadditivity"]
        ) < 1e-10
        and local["force_error_a"] < 1e-9
        and local["force_error_b"] < 1e-9
    )

    print(
        "SIZE CONSISTENCY: "
        + (
            "PASS"
            if separability_pass
            else "FAIL"
        )
    )

    if not separability_pass:
        raise SystemExit(
            "local-component H-state did not restore separability"
        )

    print()
    final = dense_comparison()
    print_dense_summary(final)


if __name__ == "__main__":
    main()
