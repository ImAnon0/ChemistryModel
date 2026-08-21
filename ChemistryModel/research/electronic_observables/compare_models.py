"""Compare a frozen candidate with baseline and QM relative energies.

Input model JSON files use a deliberately small interchange schema::

    {"records": [{"geometry_id": "...", "energy_eV": 1.23, ...}]}

Optional force and dipole arrays are compared when present in all three files.
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

from research.electronic_observables.prototype import (
    ElectronicObservableRecord,
    cancellation_warning,
    compare_candidate_rows,
    vector_rmse,
)


def _records(payload, manifest):
    metadata = {row["geometry_id"]: row for row in manifest["geometries"]}
    return [ElectronicObservableRecord(
        geometry_id=row["geometry_id"],
        family=metadata[row["geometry_id"]]["family"],
        reference_geometry_id=metadata[row["geometry_id"]]["reference_geometry_id"],
        energy_eV=float(row["energy_eV"]),
        force_eV_per_angstrom=tuple(tuple(v) for v in row.get("force_eV_per_angstrom", [])),
        dipole_debye=(None if "dipole_debye" not in row else tuple(row["dipole_debye"])),
    ) for row in payload["records"] if row.get("core_status", "ok") == "ok"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research_data/electronic_observables/manifest.json"))
    parser.add_argument("--qm", type=Path, default=Path("research_data/electronic_observables/observables.json"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in [args.qm, args.baseline, args.candidate]]
    qm, baseline, candidate = [_records(payload, manifest) for payload in payloads]
    comparisons = compare_candidate_rows(qm, baseline, candidate)
    qm_by_id = {row.geometry_id: row for row in qm}
    baseline_by_id = {row.geometry_id: row for row in baseline}
    candidate_by_id = {row.geometry_id: row for row in candidate}

    detail_rows = []
    for comparison in comparisons:
        geometry_id = comparison.geometry_id
        detail = dict(comparison.__dict__)
        force_sets = [
            qm_by_id[geometry_id].force_eV_per_angstrom,
            baseline_by_id[geometry_id].force_eV_per_angstrom,
            candidate_by_id[geometry_id].force_eV_per_angstrom,
        ]
        if all(force_sets):
            detail["baseline_force_rmse_eV_per_angstrom"] = vector_rmse(
                force_sets[1], force_sets[0]
            )
            detail["candidate_force_rmse_eV_per_angstrom"] = vector_rmse(
                force_sets[2], force_sets[0]
            )
        dipole_sets = [
            qm_by_id[geometry_id].dipole_debye,
            baseline_by_id[geometry_id].dipole_debye,
            candidate_by_id[geometry_id].dipole_debye,
        ]
        if all(value is not None for value in dipole_sets):
            detail["baseline_dipole_error_debye"] = float(np.linalg.norm(
                np.asarray(dipole_sets[1]) - np.asarray(dipole_sets[0])
            ))
            detail["candidate_dipole_error_debye"] = float(np.linalg.norm(
                np.asarray(dipole_sets[2]) - np.asarray(dipole_sets[0])
            ))
        detail_rows.append(detail)

    by_family = defaultdict(list)
    for row in detail_rows:
        by_family[row["family"]].append(row)
    families = {}
    for family, rows in sorted(by_family.items()):
        base = np.asarray([row["baseline_residual_eV"] for row in rows])
        new = np.asarray([row["candidate_residual_eV"] for row in rows])
        families[family] = {
            "count": len(rows),
            "baseline_mae_eV": float(np.abs(base).mean()),
            "candidate_mae_eV": float(np.abs(new).mean()),
            "improved": sum(row["classification"] == "improved" for row in rows),
            "regressed": sum(row["classification"] == "regressed" for row in rows),
        }
        if all("baseline_force_rmse_eV_per_angstrom" in row for row in rows):
            families[family]["baseline_force_rmse_eV_per_angstrom"] = float(np.sqrt(np.mean([
                row["baseline_force_rmse_eV_per_angstrom"] ** 2 for row in rows
            ])))
            families[family]["candidate_force_rmse_eV_per_angstrom"] = float(np.sqrt(np.mean([
                row["candidate_force_rmse_eV_per_angstrom"] ** 2 for row in rows
            ])))
    report = {
        "schema_version": 1,
        "cancellation_warning": cancellation_warning(comparisons),
        "families": families,
        "most_improved": [
            row.geometry_id for row in sorted(
                (item for item in comparisons if item.classification == "improved"),
                key=lambda x: -x.absolute_improvement_eV,
            )[:10]
        ],
        "most_regressed": [
            row.geometry_id for row in sorted(
                (item for item in comparisons if item.classification == "regressed"),
                key=lambda x: x.absolute_improvement_eV,
            )[:10]
        ],
        "rows": detail_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in detail_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"report: {args.output}")
    print(f"table: {csv_path}")
    print(f"cancellation warning: {report['cancellation_warning']}")


if __name__ == "__main__":
    main()
