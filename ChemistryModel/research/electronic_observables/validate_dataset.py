"""Quality-control and summarize an electronic-observable result file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_MANIFEST = Path("research_data/electronic_observables/manifest.json")
DEFAULT_RESULTS = Path("research_data/electronic_observables/observables.json")
DEFAULT_REPORT = Path("research_data/electronic_observables/quality_report.json")
DEFAULT_CSV = Path("research_data/electronic_observables/summary.csv")


def quality_checks(manifest, results):
    geometry = {row["geometry_id"]: row for row in manifest["geometries"]}
    records = {row["geometry_id"]: row for row in results["records"]}
    checks = []

    def add(geometry_id, check, value, tolerance, passed, detail=""):
        checks.append({
            "geometry_id": geometry_id,
            "check": check,
            "value": float(value),
            "tolerance": float(tolerance),
            "passed": bool(passed),
            "detail": detail,
        })

    for geometry_id, row in records.items():
        if row.get("core_status") != "ok":
            checks.append({
                "geometry_id": geometry_id,
                "check": "core_calculation",
                "value": 1.0,
                "tolerance": 0.0,
                "passed": False,
                "detail": row.get("error", "failed"),
            })
            continue
        source = geometry[geometry_id]
        charge = float(source["charge"])
        if row.get("s2") is not None:
            spin = 0.5 * (int(source["multiplicity"]) - 1)
            expected_s2 = spin * (spin + 1.0)
            contamination = abs(float(row["s2"]) - expected_s2)
            add(
                geometry_id,
                "spin_contamination",
                contamination,
                5e-2,
                contamination <= 5e-2,
                f"expected <S^2>={expected_s2:.6f}",
            )
        for scheme, key in [
            ("mulliken", "mulliken_charges_e"),
            ("lowdin", "lowdin_charges_e"),
            ("mbis", "mbis_charges_e"),
        ]:
            residual = abs(sum(row[key]) - charge)
            add(geometry_id, f"{scheme}_charge_conservation", residual, 5e-5, residual <= 5e-5)

        force = np.asarray(row["force_eV_per_angstrom"], dtype=float)
        translation_residual = float(np.linalg.norm(force.sum(axis=0)))
        # DFT quadrature/grid noise can leave a small translational residual
        # even with analytic gradients.  This gate detects meaningful force
        # corruption without pretending the numerical grid is exact.
        add(geometry_id, "net_force", translation_residual, 2e-2, translation_residual <= 2e-2)

        dipole_error = float(row["mbis_dipole_reconstruction_error_au"])
        add(geometry_id, "mbis_dipole_reconstruction", dipole_error, 2e-5, dipole_error <= 2e-5)

        if row.get("polarizability_status") == "ok":
            polar = row["polarizability"]
            antisymmetric = float(polar["antisymmetric_norm_au"])
            minimum = min(polar["eigenvalues_au"])
            add(geometry_id, "polarizability_symmetry", antisymmetric, 1e-3, antisymmetric <= 1e-3)
            add(geometry_id, "polarizability_positive", minimum, 0.0, minimum > 0.0)

    for geometry_id in ["h2_equilibrium", "methane_equilibrium", "n2_equilibrium"]:
        row = records.get(geometry_id)
        if row and row.get("core_status") == "ok":
            magnitude = float(row["dipole_magnitude_debye"])
            add(geometry_id, "symmetry_zero_dipole", magnitude, 2e-4, magnitude <= 2e-4)

    missing = sorted(set(geometry) - set(records))
    status_counts = Counter(row.get("status", "missing") for row in records.values())
    return {
        "schema_version": 1,
        "geometry_count": len(geometry),
        "result_count": len(records),
        "missing_geometry_ids": missing,
        "status_counts": dict(status_counts),
        "check_count": len(checks),
        "failed_check_count": sum(not row["passed"] for row in checks),
        "all_available_checks_pass": all(row["passed"] for row in checks),
        "complete": not missing and all(row.get("status") == "ok" for row in records.values()),
        "checks": checks,
    }


def write_summary_csv(path, manifest, results):
    geometry = {row["geometry_id"]: row for row in manifest["geometries"]}
    fields = [
        "geometry_id", "family", "role", "status", "energy_eV",
        "dipole_debye", "polarizability_isotropic_angstrom3", "s2",
        "wall_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results["records"]:
            source = geometry[row["geometry_id"]]
            writer.writerow({
                "geometry_id": row["geometry_id"],
                "family": source["family"],
                "role": source["role"],
                "status": row["status"],
                "energy_eV": row.get("energy_eV", ""),
                "dipole_debye": row.get("dipole_magnitude_debye", ""),
                "polarizability_isotropic_angstrom3": row.get("polarizability", {}).get("isotropic_angstrom3", ""),
                "s2": row.get("s2", ""),
                "wall_seconds": row.get("wall_seconds", ""),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = quality_checks(manifest, results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(args.csv, manifest, results)
    print(f"results: {report['result_count']}/{report['geometry_count']}")
    print(f"failed checks: {report['failed_check_count']}/{report['check_count']}")
    print(f"complete: {report['complete']}")
    print(f"report: {args.report}")
    print(f"summary: {args.csv}")


if __name__ == "__main__":
    main()
