"""Reduce raw electronic observables into auditable family-level signals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyse(manifest, results):
    records = {row["geometry_id"]: row for row in results["records"]}
    families = defaultdict(list)
    rows = []
    for source in manifest["geometries"]:
        row = records[source["geometry_id"]]
        reference = records[source["reference_geometry_id"]]
        eigenvalues = np.asarray(row["polarizability"]["eigenvalues_au"])
        isotropic = float(eigenvalues.mean())
        reduced = {
            "geometry_id": source["geometry_id"],
            "family": source["family"],
            "role": source["role"],
            "relative_energy_eV": float(row["energy_eV"] - reference["energy_eV"]),
            "dipole_magnitude_debye": float(row["dipole_magnitude_debye"]),
            "polarizability_isotropic_angstrom3": float(row["polarizability"]["isotropic_angstrom3"]),
            "polarizability_anisotropy_fraction": float(
                (eigenvalues.max() - eigenvalues.min()) / isotropic
            ),
            "mbis_charge_span_e": float(
                max(row["mbis_charges_e"]) - min(row["mbis_charges_e"])
            ),
        }
        rows.append(reduced)
        families[source["family"]].append(reduced)

    family_summary = {}
    for family, family_rows in sorted(families.items()):
        summary = {"count": len(family_rows)}
        for key in [
            "relative_energy_eV", "dipole_magnitude_debye",
            "polarizability_isotropic_angstrom3",
            "polarizability_anisotropy_fraction", "mbis_charge_span_e",
        ]:
            values = np.asarray([row[key] for row in family_rows])
            summary[key] = {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        family_summary[family] = summary

    return {
        "schema_version": 1,
        "interpretation_policy": (
            "Signals establish information content only; they are not fitted "
            "parameters and do not imply a production functional form."
        ),
        "families": family_summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research_data/electronic_observables/manifest.json"))
    parser.add_argument("--results", type=Path, default=Path("research_data/electronic_observables/observables.json"))
    parser.add_argument("--output", type=Path, default=Path("research_data/electronic_observables/observable_analysis.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    output = analyse(manifest, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"families: {len(output['families'])}")
    print(f"rows: {len(output['rows'])}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
