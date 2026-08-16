"""Build and audit the first QM residual-energy dataset.

Inputs
------
research_data/qm_residual/geometries_qm.json
research_data/qm_residual/base_results.csv
research_data/qm_residual/qm_results.csv

Output
------
research_data/qm_residual/qm_residual_dataset.csv

The target is NOT raw QM energy minus ChemistryModel energy.  Each system is
aligned to its own reactant-reference geometry first:

    base_rel(g) = E_base(g) - E_base(reactant_reference)
    qm_rel(g)   = E_qm(g)   - E_qm(reactant_reference)

    residual_target(g) = qm_rel(g) - base_rel(g)

This removes the unrelated absolute energy zeros while preserving surface
shape, barriers, and reaction energies.

The script also audits structured-grid smoothness before any ML is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/geometries_qm.json")
DEFAULT_BASE = Path("research_data/qm_residual/base_results.csv")
DEFAULT_QM = Path("research_data/qm_residual/qm_results.csv")
DEFAULT_OUTPUT = Path("research_data/qm_residual/qm_residual_dataset.csv")


def load_geometries(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("geometries", [])
    if not rows:
        raise ValueError(f"{path}: no geometries")

    by_id = {}
    for row in rows:
        gid = row["geometry_id"]
        if gid in by_id:
            raise ValueError(f"duplicate geometry id: {gid}")
        by_id[gid] = row
    return payload, by_id


def load_csv(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gid = row["geometry_id"]
            if gid in rows:
                raise ValueError(f"{path}: duplicate geometry id {gid}")
            rows[gid] = row
    return rows


def as_float(value, name: str, gid: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{gid}: invalid {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{gid}: non-finite {name}")
    return result


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def build_rows(geometry_by_id, base_by_id, qm_by_id):
    selected_ids = list(geometry_by_id)

    missing_base = [gid for gid in selected_ids if gid not in base_by_id]
    missing_qm = [gid for gid in selected_ids if gid not in qm_by_id]
    if missing_base:
        raise ValueError(f"missing {len(missing_base)} base rows")
    if missing_qm:
        raise ValueError(f"missing {len(missing_qm)} QM rows")

    bad_qm = [
        gid for gid in selected_ids
        if qm_by_id[gid].get("status") != "ok"
    ]
    if bad_qm:
        raise ValueError(
            f"{len(bad_qm)} QM rows are not status=ok; first: {bad_qm[0]}"
        )

    reference_by_system = {}
    for gid, geometry in geometry_by_id.items():
        if geometry["sample_kind"] == "reactant_reference":
            system = geometry["system"]
            if system in reference_by_system:
                raise ValueError(f"{system}: multiple reactant references")
            reference_by_system[system] = gid

    systems = sorted({row["system"] for row in geometry_by_id.values()})
    missing_refs = [s for s in systems if s not in reference_by_system]
    if missing_refs:
        raise ValueError(f"missing reactant refs: {missing_refs}")

    references = {}
    for system, ref_gid in reference_by_system.items():
        references[system] = {
            "geometry_id": ref_gid,
            "base_eV": as_float(
                base_by_id[ref_gid]["base_energy_eV"],
                "base_energy_eV",
                ref_gid,
            ),
            "qm_eV": as_float(
                qm_by_id[ref_gid]["qm_energy_eV"],
                "qm_energy_eV",
                ref_gid,
            ),
        }

    merged = []
    for gid, geometry in geometry_by_id.items():
        base = base_by_id[gid]
        qm = qm_by_id[gid]
        system = geometry["system"]
        ref = references[system]

        base_e = as_float(base["base_energy_eV"], "base_energy_eV", gid)
        qm_e = as_float(qm["qm_energy_eV"], "qm_energy_eV", gid)

        base_rel = base_e - ref["base_eV"]
        qm_rel = qm_e - ref["qm_eV"]
        residual = qm_rel - base_rel

        rc = geometry.get("reaction_coordinate", {})
        merged.append({
            "geometry_id": gid,
            "system": system,
            "split": geometry["split"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "donor_distance_angstrom": rc.get(
                "donor_distance_angstrom", ""
            ),
            "transfer_distance_angstrom": rc.get(
                "transfer_distance_angstrom", ""
            ),
            "base_energy_eV": base_e,
            "qm_energy_eV": qm_e,
            "base_reference_eV": ref["base_eV"],
            "qm_reference_eV": ref["qm_eV"],
            "base_relative_eV": base_rel,
            "qm_relative_eV": qm_rel,
            "residual_target_eV": residual,
            "base_force_max_eV_per_angstrom": as_float(
                base["base_force_max_eV_per_angstrom"],
                "base_force_max_eV_per_angstrom",
                gid,
            ),
        })

    return merged, references


FIELDNAMES = [
    "geometry_id",
    "system",
    "split",
    "sample_kind",
    "region",
    "donor_distance_angstrom",
    "transfer_distance_angstrom",
    "base_energy_eV",
    "qm_energy_eV",
    "base_reference_eV",
    "qm_reference_eV",
    "base_relative_eV",
    "qm_relative_eV",
    "residual_target_eV",
    "base_force_max_eV_per_angstrom",
]


def write_output(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def adjacent_grid_jumps(rows: list[dict]) -> list[dict]:
    """Return adjacent residual jumps along either grid coordinate."""
    grid = [row for row in rows if row["sample_kind"] == "grid"]
    by_system = defaultdict(list)
    for row in grid:
        by_system[row["system"]].append(row)

    jumps = []
    for system, group in by_system.items():
        coords = {}
        for row in group:
            donor = float(row["donor_distance_angstrom"])
            transfer = float(row["transfer_distance_angstrom"])
            coords[(donor, transfer)] = row

        donors = sorted({coord[0] for coord in coords})
        transfers = sorted({coord[1] for coord in coords})

        # Along transfer at fixed donor.
        for donor in donors:
            line = [
                coords[(donor, transfer)]
                for transfer in transfers
                if (donor, transfer) in coords
            ]
            line.sort(key=lambda row: float(row["transfer_distance_angstrom"]))
            for left, right in zip(line, line[1:]):
                delta = (
                    float(right["residual_target_eV"])
                    - float(left["residual_target_eV"])
                )
                dx = (
                    float(right["transfer_distance_angstrom"])
                    - float(left["transfer_distance_angstrom"])
                )
                jumps.append({
                    "system": system,
                    "axis": "transfer",
                    "fixed": donor,
                    "from_id": left["geometry_id"],
                    "to_id": right["geometry_id"],
                    "delta_residual_eV": delta,
                    "step_angstrom": dx,
                    "abs_slope_eV_per_angstrom": abs(delta / dx),
                })

        # Along donor at fixed transfer.
        for transfer in transfers:
            line = [
                coords[(donor, transfer)]
                for donor in donors
                if (donor, transfer) in coords
            ]
            line.sort(key=lambda row: float(row["donor_distance_angstrom"]))
            for left, right in zip(line, line[1:]):
                delta = (
                    float(right["residual_target_eV"])
                    - float(left["residual_target_eV"])
                )
                dx = (
                    float(right["donor_distance_angstrom"])
                    - float(left["donor_distance_angstrom"])
                )
                jumps.append({
                    "system": system,
                    "axis": "donor",
                    "fixed": transfer,
                    "from_id": left["geometry_id"],
                    "to_id": right["geometry_id"],
                    "delta_residual_eV": delta,
                    "step_angstrom": dx,
                    "abs_slope_eV_per_angstrom": abs(delta / dx),
                })

    return jumps


def print_summary(rows, references):
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    print("DATASET SUMMARY")
    print(
        f"{'system':14s} {'split':8s} {'N':>4s} "
        f"{'res MAE':>9s} {'res RMSE':>10s} "
        f"{'res min':>9s} {'res max':>9s} {'max |res|':>10s}"
    )
    for system in sorted(by_system):
        group = by_system[system]
        residuals = [float(row["residual_target_eV"]) for row in group]
        split = group[0]["split"]
        print(
            f"{system:14s} {split:8s} {len(group):4d} "
            f"{statistics.fmean(abs(v) for v in residuals):9.4f} "
            f"{rmse(residuals):10.4f} "
            f"{min(residuals):+9.4f} "
            f"{max(residuals):+9.4f} "
            f"{max(abs(v) for v in residuals):10.4f}"
        )

    print()
    print("PRODUCT-REFERENCE RELATIVE ENERGIES")
    print(
        f"{'system':14s} {'base/eV':>10s} {'QM/eV':>10s} "
        f"{'QM-base':>10s}"
    )
    for system in sorted(by_system):
        products = [
            row for row in by_system[system]
            if row["sample_kind"] == "product_reference"
        ]
        if len(products) != 1:
            print(f"{system:14s} expected one product reference")
            continue
        row = products[0]
        print(
            f"{system:14s} "
            f"{float(row['base_relative_eV']):+10.4f} "
            f"{float(row['qm_relative_eV']):+10.4f} "
            f"{float(row['residual_target_eV']):+10.4f}"
        )

    jumps = adjacent_grid_jumps(rows)
    print()
    print("GRID RESIDUAL SMOOTHNESS")
    print(
        "Largest adjacent changes in residual target. "
        "Large isolated jumps deserve inspection before ML."
    )
    for system in sorted(by_system):
        system_jumps = [j for j in jumps if j["system"] == system]
        if not system_jumps:
            continue
        biggest = max(
            system_jumps,
            key=lambda j: abs(j["delta_residual_eV"])
        )
        steepest = max(
            system_jumps,
            key=lambda j: j["abs_slope_eV_per_angstrom"]
        )
        print(
            f"{system:14s} "
            f"max step={abs(biggest['delta_residual_eV']):.4f} eV "
            f"({biggest['from_id']} -> {biggest['to_id']}); "
            f"max slope={steepest['abs_slope_eV_per_angstrom']:.3f} eV/A"
        )

    print()
    print("TOP 12 |RESIDUAL TARGETS|")
    ranked = sorted(
        rows,
        key=lambda row: abs(float(row["residual_target_eV"])),
        reverse=True,
    )
    for index, row in enumerate(ranked[:12], start=1):
        print(
            f"{index:2d}. {row['geometry_id']:<36s} "
            f"{float(row['residual_target_eV']):+9.4f} eV  "
            f"Fmax={float(row['base_force_max_eV_per_angstrom']):7.2f}"
        )

    print()
    print("TOP 12 ADJACENT GRID RESIDUAL JUMPS")
    ranked_jumps = sorted(
        jumps,
        key=lambda j: abs(j["delta_residual_eV"]),
        reverse=True,
    )
    for index, jump in enumerate(ranked_jumps[:12], start=1):
        print(
            f"{index:2d}. {jump['system']:<12s} "
            f"{jump['axis']:<8s} "
            f"{jump['from_id']} -> {jump['to_id']}  "
            f"dResidual={jump['delta_residual_eV']:+.4f} eV  "
            f"slope={jump['abs_slope_eV_per_angstrom']:.3f} eV/A"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--qm", type=Path, default=DEFAULT_QM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    _, geometry_by_id = load_geometries(args.geometries)
    base_by_id = load_csv(args.base)
    qm_by_id = load_csv(args.qm)

    rows, references = build_rows(
        geometry_by_id,
        base_by_id,
        qm_by_id,
    )
    write_output(args.output, rows)

    print(f"merged rows : {len(rows)}")
    print(f"wrote       : {args.output}")
    print()
    print_summary(rows, references)


if __name__ == "__main__":
    main()
