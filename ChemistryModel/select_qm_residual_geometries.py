"""Select the first QM-residual geometries for expensive QM evaluation.

The original geometry dataset is preserved unchanged.  This script creates:
  research_data/qm_residual/geometries_qm.json
  research_data/qm_residual/qm_selection.csv

For the first proof-of-concept we exclude only obviously pathological static
collisions identified by the base engine.  They remain in the original dataset
and in qm_selection.csv as stress-test rows; nothing is deleted.

Current default rule:
    exclude if base max static force > 50 eV/angstrom

The threshold is deliberately conservative.  It removes the clear high-force
water cluster while retaining the ordinary reactive/repulsive region.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_GEOMETRIES = Path("research_data/qm_residual/geometries.json")
DEFAULT_BASE = Path("research_data/qm_residual/base_results.csv")
DEFAULT_OUTPUT = Path("research_data/qm_residual/geometries_qm.json")
DEFAULT_SELECTION = Path("research_data/qm_residual/qm_selection.csv")
DEFAULT_FORCE_LIMIT = 50.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            geometry_id = row["geometry_id"]
            if geometry_id in rows:
                raise ValueError(f"duplicate base result: {geometry_id}")
            row["base_force_max_eV_per_angstrom"] = float(
                row["base_force_max_eV_per_angstrom"]
            )
            row["base_energy_eV"] = float(row["base_energy_eV"])
            rows[geometry_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, default=DEFAULT_GEOMETRIES)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--force-limit",
        type=float,
        default=DEFAULT_FORCE_LIMIT,
        help="exclude rows above this base max static force (eV/A)",
    )
    args = parser.parse_args()

    payload = json.loads(args.geometries.read_text(encoding="utf-8"))
    base = load_base(args.base)

    original = payload["geometries"]
    ids = {row["geometry_id"] for row in original}
    missing = ids - set(base)
    extra = set(base) - ids
    if missing:
        raise ValueError(f"missing base results for {len(missing)} geometries")
    if extra:
        raise ValueError(f"base results contain {len(extra)} unknown geometries")

    included = []
    audit_rows = []

    for geometry in original:
        geometry_id = geometry["geometry_id"]
        base_row = base[geometry_id]
        force = base_row["base_force_max_eV_per_angstrom"]

        include = force <= args.force_limit
        reason = "included"
        if not include:
            reason = f"stress_only_force_gt_{args.force_limit:g}_eV_per_A"

        if include:
            included.append(geometry)

        audit_rows.append({
            "geometry_id": geometry_id,
            "system": geometry["system"],
            "split": geometry["split"],
            "sample_kind": geometry["sample_kind"],
            "region": geometry["region"],
            "base_energy_eV": base_row["base_energy_eV"],
            "base_force_max_eV_per_angstrom": force,
            "qm_include": "yes" if include else "no",
            "selection_reason": reason,
        })

    output_payload = dict(payload)
    output_payload["selection"] = {
        "source_geometries": str(args.geometries),
        "source_geometries_sha256": sha256_file(args.geometries),
        "source_base_results": str(args.base),
        "source_base_results_sha256": sha256_file(args.base),
        "rule": "base_force_max_eV_per_angstrom <= force_limit",
        "force_limit_eV_per_angstrom": args.force_limit,
        "original_count": len(original),
        "included_count": len(included),
        "stress_only_count": len(original) - len(included),
    }
    output_payload["geometries"] = included

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "geometry_id",
            "system",
            "split",
            "sample_kind",
            "region",
            "base_energy_eV",
            "base_force_max_eV_per_angstrom",
            "qm_include",
            "selection_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    by_system = {}
    excluded_by_system = {}
    for row in original:
        by_system[row["system"]] = by_system.get(row["system"], 0) + 1
    for row in audit_rows:
        if row["qm_include"] == "no":
            excluded_by_system[row["system"]] = (
                excluded_by_system.get(row["system"], 0) + 1
            )

    print(f"original      : {len(original)}")
    print(f"QM included   : {len(included)}")
    print(f"stress only   : {len(original) - len(included)}")
    print(f"force limit   : {args.force_limit:.1f} eV/A")
    print("")
    for system in sorted(by_system):
        excluded = excluded_by_system.get(system, 0)
        kept = by_system[system] - excluded
        print(
            f"{system:14s} kept={kept:3d}  "
            f"stress_only={excluded:3d}"
        )
    print("")
    print(f"wrote         : {args.output}")
    print(f"audit         : {args.selection}")


if __name__ == "__main__":
    main()
