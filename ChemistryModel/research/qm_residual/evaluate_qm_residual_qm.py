"""Evaluate selected QM-residual geometries with Psi4 DFT.

Run this script inside the `chem-sapt` conda environment:

    conda activate chem-sapt
    python evaluate_qm_residual_qm.py --smoke

The script is deliberately resumable. Results are rewritten atomically after
every geometry, so an interrupted full run can simply be started again.

Default electronic-structure level for this proof of concept:
    unrestricted wB97X-D / jun-cc-pVDZ

All current systems are neutral doublets. Exact input Cartesian coordinates are
preserved with C1 symmetry, no reorientation, and no center-of-mass shift.

Inputs:
    research_data/qm_residual/geometries_qm.json

Outputs:
    research_data/qm_residual/qm_results.csv
    research_data/qm_residual/qm_results.meta.json
    research_data/qm_residual/qm_psi4.out

The later dataset builder aligns QM and ChemistryModel energies to each
system's reactant_reference. Absolute energy zeros are therefore irrelevant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path

import psi4


HARTREE_TO_EV = 27.211386245988

DEFAULT_INPUT = Path("research_data/qm_residual/geometries_qm.json")
DEFAULT_OUTPUT = Path("research_data/qm_residual/qm_results.csv")
DEFAULT_METADATA = Path("research_data/qm_residual/qm_results.meta.json")
DEFAULT_PSI4_OUTPUT = Path("research_data/qm_residual/qm_psi4.out")

DEFAULT_METHOD = "wb97x-d"
DEFAULT_BASIS = "jun-cc-pvdz"
DEFAULT_THREADS = 8
DEFAULT_MEMORY = "4 GB"

FIELDNAMES = [
    "geometry_id",
    "system",
    "split",
    "sample_kind",
    "region",
    "charge",
    "multiplicity",
    "qm_method",
    "qm_basis",
    "qm_energy_hartree",
    "qm_energy_eV",
    "s2",
    "wall_seconds",
    "status",
    "error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("geometries")

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: missing/non-empty 'geometries' list")

    ids = [row.get("geometry_id") for row in rows]
    if any(not geometry_id for geometry_id in ids):
        raise ValueError("all geometries need geometry_id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate geometry_id values")

    return payload


def molecule_from_row(row: dict):
    symbols = row["symbols"]
    coordinates = row["coordinates_angstrom"]

    if len(symbols) != len(coordinates):
        raise ValueError(
            f"{row['geometry_id']}: symbol/coordinate count mismatch"
        )

    charge = int(row["charge"])
    multiplicity = int(row["multiplicity"])

    lines = [f"{charge} {multiplicity}"]
    for symbol, xyz in zip(symbols, coordinates):
        if len(xyz) != 3:
            raise ValueError(f"{row['geometry_id']}: malformed xyz coordinate")
        x, y, z = (float(value) for value in xyz)
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError(f"{row['geometry_id']}: non-finite coordinate")
        lines.append(f"{symbol:2s} {x: .12f} {y: .12f} {z: .12f}")

    lines.extend([
        "units angstrom",
        "symmetry c1",
        "no_reorient",
        "no_com",
    ])

    return psi4.geometry("\n".join(lines))


def extract_s2(wfn):
    """Best-effort extraction; blank is acceptable if Psi4 doesn't expose it."""
    try:
        variables = wfn.scalar_variables()
    except Exception:
        return ""

    candidates = []
    for key, value in variables.items():
        normalized = str(key).upper().replace(" ", "")
        if "S^2" in normalized or "S**2" in normalized:
            candidates.append((str(key), float(value)))

    if not candidates:
        return ""

    # Prefer a variable with SCF in its name, otherwise take the first.
    candidates.sort(key=lambda pair: ("SCF" not in pair[0].upper(), pair[0]))
    return candidates[0][1]


def evaluate_row(row: dict, *, method: str, basis: str) -> dict:
    geometry_id = row["geometry_id"]
    started = time.perf_counter()

    result = {
        "geometry_id": geometry_id,
        "system": row["system"],
        "split": row["split"],
        "sample_kind": row["sample_kind"],
        "region": row["region"],
        "charge": int(row["charge"]),
        "multiplicity": int(row["multiplicity"]),
        "qm_method": method,
        "qm_basis": basis,
        "qm_energy_hartree": "",
        "qm_energy_eV": "",
        "s2": "",
        "wall_seconds": "",
        "status": "failed",
        "error": "",
    }

    try:
        molecule = molecule_from_row(row)

        energy_hartree, wfn = psi4.energy(
            f"{method}/{basis}",
            molecule=molecule,
            return_wfn=True,
        )

        energy_hartree = float(energy_hartree)
        energy_eV = energy_hartree * HARTREE_TO_EV

        if not math.isfinite(energy_hartree):
            raise RuntimeError("Psi4 returned a non-finite energy")

        result["qm_energy_hartree"] = energy_hartree
        result["qm_energy_eV"] = energy_eV
        result["s2"] = extract_s2(wfn)
        result["status"] = "ok"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        result["wall_seconds"] = time.perf_counter() - started
        # Release per-calculation scratch/intermediates before the next point.
        try:
            psi4.core.clean()
        except Exception:
            pass

    return result


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            geometry_id = row["geometry_id"]
            rows[geometry_id] = row
    return rows


def write_csv_atomic(path: Path, rows_by_id: dict[str, dict], order: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for geometry_id in order:
                if geometry_id in rows_by_id:
                    writer.writerow(rows_by_id[geometry_id])
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def choose_rows(rows: list[dict], *, smoke: bool, limit: int | None) -> list[dict]:
    chosen = rows

    if smoke:
        # One reactant reference from each system: cheap but exercises every
        # elemental composition and radical type in the experiment.
        seen = set()
        smoke_rows = []
        for row in rows:
            system = row["system"]
            if (
                system not in seen
                and row["sample_kind"] == "reactant_reference"
            ):
                smoke_rows.append(row)
                seen.add(system)
        chosen = smoke_rows

    if limit is not None:
        chosen = chosen[:limit]

    return chosen


def write_metadata(
    path: Path,
    *,
    input_path: Path,
    input_hash: str,
    method: str,
    basis: str,
    threads: int,
    memory: str,
    selected_count: int,
    results: dict[str, dict],
):
    ok_count = sum(row.get("status") == "ok" for row in results.values())
    failed_count = sum(row.get("status") == "failed" for row in results.values())

    metadata = {
        "input": str(input_path),
        "input_sha256": input_hash,
        "method": method,
        "basis": basis,
        "reference": "uks",
        "scf_type": "df",
        "threads": threads,
        "memory": memory,
        "selected_geometry_count_this_invocation": selected_count,
        "stored_ok_count": ok_count,
        "stored_failed_count": failed_count,
        "psi4_version": getattr(psi4, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "hartree_to_eV": HARTREE_TO_EV,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--psi4-output", type=Path, default=DEFAULT_PSI4_OUTPUT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate one reactant-reference geometry per system",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit selected geometries after any --smoke selection",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute rows already stored with status=ok",
    )
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    payload = load_payload(args.input)
    all_rows = payload["geometries"]
    selected = choose_rows(all_rows, smoke=args.smoke, limit=args.limit)
    input_hash = sha256_file(args.input)

    args.psi4_output.parent.mkdir(parents=True, exist_ok=True)
    psi4.set_output_file(str(args.psi4_output.resolve()), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)

    # All current systems are open-shell doublets, so use unrestricted KS.
    psi4.set_options({
        "reference": "uks",
        "scf_type": "df",
        "e_convergence": 1.0e-8,
        "d_convergence": 1.0e-8,
        "maxiter": 150,
    })

    existing = load_existing(args.output)
    selected_ids = [row["geometry_id"] for row in selected]
    all_ids = [row["geometry_id"] for row in all_rows]

    print(f"input      : {args.input}")
    print(f"sha256     : {input_hash}")
    print(f"selected   : {len(selected)} / {len(all_rows)}")
    print(f"method     : {args.method}/{args.basis}")
    print(f"reference  : UKS")
    print(f"threads    : {args.threads}")
    print(f"memory     : {args.memory}")
    print(f"resume     : {'off (--force)' if args.force else 'on'}")
    print("")

    invocation_started = time.perf_counter()
    calculated = 0
    skipped = 0
    failures = 0

    for index, row in enumerate(selected, start=1):
        geometry_id = row["geometry_id"]
        old = existing.get(geometry_id)

        if (
            not args.force
            and old is not None
            and old.get("status") == "ok"
            and old.get("qm_method") == args.method
            and old.get("qm_basis") == args.basis
        ):
            skipped += 1
            print(
                f"[{index:3d}/{len(selected):3d}] "
                f"{geometry_id:<34s} SKIP existing ok"
            )
            continue

        result = evaluate_row(
            row,
            method=args.method,
            basis=args.basis,
        )
        existing[geometry_id] = result
        calculated += 1

        if result["status"] == "ok":
            s2_text = (
                ""
                if result["s2"] == ""
                else f"  S2={float(result['s2']):.5f}"
            )
            print(
                f"[{index:3d}/{len(selected):3d}] "
                f"{geometry_id:<34s} "
                f"E={float(result['qm_energy_hartree']):+.10f} Eh  "
                f"{float(result['wall_seconds']):6.2f}s"
                f"{s2_text}"
            )
        else:
            failures += 1
            print(
                f"[{index:3d}/{len(selected):3d}] "
                f"{geometry_id:<34s} FAILED  "
                f"{result['error']}"
            )

        # Checkpoint after every calculation.
        write_csv_atomic(args.output, existing, all_ids)

    elapsed = time.perf_counter() - invocation_started

    # Ensure CSV exists even if everything was skipped.
    write_csv_atomic(args.output, existing, all_ids)
    write_metadata(
        args.metadata,
        input_path=args.input,
        input_hash=input_hash,
        method=args.method,
        basis=args.basis,
        threads=args.threads,
        memory=args.memory,
        selected_count=len(selected),
        results=existing,
    )

    ok_selected = sum(
        existing.get(geometry_id, {}).get("status") == "ok"
        for geometry_id in selected_ids
    )

    print("")
    print(f"calculated : {calculated}")
    print(f"skipped    : {skipped}")
    print(f"failed     : {failures}")
    print(f"selected ok: {ok_selected}/{len(selected)}")
    print(f"wall time  : {elapsed:.2f} s")
    print(f"results    : {args.output}")
    print(f"metadata   : {args.metadata}")
    print(f"psi4 log   : {args.psi4_output}")


if __name__ == "__main__":
    main()
