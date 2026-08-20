"""Extract reactant/TS/product endpoints from the Grambow et al. Q-Chem logs.

Source: Grambow, Pattanaik & Green, Sci. Data 2020, doi:10.1038/s41597-020-0460-4
Data:   https://zenodo.org/records/3715478  ->  wb97xd3.tar.gz

Archive layout (confirmed against the real file):

    wb97xd3/rxn009959/r009959.log      reactant
    wb97xd3/rxn009959/p009959.log      product
    wb97xd3/rxn009959/ts009959.log     transition state

Each log holds two concatenated Q-Chem jobs -- a geometry optimisation followed
by a frequency run -- so the LAST "Standard Nuclear Orientation" block is the
converged geometry and the LAST "Total energy in the final basis set" is the
SCF energy at that geometry.  Earlier occurrences are mid-optimisation and must
not be used.  Energies in the log are hartree; they are converted to eV here.

12,001 reactions at wB97X-D3/def2-TZVP, gas phase, H/C/N/O only, up to seven
heavy atoms per molecule.  Every TS in this set was verified by the authors to
have exactly one imaginary frequency, so the parsed imaginary counts double as
a check that this parser is reading the right blocks.

The archive is read as a single sequential stream -- gzip has no usable random
access -- and nothing is written to disk except the output JSON.

    python extract_grambow_endpoints.py --limit 200
    python extract_grambow_endpoints.py --check-csv wb97xd3.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("wb97xd3.tar.gz")
DEFAULT_OUTPUT = Path("research_data/benchmark/grambow_endpoints.json")

HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL = 23.060547830619026

SUPPORTED_ELEMENTS = {"H", "C", "N", "O"}

PREFIX_TO_REGION = {"r": "reactant", "ts": "transition_state", "p": "product"}
REGIONS = ("reactant", "transition_state", "product")

COORDINATE_HEADER = "Standard Nuclear Orientation"
TOTAL_ENERGY = re.compile(
    r"Total energy in the final basis set\s*=\s*(-?\d+\.\d+)"
)
COORDINATE_ROW = re.compile(
    r"^\s*\d+\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
FREQUENCY_LINE = re.compile(r"^\s*Frequency:\s*(.*)$")

# ts009959.log -> ("ts", "009959")
MEMBER_NAME = re.compile(r"/rxn(\d+)/(r|p|ts)\1\.log$")


def parse_log(text):
    """Return the converged geometry, final energy, and frequency summary."""
    lines = text.splitlines()

    energies = TOTAL_ENERGY.findall(text)
    if not energies:
        raise ValueError("no 'Total energy in the final basis set' line found")
    energy_hartree = float(energies[-1])

    starts = [
        index for index, line in enumerate(lines)
        if COORDINATE_HEADER in line
    ]
    if not starts:
        raise ValueError("no 'Standard Nuclear Orientation' block found")

    symbols = []
    coordinates = []
    for line in lines[starts[-1] + 1:]:
        match = COORDINATE_ROW.match(line)
        if match:
            symbol = match.group(1)
            if symbol not in SUPPORTED_ELEMENTS:
                raise ValueError(f"unexpected element {symbol!r}")
            symbols.append(symbol)
            coordinates.append([float(match.group(index)) for index in (2, 3, 4)])
        elif symbols:
            break

    if not symbols:
        raise ValueError("coordinate block was empty")

    frequencies = []
    for line in lines:
        match = FREQUENCY_LINE.match(line)
        if match:
            for token in match.group(1).split():
                try:
                    frequencies.append(float(token))
                except ValueError:
                    pass

    return {
        "symbols": symbols,
        "coordinates_angstrom": coordinates,
        "energy_hartree": energy_hartree,
        "reference_energy_eV": energy_hartree * HARTREE_TO_EV,
        "frequency_count": len(frequencies),
        "imaginary_count": sum(1 for value in frequencies if value < 0.0),
        "lowest_frequency_cm1": min(frequencies) if frequencies else None,
    }


def read_archive(path, limit=None):
    """Stream the tarball once, grouping logs by reaction folder."""
    pending = defaultdict(dict)
    completed = []
    skipped = []

    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if limit is not None and len(completed) >= limit:
                break
            if not member.isfile():
                continue
            match = MEMBER_NAME.search("/" + member.name)
            if not match:
                continue

            reaction_id, prefix = f"rxn{match.group(1)}", match.group(2)
            handle = tar.extractfile(member)
            if handle is None:
                continue

            try:
                pending[reaction_id][PREFIX_TO_REGION[prefix]] = parse_log(
                    handle.read().decode("utf-8", "replace")
                )
            except Exception as problem:
                skipped.append(
                    f"{member.name}: {type(problem).__name__}: {problem}"
                )
                pending[reaction_id]["__broken__"] = True
                continue

            group = pending[reaction_id]
            if all(region in group for region in REGIONS):
                if "__broken__" not in group:
                    completed.append((reaction_id, group))
                pending.pop(reaction_id)

    for reaction_id, group in pending.items():
        missing = [region for region in REGIONS if region not in group]
        if missing:
            skipped.append(f"{reaction_id}: missing {', '.join(missing)}")

    return completed, skipped


def build_rows(completed):
    geometries = []
    diagnostics = {"ts_one_imaginary": 0, "minima_zero_imaginary": 0,
                   "ts_total": 0, "minima_total": 0}

    for reaction_id, group in completed:
        formulas = {
            tuple(sorted(group[region]["symbols"])) for region in REGIONS
        }
        if len(formulas) != 1:
            continue

        for region in REGIONS:
            parsed = group[region]
            if region == "transition_state":
                diagnostics["ts_total"] += 1
                diagnostics["ts_one_imaginary"] += int(
                    parsed["imaginary_count"] == 1
                )
            else:
                diagnostics["minima_total"] += 1
                diagnostics["minima_zero_imaginary"] += int(
                    parsed["imaginary_count"] == 0
                )

            geometries.append({
                "geometry_id": f"{reaction_id}/{region}",
                "system": reaction_id,
                "reaction_id": reaction_id,
                "region": region,
                "split": "all",
                "sample_kind": "grambow_endpoint",
                "charge": 0,
                "symbols": parsed["symbols"],
                "coordinates_angstrom": parsed["coordinates_angstrom"],
                "reference_energy_eV": parsed["reference_energy_eV"],
                "imaginary_count": parsed["imaginary_count"],
                "lowest_frequency_cm1": parsed["lowest_frequency_cm1"],
            })

    return geometries, diagnostics


def cross_check(geometries, csv_path, tolerance=1.0):
    """Compare parsed barriers against the published activation energies."""
    published = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv_module.DictReader(handle):
            key = next(
                (row[name] for name in ("idx", "rxn", "reaction", "id")
                 if name in row and row[name]),
                None,
            )
            value = next(
                (row[name] for name in ("ea", "Ea", "activation_energy")
                 if name in row and row[name]),
                None,
            )
            if key is None or value is None:
                continue
            published[f"rxn{int(key):06d}"] = float(value)

    by_reaction = defaultdict(dict)
    for row in geometries:
        by_reaction[row["reaction_id"]][row["region"]] = row["reference_energy_eV"]

    compared = []
    for reaction_id, endpoints in by_reaction.items():
        if reaction_id not in published or len(endpoints) < 2:
            continue
        barrier = (
            endpoints["transition_state"] - endpoints["reactant"]
        ) * EV_TO_KCAL
        compared.append(abs(barrier - published[reaction_id]))

    if not compared:
        return {"compared": 0, "note": "no overlapping reaction ids in csv"}
    return {
        "compared": len(compared),
        "mean_absolute_kcal": sum(compared) / len(compared),
        "max_absolute_kcal": max(compared),
        "within_tolerance": sum(1 for value in compared if value <= tolerance),
        "tolerance_kcal": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-csv", type=Path, default=None,
                        help="wb97xd3.csv, to verify the parsed barriers")
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(f"{arguments.input} not found")

    completed, skipped = read_archive(arguments.input, limit=arguments.limit)
    geometries, diagnostics = build_rows(completed)

    if not geometries:
        raise SystemExit("no complete reactions parsed")

    payload = {
        "schema_version": 1,
        "source": "Grambow, Pattanaik & Green, Sci. Data 2020",
        "source_doi": "10.1038/s41597-020-0460-4",
        "reference_level": "wB97X-D3/def2-TZVP",
        "units": {"coordinates": "angstrom", "energy": "eV"},
        "energy_alignment": "within-reaction differences only",
        "counts": {
            "reactions": len(completed),
            "geometries": len(geometries),
            "skipped": len(skipped),
        },
        "frequency_diagnostics": diagnostics,
        "skipped": skipped[:50],
        "geometries": geometries,
    }

    if arguments.check_csv:
        payload["csv_cross_check"] = cross_check(geometries, arguments.check_csv)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{len(completed)} reactions -> {arguments.output}")
    print(f"TS with exactly one imaginary frequency: "
          f"{diagnostics['ts_one_imaginary']}/{diagnostics['ts_total']}")
    print(f"minima with zero imaginary frequencies: "
          f"{diagnostics['minima_zero_imaginary']}/{diagnostics['minima_total']}")
    if "csv_cross_check" in payload:
        print("csv cross-check:", json.dumps(payload["csv_cross_check"]))
    if skipped:
        print(f"{len(skipped)} skipped; first few:")
        for line in skipped[:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
