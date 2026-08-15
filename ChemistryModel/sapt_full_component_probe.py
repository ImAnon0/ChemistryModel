"""
Focused Psi4 SAPT0 component probe for the existing ChemistryModel SAPT data.

Why this exists
---------------
The current ChemistryModel nonbonded H-state wall was fitted only to
`SAPT EXCH10 ENERGY` (exchange / Pauli repulsion).

Before adding new barrier physics, test the simpler possibility that the
missing electrostatic + induction + dispersion pieces materially cancel the
exchange wall at the SAME molecular-contact geometries already used to build
the SAPT dataset.

This script deliberately reuses:
    research_data/sapt/sapt_molecular_holdout.csv

It reconstructs the stored target/probe geometries exactly, reruns
SAPT0/jun-cc-pVDZ, and records:

    SAPT ELST ENERGY
    SAPT EXCH ENERGY
    SAPT EXCH10 ENERGY
    SAPT IND ENERGY
    SAPT DISP ENERGY
    SAPT total interaction energy

Crucially, it compares the rerun EXCH10 against the EXCH10 already stored in
the CSV. If those disagree, do NOT interpret the other components yet: the
Psi4 setup is not reproducing the original calculation.

Default focused set
-------------------
10 jobs:
    CH4   : CH_bond, face      at 1.2 and 1.5 A
    CH2O  : CH_bond, carbon_end at 1.2 and 1.5 A
    H2O   : OH_bond            at 1.2 and 1.5 A

These are not radical-state SAPT calculations. They are a first-principles
audit of whether fitting ONLY exchange discarded a large attractive part of
the same closed-shell interaction data used to calibrate the wall.

Usage
-----
Focused default:
    py sapt_full_component_probe.py

All 64 molecular-holdout rows:
    py sapt_full_component_probe.py --all

Custom:
    py sapt_full_component_probe.py --systems CH4 CH2O --distances 1.2 1.5 1.85

Output
------
    research_data/sapt/sapt_component_probe.csv

The CSV is rewritten after every completed job so partial progress survives an
interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import traceback


HARTREE_TO_EV = 27.211386245988

DEFAULT_INPUT = Path(
    "research_data/sapt/sapt_molecular_holdout.csv"
)

DEFAULT_OUTPUT = Path(
    "research_data/sapt/sapt_component_probe.csv"
)

DEFAULT_BASIS = "jun-cc-pvdz"

FOCUSED_APPROACHES = {
    "CH4": {
        "CH_bond",
        "face",
    },
    "CH2O": {
        "CH_bond",
        "carbon_end",
    },
    "H2O": {
        "OH_bond",
    },
}

FOCUSED_DISTANCES = (
    1.2,
    1.5,
)

OUTPUT_FIELDS = (
    "system",
    "approach",
    "contact_distance_A",
    "stored_exch10_eV",
    "rerun_exch10_eV",
    "exch10_difference_eV",
    "exchange_component_eV",
    "electrostatics_eV",
    "induction_eV",
    "dispersion_eV",
    "nonexchange_sum_eV",
    "sapt_total_eV",
    "component_sum_eV",
    "total_minus_component_sum_eV",
    "attractive_offset_from_exchange_eV",
    "exchange_cancelled_percent",
    "elapsed_s",
    "basis",
    "method",
    "status",
    "error",
)


def _float(
    value,
):
    return float(
        value
    )


def load_rows(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"input CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    usable = [
        row
        for row in rows
        if row.get(
            "status",
            ""
        ).strip().lower()
        == "ok"
    ]

    if not usable:
        raise RuntimeError(
            f"no successful rows found in {path}"
        )

    return usable


def distance_matches(
    value,
    targets,
    tolerance=1.0e-8,
):
    return any(
        abs(
            value
            - target
        )
        <= tolerance
        for target in targets
    )


def select_rows(
    rows,
    *,
    run_all,
    systems,
    distances,
):
    if run_all:
        selected = list(
            rows
        )

    elif systems:
        allowed_systems = set(
            systems
        )

        selected = [
            row
            for row in rows
            if row[
                "system"
            ] in allowed_systems
        ]

        if distances:
            selected = [
                row
                for row in selected
                if distance_matches(
                    _float(
                        row[
                            "contact_distance_A"
                        ]
                    ),
                    distances,
                )
            ]

    else:
        selected = []

        for row in rows:
            system = row[
                "system"
            ]

            approach = row[
                "approach"
            ]

            distance = _float(
                row[
                    "contact_distance_A"
                ]
            )

            if system not in FOCUSED_APPROACHES:
                continue

            if approach not in FOCUSED_APPROACHES[
                system
            ]:
                continue

            if not distance_matches(
                distance,
                FOCUSED_DISTANCES,
            ):
                continue

            selected.append(
                row
            )

    selected.sort(
        key=lambda row: (
            row[
                "system"
            ],
            row[
                "approach"
            ],
            _float(
                row[
                    "contact_distance_A"
                ]
            ),
        )
    )

    if not selected:
        raise RuntimeError(
            "selection produced zero SAPT jobs"
        )

    return selected


def atom_lines(
    atoms,
):
    lines = []

    for atom in atoms:
        symbol = atom[
            "element"
        ]

        x, y, z = atom[
            "xyz"
        ]

        lines.append(
            f"{symbol:2s} "
            f"{float(x): .16f} "
            f"{float(y): .16f} "
            f"{float(z): .16f}"
        )

    return lines


def build_geometry_text(
    row,
):
    target = json.loads(
        row[
            "target_atoms_json"
        ]
    )

    probe = json.loads(
        row[
            "probe_atoms_json"
        ]
    )

    lines = [
        "0 1",
        *atom_lines(
            target
        ),
        "--",
        "0 1",
        *atom_lines(
            probe
        ),
        "units angstrom",
        "symmetry c1",
        "no_reorient",
        "no_com",
    ]

    return "\n".join(
        lines
    )


def get_variable(
    psi4,
    *names,
):
    """
    Psi4 renamed some aggregate SAPT total variables across versions.
    Try documented alternatives in order.
    """

    last_error = None

    for name in names:
        try:
            return float(
                psi4.core.variable(
                    name
                )
            )
        except Exception as exc:
            last_error = exc

    joined = ", ".join(
        names
    )

    raise RuntimeError(
        f"none of these Psi4 variables were available: {joined}"
    ) from last_error


def run_sapt(
    psi4,
    row,
    *,
    basis,
    threads,
    memory,
):
    psi4.core.clean()

    try:
        psi4.core.clean_variables()
    except Exception:
        pass

    psi4.set_num_threads(
        threads
    )

    psi4.set_memory(
        memory
    )

    # Quiet the enormous SAPT printout; this script reports the numbers we
    # actually need.
    psi4.core.be_quiet()

    molecule = psi4.geometry(
        build_geometry_text(
            row
        )
    )

    psi4.set_options(
        {
            "basis": basis,
            # Tight enough to make the EXCH10 replay check meaningful without
            # adding a new physical choice to the calculation.
            "e_convergence": 1.0e-9,
            "d_convergence": 1.0e-9,
        }
    )

    start = time.perf_counter()

    psi4.energy(
        "sapt0",
        molecule=molecule,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    values_h = {
        "exch10": get_variable(
            psi4,
            "SAPT EXCH10 ENERGY",
        ),
        "exchange": get_variable(
            psi4,
            "SAPT EXCH ENERGY",
        ),
        "electrostatics": get_variable(
            psi4,
            "SAPT ELST ENERGY",
        ),
        "induction": get_variable(
            psi4,
            "SAPT IND ENERGY",
        ),
        "dispersion": get_variable(
            psi4,
            "SAPT DISP ENERGY",
        ),
        "total": get_variable(
            psi4,
            "SAPT ENERGY",
            "SAPT TOTAL ENERGY",
            "SAPT0 TOTAL ENERGY",
            "SAPT SAPT0 ENERGY",
        ),
    }

    return (
        {
            name: (
                value
                * HARTREE_TO_EV
            )
            for name, value
            in values_h.items()
        },
        elapsed,
    )


def result_key(
    row,
):
    return (
        row[
            "system"
        ],
        row[
            "approach"
        ],
        f"{_float(row['contact_distance_A']):.12g}",
    )


def load_existing_results(
    path,
):
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    return {
        result_key(
            row
        ): row
        for row in rows
        if row.get(
            "status"
        )
        == "ok"
    }


def write_results(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field
                    in OUTPUT_FIELDS
                }
            )


def format_signed(
    value,
):
    return f"{value:+.6f}"


def make_result(
    source,
    components,
    elapsed,
    *,
    basis,
):
    stored_exch10 = _float(
        source[
            "exch10_eV"
        ]
    )

    rerun_exch10 = components[
        "exch10"
    ]

    electrostatics = components[
        "electrostatics"
    ]

    induction = components[
        "induction"
    ]

    dispersion = components[
        "dispersion"
    ]

    exchange = components[
        "exchange"
    ]

    total = components[
        "total"
    ]

    nonexchange = (
        electrostatics
        + induction
        + dispersion
    )

    component_sum = (
        exchange
        + nonexchange
    )

    # Positive means the omitted non-exchange terms lower the interaction
    # relative to exchange alone.
    attractive_offset = (
        exchange
        - total
    )

    if abs(
        exchange
    ) > 1.0e-12:
        cancelled_percent = (
            100.0
            * attractive_offset
            / abs(
                exchange
            )
        )
    else:
        cancelled_percent = math.nan

    return {
        "system": source[
            "system"
        ],
        "approach": source[
            "approach"
        ],
        "contact_distance_A": _float(
            source[
                "contact_distance_A"
            ]
        ),
        "stored_exch10_eV": stored_exch10,
        "rerun_exch10_eV": rerun_exch10,
        "exch10_difference_eV": (
            rerun_exch10
            - stored_exch10
        ),
        "exchange_component_eV": exchange,
        "electrostatics_eV": electrostatics,
        "induction_eV": induction,
        "dispersion_eV": dispersion,
        "nonexchange_sum_eV": nonexchange,
        "sapt_total_eV": total,
        "component_sum_eV": component_sum,
        "total_minus_component_sum_eV": (
            total
            - component_sum
        ),
        "attractive_offset_from_exchange_eV": attractive_offset,
        "exchange_cancelled_percent": cancelled_percent,
        "elapsed_s": elapsed,
        "basis": basis,
        "method": "SAPT0",
        "status": "ok",
        "error": "",
    }


def failed_result(
    source,
    exc,
    *,
    basis,
):
    return {
        "system": source[
            "system"
        ],
        "approach": source[
            "approach"
        ],
        "contact_distance_A": _float(
            source[
                "contact_distance_A"
            ]
        ),
        "stored_exch10_eV": _float(
            source[
                "exch10_eV"
            ]
        ),
        "basis": basis,
        "method": "SAPT0",
        "status": "failed",
        "error": (
            f"{type(exc).__name__}: {exc}"
        ),
    }


def print_result(
    result,
):
    label = (
        f"{result['system']:>4} "
        f"{result['approach']:<12} "
        f"{float(result['contact_distance_A']):.2f} A"
    )

    if result[
        "status"
    ] != "ok":
        print(
            f"{label}  FAILED  {result['error']}"
        )
        return

    print(
        f"{label}  "
        f"EXCH10={result['rerun_exch10_eV']:+7.3f}  "
        f"dEXCH={result['exch10_difference_eV']:+.4f}  "
        f"ELST={result['electrostatics_eV']:+7.3f}  "
        f"IND={result['induction_eV']:+7.3f}  "
        f"DISP={result['dispersion_eV']:+7.3f}  "
        f"TOTAL={result['sapt_total_eV']:+7.3f}  "
        f"offset={result['attractive_offset_from_exchange_eV']:+7.3f}"
    )


def print_summary(
    results,
):
    successful = [
        row
        for row in results
        if row.get(
            "status"
        )
        == "ok"
    ]

    if not successful:
        print()
        print(
            "No successful jobs to summarize."
        )
        return

    print()
    print(
        "SUMMARY"
    )
    print(
        "======="
    )

    max_replay_error = max(
        abs(
            float(
                row[
                    "exch10_difference_eV"
                ]
            )
        )
        for row in successful
    )

    print(
        f"max |rerun EXCH10 - stored EXCH10|  "
        f"{max_replay_error:.6f} eV"
    )

    if max_replay_error > 0.02:
        print(
            "WARNING: EXCH10 replay differs by >0.02 eV. "
            "Do not interpret the component comparison until the Psi4 setup "
            "matches the original dataset."
        )
        return

    print(
        "EXCH10 replay is close enough for this diagnostic."
    )
    print()

    for system in sorted(
        {
            row[
                "system"
            ]
            for row in successful
        }
    ):
        system_rows = [
            row
            for row in successful
            if row[
                "system"
            ]
            == system
        ]

        offsets = [
            float(
                row[
                    "attractive_offset_from_exchange_eV"
                ]
            )
            for row in system_rows
        ]

        totals = [
            float(
                row[
                    "sapt_total_eV"
                ]
            )
            for row in system_rows
        ]

        cancellation = [
            float(
                row[
                    "exchange_cancelled_percent"
                ]
            )
            for row in system_rows
            if math.isfinite(
                float(
                    row[
                        "exchange_cancelled_percent"
                    ]
                )
            )
        ]

        print(
            f"{system:>5}: "
            f"mean exchange->total lowering "
            f"{sum(offsets)/len(offsets):+.3f} eV, "
            f"range {min(offsets):+.3f}..{max(offsets):+.3f} eV, "
            f"mean total {sum(totals)/len(totals):+.3f} eV"
        )

        if cancellation:
            print(
                f"       mean fraction of exchange cancelled "
                f"{sum(cancellation)/len(cancellation):.1f}%"
            )

    print()
    print(
        "Read this diagnostic conservatively:"
    )
    print(
        "  Large positive 'offset' means electrostatics + induction + "
        "dispersion materially lower the interaction relative to exchange "
        "alone."
    )
    print(
        "  This does NOT yet prove the full SAPT total should replace the "
        "H-state wall: these are the same closed-shell molecular-probe "
        "geometries used for calibration, not radical diabatic surfaces."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--basis",
        default=DEFAULT_BASIS,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--memory",
        default="2 GB",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="rerun every successful row in the molecular holdout CSV",
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        help="optional system filter, e.g. CH4 CH2O",
    )

    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        help="optional contact-distance filter",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun jobs already present as successful in the output CSV",
    )

    args = parser.parse_args()

    try:
        import psi4
    except ImportError as exc:
        raise RuntimeError(
            "Psi4 is not importable in this Python environment. "
            "Run this from the environment you used for the previous SAPT "
            "dataset (for example your chem-sapt conda environment)."
        ) from exc

    source_rows = load_rows(
        args.input
    )

    selected = select_rows(
        source_rows,
        run_all=args.all,
        systems=args.systems,
        distances=args.distances,
    )

    existing = load_existing_results(
        args.output
    )

    print(
        "SAPT0 FULL-COMPONENT PROBE"
    )
    print(
        "=========================="
    )
    print(
        f"input : {args.input}"
    )
    print(
        f"output: {args.output}"
    )
    print(
        f"basis : {args.basis}"
    )
    print(
        f"jobs  : {len(selected)}"
    )
    print()

    results_by_key = dict(
        existing
    )

    for index, source in enumerate(
        selected,
        start=1,
    ):
        key = result_key(
            source
        )

        if (
            not args.force
            and key in existing
        ):
            result = existing[
                key
            ]

            print(
                f"[{index:02d}/{len(selected):02d}] cached ",
                end="",
            )

            print_result(
                result
            )

            continue

        print(
            f"[{index:02d}/{len(selected):02d}] running "
            f"{source['system']} {source['approach']} "
            f"{_float(source['contact_distance_A']):.2f} A...",
            flush=True,
        )

        try:
            components, elapsed = run_sapt(
                psi4,
                source,
                basis=args.basis,
                threads=args.threads,
                memory=args.memory,
            )

            result = make_result(
                source,
                components,
                elapsed,
                basis=args.basis,
            )

        except Exception as exc:
            result = failed_result(
                source,
                exc,
                basis=args.basis,
            )

            traceback.print_exc()

        results_by_key[
            key
        ] = result

        ordered = sorted(
            results_by_key.values(),
            key=lambda row: (
                row[
                    "system"
                ],
                row[
                    "approach"
                ],
                _float(
                    row[
                        "contact_distance_A"
                    ]
                ),
            ),
        )

        write_results(
            args.output,
            ordered,
        )

        print_result(
            result
        )

    selected_keys = {
        result_key(
            row
        )
        for row in selected
    }

    selected_results = [
        row
        for key, row
        in results_by_key.items()
        if key in selected_keys
    ]

    print_summary(
        selected_results
    )

    print()
    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
