"""
Run open-shell SAPT0 on the representative CURRENT minimax-path states exported
by export_sapt_minimax_path_probe.py.

Run from the Psi4 environment:

    conda activate chem-sapt
    python sapt_open_shell_minimax_path_probe.py

Inputs:
    research_data/sapt/sapt_minimax_path_probe.json

Outputs:
    research_data/sapt/sapt_open_shell_minimax_path_probe.csv
    research_data/sapt/sapt_open_shell_minimax_path_probe_psi4.out

Psi4's verbose output is redirected to the .out file, so the terminal stays
compact.

For every path point, the script evaluates BOTH physically meaningful diabatic
fragmentations:
    formaldehyde: CH2O + H   versus HCO + H2
    methane:      CH4  + H   versus CH3 + H2

It prints, as a function of path position:
    ChemistryModel signed diagonal gap (product - reactant)
    open-shell SAPT EXCH10 interaction gap
    open-shell SAPT TOTAL interaction gap
    change in the interaction gap from omitted components

No reference barrier is used and no parameter is fitted.

Important:
SAPT interaction gaps are NOT complete diabatic-state energies because the
internal electronic energies of the different fragments are not included.
This experiment tests SHAPE and state-dependent interaction physics along the
path, not an absolute replacement energy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import traceback


HARTREE_TO_EV = 27.211386245988

DEFAULT_INPUT = Path(
    "research_data/sapt/sapt_minimax_path_probe.json"
)

DEFAULT_OUTPUT = Path(
    "research_data/sapt/sapt_open_shell_minimax_path_probe.csv"
)

DEFAULT_PSI4_OUTPUT = Path(
    "research_data/sapt/sapt_open_shell_minimax_path_probe_psi4.out"
)

FIELDS = (
    "system",
    "point_label",
    "path_index",
    "donor_A",
    "transfer_A",
    "relative_path_energy_eV",
    "model_reactant_diagonal_eV",
    "model_product_diagonal_eV",
    "model_signed_gap_product_minus_reactant_eV",
    "state",
    "fragment_a_charge",
    "fragment_a_mult",
    "fragment_b_charge",
    "fragment_b_mult",
    "exch10_eV",
    "exchange_eV",
    "electrostatics_eV",
    "induction_eV",
    "dispersion_eV",
    "sapt_total_eV",
    "total_minus_exch10_eV",
    "elapsed_s",
    "status",
    "error",
)


def atom_line(
    symbol,
    xyz,
):
    return (
        f"{symbol:2s} "
        f"{float(xyz[0]): .16f} "
        f"{float(xyz[1]): .16f} "
        f"{float(xyz[2]): .16f}"
    )


def geometry_text(
    point,
    state,
):
    symbols = point[
        "symbols"
    ]

    positions = point[
        "positions_A"
    ]

    lines = [
        f"{state['charge_a']} {state['mult_a']}",
    ]

    for index in state[
        "fragment_a"
    ]:
        lines.append(
            atom_line(
                symbols[
                    index
                ],
                positions[
                    index
                ],
            )
        )

    lines.extend(
        [
            "--",
            f"{state['charge_b']} {state['mult_b']}",
        ]
    )

    for index in state[
        "fragment_b"
    ]:
        lines.append(
            atom_line(
                symbols[
                    index
                ],
                positions[
                    index
                ],
            )
        )

    lines.extend(
        [
            "units angstrom",
            "symmetry c1",
            "no_reorient",
            "no_com",
        ]
    )

    return "\n".join(
        lines
    )


def get_var(
    psi4,
    *names,
):
    last = None

    for name in names:
        try:
            return float(
                psi4.core.variable(
                    name
                )
            )
        except Exception as exc:
            last = exc

    raise RuntimeError(
        "none of these Psi4 variables were available: "
        + ", ".join(
            names
        )
    ) from last


def run_one(
    psi4,
    point,
    state,
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

    molecule = psi4.geometry(
        geometry_text(
            point,
            state,
        )
    )

    psi4.set_options(
        {
            "basis": basis,
            "reference": "uhf",
            "scf_type": "df",
            "e_convergence": 1.0e-9,
            "d_convergence": 1.0e-9,
            "maxiter": 200,
            "guess": "sad",
        }
    )

    started = time.perf_counter()

    psi4.energy(
        "sapt0",
        molecule=molecule,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    values_h = {
        "exch10": get_var(
            psi4,
            "SAPT EXCH10 ENERGY",
        ),
        "exchange": get_var(
            psi4,
            "SAPT EXCH ENERGY",
        ),
        "electrostatics": get_var(
            psi4,
            "SAPT ELST ENERGY",
        ),
        "induction": get_var(
            psi4,
            "SAPT IND ENERGY",
        ),
        "dispersion": get_var(
            psi4,
            "SAPT DISP ENERGY",
        ),
        "total": get_var(
            psi4,
            "SAPT TOTAL ENERGY",
            "SAPT ENERGY",
            "SAPT0 TOTAL ENERGY",
        ),
    }

    return (
        {
            key: value
            * HARTREE_TO_EV
            for key, value
            in values_h.items()
        },
        elapsed,
    )


def write_rows(
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
            fieldnames=FIELDS,
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
                    in FIELDS
                }
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
        "--psi4-output",
        type=Path,
        default=DEFAULT_PSI4_OUTPUT,
    )

    parser.add_argument(
        "--basis",
        default="jun-cc-pvdz",
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
        "--force",
        action="store_true",
        help="re-run rows already present in the output CSV",
    )

    args = parser.parse_args()

    try:
        import psi4
    except ImportError as exc:
        raise RuntimeError(
            "Psi4 is not importable. Run this from the chem-sapt environment."
        ) from exc

    data = json.loads(
        args.input.read_text(
            encoding="utf-8",
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.psi4_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Windows-safe quieting: redirect verbose Psi4 output to a real file.
    # Psi4 documents set_output_file for this purpose; unlike be_quiet(),
    # this does not rely on /dev/null.
    psi4.set_output_file(
        str(
            args.psi4_output
        ),
        False,
    )

    cached = {}

    if (
        args.output.exists()
        and not args.force
    ):
        with args.output.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.DictReader(
                handle
            ):
                if row.get(
                    "status"
                ) == "ok":
                    key = (
                        row[
                            "system"
                        ],
                        row[
                            "point_label"
                        ],
                        row[
                            "state"
                        ],
                    )
                    cached[
                        key
                    ] = row

    rows = []

    print(
        "OPEN-SHELL SAPT0 MINIMAX-PATH PROBE"
    )
    print(
        "===================================="
    )
    print(
        f"input : {args.input}"
    )
    print(
        f"csv   : {args.output}"
    )
    print(
        f"Psi4  : {args.psi4_output}"
    )
    print()

    total_jobs = sum(
        2
        * len(
            system_data[
                "points"
            ]
        )
        for system_data
        in data[
            "systems"
        ].values()
    )

    job_index = 0

    for system, system_data in data[
        "systems"
    ].items():
        for point in system_data[
            "points"
        ]:
            for state_name in (
                "reactant",
                "product",
            ):
                job_index += 1

                key = (
                    system,
                    point[
                        "label"
                    ],
                    state_name,
                )

                prefix = (
                    f"[{job_index:02d}/{total_jobs:02d}] "
                    f"{system:<12} "
                    f"{point['label']:<13} "
                    f"{state_name:<8}"
                )

                if key in cached:
                    row = cached[
                        key
                    ]

                    # Normalize cached numeric fields back to usable strings;
                    # final summary casts them to float.
                    rows.append(
                        row
                    )

                    print(
                        f"{prefix} cached"
                    )
                    continue

                state = point[
                    "states"
                ][
                    state_name
                ]

                print(
                    f"{prefix} running...",
                    flush=True,
                )

                try:
                    components, elapsed = run_one(
                        psi4,
                        point,
                        state,
                        basis=args.basis,
                        threads=args.threads,
                        memory=args.memory,
                    )

                    row = {
                        "system": system,
                        "point_label": point[
                            "label"
                        ],
                        "path_index": point[
                            "path_index"
                        ],
                        "donor_A": point[
                            "donor_A"
                        ],
                        "transfer_A": point[
                            "transfer_A"
                        ],
                        "relative_path_energy_eV": point[
                            "relative_path_energy_eV"
                        ],
                        "model_reactant_diagonal_eV": point[
                            "model_reactant_diagonal_eV"
                        ],
                        "model_product_diagonal_eV": point[
                            "model_product_diagonal_eV"
                        ],
                        "model_signed_gap_product_minus_reactant_eV": point[
                            "model_signed_gap_product_minus_reactant_eV"
                        ],
                        "state": state_name,
                        "fragment_a_charge": state[
                            "charge_a"
                        ],
                        "fragment_a_mult": state[
                            "mult_a"
                        ],
                        "fragment_b_charge": state[
                            "charge_b"
                        ],
                        "fragment_b_mult": state[
                            "mult_b"
                        ],
                        "exch10_eV": components[
                            "exch10"
                        ],
                        "exchange_eV": components[
                            "exchange"
                        ],
                        "electrostatics_eV": components[
                            "electrostatics"
                        ],
                        "induction_eV": components[
                            "induction"
                        ],
                        "dispersion_eV": components[
                            "dispersion"
                        ],
                        "sapt_total_eV": components[
                            "total"
                        ],
                        "total_minus_exch10_eV": (
                            components[
                                "total"
                            ]
                            - components[
                                "exch10"
                            ]
                        ),
                        "elapsed_s": elapsed,
                        "status": "ok",
                        "error": "",
                    }

                    print(
                        f"{prefix} "
                        f"EXCH10={row['exch10_eV']:+.3f} "
                        f"TOTAL={row['sapt_total_eV']:+.3f}"
                    )

                except Exception as exc:
                    traceback.print_exc()

                    row = {
                        "system": system,
                        "point_label": point[
                            "label"
                        ],
                        "path_index": point[
                            "path_index"
                        ],
                        "donor_A": point[
                            "donor_A"
                        ],
                        "transfer_A": point[
                            "transfer_A"
                        ],
                        "relative_path_energy_eV": point[
                            "relative_path_energy_eV"
                        ],
                        "model_reactant_diagonal_eV": point[
                            "model_reactant_diagonal_eV"
                        ],
                        "model_product_diagonal_eV": point[
                            "model_product_diagonal_eV"
                        ],
                        "model_signed_gap_product_minus_reactant_eV": point[
                            "model_signed_gap_product_minus_reactant_eV"
                        ],
                        "state": state_name,
                        "fragment_a_charge": state[
                            "charge_a"
                        ],
                        "fragment_a_mult": state[
                            "mult_a"
                        ],
                        "fragment_b_charge": state[
                            "charge_b"
                        ],
                        "fragment_b_mult": state[
                            "mult_b"
                        ],
                        "status": "failed",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }

                    print(
                        f"{prefix} FAILED: {row['error']}"
                    )

                rows.append(
                    row
                )

                write_rows(
                    args.output,
                    rows,
                )

    # Compact comparison from successful paired states.
    print()
    print(
        "PATH-GAP COMPARISON"
    )
    print(
        "==================="
    )
    print(
        "SAPT gaps below are INTERACTION gaps only, not complete diabatic energies."
    )
    print()
    print(
        "system        point          idx  pathE    model_gap   EXCH10_gap   "
        "TOTAL_gap   omitted_delta"
    )

    by_key = {}

    for row in rows:
        if row.get(
            "status"
        ) != "ok":
            continue

        key = (
            row[
                "system"
            ],
            row[
                "point_label"
            ],
        )

        by_key.setdefault(
            key,
            {},
        )[
            row[
                "state"
            ]
        ] = row

    for system, system_data in data[
        "systems"
    ].items():
        for point in system_data[
            "points"
        ]:
            key = (
                system,
                point[
                    "label"
                ],
            )

            states = by_key.get(
                key,
                {},
            )

            if not (
                "reactant" in states
                and "product" in states
            ):
                print(
                    f"{system:<12} {point['label']:<13} "
                    f"{point['path_index']:3d}  incomplete"
                )
                continue

            reactant = states[
                "reactant"
            ]

            product = states[
                "product"
            ]

            exch_gap = (
                float(
                    product[
                        "exch10_eV"
                    ]
                )
                - float(
                    reactant[
                        "exch10_eV"
                    ]
                )
            )

            total_gap = (
                float(
                    product[
                        "sapt_total_eV"
                    ]
                )
                - float(
                    reactant[
                        "sapt_total_eV"
                    ]
                )
            )

            omitted_delta = (
                total_gap
                - exch_gap
            )

            model_gap = float(
                point[
                    "model_signed_gap_product_minus_reactant_eV"
                ]
            )

            print(
                f"{system:<12} "
                f"{point['label']:<13} "
                f"{point['path_index']:3d}  "
                f"{point['relative_path_energy_eV']:+7.3f}  "
                f"{model_gap:+10.3f}  "
                f"{exch_gap:+11.3f}  "
                f"{total_gap:+10.3f}  "
                f"{omitted_delta:+13.3f}"
            )

    print()
    print(
        "What matters:"
    )
    print(
        "  1. Whether the model signed gap and the open-shell interaction gaps "
        "change smoothly and with related direction along each path."
    )
    print(
        "  2. Whether formaldehyde and methane show genuinely different "
        "state-dependent interaction-gap evolution rather than one universal "
        "C-H correction."
    )
    print(
        "  3. Do NOT compare the absolute SAPT interaction gap to the model "
        "diabatic gap as though they were the same energy; fragment internal "
        "energies are not included here."
    )
    print()
    print(
        f"saved CSV: {args.output}"
    )
    print(
        f"verbose Psi4 output: {args.psi4_output}"
    )


if __name__ == "__main__":
    main()
