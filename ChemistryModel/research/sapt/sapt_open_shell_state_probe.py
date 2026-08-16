"""
Open-shell SAPT0 probe of the ACTUAL diabatic fragment states.

Run this from the Psi4 environment:

    conda activate chem-sapt
    python sapt_open_shell_state_probe.py

It reads:
    research_data/sapt/sapt_state_fragment_geometries.json

and evaluates the two state fragmentations at each current frozen barrier:

formaldehyde:
    reactant-like  CH2O + H
    product-like   HCO  + H2

methane:
    reactant-like  CH4 + H
    product-like   CH3 + H2

water:
    reactant-like  H2O + OH
    product-like   OH  + H2O

This uses open-shell SAPT0/UHF directly. No barrier is fitted.

The important quantity is the state DIFFERENCE in omitted interaction:

    correction_vs_EXCH10 = SAPT_TOTAL - SAPT_EXCH10

If the product-like state gets much more negative correction than the
reactant-like state, then using exchange alone artificially lifts that
diabatic state.

Output:
    research_data/sapt/sapt_open_shell_state_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback


HARTREE_TO_EV = 27.211386245988

DEFAULT_INPUT = Path(
    "research_data/sapt/sapt_state_fragment_geometries.json"
)

DEFAULT_OUTPUT = Path(
    "research_data/sapt/sapt_open_shell_state_probe.csv"
)

FIELDS = (
    "system",
    "state",
    "donor_A",
    "transfer_A",
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
    "total_minus_exchange_eV",
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
    case,
    state,
):
    symbols = case[
        "symbols"
    ]

    positions = case[
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
    case,
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

    # Do not call psi4.core.be_quiet() here.
    # On this Windows Psi4 build it tries to open /dev/null and crashes.
    # Leaving normal Psi4 output enabled is harmless for this six-job probe.

    molecule = psi4.geometry(
        geometry_text(
            case,
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
            "SAPT ENERGY",
            "SAPT TOTAL ENERGY",
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

    rows = []

    print(
        "OPEN-SHELL SAPT0 DIABATIC STATE PROBE"
    )
    print(
        "======================================"
    )
    print(
        "These are the actual state fragmentations, not the old H2 molecular proxy."
    )
    print()

    for system, case in data[
        "cases"
    ].items():
        print(
            system.upper()
        )
        print(
            "-" * len(
                system
            )
        )

        system_rows = []

        for state_name in (
            "reactant",
            "product",
        ):
            state = case[
                "states"
            ][
                state_name
            ]

            print(
                f"  running {state_name}...",
                flush=True,
            )

            try:
                components, elapsed = run_one(
                    psi4,
                    case,
                    state,
                    basis=args.basis,
                    threads=args.threads,
                    memory=args.memory,
                )

                row = {
                    "system": system,
                    "state": state_name,
                    "donor_A": case[
                        "donor_A"
                    ],
                    "transfer_A": case[
                        "transfer_A"
                    ],
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
                    "total_minus_exchange_eV": (
                        components[
                            "total"
                        ]
                        - components[
                            "exchange"
                        ]
                    ),
                    "elapsed_s": elapsed,
                    "status": "ok",
                    "error": "",
                }

            except Exception as exc:
                traceback.print_exc()

                row = {
                    "system": system,
                    "state": state_name,
                    "donor_A": case[
                        "donor_A"
                    ],
                    "transfer_A": case[
                        "transfer_A"
                    ],
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

            rows.append(
                row
            )

            system_rows.append(
                row
            )

            write_rows(
                args.output,
                rows,
            )

            if row[
                "status"
            ] == "ok":
                print(
                    f"    EXCH10 {row['exch10_eV']:+.6f} eV"
                )
                print(
                    f"    ELST   {row['electrostatics_eV']:+.6f} eV"
                )
                print(
                    f"    IND    {row['induction_eV']:+.6f} eV"
                )
                print(
                    f"    DISP   {row['dispersion_eV']:+.6f} eV"
                )
                print(
                    f"    TOTAL  {row['sapt_total_eV']:+.6f} eV"
                )
                print(
                    f"    total-EXCH10 "
                    f"{row['total_minus_exch10_eV']:+.6f} eV"
                )
            else:
                print(
                    f"    FAILED: {row['error']}"
                )

        good = [
            row
            for row in system_rows
            if row[
                "status"
            ] == "ok"
        ]

        if len(
            good
        ) == 2:
            by_state = {
                row[
                    "state"
                ]: row
                for row in good
            }

            reactant = by_state[
                "reactant"
            ]

            product = by_state[
                "product"
            ]

            correction_gap = (
                product[
                    "total_minus_exch10_eV"
                ]
                - reactant[
                    "total_minus_exch10_eV"
                ]
            )

            exch10_gap = (
                product[
                    "exch10_eV"
                ]
                - reactant[
                    "exch10_eV"
                ]
            )

            total_gap = (
                product[
                    "sapt_total_eV"
                ]
                - reactant[
                    "sapt_total_eV"
                ]
            )

            print()
            print(
                f"  product-reactant EXCH10 interaction gap  "
                f"{exch10_gap:+.6f} eV"
            )
            print(
                f"  product-reactant TOTAL interaction gap   "
                f"{total_gap:+.6f} eV"
            )
            print(
                f"  omitted-components change in state gap   "
                f"{correction_gap:+.6f} eV"
            )

        print()

    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
