"""
Export exact ChemistryModel scanner geometries for open-shell SAPT state probes.

Run from the normal/base ChemistryModel environment (the one with torch):

    py export_sapt_state_fragment_geometries.py

This writes:
    research_data/sapt/sapt_state_fragment_geometries.json

The selected points are the current frozen SAPT barrier cells:
    formaldehyde  1.28 / 1.01 A
    water         1.10 / 1.30 A
    methane       1.44 / 0.85 A

Each point contains two physically meaningful diabatic fragmentations:

formaldehyde:
    reactant-like  CH2O + H
    product-like   HCO  + H2

methane:
    reactant-like  CH4 + H
    product-like   CH3 + H2

water:
    reactant-like  H2O + OH
    product-like   OH  + H2O

No energy calculation is done here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import hf_surface_scan as scan


OUTPUT = Path(
    "research_data/sapt/sapt_state_fragment_geometries.json"
)


CASES = {
    "formaldehyde": {
        "donor": 1.28,
        "transfer": 1.01,
        "states": {
            "reactant": {
                "fragment_a": [0, 1, 2, 3],  # CH2O
                "fragment_b": [4],           # H
                "charge_a": 0,
                "mult_a": 1,
                "charge_b": 0,
                "mult_b": 2,
            },
            "product": {
                "fragment_a": [0, 1, 3],     # HCO
                "fragment_b": [2, 4],        # H2
                "charge_a": 0,
                "mult_a": 2,
                "charge_b": 0,
                "mult_b": 1,
            },
        },
    },
    "water": {
        "donor": 1.10,
        "transfer": 1.30,
        "states": {
            "reactant": {
                "fragment_a": [0, 1, 2],     # H2O
                "fragment_b": [3, 4],        # OH
                "charge_a": 0,
                "mult_a": 1,
                "charge_b": 0,
                "mult_b": 2,
            },
            "product": {
                "fragment_a": [0, 2],        # OH
                "fragment_b": [1, 3, 4],     # H2O
                "charge_a": 0,
                "mult_a": 2,
                "charge_b": 0,
                "mult_b": 1,
            },
        },
    },
    "methane": {
        "donor": 1.44,
        "transfer": 0.85,
        "states": {
            "reactant": {
                "fragment_a": [0, 1, 2, 3, 4],  # CH4
                "fragment_b": [5],              # H
                "charge_a": 0,
                "mult_a": 1,
                "charge_b": 0,
                "mult_b": 2,
            },
            "product": {
                "fragment_a": [0, 2, 3, 4],     # CH3
                "fragment_b": [1, 5],           # H2
                "charge_a": 0,
                "mult_a": 2,
                "charge_b": 0,
                "mult_b": 1,
            },
        },
    },
}


def main():
    output = {
        "schema_version": 1,
        "note": (
            "Exact hf_surface_scan frozen geometries at current SAPT barrier "
            "cells, split into physically meaningful diabatic fragments."
        ),
        "cases": {},
    }

    for system, spec in CASES.items():
        scan.apply_system(
            system
        )

        spectators = np.asarray(
            scan.SYSTEMS[
                system
            ][
                "frozen"
            ],
            dtype=float,
        )

        symbols, positions = scan.SYSTEMS[
            system
        ][
            "geometry"
        ](
            float(
                spec[
                    "donor"
                ]
            ),
            float(
                spec[
                    "transfer"
                ]
            ),
            spectators,
        )

        output[
            "cases"
        ][
            system
        ] = {
            "donor_A": spec[
                "donor"
            ],
            "transfer_A": spec[
                "transfer"
            ],
            "spectators": spectators.tolist(),
            "symbols": list(
                symbols
            ),
            "positions_A": np.asarray(
                positions,
                dtype=float,
            ).tolist(),
            "states": spec[
                "states"
            ],
        }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"saved: {OUTPUT}"
    )

    for system, case in output[
        "cases"
    ].items():
        print()
        print(
            system.upper()
        )
        print(
            f"  donor/transfer: "
            f"{case['donor_A']:.3f}/{case['transfer_A']:.3f} A"
        )
        print(
            f"  atoms: "
            f"{' '.join(case['symbols'])}"
        )

        for state_name, state in case[
            "states"
        ].items():
            frag_a = " ".join(
                f"{case['symbols'][i]}{i}"
                for i in state[
                    "fragment_a"
                ]
            )
            frag_b = " ".join(
                f"{case['symbols'][i]}{i}"
                for i in state[
                    "fragment_b"
                ]
            )
            print(
                f"  {state_name:<8}: "
                f"[{frag_a}]  +  [{frag_b}]"
            )


if __name__ == "__main__":
    main()
