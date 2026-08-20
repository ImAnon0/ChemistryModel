from pathlib import Path
import json
import csv
import numpy as np
from collections import defaultdict


ENDPOINTS = Path(
    "research_data/benchmark/grambow_endpoints.json"
)

SCORES = Path(
    "research_data/benchmark/grambow_scores.json"
)


def distance(a, b):
    return np.linalg.norm(
        np.array(a) - np.array(b)
    )


def guess_bonds(symbols, coords):
    """
    Simple geometry bond guess.
    Not ChemistryModel physics yet.
    Used to identify geometry changes.
    """

    covalent = {
        "H": 0.37,
        "C": 0.77,
        "N": 0.75,
        "O": 0.73,
    }

    bonds = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):

            r = distance(
                coords[i],
                coords[j]
            )

            cutoff = (
                covalent[symbols[i]]
                +
                covalent[symbols[j]]
            ) * 1.25

            if r < cutoff:
                bonds.append(
                    (
                        i,
                        j,
                        symbols[i],
                        symbols[j],
                        r
                    )
                )

    return bonds


def load():
    with open(ENDPOINTS, encoding="utf8") as f:
        endpoints = json.load(f)["geometries"]

    with open(SCORES, encoding="utf8") as f:
        scores = list(csv.DictReader(f))

    return endpoints, scores


def main():

    endpoints, scores = load()

    bad = sorted(
        scores,
        key=lambda x: abs(float(x["barrier_error_eV"])),
        reverse=True
    )

    endpoint_map = defaultdict(dict)

    for e in endpoints:
        endpoint_map[
            e["reaction_id"]
        ][
            e["region"]
        ] = e


    print("="*80)
    print("GRAMBOW BOND ANALYSIS")
    print("="*80)


    for reaction in bad[:20]:

        rid = reaction["reaction_id"]

        print()
        print(
            rid,
            "barrier error",
            reaction["barrier_error_eV"]
        )


        if rid not in endpoint_map:
            continue


        for region in [
            "reactant",
            "transition_state",
            "product"
        ]:

            data = endpoint_map[rid][region]

            bonds = guess_bonds(
                data["symbols"],
                data["coordinates_angstrom"]
            )

            print(
                f"\n {region}: {len(bonds)} bonds"
            )

            for b in bonds:
                print(
                    f"   {b[2]}-{b[3]}"
                    f" ({b[4]:.2f} A)"
                )


if __name__ == "__main__":
    main()