"""Extract reactant/transition-state/product endpoints from Transition1x.

Transition1x (Schreiner et al., Sci. Data 2022, doi:10.1038/s41597-022-01870-w)
contains NEB pathways for ~10k elementary organic reactions at the
wB97x/6-31G(d) level, restricted to H/C/N/O.  That element coverage matches
ChemistryModel exactly, and the transition-state regions are the part of the
surface that near-equilibrium benchmarks never probe.

HDF5 layout, per the paper's Data Records section:

    /{split}/{formula}/{rxn}/            split in: data, train, val, test
        atomic_numbers   (m,)
        energy           (n,)            all saved NEB images
        forces           (n, m, 3)
        positions        (n, m, 3)
        /reactant/           n = 1       same structure, single configuration
        /transition_state/   n = 1
        /product/            n = 1

Only the three n=1 endpoint groups are read here.  Energies are in eV
(the dataset was produced through ASE, and the paper quotes barriers and
force thresholds in eV and eV/A).

Only energy DIFFERENCES within a reaction are ever scored downstream, so
ChemistryModel's absolute energy zero never has to match the reference.  The
composition is identical across the three endpoints of one reaction, which
makes that cancellation exact rather than approximate.

    python extract_transition1x_endpoints.py --inspect
    python extract_transition1x_endpoints.py --split test --limit 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT = Path("transition1x.h5")
DEFAULT_OUTPUT = Path("research_data/benchmark/transition1x_endpoints.json")

# Transition1x is neutral CHNO throughout.
SUPPORTED_NUMBERS = {1: "H", 6: "C", 7: "N", 8: "O"}

ENDPOINTS = ("reactant", "transition_state", "product")


def _as_symbols(numbers):
    symbols = []
    for number in np.asarray(numbers).reshape(-1).tolist():
        try:
            symbols.append(SUPPORTED_NUMBERS[int(number)])
        except KeyError as problem:
            raise ValueError(
                f"unsupported atomic number in Transition1x record: {number}"
            ) from problem
    return symbols


def _single_energy(endpoint):
    values = np.asarray(endpoint["energy"]).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"endpoint should hold one configuration, found {values.size}"
        )
    return float(values[0])


def _single_positions(endpoint, atom_count):
    coordinates = np.asarray(endpoint["positions"]).reshape(-1, 3)
    if coordinates.shape[0] != atom_count:
        raise ValueError(
            f"positions hold {coordinates.shape[0]} atoms, "
            f"atomic_numbers holds {atom_count}"
        )
    return coordinates.tolist()


def inspect(path, depth=4):
    """Print the top of the HDF5 tree so the layout can be confirmed by eye."""
    with h5py.File(path, "r") as handle:
        def walk(group, prefix="", level=0):
            if level > depth:
                return
            for key in list(group.keys())[:5]:
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(f"{prefix}{key}/  (group, {len(item)} entries)")
                    walk(item, prefix + "  ", level + 1)
                else:
                    print(f"{prefix}{key}  (dataset, shape={item.shape})")
            if len(group) > 5:
                print(f"{prefix}... {len(group) - 5} more")

        walk(handle)


def _reaction_groups(root):
    """Yield (reaction_id, group) for reaction groups holding all endpoints."""
    for formula in root.keys():
        formula_group = root[formula]
        if not isinstance(formula_group, h5py.Group):
            continue
        for reaction in formula_group.keys():
            group = formula_group[reaction]
            if not isinstance(group, h5py.Group):
                continue
            if all(name in group for name in ENDPOINTS):
                yield f"{formula}/{reaction}", group


def extract(path, split="test", limit=None):
    geometries = []
    reactions = 0
    skipped = []

    with h5py.File(path, "r") as handle:
        if split not in handle:
            raise SystemExit(
                f"split {split!r} not in file; available: {list(handle.keys())}"
            )
        for reaction_id, group in _reaction_groups(handle[split]):
            if limit is not None and reactions >= limit:
                break
            try:
                symbols = _as_symbols(group["atomic_numbers"])
                rows = [{
                    "geometry_id": f"{reaction_id}/{region}",
                    "system": reaction_id,
                    "reaction_id": reaction_id,
                    "region": region,
                    "split": split,
                    "sample_kind": "transition1x_endpoint",
                    "charge": 0,
                    "symbols": list(symbols),
                    "coordinates_angstrom": _single_positions(
                        group[region], len(symbols)
                    ),
                    "reference_energy_eV": _single_energy(group[region]),
                } for region in ENDPOINTS]

                geometries.extend(rows)
                reactions += 1
            except Exception as problem:
                skipped.append(
                    f"{reaction_id}: {type(problem).__name__}: {problem}"
                )

    return geometries, reactions, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="test",
                        choices=("data", "train", "val", "test"),
                        help="use 'test' for comparability with published numbers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inspect", action="store_true",
                        help="print the HDF5 tree and exit")
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(
            f"{arguments.input} not found. Download Transition1x.h5 from "
            "https://doi.org/10.6084/m9.figshare.19614657.v4 or via "
            "https://gitlab.com/matschreiner/Transition1x"
        )

    if arguments.inspect:
        inspect(arguments.input)
        return

    geometries, reactions, skipped = extract(
        arguments.input, split=arguments.split, limit=arguments.limit
    )

    if not geometries:
        raise SystemExit("no reactions extracted; run --inspect to check layout")

    payload = {
        "schema_version": 1,
        "source": "Transition1x (Schreiner et al., Sci. Data 2022)",
        "source_doi": "10.1038/s41597-022-01870-w",
        "reference_level": "wB97x/6-31G(d)",
        "split": arguments.split,
        "units": {"coordinates": "angstrom", "energy": "eV"},
        "energy_alignment": "within-reaction differences only",
        "counts": {
            "reactions": reactions,
            "geometries": len(geometries),
            "skipped": len(skipped),
        },
        "skipped": skipped[:50],
        "geometries": geometries,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{reactions} reactions from split {arguments.split!r} "
          f"-> {arguments.output}")
    if skipped:
        print(f"{len(skipped)} skipped; first few:")
        for line in skipped[:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
