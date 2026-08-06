import numpy as np

from ase.neighborlist import natural_cutoffs, NeighborList

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# Rough colours by element, in the style chemists expect.

ELEMENT_COLOURS = {
    "H": "#e8e8e8",
    "O": "#e8402a",
    "C": "#404040",
    "N": "#2a52e8",
    "S": "#e8d92a",
    "Ar": "#5ac8b8",
    "Si": "#c8a05a"
}

DEFAULT_COLOUR = "#a06ad0"


# Marker sizes roughly follow atomic size.

ELEMENT_SIZES = {
    "H": 55,
    "O": 130,
    "C": 120,
    "N": 120,
    "S": 150,
    "Ar": 150,
    "Si": 160
}

DEFAULT_SIZE = 110


def colours_for(atoms):
    return [
        ELEMENT_COLOURS.get(symbol, DEFAULT_COLOUR)
        for symbol in atoms.get_chemical_symbols()
    ]


def sizes_for(atoms):
    return [
        ELEMENT_SIZES.get(symbol, DEFAULT_SIZE)
        for symbol in atoms.get_chemical_symbols()
    ]


def find_molecules(atoms, bond_tolerance=1.2):
    # Two atoms count as bonded if they are closer than the sum
    # of their covalent radii times bond_tolerance. Connected
    # groups of bonded atoms are molecules.

    cutoffs = natural_cutoffs(atoms, mult=bond_tolerance)

    neighbour_list = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True
    )

    neighbour_list.update(atoms)

    connectivity = neighbour_list.get_connectivity_matrix(
        sparse=False
    )

    group_count, group_labels = connected_components(
        csr_matrix(connectivity),
        directed=False
    )

    symbols = np.array(atoms.get_chemical_symbols())

    formulas = []

    for group_index in range(group_count):
        members = symbols[group_labels == group_index]

        counts = {}

        for symbol in members:
            counts[symbol] = counts.get(symbol, 0) + 1

        formula = ""

        for symbol in sorted(counts):
            formula += symbol

            if counts[symbol] > 1:
                formula += str(counts[symbol])

        formulas.append(formula)

    return formulas


def summarise_molecules(atoms, bond_tolerance=1.2):
    formulas = find_molecules(atoms, bond_tolerance)

    tally = {}

    for formula in formulas:
        tally[formula] = tally.get(formula, 0) + 1

    parts = []

    for formula in sorted(
        tally,
        key=lambda f: (-tally[f], f)
    ):
        count = tally[formula]

        parts.append(
            f"{count}x{formula}" if count > 1 else formula
        )

    return "  ".join(parts)
