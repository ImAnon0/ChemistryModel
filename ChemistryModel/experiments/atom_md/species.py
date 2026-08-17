import numpy as np
from chemistry_format import molecular_formula

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


# ASE's NeighborList defaults to skin=0.3, which silently adds
# 0.3 A to every cutoff. The skin exists so an MD neighbour list
# stays valid for several steps; for bond detection it just means
# the real cutoff is mult * (r_i + r_j) + 0.3, not what the
# tolerance says. That is why mult=1.2 percolated the whole box
# into one fictitious molecule: the effective C-C cutoff was
# 2.12 A, not 1.82 A. It is set to zero below so this number
# means what it appears to mean.
#
# Calibrated against a box of known molecules. The binding
# constraint is H2, whose 0.74 A bond needs at least 1.19x the
# 0.62 A covalent radius sum. Below that, real H2 goes undetected.
# The upper limit is set by non-bonded contacts, which sit above
# 2.3 A, against a widest cutoff of C-C at 1.52 * mult.

DEFAULT_BOND_TOLERANCE = 1.25


# At a few hundred Kelvin atoms brush past each other constantly.
# A contact lasting one frame is a collision, not a bond, so a
# pair has to survive this many consecutive frames before it
# counts. Raise it if the tally still flickers.

DEFAULT_FRAMES_REQUIRED = 3


def find_bonded_pairs(atoms, bond_tolerance=DEFAULT_BOND_TOLERANCE):
    # Every pair closer than the sum of their covalent radii times
    # bond_tolerance, as (lower_index, higher_index).

    cutoffs = natural_cutoffs(atoms, mult=bond_tolerance)

    neighbour_list = NeighborList(
        cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=True
    )

    neighbour_list.update(atoms)

    connectivity = neighbour_list.get_connectivity_matrix(
        sparse=False
    )

    first_indices, second_indices = np.nonzero(
        np.triu(connectivity, k=1)
    )

    return set(
        zip(
            first_indices.tolist(),
            second_indices.tolist()
        )
    )


def formulas_from_pairs(atoms, bonded_pairs):
    # Connected groups of bonded atoms are molecules.

    atom_count = len(atoms)

    connectivity = np.zeros(
        (atom_count, atom_count),
        dtype=int
    )

    for first_index, second_index in bonded_pairs:
        connectivity[first_index, second_index] = 1
        connectivity[second_index, first_index] = 1

    group_count, group_labels = connected_components(
        csr_matrix(connectivity),
        directed=False
    )

    symbols = np.array(atoms.get_chemical_symbols())

    formulas = []

    for group_index in range(group_count):
        members = symbols[group_labels == group_index]

        formulas.append(molecular_formula(members))

    return formulas


def find_molecules(atoms, bond_tolerance=DEFAULT_BOND_TOLERANCE):
    # Single-frame version. Fine for a one-off check, but it has
    # no way to tell a bond from a passing collision, so prefer
    # MoleculeTracker for anything live.

    return formulas_from_pairs(
        atoms,
        find_bonded_pairs(atoms, bond_tolerance)
    )


def tally_formulas(formulas):
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


def summarise_molecules(atoms, bond_tolerance=DEFAULT_BOND_TOLERANCE):
    return tally_formulas(
        find_molecules(atoms, bond_tolerance)
    )


class MoleculeTracker:
    # Holds bond history between frames so a pair only counts once
    # it has persisted. Without this the tally flickers with every
    # thermal collision and invents species that never existed.

    def __init__(
        self,
        bond_tolerance=DEFAULT_BOND_TOLERANCE,
        frames_required=DEFAULT_FRAMES_REQUIRED
    ):
        self.bond_tolerance = bond_tolerance
        self.frames_required = frames_required

        self.consecutive_frames = {}

        self.has_started = False

    def update(self, atoms):
        present_pairs = find_bonded_pairs(
            atoms,
            self.bond_tolerance
        )

        if not self.has_started:
            # The opening configuration is taken as already
            # settled. Without this the first few readouts show
            # every atom loose, because no pair has accumulated
            # its streak yet, which reads as though the starting
            # molecules do not exist.

            self.has_started = True

            self.consecutive_frames = {
                pair: self.frames_required
                for pair in present_pairs
            }
        else:
            # Rebuilding from present_pairs rather than editing in
            # place means a pair that separates loses its streak
            # automatically.

            self.consecutive_frames = {
                pair: self.consecutive_frames.get(pair, 0) + 1
                for pair in present_pairs
            }

        confirmed_pairs = {
            pair
            for pair, frames in self.consecutive_frames.items()
            if frames >= self.frames_required
        }

        return formulas_from_pairs(atoms, confirmed_pairs)

    def summarise(self, atoms):
        return tally_formulas(self.update(atoms))
