import hashlib
from chemistry_format import molecular_formula

import numpy as np

import reactive as R


# ============================================================
# Telling isomers apart
# ============================================================
#
# Counting atoms is not enough to identify a molecule. C2OH6 is
# either ethanol, CH3-CH2-OH, or dimethyl ether, CH3-O-CH3. Same
# atoms, entirely different substances, and a report that calls
# both of them "C2OH6" has quietly merged them.
#
# It matters more the moment species become nodes in a reaction
# network, because pooling two different molecules into one node
# averages their chemistry into something that describes neither.
#
# The fix is a canonical fingerprint of the connectivity. Each
# atom starts labelled by its element, then repeatedly relabels
# itself from its own label plus its neighbours' labels. After a
# few rounds an atom's label encodes its whole local environment,
# and the sorted multiset of labels identifies the structure
# regardless of how the atoms happened to be numbered.
#
# This is Weisfeiler-Lehman refinement. It is not a perfect graph
# invariant - a few exotic pairs collide - but for small
# molecules it is reliable, and it is fast.


ROUNDS = 3


def short_hash(text):
    return hashlib.blake2s(
        text.encode(), digest_size=4
    ).hexdigest()


def fingerprint(symbols, members, bonds, orders=None):
    # members: indices of the atoms in this molecule
    # bonds: dict of atom -> list of (neighbour, order)

    labels = {
        index: symbols[index] for index in members
    }

    for _ in range(ROUNDS):
        updated = {}

        for index in members:
            # Connectivity only. Bond order was in here at
            # first, but it is computed from the surrounding
            # coordination, so the same two-atom fragment came out
            # with different orders depending on what happened to
            # be nearby, and split into phantom isomers. A pair of
            # atoms has one possible structure.

            neighbours = sorted(
                labels[other]
                for other, order in bonds.get(index, [])
                if other in labels
            )

            updated[index] = short_hash(
                labels[index] + "|" + ",".join(neighbours)
            )

        labels = updated

    return short_hash(",".join(sorted(labels.values())))


def formula_for(symbols, members):
    return molecular_formula(symbols[member] for member in members)


def describe(symbols, members, bonds):
    # A readable sketch of the connectivity: every heavy atom and
    # what hangs off it. Not a name, but enough to tell ethanol
    # from dimethyl ether at a glance.

    pieces = []

    for index in sorted(members):
        symbol = symbols[index]

        if symbol == "H":
            continue

        neighbours = bonds.get(index, [])

        hydrogens = sum(
            1 for other, order in neighbours
            if symbols[other] == "H"
        )

        heavy = sorted(
            symbols[other] for other, order in neighbours
            if symbols[other] != "H"
        )

        part = symbol

        if hydrogens:
            part += f"H{hydrogens}" if hydrogens > 1 else "H"

        if heavy:
            part += "(" + "".join(heavy) + ")"

        pieces.append(part)

    return " ".join(pieces) if pieces else "".join(
        symbols[index] for index in members
    )


def bond_table(positions, types, box_size, threshold=0.35):
    # Neighbour lists with bond orders, built once per frame.

    count = len(positions)

    offsets = positions[None, :, :] - positions[:, None, :]
    offsets -= box_size * np.round(offsets / box_size)

    distance_squared = np.sum(offsets ** 2, axis=2)
    np.fill_diagonal(distance_squared, np.inf)

    distances = np.sqrt(distance_squared)

    inner = R.CUTOFF_INNER[np.ix_(types, types)]
    outer = R.CUTOFF_OUTER[np.ix_(types, types)]

    taper = R.smooth_cutoff(distances, inner, outer)
    np.fill_diagonal(taper, 0.0)

    order, _ = R.bond_orders(taper, types)

    bonded = taper > threshold

    table = {}

    first, second = np.where(np.triu(bonded, k=1))

    for a, b in zip(first, second):
        rounded = float(np.clip(np.rint(order[a, b]), 1, 3))

        table.setdefault(int(a), []).append((int(b), rounded))
        table.setdefault(int(b), []).append((int(a), rounded))

    return table, first, second


class Registry:
    # Fingerprints are collected first and named afterwards.
    #
    # Naming as you go does not work: the first structure with a
    # given formula would be called C2OH6, and then a second
    # structure turning up later would force it to become C2OH6a,
    # leaving every label already recorded wrong. Two passes keeps
    # names stable.

    def __init__(self):
        self.by_formula = {}
        self.details = {}
        self.names = None

    def register(self, symbols, members, bonds):
        formula = formula_for(symbols, members)

        key = fingerprint(symbols, members, bonds)

        if key not in self.details:
            self.details[key] = {
                "formula": formula,
                "structure": describe(symbols, members, bonds),
                "atoms": len(members),
            }

            self.by_formula.setdefault(formula, []).append(key)

        return key

    def finalise(self):
        # A formula with one structure keeps its plain name. A
        # formula with several gets a, b, c and so on, in the
        # order the structures were first seen.

        self.names = {}

        for formula, keys in self.by_formula.items():
            if len(keys) == 1:
                self.names[keys[0]] = formula
            else:
                for position, key in enumerate(keys):
                    self.names[key] = (
                        formula + "abcdefghijkl"[position]
                    )

        return self.names

    def name(self, key):
        if self.names is None:
            self.finalise()

        return self.names.get(key, "?")

    def structure(self, key):
        return self.details.get(key, {}).get("structure", "")

    def isomers(self):
        return {
            formula: len(keys)
            for formula, keys in self.by_formula.items()
            if len(keys) > 1
        }
