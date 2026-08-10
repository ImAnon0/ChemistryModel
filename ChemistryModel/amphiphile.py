import numpy as np

import reactive as R


# ============================================================
# Looking for lipids
# ============================================================
#
# A membrane lipid is a shape before it is a molecule: a polar
# group at one end, a run of carbons trailing away from it, and
# nothing polar along the tail. That shape is what makes a
# bilayer form, and it is what the coarse-grained membrane model
# represents as one head bead and two tail beads.
#
# Real fatty acids that form vesicles start around ten carbons.
# Decanoic acid works; anything much shorter dissolves instead of
# assembling, because the tail cannot bury itself properly.
# Branching hurts too, since a polar group stuck in the middle of
# a molecule cannot sit at a membrane surface.
#
# So this measures three things about every molecule: how long
# its longest carbon chain is, whether a polar group sits at one
# end of that chain, and how much of the chain is clean of other
# polar groups. Together they say how close a run got.


POLAR = ("O", "N")

# Below this the tail is too short to bury and no membrane forms.

MINIMUM_TAIL = 4

# Roughly where real fatty acids start working.

VESICLE_TAIL = 8


def neighbours_of(members, bonds):
    return {
        index: [
            other for other, order in bonds.get(index, [])
            if other in members
        ]
        for index in members
    }


def longest_carbon_path(members, bonds, symbols):
    # Longest chain of carbons bonded end to end.
    #
    # Depth-first from every carbon. Molecules here are small
    # enough that the cost does not matter, and it handles rings
    # by refusing to revisit an atom.

    carbons = [
        index for index in members if symbols[index] == "C"
    ]

    if not carbons:
        return []

    links = neighbours_of(set(carbons), bonds)

    best = []

    def walk(current, visited, path):
        nonlocal best

        if len(path) > len(best):
            best = list(path)

        for other in links.get(current, []):
            if other in visited:
                continue

            visited.add(other)
            path.append(other)

            walk(other, visited, path)

            path.pop()
            visited.remove(other)

    for start in carbons:
        walk(start, {start}, [start])

    return best


def describe_amphiphile(members, bonds, symbols):
    # Returns None when the molecule is not amphiphile-shaped,
    # otherwise a description of how good a lipid it would make.

    chain = longest_carbon_path(members, bonds, symbols)

    if len(chain) < 2:
        return None

    links = neighbours_of(set(members), bonds)

    def polar_attached(index):
        return [
            other for other in links.get(index, [])
            if symbols[other] in POLAR
        ]

    head_end = polar_attached(chain[0])
    tail_end = polar_attached(chain[-1])

    if not head_end and not tail_end:
        return None

    # Whichever end carries the polar group becomes the head, so
    # the tail is measured running away from it.

    if head_end and not tail_end:
        tail = chain
        head = head_end
    elif tail_end and not head_end:
        tail = chain[::-1]
        head = tail_end
    else:
        # Polar at both ends is a diol or diamine: useful
        # chemistry, but it cannot sit at a membrane surface,
        # since one head would have to be buried.
        return {
            "chain": len(chain),
            "clean_tail": 0,
            "head": "both ends",
            "shape": "polar at both ends",
            "vesicle_ready": False,
        }

    # How far along the tail you can go before hitting another
    # polar group. A hydroxyl halfway down ruins the packing.

    clean = 0

    for index in tail[1:]:
        if polar_attached(index):
            break

        clean += 1

    head_symbols = "".join(sorted(symbols[i] for i in head))

    return {
        "chain": len(chain),
        "clean_tail": clean,
        "head": head_symbols,
        "shape": (
            "straight" if is_straight(tail, links, symbols)
            else "branched"
        ),
        "vesicle_ready": clean >= VESICLE_TAIL,
    }


def is_straight(chain, links, symbols):
    # A carbon in the middle of a straight chain has exactly two
    # carbon neighbours. Three means the chain branches there.

    for index in chain[1:-1]:
        carbon_neighbours = sum(
            1 for other in links.get(index, [])
            if symbols[other] == "C"
        )

        if carbon_neighbours > 2:
            return False

    return True


def scan(final, bonds, symbols):
    # Looks at every molecule in a frame and returns the best
    # amphiphile found, plus how many there were.

    best = None
    count = 0

    for entry in final:
        result = describe_amphiphile(
            entry["members"], bonds, symbols
        )

        if result is None:
            continue

        if result["clean_tail"] >= MINIMUM_TAIL:
            count += 1

        result["formula"] = entry["formula"]
        result["atoms"] = entry["atoms"]
        result["closed_shell"] = entry["closed_shell"]

        if best is None or (
            result["clean_tail"],
            result["chain"],
        ) > (best["clean_tail"], best["chain"]):
            best = result

    if best is None:
        return {
            "best_tail": 0,
            "best_chain": 0,
            "amphiphiles": 0,
            "best_formula": "",
            "best_head": "",
            "best_shape": "",
            "vesicle_ready": False,
        }

    return {
        "best_tail": best["clean_tail"],
        "best_chain": best["chain"],
        "amphiphiles": count,
        "best_formula": best["formula"],
        "best_head": best["head"],
        "best_shape": best["shape"],
        "vesicle_ready": bool(best["vesicle_ready"]),
    }
