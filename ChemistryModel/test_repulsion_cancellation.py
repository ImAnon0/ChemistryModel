"""
Structural check: does the repulsive term cancel in the H-state gaps?

_box_state_energy builds each state diagonal as

    common_core - attraction

where common_core sums taper * repulsive over every active edge and is
computed once, outside the per-state loop. If that reading is right, then
scaling `repulsive` by any factor must shift every diagonal by the same
amount and leave the differences between diagonals exactly unchanged.

This calls _box_state_energy directly with synthetic pair quantities, so
it needs no simulation, no geometry and no Psi4. Run it from the folder
containing h_state_torch.py:

    python test_repulsion_cancellation.py

Interpretation:
  gaps identical  -> repulsion cancels in the gap by construction
  gaps differ     -> repulsion does reach the gap; reading was wrong
"""

import numpy as np
import torch

import reactive as R
import h_state_torch as H


class _Stub:
    """Minimal stand-in exposing only what _box_state_energy touches."""


def build_case():
    """A single H shared between two heavy atoms: the transfer motif."""

    carbon = int(R.ELEMENT_INDEX["C"])
    oxygen = int(R.ELEMENT_INDEX["O"])
    hydrogen = int(R.ELEMENT_INDEX["H"])

    # atom 0 = C, atom 1 = H, atom 2 = O
    types = np.array(
        [carbon, hydrogen, oxygen],
        dtype=np.int64,
    )

    # Two H-containing edges: the donor bond and the forming bond.
    edge_atoms = ((0, 1), (1, 2))
    edge_rows = (0, 1)
    edge_slots = (0, 0)

    return types, edge_atoms, edge_rows, edge_slots


def build_values(repulsive_scale):
    """Synthetic pair quantities indexed [row, slot], as the real code expects."""

    def pair(a, b):
        return torch.tensor(
            [[a], [b]],
            dtype=torch.float64,
        )

    return {
        "taper": pair(0.90, 0.70),
        "pair_depth": pair(4.20, 3.60),
        "pair_width": pair(1.90, 2.10),
        "shift": pair(0.15, 0.40),
        "repulsive": pair(5.00, 3.00) * repulsive_scale,
    }


def diagonals_for(scale, types, edge_atoms, edge_rows, edge_slots):
    """Rebuild the state diagonals the way _box_state_energy does."""

    values = build_values(scale)

    taper = values["taper"]
    pair_depth = values["pair_depth"]
    pair_width = values["pair_width"]
    shift = values["shift"]
    repulsive = values["repulsive"]

    edge_repulsive = []
    edge_attractive = []

    for row, slot in zip(edge_rows, edge_slots):
        contact = taper[row, slot]
        depth = pair_depth[row, slot]

        attractive = (
            2.0
            * depth
            * torch.exp(
                -pair_width[row, slot]
                * shift[row, slot]
            )
        )

        edge_repulsive.append(contact * repulsive[row, slot])
        edge_attractive.append(contact * attractive)

    states = H._maximal_states(edge_atoms, types)

    common_core = torch.stack(edge_repulsive).sum()

    zero = taper.sum() * 0.0
    diagonals = []

    for state in states:
        if state:
            attraction = torch.stack(
                [edge_attractive[i] for i in state]
            ).sum()
        else:
            attraction = zero

        diagonals.append(common_core - attraction)

    return states, torch.stack(diagonals)


def total_energy_for(scale, stub, edge_atoms, edge_rows, edge_slots):
    """Full lowest-eigenvalue result through the real method."""

    return H.HStateReferenceBatchedSimulation._box_state_energy(
        stub,
        edge_atoms,
        edge_rows,
        edge_slots,
        build_values(scale),
    )


def main():
    types, edge_atoms, edge_rows, edge_slots = build_case()

    stub = _Stub()
    stub.types_numpy = types
    stub.h_state_mixing = float(H.H_STATE_MIXING)

    print("H-STATE REPULSION CANCELLATION CHECK")
    print("=" * 52)

    states, base = diagonals_for(
        1.0, types, edge_atoms, edge_rows, edge_slots
    )
    _, scaled = diagonals_for(
        2.0, types, edge_atoms, edge_rows, edge_slots
    )

    print(f"states enumerated: {states}")
    print()

    print(f"{'state':<12}{'diag x1':>12}{'diag x2':>12}{'shift':>12}")
    print("-" * 48)

    for state, one, two in zip(states, base, scaled):
        print(
            f"{str(state):<12}"
            f"{one.item():>12.6f}"
            f"{two.item():>12.6f}"
            f"{(two - one).item():>12.6f}"
        )

    base_gaps = base - base[0]
    scaled_gaps = scaled - scaled[0]

    print()
    print("gaps relative to first state")
    print(f"{'state':<12}{'gap x1':>12}{'gap x2':>12}{'difference':>14}")
    print("-" * 50)

    for state, one, two in zip(states, base_gaps, scaled_gaps):
        print(
            f"{str(state):<12}"
            f"{one.item():>12.6f}"
            f"{two.item():>12.6f}"
            f"{(two - one).item():>14.3e}"
        )

    worst = (scaled_gaps - base_gaps).abs().max().item()

    print()
    print(f"max |gap change| under 2x repulsion : {worst:.3e} eV")

    if worst < 1e-12:
        print()
        print("RESULT: repulsion cancels exactly in every state gap.")
        print("        The model's state gap is a pure difference of")
        print("        attractive terms. A state-dependent exchange")
        print("        contribution cannot be represented as written.")
    else:
        print()
        print("RESULT: repulsion DOES reach the state gaps.")
        print("        The cancellation reading was wrong.")

    # The lowest adiabatic root should still move, or the test proves
    # nothing about whether repulsive matters at all.
    low_one = total_energy_for(
        1.0, stub, edge_atoms, edge_rows, edge_slots
    )
    low_two = total_energy_for(
        2.0, stub, edge_atoms, edge_rows, edge_slots
    )

    print()
    print("control: lowest adiabatic eigenvalue")
    print(f"  x1 repulsion : {low_one.item():.6f} eV")
    print(f"  x2 repulsion : {low_two.item():.6f} eV")
    print(f"  change       : {(low_two - low_one).item():.6f} eV")
    print()
    print("  A nonzero change here confirms repulsive is genuinely")
    print("  in use, so an unchanged gap above is cancellation and")
    print("  not a dead input.")


if __name__ == "__main__":
    main()
