"""
Does a per-state diagonal offset actually move the barrier?

The open-shell SAPT probe measured a state-dependent exchange difference
that the model cannot represent: `common_core` is computed outside the
per-state loop in _box_state_energy, so repulsion cancels exactly in
every state gap (see test_repulsion_cancellation.py).

Before rebuilding anything, this asks the cheap question: if the product
state's diagonal is raised by hand, does the barrier respond usefully, or
does the shift wash out at the crossing?

It monkeypatches _box_state_energy with a copy that adds `offset` to any
state occupying an H-H edge -- the product state for these X-H + H -> X + H2
reactions -- and reruns the scanner's own frozen-grid saddle search.

    py state_offset_probe.py --system formaldehyde --offset 2.634
    py state_offset_probe.py --system methane --offset -0.317
    py state_offset_probe.py --system formaldehyde --sweep

Measured barrier-geometry gaps from sapt_open_shell_state_probe.csv:
    formaldehyde  +2.634 eV
    methane       -0.317 eV

Read the result as a direction, not a calibration. This hardcodes one
number at every geometry, whereas the real quantity varies along the path.
"""

import _bootstrap  # noqa: F401 - direct-execution project path

import argparse

import numpy as np
import torch

import reactive as R
import h_state_torch as H
import hf_surface_scan as S


# Set by main(); read inside the patched method on every call.
STATE_OFFSET = 0.0
OFFSET_REPORTED = False


def _state_touches_hh(state, edge_atoms, types):
    """True if this state occupies an edge between two hydrogens.

    For X-H + H -> X + H2 the product state is the one holding the newly
    formed H-H bond, which is what the open-shell probe called `product`.
    """

    hydrogen = int(R.ELEMENT_INDEX["H"])

    for edge_index in state:
        first, second = edge_atoms[edge_index]

        if (
            int(types[first]) == hydrogen
            and int(types[second]) == hydrogen
        ):
            return True

    return False


def patched_box_state_energy(
    self,
    edge_atoms,
    edge_rows,
    edge_slots,
    values,
):
    """_box_state_energy with a per-state diagonal offset added.

    Copied from h_state_torch so the only difference is the offset applied
    inside the state loop. If that file changes, re-copy this.
    """

    taper = values["taper"]
    pair_depth = values["pair_depth"]
    pair_width = values["pair_width"]
    shift = values["shift"]
    repulsive = values["repulsive"]

    edge_tapers = []
    edge_depths = []
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

        edge_tapers.append(contact)
        edge_depths.append(depth)
        edge_repulsive.append(contact * repulsive[row, slot])
        edge_attractive.append(contact * attractive)

    zero = values["taper"].sum() * 0.0

    if not edge_atoms:
        return zero

    states = H._maximal_states(
        edge_atoms,
        self.types_numpy,
    )

    common_core = torch.stack(edge_repulsive).sum()

    diagonals = []

    global OFFSET_REPORTED
    offset_flags = []

    for state in states:
        if state:
            attraction = torch.stack([
                edge_attractive[index]
                for index in state
            ]).sum()
        else:
            attraction = zero

        diagonal_value = common_core - attraction

        # THE ONE CHANGE: a state-dependent term, inside the loop, so it
        # survives the difference between diagonals instead of cancelling.
        touches = _state_touches_hh(
            state,
            edge_atoms,
            self.types_numpy,
        )

        offset_flags.append(touches)

        if touches and STATE_OFFSET != 0.0:
            diagonal_value = diagonal_value + STATE_OFFSET

        diagonals.append(diagonal_value)

    if not OFFSET_REPORTED:
        OFFSET_REPORTED = True
        print(f"    states this box: {states}")
        print(f"    offset applied to: "
              f"{[s for s, f in zip(states, offset_flags) if f]}")

    diagonal = torch.stack(diagonals)

    if len(states) == 1:
        return diagonal[0]

    transitions = {}
    weighted_degree = [zero for _ in states]

    for first in range(len(states)):
        for second in range(first + 1, len(states)):
            transfer = H._single_h_transfer(
                states[first],
                states[second],
                edge_atoms,
                self.types_numpy,
            )

            if transfer is None:
                continue

            old_index, new_index, _ = transfer

            overlap = H._contact_overlap(
                edge_tapers[old_index],
                edge_tapers[new_index],
            )

            transitions[(first, second)] = (
                old_index,
                new_index,
                overlap,
            )

            weighted_degree[first] = (
                weighted_degree[first] + overlap * overlap
            )

            weighted_degree[second] = (
                weighted_degree[second] + overlap * overlap
            )

    normalisation = torch.stack([
        H._crowding_normalisation(value)
        for value in weighted_degree
    ])

    couplings = {}

    for (first, second), (
        old_index,
        new_index,
        overlap,
    ) in transitions.items():

        depth_scale = torch.sqrt(
            torch.clamp(
                edge_depths[old_index] * edge_depths[new_index],
                min=1e-12,
            )
        )

        denominator = torch.sqrt(
            torch.clamp(
                normalisation[first] * normalisation[second],
                min=1e-12,
            )
        )

        couplings[(first, second)] = (
            self.h_state_mixing
            * depth_scale
            * overlap
            / denominator
        )

    rows = []

    for first in range(len(states)):
        row = []

        for second in range(len(states)):
            if first == second:
                value = diagonal[first]
            else:
                key = (min(first, second), max(first, second))
                value = (
                    -couplings[key]
                    if key in couplings
                    else zero
                )

            row.append(value)

        rows.append(torch.stack(row))

    hamiltonian = torch.stack(rows)

    eigenvalues = torch.linalg.eigvalsh(hamiltonian)

    return eigenvalues[0]


def barrier_for(offset, system, progress=False):
    """Frozen-grid barrier with the given per-state offset applied."""

    global STATE_OFFSET, OFFSET_REPORTED

    STATE_OFFSET = float(offset)
    OFFSET_REPORTED = False

    probe = S.SYSTEM_PROBES[system]

    donor_low, donor_high, donor_step = probe["donor"]
    transfer_low, transfer_high, transfer_step = probe["transfer"]

    donor_lengths = np.arange(donor_low, donor_high, donor_step)
    transfer_lengths = np.arange(
        transfer_low, transfer_high, transfer_step
    )

    grid = S.batched_surface(
        "h_state",
        donor_lengths,
        transfer_lengths,
        progress=progress,
        progress_label=f"offset {offset:+.3f}",
    )

    reactant, product = S.basin_seeds(
        grid,
        donor_lengths,
        transfer_lengths,
    )

    if reactant is None or product is None:
        raise RuntimeError(
            "basin_seeds could not place both basins on this grid; "
            "the offset may have deformed the surface past the probe "
            "thresholds"
        )

    # flood_saddle returns ((row, column), energy), not a bare index.
    saddle_cell, saddle_energy = S.flood_saddle(
        grid, reactant, product
    )

    if saddle_cell is None:
        raise RuntimeError(
            "flood_saddle found no connecting path between the basins; "
            "the two wells never join anywhere on this grid"
        )

    reactant_energy = float(grid[reactant])

    return {
        "barrier": float(saddle_energy - reactant_energy),
        "saddle_donor": float(donor_lengths[saddle_cell[0]]),
        "saddle_transfer": float(transfer_lengths[saddle_cell[1]]),
        "reaction": float(grid[product] - reactant_energy),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--system",
        default="formaldehyde",
        choices=sorted(S.SYSTEMS),
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="eV added to the product-state diagonal",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="scan a range of offsets instead of one value",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
    )

    options = parser.parse_args()

    S.ACTIVE_SYSTEM = options.system
    S.check_systems_registered()

    # Patch the reference model in place.
    H.HStateReferenceBatchedSimulation._box_state_energy = (
        patched_box_state_energy
    )

    print("PER-STATE DIAGONAL OFFSET PROBE")
    print("=" * 58)
    print(f"system        : {options.system}")
    print(f"mixing        : {H.H_STATE_MIXING} (untouched)")
    print()

    print("baseline (offset 0.000)")
    base = barrier_for(0.0, options.system, options.progress)
    print(f"  barrier  {base['barrier']:+.4f} eV")
    print(f"  saddle   {base['saddle_donor']:.3f} / "
          f"{base['saddle_transfer']:.3f} A")
    print(f"  reaction {base['reaction']:+.4f} eV")
    print()

    if options.sweep:
        offsets = [-1.0, -0.5, -0.317, 0.0, 0.5, 1.0, 2.0, 2.634, 4.0]
    elif options.offset is not None:
        offsets = [options.offset]
    else:
        offsets = [2.634 if options.system == "formaldehyde" else -0.317]

    print(f"{'offset':>9}{'barrier':>11}{'change':>10}"
          f"{'saddle d/t':>16}{'reaction':>11}")
    print("-" * 58)

    for offset in offsets:
        if offset == 0.0:
            result = base
        else:
            try:
                result = barrier_for(
                    offset, options.system, options.progress
                )
            except RuntimeError as error:
                print(
                    f"{offset:>+9.3f}"
                    f"{'  no saddle':>11}"
                    f"   ({error.args[0].split(';')[0]})"
                )
                continue

        change = result["barrier"] - base["barrier"]

        print(
            f"{offset:>+9.3f}"
            f"{result['barrier']:>+11.4f}"
            f"{change:>+10.4f}"
            f"{result['saddle_donor']:>9.3f} /"
            f"{result['saddle_transfer']:>6.3f}"
            f"{result['reaction']:>+11.4f}"
        )

    print()
    print("Interpretation:")
    print("  change ~ 0        -> the offset washes out at the crossing;")
    print("                       a per-state term will not fix barriers")
    print("                       and the architecture change is not the")
    print("                       answer on its own.")
    print("  change comparable -> the diagonals do drive the barrier and")
    print("    to the offset      a per-state term is worth building.")
    print()
    print("Reference barriers: formaldehyde 0.324, methane 0.640,")
    print("water 0.364-0.525 eV.")


if __name__ == "__main__":
    main()