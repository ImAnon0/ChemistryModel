import argparse
import time

import numpy as np

import torch

import build_box
import mixtures

from batched_torch import (
    BatchedReactiveSimulation,
    compare_against_single,
)

from reactive_torch import ReactiveSimulation


# ============================================================
# Is running boxes together the same as running them apart?
# ============================================================
#
#   py validate_batched.py
#   py validate_batched.py --boxes 8 --steps 200
#
# Speed is worth nothing if the answers change. This checks that
# a group of boxes computed in one tensor gives the same forces
# and the same energies as the same boxes computed one at a time,
# then that stepping them together keeps them identical, and
# finally how much faster it actually is.


def make_boxes(mixture_name, box, count, first_seed=0):
    # Each box keeps its own symbols: the builder shuffles atoms
    # as it places them, so two boxes of the same composition come
    # back in a different order.

    kind, contents = mixtures.all_mixtures()[mixture_name]

    boxes = []

    for index in range(count):
        seed = first_seed + index

        if kind == "molecules":
            symbols, positions = build_box.build(
                contents, box, random_seed=seed
            )
        else:
            symbols, positions = build_box.loose_atoms(
                contents, box,
                minimum_separation=1.25,
                random_seed=seed,
            )

        boxes.append((symbols, positions))

    return boxes


def check_forces(boxes, box):
    print("  forces and energies, grouped against alone")
    print("  " + "-" * 58)

    report = compare_against_single(boxes, box)

    worst_force = 0.0
    worst_energy = 0.0

    for entry in report:
        worst_force = max(worst_force, entry["force_error"])
        worst_energy = max(worst_energy, entry["energy_error"])

        print(
            f"    box {entry['box']}: force {entry['force_error']:.2e}"
            f"   energy {entry['energy_error']:.2e}"
            f"   ({entry['energy_alone']:.4f} vs "
            f"{entry['energy_grouped']:.4f} eV)"
        )

    print()

    good = worst_force < 1e-5 and worst_energy < 1e-5

    print(
        f"    worst force error {worst_force:.2e}, "
        f"worst energy error {worst_energy:.2e}"
    )
    print(
        "    -> " + (
            "the same to within floating point noise"
            if good else
            "DIFFERENT. Do not use this until it is understood."
        )
    )

    return good


def check_stepping(boxes, box, steps):
    print()
    print(f"  after {steps} steps, still the same?")
    print("  " + "-" * 58)

    # Two things have to be equalised before this comparison
    # means anything.
    #
    # The thermostat draws random numbers, so it is turned off.
    # And the starting velocities are drawn from one seeded
    # generator across however many atoms there are, so a box on
    # its own and the same box in a group of four do not get the
    # same draw: the first box matches and the rest start moving
    # differently. Setting every velocity to zero removes both
    # and leaves only the mechanics.

    separate = []

    for symbols, positions in boxes:
        one = ReactiveSimulation(
            symbols=symbols,
            positions=positions,
            box_size=box,
            random_seed=0,
            relax_on_start=False,
        )

        one.thermostat_is_on = False
        one.velocities = torch.zeros_like(one.velocities)

        one.step(steps)

        separate.append(one.positions_numpy.copy())

    together = BatchedReactiveSimulation(
        boxes=boxes,
        box_size=box,
        random_seed=0,
        relax_on_start=False,
    )

    together.thermostat_is_on = False
    together.velocities = torch.zeros_like(together.velocities)

    together.step(steps)

    worst = 0.0

    for index, alone in enumerate(separate):
        mine = together.positions_for(index)

        # Positions are kept wrapped into the cell, so an atom
        # sitting either side of a face looks a whole box away
        # when it has barely moved.

        offset = mine - alone
        offset -= box * np.round(offset / box)

        drift = np.abs(offset).max()

        worst = max(worst, drift)

        print(f"    box {index}: largest drift {drift:.3e} A")

    print()

    # Some drift is unavoidable and is not a mistake. Adding the
    # same numbers in a different order gives a slightly
    # different answer, a group sums over more atoms at once, and
    # molecular dynamics amplifies any difference exponentially.
    # What matters is that it starts at the size of floating
    # point noise rather than at the size of the box.

    good = worst < 0.05

    print(
        f"    worst drift {worst:.3e} A -> "
        + (
            "the same trajectory, to rounding"
            if good else
            "TRAJECTORIES DIVERGE. Something is wrong."
        )
    )

    return good


def check_speed(mixture, box, most, steps):
    print()
    print("  and how much faster is it?")
    print("  " + "-" * 58)
    print(
        f"    {'boxes':>7}{'per step':>12}{'per box-step':>15}"
        f"{'vs one':>9}"
    )

    single = None

    for count in [1, 2, 4, most]:
        if count > most and count != 1:
            continue

        boxes = make_boxes(mixture, box, count)

        simulation = BatchedReactiveSimulation(
            boxes=boxes,
            box_size=box,
            random_seed=0,
        )

        simulation.step(10)

        if simulation.positions.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        simulation.step(steps)

        if simulation.positions.device.type == "cuda":
            torch.cuda.synchronize()

        per_step = (time.perf_counter() - start) / steps

        per_box = per_step / count

        if single is None:
            single = per_box

        print(
            f"    {count:>7}{per_step * 1e3:>10.2f} ms"
            f"{per_box * 1e3:>13.3f} ms"
            f"{single / per_box:>8.2f}x"
        )

        del simulation

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Check grouped boxes against single ones."
    )

    parser.add_argument("--mixture", default="carbon rich")
    parser.add_argument("--box", type=float, default=19.0)
    parser.add_argument("--boxes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)

    options = parser.parse_args()

    print(
        f"{options.mixture} in a {options.box:g} A box, "
        f"{options.boxes} of them"
    )
    print()

    boxes = make_boxes(
        options.mixture, options.box, min(options.boxes, 4)
    )

    print(f"  {len(boxes[0][0])} atoms per box")
    print()

    forces_fine = check_forces(boxes, options.box)

    stepping_fine = check_stepping(
        boxes, options.box, min(options.steps, 60)
    )

    check_speed(
        options.mixture, options.box, options.boxes, options.steps
    )

    print()

    if forces_fine and stepping_fine:
        print("  Everything matches. Safe to use.")
    else:
        print(
            "  Something does not match. The speed is irrelevant "
            "until it does."
        )


if __name__ == "__main__":
    main()