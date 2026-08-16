import argparse
import time

import numpy as np

import torch

import build_box
import mixtures

from reactive_torch import ReactiveSimulation


# ============================================================
# Where the time goes
# ============================================================
#
#   py profile_reactive.py
#   py profile_reactive.py --atoms 410 --box 20.5
#
# A GPU sitting at a third of its capacity is waiting for
# something, and guessing which thing wastes more time than
# measuring it. This times each part of a step separately, with
# the synchronisation that timing a GPU honestly requires, and
# reports what fraction of a run each accounts for.
#
# It also tries the same work at several sizes, since the useful
# question is not how long one box takes but whether a bigger one
# takes proportionally longer. If it does not, the card is idle
# and several boxes could share it.


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Stopwatch:

    def __init__(self, device):
        self.device = device
        self.totals = {}
        self.counts = {}

    def time(self, name, function, repeats=1):
        synchronise(self.device)

        start = time.perf_counter()

        for _ in range(repeats):
            result = function()

        synchronise(self.device)

        elapsed = (time.perf_counter() - start) / repeats

        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

        return result, elapsed

    def report(self, total_name=None):
        lines = []

        reference = self.totals.get(total_name)

        for name in self.totals:
            average = self.totals[name] / self.counts[name]

            share = (
                f"{100 * average / reference:5.1f}%"
                if reference else ""
            )

            lines.append((name, average, share))

        return lines


def make(mixture_name, box, seed=0, device=None):
    kind, contents = mixtures.all_mixtures()[mixture_name]

    if kind == "molecules":
        symbols, positions = build_box.build(contents, box)
    else:
        symbols, positions = build_box.loose_atoms(
            contents, box, minimum_separation=1.25, random_seed=seed
        )

    return ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=box,
        random_seed=seed,
        device=device,
    )


def profile_one(simulation, steps=200):
    device = simulation.positions.device

    watch = Stopwatch(device)

    # A plain step, everything included.

    _, per_step = watch.time(
        "one step, everything", lambda: simulation.step(1), steps
    )

    # The force calculation on its own: the part that should
    # dominate if the card is being used properly.

    _, forces = watch.time(
        "  force calculation",
        lambda: simulation.compute_forces(),
        steps,
    )

    # Rebuilding the neighbour table. Done on the processor with
    # a k-d tree, so it costs a transfer each way as well as the
    # search itself.

    _, neighbours = watch.time(
        "  neighbour rebuild",
        lambda: simulation.build_neighbours(),
        20,
    )

    # Checking whether a rebuild is needed, which happens every
    # single step.

    _, check = watch.time(
        "  rebuild check",
        lambda: simulation.needs_rebuild(),
        steps,
    )

    # Pulling positions back for a capture.

    _, fetch = watch.time(
        "  fetch positions",
        lambda: simulation.positions_numpy,
        100,
    )

    _, energy = watch.time(
        "  read energies",
        lambda: (
            simulation.potential_energy,
            simulation.kinetic_energy,
            simulation.temperature,
        ),
        100,
    )

    return {
        "step": per_step,
        "forces": forces,
        "neighbours": neighbours,
        "check": check,
        "fetch": fetch,
        "energy": energy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find out what a step actually spends time on."
    )

    parser.add_argument("--mixture", default="carbon rich")
    parser.add_argument("--box", type=float, default=19.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--capture-every", type=int, default=40)
    parser.add_argument("--device", default=None)

    options = parser.parse_args()

    device = torch.device(
        options.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"device: {device}")

    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print()

    simulation = make(
        options.mixture, options.box, device=device
    )

    count = simulation.atom_count

    print(
        f"{options.mixture}: {count} atoms in a "
        f"{options.box:g} A box"
    )
    print()

    timings = profile_one(simulation, options.steps)

    print("  what one step costs")
    print("  " + "-" * 52)

    step = timings["step"]

    for name, key in [
        ("one step, everything", "step"),
        ("  force calculation", "forces"),
        ("  rebuild check", "check"),
    ]:
        value = timings[key]

        share = 100 * value / step if step else 0

        print(
            f"  {name:<26}{value * 1e6:9.1f} us"
            f"{share:8.1f}% of a step"
        )

    print()
    print("  and the things that happen occasionally")
    print("  " + "-" * 52)

    # How much a rebuild and a capture add, spread over the steps
    # between them.

    rebuilds = simulation.rebuild_count

    for name, key, every in [
        ("neighbour rebuild", "neighbours", 20),
        ("fetch positions", "fetch", options.capture_every),
        ("read energies", "energy", options.capture_every),
    ]:
        value = timings[key]

        spread = value / every

        share = 100 * spread / step if step else 0

        print(
            f"  {name:<26}{value * 1e6:9.1f} us"
            f"  once every {every:>3} steps"
            f"  ->{share:6.1f}% of a step"
        )

    print()

    # What a whole run would cost at this rate.

    for picoseconds in [20]:
        steps = picoseconds * 1000 / simulation.time_step

        seconds = steps * step

        print(
            f"  a {picoseconds:g} ps run at this rate: "
            f"{seconds / 60:.1f} minutes"
        )

    print()
    print()
    print("  does a bigger box cost proportionally more?")
    print("  " + "-" * 52)
    print(
        f"  {'atoms':>7}{'per step':>12}{'per atom':>12}"
        f"{'vs smallest':>14}"
    )

    smallest = None

    kind, contents = mixtures.all_mixtures()[options.mixture]

    for scale in [1, 2, 4, 8]:
        # Both the contents and the box grow, so the density
        # stays put and only the size changes. Scaling the box
        # alone would be measuring dilution instead.

        box = options.box * (scale ** (1 / 3))

        scaled = {
            name: number * scale
            for name, number in contents.items()
        }

        try:
            if kind == "molecules":
                symbols, positions = build_box.build(scaled, box)
            else:
                symbols, positions = build_box.loose_atoms(
                    scaled, box, minimum_separation=1.25,
                    random_seed=0,
                )

            bigger = ReactiveSimulation(
                symbols=symbols,
                positions=positions,
                box_size=box,
                random_seed=0,
                device=device,
            )
        except Exception as problem:
            print(f"  {scale}x failed: {problem}")
            continue

        _, per = Stopwatch(device).time(
            "step", lambda: bigger.step(1), 100
        )

        atoms = bigger.atom_count

        if smallest is None:
            smallest = per

        print(
            f"  {atoms:>7}{per * 1e6:>10.1f} us"
            f"{per / atoms * 1e6:>11.3f} us"
            f"{per / smallest:>12.2f}x"
        )

        del bigger

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print(
        "  If the cost per atom falls as the box grows, the card\n"
        "  is idle at the smaller size and several boxes could\n"
        "  share it. If it stays flat, the work is genuinely\n"
        "  compute bound and batching would not help."
    )


if __name__ == "__main__":
    main()
