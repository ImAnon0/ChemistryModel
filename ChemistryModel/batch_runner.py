import argparse
import json
import os
import time

import numpy as np


# ============================================================
# Running many boxes without a viewer
# ============================================================
#
# One result is an anecdote. Ten runs from different seeds tell
# you whether something is a real pathway in this chemistry or a
# single lucky collision.
#
# No Qt, no OpenGL, no drawing. Everything goes to disk and gets
# read back in the browser afterwards.
#
#   py batch_runner.py --mixture "H rich loose" --seeds 10 --ps 20
#
# Each run writes runs/run_003.npz plus an entry in runs/index.json.


DEFAULT_SCHEDULE = [
    # (until this many femtoseconds, target temperature in kelvin)
    #
    # Loose atoms bond immediately and heat themselves, so the
    # opening stretch is left warm to let fragments move, then
    # cooled to trap whatever formed. This mirrors what Miller's
    # condenser did: products drop out of the hot zone and stop
    # being torn apart again.
    (2000.0, 500.0),
    (4000.0, 350.0),
    (1e12, 250.0),
]


def build_simulation(mixture, box_size, seed, temperature,
                     time_step, friction, device):
    import build_box

    from reactive_torch import ReactiveSimulation

    kind, contents = mixture

    if kind == "atoms":
        symbols, positions = build_box.loose_atoms(
            contents, box_size,
            minimum_separation=1.25,
            random_seed=seed
        )
    else:
        symbols, positions = build_box.build(
            contents, box_size, random_seed=seed
        )

    return ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=box_size,
        time_step=time_step,
        target_temperature=temperature,
        friction=friction,
        device=device,
        random_seed=seed,
    )


def temperature_at(schedule, femtoseconds):
    for limit, value in schedule:
        if femtoseconds < limit:
            return value

    return schedule[-1][1]


def strike(simulation, radius, energy, generator):
    # Same cylindrical discharge channel as the viewer.

    import torch

    positions = simulation.positions_numpy

    origin = generator.uniform(0.0, simulation.box_size, size=3)

    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)

    offsets = positions - origin
    offsets -= simulation.box_size * np.round(
        offsets / simulation.box_size
    )

    along = offsets @ axis
    perpendicular = offsets - along[:, None] * axis

    distance = np.linalg.norm(perpendicular, axis=1)

    inside = distance < radius

    if not np.any(inside):
        return 0

    weight = np.zeros(len(positions))

    weight[inside] = 0.5 * (
        1.0 + np.cos(np.pi * distance[inside] / radius)
    )

    share = energy * weight / weight.sum()

    masses = simulation.masses.detach().cpu().numpy()

    speeds = np.sqrt(
        np.maximum(2.0 * share / (masses * 103.642), 0.0)
    )

    directions = generator.normal(size=positions.shape)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )

    kick = torch.tensor(
        directions * speeds[:, None],
        device=simulation.device,
        dtype=simulation.dtype
    )

    simulation.velocities += kick

    momentum = torch.sum(
        simulation.masses[:, None] * simulation.velocities, dim=0
    )

    simulation.velocities -= (
        momentum / torch.sum(simulation.masses)
    )

    return int(np.count_nonzero(inside))


def run_one(mixture, seed, options):
    from recorder import Recorder

    simulation = build_simulation(
        mixture,
        options.box,
        seed,
        DEFAULT_SCHEDULE[0][1],
        options.time_step,
        options.friction,
        options.device,
    )

    recorder = Recorder(
        simulation.symbols,
        simulation.box_size,
        maximum_frames=options.max_frames
    )

    generator = np.random.default_rng(seed + 9001)

    total_steps = int(
        options.picoseconds * 1000.0 / options.time_step
    )

    chunk = options.capture_every

    next_strike = (
        options.first_strike_fs if options.strikes > 0 else None
    )

    strikes_done = 0

    started = time.time()

    steps_done = 0

    while steps_done < total_steps:
        simulation.target_temperature = temperature_at(
            DEFAULT_SCHEDULE, simulation.elapsed_femtoseconds
        )

        simulation.step(min(chunk, total_steps - steps_done))

        steps_done += chunk

        recorder.capture(
            simulation.positions_numpy,
            simulation.elapsed_femtoseconds,
            simulation.potential_energy,
            simulation.kinetic_energy,
            simulation.temperature,
        )

        if (
            next_strike is not None
            and simulation.elapsed_femtoseconds >= next_strike
            and strikes_done < options.strikes
        ):
            strike(
                simulation,
                options.strike_radius,
                options.strike_energy,
                generator
            )

            strikes_done += 1
            next_strike += options.strike_interval_fs

        if not np.isfinite(simulation.potential_energy):
            print(f"  seed {seed}: went unstable, stopping early")
            break

    return recorder, simulation, time.time() - started, strikes_done


def main():
    parser = argparse.ArgumentParser(
        description="Run many reactive boxes headlessly."
    )

    parser.add_argument("--mixture", default="H rich loose")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--ps", dest="picoseconds", type=float,
                        default=20.0)
    parser.add_argument("--box", type=float, default=12.0)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--capture-every", type=int, default=40,
                        help="simulation steps between recorded frames")
    parser.add_argument("--max-frames", type=int, default=40000)
    parser.add_argument("--strikes", type=int, default=0)
    parser.add_argument("--first-strike-fs", type=float, default=3000.0)
    parser.add_argument("--strike-interval-fs", type=float,
                        default=3000.0)
    parser.add_argument("--strike-radius", type=float, default=2.2)
    parser.add_argument("--strike-energy", type=float, default=45.0)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=2,
                        help="frame stride used when analysing")

    options = parser.parse_args()

    from mixtures import STARTS

    if options.mixture not in STARTS:
        raise SystemExit(
            f"unknown mixture {options.mixture!r}. "
            f"options: {sorted(STARTS)}"
        )

    mixture = STARTS[options.mixture]

    os.makedirs(options.out, exist_ok=True)

    index_path = os.path.join(options.out, "index.json")

    if os.path.exists(index_path):
        with open(index_path) as handle:
            index = json.load(handle)
    else:
        index = []

    import analysis

    print(
        f"{options.seeds} runs of {options.picoseconds:.0f} ps, "
        f"mixture {options.mixture!r}, box {options.box} A"
    )
    print()

    for offset in range(options.seeds):
        seed = options.first_seed + offset

        recorder, simulation, seconds, strikes = run_one(
            mixture, seed, options
        )

        number = len(index)

        name = f"run_{number:03d}.npz"
        path = os.path.join(options.out, name)

        recorder.save(path)

        result = analysis.analyse(recorder, stride=options.stride)

        entry = {
            "number": number,
            "file": name,
            "mixture": options.mixture,
            "seed": seed,
            "box": options.box,
            "picoseconds": options.picoseconds,
            "atoms": len(simulation.symbols),
            "strikes": strikes,
            "wall_seconds": round(seconds, 1),
            "frames": len(recorder),
            "headline": analysis.headline(result),
            "final_species": sorted(
                {
                    entry["formula"]
                    for entry in result["final"]
                    if entry["heavy"] >= 2
                }
            ),
            "closed_shell": sorted(
                {
                    entry["formula"]
                    for entry in result["final"]
                    if entry["heavy"] >= 2 and entry["closed_shell"]
                }
            ),
            "species_seen": sorted(result["seen"]),
            "heavy_bonds_formed": sum(
                1 for event in result["heavy_events"]
                if event[1] == "formed"
            ),
            "final_temperature": result["temperature"]["final"],
            "final_potential": result["potential"]["final"],
        }

        index.append(entry)

        with open(index_path, "w") as handle:
            json.dump(index, handle, indent=1)

        print(
            f"run {number:03d}  seed {seed:3d}  "
            f"{seconds:6.1f} s  {len(recorder):5d} frames  "
            f"-> {entry['headline']}"
        )

    print()
    print(f"index written to {index_path}")

    # A quick tally across everything in the index.

    tally = {}

    for entry in index:
        for name in entry["closed_shell"]:
            tally[name] = tally.get(name, 0) + 1

    if tally:
        print()
        print(f"closed-shell products across {len(index)} runs:")

        for name, number in sorted(
            tally.items(), key=lambda item: -item[1]
        ):
            print(
                f"  {name:<12} in {number}/{len(index)} runs"
            )


if __name__ == "__main__":
    main()
