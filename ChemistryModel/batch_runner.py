import argparse
import json
import os
import time

import sys
import time

import numpy as np

import discharge
import running


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


class Progress:
    # A single line that rewrites itself, showing where the batch
    # is and when it will finish.
    #
    # Two bars: the run currently stepping, and the batch as a
    # whole. The estimate uses the mean of completed runs rather
    # than the last one, since the first is usually slower while
    # the neighbour lists settle.

    def __init__(self, total_runs, steps_per_run, label=""):
        self.total_runs = total_runs
        self.steps_per_run = steps_per_run
        self.label = label

        self.completed = 0
        self.durations = []
        self.started = time.time()
        self.run_started = time.time()

    @staticmethod
    def bar(fraction, width=22):
        filled = int(round(fraction * width))

        return "#" * filled + "-" * (width - filled)

    @staticmethod
    def clock(seconds):
        seconds = int(max(seconds, 0))

        if seconds < 60:
            return f"{seconds}s"

        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"

        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

    def start_run(self):
        self.run_started = time.time()

    def finish_run(self):
        self.durations.append(time.time() - self.run_started)
        self.completed += 1

    def show(self, steps_done):
        run_fraction = min(
            steps_done / max(self.steps_per_run, 1), 1.0
        )

        batch_fraction = (
            self.completed + run_fraction
        ) / max(self.total_runs, 1)

        if self.durations:
            typical = sum(self.durations) / len(self.durations)
        else:
            # Before any run has finished, guess from how far the
            # current one has got.

            elapsed = time.time() - self.run_started

            typical = (
                elapsed / run_fraction if run_fraction > 0.02
                else 0.0
            )

        remaining = 0.0

        if typical:
            remaining = typical * (
                self.total_runs - self.completed - run_fraction
            )

        line = (
            f"\r  run {self.completed + 1}/{self.total_runs} "
            f"[{self.bar(run_fraction)}] {run_fraction:4.0%}   "
            f"batch [{self.bar(batch_fraction)}] "
            f"{batch_fraction:4.0%}   "
            f"elapsed {self.clock(time.time() - self.started)}"
        )

        if remaining:
            line += f"   left {self.clock(remaining)}"

        sys.stdout.write(line + "   ")
        sys.stdout.flush()

    def clear(self):
        sys.stdout.write("\r" + " " * 110 + "\r")
        sys.stdout.flush()


def resize_toward(simulation, target, rate=0.002):
    # Scale the cell and everything in it a little at a time.
    #
    # Compressing raises the density and forces condensation;
    # expanding afterwards spreads the products out so they can be
    # told apart. At very high density every atom sits inside its
    # neighbours' bond range, so a distance-based bond test stops
    # distinguishing contact from chemistry - which is why the
    # measurement is taken after the box has been opened up again.

    difference = target - simulation.box_size

    if abs(difference) < 0.01:
        return False

    step = float(
        np.clip(
            difference,
            -rate * simulation.box_size,
            rate * simulation.box_size,
        )
    )

    new_size = simulation.box_size + step

    factor = new_size / simulation.box_size

    simulation.positions = simulation.positions * factor
    simulation.box_size = float(new_size)

    simulation.reference_positions = None
    simulation.build_neighbours()

    simulation.forces, simulation._potential_energy = (
        simulation.compute_forces()
    )

    return True


def run_one(mixture, seed, options, progress=None):
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

        this_chunk = min(chunk, total_steps - steps_done)

        simulation.step(this_chunk)

        steps_done += this_chunk

        if progress is not None and steps_done % (chunk * 20) == 0:
            progress.show(steps_done)

        recorder.capture(
            simulation.positions_numpy,
            simulation.elapsed_femtoseconds,
            simulation.potential_energy,
            simulation.kinetic_energy,
            simulation.temperature,
            velocities=(
                simulation.velocities.detach().cpu().numpy()
            ),
            box_size=simulation.box_size,
        )

        if (
            options.expand_to
            and simulation.elapsed_femtoseconds
            >= options.expand_at_fs
        ):
            resize_toward(
                simulation, options.expand_to, options.expand_rate
            )

        if (
            next_strike is not None
            and simulation.elapsed_femtoseconds >= next_strike
            and strikes_done < options.strikes
        ):
            report = discharge.apply_to(
                simulation,
                generator,
                radius=options.strike_radius,
                temperature=options.strike_temperature,
                dissociation=options.strike_dissociation,
            )

            if options.verbose:
                print(
                    f"    strike at "
                    f"{simulation.elapsed_femtoseconds:.0f} fs: "
                    f"{report['struck']} atoms, "
                    f"{report['dissociated']} bonds broken"
                )

            strikes_done += 1
            next_strike += options.strike_interval_fs

        if not np.isfinite(simulation.potential_energy):
            print(f"  seed {seed}: went unstable, stopping early")
            break

    return recorder, simulation, time.time() - started, strikes_done


def continuation_state(source_path, target_path):
    # Decides what to do with one run before spending time on it.
    #
    # Returns one of:
    #   "skip"    already extended, nothing to do
    #   "unstable" the source blew up, so extending it would only
    #              produce more of a contaminated trajectory
    #   "redo"    an output exists but is unusable, and the source
    #             is intact, so it can simply be done again
    #   "run"     not done yet

    from recorder import Recorder

    import analysis

    try:
        source = Recorder.load(source_path)
    except Exception:
        return "unreadable", None, 0

    result = analysis.analyse(
        source, stride=8, structures=False
    )

    if not result.get("stable", True):
        return "unstable", source, len(source)

    if not os.path.exists(target_path):
        return "run", source, len(source)

    # An output that will not load, or that holds no more frames
    # than the source, was never finished. The source is still
    # intact, so the cheapest answer is to do it again rather than
    # try to salvage a truncated archive: the arrays inside are
    # stored separately, so a partial write leaves positions and
    # velocities of different lengths with no way to tell which
    # frames are aligned.

    try:
        existing = Recorder.load(target_path)
    except Exception:
        os.remove(target_path)

        return "redo", source, len(source)

    if len(existing) <= len(source):
        os.remove(target_path)

        return "redo", source, len(source)

    return "skip", source, len(source)


def continue_one(path, options, progress=None):
    # Picks an existing run up where it stopped and carries on.
    #
    # The recording holds positions and velocities for every
    # frame, so the final one is a complete description of the
    # system: resuming from it produces exactly the trajectory
    # that would have followed had the run never been cut short.
    # The startup relaxation is skipped, since the atoms are
    # already settled and relaxing them would throw the state
    # away.
    #
    # New frames are appended to the same recording, so the
    # result is one continuous run rather than two stitched
    # together.

    import torch

    from recorder import Recorder

    from reactive_torch import ReactiveSimulation

    recorder = Recorder.load(path)

    # Loading sets the frame limit to whatever the recording
    # already holds, so the first appended frame would trigger
    # thinning and halve the resolution of the run being
    # continued. Room is made for the new frames plus a margin.

    expected = int(
        options.picoseconds * 1000.0
        / (options.capture_every * options.time_step)
    )

    recorder.maximum_frames = max(
        recorder.maximum_frames,
        len(recorder) + expected + 100,
    )

    last = len(recorder) - 1

    if not recorder.has_velocities:
        raise SystemExit(
            f"{path} has no velocities stored, so it cannot be "
            f"resumed exactly. Only runs recorded after velocity "
            f"capture was added can be continued."
        )

    simulation = ReactiveSimulation(
        symbols=recorder.symbols,
        positions=recorder.positions[last].astype(float),
        box_size=recorder.box_at(last),
        time_step=options.time_step,
        target_temperature=options.cool_temperature,
        friction=options.friction,
        device=options.device,
        relax_on_start=False,
    )

    simulation.velocities = torch.tensor(
        recorder.velocities[last].astype(float),
        device=simulation.device,
        dtype=simulation.dtype,
    )

    simulation.elapsed_femtoseconds = float(recorder.times[last])

    simulation.forces, simulation._potential_energy = (
        simulation.compute_forces()
    )

    generator = np.random.default_rng(
        int(recorder.times[last]) + 4711
    )

    added_steps = int(
        options.picoseconds * 1000.0 / options.time_step
    )

    chunk = options.capture_every

    next_strike = (
        simulation.elapsed_femtoseconds + options.first_strike_fs
        if options.strikes > 0 else None
    )

    strikes_done = 0

    started = time.time()
    steps_done = 0

    while steps_done < added_steps:
        this_chunk = min(chunk, added_steps - steps_done)

        simulation.step(this_chunk)

        steps_done += this_chunk

        if progress is not None and steps_done % (chunk * 20) == 0:
            progress.show(steps_done)

        recorder.capture(
            simulation.positions_numpy,
            simulation.elapsed_femtoseconds,
            simulation.potential_energy,
            simulation.kinetic_energy,
            simulation.temperature,
            velocities=(
                simulation.velocities.detach().cpu().numpy()
            ),
            box_size=simulation.box_size,
        )

        if (
            next_strike is not None
            and simulation.elapsed_femtoseconds >= next_strike
            and strikes_done < options.strikes
        ):
            discharge.apply_to(
                simulation,
                generator,
                radius=options.strike_radius,
                temperature=options.strike_temperature,
                dissociation=options.strike_dissociation,
            )

            strikes_done += 1
            next_strike += options.strike_interval_fs

        if not np.isfinite(simulation.potential_energy):
            print("  went unstable, stopping early")
            break

    return recorder, simulation, time.time() - started, strikes_done


ENTRIES = "entries"


def entry_path(folder, seed):
    return os.path.join(folder, ENTRIES, f"seed_{int(seed):06d}.json")


def write_entry(folder, entry):
    # Each run writes its own small file rather than everyone
    # rewriting one shared index.
    #
    # With several batches filling the same folder at once, a
    # shared index cannot survive: each process loads the whole
    # list at start and writes the whole list back after every
    # run, so whichever finishes last erases the others. Separate
    # files never collide, and the index below is rebuilt from
    # them, so two processes rebuilding at the same moment
    # produce the same answer.

    directory = os.path.join(folder, ENTRIES)

    os.makedirs(directory, exist_ok=True)

    path = entry_path(folder, entry.get("seed", 0))

    temporary = path + ".part"

    with open(temporary, "w") as handle:
        json.dump(entry, handle, indent=1)

    os.replace(temporary, path)


def rebuild_index(folder):
    directory = os.path.join(folder, ENTRIES)

    entries = []

    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue

            try:
                with open(os.path.join(directory, name)) as handle:
                    entries.append(json.load(handle))
            except (json.JSONDecodeError, OSError):
                continue

    entries.sort(key=lambda item: item.get("seed", 0))

    for number, entry in enumerate(entries):
        entry["number"] = number

    path = os.path.join(folder, "index.json")

    temporary = path + ".part"

    with open(temporary, "w") as handle:
        json.dump(entries, handle, indent=1)

    os.replace(temporary, path)

    return entries


def existing_seeds(folder):
    # Seeds already present, read from the entry files so it is
    # correct even while other processes are writing.

    directory = os.path.join(folder, ENTRIES)

    seeds = set()

    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.startswith("seed_") and name.endswith(".json"):
                try:
                    seeds.add(int(name[5:-5]))
                except ValueError:
                    continue

    for entry in read_existing_index(folder):
        if entry.get("seed") is not None:
            seeds.add(int(entry["seed"]))

    return seeds


def read_existing_index(folder):
    path = os.path.join(folder, "index.json")

    if not os.path.exists(path):
        return []

    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []


def migrate_index(folder):
    # Folders made before entry files existed keep their index.
    # Splitting it into entries once means everything afterwards
    # can be parallel-safe without losing what is already there.

    directory = os.path.join(folder, ENTRIES)

    if os.path.isdir(directory) and os.listdir(directory):
        return

    index = read_existing_index(folder)

    if not index:
        return

    for entry in index:
        if entry.get("seed") is None:
            continue

        write_entry(folder, entry)


def condition_key(options):
    # Everything that changes the chemistry. Two batches sharing
    # these belong in the same folder and can be pooled; anything
    # else has to be kept apart, since averaging across different
    # conditions produces a number that describes neither.

    return {
        "mixture": options.mixture,
        "box": round(float(options.box), 2),
        "picoseconds": round(float(options.picoseconds), 3),
        "strikes": int(options.strikes),
        "strike_temperature": (
            round(float(options.strike_temperature), 0)
            if options.strikes else 0.0
        ),
        "strike_dissociation": (
            round(float(options.strike_dissociation), 3)
            if options.strikes else 0.0
        ),
        "expand_to": round(float(options.expand_to), 2),
        "hot_temperature": round(float(options.hot_temperature), 0),
        "cool_temperature": round(float(options.cool_temperature), 0),
    }


def folder_name(options):
    # A readable name that encodes the conditions, so a folder
    # says what it holds without opening it.

    safe = options.mixture.replace(" ", "_").replace("+", "plus")

    parts = [
        safe,
        f"box{options.box:g}",
        f"{options.picoseconds:g}ps",
    ]

    if options.expand_to:
        parts.append(f"to{options.expand_to:g}")

    if options.strikes:
        parts.append(
            f"{options.strikes}strikes"
            f"{options.strike_temperature / 1000:g}k"
        )
    else:
        parts.append("quiet")

    if options.cool_temperature != 250.0:
        parts.append(f"cool{options.cool_temperature:g}")

    return "_".join(parts)


def existing_batch(directory):
    # Returns the index already in a folder, plus the conditions
    # its runs were made under, so a new batch can check it
    # belongs there before adding to it.

    path = os.path.join(directory, "index.json")

    if not os.path.exists(path):
        return [], None

    with open(path) as handle:
        index = json.load(handle)

    if not index:
        return [], None

    first = index[0]

    strikes = int(first.get("strikes", 0))

    # A quiet run stores whatever the strike defaults happened to
    # be, even though none were fired. Comparing those against a
    # request that zeroes them makes two identical quiet batches
    # look like different conditions, so both sides are zeroed
    # when there are no strikes.

    found = {
        "mixture": first.get("mixture"),
        "box": round(float(first.get("box", 0)), 2),
        "picoseconds": round(float(first.get("picoseconds", 0)), 3),
        "strikes": strikes,
        "strike_temperature": (
            round(float(first.get("strike_temperature", 0) or 0), 0)
            if strikes else 0.0
        ),
        "strike_dissociation": (
            round(float(first.get("strike_dissociation", 0) or 0), 3)
            if strikes else 0.0
        ),
        "expand_to": round(float(first.get("expand_to", 0) or 0), 2),
        "hot_temperature": round(
            float(first.get("hot_temperature", 500) or 500), 0
        ),
        "cool_temperature": round(
            float(first.get("cool_temperature", 250) or 250), 0
        ),
    }

    return index, found


def run_continuation(options):
    import glob
    import shutil

    from recorder import Recorder

    import analysis

    source = options.continue_from

    files = sorted(glob.glob(os.path.join(source, "run_*.npz")))

    if not files:
        raise SystemExit(f"no recordings in {source}")

    if options.out is None:
        options.out = source.rstrip("/\\") + f"_plus{options.picoseconds:g}ps"

    os.makedirs(options.out, exist_ok=True)

    running.write_lock(options.out, sys.argv)

    index_path = os.path.join(source, "index.json")

    original = {}

    if os.path.exists(index_path):
        with open(index_path) as handle:
            for entry in json.load(handle):
                original[entry.get("file")] = entry

    # Whatever is already in the output stays, so a continuation
    # that was interrupted picks up rather than starting over.

    extended = []

    output_index = os.path.join(options.out, "index.json")

    if os.path.exists(output_index):
        try:
            with open(output_index) as handle:
                extended = json.load(handle)
        except (json.JSONDecodeError, OSError):
            extended = []

    print(
        f"continuing runs from {source} "
        f"for a further {options.picoseconds:g} ps"
    )
    print(f"writing to {options.out}")
    print()

    pending = []
    skipped = 0
    unstable = []

    for path in files:
        name = os.path.basename(path)

        target = os.path.join(options.out, name)

        state, _, _ = continuation_state(path, target)

        if state == "skip":
            skipped += 1
        elif state == "unstable":
            unstable.append(name)
        elif state == "unreadable":
            unstable.append(name + " (unreadable)")
        else:
            if state == "redo":
                print(f"  {name}: previous output was unusable, "
                      f"doing it again")

            pending.append(path)

    if skipped:
        print(f"  {skipped} already extended, skipping")

    if unstable:
        print(
            f"  {len(unstable)} skipped as unstable: "
            + ", ".join(unstable[:6])
            + (" ..." if len(unstable) > 6 else "")
        )
        print(
            "    extending a run that blew up would only produce "
            "more of a contaminated trajectory."
        )

    if not pending:
        print()
        print("nothing left to do")
        return

    print(f"  {len(pending)} to run")
    print()

    progress = Progress(
        total_runs=len(pending),
        steps_per_run=int(
            options.picoseconds * 1000.0 / options.time_step
        ),
    )

    for number, path in enumerate(pending, start=len(extended)):
        name = os.path.basename(path)

        progress.start_run()

        recorder, simulation, seconds, strikes = continue_one(
            path, options, progress
        )

        progress.finish_run()
        progress.clear()

        target = os.path.join(options.out, name)

        recorder.save(target)

        result = analysis.analyse(
            recorder, stride=options.stride, structures=False
        )

        entry = dict(original.get(name, {}))

        entry.update({
            "number": number,
            "file": name,
            "frames": len(recorder),
            "picoseconds": round(
                (recorder.times[-1] - recorder.times[0]) / 1000.0, 3
            ),
            "continued_from": source,
            "added_picoseconds": options.picoseconds,
            "wall_seconds": round(seconds, 1),
            "headline": analysis.headline(result),
            "final_species": sorted({
                item["formula"] for item in result["final"]
                if item["heavy"] >= 2
            }),
            "closed_shell": sorted({
                item["formula"] for item in result["final"]
                if item["heavy"] >= 2 and item["closed_shell"]
            }),
            "species_seen": sorted(result["seen"]),
            "heavy_bonds_formed": sum(
                1 for event in result["heavy_events"]
                if event[1] == "formed"
            ),
            "late_formed": result["late_formed"],
            "late_broke": result["late_broke"],
            "turnovers": result["turnovers"],
            "largest_closed": result["largest_closed"],
            "largest_closed_heavy": result["largest_closed_heavy"],
            "largest_any": result.get("largest_any", 0),
            "largest_any_heavy": result.get("largest_any_heavy", 0),
            "most_carbon": result.get("most_carbon", 0),
            "best_tail": result.get("best_tail", 0),
            "best_chain": result.get("best_chain", 0),
            "amphiphiles": result.get("amphiphiles", 0),
            "species_count": result["species_count"],
            "stable": result.get("stable", True),
            "energy_jumps": result.get("energy_jumps", 0),
            "largest_energy_jump": result.get(
                "largest_energy_jump", 0.0
            ),
            "final_temperature": result["temperature"]["final"],
            "final_potential": result["potential"]["final"],
        })

        extended.append(entry)

        with open(
            os.path.join(options.out, "index.json"), "w"
        ) as handle:
            json.dump(extended, handle, indent=1)

        print(
            f"  {name}  now {entry['picoseconds']:g} ps  "
            f"{seconds:6.1f} s  -> {entry['headline']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run many reactive boxes headlessly."
    )

    parser.add_argument("--mixture", default="H rich loose")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--first-seed", type=int, default=None,
        help=(
            "where to start the seed sequence; by default it "
            "continues from whatever the folder already holds"
        )
    )
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
    parser.add_argument(
        "--strike-temperature", type=float, default=25000.0,
        help="channel temperature in kelvin; lightning is 20-30 thousand"
    )
    parser.add_argument(
        "--strike-dissociation", type=float, default=0.6,
        help=(
            "fraction of bonds in the channel broken directly, "
            "standing in for electron impact"
        )
    )
    parser.add_argument(
        "--expand-to", type=float, default=0.0,
        help=(
            "grow the box to this size once expand-at-fs has "
            "passed; 0 leaves it alone"
        )
    )
    parser.add_argument("--expand-at-fs", type=float, default=2000.0)
    parser.add_argument(
        "--expand-rate", type=float, default=0.002,
        help="fraction of the box size changed per captured chunk"
    )
    parser.add_argument(
        "--continue-from", default=None,
        help=(
            "carry on from every run in this folder for a further "
            "--ps picoseconds, rather than starting fresh ones"
        )
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--out", default=None,
        help=(
            "where to write; by default a folder named after the "
            "conditions, so matching batches pool and different "
            "ones stay apart"
        )
    )
    parser.add_argument(
        "--root", default="runs",
        help="where auto-named folders are created"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--hot-until-fs", type=float, default=2000.0,
        help="how long the box is held at the starting temperature"
    )
    parser.add_argument(
        "--hot-temperature", type=float, default=500.0
    )
    parser.add_argument(
        "--cool-temperature", type=float, default=250.0,
        help="temperature the run is trapped at afterwards"
    )
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

    global DEFAULT_SCHEDULE

    DEFAULT_SCHEDULE = [
        (options.hot_until_fs, options.hot_temperature),
        (
            options.hot_until_fs * 2.0,
            (options.hot_temperature + options.cool_temperature) / 2.0
        ),
        (1e12, options.cool_temperature),
    ]

    if options.continue_from:
        run_continuation(options)
        return

    if options.out is None:
        options.out = os.path.join(
            options.root, folder_name(options)
        )

    os.makedirs(options.out, exist_ok=True)

    index_path = os.path.join(options.out, "index.json")

    migrate_index(options.out)

    index, found = existing_batch(options.out)

    wanted = condition_key(options)

    if found is not None and found != wanted:
        differences = [
            f"{key}: existing {found[key]} vs requested "
            f"{wanted[key]}"
            for key in wanted
            if found.get(key) != wanted[key]
        ]

        raise SystemExit(
            "This folder already holds runs made under different "
            "conditions:\n  "
            + "\n  ".join(differences)
            + "\n\nPooling them would produce averages that "
            "describe neither. Use --out to write somewhere else."
        )

    used = existing_seeds(options.out)

    if options.first_seed is None:
        options.first_seed = (max(used) + 1) if used else 0

    if index:
        print(
            f"{len(index)} runs already here, seeds "
            f"{min(used)} to {max(used)}. "
            f"Continuing from seed {options.first_seed}."
        )

    import analysis

    # A lock in the output folder says this batch is alive, so
    # the control panel can tell a running job from one it merely
    # remembers. Removed in the finally below, whatever happens.

    running.write_lock(options.out, sys.argv)

    print(
        f"{options.seeds} runs of {options.picoseconds:.0f} ps, "
        f"mixture {options.mixture!r}, box {options.box} A"
        + (f", expanding to {options.expand_to} A"
           if options.expand_to else "")
    )
    print(f"writing to {options.out}"
    )
    print()

    progress = Progress(
        total_runs=options.seeds,
        steps_per_run=int(
            options.picoseconds * 1000.0 / options.time_step
        ),
        label=options.mixture,
    )

    planned = []

    seed = options.first_seed

    while len(planned) < options.seeds:
        if seed not in used:
            planned.append(seed)

        seed += 1

    skipped = [
        value for value in
        range(options.first_seed, planned[-1] + 1)
        if value in used
    ]

    if skipped:
        print(
            f"Skipping seeds already run here: "
            + ", ".join(str(value) for value in skipped)
        )

    try:
        run_all(planned, mixture, options, index, index_path, progress)
    finally:
        running.remove_lock(options.out)


def run_all(planned, mixture, options, index, index_path, progress):
    import analysis

    for seed in planned:
        progress.start_run()

        recorder, simulation, seconds, strikes = run_one(
            mixture, seed, options, progress
        )

        progress.finish_run()
        progress.clear()

        # Named by seed rather than by position: with several
        # processes filling one folder, position is not known
        # until everything has finished.

        name = f"run_s{seed:04d}.npz"
        path = os.path.join(options.out, name)

        recorder.save(path)

        result = analysis.analyse(recorder, stride=options.stride)

        entry = {
            "number": 0,
            "file": name,
            "mixture": options.mixture,
            "seed": seed,
            "box": options.box,
            "picoseconds": options.picoseconds,
            "atoms": len(simulation.symbols),
            "strikes": strikes,
            "strike_temperature": options.strike_temperature,
            "strike_dissociation": options.strike_dissociation,
            "expand_to": options.expand_to,
            "expand_at_fs": options.expand_at_fs,
            "final_box": round(float(simulation.box_size), 2),
            "hot_until_fs": options.hot_until_fs,
            "hot_temperature": options.hot_temperature,
            "cool_temperature": options.cool_temperature,
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
            # Only chemistry after the divergence point can tell
            # two matched conditions apart.
            "late_formed": result["late_formed"],
            "late_broke": result["late_broke"],
            "turnovers": result["turnovers"],
            "largest_closed": result["largest_closed"],
            "largest_closed_heavy": result["largest_closed_heavy"],
            "largest_any": result.get("largest_any", 0),
            "largest_any_heavy": result.get("largest_any_heavy", 0),
            "most_carbon": result.get("most_carbon", 0),
            "best_tail": result.get("best_tail", 0),
            "best_chain": result.get("best_chain", 0),
            "amphiphiles": result.get("amphiphiles", 0),
            "best_amphiphile": result.get("best_amphiphile", ""),
            "vesicle_ready": result.get("vesicle_ready", False),
            "species_count": result["species_count"],
            "stable": result.get("stable", True),
            "energy_jumps": result.get("energy_jumps", 0),
            "largest_energy_jump": result.get("largest_energy_jump", 0.0),
            "isomers": result.get("isomers", {}),
            "final_temperature": result["temperature"]["final"],
            "final_potential": result["potential"]["final"],
        }

        write_entry(options.out, entry)

        index = rebuild_index(options.out)

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