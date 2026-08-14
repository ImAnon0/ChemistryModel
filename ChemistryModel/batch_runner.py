import argparse
import json
import os
import time


import sys

import numpy as np

import build_box
import discharge
import reactive as R
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
                     time_step, friction, device, compiled_forces=False):
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

    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=box_size,
        time_step=time_step,
        target_temperature=temperature,
        friction=friction,
        device=device,
        random_seed=seed,
    )
    if compiled_forces:
        simulation.enable_compiled_forces()
    return simulation


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


def run_one(mixture, seed, options, progress=None,
            folder=None, on_progress_save=None):
    from recorder import AdaptiveRecorder, Recorder

    simulation = build_simulation(
        mixture,
        options.box,
        seed,
        DEFAULT_SCHEDULE[0][1],
        options.time_step,
        options.friction,
        options.device,
        getattr(options, "compiled_forces", False),
    )

    adaptive = bool(getattr(options, "adaptive_recording", False))
    recorder_class = AdaptiveRecorder if adaptive else Recorder
    recorder_kwargs = (
        dict(
            ordinary_interval_fs=options.capture_every * options.time_step,
            pre_event_fs=getattr(options, "adaptive_pre_event_fs", 100.0),
            post_event_fs=getattr(options, "adaptive_post_event_fs", 100.0),
            energy_jump_ev=getattr(options, "adaptive_energy_jump_ev", 20.0),
            close_contact_scale=getattr(
                options, "adaptive_close_contact_scale", 0.35
            ),
            reaction_window_fs=getattr(
                options, "adaptive_reaction_window_fs", 20.0
            ),
            chemical_context_fs=getattr(
                options, "adaptive_chemical_context_fs", 10.0
            ),
        ) if adaptive else {}
    )
    recorder = recorder_class(
        simulation.symbols, simulation.box_size,
        maximum_frames=options.max_frames, **recorder_kwargs
    )

    generator = np.random.default_rng(seed + 9001)

    total_steps = int(
        options.picoseconds * 1000.0 / options.time_step
    )

    chunk = (
        max(1, min(options.capture_every, int(round(
            getattr(options, "adaptive_candidate_fs", 2.0) / options.time_step
        )))) if adaptive else options.capture_every
    )

    next_strike = (
        options.first_strike_fs if options.strikes > 0 else None
    )

    strikes_done = 0

    started = time.time()

    steps_done = 0

    # Everything that makes the box open rather than sealed:
    # hydrogen leaving, fresh gas arriving, products condensing
    # out of reach. Applied between chunks, since each changes
    # how many atoms there are.

    from open_box import OpenBox

    opening = OpenBox(
        escape_per_ps=options.escape_per_ps,
        feed=getattr(options, "feed_ratio", None),
        trap_per_ps=options.trap_per_ps,
        trap_minimum_heavy=options.trap_minimum_heavy,
        seed=seed + 31337,
    )

    symbols = list(simulation.symbols)

    # The simulation keeps fixed array slots for speed, but an open
    # box can put a brand-new atom into an old slot. IDs preserve
    # that identity change even when H is replaced by H.
    atom_ids = np.arange(len(symbols), dtype=np.uint32)
    next_atom_id = len(symbols)

    next_save = (
        options.save_every_ps * 1000.0
        if options.save_every_ps > 0 else float("inf")
    )

    while steps_done < total_steps:
        simulation.target_temperature = temperature_at(
            DEFAULT_SCHEDULE, simulation.elapsed_femtoseconds
        )

        this_chunk = min(chunk, total_steps - steps_done)

        simulation.step(this_chunk)

        steps_done += this_chunk

        if steps_done % (chunk * 20) == 0:
            if progress is not None:
                progress.show(steps_done)

            if folder is not None:
                write_heartbeat(
                    folder, seed, steps_done, total_steps, started
                )

        record = recorder.observe if adaptive else recorder.capture
        chemical_observation = (
            simulation.chemical_observation() if adaptive else None
        )
        record(
            simulation.positions_numpy,
            simulation.elapsed_femtoseconds,
            simulation.potential_energy,
            simulation.kinetic_energy,
            simulation.temperature,
            velocities=(
                simulation.velocities.detach().cpu().numpy()
            ),
            box_size=simulation.box_size,
            symbols=symbols,
            atom_ids=atom_ids,
            **(
                {"chemical_observation": chemical_observation}
                if adaptive else {}
            ),
        )

        if opening.active:
            symbols = opening.apply(simulation, symbols)

            if opening.last_replaced_slots:
                slots = np.asarray(
                    opening.last_replaced_slots, dtype=int
                )
                count = len(slots)

                atom_ids[slots] = np.arange(
                    next_atom_id,
                    next_atom_id + count,
                    dtype=np.uint32,
                )
                next_atom_id += count

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
                # bonds_in_channel as well as dissociated: a strike that
                # catches atoms but finds no whole bond among them reports
                # zero broken, and that reads as a broken kick when it is
                # really a channel too narrow to contain both ends of
                # anything.
                print(
                    f"    strike at "
                    f"{simulation.elapsed_femtoseconds:.0f} fs: "
                    f"{report['struck']} atoms, "
                    f"{report.get('bonds_in_channel', 0)} whole bonds, "
                    f"{report['dissociated']} broken"
                )

            strikes_done += 1
            next_strike += options.strike_interval_fs

        # Written out along the way rather than only at the
        # end. The file is a complete run of however far it has
        # got, so a long one can be read while the rest is still
        # computing, and a crash costs the last few picoseconds
        # instead of all of them.

        if (
            on_progress_save is not None
            and options.save_every_ps > 0
            and simulation.elapsed_femtoseconds
            >= next_save
        ):
            on_progress_save(
                recorder, simulation, time.time() - started,
                strikes_done,
            )

            next_save += options.save_every_ps * 1000.0

        if not np.isfinite(simulation.potential_energy):
            print(f"  seed {seed}: went unstable, stopping early")
            break

    if options.verbose:
        for line in opening.report():
            print(f"    {line}")

    return recorder, simulation, time.time() - started, strikes_done


def apply_openings_to_group(simulation, openings, symbol_lists):
    # Each box decides for itself what leaves and what arrives,
    # then the whole group is updated in one go: a single call
    # rebuilds the neighbour table and the forces, and doing that
    # once for the lot rather than once per box is most of the
    # point of grouping them.

    import numpy as np

    all_slots = []
    all_symbols = []
    all_places = []

    for box, opening in enumerate(openings):
        start = box * simulation.per_box
        stop = start + simulation.per_box

        positions = simulation.positions_numpy[start:stop]

        symbols = symbol_lists[box]

        leaving, arriving, places = opening.choose(
            positions, symbols, simulation.box_size,
            simulation.elapsed_femtoseconds,
        )

        if not leaving:
            continue

        for slot, symbol in zip(leaving, arriving):
            symbols[slot] = symbol

        all_slots.extend(start + slot for slot in leaving)
        all_symbols.extend(arriving)
        all_places.extend(places)

    if all_slots:
        simulation.replace_atoms(
            all_slots, all_symbols, np.array(all_places)
        )

    return symbol_lists


def strike_group(simulation, generators, options):
    # A discharge is a column through one box, so a group of them
    # needs one per box rather than one across the lot. Each box
    # is handed its own slice of the positions and velocities and
    # struck independently, with its own random channel.

    import torch

    positions = simulation.positions_numpy
    velocities = simulation.velocities.detach().cpu().numpy()
    masses = simulation.masses.detach().cpu().numpy()
    types = simulation.types_numpy

    updated = velocities.copy()

    reports = []

    for box in range(simulation.box_count):
        start = box * simulation.per_box
        stop = start + simulation.per_box

        changed, report = discharge.strike(
            positions[start:stop],
            velocities[start:stop],
            masses[start:stop],
            types[start:stop],
            simulation.box_size,
            generators[box],
            radius=options.strike_radius,
            temperature=options.strike_temperature,
            dissociation=options.strike_dissociation,
        )

        updated[start:stop] = changed

        reports.append(report)

    simulation.velocities = torch.tensor(
        updated,
        device=simulation.device,
        dtype=simulation.dtype,
    )

    return reports


def build_group(mixture, seeds, options):
    # One box per seed, all the same composition and size.

    kind, contents = mixture

    boxes = []

    for seed in seeds:
        if kind == "molecules":
            symbols, positions = build_box.build(
                contents, options.box, random_seed=seed
            )
        else:
            symbols, positions = build_box.loose_atoms(
                contents,
                options.box,
                minimum_separation=1.25,
                random_seed=seed,
            )

        boxes.append((symbols, positions))

    return boxes


def run_group(mixture, seeds, options, progress=None, folder=None):
    # Several boxes advanced together in one process.
    #
    # A box of three hundred atoms leaves the card mostly idle, so
    # eight of them cost about twice one rather than eight times.
    # Everything else is as it would be running them one at a
    # time: each keeps its own cell, its own starting positions,
    # its own recording and its own discharges.

    from recorder import AdaptiveRecorder, Recorder

    from batched_torch import BatchedReactiveSimulation

    boxes = build_group(mixture, seeds, options)

    simulation = BatchedReactiveSimulation(
        boxes=boxes,
        box_size=options.box,
        time_step=options.time_step,
        target_temperature=DEFAULT_SCHEDULE[0][1],
        friction=options.friction,
        device=options.device,
        random_seed=seeds[0],
    )
    if getattr(options, "compiled_forces", False):
        simulation.enable_compiled_forces()

    adaptive = bool(getattr(options, "adaptive_recording", False))
    recorder_class = AdaptiveRecorder if adaptive else Recorder
    recorder_kwargs = (
        dict(
            ordinary_interval_fs=options.capture_every * options.time_step,
            pre_event_fs=getattr(options, "adaptive_pre_event_fs", 100.0),
            post_event_fs=getattr(options, "adaptive_post_event_fs", 100.0),
            energy_jump_ev=getattr(options, "adaptive_energy_jump_ev", 20.0),
            close_contact_scale=getattr(
                options, "adaptive_close_contact_scale", 0.35
            ),
            reaction_window_fs=getattr(
                options, "adaptive_reaction_window_fs", 20.0
            ),
            chemical_context_fs=getattr(
                options, "adaptive_chemical_context_fs", 10.0
            ),
        ) if adaptive else {}
    )
    recorders = [
        recorder_class(
            simulation.symbols_for(box),
            simulation.box_size,
            maximum_frames=options.max_frames,
            **recorder_kwargs,
        )
        for box in range(len(seeds))
    ]

    generators = [
        np.random.default_rng(seed + 9001) for seed in seeds
    ]

    total_steps = int(
        options.picoseconds * 1000.0 / options.time_step
    )

    chunk = (
        max(1, min(options.capture_every, int(round(
            getattr(options, "adaptive_candidate_fs", 2.0) / options.time_step
        )))) if adaptive else options.capture_every
    )

    next_strike = (
        options.first_strike_fs if options.strikes > 0 else None
    )

    strikes_done = 0

    started = time.time()

    steps_done = 0

    stopped_early = False

    seed_label = (
        str(seeds[0]) if len(seeds) == 1
        else f"{seeds[0]}-{seeds[-1]}"
    )
    if folder is not None:
        write_heartbeat(
            folder, seed_label, 0, total_steps, started,
            boxes_in_group=len(seeds),
        )

    # Each box needs its own flow: hydrogen leaving one is
    # nothing to do with another, and a trap catches whatever
    # that particular box has made.

    from open_box import OpenBox

    openings = [
        OpenBox(
            escape_per_ps=options.escape_per_ps,
            feed=getattr(options, "feed_ratio", None),
            trap_per_ps=options.trap_per_ps,
            trap_minimum_heavy=options.trap_minimum_heavy,
            seed=seed + 31337,
        )
        for seed in seeds
    ]

    symbol_lists = [
        list(simulation.symbols_for(box))
        for box in range(len(seeds))
    ]

    atom_id_lists = [
        np.arange(len(symbols), dtype=np.uint32)
        for symbols in symbol_lists
    ]
    next_atom_ids = [
        len(symbols) for symbols in symbol_lists
    ]

    # Grouped runs used to ignore --save-every-ps and only write
    # their recorders after the whole group finished. Keep the same
    # checkpoint behaviour as single-box runs so a 20 ps group can
    # leave usable 5/10/15 ps progress on disk.
    next_save = (
        options.save_every_ps * 1000.0
        if options.save_every_ps > 0 else float("inf")
    )

    if options.save_every_ps > 0:
        import analysis

    while steps_done < total_steps:
        simulation.target_temperature = temperature_at(
            DEFAULT_SCHEDULE, simulation.elapsed_femtoseconds
        )

        this_chunk = min(chunk, total_steps - steps_done)

        simulation.step(this_chunk)

        steps_done += this_chunk

        heartbeat_now = steps_done % (chunk * 20) == 0
        if heartbeat_now:
            if progress is not None:
                progress.show(steps_done)

        potentials = simulation.potential_per_box
        kinetics, temperatures = simulation.thermodynamics_per_box
        positions_per_box = simulation.positions_per_box
        velocities_per_box = simulation.velocities_per_box
        chemical_observations = (
            simulation.chemical_observations() if adaptive else None
        )

        if heartbeat_now and folder is not None:
            live = live_chemistry_summary(
                simulation, seeds, positions_per_box, temperatures
            )
            write_heartbeat(
                folder, seed_label, steps_done,
                total_steps, started,
                boxes_in_group=len(seeds), live=live,
            )

        for box, recorder in enumerate(recorders):
            record = recorder.observe if adaptive else recorder.capture
            record(
                positions_per_box[box],
                simulation.elapsed_femtoseconds,
                float(potentials[box]),
                float(kinetics[box]),
                float(temperatures[box]),
                velocities=velocities_per_box[box],
                box_size=simulation.box_size,
                symbols=symbol_lists[box],
                atom_ids=atom_id_lists[box],
                **(
                    {"chemical_observation": chemical_observations[box]}
                    if adaptive else {}
                ),
            )

        if openings and openings[0].active:
            symbol_lists = apply_openings_to_group(
                simulation, openings, symbol_lists
            )

            for box, opening in enumerate(openings):
                if not opening.last_replaced_slots:
                    continue

                slots = np.asarray(
                    opening.last_replaced_slots, dtype=int
                )
                count = len(slots)
                start_id = next_atom_ids[box]

                atom_id_lists[box][slots] = np.arange(
                    start_id,
                    start_id + count,
                    dtype=np.uint32,
                )
                next_atom_ids[box] += count

        if (
            next_strike is not None
            and simulation.elapsed_femtoseconds >= next_strike
            and strikes_done < options.strikes
        ):
            reports = strike_group(simulation, generators, options)

            if options.verbose:
                broken = sum(
                    report["dissociated"] for report in reports
                )
                available = sum(
                    report.get("bonds_in_channel", 0) for report in reports
                )

                print(
                    f"    strike at "
                    f"{simulation.elapsed_femtoseconds:.0f} fs: "
                    f"{broken} of {available} whole bonds broken across "
                    f"{len(reports)} boxes"
                )

            strikes_done += 1
            next_strike += options.strike_interval_fs

        # Checkpoint every recorder in the group at the requested
        # interval. The final state is written by run_grouped()
        # immediately after this function returns, so there is no
        # need to do a duplicate checkpoint at the exact end.
        if (
            folder is not None
            and options.save_every_ps > 0
            and simulation.elapsed_femtoseconds >= next_save
            and steps_done < total_steps
        ):
            elapsed_wall = time.time() - started
            each = elapsed_wall / max(len(seeds), 1)

            for box, seed in enumerate(seeds):
                recorder = recorders[box]

                path = os.path.join(
                    folder, f"run_s{seed:04d}.npz"
                )
                recorder.save(path)

                entry = summarise_run(
                    recorder, simulation, seed, each, strikes_done,
                    options, analysis, box=box,
                )
                entry["finished"] = False

                write_entry(folder, entry)

            rebuild_index(folder)

            if progress is not None:
                progress.clear()

            saved_ps = min(
                recorder.times[-1] for recorder in recorders
            ) / 1000.0

            print(
                f"  group {seeds[0]}-{seeds[-1]}: "
                f"{saved_ps:g} ps saved so far"
            )

            # Normally one interval is crossed at a time. A while
            # loop also handles unusually large capture chunks
            # without writing several identical checkpoints.
            while next_save <= simulation.elapsed_femtoseconds:
                next_save += options.save_every_ps * 1000.0

        # One box going bad spoils the forces for the whole group,
        # since they share a tensor. Better to stop and say which.

        if not np.all(np.isfinite(potentials)):
            bad = [
                seeds[box]
                for box in range(len(seeds))
                if not np.isfinite(potentials[box])
            ]

            print(
                f"  seeds {bad}: went unstable, stopping the "
                f"whole group early"
            )

            stopped_early = True

            break

    return (
        recorders,
        simulation,
        time.time() - started,
        strikes_done,
        stopped_early,
    )


def summarise_run(recorder, simulation, seed, seconds, strikes,
                  options, analysis, box=None):
    # Everything written about one finished run, whether it was
    # computed alone or alongside others.

    result = analysis.analyse(
        recorder, stride=options.stride, structures=False
    )

    if box is None:
        move_cap_events = int(simulation.capped_steps)
    else:
        start = box * simulation.per_box
        stop = start + simulation.per_box
        move_cap_events = int(np.sum(
            simulation.capped_atom_counts[start:stop]
        ))

    return {
        "number": 0,
        "file": f"run_s{seed:04d}.npz",
        "mixture": options.mixture,
        "seed": seed,
        "box": round(float(simulation.box_size), 2),
        "atoms": len(recorder.symbols),
        "picoseconds": round(
            (recorder.times[-1] - recorder.times[0]) / 1000.0, 3
        ),
        "requested_picoseconds": round(float(options.picoseconds), 3),
        "strikes": strikes,
        "strike_temperature": options.strike_temperature,
        "strike_dissociation": options.strike_dissociation,
        "expand_to": options.expand_to,
        "expand_at_fs": options.expand_at_fs,
        "final_box": round(float(simulation.box_size), 2),
        "hot_until_fs": options.hot_until_fs,
        "hot_temperature": options.hot_temperature,
        "cool_temperature": options.cool_temperature,
        "frames": len(recorder),
        "recording_format": int(getattr(recorder, "format_version", 1)),
        "compiled_forces": bool(getattr(options, "compiled_forces", False)),
        "adaptive_recording": bool(getattr(options, "adaptive_recording", False)),
        "adaptive_candidate_fs": (
            float(options.adaptive_candidate_fs)
            if getattr(options, "adaptive_recording", False) else None
        ),
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
        "move_cap_events": move_cap_events,
        "final_temperature": result["temperature"]["final"],
        "final_potential": result["potential"]["final"],
    }


def run_grouped(planned, mixture, options, progress):
    import analysis

    groups = [
        planned[start:start + options.group]
        for start in range(0, len(planned), options.group)
    ]

    # The progress bar counts whatever it is told a run is, and
    # in this mode a run is a whole group. Left as the number of
    # seeds it would think there were eight times as many still
    # to do, and quote a finishing time eight times too far away.

    progress.total_runs = len(groups)

    print(
        f"running {options.group} boxes at a time, "
        f"{len(groups)} group"
        + ("s" if len(groups) != 1 else "")
        + f", {len(planned)} runs in all"
    )
    print()

    for group in groups:
        progress.start_run()

        outcome = run_group(
            mixture, group, options, progress, options.out
        )

        recorders, simulation, seconds, strikes, stopped = outcome

        progress.finish_run()
        progress.clear()

        # The time is shared, so each run is charged its share.

        each = seconds / max(len(group), 1)

        for index, seed in enumerate(group):
            recorder = recorders[index]

            path = os.path.join(
                options.out, f"run_s{seed:04d}.npz"
            )

            recorder.save(path)

            entry = summarise_run(
                recorder, simulation, seed, each, strikes,
                options, analysis, box=index,
            )
            entry["finished"] = True

            write_entry(options.out, entry)

            print(
                f"seed {seed:<5d} {each:6.1f} s  "
                f"{len(recorder):5d} frames  "
                f"-> {entry['headline']}"
            )

        rebuild_index(options.out)

        print()

        if stopped:
            print(
                "  stopped early, so the rest of this group is "
                "shorter than asked for"
            )
            print()


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

    current_symbols = recorder.symbols_at(last)
    current_atom_ids = recorder.atom_ids_at(last).copy()

    if not recorder.has_velocities:
        raise SystemExit(
            f"{path} has no velocities stored, so it cannot be "
            f"resumed exactly. Only runs recorded after velocity "
            f"capture was added can be continued."
        )

    simulation = ReactiveSimulation(
        symbols=current_symbols,
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

        if steps_done % (chunk * 20) == 0:
            if progress is not None:
                progress.show(steps_done)

            if options.out:
                write_heartbeat(
                    options.out,
                    int(recorder.times[last]),
                    steps_done,
                    added_steps,
                    started,
                )

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
            symbols=current_symbols,
            atom_ids=current_atom_ids,
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


def heartbeat_path(folder):
    return os.path.join(folder, f".progress_{os.getpid()}.json")


def live_chemistry_summary(simulation, seeds, positions_per_box,
                           temperatures):
    """Compact instantaneous chemistry for the Lab's live batch panel."""
    positions = np.asarray(positions_per_box).reshape(-1, 3)
    neighbours = simulation.neighbours.detach().cpu().numpy()
    mask = simulation.neighbour_mask.detach().cpu().numpy()
    types = np.asarray(simulation.types_numpy, dtype=int)

    rows = np.broadcast_to(
        np.arange(len(positions))[:, None], neighbours.shape
    )
    keep = mask & (neighbours > rows)
    first = rows[keep]
    second = neighbours[keep]

    offsets = positions[second] - positions[first]
    offsets -= simulation.box_size * np.round(
        offsets / simulation.box_size
    )
    distances = np.linalg.norm(offsets, axis=1)
    inner = R.CUTOFF_INNER[types[first], types[second]]
    outer = R.CUTOFF_OUTER[types[first], types[second]]
    bonded = R.smooth_cutoff(distances, inner, outer) > 0.35
    first = first[bonded]
    second = second[bonded]

    parent = np.arange(len(positions))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in zip(first, second):
        a_root = find(int(a))
        b_root = find(int(b))
        if a_root != b_root:
            parent[b_root] = a_root

    components = {}
    for atom in range(len(positions)):
        components.setdefault(find(atom), []).append(atom)

    symbols = np.asarray(R.ELEMENTS, dtype=object)[types]
    candidates = []
    for members in components.values():
        member_symbols = symbols[members]
        heavy = int(np.count_nonzero(member_symbols != "H"))
        if heavy < 2:
            continue
        counts = {
            symbol: int(np.count_nonzero(member_symbols == symbol))
            for symbol in ("C", "N", "O", "H")
        }
        formula = "".join(
            symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
            for symbol in ("C", "N", "O", "H")
            if counts[symbol]
        )
        seed_index = int(members[0]) // int(simulation.per_box)
        candidates.append({
            "seed": int(seeds[seed_index]),
            "formula": formula,
            "atoms": len(members),
            "heavy": heavy,
            "carbon": counts["C"],
        })

    largest = max(
        candidates,
        key=lambda item: (item["atoms"], item["heavy"], item["carbon"]),
        default=None,
    )
    carbon = int(R.ELEMENT_INDEX["C"])
    cc_bonds = int(np.count_nonzero(
        (types[first] == carbon) & (types[second] == carbon)
    ))
    hottest = int(np.argmax(temperatures))

    per_seed = []
    atoms_per_box = int(simulation.per_box)
    edge_boxes = first // atoms_per_box
    for box, seed in enumerate(seeds):
        here = [item for item in candidates if item["seed"] == int(seed)]
        seed_largest = max(
            here,
            key=lambda item: (item["atoms"], item["heavy"], item["carbon"]),
            default=None,
        )
        seed_edges = edge_boxes == box
        seed_cc = int(np.count_nonzero(
            seed_edges
            & (types[first] == carbon)
            & (types[second] == carbon)
        ))
        per_seed.append({
            "seed": int(seed),
            "largest": seed_largest,
            "heavy_molecules": len(here),
            "cc_bonds": seed_cc,
            "temperature_K": round(float(temperatures[box])),
        })

    return {
        "time_fs": float(simulation.elapsed_femtoseconds),
        "largest": largest,
        "heavy_molecules": len(candidates),
        "cc_bonds": cc_bonds,
        "hottest_seed": int(seeds[hottest]),
        "hottest_K": round(float(temperatures[hottest])),
        "per_seed": per_seed,
    }


def write_heartbeat(folder, seed, done, total, started, boxes_in_group=1,
                    live=None):
    # How far through the current run this process is.
    #
    # The control panel cannot see the progress bar printed to the
    # console, and counting finished runs only moves once every
    # several minutes. A small file written as the run proceeds
    # gives the panel something to show in between.

    try:
        temporary = heartbeat_path(folder) + ".part"
        with open(temporary, "w") as handle:
            payload = {
                "pid": os.getpid(),
                "seed": seed,
                "steps_done": done,
                "steps_total": total,
                "run_started": started,
                "updated": time.time(),
                "boxes_in_group": int(boxes_in_group),
            }
            if live is not None:
                payload["live"] = live
            json.dump(payload, handle)
        os.replace(temporary, heartbeat_path(folder))
    except OSError:
        pass


def clear_heartbeat(folder):
    try:
        os.remove(heartbeat_path(folder))
    except OSError:
        pass


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
            if not (name.startswith("seed_") and name.endswith(".json")):
                continue

            path = os.path.join(directory, name)

            try:
                with open(path) as handle:
                    entry = json.load(handle)
            except (json.JSONDecodeError, OSError):
                continue

            # Periodic checkpoints deliberately write an entry so
            # the partial run can be inspected. They are not a
            # completed seed and must not make a restarted batch
            # skip that seed. Older entries have no flag and are
            # treated as finished for backward compatibility.
            if entry.get("finished", True) is False:
                continue

            if entry.get("seed") is not None:
                seeds.add(int(entry["seed"]))

    for entry in read_existing_index(folder):
        if entry.get("finished", True) is False:
            continue

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
        "adaptive_recording": bool(
            getattr(options, "adaptive_recording", False)
        ),
        "adaptive_candidate_fs": (
            round(float(options.adaptive_candidate_fs), 3)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_pre_event_fs": (
            round(float(options.adaptive_pre_event_fs), 3)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_post_event_fs": (
            round(float(options.adaptive_post_event_fs), 3)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_energy_jump_ev": (
            round(float(options.adaptive_energy_jump_ev), 6)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_close_contact_scale": (
            round(float(options.adaptive_close_contact_scale), 4)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_reaction_window_fs": (
            round(float(options.adaptive_reaction_window_fs), 3)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "adaptive_chemical_context_fs": (
            round(float(options.adaptive_chemical_context_fs), 3)
            if getattr(options, "adaptive_recording", False) else 0.0
        ),
        "compiled_forces": bool(getattr(options, "compiled_forces", False)),
    }


def folder_name(options):
    # A readable name that encodes the conditions, so a folder
    # says what it holds without opening it.

    safe = options.mixture.strip().replace("+", "plus")
    safe = "-".join(safe.replace("_", " ").split())

    parts = [
        safe,
        f"{options.box:g}A",
        f"{options.picoseconds:g}ps",
    ]

    if options.strikes:
        parts.append(
            f"lightning{options.strikes}x"
            f"{options.strike_temperature / 1000:g}kK"
        )

    if options.cool_temperature != 250.0:
        parts.append(f"cool{options.cool_temperature:g}K")

    if getattr(options, "adaptive_recording", False):
        candidate = float(options.adaptive_candidate_fs)
        parts.append("v2" if candidate == 2.0 else f"v2-{candidate:g}fs")
    else:
        parts.append("v1")

    if getattr(options, "compiled_forces", False):
        parts.append("compiled-experimental")

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
        "picoseconds": round(float(
            first.get("requested_picoseconds", first.get("picoseconds", 0))
        ), 3),
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
        # Indexes written before version-2 recording existed are legacy by
        # definition. Missing adaptive fields must compare as disabled rather
        # than making an old folder unreadable or raising a KeyError.
        "adaptive_recording": bool(
            first.get("adaptive_recording", False)
        ),
        "compiled_forces": bool(first.get("compiled_forces", False)),
        "adaptive_candidate_fs": round(float(
            first.get("adaptive_candidate_fs", 0) or 0
        ), 3),
        "adaptive_pre_event_fs": round(float(
            first.get("adaptive_pre_event_fs", 0) or 0
        ), 3),
        "adaptive_post_event_fs": round(float(
            first.get("adaptive_post_event_fs", 0) or 0
        ), 3),
        "adaptive_energy_jump_ev": round(float(
            first.get("adaptive_energy_jump_ev", 0) or 0
        ), 6),
        "adaptive_close_contact_scale": round(float(
            first.get("adaptive_close_contact_scale", 0) or 0
        ), 4),
        "adaptive_reaction_window_fs": round(float(
            first.get("adaptive_reaction_window_fs", 0) or 0
        ), 3),
        "adaptive_chemical_context_fs": round(float(
            first.get("adaptive_chemical_context_fs", 0) or 0
        ), 3),
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
            "recording_format": int(getattr(recorder, "format_version", 1)),
            "adaptive_recording": bool(
                getattr(options, "adaptive_recording", False)
            ),
            "adaptive_candidate_fs": (
                float(options.adaptive_candidate_fs)
                if getattr(options, "adaptive_recording", False) else None
            ),
            "picoseconds": round(
                (recorder.times[-1] - recorder.times[0]) / 1000.0, 3
            ),
            "continued_from": source,
            "added_picoseconds": options.picoseconds,
            # Recorded explicitly rather than inherited: a
            # continuation can be run at a different temperature
            # from the batch it extends, and inheriting the old
            # value would describe the wrong experiment.
            "cool_temperature": options.cool_temperature,
            "hot_temperature": options.cool_temperature,
            "hot_until_fs": 0.0,
            "strikes": strikes,
            "strike_temperature": options.strike_temperature,
            "strike_dissociation": options.strike_dissociation,
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
        "--seed-list", default=None,
        help=(
            "exactly which seeds to run, comma separated. Used "
            "when several processes share a folder and the free "
            "seeds are not contiguous"
        )
    )
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
    parser.add_argument(
        "--compiled-forces", action="store_true",
        help="experimental Triton/Inductor force evaluation (CUDA only)",
    )
    parser.add_argument("--capture-every", type=int, default=40,
                        help="simulation steps between recorded frames")
    recording_mode = parser.add_mutually_exclusive_group()
    recording_mode.add_argument(
        "--adaptive-recording", dest="adaptive_recording",
        action="store_true",
        help="use version-2 event-aware recording (the default)",
    )
    recording_mode.add_argument(
        "--legacy-recording", dest="adaptive_recording",
        action="store_false",
        help="write the original version-1 fixed-cadence format",
    )
    parser.set_defaults(adaptive_recording=True)
    parser.add_argument(
        "--adaptive-candidate-fs", type=float, default=2.0,
        help="physical-time interval between adaptive candidate observations",
    )
    parser.add_argument("--adaptive-pre-event-fs", type=float, default=100.0)
    parser.add_argument("--adaptive-post-event-fs", type=float, default=100.0)
    parser.add_argument("--adaptive-energy-jump-ev", type=float, default=20.0)
    parser.add_argument(
        "--adaptive-close-contact-scale", type=float, default=0.35,
        help="fraction of pair inner cutoff considered unusually compressed",
    )
    parser.add_argument(
        "--adaptive-reaction-window-fs", type=float, default=20.0,
        help="coalesce adjacent chemical changes into one protected window",
    )
    parser.add_argument(
        "--adaptive-chemical-context-fs", type=float, default=10.0,
        help="dense context before/after chemical reaction-window starts",
    )
    parser.add_argument("--max-frames", type=int, default=40000)
    parser.add_argument("--strikes", type=int, default=0)
    parser.add_argument("--first-strike-fs", type=float, default=3000.0)
    parser.add_argument("--strike-interval-fs", type=float,
                        default=3000.0)
    parser.add_argument("--strike-radius", type=float, default=3.5)
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expand-at-fs", type=float, default=2000.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expand-rate", type=float, default=0.002,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--continue-from", default=None,
        help=(
            "carry on from every run in this folder for a further "
            "--ps picoseconds, rather than starting fresh ones"
        )
    )
    parser.add_argument(
        "--escape-per-ps", type=float, default=0.0,
        help=(
            "how many hydrogen molecules leave the box each "
            "picosecond. H2 is light enough to escape the planet, "
            "which is why the early atmosphere grew less reducing"
        )
    )
    parser.add_argument(
        "--feed", default=None,
        help=(
            "what arrives to replace whatever leaves, as "
            "C:2,H:5,N:1,O:1. Defaults to the starting mixture, "
            "which is what a volcano venting the same rock over "
            "and over would give"
        )
    )
    parser.add_argument(
        "--trap-per-ps", type=float, default=0.0,
        help=(
            "how many formed molecules are taken out of reach "
            "each picosecond, as Miller's cold trap did"
        )
    )
    parser.add_argument(
        "--trap-minimum-heavy", type=int, default=3,
        help="smallest molecule the trap will catch"
    )
    parser.add_argument(
        "--save-every-ps", type=float, default=0.0,
        help=(
            "write the recording and its index entry this often "
            "rather than only at the end, so a long run can be "
            "read while the rest of it is still computing"
        )
    )
    parser.add_argument(
        "--group", type=int, default=16,
        help=(
            "how many boxes to advance together in one process. "
            "A single box leaves the card mostly idle; use 1 for "
            "legacy single-box scheduling"
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

    adaptive_explicit = "--adaptive-recording" in sys.argv[1:]
    options = parser.parse_args()

    if options.expand_to:
        raise SystemExit(
            "mid-simulation box expansion has been removed. Choose the "
            "desired fixed size with --box before starting the run."
        )

    if options.adaptive_recording:
        incompatible = []
        if options.escape_per_ps or options.trap_per_ps or options.feed:
            incompatible.append("open-box feed/escape/trapping")
        if options.strikes:
            incompatible.append("lightning strikes")
        if options.continue_from:
            incompatible.append("continuation")
        if incompatible:
            explanation = ", ".join(incompatible)
            if adaptive_explicit:
                raise SystemExit(
                    "adaptive recording is not yet cadence-safe with "
                    + explanation
                )
            options.adaptive_recording = False
            print(
                "using legacy recording for " + explanation,
                file=sys.stderr,
            )
        if options.adaptive_candidate_fs <= 0:
            raise SystemExit("adaptive-candidate-fs must be positive")

    if options.group < 1:
        raise SystemExit("group must be at least 1")

    from mixtures import STARTS

    if options.mixture not in STARTS:
        raise SystemExit(
            f"unknown mixture {options.mixture!r}. "
            f"options: {sorted(STARTS)}"
        )

    mixture = STARTS[options.mixture]

    # What replaces whatever leaves. Left alone, it matches the
    # mixture the box started from, so the composition is held
    # steady rather than drifting as material passes through.

    options.feed_ratio = None

    if options.feed:
        options.feed_ratio = {}

        for piece in options.feed.split(","):
            if ":" not in piece:
                continue

            name, share = piece.split(":", 1)

            try:
                options.feed_ratio[name.strip()] = float(share)
            except ValueError:
                continue
    elif mixture[0] == "atoms":
        options.feed_ratio = dict(mixture[1])

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

    if found is not None:
        def condition_differs(key):
            existing = found.get(key)
            requested = wanted[key]
            if key == "picoseconds":
                # Older indexes stored first-to-last-frame span, which is one
                # capture interval shorter than the requested run duration.
                return abs(float(existing) - float(requested)) > 0.011
            return existing != requested

        differences = [
            f"{key}: existing {found.get(key)} vs requested "
            f"{wanted[key]}"
            for key in wanted
            if condition_differs(key)
        ]
        if differences:
            raise SystemExit(
                "This folder already holds runs made under different "
                "conditions:\n  "
                + "\n  ".join(differences)
                + "\n\nPooling them would produce averages that "
                "describe neither. Use --out to write somewhere else."
            )

    used = existing_seeds(options.out)

    if options.first_seed is None:
        # Start from the lowest seed not already present rather
        # than from one past the highest. A batch that crashed
        # partway, or one split across several processes, leaves
        # gaps; jumping past them means those boxes are never run
        # and the folder ends up with holes in it.

        options.first_seed = min(used) if used else 0

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

    if options.seed_list:
        # An explicit list wins over anything worked out here.
        # Splitting a batch across processes leaves each part with
        # seeds that need not be contiguous, and guessing them
        # from a starting point plus a count gets it wrong the
        # moment there is a gap.

        planned = []

        for piece in options.seed_list.split(","):
            piece = piece.strip()

            if not piece:
                continue

            try:
                value = int(piece)
            except ValueError:
                continue

            if value not in used:
                planned.append(value)
    else:
        planned = []

        seed = options.first_seed

        while len(planned) < options.seeds:
            if seed not in used:
                planned.append(seed)

            seed += 1

    if not planned:
        print("every requested seed is already here, nothing to do")

        running.remove_lock(options.out)

        return

    skipped = sorted(
        value for value in used
        if min(planned) <= value <= max(planned)
    )

    if skipped:
        print(
            f"Already here, so skipped: "
            + ", ".join(str(value) for value in skipped)
        )

    gaps = [
        value for value in planned
        if used and value < max(used)
    ]

    if gaps:
        print(
            f"Filling gaps left by earlier runs: "
            + ", ".join(str(value) for value in gaps)
        )

    try:
        run_all(planned, mixture, options, index, index_path, progress)
    finally:
        clear_heartbeat(options.out)
        running.remove_lock(options.out)


def run_all(planned, mixture, options, index, index_path, progress):
    import analysis

    if options.group > 1:
        run_grouped(planned, mixture, options, progress)
        return

    for seed in planned:
        progress.start_run()

        def save_progress(recorder, simulation, seconds, strikes,
                          seed=seed):
            # The same writing that happens at the end, done
            # early. Whatever is on disk is a complete run of
            # however far it has got.

            path = os.path.join(
                options.out, f"run_s{seed:04d}.npz"
            )

            recorder.save(path)

            entry = summarise_run(
                recorder, simulation, seed, seconds, strikes,
                options, analysis,
            )

            entry["finished"] = False

            write_entry(options.out, entry)

            rebuild_index(options.out)

            progress.clear()

            print(
                f"  seed {seed}: "
                f"{entry['picoseconds']:g} ps saved so far"
            )

        recorder, simulation, seconds, strikes = run_one(
            mixture, seed, options, progress, options.out,
            save_progress if options.save_every_ps > 0 else None,
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
            "move_cap_events": int(simulation.capped_steps),
            "isomers": result.get("isomers", {}),
            "final_temperature": result["temperature"]["final"],
            "final_potential": result["potential"]["final"],
        }

        entry["finished"] = True

        write_entry(options.out, entry)

        index = rebuild_index(options.out)

        # Reported by seed rather than by position: with several
        # processes filling one folder, position is not settled
        # until every one of them has finished.

        print(
            f"seed {seed:<5d} {seconds:6.1f} s  "
            f"{len(recorder):5d} frames  "
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
