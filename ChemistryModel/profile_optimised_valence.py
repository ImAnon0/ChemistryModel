"""Profile the optimised-valence physics stack against reactive base.

Place in the ChemistryModel directory (next to batch_runner.py) and run:

    python profile_optimised_valence.py
    python profile_optimised_valence.py --physics reactive     # baseline
    python profile_optimised_valence.py --steps 100 --width 16

It builds the same boxes batch_runner builds, constructs the simulation
through the same grouped_simulation_class path, and profiles stepping only.
No recorder, no analysis, no disk output -- dynamics cost only.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import torch

import batch_runner
import mixtures


def build_simulation(options):
    """Mirror batch_runner.run_group's construction, without recorders."""
    mixture = mixtures.STARTS[options.mixture]
    seeds = list(range(options.seed, options.seed + options.width))

    build_options = SimpleNamespace(box=options.box)
    boxes = batch_runner.build_group(mixture, seeds, build_options)

    SimulationClass = batch_runner.grouped_simulation_class(options)

    simulation_kwargs = {}
    if options.physics == "optimised-valence":
        if options.h_s2_eigvalsh_chunk_size is not None:
            simulation_kwargs["h_s2_eigvalsh_chunk_size"] = int(
                options.h_s2_eigvalsh_chunk_size
            )
        if options.h_transition_assembly is not None:
            simulation_kwargs["h_transition_assembly"] = (
                options.h_transition_assembly
            )

    simulation = SimulationClass(
        boxes=boxes,
        box_size=options.box,
        time_step=options.time_step,
        target_temperature=batch_runner.DEFAULT_SCHEDULE[0][1],
        friction=options.friction,
        device=options.device,
        random_seed=seeds[0],
        **simulation_kwargs,
    )

    if options.compiled_forces:
        simulation.enable_compiled_forces()

    return simulation


def synchronize(simulation):
    device = getattr(simulation, "device", None)
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def measure_throughput(simulation, steps, width):
    """Plain wall-clock timing, no profiler overhead."""
    synchronize(simulation)
    started = time.perf_counter()
    simulation.step(steps)
    synchronize(simulation)
    elapsed = time.perf_counter() - started

    print()
    print("--- throughput (no profiler) ---")
    print(f"steps stepped:        {steps}")
    print(f"wall time:            {elapsed:.3f} s")
    print(f"per-box steps/s:      {steps / elapsed:.1f}")
    print(f"aggregate steps/s:    {steps * width / elapsed:.1f}")
    estimated = (20000.0 / 0.25) / (steps / elapsed) / 60.0
    print(f"implied 20 ps x {width}:  {estimated:.1f} min")
    return elapsed


def profile(simulation, steps, rows, export):
    from torch.profiler import ProfilerActivity, profile as torch_profile

    activities = [ProfilerActivity.CPU]
    device = str(getattr(simulation, "device", "cpu"))
    cuda = device.startswith("cuda")
    if cuda:
        activities.append(ProfilerActivity.CUDA)

    synchronize(simulation)
    with torch_profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as session:
        simulation.step(steps)
        synchronize(simulation)

    sort_key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    print()
    print(f"--- top {rows} operators by {sort_key} ---")
    print(
        session.key_averages().table(
            sort_by=sort_key, row_limit=rows
        )
    )

    totals = {}
    for entry in session.key_averages():
        self_time = (
            entry.self_device_time_total
            if cuda and hasattr(entry, "self_device_time_total")
            else entry.self_cpu_time_total
        )
        totals[entry.key] = totals.get(entry.key, 0.0) + float(self_time)

    grand = sum(totals.values()) or 1.0

    def share(*fragments):
        matched = sum(
            value for key, value in totals.items()
            if any(fragment in key.lower() for fragment in fragments)
        )
        return 100.0 * matched / grand

    print()
    print("--- grouped shares of self time ---")
    print(f"eigensolve (syevj/eigh/eigvalsh):   {share('eig', 'syev'):.1f}%")
    print(f"autograd scatter (index_put/add):   {share('index_put', 'index_add', 'scatter'):.1f}%")
    print(f"gather / indexing:                  {share('index_select', 'gather', 'take'):.1f}%")
    print(f"elementwise (mul/add/exp/pow):      {share('mul', 'add', 'exp', 'pow', 'sub', 'div'):.1f}%")
    print(f"reductions (sum/max/norm):          {share('sum', 'max', 'min', 'norm', 'mean'):.1f}%")
    print(f"copies / cat / stack:               {share('copy', 'cat', 'stack', 'clone'):.1f}%")

    if export:
        session.export_chrome_trace(export)
        print()
        print(f"chrome trace written: {export}  (open in chrome://tracing)")


def main():
    parser = argparse.ArgumentParser(
        description="Profile optimised-valence dynamics cost."
    )
    parser.add_argument(
        "--physics",
        choices=("optimised-valence", "reactive"),
        default="optimised-valence",
    )
    parser.add_argument("--mixture", default="carbon rich")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--box", type=float, default=19.0)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=27000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--export", default=None, help="chrome trace path")
    parser.add_argument("--compiled-forces", action="store_true")
    parser.add_argument("--h-s2-eigvalsh-chunk-size", type=int, default=None)
    parser.add_argument("--h-transition-assembly", default=None)
    parser.add_argument(
        "--skip-profile", action="store_true",
        help="throughput only, no profiler",
    )
    options = parser.parse_args()

    print(f"physics:  {options.physics}")
    print(f"mixture:  {options.mixture}")
    print(f"width:    {options.width}   box: {options.box} A")

    simulation = build_simulation(options)
    print(f"device:   {getattr(simulation, 'device', 'unknown')}")
    print(
        f"model:    "
        f"{getattr(simulation, 'physics_model_name', 'reactive_base')}"
    )

    print(f"\nwarming up ({options.warmup} steps)...")
    simulation.step(options.warmup)
    synchronize(simulation)

    measure_throughput(simulation, options.steps, options.width)

    if not options.skip_profile:
        profile(simulation, options.steps, options.rows, options.export)


if __name__ == "__main__":
    main()
