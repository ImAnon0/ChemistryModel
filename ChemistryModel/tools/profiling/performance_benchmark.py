"""Reproducible, observational ChemistryModel execution benchmark.

This never changes production physics. It constructs the normal Torch engines
with fixed seeds and times stepping, device transfers, and recorder work.
"""

import argparse
import json
import platform
import os
import tempfile
import time
import tracemalloc

import numpy as np
import torch

import build_box
import mixtures
from batched_torch import BatchedReactiveSimulation
from recorder import AdaptiveRecorder, Recorder


WORKLOADS = {
    "small": ("atoms", {"C": 8, "H": 60, "N": 6, "O": 8}, 15.0),
    "representative": (*mixtures.all_mixtures()["carbon rich"], 19.0),
    "large": ("atoms", {"C": 160, "H": 400, "N": 40, "O": 60}, 24.0),
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware():
    item = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "logical_cores": __import__("os").cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        item.update({"gpu": props.name, "vram_GB": props.total_memory / 2**30})
    return item


def boxes_for(workload, width, first_seed):
    kind, contents, box = WORKLOADS[workload]
    boxes = []
    for seed in range(first_seed, first_seed + width):
        if kind == "molecules":
            symbols, positions = build_box.build(contents, box, random_seed=seed)
        else:
            symbols, positions = build_box.loose_atoms(
                contents, box, minimum_separation=1.25, random_seed=seed,
            )
        boxes.append((symbols, positions))
    return boxes, box


def benchmark(workload, width, steps, capture_every, recording, device, seed,
              index_select="none"):
    boxes, box = boxes_for(workload, width, seed)
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    tracemalloc.start()
    began = time.perf_counter()
    simulation = BatchedReactiveSimulation(
        boxes=boxes, box_size=box, time_step=0.25,
        target_temperature=800.0, friction=0.01, device=target,
        random_seed=seed,
    )
    simulation.experimental_index_select_gather = (
        True if index_select == "all" else set(index_select.split(","))
        if index_select != "none" else set()
    )
    sync(target)
    initialisation = time.perf_counter() - began
    simulation.step(20)
    sync(target)

    recorders = []
    if recording != "none":
        cls = AdaptiveRecorder if recording == "adaptive" else Recorder
        kwargs = ({"ordinary_interval_fs": capture_every * 0.25}
                  if recording == "adaptive" else {})
        recorders = [cls(simulation.symbols_for(i), box, **kwargs)
                     for i in range(width)]

    stepping = transfer = recorder_time = 0.0
    began_total = time.perf_counter()
    chunk = min(capture_every, 8) if recording == "adaptive" else capture_every
    for start in range(0, steps, chunk):
        count = min(chunk, steps - start)
        began = time.perf_counter()
        simulation.step(count)
        sync(target)
        stepping += time.perf_counter() - began
        if recording != "none":
            began = time.perf_counter()
            positions = simulation.positions_per_box
            velocities = simulation.velocities_per_box
            potentials = simulation.potential_per_box
            kinetics, temperatures = simulation.thermodynamics_per_box
            observations = (
                simulation.chemical_observations()
                if recording == "adaptive" else None
            )
            transfer += time.perf_counter() - began
            began = time.perf_counter()
            for index, recorder in enumerate(recorders):
                fn = recorder.observe if recording == "adaptive" else recorder.capture
                fn(positions[index], simulation.elapsed_femtoseconds,
                   float(potentials[index]), float(kinetics[index]),
                   float(temperatures[index]), velocities=velocities[index],
                   **({"chemical_observation": observations[index]}
                      if observations is not None else {}))
            recorder_time += time.perf_counter() - began
    sync(target)
    wall = time.perf_counter() - began_total
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    frames = sum(len(item) for item in recorders)
    save_seconds = file_bytes = 0.0
    if recorders:
        with tempfile.TemporaryDirectory() as folder:
            began = time.perf_counter()
            for index, recorder in enumerate(recorders):
                path = os.path.join(folder, f"run_{index}.npz")
                recorder.save(path)
                file_bytes += os.path.getsize(path)
            save_seconds = time.perf_counter() - began
    return {
        "workload": workload, "atoms_per_run": len(boxes[0][0]),
        "width": width, "steps_per_run": steps, "recording": recording,
        "experimental_index_select_gather": index_select,
        "initialisation_s": initialisation, "wall_s": wall,
        "aggregate_steps_per_s": steps * width / wall,
        "simulated_ps_per_wall_s": steps * width * 0.25 / 1000.0 / wall,
        "stepping_s": stepping, "capture_transfer_s": transfer,
        "recorder_cpu_s": recorder_time, "frames": frames,
        "recording_save_s": save_seconds,
        "recording_size_MB": file_bytes / 2**20,
        "python_peak_ram_MB": peak_ram / 2**20,
        "peak_vram_MB": (torch.cuda.max_memory_allocated(target) / 2**20
                         if target.type == "cuda" else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, default="representative")
    parser.add_argument("--widths", default="1,2,4,8,16")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--capture-every", type=int, default=40)
    parser.add_argument("--recording", choices=["none", "legacy", "adaptive"], default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=24000)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--index-select", default="none",
        help="experimental gather roles: none, all, or comma-separated roles",
    )
    options = parser.parse_args()
    result = {"hardware": hardware(), "benchmarks": []}
    for width in [int(value) for value in options.widths.split(",")]:
        row = benchmark(options.workload, width, options.steps,
                        options.capture_every, options.recording,
                        options.device, options.seed, options.index_select)
        result["benchmarks"].append(row)
        print(json.dumps(row, sort_keys=True))
    if options.json:
        with open(options.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
