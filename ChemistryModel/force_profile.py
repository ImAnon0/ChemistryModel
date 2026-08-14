"""Experimental CUDA profile for the unchanged ChemistryModel force path."""

import argparse
import json
import time

import torch
from torch.profiler import ProfilerActivity, profile

from batched_torch import BatchedReactiveSimulation
from performance_benchmark import boxes_for, sync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=28100)
    parser.add_argument("--trace", default="docs/force_cuda_trace.json")
    parser.add_argument("--json", default="docs/force_profile.json")
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--index-select", default="none")
    options = parser.parse_args()

    boxes, box = boxes_for("representative", options.width, options.seed)
    simulation = BatchedReactiveSimulation(
        boxes=boxes, box_size=box, time_step=0.25,
        target_temperature=800.0, friction=0.01, device="cuda",
        random_seed=options.seed,
    )
    simulation.experimental_index_select_gather = (
        True if options.index_select == "all"
        else set(options.index_select.split(","))
        if options.index_select != "none" else set()
    )
    for _ in range(options.warmup):
        simulation.compute_forces()
    sync(simulation.device)

    began = time.perf_counter()
    for _ in range(options.calls):
        simulation.compute_forces()
    sync(simulation.device)
    force_seconds = time.perf_counter() - began

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as measured:
        for _ in range(options.calls):
            simulation.compute_forces()
        sync(simulation.device)

    measured.export_chrome_trace(options.trace)
    rows = []
    for event in measured.key_averages():
        self_device = float(getattr(event, "self_device_time_total", 0.0))
        device_total = float(getattr(event, "device_time_total", 0.0))
        if self_device <= 0 and event.self_cpu_time_total <= 0:
            continue
        rows.append({
            "name": event.key,
            "calls": int(event.count),
            "self_cuda_us": self_device,
            "cuda_total_us": device_total,
            "self_cpu_us": float(event.self_cpu_time_total),
            "cpu_total_us": float(event.cpu_time_total),
            "cuda_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
        })
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    result = {
        "width": options.width,
        "atoms_per_box": simulation.per_box,
        "force_calls": options.calls,
        "force_calls_per_s": options.calls / force_seconds,
        "mean_force_ms": force_seconds * 1000.0 / options.calls,
        "profiler_self_cuda_ms": sum(row["self_cuda_us"] for row in rows) / 1000.0,
        "operator_rows": rows,
    }
    if options.parts:
        simulation._profile_energy_part_gradients = True
        part_positions = simulation.positions.detach().requires_grad_(True)
        simulation.energy_per_atom(part_positions)
        parts = simulation._profile_energy_parts
        part_rows = {}
        for name, values in parts.items():
            sync(simulation.device)
            began = time.perf_counter()
            torch.autograd.grad(
                torch.sum(values), part_positions,
                retain_graph=name != list(parts)[-1],
            )
            sync(simulation.device)
            part_rows[name] = (time.perf_counter() - began) * 1000.0
        result["isolated_backward_ms"] = part_rows
    with open(options.json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "operator_rows"}, indent=2))
    print(measured.key_averages().table(
        sort_by="self_device_time_total", row_limit=30
    ))


if __name__ == "__main__":
    main()
