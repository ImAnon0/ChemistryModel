"""Disposable torch.compile experiment for the ChemistryModel force path."""

import argparse
import json
import time
import types

import torch

from batched_torch import BatchedReactiveSimulation
from performance_benchmark import boxes_for, sync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--seed", type=int, default=28400)
    parser.add_argument("--mode", default="default")
    parser.add_argument("--index-select", action="store_true")
    parser.add_argument("--json", default="docs/force_compile.json")
    parser.add_argument("--max-fusion-size", type=int, default=64)
    parser.add_argument("--md-steps", type=int, default=0)
    parser.add_argument("--equivalence-steps", type=int, default=0)
    options = parser.parse_args()
    torch._inductor.config.max_fusion_size = options.max_fusion_size

    boxes, box = boxes_for("representative", options.width, options.seed)
    simulation = BatchedReactiveSimulation(
        boxes=boxes, box_size=box, time_step=0.25,
        target_temperature=800.0, friction=0.01, device="cuda",
        random_seed=options.seed,
    )
    simulation._suppress_force_diagnostic_caches = True
    simulation.experimental_index_select_gather = options.index_select
    positions = simulation.positions.detach().clone()
    reference_force, reference_energy = simulation.compute_forces()
    repeat_force, repeat_energy = simulation.compute_forces()
    eager_repeat_force_error = torch.max(torch.abs(
        repeat_force - reference_force
    )).item()

    compiled_energy = torch.compile(
        simulation.energy, backend="inductor", fullgraph=True,
        mode=options.mode,
    )

    def compiled_force(value):
        differentiable = value.detach().requires_grad_(True)
        energy = compiled_energy(differentiable)
        gradient, = torch.autograd.grad(energy, differentiable)
        return -gradient.detach(), energy.detach()
    began = time.perf_counter()
    candidate_force, candidate_energy = compiled_force(positions)
    sync(simulation.device)
    compile_seconds = time.perf_counter() - began
    force_error = torch.max(torch.abs(
        candidate_force - reference_force
    )).item()
    difference = torch.abs(candidate_force - reference_force)
    rms_force_error = torch.sqrt(torch.mean(difference ** 2)).item()
    reference_force_max = torch.max(torch.abs(reference_force)).item()
    energy_error = torch.abs(candidate_energy - reference_energy).item()

    for _ in range(3):
        compiled_force(positions)
    sync(simulation.device)
    began = time.perf_counter()
    for _ in range(options.calls):
        compiled_force(positions)
    sync(simulation.device)
    measured = time.perf_counter() - began
    result = {
        "width": options.width,
        "calls": options.calls,
        "mode": options.mode,
        "max_fusion_size": options.max_fusion_size,
        "index_select": options.index_select,
        "compile_first_call_s": compile_seconds,
        "force_calls_per_s": options.calls / measured,
        "mean_force_ms": measured * 1000.0 / options.calls,
        "max_force_error_eV_per_A": force_error,
        "rms_force_error_eV_per_A": rms_force_error,
        "reference_max_force_eV_per_A": reference_force_max,
        "relative_max_force_error": force_error / max(reference_force_max, 1e-30),
        "energy_error_eV": energy_error,
        "eager_repeat_force_error_eV_per_A": eager_repeat_force_error,
        "peak_vram_MB": torch.cuda.max_memory_allocated() / 2**20,
    }
    def compiled_compute(owner):
        force, energy = compiled_force(owner.positions)
        return force, energy

    eager_compute = simulation.compute_forces
    if options.equivalence_steps:
        initial_positions = simulation.positions.detach().clone()
        initial_velocities = simulation.velocities.detach().clone()
        initial_elapsed = simulation.elapsed_femtoseconds
        initial_steps = simulation.steps_taken
        simulation.friction = 0.0
        simulation.compute_forces = types.MethodType(compiled_compute, simulation)
        simulation.forces, simulation._potential_energy = simulation.compute_forces()
        simulation.step(options.equivalence_steps)
        sync(simulation.device)
        compiled_positions = simulation.positions.detach().clone()
        compiled_velocities = simulation.velocities.detach().clone()
        compiled_final_force = simulation.forces.detach().clone()
        compiled_final_energy = simulation._potential_energy.detach().clone()

        simulation.positions = initial_positions.clone()
        simulation.velocities = initial_velocities.clone()
        simulation.elapsed_femtoseconds = initial_elapsed
        simulation.steps_taken = initial_steps
        simulation.compute_forces = eager_compute
        simulation.reference_positions = None
        simulation.build_neighbours()
        simulation.forces, simulation._potential_energy = simulation.compute_forces()
        simulation.step(options.equivalence_steps)
        sync(simulation.device)
        result["equivalence_steps"] = options.equivalence_steps
        result["position_max_error_A"] = torch.max(torch.abs(
            compiled_positions - simulation.positions
        )).item()
        result["velocity_max_error_A_per_fs"] = torch.max(torch.abs(
            compiled_velocities - simulation.velocities
        )).item()
        result["final_force_max_error_eV_per_A"] = torch.max(torch.abs(
            compiled_final_force - simulation.forces
        )).item()
        result["final_energy_error_eV"] = torch.abs(
            compiled_final_energy - simulation._potential_energy
        ).item()

    if options.md_steps:
        simulation.compute_forces = types.MethodType(compiled_compute, simulation)
        simulation.friction = 0.01

        simulation.forces, simulation._potential_energy = (
            simulation.compute_forces()
        )
        sync(simulation.device)
        began = time.perf_counter()
        simulation.step(options.md_steps)
        sync(simulation.device)
        md_seconds = time.perf_counter() - began
        result["md_steps"] = options.md_steps
        result["md_steps_per_s"] = options.md_steps / md_seconds
        result["md_wall_s"] = md_seconds
    with open(options.json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
