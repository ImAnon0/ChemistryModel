import time

import numpy as np

import torch

from membrane_sim import MembraneSimulation
from membrane_torch import TorchMembraneSimulation


# ============================================================
# Does the GPU version agree with the CPU version?
# ============================================================
#
# The numpy implementation is the one that was actually tested,
# so it is treated as the reference. This script puts identical
# positions into both, computes forces both ways, and compares.
#
# Run this once before trusting membrane_torch.py for anything.


def compare(number_of_lipids=300, box_size=20.0, device=None):
    reference = MembraneSimulation(
        number_of_lipids=number_of_lipids,
        box_size=box_size
    )

    candidate = TorchMembraneSimulation(
        number_of_lipids=number_of_lipids,
        box_size=box_size,
        skin_distance=0.5,
        device=device,
        dtype=torch.float64
    )

    # Force both to look at exactly the same configuration.

    positions = reference.positions.copy()

    candidate.positions = torch.tensor(
        positions,
        device=candidate.device,
        dtype=candidate.dtype
    )

    candidate.build_pairs(force=True)
    reference._update_pairs(force=True)

    torch_forces, torch_energy = candidate.compute_forces()

    numpy_forces, numpy_energy = reference.model.forces_and_energy(
        positions,
        reference.pair_first,
        reference.pair_second
    )

    torch_forces = torch_forces.detach().cpu().numpy()

    pair_difference = abs(
        len(candidate.pair_first) - len(reference.pair_first)
    )

    force_error = np.max(np.abs(torch_forces - numpy_forces))

    force_scale = np.max(np.abs(numpy_forces))

    energy_error = abs(
        float(torch_energy) - float(numpy_energy)
    )

    print(f"device                {candidate.device}")
    print(f"beads                 {reference.model.bead_count}")
    print(f"pairs numpy / torch   "
          f"{len(reference.pair_first)} / {len(candidate.pair_first)}"
          f"  (difference {pair_difference})")
    print(f"largest force         {force_scale:.6f}")
    print(f"max force difference  {force_error:.3e}")
    print(f"relative              {force_error / force_scale:.3e}")
    print(f"energy numpy          {float(numpy_energy):.6f}")
    print(f"energy torch          {float(torch_energy):.6f}")
    print(f"energy difference     {energy_error:.3e}")

    passed = (
        force_error / force_scale < 1e-8
        and energy_error / abs(float(numpy_energy)) < 1e-8
    )

    print()
    print("RESULT:", "match" if passed else "MISMATCH")

    if not passed:
        print()
        print("Do not use the torch version until this passes.")
        print("Worst offending beads:")

        per_bead = np.max(
            np.abs(torch_forces - numpy_forces),
            axis=1
        )

        worst = np.argsort(per_bead)[-5:][::-1]

        for index in worst:
            print(
                f"  bead {index:5d}  "
                f"type {reference.model.bead_types[index]}  "
                f"error {per_bead[index]:.3e}"
            )

    return passed


def benchmark(number_of_lipids=2000, box_size=45.0, steps=300):
    print()
    print("=" * 50)
    print(f"benchmark: {number_of_lipids} lipids")
    print("=" * 50)

    cpu_sim = MembraneSimulation(
        number_of_lipids=number_of_lipids,
        box_size=box_size
    )

    start = time.time()
    cpu_sim.step(steps)
    cpu_seconds = time.time() - start

    print(f"numpy CPU   {steps / cpu_seconds:8.1f} steps/s")

    if not torch.cuda.is_available():
        print("no CUDA device visible, skipping GPU benchmark")
        return

    gpu_sim = TorchMembraneSimulation(
        number_of_lipids=number_of_lipids,
        box_size=box_size,
        dtype=torch.float32
    )

    # Warm up: first CUDA call includes kernel compilation.

    gpu_sim.step(20)
    torch.cuda.synchronize()

    start = time.time()
    gpu_sim.step(steps)
    torch.cuda.synchronize()
    gpu_seconds = time.time() - start

    print(f"torch GPU   {steps / gpu_seconds:8.1f} steps/s")
    print(f"speedup     {cpu_seconds / gpu_seconds:8.1f}x")


def test_topology(device=None):
    # Growth, puncture and detergent all change the number or
    # type of beads, which means rebuilding every tensor on the
    # device. Nothing here checks physics against numpy; it
    # checks that the simulation survives each operation and
    # stays at a sane temperature afterwards.

    print()
    print("=" * 50)
    print("topology operations on the torch backend")
    print("=" * 50)

    simulation = TorchMembraneSimulation(
        number_of_lipids=400,
        box_size=26.0,
        start="single vesicle",
        device=device
    )

    def report(label):
        largest, clusters = simulation.cluster_report()

        temperature = simulation.temperature_kelvin

        energy = (
            simulation.potential_energy
            / max(simulation.model.number_of_lipids, 1)
        )

        sane = (
            np.isfinite(temperature)
            and temperature < 10.0
            and np.isfinite(energy)
        )

        print(
            f"{label:<22} lipids {simulation.model.number_of_lipids:5d}"
            f"  T {temperature:7.3f}"
            f"  PE/lipid {energy:8.2f}"
            f"  largest {largest:5d}"
            f"  {'ok' if sane else 'BLEW UP'}"
        )

        return sane

    simulation.step(300)
    healthy = report("baseline")

    simulation.add_lipids(40, outer_fraction=0.9)
    simulation.step(300)
    healthy &= report("after growth")

    simulation.growth_interval = 100
    simulation.growth_batch = 5
    simulation.step(600)
    simulation.growth_interval = 0
    healthy &= report("after auto-growth")

    simulation.puncture(radius=3.5)
    simulation.step(300)
    healthy &= report("after puncture")

    simulation.add_detergent(fraction=0.3)
    simulation.step(300)
    healthy &= report("after detergent")

    print()
    print("RESULT:", "all operations stable" if healthy else "UNSTABLE")

    return healthy


if __name__ == "__main__":
    print("Checking torch against numpy on CPU first")
    print("-" * 50)

    ok = compare(device="cpu")

    if ok and torch.cuda.is_available():
        print()
        print("Now checking on the GPU")
        print("-" * 50)
        ok = compare(device="cuda")

    if ok:
        benchmark()
        test_topology()