import time

import numpy as np

import torch

from atom_numpy import AtomSimulation
from atom_torch import TorchAtomSimulation


# ============================================================
# Does the GPU atom code agree with the CPU reference?
# ============================================================


def compare(cells=5, device=None):
    reference = AtomSimulation(
        unit_cells_per_side=cells,
        number_density=0.85
    )

    candidate = TorchAtomSimulation(
        unit_cells_per_side=cells,
        number_density=0.85,
        device=device,
        dtype=torch.float64
    )

    # A perfect lattice has every force cancelling by symmetry,
    # so comparing there tests almost nothing and the relative
    # error divides machine noise by machine noise. Shake the
    # atoms off their sites first.

    generator = np.random.default_rng(0)

    positions = reference.positions.copy()
    positions += generator.normal(scale=0.08, size=positions.shape)
    positions %= reference.box_size

    reference.positions = positions
    reference.reference_positions = None

    candidate.positions = torch.tensor(
        positions,
        device=candidate.device,
        dtype=candidate.dtype
    )

    candidate.build_pairs(force=True)
    reference.build_pairs(force=True)

    torch_forces, torch_energy = candidate.compute_forces()
    numpy_forces, numpy_energy = reference.compute_forces()

    torch_forces = torch_forces.detach().cpu().numpy()

    scale = np.max(np.abs(numpy_forces))
    error = np.max(np.abs(torch_forces - numpy_forces))

    energy_error = abs(float(torch_energy) - float(numpy_energy))
    virial_error = abs(candidate._virial - reference._virial)

    print(f"device                {candidate.device}")
    print(f"atoms                 {reference.particle_count}")
    print(f"pairs numpy / torch   "
          f"{len(reference.pair_first)} / {candidate.pair_first.numel()}")
    print(f"largest force         {scale:.6f}")
    print(f"max force difference  {error:.3e}")
    print(f"relative              {error / scale:.3e}")
    print(f"energy difference     {energy_error:.3e}")
    print(f"virial difference     {virial_error:.3e}")

    passed = (
        error / scale < 1e-8
        and energy_error / abs(float(numpy_energy)) < 1e-8
    )

    print()
    print("RESULT:", "match" if passed else "MISMATCH")

    return passed


def physics_checks(device=None):
    print()
    print("=" * 60)
    print("physics sanity checks")
    print("=" * 60)

    # 1. Energy conservation with the thermostat off.

    simulation = TorchAtomSimulation(
        unit_cells_per_side=5,
        time_step=0.002,
        device=device,
        dtype=torch.float64
    )

    simulation.thermostat_is_on = False

    start_energy = simulation.total_energy
    simulation.step(500)

    drift = abs(
        simulation.total_energy - start_energy
    ) / abs(start_energy)

    print(f"energy drift, 500 steps at dt=0.002:  {drift:.2e}"
          f"   {'ok' if drift < 1e-3 else 'TOO LARGE'}")

    # 2. The starting crystal should show a sharp first shell at
    #    the face centred cubic nearest neighbour distance.

    simulation = TorchAtomSimulation(
        unit_cells_per_side=6,
        number_density=0.85,
        device=device
    )

    radius, g = simulation.radial_distribution()

    cell = (4.0 / 0.85) ** (1.0 / 3.0)
    expected = cell / np.sqrt(2.0)

    found = radius[int(np.argmax(g))]

    print(f"g(r) first peak:  found {found:.3f}, "
          f"expected {expected:.3f}"
          f"   {'ok' if abs(found - expected) < 0.06 else 'WRONG'}")

    # 3. Phases should separate by how far atoms travel.

    print()
    print("phase              T*      P        D         verdict")
    print("-" * 56)

    results = {}

    for name, density, temperature in [
        ("solid", 1.05, 0.40),
        ("liquid", 0.85, 1.10),
        ("gas", 0.05, 1.50)
    ]:
        simulation = TorchAtomSimulation(
            unit_cells_per_side=6,
            number_density=density,
            target_temperature=temperature,
            time_step=0.002,
            device=device
        )

        simulation.step(1500)
        simulation.reset_msd()
        simulation.step(2500)

        diffusion = simulation.diffusion_coefficient

        results[name] = diffusion

        verdict = (
            "frozen" if diffusion < 0.005
            else "flowing" if diffusion < 0.5
            else "free"
        )

        print(f"{name:<16} {simulation.temperature:5.2f} "
              f"{simulation.pressure:8.3f} {diffusion:9.5f}   {verdict}")

    ordered = (
        results["solid"] < results["liquid"] < results["gas"]
    )

    print()
    print("RESULT:", "phases separate correctly" if ordered
          else "PHASES NOT SEPARATING")

    return ordered


def benchmark(cells=10, steps=200):
    print()
    print("=" * 60)
    print(f"benchmark: {4 * cells ** 3} atoms")
    print("=" * 60)

    cpu = AtomSimulation(unit_cells_per_side=cells)

    start = time.time()
    cpu.step(steps)
    cpu_seconds = time.time() - start

    print(f"numpy CPU   {steps / cpu_seconds:8.1f} steps/s")

    if not torch.cuda.is_available():
        print("no CUDA device visible, skipping GPU benchmark")
        return

    gpu = TorchAtomSimulation(unit_cells_per_side=cells)

    gpu.step(20)
    torch.cuda.synchronize()

    start = time.time()
    gpu.step(steps)
    torch.cuda.synchronize()
    gpu_seconds = time.time() - start

    print(f"torch GPU   {steps / gpu_seconds:8.1f} steps/s")
    print(f"speedup     {cpu_seconds / gpu_seconds:8.1f}x")


if __name__ == "__main__":
    print("Checking torch against numpy on CPU first")
    print("-" * 60)

    ok = compare(device="cpu")

    if ok and torch.cuda.is_available():
        print()
        print("Now on the GPU")
        print("-" * 60)
        ok = compare(device="cuda")

    if ok:
        physics_checks()
        benchmark()
