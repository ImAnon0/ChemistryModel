# Splits a frame into its three costs so you know what to fix:
#
#   1. MD stepping      (MACE forward passes, GPU)
#   2. species counting (neighbour search + connected components, CPU)
#   3. redraw           (matplotlib)
#
# Run it from the same folder as ChemistryModel.py:
#
#   python profile_frame.py

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_atoms(box_size_angstroms=12.0, random_seed=1):
    import numpy as np

    from ase import Atoms

    composition = {"C": 10, "H": 56, "O": 14, "N": 10}

    minimum_starting_separation = 2.2
    hydrogen_mass = 3.0

    symbols = []

    for element, count in composition.items():
        symbols += [element] * count

    random_generator = np.random.default_rng(seed=random_seed)

    sites_per_side = int(np.ceil(len(symbols) ** (1.0 / 3.0)))

    lattice_spacing = box_size_angstroms / sites_per_side

    maximum_jitter = (
        lattice_spacing - minimum_starting_separation
    ) / 2.0

    lattice_sites = np.array([
        [x_index, y_index, z_index]
        for x_index in range(sites_per_side)
        for y_index in range(sites_per_side)
        for z_index in range(sites_per_side)
    ], dtype=float) * lattice_spacing

    random_generator.shuffle(lattice_sites)

    chosen_sites = lattice_sites[:len(symbols)]

    positions = chosen_sites + random_generator.uniform(
        -maximum_jitter,
        maximum_jitter,
        size=chosen_sites.shape
    )

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[box_size_angstroms] * 3,
        pbc=True
    )

    masses = atoms.get_masses()

    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "H":
            masses[index] = hydrogen_mass

    atoms.set_masses(masses)

    return atoms


def attach_calculator(atoms):
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        from mace.calculators import mace_off

        atoms.calc = mace_off(
            model="small",
            default_dtype="float32",
            device=device
        )

        return f"mace_off small on {device}"
    except ImportError:
        from ase.calculators.lj import LennardJones

        atoms.calc = LennardJones(
            sigma=2.0,
            epsilon=0.0067,
            rc=5.9
        )

        return "lennard-jones (mace not installed)"


def synchronise_gpu():
    # CUDA calls are asynchronous, so without this the timer stops
    # when the work is queued rather than when it finishes and
    # every GPU measurement comes out near zero.

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def time_it(function, repeats):
    synchronise_gpu()

    start_time = time.perf_counter()

    for _ in range(repeats):
        function()

    synchronise_gpu()

    return (time.perf_counter() - start_time) / repeats


def main():
    steps_per_frame = 20
    species_every_frames = 5

    atoms = build_atoms()

    calculator_name = attach_calculator(atoms)

    from ase_live import AseLiveSimulation

    simulation = AseLiveSimulation(
        atoms=atoms,
        target_temperature_kelvin=800.0,
        time_step_femtoseconds=1.0,
        friction_per_femtosecond=0.005
    )

    print(f"calculator: {calculator_name}")
    print(f"atoms:      {len(atoms)}")
    print()

    # Warm up: first call pays for CUDA context, kernel autotuning
    # and lazy allocation, which would otherwise swamp everything.

    simulation.step(5)
    synchronise_gpu()

    seconds_per_step = time_it(lambda: simulation.step(1), 30)

    from species import MoleculeTracker

    tracker = MoleculeTracker()

    tracker.summarise(simulation.atoms)

    seconds_per_species = time_it(
        lambda: tracker.summarise(simulation.atoms),
        20
    )

    # Redraw, measured headlessly against the Agg backend. A real
    # window is somewhat slower, so treat this as a floor.

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.0, 7.0))

    particle_axes = figure.add_subplot(1, 2, 1, projection="3d")

    from species import colours_for, sizes_for

    coordinates = simulation.positions_in_nanometers

    markers = particle_axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=sizes_for(atoms),
        c=colours_for(atoms),
        edgecolors="black",
        linewidths=0.4,
        depthshade=True
    )

    trace_axes = figure.add_subplot(1, 2, 2)
    trace_axes.plot(range(400), range(400))

    def redraw():
        markers._offsets3d = (
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2]
        )

        figure.canvas.draw()

    redraw()

    seconds_per_redraw = time_it(redraw, 10)

    plt.close(figure)

    # ---- report ---------------------------------------------

    stepping = seconds_per_step * steps_per_frame
    counting = seconds_per_species / species_every_frames
    drawing = seconds_per_redraw

    frame_total = stepping + counting + drawing

    print(f"per MD step        {seconds_per_step * 1000:8.2f} ms")
    print(f"per species count  {seconds_per_species * 1000:8.2f} ms")
    print(f"per redraw         {seconds_per_redraw * 1000:8.2f} ms")
    print()
    print(f"one frame = {steps_per_frame} steps + counting + redraw")
    print()

    for label, cost in (
        ("MD stepping", stepping),
        ("species counting", counting),
        ("redraw", drawing)
    ):
        share = 100.0 * cost / frame_total

        bar = "#" * int(round(share / 3.0))

        print(f"  {label:<18} {cost * 1000:7.1f} ms  {share:5.1f}%  {bar}")

    print()
    print(f"  {'frame total':<18} {frame_total * 1000:7.1f} ms")
    print(f"  {'frames per second':<18} {1.0 / frame_total:7.1f}")
    print(
        f"  {'simulated per second':<18} "
        f"{steps_per_frame * 1.0 / frame_total:7.1f} fs/s"
    )
    print()

    dominant = max(
        (("MD stepping", stepping),
         ("species counting", counting),
         ("redraw", drawing)),
        key=lambda pair: pair[1]
    )

    print(f"Dominant cost: {dominant[0]}.")

    if dominant[0] == "MD stepping":
        print(
            "  Lower steps_per_frame for smoother motion, or "
            "accept the frame rate. The GPU is the limit, so "
            "render tweaks will not help."
        )
    elif dominant[0] == "redraw":
        print(
            "  Set depthshade=False, cut trace_length, and "
            "update the readout text every few frames. The GPU "
            "is idle waiting on matplotlib."
        )
    else:
        print("  Raise species_every_frames.")


if __name__ == "__main__":
    main()
