import itertools

import numpy as np

import torch

import reactive as R
import build_box

from reactive_torch import ReactiveSimulation


# ============================================================
# Is the reactive model right?
# ============================================================
#
# Two separate questions, and both matter:
#
#   1. Does the GPU code compute the same energy and forces as
#      the readable numpy reference?
#   2. Does the model actually reproduce real molecules?
#
# The second is the one that decides whether any of this means
# anything. Bond lengths and angles are measured quantities, so
# there is a right answer to check against.


def compare_energy_and_forces(device=None):
    print("torch against numpy")
    print("-" * 60)

    generator = np.random.default_rng(3)

    symbols, positions = build_box.build(
        {"H2O": 6, "CH4": 3, "NH3": 3}, 14.0
    )

    # Shake the atoms off their ideal geometry, so the forces are
    # not all sitting at zero where a comparison proves nothing.

    positions = positions + generator.normal(
        scale=0.12, size=positions.shape
    )

    types = R.types_from_symbols(symbols)

    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=14.0,
        device=device,
        dtype=torch.float64
    )

    simulation.build_neighbours()

    torch_forces, torch_energy = simulation.compute_forces()

    torch_forces = torch_forces.cpu().numpy()

    numpy_energy = R.potential_energy(positions, types, 14.0)
    numpy_forces = R.numerical_forces(positions, types, 14.0)

    energy_error = abs(float(torch_energy) - numpy_energy)

    scale = np.max(np.abs(numpy_forces))
    force_error = np.max(np.abs(torch_forces - numpy_forces))

    print(f"device                {simulation.device}")
    print(f"atoms                 {len(symbols)}")
    print(f"energy numpy          {numpy_energy:.8f} eV")
    print(f"energy torch          {float(torch_energy):.8f} eV")
    print(f"energy difference     {energy_error:.3e}")
    print(f"largest force         {scale:.5f} eV/A")
    print(f"max force difference  {force_error:.3e}")
    print(f"relative              {force_error / scale:.3e}")

    # The numpy forces are central differences, so they carry
    # their own truncation error of order 1e-7. Anything below
    # that means the two agree as well as this test can tell.

    passed = (
        energy_error < 1e-8
        and force_error / scale < 1e-5
    )

    print()
    print("RESULT:", "match" if passed else "MISMATCH")

    return passed


def relaxed_geometry(symbols, positions, box_size=20.0,
                     device=None, steps=4000):
    # Cool the molecule until it stops moving, then measure it.

    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=np.asarray(positions) + box_size / 2.0,
        box_size=box_size,
        time_step=0.2,
        target_temperature=0.5,
        friction=0.05,
        device=device,
        dtype=torch.float64
    )

    simulation.step(steps)

    return simulation.positions_numpy


def geometry_checks(device=None):
    print()
    print("=" * 60)
    print("does it reproduce real molecules?")
    print("=" * 60)
    print()
    print("molecule  quantity        model     real     error")
    print("-" * 60)

    def bond(points, first, second):
        return float(np.linalg.norm(points[first] - points[second]))

    def angle(points, centre, first, second):
        left = points[first] - points[centre]
        right = points[second] - points[centre]

        cosine = np.dot(left, right) / (
            np.linalg.norm(left) * np.linalg.norm(right)
        )

        return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

    ok = True

    def check(label, quantity, value, expected, tolerance):
        nonlocal ok

        error = value - expected

        if abs(error) > tolerance:
            ok = False

        print(f"{label:<9} {quantity:<14} {value:8.3f} "
              f"{expected:8.3f} {error:+8.3f}")

    symbols, positions = build_box.BUILDERS["H2"]()
    points = relaxed_geometry(symbols, positions + np.array([0.1, 0, 0]),
                              device=device)
    check("H2", "bond length", bond(points, 0, 1), 0.740, 0.02)

    symbols, positions = build_box.BUILDERS["H2O"]()
    points = relaxed_geometry(symbols, positions * 1.08, device=device)
    check("H2O", "O-H length", bond(points, 0, 1), 0.960, 0.02)
    check("", "H-O-H angle", angle(points, 0, 1, 2), 104.50, 2.0)

    symbols, positions = build_box.BUILDERS["CH4"]()
    points = relaxed_geometry(symbols, positions * 1.06, device=device)

    lengths = [bond(points, 0, k) for k in range(1, 5)]
    angles = [
        angle(points, 0, i, j)
        for i, j in itertools.combinations(range(1, 5), 2)
    ]

    check("CH4", "C-H length", float(np.mean(lengths)), 1.090, 0.02)
    check("", "H-C-H angle", float(np.mean(angles)), 109.47, 2.0)

    symbols, positions = build_box.BUILDERS["NH3"]()
    points = relaxed_geometry(symbols, positions * 1.06, device=device)

    lengths = [bond(points, 0, k) for k in range(1, 4)]
    angles = [
        angle(points, 0, i, j)
        for i, j in itertools.combinations(range(1, 4), 2)
    ]

    check("NH3", "N-H length", float(np.mean(lengths)), 1.010, 0.02)
    check("", "H-N-H angle", float(np.mean(angles)), 107.00, 2.5)

    symbols = ["C", "O", "O"]
    positions = np.array([
        [0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [-1.2, 0.35, 0.0]
    ])
    points = relaxed_geometry(symbols, positions, device=device)

    check("CO2", "C-O length", bond(points, 0, 1), 1.200, 0.05)
    check("", "O-C-O angle", angle(points, 0, 1, 2), 180.0, 3.0)

    symbols = ["C", "C", "H", "H", "H", "H"]
    positions = np.array([
        [0.0, 0.0, 0.0], [1.35, 0.0, 0.0],
        [-0.6, 0.95, 0.0], [-0.6, -0.95, 0.0],
        [1.95, 0.95, 0.0], [1.95, -0.95, 0.0]
    ])
    points = relaxed_geometry(symbols, positions, device=device)

    check("C2H4", "C=C length", bond(points, 0, 1), 1.340, 0.05)

    symbols = ["N", "N"]
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    points = relaxed_geometry(symbols, positions, device=device)

    check("N2", "triple bond", bond(points, 0, 1), 1.100, 0.05)

    print()
    print("RESULT:", "geometry correct" if ok else "GEOMETRY WRONG")

    return ok


def reaction_check(device=None):
    # Loose hydrogen and oxygen atoms, warm, left alone. If the
    # model works at all, water and hydrogen should appear
    # without anything being told to make them.

    print()
    print("=" * 60)
    print("do molecules form on their own?")
    print("=" * 60)

    symbols, positions = build_box.loose_atoms(
        {"H": 24, "O": 12}, 16.0, random_seed=1
    )

    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=16.0,
        time_step=0.25,
        target_temperature=400.0,
        friction=0.02,
        device=device
    )

    print(f"start: {simulation.molecule_formulas()}")

    for block in range(6):
        simulation.step(400)

        formulas = simulation.molecule_formulas()

        print(f"{simulation.elapsed_femtoseconds:7.0f} fs  "
              f"T {simulation.temperature:6.1f} K  "
              f"PE {simulation.potential_energy:9.2f} eV  "
              f"{formulas}")

    final = simulation.molecule_formulas()

    made_something = any(
        len(formula) > 2 for formula in final
    )

    print()
    print("RESULT:", "molecules formed" if made_something
          else "NOTHING FORMED")

    return made_something


def barrier_check(device=None):
    # The one number that sets every activation barrier in the
    # model, checked against the reaction it was fitted to.
    #
    # H + H2 -> H2 + H goes through a linear symmetric H3. Its
    # energy above an isolated H2 is the barrier, measured at
    # about 0.42 eV.

    print()
    print("=" * 60)
    print("activation barrier")
    print("=" * 60)

    from scipy.optimize import minimize_scalar

    types_two = R.types_from_symbols(["H", "H"])
    types_three = R.types_from_symbols(["H", "H", "H"])

    hydrogen = R.potential_energy(
        np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        types_two
    )

    def saddle(spacing):
        return R.potential_energy(
            np.array([
                [-spacing, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [spacing, 0.0, 0.0]
            ]),
            types_three
        )

    result = minimize_scalar(
        saddle, bounds=(0.6, 1.6), method="bounded"
    )

    barrier = result.fun - hydrogen

    print(f"barrier constant       {R.OVER_COORDINATION_PENALTY:.3f}")
    print(f"H + H2 barrier         {barrier:.3f} eV   "
          f"(measured 0.42)")
    print(f"H-H at the saddle      {result.x:.3f} A     "
          f"(measured 0.93)")

    # With the penalty switched off the transition state is more
    # stable than the reactants, which is a well rather than a
    # barrier and lets every reaction run with no activation at
    # all. Worth showing, since it is the whole reason the term
    # exists.

    saved = R.OVER_COORDINATION_PENALTY
    R.OVER_COORDINATION_PENALTY = 0.0

    without = minimize_scalar(
        saddle, bounds=(0.6, 1.6), method="bounded"
    ).fun - hydrogen

    R.OVER_COORDINATION_PENALTY = saved

    print(f"same, penalty off      {without:.3f} eV   "
          f"(a well, not a barrier)")

    passed = 0.3 < barrier < 0.55

    print()
    print("RESULT:", "barrier in range" if passed else "BARRIER WRONG")

    return passed


if __name__ == "__main__":
    ok = compare_energy_and_forces(device="cpu")

    if ok and torch.cuda.is_available():
        print()
        ok = compare_energy_and_forces(device="cuda")

    if ok:
        geometry_checks()
        barrier_check()
        reaction_check()