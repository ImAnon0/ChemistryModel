"""Core regressions for transferable reactive bond and angle behaviour."""

import numpy as np
import torch

import reactive as R
from batched_torch import BatchedReactiveSimulation, compare_against_single
from reactive_torch import ReactiveSimulation


BOX = 12.0
DTYPE = torch.float64

# Seed 1 at the historical CH3 + CH3 turning point. Before partial-angle
# engagement was corrected, the angle term exerted +15.5 eV/A outward here
# against -5.1 eV/A from the attractive C-C bond.
METHYL_SYMBOLS = ["H", "H", "C", "H", "H", "H", "C", "H"]
METHYL_TURNING_POINT = np.array([
    [4.8900265694, 7.2423601151, 6.3958153725],
    [6.5356173515, 6.8181324005, 7.4496626854],
    [6.0014257431, 7.3813414574, 6.6287102699],
    [6.7037630081, 7.2824635506, 5.7992815971],
    [5.5814838409, 5.5461163521, 4.6387181282],
    [5.1272864342, 5.8501877785, 6.5311083794],
    [5.7508611679, 5.2390422821, 5.7055230141],
    [5.2037549019, 4.3015189171, 5.9019498825],
], dtype=float)


def radial_term_forces(symbols, positions, pair):
    """Return relative radial force by energy term; positive is outward."""
    simulation = ReactiveSimulation(
        symbols=symbols,
        positions=positions,
        box_size=BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=DTYPE,
    )
    delta = positions[pair[1]] - positions[pair[0]]
    delta -= BOX * np.round(delta / BOX)
    unit = delta / np.linalg.norm(delta)
    epsilon = 1e-4
    samples = []
    for sign in (+1.0, -1.0):
        shifted = positions.copy()
        shifted[pair[1]] += sign * 0.5 * epsilon * unit
        shifted[pair[0]] -= sign * 0.5 * epsilon * unit
        simulation.positions = torch.tensor(shifted, dtype=DTYPE)
        simulation.build_neighbours()
        with torch.no_grad():
            simulation.energy_per_atom(simulation.positions)
        samples.append({
            name: float(value.sum())
            for name, value in simulation._energy_parts.items()
        })
    return {
        name: -2.0 * (samples[0][name] - samples[1][name]) / (2.0 * epsilon)
        for name in samples[0]
    }


def test_partial_cc_angle_does_not_recreate_capture_wall():
    parts = radial_term_forces(
        METHYL_SYMBOLS, METHYL_TURNING_POINT, pair=(2, 6)
    )
    total = sum(parts.values())
    assert parts["bond"] < 0.0, "C-C bond force is not attractive"
    assert total < 2.0, (
        "partial H-C-C angles recreate the methyl capture wall: "
        f"radial force is {total:+.2f} eV/A"
    )


def test_settled_methyl_geometry_is_unchanged_between_numpy_and_torch():
    # All active C-H tapers are one, so the engagement multiplier is exactly
    # one and the revised rule must reduce to the established angle model.
    symbols = ["C", "H", "H", "H"]
    positions = np.array([
        [6.0, 6.0, 6.0],
        [7.09, 6.0, 6.0],
        [5.6367, 7.0277, 6.0],
        [5.6367, 5.4862, 6.8900],
    ])
    types = R.types_from_symbols(symbols)
    numpy_parts = R.potential_energy(
        positions, types, BOX, return_parts=True
    )
    simulation = ReactiveSimulation(
        symbols, positions, BOX, random_seed=0, relax_on_start=False,
        device="cpu", dtype=DTYPE,
    )
    with torch.no_grad():
        simulation.energy_per_atom(simulation.positions)
    torch_angle = float(simulation._energy_parts["angle"].sum())
    assert abs(numpy_parts["angle"] - torch_angle) < 1e-10


def test_batched_matches_single_after_angle_change():
    shifted = METHYL_TURNING_POINT + np.array([0.03, -0.02, 0.01])
    boxes = [
        (METHYL_SYMBOLS, METHYL_TURNING_POINT),
        (METHYL_SYMBOLS, shifted),
    ]
    report = compare_against_single(boxes, BOX, seed=0)
    for row in report:
        assert row["force_error"] < 1e-7
        assert row["energy_error"] < 1e-7


def test_batched_reporting_reuses_current_energy_and_invalidates_safely():
    boxes = [
        (METHYL_SYMBOLS, METHYL_TURNING_POINT),
        (METHYL_SYMBOLS, METHYL_TURNING_POINT + 0.02),
    ]
    simulation = BatchedReactiveSimulation(
        boxes=boxes, box_size=BOX, random_seed=0,
        relax_on_start=False, device="cpu", dtype=DTYPE,
    )

    expected = simulation.potential_per_box.copy()
    original_energy = simulation.energy_per_atom

    def unexpected_recalculation(_positions):
        raise AssertionError("current potential was recalculated")

    simulation.energy_per_atom = unexpected_recalculation
    assert np.array_equal(simulation.potential_per_box, expected)

    simulation.energy_per_atom = original_energy
    changed = simulation.positions.clone()
    changed[0, 0] += 0.01
    simulation.positions = changed
    refreshed = simulation.potential_per_box
    with torch.no_grad():
        direct = (
            original_energy(simulation.positions)
            .reshape(simulation.box_count, simulation.per_box)
            .sum(dim=1).cpu().numpy()
        )
    assert np.array_equal(refreshed, direct)

    assert np.array_equal(
        simulation.positions_per_box[0], simulation.positions_for(0)
    )
    assert np.array_equal(
        simulation.velocities_per_box[1], simulation.velocities_for(1)
    )


def test_numpy_bond_order_is_symmetric_and_bounded():
    rng = np.random.default_rng(12345)
    for count in range(2, 12):
        taper = rng.random((count, count))
        taper = 0.5 * (taper + taper.T)
        taper[taper < 0.2] = 0.0
        np.fill_diagonal(taper, 0.0)
        types = rng.integers(0, len(R.ELEMENTS), size=count)
        order, coordination = R.bond_orders(taper, types)
        assert np.allclose(order, order.T)
        assert np.all((order >= 0.0) & (order <= 3.0))
        assert np.allclose(coordination, taper.sum(axis=1))


if __name__ == "__main__":
    tests = [
        test_partial_cc_angle_does_not_recreate_capture_wall,
        test_settled_methyl_geometry_is_unchanged_between_numpy_and_torch,
        test_batched_matches_single_after_angle_change,
        test_batched_reporting_reuses_current_energy_and_invalidates_safely,
        test_numpy_bond_order_is_symmetric_and_bounded,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as problem:
            failures += 1
            print(f"FAIL  {test.__name__}\n      {problem}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
