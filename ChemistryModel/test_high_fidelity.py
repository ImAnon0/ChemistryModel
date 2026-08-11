"""Regression tests for the high-fidelity H-transfer correction.

Run from inside the ChemistryModel package directory:

    python -m pytest test_high_fidelity.py -v
    python test_high_fidelity.py            # same checks, no pytest needed

These are physics-invariant tests, not value tests.  They do not assert any
particular barrier height, so they stay valid across v3, v4 and beyond.  What
they assert is that the correction cannot break the things a potential must
never break: smoothness, the base limit, valence, and energy conservation.

Everything runs on CPU in float64.  float32 hides step discontinuities of the
size we care about, so the dtype is not incidental.
"""

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from high_fidelity_torch import HighFidelityBatchedReactiveSimulation


BOX = 20.0
CENTRE = np.array([BOX / 2, BOX / 2, BOX / 2])

COMMON = dict(
    box_size=BOX,
    random_seed=0,
    relax_on_start=False,
    device="cpu",
    dtype=torch.float64,
)


def make(symbols, positions, high_fidelity=True):
    """Build a one-box simulation around the centre of a roomy cell."""
    cls = (
        HighFidelityBatchedReactiveSimulation if high_fidelity
        else BatchedReactiveSimulation
    )
    return cls(boxes=[(list(symbols), np.asarray(positions, float) + CENTRE)],
               **COMMON)


def energy_at(sim, positions):
    """Total potential energy at a geometry, with the neighbour table rebuilt.

    The table is built once at construction with a skin, so moving atoms by
    hand without rebuilding silently drops pairs that have wandered in.
    """
    sim.positions = torch.tensor(
        np.asarray(positions, float) + CENTRE,
        device=sim.device, dtype=sim.dtype,
    )
    sim.build_neighbours()
    with torch.no_grad():
        return float(torch.sum(sim.energy_per_atom(sim.positions)))


def collinear(donor, acceptor, separation, x):
    """donor ... H ... acceptor along a line, H at distance x from the donor."""
    return (
        [donor, "H", acceptor],
        np.array([[0, 0, 0], [x, 0, 0], [separation, 0, 0]], float),
    )


def worst_step(values, positions):
    """Largest step relative to its immediate neighbours.

    A smooth curve sampled finely has each step close to the two beside it.
    A genuine discontinuity puts one step far above both, which is what this
    returns, along with where it happened.
    """
    steps = np.diff(values)
    worst, where = 0.0, None
    for i in range(1, len(steps) - 1):
        local = 0.5 * (abs(steps[i - 1]) + abs(steps[i + 1]))
        excess = abs(steps[i]) - 3.0 * local
        if excess > worst:
            worst, where = excess, positions[i]
    return worst, where


# ----------------------------------------------------------------------
# 1.  A hydrogen with one partner must be untouched.
# ----------------------------------------------------------------------

def test_single_contact_matches_base_exactly():
    """H2 and an isolated hydroxyl must land on the base potential.

    The correction is documented as exactly zero when hydrogen has only one
    partner.  This is the cheapest way to catch a gating regression, and it
    should hold to machine precision rather than to a tolerance.
    """
    cases = [
        (["H", "H"], [[0, 0, 0], [0.74, 0, 0]]),
        (["O", "H"], [[0, 0, 0], [0.96, 0, 0]]),
        (["C", "H"], [[0, 0, 0], [1.09, 0, 0]]),
    ]

    for symbols, positions in cases:
        high = make(symbols, positions, high_fidelity=True)
        base = make(symbols, positions, high_fidelity=False)

        difference = abs(
            energy_at(high, positions) - energy_at(base, positions)
        )
        assert difference < 1e-10, (
            f"{''.join(symbols)}: correction fired on a one-bond hydrogen, "
            f"difference {difference:.3e} eV"
        )


# ----------------------------------------------------------------------
# 2.  No step discontinuities anywhere along a transfer.
# ----------------------------------------------------------------------

def test_transfer_is_continuous():
    """Walk a hydrogen across the transfer at several donor-acceptor gaps.

    The donor and competitor are chosen by argmax, so their labels swap
    partway across.  Everything downstream of that choice must be symmetric
    under the swap or the energy takes a step.  The 2.9-3.1 A range is
    deliberate: that is ordinary hydrogen-bond separation, where the swap
    lands inside the engagement window rather than safely outside it.
    """
    for donor, acceptor in [("O", "C"), ("O", "N"), ("N", "C"), ("O", "O")]:
        for separation in np.arange(2.60, 3.30, 0.05):
            symbols, _ = collinear(donor, acceptor, separation, 1.0)
            sim = make(symbols, collinear(donor, acceptor, separation, 1.0)[1])

            xs = np.arange(0.95, separation - 0.95, 0.0005)
            values = [
                energy_at(sim, collinear(donor, acceptor, separation, x)[1])
                for x in xs
            ]

            excess, where = worst_step(np.array(values), xs)
            assert excess < 2e-3, (
                f"{donor}...H...{acceptor} at {separation:.2f} A: energy step "
                f"of {excess:.4f} eV at x = {where:.4f} A"
            )


# ----------------------------------------------------------------------
# 3.  A third contact must still be resisted.
# ----------------------------------------------------------------------

def test_third_contact_is_not_bound():
    """Hydrogen must not find a stable well with three partners.

    The over-coordination term is computed from the total coordination, so a
    correction that removes all of it while replacing only two contacts hands
    the third a free unpenalised bond.  That is the three-centre product the
    whole module exists to prevent, so it is worth testing directly rather
    than trusting the construction.
    """
    def geometry(third):
        return np.array([
            [0, 0, 0],          # O donor
            [1.05, 0, 0],       # the hydrogen
            [2.30, 0, 0],       # C acceptor
            [1.05, third, 0],   # a second O closing in from above
        ], float)

    symbols = ["O", "H", "C", "O"]
    sim = make(symbols, geometry(2.20))

    far = energy_at(sim, geometry(2.20))
    closest = min(
        energy_at(sim, geometry(third))
        for third in np.arange(0.90, 1.55, 0.01)
    )

    assert closest > far - 0.30, (
        "a third contact opens a bound three-centre well: energy drops "
        f"{far - closest:.2f} eV below the two-contact geometry"
    )


# ----------------------------------------------------------------------
# 4.  The barrier must be a saddle, not a wall.
# ----------------------------------------------------------------------

def test_saddle_is_early_enough_to_be_reachable():
    """The col must open while the donor bond is still near equilibrium.

    Height alone does not decide whether a transfer happens.  A collision is
    fast compared with the C-H stretching mode, so the incoming atom arrives
    with the donor bond close to 1.09 A.  If the surface only opens a route
    after a large extension, no collision energy finds one, and the symptom
    is a bounce rather than a slow reaction.

    The 0.20 A ceiling is a judgement, not a measurement: real abstraction
    transition states sit around a tenth of an angstrom of donor stretch.
    Move it if you have a better number, but move it deliberately.
    """
    from hf_surface_scan import (
        basin_seeds, build, flood_saddle, surface,
    )

    donor_lengths = np.arange(1.00, 1.90, 0.02)
    transfer_lengths = np.arange(0.65, 1.60, 0.005)

    sim = build("high_fidelity")
    grid = surface(sim, donor_lengths, transfer_lengths)

    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    assert reactant is not None, "could not locate both basins"

    cell, saddle = flood_saddle(grid, reactant, product)
    assert cell is not None, "no connected route between reactant and product"

    barrier = saddle - grid[reactant]
    stretch = donor_lengths[cell[0]] - donor_lengths[reactant[0]]

    assert stretch < 0.20, (
        f"the col only opens after {stretch:.3f} A of C-H extension, so a "
        "thermal collision cannot reach it"
    )
    assert 0.05 < barrier < 1.5, (
        f"activation energy {barrier:.3f} eV is outside any plausible range "
        "for hydrogen abstraction"
    )


# ----------------------------------------------------------------------
# 5.  Energy must be conserved with the thermostat off.
# ----------------------------------------------------------------------

def test_gate_forces_are_continuous():
    """Forces either side of equal contact, not just energies.

    The gate takes a minimum of the two contact tapers, and a bare minimum is
    continuous in value but not in slope: at the point where the two are
    equal the derivative flips. The energy scan cannot see that, because the
    energy itself never jumps. Only the force does, and the force is what the
    integrator uses.

    So this walks a hydrogen through the geometry where the two tapers cross
    and compares autograd forces on either side. A kink shows up as a large
    change in the force between adjacent samples while the neighbouring
    changes stay small.
    """
    donor, acceptor, separation = "O", "C", 2.95

    symbols, start = collinear(donor, acceptor, separation, 1.0)
    sim = make(symbols, start)

    positions = np.arange(1.20, 1.60, 0.002)
    forces = []

    for x in positions:
        moved = torch.tensor(
            collinear(donor, acceptor, separation, x)[1] + CENTRE,
            device=sim.device, dtype=sim.dtype, requires_grad=True,
        )
        sim.positions = moved.detach()
        sim.build_neighbours()

        energy = torch.sum(sim.energy_per_atom(moved))
        gradient, = torch.autograd.grad(energy, moved)

        # Force on the transferring hydrogen along the transfer axis.
        forces.append(float(-gradient[1, 0]))

    forces = np.array(forces)
    steps = np.diff(forces)

    worst, where = 0.0, None
    for i in range(1, len(steps) - 1):
        local = 0.5 * (abs(steps[i - 1]) + abs(steps[i + 1]))
        excess = abs(steps[i]) - 3.0 * local
        if excess > worst:
            worst, where = excess, positions[i]

    assert worst < 0.5, (
        f"force kink of {worst:.3f} eV/A at x = {where:.4f} A, so the gate "
        "is continuous in energy but not in slope"
    )


def test_energy_conserved_without_thermostat():
    """The integrator test that catches everything the scans miss.

    Any discontinuity the geometry scans did not happen to cross still shows
    up here as drift.  capped_steps is checked as well: the 0.15 A move limit
    will quietly absorb a force spike and turn a crash into a slow energy
    leak, so a nonzero count means the potential misbehaved even if the run
    looked stable.
    """
    symbols, positions = formaldehyde_plus_hydrogen(1.60)
    sim = make(symbols, positions)

    sim.thermostat_is_on = False
    sim.set_temperature(300.0)

    start = sim.total_energy
    sim.step(400)
    drift = abs(sim.total_energy - start)

    assert sim.capped_steps == 0, (
        f"{sim.capped_steps} steps hit the move limit, so a force spike was "
        "absorbed rather than integrated"
    )
    assert drift < 0.05, f"total energy drifted {drift:.3f} eV over 400 steps"


def formaldehyde_plus_hydrogen(approach):
    """H2C=O with a hydrogen approaching one C-H collinearly."""
    direction = np.array([
        np.sin(np.radians(122)), np.cos(np.radians(122)), 0.0
    ])
    mirrored = np.array([-direction[0], direction[1], 0.0])

    carbon = np.zeros(3)
    oxygen = np.array([0.0, 1.20, 0.0])
    donor_h = 1.09 * direction
    other_h = 1.09 * mirrored
    incoming = donor_h + approach * direction

    return (
        ["C", "O", "H", "H", "H"],
        np.array([carbon, oxygen, donor_h, other_h, incoming]),
    )


if __name__ == "__main__":
    tests = [
        test_single_contact_matches_base_exactly,
        test_transfer_is_continuous,
        test_third_contact_is_not_bound,
        test_saddle_is_early_enough_to_be_reachable,
        test_gate_forces_are_continuous,
        test_energy_conserved_without_thermostat,
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