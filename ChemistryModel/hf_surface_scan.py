"""Map the H-transfer surface in the two distances that matter.

    py hf_surface_scan.py                       # frozen spectator geometry
    py hf_surface_scan.py --relax               # relaxed, slower, honest
    py hf_surface_scan.py --physics base
    py hf_surface_scan.py --relax --plot v3_relaxed.png

The reaction coordinate for an abstraction is two dimensional:

    r(C-H)   the donor bond, breaking
    r(H-H)   the new bond, forming

Reading the numbers
-------------------
The activation energy is quoted against the *global reactant minimum*: the
relaxed molecule with the incoming hydrogen detached.  Per-slice references
are misleading, because reaching a stretched r(C-H) costs real energy that a
per-slice reference quietly subtracts, making a late barrier look free.

The saddle is found by flooding rather than by walking a path.  Cells are
added in energy order and the saddle is the energy at which the reactant and
product basins first join.  That is the true minimax over every route across
the grid, so it cannot be fooled by a path that happens to be sampled badly.

What to look for
----------------
Two minima, near (r_CH ~ 1.1, r_HH detached) and (r_CH detached, r_HH ~ 0.74),
joined by one saddle.  Two things about that saddle matter, and they are
independent:

  height  - compare against the physical activation scale for the reaction
  timing  - how far r(C-H) has already stretched at the saddle

A barrier of sensible height that only opens after a large C-H extension is
still unreactive in dynamics.  A collision is fast compared with the C-H
stretching mode, so the incoming atom arrives while the donor bond is near
equilibrium; if there is no col at that r(C-H), no collision energy finds one.

Why --relax exists
------------------
Frozen, the scan holds C=O, the spectator C-H and every angle at their
formaldehyde values for the whole reaction.  But the carbon rehybridises as
the bond breaks, and denying it that relaxation raises the late surface and
pushes the apparent saddle later than it truly is.  Some of an apparently
late transition state can therefore be the scan rather than the potential.
Relaxing is slower, so compare the two: if the saddle moves, the frozen
number was an artefact.
"""

import argparse

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from high_fidelity_torch import HighFidelityBatchedReactiveSimulation


BOX = 20.0
CENTRE = np.array([BOX / 2, BOX / 2, BOX / 2])

SYMBOLS = ["C", "O", "H", "H", "H"]

# Spectator degrees of freedom, in the order the optimiser sees them:
# C=O length, spectator C-H length, donor angle, spectator angle.
# Angles are from the C=O axis, with the two hydrogens on opposite sides.
FROZEN = np.array([1.20, 1.09, 122.0, 122.0])

LIMITS = [(1.05, 1.75), (0.95, 1.45), (85.0, 180.0), (85.0, 180.0)]


def formaldehyde_geometry(donor_length, transfer_length, spectators=None):
    """H2C=O plus an incoming hydrogen, collinear with the donor C-H.

    `spectators` is (r_CO, r_CH_spectator, donor angle, spectator angle),
    defaulting to the frozen formaldehyde values.
    """
    if spectators is None:
        spectators = FROZEN

    length_co, length_ch, angle_donor, angle_other = spectators

    donor_axis = np.array([
        np.sin(np.radians(angle_donor)), np.cos(np.radians(angle_donor)), 0.0
    ])
    other_axis = np.array([
        -np.sin(np.radians(angle_other)), np.cos(np.radians(angle_other)), 0.0
    ])

    carbon = np.zeros(3)
    oxygen = np.array([0.0, length_co, 0.0])
    donor_h = donor_length * donor_axis
    other_h = length_ch * other_axis
    incoming = donor_h + transfer_length * donor_axis

    return SYMBOLS, np.array([carbon, oxygen, donor_h, other_h, incoming])


def build(physics="high_fidelity", mixing=None, sato=None,
          flatten=None, cap=None):
    cls = (
        HighFidelityBatchedReactiveSimulation if physics == "high_fidelity"
        else BatchedReactiveSimulation
    )

    # The correction reads these names from module scope on every call, so
    # rebinding them here changes the physics without editing the file.  Only
    # useful for measuring the knobs; set them properly in the source once you
    # know what they should be.
    import high_fidelity_torch

    if mixing is not None:
        high_fidelity_torch.H_TRANSFER_STATE_MIXING_FRACTION = float(mixing)

    if sato is not None:
        if not hasattr(high_fidelity_torch, "H_TRANSFER_SATO"):
            raise SystemExit(
                "this build of high_fidelity_torch has no H_TRANSFER_SATO; "
                "apply the v5 patch first"
            )
        high_fidelity_torch.H_TRANSFER_SATO = float(sato)

    if flatten is not None:
        if not hasattr(high_fidelity_torch, "H_TRANSFER_COUPLING_FLATTEN"):
            raise SystemExit(
                "this build of high_fidelity_torch has no "
                "H_TRANSFER_COUPLING_FLATTEN; apply the v5 patch first"
            )
        high_fidelity_torch.H_TRANSFER_COUPLING_FLATTEN = float(flatten)

    if cap is not None:
        if not hasattr(high_fidelity_torch, "H_TRANSFER_LOWERING_CAP"):
            raise SystemExit(
                "this build of high_fidelity_torch has no "
                "H_TRANSFER_LOWERING_CAP; apply the v5 patch first"
            )
        # A negative value means "no ceiling", so one sweep can include the
        # uncapped case alongside the capped ones.
        high_fidelity_torch.H_TRANSFER_LOWERING_CAP = (
            None if float(cap) < 0 else float(cap)
        )

    symbols, positions = formaldehyde_geometry(1.09, 1.60)
    return cls(
        boxes=[(symbols, positions + CENTRE)],
        box_size=BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=torch.float64,
    )


def energy_of(sim, positions):
    sim.positions = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype
    )
    # Rebuilt every point: the table is made once at construction with a skin,
    # and a stale one silently drops pairs that have moved into range.
    sim.build_neighbours()

    with torch.no_grad():
        return float(torch.sum(sim.energy_per_atom(sim.positions)))


def energy_at(sim, donor_length, transfer_length, spectators=None):
    _, positions = formaldehyde_geometry(
        donor_length, transfer_length, spectators
    )
    return energy_of(sim, positions)


def relaxed_energy(sim, donor_length, transfer_length, start=None):
    """Minimise over the spectator coordinates at fixed r(C-H) and r(H-H).

    Powell rather than a gradient method: the cost is a handful of scalars,
    the surface has flat regions where a numeric gradient is mostly noise,
    and there is no analytic derivative for the constrained coordinates.
    Warm starting from a neighbouring grid point cuts the evaluation count
    sharply, which is what makes a relaxed grid affordable at all.
    """
    from scipy.optimize import minimize

    if start is None:
        start = FROZEN

    def cost(spectators):
        for value, (low, high) in zip(spectators, LIMITS):
            if not low <= value <= high:
                return 1e6
        return energy_at(sim, donor_length, transfer_length, spectators)

    result = minimize(
        cost, np.asarray(start, float), method="Powell",
        options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 2000},
    )
    return float(result.fun), np.asarray(result.x, float)


def surface(sim, donor_lengths, transfer_lengths, relax=False, progress=True):
    """Energy grid, indexed [donor, transfer]."""
    grid = np.empty((len(donor_lengths), len(transfer_lengths)))
    carried = None

    for i, donor in enumerate(donor_lengths):
        # Each row restarts from the row above rather than from the previous
        # column: stretching r(C-H) disturbs the spectators far less than
        # sweeping r(H-H) right across the transfer does.
        row_start = carried

        for j, transfer in enumerate(transfer_lengths):
            if relax:
                value, row_start = relaxed_energy(
                    sim, donor, transfer, start=row_start
                )
                if j == 0:
                    carried = row_start
            else:
                value = energy_at(sim, donor, transfer)

            grid[i, j] = value

        if progress and relax:
            print(f"  row {i + 1}/{len(donor_lengths)}  "
                  f"r(C-H) = {donor:.3f} A", flush=True)

    return grid


# ----------------------------------------------------------------------
# Saddle by flooding
# ----------------------------------------------------------------------

def basin_seeds(grid, donor_lengths, transfer_lengths,
                detached_transfer=1.30, detached_donor=1.75,
                bonded_transfer=0.95):
    """Cells standing for the reactant and product basins.

    Reactant: the incoming hydrogen out past its cutoffs, molecule intact.
    Product:  the donor bond broken, the new H-H bond formed.
    """
    reactant_columns = np.where(transfer_lengths >= detached_transfer)[0]
    product_rows = np.where(donor_lengths >= detached_donor)[0]
    product_columns = np.where(transfer_lengths <= bonded_transfer)[0]

    if len(reactant_columns) == 0 or len(product_rows) == 0:
        return None, None
    if len(product_columns) == 0:
        return None, None

    block = grid[:, reactant_columns]
    flat = int(np.argmin(block))
    reactant = (
        flat // block.shape[1],
        int(reactant_columns[flat % block.shape[1]]),
    )

    block = grid[np.ix_(product_rows, product_columns)]
    flat = int(np.argmin(block))
    product = (
        int(product_rows[flat // block.shape[1]]),
        int(product_columns[flat % block.shape[1]]),
    )

    return reactant, product


def flood_saddle(grid, reactant, product):
    """Lowest energy at which the two basins become connected.

    Cells are added in ascending energy order, each joining whichever of its
    four neighbours are already present.  The moment the two seeds share a
    component, the cell just added is the highest point on the easiest route
    between them, which is the saddle.  No path has to be guessed, and a
    plateau or a repulsive corner cannot masquerade as a barrier.
    """
    rows, columns = grid.shape
    order = np.argsort(grid, axis=None)

    parent = list(range(rows * columns))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[first] = second

    start = reactant[0] * columns + reactant[1]
    goal = product[0] * columns + product[1]

    present = np.zeros(rows * columns, dtype=bool)

    for node in order:
        node = int(node)
        row, column = divmod(node, columns)
        present[node] = True

        for step_row, step_column in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near_row, near_column = row + step_row, column + step_column
            if not (0 <= near_row < rows and 0 <= near_column < columns):
                continue
            near = near_row * columns + near_column
            if present[near]:
                union(node, near)

        if present[start] and present[goal] and find(start) == find(goal):
            return (row, column), float(grid[row, column])

    return None, None


def ridge_positions(sim, donor_lengths, transfer_lengths):
    """Barrier position along r(H-H) for each donor bond length.

    Kept for the regression tests.  Uses the same flood construction, run on
    one row at a time: the barrier is the highest point on the easiest route
    from the product well to the entrance channel within that slice.
    """
    found = []

    for donor in donor_lengths:
        profile = np.array([
            energy_at(sim, donor, transfer) for transfer in transfer_lengths
        ])
        column = profile.reshape(1, -1)

        inside = np.where(transfer_lengths < 0.95)[0]
        if len(inside) == 0:
            found.append(None)
            continue

        product = (0, int(inside[np.argmin(profile[inside])]))
        reactant = (0, len(profile) - 1)

        cell, energy = flood_saddle(column, reactant, product)
        if cell is None or energy <= profile[reactant] + 1e-3:
            found.append(None)
        else:
            found.append(float(transfer_lengths[cell[1]]))

    return found


def fixed_donor_slice(physics, donor_lengths, transfer_lengths, relax=False,
                      mixing=None, sato=None, flatten=None, cap=None):
    """Barrier along r(H-H) with the donor bond held at each fixed length.

    The full two dimensional scan asks the thermal question: what is the
    lowest route across the surface if the molecule is free to arrange
    itself.  A collision does not get to ask that.  The encounter lasts of
    order tens of femtoseconds and a C-H stretch has a period near ten, so
    the donor bond samples whatever phase it happens to be in rather than
    obligingly extending to meet the incoming atom.

    So this holds r(C-H) fixed and reads the barrier along r(H-H) alone.
    The row at equilibrium, near 1.09 A, is the potential a fast impact
    actually meets.  If that barrier is far above the two dimensional saddle,
    the reaction is limited by how long the encounter lasts rather than by
    the height of the col, and tuning the coupling is fitting a constant to
    a barrier the dynamics never samples.

    The remaining rows show how much pre-stretch would be needed to bring the
    barrier within reach, which is the thing a slower encounter buys you.
    """
    sim = build(physics, mixing=mixing, sato=sato, flatten=flatten,
                cap=cap)

    print(f"physics: {getattr(sim, 'physics_model_name', 'reactive base')}")
    if mixing is not None:
        print(f"state mixing fraction: {mixing} (overridden)")
    print("\ndonor bond held fixed; barrier read along r(H-H) only")
    print(f"{'r(C-H)':>8} {'barrier':>11} {'at r(H-H)':>12}")

    for donor in donor_lengths:
        if relax:
            profile = np.array([
                relaxed_energy(sim, donor, transfer)[0]
                for transfer in transfer_lengths
            ])
        else:
            profile = np.array([
                energy_at(sim, donor, transfer)
                for transfer in transfer_lengths
            ])

        # Same flood construction as the full grid, on a single row: the
        # barrier is the highest point on the easiest route from the product
        # well out to the entrance channel within this slice.
        inside = np.where(transfer_lengths < 0.95)[0]
        if len(inside) == 0:
            print(f"{donor:8.3f} {'no product well':>11}")
            continue

        product = (0, int(inside[np.argmin(profile[inside])]))
        entrance = (0, len(profile) - 1)

        cell, energy = flood_saddle(profile.reshape(1, -1), entrance, product)

        if cell is None or energy <= profile[entrance[1]] + 1e-3:
            print(f"{donor:8.3f} {'no barrier':>11} {'-':>12}")
            continue

        print(f"{donor:8.3f} {energy - profile[entrance[1]]:8.3f} eV "
              f"{transfer_lengths[cell[1]]:10.3f} A")

    print("\nthe row nearest 1.09 A is what a fast collision meets;")
    print("compare it against the relaxed two dimensional saddle")


TRAP_DONOR = 1.39
TRAP_TRANSFER = 0.92
IMPACT_DONOR = 1.08

# Reference transition state for H + H2CO -> H2 + HCO, from Siai, Oueslati
# and Kerkeni (2016), Chem. Phys. 474, 44-51: CCSD(T)//MP2 geometry with a
# near collinear C-H-H arrangement.
REFERENCE_DONOR = 1.293
REFERENCE_TRANSFER = 1.008

# Computed saddle heights for the same reaction, in eV:
#   CBS/DLPNO-CCSD(T)        5.85 kcal/mol = 0.254
#   CCSD(T)/cc-pVTZ//MP2     6.69 kcal/mol = 0.290
#   CCSD(T)-F12a/cc-pVTZ-F12 6.68 kcal/mol = 0.290
#
# Experimental Arrhenius activation energies for this reaction are lower,
# around 0.17 eV, but they are not the saddle height: they fold in quantum
# tunnelling, which lets the reaction proceed below the classical barrier.
# This model is classical, so the computed saddle is the right comparison
# and the experimental number is not. Fitting to 0.17 bakes a quantum effect
# into a classical potential.
REFERENCE_BARRIER = (0.254, 0.290)

# Reaction energy from Active Thermochemical Tables dissociation energies,
# D0(H2CO-H) = 3.760 eV and D0(H-H) = 4.478 eV.
REFERENCE_REACTION = -0.718


def trap_depth(sim, transfer_lengths):
    """How far the three-centre structure sits below separated products.

    The geometry is the one a trapped trajectory actually held for 10 ps:
    the donor bond stretched but intact while the new bond is already at
    close to its equilibrium length, so the hydrogen is bonded twice at once.

    The reference is the same hydrogen with the donor removed entirely, which
    is where it should end up.  A negative number means the shared-hydrogen
    structure is genuinely bound and nothing will make it dissociate; zero or
    positive means it is at worst a shoulder that drains to products.
    """
    trapped = energy_at(sim, TRAP_DONOR, TRAP_TRANSFER)

    # Donor walked out past every cutoff, new bond left where it was.
    separated = min(
        energy_at(sim, donor, TRAP_TRANSFER)
        for donor in (2.20, 2.60, 3.00)
    )

    return trapped - separated


def impact_barrier(sim, transfer_lengths, donor_length=IMPACT_DONOR):
    """Barrier along r(H-H) with the donor bond at its equilibrium length.

    This is the potential a fast collision meets, as distinct from the
    relaxed two dimensional saddle, which assumes the donor bond has time to
    stretch to meet the incoming atom.
    """
    profile = np.array([
        energy_at(sim, donor_length, transfer) for transfer in transfer_lengths
    ])

    inside = np.where(transfer_lengths < 0.95)[0]
    if len(inside) == 0:
        return None

    product = (0, int(inside[np.argmin(profile[inside])]))
    entrance = (0, len(profile) - 1)

    cell, energy = flood_saddle(profile.reshape(1, -1), entrance, product)
    if cell is None or energy <= profile[entrance[1]] + 1e-3:
        return None

    return energy - profile[entrance[1]]


def locate_saddle(sim, donor_lengths, transfer_lengths, relax=False):
    """Find the saddle on the grid rather than assuming where it sits.

    Earlier versions checked curvature at a hardcoded point, the geometry a
    trapped trajectory happened to hold.  That was the right check in the
    wrong place: once the topology changes the saddle moves, so the location
    has to be measured too, not carried over from the configuration that was
    being diagnosed.

    Returns (donor, transfer, height above the reactant minimum, grid), or
    None if the two basins cannot be separated on this window.
    """
    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax)

    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    if reactant is None:
        return None

    cell, energy = flood_saddle(grid, reactant, product)
    if cell is None:
        return None

    return (
        float(donor_lengths[cell[0]]),
        float(transfer_lengths[cell[1]]),
        float(energy - grid[reactant]),
        grid,
    )


def saddle_character(sim, donor=TRAP_DONOR, transfer=TRAP_TRANSFER,
                     step=0.02):
    """Is the three-centre point a saddle, a minimum, or neither?

    A transition state is a first-order saddle: curvature positive along
    every direction but one, negative along the reaction coordinate.  The two
    scan axes are not the normal modes, so this diagonalises the 2x2 Hessian
    in (r_donor, r_transfer) and reports its eigenvalues.  Signs are what
    matter, not magnitudes:

        one negative, one positive  -> a genuine transition state
        both positive               -> a minimum, i.e. the trap
        both negative               -> a local maximum

    Second differences on a 0.02 A stencil.  Fine enough to resolve the sign
    on a surface this smooth, coarse enough not to be swamped by float noise.
    """
    def energy(a, b):
        return energy_at(sim, a, b)

    centre = energy(donor, transfer)

    dd = (energy(donor + step, transfer) - 2.0 * centre
          + energy(donor - step, transfer)) / (step * step)
    tt = (energy(donor, transfer + step) - 2.0 * centre
          + energy(donor, transfer - step)) / (step * step)
    dt = (
        energy(donor + step, transfer + step)
        - energy(donor + step, transfer - step)
        - energy(donor - step, transfer + step)
        + energy(donor - step, transfer - step)
    ) / (4.0 * step * step)

    hessian = np.array([[dd, dt], [dt, tt]])
    eigenvalues = np.linalg.eigvalsh(hessian)

    negative = int(np.sum(eigenvalues < 0))
    if negative == 1:
        verdict = "transition state"
    elif negative == 0:
        verdict = "MINIMUM (trap)"
    else:
        verdict = "local maximum"

    return eigenvalues, verdict, centre


def saddle_report(physics, donor_lengths, transfer_lengths, caps,
                  mixing=None, relax=False):
    """Locate the saddle at each cap, then check it against the reference.

    Three things have to be right at once, and only the first is a fit:

        height    near the computed classical barrier, 0.25 to 0.29 eV
        geometry  near r(C-H) 1.293 A, r(H-H) 1.008 A
        character one negative curvature, one positive

    Height alone can be matched by any monotonic knob, so on its own it says
    very little.  Geometry and character are free predictions: nothing in the
    fit targets them, so agreement there is evidence the mechanism is right
    rather than evidence the constant was chosen well.
    """
    low, high = REFERENCE_BARRIER

    print("reference transition state (Siai, Oueslati, Kerkeni 2016):")
    print("  r(C-H) %.3f A, r(H-H) %.3f A, barrier %.3f-%.3f eV"
          % (REFERENCE_DONOR, REFERENCE_TRANSFER, low, high))
    print("heights are above the reactant minimum on the full grid\n")

    print(f"{'cap':>7} {'r(C-H)':>9} {'r(H-H)':>9} {'height':>10} "
          f"{'offset':>9}  character")

    for cap in caps:
        sim = build(physics, mixing=mixing, cap=cap)

        found = locate_saddle(sim, donor_lengths, transfer_lengths, relax=relax)
        label = "none" if cap < 0 else f"{cap:.2f}"

        if found is None:
            print(f"{label:>7}   no route between the basins on this window")
            continue

        donor, transfer, height, _ = found

        # Curvature at the located saddle, not at any assumed point.
        eigenvalues, verdict, _ = saddle_character(
            sim, donor=donor, transfer=transfer
        )

        # Distance from the reference geometry, in the scan plane.
        offset = float(np.hypot(donor - REFERENCE_DONOR,
                                transfer - REFERENCE_TRANSFER))

        flag = "  <-- in range" if low <= height <= high else ""
        print(f"{label:>7} {donor:8.3f}A {transfer:8.3f}A {height:+8.3f} eV "
              f"{offset:8.3f}A  {verdict}{flag}")

    print("\noffset is the distance from the reference geometry in the")
    print("(r_CH, r_HH) plane; the grid step sets the floor on how small")
    print("it can be, so read it as a scale rather than a precise number.")


def cap_sweep(physics, transfer_lengths, mixings, caps):
    """Barrier and trap depth across the crossing-stabilisation ceiling.

    Unlike sato and the coupling flatten, this keys on the diabatic gap
    rather than on any distance, so it can distinguish two configurations
    that look identical to every cutoff taper.  That is the whole point: the
    trap and the collision barrier both sit where the tapers are saturated.
    """
    print("barrier at r(C-H) = %.2f A (what a fast collision meets)" %
          IMPACT_DONOR)
    print("trap = three-centre energy relative to separated products;")
    print("       negative means bound, so a trajectory can never leave\n")

    header = "".join(f"{m:>17.2f}" for m in mixings)
    print(f"{'cap':>7}{header}      <- mixing")

    for cap in caps:
        cells = []
        for mixing in mixings:
            sim = build(physics, mixing=mixing, cap=cap)
            barrier = impact_barrier(sim, transfer_lengths)
            trap = trap_depth(sim, transfer_lengths)
            shown = "  none" if barrier is None else f"{barrier:6.3f}"
            cells.append(f"{shown} /{trap:+7.3f}")

        label = "none" if cap < 0 else f"{cap:.2f}"
        print(f"{label:>7}" + "".join(f"{cell:>17}" for cell in cells))

    print("\neach cell is  barrier eV / trap eV")
    print("wanted: trap >= 0 with the smallest barrier you can get")


def knob_map(physics, transfer_lengths, mixings, satos, flatten=0.0):
    """Barrier and trap depth together, across both parameters.

    These are the two quantities that have to be satisfied at once, and under
    a single knob they moved in the same direction, which is why v4 could not
    have both.  The question this answers is whether a second parameter
    separates them.

    Read it as: find a cell whose barrier is near the physical activation
    scale AND whose trap depth is not negative.  If no such cell exists, the
    anti-Morse blend is the wrong second parameter and the functional form
    needs a different change rather than another constant.
    """
    print("barrier at r(C-H) = %.2f A (what a fast collision meets)" %
          IMPACT_DONOR)
    print("coupling flatten = %.2f" % flatten)
    print("trap = three-centre energy relative to separated products;")
    print("       negative means bound, so a trajectory can never leave\n")

    header = "".join(f"{m:>17.2f}" for m in mixings)
    print(f"{'sato':>6}{header}      <- mixing")

    for sato in satos:
        cells = []
        for mixing in mixings:
            sim = build(physics, mixing=mixing, sato=sato,
                        flatten=flatten)
            barrier = impact_barrier(sim, transfer_lengths)
            trap = trap_depth(sim, transfer_lengths)

            shown = "  none" if barrier is None else f"{barrier:6.3f}"
            cells.append(f"{shown} /{trap:+7.3f}")

        print(f"{sato:6.2f}" + "".join(f"{cell:>17}" for cell in cells))

    print("\neach cell is  barrier eV / trap eV")
    print("wanted: barrier near 0.17-0.20, trap >= 0")


def measure(sim, donor_lengths, transfer_lengths, relax=False):
    """Saddle position, activation energy and reaction energy for one model."""
    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax)

    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    if reactant is None:
        return None

    cell, saddle = flood_saddle(grid, reactant, product)
    if cell is None:
        return None

    return {
        "grid": grid,
        "reactant": reactant,
        "product": product,
        "saddle_cell": cell,
        "barrier": saddle - grid[reactant],
        "reaction": grid[product] - grid[reactant],
        "stretch": donor_lengths[cell[0]] - donor_lengths[reactant[0]],
    }


def sweep(physics, donor_lengths, transfer_lengths, values, relax=False):
    """How does the barrier respond to the state-mixing fraction?

    At the saddle the two valence states are near degenerate, so the mixed
    surface sits roughly one coupling below their average.  The barrier
    should therefore fall almost linearly in the mixing fraction while the
    saddle stays put.  If instead the saddle wanders, the coupling is doing
    something other than lowering a crossing, and the number it lands on
    would not mean much.

    This measures the knob rather than guessing it, but a value that fits one
    reaction is still a fit to one reaction.  Whatever comes out has to hold
    on a second, independent H transfer before it is a model rather than a
    formaldehyde-shaped constant.
    """
    print(f"{'mixing':>8} {'barrier':>10} {'reaction':>11} "
          f"{'saddle r(C-H)':>15} {'r(H-H)':>9}")

    for value in values:
        sim = build(physics, mixing=value)
        found = measure(sim, donor_lengths, transfer_lengths, relax=relax)
        if found is None:
            print(f"{value:8.3f} {'no route':>10}")
            continue
        cell = found["saddle_cell"]
        print(f"{value:8.3f} {found['barrier']:8.3f} eV "
              f"{found['reaction']:+9.3f} eV "
              f"{donor_lengths[cell[0]]:13.3f} A "
              f"{transfer_lengths[cell[1]]:7.3f} A")


def report(physics, donor_lengths, transfer_lengths, relax=False, plot=None,
           mixing=None, sato=None):
    sim = build(physics, mixing=mixing, sato=sato, flatten=flatten,
                cap=cap)

    print(f"physics: {getattr(sim, 'physics_model_name', 'reactive base')}")
    print(f"grid: {len(donor_lengths)} x {len(transfer_lengths)} points")
    print(f"spectators: {'relaxed' if relax else 'frozen at formaldehyde'}")
    if mixing is not None:
        print(f"state mixing fraction: {mixing} (overridden)")
    print()

    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax)

    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    if reactant is None:
        print("could not identify both basins - widen the scan window")
        return grid

    reactant_energy = grid[reactant]
    product_energy = grid[product]

    print(f"\nreactant minimum: r(C-H) {donor_lengths[reactant[0]]:.3f} A, "
          f"r(H-H) {transfer_lengths[reactant[1]]:.3f} A")
    print(f"product minimum:  r(C-H) {donor_lengths[product[0]]:.3f} A, "
          f"r(H-H) {transfer_lengths[product[1]]:.3f} A")
    print(f"reaction energy:  {product_energy - reactant_energy:+.3f} eV")

    cell, saddle_energy = flood_saddle(grid, reactant, product)
    if cell is None:
        print("\nno connected route between the basins on this grid")
        return grid

    stretch = donor_lengths[cell[0]] - donor_lengths[reactant[0]]

    print("\nSADDLE")
    print(f"  r(C-H)             {donor_lengths[cell[0]]:.3f} A")
    print(f"  r(H-H)             {transfer_lengths[cell[1]]:.3f} A")
    print(f"  activation energy  {saddle_energy - reactant_energy:.3f} eV "
          "above the reactant minimum")
    print(f"  C-H stretch at it  {stretch:+.3f} A")

    if stretch > 0.20:
        print("\n  the donor bond is already far extended before the col "
              "opens,\n  so a fast collision meets no route through")

    if plot:
        draw(grid, donor_lengths, transfer_lengths, reactant, cell, plot)
        print(f"\nwritten to {plot}")

    return grid


def draw(grid, donor_lengths, transfer_lengths, reactant, saddle, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relative = grid - grid[reactant]
    # Clipped so the deep repulsive corner does not flatten the contrast
    # across the region the reaction actually passes through.
    ceiling = float(np.percentile(relative, 90))
    floor = float(np.percentile(relative, 2))
    shown = np.clip(relative, floor, ceiling)

    figure, axes = plt.subplots(figsize=(7, 5.5))

    mesh = axes.contourf(transfer_lengths, donor_lengths, shown, levels=40)
    axes.contour(
        transfer_lengths, donor_lengths, shown,
        levels=20, colors="white", linewidths=0.4, alpha=0.5,
    )
    axes.plot(
        transfer_lengths[saddle[1]], donor_lengths[saddle[0]],
        marker="x", markersize=11, markeredgewidth=2.2, color="red",
    )
    axes.set_xlabel("r(H-H) forming, A")
    axes.set_ylabel("r(C-H) breaking, A")
    axes.set_title("H + CH2O abstraction, energy above reactant minimum")
    figure.colorbar(mesh, ax=axes, label="eV")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physics", default="high_fidelity",
        choices=["high_fidelity", "base"],
    )
    parser.add_argument(
        "--relax", action="store_true",
        help="minimise the spectator coordinates at every grid point",
    )
    parser.add_argument("--plot", default=None)
    parser.add_argument(
        "--mixing", type=float, default=None,
        help="override H_TRANSFER_STATE_MIXING_FRACTION for this run",
    )
    parser.add_argument(
        "--sweep", default=None,
        help="comma separated mixing fractions to measure, e.g. 0.45,0.55,0.65",
    )
    parser.add_argument(
        "--sato", type=float, default=None,
        help="override H_TRANSFER_SATO for this run (needs the v5 patch)",
    )
    parser.add_argument(
        "--flatten", type=float, default=None,
        help=(
            "override H_TRANSFER_COUPLING_FLATTEN for this run; applies to "
            "--knob-map as well (needs the v5 patch)"
        ),
    )
    parser.add_argument(
        "--cap", type=float, default=None,
        help=(
            "override H_TRANSFER_LOWERING_CAP for this run; negative means "
            "no ceiling (needs the v5 patch)"
        ),
    )
    parser.add_argument(
        "--saddle-check", action="store_true",
        help=(
            "report curvature and height at the three-centre point across "
            "the cap, to see whether it becomes a proper transition state"
        ),
    )
    parser.add_argument(
        "--cap-sweep", action="store_true",
        help="measure barrier and trap across the crossing-stabilisation cap",
    )
    parser.add_argument(
        "--knob-map", action="store_true",
        help=(
            "measure the impact barrier and the three-centre trap depth "
            "across both mixing and sato, to see whether any pair satisfies "
            "them at once"
        ),
    )
    parser.add_argument(
        "--fixed-donor", action="store_true",
        help=(
            "hold r(C-H) fixed and read the barrier along r(H-H) alone, "
            "which is the potential a fast collision actually meets"
        ),
    )
    parser.add_argument("--donor-min", type=float, default=1.00)
    parser.add_argument("--donor-max", type=float, default=1.90)
    parser.add_argument("--donor-step", type=float, default=None)
    parser.add_argument("--transfer-min", type=float, default=0.65)
    parser.add_argument("--transfer-max", type=float, default=1.60)
    parser.add_argument("--transfer-step", type=float, default=None)
    options = parser.parse_args()

    # Relaxing costs a constrained minimisation per cell, so the default grid
    # coarsens when it is switched on.  Both remain fine enough to place the
    # saddle well inside the accuracy the model itself claims.
    donor_step = options.donor_step or (0.04 if options.relax else 0.02)
    transfer_step = options.transfer_step or (0.02 if options.relax else 0.005)

    donor_lengths = np.arange(options.donor_min, options.donor_max, donor_step)
    transfer_lengths = np.arange(
        options.transfer_min, options.transfer_max, transfer_step
    )

    if options.saddle_check:
        saddle_report(
            options.physics, donor_lengths, transfer_lengths,
            caps=[-1.0, 1.45, 1.36, 1.32, 1.29, 1.20],
            mixing=options.mixing, relax=options.relax,
        )
        return

    if options.cap_sweep:
        cap_sweep(
            options.physics, transfer_lengths,
            mixings=[0.45, 0.63, 0.80],
            caps=[-1.0, 2.00, 1.70, 1.45, 1.20, 1.00],
        )
        return

    if options.knob_map:
        knob_map(
            options.physics, transfer_lengths,
            mixings=[0.45, 0.63, 0.80],
            satos=[0.0, 0.25, 0.50, 0.75, 1.00],
            flatten=(options.flatten or 0.0),
        )
        return

    if options.fixed_donor:
        fixed_donor_slice(
            options.physics, donor_lengths, transfer_lengths,
            relax=options.relax, mixing=options.mixing, sato=options.sato,
            flatten=options.flatten, cap=options.cap,
        )
        return

    if options.sweep:
        sweep(
            options.physics, donor_lengths, transfer_lengths,
            [float(part) for part in options.sweep.split(",")],
            relax=options.relax,
        )
        return

    report(
        options.physics,
        donor_lengths,
        transfer_lengths,
        relax=options.relax,
        plot=options.plot,
        mixing=options.mixing,
        sato=options.sato,
        flatten=options.flatten,
        cap=options.cap,
    )


if __name__ == "__main__":
    main()