
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
import time

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from high_fidelity_torch import HighFidelityBatchedReactiveSimulation
from h_state_torch import HStateReferenceBatchedSimulation


BOX = 20.0
CENTRE = np.array([BOX / 2, BOX / 2, BOX / 2])

# float64 is the default because the continuity work needed to resolve energy
# steps around 0.05 eV, which float32 noise would bury. Reading a barrier to
# three decimals does not need it, so --fast trades that precision for speed.
#
# These systems are five or six atoms, so a GPU may well be slower than the
# CPU here: per-call launch overhead dominates when there is almost nothing to
# compute. --device exists to find out rather than to assume.
SCAN_DEVICE = "cpu"
SCAN_DTYPE = torch.float64

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


# ----------------------------------------------------------------------
# Second system: a hydrogen hopping between two oxygens.
# ----------------------------------------------------------------------
#
# Everything so far is calibrated on one reaction, which leaves open whether
# the mixing fraction and the softening describe hydrogen transfer or just
# describe formaldehyde. This is the independent check, and it is chosen to
# be awkward for the correction in ways the first system was not:
#
#   different tables      O-H depths and cutoffs rather than C-H, so nothing
#                         carries over from the fitted pair
#   softening inert       neither oxygen holds a multiple bond, so the
#                         environment term contributes nothing and the
#                         correction is tested on its own
#   symmetric             donor and acceptor are the same element, so the
#                         argmax that picks donor and competitor swaps its
#                         labels exactly at the transition state, which is
#                         the geometry the gate fix was written for
#
# Being symmetric also pins the reaction energy at zero by construction. Any
# departure from zero is the correction failing to vanish in one asymptote,
# which no amount of parameter choice should be able to hide.

# H2O + OH, five atoms: the moving proton plus one spectator on each oxygen.
# Two full waters plus a proton would be H5O2+, a charged species the model
# has no way to represent, and it produced an 8 eV nonsense when tried.
WATER_SYMBOLS = ["O", "H", "H", "O", "H"]

# Spectator coordinates: the two O-H bonds that are not transferring, and the
# angle each makes with the transfer axis.
WATER_FROZEN = np.array([0.96, 0.96, 75.53, 75.53])

# These angles are measured from the transfer axis, not between the two
# bonds, so a spectator value of 75.53 is an H-O-H angle of 104.47. The two
# are complements and sum to 180. That cost a long detour: 75.53 kept
# appearing as the relaxed value and looked like the optimiser folding a
# water molecule 29 degrees away from its rest angle, when it was the
# optimiser finding the rest angle exactly.
#
# So the range below is 30 to 120 degrees of H-O-H. The floor was 90 in
# spectator terms, which is 90 of H-O-H and above the 104.47 the model
# wants, so the relaxed saddle sat on the bound and its barrier was the edge
# of the search box rather than a minimum. Widening was the right fix for
# the wrong stated reason.
WATER_LIMITS = [(0.85, 1.15), (0.85, 1.15), (60.0, 150.0), (60.0, 150.0)]


def water_pair_geometry(oxygen_separation, offset, spectators=None):
    """Two waters at a fixed separation, proton displaced along the O-O axis.

    offset is the proton's displacement from the midpoint, so 0 is the
    symmetric shared position and positive values move it toward the
    acceptor. The heavy atoms do not move.

    That fixed separation is the point. Letting the two O-H distances vary
    independently, as the two dimensional scan does, sweeps the O-O distance
    along with them, and oxygen-oxygen repulsion then dominates a surface
    meant to be about the proton. Holding the heavy atoms and moving one
    coordinate is how symmetric proton transfer is normally treated, and it
    makes the symmetry exact: the energy at +offset must equal the energy at
    -offset, for any correct potential, with no parameter able to hide a
    departure.
    """
    if spectators is None:
        spectators = WATER_FROZEN

    donor_oh, acceptor_oh, donor_angle, acceptor_angle = spectators

    donor_oxygen = np.zeros(3)
    acceptor_oxygen = np.array([oxygen_separation, 0.0, 0.0])
    moving_h = np.array([0.5 * oxygen_separation + offset, 0.0, 0.0])

    def spoke(origin, length, angle, direction, tilt):
        radians = np.radians(angle)
        return origin + length * np.array([
            direction * np.cos(radians),
            np.sin(radians) * np.cos(tilt),
            np.sin(radians) * np.sin(tilt),
        ])

    # Mirror images of each other about the midpoint, so the whole
    # arrangement is symmetric and the check below means what it says.
    donor_spokes = [
        spoke(donor_oxygen, donor_oh, donor_angle, -1.0, tilt)
        for tilt in (np.pi / 2, -np.pi / 2)
    ]
    acceptor_spokes = [
        spoke(acceptor_oxygen, acceptor_oh, acceptor_angle, 1.0, tilt)
        for tilt in (np.pi / 2, -np.pi / 2)
    ]

    return WATER_SYMBOLS, np.array([
        donor_oxygen, moving_h, donor_spokes[0],
        acceptor_oxygen, acceptor_spokes[0],
    ])


def water_geometry(donor_length, transfer_length, spectators=None):
    """H2O ... H ... OH2 with the transferring hydrogen on the O-O axis.

    donor_length is the distance from the donor oxygen to the transferring
    hydrogen; transfer_length is from that hydrogen to the acceptor oxygen.
    The two oxygens therefore sit donor_length + transfer_length apart, which
    means the scan sweeps the O-O separation as well rather than holding it
    fixed. That is deliberate: the heavy atoms move during a real proton
    transfer, and pinning them would put a wall in the way of the reaction.
    """
    if spectators is None:
        spectators = WATER_FROZEN

    donor_oh, acceptor_oh, donor_angle, acceptor_angle = spectators

    donor_oxygen = np.zeros(3)
    moving_h = np.array([donor_length, 0.0, 0.0])
    acceptor_oxygen = np.array([donor_length + transfer_length, 0.0, 0.0])

    def spoke(origin, length, angle, direction, tilt):
        radians = np.radians(angle)
        return origin + length * np.array([
            direction * np.cos(radians),
            np.sin(radians) * np.cos(tilt),
            np.sin(radians) * np.sin(tilt),
        ])

    # Both spectators take the same tilt, so the arrangement is its own
    # mirror image about the midpoint of the O-O axis.
    #
    # They did not: the donor's spoke used tilt 0 and the acceptor's used
    # pi/2, left over from when this builder returned six atoms and one spoke
    # was dropped from each list without checking the remaining pair still
    # matched. The surface was then asymmetric for a reaction that is
    # symmetric by construction, so E(a, b) did not equal E(b, a) and the
    # flood could take different routes depending on which side it came from.
    # That is the likeliest source of water's erratic barriers.
    donor_spokes = [
        spoke(donor_oxygen, donor_oh, donor_angle, -1.0, angle)
        for angle in (np.pi / 2, -np.pi / 2)
    ]
    acceptor_spokes = [
        spoke(acceptor_oxygen, acceptor_oh, acceptor_angle, 1.0, angle)
        for angle in (np.pi / 2, -np.pi / 2)
    ]

    return WATER_SYMBOLS, np.array([
        donor_oxygen, moving_h, donor_spokes[0],
        acceptor_oxygen, acceptor_spokes[0],
    ])


# ----------------------------------------------------------------------
# Third system: the control that formaldehyde cannot provide.
# ----------------------------------------------------------------------
#
# H + CH4 -> H2 + CH3 uses the identical transfer machinery to formaldehyde,
# but the carbon holds no multiple bond, so the environment softening is
# inert and the donor keeps the full generic C-H depth. Methane's C-H is one
# of the stronger bonds in ordinary chemistry and formaldehyde's is one of
# the weaker, so the barriers should differ by a wide margin in a known
# direction.
#
# Dynamics says they do not: at the same collision energy methane converted
# 63% of encounters against formaldehyde's 38%, which is the ordering
# inverted. This system exists so the same question can be asked of the
# surface directly, where there is no collision geometry to argue about.

METHANE_SYMBOLS = ["C", "H", "H", "H", "H", "H"]

# Spectator coordinates: the three C-H bonds that are not transferring, and
# the angle each makes with the donor bond.
METHANE_FROZEN = np.array([1.09, 1.09, 1.09, 109.47])

METHANE_LIMITS = [(0.95, 1.35), (0.95, 1.35), (0.95, 1.35), (95.0, 125.0)]


def methane_geometry(donor_length, transfer_length, spectators=None):
    """CH4 with a hydrogen approaching one C-H collinearly.

    Built to mirror the formaldehyde arrangement as closely as the two
    molecules allow: the donor bond lies along an axis, the incoming atom
    continues that line, and the spectators are held in a fixed
    arrangement. What differs is only the molecule, which is the point.
    """
    if spectators is None:
        spectators = METHANE_FROZEN

    first, second, third, angle = spectators

    carbon = np.zeros(3)

    # Donor bond along +z, the incoming atom beyond it on the same line.
    donor_axis = np.array([0.0, 0.0, 1.0])
    donor_h = donor_length * donor_axis
    incoming = donor_h + transfer_length * donor_axis

    # The other three arranged around the axis at the tetrahedral angle.
    radians = np.radians(angle)
    spectator_lengths = (first, second, third)
    spectators_out = []
    for index, length in enumerate(spectator_lengths):
        turn = 2.0 * np.pi * index / 3.0
        spectators_out.append(carbon + length * np.array([
            np.sin(radians) * np.cos(turn),
            np.sin(radians) * np.sin(turn),
            np.cos(radians),
        ]))

    return METHANE_SYMBOLS, np.array(
        [carbon, donor_h, *spectators_out, incoming]
    )


# ----------------------------------------------------------------------
# Fourth system: the one nothing was ever fitted to.
# ----------------------------------------------------------------------
#
# H + NH3 -> H2 + NH2 is the first reaction here involving nitrogen, and no
# parameter in the model has ever been calibrated against it. The mixing
# fraction came from formaldehyde, the environment softening from
# formaldehyde, and the bond table correction was found by testing ammonia's
# reaction energy but the depths themselves are dissociation energies read
# from CRC rather than anything fitted.
#
# So whatever this system gives is a prediction, not a fit, and it is worth
# keeping that way: if a Sato parameter or anything else is ever fitted to
# barriers, ammonia should be held out rather than included.
#
# Its reaction energy already tests well. The corrected N-H depth gives
# +0.145 eV against the HEAT and ATcT value of +0.122, with the old table
# giving -0.466 -- the wrong sign. That was what exposed the table holding
# average bond enthalpies rather than dissociation energies.

AMMONIA_SYMBOLS = ["N", "H", "H", "H", "H"]

# Spectator coordinates: the two N-H bonds that are not transferring, and
# the angle each makes with the donor bond. Ammonia is pyramidal, so 107
# degrees rather than methane's tetrahedral 109.47.
AMMONIA_FROZEN = np.array([1.01, 1.01, 107.0])

AMMONIA_LIMITS = [(0.90, 1.30), (0.90, 1.30), (95.0, 125.0)]


def ammonia_geometry(donor_length, transfer_length, spectators=None):
    """NH3 with a hydrogen approaching one N-H collinearly.

    Built to mirror methane's arrangement as closely as the two molecules
    allow, so a difference between them is the chemistry rather than the
    scan. Nitrogen has one fewer hydrogen and a slightly tighter angle, so
    there are three spectator coordinates rather than four.
    """
    if spectators is None:
        spectators = AMMONIA_FROZEN

    first, second, angle = spectators

    nitrogen = np.zeros(3)

    # Donor bond along +z, the incoming atom beyond it on the same line.
    donor_axis = np.array([0.0, 0.0, 1.0])
    donor_h = donor_length * donor_axis
    incoming = donor_h + transfer_length * donor_axis

    # The other two arranged around the axis. Their azimuthal separation is
    # derived rather than assumed: placing them on opposite sides, as
    # methane's three are, leaves the donor at the right angle to each but
    # the two of them at 146 degrees to each other rather than 107. For all
    # three pairwise angles to equal the target,
    #
    #     cos(separation) = (cos(angle) - cos(angle)^2) / sin(angle)^2
    #
    # Methane escapes this because three spectators at 120 degrees of azimuth
    # and a tetrahedral polar angle happen to give exactly 109.47 between
    # every pair. Two spectators have no such coincidence available.
    radians = np.radians(angle)
    cosine = np.cos(radians)
    separation = np.arccos(
        np.clip(
            (cosine - cosine * cosine) / max(np.sin(radians) ** 2, 1e-12),
            -1.0, 1.0,
        )
    )

    placed = []
    for index, length in enumerate((first, second)):
        turn = index * separation
        placed.append(nitrogen + length * np.array([
            np.sin(radians) * np.cos(turn),
            np.sin(radians) * np.sin(turn),
            cosine,
        ]))

    return AMMONIA_SYMBOLS, np.array(
        [nitrogen, donor_h, *placed, incoming]
    )


SYSTEMS = {
    "formaldehyde": {
        "geometry": formaldehyde_geometry,
        "frozen": FROZEN,
        "limits": LIMITS,
        "description": "H + H2CO -> H2 + HCO",
    },
    "water": {
        "geometry": water_geometry,
        "frozen": WATER_FROZEN,
        "limits": WATER_LIMITS,
        "description": "H2O + OH -> OH + H2O, symmetric proton transfer",
    },
    "methane": {
        "geometry": methane_geometry,
        "frozen": METHANE_FROZEN,
        "limits": METHANE_LIMITS,
        "description": "H + CH4 -> H2 + CH3, the strong-bond control",
    },
    "ammonia": {
        "geometry": ammonia_geometry,
        "frozen": AMMONIA_FROZEN,
        "limits": AMMONIA_LIMITS,
        "description": "H + NH3 -> H2 + NH2, nothing was fitted to this",
    },
}

# Which system the geometry helpers build. Set once from the command line
# rather than threaded through every function, because every scan works on
# one system at a time and passing it everywhere would touch code that has
# nothing else to do with the choice.
ACTIVE_SYSTEM = "formaldehyde"

# Geometry used only to construct the simulation object, before any scan
# point is evaluated. It has to be a sane arrangement for the system, not a
# scan point of interest.
SYSTEM_START = {
    "formaldehyde": (1.09, 1.60),
    "water": (0.98, 1.80),
    # Was (0.480, 0.520), a reference barrier pair that landed here from an
    # edit meant for REFERENCE_BARRIERS: both dictionaries have a "methane"
    # key and a two-number value, so a careless text replacement matched the
    # wrong one. Every methane simulation was therefore constructed with its
    # atoms half an angstrom apart. Harmless in effect, since the positions
    # are overwritten before any energy is read and the batched path does not
    # use this at all, but not something to leave sitting there.
    "methane": (1.09, 1.60),
    "ammonia": (1.01, 1.60),
}


def check_systems_registered():
    """Every system needs an entry in all four tables.

    They are separate dictionaries keyed by the same names, so a system can
    be added to one and forgotten in another, and the failure surfaces as a
    KeyError deep in a scan rather than at import. Worse, two of them hold a
    two-number tuple per system, so an edit meant for one can silently land
    in the other -- which happened: a reference barrier pair ended up as
    methane's construction geometry.

    REFERENCE_BARRIERS is deliberately allowed to be missing an entry, since
    a system may have no published barrier to compare against yet.
    """
    problems = []
    for name in SYSTEMS:
        for table, label in (
            (SYSTEM_START, "SYSTEM_START"),
            (SYSTEM_PROBES, "SYSTEM_PROBES"),
            (SPECTATOR_NAMES, "SPECTATOR_NAMES"),
        ):
            if name not in table:
                problems.append(f"{name} missing from {label}")

    # A construction geometry is a pair of bond lengths, so both entries
    # should be plausible distances. A reference barrier pair would not be.
    for name, start in SYSTEM_START.items():
        if not all(0.5 <= value <= 3.0 for value in start):
            problems.append(
                f"{name} SYSTEM_START {start} is not a pair of distances"
            )

    return problems


def active_geometry(donor_length, transfer_length, spectators=None):
    entry = SYSTEMS[ACTIVE_SYSTEM]
    return entry["geometry"](donor_length, transfer_length, spectators)


def active_frozen():
    return SYSTEMS[ACTIVE_SYSTEM]["frozen"]


def active_limits():
    return SYSTEMS[ACTIVE_SYSTEM]["limits"]


def check_water_symmetry(points=((1.00, 1.40), (1.10, 1.30), (1.20, 1.20),
                                 (0.95, 1.75))):
    """The water builder must produce its own mirror image.

    H2O + OH is symmetric, so the surface has to satisfy E(a, b) == E(b, a)
    exactly. It did not: the two spectator hydrogens were placed with
    different tilts, one in the plane and one out of it, so the arrangement
    was asymmetric by construction and the flood could route differently
    depending on which side it approached from.

    Geometry rather than energy, because it is a property of the builder and
    can be checked without a simulation. Returns the points that fail.
    """
    previous = ACTIVE_SYSTEM
    apply_system("water")

    failed = []
    try:
        for donor, transfer in points:
            _, forward = water_geometry(donor, transfer)
            _, reverse = water_geometry(transfer, donor)

            middle = 0.5 * np.linalg.norm(reverse[3] - reverse[0])
            mirrored = reverse.copy()
            mirrored[:, 0] = 2.0 * middle - mirrored[:, 0]

            one = np.array(sorted(map(tuple, np.round(forward, 6))))
            two = np.array(sorted(map(tuple, np.round(mirrored, 6))))
            if not np.allclose(one, two, atol=1e-6):
                failed.append((donor, transfer))
    finally:
        apply_system(previous)

    return failed


def build(physics="high_fidelity", mixing=None, sato=None,
          flatten=None, cap=None, softening=None,
          depth_power=None, over_weight=None, pair_depths=None,
          sato_pairs=None, boxes=None):
    if physics == "high_fidelity":
        cls = HighFidelityBatchedReactiveSimulation
    elif physics == "h_state":
        cls = HStateReferenceBatchedSimulation
    else:
        cls = BatchedReactiveSimulation

    # The h-state reference is a holdout model here: its single mixing
    # constant is calibrated against H + H2 and deliberately frozen while
    # formaldehyde, water, methane and ammonia are measured as predictions.
    if physics == "h_state" and mixing is not None:
        raise SystemExit(
            "--mixing is disabled for --physics h_state; "
            "H_STATE_MIXING is fixed at 0.472744 for validation"
        )

    # The correction reads these names from module scope on every call, so
    # rebinding them here changes the physics without editing the file.  Only
    # useful for measuring the knobs; set them properly in the source once you
    # know what they should be.
    import high_fidelity_torch
    import reactive as R

    # Every override is restored afterwards. Constructors copy these onto the
    # instance, so a simulation keeps what it was built with and the globals
    # do not need to stay changed -- leaving them changed meant a later
    # build() with no arguments silently inherited an earlier experiment.
    saved = {
        "mixing": high_fidelity_torch.H_TRANSFER_STATE_MIXING_FRACTION,
        "sato": getattr(high_fidelity_torch, "H_TRANSFER_SATO", None),
        "flatten": getattr(
            high_fidelity_torch, "H_TRANSFER_COUPLING_FLATTEN", None
        ),
        "cap": getattr(high_fidelity_torch, "H_TRANSFER_LOWERING_CAP", None),
        "softening": R.ENVIRONMENT_SOFTENING,
    }

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

    if depth_power is not None:
        if not hasattr(high_fidelity_torch, "H_TRANSFER_COUPLING_DEPTH_POWER"):
            raise SystemExit(
                "this build has no H_TRANSFER_COUPLING_DEPTH_POWER; apply "
                "the coupling patch first"
            )
        saved["depth_power"] = (
            high_fidelity_torch.H_TRANSFER_COUPLING_DEPTH_POWER
        )
        high_fidelity_torch.H_TRANSFER_COUPLING_DEPTH_POWER = float(
            depth_power
        )

    if over_weight is not None:
        if not hasattr(R, "OVER_COORDINATION_DEPTH_WEIGHT"):
            raise SystemExit(
                "this build has no OVER_COORDINATION_DEPTH_WEIGHT; apply "
                "the over-coordination patch first"
            )
        saved["over_weight"] = R.OVER_COORDINATION_DEPTH_WEIGHT
        R.OVER_COORDINATION_DEPTH_WEIGHT = float(over_weight)

    if softening is not None:
        # Same measure-before-committing pattern as the correction knobs.
        # Unlike those, this one reaches every molecule in the run, not just
        # a transferring hydrogen, so it is worth knowing what it does to the
        # reaction energy before switching it on in the source.
        import reactive as R
        R.ENVIRONMENT_SOFTENING = float(softening)

    if sato_pairs is not None:
        saved["sato_pairs"] = dict(high_fidelity_torch.H_TRANSFER_SATO_PAIRS)
        high_fidelity_torch.H_TRANSFER_SATO_PAIRS = dict(sato_pairs)

    if pair_depths:
        # Diagnostic only. Each entry is a single generic depth shared by
        # every atom of that pair, so overriding one asks what the barrier
        # does when a bond is stronger or weaker -- not what the depth ought
        # to be. The table values are dissociation energies checked against
        # CRC, so a value that improves a barrier here is evidence about the
        # barrier's dependence on depth, not evidence the table is wrong.
        #
        # The simulation copies the tables at construction, so these are set
        # before the object is built and restored immediately afterwards,
        # leaving the module untouched for anything else in the process.
        import reactive as R

        restore = []
        for pair, value in pair_depths.items():
            first, second = (part.strip() for part in pair.split("-"))
            i = int(R.ELEMENT_INDEX[first])
            j = int(R.ELEMENT_INDEX[second])
            restore.append((i, j, R.BOND_DEPTH[i, j], R.BOND_DEPTH[j, i]))
            R.BOND_DEPTH[i, j] = float(value)
            R.BOND_DEPTH[j, i] = float(value)

        prepared = boxes if boxes is not None else [
            (lambda pair: (pair[0], pair[1] + CENTRE))(
                active_geometry(*SYSTEM_START[ACTIVE_SYSTEM])
            )
        ]
        try:
            return cls(
                boxes=prepared,
                box_size=BOX,
                random_seed=0,
                relax_on_start=False,
                device=SCAN_DEVICE,
                dtype=SCAN_DTYPE,
            )
        finally:
            for i, j, first_value, second_value in restore:
                R.BOND_DEPTH[i, j] = first_value
                R.BOND_DEPTH[j, i] = second_value

    prepared = boxes if boxes is not None else [
        (lambda pair: (pair[0], pair[1] + CENTRE))(
            active_geometry(*SYSTEM_START[ACTIVE_SYSTEM])
        )
    ]
    try:
        return cls(
            boxes=prepared,
            box_size=BOX,
            random_seed=0,
            relax_on_start=False,
            device=SCAN_DEVICE,
            dtype=SCAN_DTYPE,
        )
    finally:
        high_fidelity_torch.H_TRANSFER_STATE_MIXING_FRACTION = saved["mixing"]
        if saved["sato"] is not None:
            high_fidelity_torch.H_TRANSFER_SATO = saved["sato"]
        if saved["flatten"] is not None:
            high_fidelity_torch.H_TRANSFER_COUPLING_FLATTEN = saved["flatten"]
        if hasattr(high_fidelity_torch, "H_TRANSFER_LOWERING_CAP"):
            high_fidelity_torch.H_TRANSFER_LOWERING_CAP = saved["cap"]
        R.ENVIRONMENT_SOFTENING = saved["softening"]
        if "sato_pairs" in saved:
            high_fidelity_torch.H_TRANSFER_SATO_PAIRS = saved["sato_pairs"]
        if "over_weight" in saved:
            R.OVER_COORDINATION_DEPTH_WEIGHT = saved["over_weight"]
        if "depth_power" in saved:
            high_fidelity_torch.H_TRANSFER_COUPLING_DEPTH_POWER = (
                saved["depth_power"]
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
    _, positions = active_geometry(
        donor_length, transfer_length, spectators
    )
    return energy_of(sim, positions)


def energy_and_position_gradient(sim, positions):
    """Total energy and its gradient with respect to every coordinate.

    The gradient is autograd's, so it is exact rather than a difference, and
    it costs one evaluation rather than one per coordinate.
    """
    moved = torch.tensor(
        positions + CENTRE, device=sim.device, dtype=sim.dtype,
        requires_grad=True,
    )

    # The neighbour table is built from detached positions, as elsewhere: it
    # is a discrete choice of which pairs to include, not part of the energy
    # expression, and differentiating through the search would be both
    # meaningless and expensive.
    sim.positions = moved.detach()
    sim.build_neighbours()

    total = torch.sum(sim.energy_per_atom(moved))
    gradient, = torch.autograd.grad(total, moved)

    # Detached before the conversion: torch warns about turning a tensor
    # that still carries its graph into a scalar, and the gradient has
    # already been taken by this point so there is nothing to keep.
    return float(total.detach()), gradient.detach().cpu().numpy()


def spectator_jacobian(donor_length, transfer_length, spectators, step=1e-5):
    """How every atom moves when each spectator coordinate is nudged.

    Finite difference, but only on the geometry builder, which is pure
    arithmetic with no simulation in it. So this costs a handful of numpy
    calls rather than a handful of energy evaluations, and the expensive
    derivative -- the energy's -- comes from autograd instead.

    Returns an array indexed [spectator, atom, axis].
    """
    spectators = np.asarray(spectators, dtype=float)

    columns = []
    for index in range(len(spectators)):
        forward = spectators.copy()
        backward = spectators.copy()
        forward[index] += step
        backward[index] -= step

        _, ahead = active_geometry(donor_length, transfer_length, forward)
        _, behind = active_geometry(donor_length, transfer_length, backward)

        columns.append((ahead - behind) / (2.0 * step))

    return np.stack(columns)


def relaxed_energy(sim, donor_length, transfer_length, start=None,
                   gradient_based=True):
    """Minimise over the spectator coordinates at fixed donor and transfer.

    Two methods, because they answer to different constraints.

    Powell needs no derivatives and copes with a surface that has flat
    directions, which is why it was here first. It pays for that in
    evaluations: fifty to two hundred per cell, and a relaxed grid is a
    thousand cells, so a single system takes minutes and a four system
    sweep takes most of an hour.

    The gradient method is far cheaper, and its derivative is not a finite
    difference on the energy. Autograd gives the exact gradient with respect
    to atom positions for the price of one evaluation, and the remaining
    piece -- how atoms move when a spectator coordinate changes -- is a
    difference on the geometry builder alone, which is arithmetic with no
    simulation in it. Chaining the two gives an essentially exact gradient
    for one energy evaluation rather than one per coordinate.

    Powell is kept and reachable because the two disagreeing is
    informative: on a flat surface they can settle in different local
    minima, and which one is lower says which converged better rather than
    which is right.
    """
    if start is None:
        start = active_frozen()

    start = np.clip(
        np.asarray(start, dtype=float),
        [low for low, _ in active_limits()],
        [high for _, high in active_limits()],
    )

    if not gradient_based:
        from scipy.optimize import minimize

        def cost(spectators):
            for value, (low, high) in zip(spectators, active_limits()):
                if not low <= value <= high:
                    return 1e6
            return energy_at(sim, donor_length, transfer_length, spectators)

        result = minimize(
            cost, start, method="Powell",
            options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 2000},
        )
        return float(result.fun), np.asarray(result.x, float)

    from scipy.optimize import minimize

    def cost_and_gradient(spectators):
        _, positions = active_geometry(
            donor_length, transfer_length, spectators
        )
        total, position_gradient = energy_and_position_gradient(
            sim, positions
        )

        jacobian = spectator_jacobian(
            donor_length, transfer_length, spectators
        )
        gradient = np.einsum("kij,ij->k", jacobian, position_gradient)

        return total, gradient

    result = minimize(
        cost_and_gradient, start, method="L-BFGS-B", jac=True,
        bounds=active_limits(),
        options={"ftol": 1e-10, "gtol": 1e-7, "maxiter": 200},
    )

    return float(result.fun), np.asarray(result.x, float)


BATCH_SIZE = 256


def batched_surface(physics, donor_lengths, transfer_lengths, progress=True,
                    progress_label="", batch_size=BATCH_SIZE, **build_kwargs):
    """Frozen energy grid, evaluated many geometries at a time.

    The simulation already carries several boxes at once -- that is what the
    characterisation runs use to get eight trajectories for the price of
    rather less than eight. The same machinery works for a scan: each grid
    cell becomes a box, and a whole chunk of the grid is one evaluation
    instead of one per cell.

    The neighbour table is built box by box and the indices shifted, so boxes
    never see each other despite sharing coordinates. Nothing here relies on
    that beyond the class already guaranteeing it.

    Frozen only. Relaxation needs a minimisation per cell, and the whole
    point of batching is that every cell is evaluated at the same time, so
    the two do not compose.
    """
    cells = [
        (i, j, donor, transfer)
        for i, donor in enumerate(donor_lengths)
        for j, transfer in enumerate(transfer_lengths)
    ]

    grid = np.empty((len(donor_lengths), len(transfer_lengths)))
    started = time.monotonic()

    for start in range(0, len(cells), batch_size):
        chunk = cells[start:start + batch_size]

        boxes = []
        for _, _, donor, transfer in chunk:
            symbols, positions = active_geometry(donor, transfer)
            boxes.append((symbols, positions + CENTRE))

        sim = build(physics, boxes=boxes, **build_kwargs)
        energies = sim.potential_per_box

        for (i, j, _, _), value in zip(chunk, energies):
            grid[i, j] = float(value)

        if progress:
            done = start + len(chunk)
            width = 28
            filled = int(width * done / len(cells))
            elapsed = time.monotonic() - started
            remaining = elapsed * (len(cells) - done) / max(done, 1)
            print(
                f"\r  {progress_label:<14}"
                f"[{'#' * filled}{'.' * (width - filled)}] "
                f"{done:>6}/{len(cells)} cells  "
                f"{done / max(elapsed, 1e-6):6.0f}/s  "
                f"{remaining:4.0f}s left ",
                end="", flush=True,
            )

    if progress:
        print(flush=True)

    return grid


def surface(sim, donor_lengths, transfer_lengths, relax=False, progress=True,
            record_spectators=False, progress_label="", physics=None,
            build_kwargs=None, gradient_based=True):
    """Energy grid, indexed [donor, transfer].

    With record_spectators the relaxed spectator coordinates are kept for
    every cell and returned alongside the grid. That is what lets a caller
    ask what geometry a saddle actually has, and whether the optimiser
    reached it by pinning a coordinate against its limit -- which would make
    the energy there a boundary artefact rather than a minimum.
    """
    started = time.monotonic()

    # Frozen grids go through the batched evaluator, which sends a few
    # hundred geometries per call instead of one. Relaxed grids cannot: each
    # cell needs its own minimisation, so there is nothing to batch.
    if not relax and physics is not None:
        grid = batched_surface(
            physics, donor_lengths, transfer_lengths,
            progress=progress, progress_label=progress_label,
            **(build_kwargs or {}),
        )
        if record_spectators:
            frozen = np.asarray(active_frozen(), dtype=float)
            spectators = np.broadcast_to(
                frozen,
                (len(donor_lengths), len(transfer_lengths), len(frozen)),
            ).copy()
            return grid, spectators
        return grid

    grid = np.empty((len(donor_lengths), len(transfer_lengths)))
    spectators = (
        np.empty((len(donor_lengths), len(transfer_lengths),
                  len(active_frozen())))
        if record_spectators else None
    )
    carried = None

    for i, donor in enumerate(donor_lengths):
        # Each row restarts from the row above rather than from the previous
        # column: stretching r(C-H) disturbs the spectators far less than
        # sweeping r(H-H) right across the transfer does.
        row_start = carried

        for j, transfer in enumerate(transfer_lengths):
            if relax:
                value, row_start = relaxed_energy(
                    sim, donor, transfer, start=row_start,
                    gradient_based=gradient_based,
                )
                if j == 0:
                    carried = row_start
                if spectators is not None:
                    spectators[i, j] = row_start
            else:
                value = energy_at(sim, donor, transfer)
                if spectators is not None:
                    spectators[i, j] = active_frozen()

            grid[i, j] = value

        if progress and relax:
            # Relaxation runs a minimisation at every cell, so a grid takes
            # minutes and prints nothing without this. Overwrites one line
            # rather than scrolling, and carries the label so several grids
            # in a row can be told apart.
            done = i + 1
            width = 28
            filled = int(width * done / len(donor_lengths))
            elapsed = time.monotonic() - started
            remaining = elapsed * (len(donor_lengths) - done) / max(done, 1)
            print(
                f"\r  {progress_label:<14}"
                f"[{'#' * filled}{'.' * (width - filled)}] "
                f"{done:>3}/{len(donor_lengths)}  "
                f"r {donor:.2f} A  "
                f"{elapsed:5.0f}s elapsed, {remaining:4.0f}s left ",
                end="", flush=True,
            )

    if progress and relax:
        print(flush=True)

    if spectators is not None:
        return grid, spectators
    return grid


# ----------------------------------------------------------------------
# Saddle by flooding
# ----------------------------------------------------------------------

def basin_seeds(grid, donor_lengths, transfer_lengths,
                detached_transfer=None, detached_donor=None,
                bonded_transfer=None):
    """Cells standing for the reactant and product basins.

    Reactant: the incoming hydrogen out past its cutoffs, molecule intact.
    Product:  the donor bond broken, the new H-H bond formed.
    """
    probe = SYSTEM_PROBES[ACTIVE_SYSTEM]
    if detached_transfer is None:
        detached_transfer = probe["detached_transfer"]
    if detached_donor is None:
        detached_donor = probe["detached_donor"]
    if bonded_transfer is None:
        bonded_transfer = probe["bonded_transfer"]

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

        inside = np.where(
            transfer_lengths < SYSTEM_PROBES[ACTIVE_SYSTEM]['product_below']
        )[0]
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
                      mixing=None, sato=None, flatten=None, cap=None,
                      gradient_based=True):
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
                relaxed_energy(
                    sim, donor, transfer, gradient_based=gradient_based
                )[0]
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
        inside = np.where(
            transfer_lengths < SYSTEM_PROBES[ACTIVE_SYSTEM]['product_below']
        )[0]
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


# Probe geometries, per system. The formaldehyde values are the ones a
# trapped trajectory actually held; the water ones are the symmetric shared
# proton and an equilibrium O-H, which are the analogous configurations.
SYSTEM_PROBES = {
    "formaldehyde": {"trap": (1.39, 0.92), "impact": 1.08,
                     "donor": (1.00, 1.90, 0.02),
                     "transfer": (0.65, 1.60, 0.005),
                     "product_below": 0.95, "detached_transfer": 1.30,
                     "detached_donor": 1.75, "bonded_transfer": 0.95},
    "water": {"trap": (1.20, 1.20), "impact": 0.98,
              "donor": (0.90, 1.80, 0.02),
              "transfer": (0.90, 1.80, 0.02),
              "product_below": 1.10, "detached_transfer": 1.55,
              "detached_donor": 1.55, "bonded_transfer": 1.10},
    # Same window and thresholds as formaldehyde on purpose: the two are
    # meant to be compared, and a different grid would leave any difference
    # arguable.
    "methane": {"trap": (1.39, 0.92), "impact": 1.09,
                "donor": (1.00, 1.90, 0.02),
                "transfer": (0.65, 1.60, 0.005),
                "product_below": 0.95, "detached_transfer": 1.30,
                "detached_donor": 1.75, "bonded_transfer": 0.95},
    # N-H is shorter than C-H, 1.01 against 1.09, and its outer cutoff is
    # 1.626 against 1.744, so the window starts lower and the detached donor
    # threshold comes down with it.
    "ammonia": {"trap": (1.35, 0.92), "impact": 1.01,
                "donor": (0.92, 1.85, 0.02),
                "transfer": (0.65, 1.60, 0.005),
                "product_below": 0.95, "detached_transfer": 1.30,
                "detached_donor": 1.65, "bonded_transfer": 0.95},
}

TRAP_DONOR = 1.39
TRAP_TRANSFER = 0.92
IMPACT_DONOR = 1.08


def apply_system(name):
    """Point the probe constants and basin thresholds at one system."""
    global ACTIVE_SYSTEM, TRAP_DONOR, TRAP_TRANSFER, IMPACT_DONOR
    ACTIVE_SYSTEM = name
    probe = SYSTEM_PROBES[name]
    TRAP_DONOR, TRAP_TRANSFER = probe["trap"]
    IMPACT_DONOR = probe["impact"]
    return probe

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
# Two conventions, and they must not be mixed.
#
#   electronic   the bare saddle on the potential surface, no vibrational
#                zero point energy anywhere
#   adiabatic    the saddle with zero point energy included at both the
#                reactants and the transition state
#
# From Table 1 of Kerkeni et al. 2022, gas phase columns. The unstarred row
# is electronic, the starred row includes ZPE; the surrounding prose calls
# both "barrier height", which is easy to misread.
#
#   CCSD(T)/cc-pVTZ        7.46 electronic   6.69 adiabatic
#   DLPNO-CCSD(T)/CBS      7.62 electronic   5.85 adiabatic
#   (U)CCSD(T)-F12         7.03 electronic
#
# This model is in the adiabatic convention, and not by choice: its
# BOND_DEPTH table holds dissociation energies, with H-H at 436 kJ/mol
# against a D0 of 435.99, and D0 includes zero point energy by definition.
# A potential fitted to D0 already carries ZPE inside its well depths, so
# the starred numbers are the ones to compare against.
#
# The reaction energy below is on the same footing: -0.718 eV comes from D0
# values, and the paper's ZPE-corrected gas phase exothermicity is -0.712 eV
# at CCSD(T)/cc-pVTZ. The electronic value is -0.619 eV. Switching the
# barrier to the electronic convention would mean switching this too.
REFERENCE_BARRIER = (0.254, 0.290)
REFERENCE_BARRIER_ELECTRONIC = (0.305, 0.330)

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

    inside = np.where(
            transfer_lengths < SYSTEM_PROBES[ACTIVE_SYSTEM]['product_below']
        )[0]
    if len(inside) == 0:
        return None

    product = (0, int(inside[np.argmin(profile[inside])]))
    entrance = (0, len(profile) - 1)

    cell, energy = flood_saddle(profile.reshape(1, -1), entrance, product)
    if cell is None or energy <= profile[entrance[1]] + 1e-3:
        return None

    return energy - profile[entrance[1]]


def locate_saddle(sim, donor_lengths, transfer_lengths, relax=False,
                  gradient_based=True):
    """Find the saddle on the grid rather than assuming where it sits.

    Earlier versions checked curvature at a hardcoded point, the geometry a
    trapped trajectory happened to hold.  That was the right check in the
    wrong place: once the topology changes the saddle moves, so the location
    has to be measured too, not carried over from the configuration that was
    being diagnosed.

    Returns (donor, transfer, height above the reactant minimum, grid), or
    None if the two basins cannot be separated on this window.
    """
    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax,
                   gradient_based=gradient_based)

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
                  mixing=None, relax=False, pair_depths=None,
                  gradient_based=True):
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
        sim = build(physics, mixing=mixing, cap=cap, pair_depths=pair_depths)

        found = locate_saddle(
            sim, donor_lengths, transfer_lengths, relax=relax,
            gradient_based=gradient_based,
        )
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


def locate_trap(grid, donor_lengths, transfer_lengths, saddle_cell,
                product_cell):
    """Find the spurious minimum on the product side of the saddle, if any.

    A well between the saddle and the products is not part of the reaction:
    it is an intermediate the model invented, and a trajectory that falls in
    has no reason to leave.  Rather than assume where it sits, this looks for
    a genuine local minimum in the region the reaction path passes through
    after the col, excluding the product basin itself.
    """
    rows, columns = grid.shape
    best = None

    for row in range(1, rows - 1):
        for column in range(1, columns - 1):
            # Past the col in the donor coordinate, before the products.
            if row <= saddle_cell[0] or row >= product_cell[0]:
                continue

            centre = grid[row, column]
            neighbourhood = grid[row - 1:row + 2, column - 1:column + 2]

            if centre > neighbourhood.min():
                continue

            if best is None or centre < best[2]:
                best = (row, column, float(centre))

    if best is None:
        return None

    row, column, energy = best
    return (
        float(donor_lengths[row]),
        float(transfer_lengths[column]),
        energy,
        (row, column),
    )


def escape_report(physics, donor_lengths, transfer_lengths, mixing=None,
                  cap=None, relax=False, pair_depths=None, softening=None,
                  depth_power=None, gradient_based=True):
    """How deep is the trap, and how hard is it to get out?

    The reaction has a real col.  What follows it is the problem: a bound
    minimum that captures roughly as often as the transfer completes.  Two
    numbers decide how much this matters.

        depth    how far the trap sits below the products it should become
        escape   the barrier a trapped trajectory has to climb to leave

    Depth alone is misleading.  A well 0.7 eV deep with a 0.05 eV lip is a
    speed bump: thermal motion drains it and a longer run would show the
    products forming late.  The same well behind a 0.6 eV lip is a dead end
    at any temperature this model runs at.  The escape barrier is the number
    that says which of those you have, and it is the one to fix against.
    """
    sim = build(physics, mixing=mixing, cap=cap, pair_depths=pair_depths,
                softening=softening, depth_power=depth_power)

    grid = surface(
        sim, donor_lengths, transfer_lengths, relax=relax, physics=physics,
        progress=True, progress_label=ACTIVE_SYSTEM,
        gradient_based=gradient_based,
        build_kwargs=dict(
            mixing=mixing, cap=cap, pair_depths=pair_depths,
            softening=softening, depth_power=depth_power,
        ),
    )
    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    if reactant is None:
        print("could not identify both basins - widen the scan window")
        return

    saddle_cell, saddle_energy = flood_saddle(grid, reactant, product)
    if saddle_cell is None:
        print("no connected route between the basins on this grid")
        return

    print("reactant  r(C-H) %.3f A, r(H-H) %.3f A"
          % (donor_lengths[reactant[0]], transfer_lengths[reactant[1]]))
    print("saddle    r(C-H) %.3f A, r(H-H) %.3f A, %+.3f eV above reactant"
          % (donor_lengths[saddle_cell[0]], transfer_lengths[saddle_cell[1]],
             saddle_energy - grid[reactant]))
    print("product   r(C-H) %.3f A, r(H-H) %.3f A, %+.3f eV\n"
          % (donor_lengths[product[0]], transfer_lengths[product[1]],
             grid[product] - grid[reactant]))

    found = locate_trap(grid, donor_lengths, transfer_lengths,
                        saddle_cell, product)
    if found is None:
        print("no spurious minimum between the saddle and the products")
        return

    donor, transfer, energy, cell = found

    print("TRAP")
    print("  r(C-H)            %.3f A" % donor)
    print("  r(H-H)            %.3f A" % transfer)
    print("  depth below products  %+.3f eV" % (energy - grid[product]))
    print("  height above reactant %+.3f eV" % (energy - grid[reactant]))

    # Escape barrier: flood from the trap to the products and read the level
    # at which they join.  Same construction as the reaction saddle, so it is
    # the true easiest way out rather than a guess at the exit direction.
    exit_cell, exit_energy = flood_saddle(grid, cell, product)
    if exit_cell is None:
        print("  escape barrier    no route out to the products at all")
        return

    print("  escape barrier    %+.3f eV, over r(C-H) %.3f A, r(H-H) %.3f A"
          % (exit_energy - energy,
             donor_lengths[exit_cell[0]], transfer_lengths[exit_cell[1]]))

    barrier = exit_energy - energy
    thermal = 8.617e-5 * 250.0
    print("\n  kT at 250 K is %.4f eV, so the escape barrier is %.0f kT."
          % (thermal, barrier / thermal))
    if barrier < 10 * thermal:
        print("  shallow enough that thermal motion should drain it; a longer")
        print("  run would show the products forming late rather than never")
    else:
        print("  deep enough to hold a trajectory for the whole run at this")
        print("  temperature, so this is a dead end rather than a delay")


def proton_transfer_report(physics, separations=(2.60, 2.70, 2.80, 3.00),
                           mixing=None, cap=None, softening=None):
    """Symmetric proton transfer between two waters, at fixed O-O distances.

    This is the independent check on everything fitted to formaldehyde. The
    correction has never seen these tables, the environment softening is
    inert because neither oxygen holds a multiple bond, and the donor and
    acceptor are the same element so the argmax that labels them swaps its
    choice exactly at the midpoint.

    Three things are reported and only the first is a matter of taste:

        asymmetry   the largest energy difference between mirror geometries,
                    which must be zero and is not a parameter choice
        barrier     the central barrier, which should fall as the oxygens
                    approach and vanish somewhere near 2.4 to 2.5 A
        shape       whether the midpoint is a maximum, as it must be at long
                    separation, or a minimum, which would be the shared
                    proton trapped exactly as it was in formaldehyde
    """
    sim = build(physics, mixing=mixing, cap=cap, softening=softening)

    print("symmetric proton transfer, H2O ... H ... OH2")
    print("heavy atoms fixed; the proton moves along the O-O axis\n")
    print(f"{'O-O':>6} {'asymmetry':>11} {'well at':>9} {'barrier':>10}  shape")

    for separation in separations:
        offsets = np.arange(-0.60, 0.6001, 0.005)
        energies = np.array([
            energy_of(sim, water_pair_geometry(separation, offset)[1])
            for offset in offsets
        ])

        # Mirror check: the curve reversed must equal the curve.
        asymmetry = float(np.max(np.abs(energies - energies[::-1])))

        middle = len(offsets) // 2
        centre = energies[middle]

        left = energies[:middle]
        well = int(np.argmin(left))
        barrier = centre - left[well]

        if barrier > 1e-3:
            shape = "double well, proton localised"
        elif barrier < -1e-3:
            shape = "MIDPOINT IS A MINIMUM (shared proton trapped)"
        else:
            shape = "flat"

        print(f"{separation:6.2f} {asymmetry:9.4f} eV "
              f"{offsets[well]:+8.3f}A {barrier:8.4f} eV  {shape}")

    print("\nasymmetry must be zero: the geometry is its own mirror image, so")
    print("any departure is the correction failing to vanish on one side.")


# Computed barriers, in eV, for the three systems the scanner can build.
#
# CONVENTION: classical. The bare height of the potential surface, saddle
# minus reactants, with no vibrational zero point energy anywhere. That is
# what the scanner measures, since it floods a grid of static geometries,
# and what the model can represent, since it integrates Newton's equations
# on a static surface with no vibrational levels of its own.
#
# The literature also quotes adiabatic barriers, which include the zero point
# difference between the transition state and the reactants. Mixing the two
# conventions happened three times while assembling this table, always in
# the same direction, because the adiabatic figure is the one usually
# quoted in an abstract:
#
#   formaldehyde  6.69 was taken for 7.46, the target 0.034 eV too low
#   methane       first 12.1 from a reverse barrier, then 14.1 which is the
#                 vibrationally adiabatic value, not the classical 0.64 eV
#
# So: do not convert between conventions, and do not subtract an estimated
# zero point correction. Find a value the source states as classical. The
# two are not even at the same geometry -- for H + CH4 the adiabatic maximum
# sits at s = +0.10 rather than at the saddle -- so a constant offset
# between them does not exist to apply.
#
#   formaldehyde  7.46 kcal/mol, electronic barrier, CCSD(T)/cc-pVTZ//MP2,
#                 Table 1 of Kerkeni et al. 2022 (the 6.69 in the same table
#                 is the starred, ZPE-corrected row)
#   water         8.4 kcal/mol, symmetric OH + H2O exchange, CCSD(T) with
#                 CCSDT(Q) corrections, Schaefer and co-workers 2016
#   methane       0.64 eV, classical barrier at the transition state of the
#                 permutationally invariant full-dimensional ab initio PES
#                 for H + CH4, J. Chem. Phys. 142, 204302 (2015). Quoted in
#                 eV in the source, so no conversion of mine sits between
#                 the paper and this table.
#
# Water is a range, and not because its value is uncertain. The transition
# state is the same; what differs is where the counting starts.
#
# The same paper that gives 8.4 kcal/mol from separated reactants also
# locates two entrance complexes, 3.7 and 2.1 kcal/mol below them. A real
# encounter falls into one of those and climbs from there, so measured from
# a complex the barrier is 10.5 to 12.1 kcal/mol, or 0.455 to 0.525 eV.
#
# The model has no long-range attraction: past the O-H cutoff of 1.536 A the
# two molecules stop interacting entirely, and the scan's reactant energy is
# flat from 1.70 A outward. So it cannot form either complex, and neither
# reference state is unambiguously the one it should be judged against.
# Recording a single number would bake that modelling choice into any fit
# without saying so.
#
# Note the upper figure is arithmetic on two quoted numbers rather than a
# third quoted number: 8.4 plus the complex depth. Sound, but not the same
# kind of evidence as the 8.4 itself.
#
# The other two are single published values, stored as equal pairs so the
# printing code can treat every entry the same way.
REFERENCE_BARRIERS = {
    "formaldehyde": (0.324, 0.324),
    "water": (0.364, 0.525),
    "methane": (0.640, 0.640),
}


# Grid for reading a barrier, as opposed to locating a saddle precisely.
# The fine grid is 8,550 points to extract one number; a barrier is flat
# enough near its top that a quarter of the points give the same value to
# three decimals, and the sweep builds one surface per cell.
BARRIER_DONOR_STEP = 0.04
BARRIER_TRANSFER_STEP = 0.01


SPECTATOR_NAMES = {
    "ammonia": ("r(N-H) 1", "r(N-H) 2", "angle"),
    "formaldehyde": ("r(C=O)", "r(C-H) spectator", "donor angle from C=O",
                     "spectator angle from C=O"),
    "methane": ("r(C-H) 1", "r(C-H) 2", "r(C-H) 3", "angle"),
    # The angles are from the transfer axis, so they are complements of the
    # bond angle and sum to 180 with it. Named to say so, because "donor
    # angle" reading 75.53 looks like a badly distorted water and is not.
    "water": ("r(O-H) donor", "r(O-H) acceptor", "donor 180-HOH",
              "acceptor 180-HOH"),
}


def describe_spectators(name, values, tolerance=1e-3):
    """Report the relaxed spectator coordinates and flag any at a limit.

    A coordinate sitting on its bound is not a minimum, it is the edge of
    the box the optimiser was allowed to search. The energy there says more
    about the limits than about the molecule, so anything pinned has to be
    visible rather than folded into a single number.
    """
    labels = SPECTATOR_NAMES.get(name)
    limits = SYSTEMS[name]["limits"]

    lines = []
    pinned = []
    for index, (value, (low, high)) in enumerate(zip(values, limits)):
        label = labels[index] if labels else f"spectator {index}"
        mark = ""
        if value <= low + tolerance:
            mark = f"  AT LOWER LIMIT {low:g}"
            pinned.append(label)
        elif value >= high - tolerance:
            mark = f"  AT UPPER LIMIT {high:g}"
            pinned.append(label)
        lines.append(f"    {label:>20} {value:8.3f}{mark}")

    return lines, pinned


def measure_barrier(physics, name, mixing=None, power=None, over_weight=None,
                    sato=None, sato_pairs=None, relax=False,
                    gradient_based=True,
                    donor_step=BARRIER_DONOR_STEP,
                    transfer_step=BARRIER_TRANSFER_STEP):
    """Barrier, reaction energy and saddle geometry for one setting.

    Relaxation is off by default because a relaxed grid costs a constrained
    minimisation per cell. It matters more than that cost suggests: holding
    the spectators rigid put a 0.3 eV artefact in formaldehyde's reaction
    energy and moved water's barrier by 0.15 eV, and it moved the three
    systems by different amounts, so a frozen comparison between them is not
    trustworthy even when each number looks reasonable on its own.
    """
    apply_system(name)
    sim = build(
        physics, mixing=mixing, depth_power=power, over_weight=over_weight,
        sato=sato, sato_pairs=sato_pairs,
    )

    # Window from the system, not hardcoded. These were formaldehyde's
    # bounds, which meant water was scanned from donor 1.00 when its O-H sits
    # at 0.96, clipping the reactant minimum off the edge of the grid.
    probe = SYSTEM_PROBES[name]
    donor_low, donor_high, _ = probe["donor"]
    transfer_low, transfer_high, _ = probe["transfer"]

    # Relaxing runs a minimisation at every cell, so the grid coarsens when
    # it is switched on -- the same trade the main scan makes. This matches
    # the step the relaxed escape runs use, so the two agree rather than
    # differing by grid alone. It moves a barrier by a millivolt or two,
    # which is far below the errors being measured.
    if relax:
        transfer_step = max(transfer_step, 0.02)

    donor_lengths = np.arange(donor_low, donor_high, donor_step)
    transfer_lengths = np.arange(transfer_low, transfer_high, transfer_step)

    result = surface(
        sim, donor_lengths, transfer_lengths, relax=relax, progress=relax,
        record_spectators=True, progress_label=name, physics=physics,
        gradient_based=gradient_based,
        build_kwargs=dict(
            mixing=mixing, depth_power=power, over_weight=over_weight,
            sato=sato, sato_pairs=sato_pairs,
        ),
    )
    grid, spectators = result

    reactant, product = basin_seeds(grid, donor_lengths, transfer_lengths)
    if reactant is None:
        return None

    cell, energy = flood_saddle(grid, reactant, product)
    if cell is None:
        return None

    return {
        "barrier": float(energy - grid[reactant]),
        "reaction": float(grid[product] - grid[reactant]),
        "donor": float(donor_lengths[cell[0]]),
        "transfer": float(transfer_lengths[cell[1]]),
        "spectators": spectators[cell],
        "reactant_donor": float(donor_lengths[reactant[0]]),
        "reactant_transfer": float(transfer_lengths[reactant[1]]),
    }


def agreement_report(physics, relax=False, mixing=None, power=None,
                     over_weight=None, sato=None, gradient_based=True):
    """Do the two measurement paths give the same barrier?

    measure_barrier and escape_report both locate a saddle by flooding, but
    they were written separately and diverged once already: measure_barrier
    hardcoded formaldehyde's scan window, so water was measured on the wrong
    grid and reported three different barriers from nominally identical
    settings. Nothing catches that except comparing them directly.

    Same system, same window, same parameters, same relaxation. Any
    difference beyond the grid step is a bug in one of the two.
    """
    import high_fidelity_torch

    print("barrier from each path, same settings")
    print(f"spectators: {'relaxed' if relax else 'frozen'}")
    # Printed because this used to take no parameters at all and silently
    # measured whatever the source held, which made its numbers look
    # comparable to a sweep run at a different mixing when they were not.
    print("mixing %s, depth power %s, over weight %s, sato %s" % (
        mixing if mixing is not None
        else f"{high_fidelity_torch.H_TRANSFER_STATE_MIXING_FRACTION} (source)",
        power if power is not None else "1.0 (source)",
        over_weight if over_weight is not None else "0.0 (source)",
        sato if sato is not None else "0.0 (source)",
    ))
    print()
    print(f"{'system':>14}{'measure_barrier':>18}{'escape path':>14}"
          f"{'difference':>12}")

    for name in ("formaldehyde", "water", "methane"):
        probe = apply_system(name)
        donor_low, donor_high, donor_step = probe["donor"]
        transfer_low, transfer_high, transfer_step = probe["transfer"]

        # Coarsen when relaxing, exactly as measure_barrier does. Without
        # this the check ran on the fine per-system step -- 45 rows rather
        # than 23, six grids of it, hours rather than minutes -- and worse,
        # it compared measure_barrier's coarse grid against a fine one, so a
        # difference in grid would have shown up as a difference in path.
        if relax:
            donor_step = max(donor_step, BARRIER_DONOR_STEP)
            transfer_step = max(transfer_step, 0.02)

        found = measure_barrier(
            physics, name, relax=relax, mixing=mixing, power=power,
            over_weight=over_weight, sato=sato,
            gradient_based=gradient_based,
            donor_step=donor_step, transfer_step=transfer_step,
        )

        # The escape path, inlined at the same window and step.
        apply_system(name)
        sim = build(
            physics, mixing=mixing, depth_power=power,
            over_weight=over_weight, sato=sato,
        )
        donor_lengths = np.arange(donor_low, donor_high, donor_step)
        transfer_lengths = np.arange(
            transfer_low, transfer_high, transfer_step
        )
        grid = surface(
            sim, donor_lengths, transfer_lengths, relax=relax,
            progress=relax, progress_label=f"{name} direct", physics=physics,
            gradient_based=gradient_based,
            build_kwargs=dict(
                mixing=mixing, depth_power=power, over_weight=over_weight,
                sato=sato,
            ),
        )
        reactant, product = basin_seeds(
            grid, donor_lengths, transfer_lengths
        )
        cell, energy = flood_saddle(grid, reactant, product)
        direct = float(energy - grid[reactant])

        difference = found["barrier"] - direct
        flag = "" if abs(difference) < 1e-6 else "   MISMATCH"
        print(f"{name:>14}{found['barrier']:16.4f}{direct:14.4f}"
              f"{difference:+12.6f}{flag}")

    print("\nthe two paths share flood_saddle and basin_seeds, so identical")
    print("windows must give identical answers; anything else is a bug")


def slope_report(physics, powers=(1.00, 0.75, 0.50, 0.25, 0.00),
                 mixings=None, over_weights=None, satos=None,
                 sato_pairs=None, relax=False, gradient_based=True,
                 systems=("formaldehyde", "water", "methane")):
    """Sweep both coupling knobs against every system with a known barrier.

    There are two separate errors and one knob each, which is why they have
    to be looked at together.

        depth power   how much a difference in bond strength shows up as a
                      difference in barrier. At one the coupling scales with
                      the depths it couples, which cancels most of the effect
                      and leaves an Evans-Polanyi slope near 0.09 against a
                      physical 0.3 to 0.5. Lowering it rotates the barriers
                      apart about a fixed reference depth.

        mixing        the overall size of the coupling, and so how far every
                      barrier sits below the diabatic crossing. This moves
                      all three together rather than separating them.

    They are not quite independent: mixing multiplies the whole coupling, so
    it changes the spacing a little as well as the height. Expect to iterate
    rather than read a single answer off one pass.

    Each cell shows the barrier and how far it sits from the middle of that
    system's reference range, so a column of small errors is the target
    rather than any single number.
    """
    if mixings is None:
        mixings = (0.63,)
    if over_weights is None:
        over_weights = (0.0,)
    if satos is None:
        satos = (0.0,)

    print("barrier and error against the computed reference, in eV")
    print("references: " + ", ".join(
        f"{name} {REFERENCE_BARRIERS[name][0]:.3f}"
        + ("" if REFERENCE_BARRIERS[name][0] == REFERENCE_BARRIERS[name][1]
           else f"-{REFERENCE_BARRIERS[name][1]:.3f}")
        for name in systems if name in REFERENCE_BARRIERS
    ))
    print("power 1.00 with mixing 0.63 is the current model\n")

    header = "".join(f"{name:>20}" for name in systems)
    print(f"{'mixing':>7}{'power':>7}{'over':>6}{'sato':>6}{header}"
          f"{'spread':>9}{'worst':>8}")

    measured = {}

    for sato in satos:
     for over_weight in over_weights:
      for mixing in mixings:
        for power in powers:
            cells = []
            barriers = []
            errors = []
            pinned = []

            for name in systems:
                found = measure_barrier(
                    physics, name, mixing=mixing, power=power,
                    over_weight=over_weight, sato=sato,
                    sato_pairs=sato_pairs, relax=relax,
                    gradient_based=gradient_based,
                )
                if found is None:
                    cells.append(f"{'no route':>20}")
                    continue

                # Kept so the geometry report below can use it. It used to
                # call measure_barrier again for every system, which meant a
                # single-setting relaxed sweep built each grid twice and took
                # twice as long as it needed to.
                measured[name] = found

                height = found["barrier"]
                pinned.extend(
                    (name, label)
                    for label in describe_spectators(
                        name, found["spectators"]
                    )[1]
                )

                barriers.append(height)
                low, high = REFERENCE_BARRIERS[name]

                # Distance to the reference range rather than to a midpoint.
                # A barrier inside the range has no error to report: where
                # the two ends are different reference states rather than
                # rival estimates, landing between them is not a miss.
                if height < low:
                    error = height - low
                elif height > high:
                    error = height - high
                else:
                    error = 0.0

                errors.append(abs(error))
                cells.append(f"{height:8.3f} {error:+9.3f}")

            if len(barriers) > 1:
                spread = f"{max(barriers) - min(barriers):8.3f}"
            else:
                spread = f"{'-':>8}"
            worst = f"{max(errors):7.3f}" if errors else f"{'-':>7}"

            print(f"{mixing:7.2f}{power:7.2f}{over_weight:6.2f}{sato:6.2f}"
                  + "".join(cells) + spread + worst
                  + ("   PINNED" if pinned else ""))
            for name, label in pinned:
                print(f"{'':>26}{name} saddle has {label} at a limit")

    if not relax:
        print("\nfrozen spectators: a comparison between systems on a frozen")
        print("grid is not trustworthy, since they relax by different amounts")
        print("(0.02 eV for formaldehyde, 0.15 for water). Use --relax.")

    if len(mixings) * len(powers) * len(over_weights) * len(satos) == 1:
        for name in systems:
            found = measured.get(name)
            if found is None:
                continue
            print(f"\n{name} saddle: r(donor) {found['donor']:.3f} A, "
                  f"r(transfer) {found['transfer']:.3f} A")
            lines, _ = describe_spectators(name, found["spectators"])
            for line in lines:
                print(line)

    print("\neach cell is  barrier / distance outside the reference range,")
    print("which is zero for a barrier that lands inside it")
    print("spread is the widest gap between systems; it should grow as the")
    print("power falls, since the references span 0.27 to 0.59 eV")
    print("worst is the largest single error, which is the number to minimise")


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


def measure(sim, donor_lengths, transfer_lengths, relax=False,
            gradient_based=True):
    """Saddle position, activation energy and reaction energy for one model."""
    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax,
                   gradient_based=gradient_based)

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


def sweep(physics, donor_lengths, transfer_lengths, values, relax=False,
          gradient_based=True):
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
        found = measure(
            sim, donor_lengths, transfer_lengths, relax=relax,
            gradient_based=gradient_based,
        )
        if found is None:
            print(f"{value:8.3f} {'no route':>10}")
            continue
        cell = found["saddle_cell"]
        print(f"{value:8.3f} {found['barrier']:8.3f} eV "
              f"{found['reaction']:+9.3f} eV "
              f"{donor_lengths[cell[0]]:13.3f} A "
              f"{transfer_lengths[cell[1]]:7.3f} A")


def report(physics, donor_lengths, transfer_lengths, relax=False, plot=None,
           gradient_based=True,
           mixing=None, sato=None, flatten=None, cap=None, pair_depths=None,
           softening=None):
    sim = build(physics, mixing=mixing, sato=sato, flatten=flatten,
                cap=cap, pair_depths=pair_depths, softening=softening)

    print(f"physics: {getattr(sim, 'physics_model_name', 'reactive base')}")
    print(f"grid: {len(donor_lengths)} x {len(transfer_lengths)} points")
    print(f"spectators: {'relaxed' if relax else 'frozen at formaldehyde'}")
    if mixing is not None:
        print(f"state mixing fraction: {mixing} (overridden)")
    print()

    grid = surface(sim, donor_lengths, transfer_lengths, relax=relax,
                   gradient_based=gradient_based)

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
        choices=["high_fidelity", "h_state", "base"],
    )
    parser.add_argument(
        "--powell", action="store_true",
        help=(
            "relax with Powell instead of the gradient method. Slower by a "
            "large factor and kept because the two disagreeing is "
            "informative: on a flat surface they can settle in different "
            "local minima, and whichever finds the lower energy converged "
            "better"
        ),
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
        "--softening", type=float, default=None,
        help=(
            "override ENVIRONMENT_SOFTENING for this run; 0.124 makes a\n"
            "carbonyl C-H match its 3.760 eV thermochemical depth"
        ),
    )
    parser.add_argument(
        "--pair-depth", action="append", default=None,
        metavar="PAIR=eV",
        help=(
            "override one bond depth for this run only, e.g. O-H=4.799. "
            "Repeatable. Diagnostic: the table holds checked dissociation "
            "energies, so this asks how the barrier depends on depth rather "
            "than what the depth should be"
        ),
    )
    parser.add_argument(
        "--system", default="formaldehyde", choices=sorted(SYSTEMS),
        help=(
            "which transfer to scan; water is the independent check, with "
            "different tables, no environment softening, and a symmetric "
            "crossing that must come out at zero reaction energy"
        ),
    )
    parser.add_argument(
        "--depth-power", type=float, default=None,
        help="override H_TRANSFER_COUPLING_DEPTH_POWER for this run",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device for the scan; cuda is worth trying but these "
             "systems are small enough that it may not help",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="run in float32; enough for barriers, not for continuity checks",
    )
    parser.add_argument(
        "--mixings", default=None,
        help=(
            "comma separated mixing fractions for --slope; the coupling's "
            "overall size, which moves every barrier together"
        ),
    )
    parser.add_argument(
        "--sato-pair", action="append", default=None, metavar="PAIR=value",
        help=(
            "per-pair Sato value, e.g. O-H=0.2. Repeatable. This is the one "
            "axis that can move water without moving the carbon systems, "
            "since water transfers O-H to O-H and they transfer C-H to H-H"
        ),
    )
    parser.add_argument(
        "--satos", default=None,
        help=(
            "comma separated H_TRANSFER_SATO values for --slope; blends the "
            "unoccupied partner's curve from bare Morse repulsion toward a "
            "genuinely repulsive anti-Morse. It was set aside earlier for "
            "moving the barrier and the three-centre trap together, but that "
            "was judged on formaldehyde alone; the question here is whether "
            "it separates systems rather than what it does to one"
        ),
    )
    parser.add_argument(
        "--over-weights", default=None,
        help=(
            "comma separated OVER_COORDINATION_DEPTH_WEIGHT values for "
            "--slope; scales the penalty by each atom's mean contact depth, "
            "which is the only term in the barrier that can tell one element "
            "from another"
        ),
    )
    parser.add_argument(
        "--powers", default=None,
        help="comma separated depth powers for --slope",
    )
    parser.add_argument(
        "--agreement", action="store_true",
        help=(
            "check that measure_barrier and the escape path give the same "
            "barrier at the same window and settings"
        ),
    )
    parser.add_argument(
        "--slope", action="store_true",
        help=(
            "compare formaldehyde and methane barriers across the coupling "
            "depth power, to see whether that scaling is what flattens the "
            "model's response to bond strength"
        ),
    )
    parser.add_argument(
        "--proton-transfer", action="store_true",
        help=(
            "symmetric water proton transfer at fixed O-O separations; the "
            "independent check on parameters fitted to formaldehyde"
        ),
    )
    parser.add_argument(
        "--escape", action="store_true",
        help=(
            "locate the spurious minimum past the saddle and measure how "
            "hard it is to escape from"
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

    global SCAN_DEVICE, SCAN_DTYPE
    SCAN_DEVICE = options.device
    SCAN_DTYPE = torch.float32 if options.fast else torch.float64

    sato_pairs = {}
    for item in (options.sato_pair or []):
        pair, _, value = item.partition("=")
        sato_pairs[pair] = float(value)
    options.sato_pairs = sato_pairs or None

    pair_depths = {}
    for item in (options.pair_depth or []):
        pair, _, value = item.partition("=")
        pair_depths[pair] = float(value)
    options.pair_depths = pair_depths or None

    problems = check_systems_registered()
    if problems:
        raise SystemExit(
            "system tables are inconsistent:\n  " + "\n  ".join(problems)
        )

    probe = apply_system(options.system)

    # Relaxing costs a constrained minimisation per cell, so the default grid
    # coarsens when it is switched on.  Both remain fine enough to place the
    # saddle well inside the accuracy the model itself claims.
    donor_low, donor_high, donor_default = probe["donor"]
    transfer_low, transfer_high, transfer_default = probe["transfer"]

    if options.donor_min == 1.00:
        options.donor_min = donor_low
    if options.donor_max == 1.90:
        options.donor_max = donor_high
    if options.transfer_min == 0.65:
        options.transfer_min = transfer_low
    if options.transfer_max == 1.60:
        options.transfer_max = transfer_high

    # Both coarsen by two when relaxing. The transfer step used to coarsen by
    # four, which gave water twelve columns against measure_barrier's
    # forty-five and reported a barrier of 0.004 eV where the finer grid finds
    # 0.349: too coarse to resolve a col at all. The row counts matched, so
    # both paths printed the same 23/23 and the difference was invisible.
    donor_step = options.donor_step or (
        (donor_default * 2 if options.relax else donor_default)
    )
    transfer_step = options.transfer_step or (
        (transfer_default * 2 if options.relax else transfer_default)
    )

    donor_lengths = np.arange(options.donor_min, options.donor_max, donor_step)
    transfer_lengths = np.arange(
        options.transfer_min, options.transfer_max, transfer_step
    )

    gradient_based = not options.powell

    if options.agreement:
        agreement_report(
            options.physics, relax=options.relax, mixing=options.mixing,
            gradient_based=gradient_based,
            power=options.depth_power,
            over_weight=(
                float(options.over_weights.split(",")[0])
                if options.over_weights else None
            ),
            sato=options.sato,
        )
        return

    if options.slope:
        slope_report(
            options.physics,
            powers=(
                tuple(float(part) for part in options.powers.split(","))
                if options.powers else (1.00, 0.75, 0.50, 0.25, 0.00)
            ),
            mixings=(
                tuple(float(part) for part in options.mixings.split(","))
                if options.mixings else None
            ),
            over_weights=(
                tuple(float(part) for part in options.over_weights.split(","))
                if options.over_weights else None
            ),
            satos=(
                tuple(float(part) for part in options.satos.split(","))
                if options.satos else None
            ),
            sato_pairs=options.sato_pairs,
            relax=options.relax, gradient_based=gradient_based,
        )
        return

    if options.proton_transfer:
        apply_system("water")
        proton_transfer_report(
            options.physics, mixing=options.mixing, cap=options.cap,
            softening=options.softening,
        )
        return

    if options.escape:
        escape_report(
            options.physics, donor_lengths, transfer_lengths,
            mixing=options.mixing, cap=options.cap, relax=options.relax,
            pair_depths=options.pair_depths, softening=options.softening,
            depth_power=options.depth_power, gradient_based=gradient_based,
        )
        return

    if options.saddle_check:
        saddle_report(
            options.physics, donor_lengths, transfer_lengths,
            caps=[-1.0, 1.45, 1.36, 1.32, 1.29, 1.20],
            mixing=options.mixing, relax=options.relax,
            pair_depths=options.pair_depths,
            gradient_based=gradient_based,
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
            gradient_based=gradient_based,
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
        pair_depths=options.pair_depths,
        softening=options.softening,
        gradient_based=gradient_based,
    )


if __name__ == "__main__":
    main()
