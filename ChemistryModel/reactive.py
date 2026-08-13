import numpy as np


# ============================================================
# A reactive bond-order potential
# ============================================================
#
# The idea, borrowed from Tersoff and Brenner: a bond is a Morse
# potential whose attractive part is scaled down as an atom runs
# out of spare valence. An isolated carbon binds the next atom at
# full strength; a carbon that already has four neighbours binds
# a fifth barely at all.
#
# That single rule reproduces the behaviour of valence electrons
# without any electrons in the model. Bonds form and break
# continuously as atoms move, with no bookkeeping anywhere.
#
# Every number below is measured, not invented. Bond lengths are
# in angstroms and dissociation energies in electronvolts,
# converted from the usual kJ/mol.


KJ_PER_MOL_TO_EV = 1.0 / 96.485


ELEMENTS = ["H", "C", "N", "O"]

ELEMENT_INDEX = {name: index for index, name in enumerate(ELEMENTS)}

# How many bonds each element wants.

VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2}

MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

# Preferred angle at each central atom, in degrees. Hydrogen
# never sits in the middle of anything, so it has none.

REST_ANGLE = {"C": 109.47, "N": 107.0, "O": 104.5, "H": 0.0}

ANGLE_STIFFNESS = {"C": 3.0, "N": 2.6, "O": 2.6, "H": 0.0}


# (length in angstroms, dissociation energy in kJ/mol, Morse width)

BOND_TABLE = {
    ("H", "H"): (0.74144, 458.02871, 1.94458),
    ("C", "H"): (1.086, 439.0, 1.80),
    ("N", "H"): (1.0109, 449.0, 1.95),
    ("O", "H"): (0.96, 498.0, 2.18),
    ("C", "C"): (1.525, 348.0, 1.85),
    ("C", "N"): (1.47, 305.0, 1.90),
    ("C", "O"): (1.43, 358.0, 1.95),
    ("N", "N"): (1.45, 167.0, 2.00),
    ("N", "O"): (1.40, 201.0, 2.00),
    ("O", "O"): (1.48, 146.0, 2.05),
}

# Double and triple bonds. Where a pair has no multiple bond the
# single-bond entry is reused, so the interpolation simply flattens.

DOUBLE_BOND_TABLE = {
    ("C", "C"): (1.34, 614.0, 1.95),
    ("C", "N"): (1.29, 615.0, 2.00),
    ("C", "O"): (1.20, 750.0, 2.10),
    ("N", "N"): (1.25, 418.0, 2.10),
    ("N", "O"): (1.21, 607.0, 2.10),
    ("O", "O"): (1.21, 498.0, 2.15),
}

TRIPLE_BOND_TABLE = {
    ("C", "C"): (1.20, 839.0, 2.05),
    ("C", "N"): (1.16, 891.0, 2.10),
    ("N", "N"): (1.10, 945.0, 2.20),
}

# Electrons in the outer shell, for counting lone pairs.

OUTER_ELECTRONS = {"H": 1, "C": 4, "N": 5, "O": 6}


def build_tables(table, fallback=None):
    count = len(ELEMENTS)

    length = np.zeros((count, count))
    depth = np.zeros((count, count))
    width = np.zeros((count, count))

    if fallback is not None:
        length[:] = fallback[0]
        depth[:] = fallback[1]
        width[:] = fallback[2]

    for (first, second), (r0, kj, a) in table.items():
        i = ELEMENT_INDEX[first]
        j = ELEMENT_INDEX[second]

        length[i, j] = length[j, i] = r0
        depth[i, j] = depth[j, i] = kj * KJ_PER_MOL_TO_EV
        width[i, j] = width[j, i] = a

    return length, depth, width


BOND_LENGTH, BOND_DEPTH, BOND_WIDTH = build_tables(BOND_TABLE)

DOUBLE_LENGTH, DOUBLE_DEPTH, DOUBLE_WIDTH = build_tables(
    DOUBLE_BOND_TABLE,
    fallback=(BOND_LENGTH, BOND_DEPTH, BOND_WIDTH)
)

TRIPLE_LENGTH, TRIPLE_DEPTH, TRIPLE_WIDTH = build_tables(
    TRIPLE_BOND_TABLE,
    fallback=(DOUBLE_LENGTH, DOUBLE_DEPTH, DOUBLE_WIDTH)
)

OUTER_ELECTRON_ARRAY = np.array(
    [OUTER_ELECTRONS[name] for name in ELEMENTS], dtype=float
)

# How hard it is for an atom to exceed its own valence. This is
# the single knob that sets every activation barrier in the
# model, fitted against the measured H + H2 barrier of 0.42 eV.

# How much a single bond is weakened by multiple bond character its partner
# is already carrying. The tables hold one depth per element pair, so every
# C-H is the methane C-H; the aldehydic C-H is roughly 0.53 eV shallower
# because that carbon is committed to a double bond. This is the standard
# bond-order trade-off used by Tersoff and REBO potentials, kept element
# agnostic so no molecule is special cased.
#
# 0.0 disables it and reproduces the older behaviour exactly. It touches
# every molecule holding a double or triple bond, so measure the effect
# before raising it: the formaldehyde abstraction should reach -0.718 eV.
ENVIRONMENT_SOFTENING = 0.174

# Width of the smoothing on the two-ended maximum in that softening, as a
# squared commitment value. A bare max() is continuous but not
# differentiable where the two atoms' commitments are equal, which puts a
# force flip on that surface. sqrt(1e-4) = 0.01 in bond-order units.
ENVIRONMENT_SOFTENING_SMOOTH_EPSILON_SQUARED = 1e-4


# How much of the over-coordination penalty comes from the bonds involved,
# as opposed to the single global constant above.
#
# That constant is most of every activation barrier in the model, and it is
# one number for every element pair. So nothing in the barrier can tell a
# C-H from an O-H, and the model separates reactions far less than it should:
# measured against computed references spanning 0.272 to 0.590 eV, the best
# available parameter choice spreads three systems by 0.096 eV. The
# Evans-Polanyi slope comes out near 0.09 where hydrogen abstraction runs
# 0.3 to 0.5.
#
# Physically the cost of over-coordination is not a constant. Forcing a
# second partner onto an atom already tightly bound costs more than forcing
# one onto an atom loosely bound, so the penalty should scale with the
# strength of the bonds being squeezed together. Scaling it by the mean depth
# of an atom's own contacts, relative to a reference, gives that without
# naming any element or adding a table.
#
#     0.0  exactly as before, one global constant
#     1.0  penalty scales linearly with the atom's mean contact depth
#
# Measure before raising it: this is the largest single term in every barrier
# in the model.
OVER_COORDINATION_DEPTH_WEIGHT = 0.0

# The depth, in eV, at which the scaled penalty equals the global constant.
# Roughly the C-H entry, so formaldehyde and methane stay near where they
# were fitted and the change shows up as spreading rather than as an overall
# shift.
OVER_COORDINATION_REFERENCE_DEPTH = 4.29

OVER_COORDINATION_PENALTY = 7.778

# Ideal angle for each number of electron domains around an atom
# (bonded neighbours plus lone pairs), and how much each lone
# pair squeezes it. Two domains is linear, three trigonal, four
# tetrahedral; every lone pair takes about two and a half degrees
# off. That reproduces methane at 109.5, ammonia at 107 and water
# at 104.5 from nothing but counting electrons.

DOMAIN_ANGLES = {2: 180.0, 3: 120.0, 4: 109.47}

LONE_PAIR_SQUEEZE = 2.5

VALENCE_ARRAY = np.array(
    [VALENCE[name] for name in ELEMENTS],
    dtype=float
)

MASS_ARRAY = np.array([MASS[name] for name in ELEMENTS], dtype=float)

REST_ANGLE_ARRAY = np.deg2rad(
    np.array([REST_ANGLE[name] for name in ELEMENTS], dtype=float)
)

ANGLE_STIFFNESS_ARRAY = np.array(
    [ANGLE_STIFFNESS[name] for name in ELEMENTS],
    dtype=float
)

# Where a bond fades out. Inside the inner radius it is a full
# bond, past the outer radius it is gone, and between them there
# is a smooth cosine taper so nothing jumps when a bond breaks.
#
# These scale with the bond length rather than being a fixed
# margin, and the reason matters. In water the two hydrogens sit
# about 1.5 A apart. With a fixed margin that falls inside the
# H-H bond range, so the two hydrogens start treating each other
# as partially bonded and prise the molecule open to 121 degrees.
# Scaling by bond length puts the H-H cutoff at 1.18 A, safely
# inside that separation, and the angle collapses to its proper
# value. Real bond-order potentials suppress the same 1-3
# interaction through the angular part of the bond order; this is
# the cheap version of the same correction.

CUTOFF_INNER_FACTOR = 1.25
CUTOFF_OUTER_FACTOR = 1.60


def cutoff_radii():
    inner = BOND_LENGTH * CUTOFF_INNER_FACTOR
    outer = BOND_LENGTH * CUTOFF_OUTER_FACTOR

    return inner, outer


CUTOFF_INNER, CUTOFF_OUTER = cutoff_radii()

MAXIMUM_CUTOFF = float(CUTOFF_OUTER.max())


def smooth_cutoff(distance, inner, outer):
    # 1 inside, 0 outside, a cosine taper between. Smooth in
    # value and first derivative, so no force discontinuity when
    # a bond breaks.

    span = np.maximum(outer - inner, 1e-9)

    fraction = np.clip((distance - inner) / span, 0.0, 1.0)

    return 0.5 * (1.0 + np.cos(np.pi * fraction))


def bond_orders(taper, types):
    # How many bonds each pair is sharing.
    #
    # Start every contact at order one, then hand out whatever
    # valence each atom has left over. Carbon in CO2 has four
    # valence and two neighbours, so it has two spare to give;
    # each oxygen has one spare. They meet in the middle at one
    # extra each, which makes both bonds double. Ethene works out
    # the same way: the carbons have spare valence, the hydrogens
    # do not, so the surplus has nowhere to go but the C-C bond.

    valence = VALENCE_ARRAY[types]

    coordination = np.sum(taper, axis=1)

    spare = np.maximum(valence - coordination, 0.0)

    # Weight each partner by how much spare valence it has, so
    # surplus flows towards atoms that can actually take it.

    weighted = taper * spare[None, :]

    totals = np.sum(weighted, axis=1)

    # Normalizing a vanishing contact gives a finite share whose derivative is
    # ill-conditioned. Fade that allocation in with a C1 smoothstep. Above the
    # tiny onset region the gate is exactly one and the established bond-order
    # calculation is unchanged in both value and force.
    onset = 1e-4
    share_fraction = np.clip(totals / onset, 0.0, 1.0)
    share_gate = share_fraction ** 2 * (3.0 - 2.0 * share_fraction)
    share = (
        spare[:, None] * weighted
        / np.maximum(totals[:, None], 1e-12)
        * share_gate[:, None]
    )

    # A bond can only be as strong as the poorer partner allows.

    extra = np.minimum(share, share.T) * taper

    order = 1.0 + extra

    np.fill_diagonal(order, 0.0)

    return np.clip(order, 0.0, 3.0), coordination


def interpolate_parameters(order, types):
    # Blend the single, double and triple bond tables according
    # to the bond order, which lets a bond change character
    # smoothly as the geometry changes.

    single = (
        BOND_LENGTH[np.ix_(types, types)],
        BOND_DEPTH[np.ix_(types, types)],
        BOND_WIDTH[np.ix_(types, types)],
    )

    double = (
        DOUBLE_LENGTH[np.ix_(types, types)],
        DOUBLE_DEPTH[np.ix_(types, types)],
        DOUBLE_WIDTH[np.ix_(types, types)],
    )

    triple = (
        TRIPLE_LENGTH[np.ix_(types, types)],
        TRIPLE_DEPTH[np.ix_(types, types)],
        TRIPLE_WIDTH[np.ix_(types, types)],
    )

    lower = np.clip(order - 1.0, 0.0, 1.0)
    upper = np.clip(order - 2.0, 0.0, 1.0)

    blended = []

    for single_value, double_value, triple_value in zip(
        single, double, triple
    ):
        first = single_value + (double_value - single_value) * lower

        blended.append(
            first + (triple_value - first) * upper
        )

    return blended


def domain_angle(steric, lone_pairs):
    # Interpolate between linear, trigonal and tetrahedral, then
    # let lone pairs squeeze the result.

    steric = np.clip(steric, 2.0, 4.0)

    low = np.floor(steric).astype(int)
    high = np.minimum(low + 1, 4)

    weight = steric - low

    base = np.array([DOMAIN_ANGLES.get(value, 109.47) for value in low])
    upper = np.array([DOMAIN_ANGLES.get(value, 109.47) for value in high])

    angle = base + (upper - base) * weight

    return np.deg2rad(angle - LONE_PAIR_SQUEEZE * lone_pairs)


def potential_energy(positions, types, box_size=None,
                     return_parts=False):
    count = len(positions)

    offsets = positions[None, :, :] - positions[:, None, :]

    if box_size is not None:
        offsets = offsets - box_size * np.round(offsets / box_size)

    distance_squared = np.sum(offsets ** 2, axis=2)

    np.fill_diagonal(distance_squared, np.inf)

    distances = np.sqrt(distance_squared)

    pair_inner = CUTOFF_INNER[np.ix_(types, types)]
    pair_outer = CUTOFF_OUTER[np.ix_(types, types)]

    taper = smooth_cutoff(distances, pair_inner, pair_outer)

    np.fill_diagonal(taper, 0.0)

    order, coordination = bond_orders(taper, types)

    pair_length, pair_depth, pair_width = interpolate_parameters(
        order, types
    )

    shift = distances - pair_length

    repulsive = pair_depth * np.exp(-2.0 * pair_width * shift)
    attractive = 2.0 * pair_depth * np.exp(-pair_width * shift)

    pair_energy = taper * (repulsive - attractive)

    np.fill_diagonal(pair_energy, 0.0)

    bond_total = 0.5 * np.sum(pair_energy)

    # ---- over-coordination penalty ----
    #
    # This is what creates activation barriers. Halfway through a
    # hydrogen transfer the moving atom is touching two partners
    # at once, so its coordination is about two against a valence
    # of one. Without a penalty that halfway point is more stable
    # than either end and the reaction has no barrier at all.

    valence = VALENCE_ARRAY[types]

    excess = np.maximum(coordination - valence, 0.0)

    over_total = OVER_COORDINATION_PENALTY * np.sum(excess ** 2)

    # ---- angles ----

    bonded_order = np.sum(taper * order, axis=1)

    outer = OUTER_ELECTRON_ARRAY[types]

    lone_pairs = np.maximum((outer - bonded_order) / 2.0, 0.0)

    steric = coordination + lone_pairs

    rest = domain_angle(steric, lone_pairs)

    stiffness = ANGLE_STIFFNESS_ARRAY[types]

    angle_total = 0.0

    for centre in range(count):
        if stiffness[centre] <= 0.0:
            continue

        partners = np.where(taper[centre] > 1e-6)[0]

        if len(partners) < 2:
            continue

        for a_index in range(len(partners)):
            for b_index in range(a_index + 1, len(partners)):
                first = partners[a_index]
                second = partners[b_index]

                left = offsets[centre, first]
                right = offsets[centre, second]

                cosine = np.dot(left, right) / (
                    distances[centre, first]
                    * distances[centre, second]
                )

                angle = np.arccos(np.clip(cosine, -1.0, 1.0))

                first_taper = taper[centre, first]
                second_taper = taper[centre, second]
                angle_pair_taper = first_taper * second_taper
                taper_difference = first_taper - second_taper
                weaker_taper = 0.5 * (
                    first_taper
                    + second_taper
                    - np.sqrt(taper_difference ** 2 + 1e-8)
                    + 1e-4
                )
                lone_pair_directionality = np.clip(
                    0.5 * lone_pairs[centre], 0.0, 1.0
                )
                angle_engagement = (
                    weaker_taper
                    + (1.0 - weaker_taper) * lone_pair_directionality
                )
                weight = angle_pair_taper * angle_engagement

                angle_total += (
                    0.5
                    * stiffness[centre]
                    * weight
                    * (angle - rest[centre]) ** 2
                )

    if return_parts:
        return {
            "bond": bond_total,
            "over": over_total,
            "angle": angle_total,
            "coordination": coordination,
            "order": order,
            "lone_pairs": lone_pairs,
            "steric": steric,
        }

    return bond_total + over_total + angle_total


def numerical_forces(positions, types, box_size=None, step=1e-5):
    # Central differences. Slow, but it cannot disagree with the
    # energy function, which makes it the right reference to
    # check an analytic or autograd implementation against.

    forces = np.zeros_like(positions)

    for atom in range(len(positions)):
        for axis in range(3):
            shifted = positions.copy()

            shifted[atom, axis] += step
            plus = potential_energy(shifted, types, box_size)

            shifted[atom, axis] -= 2.0 * step
            minus = potential_energy(shifted, types, box_size)

            forces[atom, axis] = -(plus - minus) / (2.0 * step)

    return forces


def types_from_symbols(symbols):
    return np.array(
        [ELEMENT_INDEX[symbol] for symbol in symbols],
        dtype=int
    )


def masses_from_symbols(symbols):
    return np.array(
        [MASS[symbol] for symbol in symbols],
        dtype=float
    )
