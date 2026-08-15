"""
Standalone reference model for ChemistryModel v2 hydrogen valence states.

PURPOSE
-------
Test a different definition of hydrogen bonding without changing production
reactive.py / reactive_torch.py.

Current production behaviour:
    every contact receives a complete Morse interaction
    hydrogen may therefore temporarily receive multiple full bonds
    over-coordination then penalises that situation

Reference behaviour:
    nearby H-containing pairs define possible covalent edges
    allowed states obey deg(H) <= 1
    each state receives repulsive/core terms from every contact
    only occupied edges receive covalent attraction
    competing one-H transfer states may mix

This first reference deliberately retains:
    - current bond tables
    - current single/double/triple interpolation
    - current radial taper
    - current Morse repulsive/attractive pieces

It is NOT yet a new production force field.
It is deliberately slow and enumerates small-state spaces exactly.

Important:
reactive.py does not currently include the Torch-only environment-softening
factor in its NumPy potential path. This reference therefore tests the
state architecture first. Scanner/Torch integration comes next.
"""

from dataclasses import dataclass
import math
import itertools

import numpy as np

import reactive as R


# Calibrated only against symmetric H + H2 exchange in this experimental
# reference. Do not refit this against formaldehyde/methane/water here.
H_STATE_MIXING = 0.472744

# Smooth region used only when crowding normalisation has to rise above one.
# This is numerical architecture, not a fitted chemistry parameter.
CROWDING_TRANSITION_WIDTH = 0.25

# Exact state enumeration is intentionally only for tiny reference systems.
MAX_REFERENCE_EDGES = 18


@dataclass
class HStateResult:
    energy: float
    states: tuple
    probabilities: np.ndarray
    occupations: dict
    hamiltonian: np.ndarray
    diagonal_energies: np.ndarray
    edges: tuple


def _minimum_image(offsets, box_size):
    if box_size is None:
        return offsets

    return offsets - box_size * np.round(offsets / box_size)


def _pair_intermediates(
    positions,
    types,
    box_size=None,
    match_torch_environment=False,
):
    """Reproduce the current pair state without changing production code.

    By default this follows the NumPy reference exactly. When
    ``match_torch_environment`` is true, it also applies the Torch engine's
    environment-softening factor so direct NumPy/Torch comparisons are
    like-for-like.
    """

    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)

    offsets = positions[None, :, :] - positions[:, None, :]
    offsets = _minimum_image(offsets, box_size)

    distance_squared = np.sum(offsets ** 2, axis=2)
    np.fill_diagonal(distance_squared, np.inf)
    distances = np.sqrt(distance_squared)

    inner = R.CUTOFF_INNER[np.ix_(types, types)]
    outer = R.CUTOFF_OUTER[np.ix_(types, types)]

    taper = R.smooth_cutoff(distances, inner, outer)
    np.fill_diagonal(taper, 0.0)

    order, coordination = R.bond_orders(taper, types)

    pair_length, pair_depth, pair_width = R.interpolate_parameters(
        order, types
    )

    if match_torch_environment and R.ENVIRONMENT_SOFTENING > 0.0:
        # Match ReactiveSimulation.environment_softening_factor() exactly,
        # but in full-matrix NumPy form. The production Torch engine uses a
        # smoothed two-ended maximum, so even zero commitment receives the
        # tiny epsilon contribution; this option exists only for exact
        # reference-vs-Torch comparisons.
        lower = np.clip(order - 1.0, 0.0, 1.0)

        commitment = np.maximum(
            np.sum(taper * (order - 1.0), axis=1),
            0.0,
        )

        own = commitment[:, None]
        partner = commitment[None, :]
        gap = own - partner

        pair_commitment = 0.5 * (
            own
            + partner
            + np.sqrt(
                gap * gap
                + R.ENVIRONMENT_SOFTENING_SMOOTH_EPSILON_SQUARED
            )
        )

        single_character = np.clip(1.0 - lower, 0.0, 1.0)

        factor = 1.0 - (
            R.ENVIRONMENT_SOFTENING
            * single_character
            * np.clip(pair_commitment, 0.0, 1.0)
        )

        pair_depth = pair_depth * factor

    shift = distances - pair_length

    repulsive = pair_depth * np.exp(
        -2.0 * pair_width * shift
    )

    attractive = 2.0 * pair_depth * np.exp(
        -pair_width * shift
    )

    np.fill_diagonal(repulsive, 0.0)
    np.fill_diagonal(attractive, 0.0)

    return {
        "distances": distances,
        "taper": taper,
        "order": order,
        "coordination": coordination,
        "pair_length": pair_length,
        "pair_depth": pair_depth,
        "pair_width": pair_width,
        "repulsive": repulsive,
        "attractive": attractive,
    }


def _hydrogen_edges(types, taper):
    """Undirected H-containing contacts with non-zero current engagement."""

    hydrogen = R.ELEMENT_INDEX["H"]
    edges = []

    for first in range(len(types)):
        for second in range(first + 1, len(types)):
            if types[first] != hydrogen and types[second] != hydrogen:
                continue

            if taper[first, second] <= 0.0:
                continue

            edges.append((first, second))

    return tuple(edges)


def _matching_is_valid(edges, types):
    """A hydrogen may occur in at most one occupied covalent edge."""

    hydrogen = R.ELEMENT_INDEX["H"]
    used_hydrogens = set()

    for first, second in edges:
        for atom in (first, second):
            if types[atom] != hydrogen:
                continue

            if atom in used_hydrogens:
                return False

            used_hydrogens.add(atom)

    return True


def _maximal_hydrogen_matchings(edges, types):
    """Enumerate maximal states satisfying deg(H) <= 1.

    Maximal means another H-containing edge cannot be added without violating
    the hydrogen valence rule.

    Examples:

        H2:
            {H-H}

        H-H-H:
            {H1-H2}
            {H2-H3}

        H2O:
            {O-H1, O-H2}

    This is deliberately exponential: it is a reference solver for tiny
    systems, not the future GPU implementation.
    """

    if len(edges) > MAX_REFERENCE_EDGES:
        raise ValueError(
            f"reference state space has {len(edges)} H-containing edges; "
            f"limit is {MAX_REFERENCE_EDGES}"
        )

    valid = []

    for count in range(len(edges) + 1):
        for chosen in itertools.combinations(edges, count):
            if _matching_is_valid(chosen, types):
                valid.append(tuple(chosen))

    maximal = []

    for state in valid:
        state_set = set(state)
        can_extend = False

        for edge in edges:
            if edge in state_set:
                continue

            if _matching_is_valid(state + (edge,), types):
                can_extend = True
                break

        if not can_extend:
            maximal.append(state)

    if not maximal:
        maximal = [tuple()]

    return tuple(maximal)


def _shared_hydrogen(first_edge, second_edge, types):
    hydrogen = R.ELEMENT_INDEX["H"]

    shared = set(first_edge).intersection(second_edge)

    shared_h = [
        atom for atom in shared
        if types[atom] == hydrogen
    ]

    if len(shared_h) != 1:
        return None

    return shared_h[0]


def _single_h_transfer(first_state, second_state, types):
    """Recognise states differing by transfer of one H bond.

    Example:

        {C-H}  <->  {H-H}

    or

        {H1-H2} <-> {H2-H3}

    States differing in several bonds are not directly coupled in this first
    reference.
    """

    first = set(first_state)
    second = set(second_state)

    removed = list(first - second)
    added = list(second - first)

    if len(removed) != 1 or len(added) != 1:
        return None

    hydrogen = _shared_hydrogen(
        removed[0], added[0], types
    )

    if hydrogen is None:
        return None

    return removed[0], added[0], hydrogen


def _contact_overlap(first_edge, second_edge, taper):
    """Same basic balanced-contact idea used by high_fidelity_torch."""

    first = float(taper[first_edge])
    second = float(taper[second_edge])

    total = first + second

    if total <= 1e-12:
        return 0.0

    balance = (
        4.0 * first * second
        / (total * total)
    )

    return math.sqrt(max(first * second, 0.0)) * balance


def _crowding_normalisation(weighted_degree):
    """Prevent N equivalent states receiving N-dependent free stabilisation.

    For one competing alternative:
        degree = 1
        normalisation = 1
        ordinary two-state result is unchanged

    With several equally strong alternatives, normalisation rises with the
    weighted state degree.

    Weak alternatives enter quadratically through the weighted degree and
    therefore barely disturb an established two-state crossing.
    """

    degree = float(weighted_degree)

    if degree <= 1.0:
        return 1.0

    width = CROWDING_TRANSITION_WIDTH

    if degree >= 1.0 + width:
        return degree

    fraction = (degree - 1.0) / width
    smooth = fraction ** 2 * (3.0 - 2.0 * fraction)

    return 1.0 + smooth * (degree - 1.0)


def hydrogen_state_energy(
    positions,
    symbols,
    box_size=None,
    mixing=H_STATE_MIXING,
    match_torch_environment=False,
):
    """Ground-state H covalent energy for a small reference geometry.

    Returns HStateResult.

    The common diagonal contribution is:

        sum(current tapered Morse repulsive pieces)

    An occupied covalent edge additionally receives:

        - current tapered Morse attractive piece

    Therefore an ordinary isolated occupied bond is exactly:

        taper * (repulsive - attractive)

    which is the existing pair potential.
    """

    positions = np.asarray(positions, dtype=float)
    types = R.types_from_symbols(symbols)

    pair = _pair_intermediates(
        positions,
        types,
        box_size=box_size,
        match_torch_environment=match_torch_environment,
    )

    taper = pair["taper"]
    pair_depth = pair["pair_depth"]
    repulsive = pair["repulsive"]
    attractive = pair["attractive"]

    edges = _hydrogen_edges(types, taper)

    if not edges:
        return HStateResult(
            energy=0.0,
            states=(tuple(),),
            probabilities=np.array([1.0]),
            occupations={},
            hamiltonian=np.zeros((1, 1)),
            diagonal_energies=np.array([0.0]),
            edges=tuple(),
        )

    states = _maximal_hydrogen_matchings(
        edges, types
    )

    common_core = sum(
        float(taper[edge] * repulsive[edge])
        for edge in edges
    )

    diagonal = np.array([
        common_core
        - sum(
            float(taper[edge] * attractive[edge])
            for edge in state
        )
        for state in states
    ])

    count = len(states)

    hamiltonian = np.diag(diagonal)

    # First construct the raw state-connectivity weights.
    overlaps = np.zeros((count, count), dtype=float)
    transitions = {}

    for first in range(count):
        for second in range(first + 1, count):
            transition = _single_h_transfer(
                states[first],
                states[second],
                types,
            )

            if transition is None:
                continue

            old_edge, new_edge, hydrogen = transition

            overlap = _contact_overlap(
                old_edge, new_edge, taper
            )

            overlaps[first, second] = overlap
            overlaps[second, first] = overlap

            transitions[(first, second)] = (
                old_edge,
                new_edge,
                hydrogen,
                overlap,
            )

    # Squaring means a vanishingly weak third state contributes only at
    # second order to crowding of an otherwise ordinary two-state crossing.
    weighted_degree = np.sum(
        overlaps ** 2, axis=1
    )

    normalisation = np.array([
        _crowding_normalisation(value)
        for value in weighted_degree
    ])

    for (first, second), transition in transitions.items():
        old_edge, new_edge, _, overlap = transition

        depth_scale = math.sqrt(
            max(
                float(pair_depth[old_edge])
                * float(pair_depth[new_edge]),
                0.0,
            )
        )

        denominator = math.sqrt(
            normalisation[first]
            * normalisation[second]
        )

        coupling = (
            float(mixing)
            * depth_scale
            * overlap
            / denominator
        )

        hamiltonian[first, second] = -coupling
        hamiltonian[second, first] = -coupling

    eigenvalues, eigenvectors = np.linalg.eigh(
        hamiltonian
    )

    ground_vector = eigenvectors[:, 0]
    probabilities = ground_vector ** 2

    occupations = {
        edge: 0.0
        for edge in edges
    }

    for probability, state in zip(
        probabilities, states
    ):
        for edge in state:
            occupations[edge] += float(probability)

    return HStateResult(
        energy=float(eigenvalues[0]),
        states=states,
        probabilities=probabilities,
        occupations=occupations,
        hamiltonian=hamiltonian,
        diagonal_energies=diagonal,
        edges=edges,
    )