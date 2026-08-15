
"""
Diagnostic: replace the old unoccupied bonded-Morse core in the small
hydrogen-state reference with the independently calibrated SAPT exchange wall.

This file does NOT modify reactive_torch.py or production MD.

Important calibration bookkeeping
----------------------------------
The SAPT wall parameters are frozen and are never fitted here.

H_STATE_MIXING = 0.472744 was historically calibrated against symmetric
H + H2 exchange under the OLD common-core radial decomposition. Therefore
H3 is not an untouched prediction of the complete state model.

This diagnostic does three separate things:

1. Measure the SAPT wall with mixing = 0.
2. Show what happens if the legacy mixing is carried over unchanged.
3. Re-anchor ONLY the state-mixing scalar to the H3 0.420 eV target under
   the new radial decomposition, then freeze it and predict water.

That keeps the independently fitted SAPT repulsion untouched and preserves
water as the actual transfer holdout.
"""

from __future__ import annotations

import heapq
import inspect
import math

import numpy as np
import torch

import reactive as R
import h_state_reference as hs
import nonbonded_continuous_torch as nb


DTYPE = torch.float64
DEVICE = "cpu"

LEGACY_MIXING = float(
    getattr(
        hs,
        "H_STATE_MIXING",
        0.472744,
    )
)

H3_TARGET_BARRIER = 0.420

# Corrected scanner interpretation:
# the spectator angle is measured from the transfer axis.
# 75.53 degrees therefore gives an H-O-H angle of 104.47 degrees.
WATER_SPECTATORS = np.array(
    [0.960, 0.960, 75.53, 75.53],
    dtype=float,
)


# ============================================================
# SMALL HELPERS
# ============================================================


def canonical_edge(edge):
    a, b = edge
    return (
        (a, b)
        if a < b
        else (b, a)
    )


def state_weight_matrix(
    atom_count,
    state,
):
    weights = torch.zeros(
        (atom_count, atom_count),
        dtype=DTYPE,
        device=DEVICE,
    )

    for edge in state:
        a, b = canonical_edge(edge)

        weights[a, b] = 1.0
        weights[b, a] = 1.0

    return weights


def sapt_pair_energy(
    fragment,
    atom_a,
    atom_b,
):
    """
    One atomic pair contribution from the SAPT-derived Slater model,
    evaluated in the full state-specific local environment.

    This mirrors the pair loop in nonbonded_continuous_torch, but returns
    only the requested pair so it can replace one unoccupied H-state core.
    """

    position_a = fragment.positions[
        atom_a
    ]

    position_b = fragment.positions[
        atom_b
    ]

    delta = (
        position_b
        - position_a
    )

    distance = torch.linalg.vector_norm(
        delta
    )

    rhat = (
        delta
        / torch.clamp(
            distance,
            min=nb.EPS,
        )
    )

    symbol_a = fragment.symbols[
        atom_a
    ]

    symbol_b = fragment.symbols[
        atom_b
    ]

    A_a = nb.effective_A(
        fragment,
        atom_a,
    )

    A_b = nb.effective_A(
        fragment,
        atom_b,
    )

    B_a = nb.effective_B(
        fragment,
        atom_a,
        rhat,
    )

    B_b = nb.effective_B(
        fragment,
        atom_b,
        -rhat,
    )

    beta = torch.sqrt(
        B_a * B_b
    )

    x = (
        beta
        * distance
    )

    radial = (
        A_a
        * A_b
        * (
            1.0
            + x
            + x*x/3.0
        )
        * torch.exp(-x)
    )

    q_a = nb.amplitude_q2(
        fragment,
        atom_a,
        rhat,
    )

    q_b = nb.amplitude_q2(
        fragment,
        atom_b,
        -rhat,
    )

    angular = torch.exp(
        nb.ELEMENT_PARAMETERS[
            symbol_a
        ].k*q_a
        + nb.ELEMENT_PARAMETERS[
            symbol_b
        ].k*q_b
    )

    return (
        radial
        * angular
    )


# ============================================================
# SAPT-WALL HYDROGEN STATE
# ============================================================


def sapt_state_energy(
    positions,
    symbols,
    *,
    mixing,
    match_torch_environment=True,
):
    """
    Hydrogen-state ground energy using:

      occupied edge:
          existing complete tapered Morse covalent bond

      unoccupied candidate edge:
          tapered SAPT exchange-repulsion pair contribution

      state coupling:
          unchanged h_state_reference overlap/crowding machinery

    This deliberately changes only the radial definition that caused the
    common-core problem.
    """

    positions = np.asarray(
        positions,
        dtype=float,
    )

    types = R.types_from_symbols(
        symbols
    )

    pair = hs._pair_intermediates(
        positions,
        types,
        box_size=None,
        match_torch_environment=(
            match_torch_environment
        ),
    )

    taper = pair["taper"]
    pair_depth = pair["pair_depth"]
    repulsive = pair["repulsive"]
    attractive = pair["attractive"]

    edges = tuple(
        canonical_edge(edge)
        for edge in hs._hydrogen_edges(
            types,
            taper,
        )
    )

    if not edges:
        return {
            "energy": 0.0,
            "states": (tuple(),),
            "probabilities": np.array(
                [1.0]
            ),
            "diagonal": np.array(
                [0.0]
            ),
            "hamiltonian": np.zeros(
                (1, 1)
            ),
            "edges": tuple(),
            "walls": np.array(
                [0.0]
            ),
            "covalent": np.array(
                [0.0]
            ),
        }

    states = tuple(
        tuple(
            canonical_edge(edge)
            for edge in state
        )
        for state in hs._maximal_hydrogen_matchings(
            edges,
            types,
        )
    )

    positions_t = torch.tensor(
        positions,
        dtype=DTYPE,
        device=DEVICE,
    )

    diagonal = []
    wall_terms = []
    covalent_terms = []

    with torch.no_grad():

        for state in states:

            occupied = set(
                state
            )

            weights = state_weight_matrix(
                len(symbols),
                state,
            )

            fragment = (
                nb.ContinuousTorchFragment(
                    symbols=list(symbols),
                    positions=positions_t,
                    bond_weights=weights,
                )
            )

            covalent = sum(
                float(
                    taper[edge]
                    * (
                        repulsive[edge]
                        - attractive[edge]
                    )
                )
                for edge in state
            )

            # Crucial replacement:
            #
            # Old common_core:
            #   every candidate edge receives bonded Morse repulsion.
            #
            # New diagnostic:
            #   only UNoccupied candidate edges receive the SAPT exchange
            #   wall, using the local covalent environment of this state.
            #
            # Keep the existing reactive taper here. This isolates the radial
            # replacement and guarantees the wall switches off smoothly at
            # the same candidate-contact boundary used by the state solver.
            wall = 0.0

            for edge in edges:

                if edge in occupied:
                    continue

                contribution = (
                    sapt_pair_energy(
                        fragment,
                        edge[0],
                        edge[1],
                    )
                )

                wall += (
                    float(
                        taper[edge]
                    )
                    * float(
                        contribution
                    )
                )

            covalent_terms.append(
                covalent
            )

            wall_terms.append(
                wall
            )

            diagonal.append(
                covalent + wall
            )

    diagonal = np.asarray(
        diagonal,
        dtype=float,
    )

    wall_terms = np.asarray(
        wall_terms,
        dtype=float,
    )

    covalent_terms = np.asarray(
        covalent_terms,
        dtype=float,
    )

    count = len(states)

    hamiltonian = np.diag(
        diagonal.copy()
    )

    overlaps = np.zeros(
        (count, count),
        dtype=float,
    )

    transitions = {}

    for first in range(count):

        for second in range(
            first + 1,
            count,
        ):

            transition = (
                hs._single_h_transfer(
                    states[first],
                    states[second],
                    types,
                )
            )

            if transition is None:
                continue

            old_edge, new_edge, hydrogen = (
                transition
            )

            old_edge = canonical_edge(
                old_edge
            )

            new_edge = canonical_edge(
                new_edge
            )

            overlap = hs._contact_overlap(
                old_edge,
                new_edge,
                taper,
            )

            overlaps[
                first,
                second,
            ] = overlap

            overlaps[
                second,
                first,
            ] = overlap

            transitions[
                (first, second)
            ] = (
                old_edge,
                new_edge,
                hydrogen,
                overlap,
            )

    weighted_degree = np.sum(
        overlaps**2,
        axis=1,
    )

    normalisation = np.array([
        hs._crowding_normalisation(
            value
        )
        for value
        in weighted_degree
    ])

    for (
        first,
        second,
    ), transition in transitions.items():

        (
            old_edge,
            new_edge,
            _,
            overlap,
        ) = transition

        depth_scale = math.sqrt(
            max(
                float(
                    pair_depth[
                        old_edge
                    ]
                )
                * float(
                    pair_depth[
                        new_edge
                    ]
                ),
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

        hamiltonian[
            first,
            second,
        ] = -coupling

        hamiltonian[
            second,
            first,
        ] = -coupling

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            hamiltonian
        )
    )

    ground = eigenvectors[:, 0]

    return {
        "energy": float(
            eigenvalues[0]
        ),
        "states": states,
        "probabilities": (
            ground**2
        ),
        "diagonal": diagonal,
        "hamiltonian": hamiltonian,
        "edges": edges,
        "walls": wall_terms,
        "covalent": covalent_terms,
    }


# ============================================================
# H3
# ============================================================


def h3_reactant_geometry():
    re = float(
        R.BOND_LENGTH[
            R.ELEMENT_INDEX["H"],
            R.ELEMENT_INDEX["H"],
        ]
    )

    return (
        ["H", "H", "H"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [re, 0.0, 0.0],
                [re + 3.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )


def h3_symmetric_geometry(r):
    return (
        ["H", "H", "H"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [r, 0.0, 0.0],
                [2.0*r, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )


def h3_barrier_sapt(
    mixing,
    *,
    r_min=0.65,
    r_max=1.30,
    points=1301,
):
    symbols, reactant_positions = (
        h3_reactant_geometry()
    )

    reactant = sapt_state_energy(
        reactant_positions,
        symbols,
        mixing=mixing,
    )["energy"]

    radii = np.linspace(
        r_min,
        r_max,
        points,
    )

    energies = np.empty_like(
        radii
    )

    for index, radius in enumerate(
        radii
    ):
        _, positions = (
            h3_symmetric_geometry(
                float(radius)
            )
        )

        energies[index] = (
            sapt_state_energy(
                positions,
                symbols,
                mixing=mixing,
            )["energy"]
        )

    saddle_index = int(
        np.argmin(
            energies
        )
    )

    radius = float(
        radii[
            saddle_index
        ]
    )

    saddle = float(
        energies[
            saddle_index
        ]
    )

    details = sapt_state_energy(
        h3_symmetric_geometry(
            radius
        )[1],
        symbols,
        mixing=mixing,
    )

    return {
        "barrier": (
            saddle
            - reactant
        ),
        "r": radius,
        "reactant": reactant,
        "saddle": saddle,
        "details": details,
    }


def h3_barrier_old_mode(
    radial_mode,
    mixing,
):
    signature = inspect.signature(
        hs.hydrogen_state_energy
    )

    if "radial_mode" not in signature.parameters:
        return None

    symbols, reactant_positions = (
        h3_reactant_geometry()
    )

    reactant = (
        hs.hydrogen_state_energy(
            reactant_positions,
            symbols,
            mixing=mixing,
            match_torch_environment=True,
            radial_mode=radial_mode,
        ).energy
    )

    radii = np.linspace(
        0.65,
        1.30,
        1301,
    )

    energies = []

    for radius in radii:

        _, positions = (
            h3_symmetric_geometry(
                float(radius)
            )
        )

        energies.append(
            hs.hydrogen_state_energy(
                positions,
                symbols,
                mixing=mixing,
                match_torch_environment=True,
                radial_mode=radial_mode,
            ).energy
        )

    energies = np.asarray(
        energies
    )

    index = int(
        np.argmin(
            energies
        )
    )

    return {
        "barrier": float(
            energies[index]
            - reactant
        ),
        "r": float(
            radii[index]
        ),
    }


def find_h3_mixing_for_target():
    """
    Re-anchor only the state-coupling scalar.

    SAPT parameters are never changed.
    """

    def residual(mixing):
        return (
            h3_barrier_sapt(
                mixing,
                points=651,
            )["barrier"]
            - H3_TARGET_BARRIER
        )

    low = 0.0
    high = 1.5

    f_low = residual(low)
    f_high = residual(high)

    if f_low == 0.0:
        return low

    if (
        f_low
        * f_high
        > 0.0
    ):
        return None

    for _ in range(45):

        middle = 0.5 * (
            low + high
        )

        f_mid = residual(
            middle
        )

        if abs(f_mid) < 1.0e-7:
            return middle

        if (
            f_low
            * f_mid
            <= 0.0
        ):
            high = middle
            f_high = f_mid
        else:
            low = middle
            f_low = f_mid

    return 0.5 * (
        low + high
    )


# ============================================================
# WATER GEOMETRY
# ============================================================


WATER_SYMBOLS = [
    "O",
    "H",
    "H",
    "O",
    "H",
]


def water_geometry(
    donor_length,
    transfer_length,
    spectators=WATER_SPECTATORS,
):
    """
    Same five-atom symmetric H2O/OH transfer geometry used by the scanner,
    with corrected spectator-axis angles supplied explicitly.
    """

    (
        donor_oh,
        acceptor_oh,
        donor_angle,
        acceptor_angle,
    ) = spectators

    donor_oxygen = np.zeros(
        3
    )

    moving_h = np.array(
        [
            donor_length,
            0.0,
            0.0,
        ]
    )

    acceptor_oxygen = np.array(
        [
            donor_length
            + transfer_length,
            0.0,
            0.0,
        ]
    )

    def spoke(
        origin,
        length,
        angle,
        direction,
        tilt,
    ):
        radians = np.radians(
            angle
        )

        return (
            origin
            + length
            * np.array(
                [
                    direction
                    * np.cos(
                        radians
                    ),
                    np.sin(
                        radians
                    )
                    * np.cos(
                        tilt
                    ),
                    np.sin(
                        radians
                    )
                    * np.sin(
                        tilt
                    ),
                ]
            )
        )

    donor_spoke = spoke(
        donor_oxygen,
        donor_oh,
        donor_angle,
        -1.0,
        np.pi/2,
    )

    acceptor_spoke = spoke(
        acceptor_oxygen,
        acceptor_oh,
        acceptor_angle,
        +1.0,
        np.pi/2,
    )

    return (
        WATER_SYMBOLS,
        np.array(
            [
                donor_oxygen,
                moving_h,
                donor_spoke,
                acceptor_oxygen,
                acceptor_spoke,
            ],
            dtype=float,
        ),
    )


def water_fixed_oo_profile(
    oxygen_separation,
    mixing,
    *,
    points=401,
):
    """
    Symmetric proton transfer at fixed O-O separation.

    Endpoints place the moving proton at approximately an equilibrium O-H
    distance from either oxygen. The midpoint is the shared proton.
    """

    endpoint_offset = max(
        0.0,
        0.5*oxygen_separation
        - 0.960,
    )

    offsets = np.linspace(
        -endpoint_offset,
        +endpoint_offset,
        points,
    )

    energies = np.empty_like(
        offsets
    )

    for index, offset in enumerate(
        offsets
    ):

        donor = (
            0.5*oxygen_separation
            + offset
        )

        transfer = (
            oxygen_separation
            - donor
        )

        symbols, positions = (
            water_geometry(
                donor,
                transfer,
            )
        )

        energies[index] = (
            sapt_state_energy(
                positions,
                symbols,
                mixing=mixing,
            )["energy"]
        )

    midpoint = int(
        np.argmin(
            np.abs(
                offsets
            )
        )
    )

    left_energy = float(
        energies[0]
    )

    right_energy = float(
        energies[-1]
    )

    endpoint = 0.5 * (
        left_energy
        + right_energy
    )

    # Path from left basin to midpoint.
    half = energies[
        :midpoint + 1
    ]

    barrier = float(
        np.max(
            half
        )
        - left_energy
    )

    centre = float(
        energies[
            midpoint
        ]
        - endpoint
    )

    symmetry_error = abs(
        left_energy
        - right_energy
    )

    return {
        "oo": oxygen_separation,
        "barrier": barrier,
        "centre_relative": centre,
        "symmetry_error": symmetry_error,
        "offsets": offsets,
        "energies": energies,
    }


# ============================================================
# COARSE 2D WATER MINIMAX
# ============================================================


def minimax_cost(
    grid,
    start,
    goal,
):
    rows, cols = grid.shape

    costs = np.full(
        grid.shape,
        np.inf,
        dtype=float,
    )

    costs[start] = float(
        grid[start]
    )

    queue = [
        (
            float(
                grid[start]
            ),
            start,
        )
    ]

    visited = np.zeros(
        grid.shape,
        dtype=bool,
    )

    # Match hf_surface_scan.flood_saddle exactly: four-connected grid.
    # Diagonal moves artificially lower a discrete minimax barrier.
    neighbours = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    while queue:

        cost, cell = heapq.heappop(
            queue
        )

        if visited[cell]:
            continue

        visited[cell] = True

        if cell == goal:
            return cost

        i, j = cell

        for di, dj in neighbours:

            ni = i + di
            nj = j + dj

            if (
                ni < 0
                or nj < 0
                or ni >= rows
                or nj >= cols
            ):
                continue

            candidate = max(
                cost,
                float(
                    grid[
                        ni,
                        nj,
                    ]
                ),
            )

            if (
                candidate
                < costs[
                    ni,
                    nj
                ]
            ):
                costs[
                    ni,
                    nj
                ] = candidate

                heapq.heappush(
                    queue,
                    (
                        candidate,
                        (ni, nj),
                    ),
                )

    return None


def water_2d_barrier(
    mixing,
    *,
    step=0.03,
):
    donor = np.arange(
        0.90,
        1.80 + 0.5*step,
        step,
    )

    transfer = np.arange(
        0.90,
        1.80 + 0.5*step,
        step,
    )

    grid = np.empty(
        (
            len(donor),
            len(transfer),
        ),
        dtype=float,
    )

    for i, d in enumerate(
        donor
    ):
        for j, t in enumerate(
            transfer
        ):

            symbols, positions = (
                water_geometry(
                    float(d),
                    float(t),
                )
            )

            grid[i, j] = (
                sapt_state_energy(
                    positions,
                    symbols,
                    mixing=mixing,
                )["energy"]
            )

    reactant_cells = [
        (i, j)
        for i, d in enumerate(
            donor
        )
        for j, t in enumerate(
            transfer
        )
        if (
            d <= 1.10
            and t >= 1.55
        )
    ]

    product_cells = [
        (i, j)
        for i, d in enumerate(
            donor
        )
        for j, t in enumerate(
            transfer
        )
        if (
            d >= 1.55
            and t <= 1.10
        )
    ]

    reactant = min(
        reactant_cells,
        key=lambda cell:
            grid[cell],
    )

    product = min(
        product_cells,
        key=lambda cell:
            grid[cell],
    )

    saddle = minimax_cost(
        grid,
        reactant,
        product,
    )

    return {
        "barrier": (
            None
            if saddle is None
            else float(
                saddle
                - grid[
                    reactant
                ]
            )
        ),
        "reaction": float(
            grid[
                product
            ]
            - grid[
                reactant
            ]
        ),
        "reactant": (
            float(
                donor[
                    reactant[0]
                ]
            ),
            float(
                transfer[
                    reactant[1]
                ]
            ),
        ),
        "product": (
            float(
                donor[
                    product[0]
                ]
            ),
            float(
                transfer[
                    product[1]
                ]
            ),
        ),
    }


# ============================================================
# REPORT
# ============================================================


def report_h3():
    print()
    print(
        "H3: H + H2 -> H2 + H"
    )
    print(
        "====================="
    )

    print(
        f"target barrier: "
        f"{H3_TARGET_BARRIER:.3f} eV"
    )

    print(
        f"legacy state mixing: "
        f"{LEGACY_MIXING:.6f}"
    )

    old_common = (
        h3_barrier_old_mode(
            "common_core",
            LEGACY_MIXING,
        )
    )

    old_zero = (
        h3_barrier_old_mode(
            "occupied_morse",
            0.0,
        )
    )

    if old_common is not None:
        print()
        print(
            "old common_core + legacy mixing"
        )

        print(
            f"  barrier = "
            f"{old_common['barrier']:+.6f} eV"
        )

        print(
            f"  symmetric r = "
            f"{old_common['r']:.6f} A"
        )

    if old_zero is not None:
        print()
        print(
            "occupied_morse, zero mixing"
        )

        print(
            f"  barrier = "
            f"{old_zero['barrier']:+.6f} eV"
        )

        print(
            f"  symmetric r = "
            f"{old_zero['r']:.6f} A"
        )

    zero = h3_barrier_sapt(
        0.0
    )

    legacy = h3_barrier_sapt(
        LEGACY_MIXING
    )

    print()
    print(
        "SAPT wall, zero mixing"
    )

    print(
        f"  barrier = "
        f"{zero['barrier']:+.6f} eV"
    )

    print(
        f"  symmetric r = "
        f"{zero['r']:.6f} A"
    )

    print()
    print(
        "SAPT wall + legacy mixing"
    )

    print(
        f"  barrier = "
        f"{legacy['barrier']:+.6f} eV"
    )

    print(
        f"  symmetric r = "
        f"{legacy['r']:.6f} A"
    )

    mixing = (
        find_h3_mixing_for_target()
    )

    print()

    if mixing is None:

        print(
            "H3 target cannot be bracketed "
            "with mixing in [0, 1.5]."
        )

        return None

    fitted = h3_barrier_sapt(
        mixing
    )

    print(
        "SAPT wall + H3-reanchored coupling"
    )

    print(
        f"  mixing  = "
        f"{mixing:.9f}"
    )

    print(
        f"  barrier = "
        f"{fitted['barrier']:+.6f} eV"
    )

    print(
        f"  symmetric r = "
        f"{fitted['r']:.6f} A"
    )

    details = fitted[
        "details"
    ]

    print(
        f"  state diagonals = "
        f"{np.array2string(details['diagonal'], precision=6)}"
    )

    print(
        f"  SAPT wall terms = "
        f"{np.array2string(details['walls'], precision=6)}"
    )

    print(
        f"  state probabilities = "
        f"{np.array2string(details['probabilities'], precision=6)}"
    )

    return mixing


def report_water(
    mixing,
):
    print()
    print(
        "WATER HOLDOUT"
    )
    print(
        "============="
    )

    print(
        "SAPT parameters frozen; "
        "mixing fixed from H3 only."
    )

    print(
        f"mixing = {mixing:.9f}"
    )

    print(
        "spectators = "
        f"{WATER_SPECTATORS.tolist()}"
    )

    print()
    print(
        "fixed O-O symmetric proton transfer"
    )

    print(
        f"{'O-O / A':>9} "
        f"{'barrier / eV':>14} "
        f"{'centre-end / eV':>17} "
        f"{'sym err / eV':>13}"
    )

    for oo in (
        2.30,
        2.35,
        2.40,
        2.45,
        2.50,
        2.55,
        2.60,
        2.70,
    ):
        result = (
            water_fixed_oo_profile(
                oo,
                mixing,
            )
        )

        print(
            f"{oo:9.2f} "
            f"{result['barrier']:14.6f} "
            f"{result['centre_relative']:17.6f} "
            f"{result['symmetry_error']:13.3e}"
        )

    print()
    print(
        "coarse 2D minimax water surface "
        "(H-state radial subsystem only)"
    )

    surface = water_2d_barrier(
        mixing,
        step=0.03,
    )

    print(
        f"  barrier  = "
        f"{surface['barrier']:+.6f} eV"
    )

    print(
        f"  reaction = "
        f"{surface['reaction']:+.6f} eV"
    )

    print(
        f"  reactant = "
        f"{surface['reactant']}"
    )

    print(
        f"  product  = "
        f"{surface['product']}"
    )

    print()
    print(
        "Published full-system water comparison used elsewhere "
        "in this project: 0.364-0.525 eV depending on reference state."
    )

    print(
        "Do NOT interpret the coarse H-state-only number as the "
        "complete molecular barrier; this diagnostic isolates the radial "
        "transfer subsystem."
    )


def main():
    print(
        "SAPT WALL TRANSFER DIAGNOSTIC"
    )

    print(
        "============================="
    )

    print(
        "No SAPT parameter is fitted to H3 or water."
    )

    print(
        "H3 is used only to re-anchor the separate state-coupling scalar "
        "because the legacy value was already H3-calibrated under the old "
        "common-core radial decomposition."
    )

    mixing = report_h3()

    if mixing is not None:
        report_water(
            mixing
        )


if __name__ == "__main__":
    main()
