"""
Audit whether the experimental state-aware helper exactly reproduces the live
ChemistryModel base energy when given the UNCHANGED ordinary taper.

This must pass before interpreting the gap between the isolated ablations and
the all-at-once state-aware model.

If:
    replay(base taper) == live base
then the all-at-once residual is a real interaction/state effect.

If not:
    the experimental helper is not a faithful copy of reactive_torch.py and
    the previous all-at-once barrier numbers must not be interpreted.

Usage:
    py sapt_state_aware_replay_audit.py
"""

from __future__ import annotations

import numpy as np
import torch

import hf_surface_scan as scan

from batched_torch import BatchedReactiveSimulation

from sapt_state_aware_valence_torch import (
    StateAwareValenceSaptHStateBatchedSimulation,
    _state_base_energy_per_atom,
)


DTYPE = torch.float64


WATER_FROZEN = np.array(
    [
        0.960,
        0.960,
        75.53,
        75.53,
    ],
    dtype=float,
)


CASES = (
    (
        "formaldehyde",
        1.28,
        1.01,
    ),
    (
        "water",
        1.10,
        1.30,
    ),
    (
        "methane",
        1.44,
        0.85,
    ),
)


def frozen_spectators(
    system,
):
    if system == "water":
        return WATER_FROZEN.copy()

    return np.asarray(
        scan.SYSTEMS[
            system
        ][
            "frozen"
        ],
        dtype=float,
    ).copy()


def build_case(
    system,
    donor,
    transfer,
):
    scan.apply_system(
        system
    )

    spectators = frozen_spectators(
        system
    )

    symbols, positions = scan.SYSTEMS[
        system
    ][
        "geometry"
    ](
        donor,
        transfer,
        spectators,
    )

    sim = StateAwareValenceSaptHStateBatchedSimulation(
        boxes=[
            (
                symbols,
                np.asarray(
                    positions,
                    dtype=float,
                )
                + scan.CENTRE,
            )
        ],
        box_size=scan.BOX,
        random_seed=0,
        relax_on_start=False,
        device="cpu",
        dtype=DTYPE,
    )

    return sim


def audit_case(
    system,
    donor,
    transfer,
):
    sim = build_case(
        system,
        donor,
        transfer,
    )

    positions = (
        sim.positions
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    # Bypass the H-state override and evaluate the live ordinary base exactly.
    live_per_atom = BatchedReactiveSimulation.energy_per_atom(
        sim,
        positions,
    )

    cached = getattr(
        sim,
        "_reactive_intermediates",
        None,
    )

    if cached is None:
        raise RuntimeError(
            f"{system}: live base did not expose reactive intermediates"
        )

    values = cached[
        1
    ]

    live_parts = getattr(
        sim,
        "_profile_energy_parts",
        None,
    )

    # The normal live engine stores detached diagnostic pieces in
    # _energy_parts. They are enough for this numerical identity audit.
    detached_parts = getattr(
        sim,
        "_energy_parts",
        None,
    )

    replay = _state_base_energy_per_atom(
        sim,
        positions=positions,
        values=values,
        state_taper=values[
            "taper"
        ],
    )

    live_total = float(
        torch.sum(
            live_per_atom
        )
        .detach()
        .cpu()
    )

    replay_total = float(
        torch.sum(
            replay[
                "total"
            ]
        )
        .detach()
        .cpu()
    )

    print(
        system.upper()
    )
    print(
        "-" * len(
            system
        )
    )
    print(
        f"geometry                 {donor:.5f} / {transfer:.5f} A"
    )
    print(
        f"live base total          {live_total:+.12f} eV"
    )
    print(
        f"replayed base total      {replay_total:+.12f} eV"
    )
    print(
        f"TOTAL DIFFERENCE         {replay_total-live_total:+.12e} eV"
    )

    if detached_parts is not None:
        for name in (
            "bond",
            "over",
            "angle",
        ):
            live_value = float(
                torch.sum(
                    detached_parts[
                        name
                    ]
                )
                .detach()
                .cpu()
            )

            replay_value = float(
                torch.sum(
                    replay[
                        name
                    ]
                )
                .detach()
                .cpu()
            )

            print(
                f"{name:>8} difference        "
                f"{replay_value-live_value:+.12e} eV"
            )

    print(
        f"max |coordination diff|  "
        f"{float(torch.max(torch.abs(replay['coordination']-values['coordination'])).detach().cpu()):.12e}"
    )

    print(
        f"max |order diff|         "
        f"{float(torch.max(torch.abs(replay['order']-values['order'])).detach().cpu()):.12e}"
    )

    print()

    return abs(
        replay_total
        - live_total
    )


def main():
    print(
        "STATE-AWARE BASE REPLAY AUDIT"
    )
    print(
        "============================="
    )
    print(
        "The unchanged ordinary taper must reproduce the live base exactly."
    )
    print()

    differences = []

    for (
        system,
        donor,
        transfer,
    ) in CASES:
        differences.append(
            (
                system,
                audit_case(
                    system,
                    donor,
                    transfer,
                ),
            )
        )

    print(
        "SUMMARY"
    )
    print(
        "-------"
    )

    for system, difference in differences:
        status = (
            "PASS"
            if difference < 1.0e-9
            else "FAIL"
        )

        print(
            f"{system:>12}: {difference:.12e} eV  {status}"
        )

    if any(
        difference >= 1.0e-9
        for _, difference
        in differences
    ):
        print()
        print(
            "Do NOT interpret the previous all-at-once state-aware barrier "
            "shift yet. The helper is not replaying the base potential "
            "identically."
        )
    else:
        print()
        print(
            "Replay is exact. The isolated-vs-all discrepancy is then a real "
            "coupled state effect and can be decomposed further."
        )


if __name__ == "__main__":
    main()
