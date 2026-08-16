"""
Check size consistency / separability of the current H-state reference.

Two identical H3 transfer systems are placed very far apart in the SAME box.
There are no physical interactions between the two clusters.

A separable potential must satisfy:

    E(A + B) = E(A) + E(B)

and the forces on A must be unchanged by the existence of distant B.

We compare:
    - BatchedReactiveSimulation (base control)
    - HStateReferenceBatchedSimulation (current reference)
    - BatchedHStateReferenceSimulation (batched execution candidate)

This is NOT a fit and changes no parameter.

Run:
    py diagnose_h_state_separability.py
"""

from __future__ import annotations

import numpy as np
import torch

from batched_torch import BatchedReactiveSimulation
from h_state_torch import HStateReferenceBatchedSimulation

try:
    from h_state_batched_torch import BatchedHStateReferenceSimulation
except ImportError:
    BatchedHStateReferenceSimulation = None


BOX_SIZE = 40.0
DTYPE = torch.float64
DEVICE = "cpu"

# Symmetric H3 competition:
#
# H0 ---0.90A--- H1 ---0.90A--- H2
#
# The two adjacent H-H contacts are both active; the end-to-end contact is
# deliberately much farther away.
H3_SPACING_A = 0.90

# Translation between the two H3 systems in the combined box.
# This is enormously larger than any ChemistryModel covalent cutoff.
CLUSTER_SEPARATION_A = 15.0


def h3_geometry(origin):
    origin = np.asarray(origin, dtype=float)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [H3_SPACING_A, 0.0, 0.0],
        [2.0 * H3_SPACING_A, 0.0, 0.0],
    ])

    return ["H", "H", "H"], positions + origin


def combined_geometry():
    symbols_a, positions_a = h3_geometry(
        [8.0, 8.0, 8.0]
    )

    symbols_b, positions_b = h3_geometry(
        [8.0, 8.0 + CLUSTER_SEPARATION_A, 8.0]
    )

    return (
        symbols_a + symbols_b,
        np.concatenate(
            [positions_a, positions_b],
            axis=0,
        ),
    )


def build(model_class, symbols, positions):
    return model_class(
        boxes=[
            (
                list(symbols),
                np.asarray(positions, dtype=float),
            )
        ],
        box_size=BOX_SIZE,
        target_temperature=0.0,
        friction=0.0,
        device=DEVICE,
        dtype=DTYPE,
        random_seed=0,
        relax_on_start=False,
    )


def evaluate(model_class):
    single_symbols, single_positions = h3_geometry(
        [8.0, 8.0, 8.0]
    )

    double_symbols, double_positions = combined_geometry()

    single = build(
        model_class,
        single_symbols,
        single_positions,
    )

    double = build(
        model_class,
        double_symbols,
        double_positions,
    )

    single_energy = float(
        single.potential_per_box[0]
    )

    double_energy = float(
        double.potential_per_box[0]
    )

    single_force = (
        single.forces
        .detach()
        .cpu()
        .numpy()
        .reshape(3, 3)
    )

    double_force = (
        double.forces
        .detach()
        .cpu()
        .numpy()
        .reshape(6, 3)
    )

    energy_nonadditivity = (
        double_energy - 2.0 * single_energy
    )

    first_cluster_force_error = float(
        np.max(
            np.abs(
                double_force[:3]
                - single_force
            )
        )
    )

    second_cluster_force_error = float(
        np.max(
            np.abs(
                double_force[3:]
                - single_force
            )
        )
    )

    return {
        "single_energy": single_energy,
        "double_energy": double_energy,
        "energy_nonadditivity": energy_nonadditivity,
        "first_cluster_force_error": first_cluster_force_error,
        "second_cluster_force_error": second_cluster_force_error,
    }


def print_result(name, result):
    print(name)

    print(
        f"  E(single H3)             = "
        f"{result['single_energy']:+.12f} eV"
    )

    print(
        f"  E(two distant H3)        = "
        f"{result['double_energy']:+.12f} eV"
    )

    print(
        f"  E(double) - 2 E(single)  = "
        f"{result['energy_nonadditivity']:+.12e} eV"
    )

    print(
        f"  max |dF| first cluster   = "
        f"{result['first_cluster_force_error']:.12e} eV/A"
    )

    print(
        f"  max |dF| second cluster  = "
        f"{result['second_cluster_force_error']:.12e} eV/A"
    )

    print()


def main():
    print("H-STATE SEPARABILITY DIAGNOSTIC")
    print()
    print(
        f"H3 spacing          : {H3_SPACING_A:.2f} A"
    )
    print(
        f"cluster separation  : {CLUSTER_SEPARATION_A:.2f} A"
    )
    print(
        f"box                 : {BOX_SIZE:.1f} A"
    )
    print(
        f"device / dtype      : {DEVICE} / {DTYPE}"
    )
    print()

    base = evaluate(
        BatchedReactiveSimulation
    )

    print_result(
        "BASE CONTROL",
        base,
    )

    reference = evaluate(
        HStateReferenceBatchedSimulation
    )

    print_result(
        "CURRENT H-STATE REFERENCE",
        reference,
    )

    if BatchedHStateReferenceSimulation is not None:
        candidate = evaluate(
            BatchedHStateReferenceSimulation
        )

        print_result(
            "BATCHED-EXECUTION H-STATE",
            candidate,
        )

        implementation_difference = abs(
            candidate["energy_nonadditivity"]
            - reference["energy_nonadditivity"]
        )

        print(
            "reference vs batched-execution "
            f"nonadditivity difference = "
            f"{implementation_difference:.12e} eV"
        )
        print()

    base_ok = (
        abs(
            base["energy_nonadditivity"]
        ) < 1.0e-10
        and base[
            "first_cluster_force_error"
        ] < 1.0e-9
        and base[
            "second_cluster_force_error"
        ] < 1.0e-9
    )

    hstate_ok = (
        abs(
            reference[
                "energy_nonadditivity"
            ]
        ) < 1.0e-10
        and reference[
            "first_cluster_force_error"
        ] < 1.0e-9
        and reference[
            "second_cluster_force_error"
        ] < 1.0e-9
    )

    print("VERDICT")

    print(
        "  base separable     : "
        + ("PASS" if base_ok else "FAIL")
    )

    print(
        "  H-state separable  : "
        + ("PASS" if hstate_ok else "FAIL")
    )

    if base_ok and not hstate_ok:
        print()
        print(
            "  The ordinary potential is size-consistent here, but the "
            "current H-state construction couples physically disconnected "
            "H-state components through its whole-box state graph."
        )
        print(
            "  That is a physics/scaling architecture issue, not merely "
            "a CUDA optimisation issue."
        )

    elif base_ok and hstate_ok:
        print()
        print(
            "  This particular disconnected-system test is separable. "
            "Do not change the H-state normalisation on the basis of the "
            "suspected issue alone."
        )

    else:
        print()
        print(
            "  The base control itself did not separate cleanly, so this "
            "geometry is not a valid diagnostic of the H-state architecture."
        )


if __name__ == "__main__":
    main()
