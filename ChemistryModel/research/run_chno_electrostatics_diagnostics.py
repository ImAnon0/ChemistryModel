from __future__ import annotations

from research.electrostatics_diagnostics import inspect_qeq_state


def chno_diagnostic_cases():
    return {
        "H2": (
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            (1, 1),
        ),
        "CH4": (
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-1.09, 0.0, 0.0],
                [0.0, 1.09, 0.0],
                [0.0, -1.09, 0.0],
            ],
            (6, 1, 1, 1, 1),
        ),
        "H2O": (
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
            (8, 1, 1),
        ),
        "CH2O": (
            [
                [0.0, 0.0, 0.0],
                [1.21, 0.0, 0.0],
                [-0.7, 0.95, 0.0],
                [-0.7, -0.95, 0.0],
            ],
            (6, 8, 1, 1),
        ),
    }


def run():
    for name, (positions, atoms) in chno_diagnostic_cases().items():
        result = inspect_qeq_state(positions, atoms)

        print()
        print("=" * 40)
        print(name)
        print("=" * 40)

        for atom, charge in zip(atoms, result["charges"]):
            print(f"{atom}: {charge.item(): .6f}")

        print(f"sum: {result['charge_sum']:.12f}")
        print(f"dipole: {result['dipole'].tolist()}")


if __name__ == "__main__":
    run()
