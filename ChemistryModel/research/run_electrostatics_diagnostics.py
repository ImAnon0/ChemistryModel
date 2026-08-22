from research.electrostatics_diagnostics import (
    inspect_qeq_state,
    known_diagnostic_cases,
)


def main():
    for name, (positions, atoms) in known_diagnostic_cases().items():
        result = inspect_qeq_state(positions, atoms)

        print()
        print("=" * 40)
        print(name)
        print("=" * 40)

        print("charges:")
        for atom, charge in zip(atoms, result["charges"]):
            print(f"  {atom}: {charge.item(): .6f}")

        print(f"charge sum: {result['charge_sum']:.12f}")
        print(f"dipole: {result['dipole'].tolist()}")


if __name__ == "__main__":
    main()