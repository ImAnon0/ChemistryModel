"""Golden direct-execution check for the frozen unified-radial ruler."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from research.unified_bond_capacity import (
    LegacyUnifiedRadialReference,
    UnifiedBondCapacityEnergyPrototype,
)
from unified_radial_equivalence import (
    compare_implementation_to_fixture,
    compare_implementations,
    load_fixture,
)


def main():
    differences = compare_implementation_to_fixture()
    fixture = load_fixture()
    differences.extend(compare_implementations(
        LegacyUnifiedRadialReference,
        UnifiedBondCapacityEnergyPrototype,
        fixture["cases"],
    ))
    if differences:
        print("UNIFIED RADIAL BASELINE EQUIVALENCE: FAIL")
        for difference in differences[:50]:
            print(f"  {difference}")
        raise SystemExit(1)
    print("UNIFIED RADIAL BASELINE EQUIVALENCE: PASS")
    print("  CPU float64 single/grouped fixture: exact within frozen tolerances")


if __name__ == "__main__":
    main()
