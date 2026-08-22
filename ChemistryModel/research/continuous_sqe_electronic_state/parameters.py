"""Provisional, unfitted parameters for the research-only C0 model.

These values are deliberately not described as a published force-field
convention or a ChemistryModel parameter fit.  They are benign numerical seed
values for testing the mathematical formulation before quantum-reference
parameterisation begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ElementC0Parameters:
    electronegativity_eV: float
    intrinsic_hardness_eV: float
    gaussian_sigma_A: float
    transfer_capacity_e2_per_eV: float
    covalent_radius_A: float


@dataclass(frozen=True)
class C0ParameterSet:
    elements: Mapping[int, ElementC0Parameters]
    capacity_radius_scale: float
    capacity_steepness: float
    support_inner_A: float = 3.5
    support_outer_A: float = 4.5
    coulomb_eV_A_per_e2: float = 14.3996454784255
    convention: str = "continuous_sqe_c0_provisional_seed_v1"

    def __post_init__(self):
        object.__setattr__(self, "elements", MappingProxyType(dict(self.elements)))
        if set(self.elements) != {1, 6, 7, 8}:
            raise ValueError("C0 requires exactly H/C/N/O elemental parameters")
        if self.elements[1].electronegativity_eV != 0.0:
            raise ValueError("the C0 electronegativity gauge fixes H to zero")
        for atomic_number, value in self.elements.items():
            if value.intrinsic_hardness_eV <= 0.0:
                raise ValueError(f"non-positive hardness for Z={atomic_number}")
            if value.gaussian_sigma_A <= 0.0:
                raise ValueError(f"non-positive Gaussian width for Z={atomic_number}")
            if value.transfer_capacity_e2_per_eV <= 0.0:
                raise ValueError(f"non-positive transfer capacity for Z={atomic_number}")
            if value.covalent_radius_A <= 0.0:
                raise ValueError(f"non-positive covalent radius for Z={atomic_number}")
        if self.capacity_radius_scale <= 0.0 or self.capacity_steepness <= 0.0:
            raise ValueError("capacity range parameters must be positive")
        if not 0.0 < self.support_inner_A < self.support_outer_A:
            raise ValueError("support radii must satisfy 0 < inner < outer")

    @property
    def independent_parameter_count(self):
        # 3 chi after one gauge + 4 hardness + 4 sigma + 4 capacity +
        # 1 radius scale + 1 steepness.
        return 17


# Electronegativity differences use the existing H/C/N/O ordering only as a
# scale-compatible initial guess.  Hardness, Gaussian width and capacity values
# are intentionally conservative numerical seeds, not a mixed literature fit.
C0_PARAMETERS = C0ParameterSet(
    elements={
        1: ElementC0Parameters(0.000, 10.0, 0.45, 0.080, 0.31),
        6: ElementC0Parameters(0.815, 8.0, 0.65, 0.100, 0.76),
        7: ElementC0Parameters(2.371, 9.0, 0.60, 0.095, 0.71),
        8: ElementC0Parameters(4.213, 10.0, 0.55, 0.090, 0.66),
    },
    capacity_radius_scale=1.25,
    capacity_steepness=2.5,
)
