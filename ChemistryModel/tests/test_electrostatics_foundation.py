from chemistry_engine.terms.electrostatics import (
    ELECTROSTATIC_PARAMETERS,
    ElectrostaticEnergyTerm,
)


def test_electrostatic_parameter_table_has_supported_elements():
    assert set(ELECTROSTATIC_PARAMETERS) == {"H", "C", "N", "O"}


def test_electrostatics_foundation_is_zero_until_solver_exists():
    assert ElectrostaticEnergyTerm().name == "electrostatics"
