from chemistry_engine.terms.electrostatics import (
    ELECTROSTATIC_PARAMETERS,
    ElectrostaticEnergyTerm,
)


def test_qeq_parameter_table_contains_chno():
    assert set(ELECTROSTATIC_PARAMETERS) == {"H", "C", "N", "O"}


def test_qeq_skeleton_has_charge_solver_boundary():
    assert hasattr(ElectrostaticEnergyTerm(), "solve_charges")
