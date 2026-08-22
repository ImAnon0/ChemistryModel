from research.electrostatics_diagnostics import known_diagnostic_cases, inspect_qeq_state


def test_qeq_diagnostic_cases_are_neutral():
    for positions, atoms in known_diagnostic_cases().values():
        result = inspect_qeq_state(positions, atoms)
        assert abs(result["charge_sum"]) < 1e-10


def test_qeq_diagnostics_return_charge_and_dipole():
    result = inspect_qeq_state(
        [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        (1, 1),
    )

    assert len(result["charges"]) == 2
    assert len(result["dipole"]) == 3
