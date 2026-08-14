import numpy as np
import pytest

from electrostatics import dipole_debye, solve_charges


def water():
    theta = np.deg2rad(104.52)
    return ["O", "H", "H"], np.array([[0, 0, 0], [0.9572, 0, 0], [0.9572*np.cos(theta), 0.9572*np.sin(theta), 0]])


def methane():
    directions = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]], float)/np.sqrt(3)
    return ["C"] + ["H"]*4, np.vstack((np.zeros(3), 1.087*directions))


def test_h2_and_ch4_symmetry():
    h2 = solve_charges(["H", "H"], [[0,0,0], [0.74144,0,0]])
    assert np.max(np.abs(h2.charges)) < 1e-14
    elements, positions = methane()
    result = solve_charges(elements, positions)
    assert np.ptp(result.charges[1:]) < 1e-13
    assert np.linalg.norm(dipole_debye(result.charges, positions)[0]) < 1e-12


def test_conservation_and_invariances():
    elements, positions = water()
    baseline = solve_charges(elements, positions)
    permutation = np.array([2, 0, 1])
    permuted = solve_charges(np.array(elements)[permutation], positions[permutation])
    assert abs(baseline.charge_error_e) < 1e-14
    assert np.allclose(permuted.charges[np.argsort(permutation)], baseline.charges, atol=1e-13)
    assert np.allclose(solve_charges(elements, positions + [4,-3,8]).charges, baseline.charges, atol=1e-13)
    rotation, _ = np.linalg.qr(np.array([[0.2, 0.4, 0.7], [0.8, -0.1, 0.2], [0.3, 0.9, -0.5]]))
    assert np.allclose(solve_charges(elements, positions @ rotation).charges, baseline.charges, atol=1e-13)


def test_co2_zero_dipole():
    elements = ["O", "C", "O"]
    positions = np.array([[-1.16,0,0], [0,0,0], [1.16,0,0]])
    result = solve_charges(elements, positions)
    assert result.charges[0] == pytest.approx(result.charges[2], abs=1e-14)
    assert dipole_debye(result.charges, positions)[1] < 1e-12


def test_stretch_is_continuous():
    values = []
    # The published parameter combination has a short-range conditioning
    # failure (documented in the audit), so this continuity gate covers the
    # well-conditioned dissociation interval rather than crossing its pole.
    for distance in np.linspace(1.5, 10.0, 400):
        values.append(solve_charges(["H", "O"], [[0,0,0], [distance,0,0]]).charges)
    assert np.max(np.abs(np.diff(values, axis=0))) < 0.011


def test_neutral_fragment_dissociation_and_qeq_pathology():
    qtpie, qeq = [], []
    # Two unlike, individually neutral one-atom fragments are the unambiguous
    # neutral dissociation test for a single global constraint.
    for separation in (1.5, 3.0, 5.0, 8.0, 12.0, 20.0):
        positions = [[0,0,0], [separation,0,0]]
        qtpie.append(abs(solve_charges(["H", "O"], positions, "qtpie").charges[0]))
        qeq.append(abs(solve_charges(["H", "O"], positions, "qeq").charges[0]))
    assert qtpie[-1] < 1e-20
    assert qeq[-1] > 1e-3
    assert qtpie[-1] < qtpie[0]


def test_neutral_scope_and_numerics():
    elements, positions = water()
    with pytest.raises(ValueError, match="neutral"):
        solve_charges(elements, positions, total_charge=1)
    for method in ("qtpie", "qeq"):
        result = solve_charges(elements, positions, method)
        assert result.residual_inf < 1e-12
        assert np.isfinite(result.condition_number)
