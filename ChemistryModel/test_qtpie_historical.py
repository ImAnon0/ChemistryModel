import numpy as np

from electrostatics import dipole_debye
from qtpie_compatibility import comparison_report
from qtpie_historical import (
    historical_matrices, solve_historical, sto_coulomb, sto_overlap,
)
from qtpie_validation import geometries


def test_rosen_integrals_are_symmetric_and_asymptotic():
    args = (1.0698,0.9745,1,2,2.0)
    np.testing.assert_allclose(
        sto_overlap(*args),sto_overlap(args[1],args[0],args[3],args[2],args[4]),rtol=1e-13
    )
    np.testing.assert_allclose(
        sto_coulomb(*args),sto_coulomb(args[1],args[0],args[3],args[2],args[4]),rtol=1e-13
    )
    assert abs(sto_coulomb(1.0698,0.9745,1,2,30.0)-1/30.0) < 1e-10


def test_original_water_ammonia_polarity_and_symmetry():
    for name, central in (("H2O",0),("NH3",0)):
        elements,positions = geometries()[name]
        result = solve_historical(elements,positions)
        assert result.charges[central] < 0
        assert result.charge_error_e < 1e-12
        assert dipole_debye(result.charges,positions)[1] > 0


def test_original_qtpie_dissociates_but_qeq_does_not():
    positions = [[0,0,0],[20,0,0]]
    qtpie = solve_historical(["H","O"],positions,"qtpie")
    qeq = solve_historical(["H","O"],positions,"qeq")
    assert abs(qtpie.charges[0]) < 1e-12
    assert abs(qeq.charges[0]) > 0.1


def test_report_exposes_projected_instability_not_just_solve_success():
    report = comparison_report()
    assert report["spectra"]["H2O"]["A_current_gaussian"]["projected_eigenvalues_eV"][0] < 0
    assert report["spectra"]["H2O"]["B_original_slater"]["projected_eigenvalues_eV"][0] > 0
    # Historical QTPIE(-H) has a real H2 stability defect despite zero charges.
    assert report["spectra"]["H2"]["B_original_slater"]["projected_eigenvalues_eV"][0] < 0
