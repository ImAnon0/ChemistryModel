"""Fast invariants protecting the current model during bond calibration."""

import numpy as np
import torch

import bond_calibration as calibration
import reactive as R
from reactive_torch import ReactiveSimulation


def test_h2_curve_has_one_well_and_no_dissociation_hump():
    result = calibration.h2_curve()
    assert abs(result["sampled_re_A"] - result["table_re_A"]) < 0.01
    assert result["sampled_well_eV"] > 4.0
    assert abs(result["energy_at_3A_eV"]) < 1e-6
    # After the minimum the energy must rise monotonically toward zero.
    assert result["post_minimum_falling_steps"] == 0


def test_numpy_table_and_morse_formula_agree_at_h2_minimum():
    i = R.ELEMENT_INDEX["H"]
    depth = float(R.BOND_DEPTH[i, i])
    length = float(R.BOND_LENGTH[i, i])
    energy = R.potential_energy(
        np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]),
        R.types_from_symbols(["H", "H"]),
    )
    assert abs(float(energy) + depth) < 1e-10


def test_h2_harmonic_diagnostic_is_finite_and_near_reference():
    predicted = calibration.h2_curve()["harmonic_cm-1"]
    assert np.isfinite(predicted)
    assert abs(predicted - calibration.H2_REFERENCE["omega_e_cm-1"]) < 250.0


def test_torch_receives_exact_numpy_h2_parameters():
    symbols = ["H", "H"]
    positions = np.array([[5.0, 5.0, 5.0], [5.74144, 5.0, 5.0]])
    simulation = ReactiveSimulation(
        symbols, positions, 12.0, target_temperature=0.0,
        device="cpu", dtype=torch.float64, random_seed=19,
    )
    i = R.ELEMENT_INDEX["H"]
    assert float(simulation.bond_length[i, i]) == float(R.BOND_LENGTH[i, i])
    assert float(simulation.bond_depth[i, i]) == float(R.BOND_DEPTH[i, i])
    assert float(simulation.bond_width[i, i]) == float(R.BOND_WIDTH[i, i])


def test_methane_ch_coordinate_has_stable_minimum_and_capture_path():
    result = calibration.methane_ch_coordinate()
    assert abs(result["sampled_minimum_A"] - result["table"]["re_A"]) < 0.01
    assert result["dissociation_coordinate_eV"] > 4.0
    assert result["short_range_energy_eV"] > 0.0
    assert result["capture_region_falling_steps"] == 0


def test_ammonia_nh_coordinate_has_stable_minimum_and_capture_path():
    result = calibration.ammonia_nh_coordinate()
    assert abs(result["sampled_minimum_A"] - result["table"]["re_A"]) < 0.01
    assert result["dissociation_coordinate_eV"] > 4.0
    assert result["short_range_energy_eV"] > 0.0
    assert result["capture_region_falling_steps"] == 0


def test_water_oh_coordinate_has_stable_minimum_and_capture_path():
    result = calibration.water_oh_coordinate()
    assert abs(result["sampled_minimum_A"] - result["table"]["re_A"]) < 0.01
    assert result["dissociation_coordinate_eV"] > 4.5
    assert result["short_range_energy_eV"] > 0.0
    assert result["capture_region_falling_steps"] == 0


def test_ethane_cc_coordinate_has_stable_minimum_and_capture_path():
    result = calibration.ethane_cc_coordinate()
    assert abs(result["sampled_minimum_A"] - result["table"]["re_A"]) < 0.03
    assert result["dissociation_coordinate_eV"] > 3.0
    assert result["short_range_energy_eV"] > 0.0
    assert result["capture_region_falling_steps"] == 0


def test_methylamine_cn_coordinate_has_stable_minimum_and_capture_path():
    result = calibration.methylamine_cn_coordinate()
    assert abs(result["sampled_minimum_A"] - result["table"]["re_A"]) < 0.04
    assert result["dissociation_coordinate_eV"] > 2.5
    assert result["short_range_energy_eV"] > 0.0
    assert result["capture_region_falling_steps"] == 0


def test_small_molecule_nve_baselines_remain_numerically_stable():
    for name in ("H2", "CH4", "NH3", "H2O"):
        result = calibration.molecule_nve(name, steps=200)
        assert abs(result["drift_eV"]) < 0.05, (name, result)
        assert result["capped_steps"] == 0, (name, result)

    ethane = calibration.ethane_nve(steps=200)
    assert abs(ethane["drift_eV"]) < 0.05, ethane
    assert ethane["capped_steps"] == 0, ethane

    methylamine = calibration.methylamine_nve(steps=200)
    assert abs(methylamine["drift_eV"]) < 0.05, methylamine
    assert methylamine["capped_steps"] == 0, methylamine


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS ", test.__name__)
        except Exception as problem:
            failures += 1
            print("FAIL ", test.__name__, problem)
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(bool(failures))
