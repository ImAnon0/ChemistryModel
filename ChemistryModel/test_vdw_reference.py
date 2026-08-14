import numpy as np
import pytest

import reactive as R
import vdw_reference as V


def central_force(function, distance, step=1e-6):
    return -(function(distance + step) - function(distance - step)) / (2 * step)


def local_minima(values):
    return np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1


def test_parameter_table_and_all_pair_combinations_are_complete():
    assert set(V.UFF_VDW) == set(V.ELEMENTS)
    assert len(list(V.unique_pairs())) == 10
    assert len(V.REACTIVE_SINGLE_BOND_LENGTH) == 10
    for first, second in V.unique_pairs():
        minimum, depth = V.pair_parameters(first, second)
        assert minimum > 0.0
        assert depth > 0.0


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_unlike_combination_is_symmetric(first, second):
    assert V.pair_parameters(first, second) == V.pair_parameters(second, first)
    distances = np.linspace(0.2, 9.0, 400)
    np.testing.assert_allclose(
        V.suppressed_vdw_energy(distances, first, second),
        V.suppressed_vdw_energy(distances, second, first),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_raw_minimum_and_derivative(first, second):
    minimum, depth = V.pair_parameters(first, second)
    assert V.raw_uff_energy(0.75 * minimum, first, second) > 0.0
    assert V.raw_uff_energy(minimum, first, second) == pytest.approx(-depth)
    assert V.raw_uff_force(minimum, first, second) == pytest.approx(0.0, abs=1e-14)
    for distance in np.linspace(0.6 * minimum, 8.4, 60):
        analytic = V.raw_uff_force(distance, first, second)
        numeric = central_force(
            lambda r: V.raw_uff_energy(r, first, second), distance
        )
        assert analytic == pytest.approx(numeric, rel=2e-6, abs=2e-8)


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_suppressed_derivative_and_finiteness(first, second):
    inner, outer = V.reactive_interval(first, second)
    distances = np.unique(np.concatenate((
        np.geomspace(0.05, inner, 100),
        np.linspace(inner - 1e-3, outer + 1e-3, 401),
        np.linspace(outer, 9.0, 500),
    )))
    energies = V.suppressed_vdw_energy(distances, first, second)
    forces = V.suppressed_vdw_force(distances, first, second)
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(forces))
    assert np.all((V.suppression_weight(distances, first, second) >= 0.0))
    assert np.all((V.suppression_weight(distances, first, second) <= 1.0))

    check = np.concatenate((
        np.linspace(inner + 2e-4, outer - 2e-4, 31),
        np.linspace(3.0, 6.8, 10),
        np.linspace(7.01, 8.49, 31),
    ))
    for distance in check:
        analytic = V.suppressed_vdw_force(distance, first, second)
        numeric = central_force(
            lambda r: V.suppressed_vdw_energy(r, first, second), distance
        )
        assert analytic == pytest.approx(numeric, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_suppression_and_cutoff_boundaries_are_c1(first, second):
    inner, outer = V.reactive_interval(first, second)
    for boundary in (inner, outer, V.CUTOFF_ON, V.CUTOFF):
        # Test the limiting values closely. A large but continuous force can
        # produce a sizeable finite energy difference at wider spacing; its
        # physical magnitude is reported separately by the diagnostics.
        delta = 1e-9
        left_e = V.suppressed_vdw_energy(boundary - delta, first, second)
        right_e = V.suppressed_vdw_energy(boundary + delta, first, second)
        left_f = V.suppressed_vdw_force(boundary - delta, first, second)
        right_f = V.suppressed_vdw_force(boundary + delta, first, second)
        assert abs(left_e - right_e) < 2e-5
        assert abs(left_f - right_f) < 2e-2
    assert V.suppressed_vdw_energy(V.CUTOFF, first, second) == 0.0
    assert V.suppressed_vdw_force(V.CUTOFF, first, second) == 0.0


def test_pair_distance_is_translation_rotation_and_permutation_invariant():
    first = np.array([0.2, -1.0, 0.5])
    second = np.array([1.7, 0.4, -0.2])
    distance = np.linalg.norm(second - first)
    base = V.suppressed_vdw_energy(distance, "C", "O")
    shift = np.array([4.0, -3.0, 2.0])
    assert V.suppressed_vdw_energy(
        np.linalg.norm((second + shift) - (first + shift)), "C", "O"
    ) == pytest.approx(base)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert V.suppressed_vdw_energy(
        np.linalg.norm(rotation @ (second - first)), "O", "C"
    ) == pytest.approx(base)


def test_off_is_exactly_zero():
    distances = np.linspace(0.01, 20.0, 100)
    assert np.array_equal(
        V.suppressed_vdw_energy(distances, "H", "O", enabled=False),
        np.zeros_like(distances),
    )
    assert np.array_equal(
        V.suppressed_vdw_force(distances, "H", "O", enabled=False),
        np.zeros_like(distances),
    )


@pytest.mark.parametrize("model", ("airebo_m", "reaxff"))
@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_literature_alternatives_match_outer_well_and_derivative(
    model, first, second
):
    energy_function, force_function = V.RAW_MODELS[model]
    minimum, depth = V.pair_parameters(first, second)
    assert energy_function(minimum, first, second) == pytest.approx(-depth)
    assert force_function(minimum, first, second) == pytest.approx(0.0, abs=1e-13)
    outer = V.OUTER_MATCH_RATIO * minimum
    assert energy_function(outer, first, second) == pytest.approx(
        V.raw_uff_energy(outer, first, second), rel=1e-12
    )
    distances = np.linspace(0.05, 8.4, 100)
    assert np.all(np.isfinite(energy_function(distances, first, second)))
    assert np.all(np.isfinite(force_function(distances, first, second)))
    for distance in np.linspace(0.2 * minimum, 8.3, 50):
        numeric = central_force(
            lambda r: energy_function(r, first, second), distance
        )
        assert force_function(distance, first, second) == pytest.approx(
            numeric, rel=5e-6, abs=2e-8
        )


@pytest.mark.parametrize("model", ("airebo_m", "reaxff"))
@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_suppressed_alternative_derivative(model, first, second):
    inner, outer = V.reactive_interval(first, second)
    distances = np.concatenate((
        np.linspace(inner + 2e-4, outer - 2e-4, 25),
        np.linspace(3.0, 6.8, 8),
        np.linspace(7.01, 8.49, 25),
    ))
    for distance in distances:
        numeric = central_force(
            lambda r: V.suppressed_vdw_energy(
                r, first, second, model=model
            ),
            distance,
        )
        analytic = V.suppressed_vdw_force(
            distance, first, second, model=model
        )
        assert analytic == pytest.approx(numeric, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize(
    "first,second", [("H", "H"), ("C", "H"), ("O", "H"),
                     ("C", "C"), ("C", "O"), ("N", "N"), ("O", "O")]
)
def test_combined_curve_has_no_transition_region_minimum(first, second):
    types = np.array([R.ELEMENT_INDEX[first], R.ELEMENT_INDEX[second]])
    distances = np.linspace(0.45, 6.5, 6000)
    reactive = np.array([
        R.potential_energy(
            np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]), types
        ) for distance in distances
    ])
    total = reactive + V.suppressed_vdw_energy(distances, first, second)
    minima = local_minima(total)
    inner, outer = V.reactive_interval(first, second)
    transition_minima = [
        index for index in minima
        if inner < distances[index] < outer
    ]
    assert transition_minima == []
