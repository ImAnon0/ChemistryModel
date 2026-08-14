import numpy as np
import pytest

import dispersion_reference as D
import vdw_reference as V


def central_force(function, distance, step=1e-6):
    return -(function(distance + step) - function(distance - step)) / (2 * step)


def test_tang_toennies_limits():
    assert D.tang_toennies_f6(0.0) == pytest.approx(0.0)
    assert D.tang_toennies_f6(100.0) == pytest.approx(1.0)


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_uff_c6_recovers_audited_long_range_tail(first, second):
    minimum, depth = V.pair_parameters(first, second)
    assert D.c6_coefficient(first, second) == pytest.approx(
        2.0 * depth * minimum ** 6
    )


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_dispersion_is_finite_and_force_matches_energy(first, second):
    for distance in np.linspace(0.05, 8.0, 80):
        energy = D.dispersion_energy(distance, first, second)
        force = D.dispersion_force(distance, first, second)
        assert np.isfinite(energy) and np.isfinite(force)
        assert force == pytest.approx(
            central_force(lambda r: D.dispersion_energy(
                r, first, second
            ), distance), rel=2e-5, abs=2e-7
        )


@pytest.mark.parametrize("convention", ("morse_proxy", "born_mayer_ip"))
@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_dispersion_recovers_minus_c6_tail(convention, first, second):
    distance = 30.0
    expected = -D.c6_coefficient(first, second) / distance ** 6
    assert D.dispersion_energy(
        distance, first, second, convention=convention
    ) == pytest.approx(
        expected, rel=1e-8
    )


def test_born_mayer_ip_equal_pair_rule_returns_atomic_exponent():
    for element in V.ELEMENTS:
        assert D.born_mayer_ip_exponent(element, element) == pytest.approx(
            D.atomic_ip_density_exponent(element)
        )


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_born_mayer_ip_dispersion_force_matches_energy(first, second):
    for distance in np.linspace(0.1, 8.0, 30):
        force = D.dispersion_force(
            distance, first, second, convention="born_mayer_ip"
        )
        numeric = central_force(
            lambda r: D.dispersion_energy(
                r, first, second, convention="born_mayer_ip"
            ), distance,
        )
        assert force == pytest.approx(numeric, rel=2e-5, abs=2e-7)
