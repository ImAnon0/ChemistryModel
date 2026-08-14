import numpy as np
import pytest

import vdw_partition as P
import vdw_reference as V


def central_force(function, distance, step=1e-6):
    return -(function(distance + step) - function(distance - step)) / (2 * step)


@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_wca_partition_exactly_reconstructs_shielded_curve(first, second):
    distances = np.linspace(0.05, 8.0, 301)
    values = P.reaxff_wca_components(distances, first, second)
    assert np.allclose(
        values["repulsive_energy"] + values["attractive_energy"],
        V.raw_reaxff_energy(distances, first, second),
    )
    assert np.allclose(
        values["repulsive_force"] + values["attractive_force"],
        V.raw_reaxff_force(distances, first, second),
    )


@pytest.mark.parametrize("architecture", P.ARCHITECTURES)
@pytest.mark.parametrize("first,second", list(V.unique_pairs()))
def test_partition_force_is_energy_derivative(architecture, first, second):
    inner, outer = V.reactive_interval(first, second)
    minimum, _ = V.pair_parameters(first, second)
    probes = np.concatenate((
        np.linspace(inner + 2e-4, outer - 2e-4, 11),
        np.array([minimum - 2e-4, minimum + 2e-4, 7.01, 8.49]),
    ))
    for distance in probes:
        energy, force = P.partition_energy_force(
            distance, first, second, architecture
        )
        numeric = central_force(
            lambda r: P.partition_energy_force(
                r, first, second, architecture
            )[0], distance,
        )
        assert np.isfinite(energy) and np.isfinite(force)
        assert force == pytest.approx(numeric, rel=3e-5, abs=3e-7)
