"""Explicit real-Psi4 smoke test; opt in because it is comparatively expensive."""

import os

import numpy as np
import pytest

import qm_structure_validator as qsv


pytestmark = pytest.mark.skipif(
    os.environ.get("CHEMISTRYMODEL_RUN_PSI4_TEST") != "1",
    reason="set CHEMISTRYMODEL_RUN_PSI4_TEST=1 in a Psi4 environment",
)


def test_h2_single_point_and_optimisation_smoke():
    pytest.importorskip("psi4")
    result = qsv.Psi4Runner(threads=1, memory="1 GB").run(
        ["H", "H"],
        np.asarray([[-0.40, 0.0, 0.0], [0.40, 0.0, 0.0]]),
        0, 1, qsv.DEFAULT_METHOD, qsv.DEFAULT_BASIS, "rhf",
    )
    assert np.isfinite(result["single_point_energy_hartree"])
    assert np.all(np.isfinite(result["gradient_hartree_per_bohr"]))
    assert result["optimisation_converged"] is True
    assert result["optimised_coordinates_A"].shape == (2, 3)
