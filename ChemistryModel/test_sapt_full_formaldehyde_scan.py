import numpy as np
import pytest

import hf_surface_scan as scan
import sapt_full_formaldehyde_scan as formaldehyde
from sapt_h_state_torch import (
    SAPT_H_STATE_MIXING,
    SaptHStateBatchedSimulation,
)


def test_configure_selects_existing_formaldehyde_system():
    previous = scan.ACTIVE_SYSTEM

    try:
        formaldehyde.configure_system()

        assert scan.ACTIVE_SYSTEM == "formaldehyde"

        assert scan.SYSTEMS[
            "formaldehyde"
        ][
            "geometry"
        ] is scan.formaldehyde_geometry

    finally:
        scan.apply_system(
            previous
        )


def test_scan_axes_use_registered_formaldehyde_window():
    donor, transfer = (
        formaldehyde.scan_axes(
            0.02,
            0.02,
        )
    )

    probe = scan.SYSTEM_PROBES[
        "formaldehyde"
    ]

    assert donor[
        0
    ] == pytest.approx(
        probe[
            "donor"
        ][
            0
        ]
    )

    assert transfer[
        0
    ] == pytest.approx(
        probe[
            "transfer"
        ][
            0
        ]
    )

    assert donor[
        -1
    ] >= (
        probe[
            "donor"
        ][
            1
        ]
        - 0.02
    )

    assert transfer[
        -1
    ] >= (
        probe[
            "transfer"
        ][
            1
        ]
        - 0.02
    )


def test_builder_uses_sapt_h_state_and_frozen_mixing():
    previous = scan.ACTIVE_SYSTEM

    try:
        model = (
            formaldehyde.build_sapt()
        )

        assert isinstance(
            model,
            SaptHStateBatchedSimulation,
        )

        assert model.h_state_mixing == pytest.approx(
            SAPT_H_STATE_MIXING
        )

        assert model.physics_model_revision == (
            "sapt-wall-v2-heavy-env"
        )

    finally:
        scan.apply_system(
            previous
        )


def test_formaldehyde_frozen_spectator_definition_is_unchanged():
    frozen = np.asarray(
        scan.SYSTEMS[
            "formaldehyde"
        ][
            "frozen"
        ],
        dtype=float,
    )

    assert np.allclose(
        frozen,
        np.array(
            [
                1.20,
                1.09,
                122.0,
                122.0,
            ]
        ),
    )


def test_project_references_are_read_from_scanner_not_refit_here():
    assert formaldehyde.reference_barrier_text() is not None

    assert scan.REFERENCE_REACTION == pytest.approx(
        -0.718
    )

    # The diagnostic imports the currently frozen mixing instead of defining
    # a new fit for formaldehyde.
    assert SAPT_H_STATE_MIXING == pytest.approx(
        0.534590721
    )
