from dataclasses import replace

from chemistry_engine.config import PhysicsSpec


def test_extension_toggle_is_possible():
    spec = PhysicsSpec.unified_radial_v1(
        {},
        capacity_temperature=0.01,
        h_regularisation_temperature=1e-4,
    )

    assert spec.enabled_extensions == ()

    enabled = replace(
        spec,
        enabled_extensions=("electrostatics",),
    )

    assert enabled.enabled_extensions == ("electrostatics",)
