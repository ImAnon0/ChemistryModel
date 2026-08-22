from chemistry_engine.terms.registry import build_extensions


def test_electrostatics_extension_is_registered():
    extensions = build_extensions(("electrostatics",))
    assert len(extensions) == 1
    assert extensions[0].name == "electrostatics"


def test_default_registry_keeps_existing_null_extension():
    extensions = build_extensions(("null",))
    assert extensions[0].name == "null"
