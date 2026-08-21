from chemistry_engine.terms.registry import build_extensions


def test_empty_extension_selection():
    assert build_extensions(()) == ()


def test_null_extension_selection():
    extensions = build_extensions(("null",))
    assert len(extensions) == 1
    assert extensions[0].name == "null"
