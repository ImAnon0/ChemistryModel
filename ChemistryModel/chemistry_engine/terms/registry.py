from __future__ import annotations

from .null import NullEnergyTerm


_EXTENSION_BUILDERS = {
    "null": NullEnergyTerm,
}


def register_extension(name, builder):
    existing = _EXTENSION_BUILDERS.get(str(name))
    if existing is not None and existing is not builder:
        raise ValueError(f"extension already registered: {name}")
    _EXTENSION_BUILDERS[str(name)] = builder


def build_extensions(names):
    return tuple(
        _EXTENSION_BUILDERS[str(name)]()
        for name in names
    )


def registered_extensions():
    return tuple(sorted(_EXTENSION_BUILDERS))
