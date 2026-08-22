from __future__ import annotations

from .electrostatics import ElectrostaticEnergyTerm
from .null import NullEnergyTerm


_EXTENSION_BUILDERS = {
    "null": NullEnergyTerm,
    "electrostatics": ElectrostaticEnergyTerm,
}


def register_extension(name, builder):
    existing = _EXTENSION_BUILDERS.get(str(name))
    if existing is not None and existing is not builder:
        raise ValueError(f"extension already registered: {name}")

    _EXTENSION_BUILDERS[str(name)] = builder


def build_extensions(names):
    extensions = []

    for name in names:
        builder = _EXTENSION_BUILDERS[str(name)]

        if str(name) == "electrostatics":
            extensions.append(builder(enabled=True))
        else:
            extensions.append(builder())

    return tuple(extensions)


def registered_extensions():
    return tuple(sorted(_EXTENSION_BUILDERS))