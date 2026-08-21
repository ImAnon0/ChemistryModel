"""Explicit registry for canonical model construction."""

from __future__ import annotations


_BUILDERS = {}


def register(model_id, builder):
    existing = _BUILDERS.get(str(model_id))
    if existing is not None and existing is not builder:
        raise ValueError(f"physics model already registered: {model_id}")
    _BUILDERS[str(model_id)] = builder


def build(model_id, *args, **kwargs):
    try:
        builder = _BUILDERS[str(model_id)]
    except KeyError as error:
        raise ValueError(f"unknown canonical physics model: {model_id}") from error
    return builder(*args, **kwargs)


def registered_models():
    return tuple(sorted(_BUILDERS))
