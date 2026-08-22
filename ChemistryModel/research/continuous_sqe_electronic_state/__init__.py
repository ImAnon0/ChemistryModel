"""Research-only continuous SQE electronic-state models."""

from .parameters import C0_PARAMETERS, C0ParameterSet

__all__ = [
    "C0_PARAMETERS",
    "C0ParameterSet",
    "C0ContinuousSQE",
    "C0Result",
]


def __getattr__(name):
    if name in {"C0ContinuousSQE", "C0Result"}:
        from .prototype import C0ContinuousSQE, C0Result

        return {"C0ContinuousSQE": C0ContinuousSQE, "C0Result": C0Result}[name]
    raise AttributeError(name)
