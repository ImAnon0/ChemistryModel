
from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from pathlib import Path
from types import SimpleNamespace

import batch_runner

from batched_torch import BatchedReactiveSimulation
from valence_state_optimised_torch import (
    OptimisedValenceStateBatchedSimulation,
)


class RunnerOptions(SimpleNamespace):
    """
    Minimal options stand-in for condition_key().

    ChemistryModel's runner grows new recording/performance options fairly
    often. They are irrelevant to this wiring smoke test, so any unrelated
    attribute that was not explicitly supplied defaults to numeric zero.

    The two fields under test -- physics and mixture -- are always explicit.
    """

    def __getattr__(self, name):
        return 0.0


def condition_options(physics):
    return RunnerOptions(
        physics=physics,
        mixture="smoke",
        box=12.0,
        picoseconds=0.05,
        hot_temperature=500.0,
        hot_until_fs=2000.0,
        cool_temperature=250.0,
        capture_every=40,
        time_step=0.25,
    )


def main():
    print("RUNNER / LAB OPT-IN PHYSICS SMOKE TEST")
    print()

    # ------------------------------------------------------------
    # 1. The selector must map the two public names to the correct
    #    concrete simulation classes.
    # ------------------------------------------------------------
    reactive_class = batch_runner.grouped_simulation_class(
        RunnerOptions(physics="reactive")
    )

    valence_class = batch_runner.grouped_simulation_class(
        RunnerOptions(physics="optimised-valence")
    )

    selector_pass = (
        reactive_class is BatchedReactiveSimulation
        and valence_class is OptimisedValenceStateBatchedSimulation
    )

    # ------------------------------------------------------------
    # 2. Physics must be part of the condition key so two otherwise
    #    identical experiments cannot be pooled into one batch.
    # ------------------------------------------------------------
    reactive_key = batch_runner.condition_key(
        condition_options("reactive")
    )

    valence_key = batch_runner.condition_key(
        condition_options("optimised-valence")
    )

    identity_pass = (
        reactive_key.get("physics") == "reactive"
        and valence_key.get("physics") == "optimised-valence"
        and reactive_key != valence_key
    )

    # Everything except physics should remain identical in this control.
    other_key_pass = (
        {
            key: value
            for key, value in reactive_key.items()
            if key != "physics"
        }
        ==
        {
            key: value
            for key, value in valence_key.items()
            if key != "physics"
        }
    )

    # ------------------------------------------------------------
    # 3. Structural guards: default remains reactive, continuation
    #    is intentionally not routed through the new model, metadata
    #    records model identity, and Lab exposes the selector.
    # ------------------------------------------------------------
    batch_source = Path("batch_runner.py").read_text(
        encoding="utf-8"
    )

    lab_source = Path("lab.py").read_text(
        encoding="utf-8"
    )

    default_pass = (
        'default="reactive"' in batch_source
        and "not wired to continuation yet" in batch_source
    )

    runner_wiring_pass = all(
        token in batch_source
        for token in (
            "def grouped_simulation_class(",
            "SimulationClass = grouped_simulation_class(options)",
            '"physics": getattr(options, "physics", "reactive")',
            '"physics_model"',
            '"physics_model_revision"',
            "_optimised_valence",
        )
    )

    lab_wiring_pass = all(
        token in lab_source
        for token in (
            "self.physics_box = QtWidgets.QComboBox()",
            '"optimised-valence"',
            '"--physics"',
            '"physics": self.physics_box.currentData()',
            'first.get("physics", "reactive")',
            "def on_physics_changed(self):",
        )
    )

    print(f"reactive selector : {reactive_class.__name__}")
    print(f"valence selector  : {valence_class.__name__}")
    print()
    print(
        "selector mapping       : "
        + ("PASS" if selector_pass else "FAIL")
    )
    print(
        "condition separation   : "
        + ("PASS" if identity_pass else "FAIL")
    )
    print(
        "only physics differs   : "
        + ("PASS" if other_key_pass else "FAIL")
    )
    print(
        "reactive default/guard : "
        + ("PASS" if default_pass else "FAIL")
    )
    print(
        "runner wiring          : "
        + ("PASS" if runner_wiring_pass else "FAIL")
    )
    print(
        "Lab wiring             : "
        + ("PASS" if lab_wiring_pass else "FAIL")
    )

    passed = all((
        selector_pass,
        identity_pass,
        other_key_pass,
        default_pass,
        runner_wiring_pass,
        lab_wiring_pass,
    ))

    print()

    if passed:
        print(
            "FINAL PASS - optimised valence is explicit opt-in for fresh "
            "grouped runs; historical reactive physics remains default, "
            "and the two physics models cannot silently pool."
        )
        return

    print(
        "FINAL FAIL - do not launch the production valence batch yet."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
