
from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from pathlib import Path

from valence_state_batched_membership_torch import (
    BatchedHeavyValenceStateBatchedSimulation,
)


def fresh_run_diagnostics():
    return {
        "evaluation_count": 0,
        "max_candidate_count": 0,
        "max_state_count": 0,
        "centre_evaluations_over_128": 0,
        "centre_evaluations_over_200": 0,
        "evaluations_with_over_128": 0,
        "evaluations_with_over_200": 0,
        "max_topology_group": 0,
        "max_total_states_solved": 0,
        "max_competitive_atom_count": 0,
        "max_state_shape": (),
    }


def dummy_solver():
    solver = object.__new__(
        BatchedHeavyValenceStateBatchedSimulation
    )
    solver._heavy_valence_run_diagnostics = (
        fresh_run_diagnostics()
    )
    return solver


def main():
    print("HEAVY-VALENCE STATE-PRESSURE DIAGNOSTICS VALIDATION")
    print()

    solver = dummy_solver()

    first = {
        "largest_candidate_count": 10,
        "largest_state_count": 210,
        "largest_state_shape": (10, 4, 210),
        "centres_over_128": 2,
        "centres_over_200": 1,
        "largest_topology_group": 3,
        "total_states_solved": 630,
        "competitive_atom_count": 5,
    }

    second = {
        "largest_candidate_count": 8,
        "largest_state_count": 70,
        "largest_state_shape": (8, 4, 70),
        "centres_over_128": 0,
        "centres_over_200": 0,
        "largest_topology_group": 7,
        "total_states_solved": 490,
        "competitive_atom_count": 8,
    }

    solver._record_heavy_valence_run_diagnostics(first)
    solver._record_heavy_valence_run_diagnostics(second)

    result = solver._heavy_valence_run_diagnostics

    expected = {
        "evaluation_count": 2,
        "max_candidate_count": 10,
        "max_state_count": 210,
        "centre_evaluations_over_128": 2,
        "centre_evaluations_over_200": 1,
        "evaluations_with_over_128": 1,
        "evaluations_with_over_200": 1,
        "max_topology_group": 7,
        "max_total_states_solved": 630,
        "max_competitive_atom_count": 8,
        "max_state_shape": (10, 4, 210),
    }

    accumulator_pass = result == expected

    print(f"accumulated : {result}")
    print(f"expected    : {expected}")
    print()
    print(
        "accumulator semantics : "
        + ("PASS" if accumulator_pass else "FAIL")
    )

    valence_source = Path(
        "valence_state_batched_membership_torch.py"
    ).read_text(encoding="utf-8")

    runner_source = Path(
        "batch_runner.py"
    ).read_text(encoding="utf-8")

    live_counter_pass = all(
        token in valence_source
        for token in (
            "largest_candidate_count = 0",
            "largest_state_count = 0",
            "if state_count > 128:",
            "if state_count > 200:",
            "_record_heavy_valence_run_diagnostics(",
        )
    )

    runner_pass = all(
        token in runner_source
        for token in (
            '"heavy_valence_group_diagnostics"',
            "heavy valence state pressure (group)",
            "max_state_count",
            "centre_evaluations_over_128",
        )
    )

    print(
        "live-counter wiring   : "
        + ("PASS" if live_counter_pass else "FAIL")
    )
    print(
        "runner persistence    : "
        + ("PASS" if runner_pass else "FAIL")
    )

    passed = (
        accumulator_pass
        and live_counter_pass
        and runner_pass
    )

    print()

    if passed:
        print(
            "FINAL PASS - state-pressure diagnostics accumulate correctly "
            "and are persisted/reported without changing the solver."
        )
        return

    print(
        "FINAL FAIL - do not use the diagnostic production run yet."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
