
from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from pathlib import Path

import numpy as np
import torch

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


def live_solver_diagnostics():
    """Prove the production membership path updates run diagnostics live.

    One carbon is surrounded by five carbon contacts in a trigonal-bipyramidal
    arrangement. Carbon has valence four, so the centre must enter the
    competitive N=5, V=4 heavy-state path (five local states).

    The constructor evaluates forces once. We then evaluate once more and
    require the cumulative evaluation counter to increase. This checks actual
    execution rather than searching source code for variable names.
    """
    centre = np.array([10.0, 10.0, 10.0], dtype=float)
    radius = 1.70

    directions = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-0.5, 0.8660254037844386, 0.0],
        [-0.5, -0.8660254037844386, 0.0],
    ], dtype=float)

    positions = np.vstack((
        centre,
        centre + radius * directions,
    ))

    simulation = BatchedHeavyValenceStateBatchedSimulation(
        boxes=[(["C"] * len(positions), positions)],
        box_size=30.0,
        target_temperature=0.0,
        friction=0.0,
        device="cpu",
        dtype=torch.float64,
        random_seed=0,
        relax_on_start=False,
    )

    first_run = dict(
        getattr(
            simulation,
            "_heavy_valence_run_diagnostics",
            {},
        )
    )
    first_current = dict(
        getattr(
            simulation,
            "_heavy_valence_diagnostics",
            {},
        )
    )

    simulation.compute_forces()

    second_run = dict(
        getattr(
            simulation,
            "_heavy_valence_run_diagnostics",
            {},
        )
    )

    passed = (
        int(first_run.get("evaluation_count", 0)) >= 1
        and int(second_run.get("evaluation_count", 0))
        > int(first_run.get("evaluation_count", 0))
        and int(first_current.get("largest_candidate_count", 0)) >= 5
        and int(first_current.get("largest_state_count", 0)) >= 5
        and int(first_current.get("competitive_atom_count", 0)) >= 1
        and int(second_run.get("max_candidate_count", 0)) >= 5
        and int(second_run.get("max_state_count", 0)) >= 5
        and int(second_run.get("max_competitive_atom_count", 0)) >= 1
    )

    return passed, first_current, first_run, second_run


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

    live_counter_pass, live_current, live_before, live_after = (
        live_solver_diagnostics()
    )

    print(f"live current         : {live_current}")
    print(f"live cumulative pre  : {live_before}")
    print(f"live cumulative post : {live_after}")
    print()

    runner_source = Path(
        "batch_runner.py"
    ).read_text(encoding="utf-8")

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
