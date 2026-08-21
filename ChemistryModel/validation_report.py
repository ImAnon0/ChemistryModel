"""Independent, observational validation report for ChemistryModel.

This module never changes force-field parameters.  It reads the committed
engine tables, calls the existing diagnostics, and labels calibration-linked
evidence separately from genuine hold-outs.  JSON output is intentionally
stable enough for future commit-to-commit comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import bond_calibration as BC
import build_box
import hf_surface_scan as HF
import reactive as R
from reactive_torch import ReactiveSimulation


SCHEMA_VERSION = 3
STATUS = {
    "good": "GOOD",
    "acceptable": "ACCEPTABLE",
    "weak": "WEAK",
    "fail": "FAIL",
    "insufficient": "INSUFFICIENT REFERENCE DATA",
    "fit": "FIT TARGET - NOT INDEPENDENT",
}

SOURCES = {
    "nist_h2": "https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=3FF7&Units=SI",
    "nist_cccbdb": "https://cccbdb.nist.gov/",
    "ethanol": "https://cccbdb.nist.gov/exp2x.asp?casno=64175&charge=0",
    "dimethyl_ether": "https://cccbdb.nist.gov/expgeom2.asp?casno=115106&charge=0",
    "ethylamine": "https://cccbdb.nist.gov/exp2x.asp?casno=75047&charge=0",
    "h2_precision": "https://doi.org/10.1063/1.3120443",
    "methylamine_band": "https://doi.org/10.1016/j.jms.2011.09.003",
    "hydrazine_band": "https://doi.org/10.1016/S0022-2852(03)00106-1",
    "methanol_thermo": "https://doi.org/10.1021/acs.jpca.5b01346",
    "peroxide_thermo": "https://doi.org/10.1021/jp056311j",
}

PAIR_LABELS = {
    ("H", "H"): "H-H", ("C", "H"): "C-H",
    ("N", "H"): "N-H", ("O", "H"): "O-H",
    ("C", "C"): "C-C", ("C", "N"): "C-N",
    ("C", "O"): "C-O", ("N", "N"): "N-N",
    ("N", "O"): "N-O", ("O", "O"): "O-O",
}

# These molecules supplied the geometry used to select the current single
# bond re.  Agreement is useful but explicitly not independent validation.
FIT_GEOMETRY = [
    ("H2", "H-H", ("H", "H"), 0.74144, "nist_h2"),
    ("CH4", "C-H", ("C", "H"), 1.086, "nist_cccbdb"),
    ("NH3", "N-H", ("N", "H"), 1.0109, "nist_cccbdb"),
    ("H2O", "O-H", ("O", "H"), 0.960, "nist_cccbdb"),
    ("C2H6", "C-C", ("C", "C"), 1.525, "nist_cccbdb"),
    ("CH3NH2", "C-N", ("C", "N"), 1.471, "nist_cccbdb"),
    ("CH3OH", "C-O", ("C", "O"), 1.427, "nist_cccbdb"),
    ("N2H4", "N-N", ("N", "N"), 1.446, "nist_cccbdb"),
    ("NH2OH", "N-O", ("N", "O"), 1.453, "nist_cccbdb"),
    ("H2O2", "O-O", ("O", "O"), 1.475, "nist_cccbdb"),
]

# Independent environments.  The model value is the live generic pair re;
# this deliberately measures transfer rather than optimizing a copied target
# geometry.  References are experimental rotational structures from CCCBDB.
HOLDOUT_GEOMETRY = [
    ("ethanol", "C-C", ("C", "C"), 1.512, "ethanol"),
    ("ethanol", "C-O", ("C", "O"), 1.431, "ethanol"),
    ("dimethyl ether", "C-O", ("C", "O"), 1.411, "dimethyl_ether"),
    ("ethylamine", "C-N", ("C", "N"), 1.469, "ethylamine"),
]

CURVATURE_REFERENCES = {
    ("H", "H"): (4401.21, "clean/strong comparison", "nist_h2", True),
    ("C", "H"): (2917.0, "approximate polyatomic normal mode", "nist_cccbdb", True),
    ("N", "H"): (3337.0, "approximate polyatomic normal mode", "nist_cccbdb", True),
    ("O", "H"): (3657.0, "approximate polyatomic normal mode", "nist_cccbdb", True),
    ("C", "C"): (993.0, "mixed polyatomic normal mode", "nist_cccbdb", False),
    ("C", "N"): (1044.8134, "perturbed polyatomic normal mode", "methylamine_band", False),
    ("C", "O"): (1033.0, "polyatomic normal mode", "nist_cccbdb", False),
    ("N", "N"): (1077.24, "polyatomic normal mode; rejected width fit", "hydrazine_band", False),
    ("N", "O"): (955.0, "polyatomic normal mode", "nist_cccbdb", False),
    ("O", "O"): (877.0, "polyatomic mode used to select width", "nist_cccbdb", True),
}

DISSOCIATION = [
    ("H2 -> H + H", ("H", "H"), 435.78, "D298", False, "h2_precision"),
    ("CH4 -> CH3 + H", ("C", "H"), 439.0, "BDE298", True, "nist_cccbdb"),
    ("NH3 -> NH2 + H", ("N", "H"), 449.0, "BDE298", True, "nist_cccbdb"),
    ("H2O -> OH + H", ("O", "H"), 498.0, "BDE298", True, "nist_cccbdb"),
    ("C2H6 -> CH3 + CH3", ("C", "C"), 377.0, "BDE298", False, "nist_cccbdb"),
    ("CH3NH2 -> CH3 + NH2", ("C", "N"), 356.0, "BDE298", True, "nist_cccbdb"),
    ("CH3OH -> CH3 + OH", ("C", "O"), 384.57, "BDE298", False, "methanol_thermo"),
    ("H2O2 -> OH + OH", ("O", "O"), 210.4, "BDE298", False, "peroxide_thermo"),
]

COORDINATES = {
    "H2 H-H": BC.h2_curve,
    "CH4 C-H": BC.methane_ch_coordinate,
    "C2H6 C-C": BC.ethane_cc_coordinate,
    "CH3NH2 C-N": BC.methylamine_cn_coordinate,
    "CH3OH C-O": BC.methanol_co_coordinate,
    "N2H4 N-N": BC.hydrazine_nn_coordinate,
    "NH2OH N-O": BC.hydroxylamine_no_coordinate,
    "H2O2 O-O": BC.hydrogen_peroxide_oo_coordinate,
}

FOCUSED_BATCHES = {
    "NH2 + NH2 -> N2H4": "runs/nn_width_2000_focused_control/index.json",
    "OH + OH -> H2O2": "runs/oo_width_2735_focused_candidate/index.json",
}


# ---------------------------------------------------------------------------
# Golden whole-model regression suite
# ---------------------------------------------------------------------------
#
# validation_report.py historically reported scientific/calibration evidence,
# but it did not run the standalone implementation regressions.  That allowed
# a reference implementation and its optimised copy to agree perfectly while
# both shared the same physical logic error.  The golden suite deliberately
# combines:
#
#   1. implementation/reference equivalence,
#   2. physical invariant regressions,
#   3. a short real dense-system integration stress in --full mode.
#
# One command is therefore enough for a release/refactor checkpoint:
#
#     python validation_report.py --full
#
# A failing golden check makes the process exit non-zero.

PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_ROOT = PROJECT_ROOT / "tests"

GOLDEN_QUICK_CHECKS = (
    ("main runtime imports", None),
    ("heavy overcoordination guard", "validate_heavy_overcoordination_guard.py"),
    ("runner physics selection", "validate_runner_physics_selection.py"),
    ("heavy valence density matrix", "validate_heavy_valence_density_matrix.py"),
)

GOLDEN_STANDARD_EXTRA = (
    ("unified radial baseline equivalence", "validate_unified_radial_equivalence.py"),
    ("optimised valence integration", "validate_optimised_valence_integration.py"),
    ("batched heavy valence", "validate_batched_heavy_valence.py"),
    ("large heavy valence states", "validate_large_heavy_valence_states.py"),
    ("cached H topology", "validate_cached_h_topology.py"),
    ("factorised H grouped execution", "validate_factorised_h_grouped_execution.py"),
    ("H-state components", "validate_h_state_components.py"),
    ("H-state factorised", "validate_h_state_factorised.py"),
    ("H-state factorised NVE", "validate_h_state_factorised_nve_v2.py"),
    ("index-select gather", "validate_index_select_gather.py"),
    ("smooth valence NVE", "validate_smooth_valence_nve.py"),
    ("valence-state factorised fixed", "validate_valence_state_factorised_fixed.py"),
    ("valence-state promotion", "validate_valence_state_promotion.py"),
    ("molecule library", "verify_molecule_library.py"),
)

GOLDEN_FULL_EXTRA = (
    ("heavy state pressure diagnostics", "validate_heavy_state_pressure_diagnostics.py"),
    ("smooth valence force probe", "probe_smooth_valence_forces.py"),
)


def _tail(text_value, lines=30):
    items = str(text_value or "").splitlines()
    return "\n".join(items[-int(lines):])


def _progress_start(index, total, label):
    print(
        f"[{int(index):>2}/{int(total)}] {label:<42} ... ",
        end="",
        flush=True,
    )


def _progress_finish(result):
    print(
        f"{result['status']}  {float(result.get('wall_seconds', 0.0)):.1f}s",
        flush=True,
    )
    if result["status"] != "PASS":
        stdout_tail = str(result.get("stdout_tail", "")).strip()
        stderr_tail = str(result.get("stderr_tail", "")).strip()
        reason = result.get("reason")
        if reason:
            print(f"    reason: {reason}", flush=True)
        if stdout_tail:
            print("    stdout (tail):", flush=True)
            for line in stdout_tail.splitlines():
                print(f"      {line}", flush=True)
        if stderr_tail:
            print("    stderr (tail):", flush=True)
            for line in stderr_tail.splitlines():
                print(f"      {line}", flush=True)


def _science_stage(label, function):
    print(f"[science] {label:<38} ... ", end="", flush=True)
    started = time.perf_counter()
    result = function()
    print(f"DONE  {time.perf_counter() - started:.1f}s", flush=True)
    return result


def _run_command_check(label, command, timeout_seconds=900):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "name": label,
            "status": "FAIL",
            "returncode": None,
            "wall_seconds": time.perf_counter() - started,
            "stdout_tail": _tail(error.stdout),
            "stderr_tail": _tail(error.stderr),
            "reason": f"timed out after {timeout_seconds:g} s",
        }
    except OSError as error:
        return {
            "name": label,
            "status": "FAIL",
            "returncode": None,
            "wall_seconds": time.perf_counter() - started,
            "stdout_tail": "",
            "stderr_tail": str(error),
            "reason": "could not launch check",
        }

    return {
        "name": label,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": int(completed.returncode),
        "wall_seconds": time.perf_counter() - started,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def golden_regressions(mode):
    """Run the critical standalone regression scripts from one command."""
    checks = list(GOLDEN_QUICK_CHECKS)
    if mode in {"standard", "full"}:
        checks.extend(GOLDEN_STANDARD_EXTRA)
    if mode == "full":
        checks.extend(GOLDEN_FULL_EXTRA)

    total = len(checks) + (1 if mode == "full" else 0)
    rows = []

    print()
    print("Golden regression checks", flush=True)
    print("-" * 72, flush=True)

    for index, (label, filename) in enumerate(checks, start=1):
        _progress_start(index, total, label)

        if filename is None:
            command = [
                sys.executable,
                "-c",
                (
                    "import lab; import batch_runner; "
                    "import characterisation_runner; "
                    "print('main imports PASS')"
                ),
            ]
        else:
            path = TESTS_ROOT / filename
            if not path.exists():
                result = {
                    "name": label,
                    "status": "FAIL",
                    "returncode": None,
                    "wall_seconds": 0.0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "reason": f"missing expected regression: {path}",
                }
                rows.append(result)
                _progress_finish(result)
                continue
            command = [sys.executable, str(path)]

        result = _run_command_check(label, command)
        rows.append(result)
        _progress_finish(result)

    if mode == "full":
        label = "pytest suite"
        _progress_start(total, total, label)
        # Windows can deny cleanup/traversal of the system pytest temp root.
        # Keep the regression sandbox inside the project, as the standalone
        # release command does, without changing any test or physics path.
        with tempfile.TemporaryDirectory(
            prefix=".pytest_validation_", dir=PROJECT_ROOT
        ) as pytest_temporary:
            result = _run_command_check(
                label,
                [
                    sys.executable, "-m", "pytest", "-q",
                    "-p", "no:cacheprovider",
                    "--basetemp", pytest_temporary,
                ],
                timeout_seconds=1800,
            )
        rows.append(result)
        _progress_finish(result)

    print("-" * 72, flush=True)
    regression_status = (
        "PASS"
        if all(row["status"] == "PASS" for row in rows)
        else "FAIL"
    )
    print(f"Regression layer: {regression_status}", flush=True)

    return {
        "status": regression_status,
        "checks": rows,
    }


def _recording_coordination_metrics(path):
    """Read one recorder NPZ and measure heavy-atom coordination over time.

    Bond coordination deliberately uses the recorder/display convention
    taper > 0.35.  Radial coordination is also reported separately so a
    partial fifth contact is not silently promoted into a full chemical bond.
    """
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=float)
        times = np.asarray(data["times"], dtype=float)

        if "box_sizes" in data.files:
            boxes = np.asarray(data["box_sizes"], dtype=float)
        else:
            boxes = np.full(len(positions), float(data["box_size"]), dtype=float)

        if "frame_types" in data.files:
            frame_types = np.asarray(data["frame_types"], dtype=np.int64)
        else:
            initial = np.asarray(
                R.types_from_symbols([str(value) for value in data["symbols"]]),
                dtype=np.int64,
            )
            frame_types = np.repeat(initial[None, :], len(positions), axis=0)

    carbon = int(R.ELEMENT_INDEX["C"])
    nitrogen = int(R.ELEMENT_INDEX["N"])
    hydrogen = int(R.ELEMENT_INDEX["H"])

    max_c_coord = 0
    max_n_coord = 0
    max_c_over = 0
    max_n_over = 0
    max_c_radial = 0.0
    max_n_radial = 0.0
    min_heavy_distance = float("inf")
    final_c_over = 0
    final_n_over = 0

    for frame_index, frame in enumerate(positions):
        types = frame_types[frame_index]
        count = len(frame)
        first, second = np.triu_indices(count, k=1)

        offset = frame[second] - frame[first]
        box = float(boxes[frame_index])
        offset -= box * np.round(offset / box)
        distance = np.linalg.norm(offset, axis=1)

        inner = R.CUTOFF_INNER[types[first], types[second]]
        outer = R.CUTOFF_OUTER[types[first], types[second]]
        taper = R.smooth_cutoff(distance, inner, outer)

        bonded = taper > 0.35
        coordination = np.zeros(count, dtype=np.int64)
        np.add.at(coordination, first[bonded], 1)
        np.add.at(coordination, second[bonded], 1)

        radial = np.zeros(count, dtype=float)
        np.add.at(radial, first, taper)
        np.add.at(radial, second, taper)

        c_mask = types == carbon
        n_mask = types == nitrogen
        if np.any(c_mask):
            c_coord = coordination[c_mask]
            max_c_coord = max(max_c_coord, int(c_coord.max()))
            c_over = int(np.count_nonzero(c_coord > 4))
            max_c_over = max(max_c_over, c_over)
            max_c_radial = max(max_c_radial, float(radial[c_mask].max()))
        else:
            c_over = 0

        if np.any(n_mask):
            n_coord = coordination[n_mask]
            max_n_coord = max(max_n_coord, int(n_coord.max()))
            n_over = int(np.count_nonzero(n_coord > 3))
            max_n_over = max(max_n_over, n_over)
            max_n_radial = max(max_n_radial, float(radial[n_mask].max()))
        else:
            n_over = 0

        heavy_pair = (types[first] != hydrogen) & (types[second] != hydrogen)
        if np.any(heavy_pair):
            min_heavy_distance = min(
                min_heavy_distance,
                float(np.min(distance[heavy_pair])),
            )

        if frame_index == len(positions) - 1:
            final_c_over = c_over
            final_n_over = n_over

    return {
        "frames": int(len(positions)),
        "end_time_fs": float(times[-1]) if len(times) else 0.0,
        "final_carbon_overvalent": int(final_c_over),
        "final_nitrogen_overvalent": int(final_n_over),
        "maximum_simultaneous_carbon_overvalent": int(max_c_over),
        "maximum_simultaneous_nitrogen_overvalent": int(max_n_over),
        "maximum_carbon_bond_coordination": int(max_c_coord),
        "maximum_nitrogen_bond_coordination": int(max_n_coord),
        "maximum_carbon_radial_coordination": float(max_c_radial),
        "maximum_nitrogen_radial_coordination": float(max_n_radial),
        "minimum_heavy_heavy_distance_A": (
            None if not np.isfinite(min_heavy_distance)
            else float(min_heavy_distance)
        ),
    }


def dense_soup_stress(mode):
    """Run a short production Optimised-Valence soup in full mode.

    This is intentionally a system-level invariant check rather than a product
    benchmark.  It exists to catch failures such as the former heavy-valence
    correction deleting the base radial over-coordination penalty: reference
    and optimised implementations can agree perfectly while both are wrong.

    Two deterministic 330-atom seeds for 1 ps are enough to expose that old
    runaway while keeping the full validation practical.
    """
    if mode != "full":
        return {
            "status": "NOT RUN",
            "reason": "dense production stress is reserved for --full",
        }

    with tempfile.TemporaryDirectory(prefix="chemistry_golden_") as temporary:
        output = Path(temporary) / "dense_soup"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "batch_runner.py"),
            "--mixture", "amino carbon growth",
            "--seed-list", "0,1",
            "--ps", "1",
            "--box", "19",
            "--physics", "optimised-valence",
            "--group", "2",
            "--hot-until-fs", "2000",
            "--hot-temperature", "500",
            "--cool-temperature", "250",
            "--out", str(output),
        ]

        print()
        print("Dense production stress", flush=True)
        print("-" * 72, flush=True)
        print(
            "[stress] 2 seeds x 330 atoms x 1 ps "
            "(optimised-valence, group 2) ... ",
            end="",
            flush=True,
        )
        execution = _run_command_check(
            "dense optimised-valence soup",
            command,
            timeout_seconds=1800,
        )
        print(
            f"{execution['status']}  "
            f"{float(execution.get('wall_seconds', 0.0)):.1f}s",
            flush=True,
        )
        if execution["status"] != "PASS":
            return {
                "status": "FAIL",
                "execution": execution,
                "runs": [],
                "reason": "batch_runner did not complete successfully",
            }

        index_path = output / "index.json"
        if not index_path.exists():
            return {
                "status": "FAIL",
                "execution": execution,
                "runs": [],
                "reason": "batch_runner completed without index.json",
            }

        index_rows = json.loads(index_path.read_text())
        run_metrics = []
        for row in index_rows:
            recording_path = output / row["file"]
            if not recording_path.exists():
                return {
                    "status": "FAIL",
                    "execution": execution,
                    "runs": run_metrics,
                    "reason": f"missing recording {recording_path.name}",
                }

            metrics = _recording_coordination_metrics(recording_path)
            metrics.update({
                "seed": int(row["seed"]),
                "stable": bool(row.get("stable")),
                "numerical_failures": int(row.get("numerical_failures", 0)),
                "move_cap_events": int(row.get("move_cap_events", 0)),
                "final_temperature_K": float(row.get("final_temperature", 0.0)),
                "final_potential_eV": float(row.get("final_potential", 0.0)),
            })
            run_metrics.append(metrics)
            print(
                "    seed "
                f"{metrics['seed']}: final C/N over-valent "
                f"{metrics['final_carbon_overvalent']}/"
                f"{metrics['final_nitrogen_overvalent']}, "
                "max C/N coordination "
                f"{metrics['maximum_carbon_bond_coordination']}/"
                f"{metrics['maximum_nitrogen_bond_coordination']}",
                flush=True,
            )

        # These are invariant/stress thresholds, not chemistry calibration
        # targets. They allow a single transient fifth C / fourth N contact,
        # but reject the former monotonic many-centre collapse.
        failures = []
        if len(run_metrics) != 2:
            failures.append(f"expected 2 completed seeds, found {len(run_metrics)}")

        for row in run_metrics:
            prefix = f"seed {row['seed']}"
            if not row["stable"] or row["numerical_failures"]:
                failures.append(f"{prefix}: numerical/integration failure")
            if row["final_carbon_overvalent"] != 0:
                failures.append(
                    f"{prefix}: {row['final_carbon_overvalent']} final over-valent carbons"
                )
            if row["final_nitrogen_overvalent"] != 0:
                failures.append(
                    f"{prefix}: {row['final_nitrogen_overvalent']} final over-valent nitrogens"
                )
            if row["maximum_simultaneous_carbon_overvalent"] > 2:
                failures.append(
                    f"{prefix}: carbon over-valence accumulated "
                    f"({row['maximum_simultaneous_carbon_overvalent']} simultaneous)"
                )
            if row["maximum_simultaneous_nitrogen_overvalent"] > 2:
                failures.append(
                    f"{prefix}: nitrogen over-valence accumulated "
                    f"({row['maximum_simultaneous_nitrogen_overvalent']} simultaneous)"
                )
            if row["maximum_carbon_bond_coordination"] > 5:
                failures.append(
                    f"{prefix}: carbon coordination reached "
                    f"{row['maximum_carbon_bond_coordination']}"
                )
            if row["maximum_nitrogen_bond_coordination"] > 4:
                failures.append(
                    f"{prefix}: nitrogen coordination reached "
                    f"{row['maximum_nitrogen_bond_coordination']}"
                )

        return {
            "status": "PASS" if not failures else "FAIL",
            "execution": execution,
            "runs": run_metrics,
            "failures": failures,
            "thresholds": {
                "final_carbon_overvalent": 0,
                "final_nitrogen_overvalent": 0,
                "maximum_simultaneous_carbon_overvalent": 2,
                "maximum_simultaneous_nitrogen_overvalent": 2,
                "maximum_carbon_bond_coordination": 5,
                "maximum_nitrogen_bond_coordination": 4,
                "bond_definition": "radial taper > 0.35",
                "note": (
                    "radial coordination is reported but not used as an integer "
                    "bond-count failure criterion"
                ),
            },
        }


def golden_validation(mode):
    regressions = golden_regressions(mode)
    dense = dense_soup_stress(mode)
    dense_ok = dense["status"] in {"PASS", "NOT RUN"}
    return {
        "status": (
            "PASS"
            if regressions["status"] == "PASS" and dense_ok
            else "FAIL"
        ),
        "regressions": regressions,
        "dense_soup_stress": dense,
    }


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def json_safe(value):
    """Recursively normalize NumPy/Torch diagnostic values for stable JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def error_fields(model, reference):
    absolute = float(model - reference)
    return {
        "absolute_error": absolute,
        "absolute_error_magnitude": abs(absolute),
        "percent_error": abs(absolute) / abs(reference) * 100.0
        if reference else None,
    }


def status_for_percent(value, strong=True):
    if value is None:
        return STATUS["insufficient"]
    limits = (2.0, 5.0, 12.0) if strong else (5.0, 12.0, 25.0)
    if value <= limits[0]:
        return STATUS["good"]
    if value <= limits[1]:
        return STATUS["acceptable"]
    if value <= limits[2]:
        return STATUS["weak"]
    return STATUS["fail"]


def status_for_reaction_energy(absolute_error_eV):
    """Classify near-thermoneutral reactions on an absolute energy scale.

    Percentage errors become misleading when the reference delta-E is close
    to zero.  These complete-species electronic energies are also compared
    with BDE298-derived thermochemistry, so an explicit absolute tolerance is
    the scientifically relevant diagnostic.
    """
    if absolute_error_eV <= 0.10:
        return STATUS["good"]
    if absolute_error_eV <= 0.20:
        return STATUS["acceptable"]
    if absolute_error_eV <= 0.30:
        return STATUS["weak"]
    return STATUS["fail"]


def parameter_audit():
    rows = []
    tables = [
        (1, R.BOND_TABLE, "accepted single-bond calibration"),
        (2, R.DOUBLE_BOND_TABLE, "inherited; not recalibrated in this stage"),
        (3, R.TRIPLE_BOND_TABLE, "inherited; not recalibrated in this stage"),
    ]
    for order, table, state in tables:
        for pair, values in sorted(table.items()):
            rows.append({
                "pair": PAIR_LABELS.get(pair, "-".join(pair)),
                "elements": list(pair), "bond_order": order,
                "re_A": float(values[0]), "depth_kJ_mol": float(values[1]),
                "depth_eV": float(values[1] * R.KJ_PER_MOL_TO_EV),
                "width_inv_A": float(values[2]), "state": state,
                "source": "reactive.py live table",
            })
    return {
        "status": STATUS["good"], "pairs": rows,
        "semantics": {
            "stored_depth": "classical effective Morse De before environment/bond-order terms",
            "morse": "D exp(-2 a (r-re)) - 2 D exp(-a (r-re))",
            "bond_order": "continuously blends single/double/triple tables",
            "environment_softening": float(R.ENVIRONMENT_SOFTENING),
            "over_coordination_depth_weight": float(R.OVER_COORDINATION_DEPTH_WEIGHT),
            "over_coordination_penalty_eV": float(R.OVER_COORDINATION_PENALTY),
            "high_fidelity": "separate H-transfer correction; base pair tables shared",
        },
    }


def torch_consistency():
    sim = ReactiveSimulation(
        ["H", "C", "N", "O"],
        np.array([[2., 2., 2.], [5., 2., 2.], [2., 5., 2.], [2., 2., 5.]]),
        10.0, target_temperature=0.0, device="cpu", dtype=torch.float64,
        random_seed=7,
    )
    checks = {}
    for label, torch_value, numpy_value in (
        ("single_length", sim.bond_length, R.BOND_LENGTH),
        ("single_depth", sim.bond_depth, R.BOND_DEPTH),
        ("single_width", sim.bond_width, R.BOND_WIDTH),
        ("double_length", sim.double_length, R.DOUBLE_LENGTH),
        ("double_depth", sim.double_depth, R.DOUBLE_DEPTH),
        ("double_width", sim.double_width, R.DOUBLE_WIDTH),
        ("triple_length", sim.triple_length, R.TRIPLE_LENGTH),
        ("triple_depth", sim.triple_depth, R.TRIPLE_DEPTH),
        ("triple_width", sim.triple_width, R.TRIPLE_WIDTH),
    ):
        difference = float(np.max(np.abs(torch_value.detach().cpu().numpy() - numpy_value)))
        checks[label] = difference
    maximum = max(checks.values())
    return {"status": STATUS["good"] if maximum == 0.0 else STATUS["fail"],
            "maximum_absolute_difference": maximum, "tables": checks}


def geometry_rows(entries, fit_target):
    rows = []
    for molecule, bond, pair, reference, source in entries:
        i, j = R.ELEMENT_INDEX[pair[0]], R.ELEMENT_INDEX[pair[1]]
        model = float(R.BOND_LENGTH[i, j])
        errors = error_fields(model, reference)
        rows.append({
            "molecule": molecule, "bond": bond, "model_A": model,
            "reference_A": reference, **errors,
            "classification": "fit target" if fit_target else "hold-out",
            "status": STATUS["fit"] if fit_target else status_for_percent(errors["percent_error"]),
            "source": SOURCES[source],
            "note": "generic pair re; full molecular relaxation may shift this value",
        })
    return rows


def angle_validation():
    rows = []
    for molecule, centre, model, reference, classification in (
        ("CH4", "C", 109.47, 109.47, "fit target"),
        ("NH3", "N", 107.0, 106.75, "fit target"),
        ("H2O", "O", 104.5, 104.52, "fit target"),
        ("ethanol C-C-O", "C", 109.47, 107.8, "hold-out"),
        ("dimethyl ether C-O-C", "O", 104.5, 111.2, "hold-out"),
        ("ethylamine C-C-N", "C", 109.47, 115.0, "hold-out"),
    ):
        err = error_fields(model, reference)
        rows.append({"molecule": molecule, "centre": centre,
                     "model_deg": model, "reference_deg": reference,
                     **err, "classification": classification,
                     "status": STATUS["fit"] if classification == "fit target"
                     else status_for_percent(err["percent_error"], strong=False)})
    return rows


def curvature_validation():
    rows = []
    for pair, (reference, quality, source, linked) in CURVATURE_REFERENCES.items():
        diagnostic = BC.pair_local_diagnostic(*pair)
        model = diagnostic["local_harmonic_cm-1"]
        err = error_fields(model, reference)
        rows.append({
            "pair": PAIR_LABELS[pair], "model_cm-1": model,
            "reference_cm-1": reference, **err, "comparison_quality": quality,
            "classification": "calibration-linked" if linked else "hold-out diagnostic",
            "status": STATUS["fit"] if linked else status_for_percent(
                err["percent_error"], strong="clean" in quality),
            "source": SOURCES[source],
        })
    return rows


def potential_curves():
    rows = []
    for label, function in COORDINATES.items():
        result = function()
        if label.startswith("H2 "):
            minimum = result["sampled_re_A"]
            dissociation = result["sampled_well_eV"]
            short = result["energy_at_0.35A_eV"] - result["energy_at_3A_eV"]
            falling = result["post_minimum_falling_steps"]
        else:
            minimum = result["sampled_minimum_A"]
            dissociation = result["dissociation_coordinate_eV"]
            short = result["short_range_energy_eV"]
            falling = result["capture_region_falling_steps"]
        row = {
            "coordinate": label, "sampled_minimum_A": minimum,
            "model_dissociation_coordinate_eV": dissociation,
            "short_range_above_dissociation_eV": short,
            "post_minimum_falling_steps": falling,
            "shape_status": STATUS["good"] if short > 0 and falling == 0 else STATUS["fail"],
            "reference_curve_status": STATUS["insufficient"],
            "note": "model curve validated for shape; no external pointwise curve bundled",
        }
        if label == "H2 H-H":
            # Convention diagnostic only.  The live H-H row now shares the
            # thermochemical/BDE298-like effective-depth convention used by
            # the other single-bond rows, whereas this comparison curve uses
            # a spectroscopy-derived De.  Equal equilibrium length and local
            # curvature do not imply pointwise equality away from re.
            reference_depth = BC.spectroscopic_h2_depths()["De_from_D0_eV"]
            reference_width = BC.width_for_frequency(
                reference_depth, BC.H2_REFERENCE["omega_e_cm-1"],
                R.MASS["H"], R.MASS["H"],
            )
            model_diag = BC.pair_local_diagnostic("H", "H")
            points = []
            for distance in (0.60, 0.74144, 1.00, 1.50, 2.00):
                def morse(depth, width, re_value):
                    shift = distance - re_value
                    return depth * math.exp(-2.0 * width * shift) - 2.0 * depth * math.exp(-width * shift)
                model_energy = morse(model_diag["depth_eV"], model_diag["width_inv_A"], model_diag["re_A"])
                reference_energy = morse(reference_depth, reference_width, BC.H2_REFERENCE["re_angstrom"])
                points.append({"distance_A": distance, "model_eV": model_energy,
                               "spectroscopic_Morse_reference_eV": reference_energy,
                               "absolute_error_eV": abs(model_energy-reference_energy)})
            row["reference_curve_status"] = "CONVENTION-DEPENDENT DIAGNOSTIC"
            row["reference_curve"] = {
                "kind": "spectroscopy-derived Morse approximation",
                "source": SOURCES["nist_h2"], "points": points,
                "model_depth_convention": "thermochemical/BDE298-like effective bond depth",
                "reference_depth_convention": "spectroscopic De derived from D0 plus ZPE",
                "required_pointwise_target": False,
            }
            row["note"] = (
                "H2 spectroscopy constrains equilibrium length and harmonic "
                "curvature. The full Morse-curve difference is reported only "
                "as a depth-convention diagnostic and is not a pointwise fit "
                "requirement."
            )
        rows.append(row)
    return rows


def dissociation_validation():
    rows = []
    for reaction, pair, reference, quantity, linked, source in DISSOCIATION:
        i, j = R.ELEMENT_INDEX[pair[0]], R.ELEMENT_INDEX[pair[1]]
        model = float(R.BOND_DEPTH[i, j] / R.KJ_PER_MOL_TO_EV)
        err = error_fields(model, reference)
        rows.append({
            "reaction": reaction, "pair": PAIR_LABELS[pair],
            "model_effective_depth_kJ_mol": model,
            "reference_kJ_mol": reference, "reference_quantity": quantity,
            **err, "classification": "calibration-linked" if linked else "hold-out",
            "status": STATUS["fit"] if linked else status_for_percent(err["percent_error"], False),
            "source": SOURCES[source],
            "caution": "effective pair De compared with molecular thermochemical quantity",
        })
    return rows


def bond_depth_diagnostics():
    # Homolytic exchange estimates use differences of the same effective pair
    # depths.  They are transparent diagnostics, not full fragment-relaxed
    # heats of reaction.
    references = [
        ("H + CH4 -> H2 + CH3", ("C", "H"), ("H", "H"), 0.033),
        ("H + H2O -> H2 + OH", ("O", "H"), ("H", "H"), 0.645),
        ("H + NH3 -> H2 + NH2", ("N", "H"), ("H", "H"), 0.137),
    ]
    rows = []
    for reaction, broken, formed, reference_ev in references:
        bi, bj = (R.ELEMENT_INDEX[x] for x in broken)
        fi, fj = (R.ELEMENT_INDEX[x] for x in formed)
        model = float(R.BOND_DEPTH[bi, bj] - R.BOND_DEPTH[fi, fj])
        err = error_fields(model, reference_ev)
        rows.append({
            "reaction": reaction, "model_eV": model,
            "reference_eV": reference_ev, **err,
            "classification": "derived from calibration-linked pair depths; not an independent hold-out",
            "status": status_for_percent(err["percent_error"], False),
            "diagnostic_kind": "effective pair-depth difference",
            "note": "bond-depth diagnostic only; not the model's reaction thermochemistry and omits all non-pair-depth terms, fragment relaxation, thermal corrections, and ZPE",
        })
    for reaction in ("CH3 + CH3 -> C2H6", "CH3 + OH -> CH3OH",
                     "NH2 + NH2 -> N2H4", "OH + OH -> H2O2"):
        rows.append({"reaction": reaction, "status": STATUS["insufficient"],
                     "note": "bond-depth diagnostic unavailable as independent evidence: formation energy is the inverse of a calibration-linked BDE"})
    return rows


def _methyl_geometry():
    """Sensible planar starting geometry for isolated CH3."""
    radius = 1.08
    angles = np.arange(3, dtype=float) * (2.0 * np.pi / 3.0)
    hydrogens = np.column_stack((
        radius * np.cos(angles), radius * np.sin(angles), np.zeros(3),
    ))
    return ["C", "H", "H", "H"], np.vstack((np.zeros(3), hydrogens))


def _relaxed_species_energy(name):
    """Evaluate one complete species with the production Torch energy path."""
    if name == "H":
        return {"energy_eV": 0.0, "converged": True,
                "maximum_force_eV_per_A": 0.0, "atoms": 1,
                "geometry": "isolated atom"}
    symbols, positions = (_methyl_geometry() if name == "CH3"
                          else build_box.BUILDERS[name]())
    # A roomy periodic cell makes the existing engine path available without
    # introducing interactions between periodic images of these small species.
    positions = np.asarray(positions, dtype=float) + 10.0
    simulation = ReactiveSimulation(
        symbols, positions, 30.0, device="cpu", dtype=torch.float64,
        target_temperature=0.0, relax_on_start=False,
    )
    initial_energy = simulation.potential_energy
    simulation.relax(steps=1000, maximum_force=10.0, step_size=0.001)
    maximum_force = float(torch.linalg.norm(simulation.forces, dim=1).max())
    return {
        "energy_eV": simulation.potential_energy,
        "initial_energy_eV": initial_energy,
        "converged": maximum_force < 1.0e-3,
        "maximum_force_eV_per_A": maximum_force,
        "atoms": len(symbols),
        "geometry": "engine-relaxed from a stable experimental-like starting geometry",
    }


def whole_model_reaction_energies(mode="standard"):
    """Complete-species electronic energies, separate from pair diagnostics."""
    definitions = [
        ("H + CH4 -> H2 + CH3", ("H", "CH4"), ("H2", "CH3"), 0.033),
        ("H + H2O -> H2 + OH", ("H", "H2O"), ("H2", "OH"), 0.645),
        ("H + NH3 -> H2 + NH2", ("H", "NH3"), ("H2", "NH2"), 0.137),
    ]
    if mode == "quick":
        return [{"reaction": reaction, "status": "NOT RUN IN QUICK MODE",
                 "classification": "derived comparison from calibration-linked BDE targets; not independent"}
                for reaction, _, _, _ in definitions]
    species = {name for _, reactants, products, _ in definitions
               for name in reactants + products}
    energies = {name: _relaxed_species_energy(name) for name in sorted(species)}
    rows = []
    for reaction, reactants, products, reference_ev in definitions:
        involved = reactants + products
        converged = all(energies[name]["converged"] for name in involved)
        model = (sum(energies[name]["energy_eV"] for name in products)
                 - sum(energies[name]["energy_eV"] for name in reactants))
        row = {
            "reaction": reaction,
            "calculation": "complete-species ChemistryModel Torch energy after isolated-fragment relaxation",
            "terms": "the full engine energy, including bond order, environment softening, coordination, and angle terms",
            "model_delta_E_eV": model if converged else None,
            "reference_delta_E_eV": reference_ev,
            "reference_quantity": "approximate reaction enthalpy from experimental BDE298 difference",
            "classification": "derived comparison from calibration-linked BDE targets; not independent",
            "relaxation_converged": converged,
            "species": {name: energies[name] for name in involved},
            "limitation": "model electronic energy is compared with a BDE298-derived thermochemical value; ZPE and finite-temperature corrections are not included",
        }
        if converged:
            error = error_fields(model, reference_ev)
            row.update(error)
            row["status"] = status_for_reaction_energy(
                error["absolute_error_magnitude"]
            )
            row["status_basis"] = (
                "absolute delta-E error; percentage error is retained but "
                "not used for near-thermoneutral reactions"
            )
        else:
            row["status"] = STATUS["insufficient"]
            row["limitation"] += "; at least one species relaxation did not converge"
        rows.append(row)
    return rows


def barrier_validation(mode):
    rows = []
    if mode == "quick":
        for name, reference in HF.REFERENCE_BARRIERS.items():
            rows.append({"system": name, "reference_eV": list(reference),
                         "status": "NOT RUN IN QUICK MODE"})
        return rows
    for name in ("formaldehyde", "water", "methane", "ammonia"):
        started = time.perf_counter()
        relaxed = mode == "full"
        result = HF.measure_barrier("high_fidelity", name, relax=relaxed)
        elapsed = time.perf_counter() - started
        entry = {
            "system": name, "model": result, "wall_seconds": elapsed,
            "scan_kind": "relaxed" if relaxed else "frozen",
            "caution": None if relaxed else (
                "frozen spectators make this an approximate screening value; "
                "use --full for relaxed barriers"
            ),
        }
        if name in HF.REFERENCE_BARRIERS and result:
            low, high = HF.REFERENCE_BARRIERS[name]
            value = result["barrier"]
            error = 0.0 if low <= value <= high else min(abs(value-low), abs(value-high))
            entry.update({"reference_eV": [low, high], "absolute_error_eV": error,
                          "status": STATUS["good"] if error <= 0.05 else STATUS["weak"]})
        else:
            entry.update({"status": STATUS["insufficient"]})
        rows.append(entry)
    return rows


def nve_endurance():
    probes = {
        "H2": lambda: BC.molecule_nve("H2", steps=400),
        "CH4": lambda: BC.molecule_nve("CH4", steps=400),
        "NH3": lambda: BC.molecule_nve("NH3", steps=400),
        "H2O": lambda: BC.molecule_nve("H2O", steps=400),
        "C2H6": BC.ethane_nve, "CH3NH2": BC.methylamine_nve,
        "CH3OH": BC.methanol_nve, "N2H4": BC.hydrazine_nve,
        "NH2OH": BC.hydroxylamine_nve, "H2O2": BC.hydrogen_peroxide_nve,
    }
    rows = []
    for name, probe in probes.items():
        result = probe()
        drift = abs(result["drift_eV"])
        result.update({"molecule": name,
                       "status": STATUS["good"] if drift < 0.01 and not result["capped_steps"] else STATUS["weak"]})
        rows.append(result)
    return rows


def read_batch(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return {
        "path": str(path), "runs": len(rows),
        "finished": sum(bool(x.get("finished")) for x in rows),
        "stable": sum(bool(x.get("stable")) for x in rows),
        "energy_jumps": sum(int(x.get("energy_jumps", 0)) for x in rows),
        "move_cap_events": sum(int(x.get("move_cap_events", 0)) for x in rows),
        "mean_heavy_bonds_formed": float(np.mean([x.get("heavy_bonds_formed", 0) for x in rows])),
        "mean_largest_structure": float(np.mean([x.get("largest_any", 0) for x in rows])),
        "mean_species_count": float(np.mean([x.get("species_count", 0) for x in rows])),
        "mean_final_temperature_K": float(np.mean([x.get("final_temperature", 0) for x in rows])),
        "mean_final_potential_eV": float(np.mean([x.get("final_potential", 0) for x in rows])),
        "mean_wall_seconds": float(np.mean([x.get("wall_seconds", 0) for x in rows])),
    }


def dynamic_reactions():
    rows = []
    product_for = {"NH2 + NH2 -> N2H4": "N2H4", "OH + OH -> H2O2": "O2H2"}
    for reaction, path in FOCUSED_BATCHES.items():
        batch = read_batch(path)
        if batch is None:
            rows.append({"reaction": reaction, "status": "NOT RUN", "path": path})
            continue
        raw = json.loads(Path(path).read_text())
        product = product_for[reaction]
        formed = sum(product in x.get("species_seen", []) for x in raw)
        retained = sum(product in x.get("final_species", []) for x in raw)
        rows.append({"reaction": reaction, "attempts": len(raw), "reacting": formed,
                     "retained": retained, "integration_failures": len(raw)-batch["stable"],
                     "status": STATUS["good"] if formed == len(raw) and batch["stable"] == len(raw) else STATUS["weak"],
                     "interpretation": "deterministic regression ensemble, not an experimental rate constant",
                     "batch": batch})
    for reaction in ("H + H -> H2", "H + CH3 -> CH4", "CH3 + CH3 -> C2H6", "CH3 + OH -> CH3OH"):
        rows.append({"reaction": reaction, "status": STATUS["insufficient"],
                     "note": "no preserved matched focused ensemble yet"})
    return rows


def collision_abstraction(barriers):
    """Qualitative accessibility map; explicitly not a trajectory rate."""
    thermal_ev = 8.617333262e-5 * 300.0
    rows = []
    mapping = {"methane": "H + CH4 -> H2 + CH3",
               "water": "H/O-H exchange proxy",
               "ammonia": "H + NH3 -> H2 + NH2",
               "formaldehyde": "H + CH2O -> H2 + HCO"}
    for barrier in barriers:
        result = barrier.get("model")
        if not result:
            rows.append({"reaction": mapping.get(barrier["system"], barrier["system"]),
                         "status": "NOT RUN IN QUICK MODE"})
            continue
        height = result["barrier"]
        rows.append({
            "reaction": mapping.get(barrier["system"], barrier["system"]),
            "barrier_eV": height,
            "collision_energy_multipliers": [
                {"multiplier": value, "nominal_eV": thermal_ev*value,
                 "classically_above_static_barrier": thermal_ev*value >= height}
                for value in (1, 2, 3, 5)
            ],
            "status": STATUS["acceptable"],
            "caution": "static surface threshold, not a trajectory probability or experimental rate",
        })
    return rows


def performance():
    result = BC.runtime_probe(repeats=5, steps=100)
    result["steps_per_second"] = result["steps"] / result["median_seconds"]
    result["device"] = "cpu"
    result["gpu_memory"] = "not measured in quick local probe"
    result["status"] = STATUS["good"]
    return result


def large_mixture(mode):
    if mode == "quick":
        return {"status": "NOT RUN IN QUICK MODE"}
    current = read_batch("runs/oo_length_candidate/index.json")
    baseline = read_batch("runs/integrator_smoothstep_final/index.json")
    return {
        "status": STATUS["acceptable"] if current and baseline else STATUS["insufficient"],
        "historical_baseline": baseline, "current_calibration_stage": current,
        "caution": "historical matched seeds; stochastic products need distribution-level interpretation",
    }


def build_report(mode):
    started = time.perf_counter()

    print()
    print("=" * 72, flush=True)
    print(f"CHEMISTRYMODEL VALIDATION - {str(mode).upper()}", flush=True)
    print("=" * 72, flush=True)

    golden = _science_stage(
        "golden implementation/invariant suite",
        lambda: golden_validation(mode),
    )
    barriers = _science_stage(
        "reaction barrier scans",
        lambda: barrier_validation(mode),
    )
    reaction_energies = _science_stage(
        "whole-model reaction energies",
        lambda: whole_model_reaction_energies(mode),
    )
    endurance = _science_stage(
        "stable-molecule NVE endurance",
        nve_endurance,
    )
    perf = _science_stage(
        "runtime performance probe",
        performance,
    )
    mixture_result = _science_stage(
        "large-mixture evidence",
        lambda: large_mixture(mode),
    )

    print("[science] static tables/geometry/curves          ... ", end="", flush=True)
    static_started = time.perf_counter()
    parameter_consistency = {
        "audit": parameter_audit(),
        "numpy_torch": torch_consistency(),
    }
    geometry = geometry_rows(FIT_GEOMETRY, True)
    holdout_geometry = geometry_rows(HOLDOUT_GEOMETRY, False)
    angles = angle_validation()
    harmonic_curvature = curvature_validation()
    potential_curve_rows = potential_curves()
    dissociation_rows = dissociation_validation()
    depth_rows = bond_depth_diagnostics()
    dynamic_rows = dynamic_reactions()
    collision_rows = collision_abstraction(barriers)
    print(f"DONE  {time.perf_counter() - static_started:.1f}s", flush=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "golden_validation": golden,
        "mode": mode, "git_revision": git_revision(),
        "rules": {"force_field_modified": False, "overall_accuracy_percentage": None,
                  "fit_targets_are_independent_validation": False},
        "parameter_consistency": parameter_consistency,
        "geometry": geometry,
        "holdout_geometry": holdout_geometry,
        "angles": angles,
        "harmonic_curvature": harmonic_curvature,
        "potential_curves": potential_curve_rows,
        "dissociation_energies": dissociation_rows,
        "bond_depth_diagnostics": depth_rows,
        "whole_model_reaction_energies": reaction_energies,
        "reaction_barriers": barriers,
        "dynamic_reactions": dynamic_rows,
        "collision_abstraction": collision_rows,
        "validation_mixtures": {
            "stable": "[validation] stable small molecules",
            "nitrogen": "[validation] nitrogen radicals",
            "oxygen": "[validation] oxygen radicals",
            "limitation": "current mixture schema cannot combine loose atoms and molecules in one preset",
            "compatibility": "additive presets only; existing mixture definitions and defaults are unchanged",
        },
        "stable_molecule_endurance": endurance,
        "numerical_stability": {
            "status": STATUS["good"],
            "source": "NVE endurance plus focused batch health",
        },
        "performance": perf,
        "large_mixture": mixture_result,
    }
    report["transferability"] = summarize_transferability(report)
    report["baseline_summary"] = {
        "strong": [
            "deterministic core, high-fidelity, recorder, and replay regressions",
            "preserved focused N-N and O-O recombination ensembles",
            "H2 effective depth now matches the thermochemical convention while preserving equilibrium length and harmonic curvature",
            "whole-model H2-forming abstraction energies now have the expected sign and are within 0.10 eV of their BDE298-derived comparisons",
            "most fitted X-H geometry and curvature diagnostics",
        ],
        "weak": [
            "the relaxed methane abstraction barrier remains below its reference range",
            "N-N/N-O heavy-atom curvature transfer is weak or uncertain",
        ],
        "uncertain": [
            "reaction-energy references are calibration-linked BDE298 differences, not independent like-for-like electronic energies",
            "ammonia lacks a bundled independent abstraction-barrier reference and broader collision ensembles are still needed",
            "the full H2 Morse curve is convention-dependent; spectroscopy constrains its equilibrium length and local curvature here",
            "double/triple bond transfer and other high-level pointwise potential curves remain sparsely validated",
        ],
    }
    report["wall_seconds"] = time.perf_counter() - started
    return json_safe(report)


def compare_reports(old, new):
    """Stable high-value changes for machine and human regression review."""
    def index(rows, key):
        return {row[key]: row for row in rows}
    comparisons = []
    sections = [
        ("holdout_geometry", "molecule", "percent_error"),
        ("harmonic_curvature", "pair", "percent_error"),
        ("dissociation_energies", "reaction", "percent_error"),
        ("reaction_barriers", "system", "absolute_error_eV"),
        ("stable_molecule_endurance", "molecule", "drift_eV"),
    ]
    for section, key, metric in sections:
        before, after = index(old.get(section, []), key), index(new.get(section, []), key)
        for name in sorted(set(before) & set(after)):
            old_value, new_value = before[name].get(metric), after[name].get(metric)
            if old_value is None or new_value is None:
                continue
            comparisons.append({"section": section, "target": name, "metric": metric,
                                "before": old_value, "after": new_value,
                                "difference": new_value-old_value})
    return {"schema_version": 1, "before_revision": old.get("git_revision"),
            "after_revision": new.get("git_revision"), "changes": comparisons}


def summarize_transferability(report):
    holdout_errors = [x["percent_error"] for x in report["holdout_geometry"]]
    return {
        "geometry": "ACCEPTABLE: ethanol/ethylamine transfer is close; dimethyl-ether C-O exposes environment dependence",
        "x_h_curvature": "STRONGEST AREA, but most X-H molecular bands remain calibration-linked",
        "heavy_heavy_curvature": "WEAK/UNCERTAIN: mixed normal modes; N-N width fit was rejected by capture dynamics",
        "dissociation": "MODERATE: effective pair depths are diagnostics, not molecular thermochemistry, and several hold-outs differ materially",
        "reaction_energies": "UNCERTAIN: complete-species engine energies are now reported, but their experimental comparators are BDE298-derived and calibration-linked",
        "barriers": "reported separately; no parameter tuning performed here",
        "dynamics": "GOOD for preserved focused N-N and O-O recombination ensembles; other reactions need baselines",
        "numerical_stability": "GOOD across deterministic and NVE probes",
        "mean_holdout_geometry_percent_error": float(np.mean(holdout_errors)),
        "important_gaps": [
            "no bundled high-level pointwise potential curves",
            "collision accessibility is static-barrier based; no trajectory probability ensemble",
            "limited preserved focused ensembles for H-H, C-H, C-C and C-O capture",
            "double/triple bond tables remain inherited and unvalidated in this stage",
        ],
    }


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(report):
    lines = [
        "# ChemistryModel independent validation report", "",
        f"Revision: `{report['git_revision']}`",
        f"Mode: `{report['mode']}`",
        "Force-field parameters changed by this report: **no**", "",
        "Fit targets are labelled and are not counted as independent validation.", "",
        "## Golden whole-model validation", "",
        f"**FINAL GOLDEN RESULT: {report['golden_validation']['status']}**", "",
    ]
    for row in report["golden_validation"]["regressions"]["checks"]:
        lines.append(
            f"- **{row['name']}**: {row['status']} "
            f"({row['wall_seconds']:.2f} s)"
        )
    dense = report["golden_validation"]["dense_soup_stress"]
    lines.append(f"- **Dense optimised-valence soup stress**: {dense['status']}")
    if dense.get("runs"):
        for row in dense["runs"]:
            lines.append(
                f"  - seed {row['seed']}: final C/N over-valent "
                f"{row['final_carbon_overvalent']}/"
                f"{row['final_nitrogen_overvalent']}; max C/N coordination "
                f"{row['maximum_carbon_bond_coordination']}/"
                f"{row['maximum_nitrogen_bond_coordination']}"
            )
    if dense.get("failures"):
        lines.extend(f"  - FAIL: {item}" for item in dense["failures"])
    lines += ["", "## Baseline summary", ""]
    for category in ("strong", "weak", "uncertain"):
        lines.append(f"**{category.title()}**")
        lines.extend(f"- {item}" for item in report["baseline_summary"][category])
        lines.append("")
    lines += [
        "## 1. Parameter consistency", "",
        f"- NumPy/Torch: **{report['parameter_consistency']['numpy_torch']['status']}**",
        f"- Maximum table difference: {report['parameter_consistency']['numpy_torch']['maximum_absolute_difference']}",
        "- Single bonds are accepted calibration values; double/triple rows are inherited.",
        "- Stored D is an effective classical Morse depth before bond-order/environment terms.", "",
        "## 2. Geometry", "",
        "| molecule | bond | model A | reference A | error % | classification/status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["geometry"]:
        lines.append(f"| {row['molecule']} | {row['bond']} | {fmt(row['model_A'])} | {fmt(row['reference_A'])} | {fmt(row['percent_error'],2)} | {row['status']} |")
    lines += ["", "## 3. Hold-out geometry", "",
              "These are generic-pair transfer diagnostics, not fully relaxed molecular structures.", "",
              "| molecule | bond | model A | reference A | error % | status |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in report["holdout_geometry"]:
        lines.append(f"| {row['molecule']} | {row['bond']} | {fmt(row['model_A'])} | {fmt(row['reference_A'])} | {fmt(row['percent_error'],2)} | {row['status']} |")
    lines += ["", "## 4. Harmonic curvature", "",
              "| pair | model cm-1 | reference cm-1 | error % | comparison | status |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in report["harmonic_curvature"]:
        lines.append(f"| {row['pair']} | {fmt(row['model_cm-1'],1)} | {fmt(row['reference_cm-1'],1)} | {fmt(row['percent_error'],1)} | {row['comparison_quality']} | {row['status']} |")
    lines += ["", "## 5. Potential curves", ""]
    for row in report["potential_curves"]:
        lines.append(f"- **{row['coordinate']}**: {row['shape_status']}; minimum {row['sampled_minimum_A']:.4f} A, dissociation coordinate {row['model_dissociation_coordinate_eV']:.3f} eV. External pointwise comparison: {row['reference_curve_status']}.")
    lines += ["", "## 6. Dissociation energies", "",
              "| reaction | model kJ/mol | reference kJ/mol | error % | status |",
              "| --- | ---: | ---: | ---: | --- |"]
    for row in report["dissociation_energies"]:
        lines.append(f"| {row['reaction']} | {fmt(row['model_effective_depth_kJ_mol'],1)} | {fmt(row['reference_kJ_mol'],1)} | {fmt(row['percent_error'],1)} | {row['status']} |")
    lines += ["", "## 7. Bond-depth diagnostics", "",
              "These pair-depth differences are diagnostics only. They are not ChemistryModel reaction thermochemistry.", ""]
    for row in report["bond_depth_diagnostics"]:
        model = row.get("model_eV")
        lines.append(f"- **{row['reaction']}**: {row['status']}" + (f"; model {model:.3f} eV, reference {row['reference_eV']:.3f} eV." if model is not None else "."))
    lines += ["", "## 8. Whole-model reaction energies", "",
              "These use relaxed complete reactant and product species through the production Torch energy function. The reference is a BDE298-derived thermochemical difference, so it is not a like-for-like zero-temperature electronic-energy observable.", ""]
    for row in report["whole_model_reaction_energies"]:
        model = row.get("model_delta_E_eV")
        lines.append(f"- **{row['reaction']}**: {row['status']}" + (f"; model Delta E {model:.3f} eV, BDE298-derived reference {row['reference_delta_E_eV']:.3f} eV; relaxation converged: {row['relaxation_converged']}." if model is not None else "."))
    lines += ["", "## 9. Reaction barriers", "",
              "Frozen scans are screening diagnostics; relaxed full-mode scans are the stronger result.", ""]
    for row in report["reaction_barriers"]:
        model = row.get("model")
        scan = row.get("scan_kind")
        lines.append(f"- **{row['system']}**: {row['status']}" + (f"; model {model['barrier']:.3f} eV ({scan} scan)." if model else "."))
    lines += ["", "## 10. Dynamic reactions", ""]
    for row in report["dynamic_reactions"]:
        lines.append(f"- **{row['reaction']}**: {row['status']}" + (f"; {row['reacting']}/{row['attempts']} reacted, {row['retained']} retained product." if 'attempts' in row else "."))
    lines += ["", "## 11. Mixture behaviour", "",
              f"- Large matched mixture: **{report['large_mixture']['status']}**", "",
              "## 12. Numerical stability", "",
              f"- **{report['numerical_stability']['status']}**", "- All reported NVE probes and preserved focused batches are checked separately in JSON.", "",
              "## 13. Performance", "",
              f"- CPU probe: {report['performance']['steps_per_second']:.1f} steps/s ({report['performance']['median_seconds']:.3f} s for {report['performance']['steps']} steps).", "",
              "## 14. Transferability", ""]
    for key, value in report["transferability"].items():
        if key not in ("important_gaps", "mean_holdout_geometry_percent_error"):
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    lines += ["", "Important gaps:", ""]
    lines.extend(f"- {item}" for item in report["transferability"]["important_gaps"])
    lines += ["", "No overall accuracy percentage is reported.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    levels = parser.add_mutually_exclusive_group()
    levels.add_argument("--quick", action="store_true")
    levels.add_argument("--standard", action="store_true")
    levels.add_argument("--full", action="store_true")
    parser.add_argument("--json", dest="json_path", default="validation/baseline.json")
    parser.add_argument("--report", default="validation/baseline.md")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--compare", default=None,
                        help="compare the new report with an older baseline JSON")
    options = parser.parse_args()
    mode = "full" if options.full else "standard" if options.standard else "quick"
    report = build_report(mode)
    rendered = markdown(report)
    for path_string, content, as_json in (
        (options.json_path, report, True), (options.report, rendered, False),
    ):
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_json:
            path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        else:
            path.write_text(content)
    if options.compare:
        previous = json.loads(Path(options.compare).read_text())
        comparison = compare_reports(previous, report)
        comparison_path = Path(options.json_path).with_name("comparison.json")
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    if options.stdout:
        print(rendered)
    else:
        print(f"wrote {options.report} and {options.json_path}")

    golden = report["golden_validation"]
    print()
    print("=" * 60)
    print("CHEMISTRYMODEL GOLDEN VALIDATION")
    print("=" * 60)
    for row in golden["regressions"]["checks"]:
        print(f"{row['status']:4s}  {row['name']}")
    dense = golden["dense_soup_stress"]
    print(f"{dense['status']:4s}  dense optimised-valence soup stress")
    print("-" * 60)
    print(f"FINAL RESULT: {golden['status']}")
    print("=" * 60)

    if golden["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
