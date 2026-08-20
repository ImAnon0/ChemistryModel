import argparse
import inspect
import json
import math
import os
import sys
import time

import numpy as np

import molecule_library as molecule_store
import running


# ============================================================
# Controlled molecule characterisation
# ============================================================
#
# A characterisation run starts from a molecule that was actually
# discovered by the reactive simulation, reconstructs that exact atomistic
# geometry in a fresh periodic box, and then hands it back to the same
# reactive Torch engine used by normal batches.
#
# Independent repeats share tensor kernels for throughput. Sixteen is a
# conservative default on the reference 4060 Ti; the final group may be
# smaller, and the command line can tune the size for another device.
DEFAULT_GROUP_SIZE = 16
ENTRIES = "entries"
DEFAULT_BOND_THRESHOLD = 0.35
DEFAULT_FORMATION_TIME_FS = 100.0
DEFAULT_DIAGNOSTIC_SAMPLE_FS = 1.0
DEFAULT_DIAGNOSTIC_FINE_WINDOW_FS = 1000.0
DEFAULT_DIAGNOSTIC_COARSE_SAMPLE_FS = 10.0
TARGET_AXIS_CANDIDATES = 96
TARGET_AXIS_FORWARD_EPS_A = 0.15
SAFE_GAP_STEP_A = 0.05
SAFE_GAP_BUFFER_A = 0.20
FIRST_ENCOUNTER_RELEASE_A = 0.05

# Microreactor early completion is tied to the real BondTracker confirmation
# time instead of a reaction-specific hardcoded lifetime. With the current
# 100 fs formation window these become 500 fs minimum runtime and 300 fs of
# continuous quiescence.
MICROREACTOR_QUIESCENCE_MIN_FORMATION_WINDOWS = 5.0
MICROREACTOR_QUIESCENCE_QUIET_FORMATION_WINDOWS = 3.0
MICROREACTOR_QUIESCENCE_VERSION = 1

from high_fidelity_torch import (
    HF_MODEL_NAME, HF_MODEL_REVISION,
    H_TRANSFER_STATE_MIXING_FRACTION,
    H_TRANSFER_GATE_START, H_TRANSFER_GATE_FULL,
    H_TRANSFER_SATO,
    H_TRANSFER_COUPLING_FLATTEN,
    H_TRANSFER_LOWERING_CAP,
    H_TRANSFER_LOWERING_CAP_SOFTNESS,
)

STANDARD_PHYSICS_MODEL = "reactive_v1"
HIGH_FIDELITY_PHYSICS_MODEL = HF_MODEL_NAME
HIGH_FIDELITY_H_TRANSFER_MIXING = H_TRANSFER_STATE_MIXING_FRACTION
HIGH_FIDELITY_H_TRANSFER_GATE_START = H_TRANSFER_GATE_START
HIGH_FIDELITY_H_TRANSFER_GATE_FULL = H_TRANSFER_GATE_FULL


def high_fidelity_parameters():
    """Everything that defines the correction, read from the physics module.

    Duplicating these as literals is what let a run report V3 while actually
    integrating V4. Reading them here means the recorded parameters cannot
    drift from the physics that produced the run.
    """
    return {
        "h_transfer_state_mixing_fraction": H_TRANSFER_STATE_MIXING_FRACTION,
        "h_transfer_gate_start": H_TRANSFER_GATE_START,
        "h_transfer_gate_full": H_TRANSFER_GATE_FULL,
        "h_transfer_sato": H_TRANSFER_SATO,
        "h_transfer_coupling_flatten": H_TRANSFER_COUPLING_FLATTEN,
        "h_transfer_lowering_cap": H_TRANSFER_LOWERING_CAP,
        "h_transfer_lowering_cap_softness": (
            H_TRANSFER_LOWERING_CAP_SOFTNESS
            if H_TRANSFER_LOWERING_CAP is not None else None
        ),
        "h_transfer_model_revision": HF_MODEL_REVISION,
    }


def simulation_class_for_physics(physics):
    """Resolve a characterisation physics identifier to its existing engine."""
    mode = str(physics)
    if mode == "standard":
        from batched_torch import BatchedReactiveSimulation
        return BatchedReactiveSimulation
    if mode == "high_fidelity":
        from high_fidelity_torch import HighFidelityBatchedReactiveSimulation
        return HighFidelityBatchedReactiveSimulation
    if mode == "optimised-valence":
        from valence_state_optimised_torch import (
            OptimisedValenceStateBatchedSimulation,
        )
        return OptimisedValenceStateBatchedSimulation
    raise ValueError(f"unknown characterisation physics: {physics!r}")


def physics_metadata(physics, simulation=None):
    """Return provenance from the class that will (or did) run the physics."""
    mode = str(physics)
    source = simulation if simulation is not None else simulation_class_for_physics(mode)
    fallback_model = {
        "standard": STANDARD_PHYSICS_MODEL,
        "high_fidelity": HIGH_FIDELITY_PHYSICS_MODEL,
    }.get(mode, mode)
    return {
        "physics_mode": mode,
        "physics_model": getattr(source, "physics_model_name", fallback_model),
        "physics_model_revision": getattr(source, "physics_model_revision", None),
        "physics_parameters": (
            high_fidelity_parameters() if mode == "high_fidelity" else {}
        ),
    }


def heartbeat_path(folder):
    return os.path.join(folder, f".progress_{os.getpid()}.json")


def write_heartbeat(folder, seed_label, done, total, started, boxes_in_group,
                    phase="simulation", results_done=None, results_total=None):
    payload = {
        "pid": os.getpid(),
        "seed": seed_label,
        "steps_done": int(done),
        "steps_total": int(total),
        "run_started": started,
        "updated": time.time(),
        "boxes_in_group": int(boxes_in_group),
        "phase": str(phase),
    }
    if results_done is not None:
        payload["results_done"] = int(results_done)
    if results_total is not None:
        payload["results_total"] = int(results_total)

    try:
        temporary = heartbeat_path(folder) + ".part"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temporary, heartbeat_path(folder))
    except OSError:
        pass


def clear_heartbeat(folder):
    try:
        os.remove(heartbeat_path(folder))
    except OSError:
        pass


def entry_path(folder, seed):
    return os.path.join(folder, ENTRIES, f"seed_{int(seed):06d}.json")


def write_entry(folder, entry):
    directory = os.path.join(folder, ENTRIES)
    os.makedirs(directory, exist_ok=True)

    path = entry_path(folder, entry.get("seed", 0))
    temporary = path + ".part"

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(entry, handle, indent=1)

    os.replace(temporary, path)


def rebuild_index(folder):
    directory = os.path.join(folder, ENTRIES)
    entries = []

    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue

            try:
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    entries.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue

    entries.sort(key=lambda item: item.get("seed", 0))

    for number, entry in enumerate(entries):
        entry["number"] = number

    path = os.path.join(folder, "index.json")
    temporary = path + ".part"

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=1)

    os.replace(temporary, path)
    return entries


def finished_seeds(folder):
    found = set()
    directory = os.path.join(folder, ENTRIES)

    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if not (name.startswith("seed_") and name.endswith(".json")):
                continue

            try:
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    entry = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if entry.get("finished", True) is not False and entry.get("seed") is not None:
                found.add(int(entry["seed"]))

    return found




def bonding_settings():
    """Read the actual BondTracker defaults used by this checkout."""

    try:
        import bonding

        formation_time = float(getattr(
            bonding, "FORMATION_TIME", DEFAULT_FORMATION_TIME_FS
        ))
        parameter = inspect.signature(
            bonding.BondTracker.__init__
        ).parameters.get("threshold")
        threshold = (
            float(parameter.default)
            if parameter is not None and parameter.default is not inspect._empty
            else DEFAULT_BOND_THRESHOLD
        )
        return threshold, formation_time
    except Exception:
        return DEFAULT_BOND_THRESHOLD, DEFAULT_FORMATION_TIME_FS


def _types_from_symbols(symbols):
    import reactive as R

    if hasattr(R, "types_from_symbols"):
        return np.asarray(R.types_from_symbols(symbols), dtype=np.int64)

    elements = [str(item) for item in np.asarray(R.ELEMENTS).tolist()]
    mapping = {symbol: index for index, symbol in enumerate(elements)}
    return np.asarray([mapping[str(symbol)] for symbol in symbols], dtype=np.int64)


def _cross_pair_arrays(positions, symbols, first_count, box_size):
    """Distances and bond taper for only molecule<->partner atom pairs."""

    import reactive as R

    positions = np.asarray(positions, dtype=np.float64)
    first_count = int(first_count)
    second_count = len(positions) - first_count

    if first_count <= 0 or second_count <= 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    first_positions = positions[:first_count]
    second_positions = positions[first_count:]
    offsets = second_positions[:, None, :] - first_positions[None, :, :]
    offsets -= float(box_size) * np.round(offsets / float(box_size))
    distances = np.linalg.norm(offsets, axis=2)

    types = _types_from_symbols(symbols)
    first_types = types[:first_count]
    second_types = types[first_count:]
    inner = R.CUTOFF_INNER[np.ix_(second_types, first_types)]
    outer = R.CUTOFF_OUTER[np.ix_(second_types, first_types)]
    taper = np.asarray(R.smooth_cutoff(distances, inner, outer), dtype=np.float64)

    return distances, taper


def _place_collision(first_positions, second_positions, centre, axis,
                     centre_distance, impact_offset=None):
    first_positions = np.asarray(first_positions, dtype=np.float64)
    second_positions = np.asarray(second_positions, dtype=np.float64)
    impact_offset = np.zeros(3, dtype=np.float64) if impact_offset is None else np.asarray(
        impact_offset, dtype=np.float64
    )

    first = first_positions + centre - 0.5 * float(centre_distance) * axis
    second = (
        second_positions + centre + 0.5 * float(centre_distance) * axis
        + impact_offset
    )
    return first, second, np.vstack([first, second])


def _safe_collision_layout(first_symbols, first_positions, second_symbols,
                           second_positions, box_size, axis, requested_gap,
                           impact_offset=None):
    """Place reactants outside the *actual* bond detector at t=0.

    The requested gap remains the starting point. If any cross-reactant pair
    is already above BondTracker's taper threshold, separation is increased
    in tiny increments until every pair is outside, then a small geometric
    buffer is added and checked again. No reaction-specific chemistry is used.
    """

    threshold, formation_time = bonding_settings()
    radius_first = _radius(first_positions)
    radius_second = _radius(second_positions)
    requested_gap = float(requested_gap)
    centre = np.full(3, float(box_size) / 2.0, dtype=np.float64)
    base = radius_first + radius_second
    gap = requested_gap
    symbols = list(first_symbols) + list(second_symbols)
    first_count = len(first_symbols)

    def evaluate(test_gap):
        first, second, combined = _place_collision(
            first_positions, second_positions, centre, axis, base + test_gap,
            impact_offset=impact_offset,
        )
        distances, taper = _cross_pair_arrays(
            combined, symbols, first_count, box_size
        )
        max_taper = float(np.max(taper)) if taper.size else 0.0
        min_distance = float(np.min(distances)) if distances.size else float("inf")
        clearance = float(np.min(np.minimum(combined, float(box_size) - combined)))
        return first, second, combined, max_taper, min_distance, clearance

    first, second, combined, max_taper, min_distance, clearance = evaluate(gap)
    requested_max_taper = float(max_taper)
    requested_min_distance = float(min_distance)
    attempts = 0

    while max_taper > threshold:
        gap += SAFE_GAP_STEP_A
        attempts += 1
        first, second, combined, max_taper, min_distance, clearance = evaluate(gap)
        if clearance < 1.5 or attempts > 400:
            raise ValueError(
                "cannot place collision partners outside the bond detector in "
                f"a {float(box_size):g} A box; increase the box size"
            )

    # Give the threshold a small spatial buffer so floating-point noise or one
    # tiny initial displacement cannot make the starting frame a contact.
    if gap > requested_gap + 1e-12:
        gap += SAFE_GAP_BUFFER_A
        first, second, combined, max_taper, min_distance, clearance = evaluate(gap)

    if max_taper > threshold or clearance < 1.5:
        raise ValueError(
            "collision start failed the bond-separation safety check; "
            "increase the box size or requested start gap"
        )

    return first, second, combined, {
        "requested_start_gap_A": requested_gap,
        "requested_gap_min_cross_distance_A": requested_min_distance,
        "requested_gap_max_bond_taper": requested_max_taper,
        "requested_gap_was_safe": bool(requested_max_taper <= threshold),
        "actual_start_gap_A": float(gap),
        "auto_added_gap_A": float(gap - requested_gap),
        "initial_min_cross_distance_A": min_distance,
        "initial_max_bond_taper": max_taper,
        "bond_threshold": float(threshold),
        "bond_formation_time_fs": float(formation_time),
        "safe_initial_separation": bool(max_taper <= threshold),
    }


def _aim_symbol(impact_target):
    return {
        "carbon": "C",
        "oxygen": "O",
        "hydrogen": "H",
    }.get(str(impact_target or "com").lower())


def _centre_layout_in_box(first, second, box_size):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    combined = np.vstack([first, second])
    low = np.min(combined, axis=0)
    high = np.max(combined, axis=0)
    shift = np.full(3, float(box_size) / 2.0) - 0.5 * (low + high)
    first = first + shift
    second = second + shift
    return first, second, np.vstack([first, second]), shift


def _safe_targeted_collision_layout(first_symbols, first_positions,
                                    second_symbols, second_positions,
                                    box_size, axis, requested_gap,
                                    target_atom, impact_offset=None):
    """Aim the partner trajectory through one selected atom, safely.

    The molecule keeps its random orientation. A random attack axis is drawn,
    then the partner COM is placed on that ray so its subsequent directed COM
    velocity points straight through the selected atom. Separation is moved
    outward until *every* cross-reactant atom pair is at least the requested
    geometric gap and outside the real BondTracker taper threshold.
    """

    threshold, formation_time = bonding_settings()
    requested_gap = float(requested_gap)
    first_positions = np.asarray(first_positions, dtype=np.float64)
    second_positions = np.asarray(second_positions, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    target_atom = int(target_atom)
    target_local = first_positions[target_atom]
    impact_offset = np.zeros(3, dtype=np.float64) if impact_offset is None else np.asarray(
        impact_offset, dtype=np.float64
    )
    symbols = list(first_symbols) + list(second_symbols)
    first_count = len(first_symbols)

    # Put the front-most partner atom requested_gap away from the selected
    # target atom along the attack ray. Other atoms in the molecule may stick
    # out farther, so the safety loop below can move the whole partner outward.
    front_projection = float(np.min(second_positions @ axis))
    base_distance = requested_gap - front_projection
    extra_distance = 0.0

    def evaluate(extra):
        partner_centre = (
            target_local + axis * (base_distance + float(extra))
            + impact_offset
        )
        first = first_positions.copy()
        second = second_positions + partner_centre
        first, second, combined, shift = _centre_layout_in_box(
            first, second, box_size
        )
        distances, taper = _cross_pair_arrays(
            combined, symbols, first_count, box_size
        )
        max_taper = float(np.max(taper)) if taper.size else 0.0
        min_distance = float(np.min(distances)) if distances.size else float("inf")
        clearance = float(np.min(np.minimum(combined, float(box_size) - combined)))
        target_global = target_local + shift
        partner_com = np.mean(second, axis=0)
        return (
            first, second, combined, max_taper, min_distance, clearance,
            target_global, partner_com,
        )

    values = evaluate(extra_distance)
    first, second, combined, max_taper, min_distance, clearance, target_global, partner_com = values
    requested_max_taper = float(max_taper)
    requested_min_distance = float(min_distance)
    requested_was_safe = bool(
        requested_min_distance + 1e-9 >= requested_gap
        and requested_max_taper <= threshold
    )

    attempts = 0
    while min_distance + 1e-9 < requested_gap or max_taper > threshold:
        extra_distance += SAFE_GAP_STEP_A
        attempts += 1
        values = evaluate(extra_distance)
        first, second, combined, max_taper, min_distance, clearance, target_global, partner_com = values
        if clearance < 1.5 or attempts > 400:
            raise ValueError(
                "cannot place targeted collision outside the bond detector in "
                f"a {float(box_size):g} A box; increase the box size"
            )

    if extra_distance > 1e-12:
        extra_distance += SAFE_GAP_BUFFER_A
        values = evaluate(extra_distance)
        first, second, combined, max_taper, min_distance, clearance, target_global, partner_com = values

    if (
        min_distance + 1e-9 < requested_gap
        or max_taper > threshold
        or clearance < 1.5
    ):
        raise ValueError(
            "targeted collision start failed the separation safety check; "
            "increase the box size or requested start gap"
        )

    actual_gap = float(min_distance)
    return first, second, combined, {
        "requested_start_gap_A": requested_gap,
        "requested_gap_min_cross_distance_A": requested_min_distance,
        "requested_gap_max_bond_taper": requested_max_taper,
        "requested_gap_was_safe": requested_was_safe,
        "actual_start_gap_A": actual_gap,
        "auto_added_gap_A": max(0.0, actual_gap - requested_gap),
        "auto_added_ray_distance_A": float(extra_distance),
        "initial_min_cross_distance_A": actual_gap,
        "initial_max_bond_taper": float(max_taper),
        "bond_threshold": float(threshold),
        "bond_formation_time_fs": float(formation_time),
        "safe_initial_separation": bool(max_taper <= threshold),
        "target_atom": target_atom,
        "target_symbol": str(first_symbols[target_atom]),
        "target_position_A": [float(value) for value in target_global],
        "partner_com_A": [float(value) for value in partner_com],
    }


def _new_contact_diagnostic(symbols, first_count, box_size, fine_sample_fs,
                            coarse_sample_fs, fine_window_fs,
                            collision_info=None):
    threshold, formation_time = bonding_settings()
    pairs = []
    for partner_local, partner_symbol in enumerate(symbols[first_count:]):
        for target_local, target_symbol in enumerate(symbols[:first_count]):
            pairs.append({
                "partner_local": int(partner_local),
                "partner_atom": int(first_count + partner_local),
                "partner_symbol": str(partner_symbol),
                "target_atom": int(target_local),
                "target_symbol": str(target_symbol),
                "closest_distance_A": float("inf"),
                "max_taper": 0.0,
                "total_contact_fs": 0.0,
                "current_contact_fs": 0.0,
                "longest_contact_fs": 0.0,
            })

    collision_info = collision_info or {}
    target_atom = collision_info.get("target_atom")
    axis = collision_info.get("axis")

    selected_target = None
    if target_atom is not None and axis is not None:
        selected_target = {
            "target_atom": int(target_atom),
            "target_symbol": str(collision_info.get("target_symbol") or symbols[int(target_atom)]),
            "axis": np.asarray(axis, dtype=np.float64),
            "closest_distance_A": float("inf"),
            "max_taper": 0.0,
            "total_contact_fs": 0.0,
            "current_contact_fs": 0.0,
            "longest_contact_fs": 0.0,
            "closest_partner_atom": None,
            "closest_partner_symbol": None,
            "initial_target_distance_A": None,
            "initial_partner_com_distance_A": None,
            "closest_partner_com_distance_A": float("inf"),
            # Legacy whole-run plane-crossing metric. v3 keeps it only for
            # backwards compatibility; first-encounter metrics below are the
            # ones shown to the user.
            "trajectory_miss_distance_A": float("inf"),
            "trajectory_axial_at_miss_A": None,
            "closest_target_time_fs": None,
            "target_position_initial_A": None,
            "target_motion_at_closest_A": None,
            # Freeze the first physical encounter once the partner has receded
            # 0.05 A from its best-so-far approach. This prevents a later
            # periodic-box crossing from being mistaken for the impact.
            "first_encounter_complete": False,
            "first_encounter_min_distance_A": float("inf"),
            "first_encounter_time_fs": None,
            "first_encounter_miss_distance_A": None,
            "first_encounter_axial_A": None,
            "first_encounter_target_motion_A": None,
            "first_encounter_partner_atom": None,
            "first_encounter_partner_symbol": None,
            "_first_encounter_best": None,
        }

    return {
        "first_count": int(first_count),
        "box_size": float(box_size),
        "fine_sample_fs": float(fine_sample_fs),
        "coarse_sample_fs": float(coarse_sample_fs),
        "fine_window_fs": float(fine_window_fs),
        "bond_threshold": float(threshold),
        "bond_formation_time_fs": float(formation_time),
        "pairs": pairs,
        "selected_target": selected_target,
        "samples": 0,
        "elapsed_fs": 0.0,
    }


def _minimum_image_vector(vector, box_size):
    vector = np.asarray(vector, dtype=np.float64)
    return vector - float(box_size) * np.round(vector / float(box_size))


def _update_contact_diagnostic(diagnostic, positions, symbols, box_size, delta_fs):
    positions = np.asarray(positions, dtype=np.float64)
    distances, taper = _cross_pair_arrays(
        positions, symbols, diagnostic["first_count"], box_size
    )
    threshold = float(diagnostic["bond_threshold"])
    delta_fs = float(max(delta_fs, 0.0))
    diagnostic["elapsed_fs"] = float(diagnostic.get("elapsed_fs", 0.0)) + delta_fs

    for pair in diagnostic["pairs"]:
        p = pair["partner_local"]
        t = pair["target_atom"]
        distance = float(distances[p, t])
        value = float(taper[p, t])
        pair["closest_distance_A"] = min(pair["closest_distance_A"], distance)
        pair["max_taper"] = max(pair["max_taper"], value)

        if value > threshold:
            pair["total_contact_fs"] += delta_fs
            pair["current_contact_fs"] += delta_fs
            pair["longest_contact_fs"] = max(
                pair["longest_contact_fs"], pair["current_contact_fs"]
            )
        else:
            pair["current_contact_fs"] = 0.0

    selected = diagnostic.get("selected_target")
    if isinstance(selected, dict):
        target_atom = int(selected["target_atom"])
        target_position = positions[target_atom]
        partner_positions = positions[int(diagnostic["first_count"]):]
        partner_distances = distances[:, target_atom]
        partner_tapers = taper[:, target_atom]

        if len(partner_distances):
            nearest_local = int(np.argmin(partner_distances))
            nearest_distance = float(partner_distances[nearest_local])
            nearest_taper = float(partner_tapers[nearest_local])

            if nearest_distance < selected["closest_distance_A"]:
                selected["closest_distance_A"] = nearest_distance
                selected["closest_partner_atom"] = int(diagnostic["first_count"] + nearest_local)
                selected["closest_partner_symbol"] = str(
                    symbols[int(diagnostic["first_count"] + nearest_local)]
                )
                selected["closest_target_time_fs"] = float(diagnostic["elapsed_fs"])
                initial_target = selected.get("target_position_initial_A")
                if initial_target is not None:
                    motion = _minimum_image_vector(
                        target_position - np.asarray(initial_target, dtype=np.float64),
                        box_size,
                    )
                    selected["target_motion_at_closest_A"] = float(np.linalg.norm(motion))

            selected["max_taper"] = max(selected["max_taper"], nearest_taper)
            if nearest_taper > threshold:
                selected["total_contact_fs"] += delta_fs
                selected["current_contact_fs"] += delta_fs
                selected["longest_contact_fs"] = max(
                    selected["longest_contact_fs"], selected["current_contact_fs"]
                )
            else:
                selected["current_contact_fs"] = 0.0

            if selected["initial_target_distance_A"] is None:
                selected["initial_target_distance_A"] = nearest_distance
                selected["target_position_initial_A"] = [float(x) for x in target_position]

        # First-collision geometry. Measure the beam at the actual first
        # closest approach to the selected atom, not at whichever target-plane
        # crossing happens later in a periodic 10 ps trajectory.
        if len(partner_positions):
            relative_atoms = np.asarray([
                _minimum_image_vector(position - target_position, box_size)
                for position in partner_positions
            ])
            relative_com = np.mean(relative_atoms, axis=0)
            com_distance = float(np.linalg.norm(relative_com))
            selected["closest_partner_com_distance_A"] = min(
                selected["closest_partner_com_distance_A"], com_distance
            )
            if selected["initial_partner_com_distance_A"] is None:
                selected["initial_partner_com_distance_A"] = com_distance

            axis = np.asarray(selected["axis"], dtype=np.float64)
            axial = float(np.dot(relative_com, axis))
            transverse = relative_com - axial * axis
            miss = float(np.linalg.norm(transverse))

            if not selected.get("first_encounter_complete"):
                # Keep the old plane metric only during the first approach.
                previous_abs_axial = selected.get("_best_abs_axial", float("inf"))
                if abs(axial) < previous_abs_axial:
                    selected["_best_abs_axial"] = abs(axial)
                    selected["trajectory_miss_distance_A"] = miss
                    selected["trajectory_axial_at_miss_A"] = axial

                # Chemical approach uses the nearest partner atom. For the H
                # benchmark this is exactly the incoming H -> selected target.
                nearest_local = int(np.argmin(partner_distances))
                encounter_distance = float(partner_distances[nearest_local])
                best = selected.get("_first_encounter_best")
                if best is None or encounter_distance < float(best["distance_A"]):
                    initial_target = selected.get("target_position_initial_A")
                    target_motion = None
                    if initial_target is not None:
                        motion = _minimum_image_vector(
                            target_position - np.asarray(initial_target, dtype=np.float64),
                            box_size,
                        )
                        target_motion = float(np.linalg.norm(motion))
                    best = {
                        "distance_A": encounter_distance,
                        "time_fs": float(diagnostic["elapsed_fs"]),
                        "miss_A": miss,
                        "axial_A": axial,
                        "target_motion_A": target_motion,
                        "partner_atom": int(diagnostic["first_count"] + nearest_local),
                        "partner_symbol": str(symbols[int(diagnostic["first_count"] + nearest_local)]),
                    }
                    selected["_first_encounter_best"] = best

                # Freeze as soon as the first pass has clearly turned around.
                best = selected.get("_first_encounter_best")
                if (
                    best is not None
                    and encounter_distance >= float(best["distance_A"]) + FIRST_ENCOUNTER_RELEASE_A
                ):
                    selected["first_encounter_complete"] = True
                    selected["first_encounter_min_distance_A"] = float(best["distance_A"])
                    selected["first_encounter_time_fs"] = float(best["time_fs"])
                    selected["first_encounter_miss_distance_A"] = float(best["miss_A"])
                    selected["first_encounter_axial_A"] = float(best["axial_A"])
                    selected["first_encounter_target_motion_A"] = best["target_motion_A"]
                    selected["first_encounter_partner_atom"] = int(best["partner_atom"])
                    selected["first_encounter_partner_symbol"] = str(best["partner_symbol"])

    diagnostic["samples"] += 1


def _finish_contact_diagnostic(diagnostic):
    formation_time = float(diagnostic["bond_formation_time_fs"])
    threshold = float(diagnostic["bond_threshold"])
    pairs = []

    for pair in diagnostic["pairs"]:
        clean = {
            key: value for key, value in pair.items()
            if key != "current_contact_fs"
        }
        if not np.isfinite(clean["closest_distance_A"]):
            clean["closest_distance_A"] = None
        clean["entered_bond_range"] = bool(clean["max_taper"] > threshold)
        clean["confirmed_contact"] = bool(
            clean["longest_contact_fs"] + 1e-9 >= formation_time
        )
        pairs.append(clean)

    closest = min(
        (pair["closest_distance_A"] for pair in pairs
         if pair["closest_distance_A"] is not None),
        default=None,
    )
    longest = max((pair["longest_contact_fs"] for pair in pairs), default=0.0)
    total = sum(pair["total_contact_fs"] for pair in pairs)
    entered = any(pair["entered_bond_range"] for pair in pairs)
    confirmed = any(pair["confirmed_contact"] for pair in pairs)

    strongest = None
    if pairs:
        strongest = max(
            pairs,
            key=lambda pair: (
                float(pair["max_taper"]),
                float(pair["longest_contact_fs"]),
                -float(pair["closest_distance_A"] or 1e9),
            ),
        )

    by_target_element = {}
    for symbol in sorted({pair["target_symbol"] for pair in pairs}):
        group = [pair for pair in pairs if pair["target_symbol"] == symbol]
        element_closest = min(
            (pair["closest_distance_A"] for pair in group
             if pair["closest_distance_A"] is not None),
            default=None,
        )
        by_target_element[symbol] = {
            "closest_distance_A": element_closest,
            "max_taper": max((pair["max_taper"] for pair in group), default=0.0),
            "total_contact_fs": sum(pair["total_contact_fs"] for pair in group),
            "longest_contact_fs": max(
                (pair["longest_contact_fs"] for pair in group), default=0.0
            ),
            "entered_bond_range": any(pair["entered_bond_range"] for pair in group),
            "confirmed_contact": any(pair["confirmed_contact"] for pair in group),
        }

    selected = diagnostic.get("selected_target")
    selected_clean = None
    if isinstance(selected, dict):
        # If the run ends before the 0.05 A receding threshold is observed,
        # still report the best first-approach sample available.
        if not selected.get("first_encounter_complete"):
            best = selected.get("_first_encounter_best")
            if isinstance(best, dict):
                selected["first_encounter_min_distance_A"] = float(best["distance_A"])
                selected["first_encounter_time_fs"] = float(best["time_fs"])
                selected["first_encounter_miss_distance_A"] = float(best["miss_A"])
                selected["first_encounter_axial_A"] = float(best["axial_A"])
                selected["first_encounter_target_motion_A"] = best["target_motion_A"]
                selected["first_encounter_partner_atom"] = int(best["partner_atom"])
                selected["first_encounter_partner_symbol"] = str(best["partner_symbol"])
        selected_clean = {
            key: value for key, value in selected.items()
            if key not in (
                "current_contact_fs", "axis", "_best_abs_axial",
                "_first_encounter_best",
            )
        }
        for key in (
            "closest_distance_A", "closest_partner_com_distance_A",
            "trajectory_miss_distance_A", "first_encounter_min_distance_A"
        ):
            value = selected_clean.get(key)
            if value is not None and not np.isfinite(value):
                selected_clean[key] = None
        selected_clean["entered_bond_range"] = bool(
            float(selected_clean.get("max_taper", 0.0)) > threshold
        )
        selected_clean["confirmed_contact"] = bool(
            float(selected_clean.get("longest_contact_fs", 0.0)) + 1e-9 >= formation_time
        )

    return {
        "diagnostic_sample_fs": float(diagnostic["fine_sample_fs"]),
        "diagnostic_fine_sample_fs": float(diagnostic["fine_sample_fs"]),
        "diagnostic_coarse_sample_fs": float(diagnostic["coarse_sample_fs"]),
        "diagnostic_fine_window_fs": float(diagnostic["fine_window_fs"]),
        "samples": int(diagnostic["samples"]),
        "bond_threshold": threshold,
        "bond_formation_time_fs": formation_time,
        "closest_cross_distance_A": closest,
        "longest_contact_fs": float(longest),
        "total_contact_fs": float(total),
        "entered_bond_range": bool(entered),
        "confirmed_contact": bool(confirmed),
        "strongest_pair": strongest,
        "selected_target": selected_clean,
        "by_target_element": by_target_element,
        "pairs": pairs,
    }


def rotation_matrix(generator):
    """Uniform random 3-D rotation from a random unit quaternion."""

    quaternion = generator.normal(size=4)
    norm = float(np.linalg.norm(quaternion))

    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)

    w, x, y, z = quaternion / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def minimum_box_size(positions, margin=6.0):
    positions = np.asarray(positions, dtype=np.float64)

    if len(positions) <= 1:
        return float(margin)

    extent = np.ptp(positions, axis=0)
    return float(np.max(extent) + margin)


def _centred_positions(payload):
    positions = np.asarray(payload["positions"], dtype=np.float64).copy()
    symbols = list(payload["symbols"])

    if not len(symbols) or len(positions) != len(symbols):
        raise ValueError("stored molecule payload has inconsistent atoms/positions")

    positions -= np.mean(positions, axis=0)
    return symbols, positions


def _radius(positions):
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(positions, axis=1)))


def _random_axis(generator):
    axis = generator.normal(size=3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


def _impact_offset(generator, axis, distance):
    """Choose a reproducible lateral offset perpendicular to an approach ray."""

    distance = float(distance)
    if distance < 0:
        raise ValueError("impact parameter cannot be negative")
    if distance == 0:
        return np.zeros(3, dtype=np.float64)

    axis = np.asarray(axis, dtype=np.float64)
    lateral = _random_axis(generator)
    lateral -= axis * float(np.dot(lateral, axis))
    norm = float(np.linalg.norm(lateral))
    if norm <= 1e-12:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, axis))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        lateral = np.cross(axis, reference)
        norm = float(np.linalg.norm(lateral))
    return lateral * (distance / norm)


def _line_of_sight_axis(generator, positions, target_atom, partner_radius=0.0):
    """Choose a collision ray that actually exposes the selected atom.

    "Aim at carbon" should not merely mean that an infinite mathematical
    line passes through carbon while another atom sits in front of it.  This
    samples many attack directions and chooses the one with the clearest
    approach corridor to the selected atom.  It changes only the experimental
    geometry; no reaction outcome or preferred chemistry is encoded.
    """

    positions = np.asarray(positions, dtype=np.float64)
    target_atom = int(target_atom)
    target = positions[target_atom]
    others = np.delete(positions - target, target_atom, axis=0)

    candidates = []

    # For an atom on the outside of a molecule, the outward radial direction
    # is usually the cleanest chemically-neutral beam direction.  Keep it as
    # one candidate, but still compare it with random directions so atoms near
    # the molecular centre (such as formaldehyde carbon) can use an out-of-
    # plane route instead.
    molecular_centre = np.mean(positions, axis=0)
    outward = target - molecular_centre
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm > 1e-10:
        candidates.append(outward / outward_norm)

    for _ in range(TARGET_AXIS_CANDIDATES):
        candidates.append(_random_axis(generator))

    best_axis = candidates[0] if candidates else np.array([1.0, 0.0, 0.0])
    best_score = (-1.0, -1.0, -1.0)
    best_clearance = None
    best_blockers = 0

    for axis in candidates:
        if len(others) == 0:
            return np.asarray(axis, dtype=np.float64), None, 0

        axial = others @ axis
        ahead = axial > TARGET_AXIS_FORWARD_EPS_A
        blockers = int(np.sum(ahead))

        if blockers == 0:
            clearance = float('inf')
        else:
            transverse = others[ahead] - axial[ahead, None] * axis[None, :]
            clearance = float(np.min(np.linalg.norm(transverse, axis=1)))

        # Prefer no atom in front of the target.  If several rays are clear,
        # prefer the one whose nearest almost-forward atom is farthest from the
        # beam.  partner_radius is subtracted only for reporting/scoring; it
        # does not alter any force or bond rule.
        effective = clearance - float(partner_radius) if np.isfinite(clearance) else 1e9
        score = (1.0 if blockers == 0 else 0.0, effective, -float(blockers))
        if score > best_score:
            best_score = score
            best_axis = np.asarray(axis, dtype=np.float64)
            best_clearance = None if not np.isfinite(clearance) else clearance
            best_blockers = blockers

    return best_axis, best_clearance, best_blockers


def prepare_box(molecule, box_size, seed):
    symbols, positions = _centred_positions(molecule)

    required = minimum_box_size(positions)

    if float(box_size) + 1e-9 < required:
        raise ValueError(
            f"{molecule['id']} needs at least about {required:.1f} A for the "
            f"isolated test; selected box is {float(box_size):g} A"
        )

    generator = np.random.default_rng(int(seed) + 47017)
    positions = positions @ rotation_matrix(generator).T
    positions += float(box_size) / 2.0

    return symbols, positions.astype(np.float32)


def atom_reactant(symbol):
    symbol = str(symbol)
    bonds = np.empty((0, 2), dtype=np.int32)
    return {
        "id": f"atom:{symbol}",
        "formula": symbol,
        "atoms": 1,
        "heavy_atoms": 0 if symbol == "H" else 1,
        "symbols": [symbol],
        "positions": np.zeros((1, 3), dtype=np.float32),
        "bonds": bonds,
        "graph_fingerprint": molecule_store.graph_fingerprint([symbol], bonds),
        "source": {"kind": "elemental reactant"},
    }


atom_partner = atom_reactant


def load_reactant(reactant_id, library):
    if not reactant_id:
        raise ValueError("experiment needs a reactant")

    if str(reactant_id).startswith("atom:"):
        symbol = str(reactant_id).split(":", 1)[1]
        if symbol not in ("H", "C", "N", "O"):
            raise ValueError(f"unsupported elemental reactant: {symbol}")
        return atom_reactant(symbol)

    return molecule_store.load_molecule(reactant_id, root=library)


load_partner = load_reactant


def minimum_pair_box_size(molecule, partner, start_gap, margin=4.0):
    _, first = _centred_positions(molecule)
    _, second = _centred_positions(partner)
    radius_first = _radius(first)
    radius_second = _radius(second)
    centre_distance = radius_first + radius_second + float(start_gap)

    # Conservative spherical envelope. It is intentionally a little larger
    # than the exact random orientation needs so one seed cannot unexpectedly
    # clip the periodic boundary when another orientation fitted.
    return float(centre_distance + 2.0 * max(radius_first, radius_second) + margin)


def prepare_collision_box(molecule, partner, box_size, seed, start_gap,
                          impact_target="com", sampling_mode=None,
                          target_atom=None, impact_parameter=0.0):
    first_symbols, first_positions = _centred_positions(molecule)
    second_symbols, second_positions = _centred_positions(partner)

    generator = np.random.default_rng(int(seed) + 67019)
    if sampling_mode is None:
        sampling_mode = "random_orientation" if impact_target == "com" else "targeted_random"
    sampling_mode = str(sampling_mode)
    if sampling_mode not in ("targeted", "random_orientation", "targeted_random"):
        raise ValueError(f"unknown sampling mode: {sampling_mode}")
    first_rotation = np.eye(3, dtype=np.float64)
    second_rotation = np.eye(3, dtype=np.float64)
    if sampling_mode != "targeted":
        first_rotation = rotation_matrix(generator)
        second_rotation = rotation_matrix(generator)
        first_positions = first_positions @ first_rotation.T
        second_positions = second_positions @ second_rotation.T

    impact_target = str(impact_target or "com").lower()
    target_symbol = _aim_symbol(impact_target)
    requested_target_atom = target_atom
    target_atom = None
    los_clearance = None
    los_blockers = None

    targeted = sampling_mode in ("targeted", "targeted_random")
    if targeted:
        if requested_target_atom is not None:
            target_atom = int(requested_target_atom)
            if target_atom < 0 or target_atom >= len(first_symbols):
                raise ValueError("target atom is outside reactant A")
            target_symbol = str(first_symbols[target_atom])
        elif target_symbol is not None:
            candidates = [index for index, symbol in enumerate(first_symbols)
                          if str(symbol) == target_symbol]
            if not candidates:
                raise ValueError(f"{molecule.get('id', 'reactant A')} has no {target_symbol} atom to aim at")
            target_atom = int(candidates[0] if sampling_mode == "targeted" else generator.choice(candidates))
        else:
            target_atom = 0 if len(first_symbols) == 1 else None

    if target_atom is None:
        if impact_target != "com" and not targeted:
            raise ValueError(f"unknown collision impact target: {impact_target}")
        axis = _random_axis(generator)
        impact_offset = _impact_offset(generator, axis, impact_parameter)
        first_positions, second_positions, combined, safety = _safe_collision_layout(
            first_symbols,
            first_positions,
            second_symbols,
            second_positions,
            box_size,
            axis,
            start_gap,
            impact_offset=impact_offset,
        )
    else:
        axis, los_clearance, los_blockers = _line_of_sight_axis(
            generator,
            first_positions,
            target_atom,
            partner_radius=_radius(second_positions),
        )
        impact_offset = _impact_offset(generator, axis, impact_parameter)
        first_positions, second_positions, combined, safety = (
            _safe_targeted_collision_layout(
                first_symbols,
                first_positions,
                second_symbols,
                second_positions,
                box_size,
                axis,
                start_gap,
                target_atom,
                impact_offset=impact_offset,
            )
        )
        safety["line_of_sight_clearance_A"] = los_clearance
        safety["line_of_sight_blockers"] = int(los_blockers or 0)
        safety["target_geometry_revision"] = 2

    first_com = np.mean(first_positions, axis=0)
    second_com = np.mean(second_positions, axis=0)
    centre_distance = float(np.linalg.norm(second_com - first_com))

    info = {
        "first_count": len(first_symbols),
        "second_count": len(second_symbols),
        "axis": axis.astype(np.float64),
        "impact_parameter_A": float(impact_parameter),
        "impact_offset_A": impact_offset.astype(np.float64),
        "first_rotation": first_rotation,
        "second_rotation": second_rotation,
        "centre_distance_A": centre_distance,
        "impact_target": impact_target,
        "sampling_mode": sampling_mode,
        "target_atom": target_atom,
        "target_symbol": target_symbol,
        "line_of_sight_clearance_A": los_clearance,
        "line_of_sight_blockers": los_blockers,
        "target_geometry_revision": 2 if target_atom is not None else 1,
        "safety": safety,
    }

    return (
        first_symbols + second_symbols,
        combined.astype(np.float32),
        info,
    )



def prepare_microcell_box(
    reactants, box_size, seed, cluster_radius_A=3.5, minimum_gap_A=1.8
):
    """Prepare a small filled reactive environment with no prescribed collision.

    Object centres are sampled uniformly through the volume of a sphere rather
    than on a loose radial shell.  Every placement still has to satisfy the
    same geometric minimum gap and the real BondTracker taper threshold.
    """
    if len(reactants) < 3:
        raise ValueError("microreactor requires at least three reactant objects")
    if len(reactants) > 8:
        raise ValueError("microreactor currently supports at most eight objects")

    threshold, formation_time = bonding_settings()
    generator = np.random.default_rng(int(seed) + 87019)
    centre = np.full(3, float(box_size) / 2.0)
    placed = []
    ranges = []
    symbols_all = []
    positions_all = []
    cursor = 0

    import reactive as R

    for object_index, reactant in enumerate(reactants):
        symbols, local = _centred_positions(reactant)
        local = local @ rotation_matrix(generator).T
        accepted = None

        # Larger environments need more rejection-sampling attempts, but the
        # deterministic seed means a retry of the same experiment is identical.
        placement_attempts = 400 + 100 * max(0, len(reactants) - 5)
        for _ in range(placement_attempts):
            direction = _random_axis(generator)

            # r = R * U^(1/3) gives a uniform distribution through sphere
            # volume. This avoids concentrating objects on the outer shell.
            radial = float(
                cluster_radius_A * generator.random() ** (1.0 / 3.0)
            )
            candidate = local + centre + direction * radial

            if float(
                np.min(np.minimum(candidate, float(box_size) - candidate))
            ) < 1.5:
                continue

            safe = True
            types_new = _types_from_symbols(symbols)
            for old_symbols, old_positions in placed:
                offsets = candidate[:, None, :] - old_positions[None, :, :]
                offsets -= float(box_size) * np.round(
                    offsets / float(box_size)
                )
                distances = np.linalg.norm(offsets, axis=2)
                if (
                    distances.size
                    and float(np.min(distances)) < float(minimum_gap_A)
                ):
                    safe = False
                    break

                types_old = _types_from_symbols(old_symbols)
                taper = R.smooth_cutoff(
                    distances,
                    R.CUTOFF_INNER[np.ix_(types_new, types_old)],
                    R.CUTOFF_OUTER[np.ix_(types_new, types_old)],
                )
                if (
                    np.size(taper)
                    and float(np.max(taper)) > float(threshold)
                ):
                    safe = False
                    break

            if safe:
                accepted = candidate
                break

        if accepted is None:
            raise ValueError(
                "could not place microreactor objects safely; "
                "increase cluster radius or box size"
            )

        placed.append((list(symbols), accepted))
        positions_all.append(accepted)
        symbols_all.extend(symbols)
        ranges.append({
            "object_index": int(object_index),
            "reactant_id": reactant.get("id"),
            "formula": reactant.get("formula"),
            "start": int(cursor),
            "stop": int(cursor + len(symbols)),
            "atoms": int(len(symbols)),
            "initial_com_A": [
                float(value) for value in np.mean(accepted, axis=0)
            ],
        })
        cursor += len(symbols)

    total_atoms = int(sum(len(item[0]) for item in placed))
    cluster_volume = (4.0 / 3.0) * math.pi * float(cluster_radius_A) ** 3

    return symbols_all, np.vstack(positions_all).astype(np.float32), {
        "experiment_family": "microcell",
        "environment_mode": "microreactor",
        "environment_version": 1,
        "object_count": len(reactants),
        "total_atoms": total_atoms,
        "object_ranges": ranges,
        "cluster_radius_A": float(cluster_radius_A),
        "cluster_volume_A3": float(cluster_volume),
        "object_density_per_A3": float(len(reactants) / cluster_volume),
        "atom_density_per_A3": float(total_atoms / cluster_volume),
        "minimum_gap_A": float(minimum_gap_A),
        "bond_threshold": float(threshold),
        "bond_formation_time_fs": float(formation_time),
        "safe_initial_separation": True,
        "placement_distribution": "uniform_sphere_volume",
    }


def apply_microcell_inward_velocities(simulation, infos, inward_factor):
    import torch
    velocities = simulation.velocities.detach().cpu().numpy().copy()
    masses = simulation.masses.detach().cpu().numpy().reshape(-1)
    positions = simulation.positions_per_box
    per_box = int(getattr(simulation, "per_box", len(velocities) // len(infos)))
    measures = []

    for box, info in enumerate(infos):
        start = box * per_box
        stop = start + per_box
        local_v = velocities[start:stop]
        local_m = masses[start:stop]
        local_p = np.asarray(positions[box], dtype=np.float64)
        seeded_com = np.average(local_v, axis=0, weights=local_m)
        thermal_rms = float(np.sqrt(np.mean(np.sum((local_v - seeded_com) ** 2, axis=1))))
        inward_speed = max(thermal_rms, 1e-8) * float(inward_factor)
        centre = np.full(3, float(simulation.box_size) / 2.0)

        for obj in info["object_ranges"]:
            sl = slice(int(obj["start"]), int(obj["stop"]))
            obj_m = local_m[sl]
            obj_v = local_v[sl]
            obj_p = local_p[sl]
            obj_com_v = np.average(obj_v, axis=0, weights=obj_m)
            local_v[sl] -= obj_com_v
            obj_com = np.average(obj_p, axis=0, weights=obj_m)
            direction = _minimum_image_vector(centre - obj_com, simulation.box_size)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                local_v[sl] += direction / norm * inward_speed

        local_v -= np.average(local_v, axis=0, weights=local_m)
        velocities[start:stop] = local_v
        measures.append({
            "thermal_rms_speed": thermal_rms,
            "inward_speed": inward_speed,
            "inward_factor": float(inward_factor),
        })

    simulation.velocities = torch.tensor(velocities, device=simulation.device, dtype=simulation.dtype)
    return measures


def _microreactor_bond_edges(symbols, positions, box_size, threshold):
    """Instantaneous graph using the same taper family as BondTracker."""
    import reactive as R

    positions = np.asarray(positions, dtype=np.float64)
    count = len(symbols)
    first, second = np.triu_indices(count, 1)
    if len(first) == 0:
        return tuple()

    offsets = positions[second] - positions[first]
    offsets -= float(box_size) * np.round(offsets / float(box_size))
    distances = np.linalg.norm(offsets, axis=1)
    types = _types_from_symbols(symbols)
    taper = np.asarray(
        R.smooth_cutoff(
            distances,
            R.CUTOFF_INNER[types[first], types[second]],
            R.CUTOFF_OUTER[types[first], types[second]],
        ),
        dtype=np.float64,
    )
    return tuple(
        (int(a), int(b))
        for a, b, value in zip(first, second, taper)
        if float(value) > float(threshold)
    )


def _components_from_edges(count, edges):
    parents = list(range(int(count)))

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for first, second in edges:
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parents[second_root] = first_root

    grouped = {}
    for index in range(int(count)):
        grouped.setdefault(find(index), []).append(index)
    return [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(grouped.items())
    ]


def _unwrap_component_positions(positions, indices, box_size):
    """Unwrap one connected component around its first atom for COM geometry."""
    positions = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    reference = positions[int(indices[0])]
    relative = positions[indices] - reference
    relative -= float(box_size) * np.round(relative / float(box_size))
    return reference + relative


def _component_state(indices, positions, velocities, masses, box_size):
    indices = np.asarray(indices, dtype=np.int64)
    local_positions = _unwrap_component_positions(
        positions, indices, box_size
    )
    local_velocities = np.asarray(velocities, dtype=np.float64)[indices]
    local_masses = np.asarray(masses, dtype=np.float64)[indices]
    total_mass = float(np.sum(local_masses))
    if total_mass <= 0.0:
        weights = np.full(len(indices), 1.0 / max(len(indices), 1))
    else:
        weights = local_masses / total_mass

    centre = np.sum(local_positions * weights[:, None], axis=0)
    velocity = np.sum(local_velocities * weights[:, None], axis=0)
    radius = float(
        np.max(np.linalg.norm(local_positions - centre, axis=1))
    ) if len(indices) else 0.0
    return {
        "indices": indices,
        "centre_A": centre,
        "velocity_A_per_fs": velocity,
        "radius_A": radius,
    }


def _future_component_encounter_possible(
    first, second, symbols, box_size, remaining_fs,
):
    """Conservative ballistic encounter test across periodic images.

    When components are already outside interaction range, early completion is
    allowed only if their current COM trajectories cannot bring their bounding
    envelopes into the largest relevant reactive cutoff anywhere in the
    remaining requested simulation time. Twenty-seven neighbouring periodic
    images are considered.
    """
    import reactive as R

    remaining_fs = max(0.0, float(remaining_fs))
    first_indices = first["indices"]
    second_indices = second["indices"]
    types = _types_from_symbols(symbols)
    outer = R.CUTOFF_OUTER[
        np.ix_(types[first_indices], types[second_indices])
    ]
    largest_outer = float(np.max(outer)) if np.size(outer) else 0.0
    encounter_radius = (
        float(first["radius_A"])
        + float(second["radius_A"])
        + largest_outer
    )

    base_relative = (
        np.asarray(second["centre_A"], dtype=np.float64)
        - np.asarray(first["centre_A"], dtype=np.float64)
    )
    relative_velocity = (
        np.asarray(second["velocity_A_per_fs"], dtype=np.float64)
        - np.asarray(first["velocity_A_per_fs"], dtype=np.float64)
    )
    speed_squared = float(np.dot(relative_velocity, relative_velocity))
    box = float(box_size)

    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                relative = base_relative + box * np.asarray(
                    [ix, iy, iz], dtype=np.float64
                )

                if speed_squared <= 1e-24:
                    closest = float(np.linalg.norm(relative))
                else:
                    closest_time = -float(
                        np.dot(relative, relative_velocity)
                    ) / speed_squared
                    closest_time = float(np.clip(
                        closest_time, 0.0, remaining_fs
                    ))
                    closest = float(np.linalg.norm(
                        relative + relative_velocity * closest_time
                    ))

                if closest <= encounter_radius:
                    return True

    return False


def _microreactor_quiescence_snapshot(
    symbols, positions, velocities, masses, box_size, remaining_fs,
    bond_threshold,
):
    """Return current topology/contact/future-encounter activity."""
    import reactive as R

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    edges = _microreactor_bond_edges(
        symbols, positions, box_size, bond_threshold
    )
    components = _components_from_edges(len(symbols), edges)

    # Any non-zero taper across separate components means they are already in
    # the reactive approach/contact region even if BondTracker has not yet
    # confirmed a bond.
    types = _types_from_symbols(symbols)
    close_contact = False
    states = [
        _component_state(
            indices, positions, velocities, masses, box_size
        )
        for indices in components
    ]

    future_encounter = False
    for first_index, first_state in enumerate(states):
        for second_state in states[first_index + 1:]:
            left = first_state["indices"]
            right = second_state["indices"]
            offsets = (
                positions[right, None, :]
                - positions[None, left, :]
            )
            offsets -= float(box_size) * np.round(
                offsets / float(box_size)
            )
            distances = np.linalg.norm(offsets, axis=2)
            taper = R.smooth_cutoff(
                distances,
                R.CUTOFF_INNER[np.ix_(types[right], types[left])],
                R.CUTOFF_OUTER[np.ix_(types[right], types[left])],
            )
            if np.size(taper) and float(np.max(taper)) > 0.0:
                close_contact = True

            if _future_component_encounter_possible(
                first_state,
                second_state,
                symbols,
                box_size,
                remaining_fs,
            ):
                future_encounter = True

    return {
        "edges": edges,
        "component_count": int(len(components)),
        "close_contact": bool(close_contact),
        "future_encounter_possible": bool(future_encounter),
    }


def _capture_microreactor_final_state(
    recorders, simulation, symbol_lists, atom_ids,
):
    """Guarantee the termination geometry exists in every recorder."""
    positions = simulation.positions_per_box
    potentials = simulation.potential_per_box
    kinetics, temperatures = simulation.thermodynamics_per_box
    velocities_per_box = simulation.velocities_per_box
    elapsed = float(simulation.elapsed_femtoseconds)

    for box, recorder in enumerate(recorders):
        already_captured = (
            len(recorder) > 0
            and abs(float(recorder.times[-1]) - elapsed) <= 1e-9
        )
        if already_captured:
            continue
        recorder.capture(
            positions[box],
            elapsed,
            float(potentials[box]),
            float(kinetics[box]),
            float(temperatures[box]),
            velocities=velocities_per_box[box],
            box_size=simulation.box_size,
            symbols=symbol_lists[box],
            atom_ids=atom_ids[box],
        )


def run_microcell_group(reactants, seeds, options, frame_observer=None):
    from recorder import Recorder

    SimulationClass = simulation_class_for_physics(options.physics)
    boxes, infos = [], []
    for seed in seeds:
        symbols, positions, info = prepare_microcell_box(
            reactants, options.box, seed,
            options.cluster_radius_A, options.minimum_gap_A,
        )
        boxes.append((symbols, positions))
        infos.append(info)

    simulation = SimulationClass(
        boxes=boxes,
        box_size=float(options.box),
        time_step=float(options.time_step),
        target_temperature=float(options.temperature),
        friction=float(options.friction),
        device=options.device,
        random_seed=int(seeds[0]),
    )
    measures = apply_microcell_inward_velocities(
        simulation, infos, options.inward_factor
    )
    recorders = [
        Recorder(
            simulation.symbols_for(i),
            simulation.box_size,
            maximum_frames=int(options.max_frames),
        )
        for i in range(len(seeds))
    ]
    symbol_lists = [
        list(simulation.symbols_for(i))
        for i in range(len(seeds))
    ]
    atom_ids = [
        np.arange(len(symbols), dtype=np.uint32)
        for symbols in symbol_lists
    ]
    _emit_frame_observer(
        frame_observer, simulation, seeds, symbol_lists, infos
    )

    time_step_fs = float(options.time_step)
    requested_fs = float(options.picoseconds) * 1000.0
    total_steps = max(1, int(requested_fs / time_step_fs))
    capture_steps = max(1, int(options.capture_every))
    sample_steps = max(
        1,
        int(round(
            float(options.diagnostic_sample_fs) / time_step_fs
        )),
    )
    sample_steps = max(1, math.gcd(capture_steps, sample_steps))

    bond_threshold, formation_time_fs = bonding_settings()
    minimum_runtime_fs = (
        MICROREACTOR_QUIESCENCE_MIN_FORMATION_WINDOWS
        * formation_time_fs
    )
    quiet_window_fs = (
        MICROREACTOR_QUIESCENCE_QUIET_FORMATION_WINDOWS
        * formation_time_fs
    )

    masses = (
        simulation.masses.detach().cpu().numpy().reshape(-1)
    )
    per_box = int(getattr(
        simulation, "per_box", len(masses) // max(len(seeds), 1)
    ))

    quiescence = []
    initial_positions = simulation.positions_per_box
    initial_velocities = simulation.velocities_per_box
    for box, symbols in enumerate(symbol_lists):
        mass_slice = masses[
            box * per_box:(box + 1) * per_box
        ]
        initial = _microreactor_quiescence_snapshot(
            symbols,
            initial_positions[box],
            initial_velocities[box],
            mass_slice,
            simulation.box_size,
            requested_fs,
            bond_threshold,
        )
        quiescence.append({
            "last_edges": initial["edges"],
            "last_activity_fs": 0.0,
            "topology_changes": 0,
            "close_contact_samples": int(initial["close_contact"]),
            "future_encounter_samples": int(
                initial["future_encounter_possible"]
            ),
            "last_snapshot": initial,
        })

    steps_done = 0
    started = time.time()
    numerical_failure = False
    quiescent_completion = False

    while steps_done < total_steps:
        simulation.target_temperature = float(options.temperature)
        chunk = min(sample_steps, total_steps - steps_done)
        simulation.step(chunk)
        steps_done += chunk

        elapsed_fs = float(simulation.elapsed_femtoseconds)
        remaining_fs = max(0.0, requested_fs - elapsed_fs)
        positions = simulation.positions_per_box
        potentials = simulation.potential_per_box
        velocities_per_box = simulation.velocities_per_box

        finite = (
            np.all(np.isfinite(np.asarray(potentials, dtype=float)))
            and np.all(np.isfinite(np.asarray(positions, dtype=float)))
            and np.all(np.isfinite(
                np.asarray(velocities_per_box, dtype=float)
            ))
        )
        if not finite:
            numerical_failure = True
            break

        all_quiet = elapsed_fs + 1e-9 >= minimum_runtime_fs

        for box, symbols in enumerate(symbol_lists):
            mass_slice = masses[
                box * per_box:(box + 1) * per_box
            ]
            snapshot = _microreactor_quiescence_snapshot(
                symbols,
                positions[box],
                velocities_per_box[box],
                mass_slice,
                simulation.box_size,
                remaining_fs,
                bond_threshold,
            )
            state = quiescence[box]

            topology_changed = (
                snapshot["edges"] != state["last_edges"]
            )
            if topology_changed:
                state["topology_changes"] += 1
                state["last_edges"] = snapshot["edges"]

            if snapshot["close_contact"]:
                state["close_contact_samples"] += 1
            if snapshot["future_encounter_possible"]:
                state["future_encounter_samples"] += 1

            active_now = bool(
                topology_changed
                or snapshot["close_contact"]
                or snapshot["future_encounter_possible"]
            )
            if active_now:
                state["last_activity_fs"] = elapsed_fs

            state["last_snapshot"] = snapshot
            quiet_for_fs = elapsed_fs - float(
                state["last_activity_fs"]
            )
            box_quiet = bool(
                elapsed_fs + 1e-9 >= minimum_runtime_fs
                and quiet_for_fs + 1e-9 >= quiet_window_fs
                and not snapshot["close_contact"]
                and not snapshot["future_encounter_possible"]
            )
            all_quiet = all_quiet and box_quiet

        capture_now = (
            steps_done % capture_steps == 0
            or steps_done >= total_steps
            or all_quiet
        )
        if capture_now:
            kinetics, temperatures = simulation.thermodynamics_per_box
            for box, recorder in enumerate(recorders):
                already_captured = (
                    len(recorder) > 0
                    and abs(
                        float(recorder.times[-1]) - elapsed_fs
                    ) <= 1e-9
                )
                if not already_captured:
                    recorder.capture(
                        positions[box],
                        elapsed_fs,
                        float(potentials[box]),
                        float(kinetics[box]),
                        float(temperatures[box]),
                        velocities=velocities_per_box[box],
                        box_size=simulation.box_size,
                        symbols=symbol_lists[box],
                        atom_ids=atom_ids[box],
                    )

        _emit_frame_observer(
            frame_observer, simulation, seeds, symbol_lists, infos
        )

        if all_quiet:
            quiescent_completion = True
            break

    # Even numerical failures and non-capture-boundary exits retain the exact
    # final finite geometry when possible. Quiescent completion in particular
    # is a successful experiment, not a failure.
    if not numerical_failure:
        _capture_microreactor_final_state(
            recorders, simulation, symbol_lists, atom_ids
        )

    actual_fs = float(simulation.elapsed_femtoseconds)
    if numerical_failure:
        termination_reason = "numerical_failure"
    elif quiescent_completion:
        termination_reason = "quiescent"
    else:
        termination_reason = "duration_complete"

    for box, info in enumerate(infos):
        state = quiescence[box]
        snapshot = state.get("last_snapshot") or {}
        info["termination"] = {
            "version": MICROREACTOR_QUIESCENCE_VERSION,
            "reason": termination_reason,
            "requested_time_fs": float(requested_fs),
            "actual_time_fs": actual_fs,
            "requested_picoseconds": float(options.picoseconds),
            "actual_picoseconds": actual_fs / 1000.0,
            "completed_early": bool(quiescent_completion),
            "minimum_runtime_fs": float(minimum_runtime_fs),
            "quiet_window_fs": float(quiet_window_fs),
            "last_activity_fs": float(state["last_activity_fs"]),
            "quiet_for_fs": max(
                0.0, actual_fs - float(state["last_activity_fs"])
            ),
            "topology_changes": int(state["topology_changes"]),
            "close_contact_samples": int(
                state["close_contact_samples"]
            ),
            "future_encounter_samples": int(
                state["future_encounter_samples"]
            ),
            "final_component_count": int(
                snapshot.get("component_count", 0)
            ),
            "final_close_contact": bool(
                snapshot.get("close_contact", False)
            ),
            "future_encounter_possible_at_end": bool(
                snapshot.get("future_encounter_possible", False)
            ),
            "criterion": (
                "after the minimum runtime, require a continuous quiet "
                "window with no topology change, no cross-component "
                "reactive contact, and no ballistic component encounter "
                "predicted within the remaining requested simulation time"
            ),
        }

    return (
        recorders,
        simulation,
        time.time() - started,
        numerical_failure,
        measures,
        [None] * len(seeds),
        infos,
    )


def outcome_for_reactants(recorder, reactants, library="molecules"):
    try:
        components = molecule_store.molecules_at(recorder, -1)
    except Exception:
        return "unclassified", []
    final = [_component_summary(component, library) for component in components]
    initial = sorted(str(r.get("graph_fingerprint")) for r in reactants)
    ending = sorted(str(c.get("graph_fingerprint")) for c in components)
    if initial == ending:
        return "no reaction", final
    expected = sum(int(r.get("atoms", len(r["symbols"]))) for r in reactants)
    found = sum(int(c.get("atoms", 0)) for c in components)
    if found != expected:
        return "unclassified", final
    return ("joined" if len(components) == 1 else "reaction"), final


def summarise_microcell(recorder, reactants, seed, options, wall_seconds, stopped_early,
                        velocity_measure=None, cell_info=None, simulation=None):
    import analysis
    result = analysis.analyse(recorder, stride=max(1, int(options.stride)), structures=False)
    outcome, final_components = outcome_for_reactants(recorder, reactants, library=options.library)
    final_time_fs = float(recorder.times[-1]) if len(recorder) else 0.0
    physics = physics_metadata(options.physics, simulation=simulation)
    return {
        "number": 0,
        "file": f"run_s{int(seed):04d}.npz",
        "seed": int(seed),
        "finished": True,
        "experiment_type": "reaction_microreactor",
        "experiment_family": "microcell",
        "environment_mode": "microreactor",
        "environment_version": 1,
        **physics,
        "temperature_K": float(options.temperature),
        "box": float(options.box),
        "atoms": sum(len(r["symbols"]) for r in reactants),
        "picoseconds": round(final_time_fs / 1000.0, 4),
        "requested_picoseconds": float(options.picoseconds),
        "frames": len(recorder),
        "group_size": int(options.group),
        "group_stopped_early": bool(stopped_early),
        "wall_seconds": round(float(wall_seconds), 2),
        "characterisation_outcome": outcome,
        "reacted": bool(outcome not in ("no reaction", "unclassified")),
        "final_components": final_components,
        "headline": f"{outcome}: {analysis.headline(result)}",
        "stable": bool(result.get("stable", True)),
        "energy_jumps": result.get("energy_jumps", 0),
        "largest_energy_jump": result.get("largest_energy_jump", 0.0),
        "final_temperature": result.get("temperature", {}).get("final"),
        "final_potential": result.get("potential", {}).get("final"),
        "species_seen": sorted(result.get("seen", [])),
        "reactant_ids": [r.get("id") for r in reactants],
        "reactant_formulas": [r.get("formula") for r in reactants],
        "cluster_radius_A": float(options.cluster_radius_A),
        "minimum_gap_A": float(options.minimum_gap_A),
        "inward_factor": float(options.inward_factor),
        "microcell_info": cell_info or {},
        "cluster_volume_A3": float((cell_info or {}).get("cluster_volume_A3", 0.0)),
        "object_density_per_A3": float(
            (cell_info or {}).get("object_density_per_A3", 0.0)
        ),
        "atom_density_per_A3": float(
            (cell_info or {}).get("atom_density_per_A3", 0.0)
        ),
        "placement_distribution": (cell_info or {}).get(
            "placement_distribution", "legacy"
        ),
        "termination": (cell_info or {}).get("termination", {
            "reason": (
                "numerical_failure"
                if stopped_early else "duration_complete"
            ),
            "completed_early": False,
        }),
        "termination_reason": (
            (cell_info or {}).get("termination") or {}
        ).get(
            "reason",
            "numerical_failure" if stopped_early else "duration_complete",
        ),
        "terminated_quiescent": bool(
            ((cell_info or {}).get("termination") or {}).get(
                "reason"
            ) == "quiescent"
        ),
        "thermal_rms_speed": float((velocity_measure or {}).get("thermal_rms_speed", 0.0)),
        "inward_speed": float((velocity_measure or {}).get("inward_speed", 0.0)),
    }



def apply_approach_velocities(simulation, infos, approach_factor):
    """Give the two objects a head-on relative COM velocity.

    The factor is deliberately expressed relative to the thermal RMS speed
    already produced by the normal simulation initialiser. That avoids a
    second, competing unit system: factor 1 means a directed relative speed
    equal to the box's initial thermal RMS speed; factor 2 means twice it.
    Internal thermal motion is preserved while the random COM drift of each
    reactant is removed. Equal and opposite momentum is assigned according to
    the two reactant masses.
    """

    import torch

    velocities = simulation.velocities.detach().cpu().numpy().copy()
    masses = simulation.masses.detach().cpu().numpy().reshape(-1)
    per_box = int(getattr(simulation, "per_box", len(velocities) // len(infos)))

    measured = []

    for box, info in enumerate(infos):
        start = box * per_box
        stop = start + per_box
        local_v = velocities[start:stop]
        local_m = masses[start:stop]

        first_count = int(info["first_count"])
        second_count = int(info["second_count"])
        if first_count + second_count != len(local_v):
            raise ValueError("collision setup atom count does not match batched box")

        first_slice = slice(0, first_count)
        second_slice = slice(first_count, first_count + second_count)

        first_mass = float(np.sum(local_m[first_slice]))
        second_mass = float(np.sum(local_m[second_slice]))
        total_mass = first_mass + second_mass
        if first_mass <= 0 or second_mass <= 0 or total_mass <= 0:
            raise ValueError("collision partner has invalid mass")

        original_v = local_v.copy()
        box_com = np.average(original_v, axis=0, weights=local_m)
        seeded_relative_rms = float(np.sqrt(np.mean(np.sum(
            (original_v - box_com) ** 2, axis=1
        ))))

        first_com = np.average(
            local_v[first_slice], axis=0, weights=local_m[first_slice]
        )
        second_com = np.average(
            local_v[second_slice], axis=0, weights=local_m[second_slice]
        )

        # Preserve internal thermal velocities, but eliminate random whole-
        # object translation before imposing the controlled collision.
        local_v[first_slice] -= first_com
        local_v[second_slice] -= second_com

        internal_thermal_rms = float(np.sqrt(np.mean(np.sum(local_v * local_v, axis=1))))
        # A one-atom reactant has no internal degrees of freedom, so removing
        # both object COM velocities makes an atom+atom scale exactly zero.
        # Only in that degenerate case, use the seeded relative thermal motion
        # from the normal initializer. Molecular collision behavior is
        # unchanged because its internal RMS remains non-zero.
        thermal_rms = (
            internal_thermal_rms
            if internal_thermal_rms > 1e-8 else seeded_relative_rms
        )
        relative_speed = max(thermal_rms, 1e-8) * float(approach_factor)
        axis = np.asarray(info["axis"], dtype=np.float64)

        first_velocity = axis * relative_speed * (second_mass / total_mass)
        second_velocity = -axis * relative_speed * (first_mass / total_mass)

        local_v[first_slice] += first_velocity
        local_v[second_slice] += second_velocity
        velocities[start:stop] = local_v

        measured.append({
            "thermal_rms_speed": thermal_rms,
            "internal_thermal_rms_speed": internal_thermal_rms,
            "seeded_relative_rms_speed": seeded_relative_rms,
            "relative_speed": relative_speed,
        })

    simulation.velocities = torch.tensor(
        velocities,
        device=simulation.device,
        dtype=simulation.dtype,
    )

    return measured


def _component_summary(component, library):
    fingerprint = component.get("graph_fingerprint")
    known = molecule_store.molecule_by_fingerprint(fingerprint, root=library)
    return {
        "id": known.get("id") if known else None,
        "formula": component.get("formula"),
        "atoms": int(component.get("atoms", 0)),
        "fingerprint": fingerprint,
    }


def outcome_for(recorder, molecule, partner=None, library="molecules"):
    """Classify the final graph without prescribing any reaction."""

    try:
        components = molecule_store.molecules_at(recorder, -1)
    except Exception:
        return "unclassified", []

    final = [_component_summary(component, library) for component in components]

    if partner is None:
        if not components:
            return "no connected structure", final

        if len(components) == 1 and int(components[0].get("atoms", 0)) == int(
            molecule.get("atoms", len(molecule["symbols"]))
        ):
            if components[0].get("graph_fingerprint") == molecule.get(
                "graph_fingerprint"
            ):
                return "intact", final
            return "rearranged", final

        return "fragmented", final

    initial_fingerprints = sorted([
        str(molecule.get("graph_fingerprint")),
        str(partner.get("graph_fingerprint")),
    ])
    final_fingerprints = sorted(
        str(component.get("graph_fingerprint")) for component in components
    )

    if final_fingerprints == initial_fingerprints:
        return "no reaction", final

    expected_atoms = int(molecule.get("atoms", len(molecule["symbols"]))) + int(
        partner.get("atoms", len(partner["symbols"]))
    )
    final_atoms = sum(int(component.get("atoms", 0)) for component in components)

    if final_atoms != expected_atoms:
        return "unclassified", final

    if len(components) == 1:
        return "joined", final

    return "reaction", final


def summarise(recorder, molecule, partner, seed, options, wall_seconds,
              stopped_early, collision_measure=None, collision_info=None,
              contact_diagnostic=None, simulation=None):
    import analysis

    result = analysis.analyse(
        recorder,
        stride=max(1, int(options.stride)),
        structures=False,
    )

    outcome, final_components = outcome_for(
        recorder,
        molecule,
        partner=partner,
        library=options.library,
    )

    ordinary_headline = analysis.headline(result)
    headline = f"{outcome}: {ordinary_headline}"

    if stopped_early:
        headline = "group stopped early; " + headline

    final_time_fs = float(recorder.times[-1]) if len(recorder) else 0.0
    atoms = len(molecule["symbols"]) + (len(partner["symbols"]) if partner else 0)

    physics = physics_metadata(options.physics, simulation=simulation)
    entry = {
        "number": 0,
        "file": f"run_s{int(seed):04d}.npz",
        "seed": int(seed),
        "finished": True,
        "experiment_type": "characterisation",
        **physics,
        "test": options.test,
        "molecule_id": molecule.get("id"),
        "formula": molecule.get("formula"),
        "initial_graph_fingerprint": molecule.get("graph_fingerprint"),
        "temperature_K": float(options.temperature),
        "box": float(options.box),
        "atoms": atoms,
        "picoseconds": round(final_time_fs / 1000.0, 4),
        "requested_picoseconds": float(options.picoseconds),
        "frames": len(recorder),
        "group_size": int(options.group),
        "group_stopped_early": bool(stopped_early),
        "wall_seconds": round(float(wall_seconds), 2),
        "characterisation_outcome": outcome,
        "reacted": bool(partner is not None and outcome not in ("no reaction", "unclassified")),
        "final_components": final_components,
        "headline": headline,
        "stable": bool(result.get("stable", True)),
        "energy_jumps": result.get("energy_jumps", 0),
        "largest_energy_jump": result.get("largest_energy_jump", 0.0),
        "final_temperature": result.get("temperature", {}).get("final"),
        "final_potential": result.get("potential", {}).get("final"),
        "species_seen": sorted(result.get("seen", [])),
        "source": molecule.get("source", {}),
    }

    if partner is not None:
        entry.update({
            "partner_id": partner.get("id"),
            "partner_formula": partner.get("formula"),
            "partner_graph_fingerprint": partner.get("graph_fingerprint"),
            "approach_factor": float(options.approach_factor),
            "impact_parameter_A": float(
                getattr(options, "impact_parameter", 0.0)
            ),
            "start_gap_A": float(options.start_gap),
            "impact_target": str(options.impact_target),
        })
        if collision_measure:
            entry["thermal_rms_speed"] = float(
                collision_measure.get("thermal_rms_speed", 0.0)
            )
            entry["relative_approach_speed"] = float(
                collision_measure.get("relative_speed", 0.0)
            )
        if collision_info:
            entry["aim_target_atom"] = collision_info.get("target_atom")
            entry["aim_target_symbol"] = collision_info.get("target_symbol")
        if collision_info and collision_info.get("safety"):
            entry["collision_start_safety"] = collision_info["safety"]
            entry["actual_start_gap_A"] = float(
                collision_info["safety"].get("actual_start_gap_A", options.start_gap)
            )
        if contact_diagnostic:
            entry["collision_diagnostics"] = contact_diagnostic

    return entry


def _emit_frame_observer(observer, simulation, seeds, symbol_lists, infos):
    if observer is None:
        return

    positions = simulation.positions_per_box
    forces = (
        simulation.forces.detach()
        .reshape(simulation.box_count, simulation.per_box, 3)
        .cpu().numpy()
    )
    potentials = simulation.potential_per_box
    kinetics, temperatures = simulation.thermodynamics_per_box
    for box, seed in enumerate(seeds):
        observer({
            "box": int(box),
            "seed": int(seed),
            "symbols": list(symbol_lists[box]),
            "positions_A": np.asarray(positions[box], dtype=np.float64).copy(),
            "forces_eV_per_A": np.asarray(forces[box], dtype=np.float64).copy(),
            "potential_energy_eV": float(potentials[box]),
            "kinetic_energy_eV": float(kinetics[box]),
            "temperature_K": float(temperatures[box]),
            "time_fs": float(simulation.elapsed_femtoseconds),
            "box_A": float(simulation.box_size),
            "collision_info": infos[box],
        })


def run_group(molecule, partner, seeds, options, frame_observer=None):
    from recorder import Recorder

    SimulationClass = simulation_class_for_physics(options.physics)

    infos = []
    boxes = []

    for seed in seeds:
        if partner is None:
            boxes.append(prepare_box(molecule, options.box, seed))
            infos.append(None)
        else:
            symbols, positions, info = prepare_collision_box(
                molecule,
                partner,
                options.box,
                seed,
                options.start_gap,
                options.impact_target,
                options.sampling_mode,
                options.target_atom_a,
                getattr(options, "impact_parameter", 0.0),
            )
            boxes.append((symbols, positions))
            infos.append(info)

    simulation = SimulationClass(
        boxes=boxes,
        box_size=float(options.box),
        time_step=float(options.time_step),
        target_temperature=float(options.temperature),
        friction=float(options.friction),
        device=options.device,
        random_seed=int(seeds[0]),
    )

    collision_measures = [None for _ in seeds]
    if partner is not None:
        collision_measures = apply_approach_velocities(
            simulation, infos, options.approach_factor
        )

    recorders = [
        Recorder(
            simulation.symbols_for(box),
            simulation.box_size,
            maximum_frames=int(options.max_frames),
        )
        for box in range(len(seeds))
    ]

    symbol_lists = [
        list(simulation.symbols_for(box))
        for box in range(len(seeds))
    ]
    atom_ids = [
        np.arange(len(symbols), dtype=np.uint32)
        for symbols in symbol_lists
    ]

    _emit_frame_observer(
        frame_observer, simulation, seeds, symbol_lists, infos
    )

    total_steps = max(
        1,
        int(float(options.picoseconds) * 1000.0 / float(options.time_step)),
    )
    capture_steps = max(1, int(options.capture_every))

    # Targeted impacts happen within the first few hundred femtoseconds at
    # these box sizes.  Sample that collision window at high resolution, then
    # fall back to the old coarse cadence for the rest of a 10 ps trajectory.
    # This catches a fast fly-by without making every later quiet picosecond
    # pay the Python-call overhead of 1 fs stepping.
    fine_steps = max(
        1,
        int(round(float(options.diagnostic_sample_fs) / float(options.time_step))),
    )
    coarse_steps = max(
        1,
        int(round(float(options.diagnostic_coarse_sample_fs) / float(options.time_step))),
    )
    fine_steps = max(1, math.gcd(capture_steps, fine_steps))
    coarse_steps = max(1, math.gcd(capture_steps, coarse_steps))
    fine_sample_fs = fine_steps * float(options.time_step)
    coarse_sample_fs = coarse_steps * float(options.time_step)
    fine_window_fs = max(0.0, float(options.diagnostic_fine_window_fs))

    diagnostics = [None for _ in seeds]
    if partner is not None:
        for box, info in enumerate(infos):
            diagnostics[box] = _new_contact_diagnostic(
                symbol_lists[box],
                info["first_count"],
                simulation.box_size,
                fine_sample_fs,
                coarse_sample_fs,
                fine_window_fs,
                collision_info=info,
            )
            # t=0 is checked too, with zero elapsed contact time. This makes
            # the saved diagnostic explicitly prove the reactants started
            # outside the bond detector rather than merely assuming it.
            _update_contact_diagnostic(
                diagnostics[box],
                simulation.positions_for(box),
                symbol_lists[box],
                simulation.box_size,
                0.0,
            )

    steps_done = 0
    started = time.time()
    stopped_early = False
    last_heartbeat_step = 0

    seed_label = (
        str(seeds[0]) if len(seeds) == 1
        else f"{seeds[0]}-{seeds[-1]}"
    )

    while steps_done < total_steps:
        simulation.target_temperature = float(options.temperature)

        elapsed_before_fs = steps_done * float(options.time_step)
        diagnostic_steps = (
            fine_steps if elapsed_before_fs < fine_window_fs else coarse_steps
        )
        # Do not jump across the fine/coarse boundary in one chunk.
        if elapsed_before_fs < fine_window_fs:
            until_boundary = max(
                1,
                int(round((fine_window_fs - elapsed_before_fs) / float(options.time_step))),
            )
            diagnostic_steps = min(diagnostic_steps, until_boundary)

        this_chunk = min(diagnostic_steps, total_steps - steps_done)
        simulation.step(this_chunk)
        steps_done += this_chunk
        elapsed_sample_fs = this_chunk * float(options.time_step)

        capture_now = (
            steps_done % capture_steps == 0
            or steps_done >= total_steps
        )

        positions_per_box = None
        if partner is not None or capture_now or frame_observer is not None:
            positions_per_box = simulation.positions_per_box

        potentials = simulation.potential_per_box

        if partner is not None:
            for box, diagnostic in enumerate(diagnostics):
                _update_contact_diagnostic(
                    diagnostic,
                    positions_per_box[box],
                    symbol_lists[box],
                    simulation.box_size,
                    elapsed_sample_fs,
                )

        if capture_now:
            kinetics, temperatures = simulation.thermodynamics_per_box
            velocities_per_box = simulation.velocities_per_box
            for box, recorder in enumerate(recorders):
                recorder.capture(
                    positions_per_box[box],
                    simulation.elapsed_femtoseconds,
                    float(potentials[box]),
                    float(kinetics[box]),
                    float(temperatures[box]),
                    velocities=velocities_per_box[box],
                    box_size=simulation.box_size,
                    symbols=symbol_lists[box],
                    atom_ids=atom_ids[box],
                )

        _emit_frame_observer(
            frame_observer, simulation, seeds, symbol_lists, infos
        )

        heartbeat_interval = capture_steps * 20
        if (
            steps_done - last_heartbeat_step >= heartbeat_interval
            or steps_done >= total_steps
        ):
            write_heartbeat(
                options.out,
                seed_label,
                steps_done,
                total_steps,
                started,
                len(seeds),
            )
            last_heartbeat_step = steps_done

        if not np.all(np.isfinite(np.asarray(potentials, dtype=float))):
            stopped_early = True
            break

    finished_diagnostics = [
        _finish_contact_diagnostic(diagnostic) if diagnostic is not None else None
        for diagnostic in diagnostics
    ]

    return (
        recorders,
        simulation,
        time.time() - started,
        stopped_early,
        collision_measures,
        finished_diagnostics,
        infos,
    )


def parse_seed_list(text):
    if not text:
        return None

    values = []

    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))

    return values


def write_experiment(folder, molecule, partner, options, requested_seeds):
    physics = physics_metadata(options.physics)
    payload = {
        "format": "ChemistryModel controlled characterisation",
        "version": 8,
        **physics,
        "test": options.test,
        "sampling_mode": str(options.sampling_mode),
        "target_atom_a": options.target_atom_a,
        "molecule_id": molecule.get("id"),
        "formula": molecule.get("formula"),
        "graph_fingerprint": molecule.get("graph_fingerprint"),
        "temperature_K": float(options.temperature),
        "duration_ps": float(options.picoseconds),
        "box_A": float(options.box),
        "time_step_fs": float(options.time_step),
        "friction": float(options.friction),
        "capture_every_steps": int(options.capture_every),
        "group_size": int(options.group),
        "diagnostic_sample_fs": float(options.diagnostic_sample_fs),
        "diagnostic_fine_window_fs": float(options.diagnostic_fine_window_fs),
        "diagnostic_coarse_sample_fs": float(options.diagnostic_coarse_sample_fs),
        "collision_diagnostic_revision": 3,
        "bond_threshold": float(bonding_settings()[0]),
        "bond_formation_time_fs": float(bonding_settings()[1]),
        "group_policy": (
            f"up to {int(options.group)} boxes per group; exactly one group "
            "active at a time; a smaller final group is allowed"
        ),
        "requested_seeds": [int(seed) for seed in requested_seeds],
        "library": os.path.normpath(options.library),
        "source": molecule.get("source", {}),
    }

    if partner is not None:
        payload.update({
            "partner_id": partner.get("id"),
            "partner_formula": partner.get("formula"),
            "partner_graph_fingerprint": partner.get("graph_fingerprint"),
            "start_gap_A": float(options.start_gap),
            "approach_factor": float(options.approach_factor),
            "impact_parameter_A": float(
                getattr(options, "impact_parameter", 0.0)
            ),
            "impact_target": str(options.impact_target),
            "target_atom_a": options.target_atom_a,
            "sampling_mode": str(options.sampling_mode),
            "target_geometry_revision": (2 if str(options.sampling_mode) != "random_orientation" else 1),
            "collision_geometry": (
                "head-on centre-of-mass approach; random independent orientations"
                if str(options.sampling_mode) == "random_orientation"
                else "partner COM trajectory aimed through the selected atom on reactant A; "
                     + ("stored orientations retained"
                        if str(options.sampling_mode) == "targeted"
                        else "independent orientations randomized around the targeted encounter")
            ),
            "approach_definition": (
                "directed relative COM speed = approach_factor times the initial "
                "thermal RMS atomic speed in that box; random object COM drift removed"
            ),
        })

    path = os.path.join(folder, "experiment.json")
    temporary = path + ".part"

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled atomistic tests on a stored molecule."
    )
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--library", default="molecules")
    parser.add_argument(
        "--test", default="isolated", choices=["isolated", "with_partner"]
    )
    parser.add_argument(
        "--physics",
        default="standard",
        choices=["standard", "high_fidelity", "optimised-valence"],
        help=(
            "Atomistic physics used for this controlled test. high_fidelity "
            "is the older competitive H-transfer model; optimised-valence "
            "selects the current factorised/cached H-state and batched "
            "heavy-valence production model."
        ),
    )
    parser.add_argument("--partner", default=None)
    parser.add_argument(
        "--sampling-mode", default=None,
        choices=["targeted", "random_orientation", "targeted_random"],
        help="How orientations and the approach axis are sampled.",
    )
    parser.add_argument("--target-atom-a", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=250.0)
    parser.add_argument("--ps", dest="picoseconds", type=float, default=10.0)
    parser.add_argument("--box", type=float, default=12.0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--group", type=int, default=DEFAULT_GROUP_SIZE,
        help=(
            "maximum independent boxes advanced together; the final group "
            "may be smaller"
        ),
    )
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--start-gap", type=float, default=2.5)
    parser.add_argument("--approach-factor", type=float, default=2.0)
    parser.add_argument(
        "--impact-parameter", type=float, default=0.0,
        help=(
            "lateral offset of the partner trajectory in angstrom; zero "
            "preserves the historical direct-impact geometry"
        ),
    )
    parser.add_argument(
        "--impact-target",
        default="com",
        choices=["com", "carbon", "oxygen", "hydrogen"],
        help=(
            "Where the partner COM trajectory is aimed. 'com' preserves the "
            "original random/centre baseline; elemental targets aim through a "
            "randomly selected atom of that element in the stored molecule."
        ),
    )
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--capture-every", type=int, default=40)
    parser.add_argument(
        "--diagnostic-sample-fs", type=float,
        default=DEFAULT_DIAGNOSTIC_SAMPLE_FS,
        help="Fine contact sampling cadence during the initial collision window.",
    )
    parser.add_argument(
        "--diagnostic-fine-window-fs", type=float,
        default=DEFAULT_DIAGNOSTIC_FINE_WINDOW_FS,
        help="How long to keep the fine collision diagnostic cadence.",
    )
    parser.add_argument(
        "--diagnostic-coarse-sample-fs", type=float,
        default=DEFAULT_DIAGNOSTIC_COARSE_SAMPLE_FS,
        help="Contact sampling cadence after the initial collision window.",
    )
    parser.add_argument("--max-frames", type=int, default=40000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)

    options = parser.parse_args()

    if options.sampling_mode is None:
        options.sampling_mode = (
            "random_orientation" if options.impact_target == "com"
            else "targeted_random"
        )

    if options.repeats < 1:
        raise SystemExit("repeats must be at least 1")

    if options.group < 1:
        raise SystemExit("group must be at least 1")

    try:
        molecule = load_reactant(options.molecule, options.library)
    except Exception as problem:
        raise SystemExit(f"cannot load reactant A: {problem}")
    partner = None
    if options.test == "with_partner":
        try:
            partner = load_partner(options.partner, options.library)
        except Exception as problem:
            raise SystemExit(f"cannot load partner: {problem}")

        if float(options.start_gap) <= 0:
            raise SystemExit("start-gap must be greater than zero")
        if float(options.approach_factor) <= 0:
            raise SystemExit("approach-factor must be greater than zero")
        if float(options.impact_parameter) < 0:
            raise SystemExit("impact-parameter cannot be negative")
        if float(options.diagnostic_sample_fs) <= 0:
            raise SystemExit("diagnostic-sample-fs must be greater than zero")
        if float(options.diagnostic_fine_window_fs) < 0:
            raise SystemExit("diagnostic-fine-window-fs cannot be negative")
        if float(options.diagnostic_coarse_sample_fs) <= 0:
            raise SystemExit("diagnostic-coarse-sample-fs must be greater than zero")

        target_symbol = _aim_symbol(options.impact_target)
        if options.target_atom_a is not None and not (
            0 <= int(options.target_atom_a) < len(molecule["symbols"])
        ):
            raise SystemExit("target-atom-a is outside reactant A")
        if options.target_atom_a is None and target_symbol is not None and target_symbol not in list(molecule["symbols"]):
            raise SystemExit(
                f"{molecule.get('id', options.molecule)} contains no "
                f"{target_symbol} atom to aim at"
            )

    requested = parse_seed_list(options.seed_list)

    if requested is None:
        requested = list(range(
            int(options.first_seed),
            int(options.first_seed) + int(options.repeats),
        ))
    else:
        requested = [int(seed) for seed in requested]

    if len(requested) != int(options.repeats):
        raise SystemExit(
            f"seed-list contains {len(requested)} seeds but --repeats is "
            f"{int(options.repeats)}"
        )

    os.makedirs(options.out, exist_ok=True)

    existing_experiment = os.path.join(options.out, "experiment.json")
    if os.path.exists(existing_experiment):
        try:
            with open(existing_experiment, encoding="utf-8") as handle:
                existing_payload = json.load(handle)
            existing_physics = str(existing_payload.get("physics_mode", "standard"))
            if existing_physics != str(options.physics):
                raise SystemExit(
                    f"output folder already contains {existing_physics} physics; "
                    f"refusing to mix it with {options.physics}"
                )
            existing_group = existing_payload.get("group_size")
            if existing_group is not None:
                # Resume interrupted experiments with their original grouping
                # so completed and rerun seeds keep the same shared RNG stream.
                options.group = max(1, int(existing_group))
        except json.JSONDecodeError:
            raise SystemExit("existing experiment.json is unreadable")

    done = finished_seeds(options.out)

    # Build groups from the full requested experiment rather than merely the
    # unfinished seeds. If an interrupted group is partial, rerun that whole
    # group so its shared random stream and trajectory remain reproducible.
    group_size = min(int(options.group), len(requested))
    requested_groups = [
        requested[start:start + group_size]
        for start in range(0, len(requested), group_size)
    ]

    groups = []
    for group in requested_groups:
        completed = [seed for seed in group if seed in done]
        if len(completed) == len(group):
            continue
        groups.append(group)

    if not groups:
        print("every requested characterisation repeat is already complete")
        return

    if partner is None:
        required = minimum_box_size(molecule["positions"])
    else:
        required = minimum_pair_box_size(
            molecule, partner, options.start_gap
        )

    if float(options.box) + 1e-9 < required:
        partner_text = "" if partner is None else f" + {partner['id']}"
        raise SystemExit(
            f"{molecule['id']}{partner_text} needs a box of about {required:.1f} A "
            f"or larger; selected {float(options.box):g} A"
        )

    write_experiment(options.out, molecule, partner, options, requested)
    running.write_lock(options.out, sys.argv)

    try:
        remaining_repeats = sum(len(group) for group in groups)
        partner_text = "" if partner is None else f" + {partner['id']} {partner.get('formula', '')}"

        print(
            f"{molecule['id']} {molecule.get('formula', '')}{partner_text}: "
            f"{remaining_repeats} repeat"
            + ("s" if remaining_repeats != 1 else "")
            + f" to run at {options.temperature:g} K"
        )
        print(
            "physics: "
            + (
                f"high fidelity ({HIGH_FIDELITY_PHYSICS_MODEL})"
                if str(options.physics) == "high_fidelity"
                else (
                    f"optimised valence ({physics_metadata(options.physics)['physics_model']})"
                    if str(options.physics) == "optimised-valence"
                    else f"standard ({STANDARD_PHYSICS_MODEL})"
                )
            )
        )
        if partner is not None:
            print(
                f"collision: gap {options.start_gap:g} A, approach "
                f"{options.approach_factor:g} x thermal RMS, sampling "
                f"{options.sampling_mode}"
                + ("" if options.target_atom_a is None else f", target A #{int(options.target_atom_a)+1}")
            )
        if len(requested) == 1:
            print("fixed grouping: single-box characterisation")
        else:
            print(
                f"grouping: up to {group_size} boxes per group, "
                f"{len(groups)} sequential group"
                + ("s" if len(groups) != 1 else "")
            )
        print()

        for group_number, seeds in enumerate(groups, start=1):
            print(
                f"group {group_number}/{len(groups)}: "
                + ", ".join(str(seed) for seed in seeds)
            )

            (
                recorders,
                simulation,
                seconds,
                stopped,
                measures,
                diagnostics,
                collision_infos,
            ) = run_group(molecule, partner, seeds, options)

            each_seconds = seconds / max(len(seeds), 1)

            # The dynamics are finished, but the run is not complete until
            # every trajectory, entry and index update is safely on disk.
            # Keep the heartbeat alive during that finalisation phase so the
            # Lab does not present 100% simulation as a mysteriously hung job.
            write_heartbeat(
                options.out,
                f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0]),
                1,
                1,
                time.time(),
                len(seeds),
                phase="saving_results",
                results_done=0,
                results_total=len(seeds),
            )

            for box, seed in enumerate(seeds):
                recorder = recorders[box]
                path = os.path.join(options.out, f"run_s{int(seed):04d}.npz")
                recorder.save(path)

                entry = summarise(
                    recorder,
                    molecule,
                    partner,
                    seed,
                    options,
                    each_seconds,
                    stopped,
                    collision_measure=measures[box],
                    collision_info=collision_infos[box],
                    contact_diagnostic=diagnostics[box],
                    simulation=simulation,
                )
                write_entry(options.out, entry)

                # Rebuild after every repeat, not only after all eight.  A
                # crash during post-processing therefore leaves every result
                # already finished visible to the Lab and resumable.
                rebuild_index(options.out)
                write_heartbeat(
                    options.out,
                    f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0]),
                    1,
                    1,
                    time.time(),
                    len(seeds),
                    phase="saving_results",
                    results_done=box + 1,
                    results_total=len(seeds),
                )

                if partner is not None and entry.get("collision_diagnostics"):
                    diagnostic = entry["collision_diagnostics"]
                    start = entry.get("collision_start_safety", {})
                    print(
                        f"  seed {seed:<5d} {entry['characterisation_outcome']:<12} "
                        f"{entry['picoseconds']:g} ps  contact "
                        f"{diagnostic.get('longest_contact_fs', 0):.1f} fs  "
                        f"closest {diagnostic.get('closest_cross_distance_A', float('nan')):.3f} A  "
                        f"start-safe {'yes' if start.get('safe_initial_separation') else 'NO'}"
                    )
                else:
                    print(
                        f"  seed {seed:<5d} {entry['characterisation_outcome']:<12} "
                        f"{entry['picoseconds']:g} ps"
                    )

            print()

            # Hard rule: the loop cannot begin the next group until this
            # BatchedReactiveSimulation has finished and every Recorder file
            # for the current eight boxes has been written.

        print("characterisation complete")
    finally:
        # If finalisation failed halfway through, keep index.json consistent
        # with every entry that did make it to disk.
        try:
            rebuild_index(options.out)
        except Exception:
            pass
        clear_heartbeat(options.out)
        running.remove_lock(options.out)


if __name__ == "__main__":
    main()
