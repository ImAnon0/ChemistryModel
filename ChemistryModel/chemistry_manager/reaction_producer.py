"""Targeted full-ChemistryModel reaction teacher-data production."""

import collections
import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

import characterisation_runner as characterisation
import molecule_scanner
import reactive as reactive_parameters

from .discovery import event_log_path, read_events, route_full_cm_events_to_qm
from .state import CandidateState
from .trust import MoleculeTrust, trust_level, trusted_molecules


FORMAT_VERSION = 1

# Windows can briefly deny an atomic rename while Defender, an indexer, an
# editor, or another reader has the freshly-written file open. Keep atomic
# replacement semantics, but tolerate those transient sharing/access locks.
_ATOMIC_REPLACE_RETRY_DELAYS_S = (
    0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00,
)


def _replace_with_retry(source, destination):
    """Atomically replace destination, retrying transient Windows file locks."""
    source = Path(source)
    destination = Path(destination)

    for attempt in range(len(_ATOMIC_REPLACE_RETRY_DELAYS_S) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as problem:
            winerror = getattr(problem, "winerror", None)
            retryable = (
                isinstance(problem, PermissionError)
                or winerror in (5, 32, 33)
            )
            if (
                not retryable
                or attempt >= len(_ATOMIC_REPLACE_RETRY_DELAYS_S)
            ):
                raise
            time.sleep(_ATOMIC_REPLACE_RETRY_DELAYS_S[attempt])
ELEMENTAL_REACTANTS = ("atom:H", "atom:C", "atom:N", "atom:O")
SPEED_RANGES = {
    "balanced": {
        "low": (0.5, 0.9), "moderate": (1.2, 2.0), "high": (2.2, 3.2),
    },
    "gentle": {
        "low": (0.35, 0.7), "moderate": (0.8, 1.3), "high": (1.4, 2.0),
    },
    "reactive": {
        "low": (0.8, 1.3), "moderate": (1.5, 2.5), "high": (2.7, 4.0),
    },
}
TEMPERATURES_K = (250.0, 500.0, 800.0)

# Exploration is deliberately soft: nothing is ever removed from the pool.
NOVELTY_FLOOR = 0.02
NOVELTY_CANDIDATES_PER_SLOT = 36
REACTANT_REUSE_EXPONENT = 0.35
PAIR_REUSE_EXPONENT = 0.90
CONFIG_REUSE_EXPONENT = 1.20
APPROACH_SIMILARITY_SCALE = 0.45
IMPACT_SIMILARITY_SCALE = 0.22
TEMPERATURE_SIMILARITY_SCALE_K = 300.0
# Microreactors are the primary discovery environment. Pair probes remain a
# deliberately smaller channel for mechanistic/targeted exploration.
PAIR_FAMILY_WEIGHT = 1.0
MICROCELL_FAMILY_WEIGHT = 3.0

# Stage-2 outcome-aware planner. A fixed exploration share ignores historical
# success and uses novelty alone, preventing early lucky results from trapping
# discovery in one chemistry family.
OUTCOME_PLANNER_VERSION = 1
OUTCOME_EXPLORATION_PROBABILITY = 0.25
OUTCOME_PRIOR_STRENGTH = 4.0
OUTCOME_SIMILARITY_MINIMUM = 0.02
OUTCOME_MULTIPLIER_MIN = 0.55
OUTCOME_MULTIPLIER_MAX = 2.75
OUTCOME_REACTION_BONUS = 0.45
OUTCOME_NO_REACTION_PENALTY = 0.35
OUTCOME_NEW_PRODUCT_BONUS = 0.70
OUTCOME_QM_VALIDATED_BONUS = 1.10
OUTCOME_CM_VALIDATED_BONUS = 0.35
OUTCOME_REJECTED_PENALTY = 0.45
OUTCOME_UNSTABLE_PENALTY = 0.35

# Stage-3 automatic local refinement. Refinement is deliberately only a share
# of the microreactor budget so successful chemistry cannot consume discovery.
REFINEMENT_PLANNER_VERSION = 1
REFINEMENT_MICROREACTOR_PROBABILITY = 0.30
REFINEMENT_MAX_DEPTH = 2
REFINEMENT_MAX_CHILDREN_BY_PARENT_DEPTH = {0: 6, 1: 3}
REFINEMENT_INWARD_STEP = 0.20
REFINEMENT_RADIUS_FRACTION = 0.12
REFINEMENT_GAP_STEP_A = 0.15
REFINEMENT_COMPOSITION_CANDIDATES = 8
REFINEMENT_PARENT_QM_VALIDATED_BONUS = 4.0
REFINEMENT_PARENT_CM_VALIDATED_BONUS = 1.5
REFINEMENT_PARENT_NEW_PRODUCT_BONUS = 2.0
REFINEMENT_PARENT_REACTION_BONUS = 1.0
REFINEMENT_PARENT_REJECTED_PENALTY = 2.0

# Stage-4 broad-envelope / wild exploration. This is a small guaranteed
# exploration channel, not a replacement for outcome-aware planning. Each wild
# experiment broadens one dominant initial-condition dimension so its result
# remains interpretable.
WILD_PLANNER_VERSION = 1
WILD_EXPLORATION_PROBABILITY = 0.08
WILD_PAIR_DIMENSIONS = (
    "collision_speed",
    "impact_parameter",
    "start_gap",
    "temperature",
)
WILD_PAIR_APPROACH_FACTOR_RANGE = (4.0, 8.0)
WILD_PAIR_IMPACT_FRACTION_RANGE = (0.80, 1.80)
WILD_PAIR_START_GAP_RANGE_A = (1.50, 5.00)
WILD_TEMPERATURE_RANGE_K = (150.0, 1600.0)

WILD_MICROCELL_DIMENSIONS = (
    "temperature",
    "density",
    "inward_factor",
    "minimum_gap",
    "composition_imbalance",
    "object_count",
)
WILD_MICROCELL_INWARD_LOW_RANGE = (0.00, 0.25)
WILD_MICROCELL_INWARD_HIGH_RANGE = (1.45, 2.00)
WILD_MICROCELL_RADIUS_TIGHT_FACTOR_RANGE = (0.65, 0.82)
WILD_MICROCELL_RADIUS_LOOSE_FACTOR_RANGE = (1.20, 1.55)
WILD_MICROCELL_GAP_TIGHT_RANGE_A = (1.60, 1.68)
WILD_MICROCELL_GAP_LOOSE_RANGE_A = (2.50, 3.20)
WILD_MICROCELL_DOMINANT_FRACTION_RANGE = (0.65, 0.85)
WILD_MICROCELL_OBJECT_COUNTS = (7, 8)

MICROCELL_OBJECT_COUNT_WEIGHTS = {
    3: 0.12, 4: 0.18, 5: 0.22, 6: 0.20, 7: 0.16, 8: 0.12,
}
# Base radius is scaled with object count so larger microreactors do not
# become artificially denser solely because they contain more objects.
MICROCELL_CLUSTER_RADIUS_RANGE_A = (2.8, 4.8)
MICROCELL_MINIMUM_GAP_RANGE_A = (1.7, 2.4)
MICROCELL_INWARD_FACTOR_RANGE = (0.35, 1.35)
MICROCELL_COMPOSITION_REUSE_EXPONENT = 1.0
MICROCELL_CONFIG_REUSE_EXPONENT = 1.1
MICROREACTOR_ENVIRONMENT_VERSION = 1


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_hash(payload, length=16):
    encoded = json.dumps(
        _plain(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def physics_source_fingerprint(root=None, physics="optimised-valence"):
    """Fingerprint the selected class's effective local source closure."""
    from physics_provenance import physics_source_identity

    simulation_class = characterisation.simulation_class_for_physics(physics)
    return physics_source_identity(
        simulation_class, project_root=root
    )["sha256"]


def _load_reactant(reactant_id, molecule_root):
    return characterisation.load_reactant(reactant_id, str(molecule_root))


def _reactant_signature(reactant):
    digest = hashlib.sha256()
    for symbol in reactant["symbols"]:
        digest.update(str(symbol).encode("ascii"))
        digest.update(b"\0")
    digest.update(np.asarray(reactant["positions"], dtype=np.float32).tobytes())
    digest.update(
        np.asarray(reactant.get("bonds", []), dtype=np.int32).reshape(-1, 2).tobytes()
    )
    return digest.hexdigest()


def allowed_reactants(molecule_root="molecules", qm_root=None):
    molecules = trusted_molecules(molecule_root, qm_root=qm_root)
    return {
        "atoms": list(ELEMENTAL_REACTANTS),
        "molecules": [row["id"] for row in molecules],
        "trusted_molecule_records": molecules,
    }


def _radius(reactant):
    positions = np.asarray(reactant["positions"], dtype=np.float64)
    if len(positions) <= 1:
        return 0.0
    positions = positions - np.mean(positions, axis=0)
    return float(np.max(np.linalg.norm(positions, axis=1)))


def _contact_distance(first_symbols, second_symbols, target_atom=None):
    first_types = characterisation._types_from_symbols(first_symbols)
    second_types = characterisation._types_from_symbols(second_symbols)
    if target_atom is not None:
        first_types = first_types[[int(target_atom)]]
    values = reactive_parameters.CUTOFF_OUTER[
        np.ix_(second_types, first_types)
    ]
    return float(np.max(values))



def _pair_key(first_id, second_id):
    return tuple(sorted((str(first_id), str(second_id))))


def _history_records(output_root):
    """Read completed experiments with their persisted chemistry outcomes.

    Historical novelty code still sees the specification fields at top level,
    while ``experiment_outcome`` makes the result side available to the future
    adaptive planner without introducing a second history database.
    """
    root = Path(output_root)
    records = []
    if not root.is_dir():
        return records
    for path in root.rglob("experiments/EXP_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "complete":
            continue
        spec = record.get("specification")
        if isinstance(spec, dict):
            row = dict(spec)
            outcome = record.get("experiment_outcome")
            if isinstance(outcome, dict):
                row["experiment_outcome"] = outcome
            records.append(row)
    return records


def _history_with_current_product_trust(
    history, molecule_root="molecules", qm_root=None,
):
    """Return history with product trust refreshed from the current library.

    ``experiment_outcome`` is an immutable record of what was known when the
    experiment finished. Planning needs a live view as well: a product that
    was WAITING_QM yesterday may be QM_VALIDATED or REJECTED today.
    """
    trust_cache = {}
    refreshed = []

    for historical in history:
        row = dict(historical)
        outcome = historical.get("experiment_outcome")
        if not isinstance(outcome, dict):
            refreshed.append(row)
            continue

        outcome = dict(outcome)
        current_trust = {}
        product_ids = sorted({
            str(value)
            for value in outcome.get("product_species", [])
            if value
        })

        for molecule_id in product_ids:
            if molecule_id not in trust_cache:
                try:
                    molecule = molecule_scanner.molecule_store.load_molecule(
                        molecule_id, root=molecule_root
                    )
                    trust_cache[molecule_id] = trust_level(
                        molecule, qm_root=qm_root
                    ).value
                except Exception:
                    trust_cache[molecule_id] = "UNKNOWN"
            current_trust[molecule_id] = trust_cache[molecule_id]

        outcome["current_product_trust"] = current_trust
        row["experiment_outcome"] = outcome
        refreshed.append(row)

    return refreshed


def _composition_overlap(first_reactants, second_reactants):
    """Weighted-Jaccard overlap between two reactant multisets."""
    first = collections.Counter(str(value) for value in first_reactants)
    second = collections.Counter(str(value) for value in second_reactants)
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    intersection = sum(min(first[key], second[key]) for key in keys)
    union = sum(max(first[key], second[key]) for key in keys)
    return float(intersection / max(union, 1))


def _microreactor_outcome_similarity(candidate, previous):
    """Similarity used only to transfer outcome evidence between environments."""
    if previous.get("experiment_family") != "microcell":
        return 0.0
    if not isinstance(previous.get("experiment_outcome"), dict):
        return 0.0

    composition = _composition_overlap(
        candidate.get("reactants", []),
        previous.get("reactants", []),
    )
    if composition <= 0.0:
        return 0.0

    temperature_delta = abs(
        float(candidate.get("temperature_K", 0.0))
        - float(previous.get("temperature_K", 0.0))
    )
    inward_delta = abs(
        float(candidate.get("inward_factor", 0.0))
        - float(previous.get("inward_factor", 0.0))
    )
    gap_delta = abs(
        float(candidate.get("minimum_gap_A", 0.0))
        - float(previous.get("minimum_gap_A", 0.0))
    )

    candidate_density = max(
        float(candidate.get("atom_density_per_A3", 0.0)), 1e-12
    )
    previous_density = max(
        float(previous.get("atom_density_per_A3", candidate_density)), 1e-12
    )
    density_log_delta = abs(np.log(candidate_density / previous_density))

    condition_similarity = np.exp(
        -temperature_delta / TEMPERATURE_SIMILARITY_SCALE_K
        -inward_delta / 0.45
        -gap_delta / 0.40
        -density_log_delta / 0.70
    )

    # Composition is the stronger signal. Squaring it prevents a single shared
    # reactant in otherwise unrelated soups from transferring much evidence.
    return float((composition ** 2) * condition_similarity)


def _historical_outcome_delta(previous):
    """Signed usefulness of one completed microreactor experiment.

    Zero is the neutral prior. Positive outcomes earn future attention; dead,
    rejected or unstable chemistry reduces it. The result never decides
    validation and never excludes an experiment by itself.
    """
    outcome = previous.get("experiment_outcome")
    if not isinstance(outcome, dict):
        return None

    reacted = bool(
        outcome.get("reacted", False)
        or int(outcome.get("reaction_event_count", 0) or 0) > 0
    )
    delta = (
        OUTCOME_REACTION_BONUS
        if reacted
        else -OUTCOME_NO_REACTION_PENALTY
    )

    new_products = {
        str(value)
        for value in outcome.get("new_product_species", [])
        if value
    }
    if new_products:
        # Reward discovering at least one new topology. Multiple products in a
        # single run do not linearly multiply influence.
        delta += OUTCOME_NEW_PRODUCT_BONUS * (
            1.0 - np.exp(-float(len(new_products)))
        )

    product_ids = {
        str(value)
        for value in outcome.get("product_species", [])
        if value
    }
    current_trust = outcome.get("current_product_trust") or {}
    if product_ids:
        denominator = float(len(product_ids))
        qm_validated = sum(
            current_trust.get(product_id)
            == MoleculeTrust.QM_VALIDATED.value
            for product_id in product_ids
        )
        cm_validated = sum(
            current_trust.get(product_id)
            == MoleculeTrust.CM_VALIDATED.value
            for product_id in product_ids
        )
        rejected = sum(
            current_trust.get(product_id)
            == MoleculeTrust.REJECTED.value
            for product_id in product_ids
        )
        delta += OUTCOME_QM_VALIDATED_BONUS * qm_validated / denominator
        delta += OUTCOME_CM_VALIDATED_BONUS * cm_validated / denominator
        delta -= OUTCOME_REJECTED_PENALTY * rejected / denominator

    if outcome.get("stable") is False:
        delta -= OUTCOME_UNSTABLE_PENALTY

    return float(delta)


def microreactor_outcome_multiplier(candidate, history):
    """Confidence-shrunk outcome multiplier and diagnostics for one candidate."""
    weighted_delta = 0.0
    evidence = 0.0
    matched_records = 0

    for previous in history:
        similarity = _microreactor_outcome_similarity(candidate, previous)
        if similarity < OUTCOME_SIMILARITY_MINIMUM:
            continue

        delta = _historical_outcome_delta(previous)
        if delta is None:
            continue

        outcome = previous.get("experiment_outcome") or {}
        reliability = (
            0.5
            if outcome.get("postprocess_status") == "warning"
            else 1.0
        )
        weight = float(similarity) * reliability
        weighted_delta += weight * float(delta)
        evidence += weight
        matched_records += 1

    if evidence <= 0.0:
        return {
            "multiplier": 1.0,
            "evidence": 0.0,
            "matched_records": 0,
            "mean_outcome_delta": 0.0,
            "confidence": 0.0,
        }

    mean_delta = weighted_delta / evidence
    confidence = evidence / (evidence + OUTCOME_PRIOR_STRENGTH)
    multiplier = float(np.clip(
        1.0 + confidence * mean_delta,
        OUTCOME_MULTIPLIER_MIN,
        OUTCOME_MULTIPLIER_MAX,
    ))

    return {
        "multiplier": multiplier,
        "evidence": float(evidence),
        "matched_records": int(matched_records),
        "mean_outcome_delta": float(mean_delta),
        "confidence": float(confidence),
    }


def _refinement_depth(spec):
    refinement = spec.get("refinement")
    if isinstance(refinement, dict):
        return int(refinement.get("depth", 0) or 0)
    return int(spec.get("refinement_depth", 0) or 0)


def _refinement_parent_score(previous):
    """Score whether one completed microreactor deserves local follow-up."""
    if previous.get("experiment_family") != "microcell":
        return 0.0

    outcome = previous.get("experiment_outcome")
    if not isinstance(outcome, dict):
        return 0.0
    if outcome.get("stable") is False:
        return 0.0

    reacted = bool(
        outcome.get("reacted", False)
        or int(outcome.get("reaction_event_count", 0) or 0) > 0
    )
    if not reacted:
        return 0.0

    if _refinement_depth(previous) >= REFINEMENT_MAX_DEPTH:
        return 0.0

    score = REFINEMENT_PARENT_REACTION_BONUS
    new_products = {
        str(value)
        for value in outcome.get("new_product_species", [])
        if value
    }
    if new_products:
        score += REFINEMENT_PARENT_NEW_PRODUCT_BONUS

    current_trust = outcome.get("current_product_trust") or {}
    relevant_products = new_products or {
        str(value)
        for value in outcome.get("product_species", [])
        if value
    }
    if relevant_products:
        qm_validated = sum(
            current_trust.get(product_id)
            == MoleculeTrust.QM_VALIDATED.value
            for product_id in relevant_products
        )
        cm_validated = sum(
            current_trust.get(product_id)
            == MoleculeTrust.CM_VALIDATED.value
            for product_id in relevant_products
        )
        rejected = sum(
            current_trust.get(product_id)
            == MoleculeTrust.REJECTED.value
            for product_id in relevant_products
        )
        denominator = float(len(relevant_products))
        score += (
            REFINEMENT_PARENT_QM_VALIDATED_BONUS
            * qm_validated / denominator
        )
        score += (
            REFINEMENT_PARENT_CM_VALIDATED_BONUS
            * cm_validated / denominator
        )
        score -= (
            REFINEMENT_PARENT_REJECTED_PENALTY
            * rejected / denominator
        )

        # If every newly-discovered product was rejected, do not spend a
        # dedicated refinement budget mapping that parent.
        if (
            new_products
            and rejected == len(new_products)
            and qm_validated == 0
            and cm_validated == 0
        ):
            return 0.0

    if outcome.get("postprocess_status") == "warning":
        score *= 0.5

    return float(max(score, 0.0))


def _refinement_child_counts(history):
    counts = collections.Counter()
    for row in history:
        refinement = row.get("refinement")
        if not isinstance(refinement, dict):
            continue
        parent_id = refinement.get("parent_experiment_id")
        if parent_id:
            counts[str(parent_id)] += 1
    return counts


def _eligible_refinement_parents(history, allowed_ids=None):
    """Completed parents that still have depth/budget and usable reactants."""
    allowed = None if allowed_ids is None else {
        str(value) for value in allowed_ids
    }
    child_counts = _refinement_child_counts(history)
    rows = []

    for previous in history:
        parent_id = previous.get("id")
        if not parent_id:
            continue

        score = _refinement_parent_score(previous)
        if score <= 0.0:
            continue

        reactants = [str(value) for value in previous.get("reactants", [])]
        if len(reactants) < 3 or len(reactants) > 8:
            continue
        if allowed is not None and any(
            reactant_id not in allowed for reactant_id in reactants
        ):
            continue

        depth = _refinement_depth(previous)
        maximum_children = int(
            REFINEMENT_MAX_CHILDREN_BY_PARENT_DEPTH.get(depth, 0)
        )
        used_children = int(child_counts[str(parent_id)])
        if used_children >= maximum_children:
            continue

        rows.append({
            "spec": previous,
            "score": float(score),
            "children_used": used_children,
            "children_limit": maximum_children,
        })

    return rows


def _refinement_mutation_key(refinement):
    if not isinstance(refinement, dict):
        return None
    mutation = refinement.get("mutation")
    if not isinstance(mutation, dict):
        return None
    return json.dumps(
        _plain(mutation), sort_keys=True, separators=(",", ":")
    )


def _existing_refinement_mutations(history, parent_id):
    found = set()
    for row in history:
        refinement = row.get("refinement")
        if not isinstance(refinement, dict):
            continue
        if str(refinement.get("parent_experiment_id")) != str(parent_id):
            continue
        key = _refinement_mutation_key(refinement)
        if key:
            found.add(key)
    return found


def _refinement_candidate_from_parent(
    parent, number, reactant_ids, temperature_K, cluster_radius_A,
    minimum_gap_A, inward_factor, generator, molecule_root, profile,
    reactant_cache, mutation,
):
    """Rebuild all derived microreactor fields for one one-variable child."""

    def cached(reactant_id):
        if reactant_id not in reactant_cache:
            reactant_cache[reactant_id] = _load_reactant(
                reactant_id, molecule_root
            )
        return reactant_cache[reactant_id]

    reactant_ids = [str(value) for value in reactant_ids]
    reactants = [cached(reactant_id) for reactant_id in reactant_ids]
    object_count = len(reactants)
    if object_count < 3 or object_count > 8:
        return None

    cluster_radius_A = float(cluster_radius_A)
    minimum_gap_A = float(minimum_gap_A)
    inward_factor = float(inward_factor)
    temperature_K = float(temperature_K)

    max_radius = max((_radius(row) for row in reactants), default=0.0)
    total_atoms = int(sum(len(row["symbols"]) for row in reactants))
    composition = {
        reactant_id: int(count)
        for reactant_id, count in sorted(
            collections.Counter(reactant_ids).items()
        )
    }
    cluster_volume = (4.0 / 3.0) * np.pi * cluster_radius_A ** 3
    count_scale = (float(object_count) / 3.0) ** (1.0 / 3.0)
    base_radius = cluster_radius_A / count_scale

    parent_refinement = parent.get("refinement")
    parent_depth = _refinement_depth(parent)
    root_id = (
        str(parent_refinement.get("root_experiment_id"))
        if isinstance(parent_refinement, dict)
        and parent_refinement.get("root_experiment_id")
        else str(parent["id"])
    )
    depth = parent_depth + 1
    refinement = {
        "version": REFINEMENT_PLANNER_VERSION,
        "parent_experiment_id": str(parent["id"]),
        "root_experiment_id": root_id,
        "depth": int(depth),
        "mutation": _plain(mutation),
    }

    return {
        "number": int(number),
        "experiment_family": "microcell",
        "environment_mode": "microreactor",
        "environment_version": MICROREACTOR_ENVIRONMENT_VERSION,
        "category": "microreactor_refinement",
        "reactants": reactant_ids,
        "composition": composition,
        "reactant_formulas": [row.get("formula") for row in reactants],
        "object_count": int(object_count),
        "total_atoms": int(total_atoms),
        "simulation_seed": int(generator.integers(0, 2**31 - 1)),
        "profile": profile,
        "temperature_K": temperature_K,
        "base_cluster_radius_A": float(base_radius),
        "cluster_radius_A": cluster_radius_A,
        "minimum_gap_A": minimum_gap_A,
        "inward_factor": inward_factor,
        "cluster_volume_A3": float(cluster_volume),
        "object_density_per_A3": float(object_count / cluster_volume),
        "atom_density_per_A3": float(total_atoms / cluster_volume),
        "box_A": float(
            max(14.0, 2.0 * (cluster_radius_A + max_radius + 2.5))
        ),
        "refinement_depth": int(depth),
        "refinement": refinement,
    }


def _build_refinement_candidates(
    parent, number, generator, pool, molecule_root, profile, reactant_cache,
    history,
):
    """Generate interpretable one-variable descendants of one parent."""
    reactant_ids = [str(value) for value in parent.get("reactants", [])]
    if len(reactant_ids) < 3 or len(reactant_ids) > 8:
        return []

    temperature = float(parent["temperature_K"])
    cluster_radius = float(parent["cluster_radius_A"])
    minimum_gap = float(parent["minimum_gap_A"])
    inward_factor = float(parent["inward_factor"])
    mutations = []

    # Temperature: immediate lower/higher level from the existing production
    # temperature set. Each child changes only temperature.
    ordered_temperatures = sorted(float(value) for value in TEMPERATURES_K)
    lower = [value for value in ordered_temperatures if value < temperature]
    higher = [value for value in ordered_temperatures if value > temperature]
    if lower:
        value = lower[-1]
        mutations.append((
            list(reactant_ids), value, cluster_radius, minimum_gap,
            inward_factor,
            {"type": "temperature_K", "from": temperature, "to": value},
        ))
    if higher:
        value = higher[0]
        mutations.append((
            list(reactant_ids), value, cluster_radius, minimum_gap,
            inward_factor,
            {"type": "temperature_K", "from": temperature, "to": value},
        ))

    # Inward bias.
    for sign in (-1.0, 1.0):
        value = float(np.clip(
            inward_factor + sign * REFINEMENT_INWARD_STEP,
            MICROCELL_INWARD_FACTOR_RANGE[0],
            MICROCELL_INWARD_FACTOR_RANGE[1],
        ))
        if abs(value - inward_factor) > 1e-9:
            mutations.append((
                list(reactant_ids), temperature, cluster_radius, minimum_gap,
                value,
                {"type": "inward_factor", "from": inward_factor, "to": value},
            ))

    # Cluster radius changes density while composition remains identical.
    for factor in (
        1.0 - REFINEMENT_RADIUS_FRACTION,
        1.0 + REFINEMENT_RADIUS_FRACTION,
    ):
        value = max(1.0, cluster_radius * factor)
        mutations.append((
            list(reactant_ids), temperature, value, minimum_gap,
            inward_factor,
            {
                "type": "cluster_radius_A",
                "from": cluster_radius,
                "to": value,
            },
        ))

    # Initial minimum separation.
    for sign in (-1.0, 1.0):
        value = float(np.clip(
            minimum_gap + sign * REFINEMENT_GAP_STEP_A,
            MICROCELL_MINIMUM_GAP_RANGE_A[0],
            MICROCELL_MINIMUM_GAP_RANGE_A[1],
        ))
        if abs(value - minimum_gap) > 1e-9:
            mutations.append((
                list(reactant_ids), temperature, cluster_radius, value,
                inward_factor,
                {"type": "minimum_gap_A", "from": minimum_gap, "to": value},
            ))

    # Composition edits stay local: remove one object, add one currently
    # allowed reactant, or replace one object. Limit generated alternatives so
    # large trusted libraries do not explode the candidate set.
    allowed = list(pool["atoms"]) + list(pool["molecules"])
    allowed = sorted({str(value) for value in allowed})

    if len(reactant_ids) > 3:
        for index, removed in enumerate(reactant_ids):
            child = list(reactant_ids)
            child.pop(index)
            mutations.append((
                child, temperature, cluster_radius, minimum_gap, inward_factor,
                {
                    "type": "composition_remove",
                    "removed": str(removed),
                    "index": int(index),
                },
            ))

    if len(reactant_ids) < 8 and allowed:
        sampled = list(allowed)
        generator.shuffle(sampled)
        for added in sampled[:REFINEMENT_COMPOSITION_CANDIDATES]:
            child = list(reactant_ids) + [str(added)]
            mutations.append((
                child, temperature, cluster_radius, minimum_gap, inward_factor,
                {"type": "composition_add", "added": str(added)},
            ))

    if allowed:
        replacement_pool = list(allowed)
        generator.shuffle(replacement_pool)
        emitted = 0
        for index, old_id in enumerate(reactant_ids):
            for new_id in replacement_pool:
                if str(new_id) == str(old_id):
                    continue
                child = list(reactant_ids)
                child[index] = str(new_id)
                mutations.append((
                    child, temperature, cluster_radius, minimum_gap,
                    inward_factor,
                    {
                        "type": "composition_replace",
                        "index": int(index),
                        "removed": str(old_id),
                        "added": str(new_id),
                    },
                ))
                emitted += 1
                break
            if emitted >= REFINEMENT_COMPOSITION_CANDIDATES:
                break

    used = _existing_refinement_mutations(history, parent["id"])
    candidates = []
    seen = set()

    for (
        child_reactants, child_temperature, child_radius, child_gap,
        child_inward, mutation,
    ) in mutations:
        mutation_key = json.dumps(
            _plain(mutation), sort_keys=True, separators=(",", ":")
        )
        if mutation_key in used or mutation_key in seen:
            continue
        seen.add(mutation_key)

        candidate = _refinement_candidate_from_parent(
            parent,
            number,
            child_reactants,
            child_temperature,
            child_radius,
            child_gap,
            child_inward,
            generator,
            molecule_root,
            profile,
            reactant_cache,
            mutation,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _choose_refinement_parent(generator, eligible):
    if not eligible:
        return None
    weights = [
        max(float(row["score"]), NOVELTY_FLOOR)
        * (
            1.0
            - float(row["children_used"])
            / max(float(row["children_limit"]), 1.0)
        )
        for row in eligible
    ]
    return _weighted_choice(generator, eligible, weights)


def _usage_counts(history):
    reactants = collections.Counter()
    pairs = collections.Counter()
    for spec in history:
        first = spec.get("reactant_a")
        second = spec.get("reactant_b")
        if first is None or second is None:
            continue
        reactants[str(first)] += 1
        reactants[str(second)] += 1
        pairs[_pair_key(first, second)] += 1
    return reactants, pairs


def _configuration_similarity(candidate, previous):
    """Soft similarity in collision-configuration space, bounded above by 1."""
    if _pair_key(candidate["reactant_a"], candidate["reactant_b"]) != _pair_key(
        previous.get("reactant_a"), previous.get("reactant_b")
    ):
        return 0.0

    target_similarity = 1.0 if (
        candidate.get("impact_target") == previous.get("impact_target")
        and candidate.get("target_atom_a") == previous.get("target_atom_a")
    ) else 0.35

    collision_similarity = (
        1.0
        if candidate.get("collision_class") == previous.get("collision_class")
        else 0.45
    )
    speed_similarity = (
        1.0
        if candidate.get("speed_class") == previous.get("speed_class")
        else 0.60
    )

    approach_delta = abs(
        float(candidate.get("approach_factor", 0.0))
        - float(previous.get("approach_factor", 0.0))
    )
    impact_delta = abs(
        float(candidate.get("impact_fraction", 0.0))
        - float(previous.get("impact_fraction", 0.0))
    )
    temperature_delta = abs(
        float(candidate.get("temperature_K", 0.0))
        - float(previous.get("temperature_K", 0.0))
    )

    continuous = (
        np.exp(-approach_delta / APPROACH_SIMILARITY_SCALE)
        * np.exp(-impact_delta / IMPACT_SIMILARITY_SCALE)
        * np.exp(-temperature_delta / TEMPERATURE_SIMILARITY_SCALE_K)
    )
    return float(
        target_similarity
        * collision_similarity
        * speed_similarity
        * continuous
    )


def novelty_weight(candidate, history):
    """Return a strictly-positive exploration weight for one proposed experiment."""
    reactant_uses, pair_uses = _usage_counts(history)
    first = str(candidate["reactant_a"])
    second = str(candidate["reactant_b"])
    pair = _pair_key(first, second)

    reactant_factor = (
        (1.0 + reactant_uses[first]) ** (-REACTANT_REUSE_EXPONENT)
        * (1.0 + reactant_uses[second]) ** (-REACTANT_REUSE_EXPONENT)
    )
    pair_factor = (1.0 + pair_uses[pair]) ** (-PAIR_REUSE_EXPONENT)

    local_reuse = sum(
        _configuration_similarity(candidate, previous)
        for previous in history
    )
    config_factor = (1.0 + local_reuse) ** (-CONFIG_REUSE_EXPONENT)

    return float(
        NOVELTY_FLOOR + reactant_factor * pair_factor * config_factor
    )


def _weighted_choice(generator, candidates, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if len(candidates) == 0:
        raise ValueError("cannot choose from an empty candidate set")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("novelty weights must be finite and strictly positive")
    probabilities = weights / np.sum(weights)
    return candidates[int(generator.choice(len(candidates), p=probabilities))]


def _choose_pair(category, generator, atom_ids, molecule_ids):
    atom = lambda: str(generator.choice(atom_ids))
    molecule = lambda: str(generator.choice(molecule_ids))
    if category == "atom_atom" or not molecule_ids:
        return atom(), atom(), "atom_atom"
    if category == "atom_molecule":
        return atom(), molecule(), category
    if category == "molecule_atom":
        return molecule(), atom(), category
    return molecule(), molecule(), "molecule_molecule"


def _build_candidate_spec(
    number, category, collision_class, speed_class, generator, pool,
    molecule_root, profile, reactant_cache,
):
    first_id, second_id, category = _choose_pair(
        category, generator, pool["atoms"], pool["molecules"]
    )

    def cached(reactant_id):
        if reactant_id not in reactant_cache:
            reactant_cache[reactant_id] = _load_reactant(
                reactant_id, molecule_root
            )
        return reactant_cache[reactant_id]

    first = cached(first_id)
    second = cached(second_id)
    simulation_seed = int(generator.integers(0, 2**31 - 1))

    use_atom_target = len(first["symbols"]) == 1 or generator.random() < 0.7
    target_atom = (
        int(generator.integers(0, len(first["symbols"])))
        if use_atom_target else None
    )
    sampling_mode = (
        "targeted_random" if target_atom is not None else "random_orientation"
    )
    impact_target = (
        "com"
        if target_atom is None
        else str(first["symbols"][target_atom]).lower()
    )
    impact_target = {
        "c": "carbon", "o": "oxygen", "h": "hydrogen", "n": "com",
    }.get(impact_target, impact_target)

    contact = _contact_distance(
        first["symbols"], second["symbols"], target_atom=target_atom
    )
    first_radius = 0.0 if target_atom is not None else _radius(first)
    scale = max(0.5, first_radius + _radius(second) + contact)

    if collision_class == "direct":
        impact_fraction = 0.0
    elif collision_class == "glancing":
        impact_fraction = float(generator.uniform(0.25, 0.75))
    else:
        impact_fraction = float(generator.uniform(1.10, 1.45))
    impact_parameter = impact_fraction * scale

    low, high = SPEED_RANGES[profile][speed_class]
    approach_factor = float(generator.uniform(low, high))
    start_gap = float(generator.uniform(2.0, 3.2))
    temperature = float(generator.choice(TEMPERATURES_K))
    minimum_box = characterisation.minimum_pair_box_size(
        first, second, start_gap
    )
    box = float(max(12.0, minimum_box + 2.0 * impact_parameter + 2.0))

    return {
        "number": int(number),
        "category": category,
        "reactant_a": first_id,
        "reactant_b": second_id,
        "reactant_a_formula": first.get("formula"),
        "reactant_b_formula": second.get("formula"),
        "reactant_a_sha256": _reactant_signature(first),
        "reactant_b_sha256": _reactant_signature(second),
        "simulation_seed": simulation_seed,
        "profile": profile,
        "collision_class": collision_class,
        "speed_class": speed_class,
        "approach_factor": approach_factor,
        "impact_parameter_A": float(impact_parameter),
        "impact_scale_A": float(scale),
        "impact_fraction": float(impact_fraction),
        "start_gap_A": start_gap,
        "temperature_K": temperature,
        "box_A": box,
        "sampling_mode": sampling_mode,
        "impact_target": impact_target,
        "target_atom_a": target_atom,
    }


def _build_wild_pair_candidate(
    number, category, collision_class, speed_class, generator, pool,
    molecule_root, profile, reactant_cache, wild_dimension=None,
):
    """Broaden exactly one pair-probe dimension beyond the normal envelope."""
    candidate = _build_candidate_spec(
        number, category, collision_class, speed_class, generator, pool,
        molecule_root, profile, reactant_cache,
    )
    dimension = str(
        wild_dimension
        if wild_dimension is not None
        else generator.choice(WILD_PAIR_DIMENSIONS)
    )
    if dimension not in WILD_PAIR_DIMENSIONS:
        raise ValueError(f"unknown wild pair dimension: {dimension}")

    before = None
    after = None
    if dimension == "collision_speed":
        before = float(candidate["approach_factor"])
        after = float(generator.uniform(*WILD_PAIR_APPROACH_FACTOR_RANGE))
        candidate["approach_factor"] = after
        candidate["speed_class"] = "wild"

    elif dimension == "impact_parameter":
        before = float(candidate["impact_fraction"])
        after = float(generator.uniform(*WILD_PAIR_IMPACT_FRACTION_RANGE))
        candidate["impact_fraction"] = after
        candidate["impact_parameter_A"] = float(
            after * float(candidate["impact_scale_A"])
        )
        candidate["collision_class"] = "wild_impact"

    elif dimension == "start_gap":
        before = float(candidate["start_gap_A"])
        after = float(generator.uniform(*WILD_PAIR_START_GAP_RANGE_A))
        candidate["start_gap_A"] = after

    elif dimension == "temperature":
        before = float(candidate["temperature_K"])
        after = float(generator.uniform(*WILD_TEMPERATURE_RANGE_K))
        candidate["temperature_K"] = after

    if dimension in ("impact_parameter", "start_gap"):
        first = reactant_cache[candidate["reactant_a"]]
        second = reactant_cache[candidate["reactant_b"]]
        minimum_box = characterisation.minimum_pair_box_size(
            first, second, float(candidate["start_gap_A"])
        )
        candidate["box_A"] = float(max(
            12.0,
            minimum_box
            + 2.0 * float(candidate["impact_parameter_A"])
            + 2.0,
        ))

    candidate["wild_exploration"] = {
        "version": WILD_PLANNER_VERSION,
        "dimension": dimension,
        "from": before,
        "to": after,
    }
    return candidate



def _composition_key(reactant_ids):
    return tuple(sorted(collections.Counter(str(v) for v in reactant_ids).items()))


def _microcell_similarity(candidate, previous):
    if previous.get("experiment_family") != "microcell":
        return 0.0
    if _composition_key(candidate["reactants"]) != _composition_key(previous.get("reactants", [])):
        return 0.0
    d = (
        abs(float(candidate["cluster_radius_A"]) - float(previous.get("cluster_radius_A", candidate["cluster_radius_A"]))) / 1.0
        + abs(float(candidate["inward_factor"]) - float(previous.get("inward_factor", candidate["inward_factor"]))) / 0.4
        + abs(float(candidate["minimum_gap_A"]) - float(previous.get("minimum_gap_A", candidate["minimum_gap_A"]))) / 0.35
        + abs(float(candidate["temperature_K"]) - float(previous.get("temperature_K", candidate["temperature_K"]))) / TEMPERATURE_SIMILARITY_SCALE_K
    )
    return float(np.exp(-d))


def microcell_novelty_weight(candidate, history):
    prior = [row for row in history if row.get("experiment_family") == "microcell"]
    key = _composition_key(candidate["reactants"])
    uses = sum(1 for row in prior if _composition_key(row.get("reactants", [])) == key)
    local = sum(_microcell_similarity(candidate, row) for row in prior)
    return float(
        NOVELTY_FLOOR
        + (1.0 + uses) ** (-MICROCELL_COMPOSITION_REUSE_EXPONENT)
        * (1.0 + local) ** (-MICROCELL_CONFIG_REUSE_EXPONENT)
    )


def _build_microcell_candidate(number, generator, pool, molecule_root, profile, reactant_cache):
    """Build one small reactive environment rather than one prescribed collision."""
    ids = list(pool["atoms"]) + list(pool["molecules"])
    if not ids:
        raise ValueError("microreactor has no allowed reactants")

    counts = list(MICROCELL_OBJECT_COUNT_WEIGHTS)
    weights = np.asarray(
        [MICROCELL_OBJECT_COUNT_WEIGHTS[x] for x in counts], dtype=float
    )
    weights /= weights.sum()
    object_count = int(generator.choice(counts, p=weights))
    reactant_ids = [str(generator.choice(ids)) for _ in range(object_count)]

    def cached(reactant_id):
        if reactant_id not in reactant_cache:
            reactant_cache[reactant_id] = _load_reactant(
                reactant_id, molecule_root
            )
        return reactant_cache[reactant_id]

    reactants = [cached(rid) for rid in reactant_ids]

    # Keep roughly comparable object density as object count grows.  This is
    # an initial-condition sampling rule only; it does not alter MD physics.
    base_radius = float(
        generator.uniform(*MICROCELL_CLUSTER_RADIUS_RANGE_A)
    )
    count_scale = (float(object_count) / 3.0) ** (1.0 / 3.0)
    cluster_radius = base_radius * count_scale

    minimum_gap = float(generator.uniform(*MICROCELL_MINIMUM_GAP_RANGE_A))
    inward_factor = float(generator.uniform(*MICROCELL_INWARD_FACTOR_RANGE))
    temperature = float(generator.choice(TEMPERATURES_K))
    max_radius = max((_radius(r) for r in reactants), default=0.0)
    total_atoms = int(sum(len(r["symbols"]) for r in reactants))
    composition = {
        reactant_id: int(count)
        for reactant_id, count in sorted(
            collections.Counter(reactant_ids).items()
        )
    }
    cluster_volume = (4.0 / 3.0) * np.pi * cluster_radius ** 3

    return {
        "number": int(number),
        # Preserve the historical family name so old summaries/scanners remain
        # compatible; environment_mode identifies the richer experiment.
        "experiment_family": "microcell",
        "environment_mode": "microreactor",
        "environment_version": MICROREACTOR_ENVIRONMENT_VERSION,
        "category": "microreactor",
        "reactants": reactant_ids,
        "composition": composition,
        "reactant_formulas": [r.get("formula") for r in reactants],
        "object_count": object_count,
        "total_atoms": total_atoms,
        "simulation_seed": int(generator.integers(0, 2**31 - 1)),
        "profile": profile,
        "temperature_K": temperature,
        "base_cluster_radius_A": base_radius,
        "cluster_radius_A": cluster_radius,
        "minimum_gap_A": minimum_gap,
        "inward_factor": inward_factor,
        "cluster_volume_A3": float(cluster_volume),
        "object_density_per_A3": float(object_count / cluster_volume),
        "atom_density_per_A3": float(total_atoms / cluster_volume),
        "box_A": float(
            max(14.0, 2.0 * (cluster_radius + max_radius + 2.5))
        ),
    }


def _rebuild_microcell_candidate(
    candidate, reactant_ids, molecule_root, reactant_cache, *,
    base_cluster_radius_A=None, minimum_gap_A=None,
    inward_factor=None, temperature_K=None,
):
    """Recompute all derived microreactor fields after one planner mutation."""

    def cached(reactant_id):
        if reactant_id not in reactant_cache:
            reactant_cache[reactant_id] = _load_reactant(
                reactant_id, molecule_root
            )
        return reactant_cache[reactant_id]

    reactant_ids = [str(value) for value in reactant_ids]
    if len(reactant_ids) < 3 or len(reactant_ids) > 8:
        raise ValueError("wild microreactor object count must remain within 3..8")

    reactants = [cached(reactant_id) for reactant_id in reactant_ids]
    object_count = len(reactants)
    total_atoms = int(sum(len(row["symbols"]) for row in reactants))

    base_radius = float(
        candidate["base_cluster_radius_A"]
        if base_cluster_radius_A is None
        else base_cluster_radius_A
    )
    count_scale = (float(object_count) / 3.0) ** (1.0 / 3.0)
    cluster_radius = float(base_radius * count_scale)
    minimum_gap = float(
        candidate["minimum_gap_A"]
        if minimum_gap_A is None else minimum_gap_A
    )
    inward = float(
        candidate["inward_factor"]
        if inward_factor is None else inward_factor
    )
    temperature = float(
        candidate["temperature_K"]
        if temperature_K is None else temperature_K
    )

    composition = {
        reactant_id: int(count)
        for reactant_id, count in sorted(
            collections.Counter(reactant_ids).items()
        )
    }
    max_radius = max((_radius(row) for row in reactants), default=0.0)
    cluster_volume = (4.0 / 3.0) * np.pi * cluster_radius ** 3

    rebuilt = dict(candidate)
    rebuilt.update({
        "reactants": reactant_ids,
        "composition": composition,
        "reactant_formulas": [row.get("formula") for row in reactants],
        "object_count": int(object_count),
        "total_atoms": int(total_atoms),
        "temperature_K": temperature,
        "base_cluster_radius_A": base_radius,
        "cluster_radius_A": cluster_radius,
        "minimum_gap_A": minimum_gap,
        "inward_factor": inward,
        "cluster_volume_A3": float(cluster_volume),
        "object_density_per_A3": float(object_count / cluster_volume),
        "atom_density_per_A3": float(total_atoms / cluster_volume),
        "box_A": float(
            max(14.0, 2.0 * (cluster_radius + max_radius + 2.5))
        ),
    })
    return rebuilt


def _build_wild_microcell_candidate(
    number, generator, pool, molecule_root, profile, reactant_cache,
    wild_dimension=None,
):
    """Broaden one microreactor dimension while preserving placement safety."""
    candidate = _build_microcell_candidate(
        number, generator, pool, molecule_root, profile, reactant_cache
    )
    dimension = str(
        wild_dimension
        if wild_dimension is not None
        else generator.choice(WILD_MICROCELL_DIMENSIONS)
    )
    if dimension not in WILD_MICROCELL_DIMENSIONS:
        raise ValueError(f"unknown wild microreactor dimension: {dimension}")

    reactant_ids = list(candidate["reactants"])
    before = None
    after = None
    metadata = {}

    if dimension == "temperature":
        before = float(candidate["temperature_K"])
        after = float(generator.uniform(*WILD_TEMPERATURE_RANGE_K))
        candidate = _rebuild_microcell_candidate(
            candidate, reactant_ids, molecule_root, reactant_cache,
            temperature_K=after,
        )

    elif dimension == "density":
        before = float(candidate["base_cluster_radius_A"])
        if generator.random() < 0.5:
            factor = float(generator.uniform(
                *WILD_MICROCELL_RADIUS_TIGHT_FACTOR_RANGE
            ))
            metadata["direction"] = "denser"
        else:
            factor = float(generator.uniform(
                *WILD_MICROCELL_RADIUS_LOOSE_FACTOR_RANGE
            ))
            metadata["direction"] = "looser"
        after = float(before * factor)
        metadata["radius_factor"] = factor
        candidate = _rebuild_microcell_candidate(
            candidate, reactant_ids, molecule_root, reactant_cache,
            base_cluster_radius_A=after,
        )

    elif dimension == "inward_factor":
        before = float(candidate["inward_factor"])
        if generator.random() < 0.5:
            after = float(generator.uniform(
                *WILD_MICROCELL_INWARD_LOW_RANGE
            ))
            metadata["direction"] = "weak"
        else:
            after = float(generator.uniform(
                *WILD_MICROCELL_INWARD_HIGH_RANGE
            ))
            metadata["direction"] = "strong"
        candidate = _rebuild_microcell_candidate(
            candidate, reactant_ids, molecule_root, reactant_cache,
            inward_factor=after,
        )

    elif dimension == "minimum_gap":
        before = float(candidate["minimum_gap_A"])
        if generator.random() < 0.5:
            after = float(generator.uniform(
                *WILD_MICROCELL_GAP_TIGHT_RANGE_A
            ))
            metadata["direction"] = "tighter"
        else:
            after = float(generator.uniform(
                *WILD_MICROCELL_GAP_LOOSE_RANGE_A
            ))
            metadata["direction"] = "looser"
        candidate = _rebuild_microcell_candidate(
            candidate, reactant_ids, molecule_root, reactant_cache,
            minimum_gap_A=after,
        )

    elif dimension == "composition_imbalance":
        before = list(reactant_ids)
        allowed = sorted({
            str(value)
            for value in list(pool["atoms"]) + list(pool["molecules"])
        })
        dominant = str(generator.choice(allowed))
        fraction = float(generator.uniform(
            *WILD_MICROCELL_DOMINANT_FRACTION_RANGE
        ))
        dominant_count = max(
            2, min(len(reactant_ids), int(np.ceil(
                fraction * len(reactant_ids)
            )))
        )
        alternatives = [value for value in allowed if value != dominant]
        child = [dominant] * dominant_count
        while len(child) < len(reactant_ids):
            source = alternatives or [dominant]
            child.append(str(generator.choice(source)))
        generator.shuffle(child)
        after = list(child)
        metadata.update({
            "dominant_reactant": dominant,
            "dominant_fraction_target": fraction,
        })
        candidate = _rebuild_microcell_candidate(
            candidate, child, molecule_root, reactant_cache,
        )

    elif dimension == "object_count":
        before = int(candidate["object_count"])
        target_count = int(generator.choice(WILD_MICROCELL_OBJECT_COUNTS))
        allowed = list(pool["atoms"]) + list(pool["molecules"])
        child = [
            str(generator.choice(allowed)) for _ in range(target_count)
        ]
        after = target_count
        candidate = _rebuild_microcell_candidate(
            candidate, child, molecule_root, reactant_cache,
        )

    candidate["category"] = "microreactor_wild"
    candidate["wild_exploration"] = {
        "version": WILD_PLANNER_VERSION,
        "dimension": dimension,
        "from": _plain(before),
        "to": _plain(after),
        **metadata,
    }
    return candidate


def _choose_experiment_family(generator):
    weights = np.asarray([PAIR_FAMILY_WEIGHT, MICROCELL_FAMILY_WEIGHT], dtype=float)
    weights /= weights.sum()
    return str(generator.choice(["pair", "microcell"], p=weights))



def generate_experiment_specs(
    count, master_seed, molecule_root="molecules", qm_root=None,
    profile="balanced", history=None, invocation_id=None,
    pair_number_offset=0, wild_probability=WILD_EXPLORATION_PROBABILITY,
):
    count = int(count)
    if count < 1:
        raise ValueError("experiment count must be at least one")
    if profile not in SPEED_RANGES:
        raise ValueError(f"unknown reaction-production profile: {profile}")
    wild_probability = float(wild_probability)
    if not 0.0 <= wild_probability <= 1.0:
        raise ValueError("wild exploration probability must be between 0 and 1")

    pool = allowed_reactants(molecule_root, qm_root=qm_root)
    generator = np.random.default_rng(int(master_seed))
    categories = ["atom_atom", "atom_molecule", "molecule_atom", "molecule_molecule"] if pool["molecules"] else ["atom_atom"]
    collisions = ["direct", "glancing", "near_miss"]
    speeds = ["low", "moderate", "high"]
    historical = _history_with_current_product_trust(
        history or [], molecule_root=molecule_root, qm_root=qm_root
    )
    working_history = [dict(row) for row in historical]
    historical_outcome_count = sum(
        isinstance(row.get("experiment_outcome"), dict)
        for row in historical
    )
    cache = {}
    specs = []
    pair_number = int(pair_number_offset)
    allowed_ids = set(pool["atoms"]) | set(pool["molecules"])
    eligible_refinement = _eligible_refinement_parents(
        historical, allowed_ids=allowed_ids
    )

    for number in range(count):
        family = _choose_experiment_family(generator)
        wild_exploration = bool(
            wild_probability > 0.0
            and generator.random() < wild_probability
        )
        candidates, weights = [], []

        if family == "pair":
            category = categories[pair_number % len(categories)]
            pair_number += 1

            if wild_exploration:
                for candidate_number in range(NOVELTY_CANDIDATES_PER_SLOT):
                    dimension = WILD_PAIR_DIMENSIONS[
                        candidate_number % len(WILD_PAIR_DIMENSIONS)
                    ]
                    candidate = _build_wild_pair_candidate(
                        number,
                        category,
                        collisions[candidate_number % len(collisions)],
                        speeds[
                            (candidate_number // len(collisions))
                            % len(speeds)
                        ],
                        generator,
                        pool,
                        molecule_root,
                        profile,
                        cache,
                        wild_dimension=dimension,
                    )
                    candidate["experiment_family"] = "pair"
                    novelty = novelty_weight(
                        candidate, working_history
                    )
                    candidate["planner"] = {
                        "version": OUTCOME_PLANNER_VERSION,
                        "wild_version": WILD_PLANNER_VERSION,
                        "mode": "wild_exploration",
                        "wild_family": "pair",
                        "wild_dimension": dimension,
                        "wild_probability": float(wild_probability),
                        "novelty_weight": float(novelty),
                        "outcome_multiplier": 1.0,
                        "final_weight": float(novelty),
                    }
                    candidates.append(candidate)
                    weights.append(novelty)
            else:
                for candidate_number in range(NOVELTY_CANDIDATES_PER_SLOT):
                    candidate = _build_candidate_spec(
                        number,
                        category,
                        collisions[candidate_number % len(collisions)],
                        speeds[
                            (candidate_number // len(collisions))
                            % len(speeds)
                        ],
                        generator, pool, molecule_root, profile, cache,
                    )
                    candidate["experiment_family"] = "pair"
                    novelty = novelty_weight(
                        candidate, working_history
                    )
                    candidate["planner"] = {
                        "version": OUTCOME_PLANNER_VERSION,
                        "mode": "pair_novelty",
                        "novelty_weight": float(novelty),
                        "outcome_multiplier": 1.0,
                        "final_weight": float(novelty),
                    }
                    candidates.append(candidate)
                    weights.append(novelty)
        else:
            if wild_exploration:
                for candidate_number in range(NOVELTY_CANDIDATES_PER_SLOT):
                    dimension = WILD_MICROCELL_DIMENSIONS[
                        candidate_number % len(WILD_MICROCELL_DIMENSIONS)
                    ]
                    candidate = _build_wild_microcell_candidate(
                        number,
                        generator,
                        pool,
                        molecule_root,
                        profile,
                        cache,
                        wild_dimension=dimension,
                    )
                    novelty = microcell_novelty_weight(
                        candidate, working_history
                    )
                    learned = microreactor_outcome_multiplier(
                        candidate, working_history
                    )
                    candidate["planner"] = {
                        "version": OUTCOME_PLANNER_VERSION,
                        "wild_version": WILD_PLANNER_VERSION,
                        "mode": "wild_exploration",
                        "wild_family": "microcell",
                        "wild_dimension": dimension,
                        "wild_probability": float(wild_probability),
                        "novelty_weight": float(novelty),
                        # Wild allocation is an exploration guarantee, so
                        # historical outcomes are recorded diagnostically but
                        # cannot suppress the experiment back into the known
                        # region.
                        "outcome_multiplier": 1.0,
                        "learned_outcome_multiplier": float(
                            learned["multiplier"]
                        ),
                        "outcome_evidence": float(learned["evidence"]),
                        "matched_history_records": int(
                            learned["matched_records"]
                        ),
                        "final_weight": float(novelty),
                    }
                    candidates.append(candidate)
                    weights.append(novelty)
            else:
                use_refinement = bool(
                    eligible_refinement
                    and generator.random()
                    < REFINEMENT_MICROREACTOR_PROBABILITY
                )

                if use_refinement:
                    parent_row = _choose_refinement_parent(
                        generator, eligible_refinement
                    )
                    parent = parent_row["spec"] if parent_row else None
                    refinement_candidates = (
                        _build_refinement_candidates(
                            parent,
                            number,
                            generator,
                            pool,
                            molecule_root,
                            profile,
                            cache,
                            working_history,
                        )
                        if parent is not None else []
                    )

                    for candidate in refinement_candidates[
                        :NOVELTY_CANDIDATES_PER_SLOT
                    ]:
                        novelty = microcell_novelty_weight(
                            candidate, working_history
                        )
                        candidate["planner"] = {
                            "version": OUTCOME_PLANNER_VERSION,
                            "refinement_version": REFINEMENT_PLANNER_VERSION,
                            "mode": "microreactor_refinement",
                            "novelty_weight": float(novelty),
                            "outcome_multiplier": 1.0,
                            "final_weight": float(novelty),
                            "parent_experiment_id": str(parent["id"]),
                            "parent_score": float(parent_row["score"]),
                            "parent_children_used": int(
                                parent_row["children_used"]
                            ),
                            "parent_children_limit": int(
                                parent_row["children_limit"]
                            ),
                            "refinement_probability": float(
                                REFINEMENT_MICROREACTOR_PROBABILITY
                            ),
                        }
                        candidates.append(candidate)
                        weights.append(novelty)

                    # If a parent has exhausted its distinct one-variable
                    # mutations, fall back to normal discovery rather than
                    # wasting the slot.
                    if not candidates:
                        use_refinement = False

                if not use_refinement:
                    # A fixed fraction is pure exploration. Before any stage-1
                    # outcome exists, all microreactors naturally remain
                    # novelty-only.
                    exploration = (
                        historical_outcome_count == 0
                        or generator.random()
                        < OUTCOME_EXPLORATION_PROBABILITY
                    )
                    planner_mode = (
                        "microreactor_exploration"
                        if exploration
                        else "microreactor_outcome_aware"
                    )

                    for _ in range(NOVELTY_CANDIDATES_PER_SLOT):
                        candidate = _build_microcell_candidate(
                            number, generator, pool, molecule_root, profile, cache
                        )
                        novelty = microcell_novelty_weight(
                            candidate, working_history
                        )
                        learned = microreactor_outcome_multiplier(
                            candidate, working_history
                        )
                        outcome_multiplier = (
                            1.0 if exploration else learned["multiplier"]
                        )
                        final_weight = float(novelty * outcome_multiplier)

                        candidate["planner"] = {
                            "version": OUTCOME_PLANNER_VERSION,
                            "mode": planner_mode,
                            "exploration_probability": (
                                OUTCOME_EXPLORATION_PROBABILITY
                            ),
                            "novelty_weight": float(novelty),
                            "outcome_multiplier": float(outcome_multiplier),
                            "learned_outcome_multiplier": float(
                                learned["multiplier"]
                            ),
                            "outcome_evidence": float(learned["evidence"]),
                            "matched_history_records": int(
                                learned["matched_records"]
                            ),
                            "mean_outcome_delta": float(
                                learned["mean_outcome_delta"]
                            ),
                            "outcome_confidence": float(learned["confidence"]),
                            "final_weight": final_weight,
                            "historical_outcome_records": int(
                                historical_outcome_count
                            ),
                        }
                        candidates.append(candidate)
                        weights.append(final_weight)

        spec = dict(_weighted_choice(generator, candidates, weights))
        spec["master_seed"] = int(master_seed)
        identity = dict(spec)
        if invocation_id is not None:
            identity["invocation_id"] = str(invocation_id)
        spec["id"] = "EXP_" + _canonical_hash(identity)
        specs.append(spec)
        working_history.append(spec)

        refinement = spec.get("refinement")
        if isinstance(refinement, dict):
            parent_id = str(refinement.get("parent_experiment_id", ""))
            for row in eligible_refinement:
                if str(row["spec"].get("id")) == parent_id:
                    row["children_used"] = int(row["children_used"]) + 1
                    break
            eligible_refinement = [
                row for row in eligible_refinement
                if int(row["children_used"]) < int(row["children_limit"])
            ]

    return specs, pool

def _instantaneous_edges(symbols, positions, box_size, threshold=0.35):
    count = len(symbols)
    first, second = np.triu_indices(count, 1)
    if len(first) == 0:
        return tuple()
    positions = np.asarray(positions, dtype=np.float64)
    offsets = positions[second] - positions[first]
    offsets -= float(box_size) * np.round(offsets / float(box_size))
    distances = np.linalg.norm(offsets, axis=1)
    types = characterisation._types_from_symbols(symbols)
    inner = reactive_parameters.CUTOFF_INNER[types[first], types[second]]
    outer = reactive_parameters.CUTOFF_OUTER[types[first], types[second]]
    taper = reactive_parameters.smooth_cutoff(distances, inner, outer)
    return tuple(
        (int(a), int(b))
        for a, b, keep in zip(first, second, taper > float(threshold))
        if keep
    )


class TeacherFrameCollector:
    def __init__(self, ordinary_interval_fs=10.0, event_window_fs=5.0):
        self.ordinary_interval_fs = float(ordinary_interval_fs)
        self.event_window_fs = float(event_window_fs)
        self.frames = []

    def __call__(self, snapshot):
        snapshot = dict(snapshot)
        info = snapshot.get("collision_info") or {}
        positions = snapshot["positions_A"]
        symbols = snapshot["symbols"]
        if info.get("experiment_family") == "microcell":
            minimum = float("inf")
            maximum = 0.0
            p = np.asarray(positions, dtype=float)
            types = characterisation._types_from_symbols(symbols)
            ranges = info.get("object_ranges", [])
            for i, left in enumerate(ranges):
                ls = slice(int(left["start"]), int(left["stop"]))
                for right in ranges[i+1:]:
                    rs = slice(int(right["start"]), int(right["stop"]))
                    offsets = p[rs, None, :] - p[None, ls, :]
                    offsets -= float(snapshot["box_A"]) * np.round(offsets / float(snapshot["box_A"]))
                    distances = np.linalg.norm(offsets, axis=2)
                    if distances.size:
                        minimum = min(minimum, float(np.min(distances)))
                    taper = reactive_parameters.smooth_cutoff(
                        distances,
                        reactive_parameters.CUTOFF_INNER[np.ix_(types[rs], types[ls])],
                        reactive_parameters.CUTOFF_OUTER[np.ix_(types[rs], types[ls])],
                    )
                    if np.size(taper):
                        maximum = max(maximum, float(np.max(taper)))
            snapshot["cross_min_distance_A"] = minimum
            snapshot["cross_max_taper"] = maximum
        else:
            first_count = int(info.get("first_count", len(symbols)))
            distances, taper = characterisation._cross_pair_arrays(
                positions, symbols, first_count, snapshot["box_A"]
            )
            snapshot["cross_min_distance_A"] = float(np.min(distances)) if distances.size else float("inf")
            snapshot["cross_max_taper"] = float(np.max(taper)) if taper.size else 0.0
        snapshot["edges"] = _instantaneous_edges(
            symbols, positions, snapshot["box_A"]
        )
        self.frames.append(snapshot)

    def selected(self):
        if not self.frames:
            return []
        event_times = []
        previous = self.frames[0]["edges"]
        for frame in self.frames[1:]:
            if frame["edges"] != previous:
                event_times.append(float(frame["time_fs"]))
            previous = frame["edges"]

        selected = []
        last_ordinary = -float("inf")
        for index, frame in enumerate(self.frames):
            reasons = []
            time_fs = float(frame["time_fs"])
            if index == 0:
                reasons.append("pre_contact")
            if index == len(self.frames) - 1:
                reasons.append("final")
            if frame["cross_max_taper"] > 0.0:
                reasons.append("close_contact")
            if any(abs(time_fs - event) <= self.event_window_fs for event in event_times):
                reasons.append("bond_change_window")
            if time_fs - last_ordinary >= self.ordinary_interval_fs - 1e-9:
                reasons.append("ordinary")
                last_ordinary = time_fs
            if reasons:
                row = dict(frame)
                row["selection_reasons"] = tuple(sorted(set(reasons)))
                selected.append(row)
        return selected


def _write_teacher_shard(path, frames, *, experiment_id="", production_id=""):
    if not frames:
        raise ValueError("teacher experiment produced no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(
        temporary,
        format_version=np.asarray([FORMAT_VERSION], dtype=np.int32),
        experiment_id=np.asarray([str(experiment_id)], dtype="U40"),
        production_id=np.asarray([str(production_id)], dtype="U40"),
        elements=np.asarray(frames[0]["symbols"], dtype="U2"),
        positions_A=np.asarray([row["positions_A"] for row in frames], dtype=np.float32),
        forces_eV_per_A=np.asarray(
            [row["forces_eV_per_A"] for row in frames], dtype=np.float32
        ),
        potential_energy_eV=np.asarray(
            [row["potential_energy_eV"] for row in frames], dtype=np.float64
        ),
        kinetic_energy_eV=np.asarray(
            [row["kinetic_energy_eV"] for row in frames], dtype=np.float64
        ),
        total_md_energy_eV=np.asarray(
            [row["potential_energy_eV"] + row["kinetic_energy_eV"] for row in frames],
            dtype=np.float64,
        ),
        time_fs=np.asarray([row["time_fs"] for row in frames], dtype=np.float64),
        temperature_K=np.asarray(
            [row["temperature_K"] for row in frames], dtype=np.float64
        ),
        box_A=np.asarray([row["box_A"] for row in frames], dtype=np.float64),
        cross_min_distance_A=np.asarray(
            [row["cross_min_distance_A"] for row in frames], dtype=np.float64
        ),
        cross_max_taper=np.asarray(
            [row["cross_max_taper"] for row in frames], dtype=np.float64
        ),
        selection_reasons=np.asarray(
            ["+".join(row["selection_reasons"]) for row in frames], dtype="U96"
        ),
    )
    _replace_with_retry(temporary, path)


def _options_for(spec, production_root, molecule_root, duration_ps, device,
                 capture_every, diagnostic_sample_fs, physics):
    return SimpleNamespace(
        physics=str(physics), test="with_partner",
        sampling_mode=spec["sampling_mode"],
        target_atom_a=spec["target_atom_a"],
        temperature=float(spec["temperature_K"]),
        picoseconds=float(duration_ps), box=float(spec["box_A"]),
        time_step=0.25, friction=0.01,
        capture_every=int(capture_every), group=1,
        diagnostic_sample_fs=float(diagnostic_sample_fs),
        diagnostic_fine_window_fs=float(duration_ps) * 1000.0,
        diagnostic_coarse_sample_fs=max(10.0, float(diagnostic_sample_fs)),
        max_frames=max(100, int(duration_ps * 1000.0 / 0.25) + 4),
        stride=2, device=device, out=str(production_root),
        library=str(molecule_root),
        start_gap=float(spec["start_gap_A"]),
        approach_factor=float(spec["approach_factor"]),
        impact_parameter=float(spec["impact_parameter_A"]),
        impact_target=spec["impact_target"],
    )



def _microcell_options_for(spec, production_root, molecule_root, duration_ps, device,
                           capture_every, diagnostic_sample_fs, physics):
    return SimpleNamespace(
        physics=str(physics), test="microcell",
        temperature=float(spec["temperature_K"]),
        picoseconds=float(duration_ps), box=float(spec["box_A"]),
        time_step=0.25, friction=0.01, capture_every=int(capture_every), group=1,
        diagnostic_sample_fs=float(diagnostic_sample_fs),
        diagnostic_fine_window_fs=float(duration_ps) * 1000.0,
        diagnostic_coarse_sample_fs=max(10.0, float(diagnostic_sample_fs)),
        max_frames=max(100, int(duration_ps * 1000.0 / 0.25) + 4),
        stride=2, device=device, out=str(production_root), library=str(molecule_root),
        cluster_radius_A=float(spec["cluster_radius_A"]),
        minimum_gap_A=float(spec["minimum_gap_A"]),
        inward_factor=float(spec["inward_factor"]),
    )



def _rebuild_recording_index(records_dir, records):
    payload = []
    for record in sorted(records, key=lambda row: row["experiment_id"]):
        entry = dict(record["run_entry"])
        entry["file"] = record["recording"]
        payload.append(entry)
    _atomic_json(Path(records_dir) / "index.json", payload)



def _local_day_string(now=None):
    current = now or datetime.now().astimezone()
    return current.astimezone().date().isoformat()


def _fresh_master_seed():
    return int(secrets.randbits(63))


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_plain(payload), sort_keys=True) + "\n")


def _daily_experiment_records(day_root):
    records = []
    directory = Path(day_root) / "experiments"
    if not directory.is_dir():
        return records
    for path in directory.glob("EXP_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(record)
    return records


def _write_day_summary(day_root):
    day_root = Path(day_root)
    records = _daily_experiment_records(day_root)
    invocations = []
    invocations_dir = day_root / "invocations"
    if invocations_dir.is_dir():
        for path in invocations_dir.glob("INV_*.json"):
            try:
                invocations.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue

    completed = [row for row in records if row.get("status") == "complete"]
    failed = [row for row in records if row.get("status") == "failed"]
    specs = [
        row.get("specification", {})
        for row in records
        if isinstance(row.get("specification"), dict)
    ]

    reactants = set()
    pairs = set()
    categories = collections.Counter()
    families = collections.Counter()
    microcell_compositions = set()
    collisions = collections.Counter()
    speeds = collections.Counter()
    temperatures = collections.Counter()

    for spec in specs:
        first, second = spec.get("reactant_a"), spec.get("reactant_b")
        if first is not None:
            reactants.add(str(first))
        if second is not None:
            reactants.add(str(second))
        if first is not None and second is not None:
            pairs.add(_pair_key(first, second))
        family = str(spec.get("experiment_family", "pair"))
        families[family] += 1
        if family == "microcell":
            microcell_compositions.add(
                _composition_key(spec.get("reactants", []))
            )
        categories[str(spec.get("category", "unknown"))] += 1
        collisions[str(spec.get("collision_class", "unknown"))] += 1
        speeds[str(spec.get("speed_class", "unknown"))] += 1
        if spec.get("temperature_K") is not None:
            temperatures[str(float(spec["temperature_K"]))] += 1

    payload = {
        "format_version": FORMAT_VERSION,
        "date": day_root.name,
        "invocations": len(invocations),
        "attempted": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "teacher_frames": int(sum(
            int(row.get("teacher_frames", 0)) for row in completed
        )),
        "unique_reactants": len(reactants),
        "unique_pairs": len(pairs),
        "unique_microcell_compositions": len(microcell_compositions),
        "experiment_families": dict(sorted(families.items())),
        "categories": dict(sorted(categories.items())),
        "collision_classes": dict(sorted(collisions.items())),
        "speed_classes": dict(sorted(speeds.items())),
        "temperatures_K": dict(sorted(temperatures.items())),
    }
    _atomic_json(day_root / "day_summary.json", payload)
    return payload


def production_summary(
    output_root="teacher_data", date=None, molecule_root="molecules", qm_root=None,
):
    """Summarise one daily teacher-data folder, including chemistry outcomes."""
    day = str(date or _local_day_string())
    root = Path(output_root) / day
    if not root.is_dir():
        return {
            "date": day,
            "root": str(root),
            "invocations": 0,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "teacher_frames": 0,
            "unique_reactants": 0,
            "unique_pairs": 0,
            "unique_microcell_compositions": 0,
            "experiment_families": {},
            "categories": {},
            "collision_classes": {},
            "speed_classes": {},
            "temperatures_K": {},
            "outcomes_by_family": {},
            "reaction_events_by_family": {},
            "unique_product_species": [],
            "untrusted_product_species": [],
            "reaction_events": 0,
        }

    summary = _write_day_summary(root)
    summary["root"] = str(root)

    records = _daily_experiment_records(root)
    complete_records = [
        row for row in records if row.get("status") == "complete"
    ]

    outcomes_by_family = {}
    experiment_family_by_id = {}

    for record in complete_records:
        experiment_id = str(record.get("experiment_id", ""))
        spec = record.get("specification") or {}
        family = str(
            record.get("experiment_family")
            or spec.get("experiment_family")
            or "pair"
        )
        if experiment_id:
            experiment_family_by_id[experiment_id] = family

        outcome = str(
            (record.get("run_entry") or {}).get(
                "characterisation_outcome", "unknown"
            )
        )
        family_counts = outcomes_by_family.setdefault(family, {})
        family_counts[outcome] = family_counts.get(outcome, 0) + 1

    events, event_errors = read_events(event_log_path(molecule_root))
    reaction_events_by_family = {}
    product_ids = set()

    day_recordings = (root / "recordings").resolve()

    for event in events:
        recording = event.get("recording")
        if not recording:
            continue
        try:
            recording_path = Path(recording).resolve()
        except (OSError, TypeError):
            continue

        if recording_path.parent != day_recordings:
            continue

        experiment_id = recording_path.stem
        family = experiment_family_by_id.get(experiment_id, "unknown")
        reaction_events_by_family[family] = (
            reaction_events_by_family.get(family, 0) + 1
        )

        for product in event.get("products", []):
            product_id = product.get("id")
            if product_id:
                product_ids.add(str(product_id))

    trusted_ids = {
        str(row["id"])
        for row in trusted_molecules(molecule_root, qm_root=qm_root)
        if row.get("id")
    }
    untrusted_products = sorted(
        product_id for product_id in product_ids
        if product_id not in trusted_ids
    )

    summary.update({
        "outcomes_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(outcomes_by_family.items())
        },
        "reaction_events_by_family": dict(
            sorted(reaction_events_by_family.items())
        ),
        "reaction_events": int(sum(reaction_events_by_family.values())),
        "unique_product_species": sorted(product_ids),
        "untrusted_product_species": untrusted_products,
        "event_log_errors": event_errors,
    })
    return summary

def _events_for_experiment(events, experiment_id):
    found = []
    for event in events:
        recording = event.get("recording")
        if not recording:
            continue
        try:
            if Path(recording).stem == str(experiment_id):
                found.append(event)
        except (OSError, TypeError):
            continue
    return found


def _product_report(
    events, molecule_root, store, *, qm_root=None, species_before=None,
):
    species_before = set(species_before or ())
    waiting_ids = set(store.product_ids(CandidateState.WAITING_QM))
    rows = {}

    for event in events:
        for product in event.get("products") or []:
            if not isinstance(product, dict) or not product.get("id"):
                continue
            molecule_id = str(product["id"])
            row = rows.setdefault(molecule_id, {
                "id": molecule_id,
                "formula": product.get("formula"),
                "new_this_experiment": molecule_id not in species_before,
                "trust": "UNKNOWN",
                "queue": "not queued",
            })

            try:
                molecule = molecule_scanner.molecule_store.load_molecule(
                    molecule_id, root=molecule_root
                )
                row["formula"] = molecule.get("formula") or row["formula"]
                row["trust"] = trust_level(
                    molecule, qm_root=qm_root
                ).value
            except Exception:
                row["trust"] = "UNKNOWN"

            if row["trust"] == MoleculeTrust.QM_VALIDATED.value:
                row["queue"] = CandidateState.QM_VALIDATED.value
            elif row["trust"] == MoleculeTrust.REJECTED.value:
                row["queue"] = CandidateState.QM_REJECTED.value
            elif row["trust"] == MoleculeTrust.CM_VALIDATED.value:
                row["queue"] = "trusted"
            elif molecule_id in waiting_ids:
                row["queue"] = CandidateState.WAITING_QM.value

    return [rows[key] for key in sorted(rows)]



def run_production(
    store, *, count=12, duration_ps=0.25, master_seed=None,
    profile="balanced", output_root="teacher_data",
    molecule_root="molecules", qm_root=None, device=None,
    ordinary_interval_fs=10.0, event_window_fs=5.0,
    diagnostic_sample_fs=1.0, capture_every=4,
    physics="optimised-valence", progress=None, result_observer=None,
    invocation_id=None, manager_run_id=None,
    planner_pair_offset=0,
    wild_probability=WILD_EXPLORATION_PROBABILITY,
):
    """Attempt exactly `count` fresh experiments for this invocation.

    An invocation is a receipt, not a resumable order. Each command creates a
    new invocation inside the local-date folder. Individual experiment failures
    are recorded and the remaining slots are still attempted.
    """
    count = int(count)
    if count < 1:
        raise ValueError("experiment count must be at least one")
    if float(duration_ps) <= 0:
        raise ValueError("duration must be greater than zero")
    if float(ordinary_interval_fs) <= 0:
        raise ValueError("ordinary frame interval must be greater than zero")
    if float(event_window_fs) < 0:
        raise ValueError("event window cannot be negative")
    if float(diagnostic_sample_fs) <= 0:
        raise ValueError("diagnostic sample interval must be greater than zero")
    if int(capture_every) < 1:
        raise ValueError("capture interval must be at least one step")
    wild_probability = float(wild_probability)
    if not 0.0 <= wild_probability <= 1.0:
        raise ValueError("wild exploration probability must be between 0 and 1")
    if str(physics) not in (
        "standard", "high_fidelity", "optimised-valence", "unified-radial"
    ):
        raise ValueError(f"unsupported characterisation physics: {physics}")

    used_master_seed = (
        _fresh_master_seed() if master_seed is None else int(master_seed)
    )
    day = _local_day_string()
    root = Path(output_root) / day
    records_dir = root / "recordings"
    experiments_dir = root / "experiments"
    invocations_dir = root / "invocations"
    root.mkdir(parents=True, exist_ok=True)

    revision = git_revision()
    teacher_physics = characterisation.physics_metadata(physics)
    source_hash = teacher_physics["physics_source_sha256"]

    if device is None:
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(device)

    if invocation_id is None:
        invocation_nonce = secrets.token_hex(6)
        invocation_id = (
            "INV_"
            + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            + "_"
            + invocation_nonce
        )
    else:
        invocation_id = str(invocation_id)

    history = _history_records(output_root)
    specs, pool = generate_experiment_specs(
        count, used_master_seed, molecule_root=molecule_root,
        qm_root=qm_root, profile=profile, history=history,
        invocation_id=invocation_id,
        pair_number_offset=planner_pair_offset,
        wild_probability=wild_probability,
    )

    trusted_pool = []
    for row in pool["trusted_molecule_records"]:
        reactant = _load_reactant(row["id"], molecule_root)
        trusted_pool.append({
            "id": row["id"],
            "formula": row.get("formula"),
            "trust_status": row.get("trust_status"),
            "reactant_sha256": _reactant_signature(reactant),
        })

    invocation = {
        "format_version": FORMAT_VERSION,
        "id": invocation_id,
        "manager_run_id": manager_run_id,
        "date": day,
        "status": "running",
        "requested": count,
        "master_seed": used_master_seed,
        "master_seed_source": (
            "random" if master_seed is None else "explicit"
        ),
        "duration_ps": float(duration_ps),
        "profile": profile,
        "physics": physics,
        "physics_model": teacher_physics["physics_model"],
        "physics_model_id": teacher_physics["physics_model_id"],
        "physics_model_revision": teacher_physics["physics_model_revision"],
        "physics_parameters": teacher_physics["physics_parameters"],
        "physics_source_algorithm": teacher_physics["physics_source_algorithm"],
        "physics_source_sha256": source_hash,
        "physics_source_files": teacher_physics["physics_source_files"],
        "physics_parameter_sha256": teacher_physics.get(
            "physics_parameter_sha256"
        ),
        "physics_enabled_terms": teacher_physics.get("physics_enabled_terms"),
        "physics_capacity_solver": teacher_physics.get(
            "physics_capacity_solver"
        ),
        "physics_geometry_convention": teacher_physics.get(
            "physics_geometry_convention"
        ),
        "device": resolved_device,
        "ordinary_interval_fs": float(ordinary_interval_fs),
        "event_window_fs": float(event_window_fs),
        "diagnostic_sample_fs": float(diagnostic_sample_fs),
        "capture_every": int(capture_every),
        "chemistrymodel_git_revision": revision,
        "planner": {
            "version": OUTCOME_PLANNER_VERSION,
            "microreactor_primary_weight": float(MICROCELL_FAMILY_WEIGHT),
            "pair_probe_weight": float(PAIR_FAMILY_WEIGHT),
            "microreactor_exploration_probability": float(
                OUTCOME_EXPLORATION_PROBABILITY
            ),
            "outcome_prior_strength": float(OUTCOME_PRIOR_STRENGTH),
            "outcome_multiplier_range": [
                float(OUTCOME_MULTIPLIER_MIN),
                float(OUTCOME_MULTIPLIER_MAX),
            ],
            "wild_exploration": {
                "version": WILD_PLANNER_VERSION,
                "probability": float(wild_probability),
                "pair_dimensions": list(WILD_PAIR_DIMENSIONS),
                "microreactor_dimensions": list(WILD_MICROCELL_DIMENSIONS),
            },
            "historical_completed_records": int(len(history)),
            "historical_outcome_records": int(sum(
                isinstance(row.get("experiment_outcome"), dict)
                for row in history
            )),
            "refinement": {
                "version": REFINEMENT_PLANNER_VERSION,
                "microreactor_probability": float(
                    REFINEMENT_MICROREACTOR_PROBABILITY
                ),
                "maximum_depth": int(REFINEMENT_MAX_DEPTH),
                "maximum_children_by_parent_depth": {
                    str(key): int(value)
                    for key, value in
                    REFINEMENT_MAX_CHILDREN_BY_PARENT_DEPTH.items()
                },
                "eligible_parents": int(len(
                    _eligible_refinement_parents(
                        _history_with_current_product_trust(
                            history,
                            molecule_root=molecule_root,
                            qm_root=qm_root,
                        ),
                        allowed_ids=(
                            set(pool["atoms"]) | set(pool["molecules"])
                        ),
                    )
                )),
            },
        },
        "allowed_atoms": pool["atoms"],
        "trusted_molecules": trusted_pool,
        "specifications": specs,
        "started_unix": datetime.now().timestamp(),
    }
    _atomic_json(invocations_dir / f"{invocation_id}.json", invocation)

    completed_now = 0
    failed_now = 0
    frames_written = 0
    queued_now = 0
    duplicate_queue_events = 0
    invocation_event_ids = set()
    invocation_product_ids = set()
    invocation_new_product_ids = set()
    completed_records = [
        row for row in _daily_experiment_records(root)
        if row.get("status") == "complete"
    ]

    for number, spec in enumerate(specs, start=1):
        if progress:
            progress_spec = spec
            if spec.get("experiment_family") == "microcell":
                formulas = spec.get("reactant_formulas") or spec.get("reactants", [])
                progress_spec = {
                    **spec,
                    "reactant_a": f"{int(spec.get('object_count', len(formulas)))} objects",
                    "reactant_b": " + ".join(str(value) for value in formulas),
                    "collision_class": "clustered",
                    "speed_class": f"inward {float(spec.get('inward_factor', 0.0)):.2f}x",
                }
            progress(number, len(specs), progress_spec)

        family = str(spec.get("experiment_family", "pair"))
        species_before = {
            str(row["id"])
            for row in molecule_scanner.molecule_store.list_molecules(molecule_root)
            if row.get("id")
        }
        collector = TeacherFrameCollector(
            ordinary_interval_fs=ordinary_interval_fs,
            event_window_fs=event_window_fs,
        )
        if family == "microcell":
            reactants = [_load_reactant(rid, molecule_root) for rid in spec["reactants"]]
            first = second = None
            options = _microcell_options_for(
                spec, root, molecule_root, duration_ps, resolved_device,
                capture_every, diagnostic_sample_fs, physics,
            )
        else:
            reactants = None
            first = _load_reactant(spec["reactant_a"], molecule_root)
            second = _load_reactant(spec["reactant_b"], molecule_root)
            options = _options_for(
                spec, root, molecule_root, duration_ps, resolved_device,
                capture_every, diagnostic_sample_fs, physics,
            )

        try:
            try:
                if family == "microcell":
                    (
                        recorders, simulation, wall_seconds, stopped_early,
                        collision_measures, diagnostics, infos,
                    ) = characterisation.run_microcell_group(
                        reactants, [spec["simulation_seed"]], options,
                        frame_observer=collector,
                    )
                else:
                    (
                        recorders, simulation, wall_seconds, stopped_early,
                        collision_measures, diagnostics, infos,
                    ) = characterisation.run_group(
                        first, second, [spec["simulation_seed"]], options,
                        frame_observer=collector,
                    )
            finally:
                characterisation.clear_heartbeat(options.out)

            recorder = recorders[0]
            recording_name = f"{spec['id']}.npz"
            records_dir.mkdir(parents=True, exist_ok=True)
            recorder.save(records_dir / recording_name)

            if family == "microcell":
                entry = characterisation.summarise_microcell(
                    recorder, reactants, spec["simulation_seed"], options,
                    wall_seconds, stopped_early,
                    velocity_measure=collision_measures[0],
                    cell_info=infos[0], simulation=simulation,
                )
            else:
                entry = characterisation.summarise(
                    recorder, first, second, spec["simulation_seed"], options,
                    wall_seconds, stopped_early,
                    collision_measures[0], infos[0], diagnostics[0],
                    simulation=simulation,
                )
            entry.update({
                "file": recording_name,
                "experiment_id": spec["id"],
                "production_id": invocation_id,
                "invocation_id": invocation_id,
                "manager_run_id": manager_run_id,
                "mixture": "reaction teacher production",
                "experiment_family": family,
                "chemistrymodel_git_revision": revision,
                "physics_source_sha256": source_hash,
            })

            frames = collector.selected()
            shard_relative = str(Path("shards") / f"{spec['id']}.npz")
            _write_teacher_shard(
                root / shard_relative, frames,
                experiment_id=spec["id"], production_id=invocation_id,
            )
            record = {
                "format_version": FORMAT_VERSION,
                "status": "complete",
                "production_id": invocation_id,
                "invocation_id": invocation_id,
                "manager_run_id": manager_run_id,
                "experiment_id": spec["id"],
                "experiment_family": family,
                "specification": spec,
                "actual_collision": infos[0],
                "velocity": collision_measures[0],
                "run_entry": entry,
                "recording": recording_name,
                "shard": shard_relative,
                "teacher_frames": len(frames),
                "chemistrymodel_git_revision": revision,
                "physics_model_id": getattr(
                    simulation, "model_id", teacher_physics["physics_model_id"]
                ),
                "physics_source_sha256": source_hash,
                "physics_source_algorithm": teacher_physics[
                    "physics_source_algorithm"
                ],
                "physics_source_files": teacher_physics["physics_source_files"],
                "physics_parameter_sha256": teacher_physics.get(
                    "physics_parameter_sha256"
                ),
                "physics_model": getattr(
                    simulation, "physics_model_name", entry["physics_model"]
                ),
                "physics_model_revision": getattr(
                    simulation, "physics_model_revision", None
                ),
                "device": str(getattr(simulation, "device", resolved_device)),
            }
            _atomic_json(experiments_dir / f"{spec['id']}.json", record)
            completed_records.append(_plain(record))
            _rebuild_recording_index(records_dir, completed_records)
            completed_now += 1
            frames_written += len(frames)

            # Full Optimised-Valence/H-state production already is the CM
            # confirmation stage. Use the known starting graph directly so an
            # early product cannot disappear inside the generic scanner warmup.
            # This post-processing is deliberately isolated from MD success.
            try:
                controlled = molecule_scanner.record_controlled_final_event(
                    recorder,
                    records_dir / recording_name,
                    reactants if family == "microcell" else [first, second],
                    library_root=str(molecule_root),
                    context={
                        "seed": spec["simulation_seed"],
                        "mixture": "reaction teacher production",
                        "batch": root.name,
                    },
                )
                controlled_event = controlled.get("event")
                experiment_events = (
                    [controlled_event] if controlled_event is not None else []
                )
                routed = route_full_cm_events_to_qm(
                    store,
                    experiment_events,
                    molecule_root,
                    qm_root=qm_root,
                    production_id=invocation_id,
                    teacher_root=root,
                    teacher_layout="live_production",
                )
                queued_now += routed["queued"]
                duplicate_queue_events += routed["duplicates"]
                invocation_event_ids.update(
                    str(event["event_id"])
                    for event in experiment_events
                    if event.get("event_id")
                )

                products_now = _product_report(
                    experiment_events,
                    molecule_root,
                    store,
                    qm_root=qm_root,
                    species_before=species_before,
                )
                for product in products_now:
                    invocation_product_ids.add(product["id"])
                    if product["new_this_experiment"]:
                        invocation_new_product_ids.add(product["id"])

                postprocess_warning = None
            except Exception as problem:
                experiment_events = []
                products_now = []
                routed = {"queued": 0, "duplicates": 0}
                postprocess_warning = f"{type(problem).__name__}: {problem}"

            experiment_outcome = {
                "schema_version": 1,
                "characterisation_outcome": entry.get(
                    "characterisation_outcome", "unknown"
                ),
                "reacted": bool(entry.get("reacted", False)),
                "stable": bool(entry.get("stable", True)),
                "reaction_event_count": int(len(experiment_events)),
                "reaction_event_ids": [
                    str(event.get("event_id"))
                    for event in experiment_events
                    if event.get("event_id")
                ],
                "product_species": [
                    str(row["id"]) for row in products_now
                    if row.get("id")
                ],
                "new_product_species": [
                    str(row["id"]) for row in products_now
                    if row.get("id") and row.get("new_this_experiment")
                ],
                "product_results": products_now,
                "queued_for_qm": int(routed["queued"]),
                "duplicate_queue_events": int(routed["duplicates"]),
                "postprocess_status": (
                    "warning" if postprocess_warning else "complete"
                ),
                "postprocess_warning": postprocess_warning,
            }
            record["experiment_outcome"] = experiment_outcome
            _atomic_json(
                experiments_dir / f"{spec['id']}.json",
                record,
            )

            if result_observer:
                termination = entry.get("termination")
                if not isinstance(termination, dict):
                    termination = {}

                actual_ps = float(entry.get(
                    "picoseconds",
                    float(getattr(simulation, "elapsed_femtoseconds", 0.0))
                    / 1000.0,
                ))
                requested_ps = float(entry.get(
                    "requested_picoseconds", duration_ps
                ))

                if family == "microcell":
                    termination_reason = str(
                        entry.get("termination_reason")
                        or termination.get("reason")
                        or (
                            "duration_complete"
                            if actual_ps + 1e-9 >= requested_ps
                            else "ended_early"
                        )
                    )
                else:
                    termination_reason = (
                        "numerical_failure"
                        if stopped_early
                        else (
                            "duration_complete"
                            if actual_ps + 1e-9 >= requested_ps
                            else "ended_early"
                        )
                    )

                result_observer({
                    "number": number,
                    "total": len(specs),
                    "experiment_id": spec["id"],
                    "family": family,
                    "outcome": entry.get(
                        "characterisation_outcome", "unknown"
                    ),
                    "reaction_events": len(experiment_events),
                    "products": products_now,
                    "queued_for_qm": routed["queued"],
                    "duplicate_queue_events": routed["duplicates"],
                    "postprocess_warning": postprocess_warning,
                    "actual_picoseconds": actual_ps,
                    "requested_picoseconds": requested_ps,
                    "termination_reason": termination_reason,
                })

        except Exception as problem:
            failed_now += 1
            failure = {
                "format_version": FORMAT_VERSION,
                "status": "failed",
                "production_id": invocation_id,
                "invocation_id": invocation_id,
                "manager_run_id": manager_run_id,
                "experiment_id": spec["id"],
                "specification": spec,
                "error": f"{type(problem).__name__}: {problem}",
                "chemistrymodel_git_revision": revision,
                "physics_source_sha256": source_hash,
                "device": resolved_device,
            }
            failure["experiment_outcome"] = {
                "schema_version": 1,
                "characterisation_outcome": "failed",
                "reacted": False,
                "stable": False,
                "reaction_event_count": 0,
                "reaction_event_ids": [],
                "product_species": [],
                "new_product_species": [],
                "product_results": [],
                "queued_for_qm": 0,
                "duplicate_queue_events": 0,
                "postprocess_status": "not_run",
                "postprocess_warning": None,
                "error": failure["error"],
            }
            _atomic_json(experiments_dir / f"{spec['id']}.json", failure)
            if result_observer:
                result_observer({
                    "number": number,
                    "total": len(specs),
                    "experiment_id": spec["id"],
                    "family": family,
                    "outcome": "failed",
                    "reaction_events": 0,
                    "products": [],
                    "queued_for_qm": 0,
                    "duplicate_queue_events": 0,
                    "error": failure["error"],
                    "requested_picoseconds": float(duration_ps),
                    "termination_reason": "failed",
                })

    _rebuild_recording_index(records_dir, completed_records)

    # Generic scanner runs once per invocation for transient/provenance
    # discovery. A scanner bookkeeping failure cannot retroactively fail MD.
    try:
        scan = molecule_scanner.scan_recordings(
            runs_root=str(root), library_root=str(molecule_root)
        )
        after_events, event_errors = read_events(event_log_path(molecule_root))
    except Exception as problem:
        scan = {
            "recordings_found": 0,
            "scanned": 0,
            "unchanged": 0,
            "formation_events": 0,
            "errors": [f"{type(problem).__name__}: {problem}"],
        }
        after_events, event_errors = read_events(event_log_path(molecule_root))
        event_errors = list(event_errors) + [
            f"scanner post-process: {type(problem).__name__}: {problem}"
        ]

    current_ids = {str(spec["id"]) for spec in specs}
    production_events = [
        row for row in after_events
        if row.get("recording")
        and Path(str(row["recording"])).stem in current_ids
    ]

    final_routed = route_full_cm_events_to_qm(
        store,
        production_events,
        molecule_root,
        qm_root=qm_root,
        production_id=invocation_id,
        teacher_root=root,
        teacher_layout="live_production",
    )
    queued_now += final_routed["queued"]
    duplicate_queue_events += final_routed["duplicates"]

    outcomes = {}
    for record in completed_records:
        if record.get("invocation_id") != invocation_id:
            continue
        outcome = record["run_entry"].get(
            "characterisation_outcome", "unknown"
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    invocation.update({
        "status": "complete",
        "completed": completed_now,
        "failed": failed_now,
        "teacher_frames": frames_written,
        "reaction_events": len(production_events),
        "queued_for_qm": queued_now,
        "product_species": sorted(invocation_product_ids),
        "new_product_species": sorted(invocation_new_product_ids),
        "completed_unix": datetime.now().timestamp(),
        "outcomes": outcomes,
    })
    _atomic_json(invocations_dir / f"{invocation_id}.json", invocation)
    _append_jsonl(root / "invocations.jsonl", {
        key: invocation[key]
        for key in (
            "id", "date", "requested", "completed", "failed",
            "master_seed", "master_seed_source", "profile", "physics",
            "device", "teacher_frames", "reaction_events",
            "started_unix", "completed_unix",
        )
    })
    day_summary = _write_day_summary(root)

    return {
        "production_id": invocation_id,
        "invocation_id": invocation_id,
        "date": day,
        "root": str(root),
        "requested": count,
        "completed_total": completed_now,
        "completed_now": completed_now,
        "failed_now": failed_now,
        "teacher_frames_written_now": frames_written,
        "new_events": len(production_events),
        "new_candidates_queued": queued_now,
        "duplicate_candidates": duplicate_queue_events,
        "queue_handoff_deferred_to_ingest": False,
        "product_species": sorted(invocation_product_ids),
        "new_product_species": sorted(invocation_new_product_ids),
        "event_log_errors": event_errors,
        "scan": scan,
        "outcomes": outcomes,
        "trusted_molecules": len(pool["molecules"]),
        "wild_probability": float(wild_probability),
        "master_seed": used_master_seed,
        "master_seed_source": (
            "random" if master_seed is None else "explicit"
        ),
        "day_summary": day_summary,
        "experiment_ids": [str(spec["id"]) for spec in specs],
    }
