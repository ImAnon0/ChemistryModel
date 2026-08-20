"""Targeted full-ChemistryModel reaction teacher-data production."""

import collections
import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import characterisation_runner as characterisation
import molecule_scanner
import reactive as reactive_parameters

from .discovery import event_log_path, read_events, route_full_cm_events_to_qm
from .state import CandidateState
from .trust import MoleculeTrust, trust_level, trusted_molecules


FORMAT_VERSION = 1
ELEMENTAL_REACTANTS = ("atom:H", "atom:C", "atom:N", "atom:O")
PHYSICS_FILES = (
    "reactive.py", "reactive_torch.py", "batched_torch.py",
    "high_fidelity_torch.py", "characterisation_runner.py",
    "h_state_reference.py", "h_state_torch.py",
    "h_state_component_torch.py", "h_state_factorised_torch.py",
    "h_state_factorised_batched_torch.py", "valence_state_torch.py",
    "valence_state_factorised_torch.py",
    "valence_state_factorised_batched_torch.py",
    "valence_state_batched_membership_torch.py",
    "valence_state_cached_h_topology_torch.py",
    "valence_state_optimised_torch.py", "heavy_valence_density.py",
)
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
PAIR_FAMILY_WEIGHT = 1.0
MICROCELL_FAMILY_WEIGHT = 1.0
MICROCELL_OBJECT_COUNT_WEIGHTS = {3: 0.55, 4: 0.30, 5: 0.15}
MICROCELL_CLUSTER_RADIUS_RANGE_A = (2.8, 4.8)
MICROCELL_MINIMUM_GAP_RANGE_A = (1.7, 2.4)
MICROCELL_INWARD_FACTOR_RANGE = (0.35, 1.35)
MICROCELL_COMPOSITION_REUSE_EXPONENT = 1.0
MICROCELL_CONFIG_REUSE_EXPONENT = 1.1


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def physics_source_fingerprint(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    digest = hashlib.sha256()
    for name in PHYSICS_FILES:
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    """Read completed teacher experiments from both legacy and daily layouts."""
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
            records.append(spec)
    return records


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
    ids = list(pool["atoms"]) + list(pool["molecules"])
    counts = list(MICROCELL_OBJECT_COUNT_WEIGHTS)
    weights = np.asarray([MICROCELL_OBJECT_COUNT_WEIGHTS[x] for x in counts], dtype=float)
    weights /= weights.sum()
    object_count = int(generator.choice(counts, p=weights))
    reactant_ids = [str(generator.choice(ids)) for _ in range(object_count)]

    def cached(reactant_id):
        if reactant_id not in reactant_cache:
            reactant_cache[reactant_id] = _load_reactant(reactant_id, molecule_root)
        return reactant_cache[reactant_id]

    reactants = [cached(rid) for rid in reactant_ids]
    cluster_radius = float(generator.uniform(*MICROCELL_CLUSTER_RADIUS_RANGE_A))
    minimum_gap = float(generator.uniform(*MICROCELL_MINIMUM_GAP_RANGE_A))
    inward_factor = float(generator.uniform(*MICROCELL_INWARD_FACTOR_RANGE))
    temperature = float(generator.choice(TEMPERATURES_K))
    max_radius = max((_radius(r) for r in reactants), default=0.0)
    return {
        "number": int(number),
        "experiment_family": "microcell",
        "category": "microcell",
        "reactants": reactant_ids,
        "reactant_formulas": [r.get("formula") for r in reactants],
        "object_count": object_count,
        "simulation_seed": int(generator.integers(0, 2**31 - 1)),
        "profile": profile,
        "temperature_K": temperature,
        "cluster_radius_A": cluster_radius,
        "minimum_gap_A": minimum_gap,
        "inward_factor": inward_factor,
        "box_A": float(max(14.0, 2.0 * (cluster_radius + max_radius + 2.5))),
    }


def _choose_experiment_family(generator):
    weights = np.asarray([PAIR_FAMILY_WEIGHT, MICROCELL_FAMILY_WEIGHT], dtype=float)
    weights /= weights.sum()
    return str(generator.choice(["pair", "microcell"], p=weights))



def generate_experiment_specs(
    count, master_seed, molecule_root="molecules", qm_root=None,
    profile="balanced", history=None, invocation_id=None,
):
    count = int(count)
    if count < 1:
        raise ValueError("experiment count must be at least one")
    if profile not in SPEED_RANGES:
        raise ValueError(f"unknown reaction-production profile: {profile}")

    pool = allowed_reactants(molecule_root, qm_root=qm_root)
    generator = np.random.default_rng(int(master_seed))
    categories = ["atom_atom", "atom_molecule", "molecule_atom", "molecule_molecule"] if pool["molecules"] else ["atom_atom"]
    collisions = ["direct", "glancing", "near_miss"]
    speeds = ["low", "moderate", "high"]
    working_history = [dict(row) for row in (history or [])]
    cache = {}
    specs = []
    pair_number = 0

    for number in range(count):
        family = _choose_experiment_family(generator)
        candidates, weights = [], []

        if family == "pair":
            category = categories[pair_number % len(categories)]
            pair_number += 1
            for candidate_number in range(NOVELTY_CANDIDATES_PER_SLOT):
                candidate = _build_candidate_spec(
                    number,
                    category,
                    collisions[candidate_number % len(collisions)],
                    speeds[(candidate_number // len(collisions)) % len(speeds)],
                    generator, pool, molecule_root, profile, cache,
                )
                candidate["experiment_family"] = "pair"
                candidates.append(candidate)
                weights.append(novelty_weight(candidate, working_history))
        else:
            for _ in range(NOVELTY_CANDIDATES_PER_SLOT):
                candidate = _build_microcell_candidate(
                    number, generator, pool, molecule_root, profile, cache
                )
                candidates.append(candidate)
                weights.append(microcell_novelty_weight(candidate, working_history))

        spec = dict(_weighted_choice(generator, candidates, weights))
        spec["master_seed"] = int(master_seed)
        identity = dict(spec)
        if invocation_id is not None:
            identity["invocation_id"] = str(invocation_id)
        spec["id"] = "EXP_" + _canonical_hash(identity)
        specs.append(spec)
        working_history.append(spec)

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
    os.replace(temporary, path)


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
            microcell_compositions.add(_composition_key(spec.get("reactants", [])))
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
    if str(physics) not in ("standard", "high_fidelity", "optimised-valence"):
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
    source_hash = physics_source_fingerprint()
    teacher_physics = characterisation.physics_metadata(physics)

    if device is None:
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(device)

    invocation_nonce = secrets.token_hex(6)
    invocation_id = (
        "INV_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        + "_"
        + invocation_nonce
    )

    history = _history_records(output_root)
    specs, pool = generate_experiment_specs(
        count, used_master_seed, molecule_root=molecule_root,
        qm_root=qm_root, profile=profile, history=history,
        invocation_id=invocation_id,
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
        "physics_model_revision": teacher_physics["physics_model_revision"],
        "physics_parameters": teacher_physics["physics_parameters"],
        "physics_source_sha256": source_hash,
        "device": resolved_device,
        "ordinary_interval_fs": float(ordinary_interval_fs),
        "event_window_fs": float(event_window_fs),
        "diagnostic_sample_fs": float(diagnostic_sample_fs),
        "capture_every": int(capture_every),
        "chemistrymodel_git_revision": revision,
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
                "physics_source_sha256": source_hash,
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

            if result_observer:
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
                })

        except Exception as problem:
            failed_now += 1
            failure = {
                "format_version": FORMAT_VERSION,
                "status": "failed",
                "production_id": invocation_id,
                "invocation_id": invocation_id,
                "experiment_id": spec["id"],
                "specification": spec,
                "error": f"{type(problem).__name__}: {problem}",
                "chemistrymodel_git_revision": revision,
                "physics_source_sha256": source_hash,
                "device": resolved_device,
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
        "master_seed": used_master_seed,
        "master_seed_source": (
            "random" if master_seed is None else "explicit"
        ),
        "day_summary": day_summary,
    }

