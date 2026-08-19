"""Targeted full-ChemistryModel reaction teacher-data production."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

import characterisation_runner as characterisation
import molecule_scanner
import reactive as reactive_parameters

from .discovery import event_log_path, read_events
from .trust import trusted_molecules


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


def generate_experiment_specs(count, master_seed, molecule_root="molecules",
                              qm_root=None, profile="balanced"):
    count = int(count)
    if count < 1:
        raise ValueError("experiment count must be at least one")
    if profile not in SPEED_RANGES:
        raise ValueError(f"unknown reaction-production profile: {profile}")

    pool = allowed_reactants(molecule_root, qm_root=qm_root)
    generator = np.random.default_rng(int(master_seed))
    categories = [
        "atom_atom", "atom_molecule", "molecule_atom", "molecule_molecule"
    ] if pool["molecules"] else ["atom_atom"]
    collision_classes = ["direct", "glancing", "near_miss"]
    speed_classes = ["low", "moderate", "high"]
    specs = []

    for number in range(count):
        category = categories[number % len(categories)]
        collision_class = collision_classes[number % len(collision_classes)]
        speed_class = speed_classes[(number + number // 3) % len(speed_classes)]
        first_id, second_id, category = _choose_pair(
            category, generator, pool["atoms"], pool["molecules"]
        )
        first = _load_reactant(first_id, molecule_root)
        second = _load_reactant(second_id, molecule_root)
        simulation_seed = int(generator.integers(0, 2**31 - 1))

        use_atom_target = len(first["symbols"]) == 1 or generator.random() < 0.7
        target_atom = (
            int(generator.integers(0, len(first["symbols"])))
            if use_atom_target else None
        )
        sampling_mode = "targeted_random" if target_atom is not None else "random_orientation"
        impact_target = "com" if target_atom is None else str(first["symbols"][target_atom]).lower()
        if impact_target == "c":
            impact_target = "carbon"
        elif impact_target == "o":
            impact_target = "oxygen"
        elif impact_target == "h":
            impact_target = "hydrogen"
        elif impact_target == "n":
            # The manual enum has no nitrogen label; an explicit target index
            # remains authoritative and is accepted with the neutral COM name.
            impact_target = "com"

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
        temperature = float(TEMPERATURES_K[number % len(TEMPERATURES_K)])
        minimum_box = characterisation.minimum_pair_box_size(
            first, second, start_gap
        )
        box = float(max(12.0, minimum_box + 2.0 * impact_parameter + 2.0))

        spec = {
            "number": number,
            "category": category,
            "reactant_a": first_id,
            "reactant_b": second_id,
            "reactant_a_formula": first.get("formula"),
            "reactant_b_formula": second.get("formula"),
            "reactant_a_sha256": _reactant_signature(first),
            "reactant_b_sha256": _reactant_signature(second),
            "simulation_seed": simulation_seed,
            "master_seed": int(master_seed),
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
        spec["id"] = "EXP_" + _canonical_hash(spec)
        specs.append(spec)

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
        first_count = int(info.get("first_count", len(symbols)))
        distances, taper = characterisation._cross_pair_arrays(
            positions, symbols, first_count, snapshot["box_A"]
        )
        snapshot["cross_min_distance_A"] = (
            float(np.min(distances)) if distances.size else float("inf")
        )
        snapshot["cross_max_taper"] = (
            float(np.max(taper)) if taper.size else 0.0
        )
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


def _rebuild_recording_index(records_dir, records):
    payload = []
    for record in sorted(records, key=lambda row: row["experiment_id"]):
        entry = dict(record["run_entry"])
        entry["file"] = record["recording"]
        payload.append(entry)
    _atomic_json(Path(records_dir) / "index.json", payload)


def run_production(store, *, count=12, duration_ps=0.25, master_seed=20260819,
                   profile="balanced", output_root="teacher_data",
                   molecule_root="molecules", qm_root=None, device=None,
                   ordinary_interval_fs=10.0, event_window_fs=5.0,
                   diagnostic_sample_fs=1.0, capture_every=4,
                   physics="optimised-valence", progress=None):
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

    specs, pool = generate_experiment_specs(
        count, master_seed, molecule_root=molecule_root,
        qm_root=qm_root, profile=profile,
    )
    revision = git_revision()
    source_hash = physics_source_fingerprint()
    teacher_physics = characterisation.physics_metadata(physics)
    if device is None:
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(device)
    trusted_pool = [
        {
            "id": row["id"], "formula": row.get("formula"),
            "trust_status": row.get("trust_status"),
            "reactant_sha256": _reactant_signature(row),
        }
        for row in pool["trusted_molecule_records"]
    ]
    production_key = {
        "format_version": FORMAT_VERSION,
        "master_seed": int(master_seed), "count": int(count),
        "duration_ps": float(duration_ps), "profile": profile,
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
        "trusted_pool": trusted_pool,
        "specifications": specs,
    }
    production_id = "PROD_" + _canonical_hash(production_key)
    root = Path(output_root) / production_id
    records_dir = root / "recordings"
    experiments_dir = root / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "production.json", {
        **production_key,
        "id": production_id,
        "chemistrymodel_git_revision": revision,
        "allowed_atoms": pool["atoms"],
        "trusted_molecules": trusted_pool,
    })

    completed_records = []
    for path in experiments_dir.glob("EXP_*.json") if experiments_dir.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") == "complete" and (root / record.get("shard", "")).is_file():
            completed_records.append(record)
    complete_ids = {row["experiment_id"] for row in completed_records}
    completed_now = 0
    frames_written = 0

    for number, spec in enumerate(specs, start=1):
        if spec["id"] in complete_ids:
            continue
        if progress:
            progress(number, len(specs), spec)
        first = _load_reactant(spec["reactant_a"], molecule_root)
        second = _load_reactant(spec["reactant_b"], molecule_root)
        collector = TeacherFrameCollector(
            ordinary_interval_fs=ordinary_interval_fs,
            event_window_fs=event_window_fs,
        )
        options = _options_for(
            spec, root, molecule_root, duration_ps, resolved_device,
            capture_every, diagnostic_sample_fs, physics,
        )
        try:
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
        entry = characterisation.summarise(
            recorder, first, second, spec["simulation_seed"], options,
            wall_seconds, stopped_early,
            collision_measures[0], infos[0], diagnostics[0],
            simulation=simulation,
        )
        entry.update({
            "file": recording_name,
            "experiment_id": spec["id"],
            "production_id": production_id,
            "mixture": "targeted reaction teacher production",
            "chemistrymodel_git_revision": revision,
            "physics_source_sha256": source_hash,
        })
        frames = collector.selected()
        shard_relative = str(Path("shards") / f"{spec['id']}.npz")
        _write_teacher_shard(
            root / shard_relative, frames,
            experiment_id=spec["id"], production_id=production_id,
        )
        record = {
            "format_version": FORMAT_VERSION,
            "status": "complete",
            "production_id": production_id,
            "experiment_id": spec["id"],
            "specification": spec,
            "actual_collision": infos[0],
            "velocity": collision_measures[0],
            "run_entry": entry,
            "recording": recording_name,
            "shard": shard_relative,
            "teacher_frames": len(frames),
            "chemistrymodel_git_revision": revision,
            "physics_source_sha256": source_hash,
            "physics_model": getattr(simulation, "physics_model_name", entry["physics_model"]),
            "physics_model_revision": getattr(simulation, "physics_model_revision", None),
            "device": str(getattr(simulation, "device", resolved_device)),
        }
        _atomic_json(experiments_dir / f"{spec['id']}.json", record)
        completed_records.append(_plain(record))
        _rebuild_recording_index(records_dir, completed_records)
        completed_now += 1
        frames_written += len(frames)

    # Rebuild even when this invocation only resumes already-complete shards:
    # an interruption may have happened after an experiment record was made
    # but before the scanner index was atomically replaced.
    _rebuild_recording_index(records_dir, completed_records)

    # Reuse the established scanner for product/event recognition. Only events
    # newly created by this production are handed to the manager queue.
    scan = molecule_scanner.scan_recordings(
        runs_root=str(root), library_root=str(molecule_root)
    )
    after_events, event_errors = read_events(event_log_path(molecule_root))
    records_absolute = records_dir.resolve()

    def belongs_to_production(event):
        recording = event.get("recording")
        if not recording:
            return False
        try:
            return Path(recording).resolve().parent == records_absolute
        except OSError:
            return False

    production_events = [
        row for row in after_events if belongs_to_production(row)
    ]
    queued = store.add_discovery_events(
        production_events, event_log_path(molecule_root)
    )

    outcomes = {}
    for record in completed_records:
        outcome = record["run_entry"].get("characterisation_outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "production_id": production_id,
        "root": str(root),
        "requested": len(specs),
        "completed_total": len(completed_records),
        "completed_now": completed_now,
        "teacher_frames_written_now": frames_written,
        "new_events": len(production_events),
        "new_candidates_queued": queued["added"],
        "duplicate_candidates": queued["duplicates"],
        "event_log_errors": event_errors,
        "scan": scan,
        "outcomes": outcomes,
        "trusted_molecules": len(pool["molecules"]),
    }
