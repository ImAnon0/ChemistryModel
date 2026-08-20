import collections
import hashlib
import json
import os
import time

import numpy as np

import bonding
import molecule_library as molecule_store
from recorder import Recorder


# ============================================================
# Incremental molecule discovery / chemistry knowledge scan
# ============================================================
#
# Scan every trustworthy recording once, then only the new tail if a
# recording is extended. The manifest records what was processed. The
# species library stores one representative geometry per structural
# fingerprint; formation_events.jsonl stores atom-conserving graph
# changes with the conditions in which they happened.


MANIFEST_NAME = "scan_manifest.json"
MANIFEST_VERSION = 2
EVENT_LOG = molecule_store.EVENT_LOG
LOCAL_ENVIRONMENT_RADIUS_A = 4.0


def _normal_path(path):
    try:
        path = os.path.relpath(os.path.abspath(path), os.getcwd())
    except ValueError:
        path = os.path.abspath(path)

    return os.path.normcase(os.path.normpath(path))


def _load_json(path, fallback):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt >= 4:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def load_manifest(library_root=molecule_store.DEFAULT_ROOT):
    path = os.path.join(library_root, MANIFEST_NAME)
    manifest = _load_json(path, {})

    if not isinstance(manifest, dict):
        manifest = {}

    manifest.setdefault("version", MANIFEST_VERSION)
    manifest.setdefault("recordings", {})
    return manifest


def save_manifest(manifest, library_root=molecule_store.DEFAULT_ROOT):
    manifest = dict(manifest)
    manifest["version"] = MANIFEST_VERSION
    manifest["updated_unix"] = time.time()
    path = os.path.join(library_root, MANIFEST_NAME)
    _write_json_atomic(path, manifest)


def manifest_summary(library_root=molecule_store.DEFAULT_ROOT):
    manifest = load_manifest(library_root)
    statuses = collections.Counter(
        item.get("status", "unknown")
        for item in manifest.get("recordings", {}).values()
    )

    return {
        "recordings": len(manifest.get("recordings", {})),
        "scanned": statuses.get("scanned", 0),
        "legacy": statuses.get("legacy", 0),
        "unstable": statuses.get("unstable", 0),
        "errors": statuses.get("error", 0),
    }


def _discover_recordings(runs_root):
    found = {}

    if not os.path.isdir(runs_root):
        return []

    for directory, _, files in os.walk(runs_root):
        if "index.json" not in files:
            continue

        index_path = os.path.join(directory, "index.json")
        entries = _load_json(index_path, [])

        if not isinstance(entries, list):
            continue

        batch = os.path.basename(os.path.normpath(directory))

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            filename = entry.get("file")

            if not filename:
                continue

            path = filename if os.path.isabs(filename) else os.path.join(directory, filename)

            if not os.path.isfile(path):
                continue

            key = _normal_path(path)

            parent_key = None
            continued_from = entry.get("continued_from")

            if continued_from:
                parent_path = os.path.join(
                    str(continued_from), os.path.basename(path)
                )
                parent_key = _normal_path(parent_path)

            found[key] = {
                "key": key,
                "path": path,
                "batch": batch,
                "entry": dict(entry),
                "parent_key": parent_key,
            }

    def depth(recording, trail=None):
        trail = set(trail or ())
        key = recording["key"]
        parent = recording.get("parent_key")

        if not parent or parent not in found or parent in trail:
            return 0

        trail.add(key)
        return 1 + depth(found[parent], trail)

    return sorted(found.values(), key=lambda item: (depth(item), item["key"]))


def _file_stat(path):
    stat = os.stat(path)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _same_file_stat(previous, current):
    return (
        previous.get("file_size") == current["size"]
        and previous.get("file_mtime_ns") == current["mtime_ns"]
    )


def _recording_signature(recorder):
    if len(recorder) == 0:
        return "empty"

    digest = hashlib.sha256()
    digest.update(np.asarray(recorder.types_at(0), dtype=np.uint8).tobytes())
    digest.update(np.asarray(recorder.atom_ids_at(0), dtype=np.uint32).tobytes())
    digest.update(np.asarray(recorder.positions[0], dtype=np.float32).tobytes())
    digest.update(np.asarray([float(recorder.times[0])], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _component_instance_key(component):
    ids = ",".join(str(int(value)) for value in sorted(component["atom_ids"]))
    return f"{component['graph_fingerprint']}|{ids}"


def _component_exact_key(component):
    ids = tuple(sorted(int(value) for value in component["atom_ids"]))
    edges = tuple(sorted(tuple(map(int, pair)) for pair in component["id_bonds"]))
    return ids, edges


def _component_descriptor(component):
    item = {
        "formula": component["formula"],
        "atoms": int(component["atoms"]),
        "heavy_atoms": int(component["heavy_atoms"]),
        "count": 1,
    }

    if component.get("species_id"):
        item["id"] = component["species_id"]

    return item


def _combine_descriptors(components):
    grouped = collections.OrderedDict()

    for component in components:
        descriptor = _component_descriptor(component)
        key = descriptor.get("id") or f"formula:{descriptor['formula']}"

        if key not in grouped:
            grouped[key] = descriptor
        else:
            grouped[key]["count"] += 1

    return list(grouped.values())


def _all_species_counts(components):
    counts = collections.Counter(component["formula"] for component in components)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _local_environment(recorder, frame_index, affected_ids, radius_A):
    positions = np.asarray(recorder.positions[frame_index], dtype=float)
    atom_ids = np.asarray(recorder.atom_ids_at(frame_index), dtype=np.uint32)
    symbols = recorder.symbols_at(frame_index)
    box = float(recorder.box_at(frame_index))

    affected_ids = set(int(value) for value in affected_ids)
    affected_slots = [
        index for index, atom_id in enumerate(atom_ids)
        if int(atom_id) in affected_ids
    ]

    if not affected_slots:
        return {}

    counts = collections.Counter()

    for slot, atom_id in enumerate(atom_ids):
        if int(atom_id) in affected_ids:
            continue

        nearest = np.inf

        for centre in affected_slots:
            delta = positions[slot] - positions[centre]
            delta -= box * np.round(delta / box)
            nearest = min(nearest, float(np.linalg.norm(delta)))

        if nearest <= radius_A:
            counts[str(symbols[slot])] += 1

    return dict(sorted(counts.items()))


def _reaction_groups(previous, current):
    """Return overlap-connected changed component groups.

    Exact unchanged components are removed first. Remaining previous/current
    components are linked if they share a real atom ID. A valid chemical
    event later requires the union of real atom IDs on both sides to match;
    that rejects open-box insertion/removal as transport instead of chemistry.
    """

    previous_exact = collections.Counter(_component_exact_key(item) for item in previous)
    current_exact = collections.Counter(_component_exact_key(item) for item in current)

    keep_previous = []
    current_budget = dict(current_exact)

    for item in previous:
        key = _component_exact_key(item)
        if current_budget.get(key, 0) > 0:
            current_budget[key] -= 1
        else:
            keep_previous.append(item)

    keep_current = []
    previous_budget = dict(previous_exact)

    for item in current:
        key = _component_exact_key(item)
        if previous_budget.get(key, 0) > 0:
            previous_budget[key] -= 1
        else:
            keep_current.append(item)

    nodes = [("p", i) for i in range(len(keep_previous))] + [
        ("c", i) for i in range(len(keep_current))
    ]

    if not nodes:
        return []

    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    previous_ids = [set(map(int, item["atom_ids"])) for item in keep_previous]
    current_ids = [set(map(int, item["atom_ids"])) for item in keep_current]

    for p_index, p_ids in enumerate(previous_ids):
        for c_index, c_ids in enumerate(current_ids):
            if p_ids & c_ids:
                union(("p", p_index), ("c", c_index))

    groups = {}

    for node in nodes:
        groups.setdefault(find(node), {"previous": [], "current": []})
        side, index = node
        groups[find(node)]["previous" if side == "p" else "current"].append(
            keep_previous[index] if side == "p" else keep_current[index]
        )

    return list(groups.values())


def _event_id(recording_key, time_fs, previous_edges, current_edges):
    payload = {
        "recording": recording_key,
        "time_fs": round(float(time_fs), 6),
        "before": sorted(previous_edges),
        "after": sorted(current_edges),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _load_existing_event_ids(library_root):
    path = os.path.join(library_root, EVENT_LOG)
    found = set()

    if not os.path.isfile(path):
        return found

    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("event_id"):
                    found.add(event["event_id"])
    except OSError:
        pass

    return found


def _append_events(events, library_root):
    if not events:
        return

    os.makedirs(library_root, exist_ok=True)
    path = os.path.join(library_root, EVENT_LOG)

    with open(path, "a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def _source_key(recording_key):
    return recording_key.replace("\\", "/")


def _fresh_species_updates():
    return {
        "observations": 0,
        "formations": 0,
        "appearances": 0,
        "longest_observed_lifetime_fs": 0.0,
        "source_observations": 0,
        "source_formations": 0,
        "source_appearances": 0,
        "source_first_fs": None,
        "source_last_fs": None,
    }


def _apply_species_updates(updates, recording_key, recording_path,
                           context, library_root):
    source_key = _source_key(recording_key)

    for species_id, change in updates.items():
        try:
            metadata = molecule_store.load_molecule(species_id, root=library_root)
        except Exception:
            continue

        # load_molecule adds payload arrays; only metadata fields belong in JSON.
        for key in (
            "symbols", "positions", "bonds", "source_atom_ids",
            "source_slots", "velocities"
        ):
            metadata.pop(key, None)

        stats = metadata.setdefault("stats", {})
        stats["observations"] = int(stats.get("observations", 0)) + int(change["observations"])
        stats["formations"] = int(stats.get("formations", 0)) + int(change["formations"])
        stats["appearances"] = int(stats.get("appearances", 0)) + int(change["appearances"])
        stats["longest_observed_lifetime_fs"] = max(
            float(stats.get("longest_observed_lifetime_fs", 0.0)),
            float(change["longest_observed_lifetime_fs"]),
        )

        sources = metadata.setdefault("sources", {})
        source = sources.setdefault(source_key, {
            "recording": os.path.normpath(recording_path),
            "seed": context.get("seed"),
            "mixture": context.get("mixture"),
            "batch": context.get("batch"),
            "observations": 0,
            "formations": 0,
            "appearances": 0,
            "first_fs": None,
            "last_fs": None,
        })

        source["observations"] = int(source.get("observations", 0)) + int(change["source_observations"])
        source["formations"] = int(source.get("formations", 0)) + int(change["source_formations"])
        source["appearances"] = int(source.get("appearances", 0)) + int(change["source_appearances"])

        first_fs = change.get("source_first_fs")
        last_fs = change.get("source_last_fs")

        if first_fs is not None:
            old = source.get("first_fs")
            source["first_fs"] = float(first_fs) if old is None else min(float(old), float(first_fs))

        if last_fs is not None:
            old = source.get("last_fs")
            source["last_fs"] = float(last_fs) if old is None else max(float(old), float(last_fs))

        stats["runs_seen"] = len(sources)
        molecule_store.save_metadata(metadata, root=library_root)


def _resolve_species(component, recording_path, frame_index, recorder,
                     context, library_root, species_by_fingerprint,
                     allow_create):
    if component["atoms"] < 2:
        component["species_id"] = None
        return False

    fingerprint = component["graph_fingerprint"]
    existing = species_by_fingerprint.get(fingerprint)

    if existing is not None:
        component["species_id"] = existing["id"]
        return False

    if not allow_create:
        component["species_id"] = None
        return False

    metadata, created = molecule_store.ensure_species(
        recording_path,
        frame_index,
        component,
        root=library_root,
        recorder=recorder,
        source_context=context,
    )

    species_by_fingerprint[fingerprint] = metadata
    component["species_id"] = metadata["id"]
    return bool(created)


def _scan_one(recording, previous_manifest, inherited_manifest, library_root,
              species_by_fingerprint, existing_event_ids, progress=None):
    path = recording["path"]
    key = recording["key"]
    entry = recording["entry"]
    context = {
        "seed": entry.get("seed"),
        "mixture": entry.get("mixture"),
        "batch": recording.get("batch"),
    }

    file_stat = _file_stat(path)

    if entry.get("stable") is False:
        return {
            "manifest": {
                "status": "unstable",
                "file_size": file_stat["size"],
                "file_mtime_ns": file_stat["mtime_ns"],
                "last_scan_unix": time.time(),
                **context,
            },
            "result": {"status": "unstable"},
        }

    if previous_manifest and _same_file_stat(previous_manifest, file_stat):
        status = previous_manifest.get("status")
        if status in ("scanned", "legacy"):
            return {
                "manifest": previous_manifest,
                "result": {"status": "unchanged"},
            }

    recorder = Recorder.load(path)

    if len(recorder) == 0:
        return {
            "manifest": {
                "status": "empty",
                "file_size": file_stat["size"],
                "file_mtime_ns": file_stat["mtime_ns"],
                "last_scan_unix": time.time(),
                **context,
            },
            "result": {"status": "empty"},
        }

    if not recorder.has_atom_history:
        return {
            "manifest": {
                "status": "legacy",
                "file_size": file_stat["size"],
                "file_mtime_ns": file_stat["mtime_ns"],
                "last_scan_unix": time.time(),
                "frames": len(recorder),
                "final_time_fs": float(recorder.times[-1]),
                **context,
            },
            "result": {"status": "legacy"},
        }

    signature = _recording_signature(recorder)
    history_manifest = previous_manifest or inherited_manifest

    if history_manifest and history_manifest.get("recording_signature") not in (None, signature):
        relation = "previously scanned recording" if previous_manifest else "continued source recording"
        raise ValueError(
            f"recording start does not match its {relation}; automatic "
            "rescanning is blocked to avoid double-counting old chemistry"
        )

    old_time = None
    if history_manifest and history_manifest.get("status") == "scanned":
        old_time = history_manifest.get("last_scanned_time_fs")
        if old_time is not None:
            old_time = float(old_time)

    final_time = float(recorder.times[-1])

    if old_time is not None and final_time <= old_time + 1e-9:
        manifest = dict(previous_manifest)
        manifest.update({
            "file_size": file_stat["size"],
            "file_mtime_ns": file_stat["mtime_ns"],
            "last_scan_unix": time.time(),
            "frames": len(recorder),
            "final_time_fs": final_time,
        })
        return {"manifest": manifest, "result": {"status": "unchanged"}}

    times = np.asarray(recorder.times, dtype=float)
    warmup = max(float(bonding.FORMATION_TIME), float(bonding.BREAKING_TIME))

    if old_time is None:
        replay_start = 0
        count_after = float(times[0]) + warmup
    else:
        replay_from_time = old_time - 2.5 * warmup
        replay_start = int(np.searchsorted(times, replay_from_time, side="left"))
        replay_start = max(0, replay_start)
        count_after = old_time

    tracker = bonding.BondTracker(
        recorder.types_at(replay_start),
        atom_ids=recorder.atom_ids_at(replay_start),
    )

    previous_components = None
    previous_time = None
    active = {}

    stored_active = {
        item.get("instance_key"): item
        for item in (history_manifest or {}).get("active_instances", [])
        if item.get("instance_key")
    }

    updates = collections.defaultdict(_fresh_species_updates)
    events_to_append = []
    new_species = 0
    frames_counted = 0
    formation_events = 0

    for index in range(replay_start, len(recorder)):
        first, second = tracker.update(
            recorder.positions[index],
            recorder.box_at(index),
            float(times[index]),
            types=recorder.types_at(index),
            atom_ids=recorder.atom_ids_at(index),
        )

        components = molecule_store.component_records(
            recorder, index, first, second, include_single=True
        )

        is_new_frame = float(times[index]) > count_after + 1e-9

        # Species that occur only in the replay/warmup part must not create
        # new library data; the prior scan already owned those frames.
        for component in components:
            created = _resolve_species(
                component,
                path,
                index,
                recorder,
                context,
                library_root,
                species_by_fingerprint,
                allow_create=is_new_frame,
            )
            new_species += int(created)

        if is_new_frame:
            frames_counted += 1
            current_time = float(times[index])

            current_active = {}

            for component in components:
                species_id = component.get("species_id")

                if not species_id:
                    continue

                instance_key = _component_instance_key(component)

                if instance_key in active:
                    started_fs = active[instance_key]["started_fs"]
                elif frames_counted == 1 and instance_key in stored_active:
                    started_fs = float(stored_active[instance_key].get("started_fs", current_time))
                else:
                    started_fs = current_time
                    updates[species_id]["appearances"] += 1
                    updates[species_id]["source_appearances"] += 1

                current_active[instance_key] = {
                    "instance_key": instance_key,
                    "species_id": species_id,
                    "started_fs": started_fs,
                    "last_seen_fs": current_time,
                    "atom_ids": [int(value) for value in sorted(component["atom_ids"])],
                }

                change = updates[species_id]
                change["observations"] += 1
                change["source_observations"] += 1
                change["source_first_fs"] = current_time if change["source_first_fs"] is None else min(change["source_first_fs"], current_time)
                change["source_last_fs"] = current_time if change["source_last_fs"] is None else max(change["source_last_fs"], current_time)
                change["longest_observed_lifetime_fs"] = max(
                    float(change["longest_observed_lifetime_fs"]),
                    current_time - started_fs,
                )

            # Reaction extraction starts only once there is an actual previous
            # graph state. The first countable frame of a fresh recording is a
            # baseline observation, not invented chemistry.
            if (
                previous_components is not None
                and (old_time is not None or frames_counted > 1)
            ):
                for group in _reaction_groups(previous_components, components):
                    before = group["previous"]
                    after = group["current"]

                    before_ids = set().union(*(set(map(int, item["atom_ids"])) for item in before)) if before else set()
                    after_ids = set().union(*(set(map(int, item["atom_ids"])) for item in after)) if after else set()

                    if before_ids != after_ids or not before_ids:
                        continue

                    before_edges = set().union(*(set(item["id_bonds"]) for item in before)) if before else set()
                    after_edges = set().union(*(set(item["id_bonds"]) for item in after)) if after else set()

                    formed = sorted(after_edges - before_edges)
                    broken = sorted(before_edges - after_edges)

                    if not formed and not broken:
                        continue

                    event_id = _event_id(key, current_time, before_edges, after_edges)

                    if event_id in existing_event_ids:
                        continue

                    # Any product first encountered because of this event gets
                    # its SP identity before the event is written.
                    for component in after:
                        if component["atoms"] >= 2 and not component.get("species_id"):
                            created = _resolve_species(
                                component,
                                path,
                                index,
                                recorder,
                                context,
                                library_root,
                                species_by_fingerprint,
                                allow_create=True,
                            )
                            new_species += int(created)

                    symbols_by_id = {
                        int(atom_id): symbol
                        for atom_id, symbol in zip(
                            recorder.atom_ids_at(index), recorder.symbols_at(index)
                        )
                    }

                    def bond_records(pairs):
                        records = []
                        for a, b in pairs:
                            records.append({
                                "atom_ids": [int(a), int(b)],
                                "symbols": [
                                    str(symbols_by_id.get(int(a), "?")),
                                    str(symbols_by_id.get(int(b), "?")),
                                ],
                            })
                        return records

                    event = {
                        "event_id": event_id,
                        "recording": key,
                        "batch": context.get("batch"),
                        "seed": context.get("seed"),
                        "mixture": context.get("mixture"),
                        "time_fs": current_time,
                        "previous_frame_time_fs": previous_time,
                        "temperature_K": float(recorder.temperature[index]),
                        "box_A": float(recorder.box_at(index)),
                        "density_atoms_per_A3": float(
                            len(recorder.positions[index]) / (float(recorder.box_at(index)) ** 3)
                        ),
                        "reactants": _combine_descriptors(before),
                        "products": _combine_descriptors(after),
                        "formed_bonds": bond_records(formed),
                        "broken_bonds": bond_records(broken),
                        "local_environment_radius_A": LOCAL_ENVIRONMENT_RADIUS_A,
                        "local_environment_elements": _local_environment(
                            recorder, index, before_ids, LOCAL_ENVIRONMENT_RADIUS_A
                        ),
                        "frame_species": _all_species_counts(components),
                    }

                    events_to_append.append(event)
                    existing_event_ids.add(event_id)
                    formation_events += 1

                    for product in after:
                        species_id = product.get("species_id")
                        if not species_id:
                            continue
                        updates[species_id]["formations"] += 1
                        updates[species_id]["source_formations"] += 1

            active = current_active

        previous_components = components
        previous_time = float(times[index])

        if progress and (
            index == replay_start
            or index == len(recorder) - 1
            or (index - replay_start) % 250 == 0
        ):
            progress({
                "stage": "frames",
                "recording": key,
                "frame": index + 1,
                "frames": len(recorder),
            })

    _append_events(events_to_append, library_root)
    _apply_species_updates(updates, key, path, context, library_root)

    manifest = {
        "status": "scanned",
        "recording_signature": signature,
        "last_scanned_time_fs": final_time,
        "frames": len(recorder),
        "final_time_fs": final_time,
        "file_size": file_stat["size"],
        "file_mtime_ns": file_stat["mtime_ns"],
        "last_scan_unix": time.time(),
        "active_instances": list(active.values()),
        "continued_from_key": recording.get("parent_key"),
        **context,
    }

    return {
        "manifest": manifest,
        "result": {
            "status": "scanned",
            "frames_counted": frames_counted,
            "new_species": new_species,
            "formation_events": formation_events,
        },
    }



def _reactant_descriptors(reactants):
    grouped = collections.OrderedDict()
    for reactant in reactants:
        item = {
            "formula": reactant.get("formula"),
            "atoms": int(reactant.get("atoms", len(reactant.get("symbols", [])))),
            "heavy_atoms": int(reactant.get(
                "heavy_atoms",
                sum(str(symbol) != "H" for symbol in reactant.get("symbols", [])),
            )),
            "count": 1,
        }
        reactant_id = reactant.get("id")
        if reactant_id and not str(reactant_id).startswith("atom:"):
            item["id"] = str(reactant_id)
        key = item.get("id") or f"formula:{item['formula']}"
        if key not in grouped:
            grouped[key] = item
        else:
            grouped[key]["count"] += 1
    return list(grouped.values())


def _initial_id_edges(reactants, atom_ids):
    atom_ids = np.asarray(atom_ids, dtype=np.uint32)
    edges = set()
    cursor = 0
    for reactant in reactants:
        count = int(reactant.get("atoms", len(reactant.get("symbols", []))))
        bonds = np.asarray(reactant.get("bonds", []), dtype=np.int32).reshape(-1, 2)
        if cursor + count > len(atom_ids):
            raise ValueError("controlled reactants exceed recorder atom count")
        for first, second in bonds:
            a = int(atom_ids[cursor + int(first)])
            b = int(atom_ids[cursor + int(second)])
            edges.add((min(a, b), max(a, b)))
        cursor += count
    if cursor != len(atom_ids):
        raise ValueError("controlled reactants do not cover recorder atom count")
    return edges


def record_controlled_final_event(
    recorder, recording_path, reactants, *,
    library_root=molecule_store.DEFAULT_ROOT, context=None,
):
    """Compare known controlled reactants directly with the persisted final graph."""
    molecule_store.require_identity_history(recorder)
    if len(recorder) == 0:
        raise ValueError("cannot inspect an empty controlled recording")

    reactants = list(reactants)
    final_index = len(recorder) - 1
    components = molecule_store.molecules_at(recorder, final_index)

    initial_fingerprints = sorted(
        str(r.get("graph_fingerprint")) for r in reactants
    )
    final_fingerprints = sorted(
        str(c.get("graph_fingerprint")) for c in components
    )
    expected_atoms = sum(
        int(r.get("atoms", len(r.get("symbols", [])))) for r in reactants
    )
    final_atoms = sum(int(c.get("atoms", 0)) for c in components)

    if final_atoms != expected_atoms:
        return {"status": "unclassified", "event": None, "created_species": []}
    if final_fingerprints == initial_fingerprints:
        return {"status": "no reaction", "event": None, "created_species": []}

    recording_path = os.path.normpath(str(recording_path))
    recording_key = _normal_path(recording_path)
    context = dict(context or {})
    species_by_fingerprint = {
        item.get("graph_fingerprint"): item
        for item in molecule_store.list_molecules(library_root)
        if item.get("graph_fingerprint")
    }

    created_species = []
    for component in components:
        if int(component.get("atoms", 0)) < 2:
            component["species_id"] = None
            continue
        created = _resolve_species(
            component, recording_path, final_index, recorder, context,
            library_root, species_by_fingerprint, allow_create=True,
        )
        if created and component.get("species_id"):
            created_species.append(str(component["species_id"]))

    atom_ids = recorder.atom_ids_at(final_index)
    before_edges = _initial_id_edges(reactants, atom_ids)
    after_edges = set().union(
        *(set(c.get("id_bonds", ())) for c in components)
    ) if components else set()
    formed = sorted(after_edges - before_edges)
    broken = sorted(before_edges - after_edges)
    if not formed and not broken:
        return {"status": "no reaction", "event": None, "created_species": created_species}

    final_time = float(recorder.times[final_index])
    event_id = _event_id(recording_key, final_time, before_edges, after_edges)
    symbols_by_id = {
        int(atom_id): str(symbol)
        for atom_id, symbol in zip(
            recorder.atom_ids_at(final_index), recorder.symbols_at(final_index)
        )
    }

    def bond_records(pairs):
        return [{
            "atom_ids": [int(a), int(b)],
            "symbols": [symbols_by_id.get(int(a), "?"), symbols_by_id.get(int(b), "?")],
        } for a, b in pairs]

    event = {
        "event_id": event_id,
        "event_kind": "controlled_final_state",
        "recording": recording_key,
        "batch": context.get("batch"),
        "seed": context.get("seed"),
        "mixture": context.get("mixture"),
        "time_fs": final_time,
        "previous_frame_time_fs": float(recorder.times[0]) if len(recorder) > 1 else None,
        "temperature_K": float(recorder.temperature[final_index]),
        "box_A": float(recorder.box_at(final_index)),
        "reactants": _reactant_descriptors(reactants),
        "products": _combine_descriptors(components),
        "formed_bonds": bond_records(formed),
        "broken_bonds": bond_records(broken),
        "frame_species": _all_species_counts(components),
        "controlled_start_graph": True,
    }
    if event_id not in _load_existing_event_ids(library_root):
        _append_events([event], library_root)
    return {
        "status": "reaction",
        "event": event,
        "created_species": sorted(created_species),
    }

def scan_recordings(runs_root="runs", library_root=molecule_store.DEFAULT_ROOT,
                    progress=None):
    """Incrementally scan all indexed recordings under runs_root."""

    os.makedirs(library_root, exist_ok=True)
    manifest = load_manifest(library_root)
    recordings_state = manifest.setdefault("recordings", {})

    recordings = _discover_recordings(runs_root)
    species_by_fingerprint = {
        item.get("graph_fingerprint"): item
        for item in molecule_store.list_molecules(library_root)
        if item.get("graph_fingerprint")
    }
    existing_event_ids = _load_existing_event_ids(library_root)

    summary = {
        "recordings_found": len(recordings),
        "scanned": 0,
        "unchanged": 0,
        "legacy": 0,
        "unstable": 0,
        "empty": 0,
        "errors": [],
        "frames_counted": 0,
        "new_species": 0,
        "formation_events": 0,
    }

    for number, recording in enumerate(recordings, start=1):
        key = recording["key"]

        if progress:
            progress({
                "stage": "recording",
                "recording": key,
                "number": number,
                "total": len(recordings),
            })

        try:
            inherited = None
            parent_key = recording.get("parent_key")

            if key not in recordings_state and parent_key:
                parent_state = recordings_state.get(parent_key)
                if parent_state and parent_state.get("status") == "scanned":
                    inherited = parent_state

            outcome = _scan_one(
                recording,
                recordings_state.get(key),
                inherited,
                library_root,
                species_by_fingerprint,
                existing_event_ids,
                progress=progress,
            )

            recordings_state[key] = outcome["manifest"]
            result = outcome["result"]
            status = result.get("status", "error")

            if status in summary and isinstance(summary[status], int):
                summary[status] += 1

            summary["frames_counted"] += int(result.get("frames_counted", 0))
            summary["new_species"] += int(result.get("new_species", 0))
            summary["formation_events"] += int(result.get("formation_events", 0))

        except Exception as problem:
            current_stat = None
            try:
                current_stat = _file_stat(recording["path"])
            except OSError:
                current_stat = {"size": None, "mtime_ns": None}

            recordings_state[key] = {
                "status": "error",
                "error": str(problem),
                "file_size": current_stat.get("size"),
                "file_mtime_ns": current_stat.get("mtime_ns"),
                "last_scan_unix": time.time(),
                "seed": recording["entry"].get("seed"),
                "mixture": recording["entry"].get("mixture"),
                "batch": recording.get("batch"),
            }
            summary["errors"].append(f"{key}: {problem}")

        save_manifest(manifest, library_root)

    summary["library_species"] = len(molecule_store.list_molecules(library_root))
    return summary
