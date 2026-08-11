import hashlib
import json
import os
import time
import zipfile

import numpy as np

import bonding
from recorder import Recorder


# ============================================================
# Molecules captured from reactive trajectories
# ============================================================
#
# The library is deliberately dumb about chemistry rules. It stores
# structures that actually existed in a reactive trajectory, plus the
# evidence gathered about where/when they were seen. The scanner and
# future characterisation runner both use this module, so molecule
# identity and storage stay in one place.


DEFAULT_ROOT = "molecules"
EVENT_LOG = "formation_events.jsonl"


def require_identity_history(recorder):
    """Reject trajectories that cannot prove per-frame atom identity."""

    if not bool(getattr(recorder, "has_atom_history", False)):
        raise ValueError(
            "molecule extraction requires per-frame atom identity history; "
            "this is a legacy recording and no molecule data will be extracted"
        )

    return True


def _ensure_root(root):
    os.makedirs(root, exist_ok=True)
    return root


def _json_write_atomic(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

    os.replace(temporary, path)


def _next_id(root):
    _ensure_root(root)
    highest = 0

    for name in os.listdir(root):
        if not (name.startswith("SP_") and name.endswith(".json")):
            continue

        try:
            highest = max(highest, int(name[3:-5]))
        except ValueError:
            continue

    return f"SP_{highest + 1:06d}"


def _formula(symbols):
    counts = {}

    for symbol in symbols:
        symbol = str(symbol)
        counts[symbol] = counts.get(symbol, 0) + 1

    ordered = ["C", "N", "O", "H"]
    extras = sorted(symbol for symbol in counts if symbol not in ordered)

    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in ordered + extras
        if symbol in counts
    )


def _heavy_count(symbols):
    return sum(1 for symbol in symbols if str(symbol) != "H")


def _components(count, first, second):
    neighbours = [[] for _ in range(count)]

    for a, b in zip(first, second):
        a = int(a)
        b = int(b)
        neighbours[a].append(b)
        neighbours[b].append(a)

    seen = np.zeros(count, dtype=bool)
    found = []

    for start in range(count):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        members = []

        while stack:
            current = stack.pop()
            members.append(current)

            for neighbour in neighbours[current]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)

        found.append(np.array(sorted(members), dtype=np.int32))

    return found


def _unwrap_component(positions, members, bonds, box_size):
    """Make a bonded molecule contiguous across periodic boundaries."""

    members = np.asarray(members, dtype=np.int32)
    raw = np.asarray(positions, dtype=np.float64)[members]

    if len(raw) <= 1:
        result = raw.copy()
        result -= np.mean(result, axis=0) if len(result) else 0.0
        return result

    neighbours = [[] for _ in range(len(raw))]

    for a, b in np.asarray(bonds, dtype=np.int32).reshape(-1, 2):
        a = int(a)
        b = int(b)
        neighbours[a].append(b)
        neighbours[b].append(a)

    unwrapped = np.zeros_like(raw)
    placed = np.zeros(len(raw), dtype=bool)
    unwrapped[0] = raw[0]
    placed[0] = True
    stack = [0]

    while stack:
        current = stack.pop()

        for neighbour in neighbours[current]:
            if placed[neighbour]:
                continue

            delta = raw[neighbour] - raw[current]
            delta -= box_size * np.round(delta / box_size)

            unwrapped[neighbour] = unwrapped[current] + delta
            placed[neighbour] = True
            stack.append(neighbour)

    # A malformed payload should not be able to crash storage. A proper
    # connected component never reaches this fallback.
    for index in np.where(~placed)[0]:
        delta = raw[index] - raw[0]
        delta -= box_size * np.round(delta / box_size)
        unwrapped[index] = unwrapped[0] + delta

    unwrapped -= np.mean(unwrapped, axis=0)
    return unwrapped


def _graph_hash(symbols, bonds):
    """Element-labelled graph fingerprint independent of atom ordering.

    This deliberately remains a structural fingerprint, not a claim of
    complete chemical identity. Bond order, formal charge, radicals and
    stereochemistry can be layered on later without changing the SP IDs
    already stored.
    """

    symbols = [str(symbol) for symbol in symbols]
    bonds = np.asarray(bonds, dtype=np.int32).reshape(-1, 2)

    neighbours = [[] for _ in symbols]

    for a, b in bonds:
        a = int(a)
        b = int(b)
        neighbours[a].append(b)
        neighbours[b].append(a)

    labels = [f"{symbol}:{len(neighbours[i])}" for i, symbol in enumerate(symbols)]

    for _ in range(max(1, len(symbols))):
        refined = []

        for i, label in enumerate(labels):
            neighbourhood = ",".join(sorted(labels[j] for j in neighbours[i]))
            text = f"{label}|{neighbourhood}"
            refined.append(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])

        labels = refined

    payload = "|".join(sorted(labels)) + f"|edges:{len(bonds)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_fingerprint(symbols, bonds):
    return _graph_hash(symbols, bonds)


def component_records(recorder, frame_index, bond_first, bond_second,
                      include_single=True):
    """Build connected-component records from already-confirmed bonds.

    The scanner calls this once per frame after advancing one BondTracker.
    That is intentionally different from molecules_at(), which is the
    convenient one-frame public helper and therefore replays persistence.
    """

    require_identity_history(recorder)

    frame_index = int(frame_index)
    count = len(recorder.positions[frame_index])
    frame_symbols = recorder.symbols_at(frame_index)
    frame_ids = recorder.atom_ids_at(frame_index)

    first = np.asarray(bond_first, dtype=int)
    second = np.asarray(bond_second, dtype=int)
    components = _components(count, first, second)
    records = []

    for number, members in enumerate(components):
        if not include_single and len(members) < 2:
            continue

        member_set = set(int(slot) for slot in members)
        local = {int(slot): i for i, slot in enumerate(members)}
        symbols = [frame_symbols[int(slot)] for slot in members]
        atom_ids = np.asarray(
            [frame_ids[int(slot)] for slot in members], dtype=np.uint32
        )

        local_bonds = []
        id_bonds = []

        for a, b in zip(first, second):
            a = int(a)
            b = int(b)

            if a not in member_set or b not in member_set:
                continue

            local_bonds.append((local[a], local[b]))

            id_a = int(frame_ids[a])
            id_b = int(frame_ids[b])
            id_bonds.append((min(id_a, id_b), max(id_a, id_b)))

        bonds = np.asarray(local_bonds, dtype=np.int32).reshape(-1, 2)

        records.append({
            "component": number,
            "members": members,
            "symbols": symbols,
            "atom_ids": atom_ids,
            "bonds": bonds,
            "id_bonds": tuple(sorted(id_bonds)),
            "formula": _formula(symbols),
            "atoms": len(members),
            "heavy_atoms": _heavy_count(symbols),
            "graph_fingerprint": _graph_hash(symbols, bonds),
        })

    records.sort(
        key=lambda item: (
            item["heavy_atoms"], item["atoms"], item["formula"]
        ),
        reverse=True,
    )

    return records


def confirmed_bonds_at(recorder, frame_index, threshold=0.35):
    """Replay the normal persistence tracker up to one stored frame."""

    require_identity_history(recorder)

    if len(recorder) == 0:
        raise ValueError("cannot inspect an empty recording")

    frame_index = int(frame_index)

    if frame_index < 0:
        frame_index += len(recorder)

    if frame_index < 0 or frame_index >= len(recorder):
        raise IndexError("frame index outside recording")

    tracker = bonding.BondTracker(
        recorder.types_at(0),
        atom_ids=recorder.atom_ids_at(0),
        threshold=threshold,
    )

    first = np.array([], dtype=int)
    second = np.array([], dtype=int)

    for index in range(frame_index + 1):
        first, second = tracker.update(
            recorder.positions[index],
            recorder.box_at(index),
            float(recorder.times[index]),
            types=recorder.types_at(index),
            atom_ids=recorder.atom_ids_at(index),
        )

    return np.asarray(first, dtype=int), np.asarray(second, dtype=int)


def molecules_at(recorder, frame_index=-1, threshold=0.35):
    """Return connected objects present in one recorded frame."""

    frame_index = int(frame_index)

    if frame_index < 0:
        frame_index += len(recorder)

    first, second = confirmed_bonds_at(recorder, frame_index, threshold)
    return component_records(recorder, frame_index, first, second)


def _empty_stats():
    return {
        "observations": 0,
        "formations": 0,
        "appearances": 0,
        "runs_seen": 0,
        "longest_observed_lifetime_fs": 0.0,
    }


def _normalise_metadata(metadata):
    metadata = dict(metadata)
    metadata.setdefault("stats", _empty_stats())

    for key, value in _empty_stats().items():
        metadata["stats"].setdefault(key, value)

    metadata.setdefault("sources", {})
    return metadata


def save_metadata(metadata, root=DEFAULT_ROOT):
    metadata = _normalise_metadata(metadata)
    molecule_id = metadata["id"]
    stored = dict(metadata)
    stored.pop("metadata_path", None)
    path = os.path.join(_ensure_root(root), f"{molecule_id}.json")
    _json_write_atomic(path, stored)
    return stored


def _save_new_component(recording_path, frame_index, selected,
                        root=DEFAULT_ROOT, note="", recorder=None,
                        source_context=None):
    if recorder is None:
        recorder = Recorder.load(recording_path)

    require_identity_history(recorder)

    members = np.asarray(selected["members"], dtype=np.int32)
    bonds = np.asarray(selected["bonds"], dtype=np.int32).reshape(-1, 2)
    frame_index = int(frame_index)

    if frame_index < 0:
        frame_index += len(recorder)

    positions = _unwrap_component(
        recorder.positions[frame_index],
        members,
        bonds,
        recorder.box_at(frame_index),
    ).astype(np.float32)

    velocities = None

    if recorder.has_velocities:
        velocities = np.asarray(
            recorder.velocities[frame_index], dtype=np.float32
        )[members].copy()
        velocities -= np.mean(velocities, axis=0)

    molecule_id = _next_id(root)
    root = _ensure_root(root)
    npz_name = f"{molecule_id}.npz"
    npz_path = os.path.join(root, npz_name)

    payload = {
        "symbols": np.asarray(selected["symbols"], dtype="U2"),
        "positions": positions,
        "bonds": bonds,
        "source_atom_ids": np.asarray(selected["atom_ids"], dtype=np.uint32),
        "source_slots": members,
    }

    if velocities is not None:
        payload["velocities"] = velocities

    np.savez_compressed(npz_path, **payload)

    context = dict(source_context or {})
    source = {
        "recording": os.path.normpath(recording_path),
        "frame": frame_index,
        "time_fs": float(recorder.times[frame_index]),
        "temperature_K": float(recorder.temperature[frame_index]),
        "box_A": float(recorder.box_at(frame_index)),
        "atom_ids": [int(value) for value in selected["atom_ids"]],
        "identity_history": True,
    }

    for key in ("seed", "mixture", "batch"):
        if context.get(key) is not None:
            source[key] = context[key]

    metadata = {
        "id": molecule_id,
        "formula": selected["formula"],
        "atoms": int(selected["atoms"]),
        "heavy_atoms": int(selected["heavy_atoms"]),
        "graph_fingerprint": selected.get(
            "graph_fingerprint", _graph_hash(selected["symbols"], bonds)
        ),
        "payload": npz_name,
        "source": source,
        "stats": _empty_stats(),
        "sources": {},
        "note": str(note or ""),
        "saved_unix": time.time(),
    }

    return save_metadata(metadata, root=root)


def save_component(recording_path, frame_index, component,
                   root=DEFAULT_ROOT, note="", recorder=None):
    """Save one selected component without deduplicating it.

    Kept for compatibility with the first manual isolator and verifier.
    Automatic scanning uses ensure_species(), which deduplicates by the
    structural fingerprint instead.
    """

    if recorder is None:
        recorder = Recorder.load(recording_path)

    require_identity_history(recorder)

    if isinstance(component, dict):
        selected = component
    else:
        molecules = molecules_at(recorder, frame_index)

        if not molecules:
            raise ValueError("no molecules found in that frame")

        component = int(component)

        if component < 0 or component >= len(molecules):
            raise IndexError("molecule selection outside frame")

        selected = molecules[component]

    return _save_new_component(
        recording_path, frame_index, selected,
        root=root, note=note, recorder=recorder,
    )


def list_molecules(root=DEFAULT_ROOT):
    _ensure_root(root)
    found = []

    for name in sorted(os.listdir(root)):
        if not (name.startswith("SP_") and name.endswith(".json")):
            continue

        path = os.path.join(root, name)

        try:
            with open(path, encoding="utf-8") as handle:
                item = _normalise_metadata(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue

        item["metadata_path"] = path
        found.append(item)

    def number(item):
        try:
            return int(str(item.get("id", "SP_0")).split("_")[-1])
        except ValueError:
            return 0

    found.sort(key=number)
    return found


def molecule_by_fingerprint(fingerprint, root=DEFAULT_ROOT):
    for item in list_molecules(root):
        if item.get("graph_fingerprint") == fingerprint:
            return item

    return None


def ensure_species(recording_path, frame_index, component,
                   root=DEFAULT_ROOT, recorder=None, source_context=None):
    """Return one canonical SP record for a structural fingerprint."""

    fingerprint = component.get("graph_fingerprint") or _graph_hash(
        component["symbols"], component["bonds"]
    )

    existing = molecule_by_fingerprint(fingerprint, root=root)

    if existing is not None:
        return existing, False

    metadata = _save_new_component(
        recording_path,
        frame_index,
        component,
        root=root,
        recorder=recorder,
        source_context=source_context,
    )

    return metadata, True


def load_molecule(molecule_id, root=DEFAULT_ROOT):
    if isinstance(molecule_id, dict):
        metadata = _normalise_metadata(dict(molecule_id))
    else:
        path = os.path.join(root, f"{molecule_id}.json")

        with open(path, encoding="utf-8") as handle:
            metadata = _normalise_metadata(json.load(handle))

    payload_path = os.path.join(root, metadata["payload"])
    data = np.load(payload_path, allow_pickle=False)

    result = dict(metadata)
    result["symbols"] = [str(value) for value in data["symbols"]]
    result["positions"] = np.asarray(data["positions"], dtype=np.float32)
    result["bonds"] = np.asarray(data["bonds"], dtype=np.int32).reshape(-1, 2)
    result["source_atom_ids"] = np.asarray(
        data["source_atom_ids"], dtype=np.uint32
    )
    result["source_slots"] = np.asarray(data["source_slots"], dtype=np.int32)

    if "velocities" in data.files:
        result["velocities"] = np.asarray(data["velocities"], dtype=np.float32)
    else:
        result["velocities"] = None

    return result


def formation_events_for_species(molecule_id, root=DEFAULT_ROOT, limit=8):
    path = os.path.join(root, EVENT_LOG)

    if not os.path.isfile(path):
        return []

    found = []

    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                product_ids = [
                    item.get("id") for item in event.get("products", [])
                ]

                if molecule_id in product_ids:
                    found.append(event)
    except OSError:
        return []

    return found[-max(1, int(limit)):]


def delete_molecule(molecule_id, root=DEFAULT_ROOT):
    metadata_path = os.path.join(root, f"{molecule_id}.json")
    payload_path = None

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)

            payload_path = os.path.join(root, metadata.get("payload", ""))
        except (OSError, json.JSONDecodeError):
            pass

    if payload_path and os.path.isfile(payload_path):
        os.remove(payload_path)

    if os.path.isfile(metadata_path):
        os.remove(metadata_path)


def export_library(destination_path, root=DEFAULT_ROOT):
    """Export the complete molecule knowledge library as one portable zip.

    The archive contains every SP metadata JSON and NPZ geometry payload,
    plus the scan manifest and formation-event log when present. It also
    includes a compact export_manifest.json so the bundle can be inspected
    without first understanding the on-disk layout. Source trajectory NPZ
    files are deliberately not copied; their paths remain recorded as
    provenance in the metadata/event records.
    """

    root = _ensure_root(root)
    destination_path = os.path.abspath(os.path.expanduser(str(destination_path)))

    if not destination_path.lower().endswith(".zip"):
        destination_path += ".zip"

    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

    molecules = list_molecules(root)
    files = []

    for item in molecules:
        metadata_path = os.path.join(root, f"{item['id']}.json")
        payload_name = item.get("payload")
        payload_path = os.path.join(root, payload_name) if payload_name else None

        if os.path.isfile(metadata_path):
            files.append(metadata_path)
        if payload_path and os.path.isfile(payload_path):
            files.append(payload_path)

    for name in ("scan_manifest.json", EVENT_LOG):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            files.append(path)

    event_count = 0
    event_path = os.path.join(root, EVENT_LOG)
    if os.path.isfile(event_path):
        try:
            with open(event_path, encoding="utf-8") as handle:
                event_count = sum(1 for line in handle if line.strip())
        except OSError:
            event_count = 0

    manifest = {
        "format": "ChemistryModel molecule library export",
        "version": 2,
        "exported_unix": time.time(),
        "species_count": len(molecules),
        "formation_event_count": event_count,
        "species": [
            {
                "id": item.get("id"),
                "formula": item.get("formula"),
                "atoms": item.get("atoms"),
                "heavy_atoms": item.get("heavy_atoms"),
                "graph_fingerprint": item.get("graph_fingerprint"),
                "stats": item.get("stats", {}),
            }
            for item in molecules
        ],
        "notes": [
            "SP_*.json contains species metadata and provenance.",
            "SP_*.npz contains exact stored atom symbols, coordinates and bonds.",
            "formation_events.jsonl contains observed bond-change/reaction events.",
            "scan_manifest.json records which trajectory ranges have already been scanned.",
            "stats.appearances counts distinct appearance episodes/instances; this is the human-facing seen count.",
            "stats.observations counts molecule-frame samples (one count per simultaneous instance per stored frame), not independent sightings.",
            "Source trajectory files are not embedded in this export.",
        ],
    }

    with zipfile.ZipFile(
        destination_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            "molecules/export_manifest.json",
            json.dumps(manifest, indent=2),
        )

        for path in files:
            archive.write(path, arcname=os.path.join("molecules", os.path.basename(path)))

    return {
        "path": destination_path,
        "species": len(molecules),
        "events": event_count,
        "files": len(files) + 1,
    }
