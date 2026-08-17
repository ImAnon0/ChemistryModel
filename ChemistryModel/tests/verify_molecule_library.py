
import _bootstrap  # noqa: F401 - direct-execution project path
import json
import os
import sys
import tempfile

import numpy as np

import molecule_library as M
from recorder import Recorder


def find_recording(root="runs"):
    """Find the first recording that has modern per-frame identity."""

    for directory, _, files in os.walk(root):
        if "index.json" not in files:
            continue

        path = os.path.join(directory, "index.json")

        try:
            with open(path) as handle:
                index = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        for entry in index:
            filename = entry.get("file")

            if not filename:
                continue

            candidate = os.path.join(directory, filename)

            if not os.path.isfile(candidate):
                continue

            try:
                recorder = Recorder.load(candidate)
            except Exception:
                continue

            if len(recorder) and recorder.has_atom_history:
                return candidate

    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_recording()

    if not path:
        print(
            "SKIP - no recording with per-frame atom identity history was "
            "found under runs/. No molecule data was extracted."
        )
        return

    print(f"recording: {path}")

    recorder = Recorder.load(path)

    if len(recorder) == 0:
        raise SystemExit("recording is empty")

    frame = len(recorder) - 1

    print(
        f"frames: {len(recorder)}  final: {float(recorder.times[frame]):.1f} fs  "
        f"identity history: {recorder.has_atom_history}"
    )

    if not recorder.has_atom_history:
        print(
            "SKIP - this is a legacy recording with no per-frame atom "
            "identity history. No molecule data was extracted."
        )
        return

    molecules = M.molecules_at(recorder, frame)

    if not molecules:
        raise SystemExit("no components found in final frame")

    chosen = molecules[0]

    print(
        f"largest/heaviest: {chosen['formula']}  "
        f"{chosen['atoms']} atoms  {chosen['heavy_atoms']} heavy  "
        f"{len(chosen['bonds'])} bonds"
    )

    with tempfile.TemporaryDirectory(prefix="chem_molecule_test_") as root:
        metadata = M.save_component(
            path, frame, chosen, root=root, note="verification",
            recorder=recorder,
        )

        loaded = M.load_molecule(metadata["id"], root=root)

        assert len(loaded["symbols"]) == chosen["atoms"]
        assert loaded["positions"].shape == (chosen["atoms"], 3)
        assert loaded["bonds"].shape[1] == 2
        assert len(loaded["source_atom_ids"]) == chosen["atoms"]
        assert np.all(np.isfinite(loaded["positions"]))

        centre = np.mean(loaded["positions"], axis=0)

        if not np.allclose(centre, 0.0, atol=1e-4):
            raise AssertionError(f"stored molecule is not centred: {centre}")

        for a, b in loaded["bonds"]:
            if not (0 <= int(a) < chosen["atoms"]):
                raise AssertionError("bond index A outside molecule")
            if not (0 <= int(b) < chosen["atoms"]):
                raise AssertionError("bond index B outside molecule")

        listed = M.list_molecules(root=root)

        assert len(listed) == 1
        assert listed[0]["id"] == metadata["id"]

        print(f"saved/reloaded: {metadata['id']}")
        print(f"fingerprint: {metadata['graph_fingerprint']}")

    print("PASS - isolation/library round trip works")


if __name__ == "__main__":
    main()
