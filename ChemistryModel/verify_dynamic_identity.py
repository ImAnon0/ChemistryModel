"""
Offline sanity check after applying the dynamic identity patch.

Run from ChemistryModel:
    py verify_dynamic_identity.py
"""

import os
import tempfile
import numpy as np

from recorder import Recorder
from bonding import BondTracker
import reactive as R


def main():
    symbols = ["H", "H", "C"]
    positions = np.array([
        [1.0, 1.0, 1.0],
        [1.7, 1.0, 1.0],
        [7.0, 7.0, 7.0],
    ])

    recorder = Recorder(symbols, 10.0)

    ids0 = np.array([0, 1, 2], dtype=np.uint32)
    recorder.capture(
        positions, 0.0, -1.0, 0.5, 250.0,
        symbols=symbols, atom_ids=ids0,
    )

    # Slot 0 is still H, but it is now a NEW hydrogen.
    ids1 = np.array([3, 1, 2], dtype=np.uint32)
    recorder.capture(
        positions, 10.0, -1.0, 0.5, 250.0,
        symbols=symbols, atom_ids=ids1,
    )

    assert int(recorder.atom_ids_at(0)[0]) == 0
    assert int(recorder.atom_ids_at(1)[0]) == 3

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "identity_test.npz")
        recorder.save(path)
        loaded = Recorder.load(path)

        assert loaded.has_atom_history
        assert int(loaded.atom_ids_at(0)[0]) == 0
        assert int(loaded.atom_ids_at(1)[0]) == 3

    types = R.types_from_symbols(symbols)
    tracker = BondTracker(
        types, formation_time=1.0, atom_ids=ids0
    )

    tracker.update(
        positions, 10.0, 0.0,
        types=types, atom_ids=ids0,
    )
    tracker.update(
        positions, 10.0, 2.0,
        types=types, atom_ids=ids0,
    )

    before = set(
        (int(a), int(b))
        for a, b in zip(*tracker.confirmed_now())
    )
    assert (0, 1) in before

    tracker.update(
        positions, 10.0, 3.0,
        types=types, atom_ids=ids1,
    )

    after = set(
        (int(a), int(b))
        for a, b in zip(*tracker.confirmed_now())
    )
    assert (0, 1) not in after

    print("PASS: per-frame atom IDs survive save/load")
    print("PASS: H -> H replacement is recognised as a new atom")
    print("PASS: the new atom does not inherit old bond persistence")


if __name__ == "__main__":
    main()
