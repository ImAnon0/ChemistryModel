"""Compatibility and parallel-array invariants for trajectory recordings."""

import os
import tempfile

import numpy as np

from recorder import AdaptiveRecorder, Recorder


def frame(recorder, index, box=10.0, atom_id_offset=0):
    recorder.capture(
        np.array([[1 + index, 2, 3], [4, 5, 6]], dtype=float) % box,
        index * 2.5, -2 + index, 1 + index, 300 + index,
        velocities=np.full((2, 3), index, dtype=float), box_size=box,
        symbols=["H", "O"],
        atom_ids=np.array([atom_id_offset, atom_id_offset + 1]),
    )


def temporary_npz(**fields):
    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    np.savez_compressed(path, **fields)
    return path


def test_unversioned_legacy_recording_loads_with_fallbacks():
    path = temporary_npz(
        symbols=np.array(["H", "O"]), box_size=10.0,
        positions=np.zeros((2, 2, 3), dtype=np.float32),
        times=np.array([0.0, 2.5]), potential=np.array([-2.0, -1.0]),
        kinetic=np.array([1.0, 2.0]), temperature=np.array([300.0, 301.0]),
    )
    try:
        loaded = Recorder.load(path)
        assert loaded.format_version == 1
        assert loaded.frame_kinds == [0, 0]
        assert loaded.symbols_at(1) == ["H", "O"]
        assert loaded.atom_ids_at(1).tolist() == [0, 1]
        assert loaded.box_at(1) == 10.0
    finally:
        os.unlink(path)


def test_version_two_round_trip_preserves_continuation_fields():
    recorder = Recorder(["H", "O"], 10.0)
    recorder.format_version = 2
    frame(recorder, 0)
    frame(recorder, 1, box=11.0, atom_id_offset=7)
    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    try:
        recorder.save(path)
        loaded = Recorder.load(path)
        assert loaded.format_version == 2
        assert loaded.frame_kinds == [0, 0]
        assert loaded.event_reasons == ["", ""]
        assert loaded.has_velocities and loaded.has_atom_history
        assert loaded.atom_ids_at(1).tolist() == [7, 8]
        assert loaded.box_at(1) == 11.0
        np.testing.assert_array_equal(loaded.positions, recorder.positions)
    finally:
        os.unlink(path)


def test_thinning_keeps_all_parallel_arrays_aligned():
    recorder = Recorder(["H", "O"], 10.0, maximum_frames=3)
    for index in range(8):
        frame(recorder, index)
    lengths = {
        len(values) for values in (
            recorder.positions, recorder.velocities, recorder.frame_types,
            recorder.frame_atom_ids, recorder.box_sizes, recorder.times,
            recorder.potential, recorder.kinetic, recorder.temperature,
            recorder.frame_kinds,
            recorder.event_reasons,
        )
    }
    assert lengths == {len(recorder)}


def test_adaptive_recorder_preserves_event_window_and_order():
    recorder = AdaptiveRecorder(
        ["H", "O"], 10.0, ordinary_interval_fs=10,
        pre_event_fs=6, post_event_fs=4, energy_jump_ev=5,
        detect_chemical_events=False,
    )
    for index in range(11):
        energy = 10.0 if index == 6 else 0.0
        recorder.observe(
            np.array([[index % 10, 0, 0], [1, 1, 1]], dtype=float),
            index * 2.0, energy, 0.0, 300.0,
        )
    assert recorder.times == sorted(set(recorder.times))
    event = recorder.frame_kinds.index(AdaptiveRecorder.KIND_EVENT)
    assert recorder.times[event] == 12.0
    assert recorder.times[event - 1] == 10.0
    assert 14.0 in recorder.times and 16.0 in recorder.times
    assert recorder.event_reasons[event].startswith("total_energy_jump")


def test_adaptive_thinning_preserves_events_and_reports_loss():
    recorder = AdaptiveRecorder(
        ["H", "O"], 10.0, maximum_frames=8,
        ordinary_interval_fs=2, pre_event_fs=4, post_event_fs=4,
        energy_jump_ev=5, detect_chemical_events=False,
    )
    for index in range(20):
        recorder.observe(
            np.array([[index % 10, 0, 0], [1, 1, 1]], dtype=float),
            index * 2.0, 10.0 if index == 10 else 0.0, 0.0, 300.0,
        )
    assert len(recorder) <= recorder.maximum_frames
    assert recorder.adaptive_dropped_frames > 0
    event_times = [
        recorder.times[index] for index, kind in enumerate(recorder.frame_kinds)
        if kind == AdaptiveRecorder.KIND_EVENT
    ]
    assert 20.0 in event_times and 22.0 in event_times
    assert recorder.times == sorted(recorder.times)


def test_authoritative_bond_event_round_trips():
    recorder = AdaptiveRecorder(
        ["H", "H"], 10.0, ordinary_interval_fs=10,
        pre_event_fs=2, post_event_fs=2, energy_jump_ev=1000,
    )
    recorder.observe(np.array([[1, 1, 1], [4, 1, 1]]), 0, 0, 0, 300)
    recorder.observe(np.array([[1, 1, 1], [1.7, 1, 1]]), 2, 0, 0, 300)
    assert recorder.events
    assert recorder.events[-1]["formed"] == [[0, 1]]
    handle, path = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    try:
        recorder.save(path)
        loaded = Recorder.load(path)
        assert loaded.events == recorder.events
    finally:
        os.unlink(path)


def test_compact_simulation_observation_matches_fallback():
    from reactive_torch import ReactiveSimulation

    simulation = ReactiveSimulation(
        ["H", "H"], np.array([[1, 1, 1], [1.7, 1, 1]], dtype=float),
        10.0, time_step=0.25, target_temperature=300,
        friction=0.0, device="cpu", relax_on_start=False,
    )
    positions = simulation.positions_numpy
    compact = simulation.chemical_observation()
    fast = AdaptiveRecorder(["H", "H"], 10.0)
    fallback = AdaptiveRecorder(["H", "H"], 10.0)
    candidate = fast._candidate(positions, 0, 0, 0, 300)
    assert fast._chemical_state(candidate, compact) == fallback._chemical_state(candidate)


def test_chemical_observation_rejects_replaced_position_tensor():
    from reactive_torch import ReactiveSimulation

    simulation = ReactiveSimulation(
        ["H", "H"], np.array([[1, 1, 1], [1.7, 1, 1]], dtype=float),
        10.0, time_step=0.25, target_temperature=300,
        friction=0.0, device="cpu", relax_on_start=False,
    )
    simulation.positions = simulation.positions.clone()
    try:
        simulation.chemical_observation()
    except RuntimeError:
        pass
    else:
        raise AssertionError("replaced position tensor accepted stale pair cache")


def test_reaction_burst_has_one_protected_window_but_keeps_events():
    recorder = AdaptiveRecorder(
        ["H", "H"], 10.0, ordinary_interval_fs=10,
        reaction_window_fs=20, detect_chemical_events=False,
    )
    for index in range(6):
        recorder.observe(
            np.array([[1, 1, 1], [3, 1, 1]]), index * 2.0,
            0, 0, 300, event_reason="bond_change",
        )
    assert len(recorder.events) == 6
    assert {event["window_id"] for event in recorder.events} == {0}
    assert sum(kind == AdaptiveRecorder.KIND_EVENT for kind in recorder.frame_kinds) == 1
    assert sum(kind == AdaptiveRecorder.KIND_POST_EVENT for kind in recorder.frame_kinds) == 5


def test_chemical_context_is_shorter_than_failure_context():
    recorder = AdaptiveRecorder(
        ["H", "H"], 10.0, ordinary_interval_fs=100,
        pre_event_fs=50, post_event_fs=50,
        chemical_context_fs=2, detect_chemical_events=False,
    )
    for time_fs in range(0, 11, 2):
        recorder.observe(
            np.array([[1, 1, 1], [3, 1, 1]]), time_fs,
            0, 0, 300,
            event_reason="bond_change" if time_fs == 10 else None,
        )
    recorder.observe(
        np.array([[1, 1, 1], [3, 1, 1]]), 12, 0, 0, 300
    )
    assert recorder.times == [0.0, 8.0, 10.0, 12.0]


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS  {name}")
            except Exception as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    print(f"\n{10 - failures} passed, {failures} failed")
    raise SystemExit(bool(failures))
