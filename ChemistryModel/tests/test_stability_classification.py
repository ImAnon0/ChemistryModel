import os
import tempfile

import numpy as np

import analysis
from recorder import Recorder


def recording(energies, temperatures=None):
    recorder = Recorder(["H"], 10.0)
    temperatures = temperatures or [300.0] * len(energies)
    for index, (energy, temperature) in enumerate(zip(energies, temperatures)):
        recorder.capture(
            np.zeros((1, 3)), index * 10.0, energy, 0.0, temperature,
            velocities=np.zeros((1, 3)),
        )
    return recorder


def test_declared_strike_jump_is_external_and_round_trips():
    recorder = recording([-100.0, 20.0])
    recorder.record_external_event(
        0.0, "strike", deposited_eV=120.0, struck_atoms=1
    )
    health = analysis.classify_stability(recorder)
    assert health["stable"]
    assert health["external_energy_injections"] == 1
    assert health["declared_external_energy_events"] == 1
    assert health["total_declared_external_energy_eV"] == 120.0
    assert health["spontaneous_energy_jumps"] == 0
    assert health["largest_external_energy_injection"] == 120.0

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "fixed.npz")
        recorder.save(path)
        loaded = Recorder.load(path)
        assert loaded.events == recorder.events
        assert analysis.classify_stability(loaded)["stable"]


def test_same_unexplained_jump_remains_unstable():
    health = analysis.classify_stability(recording([-100.0, 20.0]))
    assert not health["stable"]
    assert health["spontaneous_energy_jumps"] == 1
    assert health["external_energy_injections"] == 0


def test_event_at_next_frame_does_not_hide_preceding_jump():
    recorder = recording([-100.0, 20.0, 140.0])
    recorder.record_external_event(10.0, "strike", deposited_eV=120.0)
    health = analysis.classify_stability(recorder)
    assert health["spontaneous_energy_jumps"] == 1
    assert health["external_energy_injections"] == 1


def test_nan_is_a_numerical_failure():
    health = analysis.classify_stability(
        recording([-100.0, np.nan], [300.0, np.nan])
    )
    assert not health["stable"]
    assert health["numerical_failures"] == 1


def test_no_strike_behavior_is_unchanged():
    healthy = analysis.classify_stability(recording([-100.0, -99.0]))
    bad = analysis.classify_stability(recording([-100.0, 20.0]))
    assert healthy["stable"] and healthy["spontaneous_energy_jumps"] == 0
    assert not bad["stable"] and bad["spontaneous_energy_jumps"] == 1


def test_move_caps_remain_an_independent_summary_field():
    source = open("batch_runner.py", encoding="utf-8").read()
    assert '"move_cap_events": move_cap_events' in source
    assert '"external_energy_injections"' in source


if __name__ == "__main__":
    for name, value in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("PASS  stability classification")
