"""Regression tests for timestamp-driven, presentation-only replay."""

import numpy as np

from recorder import Recorder
from replay import ReplayClock


def make_recorder(first, second, ids=None, types=None, times=(0.0, 10.0)):
    recorder = Recorder(["H", "H"], 12.0)
    ids = ids or (
        np.array([0, 1], dtype=np.uint32),
        np.array([0, 1], dtype=np.uint32),
    )
    types = types or (
        np.array([0, 0], dtype=np.uint8),
        np.array([0, 0], dtype=np.uint8),
    )
    for index, positions in enumerate((first, second)):
        recorder.capture(
            np.asarray(positions, dtype=float), times[index],
            potential=-index, kinetic=index, temperature=300 + index,
            types=types[index], atom_ids=ids[index], box_size=12.0,
        )
    return recorder


def test_irregular_timestamps_choose_correct_interval():
    recorder = make_recorder(
        [[1, 1, 1], [2, 2, 2]], [[3, 1, 1], [2, 2, 2]],
        times=(5.0, 25.0),
    )
    clock = ReplayClock(recorder)
    clock.seek_time(10.0)
    assert clock.frame_interval() == (0, 1, 0.25)


def test_periodic_motion_takes_minimum_image_path():
    recorder = make_recorder(
        [[11.8, 1, 1], [2, 2, 2]], [[0.2, 1, 1], [2, 2, 2]],
    )
    clock = ReplayClock(recorder)
    clock.seek_time(5.0)
    midpoint = clock.interpolated_positions()[0, 0]
    assert min(abs(midpoint), abs(midpoint - 12.0)) < 1e-5


def test_replaced_atom_is_not_interpolated():
    recorder = make_recorder(
        [[1, 1, 1], [2, 2, 2]], [[9, 9, 9], [2, 2, 2]],
        ids=(
            np.array([4, 5], dtype=np.uint32),
            np.array([8, 5], dtype=np.uint32),
        ),
    )
    clock = ReplayClock(recorder)
    clock.seek_time(9.0)
    assert np.array_equal(clock.interpolated_positions()[0], [1, 1, 1])
    clock.seek_time(10.0)
    assert np.array_equal(clock.interpolated_positions()[0], [9, 9, 9])


def test_step_moves_between_real_recorded_frames():
    recorder = make_recorder(
        [[1, 1, 1], [2, 2, 2]], [[3, 1, 1], [2, 2, 2]],
    )
    clock = ReplayClock(recorder)
    clock.seek_time(4.0)
    clock.step_frame(1)
    assert clock.time_fs == 10.0
    clock.step_frame(-1)
    assert clock.time_fs == 0.0


if __name__ == "__main__":
    tests = [
        test_irregular_timestamps_choose_correct_interval,
        test_periodic_motion_takes_minimum_image_path,
        test_replaced_atom_is_not_interpolated,
        test_step_moves_between_real_recorded_frames,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} passed, 0 failed")
