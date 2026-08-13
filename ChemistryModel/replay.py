"""Timestamp-driven trajectory replay with presentation-only interpolation."""

from __future__ import annotations

import bisect
import time

import numpy as np


BASE_FEMTOSECONDS_PER_SECOND = 100.0


class ReplayClock:
    """Drive a Recorder by simulated time rather than recorded-frame FPS."""

    def __init__(self, recorder=None):
        self.recorder = None
        self.time_fs = 0.0
        self.speed = 1.0
        self.playing = False
        self._last_wall_time = time.perf_counter()
        if recorder is not None:
            self.load(recorder)

    def load(self, recorder):
        self.recorder = recorder
        self.time_fs = float(recorder.times[0]) if len(recorder) else 0.0
        self.playing = False
        self._last_wall_time = time.perf_counter()

    @property
    def frame_count(self):
        return len(self.recorder) if self.recorder is not None else 0

    @property
    def start_time(self):
        return float(self.recorder.times[0]) if self.frame_count else 0.0

    @property
    def end_time(self):
        return float(self.recorder.times[-1]) if self.frame_count else 0.0

    def seek_time(self, value):
        self.time_fs = float(np.clip(value, self.start_time, self.end_time))
        self._last_wall_time = time.perf_counter()

    def seek_fraction(self, fraction):
        self.seek_time(
            self.start_time
            + float(np.clip(fraction, 0.0, 1.0))
            * (self.end_time - self.start_time)
        )

    def seek_frame(self, index):
        if not self.frame_count:
            return
        index = int(np.clip(index, 0, self.frame_count - 1))
        self.seek_time(self.recorder.times[index])

    def step_frame(self, amount):
        lower, _, _ = self.frame_interval()
        self.seek_frame(lower + int(amount))

    def set_playing(self, playing):
        self.playing = bool(playing)
        self._last_wall_time = time.perf_counter()

    def advance(self, wall_time=None):
        now = time.perf_counter() if wall_time is None else float(wall_time)
        elapsed = max(now - self._last_wall_time, 0.0)
        self._last_wall_time = now
        if not self.playing or self.frame_count < 2:
            return
        self.time_fs += (
            elapsed * BASE_FEMTOSECONDS_PER_SECOND * float(self.speed)
        )
        if self.time_fs >= self.end_time:
            self.time_fs = self.end_time
            self.playing = False

    def frame_interval(self):
        if not self.frame_count:
            return 0, 0, 0.0
        times = self.recorder.times
        upper = bisect.bisect_right(times, self.time_fs)
        if upper <= 0:
            return 0, 0, 0.0
        if upper >= len(times):
            last = len(times) - 1
            return last, last, 0.0
        lower = upper - 1
        span = float(times[upper]) - float(times[lower])
        alpha = 0.0 if span <= 0 else (self.time_fs - times[lower]) / span
        return lower, upper, float(np.clip(alpha, 0.0, 1.0))

    def interpolated_positions(self):
        lower, upper, alpha = self.frame_interval()
        first = np.asarray(self.recorder.positions[lower], dtype=float)
        if lower == upper or alpha <= 0.0:
            return first.copy()
        second = np.asarray(self.recorder.positions[upper], dtype=float)
        first_box = float(self.recorder.box_at(lower))
        second_box = float(self.recorder.box_at(upper))
        box = first_box + alpha * (second_box - first_box)
        delta = second - first
        delta -= box * np.round(delta / box)
        result = (first + alpha * delta) % box

        # A replaced slot is a new atom, not a particle that travelled across
        # the cell between captures. Hold the old atom until the real frame.
        old_ids = self.recorder.atom_ids_at(lower)
        new_ids = self.recorder.atom_ids_at(upper)
        old_types = self.recorder.types_at(lower)
        new_types = self.recorder.types_at(upper)
        changed = (old_ids != new_ids) | (old_types != new_types)
        result[changed] = first[changed] % first_box
        return result

    def current_frame(self):
        lower, _, _ = self.frame_interval()
        return lower
