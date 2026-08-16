"""Read-only recorder baseline benchmark; never rewrites source trajectories."""

import argparse
import os
import tempfile
import time

import numpy as np

from recorder import Recorder


def benchmark(path):
    size = os.path.getsize(path)
    started = time.perf_counter()
    recorder = Recorder.load(path)
    load_seconds = time.perf_counter() - started
    handle, target = tempfile.mkstemp(suffix=".npz")
    os.close(handle)
    try:
        started = time.perf_counter()
        recorder.save(target)
        save_seconds = time.perf_counter() - started
        new_size = os.path.getsize(target)
    finally:
        os.unlink(target)
    intervals = np.diff(np.asarray(recorder.times, dtype=float))
    return {
        "version": recorder.format_version,
        "frames": len(recorder),
        "atoms": len(recorder.positions[0]) if len(recorder) else 0,
        "source_mb": size / 1048576,
        "roundtrip_mb": new_size / 1048576,
        "load_s": load_seconds,
        "save_s": save_seconds,
        "median_interval_fs": float(np.median(intervals)) if len(intervals) else 0,
        "maximum_interval_fs": float(np.max(intervals)) if len(intervals) else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    options = parser.parse_args()
    for source in options.paths:
        print(source)
        for key, value in benchmark(source).items():
            print(f"  {key:22s} {value}")
