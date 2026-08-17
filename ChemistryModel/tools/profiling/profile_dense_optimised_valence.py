"""Low-overhead profiler for the real optimised-valence batch path.

This script monkey-patches timing wrappers only for its own process. It does
not change production source, equations, parameters, recorder data, or output.
CUDA timings use sampled events and are resolved after the run.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
import sys
import time

import numpy as np
import torch


SAMPLE_EVERY = max(1, int(os.environ.get("CHEM_PROFILE_SAMPLE_EVERY", "20")))


class Profile:
    def __init__(self):
        self.calls = defaultdict(int)
        self.centres = defaultdict(int)
        self.events = defaultdict(list)
        self.wall = defaultdict(list)
        self.last_shape = None

    def cuda_call(self, label, function, *args, centres=0, **kwargs):
        call = self.calls[label]
        self.calls[label] += 1
        self.centres[label] += int(centres)
        sampled = torch.cuda.is_available() and call % SAMPLE_EVERY == 0
        if not sampled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.events[label].append((start, end))
        return result

    def wall_call(self, label, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.calls[label] += 1
            self.wall[label].append(time.perf_counter() - started)

    def stage_begin(self, label, centres=0):
        """Begin a profiling-only stage with wall time plus sampled CUDA events."""
        call = self.calls[label]
        self.calls[label] += 1
        self.centres[label] += int(centres)

        started_wall = time.perf_counter()
        start_event = None

        sampled = torch.cuda.is_available() and call % SAMPLE_EVERY == 0
        if sampled:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        return started_wall, start_event

    def stage_end(self, label, token):
        started_wall, start_event = token
        self.wall[label].append(time.perf_counter() - started_wall)

        if start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.events[label].append((start_event, end_event))

    def report(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = {"sample_every": SAMPLE_EVERY, "timings": {}}
        labels = sorted(set(self.calls) | set(self.events) | set(self.wall))
        for label in labels:
            cuda_ms = np.asarray([
                start.elapsed_time(end) for start, end in self.events[label]
            ], dtype=float)
            wall_s = np.asarray(self.wall[label], dtype=float)
            entry = {
                "calls": self.calls[label],
                "centres": self.centres[label],
                "cuda_samples": len(cuda_ms),
            }
            if len(cuda_ms):
                entry.update({
                    "cuda_median_ms": float(np.median(cuda_ms)),
                    "cuda_mean_ms": float(np.mean(cuda_ms)),
                    "cuda_sampled_total_ms": float(np.sum(cuda_ms)),
                    "cuda_estimated_total_ms": float(
                        np.mean(cuda_ms) * self.calls[label]
                    ),
                    "cuda_worst_ms": float(np.max(cuda_ms)),
                })
            if len(wall_s):
                entry.update({
                    "wall_total_s": float(np.sum(wall_s)),
                    "wall_median_s": float(np.median(wall_s)),
                    "wall_worst_s": float(np.max(wall_s)),
                })
            result["timings"][label] = entry
        return result


PROFILE = Profile()


def wrap_method(cls, name, label):
    original = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        return PROFILE.cuda_call(label, original, self, *args, **kwargs)

    setattr(cls, name, wrapped)


def install():
    import analysis
    import batch_runner
    import recorder
    import reactive_torch
    import valence_state_batched_membership_torch as heavy
    import valence_state_cached_h_topology_torch as cached_h

    # Profiling-only sinks. Normal simulations never set these class attributes.
    reactive_torch.ReactiveSimulation._reactive_profile_sink = PROFILE
    heavy.BatchedHeavyValenceStateBatchedSimulation._heavy_profile_sink = PROFILE

    wrap_method(reactive_torch.ReactiveSimulation, "compute_forces", "force.total")
    wrap_method(reactive_torch.ReactiveSimulation, "build_neighbours", "neighbours.rebuild")
    wrap_method(reactive_torch.ReactiveSimulation, "needs_rebuild", "neighbours.check")
    wrap_method(cached_h.CachedHFastValenceStateBatchedSimulation,
                "_hydrogen_state_correction", "energy.h_state")
    wrap_method(heavy.BatchedHeavyValenceStateBatchedSimulation,
                "_valence_topology_correction", "energy.heavy_topology")
    # Membership needs a profiling sink on the *live instance*.  Setting only
    # a base-class attribute is not reliable through every optimised-valence
    # inheritance path, so install it immediately around the wrapped call.
    original_membership = (
        heavy.BatchedHeavyValenceStateBatchedSimulation._local_valence_membership
    )

    def timed_membership(self, *args, **kwargs):
        had_instance_sink = "_heavy_profile_sink" in self.__dict__
        previous_sink = self.__dict__.get("_heavy_profile_sink")
        self._heavy_profile_sink = PROFILE
        try:
            return PROFILE.cuda_call(
                "heavy.membership_total",
                original_membership,
                self,
                *args,
                **kwargs,
            )
        finally:
            if had_instance_sink:
                self._heavy_profile_sink = previous_sink
            else:
                self.__dict__.pop("_heavy_profile_sink", None)

    heavy.BatchedHeavyValenceStateBatchedSimulation._local_valence_membership = (
        timed_membership
    )

    original_assemble = heavy.BatchedHeavyValenceStateBatchedSimulation._assemble_heavy_hamiltonian

    def assemble(self, diagonal, coupling, structure):
        n = int(structure["candidate_count"])
        v = int(structure["capacity"])
        s = int(structure["state_count"])
        batch = int(diagonal.shape[0])
        PROFILE.last_shape = (n, v, s, batch)
        label = f"heavy.assemble.N{n}.V{v}.S{s}"
        return PROFILE.cuda_call(
            label, original_assemble, self, diagonal, coupling, structure,
            centres=batch,
        )

    heavy.BatchedHeavyValenceStateBatchedSimulation._assemble_heavy_hamiltonian = assemble

    original_density = heavy.thermal_state_probabilities

    def timed_density(hamiltonian, temperature):
        shape = PROFILE.last_shape
        if shape is None or shape[2] != int(hamiltonian.shape[-1]):
            shape = (-1, -1, int(hamiltonian.shape[-1]), int(hamiltonian.shape[0]))
        n, v, s, batch = shape
        prefix = f"heavy.density.N{n}.V{v}.S{s}"

        return PROFILE.cuda_call(
            prefix + ".total", original_density,
            hamiltonian, temperature, centres=batch,
        )

    heavy.thermal_state_probabilities = timed_density

    original_analyse = analysis.analyse
    analysis.analyse = lambda *a, **k: PROFILE.wall_call(
        "final.analysis", original_analyse, *a, **k
    )
    original_save = recorder.Recorder.save
    recorder.Recorder.save = lambda self, *a, **k: PROFILE.wall_call(
        "final.recorder_save", original_save, self, *a, **k
    )
    original_summary = batch_runner.summarise_run
    batch_runner.summarise_run = lambda *a, **k: PROFILE.wall_call(
        "final.summarise_run", original_summary, *a, **k
    )
    original_reindex = batch_runner.rebuild_index
    batch_runner.rebuild_index = lambda *a, **k: PROFILE.wall_call(
        "final.rebuild_index", original_reindex, *a, **k
    )
    original_run_group = batch_runner.run_group
    batch_runner.run_group = lambda *a, **k: PROFILE.wall_call(
        "phase.md_group", original_run_group, *a, **k
    )

    return batch_runner


def main():
    batch_runner = install()
    try:
        batch_runner.main()
    finally:
        print("\nCHEMISTRYMODEL_PROFILE_JSON")
        print(json.dumps(PROFILE.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
