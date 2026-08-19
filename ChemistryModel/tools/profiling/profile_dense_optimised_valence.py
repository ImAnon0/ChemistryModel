"""Profiler for the real optimised-valence batch path.

This script monkey-patches timing wrappers only for its own process. It does
not change production source, equations, parameters, recorder data, or output.
CUDA timings normally use sampled events. Requested H-state regions and force
backward use explicit synchronization in this profiler process only.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

from collections import defaultdict, deque
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


SAMPLE_EVERY = max(1, int(os.environ.get("CHEM_PROFILE_SAMPLE_EVERY", "20")))

SYNCHRONIZED_LABELS = {
    "H group preparation",
    "H coupling/physics construction",
    "Hamiltonian assembly",
    "torch.linalg.eigvalsh",
    "force.autograd_backward",
}

MEMORY_CORRELATION_LABELS = {
    "energy.h_state",
    "energy.heavy_topology",
    "force.total",
}

MEMORY_STAT_PREFIXES = (
    "allocated_bytes.",
    "active_bytes.",
    "reserved_bytes.",
    "inactive_split_bytes.",
    "requested_bytes.",
    "segment.",
    "allocation.",
    "active.",
    "inactive_split.",
)

MEMORY_STAT_SCALARS = {
    "num_alloc_retries",
    "num_oom_rejections",
    "num_ooms",
    "num_device_alloc",
    "num_device_free",
    "num_sync_all_streams",
}


class Profile:
    def __init__(self):
        self.calls = defaultdict(int)
        self.centres = defaultdict(int)
        self.events = defaultdict(list)
        self.wall = defaultdict(list)
        self.last_shape = None
        self.started = time.perf_counter()
        self.memory_trace_limit = max(
            100,
            int(os.environ.get("CHEM_PROFILE_MEMORY_TRACE_LIMIT", "50000")),
        )
        self.memory_trace = deque(maxlen=self.memory_trace_limit)
        self.memory_sample_count = 0
        self.memory_trace_dropped = 0
        self.memory_jump_bytes = max(
            1,
            int(os.environ.get("CHEM_PROFILE_MEMORY_JUMP_MB", "32"))
            * 1024
            * 1024,
        )
        self.memory_trace_path = None
        self.memory_snapshot_path = None
        self.memory_history_error = None
        self.snapshot_error = None
        self.device_memory_used_error = None
        self.last_alloc_retry_count = 0
        self.retry_snapshot_paths = []
        self.retry_snapshot_errors = []

    def configure_artifacts(self, output_directory):
        output_directory = Path(output_directory)
        self.memory_trace_path = output_directory / "cuda_memory_trace.json"
        self.memory_snapshot_path = output_directory / "cuda_memory_snapshot.pickle"

    @staticmethod
    def _memory_stats_subset():
        stats = torch.cuda.memory_stats()
        selected = {}
        for key, value in stats.items():
            if key in MEMORY_STAT_SCALARS:
                selected[key] = int(value)
                continue
            if (
                not key.startswith(MEMORY_STAT_PREFIXES)
                or ".all." not in key
            ):
                continue
            if key.endswith((".current", ".peak", ".allocated", ".freed")):
                selected[key] = int(value)
        return selected

    def memory_sample(self, label, phase, context=None):
        if not torch.cuda.is_available():
            return None

        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        max_allocated = int(torch.cuda.max_memory_allocated())
        max_reserved = int(torch.cuda.max_memory_reserved())
        free_bytes, total_bytes = (
            int(value) for value in torch.cuda.mem_get_info()
        )
        physical_used = total_bytes - free_bytes

        device_memory_used = None
        if (
            hasattr(torch.cuda, "device_memory_used")
            and self.device_memory_used_error is None
        ):
            try:
                device_memory_used = int(torch.cuda.device_memory_used())
            except Exception as exc:  # optional NVML dependency on this build
                if self.device_memory_used_error is None:
                    self.device_memory_used_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

        previous = self.memory_trace[-1] if self.memory_trace else None
        allocator_stats = self._memory_stats_subset()
        retry_count = int(allocator_stats.get("num_alloc_retries", 0))
        if (
            retry_count > self.last_alloc_retry_count
            and self.memory_snapshot_path is not None
            and self.memory_history_error is None
        ):
            retry_path = self.memory_snapshot_path.with_name(
                "cuda_memory_retry_"
                f"{retry_count:03d}_snapshot.pickle"
            )
            try:
                retry_path.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.memory._dump_snapshot(str(retry_path))
                self.retry_snapshot_paths.append(str(retry_path))
            except Exception as exc:
                self.retry_snapshot_errors.append(
                    f"retry {retry_count}: {type(exc).__name__}: {exc}"
                )
        self.last_alloc_retry_count = max(
            self.last_alloc_retry_count,
            retry_count,
        )

        sample = {
            "sample": self.memory_sample_count,
            "timestamp_s": time.perf_counter() - self.started,
            "label": str(label),
            "phase": str(phase),
            "context": dict(context or {}),
            "pytorch_allocated_bytes": allocated,
            "pytorch_reserved_bytes": reserved,
            "pytorch_max_allocated_bytes": max_allocated,
            "pytorch_max_reserved_bytes": max_reserved,
            "device_free_bytes": free_bytes,
            "device_total_bytes": total_bytes,
            "device_physical_used_bytes": physical_used,
            "device_memory_used_bytes": device_memory_used,
            "physical_minus_pytorch_reserved_bytes": physical_used - reserved,
            "allocator_stats": allocator_stats,
        }

        if previous is None:
            sample["delta"] = {}
            sample["major_jump"] = False
        else:
            delta = {
                "allocated_bytes": (
                    allocated - int(previous["pytorch_allocated_bytes"])
                ),
                "reserved_bytes": (
                    reserved - int(previous["pytorch_reserved_bytes"])
                ),
                "physical_used_bytes": (
                    physical_used - int(previous["device_physical_used_bytes"])
                ),
            }
            sample["delta"] = delta
            sample["major_jump"] = any(
                abs(int(value)) >= self.memory_jump_bytes
                for value in delta.values()
            )

        self.memory_sample_count += 1
        if len(self.memory_trace) == self.memory_trace_limit:
            self.memory_trace_dropped += 1
        self.memory_trace.append(sample)

        return sample

    def print_raw_memory_checkpoint(self, label):
        if not torch.cuda.is_available():
            print(f"CUDA RAW MEMORY [{label}]: unavailable")
            return

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"CUDA RAW MEMORY [{label}]")
        print(f"  torch.cuda.memory_allocated()     = {allocated}")
        print(f"  torch.cuda.memory_reserved()      = {reserved}")
        print(f"  torch.cuda.max_memory_allocated() = {max_allocated}")
        print(f"  torch.cuda.max_memory_reserved()  = {max_reserved}")
        print(f"  torch.cuda.mem_get_info()         = ({free_bytes}, {total_bytes})")

    def start_memory_history(self):
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="alloc",
                stacks="python",
                max_entries=max(
                    1000,
                    int(os.environ.get(
                        "CHEM_PROFILE_MEMORY_HISTORY_ENTRIES",
                        "20000",
                    )),
                ),
                clear_history=True,
            )
        except Exception as exc:
            self.memory_history_error = f"{type(exc).__name__}: {exc}"

    def write_memory_artifacts(self):
        if self.memory_trace_path is None:
            return

        self.memory_trace_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None
            ),
            "pid": os.getpid(),
            "pytorch_no_cuda_memory_caching": os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING"
            ),
            "trace_limit": self.memory_trace_limit,
            "trace_dropped": self.memory_trace_dropped,
            "device_memory_used_error": self.device_memory_used_error,
            "memory_history_error": self.memory_history_error,
            "snapshot_error": self.snapshot_error,
            "retry_snapshots": self.retry_snapshot_paths,
            "retry_snapshot_errors": self.retry_snapshot_errors,
            "samples": list(self.memory_trace),
        }
        self.memory_trace_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        if torch.cuda.is_available() and self.memory_history_error is None:
            try:
                torch.cuda.memory._dump_snapshot(
                    str(self.memory_snapshot_path)
                )
            except Exception as exc:
                self.snapshot_error = f"{type(exc).__name__}: {exc}"
                payload["snapshot_error"] = self.snapshot_error
                self.memory_trace_path.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

    def cuda_call(self, label, function, *args, centres=0, **kwargs):
        call = self.calls[label]
        self.calls[label] += 1
        self.centres[label] += int(centres)
        correlate_memory = (
            label in MEMORY_CORRELATION_LABELS
            or label.startswith("heavy.assemble.")
            or label.startswith("heavy.density.")
        )
        context = None
        if label.startswith("heavy.") and self.last_shape is not None:
            n, v, s, batch = self.last_shape
            context = {
                "candidate_count": n,
                "capacity": v,
                "state_count": s,
                "batch": batch,
            }
        if correlate_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_sample(label, "before", context)

        sampled = torch.cuda.is_available() and call % SAMPLE_EVERY == 0
        try:
            if not sampled:
                return function(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            self.events[label].append((start, end))
            return result
        finally:
            if correlate_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                self.memory_sample(label, "after", context)

    def wall_call(self, label, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.calls[label] += 1
            self.wall[label].append(time.perf_counter() - started)

    def stage_begin(self, label, centres=0, context=None):
        """Begin a profiling-only stage with wall time plus sampled CUDA events."""
        call = self.calls[label]
        self.calls[label] += 1
        self.centres[label] += int(centres)

        if label in SYNCHRONIZED_LABELS and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_sample(label, "before", context)

        started_wall = time.perf_counter()
        start_event = None

        sampled = torch.cuda.is_available() and call % SAMPLE_EVERY == 0
        if sampled:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        return started_wall, start_event, context

    def stage_end(self, label, token, context=None):
        started_wall, start_event, begin_context = token

        if label in SYNCHRONIZED_LABELS and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_sample(
                label,
                "after",
                context if context is not None else begin_context,
            )

        self.wall[label].append(time.perf_counter() - started_wall)

        if start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.events[label].append((start_event, end_event))

    def report(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = {
            "sample_every": SAMPLE_EVERY,
            "memory_trace": (
                str(self.memory_trace_path)
                if self.memory_trace_path is not None
                else None
            ),
            "memory_snapshot": (
                str(self.memory_snapshot_path)
                if self.memory_snapshot_path is not None
                else None
            ),
            "memory_samples": len(self.memory_trace),
            "memory_trace_dropped": self.memory_trace_dropped,
            "memory_history_error": self.memory_history_error,
            "snapshot_error": self.snapshot_error,
            "retry_snapshots": self.retry_snapshot_paths,
            "retry_snapshot_errors": self.retry_snapshot_errors,
            "device_memory_used_error": self.device_memory_used_error,
            "timings": {},
        }
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


def output_directory_from_argv():
    for index, argument in enumerate(sys.argv[:-1]):
        if argument == "--out":
            return sys.argv[index + 1]
    for argument in sys.argv[1:]:
        if argument.startswith("--out="):
            return argument.split("=", 1)[1]
    return os.path.join("runs", "optimised_valence_profile")


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
    import h_state_factorised_batched_torch as grouped_h

    # Profiling-only sinks. Normal simulations never set these class attributes.
    reactive_torch.ReactiveSimulation._reactive_profile_sink = PROFILE
    heavy.BatchedHeavyValenceStateBatchedSimulation._heavy_profile_sink = PROFILE
    cached_h.CachedHFastValenceStateBatchedSimulation._h_state_profile_sink = PROFILE
    grouped_h.GroupedFactorisedHStateBatchedSimulation._h_state_profile_sink = PROFILE

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
    PROFILE.configure_artifacts(
        output_directory_from_argv()
    )
    PROFILE.start_memory_history()
    PROFILE.print_raw_memory_checkpoint("profiler start")
    PROFILE.memory_sample(
        "profiler",
        "start",
    )
    batch_runner = install()
    try:
        batch_runner.main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        PROFILE.memory_sample(
            "profiler",
            "quiet final",
        )
        PROFILE.print_raw_memory_checkpoint("quiet final")
        PROFILE.write_memory_artifacts()
        print("\nCHEMISTRYMODEL_PROFILE_JSON")
        print(json.dumps(PROFILE.report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
