"""CUDA microbenchmark for bounded factorised-H S=2 eigvalsh execution.

Run one chunk size per process so allocator state is independent.  This is a
diagnostic companion to the real group-16 production benchmarks, not a
replacement for them.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - direct-execution project path

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from h_state_factorised_batched_torch import (
    _bounded_factorised_eigvalsh,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--first-batch", type=int, default=640)
    parser.add_argument("--last-batch", type=int, default=1920)
    parser.add_argument("--batch-step", type=int, default=16)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if options.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    device = torch.device("cuda")
    batches = list(range(
        options.first_batch,
        options.last_batch + 1,
        options.batch_step,
    ))
    rows = []
    eigvalsh_seconds = 0.0
    started = time.perf_counter()

    for batch_size in batches:
        torch.manual_seed(100000 + batch_size)
        raw = torch.randn(
            batch_size,
            2,
            2,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        hamiltonian = 0.5 * (
            raw + raw.transpose(-1, -2)
        )

        torch.cuda.synchronize()
        eig_started = time.perf_counter()
        eigenvalues = _bounded_factorised_eigvalsh(
            hamiltonian,
            options.chunk_size,
        )
        torch.cuda.synchronize()
        eigvalsh_seconds += time.perf_counter() - eig_started

        torch.autograd.grad(
            eigenvalues[:, 0].sum(),
            raw,
        )
        torch.cuda.synchronize()

        stats = torch.cuda.memory_stats()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        rows.append({
            "batch_size": batch_size,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
            "num_oom_rejections": int(
                stats.get("num_oom_rejections", 0)
            ),
        })

        del eigenvalues, hamiltonian, raw
        gc.collect()

    torch.cuda.synchronize()
    result = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "chunk_size": options.chunk_size,
        "batches": batches,
        "wall_seconds": time.perf_counter() - started,
        "eigvalsh_seconds": eigvalsh_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "final_reserved_bytes": int(torch.cuda.memory_reserved()),
        "final_device_free_bytes": int(torch.cuda.mem_get_info()[0]),
        "allocator_retries": int(
            torch.cuda.memory_stats().get("num_alloc_retries", 0)
        ),
        "allocator_ooms": int(
            torch.cuda.memory_stats().get("num_ooms", 0)
        ),
        "allocator_oom_rejections": int(
            torch.cuda.memory_stats().get("num_oom_rejections", 0)
        ),
        "samples": rows,
    }

    output = Path(options.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in result.items()
        if key not in {"batches", "samples"}
    }, indent=2))


if __name__ == "__main__":
    main()
