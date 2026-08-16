import argparse
import os
import subprocess
import sys
import tempfile
import time


# ============================================================
# Does running several at once already help?
# ============================================================
#
#   py profile_parallel.py
#   py profile_parallel.py --steps 400 --most 6
#
# Profiling one process showed the card idle at 330 atoms, and
# eight times the atoms costing only twice the time. That says
# there is spare capacity. What it does not say is whether
# running several batches at once already claims it - which
# matters, because that costs nothing and batching inside one
# process would be a substantial rewrite of a validated energy
# function.
#
# So: run the same short simulation in one process, then two,
# then three, and see how total throughput moves. If four
# processes give something close to four times the work, the
# spare capacity is already being used and there is nothing to
# build. If it flattens out after two, the remaining gain needs
# batching.


WORKER = '''
import sys, time

# The worker is written to a temporary folder, so the project it
# needs is not on the path by default.

sys.path.insert(0, sys.argv[5])
import build_box, mixtures
from reactive_torch import ReactiveSimulation

mixture, box, steps, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

kind, contents = mixtures.all_mixtures()[mixture]

if kind == "molecules":
    symbols, positions = build_box.build(contents, box)
else:
    symbols, positions = build_box.loose_atoms(
        contents, box, minimum_separation=1.25, random_seed=seed
    )

simulation = ReactiveSimulation(
    symbols=symbols, positions=positions, box_size=box,
    random_seed=seed,
)

# A few steps first so the timing does not include warm-up.

simulation.step(20)

start = time.perf_counter()
simulation.step(steps)
elapsed = time.perf_counter() - start

print(f"{elapsed:.4f} {simulation.atom_count}")
'''


def run_group(worker_path, mixture, box, steps, count):
    started = time.perf_counter()

    processes = [
        subprocess.Popen(
            [
                sys.executable, worker_path,
                mixture, str(box), str(steps), str(700 + index),
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(count)
    ]

    times = []
    atoms = 0

    for process in processes:
        out, err = process.communicate()

        if process.returncode != 0:
            raise SystemExit(
                f"a worker failed:\n{err.strip()[-800:]}"
            )

        pieces = out.strip().split()

        times.append(float(pieces[0]))
        atoms = int(pieces[1])

    wall = time.perf_counter() - started

    return {
        "wall": wall,
        "slowest": max(times),
        "mean": sum(times) / len(times),
        "atoms": atoms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="See whether concurrent processes already use "
                    "the spare capacity."
    )

    parser.add_argument("--mixture", default="carbon rich")
    parser.add_argument("--box", type=float, default=19.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--most", type=int, default=5)

    options = parser.parse_args()

    folder = tempfile.mkdtemp()

    worker_path = os.path.join(folder, "worker.py")

    with open(worker_path, "w") as handle:
        handle.write(WORKER)

    print(
        f"{options.mixture} in a {options.box:g} A box, "
        f"{options.steps} steps each"
    )
    print()
    print(
        f"  {'at once':>8}{'wall':>10}{'per process':>14}"
        f"{'steps/s total':>16}{'vs one':>9}"
    )
    print("  " + "-" * 57)

    single = None

    for count in range(1, options.most + 1):
        result = run_group(
            worker_path, options.mixture, options.box,
            options.steps, count,
        )

        # Total useful work divided by how long it took in real
        # time: what actually matters when filling a queue.

        throughput = (
            options.steps * count / result["slowest"]
        )

        if single is None:
            single = throughput

        print(
            f"  {count:>8}{result['wall']:>9.1f}s"
            f"{result['slowest']:>13.1f}s"
            f"{throughput:>16.1f}"
            f"{throughput / single:>8.2f}x"
        )

    print()
    print(
        "  If this keeps climbing, running more at once is the\n"
        "  whole answer and nothing needs building. If it levels\n"
        "  off, the rest of the spare capacity needs several\n"
        "  boxes sharing one set of kernels."
    )

    try:
        os.remove(worker_path)
        os.rmdir(folder)
    except OSError:
        pass


if __name__ == "__main__":
    main()
