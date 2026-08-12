"""Does the gradient relaxation agree with Powell, and how much faster?

The gradient is not a finite difference on the energy. Autograd gives the
exact derivative with respect to atom positions for one evaluation, and how
the atoms move when a spectator coordinate changes is a difference on the
geometry builder alone -- arithmetic, no simulation. Chaining them gives an
essentially exact gradient at the cost of a single energy call.

Two things decide whether the change is safe. The barrier has to come out
the same, and where it does not, the relaxed energy has to be no higher:
on a flat surface the two methods can settle in different local minima, and
the lower one converged better rather than being wrong.
"""
import time

import numpy as np

from hf_surface_scan import (
    apply_system, build, energy_at, measure_barrier, relaxed_energy,
)

# Barriers from the relaxed agreement run at mixing 0.52, where both code
# paths gave identical answers. Ammonia has never been measured relaxed.
KNOWN = {"formaldehyde": 0.323, "water": 0.669, "methane": 0.414}

print("per-cell comparison at a few geometries\n")
print(f"{'system':>14}{'method':>10}{'energy':>12}{'seconds':>10}"
      f"   spectators")

for system, donor, transfer in (
    ("formaldehyde", 1.16, 1.05),
    ("water", 1.10, 1.30),
    ("methane", 1.12, 1.055),
    ("ammonia", 1.12, 1.01),
):
    apply_system(system)
    sim = build("high_fidelity", mixing=0.52)

    results = {}
    for label, gradient_based in (("Powell", False), ("gradient", True)):
        started = time.monotonic()
        value, spectators = relaxed_energy(
            sim, donor, transfer, gradient_based=gradient_based
        )
        elapsed = time.monotonic() - started
        results[label] = value

        shown = " ".join(f"{item:.2f}" for item in spectators)
        print(f"{system if label == 'Powell' else '':>14}{label:>10}"
              f"{value:12.5f}{elapsed:10.3f}   {shown}")

    difference = results["gradient"] - results["Powell"]
    verdict = (
        "identical" if abs(difference) < 1e-6
        else "gradient found lower" if difference < 0
        else "gradient found HIGHER, so it converged worse"
    )
    print(f"{'':>14}{'':>10}{difference:+12.5f}   {verdict}\n")


print("\nwhole barriers, against the known relaxed values\n")
print(f"{'system':>14}{'Powell':>10}{'gradient':>10}{'known':>10}"
      f"{'seconds':>10}")

for system in ("formaldehyde", "water", "methane", "ammonia"):
    row = {}
    for label, gradient_based in (("Powell", False), ("gradient", True)):
        started = time.monotonic()
        found = measure_barrier(
            "high_fidelity", system, mixing=0.52, relax=True,
            gradient_based=gradient_based,
        )
        row[label] = (found["barrier"], time.monotonic() - started)

    known = KNOWN.get(system)
    print(f"{system:>14}{row['Powell'][0]:10.4f}{row['gradient'][0]:10.4f}"
          f"{known if known is not None else float('nan'):10.3f}"
          f"{row['gradient'][1]:10.1f}")
    print(f"{'':>14}{row['Powell'][1]:10.1f}s Powell, "
          f"{row['gradient'][1]:.1f}s gradient, "
          f"{row['Powell'][1] / max(row['gradient'][1], 1e-9):.1f}x faster")
