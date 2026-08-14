# Reactive integrator stability

## Failure mechanism

A matched carbon-rich run provided a reproducible numerical failure. Seed
17005 jumped from 23.3 to 69,936.4 eV kinetic energy between recorded frames,
while potential changed by only 4.6 eV. Two carbon atoms acquired nearly equal
large velocities. This was not physical bond-energy release.

Live diagnostics localized the trigger to a new C-C contact at 2.439896 A,
just inside the 2.440 A outer cutoff. Its neighbor relationships were current
and reciprocal. The force was 37,660 eV/A even though rebuilding neighbors at
the nearby saved geometry produced ordinary single-digit forces.

The bond-order allocator normalized spare valence only when a contact total
exceeded `1e-9`. A vanishing cutoff contact therefore switched abruptly from
zero allocation to a finite normalized allocation. The energy change was tiny,
but the derivative through that ill-conditioned normalization was enormous.

## Root correction

Spare-valence allocation now uses a C1 smoothstep onset below total weight
`1e-4`. At zero contact both allocation and slope are zero. At and above the
onset the gate is exactly one, preserving the established-contact formula.
NumPy and Torch use the same rule.

The emergency movement limiter is also internally consistent: when a move is
capped, the unresolved force impulse is rejected instead of being stored as
velocity. Uncapped atoms execute the original Velocity-Verlet expression
exactly. Batch summaries report per-seed `move_cap_events` so future limiter
activity is visible rather than silent.

Two simpler denominator regularizations were tested and rejected. Both created
new near-zero derivative problems and catastrophic batches; neither remains in
the code.

## Validation

- reactive core: 8/8 passed
- bond calibration: 9/9 passed
- high fidelity: 7/7 passed
- randomized cutoff stress: 2,400 float32/float64 geometries, no pathological
  or non-finite forces
- exact pre-failure replay: ordinary 7--13 eV/A forces, zero cap events
- accepted-model carbon-rich CUDA batch, seeds 17000--17015: 16/16 stable,
  zero energy jumps, zero move-cap events, mean final temperature 528.83 K

The final batch retained healthy carbon chemistry: mean 57.75 heavy bonds,
11.06 atoms in the largest structure, 4.625 carbons in the largest carbon
structure, and a 3.25 mean chain score.
