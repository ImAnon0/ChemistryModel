# Unified bond-capacity research report

## Decision

**MODIFY — do not integrate yet.**

The dual unified radial model is the strongest heavy-valence formulation
tested so far. It passes the small-molecule, force, water-QM, molecule, NVE,
and complete 200-reaction gates without changing a production file or fitting
to Grambow. However, the stronger experiment that also drives the existing
angle topology from unified expected occupancies regresses the water-transfer
surface. The accepted candidate therefore unifies H/heavy radial bonding and
capacity, but still retains the established heavy-angle topology. That is a
promising architecture, not yet the requested complete replacement for the
concept of a bond.

## Implemented formulation

The reference model factorises bonding into:

- established all-valid-state H-transfer Hamiltonians;
- heavy-heavy order states 0/1/2/3 derived from the existing single, double,
  and triple Morse tables;
- one tapered finite-capacity constraint per heavy atom;
- one concave dual capacity price per heavy atom.

All factors contribute to one constrained scalar radial energy. Every pair
retains the single-bond short-range repulsive core, while attractive energy is
owned only by capacity-admissible H or heavy-order states. There is no raw
attraction followed by an overcoordination correction in the mathematical
model. The research adapter removes those old base terms only to compose the
new scalar without duplicating the unchanged geometry and angle machinery.

The dual has heavy-atom dimension rather than a product of global bond states.
It is solved with L-BFGS-B, bounded SLSQP, then Powell, and accepted only after
an independent projected KKT check below `1e-5`. Exact H degeneracies use a
`1e-4 eV` log-trace regularisation; the final water and force results measure
its effect directly.

## Small systems and safety

| gate | result |
|---|---:|
| H2 optimized distance | 0.7414400002 A (target 0.74144 A) |
| symmetric H3 minimum relative to H2 + H | +0.4196 eV |
| H + H2 approach | no bound minimum below H2 + H |
| ethane/formaldehyde/N2 expected bond order | 1.000 / 2.000 / 3.000 |
| crowded force, autograd vs finite difference | difference `4.6e-10 eV/A` |
| label permutation, energy | difference `1.1e-14 eV` |
| largest dual KKT case checked by finite difference | force difference `1.2e-7 eV/A` |
| NaN/non-finite results | none |
| 200-reaction optimizer failures | 0 |

The radial candidate preserves production energy and force, to numerical
precision, for H2, H3, methane, formaldehyde, water, ammonia, ethane, methanol,
hydroxylamine, and hydrogen peroxide. This includes accepted single, double,
and heteroatom bonds.

## Independent QM microscopes

Dense relative-energy RMSE in eV:

| system | production | local v0 | unified radial | unified topology |
|---|---:|---:|---:|---:|
| H3 | 0.2231 | 0.2231 | 0.2231 | 0.2231 |
| methane | 0.2184 | 0.2184 | 0.2184 | 0.2136 |
| formaldehyde | 0.3810 | 0.3810 | 0.3810 | 0.3330 |
| water transfer | 0.9867 | **0.2066** | **0.2136** | 0.3829 |
| all 98 dense points | 0.5874 | 0.2638 | 0.2654 | 0.3030 |

The primary model's water regression versus v0 is only `0.0071 eV` RMSE and
is far smaller than the rejected redesigned free energy (`0.4704 eV`). The
topology variant is not accepted: its `0.3829 eV` water RMSE shows that feeding
mean capacity occupancy into the current angle equation does not reproduce the
validated transfer surface.

## Matched NVE

| case | production max drift | unified max drift | caps | result |
|---|---:|---:|---:|---|
| water transfer, 250 x 0.25 fs | 0.00946 eV | 0.01541 eV | 0 | pass below 0.05 eV gate |
| crowded Grambow reactant, 250 x 0.10 fs | 0.00562 eV | 0.000122 eV | 0 | pass |
| symmetric exchange, 300 x 0.02 fs | 0.0000647 eV | 0.00000153 eV | 0 | pass |

The initial L-BFGS-only implementation did fail during water NVE. The final
solver does not accept optimizer status alone: it uses fallback algorithms and
the projected KKT residual. Re-running all trajectories produced no optimizer
failure, non-finite state, or move cap.

## Frozen Grambow 200 comparison

No prior formulation was recomputed. The new candidate was evaluated on the
same frozen endpoints and joined to the recorded baselines.

| formulation | barrier MAE | barrier RMSE | sign | reaction MAE | reaction RMSE |
|---|---:|---:|---:|---:|---:|
| production | 4.520 | 6.483 | 87.0% | 4.406 | 7.069 |
| local v0 | 2.116 | 2.761 | 95.5% | 2.723 | 3.490 |
| continuous shared edge | 2.260 | 2.902 | 94.0% | 2.876 | 3.623 |
| redesigned free energy | 2.111 | 2.649 | 98.5% | 2.500 | 3.182 |
| **unified radial** | **1.475** | **1.858** | **99.5%** | **2.233** | **2.872** |

The unified model improves/worsens/leaves unchanged 166/33/1 barriers versus
production and 142/57/1 versus local v0. It removes all 17 production barrier
errors above 10 eV and creates none. Its worst barrier error is `7.657 eV`
(`rxn008195`), down from `10.687 eV` in production and `10.698 eV` in v0.

Reaction energies improve overall but remain the weaker part of the model.
The signed reaction error is still `+1.217 eV`; the worst reaction error is
`8.159 eV`. Two new reaction errors above 5 eV appear relative to production,
so the benchmark result is not uniformly better point by point.

## Repository and production controls

- focused research tests: 21 passed;
- complete repository pytest: 339 passed, 1 skipped;
- full golden scientific checks 1-19: passed;
- dense 2 x 330 atom, 1 ps production stress: passed, zero final C/N
  over-valence in both seeds;
- the golden wrapper's embedded pytest: 49 Windows temporary-directory setup
  errors; the same suite passed with a workspace-local temporary directory;
- production force-field modules and parameters: unchanged.

The full validation report remains honestly marked `FAIL` because its wrapper
does not reinterpret the environment-only pytest setup errors.

## Why the recommendation is MODIFY

This research validates the central hypothesis: jointly allocating H and
heavy bond capacity produces a much more consistent energy landscape than
independent attraction plus a later penalty. It also shows that a factorised
dual representation can avoid global state enumeration and the earlier direct
SLSQP failure mode.

It is not ready to integrate for three reasons:

1. The winning model retains the established angle topology; full emergent
   topology has not yet been derived from the same variational scalar.
2. The direct unified-topology experiment regresses water, so promoting its
   occupancies into angles would violate the no-QM-regression rule.
3. This is a float64 SciPy reference with fallback solves, not a batched Torch
   or GPU production implementation. A production candidate needs an analytic
   or differentiable batched dual solver and identical-energy verification.

The next justified modification is to put angle/domain energy inside the
variational factor model, or derive a response-consistent topology functional,
then repeat water and NVE before any Torch/GPU port. Do not patch the current
angle layer with occupancy gates and do not tune the successful radial model
to individual Grambow reactions.

## Evidence

- `research_data/benchmark/diagnostics/unified_bond_capacity_validation.json`
- `research_data/benchmark/diagnostics/unified_bond_capacity_comparison.csv`
- `research_data/benchmark/diagnostics/unified_bond_capacity_comparison.json`
- `research_data/benchmark/diagnostics/unified_bond_capacity_production_validation.md`
- `research_data/benchmark/diagnostics/unified_bond_capacity_production_validation.json`
