# Grambow benchmark physics diagnosis

## Scope and ground truth

This investigation uses the 200-reaction Optimised-Valence result produced by
`valence_state_optimised_torch.OptimisedValenceStateBatchedSimulation` at
float64.  The stored model energies were independently reconstructed by the
per-reaction microscope to approximately `1e-14 eV`.

The current score file is named `grambow_optimised.json` but contains CSV.
The diagnostic loader detects content rather than trusting the extension.

Baseline:

| Quantity | Value |
| --- | ---: |
| Barrier MAE | 4.5195 eV |
| Barrier RMSE | 6.4831 eV |
| Barrier signed mean | +1.8843 eV |
| Barrier sign agreement | 87.0% |
| Reaction-energy MAE | 4.4058 eV |
| Reaction-energy RMSE | 7.0688 eV |
| Reaction-energy signed mean | -0.1387 eV |

No production equation or parameter was changed during this diagnosis.

## Failure classification

Endpoint bond labels are inferred from a documented covalent-radius geometry
rule.  They are useful classification evidence, not ChemistryModel bond
declarations.

| Broad family | Count | Barrier mean | Barrier MAE | Reaction mean | Reaction MAE | Sign failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bond formation/dissociation | 14 | -6.37 | 8.19 | -8.48 | 10.29 | 7 |
| Bond rearrangement | 41 | +3.24 | 4.98 | -0.37 | 4.14 | 2 |
| Hydrogen transfer | 144 | +2.30 | 4.05 | +0.72 | 3.93 | 17 |
| O-O/peroxide chemistry | 1 | +1.93 | 1.93 | +1.87 | 1.87 | 0 |

The strongest chemical-family failure is formation/dissociation, but the
energy decomposition exposes a broader geometric mechanism crossing several
reaction families.

## Measured common mechanism

In the Optimised-Valence engine, heavy-valence membership limits which
contacts participate in electron-domain/angle topology.  The base quadratic
overcoordination penalty deliberately continues to use raw radial contact
coordination.

For the catastrophic cases, ordinary intramolecular near contacts are inside
the chemical radial cutoff and therefore add almost full coordination even
when the heavy-valence state assigns them near-zero membership.  Examples in
reactant geometries include:

- `rxn006559`: C2-C6 at 1.836 A, taper 1.000, memberships 0.060/0.059.
- `rxn000096`: two C-C contacts at 2.026 A, taper 0.880, memberships 0.046/0.014.
- `rxn010742`: C2-C5 at 2.081 A, taper 0.758, memberships 0.032/0.032.
- `rxn011394`: C3-C5 at 1.959 A, taper 0.976, memberships 0.022/0.059.

These capacity-rejected contacts also retain the full radial Morse attraction.
For example, the four listed contacts contribute approximately -1.6 to
-2.9 eV each before the coarse overcoordination penalty compensates for them.

Across all 200 reactions:

- signed barrier error versus heavy-atom overcoordination change has Pearson
  `r = +0.91`;
- absolute barrier error versus absolute heavy-overcoordination change has
  `r = +0.89`;
- base overcoordination is the largest raw energy-change term in 42 of the
  worst 50 barrier errors;
- base overcoordination, base angle, and valence angle correction are strongly
  correlated because they are different responses to the same crowded radial
  contact set.

Selected microscopes:

| Reaction | Barrier error | Heavy-over change | Effective-angle change | Bond change |
| --- | ---: | ---: | ---: | ---: |
| rxn006559 | -35.502 | -41.964 | +2.025 | +6.721 |
| rxn011804 | +26.010 | +23.550 | +3.639 | +1.353 |
| rxn004353 | +22.765 | +24.354 | +1.848 | -2.930 |
| rxn000096 | -21.968 | -24.975 | +0.775 | +5.659 |
| rxn010742 | -19.573 | -26.225 | +0.973 | +7.541 |
| rxn011394 | -12.283 | -12.610 | +1.385 | +2.732 |
| rxn011223 | -11.465 | -8.423 | -1.649 | +4.309 |

The same mechanism explains both error signs: when the transition state gains
raw radial crowding, the barrier becomes too high; when the reactant is more
crowded than the transition/product endpoint, the model produces a negative
or excessively low barrier/reaction energy.

## Root-cause hypothesis

The measurements support a physics/decomposition mismatch rather than another
benchmark bug:

1. Heavy-valence competition recognises that some nearby contacts do not fit
   the atom's active valence state.
2. That state currently changes angle topology only.
3. Capacity-rejected contacts still receive full radial attraction.
4. A raw-coordinate quadratic penalty is then asked to counteract those
   attractions, but it also penalises chemically normal 1,3/ring proximity.

Consequently, endpoint energy differences can be dominated by how many
non-selected contacts happen to fall inside a radial cutoff, rather than by
the intended bond rearrangement.

## Why the tempting small change is rejected

Removing the heavy-atom overcoordination term from frozen endpoint energies is
a useful sensitivity test:

| Diagnostic-only scenario | Barrier MAE | Barrier sign | Reaction MAE |
| --- | ---: | ---: | ---: |
| Current | 4.5195 | 87.0% | 4.4058 |
| Remove heavy overcoordination | 2.2452 | 95.0% | 3.0499 |
| Remove heavy overcoordination and effective angle | 2.0957 | 93.0% | 2.6050 |

These are not valid after-model benchmark results.  They are arithmetic
term-removal diagnostics on frozen endpoints.  Simply deleting the penalty is
scientifically unsafe: capacity-rejected contacts would still keep their full
attractive Morse energy, making an extra contact attractive without a
corresponding crowding cost.  It would likely improve this benchmark while
creating collapse/overbonding elsewhere.

Likewise, globally weakening the overcoordination constant would be blind
parameter tuning and is not justified by these measurements.

## Proposed physics work

The smallest scientifically coherent direction is an energy-bearing
heavy-valence competition model, developed first as a standalone diagnostic:

1. Preserve short-range radial repulsion for every close contact.
2. Make attractive chemical bonding capacity-limited by the same smooth
   heavy-valence states that currently provide topology membership.
3. Include the heavy-state mixing/free-energy contribution explicitly rather
   than using membership only to redraw angles.
4. Retain a smooth crowding/exchange cost for genuinely penetrating rejected
   contacts, without charging normal 1,3 proximity as an extra bond.
5. Define symmetric pair bookkeeping so two local centres cannot independently
   add or remove the same pair energy.
6. Recover accepted isolated bonds, molecules, H-state transfer surfaces, and
   continuous forces exactly where no heavy-valence competition exists.

This requires a small reference-model design and microscope validation before
production integration.  The current architecture does not provide a
scientifically defensible one-line replacement for the heavy overcoordination
term.  Production physics should therefore remain unchanged at this gate.

## Required gate for a future candidate

A future candidate must pass, in order:

1. crowded-carbon/nitrogen/oxygen probes proving rejected contacts do not gain
   free attraction;
2. isolated-bond and stable-molecule energy/force equivalence;
3. dense H-transfer and heavy-valence continuity microscopes;
4. force finite differences and NVE conservation;
5. the same 200-reaction Grambow benchmark with an explicit fingerprint;
6. `validation_report.py --full` and the complete pytest suite.

Until those equations are defined and tested, the correct outcome of this
workflow is diagnosis plus a constrained proposal, not a speculative
production patch.
