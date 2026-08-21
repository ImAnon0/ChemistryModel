# Shared bond-representation formulation study

## Decision

**DO NOT INTEGRATE ANY CANDIDATE.** Production physics remains untouched.

The broader study does support one important scientific conclusion: the current
bond representation is limiting the model. A parameter-free continuous shared
bond free energy, derived from the existing single/double/triple Morse tables,
slightly beats local v0 on the frozen Grambow aggregate and substantially
improves reaction errors and barrier signs. That improvement disappears if the
model is reduced to channels of the inherited production bond order.

The redesigned candidate is not ready for production. It regresses the
independent water-transfer QM microscope and its SciPy reference optimizer
fails during one crowded NVE trajectory. The result identifies a promising
representation and the next technical/scientific obstacles; it does not clear
the required integration gates.

No force-field value, production equation, H-state implementation, integrator,
timestep, selector, or default was changed. No parameter was fitted to Grambow
or to any named reaction.

## Formulation space tested

Detailed equations and the derivation order are in `DESIGN.md`.

### 1. Inherited-order incremental channels

`SharedBondOrderChannelPrototype` splits each existing shared edge into
sigma/second/third capacity and partitions the already computed attraction by
the audited depth increments. It is conservative, symmetric, and preserves
accepted molecules.

It is not a meaningful redesign of bond order. Production computes
`spare=max(V-raw_coordination,0)` before assigning multiple-bond order. In the
crowded configurations of interest, `spare=0`, so the input has already
collapsed to order one. Only 15/200 endpoint reactions differ from the scalar
continuous model above `1e-10 eV`, and only one barrier differs materially.

### 2. Exact shared table-surface bond states

`SharedBondStateHamiltonianPrototype` makes the edge state itself carry order
0/1/2/3. State `n` uses the existing order-specific Morse surface, endpoint
valence is simultaneous, and all valid shared configurations contribute to a
finite-temperature scalar free energy.

A hard identity/state boundary initially produced a `0.069 eV` cutoff-onset
jump and was rejected. A C1 overload gate removed that discontinuity. The exact
model then exceeded 100,000 valid states on every mandatory Grambow reaction.
Exact global enumeration is therefore not a practical general representation.

### 3. Continuous shared table-surface bond free energy

`ContinuousBondFreeEnergyPrototype` replaces exponential state enumeration by
a strictly convex mean-field problem over shared continuous channel
occupancies. It uses:

- one shared channel variable at both endpoints;
- taper-weighted endpoint capacity;
- sigma/second/third hierarchy;
- the existing order-specific Morse surfaces, not the production order blend;
- the existing `0.01 eV` heavy-state scale as the entropy scale;
- a same-variable-space H-only reference, leaving H-state energy ownership
  unchanged;
- live KKT multiplier terms for geometry-dependent capacity constraints;
- a C1 parameter-free overload gate, with one valence unit as its natural
  scale.

The scalar energy—not detached occupancy—is differentiated. The established
topology/angle membership remains unchanged in this experiment so the test
isolates radial bond-energy representation. A future unified model would still
need to make topology order consume the same shared state without losing
conservative derivatives.

### 4. Heavy-overlap gated variant

`OverlapGatedBondFreeEnergyPrototype` additionally required normalized
single-Morse overlap before activating the redesigned surface. This was a
parameter-free physical hypothesis, not a threshold fit. It nearly reverted
the independent water scan to production and was rejected without further
tuning.

Linear allocation was also considered and rejected before implementation:
preference ties make the solution non-unique and create force cusps. Arbitrary
activation widths or curvature constants were not tested because they would
need independent provenance or fitting.

## Unit, symmetry, force, and preservation gates

The 26 focused tests pass. They cover:

- research-only isolation;
- exact energy and force preservation for H3, methane, formaldehyde, water,
  ethane, methanol, hydroxylamine, and hydrogen peroxide;
- shared endpoint capacity and channel hierarchy;
- exact attraction partition in the inherited-channel control;
- atom-label permutation symmetry;
- finite-difference force agreement in constrained and unconstrained regions;
- preference-exchange continuity;
- C1 continuity where an extra contact crosses the radial cutoff.

For the redesigned continuous free energy, the crowded force probe agrees with
central finite difference within `1e-4 eV/angstrom`; the observed difference
was about `5.5e-5 eV/angstrom`. Reported capacity violations are below `1e-8`.

All eight required accepted structures take the exact identity path. Their
energies agree with production within `2e-11 eV` and forces within
`2e-9 eV/angstrom`.

Complete repository pytest with a workspace-local temporary directory:

`318 passed, 1 skipped in 226.08 s`.

## Frozen 200-reaction benchmark

All continuous candidates evaluated all 200 reactions with no failures.

| formulation | barrier MAE | barrier RMSE | max barrier error | sign | reaction MAE | reaction RMSE | max reaction error |
|---|---:|---:|---:|---:|---:|---:|---:|
| production | 4.5195 | 6.4831 | 35.5021 | 87.0% | 4.4058 | 7.0688 | 49.3746 |
| local v0 | 2.1161 | 2.7611 | 10.6982 | 95.5% | 2.7231 | 3.4901 | 10.0616 |
| scalar continuous edge | 2.2595 | 2.9019 | **10.0430** | 94.0% | 2.8757 | 3.6230 | 10.0924 |
| inherited-order channels | 2.2451 | 2.8929 | **10.0430** | 94.5% | 2.8756 | 3.6229 | 10.0924 |
| redesigned bond free energy | **2.1111** | **2.6486** | 10.5142 | **98.5%** | **2.5000** | **3.1821** | **9.9922** |

The redesigned candidate improves/worsens/leaves unchanged 140/35/25 barriers
relative to production and 100/82/18 relative to v0. For reactions the counts
are 118/45/37 versus production and 102/62/36 versus v0.

It removes all 16 production barrier catastrophes over 10 eV and introduces
none. One new barrier over 5 eV appears relative to v0: `rxn011847` is
`+6.7443 eV` versus v0's `+3.8432 eV` (production was `+8.0984 eV`). Three
barrier signs remain wrong: `rxn001097`, `rxn006753`, and `rxn010413`; v0 has
nine wrong signs.

The worst redesigned barrier remains the pre-existing `rxn008195`:
`10.5142 eV` error, versus `10.6982 eV` in v0 and `10.0430 eV` in the scalar
model. No benchmark-specific response was made to this or any other tail case.

## Mandatory microscopes

Redesigned model errors, model minus reference, in eV:

| reaction | barrier error | reaction error | interpretation |
|---|---:|---:|---|
| rxn006559 | +2.5558 | +6.5012 | removes production's -35.502 barrier catastrophe |
| rxn011804 | +2.3858 | +4.1441 | improves production and v0 barrier |
| rxn004353 | +2.5676 | -0.0458 | good reaction energy; barrier worse than scalar |
| rxn000096 | +0.3237 | +4.9162 | removes production's -21.968 barrier catastrophe |
| rxn010742 | +2.3472 | +1.2074 | removes production's -19.573 barrier catastrophe |
| rxn000105 | +6.1756 | +7.2319 | modestly improves v0; production's near-zero error remains accidental cancellation |

These cases were inspected after the formulation was fixed. None supplied a
parameter, branch, molecule rule, or objective weight.

## Independent QM microscopes

Across 98 dense transfer geometries:

| formulation | MAE | RMSE | maximum absolute residual |
|---|---:|---:|---:|
| production | 0.33572 | 0.58740 | 3.56781 |
| local v0 | 0.22828 | 0.26378 | 0.56702 |
| scalar / inherited channels | **0.22624** | **0.26255** | 0.56702 |
| redesigned bond free energy | 0.27699 | 0.34991 | 1.18072 |
| overlap-gated variant | 0.33487 | 0.58298 | 3.51924 |

Formaldehyde, H3, and methane remain exactly unchanged in every candidate. The
redesigned model's regression is concentrated in the two-oxygen water-transfer
microscope:

- v0 RMSE: `0.20656 eV`;
- scalar/inherited-channel RMSE: `0.20121 eV`;
- redesigned free-energy RMSE: `0.47040 eV`;
- overlap-gated RMSE: `0.97777 eV`;
- production RMSE: `0.98667 eV`.

The ungated redesign retains a real improvement over production but fails the
explicit no-regression gate against the existing research models. The overlap
variant demonstrates that merely suppressing the redesign for weak O-O overlap
also suppresses the useful capacity correction; it is not a solution.

## Matched NVE

The redesigned continuous free energy completed two focused trajectories with
finite energies and zero move caps:

| case | max absolute drift | result |
|---|---:|---|
| water transfer, 250 x 0.25 fs | 0.0090484 eV | finite, zero caps |
| symmetric preference exchange, 300 x 0.02 fs | 0.000002378 eV | finite, zero caps |

The crowded Grambow reactant did not complete. During integration, the SciPy
reference solve returned `Inequality constraints incompatible` for its H-only
reference even though its reported endpoint-capacity and hierarchy violations
were both zero. This points to an optimizer robustness failure rather than a
demonstrated singular physical state, but it is still a failed NVE/large-motion
gate. It was not papered over by accepting an unsuccessful optimizer result.

## Golden production control

The unchanged production engine passed 19/20 named golden checks and the full
2 x 330-atom, 1 ps dense soup stress. The golden wrapper's embedded pytest step
reported 49 setup errors because its subprocess could not create files in the
default Windows temporary directory. The same repository suite, run directly
with a workspace-local temporary directory, passed 318/318 executed tests with
one skip. The failure is environmental rather than a production regression,
but the generated golden report is correctly retained as `FAIL` rather than
being relabelled.

The full validation run changed no force-field parameters. Its output was
redirected to research diagnostics so the existing baseline files were not
overwritten.

## Scientific interpretation and next step

The evidence answers the clarified question more strongly than the original
incremental-channel task:

1. **The current bond concept is not fully expressive.** When bond order is
   derived only after raw coordination exhausts spare valence, crowded
   chemistry is forced back to single-order attraction. A shared model that
   chooses bond ownership and order together improves the external reaction
   benchmark without fitting.
2. **Multiple channels alone are insufficient.** Channels that inherit the
   old order are almost exactly the scalar model.
3. **Exact global valence-bond states are too expensive.** The mandatory cases
   exceed 100,000 configurations before dynamics.
4. **A continuous shared bond free energy is the most promising architecture
   tested.** It is compact, symmetric, parameter-free relative to current
   tables, and has the best Grambow aggregate.
5. **The present coupling to H competition is incomplete.** The water QM
   regression shows that subtracting an H-only reference while changing an
   O-O bond component does not yet reproduce the validated H-transfer surface.
6. **The research solver is not production-grade.** A robust differentiable
   convex solver or analytic dual formulation is required before longer NVE or
   Torch/GPU work.

The next justified project is not another gate or benchmark adjustment. It is
to derive a unified shared H/heavy bond-capacity free energy (or an equivalent
dual formulation) in which H transfer, heavy bond order, topology order, and
radial energy come from the same constrained scalar. That derivation needs new
independent H/O and multi-centre QM configurations before production code is
touched. Until then, retain this work as research evidence and keep production
unchanged.

## Evidence files

- `research_data/benchmark/diagnostics/bond_channel_comparison.csv`
- `research_data/benchmark/diagnostics/bond_channel_comparison.json`
- `research_data/benchmark/diagnostics/bond_channel_analysis.json`
- `research_data/benchmark/diagnostics/bond_free_energy_comparison.csv`
- `research_data/benchmark/diagnostics/bond_free_energy_comparison.json`
- `research_data/benchmark/diagnostics/heavy_valence_qm_formulations.csv`
- `research_data/benchmark/diagnostics/heavy_valence_qm_formulations.json`
- `research_data/benchmark/diagnostics/heavy_valence_formulation_nve.json`
- `research_data/benchmark/diagnostics/bond_free_energy_nve.json`
- `research_data/benchmark/diagnostics/bond_channel_production_validation.json`
- `research_data/benchmark/diagnostics/bond_channel_production_validation.md`
