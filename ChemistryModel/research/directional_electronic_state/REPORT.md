# Directional electronic-state investigation report

## Decision

**REJECT the state-conditioned SAPT-P2 coupling diagnostic. Preserve unified
radial unchanged. Do not integrate or tune.**

The investigation first identified the missing variable rather than beginning
with an equation: scalar edge occupation, bond order, capacity, and capacity
price do not determine the orientation of local valence density. The smallest
general missing state is a bond-assignment-conditioned local density multipole
(dipole and/or traceless quadrupole), possibly with explicit three-centre
coherence.

Induced dipoles, split charges, and free orbital frames were stopped at the
formulation gate because the current model lacks the compatible permanent
field, parameters, or direct electronic observables required to identify them.
One no-fit falsification probe was admitted: apply the repository's
independently calibrated SAPT `P2` exchange-density anisotropy to the existing
H-state off-diagonal coupling, without changing diagonal radial energy or
settled molecules.

That probe is smooth, conservative, and numerically healthy, but it narrowly
worsens water and the frozen Grambow aggregate. Exchange-density anisotropy is
therefore not the missing transferable directional state.

## Missing-variable conclusion

Unified radial retains:

- edge state probabilities and expected order;
- scalar valence consumption and capacity prices;
- coherent mixing between alternative H assignments.

It discards:

- the orientation of lone-pair-rich density;
- local dipole/polarisation response;
- state-conditioned density anisotropy;
- whether a three-centre electronic domain is aligned with the spectator
  environment.

The current angle model attempts to reconstruct these from nuclei plus scalar
coordination. The prior geometry work showed that this inverse reconstruction
is not unique. The missing object is electronic and directional, not another
choice of angle weight.

## Candidate and controls

`StateConditionedP2CouplingPrototype` modifies only direct H-transfer
off-diagonals:

    H_st = H_st_radial sqrt(a_target_in_s * a_target_in_t)

where each `a=exp(k q2)` is evaluated in the bond-assignment state where that
target edge is unoccupied. H/C/N/O `k` values are frozen from the separate
SAPT0/jun-cc-pVDZ EXCH10 programme. No scalar multiplier is fitted or exposed.

The Boolean-off control reproduces unified radial to `1e-10 eV` in energy and
`1e-9 eV/A` in force. H3 and every settled accepted molecule are exact identity
paths.

## QM microscopes

Dense relative-energy RMSE, eV:

| model | H3 | methane | formaldehyde | water | all 98 |
|---|---:|---:|---:|---:|---:|
| production | 0.22311 | 0.21841 | 0.38103 | 0.98667 | 0.58740 |
| **unified radial** | **0.22311** | **0.21841** | 0.38103 | **0.21363** | 0.26544 |
| state-conditioned P2 | **0.22311** | **0.21841** | **0.37643** | 0.21601 | **0.26447** |

The factor is active: 81 state transitions span `0.98670` to `1.01713`.
Formaldehyde and the all-system aggregate improve slightly, but water worsens
by `0.00238 eV` RMSE and fails the explicit `<=0.21363 eV` gate. No sign or
strength reversal was tried because that would fit the diagnostic.

## Static and molecule gates

| gate | result |
|---|---|
| finite-difference force | pass; difference `4.7e-10 eV/A` |
| permutation symmetry | pass; energy `5.3e-15 eV`, force `1.1e-9 eV/A` |
| O-H outer-cutoff continuity | pass; energy span `9.2e-8 eV` |
| H2 and H3 | exact unified-radial identity |
| methane, formaldehyde, water, ammonia | exact identity when settled |
| ethane, methanol, hydroxylamine, peroxide | exact identity when settled |
| NaNs / non-finite force | none |

## Matched NVE

| case | unified radial drift | P2 drift | caps | result |
|---|---:|---:|---:|---|
| water transfer, 250 x 0.25 fs | 0.01540 eV | 0.01551 eV | 0 | pass |
| crowded Grambow reactant, 250 x 0.10 fs | 0.000122 eV | 0.000122 eV | 0 | pass |
| symmetric exchange, 300 x 0.02 fs | 0.00000153 eV | 0.00000153 eV | 0 | pass |

The probe is conservative in practice and introduces no optimizer failure or
move cap. Its rejection is scientific, not numerical.

## Frozen Grambow 200

| model | barrier MAE | barrier RMSE | sign | reaction MAE | reaction RMSE |
|---|---:|---:|---:|---:|---:|
| production | 4.51955 | 6.48308 | 87.0% | 4.40577 | 7.06884 |
| **unified radial** | **1.47462** | **1.85773** | **99.5%** | **2.23297** | **2.87249** |
| state-conditioned P2 | 1.47484 | 1.85920 | **99.5%** | **2.23297** | **2.87249** |

The probe changes 122 transition-state barriers above `1e-10 eV`, with a
maximum shift of `0.1920 eV`. Relative to unified radial it improves 56,
worsens 67, and leaves 77 barrier errors unchanged. All 200 reaction energies
are exactly unchanged because the endpoints are settled one-state structures.
There are no evaluation failures and no new error above 5 eV.

## Why P2 failed and what remains plausible

The experiment supports the architecture but rejects the descriptor:

1. A state-conditioned directional factor can be embedded in the existing
   Hamiltonian while preserving symmetry, conservative forces, cutoffs,
   molecules, NVE, and the unified radial ownership model.
2. The independently anchored exchange-density `P2` moment is weak and has the
   wrong transferability for covalent mixing: improvement in formaldehyde does
   not transfer to water or the reaction benchmark.
3. A physical induced dipole cannot be added alone. Without permanent charges
   or multipoles its source field is zero; adding bond charges would reopen the
   unresolved electrostatics parameter project.
4. The current energy-only QM dataset cannot identify dipole hardness,
   polarizability, damping, lone-pair orientation, or orbital-coupling
   parameters. Fitting any of them to the same transfer energies would create
   an underdetermined water patch.

The next justified work is **data, not another functional-form sweep**. Build
an independent electronic-observable set for neutral H/C/N/O molecules and
reactive contacts containing:

- molecular dipoles at equilibrium and along held-out distortions;
- finite-field molecular polarizability tensors;
- state/density-derived atom-centred dipole and quadrupole moments under one
  fixed partition convention;
- donor-H-acceptor angular scans at fixed radial coordinates;
- water, ammonia, formaldehyde, methane, peroxide, hydroxylamine, radicals,
  and non-water hold-outs.

Only after that dataset exists can a variational dipole/tensor or compact
three-centre state be parameterized without using Grambow or water energies as
its definition. Unified radial remains the baseline during that project.

## Evidence

- `research/directional_electronic_state/FORMULATION_GATE.md`
- `research/directional_electronic_state/DESIGN.md`
- `research/directional_electronic_state/prototype.py`
- `research/directional_electronic_state/validate.py`
- `research/directional_electronic_state/compare_grambow.py`
- `research_data/benchmark/diagnostics/directional_electronic_validation.json`
- `research_data/benchmark/diagnostics/directional_electronic_qm.csv`
- `research_data/benchmark/diagnostics/directional_electronic_grambow.json`
- `research_data/benchmark/diagnostics/directional_electronic_grambow.csv`
