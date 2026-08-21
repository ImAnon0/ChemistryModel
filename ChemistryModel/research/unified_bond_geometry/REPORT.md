# Unified bond-geometry research report

## Decision

**REJECT the four tested geometry formulations; keep unified radial as the
research baseline. Do not integrate.**

Putting geometry inside the constrained scalar is mathematically possible and
produces conservative, smooth forces. It does not, by itself, produce a better
physical geometry model. Every candidate worsens the independent water-transfer
microscope from the unified-radial `0.2136 eV` RMSE baseline. The best coupled
candidate reaches only `0.3292 eV`. Grambow was therefore not run for the
candidates: they failed the pre-benchmark scientific gate.

No production source, force-field parameter, or application selector is
modified by this work.

## Formulations tested

| formulation | role | geometry variable | one variational scalar? |
|---|---|---|---|
| unified radial | frozen baseline | established geometry layer | radial only |
| post-weighted | control | products of solved edge occupancies | no |
| variational weighted | prototype | continuous expected occupancy/order | yes |
| electron domain | prototype | soft minimum over 2/3/4-domain states | yes |
| joint local state | prototype | local joint incident-edge order distribution | yes |

All models reuse the established candidate contacts, tapers, pair energies,
capacity, valence, lone-pair squeeze, and angle stiffness. Nothing is fitted to
Grambow or to the QM microscopes.

The variational weighted models minimize radial factor free energy and geometry
energy together under factor-normalization and heavy-capacity constraints. The
joint model additionally constrains each local geometry distribution to match
the radial edge-order marginals. Its geometry energy is therefore conditional
on bonds being simultaneous rather than using a product of independent means.
Forces are gradients of the converged constrained scalar, including the
capacity-constraint envelope term.

## Independent QM microscopes

Relative-energy RMSE, eV, on the frozen dense scans:

| model | H3 | methane | formaldehyde | water | all 98 points |
|---|---:|---:|---:|---:|---:|
| production | 0.2231 | 0.2184 | 0.3810 | 0.9867 | 0.5874 |
| **unified radial** | **0.2231** | 0.2184 | 0.3810 | **0.2136** | **0.2654** |
| post-weighted | 0.2231 | 0.2136 | **0.3330** | 0.3829 | 0.3030 |
| variational weighted | 0.2231 | 0.2136 | **0.3329** | **0.3292** | 0.2837 |
| electron domain | 0.2231 | 0.2179 | 0.3824 | 0.3420 | 0.3029 |
| joint local state | 0.2231 | **0.2124** | 0.3370 | 0.3535 | 0.2931 |

The coupled weighted model is better than applying the same concept after the
radial solve (`0.3292` versus `0.3829 eV` on water). This supports variational
coupling as an architectural principle, but it does not validate the chosen
geometry variable. Formaldehyde and methane improve slightly while the critical
water surface regresses materially.

## Static physics and numerical gates

| gate | post | coupled weighted | electron domain | joint local |
|---|---:|---:|---:|---:|
| finite-difference force | pass | pass | pass | pass |
| permutation symmetry | pass | pass | pass | pass |
| outer-cutoff continuity | pass | pass | pass | pass |
| accepted molecule invariance | pass | pass | **fail** | pass |
| water not worse than radial | **fail** | **fail** | **fail** | **fail** |

The largest autograd/finite-difference discrepancy on the selected reactive
water geometry is below `4.1e-7 eV/A`. Label permutations change energies by
at most `5.6e-12 eV`; force differences are below `7.0e-6 eV/A`, consistent
with the SLSQP reference tolerance. All tested O-H outer-cutoff scans approach
zero correction force continuously and have no non-finite result.

H2 and H3 contain no heavy-centred angle, so all candidates reproduce the
unified-radial H curves. Methane, formaldehyde, water, ammonia, ethane,
methanol, hydroxylamine, and peroxide are preserved to reference-solver
tolerance by post-weighted, coupled-weighted, and joint-local models. The
electron-domain soft free energy changes accepted molecule energies by about
`0.011 eV` per active heavy centre and changes some forces, so it independently
fails molecule invariance.

## Matched dynamics

Water-transfer NVE used identical initial velocities for 250 steps at
`0.25 fs`:

| model | max absolute drift, eV | move caps | result |
|---|---:|---:|---|
| unified radial | 0.01540 | 0 | pass |
| post-weighted | 0.59403 | 0 | **fail drift gate** |
| variational weighted | 0.01310 | 0 | pass |
| electron domain | 0.01134 | 0 | pass |
| joint local state | 0.01167 | 0 | pass |

This is useful confirmation that the coupled scalar and envelope forces can be
conservative in dynamics. The sequential control is not: its solved radial
state is treated as fixed while a geometry correction depends on it, producing
the large drift expected from an incomplete response derivative. There were no
NaNs or move caps in the matched water runs.

The crowded joint-state NVE was not promoted. Its exact local state count is

    product_e (maximum_order_e + 1)

so a heavy centre with five order-0-to-3 contacts has 1,024 local states, seven
has 16,384, and eight has 65,536, beyond the prototype's 20,000-state safety
limit. A preliminary crowded solve was correspondingly impractical. Since the
same formulation already failed water, spending further compute on its crowded
dynamics would not change the decision.

## Frozen benchmark comparison

| formulation | barrier MAE | barrier RMSE | sign | reaction MAE | reaction RMSE |
|---|---:|---:|---:|---:|---:|
| production | 4.520 | 6.483 | 87.0% | 4.406 | 7.069 |
| **unified radial** | **1.475** | **1.858** | **99.5%** | **2.233** | **2.872** |
| geometry candidates | not run | not run | not run | not run | not run |

These are the frozen, already-established comparison values, not recomputed or
refitted results. Running the 200-reaction benchmark after a decisive water-QM
failure would invite selecting a geometry model for benchmark score rather than
physical transferability.

## What the failure teaches us

The core limitation is not merely that occupancy was outside the minimization.
It is the assumption that **radial bond participation and directional geometry
participation are the same variable**.

During a transfer, a contact may legitimately share attractive energy and
valence capacity without behaving as one fully directional leg of a conventional
two-bond angle. Products of mean occupancies over-count simultaneous angles.
The joint local factor fixes that correlation error, yet still forces every
simultaneous energy-bearing contact into the same pair-angle/lone-pair rule.
Its water result remains poor. A free choice among 2/3/4 ideal angles also does
not supply the missing directional electronic information.

The next scientifically justified formulation would need a distinct continuous
directional state—for example local hybrid/orbital moments or an explicit
three-centre electron-domain factor—that couples to radial capacity but is not
identical to it. It should represent a transferring three-centre domain without
enumerating every incident bond-order combination. Such a model needs its own
small-molecule geometry and transition-state reference data; it should not be
created by tuning the current angle stiffness against Grambow.

## Recommendation

Keep `UnifiedBondCapacityEnergyPrototype` as the research baseline. Reject:

1. post-solved occupancy-weighted angles because they are non-conservative in
   the matched transfer dynamics;
2. mean-field variational weighted angles because they fail water despite
   otherwise sound forces;
3. the 2/3/4 electron-domain soft minimum because it fails water and accepted
   molecule invariance;
4. exact local joint geometry states because they fail water and scale
   exponentially with local contact count.

Do not modify production physics and do not run a Grambow-driven parameter
search. If geometry research continues, investigate a compact directional or
three-centre variational representation with independent QM geometry targets.

## Reproducible evidence

- `research/unified_bond_geometry/DESIGN.md`
- `research/unified_bond_geometry/prototype.py`
- `research/unified_bond_geometry/validate.py`
- `research_data/benchmark/diagnostics/unified_bond_geometry_validation.json`
- `research_data/benchmark/diagnostics/unified_bond_geometry_qm.csv`
