# Charge-localising electrostatics acceptance-gate report

## Decision

**No candidate is safe to register without a dedicated, independently
validated H/C/N/O parameterisation.**

The engine interface remains unchanged and the audited QEq implementation
remains the opt-in, disabled-by-default comparator. It has **not** been
promoted as physically correct. A corrected Gaussian atom-space QTPIE
candidate is preserved under `research/` but is deliberately not registered.

This is a stopped acceptance gate, not a partial production rollout. The
candidate obtains the desired O/H dissociation limit, then fails conditioning,
molecular-charge and reactive-coordinate gates badly enough that its otherwise
valid autograd forces cannot make it suitable for MD.

## Candidate comparison

### Shielded QEq and ReaxFF-style QEq

Replacing `1/r` by `1/sqrt(r^2 + gamma^2)` regularises short range only. For
two atoms with one global neutral constraint, the stationary long-range charge
has the form

`q(infinity) = (chi_B - chi_A) / (eta_A + eta_B)`.

The shield tends to zero with the off-diagonal Coulomb coupling, so it cannot
make this charge vanish. ReaxFF-style shielding is therefore important for a
coherent QEq parameter convention, but is not a cure for global-QEq fractional
charge dissociation. Neither was implemented as a purported solution.

### QTPIE

QTPIE was designed around pairwise transfer/effective electronegativities whose
driving force vanishes with overlap. The original bond-space model and exact
atom-space reformulation are the most relevant topology-free candidates:

- Chen and Martinez, *QTPIE: Charge Transfer with Polarization Current
  Equalization* ([preprint](https://arxiv.org/abs/0807.2068)).
- Chen and Martinez, *A unified theoretical framework for fluctuating-charge
  models in atom-space and bond-space*
  ([preprint](https://arxiv.org/abs/0807.2174)).

Prior repository research already found that the historical Slater convention
has a negative projected H2 hardness mode at equilibrium. This investigation
therefore tested the corrected Gaussian atom-space diagnostic, including the
correct unlike-Gaussian beta mapping and explicit bohr/hartree conversion.

It passes isolated O/H localisation but fails the full gate:

| Check | Audited QEq comparator | Gaussian QTPIE candidate |
|---|---:|---:|
| O charge at 100 A from H | -0.1547 e | 0 e |
| O/H electrostatic energy at 100 A | -0.3259 eV | 0 eV |
| H2 dipole | ~0 D | 0 D |
| CH4 dipole | ~0 D | 0 D |
| CH4 carbon charge magnitude | 0.0675 e | 7.842 e |
| H2O dipole | 0.641 D | 2.394 D |
| CH2O dipole | 1.958 D | 1.052 D |
| H3 minimum projected hardness eigenvalue | +12.438 eV | **-0.733 eV** |
| CH4 projected condition number | 1.39 | 86.2 |

The zero methane dipole is only symmetry; it hides gross, cancelling charges.
The negative H3 projected eigenvalue means the constrained quadratic problem
is not a minimum even though a linear solver returns finite zero charges for
identical electronegativities.

The stored 8x8 reactive grids expose the practical consequence:

| System | Maximum adjacent delta E | Maximum adjacent atomic delta q |
|---|---:|---:|
| H + CH2O | 5.079 eV | 28.95 e |
| H + CH4 | 160.314 eV | 424.40 e |
| H + H2O | 0.176 eV | 0.144 e |

These are response-matrix singularities/near-singularities, not chemical
signals. They are unacceptable for reactive dynamics.

### SQE and ACKS2

Split-charge equilibration supplies the right localisation mechanism through
transfer variables/hardness, but the previous provenance audit found no
complete, internally compatible reactive H/C/N/O set (nitrogen and its pair
transfer terms are missing). Its traditional fixed transfer graph also needs
a smooth topology-free generalisation before use here.

ACKS2 is mathematically attractive but was stopped at the same provenance gate:
the repository investigation found no defensible complete H/C/N/O production
parameter convention to reproduce without fitting. Inventing missing numbers
would violate this task.

## Architecture and implementation status

The production path remains:

`ReactiveSimulation -> ChemistryEngine -> UnifiedRadialHamiltonian -> extensions`

`ElectrostaticEnergyTerm` is still registered exactly once. The rejected
candidate imports the same term boundary and changes only the charge functional
inside a research module. No bonded, H-state, valence-capacity, geometry,
barrier or integration parameter was changed.

The new research tests freeze four facts:

1. QEq retains separated O/H charge while QTPIE localises it.
2. The tested QTPIE H3 constrained hardness is indefinite.
3. The tested QTPIE methane response is unphysical.
4. Its differentiable energy still gives finite, translationally invariant
   forces and agrees with finite differences.

This distinction matters: the candidate is rejected for physics/conditioning,
not for an autograd implementation error.

## QM residual gate

The task required QM residual evaluation only **after** fundamental physics
validation. The candidate did not pass that gate, so it was not used to
overwrite or publish a new residual dataset. Evaluating its electrostatic term
on the already stored reaction geometries was sufficient to reveal 5--160 eV
adjacent jumps and 29--424 e charge jumps.

For context, the valid same-engine audit of the active QEq comparator was
already slightly worse than electrostatics-disabled unified radial overall
(MAE 0.98408 -> 0.98443 eV; RMSE 1.42116 -> 1.42177 eV), and water worsened
(RMSE 1.23349 -> 1.23580 eV). It remains a comparator, not accepted physics.

## Recommended next step

Do not tune ChemistryModel around either current QEq or this rejected QTPIE
candidate. The scientifically defensible next project is a dedicated,
overdetermined H/C/N/O charge-response parameterisation for a localising model,
with SQE-derived continuous transfer channels the strongest existing
mathematical starting point and EEQBC retained as a ready-made comparator.
That project must independently constrain dipoles, polarizabilities, fragment
separation, radicals and reactive contacts while enforcing positive projected
hardness across the training and hold-out geometries.

Until then, keep electrostatics opt-in and disabled by default.
