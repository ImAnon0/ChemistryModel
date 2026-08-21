# Electronic-observable foundation report

## Decision

**The next useful state variable is a local, environment-conditioned valence
density representation capable of predicting both permanent multipoles and
anisotropic response.**

The evidence does not justify another blind energy term. It also does not yet
justify selecting charges, induced dipoles, lone-pair vectors or a particular
orbital-channel Hamiltonian as the production representation. The framework
now provides the data needed to distinguish those proposals.

## Completed evidence

The version-1 dataset contains 30/30 successful wB97X-D/jun-cc-pVDZ
calculations. It includes energies, analytic forces, total dipoles, Mulliken,
Lowdin and MBIS charge proxies, MBIS atomic dipoles/quadrupoles and analytic
static polarizability tensors.

Quality result: **243/243 available checks pass**.

- all three charge schemes conserve total charge within the documented
  numerical tolerance;
- MBIS atomic charges/dipoles reconstruct the total dipole;
- all response tensors are symmetric within response tolerance and have
  positive eigenvalues;
- all UKS/RKS points pass the documented spin-contamination gate (largest
  doublet value is 0.76266 versus the exact 0.75 target);
- H2, methane and N2 have zero dipole to the symmetry tolerance;
- no calculation failed or returned a non-finite result.

Summed calculation time was approximately 108 seconds on the local
`chem-sapt` environment. This is small enough for targeted future validation,
but not a proposal to apply response calculations to the whole Grambow set.

The evaluator was also run for production and the frozen unified-radial
research model. On the four selected water-transfer rows (including the zero
reference), relative-energy MAE changes from 0.783 eV in production to 0.130
eV in unified radial, while force RMSE changes from 7.267 to 0.886 eV/A. One
transfer point improves strongly and one regresses
slightly, which the per-geometry comparison retains instead of hiding inside
the mean. Other selected families are unchanged between those two models in
this focused set. This is a framework smoke comparison, not a replacement for
the established 0.214 eV full water-microscope result.

## What the data says

### 1. Permanent polarity is informative, but atomic charges alone are not enough

At equilibrium the total dipoles distinguish electronic environments that a
radial capacity model does not expose:

- methane: 0.000 D;
- formaldehyde: 2.535 D;
- water at 104.5 degrees: 2.132 D;
- OH at 0.97 A: 1.816 D.

During the fixed-OH water angle scan, the dipole changes from 2.314 D at 90
degrees to 1.873 D at 120 degrees, while the isotropic polarizability changes
only from 1.022 to 1.008 A^3. The MBIS charge span moves in the opposite
direction, from 1.252 to 1.476 e. No single scalar charge or radial occupancy
captures all three behaviours.

Charge partitions also disagree numerically by construction. They should be
used to localise *where* redistribution happens and to test robustness across
partitions, while the molecular dipole remains the stronger target.

### 2. Polarizability carries independent directional information

The water-angle polarizability anisotropy fraction rises from 0.070 at 90
degrees to 0.290 at 120 degrees even though the isotropic response barely
changes. H2 is also strongly anisotropic despite having zero charge separation.
This separates electronic softness/direction from permanent polarity.

For H + formaldehyde transfer, the isotropic polarizability grows from 2.288
to 4.110 A^3 across the selected region while the total dipole falls from 2.535
to roughly 1.98 D. The two observables therefore constrain different missing
degrees of freedom.

### 3. Three-centre and approach regions require state redistribution

Relative to separated H + H2, the selected linear approach points change the
isotropic polarizability from 0.574 to 1.023 A^3. The perpendicular 1.5 A
geometry has a much larger 0.601 D dipole than the selected linear approaches
despite comparable close-contact physics. Orientation is therefore not
recoverable from a scalar edge strength alone.

H3 transfer similarly raises the isotropic response from 0.535 A^3 in the
separated reference to about 1.17--1.20 A^3 in the transfer region, while its
small but changing dipole and MBIS distribution track symmetry breaking. This
is the kind of three-centre electronic state the previous P2 scalar could not
represent.

### 4. Water exposes the same missing variable without being a fitting target

The water + H directional probes have relative energies of 0.000, 0.025 and
0.427 eV for the chosen out-of-plane, lone-pair-side and H-side references, but
their dipole magnitudes span 1.157--2.727 D. That is a large electronic-state
change at the same probe radius. It supports using water as a diagnostic of
transferable directionality, not adding a water-specific energy correction.

## Observable ranking for the next phase

1. **Total molecular dipoles** — strongest cheap target for permanent
   electronic state; physically observable and immediately discriminating.
2. **Static polarizability tensors** — strongest target for induced,
   directional response; crucial because isotropic and anisotropic changes are
   independent of the dipole.
3. **MBIS atomic charges plus atomic dipoles** — useful local supervision for
   assigning the molecular response to atoms/contacts, provided partition
   dependence is explicit and molecular observables remain primary.
4. **Forces and relative energies** — mandatory conservative-surface gates,
   not themselves the missing electronic variable.
5. **Density critical-point descriptors** — defer until a candidate cannot be
   distinguished by multipoles and response; their topology tracking and
   compute workflow are materially more complex.

## Architectural implication

The evidence favours a future research model with a small per-atom or
per-local-environment electronic state that can:

- produce permanent charge/dipole/multipole moments;
- respond variationally to neighbouring fields and bond-capacity state;
- rotate equivariantly with geometry;
- redistribute continuously through H and heavy-atom transfer regions;
- vanish or localise correctly at fragment separation;
- derive its energy and forces from one scalar variational energy.

That description is compatible with, but does not yet choose between, a
constrained multipolar/polarisable model, directional valence channels, or a
small shared three-centre electronic state. An induced-dipole-only term is not
enough: without a compatible permanent source it cannot reproduce the
observed zero-field molecular dipoles. A charge-only model is also not enough:
it lacks the independent response anisotropy seen here.

## What is not claimed

- The present DFT/basis level is not a final parameterisation standard.
- MBIS charges are not unique observables.
- The 30 geometries are not a fit set and do not establish transferability.
- No electronic energy, charge or force has been connected to MD.
- No production physics was changed.
- No functional form or parameter has been selected or fitted.

## Recommended next experiment

Freeze a very small candidate family before opening final holdouts. Each
candidate should predict the total dipole and polarizability tensor as well as
energy/force on the characterisation subset. Reject candidates that cannot
represent water-angle, H-H2 orientation and transfer response simultaneously.
Only then open the validation and final-holdout roles and run the existing
water, H-transfer, molecule, NVE and Grambow gates.

The framework supports that experiment; the current result is a foundation,
not permission to integrate electrostatics or directional physics.
