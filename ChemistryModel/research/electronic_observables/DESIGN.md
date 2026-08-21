# Electronic-observable validation foundation

## Scope

This directory is research-only. It does not register a physics selector,
modify production equations, add an MD force, or fit a parameter. Its purpose
is to measure which electronic information is absent from the unified radial
bond-capacity model before another functional form is proposed.

The current unified radial baseline remains frozen:

- Grambow barrier MAE: 1.475 eV
- barrier RMSE: 1.858 eV
- barrier sign agreement: 99.5%
- reaction MAE: 2.233 eV
- water-transfer QM RMSE: 0.214 eV

## Missing physical variable

Bond capacity answers **how much bonding participation an atom can allocate**.
It does not identify the orientation or redistribution of the valence density
that supplies that participation. Two geometries can have nearly identical
radial distances and scalar coordination while differing in:

- permanent charge separation;
- orientation of the molecular dipole;
- anisotropy and softness of the induced response;
- lone-pair versus bond-directed density;
- three-centre charge redistribution during transfer.

The missing state is therefore best phrased as a **state-conditioned local
valence-density response**, not another scalar angle or bond-order correction.
That hypothesis is to be constrained by observables before it is represented
by a model.

## Observable audit

| Observable | Physical information | Possible later use | Failure mode addressed | QM calculation | Scale |
|---|---|---|---|---|---|
| Relative energy and analytic force | Local potential surface and its derivative | Required scalar/force target for any conservative model | Error cancellation and incorrect forces hidden by energy-only fitting | DFT energy + analytic gradient | Cheap for this set |
| Total dipole vector | Origin-independent charge separation for neutral systems | Constrain permanent electronic state or validate a charge/multipole model | Scalar bond capacity cannot distinguish polar orientations | One-particle density property | Very cheap after SCF |
| MBIS charges and atomic multipoles | Compact, density-derived atom-centred representation | Candidate proxy for local permanent charge/multipole targets | Where charge redistributes in transfer and polar bonds | Density partition after SCF | Cheap-to-moderate grid step |
| Mulliken/Lowdin charges | Basis-space population comparators | Detect partition dependence; never authoritative alone | A seemingly good charge result that is partition-specific | Density population analysis | Negligible |
| Static polarizability tensor | Linear induced response and directional electronic softness | Constrain a future variational polarisation state | Permanent multipoles alone cannot represent environment response | Analytic DFT response | Roughly one extra SCF/response solve |
| Density/bond critical points | Real-space density topology between atoms | Potential diagnostic for shared/three-centre electronic regions | Ambiguous bond ownership during reactions | Dense wavefunction grid plus QTAIM topology | Deferred: higher workflow and robustness cost |

MBIS is intentionally treated as a **partition-dependent proxy**, not a
physical observable equal to a unique atomic charge. It is included because it
constructs a compact atom-centred pro-density from a reference electron
density and was designed for force-field development [Verstraelen et al.,
JCTC 2016](https://doi.org/10.1021/acs.jctc.6b00456). Total dipoles and
polarizabilities are the stronger molecular constraints. Psi4's documented
one-electron property and response interfaces are used for the calculations
([OEProp](https://psicode.org/psi4manual/4.0b4/oeprop),
[properties driver](https://psi4.github.io/psi4docs/master/api/psi4.driver.properties.html)).

QTAIM/bond-critical-point descriptors are not in version 1. They need a
separate topology tool, stable grid and critical-point matching across bond
formation. Adding them now would expand the project before dipoles,
polarizabilities and atom-centred multipoles have shown whether they are
sufficient.

## Dataset

`build_manifest.py` creates 30 neutral, fixed-Cartesian geometries in 12
families:

- H2 equilibrium;
- H3 transfer and separated reactant;
- linear/perpendicular H + H2 approaches;
- H + CH2O transfer and separated reactant;
- water transfer and separated reactant;
- fixed-OH water angle scan;
- OH radical stretch;
- H approaching water from lone-pair, out-of-plane and H-side directions;
- equilibrium methane, ethane, formaldehyde and N2.

The established reactive microscope coordinates are copied exactly from
`research_data/qm_residual/dense_scan_geometries.json`. Other geometries are
explicitly generated and preserved in the manifest.

Roles are fixed before calculation:

- `characterisation`: may be inspected while formulating a model;
- `validation`: rejects a frozen formulation but is not tuned point by point;
- `final_holdout`: remains unopened until a future candidate and its
  parameters are frozen.

No fitting occurs in this phase. These roles prevent the data foundation from
quietly becoming a 30-point training set.

## Quantum convention

Version 1 deliberately matches the established QM-residual level:

- unrestricted/restricted wB97X-D / jun-cc-pVDZ;
- RKS for singlets, UKS for open-shell doublets;
- density fitting;
- exact input coordinates, C1 symmetry, no reorientation, no centre-of-mass
  shift;
- analytic gradients;
- total dipole, Mulliken, Lowdin, MBIS charges, MBIS atomic dipoles and raw
  MBIS quadrupoles;
- analytic static dipole-polarizability tensor.

This economical DFT level is a **diagnostic continuity choice**, not a claim of
benchmark-quality electronic observables. A future parameter project should
audit basis/method sensitivity on a subset before fitting.

## Provenance and quality control

`compute_observables.py` is resumable and atomically checkpoints every point.
The metadata records method, basis, reference, Psi4/Python versions, manifest
hash, unit conversions and property conventions. Raw vector/tensor quantities
are retained; plots and scalar descriptors are derived later.

`validate_dataset.py` checks:

- completeness and calculation status;
- charge conservation for every partition;
- net-force residual;
- MBIS reconstruction of the total dipole;
- polarizability tensor symmetry and positive eigenvalues;
- UKS/RKS spin contamination from occupied-orbital overlaps;
- zero dipole for H2, methane and N2 by symmetry.

## Future-candidate contract

`evaluate_models.py` exports production or unified-radial energy/force results
without pretending those models predict electronic observables.
`compare_models.py` reports every improved/regressed geometry and family using
absolute residual changes, alongside a cancellation warning. The full gate
policy is machine-readable in `validation_contract.json`.

A future electronic model must remain variational/conservative, must be
permutation-equivariant, and must expose enough state to predict transferable
dipole/response changes. A lower Grambow mean alone is not acceptance.
