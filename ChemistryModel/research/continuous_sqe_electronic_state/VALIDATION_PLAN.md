# Continuous SQE electronic state: parameterisation and validation plan

## Gate order

No later gate can compensate for failure of an earlier one:

1. equation/provenance reproduction;
2. convexity, conservation and invariance;
3. dissociation and cutoff continuity;
4. independent molecular electronic observables;
5. reactive-coordinate response;
6. combined ChemistryModel energy/force residuals;
7. only then, opt-in MD/NVE evaluation.

Existing bonded, H-state, angle, barrier and valence-capacity parameters remain
frozen throughout this project.

## Quantum convention

Use restricted or unrestricted `omegaB97M-V/def2-TZVPPD` as appropriate, with
tight SCF thresholds, an ultrafine grid, stable-wavefunction checks and stored
`<S^2>` for open-shell or stretched configurations. Record program/version,
input, geometry, charge, multiplicity, convergence and density hashes.

For each selected geometry retain:

- total energy and Cartesian gradient;
- molecular dipole and quadrupole;
- symmetric finite-field dipoles/energies in three directions at two weak
  field magnitudes;
- polarizability tensor;
- MBIS charges, widths and volumes;
- external electrostatic potential on a reproducible shell grid;
- fragment-integrated charge for declared separation scans.

MBIS charges receive limited weight because atomic charges are representation
dependent. A stratified 15% DDEC6 audit measures partition dependence. Gas-
phase experimental dipoles and polarizabilities are independent hold-outs, not
fit targets. EEQBC uses its published parameters and is a frozen comparator.

## Dataset and split

Splitting is by molecule family before generating conformers or distortions.
No conformer, scan neighbour or isomer family may leak across a split.

### Training: 1,500 configurations

| Block | Count | Coverage |
|---|---:|---|
| equilibrium and thermal/normal-mode distortions | 600 | 30 neutral H/C/N/O species across coordination environments |
| controlled bond stretches | 240 | HH, HC, HN, HO, CC, CN, CO, NN, NO, OO in multiple environments |
| radicals | 180 | H, CH3, C2H5, OH, HO2, NH2, HCO, CH3O, CH2OH, CN and related environments |
| neutral fragment separation | 240 | 12 unlike pairs, multiple orientations and 20 distances including asymptotic points |
| reactive/nonreactive contacts | 240 | H abstraction, recombination, O/O, C/O, C/N and N/O approach/retreat paths |

Training families include ethane/ethene/acetylene, ethanol, dimethyl ether,
acetaldehyde, formic acid, formamide, hydrazine, nitrous acid, HCN/HNC,
acetonitrile and balanced radicals. H2 is included because the HH transfer and
response channel is otherwise unidentifiable, but most H2 distortions remain
evaluation-only.

### Model-selection validation: 400 configurations

- 160 distortions from eight molecule families absent from training;
- 80 stretches in held-out chemical environments;
- 60 held-out radical geometries;
- 50 unseen neutral-fragment pair/orientation scans;
- 50 unseen reactive approaches.

This split selects C0 versus C1 versus C2, stopping, and regularisation. It may
not be moved into training after results are inspected.

### Locked final hold-out: 450 configurations

Hold out CH4, H2O, NH3, CO, CO2, CH2O, CH3OH, CH3NH2, H2O2 and NH2OH as
complete species wherever the need to identify one elemental pair does not
make that impossible. Include:

- 100 equilibrium/distorted principal-molecule geometries;
- 100 additional normal-mode/thermal distortions;
- 80 radical geometries for CH3, OH, NH2 and HCO;
- 80 held-out bond stretches and orientations;
- 50 fragment scans: H + H2O, H + NH3, H + CH4, H/OH, H2O/NH3, CH4/O2,
  CO/H2O and related multi-atom controls;
- 40 H-abstraction and radical-recombination contact points.

The locked set is hashed before fitting. Failure creates a new model
generation and new untouched hold-out; it never triggers tuning on this set.

## Objective and weighting

Every molecule family receives equal total weight within each block.

| Target | Weight | Purpose |
|---|---:|---|
| molecular dipole vector | 25% | zero-field polarity and symmetry |
| finite-field response/polarizability | 25% | distinguish hardness/capacity from electronegativity |
| external electrostatic potential | 15% | spatial electrostatic fidelity without charge-partition dependence |
| MBIS charges | 10% | atom-resolved guidance only |
| separated-fragment net charge | 15% | localisation and asymptotic transfer |
| ordered-path charge/dipole smoothness | 5% | reactive and dissociation continuity |
| weak log-scale parameter regularisation | 5% | discourage extreme ratios without importing published values as targets |

Zero-field total QM energies are not a primary fit target because there is no
unique decomposition into bonded and electronic contributions. Finite-field
energy changes are valid electronic-response targets. Combined ChemistryModel
energies and forces are a later validation gate, not a way to retune bonded
physics around the charge model.

## Non-negotiable numerical gates

For every training, validation and hold-out geometry:

- finite energy, charges and derivatives;
- total-charge residual below `1e-12 e` in CPU float64;
- response matrix symmetry below `1e-12` relative error;
- smallest response eigenvalue at least `1 - 1e-10` for the scaled reference
  formulation;
- condition number below `1e8` hard, below `1e6` design target;
- relative linear residual below `1e-10`;
- equivalent-atom charges within `1e-10 e`;
- permutation, translation and rotation invariance;
- central finite-difference force agreement at existing backend tolerances;
- no charge/energy/force discontinuity under `1e-4 A` perturbations;
- energy and first two radial derivatives continuous at transfer support;
- separated neutral-fragment charge below `1e-8 e` once all cross channels
  are outside support, and tending smoothly toward that limit beforehand.

Also test polarizability scaling on H-(CH2)n-H, water chains and other neutral
oligomers. Superlinear growth that is not supported by QM rejects the model
even if small-molecule dipoles pass.

## Required molecular and reactive evidence

At minimum report for H2, CH4, H2O, NH3, CO, CO2, CH2O, CH3OH, CH3NH2,
H2O2 and NH2OH:

- charges and symmetry classes;
- dipole vector/magnitude;
- polarizability tensor;
- condition number and smallest eigenvalue;
- response to bond compression/stretching;
- charge and force continuity.

Reactive grids include H3, H + H2, H + CH4, H + H2O, H + NH3 and H + CH2O.
Track split charges, atomic charges, dipole, electronic energy and force. There
must be no false bound H3/H-H2 complex and no matrix softening analogous to the
rejected QTPIE candidate.

Comparators are electrostatics-disabled unified radial, current diagnostic
QEq, rejected QTPIE diagnostics, published EEQBC and fixed-topology published
SQE where its H/C/O parameters apply.

## Overdetermination

The richest preregistered model has 26 independent parameters against 2,350
base configurations. Even retaining only one dipole magnitude and one
independent charge contrast per configuration gives more than 4,700 scalar
constraints, over 180 per parameter. Dipole components, ESP points and field
response provide tens of thousands of additional stratified observations.

Observation count alone is not proof. Acceptance also requires full-rank
family-blocked Jacobians, stable bootstrap intervals, low parameter
correlation, consistent multi-start solutions and no sharp degradation when a
molecule family is omitted.

## Estimated quantum workload

Before retries, plan approximately:

- 2,350 base DFT densities/properties;
- 550 configurations with symmetric fields in three directions at two
  magnitudes: 6,600 additional SCF calculations;
- about 80 optimisations/frequency checks;
- 120 coupled-cluster-quality audit points where single-reference diagnostics
  permit;
- about 350 DDEC6 density analyses with no additional SCF.

Total: roughly 9,100 DFT SCFs, 80 optimisation/frequency jobs and 120
higher-level audit calculations. Run a frozen 10% pilot first to test SCF
reliability, storage, response linearity, parameter Jacobian rank and cost.

## Acceptance decision

A future implementation is eligible for opt-in MD testing only if it:

1. beats C0 and published EEQBC on blocked electronic-observable validation;
2. exactly localises separated neutral fragments;
3. remains uniformly convex and well-conditioned;
4. produces smooth conservative forces;
5. improves or preserves independent unified-radial QM residual shapes without
   tuning existing physics;
6. leaves electrostatics-disabled behaviour unchanged.

Only after those gates should NVE and reactive soup tests be run. Production
promotion would require a separate explicit decision.
