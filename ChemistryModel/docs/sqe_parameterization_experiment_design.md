# ChemistryModel reactive SQE parameterization: preregistered experiment design

## Scope and decision gate

This document defines the experiment before any quantum dataset is generated
or any parameter is fitted. It does not authorize an MD implementation.

The goal is a neutral-system, H/C/N/O, topology-free extension of split-charge
equilibration (SQE) that:

- conserves total charge exactly;
- gives independently neutral fragments at infinite separation;
- varies smoothly through bond formation and dissociation;
- has a positive, numerically usable response Hessian;
- predicts molecular electrostatics and response rather than merely matching
  one arbitrary atomic-charge partition; and
- uses far fewer parameters than independent reference observations.

EEQBC remains a frozen, ready-made comparator using its published 2025
parameters. QEq, historical Slater QTPIE, corrected Gaussian QTPIE, and the
published fixed-topology SQE convention remain comparators as well.

## 1. Mathematical model to parameterize

### Neutral split-charge variables

For `N` atoms, introduce one oriented split-charge variable `p_ij = -p_ji`
for every unordered atom pair. Let `B` be the atom-pair incidence matrix and

```
q = B p.
```

Because `1^T B = 0`, `sum(q) = 0` exactly without a global QEq multiplier.
The first project is deliberately restricted to neutral systems and neutral
fragment dissociation. Charged-system reference charges are not part of this
fit.

### Energy

In eV, angstrom, and elementary charge, the model is

```
E_SQE(p; R) = chi^T q
            + 1/2 sum_i eta_i q_i^2
            + 1/2 sum_(i<j) kappa_ij(r_ij) p_ij^2
            + 1/2 sum_(i != j) q_i J_ij(r_ij) q_j

q = Bp
```

The Coulomb kernel is the interaction of normalized spherical Gaussian charge
densities,

```
J_ij(r) = k_e erf(r / sqrt(2(sigma_i^2 + sigma_j^2))) / r,
```

with the analytic `r -> 0` limit. Gaussian widths are fitted elemental
response parameters; the Coulomb constant and unit conversions are fixed.

Define the positive transfer compliance `s_ij = 1/kappa_ij` as

```
s_ij(r) = s0_ab exp[-beta (r / R_ab - 1)]
R_ab    = Rcov_a + Rcov_b,
```

where `a,b` are element types, tabulated covalent radii `Rcov` are frozen and
recorded with their source, `s0_ab = s0_ba > 0`, and `beta > 0` is global.
There is no bond list, molecular identity, coordination threshold, or hard
cutoff.

This is the minimal first-generation reactive extension. Environment-
dependent electronegativities, hardnesses, pair-specific decay exponents,
bond-order inputs, and neural corrections are explicitly excluded. They may
only be proposed after a documented failure that cannot be corrected within
this fixed model.

### Stationarity and stable variable scaling

With `H` containing atomic hardness and Gaussian Coulomb response and `K =
diag(kappa)`,

```
(K + B^T H B) p = -B^T chi.
```

Direct edge variables become poorly scaled when `kappa -> infinity`. The
reference solve should therefore use `p = S^(1/2) u`, with `S = diag(s)`,

```
[I + S^(1/2) B^T H B S^(1/2)] u
    = -S^(1/2) B^T chi.
```

This form has the same physics but remains well-scaled as transfer channels
vanish. Cycle degrees of freedom are regularized by the identity term. Charges
are recovered as `q = B S^(1/2) u`.

### Exact separated-fragment limit

For every cross-fragment pair, `s_ij(r) -> 0` continuously as `r -> infinity`.
Their contribution to `q` therefore vanishes. The remaining incidence matrix
is block-separated and each fragment charge is a sum of antisymmetric internal
transfers, hence exactly zero. No finite-distance graph classification is
needed. This preserves the defining SQE localization mechanism.

The exponential has no hard cutoff. A later production approximation may use
a compact-support `C2` or smoother switching function only if it reaches zero
with zero derivatives and reproduces the untruncated reference to a declared
tolerance. That is not part of this fit.

## 2. Independent parameter count

| Parameter family | Raw count | Independent count | Rule |
|---|---:|---:|---|
| elemental electronegativity `chi_H,C,N,O` | 4 | 3 | one additive gauge fixed by `chi_H = 0` |
| elemental atomic hardness `eta` | 4 | 4 | positive |
| elemental Gaussian width `sigma` | 4 | 4 | positive |
| unordered pair compliance `s0_ab` | 10 | 10 | HH, HC, HN, HO, CC, CN, CO, NN, NO, OO; positive and symmetric |
| global decay `beta` | 1 | 1 | positive |
| fixed covalent radii and constants | - | 0 | not optimized |
| **Total** | **23** | **22** | fixed for the first fit |

No molecule-specific, atom-type, formal-bond, spin-state, or dataset-specific
parameters are permitted. Pair-specific decay rates would add ten poorly
identifiable degrees of freedom and are prohibited initially.

An ablation fit with fixed Gaussian widths may be run diagnostically, but it
does not change the preregistered 22-parameter production candidate.

## 3. Published SQE reproduction checkpoint

Before fitting, an independent fixed-topology reference implementation must
reproduce Nistor et al. method III using its complete published convention.
For the H/C/O subset the auditable values are:

| Element | `chi` | `kappa` |
|---|---:|---:|
| H | 5.0780 | 16.1954 |
| C | 5.2086 | 8.1313 |
| O | 8.5220 | 12.4062 |

| Pair | published bond hardness |
|---|---:|
| H-C | 1.2698 |
| H-O | 0.0627 |
| C-C | 1.4719 |
| C-O | 4.9727 |

The original Si parameters and Si-containing pairs should also be reproduced
in the reference test even though Si is outside ChemistryModel's fit; doing so
checks transcription and equations. The published ESP test definition and
reported training examples should be reproduced within numerical tolerance.
These numbers validate the implementation only. They are not initialization
constraints for the new Gaussian, topology-free convention and must not be
mixed into it.

## 4. Quantum-reference convention

### Primary electronic-structure level

Use unrestricted/restricted `omegaB97M-V/def2-TZVPPD` as appropriate, with:

- an ultrafine integration grid and tight SCF thresholds;
- density fitting only with a documented matching auxiliary basis;
- stable-wavefunction checks for every open-shell or stretched configuration;
- unrestricted solutions for radicals, recording `<S^2>` and rejecting severe
  spin contamination rather than silently accepting it;
- counterpoise-free isolated-supermolecule calculations for separation scans,
  using one basis convention throughout;
- Cartesian geometries, charge, multiplicity, SCF settings, program/version,
  and convergence status stored with every result.

This level matches the modern EEQBC training convention closely enough for a
fair comparator while providing diffuse and polarization flexibility required
for response and separated fragments. It is not claimed to be exact.

### Higher-level audit subset

For 80 equilibrium/distorted structures and 40 radical/separation structures,
compute `CCSD(T)`-quality dipoles or the best feasible coupled-cluster variant
with aug-cc-pVTZ, plus aug-cc-pVQZ where affordable. This subset is not used to
add parameters; it estimates DFT reference bias. Multireference diagnostics
must flag stretched H2, N2, O2, and radical-contact points where single-
reference coupled cluster or DFT becomes ambiguous.

### Charge information

Use **MBIS populations from the converged electron density** as the primary
atom-resolved guidance because MBIS is basis-robust, reproducible, and designed
for force-field electrostatics. Store MBIS valence widths/volumes as metadata,
but do not fit them unless they correspond directly to the declared Gaussian
width model.

Atomic charges are not observables. Therefore MBIS charges receive limited
weight and may not override molecular dipoles, field response, electrostatic
potential, or correct fragment localization. DDEC6 charges should be computed
on a 15% stratified audit subset to quantify partition dependence, not blended
into the target. ESP-fitted charges are not fit targets; the quantum ESP itself
is sampled outside the van der Waals surface.

For each selected geometry record:

- MBIS charges and total-charge residual;
- molecular dipole and quadrupole from the full electron density;
- ESP values on a reproducible shell grid;
- finite-field dipole derivatives/polarizability;
- energy changes under the same weak fields;
- fragment-integrated MBIS charge for declared separation experiments.

## 5. Preregistered dataset

All configurations are deduplicated by molecular graph, geometry fingerprint,
charge, and multiplicity. Splitting is by **molecule family before geometry
generation**, never by randomly separating conformers of the same molecule.

### Training: 1,500 configurations

| Block | Configurations | Contents |
|---|---:|---|
| equilibrium and thermal distortions | 600 | 30 neutral species; normal-mode and Cartesian perturbations spanning 100-1500 K-equivalent amplitudes |
| controlled single-bond stretches | 240 | 8 distances each for HH, CH, NH, OH, CC, CN, CO, NN, NO and OO environments |
| radicals | 180 | H, CH3, C2H5, OH, HO2, NH2, HCO, CH3O, CH2OH, CN and isomeric radical environments |
| neutral fragment separation | 240 | 12 unlike fragment pairs, 20 separations/orientations each, including H/O and multi-atom pairs |
| reactive contacts | 240 | approach/retreat paths for H abstraction, radical recombination, O/O, C/O, C/N and N/O contacts; both reactive and nonreactive orientations |

Training molecule families include ethane/ethene/acetylene, ethanol, ethylamine,
dimethyl ether, acetaldehyde, formic acid, formamide, hydrogen cyanide/isocyanide,
acetonitrile, hydrazine, nitrous acid, nitromethane, hydroxyl and carbon/nitrogen
radicals, plus balanced small H/C/N/O isomers. H2 is included solely because
HH response cannot be identified otherwise. Symmetric homonuclear species are
included as exact-zero controls, not high-weight sources of electronegativity.

### Tuning validation: 400 configurations

- 160 unseen conformers/distortions from eight molecule families excluded from
  training;
- 80 bond stretches with held-out bond environments;
- 60 radical geometries from held-out radicals;
- 50 neutral-fragment separations using held-out fragment pairings;
- 50 reactive approaches using held-out orientations and collision partners.

Validation selects optimization stopping and regularization strength. It may
not be moved into training after results are seen.

### Locked final hold-out: 450 configurations

The complete familiar molecular panel is held out by species wherever
identifiability permits:

- CH4, H2O, NH3, CO, CO2, CH2O, CH3OH, CH3NH2, H2O2, and NH2OH;
- H2 equilibrium/dynamics-style distortions are evaluation-only after the HH
  stretch training points have fixed the otherwise unidentifiable HH channel;
- 100 geometries from the ten principal molecules;
- 100 normal-mode/thermal distortions;
- 80 radical geometries: CH3, OH, NH2 and HCO in held-out distortions;
- 80 bond-stretch points absent from training orientations;
- 50 two-fragment scans, including H + H2O, H + NH3, H + CH4, H2O + NH3,
  CH4 + O2, and CO + H2O;
- 40 reactive-contact points near H abstraction and radical recombination.

No final-holdout result may trigger parameter adjustment. A failed hold-out
means the preregistered form is rejected or a second model generation is
designed with a new untouched hold-out.

### Experimental dipole hold-out

Gas-phase equilibrium dipoles from NIST CCCBDB and their primary references
are kept independent of the quantum fit. Include at minimum H2, CH4, CO2
(symmetry-zero), CO, H2O, NH3, HCN, HNC, CH2O, CH3OH, CH3NH2, H2O2 and NH2OH
where a reliable value and state assignment exist. Compare equilibrium
`mu_e` with equilibrium calculations and vibrationally averaged `mu_0` only
with an explicitly vibrationally averaged prediction; do not mix them.

Experimental rotational geometries, polarizabilities, and Stark data may be
secondary hold-outs. Reaction energies and bond dissociation energies are not
fit targets for this charge-only model and must not be represented as such.

## 6. Objective function

Fit normalized, robust losses with each molecule family carrying equal total
weight so large molecules and dense ESP grids cannot dominate:

| Term | Weight | Definition |
|---|---:|---|
| molecular dipole vector | 25% | component error, symmetry-projected; scale 0.10 D |
| finite-field response | 20% | dipole derivative/polarizability and selected atomic charge response; scale from reference uncertainty |
| MBIS atomic charges | 15% | per-atom robust loss after exact total-charge projection; scale 0.05 e |
| external electrostatic potential | 15% | relative/absolute robust error on stratified shell points |
| separated-fragment charge | 15% | net fragment charge along scans, with strongest emphasis on the asymptotic region |
| geometry continuity | 5% | charge and dipole first differences along ordered stretches/contacts |
| parameter regularization | 5% | weak log-scale prior against extreme parameter ratios; no published values used as fit targets |

The fit must report every component separately. A lower aggregate score cannot
compensate for failure of symmetry, charge conservation, asymptotic
localization, or matrix stability.

Finite-field calculations use symmetric `+/-F` fields in all three Cartesian
directions at two field magnitudes on a stratified 550-configuration subset.
Linearity between magnitudes is checked; nonlinear or SCF-state-switching
points are flagged rather than forced into a quadratic response target.

## 7. Hard constraints and numerical rejection rules

Parameter transformations enforce:

```
eta_i   >= 0.5 eV/e^2
sigma_i in [0.15, 2.0] angstrom
s0_ab   in [1e-4, 10] e^2/eV
beta    in [0.5, 12]
```

Bounds are deliberately broad and must be reviewed before fitting, not moved
to rescue a result. On every training and validation geometry require:

- finite charges and energy;
- `abs(sum(q)) < 1e-12 e` in float64;
- scaled response Hessian symmetric to `1e-12` relative tolerance;
- smallest eigenvalue `>= 1e-8` and preferably `>= 1e-5`;
- condition number `<= 1e8` as a hard rejection, with `<= 1e6` the design
  target;
- linear residual `<= 1e-10` relative;
- no discontinuous charge jump under `1e-4 angstrom` coordinate perturbations;
- exact symmetry-equivalent charges within `1e-10 e`;
- monotonically decreasing transfer compliance with distance;
- the envelope of fragment charge tending to zero in the asymptotic scan, with
  the final points below a preregistered tolerance. Individual finite-distance
  points need not be monotonic when physical polarization changes sign.

The final report must include eigenvalue and condition-number distributions,
not only worst cases. Cholesky failure of the scaled charge block, unexplained
near-null response, or strong dependence on numerical regularization rejects
the parameter set.

## 8. Overdetermination and identifiability

The model has 22 independent fitted parameters and 2,350 base configurations.
Even counting only one dipole magnitude and one independent charge contrast
per configuration gives more than 4,700 scalar observations, over 200 per
parameter. In practice the dipole components, atomic populations, ESP points,
and field derivatives provide tens of thousands of stratified observations.

This apparent ratio is not sufficient by itself because observations within a
molecule are correlated. Therefore report:

- family-blocked Jacobian rank and singular values;
- profile likelihood/bootstrap intervals by molecule family;
- parameter correlation matrix;
- leave-one-element-pair-out sensitivity;
- fits from at least 20 dispersed initial points;
- an ablation comparison against fewer parameters;
- training/validation/hold-out errors per physical block.

If a parameter is weakly identified, freeze or remove it in a newly declared
model generation; do not retain it because it improves training error.

## 9. Estimated calculation count and cost

Planned workload before retries:

- 2,350 base DFT densities and properties;
- 550 configurations with `+/-` fields in three directions at two magnitudes:
  6,600 additional SCF single points;
- approximately 80 equilibrium optimizations/frequency checks;
- 120 coupled-cluster audit single points, with the larger-basis subset limited
  to molecules where it is tractable;
- 15% DDEC6 audit: approximately 350 density analyses, not new SCFs.

Total: roughly **9,100 DFT SCF calculations**, 80 optimizations/frequency
jobs, and 120 higher-level audit calculations. For predominantly 2-12 atom
systems, budget approximately **25,000-70,000 CPU-core hours**, 0.5-1.5 TB of
scratch during generation, and 100-300 GB retained after deleting redundant
integral files. Wall time is approximately 2-5 days on 256 reliable CPU cores,
or 1-3 weeks on a 64-core workstation, with radical/stretched-state retries
the largest uncertainty. A 10% pilot should validate these estimates before
launching the full dataset.

GPU quantum chemistry may reduce cost if the selected program reproduces the
same functional, grid, basis, density and field response to audited tolerance;
it may not silently change the reference convention.

## 10. Required pre-generation approvals and artifacts

Before generating data, freeze:

1. exact quantum program/version and input templates;
2. MBIS and DDEC6 program/version and integration settings;
3. molecule names, geometries, multiplicities and split manifest;
4. random seeds and distortion-generation rules;
5. field magnitudes and ESP grid construction;
6. covalent-radius source and all fixed constants;
7. the 22-parameter model schema and bounds;
8. objective scales/weights and hard rejection thresholds;
9. EEQBC/QEq/QTPIE/SQE comparator versions;
10. immutable hashes for the final hold-out manifest.

Only after those artifacts are reviewed should a 10% pilot be generated. The
pilot tests convergence, reference reproducibility, response linearity,
storage, cost, and Jacobian identifiability; it does not authorize fitting the
full model or connecting electrostatics to MD.
