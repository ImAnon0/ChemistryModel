# Electronic-state correction report

## Decision

**Reject all four correction candidates. Keep unified radial as the research
baseline and do not integrate an electronic correction into production.**

The experiment was scientifically useful: low-order local electronic moments
are smooth, conservative, and easy to make safe, but the present observables do
not define a transferable correction energy. All candidates fail the
independent water-transfer gate, and none improves the complete Grambow result.

## Parameter and representation audit

The elemental source proxy was fitted to MBIS atomic charges and total dipole
components, with H fixed to zero:

| Element | Source proxy (e) | Status |
|---|---:|---|
| H | 0.000000 | fixed gauge |
| C | -0.019166 | fitted |
| O | 0.355114 | fitted |
| N | 0.000000 | unidentifiable; not fitted |

The two-column source design has rank 2 and condition number 2.30. Charge RMSE
is 0.0592 e on characterisation, 0.0821 e on validation, and 0.1599 e on the
final holdout. Dipole-component RMSE is 0.1193, 0.3869, and 0.3367 D,
respectively. The deterioration outside characterisation is an early warning
that a fixed elemental source does not capture environment-conditioned
redistribution.

Individual energy hypotheses each have one fitted coefficient and a rank-one,
well-defined design. The three-parameter combined design has rank 1 and
condition number \(5.95\times10^{16}\). On the available training paths, the
scalar, vector, and tensor features collapse to the same effective coordinate;
the combined coefficients are therefore not independently identifiable.

## Independent QM residuals

Dense energy RMSE in eV (98 total geometries):

| Model | Formaldehyde | Methane | H3 | Water | All dense |
|---|---:|---:|---:|---:|---:|
| unified radial | 0.38103 | 0.21841 | 0.22311 | **0.21363** | **0.26544** |
| local scalar | 0.38103 | 0.21841 | 0.22311 | 0.33012 | 0.29867 |
| polarisation-like vector | 0.38103 | 0.21841 | 0.22311 | 0.33012 | 0.29867 |
| multipole/density tensor | 0.38103 | 0.21841 | 0.22311 | 0.26164 | 0.27787 |
| combined | 0.38103 | 0.21841 | 0.22311 | 0.52607 | 0.37262 |

The tiny formaldehyde changes are below a meaningful improvement, methane is
unchanged to the displayed precision, and H3 is exactly unchanged because a
homonuclear H-H contact has zero source difference. Water is independent of
the energy fit and rejects every candidate. The tensor model is the least bad,
but still increases water RMSE by 22.5%.

## Grambow 200/200 comparison

Parameters were frozen before this diagnostic. No Grambow target was read by
the fitting script.

| Model | Barrier MAE | Barrier RMSE | Barrier sign | Barrier max | Reaction MAE | Reaction RMSE | Reaction max |
|---|---:|---:|---:|---:|---:|---:|---:|
| production | 4.5195 | 6.4831 | 87.0% | 35.5021 | 4.4058 | 7.0688 | 49.3746 |
| unified radial | **1.4746** | **1.8577** | **99.5%** | 7.6570 | 2.2330 | 2.8725 | 8.1591 |
| local scalar | 1.4836 | 1.8657 | 99.5% | 7.6570 | 2.2271 | 2.8703 | 8.1591 |
| polarisation-like vector | 1.4834 | 1.8647 | 99.5% | 7.6570 | 2.2275 | 2.8704 | 8.1591 |
| multipole/density tensor | 1.4764 | 1.8584 | 99.5% | 7.6570 | 2.2298 | 2.8711 | 8.1591 |
| combined | 1.5003 | 1.8963 | 99.0% | 7.6570 | 2.2272 | 2.8731 | 8.1591 |

Small reaction-MAE changes in the single-feature models do not offset their
worse barriers and failed water gate. The ill-conditioned combined model also
reduces barrier sign agreement and changes individual barriers by as much as
2.04 eV, demonstrating that cancellation can look harmless in an aggregate
reaction MAE while damaging individual chemistry.

## Physics and stability gates

All candidates pass the implementation-safety checks:

- finite-difference force errors are at most \(5.2\times10^{-10}\) eV/A;
- permutation energy errors are at most \(1.8\times10^{-15}\) eV;
- correction energy and force vanish continuously at the outer cutoff;
- accepted settled H2, CH4, H2O, ethane, formaldehyde, N2, ammonia,
  methanol, hydroxylamine, and peroxide geometries are exactly unchanged;
- H2, H3 transfer, and H + H2 are exactly identical to unified radial;
- both NVE cases remain finite, with zero move caps and no NaNs.

Water-transfer NVE maximum drift is 0.0154 eV for unified radial, 0.0164 eV
for the scalar/vector models, 0.0120 eV for the tensor model, and 0.0352 eV for
the combined model. Passing numerical safety does not rescue a candidate that
fails independent chemistry.

The exact settled-molecule preservation is a consequence of the ambiguity
window, not evidence that nitrogen or equilibrium electronic response has been
validated. In particular, the nitrogen source is unidentifiable in the current
dataset.

## Why the hypotheses failed

1. **An observable representation is not an energy functional.** MBIS charges,
   dipoles, and multipoles describe electronic density, but do not by
   themselves specify the self-energy, response kernel, damping, or coupling
   needed to turn that density into a transferable potential energy.
2. **The contact ambiguity is only a radial proxy.** \(t(1-t)\) locates a
   changing contact, but it is not the solved unified bond-state population or
   an electronic coherence/charge-flow variable. Water and formaldehyde do not
   map onto it in the same way.
3. **The data do not separate the proposed moments.** The combined design is
   rank deficient, so apparent multi-feature flexibility is parameter
   cancellation rather than evidence for three electronic degrees of freedom.
4. **Environment dependence is missing.** A fixed element source cannot
   distinguish donor, acceptor, lone-pair, radical, and coordination states.
5. **Coverage is incomplete.** There is no identifiable nitrogen response and
   too little orientational/finite-field variation to validate a general
   H/C/N/O electronic correction.

## What additional physics and data are needed

The next useful step is not another local invariant sweep. A credible
electronic layer would need:

- state-resolved charge redistribution referenced to the actual variational
  unified bond state, rather than the radial taper alone;
- finite-field energies and forces, not only dipoles/polarizabilities, across
  diverse orientations, stretches, radicals, and reactive contacts;
- explicit N-containing polar and reactive configurations;
- a positive-definite, self-consistent response Hamiltonian with exact charge
  conservation, fragment localisation, and smooth short-range damping;
- enough independent geometries to identify scalar charge-flow, vector
  polarisation, and tensor response separately;
- final holdouts containing molecule families and reaction paths absent from
  parameter generation.

A topology-free SQE-like charge-flow variable coupled to screened induced
dipoles is one physically interpretable direction, but it requires a dedicated
H/C/N/O quantum-response parameterisation and validation project. It should not
be inferred from the coefficients rejected here.

## Artifacts

- `electronic_state_parameters.json`: fit provenance, ranks, coefficients, and
  independent source/energy metrics.
- `electronic_state_validation.json` and `electronic_state_qm.csv`: microscope,
  molecule, force, cutoff, H-safety, and NVE evidence.
- `electronic_state_grambow.json` and `.csv`: frozen 200-reaction comparison and
  worst-case changes.
- `electronic_state_hypothesis_screen.csv`: compact candidate screen.

Production physics remains untouched. The correct outcome of this study is a
documented negative result and preservation of unified radial as the research
baseline.
