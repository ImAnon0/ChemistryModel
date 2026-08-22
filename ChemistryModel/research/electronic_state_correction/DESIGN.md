# Electronic-state correction study

## Scope and frozen baseline

This directory is a research-only falsification study. It does not modify or
register production physics. The frozen reference is
`UnifiedBondCapacityEnergyPrototype`, with the established 200-reaction
Grambow barrier MAE of 1.475 eV and water-transfer RMSE of 0.214 eV.

The question is whether observables in
`research_data/electronic_observables/` contain a transferable electronic
degree of freedom that can correct the unified radial energy without replacing
it. The experiment deliberately compares several minimal hypotheses before
considering a larger self-consistent electrostatic model.

## What is represented

The observable fit assigns each element a scalar source, \(\zeta_Z\), in a
hydrogen-fixed gauge. These are not permanent atomic charges. They are a
low-dimensional proxy fitted to MBIS atomic charges and molecular dipole
components from the electronic-observable dataset. Mulliken and Lowdin values
remain comparison representations rather than fit targets.

For pair \(ij\), let \(t_{ij}(r)\) be the existing smooth contact taper and

\[
    a_{ij} = t_{ij}(1-t_{ij}), \qquad
    \Delta\zeta_{ij} = \zeta_j-\zeta_i.
\]

The ambiguity factor is zero for a settled contact (\(t=1\)) and for a
separated pair (\(t=0\)). The electronic correction can therefore act only in
the contact-formation/breaking region. It cannot change a settled molecule or
create a long-range Coulomb tail.

Three rotationally invariant feature families are tested:

\[
 S_i = \sum_j a_{ij}|\Delta\zeta_{ij}|,
 \quad F_S=\sum_i S_i^2,
\]

\[
 \mathbf P_i = \sum_j a_{ij}\Delta\zeta_{ij}\hat{\mathbf r}_{ij},
 \quad F_P=\sum_i |\mathbf P_i|^2,
\]

and

\[
 \mathbf Q_i = \sum_j a_{ij}|\Delta\zeta_{ij}|
 \left(\hat{\mathbf r}_{ij}\hat{\mathbf r}_{ij}^{T}-\frac{\mathbf I}{3}\right),
 \quad F_Q=\sum_i ||\mathbf Q_i||_F^2.
\]

Each hypothesis adds one scalar energy to unified radial:

\[
 E = E_{\mathrm{unified}} + c_S F_S + c_P F_P + c_Q F_Q.
\]

Only the relevant coefficient is active in a single-feature candidate. The
combined candidate contains all three.

## Hypotheses

| Candidate | Intended physical diagnostic | Important limitation |
|---|---|---|
| no correction | Frozen unified radial control | No explicit electronic state |
| local scalar | Magnitude of environment-conditioned redistribution | No direction or response field |
| polarisation-like vector | Local oriented redistribution moment | Analytically eliminated local moment, not a self-consistent induced-dipole model |
| multipole/density tensor | Traceless local density anisotropy | No permanent multipoles or intermolecular multipole interaction |
| combined | Tests whether the three moments add independent information | Requires a full-rank design to be interpretable |

No candidate adds Coulomb energy. A proper polarisation model would require a
defined source, positive-definite response/self-energy, screened interactions,
and a separated-fragment limit. Those quantities are not justified by the
current observable dataset, so this study does not invent them.

## Fitting and leakage controls

The source fit uses non-final electronic-observable records only. Hydrogen is
the fixed gauge; carbon and oxygen are identifiable. Nitrogen is not present
in the source-fit configurations and is explicitly recorded as
unidentifiable, rather than assigned a fitted value.

Energy coefficients use 35 independent geometry observations: the
formaldehyde microscope plus alternating even-index methane points. H3, the
remaining methane points, all water-transfer points, and the electronic
dataset's final holdout remain outside the energy fit. Grambow energies and
barriers are never read during fitting. Grambow is evaluated only after
parameters are frozen.

The fit is intentionally small: one parameter for each individual hypothesis
and three for the combined hypothesis. Rank, condition number, parameter count,
fit roles, and held-out systems are serialized in
`electronic_state_parameters.json`.

## Validation and decision gates

The order is fixed:

1. Fit the observable proxy and audit rank/conditioning.
2. Fit energy coefficients without water or Grambow.
3. Test dense QM microscopes, including held-out water.
4. Test finite-difference forces, atom permutation, cutoff continuity, settled
   molecules, H2/H3/H+H2 identity, NaNs, move caps, and NVE drift.
5. Run Grambow as a frozen diagnostic even if a candidate has already failed,
   so the requested comparison remains complete. Do not retune from it.

Promotion requires water RMSE no worse than unified radial, all physics and
dynamics gates, and Grambow maintained or improved without a new catastrophic
failure. Failure of any gate means no production integration.

## Reproduction

From the repository root:

```powershell
python research/electronic_state_correction/fit_observable_parameters.py
python research/electronic_state_correction/validate.py
python research/electronic_state_correction/compare_grambow.py
python -m pytest -q research/electronic_state_correction/tests
```

The scripts write only research diagnostics under
`research_data/benchmark/diagnostics/`.
