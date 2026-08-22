# C0 continuous SQE: 10% QM-observable parameter pilot

## Decision

**RECONSIDER FORMULATION. Do not integrate C0 and do not proceed directly to
C1 environment descriptors.**

C0 remains an excellent mathematically localised split-charge kernel. The fit
improves unseen-family dipoles and polarizability, stays smooth, conserves
charge, and remains exceptionally well conditioned. It nevertheless fails two
predeclared scientific gates:

1. H, N and O intrinsic hardnesses run to the lower bound in every converged
   multistart solution; the fitted 17-parameter convention is therefore not a
   physically interior estimate.
2. Locked-holdout external-potential RMSE worsens by 205%, even though ESP is
   an explicit fit target. Atom-centred scalar charges cannot reproduce the
   missing intra-atomic dipole/multipole structure. The same structural limit
   forces both transverse H2 polarizabilities to exactly zero.

Adding only environment-dependent scalar electronegativity/hardness (C1) can
move charge magnitudes but cannot create those missing directional response
degrees of freedom. The evidence supports retaining SQE transfers in a future
**hybrid split-charge + induced atomic dipole/multipole formulation**, then
testing whether environment conditioning is still needed.

This is a research result only. No production term, ChemistryEngine extension,
bonded parameter, H-state parameter, unified-radial coupling, or MD path was
changed.

## Frozen pilot data

- 235 configurations, exactly 10% of the planned 2,350-configuration base set.
- 93 training, 59 model-selection validation, 83 locked holdout.
- 20 molecule/reaction families; every geometry from a family stays in one
  split.
- Locked-holdout hash:
  `a7f327e60b8b425ff75837d1978b94312b37daba96a48ad95e4da37be5bfb8b0`.
- QM convention: wB97X-D/jun-cc-pVDZ, RKS/UKS as appropriate, fixed Cartesian
  frame, tight SCF thresholds.
- 235/235 successful energy/gradient, dipole, MBIS, polarizability and external
  potential calculations.
- Each ESP target is the total QM molecular potential at 14 fixed points on an
  enclosing shell. It is not reconstructed from fitted or MBIS charges.

The pilot uses the already audited reduced-cost observable convention. It is
not a substitute for the planned omegaB97M-V/def2-TZVPPD production-quality
dataset. Deterministic thermal-like distortions are suitable for screening
response and identifiability, but not claimed as normal-mode samples.

### Family-blocked split

| Split | Families |
|---|---|
| Training | H2, N2, methane, ammonia, CO, methanol, OH stretch, water angle, H/formaldehyde transfer |
| Validation | H3 transfer, water transfer, ethane, HCN, H2O2 |
| Locked holdout | formaldehyde, H + H2 approach, directional water + H, methylamine, CO2, hydroxylamine |

## Objective

No QM energy was fitted. Each molecule family has equal total weight.

| Observable/gate | Role |
|---|---:|
| molecular dipole vector | 25% fit |
| full polarizability tensor | 25% fit |
| QM external potential shell | 15% fit |
| MBIS charges | 10% weak proxy |
| parameter regularisation | 5% fit |
| separated-fragment localisation | 15% independent gate |
| reactive smoothness | 5% independent gate |

MBIS charges are explicitly not treated as unique physical atomic charges.
Residual normalisations are 1 D, 1 A3, 0.005 au ESP and 0.25 e MBIS;
family equalisation is applied after those unit scales.

## Fitted parameters

H electronegativity is the fixed zero gauge. Covalent radii, Coulomb constant
and the 3.5--4.5 A compact support remain fixed.

| Element | chi (eV) | intrinsic hardness (eV) | Gaussian sigma (A) | capacity (e2/eV) |
|---|---:|---:|---:|---:|
| H | 0.000000 | **0.500000 (lower bound)** | 0.508424 | 0.270523 |
| C | 1.638615 | 3.051432 | 0.601600 | 0.243103 |
| N | 1.716862 | **0.500000 (lower bound)** | 0.828757 | 0.213420 |
| O | 1.903933 | **0.500000 (lower bound)** | 0.623337 | 0.229645 |

Global capacity radius scale `rho = 2.134518`; steepness `k_cap = 3.183317`.
The exact machine-readable result and bounds are in `fitted_parameters.json`.
These values are a rejected pilot result, not production parameters.

## Initial versus fitted metrics

Lower is better.

| Split / metric | Initial | Fitted | Change |
|---|---:|---:|---:|
| Train dipole vector RMSE (D) | 0.7563 | 0.4618 | -38.9% |
| Validation dipole vector RMSE (D) | 1.1677 | 0.9163 | -21.5% |
| Holdout dipole vector RMSE (D) | 0.6770 | 0.5201 | -23.2% |
| Train polarizability tensor RMSE (A3) | 0.9231 | 0.5542 | -40.0% |
| Validation polarizability tensor RMSE (A3) | 0.9423 | 0.6280 | -33.4% |
| Holdout polarizability tensor RMSE (A3) | 1.0397 | 0.6968 | -33.0% |
| Train ESP RMSE (au) | 0.003237 | 0.003429 | **+5.9%** |
| Validation ESP RMSE (au) | 0.002000 | 0.002489 | **+24.5%** |
| Holdout ESP RMSE (au) | 0.001258 | 0.003831 | **+204.6%** |
| Train MBIS proxy RMSE (e) | 0.3584 | 0.2294 | -36.0% |
| Validation MBIS proxy RMSE (e) | 0.2744 | 0.2129 | -22.4% |
| Holdout MBIS proxy RMSE (e) | 0.4380 | 0.4056 | -7.4% |

Thus, improvement is not confined to training data, but the observable classes
are mutually incompatible within the scalar-charge C0 representation.

## Numerical and localisation gates

| Gate | Result |
|---|---:|
| maximum total-charge residual | 2.22e-16 e |
| minimum response eigenvalue over 235 cases | 1.000000 within roundoff |
| maximum response condition number | 15.06 |
| maximum linear-solve residual | 3.75e-16 |
| numerical Jacobian rank | 17 / 17 |
| Jacobian condition number | 260.1 |
| maximum fitted OH-grid charge step | 0.00196 e per 0.01 A |
| maximum fitted OH-grid energy step | 0.00187 eV per 0.01 A |
| charge at and beyond 4.5 A support | exactly zero |

Twelve dispersed starts converge to the same training score and validation
scores within about 2e-8. The result is numerically identifiable in this
bounded problem, but three bound-active hardnesses show that numerical rank is
not sufficient evidence of a physically credible parameter convention.

At 8, 20 and 100 A separation, water and ammonia each retain zero fragment net
charge within 8.33e-17 e. OH + H also has exactly zero fragment transfer beyond
support. The nonzero reported total electronic energy is each isolated
fragment's internal C0 energy, not long-range charge transfer.

## Structural failure cases

- **H2 response:** QM polarizability eigenvalues are approximately
  `[0.17571, 0.17571, 0.95972] A3`; C0 gives
  `[0, 0, 0.82024] A3`. Pair charge flow has no transverse degree of freedom.
- **Directional water + H holdout:** dipole-magnitude MAE is 1.235 D and ESP
  RMSE is 0.00534 au. Lone-pair direction cannot be represented by scalar
  atom charges alone.
- **Formaldehyde:** dipole-magnitude error is 1.376 D despite reasonable
  response conditioning.
- **CO2 distortions:** several asymmetric distortions have 1.1--1.7 D dipole
  errors; equilibrium symmetry itself remains correct.
- **Hydroxylamine:** holdout dipole-magnitude MAE is 0.992 D, exposing weak
  transferability across mixed N/O environments.

## Recommendation

Do not continue with C0 as the complete electronic-state model. Do not connect
the fitted values to production and do not start C1 by merely adding local
scalar descriptors. Preserve C0 as the local, exactly conserving transfer
subsystem and formulate the smallest hybrid model that adds induced atomic
dipoles (and, only if independently needed, higher multipoles). Such a model
must keep C0's compact transfer support and positive response structure while
adding the directional/intra-atomic response that the pilot proves is absent.

Raw and reduced results are under
`research_data/electronic_observables/c0_pilot/`.
