# QTPIE formulation compatibility investigation

## Scope and conclusion

This investigation is standalone. No charge energy is connected to
`ReactiveSimulation`; no force, bonded parameter, integrator or MD code was
changed; and no H/C/N/O charge parameter was fitted.

The earlier 18.34 D water result is caused by a **Coulomb-integral exponent
error**, not QTPIE's effective-electronegativity equation. The implemented
Gaussian kernel used
`sqrt(2 alpha_i alpha_j/(alpha_i+alpha_j))`; Chen's fitted-Gaussian/Open Babel
kernel uses `sqrt(alpha_i alpha_j/(alpha_i+alpha_j))`. The extra square-root of
two made several hardness matrices indefinite.

The original 2007 Slater QTPIE(-H) convention has chemically correct H2O and
NH3 polarity and sensible dipoles, but it is not a safe production reference:
its H2 quadratic form is negative on the charge-conserving subspace at the
equilibrium bond length. This is consistent with the historical warning that
the omitted charge-dependent hydrogen correction was introduced to cure
nonphysical molecular hydrogen response.

**Recommendation: move to ACKS2 as the next production-candidate
investigation.** Retain the reproduced original QTPIE implementation as the
authoritative historical/dissociation reference. Do not couple either QTPIE
formulation to MD yet. A modern, complete QTPIE/ReaxFF parameterization may
also be compared, but the corrected Gaussian diagnostic is already nearly
singular for CH4 and is not enough evidence to adopt it.

## Reproduced historical convention

The implementation in `qtpie_historical.py` follows Chen and Martinez,
Chemical Physics Letters 438 (2007), equations 1--10:

- normalized single `ns` Slater orbitals,
  `phi_i = N_i r^(n_i-1) exp(-zeta_i r)`;
- exact Rosen analytic overlap and two-centre Coulomb integrals;
- QEq electronegativities and full diagonal hardnesses;
- original QTPIE attenuation `f_ij = k_ij S_ij`, with `k_ij = 1`;
- exact atom-space coefficient
  `v_i = sum_j[(chi_i-chi_j) S_ij] / N`;
- neutral KKT constraint only;
- the paper's QEq(-H) choice: no charge-dependent hydrogen radius/hardness.

| Element | n | zeta (Bohr^-1) | chi (eV) | Jii (eV) |
|---|---:|---:|---:|---:|
| H | 1 | 1.0698 | 4.528 | 13.890 |
| C | 2 | 0.8563 | 5.343 | 10.126 |
| N | 2 | 0.9089 | 6.899 | 11.760 |
| O | 2 | 0.9745 | 8.741 | 13.364 |

Sources are the original QTPIE paper, its exact atom-space reformulation, the
Rappe/Goddard QEq parameter convention, Rosen's analytic STO integrals, and
the preserved original QTPIE integral implementation in OpenMD. No ReaxFF
shield or Gaussian exponent enters this historical path.

## Hardness audit

Values below are the smallest eigenvalue of hardness projected onto
`sum(q)=0`, followed by the full KKT condition number. A positive projected
minimum is required for a constrained energy minimum.

| System | A: erroneous Gaussian min / cond | B: original Slater min / cond | C: matching Slater QEq min / cond | corrected Gaussian diagnostic min / cond |
|---|---:|---:|---:|---:|
| H2 | -2.732 / 467.5 | **-0.388 / 398.7** | -0.388 / 398.7 | 0.328 / 378.8 |
| CH4 | -3.083 / 561.1 | 1.046 / 441.3 | 1.046 / 441.3 | **0.013 / 3959.3** |
| H2O | -0.356 / 535.6 | 3.015 / 373.6 | 3.015 / 373.6 | 2.725 / 380.4 |
| NH3 | -1.867 / 543.5 | 2.137 / 417.3 | 2.137 / 417.3 | 1.325 / 434.2 |
| CO | 1.049 / 229.4 | 3.818 / 190.5 | 3.818 / 190.5 | 3.101 / 203.0 |
| CO2 | -0.793 / 315.6 | 2.658 / 238.4 | 2.658 / 238.4 | 1.764 / 257.3 |
| CH2O | -2.191 / 427.5 | 1.508 / 331.7 | 1.508 / 331.7 | 0.566 / 351.4 |

QTPIE and matching QEq share a hardness matrix, so B and C have identical
spectra and conditions; they differ in the linear electronegativity term.

## Water and ammonia comparison

All charge and linear residuals were below 2.1e-14.

### H2O

| Formulation | charges (O, H, H), e | dipole (D) | effective chi (eV) | projected minimum (eV) |
|---|---|---:|---|---:|
| A erroneous Gaussian QTPIE | +6.519, -3.259, -3.259 | 18.344 | +2.140, -1.338, -1.338 | -0.356 |
| B original Slater QTPIE | -0.620, +0.310, +0.310 | 1.745 | +1.870, -0.935, -0.935 | 3.015 |
| C matching Slater QEq | -0.931, +0.466, +0.466 | 2.621 | 8.741, 4.528, 4.528 | 3.015 |
| corrected Gaussian diagnostic | -0.851, +0.425, +0.425 | 2.395 | +2.140, -1.338, -1.338 | 2.725 |

### NH3

| Formulation | charges (N, H, H, H), e | dipole (D) | effective chi (eV) | projected minimum (eV) |
|---|---|---:|---|---:|
| A erroneous Gaussian QTPIE | +0.847, -0.282 each | 1.546 (reversed) | +1.406, about -0.702 each | -1.867 |
| B original Slater QTPIE | -0.531, +0.177 each | 0.969 | +1.134, about -0.378 each | 2.137 |
| C matching Slater QEq | -0.832, +0.277 each | 1.519 | 6.899, 4.528 each | 2.137 |
| corrected Gaussian diagnostic | -1.194, +0.398 each | 2.179 | +1.406, about -0.702 each | 1.325 |

The original QTPIE result is qualitatively correct for both molecules. Its
water dipole (1.745 D) is close to the experimental gas-phase scale; ammonia
(0.969 D) is underpolarized. These are validation comparisons, not fit targets.

## Dissociation and numerical behaviour

For neutral atomic H and O fragments, original Slater QTPIE gives H charge:

| separation (A) | QTPIE H charge (e) | matching QEq H charge (e) |
|---:|---:|---:|
| 1.5 | 0.194 | 0.402 |
| 3.0 | 0.0275 | 0.238 |
| 5.0 | 0.00176 | 0.196 |
| 20.0 | 2.95e-14 | 0.163 |

Thus the historical solver retains QTPIE's defining vanishing-transfer limit,
while matching QEq exhibits the dissociation catastrophe. The implementation
remains explicitly neutral-only and makes no charged-fragment claim.

## Interpretation

- Formula/implementation error: **confirmed** in formulation A (extra sqrt 2).
- Hardness convention mismatch: not needed to explain A; full `Jii` values are
  used consistently in B/C.
- Orbital/exponent mismatch: A mixed the right alpha table with the wrong
  Coulomb exponent mapping.
- Shield mismatch: no ReaxFF shield is used in B.
- Electronegativity mismatch: B/C use matching QEq values; not the cause.
- Intrinsic limitation: **also present**. Historical QTPIE(-H) is unstable for
  equilibrium H2 without the historical nonlinear hydrogen correction, and
  the unrefitted model underpolarizes NH3.

The original convention is therefore successfully reproduced and useful as a
reference, but it does not pass the complete positive-definiteness gate needed
for ChemistryModel production dynamics.
