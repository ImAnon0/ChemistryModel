# O-O calibration

## Reference interpretation

The committed single O-O row is `1.480 A, 146 kJ/mol, 2.05 inverse A`.
Hydrogen peroxide is the representative single-bond environment. NIST CCCBDB
reports an experimental O-O distance of 1.475 A, O-H distance 0.950 A,
H-O-O angle 94.8 degrees and H-O-O-H dihedral 119.8 degrees. The first
candidate changes only the O-O equilibrium length to 1.475 A.

The observed hydrogen-peroxide O-O stretching fundamental is about 877 cm-1.
The experimental O-O dissociation enthalpy is about 210--213 kJ/mol, much
larger than the current 146 kJ/mol effective pair depth. Both are
environment-dependent molecular observables and remain separate experiments;
depth and width are unchanged during the length test.

Sources:

- NIST CCCBDB hydrogen-peroxide geometry and vibrational data,
  https://cccbdb.nist.gov/exp2x.asp?casno=7722841
- Ruscic et al., ATcT hydroperoxyl thermochemistry,
  https://doi.org/10.1021/jp056311j
- Peroxide bond-energy review,
  https://par.nsf.gov/servlets/purl/10217581

## Length candidate

The isolated candidate changes only single-bond `re` from 1.480 to 1.475 A.
Depth, width, O=O and every other pair remain unchanged. A controlled
fixed-fragment hydrogen-peroxide coordinate and NVE probe protect local shape,
capture and integration stability before reactive validation.

## Length-candidate results

The candidate passed the calibration suite 15/15, reactive core 8/8 and high
fidelity 7/7. The controlled peroxide coordinate sampled its minimum at
1.473 A, retained a smooth 1.4391 eV dissociation coordinate and had no
post-minimum attraction hump. Its 400-step NVE drift was `+5.38e-5 eV` with
zero move caps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 17000--17015, 2 ps, a
21 A fixed box and identical settings. The control is `no_length_candidate`;
the candidate is `oo_length_candidate`.

| Quantity | 1.480 A control | 1.475 A candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 59.250 | 59.625 | +0.375 |
| mean largest structure (atoms) | 11.375 | 11.438 | +0.063 |
| mean largest heavy-atom count | 6.688 | 6.500 | -0.188 |
| mean largest carbon count | 4.500 | 4.750 | +0.250 |
| mean best carbon chain | 3.375 | 3.500 | +0.125 |
| mean species count | 50.625 | 51.188 | +0.563 |
| mean final multi-oxygen species | 3.250 | 2.938 | -0.312 |
| runs seeing hydrogen peroxide | 2 / 16 | 3 / 16 | +1 |
| mean final temperature (K) | 545.60 | 538.93 | -6.67 |
| mean final potential (eV) | -1086.01 | -1090.22 | -4.21 |

All runs in both sets formed multi-oxygen species, so oxygen chemistry was
directly exercised. Two final headlines matched, consistent with chaotic
divergence. Aggregate structure, carbon growth and numerical health were
preserved, and hydrogen peroxide was not suppressed.

Recommendation: accept 1.475 A. It improves the directly measured peroxide
geometry without changing curvature or depth and introduces no controlled or
reactive regression.

## Separate depth experiment

After committing length, an isolated candidate changes the single O-O depth
from 146 to 210.4 kJ/mol, using the experimental hydrogen-peroxide dissociation
enthalpy while holding `re = 1.475 A` and width `2.05 inverse A`. This is an
empirical test rather than an assertion that the molecular BDE is exactly the
model pair De. O=O remains unchanged. Because the change is large, controlled
stability and matched reactive peroxide formation are required before it can
be accepted.

The controlled peroxide coordinate remained smooth and NVE drift stayed small
at `+5.91e-5 eV`, but the independent high-fidelity suite failed 6/7. The
stronger O-O depth opened a bound three-centre well: adding a third contact
lowered the energy 0.25 eV below the intended two-contact geometry. This is a
whole-model reaction-surface regression, not a statistical batch fluctuation.

Recommendation: reject 210.4 kJ/mol without a reactive batch and retain the
146 kJ/mol effective depth. The molecular HOOH dissociation enthalpy cannot be
inserted directly into the baseline pair term without overbinding competing
three-centre environments. Any future attempt would require an environment
correction rather than a simple table replacement.
