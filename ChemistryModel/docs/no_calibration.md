# N-O calibration

## Reference interpretation

The committed single N-O row is `1.400 A, 201 kJ/mol, 2.00 inverse A`.
Hydroxylamine is the representative single-bond environment. NIST CCCBDB
reports `r(N-O) = 1.453 A`, with N-H 1.016 A and O-H 0.962 A. The first
candidate changes only the N-O equilibrium length to 1.453 A.

Hydroxylamine's N-O stretching mode is commonly assigned near 955 cm-1. This
is a polyatomic mode and is supporting evidence rather than a direct diatomic
width target. Reported N-O bond energies vary strongly with molecular
environment; depth and width remain unchanged during the length experiment.

Sources:

- NIST CCCBDB experimental hydroxylamine geometry,
  https://cccbdb.nist.gov/listgeomexpx.asp?casno=7803498&charge=0
- Hydroxylamine vibrational summary and N-O assignment,
  https://info.ornl.gov/sites/publications/Files/Pub162446.pdf
- N-O bond-energy environment review,
  https://doi.org/10.1021/acs.jpca.1c02741

## Length candidate

The isolated candidate changes only single-bond `re` from 1.400 to 1.453 A.
Depth, width, N=O and every other pair remain unchanged. A controlled
fixed-fragment hydroxylamine coordinate and NVE probe protect local shape,
capture and integration stability before reactive validation.

## Length-candidate results

The candidate passed the calibration suite 14/14, reactive core 8/8 and high
fidelity 7/7. The controlled hydroxylamine coordinate sampled its minimum at
1.451 A, retained a smooth 2.0975 eV dissociation coordinate and had no
post-minimum attraction hump. Its 400-step NVE drift was `+2.58e-5 eV` with
zero move caps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 17000--17015, 2 ps, a
21 A fixed box and identical thermal/recording settings. The control is
`nn_length_candidate`; the candidate is `no_length_candidate`.

| Quantity | 1.400 A control | 1.453 A candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 60.438 | 59.250 | -1.188 |
| mean largest structure (atoms) | 12.188 | 11.375 | -0.813 |
| mean largest heavy-atom count | 6.625 | 6.688 | +0.063 |
| mean largest carbon count | 4.875 | 4.500 | -0.375 |
| mean best carbon chain | 3.500 | 3.375 | -0.125 |
| mean species count | 51.000 | 50.625 | -0.375 |
| mean final N+O species | 3.250 | 3.312 | +0.062 |
| mean N+O species seen | 5.688 | 6.188 | +0.500 |
| mean final temperature (K) | 556.87 | 545.60 | -11.27 |
| mean final potential (eV) | -1093.68 | -1086.01 | +7.67 |

Every run in both sets formed N+O-bearing species, so the normal mixture
directly exercised N-O chemistry. Three final headlines matched, as expected
for a comparatively large active-length correction. The candidate preserved
heavy-atom structure size and numerical health, slightly increased the range
of N+O species seen and did not cause heating or fragmentation. Its less
negative aggregate potential is consistent with moving away from the old,
artificially short N-O equilibrium and is not treated alone as a regression.

Recommendation: accept 1.453 A. It corrects a clear hydroxylamine geometry
mismatch, is locally stable and preserves directly exercised N-O chemistry.
Depth and curvature remain separate calibration questions.
