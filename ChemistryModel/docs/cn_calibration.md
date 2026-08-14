# C-N calibration

## Reference interpretation

The original single C-N row was `1.47 A, 305 kJ/mol, 1.90 inverse A`.
Methylamine is used as the representative isolated single-bond environment.
Its experimental ground-state C-N distance is `1.471 +/- 0.003 A`, so the
existing model length is retained.

The supplied methylamine C-N dissociation enthalpy is 356 kJ/mol. This is a
finite-temperature molecular bond dissociation quantity rather than a pure
pair-potential well depth, so it is treated as a candidate requiring matched
reactive validation, not inserted as an unquestioned identity.

Methylamine has an observed C-N stretching band near 1044.8134 cm-1. That band
is strongly perturbed by torsion and wagging and is not a pure diatomic
oscillator. At the original parameters, the raw pair diagnostic is about
979.70 cm-1. Changing only the depth to 356 kJ/mol predicts approximately
1058.5 cm-1, which is encouraging supporting evidence but not a direct fit.

Sources:

- NIST CCCBDB experimental methylamine geometry,
  https://cccbdb.nist.gov/exp2x.asp?casno=74895&charge=0
- Klee et al., high-resolution methylamine C-N stretching band,
  https://doi.org/10.1016/j.jms.2011.09.003
- NIST Chemistry WebBook methylamine vibrational assignments,
  https://webbook.nist.gov/cgi/cbook.cgi?ID=C74895&Mask=1E9F

## Candidate

The isolated candidate changes only the single-bond C-N depth from 305 to
356 kJ/mol. It retains `re = 1.47 A` and width `1.90 inverse A`, and does not
alter the C=N or C#N tables. A controlled fixed-fragment methylamine coordinate
and methylamine NVE probe guard the local well, capture path and numerical
stability. Acceptance additionally requires the normal deterministic suites
and a matched reactive CUDA batch.

## Results

The candidate passed the calibration suite 10/10, reactive core 8/8 and high
fidelity 7/7. The controlled methylamine coordinate had its sampled minimum at
1.4675 A, a 3.7083 eV dissociation coordinate, positive short-range repulsion
and no post-minimum attraction hump. Its 400-step NVE probe drifted only
`+2.19e-5 eV` and used no move caps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 17000--17015, 2 ps, a
21 A fixed box and identical thermal/recording settings. The 305 kJ/mol control
is `integrator_smoothstep_final`; the candidate is `cn_depth_356_candidate`.

| Quantity | 305 kJ/mol control | 356 kJ/mol candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 57.750 | 59.188 | +1.438 |
| mean largest structure (atoms) | 11.062 | 11.875 | +0.812 |
| mean largest heavy-atom count | 6.000 | 6.562 | +0.562 |
| mean largest carbon count | 4.625 | 4.750 | +0.125 |
| mean best carbon chain | 3.250 | 3.438 | +0.188 |
| mean species count | 49.875 | 51.312 | +1.438 |
| mean final C+N species | 5.562 | 6.750 | +1.188 |
| mean final temperature (K) | 528.83 | 527.28 | -1.55 |
| mean final potential (eV) | -1084.44 | -1090.98 | -6.54 |

All runs in both sets formed C+N species. Six of sixteen final headlines
matched; divergent individual products are expected after changing an active
bond depth. The modest increase in C+N products and more negative final
potential are physically consistent with stronger C-N bonding. There was no
temperature, stability, carbon-growth or structure-size regression.

Recommendation: accept 356 kJ/mol as the effective single C-N depth. It agrees
better with the supplied methylamine dissociation evidence and brings the raw
local curvature close to the observed C-N band while remaining numerically
clean in both controlled and reactive validation.
