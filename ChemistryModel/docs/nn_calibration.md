# N-N calibration

## Reference interpretation

The committed single N-N row is `1.450 A, 167 kJ/mol, 2.00 inverse A`.
Hydrazine is the representative single-bond environment. NIST CCCBDB reports
an experimental N-N distance of 1.446 A, so the first candidate changes only
the equilibrium length to 1.446 A.

Hydrazine's N-N stretch has been unambiguously assigned at 1077.24 cm-1 in
high-resolution gas-phase spectroscopy. It is still a polyatomic normal mode
with inversion and torsional coupling, so it is used as supporting curvature
evidence rather than fitted directly as a diatomic oscillator. The current
depth of 167 kJ/mol is retained during the length experiment.

Sources:

- NIST CCCBDB experimental hydrazine geometry,
  https://cccbdb.nist.gov/listgeomexpx.asp?casno=302012&charge=0
- Gulaczyk, Kreglewski and Valentin, hydrazine N-N stretching band,
  https://doi.org/10.1016/S0022-2852(03)00106-1
- Durig et al., gas-phase Raman spectrum of hydrazine,
  https://doi.org/10.1002/jrs.1250030204

## Length candidate

The isolated candidate changes only single-bond `re` from 1.450 to 1.446 A.
Depth, width, N=N, N#N and every other pair remain unchanged. A controlled
fixed-fragment hydrazine coordinate and NVE probe protect local shape, capture
and integration stability before matched reactive validation.

## Length-candidate results

The candidate passed the calibration suite 12/12, reactive core 8/8 and high
fidelity 7/7. The controlled hydrazine coordinate sampled its minimum at
1.4455 A, retained a smooth 1.7683 eV dissociation coordinate and had no
post-minimum attraction hump. Its 400-step NVE drift was `+5.03e-5 eV` with
zero move caps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 17000--17015, 2 ps, a
21 A fixed box and identical thermal/recording settings. The control is
`co_length_candidate`; the candidate is `nn_length_candidate`.

| Quantity | 1.450 A control | 1.446 A candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 60.375 | 60.438 | +0.063 |
| mean largest structure (atoms) | 12.188 | 12.188 | 0.000 |
| mean largest heavy-atom count | 6.500 | 6.625 | +0.125 |
| mean largest carbon count | 4.812 | 4.875 | +0.063 |
| mean best carbon chain | 3.125 | 3.500 | +0.375 |
| mean species count | 51.125 | 51.000 | -0.125 |
| mean final nitrogen species | 8.188 | 8.438 | +0.250 |
| mean final temperature (K) | 527.70 | 556.87 | +29.16 |
| mean final potential (eV) | -1090.93 | -1093.68 | -2.75 |

Five of sixteen final headlines matched. The temperature difference is not
accompanied by energy jumps, caps, strikes, fragmentation or changed aggregate
structure size, and paired temperature differences are broadly chaotic. No
run in either set retained a species containing two nitrogen atoms, so this
batch establishes normal-mixture safety but does not directly validate stable
hydrazine production. That limitation is covered locally by the controlled
hydrazine coordinate and NVE probe.

Recommendation: accept 1.446 A. It corrects the table to the directly measured
hydrazine geometry, is locally stable and produces no material normal-mixture
regression. Treat N-N curvature as a separate experiment.

## Separate width experiment

At the retained 167 kJ/mol effective depth, the raw N-N pair diagnostic is
733.24 cm-1, well below hydrazine's 1077.24 cm-1 assigned N-N fundamental.
Applying the model's verified Morse curvature convention gives a provisional
width near 2.938 inverse A. This is tested separately with length and depth
fixed. It is a high-risk candidate because it increases local curvature by
about 116% and may stiffen short-range N-N encounters; the polyatomic
fundamental is not assumed to be an exact diatomic target.

The candidate passed all deterministic suites. It moved the raw diagnostic to
1077.13 cm-1, kept the controlled coordinate smooth and gave a 400-step
hydrazine NVE drift of `+4.51e-5 eV` with zero move caps.

Matched carbon-rich CUDA batches compared `nn_length_candidate` (width 2.00)
with `nn_width_2938_candidate` (width 2.938), seeds 17000--17015.

| Quantity | Width 2.00 control | Width 2.938 candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 60.438 | 59.438 | -1.000 |
| mean largest structure (atoms) | 12.188 | 12.375 | +0.187 |
| mean largest heavy-atom count | 6.625 | 6.688 | +0.063 |
| mean largest carbon count | 4.875 | 4.875 | 0.000 |
| mean best carbon chain | 3.500 | 3.438 | -0.062 |
| mean species count | 51.000 | 49.875 | -1.125 |
| mean final temperature (K) | 556.87 | 548.67 | -8.20 |
| mean final potential (eV) | -1093.68 | -1089.45 | +4.23 |

The normal-mixture result is numerically clean and structurally neutral, but
neither set recorded an N-N-containing species at any point. It therefore does
not exercise the parameter whose 47% increase is under review. Exact agreement
with the 1077.24 cm-1 fundamental is not independent validation because the
candidate was derived from that same polyatomic normal mode, not a harmonic
internal-coordinate force constant or reference potential curve.

Recommendation: reject width 2.938 for now and retain 2.00. The experiment
shows that the steeper width can run safely, but not that it is physically more
accurate overall. Revisit only with an independent hydrazine harmonic force
constant/reference potential and a validation that directly exercises N-N
formation and dissociation.
