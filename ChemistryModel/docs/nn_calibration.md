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
