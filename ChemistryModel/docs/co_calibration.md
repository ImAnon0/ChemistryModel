# C-O calibration

## Reference interpretation

The committed single C-O row is `1.430 A, 358 kJ/mol, 1.95 inverse A`.
Methanol is the representative isolated single-bond environment. NIST's
experimental structure gives `r(C-O) = 1.427 +/- 0.007 A`; the first candidate
therefore changes only the length to 1.427 A.

ATcT reports `BDE298(CH3-OH) = 384.57 +/- 0.18 kJ/mol` and 376.86 kJ/mol at
0 K. These are molecular dissociation enthalpies, not automatically the
model's effective Morse depth. Depth remains 358 kJ/mol during the length
experiment and will be evaluated separately.

Methanol's assigned gas-phase C-O stretch is 1033 cm-1. It is a polyatomic
normal mode, so it is supporting evidence rather than a literal diatomic
target. The current raw-pair local diagnostic is evaluated independently and
the width remains 1.95 pending a defensible internal-coordinate curvature.

Sources:

- NIST CCCBDB experimental methanol geometry,
  https://cccbdb.nist.gov/listgeomexpx.asp?casno=67561&charge=0
- Ruscic et al., Active Thermochemical Tables methanol dissociation analysis,
  https://doi.org/10.1021/acs.jpca.5b01346
- NIST Chemistry WebBook methanol vibrational assignment,
  https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&Mask=80CAC

## Length candidate

The isolated candidate changes only single-bond `re` from 1.430 to 1.427 A.
It leaves depth, width, C=O and all other pair tables unchanged. A controlled
fixed-fragment methanol coordinate and methanol NVE probe guard local shape,
capture and numerical stability before matched reactive validation.

## Length-candidate results

The candidate passed the calibration suite 11/11, reactive core 8/8 and high
fidelity 7/7. The controlled methanol coordinate sampled its minimum at
1.429 A, retained a smooth 3.7192 eV dissociation coordinate and had no
post-minimum attraction hump. Its 400-step NVE drift was `+8.97e-5 eV` with
zero move caps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 17000--17015, 2 ps, a
21 A fixed box and identical thermal/recording settings. The control is
`integrator_smoothstep_final`; the candidate is `co_length_candidate`.

| Quantity | 1.430 A control | 1.427 A candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 57.750 | 60.375 | +2.625 |
| mean largest structure (atoms) | 11.062 | 12.188 | +1.125 |
| mean largest heavy-atom count | 6.000 | 6.500 | +0.500 |
| mean largest carbon count | 4.625 | 4.812 | +0.188 |
| mean best carbon chain | 3.250 | 3.125 | -0.125 |
| mean species count | 49.875 | 51.125 | +1.250 |
| mean final oxygen species | 11.688 | 12.125 | +0.437 |
| mean final C+O species | 9.125 | 9.562 | +0.437 |
| mean final temperature (K) | 528.83 | 527.70 | -1.13 |
| mean final potential (eV) | -1084.44 | -1090.93 | -6.49 |

All runs in both sets retained oxygen and C+O chemistry. One of sixteen final
headlines matched, which is expected chaotic divergence after changing an
active capture distance. Aggregate temperature, carbon growth and numerical
health were preserved, while structure size rose modestly.

Recommendation: accept the 1.427 A length. It directly improves methanol
geometry, leaves curvature and depth untouched, and introduces no controlled
or reactive regression. Evaluate depth independently after committing length.

## Separate depth experiment

After committing the accepted length, a second isolated candidate changes the
single C-O depth from 358 to 385 kJ/mol while retaining `re = 1.427 A` and
width `1.95 inverse A`. This tests the supplied rounded methanol BDE target
close to ATcT's 384.57 kJ/mol value. It remains an empirical effective-depth
experiment: neither D298 nor the 376.86 kJ/mol 0 K dissociation enthalpy is
identical to a bare Morse De. C=O remains unchanged.

The 385 kJ/mol candidate passed all deterministic suites and its controlled
methanol curve remained smooth. Its NVE drift was `+1.01e-4 eV` with zero
move caps. However, holding width fixed raised the raw local-frequency
diagnostic from 1057.58 to 1096.73 cm-1, farther from methanol's assigned
1033 cm-1 C-O normal mode.

Matched CUDA batches compared `co_length_candidate` (358 kJ/mol) with
`co_depth_385_candidate` (385 kJ/mol), using carbon-rich seeds 17000--17015.

| Quantity | 358 kJ/mol control | 385 kJ/mol candidate | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps / move caps | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| mean heavy bonds formed | 60.375 | 59.938 | -0.438 |
| mean largest structure (atoms) | 12.188 | 11.062 | -1.125 |
| mean largest heavy-atom count | 6.500 | 6.375 | -0.125 |
| mean largest carbon count | 4.812 | 4.688 | -0.125 |
| mean best carbon chain | 3.125 | 3.438 | +0.312 |
| mean species count | 51.125 | 51.625 | +0.500 |
| mean final oxygen species | 12.125 | 12.312 | +0.187 |
| mean final C+O species | 9.562 | 9.688 | +0.126 |
| mean final temperature (K) | 527.70 | 551.62 | +23.92 |
| mean final potential (eV) | -1090.93 | -1093.88 | -2.95 |

Recommendation: reject 385 kJ/mol and retain 358 kJ/mol as the effective C-O
depth. The candidate was stable, but it worsened the provisional curvature
comparison, produced negligible additional C+O chemistry, raised temperature
and reduced largest-structure size. The experimental molecular BDE alone does
not justify replacing the established environment-dependent effective depth.
