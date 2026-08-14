# C-H calibration

## Reference interpretation

The model's C-H row initially used `1.09 A, 439 kJ/mol, 1.80 inverse A`.
The supplied 439 kJ/mol value is a methane dissociation quantity at finite
temperature, not automatically a universal pair `De`. The current effective
pair depth already equals that reference numerically, so it is retained rather
than applying an unsupported zero-point conversion to only one side of the
CH4 -> CH3 + H dissociation.

Methane spectroscopy is multidimensional. Its approximately 2917 cm-1
symmetric-stretch fundamental is anharmonic and collective, so it is not used
as though it were the harmonic frequency of an isolated C-H Morse oscillator.
The current pair's local two-body diagnostic is 2936.22 cm-1; width 1.80 is
therefore retained pending a defensible methane internal-coordinate force
constant.

A high-accuracy methane potential study reports equilibrium C-H values tightly
clustered around 1.086 A, including a combined experimental/ab-initio value of
1.08595(30) A. The isolated candidate changes only `re` from 1.090 to 1.086 A.

Source: Owens et al., *J. Chem. Phys.* 145, 104305 (2016),
https://doi.org/10.1063/1.4962261

## Untouched baseline

- raw pair depth: 4.549930 eV
- raw local harmonic diagnostic: 2936.22 cm-1
- controlled CH4 one-bond minimum: 1.089 A on the sampled grid
- controlled dissociation coordinate: 4.5720 eV
- short-range energy relative to dissociation: +7.7137 eV
- post-minimum falling steps: 0 (no capture/dissociation hump)
- CH4 400-step NVE drift: +0.00002414 eV; zero capped steps

Depth and width are deliberately outside this candidate's scope. The length
change also moves C-H cutoffs by 0.37%, so normal reaction dynamics must be
checked before acceptance.

## Candidate results

With `re = 1.086 A`, the full fast suites passed: calibration 6/6,
reactive-core 5/5, and high-fidelity 7/7. The controlled coordinate retained a
single well and hump-free capture path. CH4 NVE drift changed from
`+2.414e-5 eV` to `+1.994e-5 eV`, with zero capped steps in both cases.

Matched 16-seed CUDA batches used `H rich x5`, seeds 14000--14015, 2 ps, a
21 A fixed box and identical recording/thermal settings. CH4 appeared during
every trajectory in both sets.

| Quantity | Legacy 1.090 A | Candidate 1.086 A | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps | 0 / 0 | 0 / 0 | 0 / 0 |
| mean heavy bonds formed | 39.000 | 39.188 | +0.188 |
| mean largest structure (atoms) | 9.000 | 8.938 | -0.062 |
| mean species count | 45.438 | 45.000 | -0.438 |
| mean final temperature (K) | 525.79 | 519.79 | -6.00 |
| mean final potential (eV) | -1229.667 | -1231.071 | -1.404 |

Only 2/16 final headlines matched, which is normal trajectory divergence after
changing an active capture range. Aggregate chemistry remained close and both
sets were numerically clean. Timing favoured the candidate in this run, but
the batches were not interleaved and no speed claim is made.

Recommendation: accept the 1.086 A length correction. It improves the methane
equilibrium geometry reference without changing depth or curvature and causes
no material aggregate or stability regression in the matched sample.
