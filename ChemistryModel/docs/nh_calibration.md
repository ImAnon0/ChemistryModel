# N-H calibration

## Reference interpretation

The original N-H row was `1.010 A, 449 kJ/mol, 1.95 inverse A`. The supplied
449 kJ/mol value describes H-NH2 dissociation at finite temperature, not a
universal pair `De`; it is retained as the established effective depth.

Ammonia's observed stretch fundamentals are approximately 3337 and 3444
cm-1. They are anharmonic normal modes involving three N-H coordinates and the
pyramidal molecular framework, so they are not inserted directly into a
two-body Morse formula. The current raw-pair local diagnostic is 3199.14 cm-1.
Width 1.95 remains unchanged pending a defensible internal-coordinate harmonic
force constant.

The high-accuracy NH3-Y2010 surface gives equilibrium `r(N-H) = 1.0109 A` and
angle 106.75 degrees. The candidate changes only the table length from 1.0100
to 1.0109 A.

Sources:

- Szabo et al., *J. Phys. Chem. A* 116, 4356 (2012),
  https://doi.org/10.1021/jp211802y
- Huang et al., global ground-state NH3 potential surface, equilibrium
  geometry and dissociation channels, *J. Chem. Phys.* / associated NH3-Y2010
  work as summarized in the cited structure analysis.

## Untouched baseline

- raw pair depth: 4.653573 eV
- raw local harmonic diagnostic: 3199.14 cm-1
- controlled NH3 one-bond minimum: 1.0125 A on the sampled grid
- controlled dissociation coordinate: 4.44155 eV
- short-range energy relative to dissociation: +9.35277 eV
- post-minimum falling steps: 0
- NH3 400-step NVE drift: +0.00003977 eV; zero capped steps

Depth and width are outside this candidate's scope. The length change moves
N-H interaction cutoffs by only 0.09%, but normal reactive dynamics still need
validation before acceptance.

## Candidate results

With `re = 1.0109 A`, the calibration suite passed 7/7, reactive core 5/5,
and high fidelity 7/7. The coordinate remained single-welled and hump-free.
NH3 400-step NVE drift changed from `+3.977e-5 eV` to `+3.654e-5 eV`, with
zero capped steps in both cases.

Matched 16-seed CUDA batches used `H rich x5`, seeds 15000--15015, 2 ps, a
21 A fixed box and identical thermal/recording settings. NH3 appeared during
all trajectories in both sets.

| Quantity | Legacy 1.0100 A | Candidate 1.0109 A | Difference |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes / energy jumps | 0 / 0 | 0 / 0 | 0 / 0 |
| mean heavy bonds formed | 38.938 | 38.188 | -0.750 |
| mean largest structure (atoms) | 9.375 | 9.375 | 0.000 |
| mean species count | 45.625 | 46.250 | +0.625 |
| mean final temperature (K) | 527.60 | 525.64 | -1.96 |
| mean final potential (eV) | -1235.986 | -1239.424 | -3.438 |

Four of sixteen final headlines matched. The mean absolute paired differences
were 1.625 heavy bonds and 1.0 atom in the largest structure, consistent with
ordinary chaotic divergence. Aggregate structure size and numerical health
were preserved. Runtime was effectively identical (11.6 versus 11.7 seconds
per seed).

Recommendation: accept the 1.0109 A correction. It aligns the pair equilibrium
with the high-accuracy ammonia structure while leaving depth and curvature
untouched, and the matched sample shows no stability or material structural
regression.
