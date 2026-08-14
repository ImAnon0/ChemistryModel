# Bond calibration baseline

Measured on `codex/bond-calibration` before changing any engine parameter.

## H2 pair curve

| Quantity | Current model | Reference / interpretation |
| --- | ---: | ---: |
| equilibrium distance | 0.740 A (0.740875 A sampled) | 0.74144 A |
| well depth | 4.508473 eV | 4.747149 eV from D0 + second-order ZPE |
| Morse width | 1.940 inverse A | fitted below only after choosing depth |
| harmonic wavenumber | 4279.04 cm-1 | 4401.21 cm-1 |
| curvature | 33.9362 eV/A2 | diagnostic |
| post-minimum falling steps | 0 | no artificial dissociation hump |

The ideal-Morse depth inferred from both spectroscopic constants is 4.948600
eV. It is retained as a model diagnostic, not used as the experimental depth,
because real H2 is not an exact Morse oscillator.

## Short CPU NVE probes

All probes used seed 19, 100 K initial velocities, 400 steps, no thermostat.

| Molecule | energy drift (eV) | max displacement (A) | capped steps |
| --- | ---: | ---: | ---: |
| H2 | +0.000000933 | 0.10413 | 0 |
| CH4 | +0.000024139 | 0.63462 | 0 |
| NH3 | +0.000039770 | 0.33253 | 0 |
| H2O | +0.000033037 | 1.34850 | 0 |

Median runtime for the 100-step H2 CPU probe over five repetitions was
0.2442 seconds. Runtime is an environment-sensitive guardrail, not a strict
scientific assertion.

The baseline suite passed 4/4 tests. No simulation engine file was changed to
produce these measurements.

## Experimental H-H candidate

The isolated candidate changes only the H-H row to:

`re = 0.74144 A, D = 458.02871 kJ/mol, a = 1.94458 inverse A`

The depth is the precision D0 value plus the second-order spectroscopic ZPE.
After selecting that depth, the width is solved from the Morse curvature that
reproduces the measured harmonic wavenumber. This is an effective classical
pair fit; it is not a claim that BDE, activation barriers and spectroscopic De
are interchangeable.

| Quantity | Baseline | Candidate |
| --- | ---: | ---: |
| equilibrium distance (A) | 0.74000 | 0.74144 |
| depth (eV) | 4.508473 | 4.747149 |
| harmonic wavenumber (cm-1) | 4279.04 | 4401.21 |
| H2 NVE drift (eV) | +0.000000933 | +0.000000372 |
| median 100-step CPU probe (s) | 0.2442 | 0.2335 |

The runtime difference is within ordinary short-run noise and is not claimed
as a speed improvement. CH4, NH3 and H2O probe results were unchanged to the
printed precision, as expected because none contains an H-H contact.

Validation with the candidate applied:

- bond calibration regressions: 4/4 passed
- reactive-core regressions: 5/5 passed
- high-fidelity regressions: 7/7 passed
- methyl-radical scan: smoothly attractive from the capture region to a
  minimum near the configured C-C equilibrium, with no recreated capture wall

Recommendation: retain the candidate for controlled mixture comparisons. It
is materially better for isolated H2 geometry, depth and harmonic curvature,
and no tested non-H2 chemistry changed. Before committing it as a general
default, compare fixed-seed H-rich formation/dissociation statistics because a
deeper H-H well can legitimately alter reaction competition even though the
integrator and all non-H-H parameters are untouched.

## Matched H-rich CUDA comparison

Two 16-seed `H rich x5` batches used seeds 13000--13015, 2 ps, a 21 A fixed
box, identical thermal settings, recorder v2 and CUDA grouping. The only model
difference was the H-H row. H2 appeared during every run in both sets.

| Quantity | Legacy H-H | Candidate H-H | Candidate - legacy |
| --- | ---: | ---: | ---: |
| finished / stable | 16 / 16 | 16 / 16 | 0 |
| strikes | 0 | 0 | 0 |
| energy jumps | 0 | 0 | 0 |
| mean heavy bonds formed | 37.812 | 37.875 | +0.063 |
| mean largest structure (atoms) | 9.188 | 9.250 | +0.062 |
| mean species count | 45.562 | 45.438 | -0.125 |
| mean final temperature (K) | 518.67 | 525.15 | +6.48 |
| mean final potential (eV) | -1220.906 | -1238.311 | -17.405 |
| mean wall time per seed (s) | 14.7 | 11.5 | -3.2 |

Individual final headlines differ in most seeds, which is expected for a
chaotic reactive trajectory after changing a physically active bond surface.
The aggregate structure and reaction-count observables remain very close. The
more negative candidate potential is expected because many H-H contacts sample
a well that is 0.239 eV deeper; it is not evidence of numerical energy loss.
The timing difference is favourable but is not attributed to the parameter
change without repeated interleaved benchmarks.

Final recommendation: accept this H-H candidate. It matches the selected H2
reference observables substantially better, remains stable, preserves the
tested non-H2 force behaviour, and does not materially shift aggregate output
in the matched H-rich sample. Keep the old constants documented so this result
can be reproduced or reversed.
