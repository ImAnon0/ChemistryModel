# C-C calibration

## Reference interpretation

The original single C-C row is `1.540 A, 348 kJ/mol, 1.85 inverse A`.
Ethane is the relevant single-bond environment. Modern high-level structures
place its equilibrium C-C distance near 1.525 A, while older ground-state
spectroscopic distances around 1.536 A include vibrational averaging.

The supplied `CH3-CH3` dissociation enthalpy is 377 kJ/mol. A modern
thermochemical analysis distinguishes an approximately 90.0 kcal/mol bond
dissociation enthalpy from a 97.4 kcal/mol electronic dissociation energy.
Neither may be inserted uncritically as the model pair `De`; depth is therefore
held at 348 kJ/mol during the geometry experiment.

Ethane's approximately 993 cm-1 C-C stretching feature is a molecular normal
mode rather than a pure two-carbon oscillator. The current raw-pair diagnostic
is 1057.31 cm-1. Width remains 1.85 pending a clean internal-coordinate force
constant or reference potential scan.

Sources:

- Blokker et al., methyl substitution and alkyl bond dissociation,
  https://doi.org/10.1002/anie.202207477
- Ethane structural spectroscopy reports a ground-state C-C distance of
  1.536(2) A; modern equilibrium calculations use approximately 1.525 A.

## Untouched baseline

- raw pair depth: 3.606778 eV
- raw local harmonic diagnostic: 1057.31 cm-1
- controlled fixed-CH3 minimum: 1.540 A
- controlled dissociation coordinate: 3.650919 eV
- short-range energy relative to dissociation: +6.752492 eV
- post-minimum falling steps: 0

The first candidate changes only single-bond `re` from 1.540 to 1.525 A.
Double- and triple-bond tables, depth and width remain unchanged. Since this
shortens C-C capture cutoffs by 0.97%, methyl recombination and normal mixture
dynamics require explicit validation.

## Length-candidate results

With single-bond `re = 1.525 A`, the calibration suite passed 9/9, reactive
core 5/5, and high fidelity 7/7. The explicit methyl scan remained smoothly
attractive through capture. The ethane 400-step NVE probe drifted only
`+2.107e-5 eV` with zero capped steps.

Matched 16-seed CUDA batches used `carbon rich`, seeds 16000--16015, 2 ps, a
21 A fixed box and identical thermal/recording settings. Every run in both
sets formed multi-carbon structures.

| Quantity | Legacy 1.540 A | Candidate 1.525 A | Difference |
| --- | ---: | ---: | ---: |
| finished | 16 / 16 | 16 / 16 | 0 |
| stable | 15 / 16 | 16 / 16 | +1 |
| energy jumps | 2 | 0 | -2 |
| mean heavy bonds formed | 61.125 | 60.688 | -0.437 |
| mean largest structure (atoms) | 12.125 | 12.250 | +0.125 |
| mean largest carbon count | 5.000 | 5.375 | +0.375 |
| mean best carbon chain | 4.000 | 3.625 | -0.375 |
| mean species count | 54.812 | 54.375 | -0.437 |
| mean final temperature (K) | 530.91 | 541.79 | +10.88 |
| mean wall time per seed (s) | 10.8 | 10.5 | -0.3 |

Legacy seed 16011 had two energy jumps, including a 40,382.3 eV event, and
was marked unstable. The matched candidate seed was stable with zero jumps and
a largest reported change of 1.0 eV. This single event is not claimed as proof
of a universal stability improvement, but the candidate clearly introduces no
stability penalty.

Four of sixteen final headlines matched, as expected for a changed active
capture range. Aggregate chemistry remained close; the candidate slightly
increased largest carbon count while slightly reducing the chain metric.

Recommendation: accept the 1.525 A single-bond length. It improves ethane
equilibrium geometry, preserves methyl capture and aggregate carbon chemistry,
and was numerically clean. Keep depth 348 kJ/mol and width 1.85 unchanged until
the dissociation-energy and internal-coordinate-curvature questions are tested
independently.

## Separate depth experiment

After committing the accepted length, a second isolated experiment tests the
supplied CH3-CH3 value by changing depth `348 -> 377 kJ/mol` while holding
`re = 1.525 A` and width `1.85 inverse A`. This is an empirical candidate, not
an assertion that the finite-temperature dissociation enthalpy equals Morse
`De`. Because width is held fixed, the deeper well also raises local curvature
and frequency; both must be considered in the acceptance decision.

The 377 kJ/mol candidate passed all deterministic suites and preserved methyl
capture, but raised the controlled well from 3.651 to 3.951 eV and the raw
local-frequency diagnostic from 1057 to 1100 cm-1, farther from the provisional
993 cm-1 ethane mode.

Matched 16-seed carbon-rich CUDA batches then isolated the depth change at the
accepted 1.525 A length:

| Quantity | 348 kJ/mol control | 377 kJ/mol candidate |
| --- | ---: | ---: |
| finished | 16 / 16 | 16 / 16 |
| stable | 16 / 16 | 15 / 16 |
| energy jumps | 0 | 1 |
| mean heavy bonds formed | 59.500 | 60.688 |
| mean largest structure (atoms) | 11.438 | 11.438 |
| mean largest carbon count | 4.750 | 4.625 |
| mean best carbon chain | 3.188 | 3.188 |
| mean species count | 50.312 | 52.188 |

Candidate seed 17005 produced a 69,917.6 eV total-energy jump and ended at
2747.6 K. The exact matched 348 kJ/mol run was stable, with a largest energy
change of only 1.3 eV. Recorded frames show kinetic energy jumping from 23.3
to 69,936.4 eV while potential changed by only 4.6 eV; this is a numerical
impulse, not plausible C-C dissociation energy.

Recommendation: reject 377 kJ/mol and retain the committed 348 kJ/mol effective
depth. It worsens the provisional curvature comparison and caused a matched
numerical regression without improving largest-structure or chain output.
The failure also exposes a separate integrator safeguard issue: capped position
moves still receive the full uncapped Velocity-Verlet velocity update. That
general failure mechanism should be corrected independently from calibration.
