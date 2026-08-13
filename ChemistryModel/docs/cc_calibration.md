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
