# O-H calibration

## Reference interpretation

The original O-H row was `0.960 A, 498 kJ/mol, 2.18 inverse A`. The supplied
498 kJ/mol is the finite-temperature H-OH dissociation reference, not directly
a universal pair `De`, and the model already uses it as its effective depth.
It is retained.

Gas-phase water fundamentals are 3656.65 cm-1 (symmetric stretch) and 3755.79
cm-1 (antisymmetric stretch). They are coupled, anharmonic molecular modes.
The current raw-pair local diagnostic is 3750.80 cm-1, already bracketed by the
observed stretches, so width 2.18 remains unchanged rather than being fitted to
one mode.

A multi-isotopomer rotation-vibration fit gives equilibrium
`r(O-H) = 0.957848(16) A` and angle `104.5424(46) degrees`. The candidate
changes only the table length from 0.9600 to 0.95785 A.

Sources:

- Jensen et al., refined water ground-state potential, *J. Mol. Spectrosc.*,
  equilibrium geometry `0.957848(16) A`.
- NIST Chemistry WebBook water vibrational compilation:
  https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1883&Units=SI

## Untouched baseline

- raw pair depth: 5.161424 eV
- raw local harmonic diagnostic: 3750.80 cm-1
- controlled H2O one-bond minimum: 0.9600 A on the sampled grid
- controlled dissociation coordinate: 5.161424 eV
- short-range energy relative to dissociation: +16.23121 eV
- post-minimum falling steps: 0
- H2O 400-step NVE drift: +0.00003304 eV; zero capped steps

Depth and width are outside this candidate's scope. The 0.22% length change
also moves O-H cutoffs and therefore requires normal reactive validation.

## Candidate result: rejected

The isolated `re = 0.95785 A` candidate retained a smooth water coordinate and
passed the eight calibration regressions and five reactive-core regressions.
H2O NVE drift remained small (`+3.509e-5 eV` versus `+3.304e-5 eV`) with zero
capped steps.

However, the established high-fidelity suite failed its deterministic transfer
gate force-continuity check:

`force kink of 0.691 eV/A at x = 1.3820 A`

Restoring `re = 0.960 A` immediately restored the high-fidelity result to 7/7
passes. This is mechanistically credible: changing O-H length also shifts its
cutoffs, while the separate hydrogen-transfer gate was calibrated around the
existing base surface. Adjusting that gate merely to permit this small geometry
change would expand the candidate beyond one independently constrained pair
parameter and could change reaction barriers.

Recommendation: retain `re = 0.960 A`, depth 498 kJ/mol and width 2.18. The
0.00215 A isolated-geometry improvement is not worth an unexplained
whole-model force-continuity regression. Revisit O-H only as a joint,
explicitly benchmarked transfer-surface recalibration—not as a table-only
change. No stochastic mixture batch is required for this rejected candidate.
