# Bond calibration audit

## Current parameter meaning

`reactive.py` is the source table. Each entry contains equilibrium distance in
angstrom, an energy labelled as a dissociation energy in kJ/mol, and a Morse
width in inverse angstrom. `build_tables()` converts the energy to eV. The pair
energy is

`D exp(-2 a (r-re)) - 2 D exp(-a (r-re))`,

so its minimum is exactly `-D` before environment and bond-order effects. In
the mathematical potential, the stored depth therefore behaves as a classical
effective `De`, even though the source numbers appear to be thermochemical
dissociation values such as `D298`. Those quantities must not be conflated.

The single, double and triple tables independently contain `re`, depth and
width. Continuous bond order blends all three tables. The cutoff also depends
on the single-bond `re` through fixed inner/outer factors, so changing a length
also changes interaction range.

Environment softening can reduce the blended depth for single-character bonds
next to multiple-bond commitment. It is currently globally disabled at weight
zero. The over-coordination barrier can optionally scale with contact depth;
that feature is also disabled at weight zero. Angle and over-coordination terms
remain separate from the pair well.

`reactive_torch.py` copies the NumPy tables into tensors at construction and
implements the same Morse expression. `BatchedReactiveSimulation` subclasses
that implementation and changes neighbour indexing and reporting, not pair
parameters. The existing consistency tests compare grouped and single paths.

The high-fidelity model starts from the same Torch pair terms. Its separate
valence-state correction activates only for a hydrogen with competing heavy
partners. Isolated H2 and ordinary single-contact bonds remain on the base
surface.

## H2 reference interpretation

NIST reports `re = 0.74144 A`, `omega_e = 4401.21 cm-1`, and
`omega_e x_e = 121.33 cm-1` for ground-state H2. The precision dissociation
result is `D0 = 36118.06962 cm-1`. A depth-like `De` requires adding zero-point
energy; to second spectroscopic order, `G(0) = omega_e/2 - omega_e x_e/4`.

For an ideal Morse oscillator, `De/(hc) = omega_e^2/(4 omega_e x_e)`. Real H2
is not an exact Morse oscillator, so this second estimate is recorded as a
model-consistency diagnostic rather than treated as an experimental truth.

Sources:

- NIST Chemistry WebBook, Hydrogen, constants of diatomic molecules:
  https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=3FF7&Units=SI
- Liu et al., *J. Chem. Phys.* 130, 174306 (2009), precision D0:
  https://doi.org/10.1063/1.3120443

## Safety boundary

The calibration diagnostics are offline tools. The engine does not import
them. No force, integration, thermostat, reaction, or default parameter has
been changed in establishing this audit and baseline suite.
