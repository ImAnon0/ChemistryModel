# Damped dispersion-only investigation

## Scope

This work asks whether the existing reactive energy can own all short-range
interaction physics while a new term supplies only missing asymptotic
dispersion. It is standalone and uncommitted. No production, Torch, MD,
integrator, or bonded parameter was changed.

## Base reactive-pair audit

An isolated two-atom calculation was scanned using the actual
`reactive.potential_energy`, including its continuous contact and bond-order
parameter interpolation. The wall has no finite maximum: its exponential
Morse repulsion continues upward as separation tends to zero. To report a
reproducible magnitude, the table gives energy and outward force at `0.5` of
the accepted single-bond length, plus the inner zero crossing. The reactive
term is exactly zero beyond its pair-specific outer contact cutoff.

| pair | inner zero (A) | E at 0.5 re (eV) | outward force (eV/A) | reactive zero from (A) | collapse? |
|---|---:|---:|---:|---:|---|
| H-H | 0.394 | 0.89 | 41.26 | 1.186 | no |
| H-C | 0.701 | 7.95 | 72.13 | 1.738 | no |
| H-N | 0.655 | 8.48 | 81.72 | 1.617 | no |
| H-O | 0.642 | 12.46 | 118.43 | 1.536 | no |
| C-C | 0.862 | 9.63 | 126.91 | 2.440 | no |
| C-N | 0.830 | 9.96 | 136.51 | 2.352 | no |
| C-O | 0.870 | 16.81 | 161.29 | 2.283 | no |
| N-N | 0.785 | 6.55 | 127.61 | 2.314 | no |
| N-O | 0.880 | 13.21 | 128.41 | 2.325 | no |
| O-O | 0.887 | 10.85 | 107.95 | 2.360 | no |

At the shortest scanned separations (`0.15 re`) energies range from 24 eV for
H-H to approximately 370 eV for C-O and forces remain outward. Every pair
therefore has a strong exclusion wall; nothing in these pair scans requires a
second generic nonbonded repulsive core. This does not prove that every
many-body reactive encounter is quantitatively accurate, but it rules out the
specific collapse prerequisite posed here.

## Literature damping and coherent parameters

Tang and Toennies damp each asymptotic coefficient as

    E_disp,6(r) = -C6 f6(br) / r^6
    f6(x) = 1 - exp(-x) sum(k=0..6) x^k/k!.

The function tends to zero at short range and one at long range. In its
original setting, `b` is the exponent of the Born-Mayer exchange-repulsion
term, so it is not a freely interchangeable switching radius.

Source: Tang and Toennies, *J. Chem. Phys.* **80**, 3726 (1984),
[doi:10.1063/1.447150](https://doi.org/10.1063/1.447150).

The C6 family is derived coherently from the already audited UFF 12-6 values:

    D[(x/r)^12 - 2(x/r)^6]  =>  C6 = 2 D x^6.

UFF geometric mixing of `D` and `x` is retained. No external C6 table is mixed
in. Source: Rappe et al., *JACS* **114**, 10024 (1992),
[doi:10.1021/ja00051a040](https://doi.org/10.1021/ja00051a040).

For the first no-fit compatibility test, `b` is the exponent of the repulsive
exponential already present in ChemistryModel's single-bond Morse curve,
`b_ij = 2 a_ij`. This follows the Tang-Toennies mapping literally and uses no
new pair parameters. It is a compatibility hypothesis, not a validated claim
that a reactive Morse exponent equals the physical density-overlap exponent.

| pair | C6 (eV A^6) | b (A^-1) |
|---|---:|---:|
| H-H | 2.2049 | 3.9872 |
| H-C | 8.0927 | 3.6000 |
| H-N | 5.6318 | 3.9000 |
| H-O | 4.5926 | 4.3600 |
| C-C | 29.7024 | 3.7000 |
| C-N | 20.6701 | 3.8000 |
| C-O | 16.8560 | 3.9000 |
| N-N | 14.3845 | 4.0000 |
| N-O | 11.7303 | 4.0000 |
| O-O | 9.5658 | 5.4700 |

## Pair-curve result

The damped term is finite, has no shelf, recovers `-C6/r^6`, and its analytic
force agrees with finite differences. Nevertheless it is much too strong in
the chemical region:

| pair | minimum shift (A) | well-energy change (eV) |
|---|---:|---:|
| H-H | -0.02181 | -0.4252 |
| H-C | -0.03279 | -0.5174 |
| H-N | -0.03284 | -0.5731 |
| H-O | -0.03704 | -0.8045 |
| C-C | -0.04439 | -1.6838 |
| C-N | -0.03422 | -1.3865 |
| C-O | -0.03452 | -1.1317 |
| N-N | -0.02954 | -1.3139 |
| N-O | -0.03258 | -0.8345 |
| O-O | -0.10155 | -1.7755 |

These changes are far beyond a dispersion-scale perturbation to an accepted
chemical well. The mathematical damping form passes, but its required overlap
exponent is not supplied by the currently audited UFF `x,D` data, and the
reactive Morse exponent is not a compatible substitute.

UFF can be represented with exponential-6 alternatives, but selecting or
deriving an exponential hardness solely to make these curves pass would add a
new damping parameter convention. It would need independent interaction-energy
or density-overlap references rather than tuning against the desired curve.

## Representative fixed-orientation dimers

The same conclusion appears in cross-molecule scans. Values below compare the
principal base interaction minimum with the damped-dispersion total; these are
diagnostic orientations, not fitted dimer benchmarks and do not include the
parked electrostatics model.

| dimer | base minimum (eV) | with dispersion (eV) |
|---|---:|---:|
| H2...H2 | -0.163 | -0.411 |
| CH4...CH4 | -0.021 | -0.764 |
| H2O...H2O | approximately 0 | -0.229 |
| NH3...NH3 | -0.028 | -0.596 |
| CH4...H2O | -0.012 | -0.366 |
| NH3...H2O | -0.096 | -0.423 |

The correction overbinds every representative orientation. Several scans also
retain the base model's reactive entrance structure; the correction is not a
validated molecular interaction model merely because it is always attractive.

## Interaction-context design (not implemented)

The existing `_reactive_intermediates` mechanism should become an ephemeral,
internal context produced once per energy evaluation:

    PairGeometry
        pair/neighbour indices, validity mask, batch identity
        minimum-image vectors, distances
        centre and partner element types

    ReactivePairState
        inner/outer contact radii, continuous contact taper
        continuous bond order
        order-blended length, depth and width

    ReactiveLocalState
        coordination, valence, bonded order
        lone-pair and steric descriptors

    InteractionContext
        geometry, pair_state, local_state

Requirements:

- tensors remain attached to the one current autograd graph;
- the context is not cached across integration steps;
- NumPy may use dense matrices and Torch padded neighbours behind the same
  semantic field names;
- pair double-counting ownership is explicit;
- diagnostics receive an explicitly detached snapshot;
- existing equations remain unchanged while new standalone consumers prove
  the interface.

The smallest migration is to replace the private high-fidelity tuple with this
named internal object and verify parity. That design should not be implemented
until a scientifically acceptable dispersion term exists; it is not the cause
of the current parameter failure.

## Recommendation

REVISE dispersion damping/parameters
