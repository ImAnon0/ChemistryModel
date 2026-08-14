# Dispersion damping-parameter provenance gate

## Scope and fixed quantities

This investigation changes neither the Tang-Toennies `f6` form nor the
previously audited UFF-derived coefficients

    C6_ij = 2 D_ij r_min,ij^6.

Only the damping exponent `b_ij` is examined. No value is fitted to a
ChemistryModel bond minimum, reaction, or simulation outcome. Production,
Torch, MD, and the proposed interaction context remain untouched.

## Provenance search

### 1. Free-atom ionization-potential / Born-Mayer-IP convention

This is the only complete, internally consistent elemental convention found
that directly covers arbitrary H/C/N/O pairs without adding chemical atom
types. The asymptotic free-atom density exponent is

    B_i = 2 sqrt(2 IP_i)       [bohr^-1]

and the published unlike-pair rule used in SAPT-derived Born-Mayer force
fields is

    B_ij = (B_i + B_j) B_i B_j / (B_i^2 + B_j^2).

The values use neutral ground-state ionization energies from the
[NIST Atomic Spectra Database](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html):
H `13.598434599702`, C `11.2602880`, N `14.53413`, O `13.618055` eV.

The equation mapping and comparison against density-derived approaches are
documented by Van Vleet et al., *JCTC* **12**, 3851 (2016),
[doi:10.1021/acs.jctc.6b00209](https://doi.org/10.1021/acs.jctc.6b00209).

Resulting complete pair set:

| pair | b (A^-1) |
|---|---:|
| H-H | 3.77844 |
| H-C | 3.59237 |
| H-N | 3.84023 |
| H-O | 3.77981 |
| C-C | 3.43830 |
| C-N | 3.64259 |
| C-O | 3.59349 |
| N-N | 3.90628 |
| N-O | 3.84169 |
| O-O | 3.78117 |

This convention is provenance-complete, but its limitation is explicit:
free-atom tails do not include the redistribution of density in molecules.
The cited comparison finds its Tang-Toennies dispersion less predictive than
ISA density-derived damping.

### 2. Slater-ISA / scaled-ISA

Slater-ISA obtains exponents by fitting exponential tails of partitioned
atom-in-molecule densities. It is the strongest physical provenance found and
improves dispersion damping against SAPT decomposition. It does **not** supply
one universal `H`, `C`, `N`, and `O` exponent: carbonyl C, methyl C, molecular
O, and H can have different density tails, and those tails change with the
molecular environment. Published examples therefore form molecule/site
parameterizations rather than a complete arbitrary-reactive-geometry table.

For example, a published water model reports `OO=1.7794`, `OH=1.9011`, and
`HH=2.0227` in atomic units. That is traceable but covers water sites only; it
cannot populate C/N or establish transferability across radicals and changing
coordination. Applying its values element-wide would discard the feature ISA
was introduced to represent.

### 3. Other located sets

- PAH Tang-Toennies potentials provide C/H values tied to a hydrocarbon/PAH
  model, leaving N/O uncovered.
- SAPT force fields for water, acetonitrile, ionic liquids, and other named
  molecules provide site-specific exponents fitted or derived for those
  monomers, not universal reactive elements.
- DFT-D and related rational damping families have complete element coverage,
  but their damping and C6 values are jointly tied to a particular electronic
  structure method and coordination convention. Combining their damping with
  fixed UFF C6 would violate the required internally coherent provenance.
- Scaling ISA exponents by a published global factor still requires the ISA
  atom-in-molecule exponents that are absent for arbitrary ChemistryModel
  environments.

No missing pair was filled by interpolation or by mixing these families.

## Standalone behavior of the complete Born-Mayer-IP candidate

### Atomic pair curves

| pair | minimum shift (A) | bonded-well change (eV) |
|---|---:|---:|
| H-H | -0.01566 | -0.3294 |
| H-C | -0.03251 | -0.5130 |
| H-N | -0.03033 | -0.5384 |
| H-O | -0.01810 | -0.4522 |
| C-C | -0.03143 | -1.2743 |
| C-N | -0.02815 | -1.1835 |
| C-O | -0.02375 | -0.8428 |
| N-N | -0.02623 | -1.2033 |
| N-O | -0.02733 | -0.7263 |
| O-O | -0.02428 | -0.5594 |

The correction is continuous, shelf-free, has no secondary pair minimum or
new pair barrier, and recovers `-C6/r^6`. Those mathematical gates pass. The
scientific magnitude gate fails: this is not a small long-range correction.
It materially moves every accepted chemical minimum and introduces eV-scale
deepening for C-C, C-N, and N-N.

### Representative fixed-orientation dimers

| dimer | base principal minimum (eV) | Born-Mayer-IP total (eV) |
|---|---:|---:|
| H2...H2 | -0.163 | -0.374 |
| CH4...CH4 | -0.021 | -0.726 |
| H2O...H2O | approximately 0 | -0.190 |
| NH3...NH3 | -0.028 | -0.552 |
| CH4...H2O | -0.012 | -0.315 |
| NH3...H2O | -0.096 | -0.356 |

Every diagnostic orientation is overbound. Existing reactive entrance
structure is not cured by the damping and must not be mistaken for a validated
outer dispersion well. Polar dimers also lack the deliberately parked
electrostatics model, so their absolute minima are diagnostics rather than
experimental predictions.

Analytic forces agree with finite-difference energy derivatives, and both
damping conventions recover the fixed UFF `-C6/r^6` asymptote. Thus rejection
is based on physical distortion, not implementation failure.

## Provenance-gate decision

A complete sourced free-atom convention exists but fails the strict behavior
gates. The better-supported ISA approach does not provide a universal,
ready-made H/C/N/O set for topology-changing reactive environments. Creating
such exponents would be a new quantum-density/SAPT parameter-generation
project, which is outside this provenance search.

No production vdW or interaction-context refactor should proceed from the
available parameters.

NO DEFENSIBLE DAMPING SET — park vdW
