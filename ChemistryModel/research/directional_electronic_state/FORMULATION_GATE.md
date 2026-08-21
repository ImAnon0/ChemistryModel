# Directional electronic-state formulation gate

## Frozen starting point

This investigation is research-only. `UnifiedBondCapacityEnergyPrototype` is
the frozen radial reference. Production physics, the established angle layer,
radial tables, H-state mixing, capacity constraints, integrator, and selectors
are outside scope.

## What information is actually missing

The unified radial model solves scalar questions:

- which edge owns attractive energy;
- expected order/occupation of that edge;
- how much scalar valence capacity each endpoint consumes;
- the coherent mixture of alternative H-bond assignments;
- a scalar capacity price at each heavy atom.

Those quantities do **not** determine the orientation of the local valence
electron density. Two local electronic environments can have identical edge
occupations, expected bond order, total coordination, and remaining valence,
yet differ in:

- the direction of a lone-pair-rich region;
- the anisotropy of the occupied valence density;
- the permanent or induced dipole;
- the alignment of donor and acceptor orbitals;
- the coherence of a three-centre transfer state.

The current angle layer reconstructs geometry from nuclei and scalar
coordination. Its “lone-pair count” is only

    max((outer_electrons - scalar_bond_order) / 2, 0)

and has no direction. The H-state Hamiltonian is coherent, but its off-diagonal
coupling uses only a balanced product of two radial tapers. Rotating the
spectator environment while holding the two transfer distances fixed does not
change that coupling. The model therefore has no electronic variable capable
of distinguishing radial access from directional electronic access.

The missing variable is best stated as a **state-conditioned local valence
density anisotropy**. Its lowest useful moments are:

- a vector `mu_i^s` for polar response/dipole direction;
- a symmetric traceless tensor `Q_i^s` for non-polar directional density;
- optionally, off-diagonal coherence between competing bond states.

The superscript `s` matters: averaging edge occupations first loses the
correlation between an electronic direction and the bond assignment that
created it.

## Why this is not another angle term

An angle term is a scalar function of three nuclear positions. A directional
electronic state is an internal variable that responds to the full local
environment and is minimized with the energy. It may produce angular forces,
but it can also distinguish environments with the same bond angle and different
polarity, bond order, lone-pair occupancy, or transfer-state coherence.

A quadratic atom-centred tensor with an isotropic self-energy and a source
formed only from bond axes can be eliminated analytically. The result is a
sum of pairwise angular invariants. That construction is only a disguised
weighted-angle model and is rejected by this gate unless it has an independently
defined anisotropic response or a directly validated electronic observable.

## Candidate audit before implementation

### A. Induced atomic dipoles alone — reject as a first prototype

The standard variational induced-dipole functional has the form

    E(mu) = 1/2 mu^T A mu - mu^T E_perm

with a positive-definite, short-range-damped response matrix `A`. This is a
sound variational electronic degree of freedom and gives analytic envelope
forces when solved tightly. However, ChemistryModel currently has no coherent
permanent charge/multipole field `E_perm`. With no source field the neutral
minimum is `mu=0`, so adding polarizabilities alone cannot give water or ammonia
their polarity. Inventing bond charges would silently restart the parked
electrostatics parameter project. Short-range damping and positive definiteness
would also need independent parameters. See the published induced-dipole
functional and conditioning discussion in
[Aviat et al.](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00981) and the
many-body/damping analysis in
[Wang and Skeel](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00225).

### B. Bond dipoles or split charge plus induced dipoles — defer

This supplies the missing source field and could represent both permanent
polarity and response. It is not parameter-free: electronegativity/charge-flow,
hardness, short-range electrostatics, atomic polarizability, and damping must be
one compatible convention. Prior ChemistryModel research already found the
QEq dissociation pathology and stopped QTPIE/ACKS2/SQE at formulation or
parameter-provenance gates. QTPIE's motivation and correct separated-fragment
limit are documented by
[Chen and Martinez](https://www.sciencedirect.com/science/article/pii/S0009261407002618).
This geometry task must not bypass that result by introducing unsourced bond
charges.

### C. Free local orbital frame with angle-like self-energy — reject

Four freely rotating “sp3 slots”, or analogous 2/3/4-domain vectors, need
hybridization gaps, occupancy energies, orbital widths, and Pauli terms. Without
independent density/orbital targets these parameters are unidentifiable from
the current energy-only microscopes. If their energy depends only on dot
products with bond axes, eliminating the frame again produces a many-angle
potential. The previous electron-domain and joint-local-state failures already
falsify the parameter-free version of this idea.

### D. State-conditioned three-centre/density-anisotropy coupling — admit as a
diagnostic

Unified radial already has the appropriate electronic-state carrier for H
transfer: a Hamiltonian over alternative bond assignments. What it lacks is a
directional descriptor in the off-diagonal coupling. A minimal diagnostic may
multiply the existing radial coupling by a **state-conditioned, independently
anchored density-anisotropy factor**, while leaving diagonal radial energies,
capacity, and all settled one-state molecules exactly unchanged.

The repository contains a frozen continuous SAPT exchange-density descriptor:
an atom-centred `P2` density moment and H/C/N/O anisotropy coefficients fitted
to `SAPT0/jun-cc-pVDZ EXCH10`, not to reaction barriers or the water microscope.
This descriptor has previously been used only for exchange diagnostics. Using
it in covalent state coupling is not asserted as established physics; it is a
falsification probe asking whether state-conditioned density direction carries
the missing signal. A zero-response control must reproduce unified radial
exactly.

Empirical valence-bond literature confirms that an elementary reaction can be
represented by a small Hamiltonian, but also warns that off-diagonal coupling
is an additional, normally parameterized quantity rather than something fixed
by the diagonal force fields; see the methodology summary in
[Bergonzo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10500987/) and the
coupling discussion in
[Aqvist and co-workers](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00126).
Accordingly this diagnostic may reuse only the already frozen anisotropy map;
it may not introduce a fitted strength.

## Admitted prototype invariant

For a direct transfer between states `a` and `b`, let `g_old->new` be the
published/frozen SAPT `P2` angular density factor of the currently unoccupied
target centre in state `a`, evaluated toward the transferring H. Define the
symmetric transition factor

    g_ab = sqrt(g_old->new * g_new->old)

and

    H_ab = H_ab_radial * g_ab.

Required properties:

- `g_ab = g_ba`;
- H3 gives exactly `g_ab=1` because no heavy directional environment exists;
- a one-state settled molecule is exactly unchanged;
- radial cutoff and separated-fragment limits remain those of the baseline;
- no parameter is fitted to water, Grambow, or any named reaction;
- the live Torch scalar, not a detached descriptor, supplies forces.

Failure of this diagnostic does not reject polarisation in general. It rejects
the narrower claim that the existing frozen `P2` density anisotropy is the
missing state variable for unified radial.

## Observability limitation

The frozen 106-geometry QM dataset contains total energies only. It has no
dipoles, polarizabilities, atomic multipoles, density moments, or bond-order
observables. Energy agreement alone cannot identify a new electronic state.
Therefore:

1. the admitted no-fit diagnostic can be screened against existing energies;
2. no new electronic-response parameter may be fitted in this task;
3. a successful energy screen would still require a separate QM dataset with
   neutral-molecule dipoles, finite-field polarizabilities, and preferably
   density-derived local multipoles before the state could be interpreted as
   physical polarisation.

## Promotion order

1. exact zero-response/baseline identity;
2. finite forces, permutation symmetry, cutoff continuity, and molecule
   preservation;
3. frozen H3/methane/formaldehyde/water microscopes;
4. water must be no worse than `0.213629 eV` RMSE;
5. matched NVE;
6. only then, frozen Grambow 200.

Any failure stops promotion. A better benchmark score cannot rescue a failed
electronic-observability, molecule, water, or dynamics gate.
