# Generic nonbonded van der Waals design

## Status and scope

This is a design-only milestone. It does not alter `ReactiveSimulation`,
energies, forces, neighbour lists, CUDA batching, or any calibrated bonded
parameter. Production implementation begins only after this design and the
standalone pair-curve milestone are reviewed.

The first vdW implementation must be optional and default to `OFF`. With vdW
off, the existing model must remain unchanged within its deterministic
numerical tolerances.

## What atoms experience today

The production reactive model supplies:

- a smoothly tapered Morse pair term;
- continuous valence allocation and single/double/triple bond interpolation;
- environment softening of bond depths;
- quadratic over-coordination energy;
- continuous angular energy based on coordination and lone pairs; and
- the high-fidelity competing-contact correction for hydrogen transfer.

The same minimum-image periodic displacement is used by the NumPy reference
and Torch implementation. Chemical pair tapers are one inside `1.25 r_e`,
fall by a cosine taper, and are zero outside `1.60 r_e`. The largest current
outer radius is the C-C single-bond value, approximately 2.44 A.

Consequently:

| Separation/state | Current interaction |
|---|---|
| bonded | Morse/bond-order, coordination, angle, and applicable transfer terms |
| reactive near-contact | tapered Morse plus continuously changing coordination/bond order and penalties |
| moderately separated, beyond chemical outer radius | **none** |
| far apart inside the neighbour/search cell | **none** |

There is no generic London dispersion, no nonbonded excluded-volume/Pauli
term outside the chemical contact range, no electrostatics, and no hydrogen-
bond interaction.

## Why the old LJ code is not suitable

`interactions.py` is an isolated educational/reference implementation and is
not called by `ReactiveSimulation.energy_per_atom()`. Enabling it would be
scientifically and architecturally unsafe because it has:

- one unproven global `epsilon` and `sigma` for every atom;
- unshielded `r^-12` short-range divergence;
- only an energy shift at the cutoff, leaving force discontinuity;
- no suppression during bonded/reactive contact;
- a dense all-pairs path unrelated to current Torch batching; and
- no matching energy decomposition or OFF/ON compatibility contract.

It remains unused.

## Alternatives considered

### Lennard-Jones 12-6

Advantages: simple, auditable, inexpensive, differentiable away from zero,
widely validated as a baseline, and maps directly to pairwise Torch kernels.
Its repulsive exponent is only an empirical Pauli proxy and can be too stiff
during high-energy collision, so short-range suppression is mandatory.

### Buckingham / exp-6

The exponential repulsion is physically preferable to `r^-12`, but a single
coherent, topology-free H/C/N/O parameter family suitable for transplant into
this reactive model was not identified. The attractive `-C6/r^6` term still
needs short-range damping. It is not selected for the first milestone.

### ReaxFF-style shielded vdW

ReaxFF demonstrates the right architecture: nonbonded interactions between
all atoms, shielded at short range, while bonded terms vary with bond order.
However, its vdW parameters and shield exponents are fitted jointly with a
particular ReaxFF parameter set. Copying them into ChemistryModel would break
that convention and obscure double counting.

### Damped dispersion-only forms

Models such as D3 provide defensible long-range attraction, but not the
matching excluded-volume repulsion required for stable neutral encounters.
They are inappropriate as the complete initial nonbonded layer.

## Selected initial form

Use the Universal Force Field (UFF) 12-6 vdW convention from Rappe et al.,
JACS **114**, 10024 (1992), DOI 10.1021/ja00051a040:

```
V_UFF,ij(r) = D_ij [(x_ij/r)^12 - 2(x_ij/r)^6]
```

`x_ij` is the pair minimum and `D_ij` is the well depth. UFF is selected as a
coherent baseline, not as a claim of final condensed-phase accuracy.

### H/C/N/O parameters

| Element | UFF `x_i` (A) | UFF `D_i` (kcal/mol) | `D_i` (eV) |
|---|---:|---:|---:|
| H | 2.886 | 0.044 | 0.001908 |
| C | 3.851 | 0.105 | 0.004553 |
| N | 3.660 | 0.069 | 0.002992 |
| O | 3.500 | 0.060 | 0.002602 |

The ordinary H, C, N, and O UFF types use these vdW values; the relevant C,
N, and O hybridization variants do not require a new vdW table. ChemistryModel
therefore does not need fixed atom typing for this term.

Use UFF's geometric combining rules without alteration:

```
x_ij = sqrt(x_i x_j)
D_ij = sqrt(D_i D_j)
```

No UFF bonded, angle, torsion, charge, or other parameters are imported.

## Smooth bonded/reactive suppression

Let `tau_ij` be the existing chemical contact taper: one at mature chemical
contact and zero beyond the chemical outer radius. Define

```
g_chemical(tau) = 1 - 3 tau^2 + 2 tau^3.
```

This complement smoothstep is zero with zero slope at `tau=1`, and one with
zero slope at `tau=0`. The added energy is

```
V_added,ij(r) = g_chemical(tau_ij)
                S_cut(r)
                V_UFF,ij(r_safe).
```

This is a ChemistryModel-specific blend. It is not part of UFF.

`r_safe` is a numerical floor below the smallest chemical inner radius. The
floor is reached only where `g_chemical` is identically zero, so it prevents
`0 * infinity` and NaNs without supplying physical force in the suppressed
region. Implementations must avoid evaluating divergent powers before the
safe distance is applied.

The blend must be rejected if any pair curve develops:

- a second attractive minimum competing with the chemical bond;
- a barrier or force reversal caused only by the handoff;
- an energy or force discontinuity at either chemical taper endpoint; or

- inadequate repulsion during an unbonded high-energy approach.

Increasing a suppression exponent or changing the handoff is not automatic
retuning; it requires renewed curve and collision validation.

## Long-range switch and periodic treatment

Use a potential switch from `r_on = 7.0 A` to `r_cut = 8.5 A`. For
`t = (r-r_on)/(r_cut-r_on)`:

```
S_cut = 1                              r <= r_on
S_cut = 1 - 10t^3 + 15t^4 - 6t^5     r_on < r < r_cut
S_cut = 0                              r >= r_cut
```

The quintic switch has zero first and second derivatives at both endpoints,
giving continuous energy and force and a smoother force derivative than a
simple energy shift. Plain truncation and energy shifting are rejected because
their force is discontinuous; force shifting perturbs the force throughout the
active interval.

Continue using the minimum-image convention. Enforce
`r_cut <= box_size/2`; therefore the initial 8.5 A cutoff is valid for typical
19 A and 21 A boxes. Smaller boxes must reject vdW ON or reduce the cutoff only
through an explicit, recorded configuration—not silently.

No analytical long-range tail correction is included initially because the
system is reactive, inhomogeneous, and multi-component. The finite cutoff is
part of the declared model and must be considered when interpreting pressure
and density.

## Neighbour and GPU design

The current chemical table searches only to approximately 2.44 A plus skin
and stores at most 64 neighbours. Extending that same padded table to 8.5 A is
unsafe: a 330-atom, 19 A box can exceed 64 neighbours per atom, and every
bond-order and angle kernel would then process many irrelevant distant rows.

Use a second-tier vdW neighbour representation:

- retain the chemical neighbour table unchanged;
- construct one undirected `(i,j)` vdW edge list per box using a cell list or
  radius query and a skin;
- concatenate box edge lists with shifted indices for batched Torch execution;
- compute each vdW pair once and scatter half its energy to both atoms;
- rebuild using the existing displacement/skin principle;
- ensure no edges connect independent batch boxes; and
- expose overflow/rebuild diagnostics rather than silently dropping pairs.

This keeps production work proportional to the number of pairs within the vdW
cutoff instead of full `O(N^2)`. A dense all-pairs implementation is acceptable
only as the small-system NumPy/Torch validation oracle.

Benchmark matched workloads:

1. frozen baseline, vdW OFF;
2. vdW pair-list construction plus energy only;
3. vdW energy and autograd forces;
4. complete stepping with realistic rebuild frequency;

especially for approximately 330 atoms x 16 CUDA seeds. Report elapsed step
time, pair count, rebuild cost, peak device memory, and throughput. This is an
incremental measurement, not a broad GPU optimization project.

## Validation plan

### Mathematical and implementation tests

- NumPy scalar reference versus Torch float64 energy.
- Autograd force versus central finite differences.
- Pair symmetry and equal/opposite forces.
- Translational, rotational, and permutation invariance.
- Minimum-image equivalence across every box face.
- Energy and force immediately below/above:
  - the numerical safety radius;
  - chemical inner radius;
  - chemical outer radius;
  - 7.0 A switch onset;
  - 8.5 A cutoff.
- No NaN/Inf for coincident or extremely close validation coordinates.
- OFF mode identical to the frozen baseline and existing tests.

For every H/C/N/O pair, sample the complete combined reactive-plus-vdW curve,
its first derivative, and numerical second derivative. Automatically detect
extra minima, cusps, force sign changes, and excessive curvature.

### Isolated molecular dimers

- H2 ... H2
- CH4 ... CH4
- H2O ... H2O
- NH3 ... NH3
- CH4 ... H2O
- NH3 ... H2O

Scan centre separation and several orientations. Judge the layer on sensible
long-range attraction, approach to zero, finite short-range handling,
continuous force, reasonable UFF order of magnitude, and absence of a competing
chemical well.

Water and ammonia dimer binding must not be compared as though vdW were the
complete real interaction: electrostatics, induction and directional hydrogen
bonding remain absent. Published or quantum dimer curves may be plotted to
show the missing physics, but are not vdW-only fit targets.

### Preservation of bonded/reactive behaviour

Repeat existing core, high-fidelity, bond-calibration and H-transfer tests with
vdW OFF. With vdW ON verify:

- equilibrium bond lengths and harmonic curvature remain within declared
  tolerance;
- no new secondary bonded minima;
- stable H2, CH4, H2O and NH3 endurance;
- abstraction barrier changes are measured and attributed, not silently
  accepted; and
- bond formation/dissociation crosses the blend without impulses.

### Soup and condensed comparisons

Use identical starting coordinates, velocities and seeds for OFF/ON runs.
Compare:

- virial pressure and density response;
- radial distribution functions by element pair;
- clustering and largest-cluster distributions;
- stable-molecule encounter duration and collision frequency;
- reaction and molecule-survival statistics;
- temperature/energy drift in NVE checks;
- move-cap activity, integration failures and NaNs; and
- runtime/memory overhead.

Start with short diagnostic boxes, then the representative approximately
330-atom x 16-seed CUDA workload. Do not interpret a changed reaction yield as
an improvement until the changed collision/clustering statistics are understood.

## Known missing physics after this layer

- permanent electrostatics and charge transfer;
- induction/polarization;
- directional hydrogen bonding;
- many-body dispersion;
- environment-dependent Pauli repulsion and dispersion coefficients;
- rigorous long-range dispersion treatment;
- torsional energy;
- validated condensed-phase thermodynamics; and
- systematic double/triple-bond and broader reaction calibration.

UFF vdW will improve the zero-interaction gap between molecules, but it cannot
make polar chemistry quantitatively complete.

## Smallest safe implementation milestone

1. Add a standalone NumPy UFF pair curve, switch, and reactive suppression.
2. Add exhaustive scalar energy/force continuity tests for all ten element
   pairs.
3. Add isolated dimer scans with vdW OFF/ON and inspect for extra minima.
4. Report those results for review.

Only after that review should a disabled-by-default Torch term and separate vdW
neighbour list be added. Production vdW must not be enabled by default until
the bonded regressions, soup comparisons, stability tests, and incremental
CUDA benchmark pass.

## Option A ordering

1. generic nonbonded vdW;
2. torsional physics;
3. double/triple bond validation and justified recalibration;
4. broader reaction/collision validation;
5. stronger environment dependence and hydrogen-bond treatment;
6. return to dynamic electrostatics.

Each layer remains separately switchable and validated so its effects can be
attributed.
