# Reactive/nonbonded energy-partition investigation

## Decision summary

Nothing in ChemistryModel's integrator, autograd use, neighbour-table layout,
or continuous-reactive premise fundamentally prevents normal vdW physics. The
failure is narrower: the present reactive Morse term and a generic vdW core
both represent close-range repulsion, and the attempted handoff transfers full
responsibility to vdW while that vdW curve is still strongly repulsive.

No tested standalone partition is ready for Torch parity. The physically clean
next step is to define the missing bonded/nonbonded energy responsibility, not
to tune another switch.

## Current engine audit

The NumPy reference constructs a full pair matrix and calculates, in order:

1. pair vectors and distances;
2. a pair-specific cosine contact taper from `1.25 r_e` to `1.60 r_e`;
3. coordination from the sum of those contacts;
4. continuous bond order and order-dependent Morse parameters;
5. tapered Morse pair energy;
6. over-coordination and angle/environment energies using the same taper.

The Torch engine computes the same state over a padded neighbour table. It
already has two partial sharing mechanisms:

- `_chemical_pair_cache` exposes detached distance/contact diagnostics;
- `_reactive_intermediates` exposes differentiable pair state to the optional
  high-fidelity correction when requested.

Thus there are not several contradictory production definitions of chemical
contact. There are, however, different *representations* of the same state
(dense NumPy matrices, padded Torch neighbours, detached diagnostics), and no
small public contract for a new differentiable energy consumer.

The production neighbour table is a computational candidate list, not a bond
definition. The contact taper is continuous reactive character. Bond order is
an environment-dependent refinement of that contact. Treating any one of
these as a permanent bonded/nonbonded label would lose the model's intended
reactivity.

## Physical overlap

ChemistryModel's pair term is

    E_reactive = c(r) D [exp(-2 a (r-r_e)) - 2 exp(-a (r-r_e))].

It already supplies chemical attraction and exponential close-range
repulsion. Coordination and angle terms add environment-dependent resistance,
but do not replace that pair repulsion.

The UFF, shielded ReaxFF-style, and AIREBO-M curves each contain both an outer
attractive well and a generic repulsive core. Adding a complete such curve
therefore duplicates close-range repulsion. Long-range dispersion attraction
is the clearly missing component; how much residual closed-shell repulsion is
missing has not been established independently.

## Why whole-potential suppression fails

The contact complement reaches one at the reactive cutoff. For every tested
pair that cutoff lies well inside the zero crossing of the UFF-matched vdW
curve. Consequently the handoff insists that the full positive vdW core is
already active just as the negative reactive energy reaches zero. The product
can be perfectly differentiable while still creating a positive local maximum.
Shielding makes that maximum smaller; it cannot change this responsibility
error.

## Established reactive-force-field partitioning

### ReaxFF

The original ReaxFF energy expression applies shielded vdW interactions to all
pairs, bonded and nonbonded, because discrete bonded exclusions would become
discontinuous during reactions. Bond, coordination, angle and related terms
are bond-order dependent, while vdW uses its own shielded distance and a
long-range taper. These terms coexist rather than using ChemistryModel's
whole-potential contact complement. Crucially, ReaxFF's bonded and vdW
parameters are fitted as one force field; adding only its vdW architecture to
an independently calibrated Morse bond term does not preserve that balance.

Source: van Duin et al., *J. Phys. Chem. A* **105**, 9396 (2001),
[doi:10.1021/jp004368u](https://doi.org/10.1021/jp004368u).

### AIREBO/AIREBO-M

AIREBO adds nonbonded energy to REBO but does not use a single radial
complement. Its adaptive term combines a distance switch, a hypothetical bond
order evaluated for the candidate pair, and a connectivity factor. First- and
second-neighbour interactions are excluded and 1-4 responsibility is assigned
to torsions, with continuous bond weights allowing those relationships to
change. This is environment-aware, but its connectivity machinery is much
closer to transient topology than ChemistryModel currently wants and is not a
drop-in generic H/C/N/O solution. AIREBO-M changes the singular LJ curve to
Morse; it retains the surrounding AIREBO partition logic.

Sources: Stuart, Tutein, and Harrison, *J. Chem. Phys.* **112**, 6472 (2000),
[original AIREBO paper](https://www.usna.edu/Users/chemistry/jah/_files/Papers/AIREBO_JCPversion.pdf);
O'Connor, Andzelm, and Robbins, *J. Chem. Phys.* **142**, 024903 (2015),
[doi:10.1063/1.4905549](https://doi.org/10.1063/1.4905549).

## Standalone partition experiments

All candidates use the same UFF-matched ReaxFF-style shielded curve and the
same long-range cutoff. No production code or parameters are involved.

- **A, whole suppressed:** rejected baseline `g(contact) E_vdw`.
- **B, all-pairs shielded:** published ReaxFF architectural idea, without
  pretending the independently calibrated parameters are jointly fitted.
- **C1, split WCA:** exact WCA-style algebraic separation at the vdW minimum;
  the attractive component persists while only residual repulsion is contact
  suppressed.
- **C2, dispersion-only diagnostic:** retain only the WCA attractive component
  to isolate what happens if ChemistryModel owns all short-range repulsion.
- **D, shared-contact linear split:** same C1 components, but reuse `1-contact`
  directly. This explicitly tests—and rejects—the tempting assumption that
  sharing a descriptor alone makes the partition physical.

### Transition maxima and chemical-minimum shifts

| pair | A barrier (eV) | B barrier (eV) | C1 barrier (eV) | D barrier (eV) | B minimum shift (A) |
|---|---:|---:|---:|---:|---:|
| H-H | 0.2682 | 0.2682 | 0.2682 | 0.2681 | 0.05269 |
| C-H | 0.1598 | 0.1598 | 0.1598 | 0.1597 | 0.06075 |
| O-H | 0.1702 | 0.1702 | 0.1702 | 0.1701 | 0.03252 |
| C-C | 0.0795 | 0.0795 | 0.0795 | 0.0794 | 0.03529 |
| C-O | 0.0681 | 0.0681 | 0.0681 | 0.0681 | 0.02697 |
| N-N | 0.0534 | 0.0534 | 0.0534 | 0.0533 | 0.01941 |
| O-O | 0.0288 | 0.0288 | 0.0288 | 0.0287 | 0.02773 |

The maxima are effectively identical because all architectures restore the
same repulsive curve by the outer edge of chemical contact. B additionally
shifts every chemical minimum, demonstrating double counting in the current
independently calibrated potential.

C2 is the only numerical curve with no positive transition maximum and no
chemical-minimum shift. It is nevertheless not acceptable: the WCA attractive
component is constant from the chemical region to `r_min`. It therefore
creates a force-free attractive shelf instead of a well that smoothly draws
nonbonded atoms into reactive contact. Removing repulsion alone avoids the
symptom but does not supply a defensible association surface.

All candidates are finite and continuous. Maximum analytic-versus-numerical
force disagreement is `2.4e-9 eV/A` (C2; all others below `6.2e-10 eV/A`). The
intended outer UFF-matched depths and positions remain unchanged where an
isolated outer minimum exists.

## Continuous-reactive difficulty

A fixed-topology force field can simply exclude 1-2/1-3 pairs and assign 1-4
scaling. ChemistryModel cannot: those relationships must emerge and disappear
smoothly. It therefore needs either:

1. a nonbonded term whose short-range limit was calibrated jointly with its
   reactive term, as in ReaxFF; or
2. a physically derived, smoothly damped dispersion-only contribution whose
   overlap with the existing Morse energy is explicitly defined and validated.

Neither follows merely from `1 - contact` or `1 - bond_order`. The damping or
joint decomposition remains model-specific ChemistryModel physics and will
require independent reference surfaces, not curve cosmetics.

## Minimal shared interaction context

A shared context would materially simplify future terms and prevent redundant
distance/contact calculations, but it does not solve the vdW science by
itself. The smallest incremental interface is a lightweight internal value
object produced inside the existing energy pass:

    PairGeometry:
        neighbours / pair indices, mask, vectors, distances, element pairs

    ReactiveState:
        contact taper, continuous bond order, coordination,
        order-blended pair parameters

New energy consumers should receive these tensor views without detaching them,
so their contribution remains in the same autograd graph. Diagnostics can
request a detached projection. NumPy may retain dense matrices and Torch its
padded layout behind the interface. Existing equations should remain inline
initially; the already-working `_reactive_intermediates` path proves that a
consumer can reuse the state without a broad rewrite.

Suggested migration:

1. replace the private tuple used by high-fidelity code with a named internal
   context carrying the same tensors;
2. verify bitwise/close parity with the current high-fidelity and batched paths;
3. let the next standalone vdW Torch prototype consume that context;
4. migrate existing energy terms only if profiling or reuse justifies it.

This is an architectural hygiene improvement, not the next scientific answer.
The underlying bonded/nonbonded decomposition must be resolved before Torch
parity work is worthwhile.

## Final recommendation

REVISE energy partition further
