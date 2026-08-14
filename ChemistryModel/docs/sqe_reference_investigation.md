# SQE reference investigation: mathematics, topology, and parameter gate

## Outcome

The published split-charge equilibration (SQE) model has been reproduced at
the equation level. Its dissociation mechanism is materially different from
QEq and is suitable in principle: charge is transferred only through declared
bond channels, and the transfer hardness of a breaking bond tends to infinity.
When the transfer graph separates, each neutral connected component therefore
retains its own charge and long-range inter-fragment transfer vanishes.

The parameter gate does **not** permit the requested H/C/N/O solver or
validation set. The original transferable fit by Nistor et al. publishes a
single internally compatible convention for Si/C/O/H, including atomic and
bond-hardness parameters, but contains no nitrogen. No matching published
values were found for atomic N or the H-N, C-N, N-N, and N-O transfer
hardnesses. Later peptide and materials SQE parameterizations are
topology/atom-type-specific fits and cannot be mixed into the original
elemental convention.

Consequently no SQE numbers are invented, no partial H/C/O-only result is
presented as an H/C/N/O candidate, and no standalone solver is implemented.
The requested decision is:

**SQE mathematics is promising but requires a new parameter-generation
project.**

No electrostatics were connected to MD or forces. `ReactiveSimulation`, the
bonded force field, integrator, and CUDA paths were not changed.

## Published SQE convention

Primary source: Nistor, Polihronov, Muser and Mosey, *A generalization of the
charge equilibration method for nonmetallic materials*, J. Chem. Phys. **125**,
094108 (2006), DOI 10.1063/1.2346671. The interpretation of bond hardness and
dielectric response was checked against Nistor and Muser, *Dielectric
properties of solids in regular and split-charge equilibration formalisms*
(2009).

The reported molecular parameter convention uses elementary charge for charge
and electron-volts for the energy coefficients: electronegativity is in eV/e,
and atomic and transfer hardnesses are in eV/e^2. Distances enter the selected
Coulomb interaction and any distance dependence assigned to transfer
hardness; those choices must remain part of one parameter convention.

### Variables

- `q_ij = -q_ji`: split charge transferred from atom `j` to atom `i` along
  one allowed bond/transfer channel.
- `Q_i = sum_j q_ij`: net atomic charge for the neutral reference model.
- `chi_i`: atomic electronegativity.
- `kappa_i`: atomic hardness.
- `kappa_s,ij`: bond or transfer hardness for channel `ij`.
- `J_ij`: Coulomb interaction between atomic charges, including whatever
  short-range screening convention belongs to the fitted model.

With an arbitrary orientation assigned to every allowed edge, let `p` be the
vector of independent split charges and `T` the atom-edge incidence matrix.
Then

```
Q = T p
1^T Q = 1^T T p = 0
```

so neutrality is exact by construction. This first investigation is restricted
to neutral systems and neutral-fragment dissociation. Charged references need
an explicit reference-charge/oxidation-state extension and are not justified
by this neutral model alone.

### Energy functional

The default SQE energy is

```
E = sum_(ij in bonds) 1/2 kappa_s,ij q_ij^2
  + sum_i [chi_i Q_i + 1/2 kappa_i Q_i^2]
  + V_C(Q, R)
```

For a quadratic Coulomb model, define `H_ii = kappa_i` and put the Coulomb
kernel in the off-diagonal entries of `H`. Then

```
E(p) = chi^T T p + 1/2 p^T K_s p
     + 1/2 (T p)^T H (T p)
```

and stationarity is

```
[K_s + T^T H T] p = -T^T chi.
```

The atomic charges follow from `Q = T p`. A faithful reference solver must
report the split charges as well as atomic charges, the atomic,
transfer-hardness and Coulomb energy components, `sum(Q)`, the stationarity
residual, the condition number, and the smallest eigenvalue of
`K_s + T^T H T` after removing redundant cycle directions if necessary.
Merely obtaining a solution from a linear solver is not a stability test.

At stationarity, the difference in atomic electrochemical potentials across
an active channel is balanced by its transfer-hardness penalty. Unlike QEq,
SQE does not impose unrestricted global chemical-potential equalization.

## Why SQE has the correct dissociation limit

QEq optimizes one independent charge per atom under only a global total-charge
constraint. Even when two fragments are infinitely separated, different
fragment electronegativities can therefore be lowered by a finite transfer of
charge; only their Coulomb interaction disappears.

SQE permits transfer only through its edge variables. For a breaking channel
`ij`, the original construction makes

```
kappa_s,ij(r_ij) -> infinity as r_ij -> infinity.
```

The stationary split charge consequently satisfies `q_ij -> 0`. Once all
cross-fragment channels have zero transfer compliance (`1/kappa_s = 0`), the
incidence matrix is block diagonal and the total charge of every connected
component is independently conserved. Two initially neutral fragments remain
neutral regardless of their electronegativity difference.

This result is not produced by distance alone. Published SQE relies on all of:

1. a molecular/neighbour topology identifying permitted transfer channels;
2. explicit bond-edge split-charge variables;
3. infinite hardness (or omission) for nonbonded pairs; and
4. a bond hardness that diverges as a permitted bond dissociates.

A fixed finite bond hardness would retain residual transfer. A hard change in
the edge list would give discontinuous charges. These details are the central
architectural issue for a topology-changing model.

## Parameter provenance audit

The original paper compares multiple fitting strategies. The complete
variable-charge SQE convention (its method III) jointly fits the following
atomic and bond-hardness values. These values must be treated as a set, not as
an invitation to insert unrelated QEq or force-field values.

### Atomic parameters (method III)

| Element | `chi` (eV/e) | `kappa` (eV/e^2) | Classification |
|---|---:|---:|---|
| H  | 5.0780 | 16.1954 | fitted atomic parameter |
| C  | 5.2086 | 8.1313  | fitted atomic parameter |
| O  | 8.5220 | 12.4062 | fitted atomic parameter |
| Si | 4.3850 | 6.7348  | fitted atomic parameter |
| N  | unavailable | unavailable | provenance gate failure |

### Pair transfer hardnesses (method III)

| Pair | `kappa_s` (eV/e^2) | Classification |
|---|---:|---|
| H-C   | 1.2698 | fitted pair/bond parameter |
| H-O   | 0.0627 | fitted pair/bond parameter |
| H-Si  | 2.1629 | fitted pair/bond parameter |
| C-C   | 1.4719 | fitted pair/bond parameter |
| C-O   | 4.9727 | fitted pair/bond parameter |
| C-Si  | 2.7455 | fitted pair/bond parameter |
| O-Si  | 4.0194 | fitted pair/bond parameter |
| Si-Si | 4.9988 | fitted pair/bond parameter |
| H-N, C-N, N-N, N-O | unavailable | provenance gate failure |

The elemental identity selects `chi` and `kappa`; the declared covalent bond
type selects `kappa_s`. The topology and the distance law used to turn off a
breaking channel are also model inputs. They are not universal elemental
parameters and cannot be silently inferred from ChemistryModel bond order.

The paper's Si/C/O/H scope is explicit. Its parameter table cannot evaluate
NH3, CH3NH2, NH2OH, or any other required N-containing molecule. Peptide SQE
work uses molecular topology and chemistry-specific atom types fitted to
peptide charge targets. MOF and other materials parameterizations likewise
define different types, objectives, and topology assumptions. Taking an N
value from one of these while retaining the original method-III H/C/O values
would create an unpublished hybrid convention.

## Requested validation: stopped at the gate

| Requested evidence | SQE status |
|---|---|
| H2, CH4, H2O, CO, CO2, CH2O, CH3OH, H2O2 | parameters partly available, but not run as a substitute for the required complete convention |
| NH3, CH3NH2, NH2OH | blocked by missing N and N-pair parameters |
| symmetry, dipoles, continuity, conditioning | blocked for the complete validation set |
| separated H/O scan | mathematically supported, but no selectively reported SQE numerical result |
| multi-atom fragment separation | mathematically supported, but full H/C/N/O comparison blocked |
| QEq/QTPIE/SQE numeric comparison | existing QEq/QTPIE references retained; SQE column blocked |

Stopping here prevents an incomplete subset from being mistaken for a viable
production convention. Existing QEq, historical Slater QTPIE, and corrected
Gaussian QTPIE implementations remain unchanged alongside this report.

## Reactive-topology feasibility (analysis only)

Original SQE is a topology-based model. A ChemistryModel application cannot
use permanent molecule definitions, so it would need a continuously varying
transfer network. One possible mathematical direction is to define a
nonnegative transfer compliance

```
s_ij(R) = 1 / kappa_s,ij(R)
```

for candidate atom pairs. `s_ij` would approach zero smoothly as contact or
bond character disappears and remain finite in a bonded environment. Using an
all-pairs incidence representation with exactly zero or negligibly small
smooth compliance would avoid a chemically classified graph flip; the
incidence construction would continue to conserve total charge exactly. When
all cross-fragment compliances vanish, separated fragments recover independent
charge conservation.

Potential future signals include distance, continuous bond order, or contact
strength, but none can be selected merely because it already exists in
ChemistryModel. A defensible model would require:

- a smooth transfer-compliance law and smooth derivatives;
- proof or numerical evidence of the neutral-fragment limit;
- no hard neighbour-list discontinuity in charge or energy;
- stable conditioning as channels vanish and appear;
- a defined treatment of cycles and redundant split-charge variables;
- H/C/N/O atomic and pair parameters fitted under that same law;
- independent molecular polarity, response, and dissociation validation.

The incidence-variable energy, exact conservation law, and the principle that
vanishing compliance localizes charge are published SQE. Replacing its fixed
covalent graph and fitted bond-type hardnesses with ChemistryModel distance,
bond-order, or contact functions would be a **new ChemistryModel-specific SQE
approximation**. It must be named and validated as such. No such generalization
is implemented in this investigation.

## Comparison and decision

| Model | Current conclusion |
|---|---|
| QEq | rejected as production foundation: finite long-range charge transfer |
| QTPIE | useful dissociation-correct reference; current complete conventions are unstable or ill-conditioned for key molecules |
| ACKS2 | mathematically promising; blocked by complete public H/C/N/O parameter provenance |
| SQE | correct localization mechanism; original convention is topology-bound and lacks N/N-pair parameters |

**Decision: SQE mathematics is promising but requires a new
parameter-generation project.** A future project would have to generate and
validate nitrogen plus all needed transfer-hardness pairs and, separately,
derive a smooth reactive transfer law. Neither task is a lookup or a minor
implementation step, so both remain outside this reference investigation.

