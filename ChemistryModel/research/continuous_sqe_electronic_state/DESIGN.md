# Continuous SQE-derived electronic state: preregistered design

## Status and scope

This is a formulation and parameterisation design only. It does not implement
a solver, register an extension, connect electrostatics to forces or MD, or
change any existing ChemistryModel parameter.

The first model generation is restricted to neutral H/C/N/O systems and
neutral-fragment dissociation. A non-neutral system may eventually use a
declared fixed reference-charge vector `q_ref` with `sum(q_ref) = Q_total`, but
this design does not invent how net charge is partitioned. A single global KKT
constraint is deliberately excluded because it would reintroduce artificial
inter-fragment transfer.

The design combines:

- SQE's antisymmetric local transfer variables and exact component-wise charge
  conservation;
- EEQBC's topology-free, smooth distance-dependent transfer capacity idea;
- optional continuous ChemistryModel bond participation as a tested feature,
  never as a discrete bond list or the sole transfer gate.

Primary references are Nistor et al., *A generalization of the charge
equilibration method for nonmetallic materials*, JCP 125, 094108 (2006), DOI
10.1063/1.2346671; Mikulski et al., *Merging bond-order potentials with charge
equilibration*, JCP 131, 241105 (2009), DOI 10.1063/1.3271798; and Froitzheim
et al., *The bond capacity electronegativity equilibration charge model*, JCP
162, 214109 (2025), DOI 10.1063/5.0268978.

## Candidate architecture comparison

| Model/information source | State location | Localisation | Reactive smoothness | Use in this design |
|---|---|---|---|---|
| global QEq | independent atomic charges plus one global constraint | fails for separated unlike fragments | smooth but globally coupled | diagnostic comparator only |
| published SQE | antisymmetric declared-bond transfers; atomic charges derived | exact when transfer graph disconnects | fixed topology is unsuitable as published | conservation/localisation mathematics |
| EEQBC | atomic charges controlled by smooth all-pair bond capacities | strongly suppresses but does not exactly remove global residual transfer | topology-free and smooth | shape/combining-rule inspiration and frozen comparator |
| BOP/SQE | bond-centred transfer limited by continuous bond order | local and reactive within its BOP convention | potentially smooth | evidence that bond state can modulate transfer |
| unified-radial participation | continuous solved chemical-energy ownership | not itself a charge model | smooth and already available | optional C2 descriptor only |
| proposed model | local pair transfers with derived atomic charges | exact component net charge once cross compliance is zero | topology-free smooth capacity with compact support | reference candidate |

EEQBC solves the topology problem but retains a global atomic-charge solve.
Published SQE solves localisation but assumes a transfer topology. The proposed
form keeps SQE conservation while borrowing only the smooth capacity concept,
not EEQBC's global charge constraint or fitted numbers.

## 1. What the electronic state represents

The primary state lives on atom-pair transfer channels. Atomic charges are
derived observables:

```text
local antisymmetric transfer amplitudes
                  |
                  v
      atomic charge q = B p
                  |
                  v
       dipole, Coulomb response, energy
```

For an oriented candidate-pair incidence matrix `B`, every column contains one
`+1` and one `-1`. Therefore

```text
1^T B = 0
q = q_ref + B p
sum(q) = sum(q_ref) = Q_total.
```

For the neutral first generation, `q_ref = 0`. Transfer is local, bonds do not
own charge, and atoms do not independently exchange charge through one global
reservoir. A bond-centred variable is the correct conservation mechanism;
atomic charge remains the correct quantity for Coulomb energy and molecular
observables. The model is consequently a hybrid bond/atom state.

## 2. Variational energy

Let `c_e(R) >= 0` be the transfer compliance of edge `e`, `C = diag(c_e)`,
and use the scaled variable

```text
p = C^(1/2) u
q = q_ref + B C^(1/2) u.
```

The proposed electronic energy is

```text
E_el(u; R) = 1/2 u^T u
           + chi(R)^T q
           + 1/2 q^T H(R) q

H(R) = diag(kappa(R)) + J_G(R).
```

`J_G` is the complete Coulomb Gram matrix of normalized spherical Gaussian
charge densities, including its analytic self terms. `diag(kappa)` is a
strictly positive intrinsic chemical-hardness contribution. Gaussian widths,
Coulomb units and hardnesses must be fitted and validated as one convention;
the QTPIE failure showed why unrelated diagonal and off-diagonal conventions
cannot be combined.

The stationary equation is

```text
X = B C^(1/2)
A u = -X^T (chi + H q_ref)
A = I + X^T H X.
```

With positive `kappa` and the positive-semidefinite Gaussian Coulomb Gram
matrix, `H` is positive definite and hence `A` has eigenvalues at least one.
This makes convexity structural rather than an empirical hope. Every fitted
candidate must still report eigenvalue and condition-number distributions.

The solved scalar energy is differentiated through the solve. Forces include
derivatives of compliance, environment-conditioned parameters and Gaussian
Coulomb interactions. No charge lag, detached charge, or force correction is
part of the reference formulation.

## 3. Smooth transfer capacity and dissociation

The distance-only capacity is inspired by EEQBC's smooth bond-capacity law but
used as an SQE compliance:

```text
R_ij = Rcov_i + Rcov_j
x_ij = (r_ij / (rho R_ij) - 1)
f_cap = 1/2 erfc(k_cap x_ij)
c_ij^dist = sqrt(xi_i xi_j) f_cap [f_support(r_ij)]^2.
```

`Rcov` values and the `C2` or smoother compact-support interval are frozen
hyperparameters with recorded provenance. `f_support = 1` inside the inner
radius and reaches exactly zero, with zero first and second derivatives, at
the outer radius. The solver constructs `C^(1/2)` directly and uses
`f_support` as its support amplitude; this avoids differentiating a square root
at zero and gives the squared support in `c_ij`. The neighbour list must include
the complete support plus a skin, so an edge can enter or leave the list only
while its energy and required derivatives are exactly zero.

At fragment separation every cross-fragment `c_ij` becomes zero. Then `X` is
block separated and each fragment charge is a sum of antisymmetric internal
transfers. Every initially neutral fragment therefore has exactly zero net
charge. This remains true even when fragment electronegativities differ.
The Gaussian Coulomb matrix may still couple the fragments and induce physical
within-fragment polarization; incidence conservation prevents that response
from becoming net electron transfer.

The reference solver should retain zero-capacity edges only conceptually; it
may omit their zero columns without changing the energy. Warm starts and sparse
execution are backend details, not physics.

## 4. Local environment and ChemistryModel bond state

Three preregistered nested variants separate genuine information gain from
extra flexibility.

### C0: elemental, distance-only SQE

```text
chi_i = chi_Z
kappa_i = kappa_Z
c_ij = c_ij^dist.
```

This is the identifiability and stability control. It must be evaluated even
if a richer model performs better.

### C1: environment-conditioned atom state

Define a topology-free continuous coordination descriptor using a separate
fixed smooth radial function:

```text
n_i = sum_j f_CN(r_ij)
z_i = tanh(n_i - n_ref,Z)
chi_i = chi_Z + a_Z z_i
kappa_i = kappa_min + softplus(h_Z + b_Z z_i).
```

`n_ref,Z` and the descriptor shape are frozen before fitting. This lets
under-coordinated radicals, ordinary molecules and crowded contacts have
different charge response without molecule labels. Electronegativity changes
the zero-field drive; hardness changes susceptibility. Both are needed because
dipoles alone cannot distinguish them.

No direct dependence on solved charge is allowed in generation one. It would
make the state nonlinear, complicate convexity, and add a new identifiability
problem before the quadratic model is understood.

### C2: continuous bond-participation ablation

Only after C1, test whether the existing unified-radial participation
`m_ij in [0,1]` supplies transferable information beyond distance:

```text
c_ij = c_ij^dist [(1 - alpha) + alpha m_ij]^2,
0 <= alpha <= 1.
```

`alpha = 0` recovers C1. This is never a binary topology decision, and the
distance support still guarantees localisation. A nonzero distance channel is
retained because polarization/charge transfer may begin before a chemical bond
has appreciable occupancy.

C2 is rejected if bond participation introduces force roughness, circular
state ownership, worse dissociation/contact transferability, or no blocked-
family validation gain. ChemistryModel's bond state is valuable evidence, but
previous geometry research already showed that one scalar participation need
not represent every electronic degree of freedom.

## 5. Parameter list

All parameters are global elemental or global scalar quantities. There are no
molecule, reaction, formal-bond, spin-state or benchmark-specific parameters.

| Family | C0 | C1 | C2 | Constraint / role |
|---|---:|---:|---:|---|
| elemental electronegativity `chi_Z` | 3 | 3 | 3 | four values with one additive gauge fixed |
| intrinsic hardness raw value `h_Z` | 4 | 4 | 4 | transformed to `kappa >= kappa_min` |
| Gaussian width `sigma_Z` | 4 | 4 | 4 | positive, matched to the Coulomb convention |
| elemental capacity `xi_Z` | 4 | 4 | 4 | positive; pair amplitude is geometric mean |
| capacity range scale `rho` | 1 | 1 | 1 | positive global scalar |
| capacity steepness `k_cap` | 1 | 1 | 1 | positive global scalar |
| electronegativity environment response `a_Z` | 0 | 4 | 4 | signed |
| hardness environment response `b_Z` | 0 | 4 | 4 | bounded through positive hardness transform |
| bond-participation blend `alpha` | 0 | 0 | 1 | `[0,1]`; C2 only |
| **independent fitted parameters** | **17** | **25** | **26** | |

Fixed, versioned inputs are the Coulomb constant, units, Gaussian definition,
covalent radii, `f_CN`, `n_ref`, `kappa_min`, and compact-support interval.
Changing any of them defines a new model generation.

Pair-specific capacities or decay exponents are prohibited initially. Adding
ten pair corrections could hide a defective combining rule and would be
considered only after leave-one-pair-out evidence identifies a reproducible
pair-specific failure.

## 6. Identifiability analysis

The important parameter confounders and separating observations are:

| Potential confounding | Separating evidence |
|---|---|
| additive `chi` gauge | fix `chi_H = 0`; only differences are physical |
| `Delta chi` versus capacity `xi` | zero-field dipoles constrain their product; finite-field response and polarizability constrain susceptibility independently |
| hardness `kappa` versus Gaussian width `sigma` | molecular response plus compressed geometries and external ESP distinguish local hardness from spatial Coulomb screening |
| capacity amplitude `xi` versus range `rho/k_cap` | dense separation and approach scans locate onset/decay; equilibrium data alone are insufficient |
| `a_Z` versus baseline `chi_Z` | same element sampled across multiple coordination/radical environments, with molecule-family blocking |
| `b_Z` versus baseline hardness | finite-field response at multiple coordinations, not only static charges |
| `alpha` versus distance capacity | matched geometries having similar distances but different unified bond participation; profile likelihood must exclude an unconstrained blend |

Homonuclear molecules cannot identify electronegativity differences, but they
are essential zero-dipole, response, width and capacity controls. Atomic charge
partitions alone cannot identify the model: MBIS is guidance, not truth.

For each fitted variant report:

- family-blocked Jacobian rank and singular values;
- parameter correlation and profile likelihoods;
- bootstrap intervals resampling molecule families, not individual points;
- at least 20 dispersed initialisations;
- leave-one-element-pair-out sensitivity;
- C0/C1/C2 likelihood and validation comparison with complexity penalty;
- prediction sensitivity to MBIS versus a DDEC6 audit subset.

A weakly identified parameter is removed or frozen in a newly declared model;
it is not retained because it improves training loss.

## 7. Architecture handoff for a future implementation

The future term should preserve the existing boundary:

```text
ChemistryEngine
  -> Hamiltonian
    -> ElectronicStateEnergyTerm.energy(context, current_energy)
```

It should return one scalar contribution and diagnostics through
`EnergyResult.state`, including atomic charges, nonzero split charges,
compliances, dipole, energy components, residual, eigenvalue bounds and
condition number. C0/C1 require only positions, elements, box assignments and
neighbour support from `InteractionContext`.

C2 requires an explicit differentiable state dependency. It must not reach
into simulation private fields. The canonical Hamiltonian would expose
unified participation through a documented context/result channel before the
extension is evaluated. If that clean dependency cannot be provided without
duplicating or detaching the capacity solve, C2 is not implemented.

Independent simulation boxes are solved independently. The extension remains
opt-in, and `enabled_extensions=()` must remain bitwise/numerically equivalent
to frozen unified radial.

## Decision before implementation

The recommended first reference implementation is **C0 followed by C1**.
C2 is a preregistered ablation, not the assumed final answer. This ordering
isolates whether environment-conditioned electronic response is necessary and
whether ChemistryModel bond participation contains extra transferable signal.

The design is suitable to proceed to a small quantum-data pilot, but not yet
to production implementation or MD coupling.
