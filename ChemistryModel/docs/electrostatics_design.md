# Dynamic charge and electrostatics design

## Decision

ChemistryModel should investigate an **atom-space QTPIE fluctuating-charge
model with a shielded hardness matrix**. It should not put ordinary QEq into
production dynamics.

QEq is useful as a reference implementation because its constrained quadratic
energy is simple and well established, but it leaves finite charge transfer
between unlike atoms at infinite separation. That is a decisive defect for a
reactive model. QTPIE replaces fixed atomic electronegativities with
overlap-weighted, geometry-dependent effective electronegativities, so charge
transfer vanishes as contacts dissociate without requiring a fixed molecular
topology. Chen and Martinez also showed that the original bond-space method
has an exactly equivalent atom-space formulation, avoiding its redundant,
rank-deficient O(N^2) transfer-variable system.

ACKS2 is the strongest alternative to retain as a research comparator. It
addresses both dissociation and the superlinear polarizability scaling of EEM,
with only a modest formal increase in cost. It is not the first implementation
choice because its atom-in-molecule response parameters form a different
convention and ChemistryModel does not yet have a compatible H/C/N/O parameter
set. Importing unrelated ReaxFF or EEM numbers into ACKS2 would not be valid.

Fixed partial charges are rejected for the reactive core: they require
topology/environment assignments that become ambiguous as bonds form and
break, and they cannot represent charge redistribution during a collision.

Primary references:

- [Rappe and Goddard, original QEq](https://doi.org/10.1021/j100161a070)
- [Chen and Martinez, QTPIE and correct dissociation asymptotics](https://arxiv.org/pdf/0807.2068)
- [Chen and Martinez, exact atom-space reformulation](https://arxiv.org/pdf/0807.2174)
- [Verstraelen et al., ACKS2](https://doi.org/10.1063/1.4791569)
- [van Duin et al., reactive force field and shielded nonbond terms](https://doi.org/10.1021/jp004368u)
- [LAMMPS QTPIE formulation and parameter units](https://docs.lammps.org/fix_qtpie_reaxff.html)

## Proposed isolated model

Positions are in angstrom, charges `q_i` in elementary-charge units, and
energies in eV. For atom types `t_i`, define absolute electronegativity
`chi_i` (eV), self hardness `eta_i` (eV), a short-range shielding parameter
`gamma_i` (inverse angstrom under the selected shield convention), and a
normalized Gaussian overlap exponent `alpha_i` (inverse Bohr squared).

The normalized-orbital overlap `S_ij(R_ij)` gives the QTPIE effective
electronegativity

```
chi_tilde_i = sum_j [(chi_i - chi_j) S_ij] / sum_m S_im.
```

The atom-space energy to minimize is

```
E_charge(q; R) = chi_tilde(R)^T q + 1/2 q^T H(R) q
sum_i q_i = Q_requested
```

with

```
H_ii = eta_i
H_ij = J_ij(R_ij), i != j.
```

`J_ij` is a shielded Coulomb interaction in eV per squared elementary charge.
The first reference implementation should use the selected QTPIE/QEq orbital
integral convention exactly. A ReaxFF-style algebraic shield is a separate
candidate, not an interchangeable formula:

```
J_ij(r) = k_e / [r^3 + gamma_ij^(-3)]^(1/3).
```

Whichever form is selected after the parameter audit must be used consistently
in both the hardness matrix and the reported pair electrostatic energy.

The constrained stationary point is obtained from the KKT system

```
[ H   1 ] [ q      ] = [ -chi_tilde ]
[ 1^T 0 ] [ lambda ]   [ Q_requested ]
```

where `lambda` is the charge chemical potential. The initial isolated solver
will use a float64 symmetric dense solve with residual and condition-number
reporting. It must reject non-finite, singular, poorly conditioned, or
constraint-violating solutions rather than returning fallback charges.

Standard QTPIE transfer variables conserve a neutral starting charge. The KKT
form above generalizes the atom-space solve to requested nonzero total charge,
but charged-fragment dissociation is a mandatory validation case: a single
global constraint can still put charge on the wrong separated fragment. No
charged periodic production run is approved until that behaviour is resolved,
potentially through explicit fragment/electrode constraints in a later design.

## Energy decomposition and forces

The engine decomposition will remain explicit:

```
E_total = E_existing_reactive
        + E_charge_self
        + E_electrostatic_pair

E_charge_self = sum_i [chi_tilde_i q_i + 1/2 eta_i q_i^2]
E_electrostatic_pair = sum_(i<j) q_i q_j J_ij.
```

The two new terms must be individually observable. They must not be folded
into Morse depths, bond order, coordination, angles, environment softening, or
the H-transfer correction.

At a fully converged variational charge minimum, the envelope theorem permits
forces from the partial position derivative of the minimized energy. The
reference path will nevertheless differentiate the complete solve and compare
with central finite differences. Tests must cover energy/force continuity,
charge residual sensitivity, and the effect of deliberately incomplete solves.
Skipped charge updates do not follow the same Born-Oppenheimer surface and may
damage NVE conservation; therefore **every-step equilibration is the scientific
default** until cadence experiments demonstrate otherwise.

## Short range and boundaries

Bare `q_i q_j/r` is unsuitable at reactive contact distances. The first solver
uses the same finite shield in `H_ij` and the pair energy. It will test `r -> 0`,
normal bond distances, bond stretching, and smooth geometry perturbations.

Development proceeds in this order:

1. isolated/non-periodic molecules with no cutoff;
2. isolated fragment-separation tests;
3. periodic neutral boxes using a minimum-image, damped shifted-force candidate
   whose potential and first derivative both reach zero at a cutoff below half
   the smallest box length;
4. comparison of that approximation against an Ewald reference before soup
   claims are made.

The damped shifted-force method is a prototype, not assumed truth. It gives
continuous energies and forces and maps naturally to pairwise GPU kernels, but
published tests show that accuracy can worsen as polarity increases. PME/Ewald
therefore remains the reference for periodic validation rather than an
immediate implementation requirement.

References:

- [Fennell/Gezelter-style DSF discussion and energy-force consistency](https://pmc.ncbi.nlm.nih.gov/articles/PMC4636498/)
- [GPU Wolf assessment and polarity limitation](https://doi.org/10.1039/C4FD00012A)

## Locked standalone reference convention and parameter audit

The standalone reference is locked to Chen, Hundertmark and Martinez,
*The dissociation catastrophe in fluctuating-charge models and its
implications for the concept of atomic electronegativity* (2008), together
with their exact atom-space reformulation. It uses equations 22 and 28--30,
the primitive s-Gaussian Coulomb integral in equation 33, and the fitted
Gaussian exponents in Table 1. This is not the newer LAMMPS/ReaxFF shield.

Equation mapping in `electrostatics.py` is:

- normalized primitive-orbital overlap -> `gaussian_overlap_matrix`;
- equation 30 -> `effective_electronegativity`;
- equation 33 generalized to unlike Gaussian exponents ->
  `hardness_matrix`, with `J_ii` replaced by the published atomic hardness;
- equations 28 and 37 -> the float64 KKT system in `solve_charges`;
- plain QEq comparator -> the same hardness/KKT system with bare `chi_i`.

The unlike-Gaussian result follows by the Gaussian product theorem. For
normalized primitive orbitals `phi_i ~ exp(-alpha_i r^2)`, their charge
densities have exponents `2 alpha_i`, hence

```
beta_ij = 2 alpha_i alpha_j / (alpha_i + alpha_j)
J_ij(R) = erf(sqrt(beta_ij) R) / R       (atomic units)
```

and the equal-exponent limit is exactly published equation 33. Positions are
converted from angstrom to Bohr and Hartree to eV exactly once.

| element | chi (eV) | full J_ii (eV) | Slater exponent | Gaussian alpha (Bohr^-2) | provenance |
|---|---:|---:|---:|---:|---|
| H | 4.528 | 13.890 | 1.0698 | 0.5434 | QEq values reused by QTPIE; Chen et al. Table 1 |
| C | 5.343 | 10.126 | 0.8563 | 0.2069 | same |
| N | 6.899 | 11.760 | 0.9089 | 0.2214 | same |
| O | 8.741 | 13.364 | 0.9745 | 0.2240 | same |

No value in this table was fitted to ChemistryModel. The Slater column is
recorded for provenance but is not used by the Gaussian reference.

The compatibility audit subsequently identified an extra factor of two inside
the Gaussian Coulomb error-function argument in this first implementation.
That error, retained as formulation A for regression evidence, makes the water
hardness matrix indefinite and causes the unphysical +6.52 e oxygen result.
The corrected published Gaussian mapping restores water polarity, but is only
a diagnostic pending selection of a complete production parameter convention.
See `docs/qtpie_compatibility_report.md`.

Within its well-conditioned dissociation interval, the exact convention is
continuous and shows the intended QTPIE asymptotic result: neutral H and O
fragments lose charge transfer, while the otherwise-identical QEq comparator
retains fractional charge. Molecular-fragment claims and all charged-fragment
claims remain out of scope for a single global KKT constraint.

A production-compatible QTPIE parameter set would require, as one
convention-matched unit:

- `chi`: absolute electronegativity, eV;
- `eta`: full diagonal self-Coulomb hardness, eV (not the half-hardness used by
  some ReaxFF files);
- `gamma` or orbital radius/exponent for the chosen shield;
- `alpha`: Gaussian overlap exponent, inverse Bohr squared;
- optional pair overlap scale `k_ij`, initially fixed at one and fitted only if
  a declared fit/hold-out split justifies it.

The current LAMMPS QTPIE implementation separately combines these overlaps
with ReaxFF-style `gamma` shielding. That is a different convention and is not
silently substituted here. If a fit is later authorized, proposed fit targets
are H2O, NH3, CO and CH2O dipoles plus
small geometry-response curves; CH3OH, CH3NH2, H2O2 and NH2OH remain hold-outs.

## Validation plan

The isolated solver API will accept elements, positions, boundary/box,
requested total charge, parameter set and tolerances, and return charges,
energy decomposition, charge error, residual, condition estimate, convergence
state and iteration count.

Required invariants:

- exact total-charge conservation within declared tolerance;
- deterministic output;
- permutation, translation and rotation invariance;
- equivalent charges on symmetry-equivalent atoms;
- continuous charge, energy and force under small geometry changes;
- finite short-range behaviour;
- vanishing spurious charge transfer between separated neutral fragments;
- explicit failure on a singular or ill-conditioned solve.

Molecules: H2, CH4, NH3, H2O, CO, CO2, CH2O, CH3OH, CH3NH2, H2O2 and NH2OH.
Qualitative signs and symmetry are sanity checks, not charge fit targets.
Dipoles use

```
mu = sum_i q_i (R_i - R_origin)
1 e Angstrom = 4.803204712 Debye.
```

Neutral-molecule dipoles are origin independent. Charged-system dipoles must
state their origin. NIST CCCBDB supplies experimental comparisons including
NH3 1.476 D, CH2O 2.332 D and H2O2 1.770 D; source and molecular geometry/state
will be stored with each target rather than silently mixing equilibrium and
vibrationally averaged values.

Reference: [NIST CCCBDB experimental dipole list](https://cccbdb.nist.gov/diplistx.asp).

## Architecture and performance

The implementation belongs in a dedicated module, initially
`electrostatics.py`, with NumPy float64 reference code. A later
`electrostatics_torch.py` will provide batched production operations. The OFF
path will return exactly zero new energy and preserve the frozen baseline
bit-for-bit.

`ReactiveSimulation.energy_per_atom` is already the differentiable energy
boundary and `compute_forces` obtains its gradient with Torch autograd. The
eventual ON path can add per-atom charge energy there without changing the
integrator order. Batched runs concatenate independent seed boxes, so charge
constraints and solves must be segmented per seed; a single constraint over
the concatenated tensor would permit unphysical charge transfer between runs.

A dense solve is O(N^3) and is only acceptable for the small reference set.
The intended 300-700 atom path is a batched, warm-started iterative constrained
solve (matrix-free where useful), with symmetric pair construction reusable by
the Coulomb calculation. Expected cost cannot be stated credibly before the
parameterized condition numbers are known. The gate is a benchmark reporting
solver ms/step, pair energy ms/step, iterations, VRAM and total slowdown for
330 atoms x 16 CUDA seeds. Every-step, 2-, 4- and 8-step cadence tests will
compare charge, energy, force, NVE and reaction outcomes; speed alone cannot
select cadence.

## Smallest useful milestone

1. Add the standalone float64 atom-space QTPIE solver with no MD coupling.
2. Add a convention-locked parameter file only after provenance review.
3. Pass invariance, charge conservation, symmetry, continuity, condition and
   neutral-fragment dissociation tests.
4. Report charges and isolated dipoles for the molecular validation set.
5. Compare QTPIE against plain QEq on stretched H2O and separated polar
   fragments to demonstrate why the selected model matters.

Only after that evidence is satisfactory should an electrostatic energy be
connected to forces. Periodic soup dynamics, cadence optimisation and bonded
parameter rebalance are explicitly outside this first milestone.
