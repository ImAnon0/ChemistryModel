# ACKS2 reference-model investigation: equation and parameter gate

## Outcome

The equation audit is complete, but the parameter gate does **not** permit a
scientific H/C/N/O solver comparison yet. The original ACKS2 model does not
publish a transferable elemental parameter table. Its parameters are
atoms-in-molecules (AIM) expectation values derived for a particular molecular
reference density. The modern LAMMPS `acks2/reaxff` implementation is a
separate empirical approximation whose extra response parameters are fitted
as part of a specific ReaxFF force field.

LAMMPS ships one explicit ACKS2 force field, `acks2_ff.water`, containing H and
O only. No complete, public, internally compatible H/C/N/O ACKS2 parameter set
was found in the original paper, its public supporting material, LAMMPS's
potential collection, or LAMMPS's ACKS2 examples. Consequently, no values are
invented for C or N and no ACKS2 dipole, stability, dissociation or geometry
comparison is reported for the requested molecular set.

This is a parameter/provenance failure, not evidence that ACKS2's mathematics
fails. The requested decision is therefore **another model required**, unless
a complete ACKS2 parameter source can be obtained and independently audited.

No electrostatics were connected to MD. `ReactiveSimulation`, the bonded force
field, integrator, H-transfer correction and CUDA paths were not changed.

## Original Verstraelen ACKS2 convention

Primary source: Verstraelen, Ayers, Van Speybroeck and Waroquier, *J. Chem.
Phys.* **138**, 074108 (2013), DOI 10.1063/1.4791569. Atomic units are used in
the paper.

### Degrees of freedom

- `N_A`: AIM electron population from a selected density partition.
- `N_A^0`: reference AIM population.
- `Delta_A = N_A - N_A^0`: relative electron population.
- `q_A = Z_A - N_A`; for a neutral-atom reference, `q_A = -Delta_A`.
- `U_A`: relative atom-condensed Kohn-Sham potential, dual to `Delta_A`.
- `mu_A`: first-order atomic chemical-potential parameter.
- `eta^e_AB`: second-order EEM-like hardness excluding the kinetic-energy
  response that ACKS2 treats through `U`.
- `X^s_AB`: atom-condensed non-interacting Kohn-Sham response kernel.

The AIM definition is part of the convention. The paper permits schemes such
as Hirshfeld, Hirshfeld-I, ISA or QTAIM, but changing the partition changes the
parameters. They are therefore not interchangeable elemental constants.

### Energy and constraints

For the neutral reference used in the dissociation derivation, paper equation
72 is the saddle functional

```
E_ACKS2 - E_ref - E_nn =
    min_(Delta, 1^T Delta = 0) max_(U, 1^T U = 0)
    [ mu^T Delta - U^T Delta
      + 1/2 Delta^T eta^e Delta
      + 1/2 U^T X^s U ]
```

`X^s` is a density-response kernel and is negative semidefinite on the
zero-sum-potential subspace, which is why the `U` operation is a maximization.
There are two independent constraints:

1. `1^T Delta = 0`, fixing total electronic population relative to the
   reference (equivalently the requested total charge);
2. `1^T U = 0`, fixing the arbitrary additive potential gauge.

With multipliers `mu_mol` and `lambda_U`, stationarity gives

```
eta^e Delta - U + mu - 1 mu_mol = 0
-Delta + X^s U - 1 lambda_U = 0
1^T Delta = 0
1^T U = 0
```

This is a `(2N+2)`-dimensional symmetric indefinite saddle-point problem, not
an ordinary positive-definite charge-only minimization. Physical stability
must test convexity of `eta^e` in allowed population directions together with
concavity of `X^s` in gauge-fixed potential directions, or equivalently the
Schur-complement charge response. A small linear residual alone is not a
stability test.

The paper's parameters are defined from a chosen molecular reference:

```
mu_A       = first derivative of the population-constrained energy
eta^e_AB   = condensed second derivative excluding KS kinetic response
X^s_AB     = integral integral X^s(r,r') w_A(r) w_B(r') dr dr'
```

The authors explicitly note that a transferable construction for the
Kohn-Sham contribution to `mu_A` is not supplied. Instead, it must be computed
from the molecular reference density and a reconstructed reference KS
potential. This prevents extraction of a universal H/C/N/O table from the
paper.

## LAMMPS `acks2/reaxff` convention

LAMMPS is an implementation reference, not the original AIM parameterization.
It solves the charge-sign version of the same block structure:

```
[ H   I   0   1 ] [ q        ]   [ -chi ]
[ I   X   1   0 ] [ U        ] = [   0  ]
[ 0  1^T  0   0 ] [ lambda_U ]   [   0  ]
[ 1^T 0   0   0 ] [ lambda_q ]   [   0  ]
```

where `q = -Delta`. The exact row ordering differs internally, but source code
constructs the `H`, `X`, two identity cross blocks, charge constraint and
potential-gauge constraint explicitly.

### ReaxFF hardness block

`H` is the same shielded QEq/ReaxFF matrix used by `fix qeq/reaxff`:

- `H_ii = eta_i`, where LAMMPS standalone files require the **full**
  self-Coulomb value (twice the value stored in a ReaxFF file);
- off-diagonal interactions use ReaxFF's shield convention and `gamma_i`;
- `chi_i`, `eta_i` and `gamma_i` are ReaxFF charge-model parameters, not the
  original ACKS2 AIM matrices.

### Empirical ACKS2 response block

For atom types `i,j`, LAMMPS forms

```
bcut_ij = (bcut_i + bcut_j) / 2
d       = r_ij / bcut_ij
X_ij    = Lambda d^3 (1-d)^6       for r_ij <= bcut_ij
X_ij    = 0                         otherwise
X_ii    = -sum_(j != i) X_ij
```

`Lambda` is the global `bond_softness` parameter. Thus `X` is a negative
weighted graph Laplacian with the constant vector as its gauge null mode.
Both `Lambda` and each type's `bcut_i` are ACKS2/ReaxFF-specific fitted
parameters. They are not derived from QEq electronegativities, Gaussian
overlaps or the original QTPIE parameters.

LAMMPS standalone parameter-file units are fixed to angstrom, eV and
elementary charge:

```
bond_softness
itype chi eta gamma bcut
```

The solver is iterative BiCGStab in production, but a float64 reference would
use the same dense `(2N+2)` block matrix and report both constraint residuals,
the full stationarity residual, inertia/projected spectra and Schur-complement
stability.

## Provenance table

| Quantity | Original ACKS2 | LAMMPS/ReaxFF ACKS2 | Availability for H/C/N/O |
|---|---|---|---|
| electronegativity / chemical potential | molecular AIM expectation value | ReaxFF `chi_i` | no universal original table; force-field-specific in LAMMPS |
| hardness | full molecular `eta^e_AB` matrix | ReaxFF shielded `H` from `eta_i,gamma_i` | force-field-specific, not freely mixable |
| response kernel | molecular AIM `X^s_AB` | empirical cutoff graph Laplacian | no original elemental table |
| bond softness | not a universal scalar in original equations | global fitted `Lambda` | available only with ACKS2-enabled ReaxFF sets |
| response cutoff | contained in molecular `X^s_AB` | fitted per-type `bcut_i` | available only with ACKS2-enabled ReaxFF sets |
| charge constraint | fixed relative population | zero total charge in LAMMPS fix group | defined |
| potential constraint | `sum U_A = 0` | same gauge row | defined |

The public LAMMPS `acks2_ff.water` example is attributed to the ACKS2 water
ReaxFF work and supplies only H/O plus its jointly fitted global softness. It
cannot support CH4, NH3, CO, CO2, CH2O, CH3OH, CH3NH2 or NH2OH. Ordinary CHO
or CHON ReaxFF files are not automatically ACKS2-compatible merely because
they contain QEq `chi`, `eta` and `gamma`: the ACKS2 softness and per-type
response cutoffs must also have been fitted and declared.

Some commercial ReaxFF distributions advertise ACKS2-enabled CHONS force
fields, but their parameters are force-field-specific and were not found as a
complete, independently documented public reference suitable for this audit.
Using them would also answer whether that particular ReaxFF charge model works,
not whether original ACKS2 supplies a transferable foundation for
ChemistryModel.

## Requested comparison: honest status

| Criterion | QEq | historical QTPIE | corrected Gaussian QTPIE | ACKS2 |
|---|---|---|---|---|
| mathematical implementation | complete | complete | diagnostic complete | equations reproduced; H/C/N/O parameters unavailable |
| long-range dissociation | fails | passes | passes | theory passes; requested numerical test blocked |
| small-set stability | conventional QEq limitations | H2 negative mode | CH4 nearly singular | unknown for H/C/N/O |
| polarity/dipoles | available | available | available | blocked beyond H/O |
| geometry continuity | available | available | available | blocked beyond H/O |
| parameter provenance | matching QEq convention | original published convention | corrected Chen Gaussian diagnostic | original values are molecular AIM; modern values are ReaxFF-fit-specific |
| dense reference complexity | O(N^3), N+1 system | O(N^3), N+1 system | O(N^3), N+1 system | O(N^3), 2N+2 saddle system |
| later sparse/batched structure | pair H solve | pair H plus overlap | pair H plus overlap | two sparse pair blocks plus two constraints; feasible structurally |

## Decision and next evidence needed

**Recommendation: another model required.** ACKS2 cannot be promoted or
rejected numerically without a defensible complete parameter convention. The
allowed next steps are one of:

1. obtain a published/open ACKS2-enabled H/C/N/O ReaxFF set whose `chi`, full
   `eta`, `gamma`, global softness and all `bcut` values were jointly defined,
   then validate that exact set as a modern ReaxFF-compatible candidate; or
2. derive original AIM ACKS2 matrices from a declared quantum-chemistry level,
   density partition and reference-state protocol. This is parameter
   generation, not elemental table lookup, and requires separate permission;
   or
3. investigate another dissociation-correct model with an available complete
   H/C/N/O parameterization.

No fitting should begin until one of these routes is explicitly selected.
