# C0 continuous SQE electronic-state prototype report

> The subsequent 235-configuration parameter pilot is documented in
> `PILOT_REPORT.md`. It concludes that C0 should be retained as a localising
> transfer kernel but reconsidered as a complete electronic-state model.

## Outcome

**C0 passes the mathematical/stability gate and is worth progressing to a
blocked C0-versus-C1 parameterisation study.**

It is not ready for production, MD coupling, or scientific charge claims. The
17 values in `parameters.py` are explicitly provisional numerical seeds. No QM
fit was performed and no existing ChemistryModel parameter or production file
was changed.

## Implementation

The research-only implementation consists of:

- `parameters.py`: immutable 17-parameter H/C/N/O C0 schema and fixed support
  conventions;
- `solver.py`: incidence construction, Gaussian Coulomb/hardness matrix,
  smooth transfer compliance and differentiable scaled solve;
- `prototype.py`: standalone evaluation facade returning energy, charges,
  forces, dipole, transfers and response state;
- `diagnostics.py`: eigenvalue, conditioning, residual, charge and continuity
  reporting plus the stored-grid audit;
- `tests/test_c0.py`: conservation, molecule, dissociation, symmetry, force and
  continuity gates.

For oriented pair incidence `B`, the implementation solves

```text
X = B C^(1/2)
A = I + X^T H X
A u = -X^T chi
p = C^(1/2) u
q = B p.
```

The scalar energy is

```text
E = 1/2 u^T u + chi^T q + 1/2 q^T H q.
```

`H` contains strictly positive intrinsic elemental hardness plus the complete
Gaussian charge-density Coulomb Gram matrix, including self interactions. The
response is structurally positive definite. Forces are obtained by autograd
through the same scalar and linear solve.

Transfer compliance is distance- and element-only. The solver constructs
`C^(1/2)` directly with an error-function capacity and a quintic compact-
support amplitude. This gives exactly zero transfer outside 4.5 A with smooth
energy and forces. C0 contains no environment descriptor, bond order, unified
membership, topology classification, molecule identity or production state.

The first version is neutral-only. A nonzero requested total charge is rejected
instead of being distributed through a global KKT constraint. A future charged
model requires explicit reference/source charges and is outside C0.

## Focused test result

```text
17 passed
```

Covered:

- exact incidence-based total-charge conservation;
- independently neutral separated O/H and H/OH fragments;
- finite H2, H3, CH4, H2O and CH2O energy/forces;
- positive hardness and response spectra;
- deterministic repeated solves;
- translation and rotation invariance;
- atom-permutation consistency;
- net-zero force;
- central finite-difference force agreement;
- smooth O/H distance scan;
- energy/force continuity at the 3.5 A inner and 4.5 A outer support points;
- explicit rejection of unsupported non-neutral use.

## Numerical diagnostics

### Representative systems

| System | min eig(A) | cond(A) | min eig(H), eV | max absolute q, e | Electronic energy, eV |
|---|---:|---:|---:|---:|---:|
| H2 | 2.204 | 1.000 | 13.36 | 0 | 0 |
| H3 | 1.000 | 2.301 | 12.00 | 0 | 0 |
| CH4 | 1.000 | 4.318 | 9.12 | 0.055 | -0.022 |
| H2O | 1.000 | 3.449 | 12.51 | 0.159 | -0.336 |
| CH2O | 1.000 | 3.712 | 10.00 | 0.102 | -0.197 |

Solve residuals are at or below `1.5e-16` and total-charge residuals are at or
below `1.1e-17 e` in these cases. The minimum response eigenvalues reported as
slightly below one in aggregate (`1 - 2.1e-15`) are eigensolver roundoff, not
negative modes.

### O/H dissociation

| Separation | O charge | Electronic energy |
|---:|---:|---:|
| 0.8 A | -0.10665 e | -0.22465 eV |
| 1.5 A | -0.04533 e | -0.09548 eV |
| 2.0 A | -0.00374 e | -0.00788 eV |
| 3.0 A | -3.34e-8 e | -7.03e-8 eV |
| 4.5 A | 0 | 0 |
| 10-100 A | 0 | 0 |

This directly fixes the diagnostic QEq failure mode without relying on a
distance-based fragment label.

### Stored reactive grids

| System | max adjacent delta E | max adjacent atomic delta q | min eig(A) | max cond(A) |
|---|---:|---:|---:|---:|
| H + CH2O | 0.00349 eV | 0.00476 e | 1.000 | 4.053 |
| H3 | 0 | 0 | 1.000 | 2.735 |
| H + CH4 | 0.000573 eV | 0.00196 e | 1.000 | 4.309 |
| H + H2O | 0.01715 eV | 0.01140 e | 1.000 | 4.252 |

All 256 grid values are finite. There is no analogue of the rejected Gaussian
QTPIE methane result (`160 eV` and `424 e` adjacent jumps).

The complete machine-readable result is `c0_diagnostics.json`.

## Limitations

- The initial parameters are not scientifically calibrated. Dipoles and
  charges must not be presented as predictions.
- C0 can represent scalar atomic charge redistribution but not explicit atomic
  dipoles, lone-pair directionality or higher multipoles.
- The dense all-pair reference uses `N(N-1)/2` transfer variables and a dense
  solve. It is intentionally correctness-first, not a production scaling
  design.
- The compact support is a versioned provisional hyperparameter. The QM pilot
  must determine whether physically relevant polarization/transfer is being
  truncated too early.
- Neutral fragments retain exact net charge while their internal charges may
  still polarize through long-range Coulomb coupling; this desired behaviour
  needs explicit dimer validation.
- Charged systems are not supported by this generation.

## Recommendation

Proceed, but in this order:

1. Freeze and run the planned 10% quantum-observable pilot for C0.
2. Verify parameter Jacobian rank and whether 17 C0 parameters can separately
   identify electronegativity, hardness, Gaussian width and capacity range.
3. Implement C1 only as a nested comparator using the same solver and dataset.
4. Promote C1 over C0 only if molecule-family-blocked dipoles, response and
   reactive contacts improve without weakening localisation, conditioning or
   force continuity.

Do not connect C0 to the chemistry engine or MD yet. The result supports the
mathematical architecture, not the provisional parameter values.
