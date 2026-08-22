# ChemistryModel electrostatics integration audit

## Decision

**Architecture after the narrow fixes below: correct for an opt-in extension.**

**Current electrostatic formulation: not ready for tuning or production MD.**

The term is registered once and added once after the frozen unified-radial
composition. With `enabled_extensions=()` the 21-case frozen equivalence suite
is unchanged. The enabled implementation conserves charge, is deterministic,
has conservative forces, and is smooth on the stored reaction grids.

However, it is not a complete, internally documented QEq convention. The
H/C/N/O table supplies electronegativity and self-hardness values while the
off-diagonal interaction is an unshielded numerical `1 / r` with distances in
angstrom. The corresponding atomic size/shielding convention and explicit
Coulomb conversion are absent. The current values therefore cannot yet be
described as a coherent eV-valued published QEq energy.

More importantly, global QEq has the wrong separated-fragment limit. In the
local diagnostic, a nominally neutral separated O/H pair retains charges of
approximately `-0.1547 e / +0.1547 e` and an energy of `-0.3259 eV` at 100 Å.
That is unacceptable for a reactive soup model in which neutral fragments form
and separate.

## Architecture audit

The active opt-in path is:

`UnifiedBondCapacityEnergyPrototype`
→ `ChemistryEngine`
→ `UnifiedRadialHamiltonian`
→ frozen unified-radial terms
→ registered extension terms
→ `EnergyResult`

Electrostatics is not added by `ReactiveSimulation` or a legacy force path. It
is instantiated by `chemistry_engine.terms.registry`, evaluated once in the
Hamiltonian extension loop, distributed once into the per-atom total, and then
differentiated with the full scalar Hamiltonian.

The audit found and fixed four integration defects without changing QEq
parameters:

1. A grouped batch was previously solved as one global charge system, including
   cross-box Coulomb terms and one global constraint. It now solves every
   independent simulation box separately and enforces the requested charge per
   box.
2. QEq distances previously ignored the simulation's periodic minimum image.
   Equivalent wrapped and unwrapped molecular positions now agree.
3. Charge state was held only inside the term and was missing from
   `EnergyResult`. It is now exposed under
   `result.state["extensions"]["electrostatics"]`.
4. Run provenance recorded enabled base terms but not enabled extensions or
   extension parameters. It now records the extension selection and a separate
   extension-parameter payload/hash.

The existing extension diagnostic also reconstructed scalar components by
broadcasting the scalar once per atom, falsely reporting a composition error.
It now sums every component exactly once. Reconstruction error is zero for
water and `3.6e-15 eV` for formaldehyde.

## Disabled regression

The frozen `unified_radial_v1` equivalence harness passes all 21 cases with no
extension selected. This checks total energy, components, forces, memberships,
bond orders, probabilities, dual variables and KKT diagnostics. The extension
default remains empty, so no existing user path silently enables charges.

## Charge and matrix checks

All values below use CPU float64 and the current implementation.

| System | Representative charges (e) | Dipole (D) | Minimum projected hardness eigenvalue | Projected condition |
|---|---:|---:|---:|---:|
| H2 | approximately 0, 0 | approximately 0 | 12.5386 | 1.000 |
| H3 | approximately 0, 0, 0 | approximately 0 | 12.4378 | 1.071 |
| CH4 | C -0.0675; H +0.0169 each | approximately 0 | 9.6616 | 1.386 |
| H2O | O -0.2270; H +0.1135 each | 0.641 | 12.3703 | 1.070 |
| CH2O | C +0.0470; O -0.2306; H +0.0918 | 1.958 | 10.0016 | 1.336 |

Neutral charge residuals are below `5e-16 e`. Repeated evaluations are
bit-identical. Energies, charges, and forces are finite in all audited cases.
The positive projected eigenvalues show that these small examples are stable
under the *current weak unshielded matrix*. They do not validate a future
corrected Coulomb/shielding convention, whose conditioning must be retested.

## Force validation

The electrostatic force passed:

- central finite differences on every CH2O Cartesian coordinate;
- translation invariance;
- rigid-rotation covariance;
- Newton's third law / net force approximately zero;
- periodic wrapped/unwrapped equivalence;
- independent grouped-box equivalence.

The finite-difference tolerance was `2e-8 eV/Å` absolute and `2e-7` relative.
The force is obtained by autograd through the variational charge solution. At a
converged constrained minimum this is consistent with the stationary energy.

## Reaction-coordinate behaviour

The stored 8×8 transfer grids are finite and continuous at their sampling
resolution:

| System | Maximum adjacent electrostatic ΔE | Maximum adjacent slope | Maximum adjacent atomic Δq |
|---|---:|---:|---:|
| H3 | ~`2e-15 eV` | ~`1e-14 eV/Å` | ~`2e-16 e` |
| H + CH2O / formaldehyde | `0.00124 eV` | `0.00916 eV/Å` | `0.000820 e` |
| H + CH4 / methane | `0.000115 eV` | `0.000892 eV/Å` | `0.000461 e` |
| H + H2O / water | `0.00258 eV` | `0.0201 eV/Å` | `0.00170 e` |

No discontinuity or sudden QEq energy jump was found. This is necessary but
not sufficient: the very small relative changes also explain why the term does
almost nothing to the QM residual shapes.

## QM residual comparison

The originally requested comparison between `base_results.csv` and
`electrostatics_results.csv` is invalid. `base_results.csv` uses the old
`ReactiveSimulation` base potential, while `electrostatics_results.csv` uses
the unified-radial model plus electrostatics. Even after subtracting the
electrostatic component, the model mismatch has RMSE values from `0.594 eV`
for H3 to `6.402 eV` for water. Those differences are not electrostatic effects.

The valid comparison reconstructs disabled and enabled energies from the same
unified-radial evaluation. Across the 347 geometries with successful QM data:

| Scope | Disabled MAE / RMSE (eV) | Enabled MAE / RMSE (eV) | Result |
|---|---:|---:|---|
| Overall | 0.98408 / 1.42116 | 0.98443 / 1.42177 | slightly worse |
| Formaldehyde | 0.89564 / 1.30838 | 0.89511 / 1.30893 | mixed; negligible |
| H3 | 1.33086 / 1.80501 | 1.33086 / 1.80501 | unchanged |
| Methane | 0.86875 / 1.23103 | 0.86877 / 1.23114 | slightly worse |
| Water | 0.81690 / 1.23349 | 0.81907 / 1.23580 | worse; 73 worsened, 2 improved |

By region, product-like points improve slightly (`0.38349 → 0.38215 eV` MAE),
while the transfer region worsens (`1.20097 → 1.20188 eV` MAE). There is no
material, systematic evidence that the current term adds missing transferable
physics.

The electrostatics evaluator was corrected so future runs report a same-engine
disabled energy, enabled energy, actual enabled forces, disabled forces and the
isolated electrostatic force. Previously its force columns came from forces
computed before the enabled engine was installed.

## Parameter/formulation provenance gate

The original QEq model uses electronegativity, atomic self-Coulomb hardness and
shielded Coulomb interactions between atomic charge densities. The current
table resembles a published QEq H/C/N/O table, but that convention also
contains atomic radii/size information that is absent here. The original paper
describes shielded interactions rather than this bare mixed-unit inverse
distance ([Rappé and Goddard, 1991](https://paperzz.com/doc/7664319/charge-equilibration-for-molecular-dynamics)).

Modern ReaxFF/QEq documentation likewise treats electronegativity, self-Coulomb
hardness and the shielding exponent as a matched set
([LAMMPS QEq/ReaxFF documentation](https://docs.lammps.org/fix_qeq_reaxff.html));
the ReaxFF equations use a pair shielding term to avoid unbounded short-range
electrostatics
([ReaxFF/AMBER formulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8145783/)).
The official ReaxFF developer documentation also notes that original QEq uses
1s Slater-type charge densities and that ReaxFF uses an analytic approximation,
with an explicit Coulomb unit conversion
([SCM ReaxFF developer documentation](https://www.scm.com/doc/ReaxFF/_downloads/96fcdbe2e9ecc2c7b4a2af885be12ff2/ReaxFF_SCM_Developer_Doc.pdf)).

Global QEq's fractional-charge dissociation problem is intrinsic, not a local
coding mistake. QTPIE was introduced specifically to correct that asymptotic
behaviour
([Chen and Martínez](https://arxiv.org/abs/0812.1543)).

## Answers and next steps

### 1. Is the implementation correct?

The engine integration, batching, periodic distances, diagnostics and forces
are correct after the narrow fixes. The physical electrostatic convention is
not complete, so the overall extension is not scientifically correct yet.

### 2. Is it physically helping?

No convincing evidence. The valid same-engine residual comparison is slightly
worse overall and notably worse for most water points. H-transfer surfaces are
smooth, but the correction is nearly constant and very small along them.

### 3. Are the current values reasonable?

They are recognizable electronegativity/self-hardness values, but values cannot
be assessed independently of the missing Coulomb units and shielding/atomic
size convention. They must not be tuned in the current hybrid formulation.

### 4. What should eventually be fitted?

Only after selecting a complete charge-localising formulation:

- H/C/N/O electronegativities;
- self-hardnesses;
- all required atomic size/shielding or transfer-hardness parameters;
- any smooth reactive connectivity/localisation parameters;
- explicit total/fragment charge conventions.

Fit against independent QM charge-response information, molecular dipoles,
finite-field polarizabilities, charge redistribution along reactions, energies
and forces. Keep molecule families and dissociation scans as untouched
hold-outs. Do not fit directly to Grambow aggregate barriers.

### 5. Which existing ChemistryModel terms may later need retuning?

If a validated electrostatic model is adopted, the radial/capacity surfaces and
established geometry terms will need a joint residual audit because the frozen
effective model already absorbs some electronic effects. Barrier and reaction
energies must then be revalidated, but no existing parameter should be changed
before the electrostatic convention passes its own gates.

### Must fix before tuning

1. Choose and document one complete H/C/N/O electrostatic convention, including
   units, shielding and gradients.
2. Replace global QEq with a model having the correct separated-neutral-fragment
   limit, or explicitly justify a smooth reactive fragment constraint.
3. Re-run conditioning and short-range stability after that formulation change.
4. Generate same-engine disabled/enabled energy **and force** datasets with
   immutable metadata.
5. Validate dipoles, polarizabilities and charge response independently of
   energy residuals.

### Safe now

- Keep the extension opt-in and disabled by default.
- Retain the architecture, per-box solver boundary, diagnostics and audit tests.
- Use the current implementation only as a QEq comparator/diagnostic.

### Future improvements

- Evaluate the already researched charge-localising candidates under this
  engine extension interface.
- Add scalable batched linear algebra only after the reference formulation and
  gradients are fixed.
- Jointly refit the complete energy landscape only after independent electronic
  validation passes.

**Recommendation: do not tune and do not enable in production. Preserve this
QEq implementation as a diagnostic comparator while selecting a coherent,
charge-localising electrostatic model.**
