# Unified continuous bond-capacity free energy

## Status and scope

This is a research-only formulation. It is not registered in the application,
batch runner, or production physics selector. It does not modify the bonded
tables, H-transfer coupling, integrator, or production modules.

## Scalar formulation

Every candidate interaction retains a state-independent short-range repulsive
core. Attractive bonding is represented by factors coupled through one finite
capacity constraint for every heavy atom.

For an H competition component, the basis is the established set of all
one-valence-valid H matchings. Its Hamiltonian is the validated factorised
H-transfer Hamiltonian, but state `s` also has a capacity vector `c_s` whose
entry is the tapered occupancy of every heavy endpoint.

For a heavy-heavy edge, states are bond orders `n = 0, 1, 2, 3` where supported
by the existing H/C/N/O tables. Their energies are telescoping differences of
the existing single, double, and triple Morse surfaces relative to the common
single-bond repulsive core. State `n` consumes tapered capacity `n t_ij` at
both endpoints.

The constrained factor free energy is evaluated through non-negative dual
prices `lambda_i`:

    G(lambda) = sum_a epsilon_min(H_a + diag(C_a lambda))
              + sum_e [-tau log sum_n exp(-(E_en + c_en.lambda)/tau)]
              - lambda.V

and the physical radial energy is:

    E_radial = sum_pair R_1(r) + max_(lambda >= 0) G(lambda)

The derivative of `G` is expected capacity use minus elemental capacity. The
dual is concave and has only one scalar variable per heavy atom. A bounded
L-BFGS solve therefore replaces global bond-state enumeration and the previous
coupled SLSQP occupancy solve. Geometry gradients of the optimized radial
scalar follow from the envelope theorem; the converged dual price is held
constant in the live Torch expression and all distance/taper derivatives stay
in autograd.

The reference solver verifies projected KKT stationarity independently of the
optimizer's success flag and requires a residual below `1e-5`. L-BFGS-B,
bounded SLSQP, and Powell are attempted in sequence; a reported optimizer
success never bypasses the KKT gate.

No Grambow value, molecule identity, or reaction rule enters this expression.
The heavy bond-order free-energy scale is the already established
heavy-valence research temperature (0.01 eV). H mixing remains the published
ChemistryModel value. A separate `1e-4 eV` log-trace regularisation is applied
to the H Hamiltonian only to make the dual derivative unique at exact
degeneracy. It is sub-meV and far below ordinary chemical energies;
its effect is measured against the unregularised H microscopes rather than
assumed harmless.

## Two deliberately separated experiments

`UnifiedBondCapacityEnergyPrototype` tests the new radial representation while
retaining the established heavy-valence topology/angle layer. This is the
clean control for energy allocation and has conservative envelope forces.

`UnifiedBondCapacityTopologyPrototype` also sends expected factor occupancies
to the existing angle equation. It tests the stronger emergent-topology idea,
but is not assumed valid: because angle energy is not part of the dual
minimisation, capacity-price response can be missing from angle derivatives
when a constraint is active. Finite-difference tests decide whether that
approximation is acceptable. Failure means a future formulation must include
angles in the variational state or differentiate the full stationarity system.

## Gates

The order is fixed: analytic/small molecule safety, water-transfer QM,
accepted molecule preservation, NVE, then (and only then) frozen Grambow. A
benchmark gain cannot override a failure in an earlier gate.
