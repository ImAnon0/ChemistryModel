# Unified bond geometry: variational formulation

## Scope

This directory is research-only. No class is registered in the application or
production physics selector, and no force-field table is changed.

## Controls

The frozen control is `UnifiedBondCapacityEnergyPrototype`: its H-transfer and
heavy bond-order factors share heavy-atom capacity, while the established
heavy topology supplies angle energy.

`PostSolvedWeightedGeometryPrototype` records the already-observed sequential
alternative: solve radial capacity first, then weight the current angle layer
by expected bond participation. It is not variationally coupled and is kept to
separate “better weights” from a genuinely shared minimisation.

## Joint factor variables

For every H component, a probability `p_as` is assigned to each established
all-valid H matching. Positive amplitudes `sqrt(p_as)` reproduce the lowest
state of the stoquastic H-transfer Hamiltonian:

    F_H = sqrt(p)^T H sqrt(p)

For every heavy-heavy edge, `q_en` is a probability over order states
`n=0,1,2,3` supported by the existing tables:

    F_e = sum_n q_en E_en + tau sum_n q_en log(q_en)

Each factor probability is normalized. Expected tapered bond order consumes
the same finite heavy capacity as the unified radial model.

## Geometry inside the same objective

Expected bond participation and order are linear functions of `p` and `q`.
They determine bonding-domain count, bonded order, lone-pair count, angular
engagement, and stiffness inside the minimised scalar—not after it.

The weighted formulation uses the current continuous steric interpolation
between 180, 120, and 109.47 degrees. The electron-domain formulation instead
forms three local domain-state angle energies and uses their free energy:

    F_domain = -tau log sum_d exp(-E_d/tau) + tau log(3)
    d in {2,3,4}

The additive normalization makes an atom with no engaged angle contribute
zero. It is not fitted to a molecule or benchmark.

The complete minimized reference scalar is:

    E = sum_pair R_1(r) + sum_H F_H + sum_HH F_e + sum_atom F_geometry

subject to factor normalization and elemental heavy-capacity inequalities.
There is no separate overcoordination or post-solved angle term in this
mathematical energy.

## Forces

The float64 reference uses constrained SLSQP. A solution is rejected if its
simplex or capacity violation exceeds `2e-7`. The live Torch scalar is rebuilt
at the converged state, and capacity multipliers restore the geometry
derivative of taper-dependent constraints. Autograd then differentiates that
single Lagrangian-envelope scalar with respect to positions. Finite-difference
agreement is a mandatory gate; optimizer success alone is insufficient.

This is a formulation probe, not a production solver. Even a scientifically
successful candidate would still require a robust batched differentiable
solver and GPU-equivalence work.

## Joint local-state formulation

The product-of-marginals approximation cannot represent the fact that two
capacity-competing bonds may be mutually exclusive. The third prototype adds
one local state factor per heavy atom. A local state assigns integer order to
every incident candidate. Its probabilities are constrained to reproduce every
incident radial factor's order marginals, while the shared expected-order
capacity constraint remains the same continuous constraint as the radial
baseline. Individual local states are not treated as permanent molecular
topologies or filtered by a geometry-dependent hard capacity cutoff.
Angle and lone-pair energy are evaluated conditionally inside each joint state.

This is a local-polytope variational model: radial H/heavy factors and local
geometry factors are minimized together, and simultaneous angle engagement is
a joint probability rather than a product of independent means. Its equality
constraints are geometry-independent, so the ordinary envelope theorem gives
forces directly from the rebuilt Torch scalar.
