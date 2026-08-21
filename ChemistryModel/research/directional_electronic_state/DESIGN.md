# State-conditioned directional electronic diagnostic

## Scope and control

The production engine and `UnifiedBondCapacityEnergyPrototype` are frozen.
`StateConditionedP2CouplingPrototype` is an unregistered research subclass.
Its Boolean control `directional_response=False` executes the inherited
unified-radial Hamiltonian exactly. There is no continuous response-strength
parameter and no water/Grambow fit.

The physical-variable audit preceding this design is in
`FORMULATION_GATE.md`.

## Baseline electronic state

For each connected H-contact component, unified radial enumerates valid bond
assignments `s`. Its Hamiltonian contains:

    H_ss = sum of occupied tapered radial attractions

and, when states differ by transfer of one H between an old and new edge,

    H_st = -c_H sqrt(D_old D_new) O(t_old, t_new) / N

where `O` is the established balanced simultaneous-contact gate and `N` is the
established crowding normalization. The lowest regularized state free energy
participates in the same heavy-capacity dual as heavy bond order.

This is already a coherent electronic-state model. Its missing information is
that `O` depends only on two scalar radial tapers.

## State-conditioned density descriptor

For each state `s`, construct the repository's frozen continuous SAPT
environment matrix `w^s_ij`:

- heavy-heavy contacts use their existing continuous taper;
- an H contact appears only if it is occupied in `s`, weighted by its taper;
- no hard topology threshold is introduced.

At target centre `i`, evaluated toward the transferring H direction `u`, use
the normalized second Legendre moment

    q2_i^s(u) = presence_i
                sum_j w^s_ij P2(u_ij dot u)
                / (sum_j w^s_ij + epsilon)

with

    P2(x) = (3 x^2 - 1) / 2.

The frozen SAPT exchange-amplitude map is

    a_i^s(u) = exp(k_i q2_i^s(u)),

where `k_i` is the existing H/C/N/O coefficient independently fitted in the
SAPT research programme to `SAPT0/jun-cc-pVDZ EXCH10`, not a reaction energy.

For a transition `s -> t` that exchanges old and new edges, evaluate:

- the new target centre in state `s`, where the new edge is unoccupied;
- the old target centre in state `t`, where the old edge is unoccupied.

The symmetric factor is

    g_st = sqrt(a_new^s * a_old^t) = g_ts

and the diagnostic Hamiltonian uses

    H_st_directional = H_st_radial g_st.

All diagonal energies, radial tapers, capacity loads, dual constraints,
regularization temperatures, and crowding normalization remain unchanged.

## Invariants

- H3: target atoms have no heavy directional environment, so `q2=0`, `g=1`.
- Settled one-state molecules: no off-diagonal transition exists, so energy and
  force are bit-for-bit the inherited result.
- Permutation: all environment moments are sums and the transition factor is
  symmetric.
- Rotation: only unit-vector dot products enter.
- Cutoff: the inherited radial overlap still makes coupling vanish; the
  descriptor weights also taper continuously.
- Forces: positions, tapers, density moments, and the modified Hamiltonian stay
  in Torch; the regularized eigenvalue and capacity envelope are differentiated
  normally.

## Interpretation boundary

The frozen `P2` map describes anisotropy of short-range exchange density. Its
use as a multiplier of covalent state coupling is a diagnostic analogy, not a
published identity. Passing would show that this independently anchored local
density moment carries useful missing information. Failing rejects this map,
not all directional electronic or polarization models.

The prototype does not output physical atomic charges, molecular dipoles, or
polarizabilities and must not be described as an electrostatic model.
