# Shared incremental bond-order channels

## Status and scope

Research-only formulation. It is not registered in Lab, `batch_runner.py`, or
the production physics selector. It changes no force-field parameter, H-state
Hamiltonian, integrator, timestep, or production source file.

This study asks only whether replacing one scalar shared-edge attenuation by
separate shared sigma/second/third bond-order channels improves the physical
description of crowded heavy-heavy bonding.

## Existing quantities

For undirected heavy-heavy edge `e=(i,j)`, the current engine supplies smooth
directed taper `t_ij`, continuous bond order `o_ij`, and the fully blended,
environment-softened attractive Morse magnitude `A_ij`. Symmetric edge values
are direction averages. The element-pair tables also supply the existing
single, double, and triple well depths `D1`, `D2`, and `D3`.

The ten H/C/N/O tables are monotone: `D1 <= D2 <= D3`. C-O, N-O, and O-O have
`D3=D2`; their oxygen endpoint limits ordinary order to at most two, so the
zero third increment is not expected to create an active preferred channel.
The prototype records and rejects any active zero-energy preferred channel
rather than silently inventing an energy for it.

## Preferred shared channels

Let

`p_e1 = clamp(o_e,     0, 1)`

`p_e2 = clamp(o_e - 1, 0, 1)`

`p_e3 = clamp(o_e - 2, 0, 1)`.

The preferred taper-weighted capacity of channel `k` is

`q_ek = t_e p_ek`.

The live shared capacity is `0 <= b_ek <= q_ek`, with occupancy
`x_ek=b_ek/q_ek`. Every `b_ek` belongs to the undirected edge and is used
identically at both endpoints. Total effective edge bond order is

`beta_e = sum_k b_ek / t_e`.

Thus a single bond has only channel 1, a double bond adds channel 2, and a
triple bond adds channel 3. Partial bond character is represented by
fractional channel capacity rather than fractional attenuation of the whole
edge.

## Valence and hierarchy constraints

The validated H-state energy remains untouched. Raw H contact taper retains
its existing heavy-centre load

`h_i = sum_(a=H) t_ia`.

Heavy-channel capacity satisfies

`sum_(e incident on i) sum_k b_ek <= max(V_i-h_i, 0)`.

Channel hierarchy is explicit:

`x_e3 <= x_e2 <= x_e1`.

This prevents a pi-like channel from surviving more strongly than the lower
channel that supports it. Both the endpoint constraints and hierarchy are
linear in `b` for fixed live `q`, so the allocation remains a convex quadratic
programme.

## Incremental attraction

Existing depth increments are

`d1=D1`, `d2=D2-D1`, and `d3=D3-D2`.

For the channels present at the current order, define unnormalised energetic
shares `w_ek=d_k q_ek`. The exact current attractive magnitude is partitioned
without changing it:

`A_ek = A_e w_ek / sum_l w_el`.

Therefore `sum_k A_ek=A_e` at every geometry. No new energy or fitted
coefficient is introduced; depth increments determine only which established
part of the current attraction is associated with each bond-order channel.

Each channel uses the same unique parameter-free quadratic completion tested
by the scalar reference:

`E_ek(b) = -A_ek [2 x_ek - x_ek^2]`.

It obeys `E(0)=0`, `E(q)=-A_ek`, and `dE/db=0` at the accepted preferred
capacity. The full Morse repulsive branch is unchanged. When endpoint capacity
is not contested, all `b=q` and the exact current attraction, topology, energy,
and first derivative are recovered.

The allocation minimizes, up to the constant accepted energy,

`sum_(e,k) (A_ek/q_ek^2) (b_ek-q_ek)^2`

under shared endpoint, bound, and hierarchy constraints. The distinction from
the scalar model is physical rather than parametric: competition can surrender
an incremental pi channel while retaining more of the sigma channel instead
of attenuating all bond character through one occupancy.

## Topology and scalar energy

The existing angle implementation expects one contact membership. The channel
model supplies

`m_e = sum_k b_ek / sum_k q_ek`.

Consequently its topology bonded order is exactly the allocated total channel
capacity, while `m_e=1` in every unconstrained accepted structure. One edge
energy is split equally between its two endpoints only for per-atom accounting;
the undirected total is counted once.

The heavy overcoordination replacement uses the identical constrained load
`h_i + sum_(e,k)b_ek`. It is a consistency guard, not a compensating penalty
on attraction rejected by another representation.

## Conservative gradients and symmetry

The slow reference uses float64 SLSQP to identify active inequalities, then
reconstructs the active KKT system from live Torch tensors. The scalar energy
is differentiated through current taper, order, attraction shares, endpoint
capacity, and hierarchy coefficients. No detached solution is used as an
energy input.

Within an active region the result is differentiable. The value function of
the convex programme is continuously differentiable at nondegenerate active
set changes; finite-difference, channel-crossing, label-permutation, and NVE
tests are required to verify the numerical implementation.

Permutation symmetry follows from one sorted undirected edge, direction-
averaged edge quantities, one incidence column per shared channel, and channel
labels defined by bond-order rank rather than atom identity.

## Falsification gates

Reject the prototype if it changes any required accepted molecule, fails force
or continuity checks, loses the scalar model's catastrophic-error removal,
performs worse than local v0 on the combined benchmark gates, or only improves
the named microscope cases. No energy share or constraint may be tuned against
Grambow.

## Clarified formulation-space study

The incremental-QP result revealed an inherited limitation rather than a
positive result for the current bond representation. In crowded environments
the production order builder first computes `spare = max(V-coordination, 0)`.
Exactly where capacity competition is needed, raw coordination is normally at
or above valence, `spare` is zero, and every preferred heavy-heavy edge is
therefore reduced to order one before the channel model sees it. Splitting
that already-collapsed attraction cannot test whether multiple bond states are
needed. This explains why the first channel reference differs materially from
the scalar model in only one of 200 reactions.

The broader study will therefore compare these formulation classes without
changing production:

1. **Inherited-order channel QP (implemented v0).** Useful control; it tests
   shared incremental allocation but not a new definition of bond order.
2. **Linear shared allocation.** Directly assigns capacity to the most
   favourable edges. It resembles the successful local-v0 selection but has
   non-unique states and force cusps at exact preference crossings, so it is
   rejected as a conservative MD candidate without implementation.
3. **Empirical curvature/activation functions.** These could make new channel
   preferences from distance, but require a new width, threshold, or chemical
   potential. They are rejected at this stage because no independent source
   fixes those values and Grambow fitting is forbidden.
4. **Shared bond-state free-energy Hamiltonian (selected v1 experiment).** Bond order is a
   state of the undirected edge and is derived from the existing single,
   double, and triple Morse surfaces rather than the raw-coordination order
   heuristic. Existing Hamiltonian mixing and heavy-state temperature are
   reused; no new coefficient is introduced.

### Table-surface shared bond Hamiltonian

For each heavy-heavy edge `e`, an integer state `n_e in {0,1,2,3}` is
represented by hierarchical unit channels. Endpoint constraints are

`sum_(e incident on i) n_e + sum_(H contacts h at i) z_h <= V_i`.

Hydrogen contact variables are present only to make heavy capacity competition
consistent. Their actual H-state energy remains owned by the unchanged H-state
model. The research correction subtracts a same-basis H-only reference, so it
adds no standalone H energy.

Let `V_en(r)` be the existing Morse surface built from the audited length,
depth, and width table for order `n`. The no-bond surface retains the
single-table repulsive core `R_e1(r)`. State-dependent increments are

`epsilon_e0 = 0`,

`epsilon_en = V_en(r) - R_e1(r), n=1,2,3`.

Thus the full edge energy is exactly `R_e1` in state zero and exactly the
existing order-specific Morse surface in integer state `n`. This differs
fundamentally from partitioning the already blended production attraction.
It lets endpoint capacity decide bond order at the same time as bond ownership.

All valid (not merely maximal) component states are retained. Omitting
non-maximal states would force a vanishing cutoff contact to consume capacity.
The diagonal is the sum of selected H attraction and the `epsilon_en` for the
highest occupied channel on each heavy-heavy edge. The first falsification
reference keeps the Hamiltonian diagonal and uses the existing heavy-state
temperature to form an exact state free energy. This is the unique
parameter-free smooth mixture available from the current heavy-state
infrastructure. Adding off-diagonal sigma/pi transfer coupling would require
deciding how a bond-formation state couples to the no-bond state; the H-transfer
coupling does not define that quantity, so it is not silently reused or fitted.

A same-state-space reference retains only H diagonal terms. The heavy-bond scalar is

`E_HH = sum_e t_e R_e1 + F(H_full) - F(H_H-only)`,

with the existing `0.01 eV` heavy-state free-energy scale. The same-basis
subtraction makes this exactly zero beyond the deterministic repulsive core
when heavy-bond terms and couplings vanish; it also prevents state-count
entropy from becoming a fictitious bond energy.

The density-matrix diagonal supplies shared channel occupations. Sigma
occupation defines contact/angle membership; expected channel count is the
bond order and capacity load. Energy and forces come from the scalar free
energy, not from detached occupations. Exact eigenspectrum differentiation is
conservative, and the thermal density is basis-independent at degeneracy.

For required accepted molecules the production graph is feasible and no heavy
centre is competitive. The reference takes an exact identity path, preserving
the current energy and force rather than adding artificial finite-temperature
mixing to settled molecules. A hard identity-to-Hamiltonian switch was tested
and rejected because it produced a finite energy jump when a vanishing fifth
contact entered the cutoff.

The corrected experiment uses the continuous raw overload
`u_i=clamp(sum_j t_ij-V_i,0,1)`, the C1 smoothstep `s_i=u_i^2(3-2u_i)`, and the
symmetric component gate `g=1-product_i(1-s_i)`. The scalar energy is

`E=(1-g) E_current + g E_state`.

There is no fitted width: one valence unit is the natural capacity scale.
Both `g` and its first derivative are zero at competition onset and `g=1`
with zero first derivative when any centre reaches one full excess unit.
Topology membership and overcoordination replacement use the same gate. This
is a single conservative scalar interpolation, not detached force blending.
Finite-difference and dense onset scans must still verify the construction.

This v1 is a falsification reference, not a production architecture. Exact
valid-state enumeration is exponential and may itself demonstrate that a
future production model needs a factorised or continuous solver. It is rejected
if accepted structures change, the onset boundary is not conservative, state
counts exceed the safety bound on the frozen benchmark, or its independent
metrics do not justify the extra representation.

### Continuous mean-field reduction

The exact v1 exceeded 100,000 states on every mandatory reaction, so v2 keeps
the same table-surface energy but replaces the global configuration sum by a
strictly convex mean-field free energy. For every H contact or heavy bond-order
channel, use occupancy `0<x_a<1`, taper capacity `q_a=t_a`, and energy
increment `epsilon_a`. Minimise

`F(x)=sum_a [epsilon_a x_a + tau q_a (x_a ln x_a +(1-x_a)ln(1-x_a))]`

subject to

`sum_(a incident on i) q_a x_a <= V_i`

and the channel hierarchy `x_3<=x_2<=x_1`. Here `tau=0.01 eV` is the existing
heavy-state scale. Taper multiplies both energy/entropy and capacity, so a
contact disappearing at the cutoff contributes neither a fictitious state
entropy nor finite capacity load.

The same variable space is solved once with full H/heavy increments and once
with H increments only. Their free-energy difference isolates the new heavy
bond term while the production H-state energy remains untouched. Because the
capacity matrix depends on live taper, the scalar includes the SLSQP endpoint
KKT multipliers as zero-valued gradient terms; finite differences test this
envelope derivative. Occupancies remain diagnostics. Forces are derived from
the scalar optimum value rather than differentiating a detached topology.

V2 deliberately retains the existing topology/angle membership. This isolates
the radial bond-energy hypothesis and avoids pretending that a detached
mean-field occupancy supplies conservative angle forces. A future unified
model would need topology order and angles to consume the same shared state.

V2 produces the strongest frozen Grambow aggregate tested, but fails the
independent water-transfer and crowded-NVE gates. A v3 overlap gate based on
normalised existing single-Morse overlap was tested and rejected because it
reverted water transfer almost to production. No threshold or gate coefficient
was fitted after that falsification.
