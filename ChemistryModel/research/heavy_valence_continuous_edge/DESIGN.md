# Continuous shared-edge heavy-valence capacity

## Status and boundary

This is a research-only formulation. It is not registered in `batch_runner.py`,
Lab, the benchmark physics selector, or any production default. It changes no
force-field table, equation, H-state Hamiltonian, integrator, or timestep.

## Audit of the current energy path

`reactive_torch.py` constructs a smooth radial contact `t_ij`, raw radial
coordination `c_i = sum_j t_ij`, and a continuous pair order `o_ij` from spare
valence shared at both endpoints. The order interpolates the existing
single/double/triple Morse length, depth, and width tables. Every radial
contact then receives

`E_Morse,ij = t_ij [R_ij(o_ij) - A_ij(o_ij)]`.

The base overcoordination term is evaluated from every radial contact. The
separate topology taper can exclude configured nonchemical pairs, but it does
not change the radial Morse energy.

`h_state_torch.py` replaces the energy of H-containing contacts by a
capacity-one adiabatic H-state Hamiltonian. It removes the hydrogen share of
the base overcoordination penalty but deliberately leaves heavy-centred
overcoordination intact.

`valence_state_torch.py` / the optimised batched implementation enumerate
size-`V_i` local contact states only for heavy-centred topology. Their thermal
membership changes electron-domain counting and angles. It does not replace
heavy-heavy radial attraction, and production intentionally retains the raw
heavy overcoordination penalty. Consequently, a heavy-heavy contact may be
rejected by topology while retaining full radial attraction and adding to the
penalty.

## Shared continuous variable

For one undirected heavy-heavy edge `e=(i,j)`, define its preferred capacity

`q_e = mean_directed(t_ij o_ij)`.

This is the current model's continuous, taper-weighted bond order and contains
no new fitted value. The prototype introduces one shared capacity

`0 <= b_e <= q_e`,

used at both endpoints. Its topology occupancy is `x_e=b_e/q_e` (zero when
`q_e=0`), and its effective bond order is `b_e/t_e`. Thus fractional capacity,
single-bond capacity, and multiple-bond character are represented by the same
edge quantity rather than two atom-local binary decisions.

The H-state energy is not changed. Raw H contact taper supplies the H load

`h_i = sum_(k=H) t_ik`.

The shared heavy-edge constraints are

`sum_(e incident on i) b_e <= max(V_i - h_i, 0)`.

There is one column per undirected edge in the incidence matrix, so the same
`b_ij` consumes capacity at both atoms. No molecule graph, reaction identity,
or element-pair exception is introduced.

## Capacity-bearing attraction

Let `A_e` be the positive magnitude of the existing full heavy-heavy Morse
attraction, averaged over its two directed copies. The continuous attractive
energy is

`E_e(b_e) = -A_e [2 x_e - x_e^2]`, where `x_e=b_e/q_e`.

Equivalently, apart from the constant `-A_e`, the allocation minimizes the
strictly convex weighted projection

`sum_e (A_e/q_e^2) (b_e-q_e)^2`

under the shared capacity constraints.

This is the unique parameter-free quadratic completion with all three useful
anchors: `E(0)=0`, `E(q)=-A`, and `dE/db=0` at the accepted preferred bond.
It avoids the discontinuous winner selection of a linear programme and uses no
temperature or Grambow-tuned smoothing value. This quadratic completion is a
new modelling hypothesis and must be judged by the evidence gates; it is not
claimed as a derived electronic Hamiltonian.

The existing Morse repulsive term is never changed. At `b=q`, attraction and
its first derivative are exactly the current accepted interaction. When
capacity is crowded, reducing `b` continuously removes attraction while
retaining short-range exclusion. The heavy overcoordination replacement is
evaluated from the exact constrained load `h_i + sum_e b_e`; it is therefore a
consistency guard, not a second compensation for attraction that the capacity
model rejected.

## Conservative gradients and smoothness

The allocation is a strictly convex quadratic programme. The research solver
uses float64 SLSQP only to identify the active set, then reconstructs that
active KKT system with Torch tensors. Energy is evaluated as one scalar from
the reconstructed `b`, so autograd differentiates through the live geometry,
preferred capacities, attractions, and active capacity constraints.

Within an active region the solution and energy are differentiable. Across an
active-set boundary the value function of this convex projection is
continuously differentiable under the usual nondegeneracy conditions, although
second derivatives may change. Finite differences, label-exchange scans, and
NVE tests are required rather than assuming that property numerically.

Permutation symmetry follows from unique sorted undirected edges, a symmetric
incidence matrix, and direction-averaged pair quantities. Atom indices choose
only deterministic storage order, not an energetic preference.

## Exact preservation conditions

If `M q <= V-h`, the QP is inactive and returns `b=q` without a numerical
solve. Therefore heavy-heavy radial attraction, heavy topology membership,
the overcoordination correction, and forces reduce exactly to the current
Optimised-Valence result. This is the intended path for H3, methane,
formaldehyde, water, ethane, methanol, hydroxylamine, and peroxide at their
accepted geometries.

H-state energy and every H-containing radial term remain owned by the existing
validated H-state implementation. Raw H taper is retained in the heavy-centre
capacity load so this prototype cannot make H-only crowding energetically free
or introduce a second H-state decision into the shared-edge solve.

## Complexity and production implications

For `E` candidate heavy-heavy edges and `N` heavy atoms, the reference solves a
convex `E`-variable QP and a small active KKT system. It does not enumerate
`2^E` binary edge states, so it avoids the joint-state combinatorial tail.
This SciPy active-set implementation is intentionally unsuitable for
production Torch/GPU batching. A production decision would require an
equivalent batched differentiable QP solver and a separate equivalence audit.

## Falsification gates

Reject this formulation if its apparent gain comes only from deleting the
production penalty; if accepted molecule energies or forces change; if
permutation, finite-difference, continuity, or NVE checks fail; if it creates
new catastrophic Grambow errors; or if improvement is confined to the named
microscope reactions. No parameter may be tuned against Grambow.
