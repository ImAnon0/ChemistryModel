# Heavy-valence state energy prototype v0

## Status

Research-only. It is not registered in `batch_runner.py`, Lab, or the normal
benchmark selector and cannot change default simulation behaviour.

## Existing inconsistency

The production Optimised-Valence model uses smooth local state membership for
heavy-centred angle topology, while radial attraction and the quadratic
overcoordination penalty still use every radial contact. Capacity-rejected
contacts therefore remain attractive and simultaneously create a penalty.

## Prototype equation

For directed contact `i -> j`, let:

- `A_ij = taper_ij * attractive_ij` be the positive magnitude of the current
  Morse attraction;
- `m_ij` be the existing heavy-valence state membership in `[0, 1]`;
- `H_ij` select heavy-heavy contacts.

The current undirected attraction is represented by two half contributions.
The prototype uses:

`E_attr = -1/2 sum_ij H_ij A_ij m_ij`

instead of:

`E_attr,current = -1/2 sum_ij H_ij A_ij`.

This is symmetric after summing both directions. Each centre can contribute
only the attraction admitted by its own capacity-limited state distribution.
No new parameter is introduced.

Heavy-centred effective coordination is:

`c_i = sum_(j heavy) taper_ij m_ij + sum_(j H) taper_ij`.

The existing overcoordination equation and depth scale are then evaluated at
`c_i`. Hydrogen contacts remain unchanged because the H-state model owns their
transfer energetics.

## Preserved behaviour by construction

- All repulsive Morse terms are unchanged.
- H-containing radial attraction and H-state corrections are unchanged.
- With no heavy-valence competition, `m_ij = 1`, so energy and force are
  exactly the production Optimised-Valence result.
- Existing taper, depth, width, environment softening, state Hamiltonian,
  temperature, and membership equations are reused.

## Limitations

This is a mean-state energy, not yet the free energy of a globally coupled
heavy-bond Hamiltonian. Local states at the two ends of a pair are not jointly
correlated. A contact accepted at one end and rejected at the other retains
half attraction. H-heavy competition is intentionally left to later work to
avoid silently changing validated H transfer.

The prototype is acceptable only as a diagnostic candidate. Integration would
require benchmark improvement plus force, continuity, molecule, H-transfer,
and NVE gates.

## Formulation study

Let `A_ij = taper_ij * attractive_ij > 0`. The heavy-centred candidate set
contains both heavy-heavy contacts and H contacts, because both consume heavy
valence. H-state remains the sole owner of H-containing radial energy.

### A. Local mean-state v0

For centre `i`, the existing Hamiltonian produces local state probabilities
`p_i(s)` and directed membership

`m_ij = sum_s p_i(s) 1[(i,j) in s]`.

The heavy-heavy attraction is

`E_A = -1/2 sum_(i,j heavy) A_ij m_ij`.

Thus an undirected edge receives

`E_A,ij = -A_ij (m_ij + m_ji)/2`.

This is a scalar and is symmetric after both directed halves are summed, but
the occupancy is not shared: `m_ij` and `m_ji` may differ. Moreover, v0
differentiates the product of a state-dependent membership and attraction;
it is not the free energy whose derivative generated that membership.

### B. Local free energy

For every competing heavy centre, enumerate the existing size-`V_i` local
states. Define diagonal state energies

`epsilon_i(s) = -sum_(ij selected in s) A_ij`

and use the already established heavy-state scale `tau = 0.01 eV`:

`F_i = -tau log sum_s exp(-epsilon_i(s)/tau)`.

To keep H energy isolated, the heavy-heavy contribution is a conditional
free-energy difference:

`E_B,i = 1/2 [F_i(A_H + A_HH) - F_i(A_H)]`.

The subtraction uses the identical state space, so state entropy and H-only
selection bias cancel. With one allowed state this reduces exactly to the
ordinary half-edge attraction. Its derivative gives a thermodynamically
consistent local occupancy. Endpoint free energies remain independent, so
this formulation still does not impose `m_ij = m_ji`.

For `N` candidates and integer capacity `V`, the local state count is
`C(N,V)` when `N > V`, otherwise one. Examples are `C(5,4)=5`,
`C(6,4)=15`, and `C(7,4)=35`. The existing grouped `(N,V)` execution can
eventually batch this formulation.

As `tau -> 0`, `F_i` tends to the lowest local state energy. Finite `tau`
removes force ambiguity at degenerate preference exchanges. No new smoothing
parameter was introduced.

### C. Joint shared-edge free energy

Construct one candidate variable for each heavy-heavy edge and a local stub
for each H-heavy contact. A global state is a maximal subset satisfying

`sum_(e incident on i) n_e <= V_i`

for every heavy atom, with `n_ij` shared by both endpoints. Maximality prevents
the empty-state pathology and exactly reproduces ordinary attraction when no
capacity conflict exists.

The full and H-reference state energies are

`epsilon(s) = -sum_(HH selected) A_ij - sum_(H stubs selected) A_iH`

`epsilon_H(s) = -sum_(H stubs selected) A_iH`.

For each connected heavy component:

`E_C = F(epsilon) - F(epsilon_H)`.

Every heavy-heavy attraction is counted once, and one occupancy is constrained
at both endpoints. This is a globally coherent finite-temperature
capacity-constrained b-matching partition function. Exact component
enumeration is used only as a slow reference; production execution would need
factorisation or a dedicated batched solver.

### Non-equivalence

Local and joint free energies are equivalent only if the heavy-heavy candidate
graph factorises into independent stars, or if every shared edge has a fixed
occupancy. With two simultaneously competing endpoints, the local partition
function includes combinations in which the two centres disagree about the
same edge. The joint partition function excludes those combinations. The
different Grambow energies and shared-edge occupancies numerically confirm
that the formulations are not generally equivalent.

### Shared overcoordination policy

All three references preserve the full Morse repulsive branch. Heavy-heavy
effective coordination is `sum taper_ij * membership_ij`; H contacts retain
raw taper in the radial overcoordination term so the investigation does not
silently take ownership of validated H-state energy. This isolation is a
deliberate boundary and is also a remaining limitation for a future unified
global valence Hamiltonian.
