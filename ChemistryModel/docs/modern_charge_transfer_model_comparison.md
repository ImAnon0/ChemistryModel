# Modern restricted charge-transfer model comparison

## Decision

The 2025 bond-capacity electronegativity-equilibration model (EEQBC) is the
first investigated modern model that clears the public H/C/N/O parameter,
topology-free geometry, analytical-gradient, and practical-solver gates at the
same time. It is a strong **diagnostic/reference-charge model**.

It does not, however, clear ChemistryModel's strict dissociation gate. EEQBC
deliberately suppresses rather than mathematically eliminates inter-fragment
charge transfer. Its authors state that the global Lagrange multiplier leaves
a small second-order coupling, limited artificial long-range transfer, and the
non-linear polarizability scaling inherited from EEQ. That is substantially
better than QEq, but it is not the exact independently neutral fragment limit
already required for ChemistryModel.

The model search therefore stops here. No implementation or fit is started.

**Recommendation: give SQE a dedicated ChemistryModel H/C/N/O
quantum-reference parameterization project**, including a smooth reactive
transfer-compliance law. SQE has the clearest exact localization mechanism;
the missing work is explicit and testable. EEQBC should remain the strongest
publicly parameterized comparator for that project, not the production
foundation.

## EEQBC source and convention

Primary source: Froitzheim, Muller, Hansen and Grimme, *The bond capacity
electronegativity equilibration charge model (EEQBC) for the elements Z =
1-103*, J. Chem. Phys. **162**, 214109 (2025), DOI 10.1063/5.0268978.
Equations, analytical nuclear derivatives, supporting data, and an open-source
implementation are publicly available.

The model retains the constrained variational EEQ energy

```
E(q) = q^T (1/2 A q - b)

[ A   1 ] [q]      [b]
[1^T  0 ] [lambda] [Q_total]
```

but modifies both `A` and `b` with a distance-dependent Maxwell capacitance
matrix `C`. In the paper's notation,

```
A = I + C .* J
b = C (chi + psi)
```

where `.*` is the symmetric Hadamard product and `J` is the Gaussian-charge
Coulomb/hardness matrix. The off-diagonal capacities are smooth all-pair
functions,

```
C_ij = -sqrt(xi_i xi_j)
       [1 + erf(-k_BC (r_ij - RvdW_ij) / RvdW_ij)]
```

with `C_ii = -sum_(j != i) C_ij`. Coordination-dependent atomic radii,
hardnesses, and electronegativities add local geometry response. The exact
definitions and charge/energy nuclear derivatives are supplied in the
supporting information and implementation.

## Requirements audit

| ChemistryModel requirement | EEQBC finding | Verdict |
|---|---|---|
| complete H/C/N/O parameters | Published implementation contains eight element-wise parameters for every element Z=1-103 plus four global parameters | **Pass** |
| public, coherent provenance | One joint fit; values are available in the open-source implementation | **Pass** |
| separated neutral fragments | First-order electronegativity driving term decouples; global constraint leaves small second-order transfer | **Improved, not exact** |
| no fixed molecular topology | `C_ij` is constructed from element identities and continuous interatomic distances for all pairs | **Pass** |
| arbitrary reactive geometries | No Lewis structure, molecule identity, or predefined bond list is required | **Pass in declared domain** |
| smooth contact formation/breaking | Error-function capacities, smooth coordination numbers, Gaussian Coulomb kernel | **Pass mathematically** |
| charge equations available | Symmetric `(N+1)` constrained linear system is published | **Pass** |
| nuclear gradients available | Analytical electrostatic-energy and atomic-charge derivatives are derived and implemented | **Pass** |
| exact total charge | Enforced by one Lagrange constraint | **Pass globally** |
| exact fragment charge at dissociation | Not independently constrained; residual coupling is acknowledged | **Fail for strict requirement** |
| MD conditioning evidence | Symmetric variational system is favorable, but the publication does not provide the condition-number/endurance evidence required for reactive MD | **Promising, unproven** |
| Torch/GPU batching | Dense all-pair construction and batched symmetric solves map directly to Torch; `N+1` variables | **Practical for small/medium systems** |

### Parameter provenance

EEQBC uses eight empirical parameters per element:

- base hardness `eta0`;
- local-charge hardness response `k_eta`;
- base Gaussian radius `a0`;
- base electronegativity `chi0`;
- coordination electronegativity response `k_chiCN`;
- local-charge electronegativity response `k_chiq`;
- elemental bond capacity `xi`;
- covalent radius used for coordination number.

It additionally uses four global fitted constants controlling radius response,
coordination normalization, local-charge construction, and capacity decay.
The parameters were fitted jointly to Hirshfeld charges computed primarily at
`omegaB97M-V/def2-TZVPPD` (with a different declared basis for actinides).
They are therefore complete and usable for H/C/N/O, but they reproduce a
specific population-analysis and electronic-structure convention. They are
not experimental elemental constants and should not be mixed with QEq,
QTPIE, ACKS2, or SQE values.

The authors demonstrate broad molecular transferability, including random
PubChem and deliberately unusual structures. That supports use on arbitrary
geometries more strongly than a molecule-specific fit, but it does not prove
accuracy for ChemistryModel's hot radical, collision, and transition-state
distribution. Such states would need independent hold-out validation.

## Dissociation: why it is better than QEq but not exact

As two fragments separate, their cross-fragment capacities decay smoothly.
Because `b = C chi` and every isolated fragment's capacitance block has zero
row/column sum, the direct electronegativity drive for inter-fragment transfer
vanishes. This cures the dominant first-order QEq pathology without a fixed
fragment graph.

EEQBC retains one global total-charge multiplier, however. Its second-order
charge response is not independently charge-conserving per disconnected
fragment. The paper explicitly describes the remaining long-range transfer as
small or negligible rather than zero, and notes that non-linear polarizability
scaling remains. Thus:

- QEq: finite, generally substantial asymptotic transfer;
- EEQBC: strongly attenuated, small residual asymptotic/global coupling;
- QTPIE/SQE: transfer tends to exactly zero under their defining localization
  construction.

For an initial charge guess, dispersion correction, or many equilibrium
molecular uses, EEQBC's approximation can be entirely reasonable. For a
reactive MD model whose explicit acceptance rule is independently neutral
separated fragments, “negligible” is not interchangeable with “zero”.

## Smoothness and reactive geometry

EEQBC is more naturally compatible with topology-changing geometry than
published SQE:

- every pair is treated through a continuous distance function;
- there is no discrete bond classification;
- capacity appears and disappears through an error function;
- local environment dependence uses continuous coordination descriptors;
- Gaussian Coulomb interactions remain finite at short range.

No graph event is needed when atoms collide or separate. This is a genuine
architectural advantage for ChemistryModel. It also means the dense reference
model is `O(N^2)` to build and conventionally `O(N^3)` to solve. Batched small
systems are straightforward on a GPU. Large sparse scaling is less automatic:
the published error-function capacities are mathematically nonzero at finite
distance, and introducing a hard cutoff would require a smooth truncation and
new validation.

The constrained matrix is symmetric, unlike the original nonvariational BC
system. If `C` and the Gaussian Coulomb/hardness kernel are positive
semidefinite, the Schur-product construction plus the identity term gives a
well-behaved positive charge block; the augmented constraint system remains
indefinite in the standard KKT sense. This is structurally suitable for direct
float64 solves. The publication reports efficiency and large-system timings,
but does not establish condition-number distributions, trajectory-to-
trajectory continuity, force drift, or long NVE stability. Those would remain
mandatory before MD coupling.

## Direct comparison

| Model | H/C/N/O provenance | Exact neutral dissociation | Fixed topology | Smooth reactive geometry | Gradients | Numerical/production outlook |
|---|---|---|---|---|---|---|
| QEq | complete | **No** | no | yes | available | simplest and GPU-friendly; scientifically rejected |
| historical Slater QTPIE | complete published convention | **Yes** | no in atom-space form | overlap-smooth | derivable | investigated convention has negative/ill-conditioned modes |
| corrected Gaussian QTPIE | complete diagnostic convention | **Yes** | no | overlap-smooth | derivable | useful diagnostic; key molecular conditioning/polarity failures |
| original ACKS2 | no transferable H/C/N/O table | theoretical yes | no | response-kernel dependent | available in theory | blocked at provenance gate |
| ReaxFF ACKS2 | only audited public H/O set | yes for its response construction | distance graph | smooth within its fitted convention | implemented | H/C/N/O provenance blocked |
| published SQE | H/C/O/Si, missing N and N-pairs | **Yes** | **yes** | no, not as published | available | best localization mechanism; needs reactive law and parameters |
| EEQBC (2025) | **complete Z=1-103** | **No: residual is acknowledged** | **no** | **yes** | **published and implemented** | strongest ready comparator; misses strict fragment limit |

## Why the next project should be SQE parameterization

No investigated off-the-shelf model simultaneously provides:

1. a complete coherent H/C/N/O convention;
2. exact independent neutral-fragment localization;
3. topology-free smooth reactive channels;
4. demonstrated stable molecular polarity and conditioning; and
5. analytical MD-ready energy derivatives.

SQE is the best mathematical starting point for a dedicated project because
its localization proof is simple and exact: split-charge compliance between
fragments vanishes, so each disconnected component conserves its charge. Its
known deficits are concrete rather than hidden—a missing nitrogen/pair fit and
a published fixed-topology transfer network.

The parameterization project should jointly define H/C/N/O atomic response,
all required pair transfer capacities/hardnesses, Coulomb screening, and a
smooth distance-dependent reactive compliance. It should use a declared
quantum level and charge-partition convention, train on molecules, radicals,
distorted structures, transition-like contacts, and separated neutral
fragments, and reserve independent polarity/dissociation sets as hold-outs.
Adopting a smooth all-pair law would be a ChemistryModel-specific reactive SQE
extension and must be labelled accordingly.

EEQBC should be implemented only if a future task explicitly requests a
standalone comparator. Its public H/C/N/O parameters and smooth topology-free
construction make it an excellent benchmark against which the new SQE fit can
be judged, but its residual asymptotic transfer prevents recommending it now
as the production electrostatics foundation.

