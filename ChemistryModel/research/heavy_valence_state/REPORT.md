# Heavy-valence competition formulation study

## Decision

**C — the rigorous formulation exposes that the original hypothesis is
incomplete. Do not integrate any candidate yet.**

The original attraction/overcoordination inconsistency is real. All three
research formulations remove its catastrophic baseline tail and improve the
independent water microscope. The exact shared-edge model is the most
internally coherent, permutation-symmetric scalar energy, but it creates a
new systematic high-barrier tail in multi-centre C/N/O networks. The existing
binary contact-capacity state model is therefore not yet a defensible global
heavy-bond Hamiltonian, even when its bookkeeping is made exact.

No production physics, selector, force-field parameter, equation, or H-state
implementation was changed.

## 1. Formulations tested

Detailed equations are in `DESIGN.md`.

- **Production:** full Morse attraction for every radial contact plus raw
  radial heavy-overcoordination; heavy membership affects topology/angles.
- **Local mean-state v0:** each heavy centre independently weights its half of
  heavy-heavy attraction by existing membership. It is a differentiable scalar
  but endpoint occupancies may disagree and its energy is not the free energy
  that generated the membership.
- **Local free energy:** explicit `-tau logsumexp(-E_s/tau)` over each centre's
  local states, using the existing 0.01 eV heavy-state scale. An H-only
  reference partition function is subtracted so H radial energy remains owned
  by H-state.
- **Joint edge:** exact finite-temperature partition function over maximal
  capacity-constrained shared-edge states in each connected heavy component.
  One heavy-heavy occupancy is constrained simultaneously at both endpoints
  and its attraction is counted once.

The local and joint free energies are not generally equivalent. They coincide
only where shared occupancies are fixed or the graph factorises into independent
stars. Simultaneously competing endpoints produce different state spaces and
different measured energies.

## 2. Frozen Grambow benchmark

All formulations evaluated the same 200 reactions with zero failures.

| metric | production | mean v0 | local free energy | joint edge |
| --- | ---: | ---: | ---: | ---: |
| Barrier MAE | 4.5195 | **2.1161** | 2.3410 | 2.3448 |
| Barrier RMSE | 6.4831 | **2.7611** | 3.1728 | 3.3621 |
| Barrier median AE | 3.1874 | **1.6262** | 1.7643 | 1.6715 |
| Barrier max AE | 35.5021 | **10.6982** | 10.7133 | 12.9295 |
| Barrier signed mean | +1.8843 | +0.7353 | +0.8902 | +0.9164 |
| Reaction MAE | 4.4058 | 2.7231 | 2.8217 | **2.7278** |
| Reaction RMSE | 7.0688 | 3.4901 | 3.6153 | **3.4396** |
| Reaction median AE | 3.0473 | **2.2186** | 2.3232 | 2.2628 |
| Reaction max AE | 49.3746 | 10.0616 | 10.3609 | 10.0853 |
| Reaction signed mean | -0.1387 | +0.7619 | +0.7052 | +0.7094 |
| Barrier sign agreement | 87.0% | **95.5%** | 95.0% | **95.5%** |

Against production, barrier errors improved/worsened/unchanged in:

- v0: 119 / 63 / 18;
- local free energy: 113 / 69 / 18;
- joint edge: 115 / 67 / 18.

Reaction errors improved/worsened/unchanged in:

- v0: 99 / 65 / 36;
- local free energy: 96 / 68 / 36;
- joint edge: 96 / 68 / 36.

All three remove 16 old barrier errors above 10 eV. New barrier errors above
10 eV relative to production are 0 for v0, 2 for local free energy, and 5 for
joint edge. The joint model's five new >10 eV cases are the decisive failed
tail gate.

### Barrier absolute-error distribution

| percentile | production | mean v0 | local free energy | joint edge |
| --- | ---: | ---: | ---: | ---: |
| 50% | 3.187 | **1.626** | 1.764 | 1.672 |
| 75% | 6.152 | **3.093** | 3.226 | 3.166 |
| 90% | 9.258 | **4.636** | 5.002 | 4.805 |
| 95% | 11.768 | **5.394** | 6.399 | 6.226 |
| 99% | 22.798 | **6.967** | 10.116 | 12.620 |
| max | 35.502 | **10.698** | 10.713 | 12.929 |

The improvement is broad through the 95th percentile and also eliminates old
catastrophes. It is not merely moving one or two outliers. The rigorous
formulations nevertheless reintroduce a distinct extreme tail.

## 3. The 63 v0 barrier regressions

Severity by increase in absolute error:

- minor, <=0.5 eV: 35;
- moderate, 0.5–2 eV: 21;
- major, 2–5 eV: 6;
- new catastrophic/outlier, >5 eV: 1.

Seventeen have no net >0.5 topology edge change. Across the inferred bond
changes, C-C is involved 50 times, C-N 20, C-O 17, N-O 5, and N-N once.
Forty-three have at least one transition-state heavy edge for which both
endpoints compete. Fifty-seven have endpoint membership asymmetry above 0.5.
This is not one clean reaction family: it is concentrated in crowded,
multi-centre C/N/O networks where independent endpoint assignment is most
questionable.

The complete per-reaction forensic table records composition, inferred bond
changes, membership asymmetry, attraction correction and overcoordination
changes.

## 4. New 6.30 eV v0 outlier

`rxn000105` is a C5H7N C-C breaking reaction.

- production barrier error: -0.2045 eV;
- v0 barrier error: +6.3042 eV;
- absolute-error regression: 6.0997 eV.

The reactant contains four carbon centres with raw radial coordination about
4.544. Production assigns about 2.30 eV heavy-overcoordination to each, or
9.2096 eV total. V0 recognises their excess proximity as capacity-rejected,
sets their effective heavy overcoordination to zero, and retains 2.7362 eV as
rejected-attraction energy. The transition state has only 0.0031 eV production
heavy-overcoordination and 0.0383 eV rejected attraction.

Consequently the overcoordination replacement raises the barrier by about
9.207 eV while the attraction replacement lowers it by about 2.698 eV: net
approximately +6.509 eV. The outlier is therefore not a numerical failure or
one rogue asymmetric edge. Production happened to obtain a good barrier from
a large crowded-reactant penalty that the corrected bookkeeping removes. This
reveals missing compensating physics or an inadequate capacity representation;
retaining the old penalty solely to recover this score would be unjustified.

## 5. Why the joint model's tail worsens

The new joint-edge >10 eV barrier cases are `rxn011246`, `rxn005670`,
`rxn008551`, `rxn000920`, and `rxn011847`. Their transition structures contain
several simultaneous C-C/C-N/C-O/N-N contacts and strongly inconsistent local
endpoint memberships. The joint model correctly removes locally inconsistent
state combinations, but the remaining binary, unit-capacity edge states raise
these transition energies by roughly 2–5 eV relative to v0.

This is evidence that a global binary contact count is too restrictive for
delocalised, ring-like, multiple-bond and multi-centre transition structures.
The current local state machinery uses integer elemental contact capacity even
though radial pair depth/order already distinguishes multiple-bond character.
Making those binary states globally consistent exposes that mismatch rather
than curing it.

## 6. Independent QM microscopes

| system RMSE | production | mean v0 | local free energy | joint edge |
| --- | ---: | ---: | ---: | ---: |
| H3 | 0.2231 | 0.2231 | 0.2231 | 0.2231 |
| methane | 0.2184 | 0.2184 | 0.2184 | 0.2184 |
| formaldehyde | 0.3810 | 0.3810 | 0.3810 | 0.3810 |
| water transfer | 0.9867 | 0.2066 | **0.2018** | **0.2018** |
| all dense scans | 0.5874 | 0.2638 | **0.2627** | **0.2627** |

All formulations leave H3, methane and formaldehyde unchanged. Water's worst
adjacent residual step improves from 0.8505 eV to 0.1811 eV in every
competition formulation. Independent evidence therefore supports the core
capacity-limited-attraction hypothesis, but does not distinguish local from
joint treatment on the available microscopes.

## 7. Molecules, forces and continuity

H2, CH4, H2O, ethane, methylamine, methanol, hydrazine, hydroxylamine and
hydrogen peroxide retain current energies and forces at float64 tolerances
when no heavy competition exists.

For both rigorous formulations:

- atom permutation leaves energy invariant and permuted forces identical;
- autograd forces match central finite differences within 1e-5 eV/A at
  competition onset, strong competition and a shared mixed H/heavy edge;
- energy, force and occupancy remain finite through preference exchange;
- a resolved +/-1e-6 A equivalent-contact crossing has a continuous force
  limit below the 0.01 eV/A test gate.

The preference exchange is steep at `tau=0.01 eV`, especially for the exact
joint model, but it is smooth rather than a label-swap cusp.

## 8. NVE dynamics

Maximum absolute total-energy drift, with zero move caps in every run:

| case | production | mean v0 | local free energy | joint edge |
| --- | ---: | ---: | ---: | ---: |
| water transfer, 62.5 fs | 9.46e-3 | 7.11e-3 | 5.85e-3 | **5.12e-3** |
| crowded Grambow reactant, 25 fs | 5.62e-3 | 8.97e-5 | **6.66e-5** | **6.66e-5** |
| symmetric dominance exchange, 6 fs | 6.47e-5 | 2.67e-6 | 2.74e-6 | **1.11e-6** |

No collapse, non-finite state, force discontinuity or move-cap event occurred.
The joint reference is dynamically conservative on the focused cases.

The exact joint enumerator was intentionally not wired into the 2 x 330 atom
production runner: it is a correctness reference with exponential worst-case
state enumeration, and it had already failed the external tail gate. The
unchanged production engine's established 2 x 330 atom, 1 ps stress remains a
golden PASS.

Repository-wide pytest after adding the formulation references and guards:
**278 passed, 1 skipped**. The only warning was the existing inability to write
`.pytest_cache`; it did not affect test execution. `git diff --check` is clean.
The immediately preceding full golden validation on this unchanged production
engine passed all 20 checks and the dense production stress. It was not rerun
after the research-only references failed the external decision gate, because
they are not reachable from production execution.

## 9. Recommendation

**Do not integrate yet.**

Do not choose v0 solely for its best benchmark score: its endpoint occupancies
are not globally coherent. Do not choose the current joint model solely for
its superior mathematics: its new multi-centre barrier tail demonstrates that
the binary unit-edge capacity model is incomplete.

The next experiment should stay research-only and generalise the shared-edge
state variable from binary contact occupancy to bond-order/electron-pair
capacity. It must use existing continuous bond order and elemental valence,
preserve one shared edge energy, keep H-state ownership explicit, and reduce to
the present accepted-molecule limit. Compare that model specifically on the
five joint-tail reactions, `rxn000105`, the frozen QM microscopes and the same
force/NVE gates before repeating the full 200-reaction decision table.
