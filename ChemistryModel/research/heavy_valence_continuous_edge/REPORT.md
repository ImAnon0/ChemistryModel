# Continuous shared-edge capacity report

## Decision

**DO NOT INTEGRATE v0.** The continuous shared-edge QP is mathematically and
numerically healthier than binary shared-edge enumeration, and it remains a
useful research direction. It does not clear the production evidence gate:
barrier MAE/RMSE are worse than local v0, reaction metrics are worse, and
barrier-sign agreement falls from 95.5% to 94.0%. No parameter was tuned and no
production physics was changed.

## Prototype tested

The complete formulation is in `DESIGN.md`. In brief, every undirected
heavy-heavy contact owns one continuous taper-weighted capacity `b_ij`, shared
at both endpoints and constrained by elemental valence after the inherited H
topology load. The accepted current bond order is `q_ij=taper*order`. The
capacity-bearing attraction is the parameter-free convex completion

`E_ij=-A_ij[2(b/q)-(b/q)^2]`.

It is zero at no capacity, exactly recovers current attraction and its first
derivative at `b=q`, and preserves the full Morse repulsive branch. A float64
SLSQP solve identifies the active set; Torch reconstructs the KKT solution so
the final scalar remains differentiable. This is a slow standalone reference,
not a proposed production implementation.

## Frozen Grambow comparison (200 reactions)

| formulation | barrier MAE | barrier RMSE | max barrier error | sign | reaction MAE | reaction RMSE | max reaction error |
|---|---:|---:|---:|---:|---:|---:|---:|
| production | 4.5195 | 6.4831 | 35.5021 | 87.0% | 4.4058 | 7.0688 | 49.3746 |
| local mean-state v0 | **2.1161** | **2.7611** | 10.6982 | **95.5%** | **2.7231** | 3.4901 | 10.0616 |
| joint binary edge | 2.3448 | 3.3621 | 12.9295 | 95.5% | 2.7278 | **3.4396** | 10.0853 |
| continuous shared edge | 2.2595 | 2.9019 | **10.0430** | 94.0% | 2.8757 | 3.6230 | 10.0924 |

Against production, continuous edge improved/worsened/left unchanged 120/62/18
barriers, removed all 16 production barrier errors above 10 eV, and introduced
no new barrier error above 10 eV. It left one pre-existing >10 eV barrier
(`rxn008195`, 10.0430 eV). Against v0 it improved/worsened/left unchanged
78/104/18 barriers. Its 12 sign failures, versus nine for v0, are a failed
acceptance gate even though its error tail is narrower.

The reaction result is not better than v0: 2.8757 versus 2.7231 eV MAE and
3.6230 versus 3.4901 eV RMSE. One reaction error crosses 10 eV relative to v0
(`rxn007313`, -10.0924 eV); it was already worse than 10 eV in the local free
energy reference and is not a production integration regression because the
prototype is unreachable from production.

## Required microscopes

Barrier errors in eV:

| reaction | production | v0 | joint binary | continuous |
|---|---:|---:|---:|---:|
| rxn006559 | -35.5021 | +0.0196 | -2.4616 | -1.1849 |
| rxn011804 | +26.0104 | +4.1819 | +4.4983 | +3.0301 |
| rxn004353 | +22.7650 | +2.8944 | +2.1797 | **+0.1983** |
| rxn000096 | -21.9676 | -0.0268 | +0.1665 | -0.5363 |
| rxn010742 | -19.5727 | +1.2060 | -0.4546 | +0.5797 |
| rxn000105 | **-0.2045** | +6.3042 | +6.4536 | +8.2091 |

The first five confirm that a shared continuous capacity removes the dominant
attraction/penalty inconsistency without binary-state tail failures. The
`rxn000105` C-C-breaking case becomes still worse than v0. Production's near
zero error there is known to be accidental cancellation, but continuous v0
does not supply a better physical description of the missing stabilization;
that is evidence against integration, not a reason to preserve the production
penalty.

## Independent QM microscopes

The 106-geometry / 98-dense independent residual set was rerun.

| system | production RMSE | v0 RMSE | joint RMSE | continuous RMSE |
|---|---:|---:|---:|---:|
| H3 | 0.2231 | 0.2231 | 0.2231 | 0.2231 |
| methane | 0.2184 | 0.2184 | 0.2184 | 0.2184 |
| formaldehyde | 0.3810 | 0.3810 | 0.3810 | 0.3810 |
| water transfer | 0.9867 | 0.2066 | 0.2018 | **0.2012** |
| all dense | 0.5874 | 0.2638 | 0.2627 | **0.2626** |

Continuous edge preserves the noncompeting systems exactly and retains nearly
all of the water-transfer improvement. It is marginally best on this compact
QM residual set, but the difference from free/joint is only about 0.0001 eV
RMSE and does not outweigh the independent Grambow regression.

## Force, continuity, and chemistry gates

Fourteen focused tests passed. They cover:

- exact energy/force preservation for H3, methane, formaldehyde, water,
  ethane, methanol, hydroxylamine, and peroxide;
- one shared capacity at both endpoints and exact endpoint capacity bounds;
- atom-permutation symmetry;
- float64 autograd versus central finite-difference force in both onset and
  strong competition;
- continuous energy and force through an equivalent-contact preference
  crossing;
- explicit research-only status.

No molecule-specific rule or fitted parameter was introduced.

## Matched NVE

All runs had zero move caps and finite forces.

| case | production max drift | v0 | joint | continuous |
|---|---:|---:|---:|---:|
| water transfer, 250 x 0.25 fs | 9.464e-3 | 7.109e-3 | 5.118e-3 | 5.932e-3 |
| rxn000105 crowded reactant, 250 x 0.10 fs | 5.616e-3 | 8.973e-5 | 6.659e-5 | 6.224e-5 |
| symmetric exchange, 300 x 0.02 fs | 6.465e-5 | 2.673e-6 | 1.105e-6 | 2.472e-6 |

The active-set/KKT force is conservative in these probes. The reference uses
SciPy and is deliberately not a performance candidate.

## Regression and golden validation

- Focused continuous-edge tests: `14 passed`.
- Full pytest with a workspace-local temp root: `292 passed, 1 skipped`.
- `validation_report.py --full`: all 20 named regression checks PASS.
- Dense unchanged production stress: PASS, 2 seeds x 330 atoms x 1 ps;
  zero final C/N over-valence in both seeds.
- Final golden result: PASS.

The initial pytest attempts produced fixture setup errors only because the
default Windows temp directory was inaccessible (and the first workspace temp
parent had not yet been created). The corrected run had no test failures.

## Interpretation and next research step

The shared convex QP solves the architectural consistency problem: attraction,
capacity, and effective bond order arise from one undirected variable; state
count no longer grows exponentially; repulsion and accepted chemistry are
preserved; and forces are stable. It also confirms that binary edge occupancy
was too rigid.

Its weakness is now specific: scaling an entire multi-bond edge by one
quadratic occupancy over-stabilizes some partial/crowded transition states and
does not represent separate sigma/pi capacity or delocalised multi-centre
sharing. That shows up as five new negative-barrier sign failures versus v0,
the `rxn000105` worsening, and poorer ensemble barrier/reaction metrics despite
the marginal improvement on the much smaller QM residual set.

If this line of research continues, the next hypothesis should be a shared
incremental-capacity model (separate continuous first/second/third bond-order
channels with one edge identity), or an explicitly delocalised component
capacity functional. It should be derived before looking at benchmark scores;
no coefficient in the current quadratic should be tuned to Grambow.
