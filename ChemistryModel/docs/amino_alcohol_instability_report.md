# Amino-alcohol batch instability forensic report

## Verdict

The reported 11/16 instability rate is a **stability-detector false-positive caused by intentional strike energy injection**. All 15 detected energy jumps occur in the first fixed-cadence frame after one of the ten 30,000 K strikes (strike time + 10 fs). No threshold-crossing jump occurs during ordinary chemistry. Across all seeds, the largest positive 10 fs energy change outside the strike train is only 4.3 eV.

This evidence supports **STABILITY-DETECTOR FIX / NO PHYSICS CHANGE** as the next action. It does not establish a force-field defect or timestep failure. The strike protocol is deliberately violent and does create large displacements and move-cap activity, but those are distinct from why these runs were labelled unstable.

## Batch and timing

- 16/16 runs finished; 5 labelled stable and 11 labelled unstable.
- Labelled stable: seeds 5, 7, 8, 9, 14.
- Labelled unstable: seeds 0, 1, 2, 3, 4, 6, 10, 11, 12, 13, 15.
- 15/15 threshold events are at 15010--15910 fs, exactly 10 fs after a scheduled strike at 15000--15900 fs.
- 0/15 events occur outside the strike train.
- The legacy recorder writes every 10 fs. The simulation applies a strike after capturing the strike-time frame, so its effect first appears in the following recorded frame.

The detector threshold is `max(80 eV, 8% of |final potential energy|)`. Consequently, almost identical intentional deposits straddle a classification boundary: stable seed 7 reaches 91.4 eV and seed 9 reaches 88.2 eV, while seed 3 is marked unstable at 94.2 eV. This is not a meaningful physical separation.

## Event character

The jumps are predominantly kinetic injection followed by a smaller potential response. Representative cases are:

| seed | recorded time (fs) | delta total (eV) | delta potential (eV) | delta kinetic (eV) | temperature K, before to after |
|---:|---:|---:|---:|---:|---:|
| 1 | 15310 | 120.5 | 34.5 | 85.9 | 470 to 2496 |
| 1 | 15510 | 100.0 | 30.1 | 69.9 | 727 to 2375 |
| 2 | 15910 | 95.9 | 33.6 | 62.3 | 721 to 2190 |
| 12 | 15510 | 157.3 | 57.6 | 99.7 | 443 to 2794 |

The largest-displacement atom in these cases is hydrogen (3.7--6.0 A over the recorded 10 fs interval). Reconstructed pre-event maximum forces are finite, about 10.7--28.9 eV/A in these representative events. Energy decomposition contains the existing Morse/bond, over-coordination, and angle terms; no singular or non-finite term was found. The potential increase is distributed over the post-strike evolution rather than attributable from the saved cadence to one pathological pair.

The closest pairs around every event are H-H at roughly 0.596--0.706 A. These are mostly compact H2-like bonded contacts, not evidence by themselves of atoms crossing a repulsive wall; the current H-H inner cutoff is 0.927 A and the equilibrium H-H distance is about 0.74 A. The globally closest pair is frequently not the strongest-force atom or the atom with the largest displacement. Therefore the recordings do not support assigning the jumps to compressed H-H collisions.

No disproportionate signature implicates C-C taper behavior, N-N, N-O, O-O, environment softening, over-coordination, angle collapse, or the high-fidelity H-transfer correction. The observed scalar jump is explained directly by the discharge operation, which replaces/blends velocities in a random channel and can add bond-dissociation kicks.

## Move-cap interpretation

`limit_move` caps a proposed per-integration-step atomic displacement at 0.15 A. It is a protective integration limiter and changes the realized dynamics when activated. Scaling a proposed displacement down can remove kinetic energy; it is not an energy source. A cap is a warning about a fast proposed move, not synonymous with run failure.

The batch itself demonstrates the distinction:

- seed 2: one labelled jump, zero caps;
- seeds 7 and 8: zero labelled jumps, 48 and 42 caps;
- seed 1: two labelled jumps, 121 caps.

Thus caps are neither necessary nor sufficient for the unstable label. Legacy recordings store only the final aggregate cap count, not cap time or atom identity, so whether a particular cap occurred immediately before or after a particular strike cannot be recovered honestly from these files.

## Case studies

### Seed 2 -- clean classification case

The only event is 15910 fs, 10 fs after the final strike: +95.9 eV total, comprising +62.3 eV kinetic and +33.6 eV potential, with temperature rising from 721 K to 2190 K. There are zero move caps. The pre-event reconstructed maximum force is finite (18.6 eV/A). This is the clearest proof that the unstable label can be produced solely by counting an intentional strike deposit as an unexplained integration jump.

### Seed 1 -- high-cap case

Events occur at 15310 and 15510 fs, each 10 fs after strikes, at +120.5 and +100.0 eV. The 121 caps cannot be temporally localized from the legacy file. The second event has the largest representative pre-event force (28.9 eV/A), but it remains finite. Seed 1 may have experienced a longer violent post-strike cascade, yet its two detector events have the same direct strike timing as seed 2.

### Seed 12 -- largest jump

The +157.3 eV event occurs at 15510 fs, 10 fs after a strike. It consists of +99.7 eV kinetic and +57.6 eV potential, with temperature rising from 443 K to 2794 K. The largest 10 fs displacement is a hydrogen atom at the recorder's periodic-distance ceiling (6.0 A); the pre-event maximum force is only 15.3 eV/A. This looks like a many-atom discharge deposit and subsequent response, not a single divergent compressed pair.

## Stable controls

Stable seeds encounter the same strike schedule and large deposits. Their maximum strike-adjacent rises are 55.7 (seed 5), 91.4 (7), 83.6 (8), 88.2 (9), and 87.7 eV (14). Seeds 7 and 8 also have substantial cap activity. The decisive difference is simply whether a strike-dependent rise crosses the run-specific analyser threshold. It is not the presence of strikes, caps, or a uniquely identified chemical motif.

## What the legacy recordings cannot establish

The files preserve positions, velocities, atom identities, box, energies, temperature, and fixed frame times. They do **not** preserve strike channel membership/origin/axis, per-step forces, cap timestamps/atoms, integration warnings, or states inside each 10 fs recording interval. Consequently:

- directly energized atom IDs cannot be recovered reliably;
- bond formation/breaking within the unseen 10 fs interval cannot be ordered relative to the deposit;
- cap clustering and cap energy effects cannot be reconstructed;
- an exact deterministic replay of the original strike and thermostat trajectory cannot start from the saved pre-strike frame;
- a timestep or reduced-strike comparison from that frame would be a different stochastic event, not a controlled reproduction of this one.

For those reasons, no claimed timestep-sensitivity or strike-sensitivity result is manufactured from non-equivalent states. The existing evidence already answers the classification question without changing the simulator.

## Chemistry assessment

The recordings show extensive fragmentation/recombination and mixed CHNO connectivity under the very energetic protocol, including C-C, C-N, and C-O incorporation. Formula strings alone are insufficient to identify methanol, hydroxylamine, methylamine, ethylene glycol, or ethanolamine. A defensible named-structure inventory requires persistent bond connectivity and geometry over time; it is independent of the instability label and was not inferred from formulas in this diagnosis.

## Classification and next action

- 11/11 labelled-unstable seeds: **F -- stability-detector false positive**, directly evidenced.
- 0/11: proven force-field pathology.
- 0/11: proven timestep/integrator failure.
- Strike protocol: causal for the energy deposits, but not evidence that the resulting dynamics numerically failed.
- Move-cap behavior: a separate diagnostic concern with insufficient legacy timing data for event attribution.

Recommended next action: teach stability analysis to distinguish declared external energy-injection windows from spontaneous energy creation, while continuing to report the size of strike deposits and move caps separately. That should be designed and tested in a separate change; this forensic task does not alter thresholds or analysis logic.

No force-field, integrator, strike, recorder, or analysis code was changed. No bad-state regression fixture was added because the recordings reveal a missing event annotation/classification distinction, not a reproducible pathological state. The machine-readable reconstruction is in `docs/amino_alcohol_instability.json`, generated by the read-only `forensic_instability.py` utility.
