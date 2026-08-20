# Reaction teacher-data production

`python -m chemistry_manager produce_reactions` creates small, targeted
full-ChemistryModel collision experiments for a future discovery surrogate.
It does not train a model and does not alter the reactive potential.

Its default teacher is `optimised-valence`: the current validated production
engine with factorised/grouped/cached H-state physics, batched heavy-valence
membership, and the device-aware neighbour gather backend. This is distinct
from characterisation's historical `standard` model and its older experimental
`high_fidelity` H-transfer model. Research/debug runs may override the teacher
with `--physics standard` or `--physics high_fidelity`; manual Molecule Lab
defaults are unchanged.

## Shared collision semantics

The producer calls `characterisation_runner.run_group`; it does not implement
another simulator. Reactant A is initially centred but is **not fixed**. Both
reactants keep their internal thermal motion, random object centre-of-mass
drift is removed, and equal/opposite approach momentum is assigned by mass.
All atoms then evolve under the selected characterisation physics.

The existing seeded random rotations, exposed-target ray selection, safe
starting-separation checks, recorder, bond tracker, outcome analysis, and
molecule scanner remain authoritative. A non-negative impact parameter adds a
seeded offset perpendicular to the existing approach ray. Zero preserves the
historical direct-impact geometry.

## Reactant trust gate

H, C, N, and O atoms are always eligible. A stored molecule is eligible only
when either:

- its metadata explicitly says `CM_VALIDATED` or `QM_VALIDATED`; or
- an existing completed QM validation preserved connectivity and did not
  fragment or rearrange the molecule.

Failed QM jobs do not imply rejection. Discovered products remain untrusted
until they pass this gate. Their full-CM trajectory frames are still valid
teacher labels for learning the current model.

## Reproducibility and layout

A master seed deterministically generates a stratified set of reactant order,
direct/glancing/near-miss class, speed class, temperature, target, rotations,
approach direction, start gap, and simulation seed. The production identity
also includes reactant geometry hashes, the trusted pool, frame-selection
settings, device, physics mode, and a hash of the relevant local physics
sources. The manifest additionally records the concrete runtime physics model
name and revision plus the ChemistryModel Git revision, so datasets from
different backends cannot share a production identity silently.

Each production is append/resume friendly:

```text
teacher_data/PROD_<identity>/
    production.json
    experiments/EXP_<identity>.json
    recordings/index.json
    recordings/EXP_<identity>.npz
    shards/EXP_<identity>.npz
```

Each generic NPZ shard contains elements, positions, full-model forces,
potential energy, kinetic and total MD energy, time, temperature, box size,
contact diagnostics, and frame-selection reasons. Frames are selected from
sparse ordinary coverage plus close contacts and windows around instantaneous
bond-graph changes.

After a production, the existing molecule scanner analyses its recordings.
Events belonging to full Optimised-Valence teacher production are routed to
the manager's `WAITING_QM` queue. The scanner's structural fingerprinting
continues to own molecule identity and deduplication.

## Autonomous manager loop

The sequential manager command plans exactly one experiment from current
history, runs and persists it through the same production pipeline, then reads
the updated history before planning the next experiment:

```powershell
py -m chemistry_manager run --count 100 --duration 2.0 --qm-every 5 --device cuda
```

After each configured number of completed simulations it pauses MD and checks
the existing `WAITING_QM` queue. An empty queue is skipped; otherwise the
existing QM validator runs synchronously and its accepted/rejected trust state
is visible to the very next planner decision. A final queue check occurs after
the experiment budget.

Each run has an atomic receipt under
`teacher_data/<date>/manager_runs/RUN_....json`. It records deterministic
per-step planning seeds, child invocation and experiment IDs, completion and
failure counts, QM checkpoints, settings, and an active-step marker. Resume by
ID without changing the original settings:

```powershell
py -m chemistry_manager run --resume RUN_...
```

Completed experiment JSON files remain authoritative during crash recovery;
an incomplete active step may be retried, but a completed one is reconciled
without repeating its MD trajectory.
