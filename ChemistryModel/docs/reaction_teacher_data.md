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
Events belonging to that production are added to the manager's
`WAITING_CHARACTERISATION` queue. The scanner's structural fingerprinting
continues to own molecule identity and deduplication.
