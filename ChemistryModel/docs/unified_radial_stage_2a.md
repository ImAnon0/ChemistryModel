# Unified radial Stage 2A checkpoint

## Status

`unified_radial_v1` is the frozen scientific reference for the future engine
migration. It remains opt-in. Historical batch, Lab, characterisation and
benchmark defaults have not changed.

This stage registers the existing implementation and measures it. It does not
move, simplify or reimplement any physics. The inherited implementation in
`research/unified_bond_capacity/prototype.py` remains authoritative.

Frozen scientific headline:

- Grambow barrier MAE: 1.474616 eV;
- Grambow barrier RMSE: 1.85773 eV;
- barrier sign agreement: 99.5%;
- reaction MAE: 2.23297 eV;
- water-transfer RMSE: 0.213629 eV;
- validated H2/H3/H + H2 and NVE behaviour.

It is not a default because it is still the float64/SciPy reference
implementation, retains the established angle-topology layer, and has not yet
been migrated to the proposed canonical engine/backend boundary.

## Explicit selection

Fresh grouped batch:

```powershell
python batch_runner.py --physics unified-radial --group 2 <other arguments>
```

Characterisation:

```powershell
python characterisation_runner.py --physics unified-radial <other arguments>
```

Benchmark:

```powershell
python research/benchmark/benchmark_reaction_barriers.py --physics unified-radial
```

The Lab selectors expose the same opt-in name. Continuation and single-box
batch paths retain their existing guards.

## Effective-source provenance

`physics_provenance.py` derives a manifest from the selected class rather than
using a manually maintained filename list. It captures:

- every class in the effective MRO;
- the original source of copied/aliased method objects;
- project-local functions referenced by those methods;
- the transitive project-local import closure.

The sorted relative paths and their bytes produce a SHA-256 identity. Run
metadata stores the algorithm, hash and complete manifest. Environment
packages are excluded; dependency versions remain environment provenance, not
ChemistryModel source identity.

## Equivalence fixture

`tests/fixtures/unified_radial_v1_reference.json` contains 21 self-contained
CPU-float64 cases:

- H2, H3 and H + H2;
- accepted stable H/C/N/O molecules;
- a water-transfer midpoint;
- reactant, transition-state and product geometries from three representative
  Grambow reactions.

For single and grouped execution it records:

- total, per-box and per-atom energy;
- base bond, overcoordination and angle terms;
- capacity and topology corrections;
- forces for every atom;
- dual variables, KKT/solver information and capacity use;
- H and heavy state probabilities;
- directed memberships and expected heavy bond orders.

The fixture generator refuses to overwrite the reference unless `--replace`
is supplied. Replacement is a scientific-baseline operation, not part of an
ordinary refactor.

## Running the ruler

Current reference against the frozen fixture:

```powershell
python unified_radial_equivalence.py
```

Future canonical candidate:

```powershell
python unified_radial_equivalence.py --candidate package.module:ClassName
```

Include a live same-device CUDA comparison where available:

```powershell
python unified_radial_equivalence.py --candidate package.module:ClassName --cuda
```

CPU float64 tolerances are `1e-10 eV` for energy, `1e-8 eV/A` for forces and
`1e-10` for state quantities. CUDA float32 uses the existing same-device
execution tolerances: `2e-5 eV`, `2e-4 eV/A` and `2e-5` for state quantities.
The fixture does not require cross-device bit identity.

## Stage boundary

Completed in Stage 2A:

- explicit model ID and opt-in selectors;
- automatic source provenance stored with runs;
- frozen scientific inputs and outputs;
- strict single/grouped and CPU/CUDA-aware comparator;
- unit and golden validation registration.

Stage 2B is documented separately in `unified_radial_stage_2b.md`. This file
continues to describe the immutable pre-extraction checkpoint and fixture.

Known migration risks remain the base-plus-correction evaluation order,
directed pair accounting, inherited angle membership, warm-started dual solve,
autograd lifetime of shared tensors, box-local indexing, and legacy metadata.
The Stage 2A fixture now makes those changes observable.
