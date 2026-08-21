# Unified radial Stage 2B engine extraction

## Outcome

`unified_radial_v1` now routes its energy composition through a canonical
`ChemistryEngine` and `UnifiedRadialHamiltonian`. The equations, solver,
parameters, inherited runtime, neighbours, force differentiation, integrator
and thermostat are unchanged.

The frozen Stage 2A JSON fixture was not regenerated. It remains the external
scientific ruler created from the pre-extraction implementation.

## Boundaries

The new package separates:

- `PhysicsSpec`: immutable model ID, parameter identity, capacity-solver
  convention, geometry convention and enabled terms;
- `ExecutionConfig`: device, dtype, batch layout, neighbour strategy, caching
  and solver execution mode;
- `InteractionContext`: positions, element identity, neighbours, box/batch
  information and the existing required tensors;
- `EnergyResult`: per-atom scalar energy, named components and state outputs;
- `ChemistryEngine`: the selected immutable physics spec plus Hamiltonian;
- runtime protocols for force, neighbour, batching, integration and thermostat
  responsibilities.

The canonical Hamiltonian does not build neighbours, select a device, advance
time, manage recording, integrate positions or apply a thermostat.

## Preserved composition

The current implementation intentionally retains the original call and
floating-point order:

```text
base = existing BatchedReactiveSimulation energy

capacity_correction =
    unified capacity
    - removed base pair attraction
    - removed base overcoordination

topology_correction =
    established geometry
    - removed base geometry

total = base + capacity_correction + topology_correction
```

The base is evaluated exactly once. Its live reactive intermediates remain
available to both corrections and are cleared in the same `finally` boundary.
The SciPy L-BFGS-B dual solver and all existing state construction remain in
their authoritative implementation.

Stage 2B uses narrow adapters for these existing operations. This is
intentional: moving their internal equations is a later migration and would
combine architecture risk with physics risk.

## Compatibility and comparison

`UnifiedBondCapacityEnergyPrototype` is the canonical-route compatibility
adapter used by all Stage 2A selectors. `LegacyUnifiedRadialReference` retains
the pre-extraction composition path solely for equivalence testing. Existing
reactive, H-state, valence-state, recording and replay APIs remain present.

The ruler now performs both comparisons:

1. canonical engine against the frozen Stage 2A outputs;
2. retained legacy route against canonical engine on identical inputs.

Run it with:

```powershell
python unified_radial_equivalence.py
```

On a CUDA-enabled runtime:

```powershell
python unified_radial_equivalence.py --cuda
```

Both single-box and grouped execution are checked. CPU float64 tolerances
remain `1e-10 eV` for energy, `1e-8 eV/A` for forces and `1e-10` for state.
CUDA uses the unchanged Stage 2A same-device tolerances.

## Remaining migration work

This stage does not delete or flatten the inherited runtime hierarchy. The
existing modules still own their validated internal equations, neighbour
construction, autograd force calculation, integration and thermostat. Future
cleanup may migrate those implementations behind the interfaces only after a
separate equivalence checkpoint.

No electronic physics, additional elements, parameter changes, solver changes
or performance work are part of Stage 2B.
