# Unified radial Stage 2C runtime separation

Stage 2C adds a concrete `SimulationRuntime` without changing chemistry.

The runtime owns the live dynamics tensors, counters, random generators,
neighbour state and known execution caches. It coordinates the unchanged
velocity-Verlet sequence through `ExistingVelocityVerletIntegrator`, the
unchanged Langevin update through `ExistingLangevinThermostat`, and existing
force and neighbour implementations through narrow adapters.

`ReactiveSimulation.step()` remains the public compatibility API and delegates
to `SimulationRuntime.step()`. Existing reactive, batched, H-state, valence and
unified-radial subclasses therefore continue to work without API changes.
Their historical attribute names are compatibility views onto runtime-owned
state.

The execution flow is now:

```text
SimulationRuntime
    -> Integrator / Thermostat / NeighbourBackend / ForceBackend
    -> compatibility simulation adapter
    -> ChemistryEngine
    -> UnifiedRadialHamiltonian
```

The runtime does not select a chemistry model or inspect Hamiltonian terms.
Its force backend invokes the existing force calculation, which reaches the
already-selected `ChemistryEngine` through the compatibility simulation.

For migration checking, `_legacy_step()` and `_legacy_apply_langevin()` retain
the pre-Stage-2C operations. Tests compare seeded thermostatted trajectories
bit-for-bit. They are reference paths, not alternate user-facing modes.

No velocity-Verlet operation, Langevin draw, random seed, timestep, neighbour
rule, force equation, energy term, solver or default changed in this stage.
