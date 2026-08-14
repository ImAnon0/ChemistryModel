# Force-kernel performance investigation

Measured 14 August 2026 on the local i7-14700F / RTX 4060 Ti system. The
reference force field, parameters, timestep, neighbour set, thermostat,
recorder cadence, and chemistry were not simplified or tuned.

## Decision

**Keep the Triton/Inductor implementation experimental for now.** It exceeds
the performance target by a large margin and completes the full production
workload cleanly, but compiled float32 reductions do not reproduce eager
forces bit-for-bit and the project does not yet ship a supported Triton
dependency. The default eager implementation remains available and unchanged.

The simple `index_select` gather candidate is rejected as a standalone
production change: it is clean but its end-to-end MD gain is only about 5.7%.

## Reference profile

Workload: 16 simultaneous 330-atom `carbon rich` boxes, 19 A, float32 CUDA.

- Reference sustained short benchmark: 402--430 aggregate MD steps/s.
- Isolated force evaluation: about 13.4 ms/call, 74.6 calls/s.
- Eight profiled calls launched approximately 4,384 CUDA kernels, about 548
  kernels per force call.
- Autograd indexed-gradient accumulation (`index_put` and its CUDA indexing
  kernels) consumed about 88.7% of CUDA self time.
- Multiplication was 3.6%, reductions 1.3%, division 0.8%, and ordinary
  indexing below 0.5% of CUDA self time in the representative profile.

The dominant cost is therefore autograd's backward scatter/accumulation for
padded neighbour gathers, followed by launch overhead from hundreds of small
forward/backward kernels. Morse exponentials, neighbour construction, and
angle trigonometry are not individually the dominant kernel cost.

## Safe eager candidate: index-select gathers

Changing differentiable neighbour gathers from advanced indexing to flattened
`index_select` replaces the expensive sorted `index_put` backward path with
`index_add`.

| metric | reference | index-select | result |
|---|---:|---:|---:|
| isolated force calls/s | 74.6 | 86.9 | +16.4% |
| matched width-16 MD steps/s | 402.3 | 425.1 | +5.7% |

Energy was bit-identical on the deterministic comparison. Force differences
were at float32 accumulation-order scale and five NVE steps remained within
strict test tolerances. Because the realistic gain is below the project's
10--15% usefulness range, this remains an experimental benchmark option and
is not enabled in production.

## torch.compile / Triton

The installed PyTorch 2.13 CUDA build had no Triton package. A disposable
`triton-windows 3.7.1.post27` installation was tested without changing the
project environment. It required explicit writable caches and its bundled
TinyCC. Default Inductor fusion failed compiling the large angle backward
kernel; limiting `torch._inductor.config.max_fusion_size` to 8 avoided that
compiler failure.

Steady isolated results:

| path | force time | force calls/s |
|---|---:|---:|
| eager autograd | 13.4 ms | 74.6 |
| compiled, default mode/fusion 8 | 0.73--1.36 ms | 737--1,368 |
| compiled, reduce-overhead/fusion 8 | 0.66 ms | 1,516 |

Initial compilation took 18--34 seconds in the successful configuration;
cached startup was about 2--3 seconds. `reduce-overhead`/CUDA Graph operation
survived repeated neighbour-table rebuilds and gave 155.5 MD steps/s in the
focused width-16 loop versus about 25 steps/s eager.

## Numerical equivalence

At the initial deterministic production-sized state:

- potential-energy error: exactly 0 eV;
- maximum force error: 0.002389 eV/A;
- RMS force error: 0.0000811 eV/A;
- reference maximum force: 9.166 eV/A;
- relative maximum force error: 0.0261%;
- repeated eager force error: exactly zero.

The discrepancy comes from compiled float32 reduction ordering and is
independent of the index-select experiment. After 20 NVE steps, maximum
position error was 1.05e-5 A and velocity error 1.78e-6 A/fs. After 100 steps,
chaotic divergence reached 0.025 A, so long trajectories cannot be compared
atom-for-atom. This requires ensemble-level validation before mainstream use.

A matched 16-seed 500 fs ensemble showed:

| metric | eager | compiled |
|---|---:|---:|
| finished/stable | 16/16 | 16/16 |
| energy jumps | 0 | 0 |
| move caps | 0 | 0 |
| mean final temperature | 954.7 K | 913.2 K |
| mean final potential | -1007.3 eV | -1008.8 eV |
| mean species count | 49.12 | 49.06 |
| mean largest structure | 11.31 | 11.38 |
| mean adaptive frames | 157.6 | 160.7 |
| dynamics accounting | 44.8 s | 14.4 s |

The temperature difference is within one small 16-run stochastic ensemble's
broad spread but needs repeated matched ensembles before declaring statistical
equivalence.

## Full production candidate

Command conditions: `carbon rich`, seeds 28800--28815, 330 atoms/run, 19 A,
20 ps, width 16, CUDA, adaptive recorder v2.

| metric | eager baseline | compiled candidate |
|---|---:|---:|
| end-to-end wall time | 20.15 min | 11.85 min |
| grouped dynamics | 19.01 min | 10.48 min |
| post-processing | 1.14 min | 1.37 min |
| minutes saved, total | -- | 8.30 min |
| wall-time reduction | -- | 41.2% |
| dynamics time reduction | -- | 44.9% |
| dynamics throughput gain | -- | 81.5% |
| finished/stable | 16/16 | 16/16 |
| energy jumps | 0 | 0 |
| move caps | 0 | 0 |
| mean frames/run | 2382.6 | 2371.4 |
| recording size | 264.1 MB | 263.0 MB |

Candidate final temperature averaged 250.2 K (232.7--264.7 K). Mean final
potential was -1189.5 eV. No non-finite state, integration failure, unexplained
energy jump, or move cap occurred.

## Analytical/custom-kernel feasibility

An analytical Morse-only derivative would not remove the dominant cost: bond
order, environment softening, over-coordination, and angle terms all propagate
through the same gathered coordination graph. A correct larger analytical
implementation would need the full coupled derivative and would introduce much
more scientific-maintenance risk. It was not pursued after compilation removed
the launch/scatter bottleneck with much less code.

A custom Triton/CUDA force kernel is likewise not justified yet. Inductor
already demonstrates that fusion/graph capture can exceed the stretch target.
Custom work should be reconsidered only if compiled reduction equivalence or
dependency deployment cannot be made acceptable.

High-fidelity H-transfer corrections are not active in the normal soup force
path. Existing high-fidelity regressions pass unchanged; compiled forces are
explicitly rejected for that subclass until separately validated.

## Regression status

Passed after the experimental integration:

- reactive core: 8/8;
- high fidelity: 7/7;
- recorder compatibility: 10/10;
- stability classification;
- reaction experiments: 6/6;
- Results analysis;
- validation report `--quick`;
- compiled adaptive-recording smoke, 500 fs ensemble, 2 ps ensemble, and full
  20 ps production workload.

## Recommendation

Keep `--compiled-forces` opt-in and experimental. It is a credible path to a
mainstream 10--12 minute workload and has already passed the performance and
basic stability gates. Before making it the Lab/default path:

1. provide a supported, reproducible Triton-Windows dependency environment;
2. repeat matched 16-seed ensembles to bound temperature and chemistry
   distribution differences;
3. run focused capture/continuity and stable-molecule NVE comparisons through
   the compiled path itself;
4. decide and document an acceptable float32 force-equivalence tolerance;
5. validate high-fidelity separately or continue rejecting that combination.
