# ChemistryModel ensemble performance investigation

Measured 13 August 2026 on the current local repository. This is an
observational performance study; no force-field parameter or equation was
changed.

## Machine and execution path

- Intel Core i7-14700F, 20 physical / 28 logical cores
- 15.8 GB system RAM
- NVIDIA GeForce RTX 4060 Ti, 8 GB VRAM, driver 610.74
- PyTorch 2.13.0+cu132, CUDA runtime 13.2, Python 3.12.5
- Normal Lab batches launch one `batch_runner.py` process per requested batch.
- A process advances up to 16 independent equal-sized seeds in one
  `BatchedReactiveSimulation`. Boxes are flattened into one tensor but have
  independent neighbour tables, periodic boundaries, initial states,
  thermostat draws, recorders and result entries. Groups run sequentially.
- Full molecular analysis and index generation run per seed after dynamics.
  Adaptive event observation and sparse Lab heartbeat summaries occur during
  dynamics. Formula naming, structure recognition and connected-component
  reporting are not performed every MD step.

## Measured 330-atom tensor scaling

Fixed seeds, 19 A box, standard float32 CUDA physics, no recording. Values are
aggregate MD steps across all seeds per wall second.

| tensor width | steps/s | relative to width 1 |
| ---: | ---: | ---: |
| 1 | 124.6 | 1.00x |
| 2 | 198.8 | 1.60x |
| 4 | 287.7 | 2.31x |
| 8 | 365.0 | 2.93x |
| 16 | 425.1 | 3.41x |
| 24 | 437.0 | 3.51x |
| 32 | 440.2 | 3.53x |

Width 16 captures nearly all useful gain for a 16-seed request. Width 24 is
only 2.8% faster and width 32 only 3.5% faster, while initialization rises
from 6.7 s at width 16 to 11.9 s at width 32. The current width-16 default is
therefore appropriate for this workload.

## CUDA process concurrency

Independent 330-atom CUDA processes, 400 measured steps each:

| processes | aggregate steps/s | vs one |
| ---: | ---: | ---: |
| 1 | 172.3 | 1.00x |
| 2 | 264.4 | 1.53x |
| 3 | 284.9 | 1.65x |
| 4 | 277.4 | 1.61x |

One width-16 tensor process is about 1.49x faster than the best three-process
result and avoids duplicated CUDA contexts. Lab now prevents two grouped GPU
jobs from competing concurrently; CPU and explicit group-1 jobs retain the
user's normal concurrency setting.

## Hot-path profile

Synchronized single-seed 330-atom measurements:

| work | measured share/cost |
| --- | ---: |
| force calculation | 79.8% of a step |
| per-step neighbour rebuild check | 2.2% |
| neighbour rebuild, amortized over 20 steps | 1.1% |
| position fetch at 40-step cadence | below 0.1% |
| energy reads at 40-step cadence | about 0.1% |

The energy uses a CPU `cKDTree` neighbour build with a skin and a maximum of
12 neighbours per atom. Pair/bond-order work is O(N*k), and angle work is
O(N*k^2) for fixed k=12; it does not construct an N-by-N force matrix. The
force/autograd kernels, not neighbour search or post-run analysis, are the
primary cost.

At width 16, sampled GPU utilization averaged 44.8%, peaked at 81%, used about
1968 MB process VRAM, and averaged 32.1 W during the short steady benchmark.
The card is not fully saturated continuously; kernel launch/autograd overhead
and CPU neighbour/startup work remain visible. Width growth beyond 16 provides
little throughput improvement for a 16-seed workload.

## Atom-count scaling

Single-seed synchronized step cost rose only about 12% from 330 to 660 atoms.
At width 16, aggregate throughput was 425 steps/s for 330 atoms and 201 steps/s
for 660 atoms. This under-filled small-box behavior is why cross-seed tensor
batching is effective.

## Recording

At width 16 and production-like cadence:

- Legacy fixed-cadence observation was within run-to-run noise of recording
  disabled; transfer plus recorder CPU time was below 0.1% in the measured
  800-step probe.
- Adaptive observation at its real 2 fs candidate cadence measured about
  7-9% wall overhead. Most was CPU chemical-event/retention work; compact
  device transfer was small. The initial apparent 5.5x overhead was rejected
  because the benchmark had omitted the production compact chemical
  observation and forced the recorder's all-pairs compatibility fallback.
- Adaptive recording remains observational and was not weakened or given a
  coarser event cadence for speed.
- In a very short 200-step/seed probe across 16 seeds, compressed legacy
  output totalled 0.60 MB and adaptive output 2.81 MB. This opening interval
  is deliberately event-dense and too short to extrapolate file size linearly;
  adaptive retention becomes sparse during quiet periods. Save time was
  0.03 s and 0.12 s respectively.

## Precision, compilation, and equivalence

- Production Torch dynamics already use float32. Float64 is used selectively
  in scientific diagnostics, not in the normal batch hot path. There is no
  obvious safe float64-to-float32 production conversion to claim.
- Float16 was rejected without experimentation because the core has steep
  Morse repulsion, thresholds and stability-sensitive gradients.
- `torch.compile` cannot currently use the installed Inductor path because a
  working Triton installation is absent. No dependency or compiled path was
  added speculatively.
- The existing grouped-versus-single deterministic check produced exactly
  zero force and energy error for four fixed independent boxes.
- Reactive core, high-fidelity, recorder and Replay regression suites passed
  during this work.

## Failure and heterogeneous-job behavior

Tensor batching requires equal atom count, box and settings; heterogeneous
jobs retain separate processes/groups. Per-seed recordings and analysis remain
separate. A NaN currently identifies the affected seed but stops its whole
active tensor group. Per-seed masking/extraction is not implemented because it
would materially complicate integration state and needs its own correctness
study; stopping the group remains safer than allowing a bad trajectory to
contaminate reporting.

## Recommendation

The fastest maintainable architecture on this machine is the architecture
already present: one CUDA process containing one width-16 true tensor batch,
with sequential remainder groups and separate per-seed recorders/results.
The production improvement justified by measurement is scheduler-level:
do not overlap multiple already-grouped CUDA jobs. A hybrid of several
processes each holding tensor batches is not supported by these results and
would add contention.

## Full production workload

The requested production benchmark completed using `carbon rich`, seeds
27000-27015, 330 atoms/run, 19 A, 20 ps, width 16, CUDA and adaptive recorder
v2:

| result | measured |
| --- | ---: |
| complete end-to-end wall time | 20.15 min |
| grouped dynamics | 19.01 min |
| final analysis/compression/index work | 1.14 min |
| aggregate throughput | 1,122 MD steps/s |
| finished/stable | 16/16 |
| energy jumps | 0 |
| move-cap events | 0 |
| mean recorded frames/run | 2,382.6 |
| recording size | 264.1 MB total / 16.5 MB mean |

The earlier synchronized short probe was pessimistic because explicit timing
synchronization and dense opening chemistry dominated it. It is superseded by
this production measurement for runtime estimates.

### Threshold verdict

- **Below 20 minutes:** nearly achievable by removing all post-run latency,
  but that would only reach the measured 19.01-minute dynamics floor.
- **Below 15 minutes:** not supported by any regression-safe optimization
  measured here. It requires at least a further 27% dynamics-throughput gain,
  beyond recorder or width tuning.
- **Below 10 minutes:** would require roughly doubling full-run dynamics
  throughput. Current width scaling, process scaling and GPU measurements do
  not support this without a deeper force-kernel implementation project.
- **Below 5 minutes:** not realistic on the measured engine and hardware.

The honest default is therefore width-16 tensor batching plus exclusive
grouped-GPU scheduling. A future sub-15-minute effort should focus narrowly on
the 80%-of-step force/autograd path (for example a supported compiled/custom
kernel implementation), with the complete physics-equivalence suite as its
acceptance gate. Recorder, analysis, file layout and additional CUDA processes
cannot supply the required gain.
