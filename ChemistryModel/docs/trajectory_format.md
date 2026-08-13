# ChemistryModel trajectory compatibility baseline

## Legacy format (version 1)

Legacy recordings are compressed NumPy NPZ archives with no version field.
`Recorder.load()` identifies them by the absence of `format_version`.

Required arrays:

- `symbols`: starting element symbols
- `box_size`: nominal scalar periodic-cell size
- `positions`: float32 `[frame, atom, xyz]`
- `times`, `potential`, `kinetic`, `temperature`: one scalar per frame

Optional arrays added while retaining the unversioned layout:

- `velocities`: float32 `[frame, atom, xyz]`, required for exact continuation
- `box_sizes`: per-frame periodic-cell size
- `frame_types`: uint8 per-frame element types for open-box replacement
- `frame_atom_ids`: uint32 stable identities for open-box replacement

Missing optional fields fall back to the starting symbols, array-slot identity,
nominal box size, and no exact continuation respectively. Bonds and species are
not stored; readers reconstruct them from recorded positions using the current
bond taper threshold.

## Capture behavior baseline

Callers choose the simulation interval and call `Recorder.capture`. Recorder
then applies an internal `stride`, initially one. When `maximum_frames` is
exceeded, every parallel frame array is thinned by two and future input calls
are retained at twice the prior stride. This preserves the full time span but
progressively reduces temporal resolution everywhere, including old events.

Batch and grouped-batch runners call capture every `capture_every` integration
steps. Live standalone Lab subdivides each display update by its recording
interval. Characterisation has separate fine/coarse stepping for contact
diagnostics but stores ordinary trajectory frames at `capture_every`.

## Version 2 compatibility foundation

Opt-in version-2 saves include explicit `format_version=2`, parallel
`frame_kinds` uint8 metadata, and `event_reasons` strings. Kind zero means an
ordinary scheduled frame; kinds 1/2/3 mean pre-event/event/post-event. Legacy files remain
read-only compatible and are never rewritten during loading. Version 2 retains
the in-memory Recorder interface used by analysis, reindex, molecule tools,
continuation, run browser and Replay.

Future adaptive kinds and event metadata must remain observational: capture
decisions may copy already-computed state but must not step the simulation,
modify forces, alter thermostat state, or change integration order.

`AdaptiveRecorder` retains ordinary frames at a chosen
physical-time cadence, keeps a bounded rolling pre-event buffer, and flushes
dense candidates before and after explicit events or total-energy jumps. It is
the default for sealed batch and Lab runs. `--legacy-recording` retains the
original writer explicitly, and unsupported workflows fall back to it.

Authoritative version-2 events store stable-ID formed/broken bond pairs,
newly-entered unusually compressed contacts, event reason, time and energy
delta. Bond detection uses the existing taper with 0.65/0.15 form/break
hysteresis; compressed-contact detection defaults to 0.35 times the pair inner
cutoff. Replay prefers these stored events and reconstructs only legacy files.

The production detector reuses detached neighbour, distance, inner-cutoff and
taper tensors already evaluated by the force calculation. It filters unique
active pairs on-device and performs one compact host transfer; grouped runs
split that transfer by box. A 410-atom check transferred 63 active pairs rather
than evaluating all 83,845 pairs and cost about 245 microseconds per
observation. The old host all-pairs prototype cost about 24.5 milliseconds.

Dense retention, not detection, is now the limiting overhead during the first
rapid assembly window: a short event-heavy two-box CPU check was about 36%
slower at 2 fs candidates, while preserving final state exactly. This is why
authoritative chemical detection remains opt-in pending long GPU measurements
and a policy for coalescing sustained assembly bursts without losing evidence.

Sustained chemical changes are now coalesced into configurable reaction windows
(20 fs by default). Every exact change keeps its timestamp and stable-ID pairs
in event metadata, while only the first frame is protected as an exact event;
subsequent changes remain dense post-event frames. Replay shows one marker per
window and aggregates all changed pairs for inspection. Energy/failure events
remain independently protected and are never absorbed into chemical windows.

Chemical windows use a separate dense context (`chemical_context_fs`, 10 fs
by default) rather than the longer failure-debugging pre/post context. Measured
against the 16-run, 2 ps H-rich GPU benchmark, retaining every exact event plus
10 fs chemical context projects to about 7,532 frames / 64.8 MB, versus 13,889
frames / 119.6 MB for 50 fs chemical context and 3,200 frames / 27.6 MB legacy.

Batch runner integration defaults to version 2. Candidate,
pre/post-window and energy-jump settings are part of batch identity, and
auto-named output folders carry an `adaptiveNfs` suffix. The first safe path is
sealed, fixed-box, non-continuation chemistry. Open-box operations, lightning,
and continuation automatically retain version 1 unless version 2 was requested
explicitly, in which case the runner reports that the combination is not yet
cadence-safe. `--legacy-recording` is always available. The standalone live
runner also remains on the legacy writer.

Matched CPU benchmarks preserved final positions and velocities exactly. A
short 82-atom single-box run cost about 20% more wall time at 2 fs candidates;
a short two-box grouped run cost about 10% more. Quiet runs retained the same
frame count. Synthetic event coverage improved from 10 fs to 2 fs while file
size rose about 27% for a trajectory containing a protected event window.

Adaptive frame-limit handling preserves every exact event frame plus the first
and last trajectory frames. It reduces ordinary frames first, then post-event
and pre-event context only when required, distributing removals over time.
`adaptive_dropped_frames` and `thinned_count` report this loss explicitly. If
protected event frames alone exceed the limit, recording stops with a clear
request to increase `--max-frames`.

## Reader/writer inventory

Writers: `batch_runner.py` (single, grouped, continuation and checkpoints),
`characterisation_runner.py`, and `run_reactive_gl.py`.

Readers: `batch_runner.py` continuation/merge, `analysis.py` through callers,
`reindex.py`, `lab.py`, `lab_replay.py`, `run_browser.py`,
`run_reactive_gl.py`, `molecule_scanner.py`, and `molecule_library.py`.

`molecule_library.py` also writes its own separate molecule payload NPZ files;
those are not trajectory recordings and must not be migrated as such.
