"""Queue-oriented command line interface for Chemistry Manager v1."""

import argparse

from .config import (
    DEFAULT_MOLECULE_ROOT, DEFAULT_RUNS_ROOT, DEFAULT_STATE_PATH,
    DEFAULT_TEACHER_ROOT,
)
from .discovery import discover
from .state import CandidateState
from .store import ManagerStore


LABELS = {
    CandidateState.WAITING_CHARACTERISATION: "Waiting for characterisation",
    CandidateState.WAITING_QM: "Waiting for QM validation",
    CandidateState.QM_VALIDATED: "QM validated",
    CandidateState.QM_REJECTED: "QM rejected",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m chemistry_manager",
        description="Persistent queue manager for existing ChemistryModel workflows.",
    )
    parser.add_argument(
        "--state-file", default=str(DEFAULT_STATE_PATH),
        help="manager JSON overlay (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show queue counts")

    ingest_parser = commands.add_parser(
        "ingest", help="scan existing runs and queue discovered reactions"
    )
    ingest_parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ingest_parser.add_argument(
        "--molecule-root", default=str(DEFAULT_MOLECULE_ROOT)
    )

    commands.add_parser(
        "discover",
        help="reserved for future large-scale surrogate soup discovery",
    )

    produce = commands.add_parser(
        "produce_reactions",
        help="generate targeted full-ChemistryModel teacher experiments",
    )
    produce.add_argument("--count", type=int, default=12)
    produce.add_argument("--duration", type=float, default=0.25)
    produce.add_argument("--master-seed", type=int, default=20260819)
    produce.add_argument(
        "--profile", choices=("balanced", "gentle", "reactive"),
        default="balanced",
    )
    produce.add_argument("--output-root", default=str(DEFAULT_TEACHER_ROOT))
    produce.add_argument("--molecule-root", default=str(DEFAULT_MOLECULE_ROOT))
    produce.add_argument("--qm-root", default=None)
    produce.add_argument("--device", default=None)
    produce.add_argument(
        "--physics",
        choices=("optimised-valence", "standard", "high_fidelity"),
        default="optimised-valence",
        help=(
            "teacher physics (default: current validated Optimised-Valence "
            "H-state model)"
        ),
    )
    produce.add_argument("--ordinary-frame-fs", type=float, default=10.0)
    produce.add_argument("--event-window-fs", type=float, default=5.0)
    produce.add_argument("--diagnostic-sample-fs", type=float, default=1.0)
    produce.add_argument("--capture-every", type=int, default=4)
    ingest_parser.add_argument(
        "--no-scan", action="store_true",
        help="only import the scanner's existing formation-event log",
    )

    commands.add_parser(
        "validate", help="process candidates waiting for full-CM validation"
    )
    commands.add_parser("qm", help="process candidates waiting for QM validation")
    return parser


def print_status(store):
    counts = store.counts()
    print("Chemistry Manager")
    print()
    for state in CandidateState:
        print(f"{LABELS[state] + ':':34} {counts[state]:>6}")


def _deferred_queue(store, state, empty_message, deferred_message):
    waiting = store.candidates(state)
    if not waiting:
        print(empty_message)
        return 0
    print(f"{len(waiting)} candidate(s) waiting.")
    print(deferred_message)
    print("No candidate state was changed.")
    return 0


def main(argv=None):
    options = build_parser().parse_args(argv)
    store = ManagerStore(options.state_file)

    try:
        if options.command == "status":
            print_status(store)
            return 0

        if options.command == "ingest":
            result = discover(
                store,
                options.runs_root,
                options.molecule_root,
                scan=not options.no_scan,
            )
            if result["scan"] is not None:
                scan = result["scan"]
                print(
                    "Scanner: "
                    f"{scan['scanned']} scanned, {scan['unchanged']} unchanged, "
                    f"{scan['formation_events']} new event(s)."
                )
            print(
                f"Queued {result['queued']} reaction candidate(s); "
                f"{result['already_known']} already known."
            )
            if result["errors"]:
                print(
                    f"Ignored {len(result['errors'])} malformed event-log line(s)."
                )
            return 0

        if options.command == "discover":
            print(
                "The discover command is reserved for future large-scale "
                "surrogate soup discovery."
            )
            print("Use ingest for existing scanner data or produce_reactions for teacher data.")
            return 0

        if options.command == "produce_reactions":
            from .reaction_producer import run_production

            if options.duration <= 0:
                raise ValueError("duration must be greater than zero")
            if options.diagnostic_sample_fs <= 0:
                raise ValueError("diagnostic-sample-fs must be greater than zero")
            if options.capture_every < 1:
                raise ValueError("capture-every must be at least one")

            def progress(number, total, spec):
                print(
                    f"[{number}/{total}] {spec['category']}  "
                    f"{spec['reactant_a']} + {spec['reactant_b']}  "
                    f"{spec['collision_class']} / {spec['speed_class']}"
                )

            result = run_production(
                store,
                count=options.count,
                duration_ps=options.duration,
                master_seed=options.master_seed,
                profile=options.profile,
                output_root=options.output_root,
                molecule_root=options.molecule_root,
                qm_root=options.qm_root,
                device=options.device,
                ordinary_interval_fs=options.ordinary_frame_fs,
                event_window_fs=options.event_window_fs,
                diagnostic_sample_fs=options.diagnostic_sample_fs,
                capture_every=options.capture_every,
                physics=options.physics,
                progress=progress,
            )
            print()
            print("Reaction Teacher Production")
            print()
            print(f"Production:              {result['production_id']}")
            print(f"Experiments requested:   {result['requested']}")
            print(f"Completed this run:       {result['completed_now']}")
            print(f"Completed total:          {result['completed_total']}")
            print(f"Teacher frames written:  {result['teacher_frames_written_now']}")
            print(f"New reactions detected:  {result['new_events']}")
            print(f"New candidates queued:   {result['new_candidates_queued']}")
            print(f"Trusted molecules used:  {result['trusted_molecules']}")
            print(f"Dataset:                  {result['root']}")
            return 0

        if options.command == "validate":
            return _deferred_queue(
                store,
                CandidateState.WAITING_CHARACTERISATION,
                "Nothing waiting for controlled characterisation.",
                "Automated controlled characterisation is not connected in v1.",
            )

        if options.command == "qm":
            return _deferred_queue(
                store,
                CandidateState.WAITING_QM,
                "Nothing waiting for QM validation.",
                "Automated QM execution is not connected in v1.",
            )
    except (OSError, ValueError) as problem:
        raise SystemExit(f"Chemistry Manager: {problem}") from problem

    raise AssertionError(f"unhandled command: {options.command}")
