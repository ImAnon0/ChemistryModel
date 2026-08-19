"""Queue-oriented command line interface for Chemistry Manager v1."""

import argparse

from .config import DEFAULT_MOLECULE_ROOT, DEFAULT_RUNS_ROOT, DEFAULT_STATE_PATH
from .discovery import discover
from .state import CandidateState
from .store import ManagerStore


LABELS = {
    CandidateState.WAITING_FULL_CM: "Waiting for full-CM validation",
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

    discover_parser = commands.add_parser(
        "discover", help="scan existing runs and queue discovered reactions"
    )
    discover_parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    discover_parser.add_argument(
        "--molecule-root", default=str(DEFAULT_MOLECULE_ROOT)
    )
    discover_parser.add_argument(
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

        if options.command == "discover":
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

        if options.command == "validate":
            return _deferred_queue(
                store,
                CandidateState.WAITING_FULL_CM,
                "Nothing waiting for full-ChemistryModel validation.",
                "Automated full-ChemistryModel execution is not connected in v1.",
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

