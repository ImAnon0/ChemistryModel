"""Queue-oriented command line interface for Chemistry Manager v1."""

import argparse

from .config import (
    DEFAULT_MOLECULE_ROOT, DEFAULT_RUNS_ROOT, DEFAULT_STATE_PATH,
    DEFAULT_TEACHER_ROOT,
)
from .discovery import discover, ingest_teacher_data
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

    cleanup = commands.add_parser(
        "cleanup",
        help="preview or remove legacy manager queue entries",
    )
    cleanup.add_argument(
        "--legacy-characterisation",
        action="store_true",
        help=(
            "target only WAITING_CHARACTERISATION candidates whose "
            "source.kind is formation_event"
        ),
    )
    cleanup.add_argument(
        "--confirm",
        action="store_true",
        help="actually apply the requested cleanup; without this flag only preview",
    )

    ingest_parser = commands.add_parser(
        "ingest", help="scan existing runs and queue discovered reactions"
    )
    ingest_parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ingest_parser.add_argument(
        "--molecule-root", default=str(DEFAULT_MOLECULE_ROOT)
    )
    ingest_parser.add_argument(
        "--teacher-data", action="store_true",
        help="register full-CM teacher datasets and queue new products directly for QM",
    )
    ingest_parser.add_argument(
        "--teacher-root", default=str(DEFAULT_TEACHER_ROOT),
        help="teacher-data root used with --teacher-data",
    )
    ingest_parser.add_argument(
        "--qm-root", default=None,
        help="optional QM-validation root used to determine trusted molecules",
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
    produce.add_argument(
        "--master-seed", type=int, default=None,
        help="explicit reproducible seed; omitted = fresh random seed",
    )
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

    production = commands.add_parser(
        "production",
        help="show reaction-production coverage for a calendar day",
    )
    production.add_argument("--output-root", default=str(DEFAULT_TEACHER_ROOT))
    production.add_argument(
        "--molecule-root", default=str(DEFAULT_MOLECULE_ROOT),
        help="molecule library used to resolve reaction products",
    )
    production.add_argument(
        "--qm-root", default=None,
        help="optional QM-validation root used to determine trusted products",
    )
    production.add_argument(
        "--date", default=None,
        help="local date YYYY-MM-DD (default: today)",
    )

    commands.add_parser(
        "validate", help="characterise future cheap-discovery candidates with full CM"
    )
    qm = commands.add_parser(
        "qm", help="process candidates waiting for QM validation"
    )
    qm.add_argument("--molecule-root", default=str(DEFAULT_MOLECULE_ROOT))
    qm.add_argument("--qm-root", default=None)
    qm.add_argument("--method", default="wb97x-d")
    qm.add_argument("--basis", default="jun-cc-pvdz")
    qm.add_argument("--threads", type=int, default=8)
    qm.add_argument("--memory", default="4 GB")
    qm.add_argument("--limit", type=int, default=None)
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

        if options.command == "cleanup":
            if not options.legacy_characterisation:
                print("No cleanup target selected.")
                print("Use --legacy-characterisation to preview the legacy queue cleanup.")
                return 0

            targeted = store.legacy_characterisation_candidates()
            counts = store.counts()

            print("Legacy Characterisation Cleanup")
            print()
            print(f"Candidates targeted:             {len(targeted)}")
            print(
                f"Waiting for QM (untouched):      "
                f"{counts[CandidateState.WAITING_QM]}"
            )
            print(
                f"QM validated (untouched):        "
                f"{counts[CandidateState.QM_VALIDATED]}"
            )
            print(
                f"QM rejected (untouched):         "
                f"{counts[CandidateState.QM_REJECTED]}"
            )
            print()
            print(
                "Target rule: state=WAITING_CHARACTERISATION and "
                "source.kind=formation_event"
            )

            if not options.confirm:
                print()
                print("No changes made.")
                print(
                    "Re-run with --confirm to remove only the candidates listed "
                    "by this rule."
                )
                return 0

            result = store.remove_legacy_characterisation_candidates()
            print()
            print(f"Removed legacy candidates:       {result['removed']}")
            print("All other candidate states and source kinds were preserved.")
            return 0

        if options.command == "ingest":
            if options.teacher_data:
                result = ingest_teacher_data(
                    store,
                    options.teacher_root,
                    options.molecule_root,
                    qm_root=options.qm_root,
                    state_file=options.state_file,
                    scan=not options.no_scan,
                )
                print("Teacher Data Ingest")
                print()
                print(f"Productions found:         {result['productions_found']}")
                print(f"New productions:           {result['productions_added']}")
                print(f"New experiments:           {result['experiments_added']}")
                print(f"New teacher frames:        {result['teacher_frames_added']}")
                print(f"Reaction events seen:      {result['events_seen']}")
                print(f"Queued directly for QM:    {result['queued_for_qm']}")
                print(f"Already-trusted events:    {result['already_trusted_events']}")
                print(f"Duplicate candidates:      {result['duplicate_candidates']}")
                print(f"Registry:                  {result['registry']}")
                if result["invalid"]:
                    print(f"Invalid teacher records:   {len(result['invalid'])}")
                if result["event_log_errors"]:
                    print(
                        f"Malformed event-log lines:  {len(result['event_log_errors'])}"
                    )
                return 0

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
            print(f"Invocation:              {result['invocation_id']}")
            print(f"Date:                    {result['date']}")
            print(f"Experiments requested:   {result['requested']}")
            print(f"Completed:               {result['completed_now']}")
            print(f"Failed:                  {result['failed_now']}")
            print(f"Master seed:             {result['master_seed']} ({result['master_seed_source']})")
            print(f"Teacher frames written:  {result['teacher_frames_written_now']}")
            print(f"New reactions detected:  {result['new_events']}")
            print(f"Trusted molecules used:  {result['trusted_molecules']}")
            print(f"Daily dataset:           {result['root']}")
            print("View: python -m chemistry_manager production")
            print("Next: python -m chemistry_manager ingest --teacher-data")
            return 0

        if options.command == "production":
            from .reaction_producer import production_summary

            result = production_summary(
                output_root=options.output_root,
                date=options.date,
                molecule_root=options.molecule_root,
                qm_root=options.qm_root,
            )
            print(f"Reaction Production — {result['date']}")
            print()
            print(f"Invocations:             {result['invocations']}")
            print(f"Experiments attempted:   {result['attempted']}")
            print(f"Completed:               {result['completed']}")
            print(f"Failed:                  {result['failed']}")
            print(f"Teacher frames:          {result['teacher_frames']}")
            print(f"Unique reactants:        {result['unique_reactants']}")
            print(f"Unique pairs:            {result['unique_pairs']}")
            print(
                f"Microcell compositions:  "
                f"{result.get('unique_microcell_compositions', 0)}"
            )

            if result.get("experiment_families"):
                print()
                print("Experiment families")
                for name, count in result["experiment_families"].items():
                    print(f"  {name + ':':24} {count}")

            if result.get("outcomes_by_family"):
                print()
                print("Chemistry outcomes")
                for family, counts in result["outcomes_by_family"].items():
                    total = sum(counts.values())
                    reacted = sum(
                        value for name, value in counts.items()
                        if name not in ("no reaction", "unclassified", "unknown")
                    )
                    print(
                        f"  {family}: {reacted}/{total} reacted"
                    )
                    for name, count in counts.items():
                        print(f"    {name + ':':22} {count}")

            print()
            print(f"Reaction events:         {result.get('reaction_events', 0)}")
            if result.get("reaction_events_by_family"):
                for family, count in result["reaction_events_by_family"].items():
                    print(f"  {family + ':':24} {count}")

            products = result.get("unique_product_species", [])
            untrusted = result.get("untrusted_product_species", [])
            print(f"Unique product species:  {len(products)}")
            print(f"Untrusted/new products:  {len(untrusted)}")
            if untrusted:
                print("  " + ", ".join(untrusted))

            if result["categories"]:
                print()
                print("Categories")
                for name, count in result["categories"].items():
                    print(f"  {name + ':':24} {count}")

            if result["collision_classes"]:
                print()
                print("Collision classes")
                for name, count in result["collision_classes"].items():
                    print(f"  {name + ':':24} {count}")

            if result["speed_classes"]:
                print()
                print("Speed classes")
                for name, count in result["speed_classes"].items():
                    print(f"  {name + ':':24} {count}")

            if result.get("event_log_errors"):
                print()
                print(
                    f"Malformed event-log lines: "
                    f"{len(result['event_log_errors'])}"
                )

            print()
            print(f"Dataset:                 {result['root']}")
            return 0

        if options.command == "validate":
            return _deferred_queue(
                store,
                CandidateState.WAITING_CHARACTERISATION,
                "Nothing waiting for controlled characterisation.",
                "Automated controlled characterisation is not connected in v1.",
            )

        if options.command == "qm":
            from .qm import process_qm_queue

            waiting = store.candidates(CandidateState.WAITING_QM)
            if not waiting:
                print("Nothing waiting for QM validation.")
                return 0

            def qm_progress(number, total, candidate, products):
                product_text = ", ".join(products) if products else "no product IDs"
                print(
                    f"[{number}/{total}] {candidate['id']}  "
                    f"products: {product_text}"
                )

            result = process_qm_queue(
                store,
                molecule_root=options.molecule_root,
                qm_root=options.qm_root,
                method=options.method,
                basis=options.basis,
                threads=options.threads,
                memory=options.memory,
                limit=options.limit,
                progress=qm_progress,
            )
            print()
            print("QM Queue")
            print()
            print(f"Candidates processed:      {result['candidates_seen']}")
            print(f"QM validated:              {result['validated']}")
            print(f"QM rejected:               {result['rejected']}")
            print(f"Still waiting:             {result['still_waiting']}")
            print(f"New molecules validated:   {result['molecules_validated']}")
            print(f"New molecules rejected:    {result['molecules_rejected']}")
            print(f"Existing results reused:   {result['molecules_reused']}")
            if result["errors"]:
                print(f"Errors/blocked:             {len(result['errors'])}")
                for problem in result["errors"][:8]:
                    print(f"  - {problem}")
            return 0
    except (OSError, ValueError) as problem:
        raise SystemExit(f"Chemistry Manager: {problem}") from problem

    raise AssertionError(f"unhandled command: {options.command}")
