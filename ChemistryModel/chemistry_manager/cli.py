"""Queue-oriented command line interface for Chemistry Manager v1."""

import argparse

import molecule_library

from .config import (
    DEFAULT_MOLECULE_ROOT, DEFAULT_RUNS_ROOT, DEFAULT_STATE_PATH,
    DEFAULT_TEACHER_ROOT,
)
from .discovery import discover, ingest_teacher_data
from .state import CandidateState
from .store import ManagerStore
from .trust import MoleculeTrust, trust_level


LABELS = {
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
    status = commands.add_parser(
        "status", help="show actionable molecule and queue state"
    )
    status.add_argument("--molecule-root", default=str(DEFAULT_MOLECULE_ROOT))
    status.add_argument("--qm-root", default=None)

    ingest_parser = commands.add_parser(
        "ingest", help="scan existing runs and queue discovered reactions"
    )
    ingest_parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ingest_parser.add_argument(
        "--molecule-root", default=str(DEFAULT_MOLECULE_ROOT)
    )
    ingest_parser.add_argument(
        "--teacher-data", action="store_true",
        help="import/recover existing full-CM teacher datasets and queue untrusted products for QM",
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
    produce.add_argument(
        "--wild-probability", type=float, default=0.08,
        help=(
            "share of experiments reserved for one-dimension broad-envelope "
            "exploration (default: %(default)s)"
        ),
    )
    produce.add_argument("--output-root", default=str(DEFAULT_TEACHER_ROOT))
    produce.add_argument("--molecule-root", default=str(DEFAULT_MOLECULE_ROOT))
    produce.add_argument("--qm-root", default=None)
    produce.add_argument("--device", default=None)
    produce.add_argument(
        "--physics",
        choices=(
            "optimised-valence", "standard", "high_fidelity",
            "unified-radial",
        ),
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

    autonomous = commands.add_parser(
        "run",
        help="run the sequential autonomous discovery/QM loop",
    )
    autonomous.add_argument("--count", type=int, default=None)
    autonomous.add_argument("--duration", type=float, default=None)
    autonomous.add_argument("--qm-every", type=int, default=None)
    autonomous.add_argument("--master-seed", type=int, default=None)
    autonomous.add_argument(
        "--profile", choices=("balanced", "gentle", "reactive"),
        default=None,
    )
    autonomous.add_argument(
        "--wild-probability", type=float, default=None,
        help=(
            "broad-envelope exploration share; omitted uses the current "
            "default for a new run and the saved value on resume"
        ),
    )
    autonomous.add_argument("--output-root", default=str(DEFAULT_TEACHER_ROOT))
    autonomous.add_argument("--molecule-root", default=None)
    autonomous.add_argument("--qm-root", default=None)
    autonomous.add_argument("--device", default=None)
    autonomous.add_argument(
        "--physics",
        choices=(
            "optimised-valence", "standard", "high_fidelity",
            "unified-radial",
        ),
        default=None,
    )
    autonomous.add_argument("--ordinary-frame-fs", type=float, default=None)
    autonomous.add_argument("--event-window-fs", type=float, default=None)
    autonomous.add_argument("--diagnostic-sample-fs", type=float, default=None)
    autonomous.add_argument("--capture-every", type=int, default=None)
    autonomous.add_argument("--method", default=None)
    autonomous.add_argument("--basis", default=None)
    autonomous.add_argument("--threads", type=int, default=None)
    autonomous.add_argument("--memory", default=None)
    autonomous.add_argument(
        "--resume", default=None,
        help="resume a RUN_... receipt from the selected teacher root",
    )
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


def _candidate_product_ids(candidates):
    found = set()
    for candidate in candidates:
        for product in candidate.get("products") or []:
            if isinstance(product, dict) and product.get("id"):
                found.add(str(product["id"]))
    return found


def _print_reaction_result(item):
    if item.get("outcome") == "failed":
        requested = item.get("requested_picoseconds")
        runtime_text = (
            "" if requested is None
            else f"  requested {float(requested):.2f} ps"
        )
        print(
            f"       FAILED:{runtime_text}  "
            f"{item.get('error', 'unknown error')}"
        )
        return

    actual = item.get("actual_picoseconds")
    requested = item.get("requested_picoseconds")
    reason = str(item.get("termination_reason") or "unknown")
    if actual is not None and requested is not None:
        reason_text = {
            "quiescent": "quiescent",
            "duration_complete": "full duration",
            "numerical_failure": "numerical failure",
            "ended_early": "ended early",
        }.get(reason, reason.replace("_", " "))
        print(
            f"       runtime: {float(actual):.2f} / "
            f"{float(requested):.2f} ps  ({reason_text})"
        )

    events = int(item.get("reaction_events", 0))
    products = item.get("products") or []
    print(
        f"       result: {item.get('outcome', 'no reaction')}; "
        + (
            "no reaction event" if events == 0
            else f"{events} reaction event(s)"
        )
    )
    for product in products:
        novelty = "NEW" if product.get("new_this_experiment") else "known"
        print(
            f"       {product['id']} {product.get('formula') or '?'}  "
            f"{novelty} / {product.get('trust', 'UNKNOWN')}"
        )
        queue = product.get("queue")
        if queue == CandidateState.WAITING_QM.value:
            print("         -> WAITING_QM")
        elif queue == CandidateState.QM_VALIDATED.value:
            print("         -> QM_VALIDATED / trusted")
        elif queue == CandidateState.QM_REJECTED.value:
            print("         -> QM_REJECTED")
        elif queue == "trusted":
            print("         -> trusted reactant")


def print_status(store, molecule_root=DEFAULT_MOLECULE_ROOT, qm_root=None):
    counts = store.counts()
    waiting_candidates = store.candidates(CandidateState.WAITING_QM)
    candidate_waiting_ids = _candidate_product_ids(waiting_candidates)

    molecules = molecule_library.list_molecules(root=molecule_root)
    rows = []
    for molecule in molecules:
        try:
            level = trust_level(molecule, qm_root=qm_root)
        except Exception:
            level = MoleculeTrust.UNVALIDATED
        rows.append((molecule, level))

    level_by_id = {
        str(molecule.get("id")): level
        for molecule, level in rows
        if molecule.get("id")
    }
    waiting_ids = {
        molecule_id for molecule_id in candidate_waiting_ids
        if level_by_id.get(molecule_id, MoleculeTrust.UNVALIDATED)
        == MoleculeTrust.UNVALIDATED
    }

    trusted = [
        molecule for molecule, level in rows
        if level in (MoleculeTrust.CM_VALIDATED, MoleculeTrust.QM_VALIDATED)
    ]
    rejected = [
        molecule for molecule, level in rows
        if level == MoleculeTrust.REJECTED
    ]
    idle = [
        molecule for molecule, level in rows
        if level == MoleculeTrust.UNVALIDATED
        and str(molecule.get("id")) not in waiting_ids
    ]

    print("Chemistry Manager")
    print()
    print("Molecules")
    print(f"  Stored species:                   {len(molecules):>6}")
    print(f"  Trusted reactants:                {len(trusted):>6}")
    print(f"  Waiting for QM:                   {len(waiting_ids):>6}")
    print(f"  Unvalidated, not queued:          {len(idle):>6}")
    print(f"  Rejected:                         {len(rejected):>6}")

    if waiting_ids:
        by_id = {str(m.get("id")): m for m, _ in rows}
        print()
        print("Waiting for QM")
        for molecule_id in sorted(waiting_ids):
            molecule = by_id.get(molecule_id, {})
            print(f"  {molecule_id:12} {molecule.get('formula', '?')}")
        print()
        print("Next: python -m chemistry_manager qm")

    if idle:
        print()
        print("Unvalidated and not queued")
        for molecule in idle[:20]:
            print(
                f"  {str(molecule.get('id')):12} "
                f"{molecule.get('formula', '?')}"
            )
        if len(idle) > 20:
            print(f"  ... and {len(idle) - 20} more")
        print(
            "  These structures exist in the library but are not currently "
            "represented in WAITING_QM."
        )

    print()
    print("Event candidate states")
    for state in CandidateState:
        print(f"  {LABELS[state] + ':':32} {counts[state]:>6}")


def main(argv=None):
    options = build_parser().parse_args(argv)
    store = ManagerStore(options.state_file)

    try:
        migration = store.migrate_legacy_wait_states()
        if options.command == "status":
            if migration["migrated"]:
                print(
                    f"Migrated {migration['migrated']} legacy candidate(s) "
                    "to WAITING_QM."
                )
                print()
            print_status(
                store,
                molecule_root=options.molecule_root,
                qm_root=options.qm_root,
            )
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
                print(
                    f"Existing candidates refreshed: "
                    f"{result.get('refreshed_candidates', 0)}"
                )
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
                f"Queued {result['queued']} reaction candidate(s) for QM; "
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

        if options.command == "run":
            from .manager_runner import run_manager

            def manager_progress(number, total, spec):
                planner = spec.get("planner") or {}
                mode = str(planner.get("mode", "planning")).replace("_", " ")
                print(
                    f"[{number}/{total}] {mode}  {spec.get('category', 'experiment')}  "
                    f"{spec.get('reactant_a', '?')} + {spec.get('reactant_b', '?')}"
                )

            def checkpoint_output(checkpoint):
                print()
                print(
                    f"QM {checkpoint['kind']} checkpoint after "
                    f"{checkpoint['after_completed_experiments']} simulations"
                )
                print(f"  waiting:          {checkpoint['waiting_before']}")
                print(f"  validated:        {checkpoint['validated']}")
                print(f"  rejected:         {checkpoint['rejected']}")
                print(f"  still waiting:    {checkpoint['still_waiting']}")
                print(f"  trusted reactants:{checkpoint['trusted_reactants']:>5}")
                if checkpoint["kind"] != "final":
                    print("continuing...")
                print()

            result = run_manager(
                store,
                count=options.count,
                duration_ps=options.duration,
                qm_every=options.qm_every,
                device=options.device,
                master_seed=options.master_seed,
                profile=options.profile,
                output_root=options.output_root,
                molecule_root=options.molecule_root,
                qm_root=options.qm_root,
                physics=options.physics,
                ordinary_interval_fs=options.ordinary_frame_fs,
                event_window_fs=options.event_window_fs,
                diagnostic_sample_fs=options.diagnostic_sample_fs,
                capture_every=options.capture_every,
                wild_probability=options.wild_probability,
                qm_method=options.method,
                qm_basis=options.basis,
                qm_threads=options.threads,
                qm_memory=options.memory,
                resume=options.resume,
                progress=manager_progress,
                result_observer=_print_reaction_result,
                qm_observer=checkpoint_output,
            )
            print("Autonomous Chemistry Manager")
            print()
            print(f"Run:                     {result['run_id']}")
            print(f"Status:                  {result['status']}")
            print(f"Requested experiments:   {result['requested_experiments']}")
            print(f"Completed experiments:   {result['completed_experiments']}")
            print(f"Failed experiments:      {result['failed_experiments']}")
            print(f"QM validated:            {result['qm_validated_count']}")
            print(f"QM rejected:             {result['qm_rejected_count']}")
            print(f"Master seed:             {result['master_seed']}")
            print(
                f"Wild exploration:        "
                f"{100.0 * float(result['wild_probability']):.1f}%"
            )
            print(f"Device:                  {result['device']}")
            print(f"Receipt:                 {result['receipt_path']}")
            return 0

        if options.command == "produce_reactions":
            from .reaction_producer import run_production

            if options.duration <= 0:
                raise ValueError("duration must be greater than zero")
            if options.diagnostic_sample_fs <= 0:
                raise ValueError("diagnostic-sample-fs must be greater than zero")
            if options.capture_every < 1:
                raise ValueError("capture-every must be at least one")
            if not 0.0 <= options.wild_probability <= 1.0:
                raise ValueError(
                    "wild-probability must be between 0 and 1"
                )

            def progress(number, total, spec):
                print(
                    f"[{number}/{total}] {spec['category']}  "
                    f"{spec['reactant_a']} + {spec['reactant_b']}  "
                    f"{spec['collision_class']} / {spec['speed_class']}"
                )

            def result_observer(item):
                _print_reaction_result(item)

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
                wild_probability=options.wild_probability,
                progress=progress,
                result_observer=result_observer,
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
            print(
                f"Wild exploration:        "
                f"{100.0 * float(result['wild_probability']):.1f}%"
            )
            print(f"Teacher frames written:  {result['teacher_frames_written_now']}")
            print(f"New reactions detected:  {result['new_events']}")
            print(f"Queued for QM now:       {result['new_candidates_queued']}")
            print(f"New product species:     {len(result.get('new_product_species', []))}")
            print(f"Trusted molecules used:  {result['trusted_molecules']}")
            print(f"Daily dataset:           {result['root']}")
            print("View: python -m chemistry_manager production")
            if result["new_candidates_queued"]:
                print("Next: python -m chemistry_manager qm")
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
            print(f"Untrusted products:      {len(untrusted)}")
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

            def qm_molecule_progress(number, total, candidate, result):
                molecule_id = result.get("molecule_id", "?")
                outcome = result.get("outcome", "unknown")
                reused = " (reused)" if result.get("reused") else ""
                print(f"       {molecule_id}: {outcome}{reused}")
                if outcome == "validated":
                    print("         -> trusted reactant")
                elif outcome == "rejected":
                    print("         -> stored but excluded from trusted pool")

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
                molecule_progress=qm_molecule_progress,
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
