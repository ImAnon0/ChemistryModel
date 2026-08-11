import json
import os
from collections import Counter


DEFAULT_ROOT = "characterisation"


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def _finished_entries(folder):
    index = _read_json(os.path.join(folder, "index.json"), [])
    if not isinstance(index, list):
        return []

    entries = [
        entry for entry in index
        if isinstance(entry, dict) and entry.get("finished", True) is not False
    ]
    entries.sort(key=lambda entry: int(entry.get("seed", 0)))
    return entries


def _latest_stamp(folder):
    stamp = 0.0
    for name in ("experiment.json", "index.json"):
        path = os.path.join(folder, name)
        try:
            stamp = max(stamp, os.path.getmtime(path))
        except OSError:
            pass
    return stamp


def _outcomes(entries):
    counts = Counter()
    for entry in entries:
        outcome = str(entry.get("characterisation_outcome") or "unclassified")
        counts[outcome] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _final_species(entry):
    components = entry.get("final_components", [])
    if not isinstance(components, list):
        return "?"

    names = []
    for component in components:
        if not isinstance(component, dict):
            continue
        formula = component.get("formula") or "?"
        species_id = component.get("id")
        atoms = component.get("atoms")
        label = f"{species_id} {formula}" if species_id else str(formula)
        names.append(f"{label}({atoms})" if atoms is not None else label)

    return " + ".join(names) if names else "none"


def _contact_rollup(entries):
    rows = []
    safe_rows = []

    for entry in entries:
        diagnostic = entry.get("collision_diagnostics")
        if isinstance(diagnostic, dict):
            rows.append(diagnostic)
        safety = entry.get("collision_start_safety")
        if isinstance(safety, dict):
            safe_rows.append(safety)

    if not rows and not safe_rows:
        return None

    trials = len(rows)
    closest_values = [
        float(item["closest_cross_distance_A"])
        for item in rows
        if item.get("closest_cross_distance_A") is not None
    ]
    longest_values = [float(item.get("longest_contact_fs", 0.0)) for item in rows]

    target_symbols = sorted({
        symbol
        for item in rows
        for symbol in (item.get("by_target_element") or {}).keys()
    })
    by_target = {}

    for symbol in target_symbols:
        element_rows = [
            (item.get("by_target_element") or {}).get(symbol)
            for item in rows
        ]
        element_rows = [item for item in element_rows if isinstance(item, dict)]
        closest = [
            float(item["closest_distance_A"])
            for item in element_rows
            if item.get("closest_distance_A") is not None
        ]
        by_target[symbol] = {
            "trials": len(element_rows),
            "entered": sum(bool(item.get("entered_bond_range")) for item in element_rows),
            "confirmed": sum(bool(item.get("confirmed_contact")) for item in element_rows),
            "longest_contact_fs": max(
                (float(item.get("longest_contact_fs", 0.0)) for item in element_rows),
                default=0.0,
            ),
            "closest_distance_A": min(closest) if closest else None,
        }

    actual_gaps = [
        float(item["actual_start_gap_A"])
        for item in safe_rows
        if item.get("actual_start_gap_A") is not None
    ]
    auto_added = [
        float(item.get("auto_added_gap_A", 0.0))
        for item in safe_rows
    ]

    selected_rows = [
        item.get("selected_target") for item in rows
        if isinstance(item.get("selected_target"), dict)
    ]
    selected_rollup = None
    if selected_rows:
        selected_closest = [
            float(item["closest_distance_A"]) for item in selected_rows
            if item.get("closest_distance_A") is not None
        ]
        selected_first_distance = [
            float(item["first_encounter_min_distance_A"]) for item in selected_rows
            if item.get("first_encounter_min_distance_A") is not None
        ]
        selected_first_miss = [
            float(item["first_encounter_miss_distance_A"]) for item in selected_rows
            if item.get("first_encounter_miss_distance_A") is not None
        ]
        selected_first_time = [
            float(item["first_encounter_time_fs"]) for item in selected_rows
            if item.get("first_encounter_time_fs") is not None
        ]
        selected_target_motion = [
            float(item["first_encounter_target_motion_A"]) for item in selected_rows
            if item.get("first_encounter_target_motion_A") is not None
        ]
        selected_initial = [
            float(item["initial_target_distance_A"]) for item in selected_rows
            if item.get("initial_target_distance_A") is not None
        ]
        selected_rollup = {
            "trials": len(selected_rows),
            "target_symbol": selected_rows[0].get("target_symbol"),
            "entered": sum(bool(item.get("entered_bond_range")) for item in selected_rows),
            "confirmed": sum(bool(item.get("confirmed_contact")) for item in selected_rows),
            "longest_contact_fs": max(
                (float(item.get("longest_contact_fs", 0.0)) for item in selected_rows),
                default=0.0,
            ),
            "closest_distance_A": min(selected_closest) if selected_closest else None,
            "first_encounter_closest_A": min(selected_first_distance) if selected_first_distance else None,
            "first_encounter_farthest_closest_A": max(selected_first_distance) if selected_first_distance else None,
            "smallest_first_encounter_miss_A": min(selected_first_miss) if selected_first_miss else None,
            "largest_first_encounter_miss_A": max(selected_first_miss) if selected_first_miss else None,
            "earliest_first_encounter_fs": min(selected_first_time) if selected_first_time else None,
            "latest_first_encounter_fs": max(selected_first_time) if selected_first_time else None,
            "max_target_motion_at_first_encounter_A": max(selected_target_motion) if selected_target_motion else None,
            "min_initial_target_distance_A": min(selected_initial) if selected_initial else None,
            "max_initial_target_distance_A": max(selected_initial) if selected_initial else None,
            "max_taper": max(
                (float(item.get("max_taper", 0.0)) for item in selected_rows),
                default=0.0,
            ),
        }

    first = rows[0] if rows else {}
    return {
        "diagnostic_trials": trials,
        "start_checks": len(safe_rows),
        "start_safe": sum(bool(item.get("safe_initial_separation")) for item in safe_rows),
        "auto_adjusted": sum(value > 1e-9 for value in auto_added),
        "max_actual_start_gap_A": max(actual_gaps) if actual_gaps else None,
        "bond_threshold": first.get("bond_threshold"),
        "formation_time_fs": first.get("bond_formation_time_fs"),
        "sample_fs": first.get("diagnostic_sample_fs"),
        "entered": sum(bool(item.get("entered_bond_range")) for item in rows),
        "confirmed": sum(bool(item.get("confirmed_contact")) for item in rows),
        "longest_contact_fs": max(longest_values) if longest_values else 0.0,
        "closest_distance_A": min(closest_values) if closest_values else None,
        "by_target_element": by_target,
        "selected_target": selected_rollup,
    }


def _contact_tooltip(entry):
    diagnostic = entry.get("collision_diagnostics")
    if not isinstance(diagnostic, dict):
        return ""

    impact_target = str(entry.get("impact_target") or "com")
    target_atom = entry.get("aim_target_atom")
    target_symbol = entry.get("aim_target_symbol")
    lines = [
        f"aim: {'random / COM' if impact_target == 'com' else impact_target}",
        (
            f"selected target atom: {target_symbol} #{target_atom}"
            if target_atom is not None else "selected target atom: COM"
        ),
        f"bond threshold: taper > {diagnostic.get('bond_threshold', '?')}",
        f"confirmation time: {diagnostic.get('bond_formation_time_fs', '?')} fs",
        f"diagnostic fine sample: {diagnostic.get('diagnostic_fine_sample_fs', diagnostic.get('diagnostic_sample_fs', '?'))} fs",
        f"fine window: {diagnostic.get('diagnostic_fine_window_fs', '?')} fs",
        f"coarse sample: {diagnostic.get('diagnostic_coarse_sample_fs', '?')} fs",
    ]

    strongest = diagnostic.get("strongest_pair")
    if isinstance(strongest, dict):
        lines += [
            "",
            "strongest contact",
            f"  {strongest.get('partner_symbol', '?')} -> {strongest.get('target_symbol', '?')} "
            f"(target atom {strongest.get('target_atom', '?')})",
            f"  closest {strongest.get('closest_distance_A', '?')} A",
            f"  max taper {strongest.get('max_taper', '?')}",
            f"  longest {strongest.get('longest_contact_fs', '?')} fs",
        ]

    selected = diagnostic.get("selected_target")
    if isinstance(selected, dict):
        closest = selected.get("closest_distance_A")
        miss = selected.get("trajectory_miss_distance_A")
        lines += [
            "",
            "selected aim target",
            f"  {selected.get('target_symbol', '?')} atom #{selected.get('target_atom', '?')}",
            f"  initial distance {selected.get('initial_target_distance_A', '?')} A",
            f"  closest {closest if closest is not None else '?'} A",
            f"  max taper {selected.get('max_taper', '?')}",
            f"  longest {selected.get('longest_contact_fs', '?')} fs",
            f"  beam miss at target plane {miss if miss is not None else '?'} A",
            f"  closest at {selected.get('closest_target_time_fs', '?')} fs",
            f"  target moved {selected.get('target_motion_at_closest_A', '?')} A by then",
        ]

    by_target = diagnostic.get("by_target_element") or {}
    if by_target:
        lines += ["", "by target element"]
        for symbol, data in sorted(by_target.items()):
            closest = data.get("closest_distance_A")
            closest_text = "?" if closest is None else f"{float(closest):.3f} A"
            lines.append(
                f"  -> {symbol}: closest {closest_text}, longest "
                f"{float(data.get('longest_contact_fs', 0.0)):.1f} fs, "
                f"entered {'yes' if data.get('entered_bond_range') else 'no'}, "
                f"confirmed {'yes' if data.get('confirmed_contact') else 'no'}"
            )

    safety = entry.get("collision_start_safety")
    if isinstance(safety, dict):
        lines += [
            "",
            "starting separation",
            f"  requested gap {safety.get('requested_start_gap_A', '?')} A",
            f"  requested gap safe {'yes' if safety.get('requested_gap_was_safe') else 'NO'}",
            f"  requested max taper {safety.get('requested_gap_max_bond_taper', '?')}",
            f"  actual gap {safety.get('actual_start_gap_A', '?')} A",
            f"  final start max taper {safety.get('initial_max_bond_taper', '?')}",
            f"  safe {'yes' if safety.get('safe_initial_separation') else 'NO'}",
        ]
        if safety.get("target_geometry_revision") is not None:
            lines += [
                f"  target geometry v{safety.get('target_geometry_revision')}",
                f"  line-of-sight blockers {safety.get('line_of_sight_blockers', '?')}",
                (
                    "  line-of-sight clearance clear"
                    if safety.get("line_of_sight_clearance_A") is None
                    else f"  line-of-sight clearance {float(safety.get('line_of_sight_clearance_A')):.3f} A"
                ),
            ]

    return "\n".join(lines)


def list_experiments(molecule_id=None, root=DEFAULT_ROOT):
    """Return every controlled experiment recorded for a stored species.

    Characterisation data deliberately lives outside the discovery run tree.
    This reader only consumes experiment/index files and never feeds anything
    back into the molecule discovery library.
    """

    if not os.path.isdir(root):
        return []

    experiments = []

    for folder, directories, files in os.walk(root):
        directories.sort()
        if "experiment.json" not in files:
            continue

        experiment = _read_json(os.path.join(folder, "experiment.json"), {})
        if not isinstance(experiment, dict):
            continue

        found_id = experiment.get("molecule_id")
        if molecule_id is not None and found_id != molecule_id:
            continue

        entries = _finished_entries(folder)
        outcomes = _outcomes(entries)
        stable = sum(1 for entry in entries if entry.get("stable") is not False)

        experiments.append({
            "folder": os.path.normpath(folder),
            "molecule_id": found_id,
            "formula": experiment.get("formula"),
            "test": experiment.get("test", "isolated"),
            "physics_mode": experiment.get("physics_mode", "standard"),
            "physics_model": experiment.get("physics_model", "reactive_v1"),
            "physics_parameters": experiment.get("physics_parameters", {}),
            "temperature_K": experiment.get("temperature_K"),
            "duration_ps": experiment.get("duration_ps"),
            "box_A": experiment.get("box_A"),
            "group_size": experiment.get("group_size"),
            "group_policy": experiment.get("group_policy"),
            "partner_id": experiment.get("partner_id"),
            "partner_formula": experiment.get("partner_formula"),
            "approach_factor": experiment.get("approach_factor"),
            "start_gap_A": experiment.get("start_gap_A"),
            "impact_target": experiment.get("impact_target", "com"),
            "target_geometry_revision": experiment.get("target_geometry_revision"),
            "collision_diagnostic_revision": experiment.get("collision_diagnostic_revision"),
            "entries": entries,
            "trials": len(entries),
            "stable_trials": stable,
            "outcomes": outcomes,
            "contacts": _contact_rollup(entries),
            "latest_stamp": _latest_stamp(folder),
            "experiment": experiment,
        })

    experiments.sort(key=lambda item: item.get("latest_stamp", 0.0), reverse=True)
    return experiments


def aggregate(experiments):
    outcomes = Counter()
    trials = 0
    stable = 0

    for experiment in experiments:
        trials += int(experiment.get("trials", 0))
        stable += int(experiment.get("stable_trials", 0))
        outcomes.update(experiment.get("outcomes", {}))

    return {
        "experiments": len(experiments),
        "trials": trials,
        "stable_trials": stable,
        "outcomes": dict(sorted(outcomes.items(), key=lambda item: (-item[1], item[0]))),
    }


def experiment_label(experiment):
    test = experiment.get("test", "?")
    temperature = experiment.get("temperature_K")
    duration = experiment.get("duration_ps")
    trials = int(experiment.get("trials", 0))

    temperature_text = "? K" if temperature is None else f"{float(temperature):g} K"
    duration_text = "? ps" if duration is None else f"{float(duration):g} ps"

    outcomes = experiment.get("outcomes", {})
    outcome_text = ", ".join(
        f"{name} {count}" for name, count in list(outcomes.items())[:2]
    )
    if not outcome_text:
        outcome_text = "no completed trials"

    partner = experiment.get("partner_id")
    if test == "with_partner":
        partner_text = experiment.get("partner_formula") or partner or "?"
        impact_target = str(experiment.get("impact_target") or "com")
        aim_text = "COM" if impact_target == "com" else impact_target
        revision = experiment.get("target_geometry_revision")
        if impact_target != "com" and revision:
            aim_text += f" v{revision}"
        test_text = f"with {partner_text} -> {aim_text}"
    else:
        test_text = test

    physics_mode = str(experiment.get("physics_mode") or "standard")
    physics_text = "HF" if physics_mode == "high_fidelity" else "STD"

    return (
        f"[{physics_text}] {test_text}  {temperature_text}  {duration_text}  |  "
        f"{trials} trial{'s' if trials != 1 else ''}  |  {outcome_text}"
    )


def experiment_summary_lines(experiment):
    if not experiment:
        return ["No characterisation experiment selected."]

    trials = int(experiment.get("trials", 0))
    stable = int(experiment.get("stable_trials", 0))
    outcomes = experiment.get("outcomes", {})

    physics_mode = str(experiment.get("physics_mode") or "standard")
    physics_model = str(experiment.get("physics_model") or "reactive_v1")

    lines = [
        f"{experiment.get('molecule_id', '?')}  {experiment.get('formula', '?')}",
        f"test          {experiment.get('test', '?')}",
        f"physics       {physics_mode}",
        f"model         {physics_model}",
    ]

    physics_parameters = experiment.get("physics_parameters") or {}
    # Keep old v1 result folders readable while showing the v2 control with
    # terminology that matches the actual model.
    if "h_transfer_conjugation_alpha" in physics_parameters:
        lines.append(
            "hf alpha      "
            f"{float(physics_parameters['h_transfer_conjugation_alpha']):g}"
        )
    if "h_transfer_state_mixing_fraction" in physics_parameters:
        lines.append(
            "hf mixing     "
            f"{float(physics_parameters['h_transfer_state_mixing_fraction']):g}"
        )
    if "h_transfer_gate_start" in physics_parameters and "h_transfer_gate_full" in physics_parameters:
        lines.append(
            "hf gate       "
            f"{float(physics_parameters['h_transfer_gate_start']):g}-"
            f"{float(physics_parameters['h_transfer_gate_full']):g} taper"
        )

    if experiment.get("test") == "with_partner":
        partner = experiment.get("partner_formula") or experiment.get("partner_id") or "?"
        lines.append(f"partner       {partner}")
        impact_target = str(experiment.get("impact_target") or "com")
        aim_label = 'random / COM' if impact_target == 'com' else impact_target
        revision = experiment.get("target_geometry_revision")
        if impact_target != "com" and revision:
            aim_label += f" (target geometry v{revision})"
        lines.append(f"aim           {aim_label}")
        lines.append(f"approach      {experiment.get('approach_factor', '?')} x thermal RMS")
        lines.append(f"start gap     {experiment.get('start_gap_A', '?')} A")

    lines += [
        f"temperature   {experiment.get('temperature_K', '?')} K",
        f"duration      {experiment.get('duration_ps', '?')} ps",
        f"box           {experiment.get('box_A', '?')} A",
        f"trials        {trials}",
        f"stable        {stable}/{trials}" if trials else "stable        0/0",
    ]

    contacts = experiment.get("contacts")
    if isinstance(contacts, dict):
        checked = int(contacts.get("start_checks", 0))
        diagnostic_trials = int(contacts.get("diagnostic_trials", 0))
        formation = contacts.get("formation_time_fs")
        formation_text = "?" if formation is None else f"{float(formation):g}"
        closest = contacts.get("closest_distance_A")
        closest_text = "?" if closest is None else f"{float(closest):.3f} A"

        lines += [
            "",
            "contact diagnostics" + (
                f" v{experiment.get('collision_diagnostic_revision')}"
                if experiment.get('collision_diagnostic_revision') else ""
            ),
            f"  diagnostic trials         {diagnostic_trials}/{trials}",
            f"  start outside bond range  {contacts.get('start_safe', 0)}/{checked}",
            f"  auto-adjusted start gap   {contacts.get('auto_adjusted', 0)}/{checked}",
            f"  entered bond range        {contacts.get('entered', 0)}/{diagnostic_trials}",
            f"  held >= {formation_text} fs          {contacts.get('confirmed', 0)}/{diagnostic_trials}",
            f"  longest contact           {float(contacts.get('longest_contact_fs', 0.0)):.1f} fs",
            f"  closest approach          {closest_text}",
        ]

        selected = contacts.get("selected_target")
        if isinstance(selected, dict):
            selected_trials = int(selected.get("trials", 0))
            selected_closest = selected.get("closest_distance_A")
            first_closest_min = selected.get("first_encounter_closest_A")
            first_closest_max = selected.get("first_encounter_farthest_closest_A")
            first_miss_min = selected.get("smallest_first_encounter_miss_A")
            first_miss_max = selected.get("largest_first_encounter_miss_A")
            first_time_min = selected.get("earliest_first_encounter_fs")
            first_time_max = selected.get("latest_first_encounter_fs")
            target_motion_max = selected.get("max_target_motion_at_first_encounter_A")
            selected_closest_text = (
                "?" if selected_closest is None else f"{float(selected_closest):.3f} A"
            )
            first_closest_text = (
                "?" if first_closest_min is None else
                f"{float(first_closest_min):.3f}-{float(first_closest_max):.3f} A"
            )
            first_miss_text = (
                "?" if first_miss_min is None else
                f"{float(first_miss_min):.3f}-{float(first_miss_max):.3f} A"
            )
            first_time_text = (
                "?" if first_time_min is None else
                f"{float(first_time_min):.1f}-{float(first_time_max):.1f} fs"
            )
            motion_text = (
                "?" if target_motion_max is None else f"{float(target_motion_max):.3f} A"
            )
            lines += [
                f"  selected target ({selected.get('target_symbol', '?')})   entered "
                f"{selected.get('entered', 0)}/{selected_trials}, held "
                f"{selected.get('confirmed', 0)}/{selected_trials}",
                f"  selected target closest   {selected_closest_text}",
                f"  first encounter closest   {first_closest_text}",
                f"  first encounter miss      {first_miss_text}",
                f"  first encounter time      {first_time_text}",
                f"  target motion by impact   <= {motion_text}",
            ]

        partner_label = experiment.get("partner_formula") or experiment.get("partner_id") or "partner"
        for symbol, data in (contacts.get("by_target_element") or {}).items():
            element_trials = int(data.get("trials", 0))
            element_closest = data.get("closest_distance_A")
            element_closest_text = (
                "?" if element_closest is None else f"{float(element_closest):.3f} A"
            )
            lines.append(
                f"  {partner_label} -> {symbol:<2}  entered {data.get('entered', 0)}/{element_trials}, "
                f"held {data.get('confirmed', 0)}/{element_trials}, "
                f"longest {float(data.get('longest_contact_fs', 0.0)):.1f} fs, "
                f"closest {element_closest_text}"
            )

    if outcomes:
        lines.append("")
        lines.append("outcomes")
        for name, count in outcomes.items():
            fraction = (100.0 * count / trials) if trials else 0.0
            lines.append(f"  {name:<18} {count:>4}/{trials:<4}  {fraction:5.1f}%")
    else:
        lines += ["", "No completed trials recorded yet."]

    return lines


def run_row(entry):
    diagnostic = entry.get("collision_diagnostics")
    if isinstance(diagnostic, dict):
        contact_fs = float(diagnostic.get("longest_contact_fs", 0.0))
        closest = diagnostic.get("closest_cross_distance_A")
        confirmed = bool(diagnostic.get("confirmed_contact"))
        entered = bool(diagnostic.get("entered_bond_range"))
    else:
        contact_fs = None
        closest = None
        confirmed = False
        entered = False

    return {
        "seed": entry.get("seed", "?"),
        "outcome": entry.get("characterisation_outcome", "unclassified"),
        "stable": entry.get("stable") is not False,
        "contact_fs": contact_fs,
        "closest_A": closest,
        "entered_bond_range": entered,
        "confirmed_contact": confirmed,
        "contact_tooltip": _contact_tooltip(entry),
        "final_temperature": entry.get("final_temperature"),
        "final_species": _final_species(entry),
        "headline": entry.get("headline", ""),
        "file": entry.get("file"),
    }


def recording_path(experiment, entry):
    if not experiment or not entry:
        return None

    name = entry.get("file")
    if not name:
        return None

    path = os.path.join(experiment.get("folder", ""), name)
    return os.path.abspath(path) if os.path.exists(path) else None
