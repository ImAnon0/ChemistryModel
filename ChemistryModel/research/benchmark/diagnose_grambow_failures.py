"""Classify Grambow benchmark failures without changing model physics.

Bond changes in this report are geometry-derived diagnostics, not
ChemistryModel bond declarations.  They use the same deliberately simple
covalent-radius rule for every endpoint and retain atom indices so that
rearrangements with unchanged aggregate bond counts remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_SCORES = Path("research_data/benchmark/grambow_optimised.json")
DEFAULT_ENDPOINTS = Path("research_data/benchmark/grambow_endpoints.json")
DEFAULT_OUTPUT = Path("research_data/benchmark/diagnostics")

COVALENT_RADII = {"H": 0.37, "C": 0.77, "N": 0.75, "O": 0.73}
GEOMETRIC_BOND_SCALE = 1.25
EXPECTED_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2}


def _load_score_rows(path: Path):
    """Load either genuine JSON or the current CSV-with-.json-suffix file."""
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    detected_format = "json" if stripped.startswith(("{", "[")) else "csv"
    if detected_format == "json":
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("rows", payload.get("scores", []))
    else:
        rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"No score rows found in {path}")
    return rows, detected_format


def _distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _pair_type(first, second):
    return "-".join(sorted((first, second)))


def _guess_bonds(geometry):
    symbols = geometry["symbols"]
    coordinates = geometry["coordinates_angstrom"]
    bonds = {}
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            cutoff = GEOMETRIC_BOND_SCALE * (
                COVALENT_RADII[symbols[first]]
                + COVALENT_RADII[symbols[second]]
            )
            distance = _distance(coordinates[first], coordinates[second])
            if distance < cutoff:
                bonds[(first, second)] = {
                    "atoms": [first, second],
                    "elements": [symbols[first], symbols[second]],
                    "type": _pair_type(symbols[first], symbols[second]),
                    "distance_angstrom": distance,
                    "cutoff_angstrom": cutoff,
                }
    return bonds


def _formula(element_counts):
    order = []
    if element_counts.get("C"):
        order.append("C")
    if element_counts.get("H"):
        order.append("H")
    order.extend(sorted(key for key in element_counts if key not in {"C", "H"}))
    return "".join(
        symbol + (str(element_counts[symbol]) if element_counts[symbol] != 1 else "")
        for symbol in order
    )


def _component_formulas(geometry, bonds):
    symbols = geometry["symbols"]
    adjacency = [set() for _ in symbols]
    for first, second in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)

    unseen = set(range(len(symbols)))
    formulas = []
    components = []
    while unseen:
        root = min(unseen)
        stack = [root]
        members = []
        unseen.remove(root)
        while stack:
            atom = stack.pop()
            members.append(atom)
            for neighbour in adjacency[atom]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        counts = Counter(symbols[index] for index in members)
        formulas.append(_formula(counts))
        components.append(sorted(members))
    ordering = sorted(range(len(formulas)), key=lambda index: (-len(components[index]), formulas[index]))
    return [formulas[index] for index in ordering], [components[index] for index in ordering]


def _bond_text(record):
    first, second = record["atoms"]
    return (
        f"{record['elements'][0]}{first}-{record['elements'][1]}{second}"
        f"@{record['distance_angstrom']:.3f}A"
    )


def _change_summary(source_bonds, target_bonds):
    formed = [target_bonds[key] for key in sorted(set(target_bonds) - set(source_bonds))]
    broken = [source_bonds[key] for key in sorted(set(source_bonds) - set(target_bonds))]
    return {
        "formed": formed,
        "broken": broken,
        "formed_types": dict(Counter(record["type"] for record in formed)),
        "broken_types": dict(Counter(record["type"] for record in broken)),
    }


def _has_hydrogen(record):
    return "H" in record["elements"]


def _hydrogen_partner(record):
    return next((element for element in record["elements"] if element != "H"), "H")


def _classify(reactant, product, change):
    formed = change["formed"]
    broken = change["broken"]
    formed_types = set(change["formed_types"])
    broken_types = set(change["broken_types"])
    changed_types = formed_types | broken_types
    tags = []

    if any("N" in record["elements"] for record in formed + broken):
        tags.append("N chemistry")
    if any("O" in record["elements"] for record in formed + broken):
        tags.append("O chemistry")
    if "O-O" in changed_types:
        tags.append("O-O/peroxide topology")

    h_formed = [record for record in formed if _has_hydrogen(record)]
    h_broken = [record for record in broken if _has_hydrogen(record)]
    if h_formed and h_broken:
        donors = "/".join(sorted({_hydrogen_partner(record) for record in h_broken}))
        acceptors = "/".join(sorted({_hydrogen_partner(record) for record in h_formed}))
        primary = f"H transfer ({donors}-H -> {acceptors}-H)"
        tags.append("H transfer")
    elif "O-O" in changed_types:
        primary = "O-O formation" if "O-O" in formed_types else "O-O dissociation"
    elif formed and not broken:
        primary = f"{formed[0]['type']} formation" if len(formed_types) == 1 else "multi-bond formation"
    elif broken and not formed:
        primary = f"{broken[0]['type']} dissociation" if len(broken_types) == 1 else "multi-bond dissociation"
    elif formed or broken:
        primary = "bond rearrangement"
        tags.append("rearrangement")
    else:
        primary = "no inferred endpoint bond change"

    reactant_bonds = _guess_bonds(reactant)
    product_bonds = _guess_bonds(product)
    reactant_formulas, _ = _component_formulas(reactant, reactant_bonds)
    product_formulas, _ = _component_formulas(product, product_bonds)
    if len(product_formulas) < len(reactant_formulas):
        tags.append("association")
    elif len(product_formulas) > len(reactant_formulas):
        tags.append("fragmentation")

    for geometry, bonds in ((reactant, reactant_bonds), (product, product_bonds)):
        degree = Counter(index for pair in bonds for index in pair)
        if any(degree[index] > EXPECTED_VALENCE[symbol] for index, symbol in enumerate(geometry["symbols"])):
            tags.append("geometrically high coordination")
            break
    return primary, sorted(set(tags))


def _aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    result = []
    for label, members in groups.items():
        barrier_errors = [row["barrier_error_eV"] for row in members]
        reaction_errors = [row["reaction_error_eV"] for row in members]
        result.append({
            key: label,
            "count": len(members),
            "barrier_signed_mean_eV": sum(barrier_errors) / len(members),
            "barrier_mae_eV": sum(abs(value) for value in barrier_errors) / len(members),
            "reaction_signed_mean_eV": sum(reaction_errors) / len(members),
            "reaction_mae_eV": sum(abs(value) for value in reaction_errors) / len(members),
            "barrier_sign_failures": sum(not row["barrier_sign_agrees"] for row in members),
        })
    return sorted(result, key=lambda row: (-row["barrier_mae_eV"], -row["count"], row[key]))


def _family(primary):
    if primary.startswith("H transfer"):
        return "hydrogen transfer"
    if primary.startswith("O-O"):
        return "O-O/peroxide chemistry"
    if primary == "bond rearrangement":
        return "bond rearrangement"
    if "formation" in primary or "dissociation" in primary:
        return "bond formation/dissociation"
    return "unclassified endpoint topology"


def _tag_summary(rows):
    expanded = []
    for row in rows:
        for tag in row["tags"]:
            tagged = dict(row)
            tagged["tag"] = tag
            expanded.append(tagged)
    return _aggregate(expanded, "tag") if expanded else []


def _markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def diagnose(scores_path, endpoints_path, top_count=50):
    raw_scores, detected_format = _load_score_rows(scores_path)
    endpoint_payload = json.loads(endpoints_path.read_text(encoding="utf-8"))
    endpoints = defaultdict(dict)
    for geometry in endpoint_payload["geometries"]:
        endpoints[geometry["reaction_id"]][geometry["region"]] = geometry

    rows = []
    for raw in raw_scores:
        reaction_id = raw["reaction_id"]
        geometries = endpoints[reaction_id]
        missing = {"reactant", "transition_state", "product"} - set(geometries)
        if missing:
            raise ValueError(f"{reaction_id} missing endpoints: {sorted(missing)}")
        reactant = geometries["reactant"]
        transition = geometries["transition_state"]
        product = geometries["product"]
        reactant_bonds = _guess_bonds(reactant)
        transition_bonds = _guess_bonds(transition)
        product_bonds = _guess_bonds(product)
        endpoint_change = _change_summary(reactant_bonds, product_bonds)
        transition_change = _change_summary(reactant_bonds, transition_bonds)
        primary, tags = _classify(reactant, product, endpoint_change)
        reactant_formulas, reactant_components = _component_formulas(reactant, reactant_bonds)
        product_formulas, product_components = _component_formulas(product, product_bonds)
        counts = Counter(reactant["symbols"])
        rows.append({
            "reaction_id": reaction_id,
            "atom_count": int(raw["atom_count"]),
            "reference_barrier_eV": float(raw["reference_barrier_eV"]),
            "model_barrier_eV": float(raw["model_barrier_eV"]),
            "barrier_error_eV": float(raw["barrier_error_eV"]),
            "reference_reaction_eV": float(raw["reference_reaction_eV"]),
            "model_reaction_eV": float(raw["model_reaction_eV"]),
            "reaction_error_eV": float(raw["reaction_error_eV"]),
            "barrier_sign_agrees": bool(int(raw["barrier_sign_agrees"])),
            "element_counts": dict(sorted(counts.items())),
            "formula": _formula(counts),
            "reactant_composition": " + ".join(reactant_formulas),
            "product_composition": " + ".join(product_formulas),
            "reactant_components": reactant_components,
            "product_components": product_components,
            "primary_class": primary,
            "failure_family": _family(primary),
            "tags": tags,
            "formed_bonds": [_bond_text(record) for record in endpoint_change["formed"]],
            "broken_bonds": [_bond_text(record) for record in endpoint_change["broken"]],
            "formed_bond_types": endpoint_change["formed_types"],
            "broken_bond_types": endpoint_change["broken_types"],
            "ts_formed_bonds": [_bond_text(record) for record in transition_change["formed"]],
            "ts_broken_bonds": [_bond_text(record) for record in transition_change["broken"]],
        })

    barrier_sorted = sorted(rows, key=lambda row: abs(row["barrier_error_eV"]), reverse=True)
    reaction_sorted = sorted(rows, key=lambda row: abs(row["reaction_error_eV"]), reverse=True)
    barrier_rank = {row["reaction_id"]: rank for rank, row in enumerate(barrier_sorted, 1)}
    reaction_rank = {row["reaction_id"]: rank for rank, row in enumerate(reaction_sorted, 1)}
    for row in rows:
        row["barrier_abs_error_rank"] = barrier_rank[row["reaction_id"]]
        row["reaction_abs_error_rank"] = reaction_rank[row["reaction_id"]]
        row["top_barrier_failure"] = row["barrier_abs_error_rank"] <= top_count
        row["top_reaction_failure"] = row["reaction_abs_error_rank"] <= top_count

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scores": str(scores_path),
        "scores_detected_format": detected_format,
        "endpoints": str(endpoints_path),
        "bond_inference": {
            "kind": "geometry diagnostic only; not ChemistryModel topology",
            "covalent_radii_angstrom": COVALENT_RADII,
            "cutoff_rule": f"r < {GEOMETRIC_BOND_SCALE} * (r_cov_i + r_cov_j)",
        },
        "row_count": len(rows),
        "top_count": top_count,
        "rows": rows,
        "class_summary": _aggregate(rows, "primary_class"),
        "family_summary": _aggregate(rows, "failure_family"),
        "tag_summary": _tag_summary(rows),
        "formula_summary": _aggregate(rows, "formula"),
        "worst_barriers": barrier_sorted[:top_count],
        "worst_reactions": reaction_sorted[:top_count],
        "sign_failures": [row for row in rows if not row["barrier_sign_agrees"]],
    }


def _write_outputs(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "grambow_failure_diagnosis.json"
    csv_path = output_dir / "grambow_failure_rows.csv"
    report_path = output_dir / "grambow_failure_report.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for row in result["rows"]:
        flat = dict(row)
        for key in (
            "element_counts", "reactant_components", "product_components",
            "tags", "formed_bonds", "broken_bonds", "formed_bond_types",
            "broken_bond_types", "ts_formed_bonds", "ts_broken_bonds",
        ):
            flat[key] = json.dumps(flat[key], separators=(",", ":"))
        csv_rows.append(flat)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    class_columns = [
        ("primary_class", "Failure class"), ("count", "Count"),
        ("barrier_signed_mean_eV", "Barrier mean"), ("barrier_mae_eV", "Barrier MAE"),
        ("reaction_signed_mean_eV", "Reaction mean"), ("reaction_mae_eV", "Reaction MAE"),
        ("barrier_sign_failures", "Sign failures"),
    ]
    worst_columns = [
        ("reaction_id", "Reaction"), ("primary_class", "Class"),
        ("reactant_composition", "Reactants"), ("product_composition", "Products"),
        ("model_barrier_eV", "Model"), ("reference_barrier_eV", "Reference"),
        ("barrier_error_eV", "Error"),
    ]
    reaction_columns = [
        ("reaction_id", "Reaction"), ("primary_class", "Class"),
        ("reactant_composition", "Reactants"), ("product_composition", "Products"),
        ("model_reaction_eV", "Model"), ("reference_reaction_eV", "Reference"),
        ("reaction_error_eV", "Error"),
    ]
    report = [
        "# Grambow optimised-valence failure diagnosis",
        "",
        f"Rows: {result['row_count']}. Score format detected: {result['scores_detected_format']}.",
        "",
        "Bond labels below are geometry-derived diagnostics, not ChemistryModel bond declarations.",
        "",
        "## Broad failure families",
        "",
        _markdown_table(result["family_summary"], [
            ("failure_family", "Family"), *class_columns[1:]
        ]),
        "",
        "## Detailed failure classes",
        "",
        _markdown_table(result["class_summary"], class_columns),
        "",
        f"## Worst {result['top_count']} barrier errors",
        "",
        _markdown_table(result["worst_barriers"], worst_columns),
        "",
        f"## Worst {result['top_count']} reaction-energy errors",
        "",
        _markdown_table(result["worst_reactions"], reaction_columns),
        "",
        "## Barrier-sign failures",
        "",
        _markdown_table(result["sign_failures"], worst_columns),
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    return json_path, csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=50)
    arguments = parser.parse_args()
    result = diagnose(arguments.scores, arguments.endpoints, arguments.top)
    paths = _write_outputs(result, arguments.output_dir)
    print(f"analysed {result['row_count']} reactions")
    print(f"score format: {result['scores_detected_format']}")
    print(f"barrier sign failures: {len(result['sign_failures'])}")
    print("largest failure classes by barrier MAE:")
    for row in result["class_summary"][:10]:
        print(
            f"  {row['primary_class']:<34s} n={row['count']:3d}  "
            f"barrier mean={row['barrier_signed_mean_eV']:+7.3f}  "
            f"MAE={row['barrier_mae_eV']:7.3f} eV"
        )
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
