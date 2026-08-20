"""Quantify which current energy terms track Grambow endpoint errors.

The term-removal figures are sensitivity diagnostics only.  They are not
candidate force fields: removing an energy term from frozen endpoint scores
does not test forces, continuity, dynamics, or transferability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_COMPONENTS = Path(
    "research_data/benchmark/diagnostics/grambow_full_component_audit_components.csv"
)
DEFAULT_CLASSES = Path(
    "research_data/benchmark/diagnostics/grambow_failure_diagnosis.json"
)
DEFAULT_OUTPUT = Path(
    "research_data/benchmark/diagnostics/grambow_component_bias.json"
)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _pearson(first, second):
    first_mean = _mean(first)
    second_mean = _mean(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    denominator = math.sqrt(
        sum((value - first_mean) ** 2 for value in first)
        * sum((value - second_mean) ** 2 for value in second)
    )
    return numerator / denominator if denominator else 0.0


def _metrics(errors, model_values=None, reference_values=None):
    result = {
        "count": len(errors),
        "mae_eV": _mean([abs(value) for value in errors]),
        "rmse_eV": math.sqrt(_mean([value * value for value in errors])),
        "signed_mean_eV": _mean(errors),
    }
    if model_values is not None and reference_values is not None:
        result["sign_agreement"] = _mean([
            (model > 0.0) == (reference > 0.0)
            for model, reference in zip(model_values, reference_values)
        ])
    return result


def _float_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for source in raw:
        row = {}
        for key, value in source.items():
            if key in {"reaction_id"}:
                row[key] = value
            elif key == "barrier_sign_agrees":
                row[key] = value.lower() in {"1", "true"}
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
        rows.append(row)
    return rows


def _kind_analysis(rows, kind):
    error_key = f"{kind}_error_eV"
    model_key = f"scored_model_{kind}_eV"
    reference_key = f"reference_{kind}_eV"
    component_names = (
        "base_bond", "base_overcoordination", "base_angle",
        "h_state_correction", "valence_topology_correction",
        "effective_angle_total",
    )
    errors = [row[error_key] for row in rows]
    correlations = []
    for component in component_names:
        values = [row[f"{kind}_{component}_eV"] for row in rows]
        correlations.append({
            "component": component,
            "signed_error_correlation": _pearson(errors, values),
            "absolute_error_correlation": _pearson(
                [abs(value) for value in errors],
                [abs(value) for value in values],
            ),
            "mean_absolute_contribution_eV": _mean([abs(value) for value in values]),
        })
    correlations.sort(key=lambda row: abs(row["signed_error_correlation"]), reverse=True)

    pressure_metrics = (
        "heavy_base_overcoordination_eV", "hydrogen_base_overcoordination_eV",
        "coordination_gap_sum", "max_coordination_gap",
        "radially_overcoordinated_atom_count", "max_radial_valence_excess",
        "suppressed_contact_count",
    )
    pressure_correlations = []
    for metric in pressure_metrics:
        values = [row[f"{kind}_pressure_{metric}"] for row in rows]
        pressure_correlations.append({
            "metric": metric,
            "signed_error_correlation": _pearson(errors, values),
            "absolute_error_correlation": _pearson(
                [abs(value) for value in errors],
                [abs(value) for value in values],
            ),
        })
    pressure_correlations.sort(
        key=lambda row: abs(row["signed_error_correlation"]), reverse=True
    )

    scenarios = {
        "current": [0.0 for _ in rows],
        "remove_heavy_base_overcoordination_diagnostic": [
            row[f"{kind}_pressure_heavy_base_overcoordination_eV"] for row in rows
        ],
        "remove_all_base_overcoordination_diagnostic": [
            row[f"{kind}_base_overcoordination_eV"] for row in rows
        ],
        "remove_effective_angle_diagnostic": [
            row[f"{kind}_effective_angle_total_eV"] for row in rows
        ],
        "remove_h_state_correction_diagnostic": [
            row[f"{kind}_h_state_correction_eV"] for row in rows
        ],
        "remove_heavy_over_and_effective_angle_diagnostic": [
            row[f"{kind}_pressure_heavy_base_overcoordination_eV"]
            + row[f"{kind}_effective_angle_total_eV"]
            for row in rows
        ],
    }
    sensitivity = []
    for name, removed in scenarios.items():
        changed_errors = [error - delta for error, delta in zip(errors, removed)]
        changed_models = [row[model_key] - delta for row, delta in zip(rows, removed)]
        metrics = _metrics(
            changed_errors, changed_models, [row[reference_key] for row in rows]
        )
        metrics["scenario"] = name
        metrics["warning"] = "frozen-endpoint term-removal diagnostic; not a physical model"
        sensitivity.append(metrics)

    largest = Counter()
    largest_top50 = Counter()
    ordered = sorted(rows, key=lambda row: abs(row[error_key]), reverse=True)
    for index, row in enumerate(ordered):
        component = max(
            component_names,
            key=lambda name: abs(row[f"{kind}_{name}_eV"]),
        )
        largest[component] += 1
        if index < 50:
            largest_top50[component] += 1

    return {
        "current_metrics": _metrics(
            errors,
            [row[model_key] for row in rows],
            [row[reference_key] for row in rows],
        ),
        "component_correlations": correlations,
        "pressure_correlations": pressure_correlations,
        "largest_component_counts_all": dict(largest),
        "largest_component_counts_top50": dict(largest_top50),
        "term_removal_sensitivity": sensitivity,
    }


def analyse(component_path, classes_path):
    rows = _float_rows(component_path)
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    class_by_id = {
        row["reaction_id"]: row["primary_class"] for row in classes["rows"]
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[class_by_id.get(row["reaction_id"], "unknown")].append(row)
    class_pressure = []
    for label, members in grouped.items():
        class_pressure.append({
            "primary_class": label,
            "count": len(members),
            "barrier_error_mean_eV": _mean([row["barrier_error_eV"] for row in members]),
            "barrier_heavy_over_delta_mean_eV": _mean([
                row["barrier_pressure_heavy_base_overcoordination_eV"] for row in members
            ]),
            "reaction_error_mean_eV": _mean([row["reaction_error_eV"] for row in members]),
            "reaction_heavy_over_delta_mean_eV": _mean([
                row["reaction_pressure_heavy_base_overcoordination_eV"] for row in members
            ]),
        })
    class_pressure.sort(key=lambda row: abs(row["barrier_error_mean_eV"]), reverse=True)
    return {
        "component_source": str(component_path),
        "classification_source": str(classes_path),
        "row_count": len(rows),
        "interpretation_warning": (
            "Correlations and frozen-endpoint term removals identify candidate causes only; "
            "they are not parameter fits or acceptable production changes."
        ),
        "barrier": _kind_analysis(rows, "barrier"),
        "reaction": _kind_analysis(rows, "reaction"),
        "class_pressure_summary": class_pressure,
    }


def _write_report(result, json_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path = json_path.with_suffix(".md")
    lines = [
        "# Grambow component-bias audit", "",
        result["interpretation_warning"], "",
    ]
    for kind in ("barrier", "reaction"):
        section = result[kind]
        current = section["current_metrics"]
        lines.extend([
            f"## {kind.title()}", "",
            f"Current MAE {current['mae_eV']:.6f} eV; RMSE {current['rmse_eV']:.6f} eV; "
            f"signed mean {current['signed_mean_eV']:+.6f} eV.", "",
            "### Signed-error correlations", "",
            "| Component | r(error, component) | r(abs error, abs component) | Mean abs contribution |",
            "| --- | --- | --- | --- |",
        ])
        for row in section["component_correlations"]:
            lines.append(
                f"| {row['component']} | {row['signed_error_correlation']:+.4f} | "
                f"{row['absolute_error_correlation']:+.4f} | "
                f"{row['mean_absolute_contribution_eV']:.4f} eV |"
            )
        lines.extend([
            "", "### Frozen-endpoint sensitivity (diagnostic only)", "",
            "| Scenario | MAE | RMSE | Signed mean | Sign agreement |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in section["term_removal_sensitivity"]:
            lines.append(
                f"| {row['scenario']} | {row['mae_eV']:.4f} | {row['rmse_eV']:.4f} | "
                f"{row['signed_mean_eV']:+.4f} | {row['sign_agreement']:.1%} |"
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = analyse(arguments.components, arguments.classes)
    markdown = _write_report(result, arguments.output)
    print(f"analysed {result['row_count']} reactions")
    for kind in ("barrier", "reaction"):
        strongest = result[kind]["component_correlations"][0]
        print(
            f"{kind}: strongest signed correlation {strongest['component']} "
            f"r={strongest['signed_error_correlation']:+.4f}"
        )
        for row in result[kind]["term_removal_sensitivity"]:
            print(f"  {row['scenario']}: MAE={row['mae_eV']:.4f} eV")
    print(f"wrote {arguments.output}")
    print(f"wrote {markdown}")


if __name__ == "__main__":
    main()
