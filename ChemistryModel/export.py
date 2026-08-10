import argparse
import csv
import json
import os

import numpy as np


# ============================================================
# Everything in one file
# ============================================================
#
# Walks every batch, reads each index, and writes one row per
# run to a CSV. Also writes a species table showing how many
# runs of each batch contained each molecule, and a summary of
# the per-batch averages with a t-test between the two largest
# batches.
#
#   py export.py runs
#   py export.py runs --out results
#
# The CSV opens in Excel or anything else, so the numbers stop
# being locked inside the browser.


NUMERIC_COLUMNS = [
    "atoms",
    "box",
    "picoseconds",
    "strikes",
    "strike_temperature",
    "strike_dissociation",
    "frames",
    "heavy_bonds_formed",
    "late_formed",
    "late_broke",
    "turnovers",
    "largest_closed",
    "largest_closed_heavy",
    "largest_any",
    "largest_any_heavy",
    "most_carbon",
    "best_tail",
    "best_chain",
    "amphiphiles",
    "species_count",
    "final_temperature",
    "final_potential",
    "wall_seconds",
    "energy_jumps",
    "largest_energy_jump",
]


def find_batches(root):
    found = []

    if os.path.exists(os.path.join(root, "index.json")):
        found.append((os.path.basename(os.path.abspath(root)), root))

    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)

            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "index.json")
            ):
                found.append((name, path))

    return found


def load(batches):
    rows = []

    for label, path in batches:
        with open(os.path.join(path, "index.json")) as handle:
            for entry in json.load(handle):
                record = {"batch": label}

                record["stable"] = entry.get("stable", True)
                record["mixture"] = entry.get("mixture", "")
                record["seed"] = entry.get("seed", -1)
                record["run"] = entry.get("number", -1)

                for column in NUMERIC_COLUMNS:
                    record[column] = entry.get(column, "")

                record["closed_shell"] = " ".join(
                    entry.get("closed_shell", [])
                )
                record["final_species"] = " ".join(
                    entry.get("final_species", [])
                )
                record["headline"] = entry.get("headline", "")

                rows.append(record)

    return rows


def write_runs(rows, path):
    if not rows:
        return

    columns = (
        ["batch", "run", "seed", "stable", "mixture"]
        + NUMERIC_COLUMNS
        + ["headline", "closed_shell", "final_species"]
    )

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_species(rows, path):
    # One row per species, one column per batch, holding the
    # number of runs in that batch where it survived to the end.

    batches = sorted({row["batch"] for row in rows})

    counts = {}
    totals = {label: 0 for label in batches}

    for row in rows:
        totals[row["batch"]] += 1

        for name in row["final_species"].split():
            counts.setdefault(name, {}).setdefault(row["batch"], 0)
            counts[name][row["batch"]] += 1

    closed = {}

    for row in rows:
        for name in row["closed_shell"].split():
            closed.setdefault(name, {}).setdefault(row["batch"], 0)
            closed[name][row["batch"]] += 1

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)

        header = ["species"]

        for label in batches:
            header += [
                f"{label} present", f"{label} closed shell"
            ]

        header += ["runs total"]

        writer.writerow(header)

        ordered = sorted(
            counts,
            key=lambda name: (
                -sum(counts[name].values()), -len(name)
            )
        )

        for name in ordered:
            row = [name]

            for label in batches:
                row += [
                    counts[name].get(label, 0),
                    closed.get(name, {}).get(label, 0),
                ]

            row.append(sum(counts[name].values()))

            writer.writerow(row)

        writer.writerow([])
        writer.writerow(
            ["runs in each batch"]
            + [
                value
                for label in batches
                for value in (totals[label], "")
            ]
        )


def paired_comparison(rows, columns):
    # Matched seeds are worth far more than the same number of
    # unmatched runs.
    #
    # Two runs from the same seed start from an identical box, so
    # everything before the first discharge is the same trajectory.
    # Comparing them as independent groups throws that away and
    # leaves seed-to-seed variation as the dominant noise: these
    # controls ranged from six to nine atoms purely on starting
    # configuration, which is larger than any effect being looked
    # for. Subtracting within a pair cancels it entirely.

    from scipy.stats import ttest_rel, wilcoxon

    by_condition = {}

    for row in rows:
        strikes = row.get("strikes", "")

        try:
            strikes = int(float(strikes))
        except (TypeError, ValueError):
            continue

        condition = "strike" if strikes > 0 else "quiet"

        key = (
            row.get("mixture", ""),
            str(row.get("atoms", "")),
            str(row.get("picoseconds", "")),
        )

        by_condition.setdefault(key, {}).setdefault(
            condition, {}
        )[row.get("seed")] = row

    lines = []

    for key, conditions in sorted(by_condition.items()):
        if len(conditions) < 2:
            continue

        quiet = conditions.get("quiet", {})
        strike = conditions.get("strike", {})

        shared = sorted(set(quiet) & set(strike))

        if len(shared) < 3:
            continue

        mixture, atoms, picoseconds = key

        lines.append("")
        lines.append("")
        lines.append("PAIRED BY SEED")
        lines.append("=" * 78)
        lines.append("")
        lines.append(
            f"  {mixture}, {atoms} atoms, {picoseconds} ps"
        )
        lines.append(
            f"  {len(shared)} seeds run under both conditions: "
            + ", ".join(str(seed) for seed in shared[:12])
            + (" ..." if len(shared) > 12 else "")
        )

        unpaired = (set(quiet) ^ set(strike))

        if unpaired:
            lines.append(
                f"  {len(unpaired)} runs left out for having no "
                f"partner at the same seed."
            )

        lines.append("")
        lines.append(
            f"  {'metric':<18}{'quiet':>9}{'strike':>9}"
            f"{'mean diff':>11}{'better':>8}{'paired p':>10}"
            f"{'signed p':>10}"
        )
        lines.append("  " + "-" * 73)

        for column, title in columns:
            first = []
            second = []

            for seed in shared:
                try:
                    first.append(float(quiet[seed][column]))
                    second.append(float(strike[seed][column]))
                except (TypeError, ValueError, KeyError):
                    first = []
                    break

            if len(first) < 3:
                continue

            a = np.array(first)
            b = np.array(second)

            differences = b - a

            better = int(np.sum(differences > 0))
            worse = int(np.sum(differences < 0))

            try:
                p1 = ttest_rel(a, b).pvalue
            except Exception:
                p1 = float("nan")

            try:
                p2 = wilcoxon(a, b).pvalue
            except Exception:
                p2 = float("nan")

            mark = ""

            if p2 < 0.05 or p1 < 0.05:
                mark = "  *"

            lines.append(
                f"  {title:<18}{a.mean():>9.2f}{b.mean():>9.2f}"
                f"{differences.mean():>+11.2f}"
                f"{better:>4}/{worse:<3}{p1:>10.3f}{p2:>10.3f}{mark}"
            )

        lines.append("")
        lines.append(
            "  The 'better' column counts seeds where strikes gave"
        )
        lines.append(
            "  a higher value, against those where they gave a"
        )
        lines.append(
            "  lower one. Ties are in neither. A consistent"
        )
        lines.append(
            "  direction across seeds matters more than the size of"
        )
        lines.append(
            "  the average difference."
        )

    return lines


def summarise(all_rows, path):
    # Runs where the integrator failed are reported but kept out
    # of the averages. Their chemistry happened in a box that was
    # briefly thousands of degrees hotter than intended, so they
    # are not measuring the condition they were meant to.

    broken = [
        row for row in all_rows if row.get("stable") is False
    ]

    rows = [row for row in all_rows if row.get("stable") is not False]

    batches = sorted({row["batch"] for row in rows})

    lines = []

    if broken:
        lines.append("EXCLUDED: INTEGRATION FAILURES")
        lines.append("=" * 78)
        lines.append("")

        for row in broken:
            lines.append(
                f"  {row['batch']}  run {row['run']}  "
                f"seed {row['seed']}   energy jumped "
                f"{row.get('largest_energy_jump', '?')} eV"
            )

        lines.append("")
        lines.append(
            f"  {len(broken)} of {len(all_rows)} runs excluded "
            f"from everything below."
        )
        lines.append("")
        lines.append("")

    lines.append("BATCH AVERAGES")
    lines.append("=" * 78)
    lines.append("")

    columns = [
        ("heavy_bonds_formed", "bonds"),
        ("late_formed", "late+"),
        ("late_broke", "late-"),
        ("turnovers", "turnover"),
        ("largest_closed", "closed"),
        ("largest_any", "any"),
        ("most_carbon", "carbons"),
        ("best_chain", "chain"),
        ("best_tail", "tail"),
        ("species_count", "species"),
    ]

    header = f"{'batch':<26}{'n':>4}"

    for key, title in columns:
        header += f"{title:>10}"

    lines.append(header)
    lines.append("-" * len(header))

    values = {}

    for label in batches:
        subset = [row for row in rows if row["batch"] == label]

        line = f"{label[:26]:<26}{len(subset):>4}"

        values[label] = {}

        for key, title in columns:
            numbers = [
                float(row[key]) for row in subset
                if row[key] not in ("", None)
            ]

            values[label][key] = numbers

            if numbers:
                line += f"{np.mean(numbers):>10.1f}"
            else:
                line += f"{'-':>10}"

        lines.append(line)

    # If there are exactly two batches worth comparing, test them.

    biggest = sorted(
        batches,
        key=lambda label: -len(
            [row for row in rows if row["batch"] == label]
        )
    )[:2]

    if len(biggest) == 2:
        from scipy.stats import ttest_ind, mannwhitneyu

        first, second = biggest

        lines.append("")
        lines.append("")
        lines.append(f"{first}  vs  {second}")
        lines.append("=" * 78)
        lines.append("")
        lines.append(
            f"{'metric':<22}{'mean 1':>10}{'mean 2':>10}"
            f"{'diff':>10}{'t-test':>10}{'rank test':>12}"
        )
        lines.append("-" * 74)

        for key, title in columns:
            a = np.array(values[first][key], float)
            b = np.array(values[second][key], float)

            if len(a) < 2 or len(b) < 2:
                continue

            try:
                p1 = ttest_ind(a, b, equal_var=False).pvalue
            except Exception:
                p1 = float("nan")

            try:
                p2 = mannwhitneyu(a, b).pvalue
            except Exception:
                p2 = float("nan")

            lines.append(
                f"{title:<22}{a.mean():>10.2f}{b.mean():>10.2f}"
                f"{b.mean() - a.mean():>+10.2f}"
                f"{p1:>10.3f}{p2:>12.3f}"
            )

        lines.append("")
        lines.append(
            "  A p value below 0.05 is the usual threshold for"
        )
        lines.append(
            "  calling a difference real. With small samples the"
        )
        lines.append(
            "  rank test is the more trustworthy of the two, and"
        )
        lines.append(
            "  the t-test misbehaves when one group has no spread."
        )

    lines += paired_comparison(rows, columns)

    with open(path, "w") as handle:
        handle.write("\n".join(lines))

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Export every run to CSV."
    )

    parser.add_argument("directory", nargs="?", default="runs")
    parser.add_argument("--out", default="results")

    options = parser.parse_args()

    batches = find_batches(options.directory)

    if not batches:
        raise SystemExit(
            f"no batches found under {options.directory}"
        )

    rows = load(batches)

    if not rows:
        raise SystemExit("batches found but no runs in them")

    os.makedirs(options.out, exist_ok=True)

    runs_path = os.path.join(options.out, "runs.csv")
    species_path = os.path.join(options.out, "species.csv")
    summary_path = os.path.join(options.out, "summary.txt")

    write_runs(rows, runs_path)
    write_species(rows, species_path)

    lines = summarise(rows, summary_path)

    print("\n".join(lines))
    print()
    print(f"{len(rows)} runs from {len(batches)} batches")
    print(f"  {runs_path}      one row per run")
    print(f"  {species_path}   which molecules in which batch")
    print(f"  {summary_path}    the above, as a file")


if __name__ == "__main__":
    main()