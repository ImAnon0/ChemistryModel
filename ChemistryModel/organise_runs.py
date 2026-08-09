import argparse
import json
import os
import shutil


# ============================================================
# Sorting a mixed runs folder
# ============================================================
#
# The batch runner appends to whatever index it finds, so several
# experiments pile into one folder and the frequencies stop
# meaning anything: "methanol in 5/18 runs" is worthless if those
# eighteen runs used three different box sizes.
#
# This groups them by the settings that actually change the
# chemistry - mixture, atom count, duration, strikes - and gives
# each group its own folder and index.
#
#   py organise_runs.py runs              show what is there
#   py organise_runs.py runs --apply      actually move things


def group_key(entry):
    return (
        entry.get("mixture", "?"),
        entry.get("atoms", 0),
        entry.get("picoseconds", 0),
        entry.get("strikes", 0),
    )


def group_label(key):
    mixture, atoms, picoseconds, strikes = key

    safe = mixture.replace(" ", "_").replace("+", "plus")

    return (
        f"{safe}_{atoms}atoms_{picoseconds:g}ps_"
        f"{strikes}strikes"
    )


def organise(directory, apply=False):
    index_path = os.path.join(directory, "index.json")

    if not os.path.exists(index_path):
        raise SystemExit(f"no index.json in {directory}")

    with open(index_path) as handle:
        index = json.load(handle)

    groups = {}

    for entry in index:
        groups.setdefault(group_key(entry), []).append(entry)

    print(f"{len(index)} runs in {directory} fall into "
          f"{len(groups)} groups:")
    print()

    for key, entries in sorted(
        groups.items(), key=lambda item: -len(item[1])
    ):
        mixture, atoms, picoseconds, strikes = key

        print(
            f"  {len(entries):3d} runs   {mixture:<16} "
            f"{atoms:4d} atoms   {picoseconds:g} ps   "
            f"{strikes} strikes"
        )
        print(f"        -> {group_label(key)}")

        seeds = sorted(entry.get("seed", -1) for entry in entries)

        duplicates = len(seeds) - len(set(seeds))

        if duplicates:
            print(
                f"        warning: {duplicates} repeated seeds "
                f"in this group, so some runs are identical"
            )

    if not apply:
        print()
        print("nothing moved. run again with --apply to sort them.")
        return

    print()

    for key, entries in groups.items():
        label = group_label(key)

        target = os.path.join(directory, label)

        os.makedirs(target, exist_ok=True)

        renumbered = []

        for number, entry in enumerate(
            sorted(entries, key=lambda item: item["number"])
        ):
            source_file = os.path.join(directory, entry["file"])

            new_name = f"run_{number:03d}.npz"

            if os.path.exists(source_file):
                shutil.move(
                    source_file,
                    os.path.join(target, new_name)
                )
            else:
                print(f"  missing {entry['file']}, skipping")
                continue

            moved = dict(entry)
            moved["number"] = number
            moved["file"] = new_name

            renumbered.append(moved)

        with open(
            os.path.join(target, "index.json"), "w"
        ) as handle:
            json.dump(renumbered, handle, indent=1)

        print(f"  {label}: {len(renumbered)} runs")

    # The old index is no longer valid, and leaving it behind
    # would make the browser show the parent folder as a batch
    # containing files that have all moved.

    os.remove(index_path)

    print()
    print(f"removed the old {index_path}")
    print("browse with:  py run_browser.py " + directory)


def main():
    parser = argparse.ArgumentParser(
        description="Sort a mixed runs folder into subfolders."
    )

    parser.add_argument("directory", nargs="?", default="runs")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move the files; without this it only reports"
    )

    options = parser.parse_args()

    organise(options.directory, apply=options.apply)


if __name__ == "__main__":
    main()
