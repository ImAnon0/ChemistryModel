import argparse
import glob
import json
import os

import numpy as np

from recorder import Recorder

import analysis


# ============================================================
# Rebuilding an index
# ============================================================
#
# index.json is a cache. It is written once when a run finishes
# and never revisited, so it goes stale in two ways:
#
#   1. Files move or get deleted, and the index still lists them.
#   2. The analysis code improves, and the stored summaries were
#      produced by the old version.
#
# Both are fixed by throwing the index away and rebuilding it
# from the recordings, which are the actual data. Metadata that
# only exists in the index - the mixture name, the seed, how many
# strikes - is carried across where the old entry still matches.
#
#   py reindex.py runs/my_batch
#   py reindex.py runs --all        every batch under runs/


def rebuild(directory, stride=4):
    files = sorted(glob.glob(os.path.join(directory, "run_*.npz")))

    if not files:
        return None

    index_path = os.path.join(directory, "index.json")

    previous = {}

    if os.path.exists(index_path):
        try:
            with open(index_path) as handle:
                for entry in json.load(handle):
                    previous[entry.get("file")] = entry
        except (json.JSONDecodeError, OSError):
            print(f"  could not read the old index, starting fresh")

    rebuilt = []

    for number, path in enumerate(files):
        name = os.path.basename(path)

        recorder = Recorder.load(path)
        result = analysis.analyse(recorder, stride=stride)

        old = previous.get(name, {})

        entry = {
            "number": number,
            "file": name,
            # Carried over from the old entry when it exists,
            # since none of this is stored in the recording.
            "mixture": old.get("mixture", "unknown"),
            "seed": old.get("seed", -1),
            "strikes": old.get("strikes", 0),
            "wall_seconds": old.get("wall_seconds", 0.0),
            # These come straight from the recording, so they are
            # always right even for an entry that never existed.
            "box": float(recorder.box_size),
            "atoms": len(recorder.symbols),
            "picoseconds": round(
                (recorder.times[-1] - recorder.times[0]) / 1000.0, 3
            ),
            "frames": len(recorder),
            "headline": analysis.headline(result),
            "final_species": sorted({
                item["formula"] for item in result["final"]
                if item["heavy"] >= 2
            }),
            "closed_shell": sorted({
                item["formula"] for item in result["final"]
                if item["heavy"] >= 2 and item["closed_shell"]
            }),
            "species_seen": sorted(result["seen"]),
            "heavy_bonds_formed": sum(
                1 for event in result["heavy_events"]
                if event[1] == "formed"
            ),
            "final_temperature": result["temperature"]["final"],
            "final_potential": result["potential"]["final"],
        }

        rebuilt.append(entry)

    with open(index_path, "w") as handle:
        json.dump(rebuilt, handle, indent=1)

    missing = set(previous) - {entry["file"] for entry in rebuilt}

    return rebuilt, missing


def report(directory, rebuilt, missing):
    print(f"{directory}")
    print(f"  {len(rebuilt)} recordings re-analysed")

    if missing:
        print(
            f"  dropped {len(missing)} stale entries whose files "
            f"are gone: {', '.join(sorted(missing)[:6])}"
            + (" ..." if len(missing) > 6 else "")
        )

    unknown = sum(
        1 for entry in rebuilt if entry["mixture"] == "unknown"
    )

    if unknown:
        print(
            f"  {unknown} runs had no previous entry, so their "
            f"mixture and seed are unknown"
        )

    tally = {}

    for entry in rebuilt:
        for name in entry["closed_shell"]:
            tally[name] = tally.get(name, 0) + 1

    if tally:
        top = sorted(tally.items(), key=lambda item: -item[1])[:8]

        print(
            "  closed shell: "
            + "   ".join(
                f"{name} {number}/{len(rebuilt)}"
                for name, number in top
            )
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild run indexes from the recordings."
    )

    parser.add_argument("directory", nargs="?", default="runs")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also rebuild every subfolder containing recordings"
    )
    parser.add_argument("--stride", type=int, default=4)

    options = parser.parse_args()

    targets = [options.directory]

    if options.all and os.path.isdir(options.directory):
        for name in sorted(os.listdir(options.directory)):
            path = os.path.join(options.directory, name)

            if os.path.isdir(path):
                targets.append(path)

    done = 0

    for target in targets:
        outcome = rebuild(target, stride=options.stride)

        if outcome is None:
            index_path = os.path.join(target, "index.json")

            # A folder with an index but no recordings is left
            # over from sorting. The index points at nothing.

            if os.path.exists(index_path):
                os.remove(index_path)

                print(f"{target}")
                print(
                    "  no recordings here, removed the orphaned "
                    "index.json"
                )
                print()

            continue

        rebuilt, missing = outcome

        report(target, rebuilt, missing)

        done += 1

    if done == 0:
        print("no recordings found")


if __name__ == "__main__":
    main()
