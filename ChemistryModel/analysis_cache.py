import hashlib
import json
import os


# ============================================================
# Remembering what has already been worked out
# ============================================================
#
# Analysis is deterministic: the same recording, the same
# settings and the same code always give the same answer. So
# there is no reason to do it twice, and browsing is repetitive
# by nature - the whole point is clicking back and forth between
# runs.
#
# Results are written next to the recording as a small JSON file.
# The cache is keyed on the recording's size and modification
# time plus the analysis settings, so replacing a recording or
# changing how it is analysed produces a different key and the
# old answer is simply never looked up again.
#
# Anything that cannot survive a round trip through JSON is
# dropped rather than stored badly: the per-molecule member
# lists, for instance, which the browser recomputes anyway.


CACHE_VERSION = 4

FOLDER = ".analysis_cache"


def key_for(path, settings):
    stat = os.stat(path)

    parts = [
        os.path.basename(path),
        str(stat.st_size),
        str(int(stat.st_mtime)),
        str(CACHE_VERSION),
    ]

    for name in sorted(settings):
        parts.append(f"{name}={settings[name]}")

    return hashlib.blake2s(
        "|".join(parts).encode(), digest_size=10
    ).hexdigest()


def cache_path(path, settings):
    directory = os.path.join(os.path.dirname(path), FOLDER)

    return os.path.join(directory, key_for(path, settings) + ".json")


def load(path, settings):
    target = cache_path(path, settings)

    if not os.path.exists(target):
        return None

    try:
        with open(target) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def save(path, settings, result):
    target = cache_path(path, settings)

    os.makedirs(os.path.dirname(target), exist_ok=True)

    try:
        with open(target, "w") as handle:
            json.dump(make_storable(result), handle)
    except (OSError, TypeError):
        # A cache that cannot be written is an inconvenience,
        # not a failure. The answer is still correct.
        pass

    return result


def make_storable(result):
    # Tuples become lists, numpy scalars become plain numbers, and
    # anything that still will not serialise is left out.

    def convert(value):
        if isinstance(value, dict):
            return {
                str(name): convert(item)
                for name, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]

        if hasattr(value, "item") and not isinstance(
            value, (str, bytes)
        ):
            try:
                return value.item()
            except (ValueError, AttributeError):
                return None

        if isinstance(value, (int, float, str, bool)) or value is None:
            return value

        return None

    return convert(result)


def analyse_cached(recorder, path, analyse, **settings):
    # Looks the result up, and works it out only if it is not
    # already known.

    stored = load(path, settings)

    if stored is not None:
        stored["from_cache"] = True

        return stored

    result = analyse(recorder, **settings)

    result["from_cache"] = False

    save(path, settings, result)

    return result


def clear(directory):
    # Removes every cached answer under a folder, for when the
    # analysis code has changed in a way the version number did
    # not capture.

    removed = 0

    for root, folders, files in os.walk(directory):
        if os.path.basename(root) != FOLDER:
            continue

        for name in files:
            os.remove(os.path.join(root, name))
            removed += 1

    return removed
