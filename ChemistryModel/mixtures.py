# ============================================================
# Starting mixtures
# ============================================================
#
# Kept in their own module so the headless batch runner does not
# have to import the viewer, and therefore does not need Qt or
# OpenGL installed to run.
#
#   "atoms"     scatters loose atoms of the given elements
#   "molecules" builds real geometry from build_box.BUILDERS
#
# Loose atoms bond immediately and heat themselves, so they need
# no spark. Molecular mixtures are stable until something breaks
# them open, which is what the lightning channel is for.

import json
import os


# Anything defined in mixtures.json is merged in on top of the
# built-in list. The control panel writes that file rather than
# editing this one, so a mixture invented in the interface is
# visible to the batch runner and the viewer without any of them
# having to edit Python.

CUSTOM_FILE = "mixtures.json"


BUILT_IN = {
    "loose H + O": ("atoms", {"H": 40, "O": 20}),
    "loose C H N O": ("atoms", {"C": 8, "H": 44, "N": 6, "O": 8}),
    "Miller-Urey": (
        "molecules", {"CH4": 6, "NH3": 4, "H2O": 6, "H2": 8}
    ),
    "water box": ("molecules", {"H2O": 24}),
    "methane box": ("molecules", {"CH4": 12}),
    "H rich loose": ("atoms", {"C": 8, "H": 60, "N": 6, "O": 8}),
    "H rich x5": ("atoms", {"C": 40, "H": 200, "N": 20, "O": 30}),
    "carbon rich": ("atoms", {"C": 80, "H": 200, "N": 20, "O": 30}),
}


def load_custom(path=CUSTOM_FILE):
    if not os.path.exists(path):
        return {}

    try:
        with open(path) as handle:
            stored = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

    mixtures = {}

    for name, entry in stored.items():
        kind = entry.get("kind", "atoms")
        contents = entry.get("contents", {})

        if kind in ("atoms", "molecules") and contents:
            mixtures[name] = (kind, contents)

    return mixtures


def save_custom(mixtures, path=CUSTOM_FILE):
    stored = {
        name: {"kind": kind, "contents": contents}
        for name, (kind, contents) in mixtures.items()
    }

    with open(path, "w") as handle:
        json.dump(stored, handle, indent=1)


def all_mixtures():
    combined = dict(BUILT_IN)
    combined.update(load_custom())

    return combined


STARTS = all_mixtures()