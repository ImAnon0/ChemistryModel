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

STARTS = {
    "loose H + O": ("atoms", {"H": 40, "O": 20}),
    "loose C H N O": ("atoms", {"C": 8, "H": 44, "N": 6, "O": 8}),
    "Miller-Urey": (
        "molecules", {"CH4": 6, "NH3": 4, "H2O": 6, "H2": 8}
    ),
    "water box": ("molecules", {"H2O": 24}),
    "methane box": ("molecules", {"CH4": 12}),
    "H rich loose": ("atoms", {"C": 8, "H": 60, "N": 6, "O": 8}),
    "H rich x5": ("atoms", {"C": 40, "H": 300, "N": 30, "O": 40}),
}
