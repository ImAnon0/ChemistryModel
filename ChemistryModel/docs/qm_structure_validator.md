# QM structure validator v1

The validator answers one deliberately narrow question: does quantum chemistry
support the exact connected structure saved by ChemistryModel?

`molecule_library.py` remains the owner of the immutable recorded structure.
`qm_structure_validator.py` loads that structure, requires an explicit charge
and multiplicity, evaluates its unchanged Cartesian geometry, and starts a
Psi4 geometry optimisation from it. The default project convention is
`wb97x-d/jun-cc-pvdz`, C1 symmetry, with Psi4 reorientation and centre-of-mass
translation disabled.

Each attempt creates a separate append-only record under:

    molecules/qm_validations/SP_XXXXXX/QV_*.json
    molecules/qm_validations/SP_XXXXXX/QV_*.npz

The JSON contains status, electronic state, method, provenance, energies,
force summaries and structural comparison. The NPZ contains the original
recorded coordinates, original bonds, exact-geometry QM gradient/forces and,
when optimisation succeeds, the optimised coordinates. The original molecule
NPZ is never rewritten. Multiple methods or repeated attempts therefore remain
distinguishable and reproducible.

Status is `running`, `complete`, or `failed`; absence of a record means
untested. If optimisation fails after the exact-geometry calculation, that
single-point energy and gradient remain persisted in the failed record.

Molecule Lab runs the worker in a separate process so the interface remains
responsive. It displays the newest result and can toggle the existing 3D
viewer between the ChemistryModel and QM-optimised geometries. Charge and
multiplicity are intentionally blank until supplied by the user because the
current graph-only species identity does not safely establish them.

Command-line use:

    python qm_structure_validator.py SP_000001 --charge 0 --multiplicity 1

Run this from a Python environment containing Psi4. The deterministic unit
tests use injected runners and do not require Psi4. The explicit real-Psi4
smoke test is enabled with `CHEMISTRYMODEL_RUN_PSI4_TEST=1`.

Molecule Lab automatically looks for a conventional `chem-sapt` conda
environment. If Psi4 lives elsewhere, set `CHEMISTRYMODEL_PSI4_PYTHON` to that
environment's Python executable before starting Lab. The worker remains a
separate process; it does not load Psi4 into the simulation/UI process.
