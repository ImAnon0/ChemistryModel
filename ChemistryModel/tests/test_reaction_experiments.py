"""Workflow invariants for Molecules-tab reaction experiments."""

import tempfile

import numpy as np

import characterisation_runner as C
import molecule_library as M
from chemistry_format import molecular_formula


def test_hill_formula_is_independent_of_atom_order():
    assert molecular_formula(["C", "C", "O", "H", "H", "H", "H", "H", "H"]) == "C2H6O"
    assert molecular_formula(["C", "N", "O", "O", "H", "H", "H"]) == "CH3NO2"
    assert molecular_formula(["H", "O", "H"]) == "H2O"


def test_atom_can_be_either_reactant_with_empty_library():
    with tempfile.TemporaryDirectory() as root:
        hydrogen = C.load_reactant("atom:H", root)
        oxygen = C.load_reactant("atom:O", root)
    assert hydrogen["symbols"] == ["H"]
    assert oxygen["symbols"] == ["O"]


def test_atom_atom_collision_uses_normal_collision_layout():
    h = C.atom_reactant("H")
    o = C.atom_reactant("O")
    symbols, positions, info = C.prepare_collision_box(
        h, o, 12.0, 7, 2.5, sampling_mode="random_orientation"
    )
    assert symbols == ["H", "O"]
    assert positions.shape == (2, 3)
    assert info["sampling_mode"] == "random_orientation"


def test_sampling_switch_is_seed_reproducible_and_preserves_reactants():
    a = C.atom_reactant("H")
    b = C.atom_reactant("O")
    one = C.prepare_collision_box(a, b, 12.0, 19, 2.5, sampling_mode="random_orientation")
    two = C.prepare_collision_box(a, b, 12.0, 19, 2.5, sampling_mode="random_orientation")
    targeted = C.prepare_collision_box(a, b, 12.0, 19, 2.5, sampling_mode="targeted", target_atom=0)
    assert np.array_equal(one[1], two[1])
    assert one[0] == targeted[0]
    assert targeted[2]["target_atom"] == 0


def test_atom_molecule_and_molecule_molecule_layouts_remain_supported():
    water = {
        "id": "water", "symbols": ["O", "H", "H"],
        "positions": np.array([[0, 0, 0], [.96, 0, 0], [-.24, .93, 0.0]]),
    }
    atom = C.atom_reactant("H")
    assert len(C.prepare_collision_box(water, atom, 15, 2, 2.5)[0]) == 4
    assert len(C.prepare_collision_box(water, water, 15, 2, 2.5)[0]) == 6


def test_product_component_extraction_uses_canonical_formula():
    class Recorded:
        has_atom_history = True
        positions = [np.zeros((3, 3))]
        def symbols_at(self, unused):
            return ["O", "H", "H"]
        def atom_ids_at(self, unused):
            return np.arange(3, dtype=np.uint32)
    records = M.component_records(Recorded(), 0, [0, 0], [1, 2])
    assert records[0]["formula"] == "H2O"
    assert records[0]["bonds"].shape == (2, 2)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS ", test.__name__)
        except Exception as problem:
            failures += 1
            print("FAIL ", test.__name__, problem)
    print(f"\n{len(tests)-failures} passed, {failures} failed")
    raise SystemExit(bool(failures))
