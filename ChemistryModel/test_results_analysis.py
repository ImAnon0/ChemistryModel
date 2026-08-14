"""Sparse-output and Results-tab analysis invariants."""

import analysis


def test_optional_structure_fields_are_not_encoded_as_empty_placeholders():
    source = open(analysis.__file__, encoding="utf-8").read()
    assert '"amino_structure": None' not in source
    assert '"amino_structure": 0' not in source
    assert '"amino_structure": False' not in source
    assert '"structures": {}' not in source
    assert '"isomers": {}' not in source


if __name__ == "__main__":
    test_optional_structure_fields_are_not_encoded_as_empty_placeholders()
    print("PASS  test_optional_structure_fields_are_not_encoded_as_empty_placeholders")
