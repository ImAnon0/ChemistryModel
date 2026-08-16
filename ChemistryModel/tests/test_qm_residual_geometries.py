"""Tests for generate_qm_residual_geometries.py."""

import math

import generate_qm_residual_geometries as G


def test_dataset_shape_and_split():
    payload = G.build_payload(G.DEFAULT_SEED)
    rows = payload["geometries"]

    # Per system: 2 references + 64 grid + 24 jitter = 90.
    assert len(rows) == 4 * 90

    methane = [row for row in rows if row["system"] == "methane"]
    non_methane = [row for row in rows if row["system"] != "methane"]

    assert methane
    assert all(row["split"] == "holdout" for row in methane)
    assert all(row["split"] == "train" for row in non_methane)


def test_every_system_has_one_reactant_reference():
    rows = G.build_payload(G.DEFAULT_SEED)["geometries"]
    for system_name in G.SYSTEMS:
        refs = [
            row for row in rows
            if row["system"] == system_name
            and row["sample_kind"] == "reactant_reference"
        ]
        assert len(refs) == 1


def test_geometry_ids_are_unique():
    rows = G.build_payload(G.DEFAULT_SEED)["geometries"]
    ids = [row["geometry_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_coordinates_are_finite_and_match_symbols():
    rows = G.build_payload(G.DEFAULT_SEED)["geometries"]
    for row in rows:
        assert len(row["symbols"]) == len(row["coordinates_angstrom"])
        for xyz in row["coordinates_angstrom"]:
            assert len(xyz) == 3
            assert all(math.isfinite(value) for value in xyz)


def test_generation_is_reproducible():
    assert G.build_payload(12345) == G.build_payload(12345)


def test_methane_has_six_atoms():
    rows = G.build_payload(G.DEFAULT_SEED)["geometries"]
    methane = next(row for row in rows if row["system"] == "methane")
    assert methane["symbols"] == ["C", "H", "H", "H", "H", "H"]


def test_all_systems_are_neutral_doublets():
    payload = G.build_payload(G.DEFAULT_SEED)
    for metadata in payload["systems"].values():
        assert metadata["charge"] == 0
        assert metadata["multiplicity"] == 2
