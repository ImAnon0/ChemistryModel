"""Build the frozen, molecule-family-blocked C0 parameter-pilot manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "research_data/electronic_observables/manifest.json"
OUTPUT = ROOT / "research_data/electronic_observables/c0_pilot/manifest.json"


SPLITS = {
    "training": {
        "h2", "methane", "h_formaldehyde_transfer", "water_angle",
        "oh_stretch", "ammonia", "carbon_monoxide", "methanol", "n2",
    },
    "validation": {
        "water_transfer", "h3_transfer", "ethane", "hydrogen_cyanide",
        "hydrogen_peroxide",
    },
    "locked_holdout": {
        "formaldehyde", "h_h2_approach", "water_h_direction",
        "methylamine", "carbon_dioxide", "hydroxylamine",
    },
}


BASES = {
    "ammonia": (["N", "H", "H", "H"], [
        [0.0, 0.0, 0.116], [0.0, 0.938, -0.273],
        [0.812, -0.469, -0.273], [-0.812, -0.469, -0.273],
    ]),
    "carbon_monoxide": (["C", "O"], [[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]]),
    "methanol": (["C", "O", "H", "H", "H", "H"], [
        [0.0, 0.0, 0.0], [1.427, 0.0, 0.0], [-0.363, 1.027, 0.0],
        [-0.363, -0.514, 0.890], [-0.363, -0.514, -0.890],
        [1.785, 0.891, 0.0],
    ]),
    "hydrogen_cyanide": (
        ["H", "C", "N"], [[-1.066, 0.0, 0.0], [0.0, 0.0, 0.0], [1.153, 0.0, 0.0]]
    ),
    "hydrogen_peroxide": (["O", "O", "H", "H"], [
        [-0.727, 0.0, 0.0], [0.727, 0.0, 0.0],
        [-1.020, 0.812, 0.433], [1.020, -0.812, 0.433],
    ]),
    "methylamine": (["C", "N", "H", "H", "H", "H", "H"], [
        [0.0, 0.0, 0.0], [1.471, 0.0, 0.0], [-0.363, 1.027, 0.0],
        [-0.363, -0.514, 0.890], [-0.363, -0.514, -0.890],
        [1.827, 0.465, 0.823], [1.827, 0.465, -0.823],
    ]),
    "carbon_dioxide": (
        ["O", "C", "O"], [[-1.160, 0.0, 0.0], [0.0, 0.0, 0.0], [1.160, 0.0, 0.0]]
    ),
    "hydroxylamine": (["N", "O", "H", "H", "H"], [
        [0.0, 0.0, 0.0], [1.450, 0.0, 0.0],
        [-0.450, 0.805, 0.390], [-0.450, -0.805, 0.390],
        [1.805, 0.820, -0.330],
    ]),
}


def _scale(coords, factor):
    anchor = coords[0]
    return [
        [anchor[k] + factor * (point[k] - anchor[k]) for k in range(3)]
        for point in coords
    ]


def _new_rows():
    rows = []
    for family, (symbols, coordinates) in BASES.items():
        split = next(name for name, families in SPLITS.items() if family in families)
        factors = np.linspace(0.90, 1.10, 9)
        for factor in factors:
            factor = float(factor)
            rows.append({
                "geometry_id": f"c0_{family}_{factor:.2f}".replace(".", "p"),
                "family": family,
                "role": split,
                "split": split,
                "comparison_group": family,
                "reference_geometry_id": f"c0_{family}_1p00",
                "purpose": "C0 10% parameter-pilot molecular response.",
                "charge": 0,
                "multiplicity": 1,
                "symbols": symbols,
                "coordinates_angstrom": _scale(coordinates, factor),
                "scan_coordinate": {"uniform_internal_scale": factor},
            })
        # Deterministic off-symmetry distortions provide response information
        # beyond a redundant uniform bond scan.  They are deliberately kept in
        # the same family split as their parent molecule.
        target_count = 30 if family == "ammonia" else 25
        rng = np.random.default_rng(sum((i + 1) * ord(c) for i, c in enumerate(family)))
        base = np.asarray(coordinates, dtype=float)
        for index in range(target_count - len(factors)):
            amplitude = 0.015 + 0.045 * (index % 6) / 5.0
            displacement = rng.normal(size=base.shape)
            displacement -= displacement.mean(axis=0, keepdims=True)
            norm = np.sqrt(np.mean(displacement * displacement))
            distorted = base + amplitude * displacement / norm
            rows.append({
                "geometry_id": f"c0_{family}_distort_{index:02d}",
                "family": family,
                "role": split,
                "split": split,
                "comparison_group": family,
                "reference_geometry_id": f"c0_{family}_1p00",
                "purpose": "C0 10% parameter-pilot deterministic thermal-like distortion.",
                "charge": 0,
                "multiplicity": 1,
                "symbols": symbols,
                "coordinates_angstrom": distorted.round(10).tolist(),
                "scan_coordinate": {"distortion_index": index, "rms_amplitude_A": amplitude},
            })
    return rows


def build():
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for original in source["geometries"]:
        row = dict(original)
        family = row["family"]
        split = next(name for name, families in SPLITS.items() if family in families)
        row["role"] = split
        row["split"] = split
        rows.append(row)
    rows.extend(_new_rows())
    assert len(rows) == 235, len(rows)
    family_to_split = {
        family: split for split, families in SPLITS.items() for family in families
    }
    assert {row["family"] for row in rows} == set(family_to_split)
    assert all(row["split"] == family_to_split[row["family"]] for row in rows)
    payload = {
        "schema_version": 1,
        "purpose": "C0 continuous-SQE 10% screening pilot; family-blocked",
        "qm_convention": "wB97X-D/jun-cc-pVDZ, matching the audited seed observable set",
        "fitting_policy": "Dipoles and response; MBIS charges are low-weight proxies only.",
        "splits": {name: sorted(families) for name, families in SPLITS.items()},
        "geometries": rows,
    }
    holdout = [row for row in rows if row["split"] == "locked_holdout"]
    canonical = json.dumps(holdout, sort_keys=True, separators=(",", ":")).encode()
    payload["locked_holdout_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main():
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = {
        split: sum(row["split"] == split for row in payload["geometries"])
        for split in SPLITS
    }
    print(json.dumps({"output": str(OUTPUT), "counts": counts,
                      "locked_holdout_sha256": payload["locked_holdout_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
