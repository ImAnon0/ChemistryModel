r"""Compute the version-1 electronic-observable dataset with Psi4.

Run in the project's separate quantum-chemistry environment, for example::

    C:\Users\Mikey\miniforge3\envs\chem-sapt\python.exe \
        research/electronic_observables/compute_observables.py

The calculation is resumable and rewrites JSON atomically after every
geometry.  A failure of an optional response calculation does not discard a
successful energy/gradient/density calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import psi4


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_AU_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
DIPOLE_AU_TO_DEBYE = 2.541746473
POLARIZABILITY_AU_TO_ANGSTROM3 = BOHR_TO_ANGSTROM ** 3

DEFAULT_INPUT = Path("research_data/electronic_observables/manifest.json")
DEFAULT_OUTPUT = Path("research_data/electronic_observables/observables.json")
DEFAULT_METADATA = Path("research_data/electronic_observables/observables.meta.json")
DEFAULT_PSI4_OUTPUT = Path("research_data/electronic_observables/psi4.out")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_list(value):
    if hasattr(value, "np"):
        value = value.np
    return np.asarray(value, dtype=float).squeeze().tolist()


def _finite(value):
    return bool(np.isfinite(np.asarray(value, dtype=float)).all())


def _molecule(row):
    lines = [f"{int(row['charge'])} {int(row['multiplicity'])}"]
    for symbol, xyz in zip(row["symbols"], row["coordinates_angstrom"]):
        x, y, z = map(float, xyz)
        lines.append(f"{symbol:2s} {x: .12f} {y: .12f} {z: .12f}")
    lines += ["units angstrom", "symmetry c1", "no_reorient", "no_com"]
    return psi4.geometry("\n".join(lines))


def _s2_from_orbitals(alpha, beta, overlap):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    n_alpha = alpha.shape[1]
    n_beta = beta.shape[1]
    spin_z = 0.5 * (n_alpha - n_beta)
    alpha_beta_overlap = alpha.T @ overlap @ beta
    return float(
        spin_z * (spin_z + 1.0)
        + n_beta
        - np.square(alpha_beta_overlap).sum()
    )


def _extract_s2(wfn):
    for key, value in wfn.scalar_variables().items():
        normalized = str(key).upper().replace(" ", "")
        if "S^2" in normalized or "S**2" in normalized:
            return float(value)
    # Psi4 1.11 prints <S^2> for UKS but does not expose it as a scalar
    # variable.  Evaluate the standard UHF expression from occupied alpha/beta
    # orbitals in the non-orthogonal AO basis.
    try:
        alpha = np.asarray(wfn.Ca_subset("AO", "OCC").np, dtype=float)
        beta = np.asarray(wfn.Cb_subset("AO", "OCC").np, dtype=float)
        overlap = np.asarray(wfn.S().np, dtype=float)
        return _s2_from_orbitals(alpha, beta, overlap)
    except Exception:
        return None


def _array_variable(wfn, key):
    variables = wfn.array_variables()
    if key not in variables:
        raise KeyError(f"Psi4 did not expose {key!r}")
    return np.asarray(
        variables[key].np if hasattr(variables[key], "np") else variables[key],
        dtype=float,
    ).squeeze()


def _density_observables(wfn, coordinates_angstrom):
    psi4.oeprop(
        wfn,
        "DIPOLE",
        "MULLIKEN_CHARGES",
        "LOWDIN_CHARGES",
        "MBIS_CHARGES",
        "MBIS_DIPOLES",
        "MBIS_QUADRUPOLES",
    )
    dipole = _array_variable(wfn, "CURRENT DIPOLE").reshape(3)
    mulliken = _array_variable(wfn, "MULLIKEN CHARGES").reshape(-1)
    lowdin = _array_variable(wfn, "LOWDIN CHARGES").reshape(-1)
    mbis = _array_variable(wfn, "MBIS CHARGES").reshape(-1)
    mbis_dipoles = _array_variable(wfn, "MBIS DIPOLES").reshape((-1, 3))
    mbis_quadrupoles = _array_variable(wfn, "MBIS QUADRUPOLES")

    coordinates_bohr = np.asarray(coordinates_angstrom) / BOHR_TO_ANGSTROM
    reconstructed = (mbis[:, None] * coordinates_bohr).sum(axis=0)
    reconstructed += mbis_dipoles.sum(axis=0)

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    centre = coordinates.mean(axis=0)
    directions = np.asarray([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        *[[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    shell_radius = float(np.max(np.linalg.norm(coordinates - centre, axis=1)) + 2.0)
    esp_points_angstrom = centre + shell_radius * directions
    esp = psi4.core.ESPPropCalc(wfn).compute_esp_over_grid_in_memory(
        psi4.core.Matrix.from_array(esp_points_angstrom / BOHR_TO_ANGSTROM)
    )
    esp = np.asarray(esp.np if hasattr(esp, "np") else esp, dtype=float).reshape(-1)

    return {
        "dipole_au": dipole.tolist(),
        "dipole_debye": (dipole * DIPOLE_AU_TO_DEBYE).tolist(),
        "dipole_magnitude_debye": float(np.linalg.norm(dipole) * DIPOLE_AU_TO_DEBYE),
        "mulliken_charges_e": mulliken.tolist(),
        "lowdin_charges_e": lowdin.tolist(),
        "mbis_charges_e": mbis.tolist(),
        "mbis_atomic_dipoles_au": mbis_dipoles.tolist(),
        "mbis_atomic_quadrupoles_au": mbis_quadrupoles.tolist(),
        "mbis_reconstructed_dipole_au": reconstructed.tolist(),
        "mbis_dipole_reconstruction_error_au": float(np.linalg.norm(reconstructed - dipole)),
        "external_potential": {
            "points_angstrom": esp_points_angstrom.tolist(),
            "potential_au": esp.tolist(),
            "shell_radius_angstrom": shell_radius,
            "convention": "Psi4 total molecular ESP on a 14-point enclosing shell",
        },
    }


def _polarizability(method, basis, molecule):
    try:
        psi4.core.clean_variables()
    except Exception:
        pass
    psi4.properties(
        f"{method}/{basis}",
        molecule=molecule,
        properties=["DIPOLE_POLARIZABILITIES"],
    )
    axes = "XYZ"
    tensor = np.asarray([
        [float(psi4.core.variable(f"DIPOLE POLARIZABILITY {a}{b}")) for b in axes]
        for a in axes
    ])
    symmetric = 0.5 * (tensor + tensor.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return {
        "tensor_au": tensor.tolist(),
        "tensor_angstrom3": (tensor * POLARIZABILITY_AU_TO_ANGSTROM3).tolist(),
        "isotropic_au": float(np.trace(tensor) / 3.0),
        "isotropic_angstrom3": float(
            np.trace(tensor) / 3.0 * POLARIZABILITY_AU_TO_ANGSTROM3
        ),
        "eigenvalues_au": eigenvalues.tolist(),
        "antisymmetric_norm_au": float(np.linalg.norm(tensor - tensor.T)),
    }


def evaluate(row, method, basis):
    started = time.perf_counter()
    result = {
        "geometry_id": row["geometry_id"],
        "status": "failed",
        "core_status": "failed",
        "polarizability_status": "not_attempted",
        "error": "",
        "polarizability_error": "",
    }
    try:
        molecule = _molecule(row)
        reference = "rks" if int(row["multiplicity"]) == 1 else "uks"
        psi4.set_options({
            "reference": reference,
            "scf_type": "df",
            "e_convergence": 1.0e-9,
            "d_convergence": 1.0e-9,
            "maxiter": 200,
        })
        gradient, wfn = psi4.gradient(
            f"{method}/{basis}", molecule=molecule, return_wfn=True
        )
        gradient = np.asarray(gradient.np, dtype=float)
        energy = float(wfn.energy())
        density = _density_observables(wfn, row["coordinates_angstrom"])
        if (
            not _finite(energy)
            or not _finite(gradient)
            or not _finite(density["dipole_au"])
        ):
            raise RuntimeError("non-finite core observable")
        result.update({
            "core_status": "ok",
            "energy_hartree": energy,
            "energy_eV": energy * HARTREE_TO_EV,
            "gradient_hartree_per_bohr": gradient.tolist(),
            "force_eV_per_angstrom": (-gradient * FORCE_AU_TO_EV_PER_ANGSTROM).tolist(),
            "s2": _extract_s2(wfn),
            **density,
        })

        polar_started = time.perf_counter()
        try:
            polar = _polarizability(method, basis, molecule)
            if not _finite(polar["tensor_au"]):
                raise RuntimeError("non-finite polarizability")
            result["polarizability"] = polar
            result["polarizability_status"] = "ok"
        except Exception as exc:
            result["polarizability_status"] = "failed"
            result["polarizability_error"] = f"{type(exc).__name__}: {exc}"
        result["polarizability_wall_seconds"] = time.perf_counter() - polar_started
        result["status"] = (
            "ok" if result["polarizability_status"] == "ok" else "partial"
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        try:
            psi4.core.clean()
        except Exception:
            pass
    return result


def _write_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--psi4-output", type=Path, default=DEFAULT_PSI4_OUTPUT)
    parser.add_argument("--method", default="wb97x-d")
    parser.add_argument("--basis", default="jun-cc-pvdz")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="4 GB")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--geometry-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.input.read_text(encoding="utf-8"))
    rows = manifest["geometries"]
    if args.geometry_id:
        selected = set(args.geometry_id)
        rows = [row for row in rows if row["geometry_id"] in selected]
        missing = selected - {row["geometry_id"] for row in rows}
        if missing:
            raise SystemExit(f"unknown --geometry-id: {sorted(missing)}")
    if args.limit:
        rows = rows[: args.limit]

    stored = {"schema_version": 1, "records": []}
    if args.output.exists():
        stored = json.loads(args.output.read_text(encoding="utf-8"))
    records = {row["geometry_id"]: row for row in stored.get("records", [])}

    args.psi4_output.parent.mkdir(parents=True, exist_ok=True)
    psi4.set_output_file(str(args.psi4_output.resolve()), False)
    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)

    for index, row in enumerate(rows, 1):
        old = records.get(row["geometry_id"])
        if not args.force and old and old.get("status") == "ok":
            print(f"[{index:02d}/{len(rows):02d}] {row['geometry_id']:<38s} SKIP")
            continue
        result = evaluate(row, args.method, args.basis)
        records[row["geometry_id"]] = result
        stored = {
            "schema_version": 1,
            "input_sha256": _sha256(args.input),
            "method": args.method,
            "basis": args.basis,
            "records": [
                records[key] for key in [item["geometry_id"] for item in manifest["geometries"]]
                if key in records
            ],
        }
        _write_atomic(args.output, stored)
        print(
            f"[{index:02d}/{len(rows):02d}] {row['geometry_id']:<38s} "
            f"{result['status']:<7s} {result['wall_seconds']:6.2f}s"
        )

    metadata = {
        "schema_version": 1,
        "manifest": str(args.input),
        "manifest_sha256": _sha256(args.input),
        "method": args.method,
        "basis": args.basis,
        "references": {"singlet": "RKS", "open_shell": "UKS"},
        "scf_type": "DF",
        "threads": args.threads,
        "memory": args.memory,
        "psi4_version": getattr(psi4, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "constants": {
            "hartree_to_eV": HARTREE_TO_EV,
            "bohr_to_angstrom": BOHR_TO_ANGSTROM,
            "dipole_au_to_debye": DIPOLE_AU_TO_DEBYE,
            "polarizability_au_to_angstrom3": POLARIZABILITY_AU_TO_ANGSTROM3,
        },
        "property_conventions": {
            "charges": "Mulliken, Lowdin, and MBIS are partition-dependent proxies.",
            "dipole": "Psi4 total electric dipole in the fixed input frame.",
            "polarizability": "Static analytic dipole polarizability in atomic units.",
            "mbis_multipoles": "Raw Psi4 component ordering and atomic-unit convention retained.",
            "external_potential": "Psi4 total molecular ESP in atomic units on a fixed enclosing 14-point shell.",
        },
    }
    _write_atomic(args.metadata, metadata)
    print(f"results: {args.output}")
    print(f"metadata: {args.metadata}")


if __name__ == "__main__":
    main()
