"""Persistent, molecule-library-linked quantum structure validation.

The validator is deliberately independent of the Lab UI.  It preserves the
recorded ChemistryModel geometry, evaluates that exact geometry, starts a QM
optimisation from it, and stores the two structures and their comparison as a
new validation record.  Psi4 is imported lazily so ordinary Lab use and unit
tests do not require a Psi4 installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import molecule_library


DEFAULT_METHOD = "wb97x-d"
DEFAULT_BASIS = "jun-cc-pvdz"
DEFAULT_REFERENCE = "uhf"
DEFAULT_ROOT = Path(molecule_library.DEFAULT_ROOT) / "qm_validations"
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

COVALENT_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}


def psi4_worker_python():
    """Find the configured/project Psi4 interpreter, falling back explicitly."""
    configured = os.environ.get("CHEMISTRYMODEL_PSI4_PYTHON")
    if configured:
        configured = os.path.abspath(os.path.expanduser(configured))
        if not os.path.isfile(configured):
            raise ValueError(
                f"CHEMISTRYMODEL_PSI4_PYTHON does not exist: {configured}"
            )
        return configured
    user = os.path.expanduser("~")
    candidates = [
        os.path.join(user, distribution, "envs", "chem-sapt", "python.exe")
        for distribution in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:
        return None


def geometry_sha256(symbols, coordinates):
    digest = hashlib.sha256()
    digest.update("\0".join(map(str, symbols)).encode("utf-8"))
    digest.update(np.asarray(coordinates, dtype="<f8").tobytes())
    return digest.hexdigest()


def build_psi4_geometry(symbols, coordinates, charge, multiplicity):
    """Return a Psi4 molecule specification without altering coordinates."""
    if charge is None or multiplicity is None:
        raise ValueError("charge and multiplicity are required; they are not inferred")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("symbols and coordinates must describe the same N x 3 geometry")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("geometry contains a non-finite coordinate")
    if int(multiplicity) < 1:
        raise ValueError("multiplicity must be at least one")
    lines = [f"{int(charge)} {int(multiplicity)}"]
    for symbol, (x, y, z) in zip(symbols, coordinates):
        lines.append(f"{str(symbol):2s} {x: .12f} {y: .12f} {z: .12f}")
    lines += ["units angstrom", "symmetry c1", "no_reorient", "no_com"]
    return "\n".join(lines)


def inferred_bonds(symbols, coordinates, scale=1.25):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    bonds = []
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            radius = COVALENT_RADIUS.get(str(symbols[first]))
            other = COVALENT_RADIUS.get(str(symbols[second]))
            if radius is None or other is None:
                continue
            distance = np.linalg.norm(coordinates[first] - coordinates[second])
            if distance <= scale * (radius + other):
                bonds.append((first, second))
    return np.asarray(bonds, dtype=np.int32).reshape(-1, 2)


def _kabsch(moving, target):
    moving = np.asarray(moving, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return (moving - moving_center) @ rotation + target_center


def _assignment(cost):
    """Minimum assignment, using SciPy when available and exact DP otherwise."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, columns = linear_sum_assignment(cost)
        return columns[np.argsort(rows)]
    except ImportError:
        pass
    cost = np.asarray(cost, dtype=np.float64)
    count = len(cost)
    if count > 16:
        # Large all-equivalent groups are unusual here. Repeated nearest
        # assignment remains deterministic but is explicitly a fallback.
        available = set(range(count))
        result = []
        for row in range(count):
            chosen = min(available, key=lambda column: cost[row, column])
            result.append(chosen)
            available.remove(chosen)
        return np.asarray(result, dtype=int)
    states = {0: (0.0, ())}
    for row in range(count):
        following = {}
        for mask, (total, chosen) in states.items():
            for column in range(count):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (total + cost[row, column], chosen + (column,))
                old = following.get(mask | bit)
                if old is None or candidate[0] < old[0]:
                    following[mask | bit] = candidate
        states = following
    return np.asarray(states[(1 << count) - 1][1], dtype=int)


def align_equivalent_atoms(symbols, original, optimised, iterations=8):
    """Align and permute equivalent elements; return aligned coordinates/map."""
    original = np.asarray(original, dtype=np.float64)
    optimised = np.asarray(optimised, dtype=np.float64)
    if original.shape != optimised.shape or original.shape != (len(symbols), 3):
        raise ValueError("the two geometries must have matching N x 3 coordinates")
    mapping = np.arange(len(symbols), dtype=int)
    aligned = _kabsch(optimised, original)
    for _ in range(iterations):
        previous = mapping.copy()
        for symbol in sorted(set(map(str, symbols))):
            slots = np.asarray([i for i, value in enumerate(symbols) if str(value) == symbol])
            cost = np.linalg.norm(
                original[slots, None, :] - aligned[None, slots, :], axis=-1
            )
            mapping[slots] = slots[_assignment(cost)]
        aligned = _kabsch(optimised[mapping], original)
        if np.array_equal(previous, mapping):
            break
    return aligned, mapping


def _angles_from_bonds(bonds, count):
    neighbours = [[] for _ in range(count)]
    for first, second in np.asarray(bonds, dtype=int).reshape(-1, 2):
        neighbours[first].append(second)
        neighbours[second].append(first)
    return [
        (first, centre, second)
        for centre, linked in enumerate(neighbours)
        for offset, first in enumerate(linked)
        for second in linked[offset + 1:]
    ]


def _angle(coordinates, triple):
    first, centre, second = triple
    a = coordinates[first] - coordinates[centre]
    b = coordinates[second] - coordinates[centre]
    cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def compare_structures(symbols, original, original_bonds, optimised):
    aligned, mapping = align_equivalent_atoms(symbols, original, optimised)
    original = np.asarray(original, dtype=np.float64)
    original_bonds = np.asarray(original_bonds, dtype=int).reshape(-1, 2)
    differences = aligned - original
    atom_rmsd = float(np.sqrt(np.mean(np.sum(differences ** 2, axis=1))))
    heavy = np.asarray([str(symbol) != "H" for symbol in symbols])
    heavy_rmsd = (
        float(np.sqrt(np.mean(np.sum(differences[heavy] ** 2, axis=1))))
        if np.any(heavy) else None
    )
    original_edges = {tuple(sorted(map(int, edge))) for edge in original_bonds}
    optimised_edges_raw = inferred_bonds(symbols, optimised)
    inverse = np.empty(len(mapping), dtype=int)
    inverse[mapping] = np.arange(len(mapping))
    optimised_edges = {
        tuple(sorted((int(inverse[a]), int(inverse[b]))))
        for a, b in optimised_edges_raw
    }
    bond_errors = []
    for first, second in original_edges & optimised_edges:
        before = np.linalg.norm(original[first] - original[second])
        after = np.linalg.norm(aligned[first] - aligned[second])
        bond_errors.append(after - before)
    angle_errors = [
        _angle(aligned, triple) - _angle(original, triple)
        for triple in _angles_from_bonds(original_bonds, len(symbols))
    ]
    components = _component_count(len(symbols), optimised_edges)
    return {
        "atom_mapping_optimised_index_by_original": mapping.tolist(),
        "all_atom_rmsd_A": atom_rmsd,
        "heavy_atom_rmsd_A": heavy_rmsd,
        "connectivity_preserved": original_edges == optimised_edges,
        "connectivity_changed": original_edges != optimised_edges,
        "fragmented": components > _component_count(len(symbols), original_edges),
        "rearranged": original_edges != optimised_edges and components == _component_count(len(symbols), original_edges),
        "original_bond_count": len(original_edges),
        "optimised_bond_count": len(optimised_edges),
        "bond_length_rms_error_A": float(np.sqrt(np.mean(np.square(bond_errors)))) if bond_errors else None,
        "angle_rms_error_deg": float(np.sqrt(np.mean(np.square(angle_errors)))) if angle_errors else None,
        "angle_max_abs_error_deg": float(np.max(np.abs(angle_errors))) if angle_errors else None,
    }


def _component_count(count, edges):
    parents = list(range(count))
    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for first, second in edges:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root
    return len({find(index) for index in range(count)})


class Psi4Runner:
    def __init__(self, threads=8, memory="4 GB"):
        self.threads = int(threads)
        self.memory = str(memory)

    def run(self, symbols, coordinates, charge, multiplicity, method, basis, reference):
        try:
            import psi4
        except ImportError as exc:
            raise RuntimeError(
                "Psi4 is not installed in this Python environment. Run Lab from "
                "the project's Psi4/chem-sapt environment."
            ) from exc
        psi4.set_num_threads(self.threads)
        psi4.set_memory(self.memory)
        psi4.core.set_output_file(os.devnull, False)
        psi4.set_options({"reference": reference})
        molecule = psi4.geometry(
            build_psi4_geometry(symbols, coordinates, charge, multiplicity)
        )
        try:
            gradient, wavefunction = psi4.gradient(
                f"{method}/{basis}", molecule=molecule, return_wfn=True
            )
            energy = float(wavefunction.energy())
            gradient_array = np.asarray(gradient, dtype=np.float64)
            result = {
                "single_point_energy_hartree": energy,
                "gradient_hartree_per_bohr": gradient_array,
                "psi4_version": getattr(psi4, "__version__", None),
            }
            try:
                optimised_energy, optimised_wfn = psi4.optimize(
                    f"{method}/{basis}", molecule=molecule, return_wfn=True
                )
            except Exception as exc:
                result.update({
                    "optimisation_converged": False,
                    "optimisation_error": f"{type(exc).__name__}: {exc}",
                })
                return result
            optimised = (
                np.asarray(optimised_wfn.molecule().geometry(), dtype=np.float64)
                * BOHR_TO_ANGSTROM
            )
            variables = {
                str(key): float(value)
                for key, value in optimised_wfn.scalar_variables().items()
                if np.isscalar(value)
            }
            result.update({
                "optimisation_converged": True,
                "optimised_energy_hartree": float(optimised_energy),
                "optimised_coordinates_A": optimised,
                "optimisation_metadata": variables,
            })
            return result
        finally:
            psi4.core.clean()


def validation_directory(molecule_id, root=DEFAULT_ROOT):
    return Path(root) / str(molecule_id)


def list_validations(molecule_id, root=DEFAULT_ROOT):
    directory = validation_directory(molecule_id, root)
    found = []
    if directory.is_dir():
        for path in directory.glob("QV_*.json"):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(found, key=lambda row: row.get("created_unix", 0), reverse=True)


def load_validation(molecule_id, validation_id, root=DEFAULT_ROOT):
    path = validation_directory(molecule_id, root) / f"{validation_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_validation(molecule_id, charge, multiplicity, *, method=DEFAULT_METHOD,
                   basis=DEFAULT_BASIS, reference=None, root=DEFAULT_ROOT,
                   molecule_root=molecule_library.DEFAULT_ROOT, runner=None):
    molecule = molecule_library.load_molecule(molecule_id, root=molecule_root)
    symbols = molecule["symbols"]
    original = np.asarray(molecule["positions"], dtype=np.float64).copy()
    original_bonds = np.asarray(molecule["bonds"], dtype=np.int32).copy()
    charge, multiplicity = int(charge), int(multiplicity)
    if reference is None:
        reference = "rhf" if multiplicity == 1 else DEFAULT_REFERENCE
    validation_id = "QV_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = validation_directory(molecule_id, root)
    record_path = directory / f"{validation_id}.json"
    payload_path = directory / f"{validation_id}.npz"
    source = molecule.get("source", {})
    record = {
        "format_version": 1,
        "id": validation_id,
        "molecule_id": molecule_id,
        "status": "running",
        "created_unix": time.time(),
        "method": method,
        "basis": basis,
        "reference": reference,
        "charge": charge,
        "multiplicity": multiplicity,
        "source": {key: source.get(key) for key in ("recording", "batch", "seed", "frame", "time_fs")},
        "chemistrymodel_git_revision": _git_revision(),
        "physics_model_revision": molecule.get("physics_model_revision"),
        "original_geometry_sha256": geometry_sha256(symbols, original),
        "host": {"python": sys.version, "platform": platform.platform()},
        "payload": payload_path.name,
    }
    _atomic_json(record_path, record)
    started = time.perf_counter()
    try:
        raw = (runner or Psi4Runner()).run(
            symbols, original.copy(), charge, multiplicity, method, basis, reference
        )
        gradient = np.asarray(raw["gradient_hartree_per_bohr"], dtype=np.float64)
        forces = -gradient * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        original_energy = float(raw["single_point_energy_hartree"])
        single_point = {
            "energy_hartree": original_energy,
            "energy_eV": original_energy * HARTREE_TO_EV,
            "max_force_eV_per_A": float(np.max(np.linalg.norm(forces, axis=1))),
            "rms_force_eV_per_A": float(np.sqrt(np.mean(np.sum(forces ** 2, axis=1)))),
        }
        if not raw.get("optimisation_converged", True):
            np.savez_compressed(
                payload_path, symbols=np.asarray(symbols, dtype="U2"),
                original_coordinates_A=original, original_bonds=original_bonds,
                qm_gradient_hartree_per_bohr=gradient,
                qm_forces_eV_per_A=forces,
            )
            record.update({
                "status": "failed",
                "completed_unix": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "single_point": single_point,
                "optimisation": {
                    "converged": False,
                    "error": raw.get("optimisation_error", "QM optimisation failed"),
                },
                "error": raw.get("optimisation_error", "QM optimisation failed"),
                "psi4_version": raw.get("psi4_version"),
            })
        else:
            optimised = np.asarray(raw["optimised_coordinates_A"], dtype=np.float64)
            comparison = compare_structures(symbols, original, original_bonds, optimised)
            np.savez_compressed(
                payload_path, symbols=np.asarray(symbols, dtype="U2"),
                original_coordinates_A=original, original_bonds=original_bonds,
                qm_gradient_hartree_per_bohr=gradient,
                qm_forces_eV_per_A=forces,
                optimised_coordinates_A=optimised,
            )
            optimised_energy = float(raw["optimised_energy_hartree"])
            record.update({
                "status": "complete",
                "completed_unix": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "single_point": single_point,
                "optimisation": {
                    "converged": True,
                    "energy_hartree": optimised_energy,
                    "energy_eV": optimised_energy * HARTREE_TO_EV,
                    "relaxation_energy_hartree": optimised_energy - original_energy,
                    "relaxation_energy_eV": (optimised_energy - original_energy) * HARTREE_TO_EV,
                    "metadata": raw.get("optimisation_metadata", {}),
                },
                "comparison": comparison,
                "psi4_version": raw.get("psi4_version"),
            })
    except Exception as exc:
        record.update({
            "status": "failed",
            "completed_unix": time.time(),
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        })
    _atomic_json(record_path, record)
    return record


def load_geometries(record, root=DEFAULT_ROOT):
    path = validation_directory(record["molecule_id"], root) / record["payload"]
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule_id")
    parser.add_argument("--charge", type=int, required=True)
    parser.add_argument("--multiplicity", type=int, required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    args = parser.parse_args()
    result = run_validation(
        args.molecule_id, args.charge, args.multiplicity,
        method=args.method, basis=args.basis,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
