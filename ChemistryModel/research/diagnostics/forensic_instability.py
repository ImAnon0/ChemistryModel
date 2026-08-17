"""Read-only forensic analysis of legacy batch instability recordings."""

import _bootstrap  # noqa: F401 - direct-execution project path

import argparse
import json
import os

import numpy as np
import torch

import reactive as R
from reactive_torch import ReactiveSimulation
from recorder import Recorder


PAIR_NAMES = {tuple(sorted(pair)): "-".join(sorted(pair)) for pair in (
    ("H", "H"), ("C", "H"), ("H", "N"), ("H", "O"),
    ("C", "C"), ("C", "N"), ("C", "O"), ("N", "N"),
    ("N", "O"), ("O", "O"),
)}


def minimum_image(offsets, box):
    return offsets - box * np.round(offsets / box)


def pair_geometry(recorder, index):
    positions = np.asarray(recorder.positions[index], dtype=float)
    symbols = recorder.symbols_at(index)
    ids = recorder.atom_ids_at(index)
    box = recorder.box_at(index)
    first, second = np.triu_indices(len(positions), 1)
    offsets = minimum_image(positions[second] - positions[first], box)
    distances = np.linalg.norm(offsets, axis=1)
    order = np.argsort(distances)[:8]
    rows = []
    types = recorder.types_at(index)
    inner = R.CUTOFF_INNER[types[first], types[second]]
    outer = R.CUTOFF_OUTER[types[first], types[second]]
    taper = R.smooth_cutoff(distances, inner, outer)
    for at in order:
        a, b = int(first[at]), int(second[at])
        rows.append({
            "slots": [a, b], "atom_ids": [int(ids[a]), int(ids[b])],
            "symbols": [symbols[a], symbols[b]],
            "pair": PAIR_NAMES.get(tuple(sorted((symbols[a], symbols[b]))),
                                   "-".join(sorted((symbols[a], symbols[b])))),
            "distance_A": float(distances[at]), "taper": float(taper[at]),
            "inner_A": float(inner[at]), "outer_A": float(outer[at]),
        })
    return rows


def frame_diagnostics(recorder, index):
    positions = np.asarray(recorder.positions[index], dtype=float)
    velocities = np.asarray(recorder.velocities[index], dtype=float)
    symbols = recorder.symbols_at(index)
    ids = recorder.atom_ids_at(index)
    box = recorder.box_at(index)
    simulation = ReactiveSimulation(
        symbols, positions, box, device="cpu", dtype=torch.float64,
        target_temperature=0.0, relax_on_start=False,
    )
    force_norm = torch.linalg.norm(simulation.forces, dim=1).detach().cpu().numpy()
    parts = {name: float(torch.sum(values))
             for name, values in simulation._energy_parts.items()}
    fastest = np.argsort(np.linalg.norm(velocities, axis=1))[-5:][::-1]
    strongest = np.argsort(force_norm)[-5:][::-1]
    return {
        "reconstructed_potential_eV": simulation.potential_energy,
        "energy_terms_eV": parts,
        "maximum_force_eV_per_A": float(force_norm[strongest[0]]),
        "strongest_atoms": [{"slot": int(i), "atom_id": int(ids[i]),
                              "symbol": symbols[i], "force_eV_per_A": float(force_norm[i])}
                             for i in strongest],
        "fastest_atoms": [{"slot": int(i), "atom_id": int(ids[i]),
                            "symbol": symbols[i],
                            "speed_A_per_fs": float(np.linalg.norm(velocities[i]))}
                           for i in fastest],
        "closest_pairs": pair_geometry(recorder, index),
    }


def strike_relation(time_fs, first=15000.0, interval=100.0, count=10):
    strikes = np.arange(count, dtype=float) * interval + first
    previous = strikes[strikes <= time_fs]
    nearest = strikes[np.argmin(np.abs(strikes - time_fs))]
    return {"nearest_strike_fs": float(nearest),
            "offset_from_nearest_strike_fs": float(time_fs - nearest),
            "since_previous_strike_fs": (None if not len(previous)
                                          else float(time_fs - previous[-1]))}


def analyse_recording(path, seed):
    recorder = Recorder.load(path)
    total = np.asarray(recorder.potential) + np.asarray(recorder.kinetic)
    rises = np.diff(total)
    threshold = max(80.0, 0.08 * abs(float(recorder.potential[-1])))
    events = []
    for before in np.where(rises > threshold)[0]:
        after = before + 1
        box = recorder.box_at(before)
        displacement = minimum_image(
            np.asarray(recorder.positions[after]) - np.asarray(recorder.positions[before]), box
        )
        magnitudes = np.linalg.norm(displacement, axis=1)
        largest = int(np.argmax(magnitudes))
        ids = recorder.atom_ids_at(before)
        symbols = recorder.symbols_at(before)
        time_fs = float(recorder.times[after])
        events.append({
            "seed": seed, "before_frame": int(before), "after_frame": int(after),
            "time_fs": time_fs, "threshold_eV": threshold,
            "energy_before_eV": float(total[before]), "energy_after_eV": float(total[after]),
            "delta_energy_eV": float(rises[before]),
            "potential_before_eV": float(recorder.potential[before]),
            "potential_after_eV": float(recorder.potential[after]),
            "kinetic_before_eV": float(recorder.kinetic[before]),
            "kinetic_after_eV": float(recorder.kinetic[after]),
            "temperature_before_K": float(recorder.temperature[before]),
            "temperature_after_K": float(recorder.temperature[after]),
            "maximum_10fs_displacement_A": float(magnitudes[largest]),
            "largest_displacement_atom": {"slot": largest, "atom_id": int(ids[largest]),
                                           "symbol": symbols[largest]},
            "strike_relation": strike_relation(time_fs),
            "before": frame_diagnostics(recorder, before),
            "after": frame_diagnostics(recorder, after),
        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--output", default="docs/amino_alcohol_instability.json")
    options = parser.parse_args()
    with open(os.path.join(options.folder, "index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    events = []
    for entry in index:
        events.extend(analyse_recording(
            os.path.join(options.folder, entry["file"]), int(entry["seed"])
        ))
    payload = {"source": os.path.normpath(options.folder), "events": events,
               "limitations": [
                   "legacy frames are 10 fs apart; exact sub-frame event state is unavailable",
                   "cap timestamps/atom IDs and strike channel membership were not recorded",
                   "forces and term energies are reconstructed at saved frames using the current engine",
               ]}
    os.makedirs(os.path.dirname(options.output) or ".", exist_ok=True)
    with open(options.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {len(events)} events to {options.output}")


if __name__ == "__main__":
    main()
