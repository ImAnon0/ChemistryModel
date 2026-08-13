import numpy as np

import torch

from scipy.spatial import cKDTree

import reactive as R

from reactive_torch import (
    MAXIMUM_NEIGHBOURS,
    ReactiveSimulation,
)


# ============================================================
# Several boxes at once
# ============================================================
#
# A box of three hundred atoms leaves the card mostly idle: eight
# times as many atoms cost only twice the time, so most of a step
# is spent starting work rather than doing it. Running several
# batches as separate processes claims about half of that, and
# then stops - measured on a 4060 Ti, three processes reach 1.95
# times the throughput of one and a fourth makes it slightly
# worse.
#
# The rest needs the boxes to share the same kernels. They are
# laid end to end in one tensor, and the neighbour table is built
# for each box separately, so no atom ever sees one from another
# box. Every force, every bond and every angle is then computed
# in one pass over the lot.
#
# Nothing about the physics changes. Each box keeps its own
# periodic cell of the same size, its own starting positions and
# its own thermostat draw. Energies and temperatures come back
# per box because the per-atom terms are simply reshaped and
# summed within each block.
#
# The one requirement is that every box in a group has the same
# number of atoms and the same cell, which is exactly what a
# batch of runs under one condition already is.


class BatchedReactiveSimulation(ReactiveSimulation):

    def __init__(self, boxes, box_size, **rest):
        # boxes: a list of (symbols, positions), one per box.
        #
        # Each box carries its own symbols rather than sharing
        # one list, because the builder shuffles atoms as it
        # places them: two boxes of the same composition come back
        # with the same atoms in a different order. Assuming a
        # shared order would give every box after the first the
        # wrong elements, silently, everywhere.

        self.box_count = len(boxes)

        counts = {len(symbols) for symbols, _ in boxes}

        if len(counts) != 1:
            raise ValueError(
                "every box in a group must have the same number "
                f"of atoms, but got {sorted(counts)}"
            )

        self.per_box = counts.pop()

        all_symbols = []

        for symbols, _ in boxes:
            all_symbols.extend(symbols)

        stacked = np.concatenate(
            [np.asarray(positions, dtype=float)
             for _, positions in boxes],
            axis=0,
        )

        super().__init__(
            symbols=all_symbols,
            positions=stacked,
            box_size=box_size,
            **rest,
        )

    def symbols_for(self, box):
        start = box * self.per_box

        return self.symbols[start:start + self.per_box]

    # --------------------------------------------------------

    def build_neighbours(self):
        # One table for the lot, but built box by box.
        #
        # A single search over everything would pair atoms from
        # different boxes, since they share the same coordinates.
        # Searching each box on its own and shifting the indices
        # keeps them apart without the force calculation needing
        # to know anything about it.

        positions = self.positions_numpy % self.box_size

        total = len(positions)

        table = np.zeros(
            (total, MAXIMUM_NEIGHBOURS), dtype=np.int64
        )
        mask = np.zeros(
            (total, MAXIMUM_NEIGHBOURS), dtype=bool
        )

        radius = self.maximum_cutoff + self.neighbour_skin

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box

            here = positions[start:stop]

            tree = cKDTree(here, boxsize=self.box_size)

            lists = tree.query_ball_point(here, r=radius)

            for index, entries in enumerate(lists):
                others = [
                    item for item in entries if item != index
                ]

                if len(others) > MAXIMUM_NEIGHBOURS:
                    offsets = here[others] - here[index]
                    offsets -= self.box_size * np.round(
                        offsets / self.box_size
                    )

                    order = np.argsort(
                        np.sum(offsets ** 2, axis=1)
                    )

                    others = [
                        others[position]
                        for position in order[:MAXIMUM_NEIGHBOURS]
                    ]

                row = start + index

                table[row, :len(others)] = [
                    start + item for item in others
                ]
                mask[row, :len(others)] = True

        self.neighbours = torch.tensor(
            table, device=self.device, dtype=torch.long
        )

        self.neighbour_mask = torch.tensor(
            mask, device=self.device, dtype=torch.bool
        )
        self._neighbour_weight = self.neighbour_mask.to(self.dtype)

        self.reference_positions = self.positions.clone()
        self.rebuild_count += 1

    # --------------------------------------------------------
    # Per-box quantities

    def by_box(self, values):
        return values.reshape(self.box_count, self.per_box)

    @property
    def potential_per_box(self):
        per_atom = getattr(self, "_potential_per_atom", None)
        cache_is_current = (
            per_atom is not None
            and getattr(self, "_potential_cache_source", None)
            is self.positions
            and getattr(self, "_potential_cache_version", None)
            == self.positions._version
        )

        if not cache_is_current:
            with torch.no_grad():
                per_atom = self.energy_per_atom(self.positions)

        return (
            torch.sum(self.by_box(per_atom), dim=1)
            .detach().cpu().numpy()
        )

    @property
    def kinetic_per_box(self):
        kinetic, _ = self.thermodynamics_per_box
        return kinetic

    @property
    def thermodynamics_per_box(self):
        """Kinetic energy and temperature with one host transfer."""
        energies = 0.5 * self.masses * torch.sum(
            self.velocities ** 2, dim=1
        )

        kinetic = torch.sum(self.by_box(energies), dim=1) * 103.642

        freedom = max(3 * self.per_box - 3, 1)
        temperature = kinetic / (0.5 * freedom * 8.617333e-5)

        values = torch.stack((kinetic, temperature)).detach().cpu().numpy()
        return values[0], values[1]

    @property
    def temperature_per_box(self):
        # Three degrees of freedom per atom, less the three taken
        # by the box as a whole not moving.

        _, temperature = self.thermodynamics_per_box
        return temperature

    def positions_for(self, box):
        start = box * self.per_box

        return self.positions_numpy[start:start + self.per_box]

    @property
    def positions_per_box(self):
        """All box positions with one device-to-host transfer."""
        return (
            self.positions.detach()
            .reshape(self.box_count, self.per_box, 3)
            .cpu().numpy()
        )

    def velocities_for(self, box):
        start = box * self.per_box

        return (
            self.velocities[start:start + self.per_box]
            .detach().cpu().numpy()
        )

    @property
    def velocities_per_box(self):
        """All box velocities with one device-to-host transfer."""
        return (
            self.velocities.detach()
            .reshape(self.box_count, self.per_box, 3)
            .cpu().numpy()
        )

    def chemical_observations(self):
        """Split one compact device transfer into per-box pair diagnostics."""
        combined = super().chemical_observation()
        result = []
        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box
            keep = (
                (combined["first"] >= start)
                & (combined["first"] < stop)
            )
            result.append({
                "first": combined["first"][keep] - start,
                "second": combined["second"][keep] - start,
                "distance": combined["distance"][keep],
                "inner": combined["inner"][keep],
                "taper": combined["taper"][keep],
            })
        return result


def split_into_groups(seeds, group_size):
    # Whatever is left over goes into a smaller final group
    # rather than being padded out, since padding would mean
    # computing boxes nobody asked for.

    return [
        seeds[start:start + group_size]
        for start in range(0, len(seeds), group_size)
    ]


def compare_against_single(boxes, box_size, seed=0):
    # Does a group of boxes give the same answer as running each
    # on its own?
    #
    # This is the only thing that matters about the whole idea.
    # If the forces differ at all, every result computed this way
    # would be quietly wrong, and the speed would be worthless.

    separate = []

    for symbols, positions in boxes:
        one = ReactiveSimulation(
            symbols=symbols,
            positions=positions,
            box_size=box_size,
            random_seed=seed,
            relax_on_start=False,
        )

        forces, energy = one.compute_forces()

        separate.append(
            (forces.detach().cpu().numpy(), float(energy))
        )

    together = BatchedReactiveSimulation(
        boxes=boxes,
        box_size=box_size,
        random_seed=seed,
        relax_on_start=False,
    )

    all_forces, _ = together.compute_forces()

    all_forces = all_forces.detach().cpu().numpy()

    energies = together.potential_per_box

    report = []

    per_box = len(boxes[0][0])

    for index, (forces, energy) in enumerate(separate):
        start = index * per_box

        mine = all_forces[start:start + per_box]

        scale = max(np.abs(forces).max(), 1e-12)

        force_error = np.abs(mine - forces).max() / scale

        energy_error = abs(energies[index] - energy) / max(
            abs(energy), 1e-12
        )

        report.append({
            "box": index,
            "force_error": force_error,
            "energy_error": energy_error,
            "energy_alone": energy,
            "energy_grouped": float(energies[index]),
        })

    return report
