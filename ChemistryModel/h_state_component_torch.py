"""
Local-component H-state model.

This is a physics correction to the experimental H-state reference.

Problem in the whole-box reference
----------------------------------
h_state_torch.py builds one H-state graph for every active H-containing
contact in an entire simulation box. Independent hydrogen-transfer networks
therefore enter one Cartesian-product state space and the crowding
normalisation can make physically disconnected events alter one another's
energy.

This module factorises the H-state correction into independent hydrogen
competition components.

Component definition
--------------------
Candidate edges are connected ONLY when they share a hydrogen atom.

That is intentionally narrower than ordinary molecular connectivity:

    C-H1 and C-H2
        share carbon but NOT hydrogen
        -> independent H-valence components

    C-H1 and H1-H2
        share H1
        -> same component

    H1-H2 and O-H2
        share H2
        -> same component

Thus an H-H edge can join what would otherwise be separate hydrogen
competition networks, while a shared heavy atom does not.

Within every component this class calls the unchanged
HStateReferenceBatchedSimulation._box_state_energy(), so:
    - allowed one-valence H states are unchanged
    - contact overlap is unchanged
    - crowding normalisation is unchanged
    - H_STATE_MIXING is unchanged
    - the local Hamiltonian/eigensolve is unchanged

The only physics change is restoring locality / size consistency between
disconnected H-valence problems.

The existing h_state_torch.py remains the historical whole-box reference.
"""

from __future__ import annotations

import torch

import reactive as R

from h_state_torch import HStateReferenceBatchedSimulation


H_STATE_COMPONENT_MODEL_NAME = "reactive_v2_h_state_local_components"
H_STATE_COMPONENT_MODEL_REVISION = 0


class HStateComponentBatchedSimulation(
    HStateReferenceBatchedSimulation
):
    """H-state reference factorised into independent H competition networks."""

    physics_model_name = H_STATE_COMPONENT_MODEL_NAME
    physics_model_revision = H_STATE_COMPONENT_MODEL_REVISION

    def _hydrogen_edge_components(self, edge_atoms):
        """
        Return tuples of edge indices connected through shared H atoms.

        Connectivity is on the H-valence conflict graph, NOT the ordinary
        atom-bond graph. Heavy atoms never connect two components by
        themselves.
        """

        hydrogen = int(R.ELEMENT_INDEX["H"])

        hydrogen_to_edges = {}
        edge_hydrogens = []

        for edge_index, (first, second) in enumerate(edge_atoms):
            hydrogens = tuple(
                atom
                for atom in (first, second)
                if int(self.types_numpy[atom]) == hydrogen
            )

            if not hydrogens:
                raise RuntimeError(
                    "H-state component builder received a non-H edge"
                )

            edge_hydrogens.append(hydrogens)

            for atom in hydrogens:
                hydrogen_to_edges.setdefault(
                    atom,
                    [],
                ).append(edge_index)

        visited = set()
        components = []

        for seed in range(len(edge_atoms)):
            if seed in visited:
                continue

            stack = [seed]
            visited.add(seed)
            component = []

            while stack:
                edge_index = stack.pop()
                component.append(edge_index)

                for hydrogen_atom in edge_hydrogens[edge_index]:
                    for neighbour_edge in hydrogen_to_edges[
                        hydrogen_atom
                    ]:
                        if neighbour_edge in visited:
                            continue

                        visited.add(neighbour_edge)
                        stack.append(neighbour_edge)

            components.append(
                tuple(sorted(component))
            )

        components.sort(
            key=lambda component: component[0]
        )

        return tuple(components)

    def _hydrogen_state_correction(
        self,
        positions,
        base_per_atom,
    ):
        cached = getattr(
            self,
            "_reactive_intermediates",
            None,
        )

        if cached is None or cached[0] is not positions:
            raise RuntimeError(
                "component H-state requires current reactive intermediates"
            )

        values = cached[1]

        neighbours_numpy = (
            values["neighbours"]
            .detach()
            .cpu()
            .numpy()
        )

        active_numpy = (
            (
                values["taper"]
                .detach()
                .cpu()
                .numpy()
                > 1e-12
            )
            & self.neighbour_mask
            .detach()
            .cpu()
            .numpy()
        )

        taper = values["taper"]
        pair_depth = values["pair_depth"]
        pair_width = values["pair_width"]
        shift = values["shift"]
        repulsive = values["repulsive"]

        attractive = (
            2.0
            * pair_depth
            * torch.exp(
                -pair_width * shift
            )
        )

        pair_morse = (
            taper
            * (repulsive - attractive)
        )

        # Same base-H overcoordination term removed by the whole-box
        # reference. The only change is assigning each participating H to
        # its local component exactly once.
        over_scale = self.over_coordination_scale(
            taper,
            values["unsoftened_depth"],
            values["mask"],
            cache_key=positions,
        )

        excess = torch.clamp(
            values["coordination"]
            - values["valence"],
            min=0.0,
        )

        base_over = (
            self.over_penalty
            * over_scale
            * excess
            * excess
        )

        hydrogen = int(R.ELEMENT_INDEX["H"])

        correction = torch.zeros_like(
            base_per_atom
        )

        all_indices = torch.arange(
            len(base_per_atom),
            device=base_per_atom.device,
        )

        # Diagnostics are detached bookkeeping only.
        component_counts = []
        largest_component_edges = 0

        for box in range(self.box_count):
            start = box * self.per_box
            stop = start + self.per_box

            (
                edge_atoms,
                edge_rows,
                edge_slots,
            ) = self._active_edges_for_box(
                box,
                values,
                neighbours_numpy,
                active_numpy,
            )

            if not edge_atoms:
                component_counts.append(0)
                continue

            components = self._hydrogen_edge_components(
                edge_atoms
            )

            component_counts.append(
                len(components)
            )

            for component in components:
                largest_component_edges = max(
                    largest_component_edges,
                    len(component),
                )

                component_edge_atoms = tuple(
                    edge_atoms[index]
                    for index in component
                )

                component_edge_rows = tuple(
                    edge_rows[index]
                    for index in component
                )

                component_edge_slots = tuple(
                    edge_slots[index]
                    for index in component
                )

                # Exact historical H-state Hamiltonian, now local.
                state_energy = self._box_state_energy(
                    component_edge_atoms,
                    component_edge_rows,
                    component_edge_slots,
                    values,
                )

                base_pair_terms = [
                    pair_morse[row, slot]
                    for row, slot in zip(
                        component_edge_rows,
                        component_edge_slots,
                    )
                ]

                base_h_pair = torch.stack(
                    base_pair_terms
                ).sum()

                component_hydrogens = sorted({
                    atom
                    for first, second in component_edge_atoms
                    for atom in (first, second)
                    if int(self.types_numpy[atom]) == hydrogen
                })

                if component_hydrogens:
                    base_h_over = torch.stack([
                        base_over[atom]
                        for atom in component_hydrogens
                    ]).sum()
                else:
                    base_h_over = state_energy * 0.0

                delta = (
                    state_energy
                    - base_h_pair
                    - base_h_over
                )

                anchor = (
                    component_hydrogens[0]
                    if component_hydrogens
                    else start
                )

                weight = (
                    all_indices == anchor
                ).to(base_per_atom.dtype)

                correction = (
                    correction
                    + weight * delta
                )

        self._h_component_diagnostics = {
            "component_counts_per_box": tuple(component_counts),
            "largest_component_edges": int(largest_component_edges),
        }

        return correction


# Short aliases for experiments.
ComponentHStateBatchedSimulation = HStateComponentBatchedSimulation
LocalHStateBatchedSimulation = HStateComponentBatchedSimulation
