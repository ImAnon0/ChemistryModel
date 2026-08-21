"""State-conditioned density-anisotropy probe for unified radial bonding.

This is deliberately a falsification reference.  It does not add charges,
dipoles, an angle correction, or a fitted response strength.  The only new
signal is the frozen SAPT P2 exchange-density anisotropy evaluated separately
in the two bond-assignment states of an existing H transfer.
"""

from __future__ import annotations

from dataclasses import replace

import torch

import nonbonded_continuous_torch as density
import reactive as R
from h_state_torch import _single_h_transfer
from research.sapt.sapt_h_state_torch import _descriptor_weights_for_state
from research.unified_bond_capacity.prototype import (
    UnifiedBondCapacityEnergyPrototype,
)


class StateConditionedP2CouplingPrototype(UnifiedBondCapacityEnergyPrototype):
    """Expose H-state coupling to a frozen local density P2 moment.

    ``directional_response`` is a Boolean scientific control, not a tunable
    strength.  False executes the inherited Hamiltonian exactly; True applies
    the complete frozen P2 angular factor with its independently calibrated
    H/C/N/O coefficients.
    """

    physics_model_name = "research_state_conditioned_p2_coupling_v0"
    physics_model_revision = 0
    research_only = True

    def __init__(self, *args, directional_response=True, **kwargs):
        if not isinstance(directional_response, bool):
            raise TypeError("directional_response must be a Boolean control")
        self.directional_response = directional_response
        self._directional_probe_records = []
        self._directional_electronic_diagnostics = None
        super().__init__(*args, **kwargs)

    def energy_per_atom(self, positions):
        self._directional_probe_records = []
        result = super().energy_per_atom(positions)
        factors = [
            row["transition_factor"] for row in self._directional_probe_records
        ]
        self._directional_electronic_diagnostics = {
            "formulation": "state_conditioned_frozen_sapt_p2_coupling",
            "directional_response": self.directional_response,
            "transition_count": len(self._directional_probe_records),
            "minimum_transition_factor": min(factors, default=1.0),
            "maximum_transition_factor": max(factors, default=1.0),
            "transitions": tuple(self._directional_probe_records),
            "parameter_source": "frozen SAPT0/jun-cc-pVDZ EXCH10 descriptor",
        }
        return result

    def _state_fragment(self, factor, state, positions, values):
        first_atom = min(atom for pair in factor.atoms for atom in pair)
        box = first_atom // self.per_box
        start = box * self.per_box
        stop = start + self.per_box
        local_positions = positions[start:stop]

        # Give the density descriptor one continuous minimum-image molecular
        # image. This is a coordinate representation only; no topology or
        # energy is detached from the live Torch graph.
        anchor = local_positions[:1]
        offsets = local_positions - anchor
        local_positions = anchor + offsets - self.box_size * torch.round(
            offsets / self.box_size
        )
        symbols_for = {
            int(index): symbol for symbol, index in R.ELEMENT_INDEX.items()
        }
        symbols = [
            symbols_for[int(self.types_numpy[atom])]
            for atom in range(start, stop)
        ]
        weights = _descriptor_weights_for_state(
            box=box,
            per_box=self.per_box,
            types_numpy=self.types_numpy,
            neighbours=values["neighbours"],
            neighbour_mask=self.neighbour_mask,
            taper=values["taper"],
            edge_atoms=factor.atoms,
            edge_tapers=[
                values["taper"][row, slot]
                for row, slot in zip(factor.rows, factor.slots)
            ],
            state=state,
        )
        return density.ContinuousTorchFragment(
            symbols=symbols,
            positions=local_positions,
            bond_weights=weights,
        ), start

    def _target_density_factor(
        self, factor, state_index, target_edge, hydrogen, positions, values
    ):
        state = factor.states[state_index]
        fragment, start = self._state_fragment(factor, state, positions, values)
        first, second = factor.atoms[target_edge]
        target = second if first == hydrogen else first
        local_target = target - start
        local_hydrogen = hydrogen - start
        direction = (
            fragment.positions[local_hydrogen]
            - fragment.positions[local_target]
        )
        q2 = density.amplitude_q2(fragment, local_target, direction)
        symbol = fragment.symbols[local_target]
        coefficient = float(density.ELEMENT_PARAMETERS[symbol].k)
        angular_factor = torch.exp(q2 * coefficient)
        return angular_factor, q2, symbol, target

    def _h_factor(self, edge_atoms, edge_rows, edge_slots, values, heavy_index):
        factor = super()._h_factor(
            edge_atoms, edge_rows, edge_slots, values, heavy_index
        )
        if not self.directional_response or len(factor.states) < 2:
            return factor
        cached = getattr(self, "_reactive_intermediates", None)
        if cached is None:
            raise RuntimeError("directional P2 probe requires live intermediates")
        positions = cached[0]
        transition_factors = {}
        transition_records = []
        for first_state in range(len(factor.states)):
            for second_state in range(first_state + 1, len(factor.states)):
                transfer = _single_h_transfer(
                    factor.states[first_state],
                    factor.states[second_state],
                    factor.atoms,
                    self.types_numpy,
                )
                if transfer is None:
                    continue
                old_edge, new_edge, hydrogen = transfer
                forward, forward_q2, forward_symbol, forward_target = (
                    self._target_density_factor(
                        factor, first_state, new_edge, hydrogen, positions, values
                    )
                )
                reverse, reverse_q2, reverse_symbol, reverse_target = (
                    self._target_density_factor(
                        factor, second_state, old_edge, hydrogen, positions, values
                    )
                )
                directional = torch.sqrt(torch.clamp(forward * reverse, min=0.0))
                transition_factors[first_state, second_state] = directional
                transition_records.append({
                    "states": (first_state, second_state),
                    "hydrogen": hydrogen,
                    "targets": (forward_target, reverse_target),
                    "target_symbols": (forward_symbol, reverse_symbol),
                    "forward_q2": float(forward_q2.detach().cpu()),
                    "reverse_q2": float(reverse_q2.detach().cpu()),
                    "transition_factor": float(directional.detach().cpu()),
                })

        if not transition_factors:
            return factor
        rows = []
        for first_state in range(len(factor.states)):
            row = []
            for second_state in range(len(factor.states)):
                if first_state == second_state:
                    value = factor.hamiltonian[first_state, second_state]
                else:
                    key = (min(first_state, second_state), max(first_state, second_state))
                    value = factor.hamiltonian[first_state, second_state]
                    if key in transition_factors:
                        value = value * transition_factors[key]
                row.append(value)
            rows.append(torch.stack(row))
        self._directional_probe_records.extend(transition_records)
        return replace(factor, hamiltonian=torch.stack(rows))
