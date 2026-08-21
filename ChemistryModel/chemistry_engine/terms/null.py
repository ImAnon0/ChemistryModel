from __future__ import annotations

import torch


class NullEnergyTerm:
    """Explicit zero contribution extension.

    The extension boundary receives the current energy tensor and must
    return a contribution with identical device/dtype semantics.
    """

    name = "null"

    def energy(self, context, current_energy):
        return torch.zeros_like(current_energy)
