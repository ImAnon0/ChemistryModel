"""Batch-layout interface kept outside physics configuration."""

from __future__ import annotations

from typing import Protocol


class BatchingBackend(Protocol):
    def assignment(self, atom_count: int):
        ...
