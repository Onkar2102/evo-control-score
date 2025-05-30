"""
variation_operators.py
----------------------
Abstract interface for variation operators used by the evolutionary engine.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import random


class VariationOperator(ABC):
    """Base class for mutation / crossover operators on text strings."""

    name: str = "VariationOperator"
    arity: int = 1                   # 1 = mutation, 2 = crossover …

    @abstractmethod
    def apply(self, parents: Sequence[str]) -> str:
        """Return a single child string given *arity* parent strings."""
        raise NotImplementedError

    # Optional helper for probabilistic application
    def maybe_apply(self, parents: Sequence[str], p: float = 1.0) -> str:
        if random.random() < p:
            return self.apply(parents)
        return parents[0] if self.arity == 1 else random.choice(parents)