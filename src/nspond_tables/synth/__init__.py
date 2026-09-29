"""Synthetic tables built as Grids, so they render through the real serialiser."""

from .build import Table, Truth, TruthAnalysis, TruthPoint, build
from .weights import DEFAULT, Weights

__all__ = ["Table", "Truth", "TruthAnalysis", "TruthPoint", "build",
           "DEFAULT", "Weights"]
