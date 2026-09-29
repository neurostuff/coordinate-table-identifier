"""Synthetic tables built as Grids, so they render through the real serialiser."""

from .build import Table, Truth, TruthAnalysis, TruthPoint, build
from .empty import build_empty
from .mix import build_mixed
from .trainset import EMPTY_TARGET, Example, build_trainset, census, write_jsonl
from .weights import DEFAULT, Weights

__all__ = ["Table", "Truth", "TruthAnalysis", "TruthPoint", "build",
           "build_empty", "build_mixed", "build_trainset", "census",
           "write_jsonl", "Example", "EMPTY_TARGET", "DEFAULT", "Weights"]
