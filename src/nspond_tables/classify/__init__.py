"""A high-recall gate deciding whether a table could hold coordinates.

    from nspond_tables.classify import Gate, features, dataset
    rows, labels = dataset.from_jsonl("coord_labels.jsonl")
    gate = model.fit(rows, labels)
    gate.predict(serialised_text, caption=caption)

Why a gate and not a classifier: `create_analyses` drops a table with no
coordinates, so a miss loses the article for good while a false alarm costs one
call that returns nothing. Tune on recall at a precision floor.
"""

from . import dataset, features, model
from .model import Gate

__all__ = ["Gate", "dataset", "features", "model"]
