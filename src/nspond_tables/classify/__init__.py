"""A high-recall gate deciding whether a table could hold coordinates.

    from nspond_tables.classify import fit_routed, dataset
    gate = fit_routed(list(dataset.rows_from_jsonl("coord_labels.jsonl")))
    gate.predict(serialised_text, caption=caption)

Two gates, routed on whether the reader read a triple out of the table: they
want different operating points, and one threshold cannot serve both.

Why a gate and not a classifier: `create_analyses` drops a table with no
coordinates, so a miss loses the article for good while a false alarm costs one
call that returns nothing. Tune on recall at a precision floor.
"""

from . import dataset, features, model
from .model import Gate, RoutedGate, fit_routed

__all__ = ["Gate", "RoutedGate", "fit_routed", "dataset", "features", "model"]
