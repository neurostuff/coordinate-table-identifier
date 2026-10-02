"""Fields a reader can fill without a model, from the header and the context.

Each module answers one question and says when it cannot. Returning None is a
result: a target that guesses a space on a table that never states one teaches a
model to invent them, which is why `space` moved from the model's input to its
output in the first place.
"""

from .laterality import Side, laterality, sign_agrees
from .measure import cluster_measure
from .space import visible_space
from .statistic import STATISTIC_PRIORITY, best_of, statistic_type

__all__ = ["Side", "cluster_measure", "laterality", "sign_agrees",
           "STATISTIC_PRIORITY", "best_of", "statistic_type", "visible_space"]
