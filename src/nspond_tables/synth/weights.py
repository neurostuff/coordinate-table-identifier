"""How often the generator does each thing, and why.

Every rate carries the real rate beside it and one of three verdicts:

  MATCH      reproduce what papers do
  OVER       deliberately above the real rate, so a shortcut meets enough
             counterexamples to break. A cue that is 90% reliable is not
             unlearned by seeing it fail 10% of the time.
  UNDER      deliberately below, so the model cannot come to depend on a cue
             that is often absent in the wild

Real rates are measured over the curated half of the training set unless noted.
The scripts are in `scans/` on the training host; the section numbers refer to
`scans/PRE_V17.md`.

A rate with no verdict and no real rate is a bug. Do not add one.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    # -- structure --------------------------------------------------------
    # A spanning banner row is the main grouping cue. Real multi-analysis
    # tables carry 2.23 per table; the generator managed 1.58 (34(c)).
    banners_per_table: float = 2.2                     # real 2.23  MATCH

    # A divider rendered as a partial span with the rest of the row empty --
    # `<2:Occipito-temporal cortex | | | |`. Real 1.0%, generator 0.0%, and it
    # is the form the naturalistic table uses on every one of its four
    # sections (36).
    partial_span_divider: float = 0.33                 # real ~0.35 of dividers  MATCH

    # A rowspan in column 0 carries a group label down several rows, which the
    # model must propagate. A rowspan in a later column decorates a statistic.
    # Real tables put 25.6% in column 0 and 11.0% later; the generator had that
    # inverted at 8.6% and 35.3% (34(c)).
    rowspan_in_first_column: float = 0.26              # real 25.6%  MATCH
    rowspan_in_later_column: float = 0.11              # real 11.0%  MATCH

    # Two analyses with identical column structure, so the banner text is the
    # only evidence separating them. Nothing made grouping locally undecidable
    # before, which is why a model can group correctly without reading the
    # banner at all (34(j)).
    structurally_identical_neighbours: float = 0.30    # real unmeasured  OVER

    # -- where each signal lives -----------------------------------------
    # The generator used to put each signal in one fixed home, so the model
    # could learn a routing table instead of a reading strategy (34(g)).
    space_in_table: float = 0.70                       # real 84.1%  UNDER
    space_in_context_only: float = 0.20                # real 15.7%  MATCH
    # leaves ~10% stating it nowhere

    statistic_in_header: float = 0.45                  # real 44.2%  MATCH
    statistic_in_footnote: float = 0.08                # real 3.8%   OVER
    # leaves ~47% naming it nowhere, against 52.0% real

    # Naming the axes is a locating cue. Handing it over more often than
    # reality does teaches the model to look for it and then fail on the 42% of
    # real tables that never name them (34(d)).
    axes_named_in_header: float = 0.45                 # real 57.6%  UNDER

    caption_present: float = 0.84                      # real 83.6%  MATCH
    footer_chars: int = 120                            # real 120    MATCH

    # -- values -----------------------------------------------------------
    # A statistic printed as a cell, so the target can truthfully assert one.
    # `scrub_unstated` correctly nulled the ones that were never printed, but
    # accepting the null left the field half-supervised at twice the real rate
    # (34(k)).
    statistic_printed: float = 0.80                    # real 80.0%  MATCH

    # A numeric column immediately left of x. v14 located coordinates by
    # looking for a minus sign and scored 0/7 on positive-x rows; this is the
    # lever that broke it, and it works (34(n)).
    numeric_column_before_x: float = 0.74              # real ~28%   OVER

    # A midline coordinate is the one value where sign carries no information,
    # so every sign-based heuristic fails on it at once. The shape
    # `| extent | 0 | y | z |` appeared 30 times in 200,490 points (34(n)).
    zero_coordinate: float = 0.018                     # real 1.83%  MATCH
    zero_x_after_extent: float = 0.25                  # of zero-x rows  OVER

    # Coordinates are drawn from the region name, so laterality and sign agree
    # and nothing lands outside a head. The generator drew them independently:
    # 91.2% agreement against 97.3% real, and 0.59% outside the brain (34(p)).
    laterality_agrees: float = 0.973                   # real 97.3%  MATCH
    non_integer_coordinates: float = 0.11              # real 3.85%  OVER

    decimal_statistics: float = 0.20                   # real ~5.5%  OVER
    six_or_more_analyses: float = 0.14                 # real 7%     OVER
    statistic_is_t: float = 0.30                       # real 66%    UNDER

    # -- brain bounds -----------------------------------------------------
    x_limit: int = 72
    y_limit: int = 104
    z_limit: int = 76


DEFAULT = Weights()
