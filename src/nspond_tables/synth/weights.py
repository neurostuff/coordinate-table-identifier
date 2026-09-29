"""How often the generator does each thing, and why.

Every rate carries the real rate beside it and one of three verdicts:

  MATCH      reproduce what papers do
  OVER       deliberately above the real rate, so a shortcut meets enough
             counterexamples to break. A cue that is 90% reliable is not
             unlearned by seeing it fail 10% of the time.
  UNDER      deliberately below, so the model cannot come to depend on a cue
             that is often absent in the wild

Real rates are measured over the 745 real coordinate tables in the label set
after the re-extraction -- ACE scanning source html for every table, and
captions and footnotes kept on all four sources. Several rates moved when that
landed, and the ones that did say so.
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
    banners_per_table: float = 2.2                     # real 1.45  OVER
    # Was MATCH against 2.23. The re-extracted corpus averages 1.45, because
    # the rescued tables are smaller and often hold one analysis. Grouping is
    # the whole reason to fine-tune rather than prompt, so the rate stays where
    # it was and the verdict changes instead.

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
    # Both fell sharply on the re-extracted corpus: the rescued tables state
    # the space far less often than the ones ACE used to parse.
    space_in_table: float = 0.45                       # real 45.2%  MATCH
    space_in_context_only: float = 0.12                # real 12.1%  MATCH
    # leaves ~43% stating it nowhere, against ~43% real

    statistic_in_header: float = 0.45                  # real 44.2%  MATCH
    statistic_in_footnote: float = 0.08                # real 3.8%   OVER
    # leaves ~47% naming it nowhere, against 52.0% real

    # Naming the axes is a locating cue. Handing it over more often than
    # reality does teaches the model to look for it and then fail on the 42% of
    # real tables that never name them (34(d)).
    axes_named_in_header: float = 0.45                 # real 64.6%  UNDER

    # A single header spanning three columns -- `MNI coordinates` with no x, y
    # or z beneath it -- is what `axes_named_in_header` NOT firing already
    # produces, at 55%. Real rate 7.2%, so this is heavily OVER on purpose and
    # does not need a second knob.

    # The whole triple in one cell: `-42, -55, -18`, `(-42, -55, -18)`,
    # `-42-55 -18`, or several peaks separated by a semicolon. Nearly a quarter
    # of real coordinate tables, and the generator has never produced one.
    coordinates_packed_in_one_cell: float = 0.22       # real 22.4%  MATCH
    packed_in_brackets: float = 0.35                   # of packed  MATCH

    # The header written in <td>, so nothing in the table is marked a header.
    # Three of the seventeen in the uncertain band.
    header_row_unmarked: float = 0.15                  # real 7.0%   OVER

    caption_present: float = 0.89                      # real 88.9%  MATCH

    # The label set carried no footnotes at all until they were joined back on
    # from the re-extraction, so every rate below that mentions a footnote was
    # previously fitted against nothing.
    footer_present: float = 0.47                       # real 47.0%  MATCH
    footer_chars: int = 112                            # real 112    MATCH

    # -- values -----------------------------------------------------------
    # A statistic printed as a cell, so the target can truthfully assert one.
    # `scrub_unstated` correctly nulled the ones that were never printed, but
    # accepting the null left the field half-supervised at twice the real rate
    # (34(k)).
    statistic_printed: float = 0.73                    # real 72.7%  MATCH

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

    # -- tables that hold nothing to extract -------------------------------
    # No version before v19 has ever been shown a table whose correct answer is
    # nothing, which is why the model invents an analysis when handed one. The
    # target for these is the empty structure, not a missing field and not an
    # analysis with an empty point list.
    #
    # They are built from the same vocabulary and the same layout machinery as
    # the positives -- spans, banner rows, sub-headers -- so the model cannot
    # separate them on shape. `Patients | 24 | 34.2 | 12` already trips a
    # triple detector, which is the case worth over-weighting.
    no_coordinates: float = 0.25                       # real ~60%   UNDER
    # UNDER because the gate already removes most of them before the model is
    # called; what the model needs is enough to learn the empty answer, not the
    # corpus proportion.

    # The near-misses, as a share of the tables that hold nothing. These are
    # the three that beat the gate at 0.86 or better in the hand review: a
    # brain-template comparison full of MNI names and millimetres, an
    # activation table that reports regions and t values and no coordinates,
    # and a statistic printed beside its interval, which parses as a triple.
    near_miss: float = 0.45                            # real unmeasured  OVER

    # -- brain bounds -----------------------------------------------------
    x_limit: int = 72
    y_limit: int = 104
    z_limit: int = 76


DEFAULT = Weights()
