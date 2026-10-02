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
    # The realised count is about 3.1, because it sets the odds a banner is
    # written for the first analysis and every later analysis always gets one.
    # Left over: grouping is the whole reason to fine-tune rather than prompt,
    # and a table with more analyses in it is a harder one to group.
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

    # An analysis announced by a spanning header rather than a banner, so two
    # analyses share every row and only the column they sit in says which is
    # which. 8.4% of multi-analysis curated tables are built this way and the
    # generator made none, which left the only grouping cue it taught the one
    # a banner gives. Over, because this is the harder of the two.
    analyses_in_column_groups: float = 0.16            # real 8.4%   OVER

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
    # Realised nearer 36%, because a packed triple has no axis columns to
    # name and takes 22% of tables with it. Both push the same way: the reader
    # has to find the coordinates without being told where they are.

    # A single header spanning three columns -- `MNI coordinates` with no x, y
    # or z beneath it -- is what `axes_named_in_header` NOT firing already
    # produces, at 55%. Real rate 7.2%, so this is heavily OVER on purpose and
    # does not need a second knob.

    # The whole triple in one cell: `-42, -55, -18`, `(-42, -55, -18)`,
    # `-42-55 -18`, or several peaks separated by a semicolon. Nearly a quarter
    # of real coordinate tables, and the generator has never produced one.
    coordinates_packed_in_one_cell: float = 0.22       # real 22.4%  MATCH
    # How a packed cell is written. Every one of these was read off a real
    # table and none of them existed in the generator, so a model trained on
    # it met them cold. Shares of the packed cells, summing under one; what is
    # left over is the plain `-42, -55, -18`.
    packed_in_brackets: float = 0.22                   # `(-42, -55, -18)`
    packed_signs_run_on: float = 0.14                  # `-10-42 16`
    packed_semicolons: float = 0.08                    # `10.2; 20.5; 19.5`
    packed_after_a_statistic: float = 0.14             # `9.91 [3, 15, 51]`
    packed_with_a_note: float = 0.10                   # `-56 -30 28 (OP1)`

    # An axis header that says what kind of coordinate it is. `X coor` is
    # three headers in 6,897 and losing it lost a whole table.
    axis_names_its_kind: float = 0.12                  # `X coor`, `y coordinate`

    # Each region or contrast gets a column and every cell holds a triple, so
    # one row carries a peak per column. 157 of the tables the old filter kept
    # are built this way.
    triple_column_per_group: float = 0.08              # real ~7% of losses  OVER

    # -- shapes the reader reads NOTHING out of -----------------------------
    # Read off 98 residual-route tables judged by hand: 41 held coordinates
    # and the reader found none of them. These are the residual gate's whole
    # positive class, and it had 34 examples of it.
    #: Coordinates as voxel indices rather than millimetres -- `92 | 132 | 96`
    #: under a header reading `Peak MNI`, or `COG-x (vox)`. Every value is
    #: positive and outside a head, so the reader's mm bounds reject the lot.
    voxel_indices: float = 0.05                        # 3 of 41  OVER
    #: `- 34 - 68 - 28`: a space between the sign and the digits, which is how
    #: several publishers' HTML arrives.
    signs_spaced: float = 0.06                         # 2 of 41  OVER
    #: The triple comes first and the region is named after it.
    coordinates_before_the_region: float = 0.05        # 2 of 41  OVER
    #: `Insula (-33, 21, 3)` -- the region cell carries its own coordinates,
    #: and no column holds them.
    coordinates_in_the_region_name: float = 0.05       # 2 of 41  OVER
    #: One header cell naming all three axes over three columns, with nothing
    #: marking the span: `Coordinates (x, y, z mm)`.
    span_without_a_marker: float = 0.05                # 2 of 41  OVER

    # -- shapes nothing can read yet ---------------------------------------
    # These are here because only a model can do them. The reader cannot, and
    # generating them is the only way the model will ever see one.
    coordinates_along_rows: float = 0.03               # axes are rows, not columns
    roi_centroid_in_header: float = 0.03               # `#<2:Left DLPFC (-45, 15, 35)`
    bilateral_pair: float = 0.03                       # `(±30 -80 6)`, per TABLE
    ras_instead_of_xyz: float = 0.03                   # `#R | #A | #S`

    # The header written in <td>, so nothing in the table is marked a header.
    # Three of the seventeen in the uncertain band.
    header_row_unmarked: float = 0.15                  # real 7.0%   OVER

    caption_present: float = 0.89                      # real 88.9%  MATCH

    # The label set carried no footnotes at all until they were joined back on
    # from the re-extraction, so every rate below that mentions a footnote was
    # previously fitted against nothing.
    footer_present: float = 0.33                       # real 47.0%  UNDER
    # The realised rate is higher than this: a footnote that has to carry the
    # space or the statistic is written whatever this says. 0.33 lands the
    # total near 47%. Going lower would be harder still -- a missing footnote
    # is a missing sentence the reader wanted -- but 47% is what papers do.
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
    zero_coordinate: float = 0.018                     # real 2.1%   OVER
    # Applied per row, and a table has many rows, so the realised share of
    # POINTS with x = 0 is about four times this. Left as it is: a midline
    # coordinate is the one value where sign carries no information, so every
    # sign-based shortcut fails on it at once, and more of them is harder.
    zero_x_after_extent: float = 0.25                  # of zero-x rows  OVER

    # Coordinates are drawn from the region name, so laterality and sign agree
    # and nothing lands outside a head. The generator drew them independently:
    # 91.2% agreement against 97.3% real, and 0.59% outside the brain (34(p)).
    laterality_agrees: float = 0.973                   # real 97.3%  MATCH
    non_integer_coordinates: float = 0.11              # real 3.9%   OVER
    # Per point. It used to be rolled per axis, which made 29.5% of points
    # fractional rather than 11%.

    decimal_statistics: float = 0.20                   # real ~5.5%  OVER
    six_or_more_analyses: float = 0.14                 # real 7%     OVER
    #: A second statistic column beside the first, almost always a
    #: p-value. 1,529 of the real tables scanned print a p; 853 of
    #: those print a test statistic beside it.
    second_statistic_printed: float = 0.25
    #: Real 55.7%, measured on the re-serialised curated set where the
    #: document names the statistic for 98.1% of points. The earlier
    #: "real 66%" came from luna's labels, which answer T 78.3% of the
    #: time and agree with the header 17% of the time.
    #: Of the sub-headings that sit inside an analysis, how many name a
    #: threshold rather than a lobe. 828 of 53,653 production analyses (1.54%)
    #: are named after a threshold the model took for the contrast.
    subheading_is_a_threshold: float = 0.4
    statistic_is_t: float = 0.45                       # real 55.7%  UNDER

    # -- the mess ----------------------------------------------------------
    # Real tables are not tidy, and every version before v19 was trained as if
    # they were. Measured over the 748 real coordinate tables in the label set.
    #
    # A superscript on a value -- `4.14*`, `-26a`, `Cuneusa` -- is on 4.46% of
    # cells and 93.3% of TABLES. The generator had never produced one, so the
    # commonest thing a real table does to a number was absent from training.
    footnote_markers: float = 0.07                     # real 4.46%  OVER
    # Per cell, and what matters is how many TABLES carry one: 93.3% of
    # real ones do, and at 0.05 the generator managed 79.7%.

    # A header wears one far less often than a value does: 0.54% of real
    # headers carry a symbol and 0.04% a bare letter after a bracket. Rare,
    # and the rare one is what cost a table -- `X (mm)d` hid the axis and the
    # whole table went with it -- so this is over the real rate but nowhere
    # near the rate for values.
    #
    # A letter that follows a letter is not a marker: `T-value`, `Side`,
    # `Anatomic site` end that way because the word does, and 1,613 real
    # headers would be mangled by treating them as marked.
    header_markers: float = 0.02                       # real 0.58%  OVER

    # An empty cell. 12.89% of real cells, and over half of tables have some.
    blank_cells: float = 0.13                          # real 12.89% MATCH

    # A row that stops short of the full width. 44.8% of real tables have one.
    ragged_rows: float = 0.10                          # real ~10% of rows  MATCH

    # `-`, `n.s.`, `N/A` where a number was expected.
    dash_for_missing: float = 0.01                     # real 0.91%  MATCH

    # A row naming a region and stating no coordinates for it -- a sub-heading
    # that is not a banner. Its point is absent from the target, so a model
    # that invents one is wrong.
    rows_without_coordinates: float = 0.04             # real unmeasured  OVER

    # A contrast that was run and found nothing. The table names it and then
    # says `n.s.`, `No significant activation`, or fills the row with dashes.
    # That is an ANALYSIS with no points, not an absence of one: the paper
    # ran the contrast and is reporting its result. Five of about 120 real
    # tables read by hand carry one, and no curated target in 10,015 has ever
    # held one, so the model has never been taught the case.
    analysis_found_nothing: float = 0.04               # real ~4% of TABLES  OVER

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
