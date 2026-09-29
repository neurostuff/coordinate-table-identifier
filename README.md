# nspond-tables

One path from a publisher's table to structured coordinates, and a synthetic
generator that produces tables of the same shape.

    HTML ─┐
    CALS ─┼→ Grid → render() → serialised text → read() → fields
    CSV  ─┘    ↑
          synth.build()

## Why a Grid in the middle

The previous pipeline had two renderers: real tables went through a serialiser,
synthetic ones were written as strings by the generator. They drifted. Spanning
banners were never widened in synthetic data, rowspan-covered cells were blanked
where real ones were omitted, and targets asserted statistics the table never
printed. Every one of those is a symptom of the same split.

Here both sides build a `Grid` and a single `render()` emits the text, so the two
cannot disagree about the format.

## What is deterministic

`read()` resolves each cell to a column index, finds the axis header, and reads
coordinates straight down the column. On real tables this is exact for 62% of
them and partially right for 74%, measured over 240 tables re-serialised from
source. Space, statistic type and cluster measure are read from the header,
caption and footer by the rules in `fields/`. A model is needed for what is left:
grouping rows into analyses, and tables whose header does not name its axes.

## Usage

    from nspond_tables import serialize, read
    grid = serialize.from_html(raw_html)
    text = serialize.render(grid)
    result = read.extract(text, caption=caption, footer=footer)

    from nspond_tables.synth import build
    grid, truth = build(seed=0)

## Layout

    grid.py        Cell and Grid; column resolution; the rendering contract
    serialize.py   from_html / from_cals / from_csv; render
    read.py        deterministic extraction from a rendered grid
    fields/        space, statistic type, cluster measure, laterality
    synth/         grid-building generator and its difficulty weights

## Round trip

`read.extract` over 1,000 generated tables recovers, on the 430 where it can
locate the axes, **100% of coordinates, statistic values, statistic types and
extents, inventing none of any field**. That is a test of agreement between the
generator and the reader, not of either against reality -- but a disagreement
there is always a bug in one of them, and it found six:

* the truth asserted a space the document never named
* a footnote named a statistic the truth called null, and vice versa
* the `Z` statistic column was read as the `z` coordinate
* a threshold sentence (`p<0.05`) was read as naming the statistic
* the side column `L/R` was claimed as an `R` statistic
* a region rowspan covered rows generated from different regions

## Measurements

`evalkit/` holds the scripts. Every number in a docstring names one.

## The coordinate gate

`classify/` decides whether a table could hold coordinates at all, so a table
the extractor missed can be found again. It is a **gate, not a classifier**:
`create_analyses` drops a table with no coordinates, so a false negative loses
the article permanently while a false positive costs one call that returns
nothing. The threshold is the lowest that still clears a precision floor, and
the number reported is recall at that point.

Labels come without hand review. `detect.detect` commits to NEGATIVE only on
both structural and topical evidence; positives come from the extractor or from
`read.extract` locating three points. The UNCERTAIN band is excluded from
training, because it is the population the gate exists to sort.

Fitted on 1,755 tables (674 articles, split by article so a paper's tables never
straddle the boundary):

| precision floor | test precision | test recall | misses |
|---|---|---|---|
| 0.95 | 96.1% | **98.7%** | 3 of 224 |
| 0.90 | 90.2% | **99.1%** | 2 of 224 |

On the 591 UNCERTAIN tables it was never trained on, it splits them 87% pass /
13% drop, and the calls hold up by eye: it passes `#<4:Peak MNI coordinates` and
`Cluster | Brain Region | Ke | L/R% | x | y | z | t | P`, and drops "Features of
Conventional Radiomics Model" and "studies included in the meta-analysis".

**What this number is not.** The labels come from a heuristic, so the score
measures whether a cheap linear model reproduces that heuristic and generalises
across articles -- not whether it finds tables the heuristic cannot. Establishing
that needs hand review of the uncertain band.
