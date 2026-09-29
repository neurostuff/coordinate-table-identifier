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
