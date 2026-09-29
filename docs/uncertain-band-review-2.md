# Second hand review of the uncertain band

Fifty tables drawn from the gate's UNCERTAIN band on the **re-extracted**
corpus: ACE scanning source HTML for every table, captions and footnotes kept
on all four sources, and the gate re-fitted on the rebuilt labels. Five tables
from each of ten score deciles, so the sample is stratified and its overall
precision is not the band's rate. Judgments are in
`uncertain-band-judgments-2.json`.

The first review, on the old artifacts, is in `uncertain-band-review.md`.

## The gate selected correctly

**17 of the 50 hold coordinates, and the gate kept all 17.** Recall is 100%, as
it was on the old corpus.

| deciles | score | tables | hold coordinates |
| --- | --- | ---: | ---: |
| 1–6 | 0.007 – 0.61 | 30 | **0** |
| 7–10 | 0.61 – 1.00 | 20 | **17 (85%)** |

The lowest-scoring real table scores **0.6863**; the deployed threshold is
0.066. Raising it to the band's 60th percentile loses nothing here and takes
precision on the band from about a third to 85%. The first review found the
same shape with less room (0.269), so the re-fit moved the real tables up, not
the threshold.

The three false positives above 0.61 are the interesting ones, because they are
what a classifier built on topic alone cannot do:

* a brain-template comparison — AC-PC length, MNI-305 against ICBM-152, every
  number in millimetres, no points anywhere;
* a real activation table that reports region, side, voxels and t, and no
  coordinates;
* an ACE supplementary-file listing, whose *descriptions* are full of
  anatomy. It is not a data table at all.

Captions now reach 47 of the 50 and 15 of the 17 real ones, so the caption
artefact the first review blamed for the gate's weights is gone.

## Why the reader saw none of them

A table is in this band because the reader found nothing, so 0 of 17 is
definitional. The obstacles are not. Four of them, all fixed:

1. **A span names the axes** (6 tables). `#<3:MNI coordinates (X Y Z)` and
   nothing beneath it. The span says which three columns; the body says whether
   that reading is sane.
2. **The header is written in `<td>`** (3 tables). The axis row is right there
   and reads the same. Only the tag differs.
3. **The sign is printed, or detached** (4 tables). `+68 -16 34`,
   `-44 -90 + 12`.
4. **A row states more than one peak**. `-8/12//-6 | 56/48/52 | 38/48/30` is
   three points; reading the cell whole lost all three.

Two of the four were really serialiser bugs:

* Oxford Academic hides a sort marker in every `<th>`:
  `<span aria-hidden="true" style="display:none"> . </span>`. It rode into the
  cell, and `x .` is not an axis name.
* `&#xA0;` has an uppercase A, and the entity pattern was lowercase-only, so
  `-&#xA0;45` was not a number.

Measured across the 2,384-table label set after the fixes: positives hold at
498 of 498 and 7,648 points, negatives at 0 of 1,130, and the uncertain band
goes from 0 tables to **93, worth 1,653 points**.

Fifteen of the seventeen now read. The two that do not are pdfs docling
shredded, and there the OCR turned a leading minus into a `2` — `232` for −32,
`284` for −84. That is silent corruption in the pdf path, not a reader gap.

## What this leaves for v19

The 33 tables that hold no coordinates are hand-confirmed negatives, and the 30
below decile 7 are the ordinary ones: ANOVA effect tables, MRI acquisition
parameters, demographics, correlation matrices, ROC curves. The model has never
been shown a table whose correct answer is nothing.
