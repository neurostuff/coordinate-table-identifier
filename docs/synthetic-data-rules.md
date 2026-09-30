# The rules a synthetic example has to obey

Every rule here was added because breaking it taught the model something
false, and most of them were found by reading generated tables one at a time.
Each says what it is, why, and where it is enforced and tested.

A rate is not a rule. Rates live in [`synth/weights.py`](../src/nspond_tables/synth/weights.py),
each carrying the real corpus figure and whether the generator sits over,
under or on it. The shapes those rates select live in
[`table-shapes.md`](table-shapes.md). This page is the invariants.

## 1. The target says only what the document says

| rule | why | where |
| --- | --- | --- |
| A target may not assert a **statistic type** the document never names. | A row printing `4.55` under a header reading `Value`, with silent footnotes, does not say whether that is a *t* or a *Z*. Claiming one teaches the model to invent statistic types. | `build._data_row`, `_Layout.stat_in_header` / `stat_in_footnote` |
| A target may not assert a **coordinate space** nothing states. | 53.5% of rows carry no visible cue. A space taken from article metadata teaches the model to invent one on half its examples. | `trainset.set_space_from_what_is_visible`, `context.visible_space` |
| Where two parts of the document **disagree** about the space, the target is null. | That is the rule the production prompt already gives: *if not stated or not confident, set space = null.* | `context.visible_space` |
| An **analysis with no banner** must be named by the caption, or the table is not usable as a positive. | Otherwise the target names something no part of the document does. | `build.build`, `trainset._reader_target` |
| A cell that is **blanked or dashed** loses its value from the target. | A dash stands where a number was expected. A target still asserting it teaches reading a value that is not there. | `build._rough_up` |
| A **cosmetic** mark does not change the target. | A footnote marker on a number leaves the number unchanged. | `build._rough_up` |

## 2. The table is messy, but never lying

Real tables carry markers, blanks, ragged rows, rowspans and packed cells, and
a model that only meets a structural cue in clean conditions learns it only in
clean conditions. The line is this: **mess may not move a coordinate from one
analysis to another, or invent one.**

* A row covered by a rowspan keeps the identity of the row above it; drawing a
  fresh region for it made the inherited label describe the wrong structure.
* A blanked row must keep its own first cell. An all-empty row vanishes when
  the grid renders, which shifts every following row one column along.
* A row with no coordinates takes its point out of the target with it.
* A table names one space. A `Talairach` column under an MNI caption is a
  contradiction no paper prints.
* A statistic header may not collide with an axis header: a transposed table
  headed `z` above a row headed `z (mm)` is ambiguous by construction.
* One form per table, and per column. A paper does not alternate between
  `(-42, -55, -18)` and `-42-55 -18`, or head the same quantity
  `Coordinates`, `Peak voxel` and `Location` across three contrasts.
* A region appears once. Two columns headed `Vermis` in one contrast hands the
  model a cue for the wrong reason.

## 3. A table that holds nothing gets the empty answer

The target is `{"space": null, "analyses": []}` — not a missing field and not
a refusal. Every negative carries exactly that, whatever produced it.

**But "no analyses" and "an analysis that found nothing" are different
answers, and the difference is the whole of this section.**

| the table | the target |
| --- | --- |
| a demographics table, a correlation matrix, a figure legend | `{"space": null, "analyses": []}` |
| `Previous down-regulation vs. look aversive` / `no significant results` | an analysis with that name and **no points** |

A contrast the paper ran and reports as `n.s.` is an analysis. The table names
it, so the document states that it was run, and a target that drops it says
the paper never looked. Five of roughly 120 real tables read by hand carry
one, and **no curated target in 10,015 has ever held one**, so this is a case
the model has never been taught.

* **The empty target keeps a null space** even when the table is headed
  `MNI-305`. A table stating no coordinates states no space for them, and the
  template-comparison negatives exist to teach answering nothing.
* **A negative still sits in a paper.** Giving the empty examples no abstract
  made 84.7% of the rows lacking one answerable without reading the table —
  a shortcut worth a quarter of the set.
* **Real negatives before generated ones.** A synthetic negative is a guess
  about what a non-coordinate table looks like; a real one is not. The 143
  tables read by eye go in first.

## 4. The prose belongs to this table

The prompt carries a title and an abstract. Both are composed from the table
in [`synth/context.py`](../src/nspond_tables/synth/context.py), and the module
docstring holds the measured rates.

* The abstract **names the contrasts the table is split on** and **spells out
  what the table abbreviates**: `DLPFC` in a row beside `dorsolateral
  prefrontal cortex` in the prose is the cross-reference the model must learn
  to follow.
* It is **not a description of the table**. Most of what it names is the rest
  of the paper.
* It **almost never settles the space** (1.2%): the normalisation is stated in
  the Methods, and the Methods are not in the prompt.
* It **does not quote an analysis label** more than 8.7% of the time. Quoting
  it makes the prose an exact-match key to the answer.
* The **title is drawn from the analyses**, not the caption. A caption's words
  are boilerplate: `Local maxima of significant clusters` yields "local".
* A **real article's own** title and abstract are used where the row has one.
  Composing is the last resort, not the first.

## 5. Nothing may be separable on anything but the table

The recurring failure is a shortcut: some field that answers the question
without reading the table. Each was found by measuring, not by review.

* the empty-abstract shortcut above;
* a caption reproducing the analysis name verbatim;
* a footer defining abbreviations the table never uses, which teaches that the
  footer is unrelated to the table;
* a column whose values are drawn per cell rather than per column, which
  teaches that a header constrains nothing.

**When you add a field, measure `P(target | field)` before you ship it.**

## 6. Determinism

`build(seed=n)` and `build_empty(seed=n)` return the same table every time,
and every function that needs randomness takes an `rng` rather than reaching
for a module-level one. A generator that cannot be replayed cannot be
bisected when a defect turns up in the training set.

## 7. Read the output

Most of the rules above came from reading samples, not from reasoning about
the code. The generator is checked by `tests/test_synth.py` and
`tests/test_context.py`, which pin rates and shapes — but a test only
catches what someone already thought of. Read tables after every change, and
keep reading until the only defects left are the ones you put there.
