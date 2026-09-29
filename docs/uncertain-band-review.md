# Hand review of 50 uncertain tables

The coordinate gate was fitted on labels a heuristic produced, so its held-out
score measured whether a cheap linear model reproduces that heuristic — not
whether it decides correctly where the heuristic would not commit. This is that
second question, answered by reading 50 tables.

**Sample.** The 591 tables `detect` returned UNCERTAIN for, sorted by gate score
and sampled five from each of ten score bands. Stratified rather than random,
because a random draw from a band that is 87% pass tells you least about the
boundary, which is the only place a threshold matters. Scores span 0.021 to
1.000. Every table was read with its caption and first rows; the judgment is
"does this table report coordinates", not "would the model find them".

## What the 50 are

| | count |
|---|---|
| hold coordinates | 24 |
| hold coordinates but the serialised form is too damaged to read them | 2 |
| do not hold coordinates | 24 |

So the uncertain band is close to an even split — it is genuinely the hard
population, not a mislabelled easy one.

## How the gate did

| threshold | tp | fp | fn | tn | precision | recall | |
|---|---|---|---|---|---|---|---|
| 0.055 | 26 | 19 | 0 | 5 | 57.8% | **100.0%** | deployed (0.90 precision floor) |
| 0.146 | 26 | 10 | 0 | 14 | 72.2% | **100.0%** | 0.95 floor |
| 0.269 | 26 | 7 | 0 | 17 | 78.8% | **100.0%** | lowest-scoring positive |
| 0.500 | 24 | 3 | 2 | 21 | 88.9% | 92.3% | |

**Recall is 100% at the deployed threshold.** Not one of the 26 coordinate
tables was dropped, including the two whose serialisation is wrecked. For a gate
whose whole purpose is that a miss loses the article permanently, that is the
number that mattered, and it held.

**Precision is 57.8%, against the 90% the fit promised.** That is not a broken
model — it is the band. `detect` refused to commit on these precisely because
they carry mixed evidence, so the negatives here are the hardest negatives in
the corpus. Precision measured on the labelled set does not transfer to the
band the gate is actually for.

### The threshold is set too low

The lowest-scoring true positive is **0.269**. Every negative below that is a
free rejection:

> Moving the threshold from 0.055 to 0.269 loses nothing and removes 12 of the
> 19 false positives — precision 57.8% → 78.8%, recall unchanged at 100%.

The floor-based rule picked 0.055 from the labelled set, where scores separate
cleanly and the lowest positive sits much lower. Re-fitting the threshold on
reviewed uncertain tables rather than on heuristic labels is the single change
this review argues for.

Above 0.269 there is no clean cut: positives and negatives overlap between
**0.269 and 0.904**, ten tables deep. Pushing past it starts costing real
tables — at 0.50, two coordinate tables are dropped to gain 10 points of
precision, which is the wrong trade here.

## What the hard cases look like

The false positives are not noise. They are tables that genuinely resemble
coordinate tables:

- **#27, score 0.904** — scanner acquisition parameters: matrix size, resolution,
  TR, b-value, per subject. Many numeric columns, many rows, an imaging
  vocabulary. The worst false positive in the sample.
- **#24, score 0.558** — segmentation ROI volumes in mm³ beside region names
  ("Left accumbens area", "Left amygdala"). Region words and three numeric
  columns; only the absence of an axis header separates it.
- **#26, score 0.811** — seed-to-voxel functional connectivity with region
  names, t values and p-FDR, but no coordinates reported at all.
- **#9, score 0.073** — names Left NAcc, Right NAcc and Thalamus, holds
  correlation coefficients. Correctly dropped, and a good sign: region
  vocabulary alone did not carry it.

The common shape is *a real neuroimaging results table that reports something
other than coordinates*. A model trained on this band would need to learn that
distinction, not the coarse one.

## Findings beyond the gate

**Transposed coordinate tables exist and nothing handles them.** #30 and #38 lay
their coordinates along rows rather than columns — a row labelled `z` followed
by z values, a row labelled `y` followed by y values. `read.extract` looks for
an axis header and reads down a column, so it finds nothing. The gate scored
both above 0.92 on other evidence, so they are not lost, but the deterministic
reader cannot use them.

**One table collapsed to a single serialised row.** #33 renders as one line
beginning `Table 1Identification of BOLD Signal Increases during Encoding Tasks
versus Fixation...` — caption and contents concatenated with no cell structure
at all. Worth tracing: a whole table's coordinates are unreachable.

**Two pdf tables are readable as images but not as text.** #19 and #38 have
their columns interleaved by the pdf conversion. These were extracted before the
HTML switch; both should be re-checked against the re-extracted artifacts.

**Detached minus signs and packed triples both appear in the wild**, confirming
two fixes already made: #48 writes x as `- 10`, and #47 packs coordinates as
`[6 9 -9]` in one cell.

## Source skew

| source | hold coordinates | do not |
|---|---|---|
| ace | 15 | 1 |
| pdf | 7 | 3 |
| pubget | 2 | 8 |
| elsevier | 2 | 12 |

ace's uncertain tables are almost all coordinate tables; elsevier's and
pubget's are mostly not. That is consistent with ace being a coordinate-focused
scraper whose article set is already enriched, and it suggests the gate could
take source as a feature — though doing so would bake in a sampling artefact
rather than a property of papers, so I have not.

## What I would do next

1. **Re-fit the threshold on reviewed tables, not heuristic labels.** 0.269 is
   free. This review is 50 tables; another 50 would set it with more confidence.
2. **Handle transposed tables in `read.extract`** — look for an axis name in the
   first *column* as well as the header, then read across.
3. **Trace the collapsed table (#33)** back to its raw source and find out which
   path produced a single row.
4. **Leave the feature set alone.** It is doing the work: the positives it ranks
   highest are unambiguous, the negatives it ranks lowest are unambiguous, and
   the overlap band is genuinely ambiguous rather than carelessly scored.

## The table

`pass`/`drop` is at the deployed threshold of 0.055. `yes*` means the table
holds coordinates but its serialised form is too damaged to read them.

| # | score | source | coords? | gate | caption | note |
|---|---|---|---|---|---|---|
| 1 | 0.021 | pubget | no | drop | Contrast-Enhanced Cardiac MRI Findings During Index Hospit |  |
| 2 | 0.025 | pubget | no | drop | MAE values between pseudo-CT and reference-CT for the over |  |
| 3 | 0.038 | ace | no | drop | (TUS group) The mean ADC values pre- and post-TUS in diffe |  |
| 4 | 0.040 | elsevier | no | drop | Results of the 4-way ANOVA performed on MTR values in 30 t |  |
| 5 | 0.040 | elsevier | no | drop | Estimating the agreement between radiologist A and F using |  |
| 6 | 0.068 | elsevier | no | pass | The number of small vessels and micro-hemorrhage volume sh |  |
| 7 | 0.070 | pdf | no | pass | Table 1. Normalized EEG discriminator amplitudes used to m |  |
| 8 | 0.071 | pubget | no | pass | Rotation of the parcellation (adult 4). |  |
| 9 | 0.073 | pubget | no | pass | Correlations between questionnaire scores and brain activa | names Left/Right NAcc and Thalamus but holds correlation coefficients |
| 10 | 0.085 | elsevier | no | pass | Estimating the agreement between MRA-2D and IA using CIAs  |  |
| 11 | 0.110 | pubget | no | pass | Estimated mean CoV across the five random NH subjects for  |  |
| 12 | 0.111 | elsevier | no | pass | The correlation analysis of the left, the right and the to |  |
| 13 | 0.133 | elsevier | no | pass | Characteristics of the 3 Patients Whose Magnetic Resonance |  |
| 14 | 0.139 | pdf | no | pass | Table 2. Neuropsychological test results, salivary cortiso |  |
| 15 | 0.160 | elsevier | no | pass | Characteristics of the patients |  |
| 16 | 0.181 | pdf | no | pass | Table 1. Response patterns for incongruent trials |  |
| 17 | 0.191 | pubget | no | pass | Review of some main contributions on the relationship betw |  |
| 18 | 0.269 | ace | yes | pass | Brain regions showing significantly correlation between FC |  |
| 19 | 0.270 | pdf | yes* | pass | Table 3. Areas showing activity in the contrast of the par | activation table, but the pdf conversion destroyed the columns |
| 20 | 0.299 | elsevier | no | pass | Craving scores and CO levels in both sessions as well as t |  |
| 21 | 0.352 | elsevier | no | pass | Pathological Findings |  |
| 22 | 0.403 | pubget | no | pass | MRS Sequence Protocol |  |
| 23 | 0.487 | pubget | no | pass | T and N staging of all colorectal cancers |  |
| 24 | 0.558 | elsevier | no | pass | A comparison of the average segmentation volumes of MRI sc | ROI volumes in mm3 beside region names -- structurally very close to a coordinate table |
| 25 | 0.632 | pdf | yes | pass | Table 1. Regions showing repetition suppression for goal |  |
| 26 | 0.811 | elsevier | no | pass | Post-hoc analysis of seed to voxel functional connectivity | seed-to-voxel FC with region names and statistics, but no coordinates reported |
| 27 | 0.904 | elsevier | no | pass | _(no caption)_ | scanner acquisition parameters; the worst false positive in the sample |
| 28 | 0.906 | ace | yes | pass | Meta analysis results for anatomical regions with differen |  |
| 29 | 0.918 | ace | yes | pass | Activation likelihood estimation results for facial emotio |  |
| 30 | 0.929 | pdf | yes | pass | _(no caption)_ | coordinates run along ROWS, not columns -- a transposed table |
| 31 | 0.968 | pdf | yes | pass | Table 1. Significant main effect areas of activation for d |  |
| 32 | 0.970 | pdf | yes | pass | Table4 Functionalconnectivityanalysis |  |
| 33 | 0.981 | elsevier | yes* | pass | Identification of BOLD Signal Increases during Encoding Ta | the whole table collapsed into a single serialised row |
| 34 | 0.982 | ace | yes | pass | Anatomical regions with significant white matter volume ch |  |
| 35 | 0.984 | pdf | yes | pass | Table 2: Regions of face-voice integration according to: a |  |
| 36 | 0.988 | elsevier | yes | pass | Think/No-Think Task fMRI results. |  |
| 37 | 0.991 | pubget | yes | pass | Regional Activations while perceiving an odor: Smelled sti |  |
| 38 | 0.996 | pdf | yes | pass | _(no caption)_ | transposed, and the pdf conversion interleaved the columns |
| 39 | 0.996 | ace | yes | pass | _(no caption)_ |  |
| 40 | 0.997 | ace | yes | pass | _(no caption)_ |  |
| 41 | 0.997 | pubget | yes | pass | Maintenance-related neural distinctiveness. |  |
| 42 | 0.997 | ace | yes | pass | _(no caption)_ |  |
| 43 | 0.997 | ace | yes | pass | Results of the whole brain analysis for different spatial  |  |
| 44 | 0.998 | ace | yes | pass | . Coordinates of the brain regions with significant differ |  |
| 45 | 0.999 | ace | yes | pass | _(no caption)_ |  |
| 46 | 1.000 | ace | yes | pass | . Brain regions showing the effect of maternal depressive  |  |
| 47 | 1.000 | ace | yes | pass | Effects of silent music reading expertise (singers > contr | coordinates packed as [6 9 -9] in one cell |
| 48 | 1.000 | ace | yes | pass | Region activated for an interaction between Proportion and | x written as '- 10', the detached minus |
| 49 | 1.000 | ace | yes | pass | Cluster maxima of the functional connectivity analysis wit |  |
| 50 | 1.000 | ace | yes | pass | Choice phase: 2×2 ANOVA results. |  |
---

Sample and scores reproducible with `evalkit/sample_uncertain.py` against
`train-data/coord_labels.jsonl` and `train-data/coord_gate.json`. The judgments
are mine, recorded in `docs/uncertain-band-judgments.json`.
