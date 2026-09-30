# The shapes a coordinate table comes in

Every entry here was found by reading real tables, not by imagining what a
table might do. Each one names a shape, gives a real example, and says whether
the reader handles it and whether the generator makes it. A shape the
generator cannot make is a shape the model will meet cold.

The machine-readable judgements are in `judged-tables.json`: 249 tables read
by hand, 106 holding coordinates and 143 not.

## Shapes that hold coordinates

| shape | example | reader | generator |
| --- | --- | --- | --- |
| three named axis columns | `#Region \| #x \| #y \| #z` | yes | yes |
| one span, no axes beneath | `#<3:MNI coordinates` | yes | yes |
| a span headed with no axis word | `#<3:Location`, `#<3:Peak voxel` | yes | yes |
| an axis naming its kind | `X coor`, `y coordinate`, `Z (Talairach)` | yes | yes, 5.6% |
| the header written in `<td>` | nothing marked, axes on row 1 | yes | yes |
| only the second header row unmarked | `#<3:Individual ROIs` over `ROI \| x \| y \| z` | yes | yes, 17% |
| an axis wearing a marker | `X (mm)d`, `Peak (x, y, z)c` | yes | yes |
| the whole triple in one cell | `-42, -55, -18` | yes | yes |
| that triple in brackets | `(-42, -55, -18)` | yes | yes, 4.2% |
| signs run together | `-10-42 16`, `(34-34 72)` | yes | yes, 2.5% |
| semicolons between axes | `10.2; 20.5; 19.5` | yes | yes, 1.3% |
| several peaks in one cell | `-16, -54, 46; -22, -54, 52` | yes | yes |
| a statistic then its peak | `9.91 [3, 15, 51]`, `4.3 (44, -12, 6)` | yes | yes, 2.8% |
| a triple then a note | `-56 -30 28 (OP1)`, `12 -44 8 (BA 40)` | yes | yes, 1.8% |
| several peaks per row, split | `-8/12//-6 \| 56/48/52 \| 38/48/30` | yes | no |
| a triple column per region | `Target \| -16 -100 -8 \| 18 -96 2` | yes | yes, 0.9% |
| analyses as column groups | `#<4:TD group \| #<4:ASD group` | no, by design | yes, 15% |
| coordinates along rows | `z \| 4 \| 4 \| 2` | no | yes, 2.8% |
| an ROI centroid in the header | `#<2:Left DLPFC (-45, 15, 35)` | no | yes, 3.8% |
| a bilateral pair | `Visual: BA 17, 19 (±30 -80 6)` | no | yes, 1.8% |
| RAS instead of xyz | `#Peak Z \| #R \| #A \| #S` | no | yes, 1.3% |

## Shapes that do not, and are mistaken for them

The reader is meant to over-generate and the gate to reject, so most of these
are read and then thrown out. They are the negatives worth generating.

| shape | example | why it reads as a coordinate |
| --- | --- | --- |
| an estimate and its interval | `15 (9.3, 24.4)`, `1.5 (0.3-7.8)` | three numbers, all in head bounds |
| an odds ratio with its CI | `0.31(0.11,0.86)` | the same |
| an effect size in brackets | `0.63 [0.41, 0.84]` | the same |
| an F with its degrees of freedom | `F(1,67) = 28.05` | the same |
| a median with its range | `74.11 (72.85, 75.37)` | the same |
| a count triple | `Handedness \| 39, 6, 1` | the same |
| mean, SD and a range | `35.56, 8.11, 21-49` | the same |
| scientific notation | `3.6 x 10-6` | a minus that follows a digit |
| voxel or matrix dimensions | `62 x 48 x 58`, `2, 2, 2` | three small integers |
| genomic positions | `Chr6 \| 168,336,080 \| 168,597,552` | large integers, and the word position |
| equilibrium points | `E1 (0,0,0)`, `E2 (1,0,0)` | a bracketed triple |
| a figure legend as a table | `View larger version(87K) \| Figure 5. ...` | coordinate words in the prose |
| a supplementary file listing | `suppinfofigure1.tif \| 12372K \| ...` | anatomy in the descriptions |
| a template comparison | `AC-PC length [mm] \| MNI-305 \| ICBM-152` | MNI and millimetres throughout |
| an activation table with no peaks | `Region \| Side \| Voxels \| T \| p` | every word of a coordinate table |

Every negative shape above is generated. The percentages in the positive
table are of all generated tables, measured over 3,000 seeds, and most of them
sit over the corpus rate on purpose -- see the note below.

## What a shape costs when it is missing

`X (mm)d` is 3 headers in 6,897, and it lost a whole table. Rarity is not the
same as unimportance: the reader either handles a shape or the article is
gone. That is why several rates in `synth/weights.py` are deliberately over
the real ones.
