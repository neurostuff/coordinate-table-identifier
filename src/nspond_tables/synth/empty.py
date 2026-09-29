"""Tables that hold nothing to extract, and the empty answer they deserve.

No version before v19 has been shown a table whose correct answer is nothing,
which is why the model invents an analysis when handed a demographics table.
The target here is the empty structure -- `analyses: []` -- not a missing
field, not a refusal, and not an analysis with an empty point list.

Three kinds, in rising order of how hard they are to tell from a real one:

* **ordinary** -- an ANOVA effect table, MRI acquisition parameters,
  demographics, a correlation matrix, a ROC curve. Nothing about them looks
  like coordinates except that they are tables of numbers from the same
  papers.
* **triple** -- a table whose cells parse as coordinate triples but are not:
  a median with its range, an odds ratio with its interval, an F with its
  degrees of freedom. `Patients | 24 | 34.2 | 12` trips a triple detector
  exactly as an activation row does, and among the tables the reader reads
  something out of, this shape accounts for 75 of the 87 that hold no
  coordinates.
* **near miss** -- the three that beat the gate at 0.86 or better when the
  uncertain band was read by hand: a brain-template comparison full of MNI
  names and millimetres, an activation table that reports regions and t values
  and no coordinates at all, and a supplementary-file listing whose
  descriptions are thick with anatomy.

They are built from the same vocabulary and the same layout machinery as the
positives -- spans, banner rows, sub-headers -- so the model cannot separate
them on shape.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

from ..grid import Cell, Grid
from . import vocab
from .weights import DEFAULT, Weights

KINDS = ("ordinary", "triple", "near_miss")

_ORDINARY = (
    ("anova", "Effects of {a} and {b} on {m}",
     ["Effect", "F", "df", "p", "partial eta2"]),
    ("acquisition", "MRI acquisition parameters",
     ["Parameter", "T1-weighted", "T2-weighted", "DWI"]),
    ("demographics", "Participant characteristics by group",
     ["Measure", "{g1} (n = {n1})", "{g2} (n = {n2})", "p"]),
    ("correlation", "Correlations between {m} and behavioural measures",
     ["Measure", "1", "2", "3", "4"]),
    ("roc", "Receiver operating characteristic analysis",
     ["Variable", "AUC (95% CI)", "p", "Sensitivity (%)", "Specificity (%)"]),
    ("studies", "Studies included in the meta-analysis",
     ["Study", "Year", "n", "Task", "Modality"]),
)

_NEAR_MISS = (
    ("template", "Global brain features of the {t} template against existing templates",
     ["Parameter", "MNI-305", "ICBM-152", "ICBM-452", "CBA-56"]),
    ("activation_no_coords", "Cortical patterns of activation during {c}",
     ["Region", "Side", "Voxels", "T-value", "p-value"]),
    ("file_listing", "",
     ["Filename", "Format", "Size", "Description"]),
)

_TEMPLATE_ROWS = ["AC-PC length [mm]", "Length (L) [mm]", "Width (W) [mm]",
                  "Height (H) [mm]", "W/L", "H/L", "Brain volume [cm3]"]
_ACQ_ROWS = ["TR (ms)", "TE (ms)", "Flip angle", "Slice thickness (mm)",
             "Field of view (mm)", "Matrix", "Bandwidth (Hz/pixel)", "Slices"]
_DEMO_ROWS = ["Age (years)", "Sex (M:F)", "Education (years)", "Handedness (R:L)",
              "IQ", "Symptom score", "Medication (n)", "Illness duration (years)"]
_MEASURES = ["reaction time", "accuracy", "working memory span", "cortical thickness",
             "grey matter volume", "fractional anisotropy"]

# Which tables plausibly print an estimate beside its interval. A flip angle
# does not come with a confidence interval, and a table that pretends otherwise
# teaches the model nothing except that the generator is careless.
_TAKES_AN_INTERVAL = ("anova", "demographics", "correlation", "roc")

# A banner has to name something the table is actually divided by.
_BANNERS = {
    "anova": ["Within-subject effects", "Between-subject effects"],
    "demographics": ["Clinical", "Demographic", "Cognitive"],
    "correlation": ["Behavioural measures", "Imaging measures"],
    "roc": ["Univariable", "Multivariable"],
    "acquisition": ["Structural", "Functional", "Diffusion"],
    "studies": ["Task-based", "Resting state"],
    "template": ["Global measures", "Regional measures"],
    "activation_no_coords": ["Controls", "Patients"],
    "file_listing": ["Figures", "Videos"],
}


# Which interval form each kind of table actually prints. An age does not come
# with degrees of freedom, and an ANOVA effect does not come with an IQR.
#: Where each kind of table prints an estimate beside its interval, and in
#: what form. One column, the one whose header says so: an odds ratio in an
#: `AUC (95% CI)` column or an interquartile range in a `p` column is not a
#: near miss, it is a mistake.
_INTERVAL_FORMS = {
    "anova": ("f",), "demographics": ("iqr",), "correlation": ("bracket",),
    "roc": ("auc", "or"),
}
_INTERVAL_COLUMN = {"anova": (1, "f"), "demographics": (1, "iqr"),
                    "correlation": (2, "bracket"), "roc": (1, "auc")}


def _stat_with_interval(rng: random.Random, flavour: str = "roc",
                        form: str = "") -> str:
    """A number and its interval, which reads as a coordinate triple.

    `15 (9.3, 24.4)`, `0.31(0.11,0.86)`, `0.63 [0.41, 0.84]` and
    `F(1,67) = 28.05` all parse as three numbers inside a head. This is the
    single shape behind 75 of the 87 candidate tables that hold none.
    """
    form = form or rng.choice(_INTERVAL_FORMS.get(flavour, ("iqr",)))
    if form == "iqr":
        v = round(rng.uniform(1, 80), 1)
        return "%s (%s, %s)" % (v, round(v * 0.6, 1), round(v * 1.5, 1))
    if form == "or":
        v = round(rng.uniform(0.05, 4.0), 2)
        return "%s(%s,%s)" % (v, round(v * 0.35, 2), round(v * 2.8, 2))
    if form == "bracket":
        v = round(rng.uniform(-0.8, 0.9), 2)
        return "%s [%s, %s]" % (v, round(v - 0.2, 2), round(v + 0.2, 2))
    if form == "auc":
        v = round(rng.uniform(0.55, 0.95), 2)
        return "%s (%s-%s)" % (v, round(v - 0.11, 2), round(v + 0.06, 2))
    return "F(%d,%d) = %s" % (rng.randint(1, 4), rng.randint(20, 120),
                              round(rng.uniform(2, 40), 2))


#: What a column holds, by what its header says. A near-miss table is only
#: hard if its numbers belong under its headers: a `p` column holding 108 and
#: a `df` column holding 0.174 are dismissed at a glance, and worse, they
#: teach that a header does not constrain what sits beneath it.
_BY_HEADER = (
    (re.compile(r"\bp[-\s]?value|^p$|^sig", re.I),
     lambda r: "%.3f" % r.uniform(0.0001, 0.9)),
    (re.compile(r"^df$|degrees of freedom", re.I),
     lambda r: "%d,%d" % (r.randint(1, 4), r.randint(18, 120))),
    (re.compile(r"^F$|F[-\s]?value|F[-\s]?stat", re.I),
     lambda r: "%.2f" % r.uniform(0.3, 42)),
    (re.compile(r"eta2|η2|cohen|effect size", re.I),
     lambda r: "%.3f" % r.uniform(0.005, 0.62)),
    (re.compile(r"\bAUC\b", re.I),
     lambda r: "%.2f (%.2f-%.2f)" % ((lambda v: (v, v - 0.11, v + 0.06))(
         round(r.uniform(0.55, 0.95), 2)))),
    (re.compile(r"sensitivity|specificity|\bPPV\b|\bNPV\b|%\)|percent", re.I),
     lambda r: "%.1f" % r.uniform(38, 99)),
    (re.compile(r"citations|documents|^n$|\bcount\b|number", re.I),
     lambda r: str(r.randint(3, 480))),
    (re.compile(r"\byear\b|^20\d\d$", re.I),
     lambda r: str(r.randint(1998, 2025))),
    (re.compile(r"\[mm\]|\(mm\)|length|width|height", re.I),
     lambda r: "%.2f" % r.uniform(18, 185)),
    (re.compile(r"W/L|H/L|ratio", re.I),
     lambda r: "%.2f" % r.uniform(0.55, 0.95)),
    (re.compile(r"\bTR\b|\bTE\b|flip|thickness|field of view|bandwidth|matrix",
                re.I),
     lambda r: r.choice(["%.1f" % r.uniform(0.9, 9.5), str(r.randint(20, 4000)),
                         "%d x %d" % ((r.choice([64, 128, 256]),) * 2)])),
)


def _plain(rng: random.Random, header: str = "", label: str = "") -> str:
    """A value that belongs under `header`, or beside `label`.

    A template comparison puts the unit in the row label -- `AC-PC length
    [mm]` -- and the template's name in the header, so the column says nothing
    about what belongs in it and the row says everything.
    """
    for pattern, make in _BY_HEADER:
        if pattern.search(header or "") or pattern.search(label or ""):
            return make(rng)
    return rng.choice(["%.2f" % rng.uniform(0, 5), "%.3f" % rng.uniform(0, 1),
                       str(rng.randint(2, 240)),
                       "%.1f \u00b1 %.1f" % (rng.uniform(20, 60), rng.uniform(1, 12))])


def _caption_and_footer(rng: random.Random, kind: str, name: str, title: str,
                        w: Weights) -> Tuple[str, str]:
    caption = ""
    if title and rng.random() < w.caption_present:
        caption = "Table %d. %s." % (rng.randint(1, 8), title.format(
            a=rng.choice(vocab.CONDITIONS), b=rng.choice(vocab.GROUPS),
            m=rng.choice(_MEASURES), c=rng.choice(vocab.CONDITIONS),
            t=rng.choice(["BRAHMA", "SCBT-2020", "NIHPD"]),
            g1=rng.choice(vocab.GROUPS), g2=rng.choice(vocab.GROUPS),
            n1=rng.randint(12, 60), n2=rng.randint(12, 60)))
    footer = ""
    if rng.random() < w.footer_present:
        footer = rng.choice([
            "Values are mean ± SD unless stated otherwise.",
            "L: left; R: right.",
            "* p < 0.05, ** p < 0.01.",
            "Abbreviations: FA, fractional anisotropy; MD, mean diffusivity.",
            "The statistical threshold was set at p<0.05 (FWE-corrected).",
        ])[:w.footer_chars]
    return caption, footer


def _rows_for(rng: random.Random, name: str, headers: List[str],
              triples: bool) -> List[List[str]]:
    width = len(headers)
    if name == "template":
        labels = _TEMPLATE_ROWS
    elif name == "acquisition":
        labels = _ACQ_ROWS
    elif name == "demographics":
        labels = _DEMO_ROWS
    elif name == "activation_no_coords":
        labels = [("%s %s" % (rng.choice(["L", "R"]), r.name))
                  for r in rng.sample(vocab.REGIONS, min(8, len(vocab.REGIONS)))]
    elif name == "file_listing":
        labels = ["suppinfofigure%d.tif" % i for i in range(1, 8)]
    elif name == "correlation":
        labels = _MEASURES
    else:
        labels = rng.sample(vocab.EFFECTS, min(6, len(vocab.EFFECTS)))

    # One form per column, not per cell: a column headed `p` does not alternate
    # between an odds ratio and an interquartile range down its length. And a
    # column whose header names what it holds keeps holding that: an `F` column
    # is not a place to print an interquartile range.
    forms = [rng.choice(_INTERVAL_FORMS.get(name, ("iqr",)))
             for _ in range(width)]
    # One column carries the interval, the way a real table has an `OR (95%
    # CI)` among ordinary ones. Filling every column with it made the table
    # obviously synthetic, and filling none -- which is what happened once the
    # headers started constraining their columns -- left the `triple` kind
    # with no triples in it at all.
    carries, carried_form = _INTERVAL_COLUMN.get(name, (0, "iqr"))
    if carries >= width:
        carries = 0
    out = []
    for label in labels[:rng.randint(4, 8)]:
        if name == "file_listing":
            out.append([label, "", "%dK" % rng.randint(900, 14000),
                        "Supplemental figure showing %s parcellation of the %s"
                        % (rng.choice(["Brodmann", "AAL", "Desikan"]),
                           rng.choice(vocab.REGIONS).name)])
            continue
        cells = [label]
        for col in range(1, width):
            head = headers[col] if col < len(headers) else ""
            if triples and col == carries:
                cells.append(_stat_with_interval(rng, name, carried_form))
            else:
                cells.append(_plain(rng, head, label))
        out.append(cells)
    return out


def build_empty(seed: int = 0, weights: Weights = DEFAULT,
                kind: Optional[str] = None):
    """One table that holds no coordinates, and the empty target it deserves.

    Returns the same `Table` the positive generator returns, so the two can be
    shuffled into one training set without special-casing either.
    """
    from .build import Table, Truth          # noqa: PLC0415  (circular at module level)

    rng = random.Random(seed)
    w = weights
    if kind is None:
        kind = ("near_miss" if rng.random() < w.near_miss
                else rng.choice(("ordinary", "triple")))
    if kind == "near_miss":
        pool = _NEAR_MISS
    elif kind == "triple":
        # An interval belongs beside an estimate, not beside a flip angle.
        pool = [t for t in _ORDINARY if t[0] in _TAKES_AN_INTERVAL]
    else:
        pool = _ORDINARY
    name, title, headers = rng.choice(pool)

    grid = Grid()
    g1, g2 = rng.sample(vocab.GROUPS, 2)
    grid.add([Cell(text=h.format(g1=g1, g2=g2, n1=rng.randint(12, 60),
                                 n2=rng.randint(12, 60)), header=True)
              for h in headers])

    body = _rows_for(rng, name, headers, triples=(kind == "triple"))
    # A banner row, so the shape matches a multi-analysis coordinate table and
    # the model cannot use a divider as evidence that coordinates follow.
    if rng.random() < 0.4 and len(body) > 2:
        cut = rng.randint(1, len(body) - 1)
        body.insert(cut, None)
    for row in body:
        if row is None:
            grid.add([Cell(text=rng.choice(_BANNERS[name]),
                           colspan=len(headers))])
        else:
            grid.add([Cell(text=c) for c in row])

    caption, footer = _caption_and_footer(rng, kind, name, title, w)
    return Table(grid=grid, truth=Truth(space=None, analyses=[]),
                 caption=caption, footer=footer,
                 notes={"empty": True, "kind": kind, "flavour": name,
                        "layout": list(headers)})
