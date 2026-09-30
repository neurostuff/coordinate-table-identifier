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
    # Large integers under a column headed `Position`, and a p written
    # `3.6 x 10-6` whose minus follows a digit. Both read as coordinates.
    ("genetics", "Genome-wide association results for {m}",
     ["SNP", "Chr", "Position", "Nearest gene", "Beta (SE)", "p"]),
    # Four of 98 hand-read residual tables were one of these. A neural-mass or
    # haemodynamic model prints Greek letters beside numbers, and nothing
    # about it is a place.
    ("parameters", "Prior expectations of model parameters",
     ["Parameter", "Physiological interpretation", "Value", "Units"]),
    # An index from ROI number to lobe. It names every region a coordinate
    # table names and locates none of them.
    ("roi_index", "Index of ROI numbers",
     ["ROI number", "ROI location", "Volume (mm3)"]),
)

_NEAR_MISS = (
    ("template", "Global brain features of the {t} template against existing templates",
     ["Parameter", "MNI-305", "ICBM-152", "ICBM-452", "CBA-56"]),
    ("activation_no_coords", "Cortical patterns of activation during {c}",
     ["Region", "Side", "Voxels", "T-value", "p-value"]),
    ("file_listing", "",
     ["Filename", "Format", "Size", "Description"]),
    # A bracketed triple beside a label, which is the shape of a packed peak
    # and none of the numbers is one.
    ("equilibrium", "Equilibrium points of the {d} model and their stability",
     ["Point", "Coordinates", "Eigenvalues", "Stability"]),
    # Prose with a figure number in front of it. 11.5% of the tables the old
    # filter kept and triage drops are these.
    ("figure_legend", "",
     ["", "Figure"]),
    # Three of 98. A spatial scan statistic over a country prints coordinates
    # that are latitudes and longitudes, under a header reading `Coordinate`.
    ("scan_statistic", "Significant clusters of {c} detected by a spatial scan",
     ["Cluster type", "Enumeration areas", "Coordinate / radius", "Population",
      "Cases", "RR", "P-value"]),
    # Mask overlap in cubic millimetres. Everything is a volume and a Dice
    # index; the gate accepted one of these.
    ("mask_overlap", "Overlap between somatotopic masks and the peak clusters",
     ["Anatomic area", "Somatotopic label", "Mask volume (mm3)",
      "Overlap volume (mm3)", "Sorensen-Dice"]),
)

_TEMPLATE_ROWS = ["AC-PC length [mm]", "Length (L) [mm]", "Width (W) [mm]",
                  "Height (H) [mm]", "W/L", "H/L", "Brain volume [cm3]"]
_ACQ_ROWS = ["TR (ms)", "TE (ms)", "Flip angle", "Slice thickness (mm)",
             "Field of view (mm)", "Matrix", "Bandwidth (Hz/pixel)", "Slices"]
_DEMO_ROWS = ["Age (years)", "Sex (M:F)", "Education (years)", "Handedness (R:L)",
              "IQ", "Symptom score", "Medication (n)", "Illness duration (years)"]
#: What a neural-mass or haemodynamic model prints. Greek letters beside
#: numbers, and not one of them is a place.
_PARAMETERS = (
    ("\u03ba\u2091, \u03ba\u1d62", "Postsynaptic time constants", "1/4, 1/28", "ms-1"),
    ("\u03b1\u2081\u2083, \u03b1\u2082\u2083", "Amplitude of intrinsic connectivity kernels",
     "2000, 8000", "-"),
    ("c\u1d62\u2c7c", "Intrinsic connectivity decay constant", "0.32", "mm-1"),
    ("r, \u03b7, g", "Sigmoid parameters", "0.54, 0, 0.135", "-"),
    ("s\u1d62\u2c7c", "Conduction velocity", "3", "m/s"),
    ("\u03c4\u1d62", "Haemodynamic transit time", "0.98", "s"),
    ("\u03b1", "Grubb's exponent", "0.32", "-"),
    ("E\u2080", "Resting oxygen extraction fraction", "0.34", "-"),
    ("V\u2080", "Resting blood volume fraction", "0.02", "-"),
    ("\u03b3\u1d62", "Rate of flow-dependent elimination", "0.41", "s-2"),
    ("m\u2091, m\u1d62", "Maximum postsynaptic depolarization", "8, 32", "mV"),
    ("H\u2091", "Maximum post-synaptic potential", "4", "mV"),
)

_LOBES = ["frontal lobe", "parietal cortex", "somatosensory cortex",
          "motor cortex", "visual cortex", "occipital lobe", "temporal lobe",
          "cingulate cortex", "insula", "cerebellum"]

_MEASURES = ["reaction time", "accuracy", "working memory span", "cortical thickness",
             "grey matter volume", "fractional anisotropy"]

# Which tables plausibly print an estimate beside its interval. A flip angle
# does not come with a confidence interval, and a table that pretends otherwise
# teaches the model nothing except that the generator is careless.
#: Which demographics rows can carry which form. `Sex (M:F)` is a count and
#: `Age (years)` is a measurement; swapping them prints an age as three
#: integers, which is the generator inventing a shape no paper has.
_COUNTED = re.compile(r"\(M:F\)|\(R:L\)|\(n\)|^Sex|handedness", re.I)

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
    "genetics": ["Discovery sample", "Replication sample"],
    "equilibrium": ["Trivial", "Non-trivial"],
    "figure_legend": ["Main figures", "Supplementary figures"],
    "parameters": ["Domain and indices", "Model", "Observation"],
    "roi_index": ["Left hemisphere", "Right hemisphere"],
    "scan_statistic": ["Primary", "Secondary"],
    "mask_overlap": ["Motor (precentral gyrus)", "Somatosensory (postcentral gyrus)"],
}


# Which interval form each kind of table actually prints. An age does not come
# with degrees of freedom, and an ANOVA effect does not come with an IQR.
#: Where each kind of table prints an estimate beside its interval, and in
#: what form. One column, the one whose header says so: an odds ratio in an
#: `AUC (95% CI)` column or an interquartile range in a `p` column is not a
#: near miss, it is a mistake.
_INTERVAL_FORMS = {
    "anova": ("f",), "demographics": ("iqr", "counts", "mean_sd_range"),
    "correlation": ("bracket",), "roc": ("auc", "or"),
    "genetics": ("beta",),
}
_INTERVAL_COLUMN = {"anova": (1, ("f",)),
                    "demographics": (1, ("iqr", "iqr", "counts",
                                        "mean_sd_range")),
                    "correlation": (2, ("bracket",)), "roc": (1, ("auc", "or")),
                    "genetics": (4, ("beta",))}


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
    if form == "counts":
        # `Handedness | 39, 6, 1`. Three small integers with commas between
        # them and nothing to say they are not a peak.
        return "%d, %d, %d" % (rng.randint(8, 90), rng.randint(1, 30),
                               rng.randint(0, 12))
    if form == "mean_sd_range":
        m = round(rng.uniform(18, 70), 2)
        return "%s, %s, %d-%d" % (m, round(rng.uniform(1, 12), 2),
                                  int(m * 0.6), int(m * 1.4))
    if form == "beta":
        return "%.3f (%.3f)" % (rng.uniform(-0.6, 0.6), rng.uniform(0.01, 0.2))
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
    (re.compile(r"eigenvalue", re.I),
     lambda r: "%.2f, %.2f" % (r.uniform(-3, 1), r.uniform(-3, 1))),
    (re.compile(r"stability|stable", re.I),
     lambda r: r.choice(["stable node", "saddle", "unstable focus",
                         "stable focus"])),
    (re.compile(r"^df$|degrees of freedom", re.I),
     lambda r: "%d,%d" % (r.randint(1, 4), r.randint(18, 120))),
    (re.compile(r"^F$|F[-\s]?value|F[-\s]?stat", re.I),
     lambda r: "%.2f" % r.uniform(0.3, 42)),
    # A statistic is one number. `T-value | 27.1 +/- 3.2` is a mean with a
    # standard deviation wearing a statistic's header.
    (re.compile(r"^[TZtz]$|[TZ][-\s]?value|[TZ][-\s]?stat|^t\(\d", re.I),
     lambda r: "%.2f" % r.uniform(0.4, 9.5)),
    (re.compile(r"eta2|η2|cohen|effect size", re.I),
     lambda r: "%.3f" % r.uniform(0.005, 0.62)),
    (re.compile(r"\bAUC\b", re.I),
     lambda r: "%.2f (%.2f-%.2f)" % ((lambda v: (v, v - 0.11, v + 0.06))(
         round(r.uniform(0.55, 0.95), 2)))),
    (re.compile(r"sensitivity|specificity|\bPPV\b|\bNPV\b|%\)|percent", re.I),
     lambda r: "%.1f" % r.uniform(38, 99)),
    # A `Side` column holding 4.10 and a `Voxels` column holding 29.3 +/- 9.3
    # make the table separable on nonsense rather than on what it holds, which
    # is the same shortcut as any other.
    (re.compile(r"^side$|hemisphere|^hemi|^lat\.?$|^L/R$|^H$", re.I),
     lambda r: r.choice(["L", "R", "Left", "Right", "B"])),
    (re.compile(r"voxels|^k$|extent|cluster size|^size$|mm3|mm\^3", re.I),
     lambda r: str(r.choice([8, 14, 22, 31, 48, 76, 120, 210, 380, 640, 1180]))),
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
     # `62 x 48 x 58` is three small integers in one cell, which is exactly
     # what a packed peak looks like to anything counting numbers.
     lambda r: r.choice(["%.1f" % r.uniform(0.9, 9.5), str(r.randint(20, 4000)),
                         "%d x %d" % ((r.choice([64, 128, 256]),) * 2),
                         "%d x %d x %d" % (r.randint(48, 96), r.randint(48, 96),
                                           r.randint(30, 70))])),
)


#: The fallback forms, one chosen per column rather than per cell. A column
#: alternating between `4.61` and `28.9 +/- 5.8` down its length is not what a
#: table looks like, and it teaches that a column means nothing.
_FALLBACKS = (
    lambda r: "%.2f" % r.uniform(0, 5),
    lambda r: "%.3f" % r.uniform(0, 1),
    lambda r: str(r.randint(2, 240)),
    lambda r: "%.1f \u00b1 %.1f" % (r.uniform(20, 60), r.uniform(1, 12)),
)

#: A genome-wide p is written `3.6 x 10-6`, and the minus that follows the 10
#: reads as the sign of a coordinate. Only a genetics table prints one: a
#: demographics table reporting `1.1 x 10-7` for a sex difference is not a
#: near miss, it is nonsense.
_SCIENTIFIC = re.compile(r"\bp[-\s]?value|^p$", re.I)

#: Matched against the column header alone. A genetics table's row label is
#: itself an rs number, so matching these against the label too printed one in
#: every column.
_BY_COLUMN_ONLY = (
    (re.compile(r"^Chr$|chromosome", re.I),
     lambda r: "Chr%d" % r.randint(1, 22)),
    (re.compile(r"position|\bbp\b|locus", re.I),
     lambda r: "{:,}".format(r.randint(1_000_000, 240_000_000))),
    (re.compile(r"beta \(se\)|^beta$|\bOR \(SE\)", re.I),
     lambda r: "%.3f (%.3f)" % (r.uniform(-0.6, 0.6), r.uniform(0.01, 0.2))),
    (re.compile(r"nearest gene|^gene$", re.I),
     lambda r: r.choice(["BDNF", "COMT", "CACNA1C", "ZNF804A", "DRD2",
                         "APOE", "FKBP5", "OXTR", "TCF4", "SLC6A4"])),
)


def _plain(rng: random.Random, header: str = "", label: str = "",
           variant: int = -1, flavour: str = "") -> str:
    """A value that belongs under `header`, or beside `label`.

    A template comparison puts the unit in the row label -- `AC-PC length
    [mm]` -- and the template's name in the header, so the column says nothing
    about what belongs in it and the row says everything.
    """
    for pattern, make in _BY_COLUMN_ONLY:
        if pattern.search(header or ""):
            return make(rng)
    if flavour == "genetics" and _SCIENTIFIC.search(header or ""):
        return "%.1f x 10-%d" % (rng.uniform(1, 9.9), rng.randint(4, 12))
    for pattern, make in _BY_HEADER:
        if pattern.search(header or "") or pattern.search(label or ""):
            return make(rng)
    if variant < 0:
        variant = rng.randrange(len(_FALLBACKS))
    return _FALLBACKS[variant % len(_FALLBACKS)](rng)


def _caption_and_footer(rng: random.Random, kind: str, name: str, title: str,
                        w: Weights) -> Tuple[str, str]:
    caption = ""
    if title and rng.random() < w.caption_present:
        caption = "Table %d. %s." % (rng.randint(1, 8), title.format(
            a=rng.choice(vocab.CONDITIONS), b=rng.choice(vocab.GROUPS),
            m=rng.choice(_MEASURES), c=rng.choice(vocab.CONDITIONS),
            d=rng.choice(["excitatory-inhibitory", "Wilson-Cowan",
                          "predator-prey", "two-compartment",
                          "neural mass"]),
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
    elif name == "genetics":
        labels = ["rs%d" % rng.randint(1000, 99999999) for _ in range(8)]
    elif name == "equilibrium":
        labels = ["E%d" % i for i in range(1, 8)]
    elif name == "figure_legend":
        labels = ["" for _ in range(4)]
    elif name == "parameters":
        labels = list(_PARAMETERS)
    elif name == "roi_index":
        labels, n = [], 1
        for _ in range(9):
            width = rng.randint(1, 3)
            labels.append(", ".join(str(n + k) for k in range(width)))
            n += width
    elif name == "scan_statistic":
        labels = ["Primary cluster"] + ["%s secondary cluster" % o for o in
                                        ("1st", "2nd", "3rd", "4th", "5th", "6th")]
    elif name == "mask_overlap":
        labels = ["Lips", "Upper limb", "Trunk", "Lower limb", "Face", "Tongue"]
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
    carries, carried_form = _INTERVAL_COLUMN.get(name, (0, ("iqr",)))
    carried_form = rng.choice(carried_form)
    if carries >= width:
        carries = 0
    # One fallback form per column, chosen once.
    fallbacks = [rng.randrange(4) for _ in range(width)]
    places: list = []
    out = []
    for label in labels[:rng.randint(4, 8)]:
        if name == "parameters":
            sym, meaning, value, unit = rng.choice(_PARAMETERS)
            out.append([sym, meaning, value, unit])
            continue
        if name == "roi_index":
            if not places:
                places = [(side, lobe) for side in ("Left", "Right")
                          for lobe in _LOBES]
                rng.shuffle(places)
            side, lobe = places.pop()
            out.append([label, "%s %s" % (side, lobe),
                        str(rng.randint(400, 9000))])
            continue
        if name == "scan_statistic":
            out.append([label.split()[0],
                        ", ".join(str(rng.randint(1, 280))
                                  for _ in range(rng.randint(4, 12))),
                        "(%.6f N, %.6f E) / %.2f km"
                        % (rng.uniform(3.5, 14.5), rng.uniform(33.0, 43.0),
                           rng.uniform(10, 190)),
                        str(rng.randint(35, 450)), str(rng.randint(30, 380)),
                        "%.2f" % rng.uniform(1.1, 2.4),
                        "%.4f" % rng.uniform(0.0001, 0.02)])
            continue
        if name == "mask_overlap":
            out.append([label, rng.choice(["Lips", "Upper limb", "Trunk"]),
                        "{:,}".format(rng.randint(1500, 9800)),
                        "{:,}".format(rng.randint(40, 1800)),
                        "%.2f" % rng.uniform(0.0, 0.25)])
            continue
        if name == "equilibrium":
            # `E1 (0, 0, 0)`: a label and a bracketed triple, which is the
            # packed-peak shape with nothing anatomical anywhere near it.
            out.append([label,
                        "(%d, %d, %d)" % tuple(rng.randint(0, 3) for _ in range(3)),
                        _plain(rng, "Eigenvalues"), _plain(rng, "Stability")])
            continue
        if name == "figure_legend":
            out.append([
                "View larger version(%dK)" % rng.randint(20, 180),
                "Figure %d. %s activation in the %s during %s. Coordinates "
                "are shown in %s space; the slice is at z = %d."
                % (rng.randint(1, 8), rng.choice(["Group", "Mean", "Peak"]),
                   rng.choice(vocab.REGIONS).name,
                   rng.choice(vocab.CONDITIONS),
                   rng.choice(["MNI", "Talairach"]), rng.randint(-40, 60))])
            continue
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
                form = carried_form
                if name == "demographics":
                    counted = bool(_COUNTED.search(label))
                    if counted and form != "counts":
                        form = "counts"
                    elif not counted and form == "counts":
                        form = rng.choice(("iqr", "mean_sd_range"))
                cells.append(_stat_with_interval(rng, name, form))
            else:
                cells.append(_plain(rng, head, label, fallbacks[col], name))
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
