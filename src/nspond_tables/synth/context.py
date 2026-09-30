"""The title and abstract a synthetic table's paper would carry.

A table does not arrive alone. The prompt carries the article's title and
abstract, and every rule here exists because getting one of them wrong teaches
something false:

* **The abstract talks about this table.** It names the contrasts the table is
  split on and spells out the abbreviations the table gives only in short form
  -- a row reading `DLPFC` under an abstract reading `dorsolateral prefrontal
  cortex` is the cross-reference the model has to learn to follow. An abstract
  lifted from a different article teaches the opposite.

* **But it is not a description of the table.** An abstract discusses a whole
  paper: other tables, prior work, the discussion. Real abstracts name a
  region from their own table only **14%** of the time they name a region at
  all. An earlier generator scored 100%, which would have taught that anything
  the abstract mentions is in the table.

* **It rarely settles the coordinate space.** Only **1.2%** of real abstracts
  name one, because the normalisation is stated in the Methods and the Methods
  are not in the prompt.

* **The target's space is what the visible text says**, and nothing else --
  `visible_space` over the abstract, caption, footer and table. Not article
  metadata: 53.5% of rows carry no visible cue, so a metadata-derived target
  would teach the model to invent a space on half its examples. Null when
  nothing states it, and null when two parts of the document disagree.

* **A table that holds nothing still sits in a paper.** Its abstract is about
  that paper, and says nothing about this table -- which is the whole
  discrimination the empty examples exist to teach. Leaving them without one
  would hand the model a shortcut: in the first v19 build, 84.7% of the rows
  with no abstract had an empty target, and none of the rows with one did.

* **An empty target keeps `space: null`.** A table stating no coordinates
  states no space for them, whatever its abstract mentions.

The vocabulary and the rates are v18's, measured against the corpus rather
than assumed, and are moved here so the rules live with the generator instead
of in a script beside it.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import vocab
from .vocab import GROUPS, REGIONS

#: What a real footer spells out, and what the abstract therefore has to be
#: able to expand. The pairing is the point: a row reading `DLPFC` under an
#: abstract reading `dorsolateral prefrontal cortex` is the cross-reference
#: the model must learn to follow.
FOOT_DEFS = {
    "BA": "Brodmann area", "L": "left", "R": "right",
    "MNI": "Montreal Neurological Institute",
    "ROI": "region of interest", "FWE": "family-wise error",
    "FDR": "false discovery rate",
    "SMA": "supplementary motor area", "IFG": "inferior frontal gyrus",
    "MFG": "middle frontal gyrus", "STG": "superior temporal gyrus",
    "IPL": "inferior parietal lobule", "ACC": "anterior cingulate cortex",
    "PCC": "posterior cingulate cortex",
    "DLPFC": "dorsolateral prefrontal cortex",
    "TAL": "Talairach", "k": "cluster extent in voxels",
}

#: Participant groups. Not regions: an abstract names these where it says what
#: was compared, never where it says where activation was found. They reach the
#: table through the harvested contrast vocabulary -- HC alone appears in 450
#: analysis names per 1,500 tables -- and had no long form anywhere.
GROUP_DEFS = {
    "HC": "healthy controls", "HCs": "healthy controls",
    "NC": "normal controls", "TD": "typically developing children",
    "MDD": "major depressive disorder", "ASD": "autism spectrum disorder",
    "SCZ": "schizophrenia", "PTSD": "post-traumatic stress disorder",
    "PD": "Parkinson's disease", "AD": "Alzheimer's disease",
    "MCI": "mild cognitive impairment",
    "ADHD": "attention-deficit/hyperactivity disorder",
    "OCD": "obsessive-compulsive disorder", "BD": "bipolar disorder",
    "FEP": "first-episode psychosis", "UHR": "ultra-high risk",
}

#: Networks are spatial, so they belong with the regions -- but an abstract
#: reports activation *within* a network rather than at one.
NET_DEFS = {
    "DMN": "default mode network", "CCN": "cognitive control network",
    "DAN": "dorsal attention network", "ECN": "executive control network",
    "SN": "salience network", "FPN": "frontoparietal network",
}

#: Methodological abbreviations. These are neither regions nor groups: an
#: abstract that reports activation "in the region of interest (ROI)" is
#: listing a method as if it were a place. They get a methods sentence, which
#: still spells them out where the table only abbreviates.
METHOD_PHRASE = {
    "ROI": "Region-of-interest (ROI) analyses were conducted.",
    "FWE": "Clusters survived family-wise error (FWE) correction.",
    "FDR": "Thresholds were corrected using the false discovery rate (FDR).",
    "BA": "Peaks are labelled by Brodmann area (BA).",
}


#: How often an abstract reproduces an analysis label word for word. Real
#: captions share vocabulary with the analyses 37% of the time but reproduce
#: the label verbatim only 8.7%. Quoting it every time made the prose an
#: exact-match key to the answer: a shortcut, not information.
VERBATIM = 0.087


def paraphrase(rng: random.Random, name: str) -> str:
    """Say what an analysis is without quoting its label."""
    n = (name or "").strip()
    if rng.random() < VERBATIM:
        return n
    n = re.sub(r"\s*>\s*",
               rng.choice([" compared with ", " versus ", " relative to "]), n)
    n = re.sub(r"\s*<\s*", rng.choice([" compared with ", " versus "]), n)
    if rng.random() < 0.5 and n:
        n = n[0].lower() + n[1:]
    return n


def join(names: Sequence[str]) -> str:
    names = [n for n in names if n]
    if len(names) == 1:
        return "the %s analysis" % names[0]
    return "the %s and %s analyses" % (", ".join(names[:-1]), names[-1])


def fix_acronym(name):
    """paraphrase() can lower-case an acronym's first letter: DMN -> dMN."""
    if len(name) > 1 and name[0].islower() and name[1].isupper():
        return name[0].upper() + name[1:]
    return name


#: An abstract discusses a whole paper -- other tables, prior work, the
#: discussion -- so most anatomy it names is not in the table beside it. Real
#: abstracts name a region from their own table only 14% of the time they name
#: a region at all; the first version of this generator scored 100%, which
#: would have taught that anything the abstract mentions is in the table.
ABS_BACKGROUND = [
    "Previous work has implicated the %s in this process.",
    "Earlier studies reported involvement of the %s.",
    "The %s have been linked to these functions.",
]
ABS_ELSEWHERE = [
    "Further clusters were observed in the %s in secondary analyses.",
    "Additional effects outside the present contrast involved the %s.",
    "Exploratory whole-brain analyses also implicated the %s.",
]
ABS_DISCUSS = [
    "The involvement of the %s is discussed in relation to prior reports.",
    "We consider these results alongside findings in the %s.",
]
ABS_MORE = [
    "Connectivity analyses implicated the %s.",
    "Structural measures differed in the %s.",
    "A conjunction across conditions involved the %s.",
    "Region-of-interest analyses covered the %s.",
    "Effects of covariates were seen in the %s.",
    "Control analyses examined the %s.",
]
ABS_LIMIT = [
    "Interpretation is limited by sample size and by the spatial resolution of the acquisition.",
    "Replication in an independent cohort will be needed before these effects are considered settled.",
    "The cross-sectional design precludes conclusions about the direction of these associations.",
]


#: Prose that carries no anatomy. Real abstracts run ~233 words against the
#: ~141 the anatomical sentences alone produce, and the difference is
#: hypotheses, acquisition detail and clinical framing. Adding words here
#: rather than more regions keeps the precision figure where it was measured.
#: Methods prose. Belongs after the sample sentence, never before the aim.
ABS_METHODS_FILL = [
    "Images were acquired on a 3 T scanner using a gradient-echo echo-planar sequence.",
    "Data were preprocessed with standard realignment, coregistration and smoothing steps.",
    "Motion parameters were included as nuisance regressors in all first-level models.",
    "Statistical maps were thresholded at p < 0.001 uncorrected at the voxel level.",
    "Age, sex and years of education were entered as covariates of no interest.",
    "Participants gave written informed consent and the protocol was approved by the local ethics committee.",
]

#: Framing prose. Belongs at the end, with the discussion.
ABS_FRAME_FILL = [
    "Behavioural performance did not differ between conditions, so the imaging effects are unlikely to reflect difficulty.",
    "The findings bear on models that treat these processes as functionally separable.",
    "Understanding this dissociation may inform how such deficits are characterised clinically.",
]

ABS_HYPOTHESIS = [
    "We hypothesised that the two conditions would dissociate along a rostro-caudal gradient.",
    "We predicted that the effect would be strongest under the more demanding condition.",
]

ABS_FILLER = [
    "Images were acquired on a 3 T scanner using a gradient-echo echo-planar sequence.",
    "We hypothesised that the two conditions would dissociate along a rostro-caudal gradient.",
    "Statistical maps were thresholded at p < 0.001 uncorrected at the voxel level.",
    "Motion parameters were included as nuisance regressors in all first-level models.",
    "Participants gave written informed consent and the protocol was approved by the local ethics committee.",
    "Behavioural performance did not differ between conditions, so the imaging effects are unlikely to reflect difficulty.",
    "Age, sex and years of education were entered as covariates of no interest.",
    "The findings bear on models that treat these processes as functionally separable.",
    "Understanding this dissociation may inform how such deficits are characterised clinically.",
    "Data were preprocessed with standard realignment, coregistration and smoothing steps.",
]


def distractors(rng, exclude, k):
    """Regions for the rest of the paper, never the ones in this table."""
    out, seen = [], {w.lower() for w in exclude}
    for cand in rng.sample(REGIONS, min(len(REGIONS), k * 8)):
        c = prose_region(str(cand))
        # Skip anything still carrying table punctuation or a bare index: with
        # 4,000 regions to draw from, the clean ones are plentiful.
        if not c or not re.fullmatch(r"[A-Za-z][A-Za-z \-']{3,48}", c):
            continue
        if re.match(r"(?i)(lh|rh)-", c):
            continue
        if any(w in seen for w in c.lower().split()):
            continue
        seen.update(c.lower().split())
        out.append(c)
        if len(out) == k:
            break
    return out


ABS_GROUPS = [
    "We compared %s.",
    "Analyses contrasted %s.",
    "The sample comprised %s.",
    "Participants were %s.",
]


#: Sentence frames for the synthetic abstract. Kept plain: the point is the
#: words they carry, not the prose.
ABS_AIM = [
    "We used functional MRI to examine %s.",
    "The present study investigated %s.",
    "This study examined %s using functional MRI.",
    "We investigated %s in a whole-brain analysis.",
]
ABS_METHOD = [
    "%d participants completed the task during scanning.",
    "A total of %d volunteers were scanned.",
    "Imaging data were acquired from %d participants.",
    "%d subjects underwent functional imaging.",
]
ABS_RESULT = [
    "Significant effects were observed in the %s.",
    "We found reliable activation in the %s.",
    "Clusters were identified in the %s.",
    "The contrast revealed responses in the %s.",
]
ABS_CLOSE = [
    "These findings indicate a distributed response across the regions reported below.",
    "Together the results implicate this network in the processes examined here.",
    "The pattern is consistent with the regions listed in the table.",
]
ABS_SPACE = {
    "MNI": ["Images were normalised to MNI space.",
            "Data were spatially normalised to the MNI152 template.",
            "All coordinates are reported in MNI space."],
    "TAL": ["Coordinates were converted to Talairach space.",
            "Images were normalised to the Talairach and Tournoux atlas."],
}


def conjoin(items):
    """"a, b and c" -- join() is for analysis names and says "analyses"."""
    if len(items) == 1:
        return items[0]
    return "%s and %s" % (", ".join(items[:-1]), items[-1])


def prose_region(cell):
    """A region name as an abstract would write it.

    Lower-cased, because a sentence is not a table cell, but any run of
    capitals is left alone: those are the abbreviations whose long forms are
    the reason the abstract is here at all.
    """
    c = re.sub(r"\s*\(.*?\)", "", cell).strip().rstrip(",;.")
    # Atlas labels are table vocabulary, not prose: a paper writes "left
    # cerebellar crus II", never "Cerebellum_Crus2_L".
    c = c.replace("_", " ").replace("/", " and ")
    c = re.sub(r"(?i)(^|\s)([LR])[.\s]+", lambda m: m.group(1) + (
        "left " if m.group(2).upper() == "L" else "right "), c)
    c = re.sub(r"(?i)\s+([LR])$", lambda m: "", c) if re.search(r"(?i)\s[LR]$", c) else c
    # AAL-style labels abbreviate the modifier and put it last: Parietal_Inf_L.
    # An abstract says "inferior parietal", so expand then move it forward.
    ATLAS = {"inf": "inferior", "sup": "superior", "mid": "middle",
             "orb": "orbital", "med": "medial", "ant": "anterior",
             "post": "posterior", "lat": "lateral", "tri": "triangularis",
             "oper": "opercularis", "vstriatum": "ventral striatum"}
    c = " ".join(ATLAS.get(w.lower(), w) for w in c.split())
    parts = c.split()
    if len(parts) > 1 and parts[-1].lower() in set(ATLAS.values()):
        parts = [parts[-1]] + parts[:-1]
        c = " ".join(parts)
    # A footnote marker sticks to the last word: "gyrusa", "insulaa".
    c = re.sub(r"(?i)\b(gyrus|cortex|lobe|lobule|area|nucleus|insula|cerebellum"
               r"|operculum|cingulate|precuneus|thalamus|putamen)[a-z]\b", r"\1", c)
    c = re.sub(r"\s{2,}", " ", c).strip()
    c = re.sub(r"^([LR])[.\s]+", lambda m: "left " if m.group(1) == "L" else "right ", c)
    # Two capitals, or a letter beside a digit: DLPFC, A1+, V1, BA9 are labels,
    # not words, and an abstract writes them the way the table does.
    keep = re.compile(r"[A-Z]{2,}|[A-Za-z]\d|Brodmann")
    return " ".join(w if keep.search(w) else w.lower() for w in c.split())


def compose_abstract(rng, lines, analyses, space):
    """An abstract that talks about this table, not about some other paper.

    Two things make it worth having. It names the contrasts the table is split
    on, and it spells out the abbreviations the table gives only in short form
    -- a row reading `DLPFC` under an abstract reading `dorsolateral prefrontal
    cortex` is the cross-reference the model has to learn to follow. An
    abstract lifted from another article would teach the opposite, the way a
    borrowed footer teaches that footers are unrelated to their tables.

    Returns "" when the table offers nothing to talk about, which keeps the
    empty case honest rather than inventing content.
    """
    body = "\n".join(lines)

    def used(abbr):
        return re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(abbr), body)

    # Region names as the table writes them, minus headers and banners.
    seen = []
    for l in lines:
        if not l or l[0] in "#<" or "|" not in l:
            continue
        cell = l.split("|")[0].strip().lstrip("#<0123456789^").strip()
        if cell and re.search(r"[A-Za-z]{3}", cell) and cell not in seen:
            seen.append(cell)

    # Prefer the long form of anything the table abbreviates: that pairing is
    # the whole reason the abstract is in the prompt.
    long_forms, plain = [], []
    for abbr, full in FOOT_DEFS.items():
        if abbr in ("k", "L", "R", "MNI", "TAL") or abbr in METHOD_PHRASE \
                or not used(abbr):
            continue
        long_forms.append(full if rng.random() < 0.72 else "%s (%s)" % (full, abbr))
    for cell in seen:
        c = prose_region(cell)
        # Some layouts put a whole row in the first cell, which arrived in the
        # abstract as "middle temporal gyrus 20 right 40 -18 -20 4.55 31".
        # A region name has no numbers in it.
        if not re.fullmatch(r"[A-Za-z][A-Za-z \-']{3,48}", c or ""):
            continue
        if not any(c.lower() in f.lower() for f in long_forms):
            plain.append(c)

    # Networks read as regions do, so they join the results sentence.
    for abbr, full in NET_DEFS.items():
        if used(abbr):
            # The frame already supplies "in the ...".
            long_forms.append("%s (%s)" % (full, abbr)
                              if rng.random() < 0.4 else full)
    # Groups get their own sentence, and dedupe by expansion so HC and HCs do
    # not both spell out "healthy controls".
    groups, said = [], set()
    for abbr, full in GROUP_DEFS.items():
        if used(abbr) and full not in said:
            said.add(full)
            groups.append(full if rng.random() < 0.65 else "%s (%s)" % (full, abbr))

    rng.shuffle(plain)
    keep = rng.choice([1, 2, 2, 3])
    # An expansion always earns its place: `DLPFC` in the table beside
    # `dorsolateral prefrontal cortex` in the prose is the cross-reference the
    # abstract is there for. Repeating a region the table already spells out
    # earns nothing and, done every time, teaches that anything the abstract
    # names is in the table.
    mentions = long_forms[:keep]
    if rng.random() < NAMES_ITS_OWN_REGION:
        mentions += plain[: max(0, keep - len(mentions))]
    if not mentions and not groups:
        return ""
    rng.shuffle(mentions)

    names = [n for n, _ in analyses if n and n.upper() != "UNKNOWN"]
    topic = paraphrase(rng, rng.choice(names)) if names else "task-related activation"
    topic = fix_acronym(topic)

    out = [rng.choice(ABS_AIM) % topic,
           rng.choice(ABS_METHOD) % rng.randint(12, 64)]
    if groups:
        out.append(rng.choice(ABS_GROUPS) % conjoin(groups[:3]))
    meth = [ph for ab, ph in METHOD_PHRASE.items() if used(ab)]
    if meth:
        out.extend(rng.sample(meth, min(len(meth), rng.randint(1, 2))))
    if space and rng.random() < ABS_STATES_SPACE:
        out.append(rng.choice(ABS_SPACE[space]))
    # Everything above is about this table. Everything below is the rest of the
    # paper, which is most of a real abstract.
    far = distractors(rng, set(" ".join(mentions + plain).split()), 15)
    if far[:3]:
        out.insert(0, rng.choice(ABS_BACKGROUND) % conjoin(far[:3]))
    if mentions:
        out.append(rng.choice(ABS_RESULT) % conjoin(mentions))
    # The rest of the paper. Frames are drawn without replacement so a long
    # abstract does not repeat one sentence shape.
    pool = rng.sample(ABS_MORE, len(ABS_MORE)) + [rng.choice(ABS_ELSEWHERE),
                                                  rng.choice(ABS_DISCUSS)]
    for i, lo in enumerate(range(3, 15, 3)):
        chunk = far[lo:lo + 3]
        if chunk and i < len(pool):
            out.append(pool[i] % conjoin(chunk))
    if rng.random() < 0.55:
        out.append(rng.choice(ABS_LIMIT))
    # Background, aim, hypothesis, methods, results, discussion -- an abstract
    # that opens on its preprocessing reads as generated, whatever it contains.
    if rng.random() < 0.6:
        out.insert(2, rng.choice(ABS_HYPOTHESIS))
    for i, extra in enumerate(rng.sample(ABS_METHODS_FILL, rng.randint(2, 4))):
        out.insert(min(3 + i, len(out)), extra)
    out.extend(rng.sample(ABS_FRAME_FILL, rng.randint(1, 2)))
    if len(names) > 1 and rng.random() < 0.45:
        out.append("Analyses compared %s."
                   % join([fix_acronym(paraphrase(rng, n)) for n in names[:3]]))
    if rng.random() < 0.38:
        out.append(rng.choice(ABS_CLOSE))
    return " ".join(out)


#: Measured, not assumed: only 0.8% of real abstracts name a coordinate space.
#: The normalisation is stated in the Methods, which is not in the prompt. An
#: earlier value of 0.55 here came from assuming the opposite, and would have
#: taught the model that the abstract reliably settles the space.
ABS_STATES_SPACE = 0.012

#: How often the results sentence repeats a region the table already names.
#:
#: Three numbers matter here and only one of them is a target. Real abstracts
#: name a region from their own table **14%** of the time they name one at
#: all. v18 shipped at **73%**. Holding this knob at 1.0 reproduces v18.
#:
#: It stays at 1.0. v19 changes the data in three deliberate ways and exists
#: to measure them; moving this at the same time would confound the two, and
#: lowering it also costs the abbreviation expansions, which are the reason
#: the abstract is in the prompt at all. Bringing it toward 0.14 is v20's
#: work, and wants the expansions separated from the plain repeats first --
#: at 0.62 the share of abstracts naming any region fell from 99% to 55%,
#: which is a second deviation, not a fix.
NAMES_ITS_OWN_REGION = 1.0


# -- the space a reader could actually get from the document ---------------


# A serialised header cell carries its span inline -- `<3MNI coordinates` -- and
# \b needs a word boundary, which a digit beside a letter does not give. That
# hid the cue on 3,030 of 24,409 rows (12.4%), all of them labelled null while
# their own table named the space. Talairach was unaffected, having no leading
# \b, so the loss fell entirely on MNI. Look for a letter boundary instead.
_NB = r"(?<![A-Za-z])"
_NA = r"(?![A-Za-z])"
MNI_RE = re.compile(_NB + r"(?:MNI|ICBM)" + _NA + r"|MNI-?152|montreal neuro", re.I)
TAL_RE = re.compile(r"talairach|tournoux|" + _NB + r"(?:Tal|TT)" + _NA, re.I)


def visible_space(*texts):
    """"MNI", "TAL", or None -- from the visible text alone."""
    t = " ".join(str(x or "") for x in texts)
    mni, tal = bool(MNI_RE.search(t)), bool(TAL_RE.search(t))
    if mni and not tal:
        return "MNI"
    if tal and not mni:
        return "TAL"
    return None


def title_and_abstract(rng: random.Random, table, target: Dict,
                       *, caption: str = "", footer: str = "",
                       ) -> Tuple[str, str]:
    """What the paper around this table would be called, and say.

    `target` is the table's own target, so the abstract is composed from what
    the table actually holds. A table with no analyses gets an abstract about
    its paper rather than about itself -- see the module docstring.
    """
    analyses = [(a.get("name") or "", a.get("points") or [])
                for a in (target.get("analyses") or [])]
    lines = (table or "").split("\n")
    if analyses:
        abstract = compose_abstract(rng, lines, analyses,
                                    target.get("space") or rng.choice(["MNI"] * 7
                                                                      + ["TAL"]))
    else:
        abstract = abstract_for_a_table_that_holds_nothing(rng)
    title = title_for(rng, [n for n, _ in analyses], caption)
    return title, abstract


def abstract_for_a_table_that_holds_nothing(rng: random.Random) -> str:
    """An abstract for the paper, not for the table.

    A demographics table sits in an imaging paper, and that paper's abstract
    reports activation somewhere -- in a different table. The model has to
    answer from the table in front of it and not from the prose around it,
    which it can only learn if the prose is there.

    Built from regions and contrasts drawn at random, so nothing in it can be
    found in the table it accompanies.
    """
    pool = [r.name for r in REGIONS]
    picked = rng.sample(pool, min(len(pool), rng.randint(2, 4)))
    names = [("%s > %s" % (a, b)) for a, b in
             [rng.sample(vocab.CONDITIONS, 2) for _ in range(rng.randint(1, 2))]]
    analyses = [(n, []) for n in names]
    lines = ["#Region | #x | #y | #z"] + ["%s | 0 | 0 | 0" % p for p in picked]
    return compose_abstract(rng, lines, analyses,
                            rng.choice(["MNI"] * 7 + ["TAL"]))


def title_for(rng: random.Random, analyses: Sequence[str],
              caption: str = "") -> str:
    """A title about the study, drawn from the contrasts it ran.

    Real title/caption pairs share a content word 67% of the time; two drawn
    independently share one 20%. The shared word comes from the analyses
    rather than from the caption, because a caption's own words are mostly
    boilerplate -- `Local maxima of significant clusters` yields "local",
    and a title reading "the neural correlates of local" is not a title.
    """
    words: list = []
    for name in analyses:
        words += [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", name or "")
                  if w.lower() not in _STOP]
    if not words:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", caption or "")
                 if w.lower() not in _STOP]
    region = rng.choice([r.name for r in REGIONS])
    if not words:
        return "%s%s" % (rng.choice(_TITLE_OPENINGS), region)
    head = rng.choice(words)
    head = head if head.isupper() else head.lower()
    return rng.choice([
        "%s%s" % (rng.choice(_TITLE_OPENINGS), head),
        "%s%s in the %s" % (rng.choice(_TITLE_OPENINGS), head, region),
        "%s and the %s" % (head[0].upper() + head[1:], region),
    ])


#: Words a caption or an analysis name carries that say nothing about the
#: study: every coordinate table's caption has them.
_STOP = {"table", "regions", "region", "showing", "during", "results",
         "coordinates", "coordinate", "significant", "analysis", "analyses",
         "effects", "effect", "activation", "activations", "cluster",
         "clusters", "peaks", "peak", "areas", "area", "brain", "space",
         "between", "greater", "local", "maxima", "maximum", "main",
         "versus", "compared", "with", "than", "that", "this", "from",
         "more", "less", "group", "groups",
         # Bare adjectives. "Functional organisation of older" is not a title.
         "older", "young", "younger", "healthy", "positive", "negative",
         "adults", "children", "male", "female", "interaction", "time"}

_TITLE_OPENINGS = ["Neural correlates of ", "Functional organisation of ",
                   "Variables influencing the neural correlates of ",
                   "Dissociable contributions of ", "An fMRI study of ",
                   "Individual differences in "]
