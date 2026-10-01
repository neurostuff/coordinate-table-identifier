"""Words a real table uses, and where in the brain each region sits.

A region carries its own coordinate box so a drawn coordinate lands inside a
head and on the side its name claims. The generator used to pick the name and
the numbers independently, which left 8.8% of rows with a laterality marker
contradicting the x sign -- three times the real rate -- and 0.59% of
coordinates outside the brain entirely (34(p) in scans/PRE_V17.md).

The boxes are coarse on purpose. They need to be right about hemisphere and
roughly right about lobe; a table does not need anatomically exact peaks to
exercise a reader.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Region:
    name: str
    x: Tuple[int, int]
    y: Tuple[int, int]
    z: Tuple[int, int]
    lobe: str
    midline: bool = False

    def sided(self, side: Optional[str]) -> "Region":
        """The same region on a given side, name and x box together."""
        if self.midline or side is None:
            return self
        lo, hi = self.x
        span = (abs(lo) + abs(hi)) // 2 or 20
        base = max(4, min(abs(lo), abs(hi)))
        box = (base, base + span) if side == "R" else (-(base + span), -base)
        return Region(name=self.name, x=box, y=self.y, z=self.z, lobe=self.lobe)


# x boxes are written right-handed; `sided` mirrors them for the left.
REGIONS: List[Region] = [
    Region("fusiform gyrus", (28, 46), (-70, -40), (-24, -8), "temporal"),
    Region("inferior temporal gyrus", (42, 60), (-64, -20), (-30, -8), "temporal"),
    Region("middle temporal gyrus", (48, 66), (-60, -12), (-16, 4), "temporal"),
    Region("superior temporal gyrus", (46, 64), (-44, 8), (-8, 12), "temporal"),
    Region("parahippocampal gyrus", (18, 32), (-44, -14), (-28, -12), "temporal"),
    Region("hippocampus", (20, 34), (-40, -12), (-24, -8), "temporal"),
    Region("amygdala", (16, 28), (-10, 2), (-24, -10), "temporal"),
    Region("inferior frontal gyrus", (38, 56), (10, 40), (-6, 26), "frontal"),
    Region("middle frontal gyrus", (28, 48), (14, 52), (18, 48), "frontal"),
    Region("superior frontal gyrus", (12, 28), (10, 56), (36, 60), "frontal"),
    Region("precentral gyrus", (32, 56), (-24, 4), (30, 60), "frontal"),
    Region("orbitofrontal cortex", (18, 40), (30, 56), (-20, -4), "frontal"),
    Region("insula", (30, 44), (-16, 20), (-8, 14), "frontal"),
    Region("postcentral gyrus", (34, 58), (-36, -18), (36, 60), "parietal"),
    Region("inferior parietal lobule", (36, 56), (-66, -40), (32, 52), "parietal"),
    Region("superior parietal lobule", (18, 34), (-72, -50), (46, 64), "parietal"),
    Region("supramarginal gyrus", (46, 62), (-54, -30), (20, 38), "parietal"),
    Region("angular gyrus", (38, 54), (-70, -52), (26, 42), "parietal"),
    Region("lingual gyrus", (10, 26), (-90, -60), (-12, 4), "occipital"),
    Region("middle occipital gyrus", (26, 44), (-92, -68), (-4, 24), "occipital"),
    Region("superior occipital gyrus", (18, 32), (-94, -74), (16, 32), "occipital"),
    Region("cuneus", (6, 18), (-88, -68), (18, 34), "occipital"),
    Region("caudate", (10, 20), (-4, 18), (2, 18), "subcortical"),
    Region("putamen", (18, 30), (-10, 14), (-6, 10), "subcortical"),
    Region("thalamus", (6, 18), (-30, -8), (2, 16), "subcortical"),
    Region("cerebellum", (14, 40), (-76, -48), (-42, -22), "cerebellum"),
    Region("precuneus", (4, 14), (-76, -52), (34, 52), "parietal", midline=True),
    Region("anterior cingulate", (0, 10), (18, 44), (4, 26), "limbic", midline=True),
    Region("posterior cingulate", (0, 10), (-58, -30), (18, 36), "limbic", midline=True),
    Region("medial frontal gyrus", (0, 12), (30, 56), (18, 44), "frontal", midline=True),
    Region("vermis", (0, 6), (-64, -44), (-38, -22), "cerebellum", midline=True),
    Region("brainstem", (0, 8), (-32, -18), (-32, -14), "subcortical", midline=True),
    Region("supplementary motor area", (0, 12), (-8, 14), (48, 66), "frontal", midline=True),
]

LOBE_SECTIONS = {
    "frontal": ["Frontal cortex", "Frontal lobe", "Frontal regions"],
    "temporal": ["Temporal cortex", "Temporal lobe", "Occipito-temporal cortex"],
    "parietal": ["Parietal cortex", "Parietal lobe", "Parieto-occipital regions"],
    "occipital": ["Occipital cortex", "Occipital lobe", "Visual cortex"],
    "subcortical": ["Subcortical structures", "Subcortical regions", "Basal ganglia"],
    "limbic": ["Limbic regions", "Cingulate cortex"],
    "cerebellum": ["Cerebellar cortex", "Cerebellum"],
}

CONDITIONS = [
    "faces", "houses", "words", "objects", "scenes", "tools", "bodies",
    "reward", "loss", "neutral", "fear", "happy", "angry", "sad",
    "encoding", "retrieval", "rest", "baseline", "fixation", "control",
    "congruent", "incongruent", "go", "no-go", "switch", "repeat",
]
GROUPS = ["patients", "controls", "HC", "MDD", "ASD", "ADHD", "older adults",
          "younger adults", "experts", "novices"]
EFFECTS = ["Main effect of task", "Main effect of group", "Group x task interaction",
           "Effect of time", "Positive correlation", "Negative correlation",
           "Conjunction across conditions", "Average effect of condition"]

REGION_HEADERS = ["Region", "Anatomical region", "Brain region", "Area",
                  "Anatomical location", "Location", "Structure"]
SIDE_HEADERS = ["Side", "Hemisphere", "Hemi.", "Lat.", "L/R", "H"]
EXTENT_HEADERS = {
    "voxels": ["Cluster size (voxels)", "k", "kE", "No. of voxels", "N voxels",
               "Cluster extent (voxels)", "Voxels"],
    "mm^3": ["Cluster size (mm3)", "Volume (mm3)", "Cluster volume (mm3)",
             "Extent (mm3)"],
}
#: A statistic column whose header does NOT name its kind. `stat_in_header`
#: decides which set is drawn from; before, the kinded header was printed
#: whatever that flag said, so a table could head a column `T-stat` and still
#: carry a target claiming no statistic type -- teaching the model not to read
#: a header it should read.
UNNAMED_STAT_HEADERS = ["Value", "Statistic", "Stat.", "Peak value",
                        "Max", "Peak", "Value at peak"]
STAT_HEADERS = {
    "T": ["T", "t", "T value", "t-value", "T-stat", "Peak T", "t(38)"],
    "Z": ["Z", "Z value", "Z-score", "Peak Z", "z"],
    "F": ["F", "F value", "F(2,38)", "Peak F"],
    "P": ["p", "p value", "p(FWE cor.)", "p(unc.)", "p-corrected", "pFDR"],
    "R": ["r", "r value", "Correlation"],
    "B": ["Beta", "b", "Beta weight", "Coefficient"],
    "D": ["Cohen's d", "d", "Effect size (d)", "Cohen's d at the peak voxel"],
    "G": ["Hedges' g", "Hedge's g", "Hedges g"],
}
SPACE_HEADERS = {
    "MNI": ["MNI coordinates", "MNI", "MNI coordinates (mm)", "Coordinates (MNI)",
            "MNI152 coordinates", "Peak MNI coordinate"],
    "TAL": ["Talairach coordinates", "Talairach", "Talairach coordinates (mm)",
            "Coordinates (Talairach)", "TAL"],
}
AXIS_HEADERS = [("x", "y", "z"), ("X", "Y", "Z"), ("x (mm)", "y (mm)", "z (mm)")]
#: An axis that says what kind of coordinate it is. Read off real tables;
#: `X coor` is three headers in 6,897 and losing it lost a whole table.
AXIS_HEADERS_KINDED = [
    ("X coor", "Y coor", "Z coor"),
    ("x coordinate", "y coordinate", "z coordinate"),
    ("X (MNI)", "Y (MNI)", "Z (MNI)"),
    ("x (Talairach)", "y (Talairach)", "z (Talairach)"),
    ("X coord", "Y coord", "Z coord"),
]
#: The anatomical axes, used instead of x, y and z by some papers.
RAS_HEADERS = [("R", "A", "S"), ("Right", "Anterior", "Superior")]
BARE_COORD_HEADERS = ["Coordinates", "Peak coordinate", "Peak voxel", "Location"]

FOOTER_PARTS = [
    "The statistical threshold was set at p<0.05 (FWE-corrected).",
    "Cluster size threshold was {k} voxels.",
    "Coordinates are reported in {space} space.",
    "L: left; R: right.",
    "All reported clusters survive correction for multiple comparisons.",
    "{stat} values are given for the peak voxel of each cluster.",
    "Anatomical labels follow the AAL atlas.",
]
