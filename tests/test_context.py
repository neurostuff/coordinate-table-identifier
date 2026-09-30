"""The rules that keep the title and abstract honest about the table."""

from __future__ import annotations

import random

from nspond_tables.synth import build, context
from nspond_tables.synth.empty import build_empty
from nspond_tables.synth.trainset import (Example, add_context,
                                          set_space_from_what_is_visible)


def _composed(n=250):
    out = []
    for seed in range(n):
        rng = random.Random(seed)
        t = build(seed=seed)
        target = t.truth.as_target()
        title, abstract = context.title_and_abstract(
            rng, t.grid.render(), target, caption=t.caption, footer=t.footer)
        out.append((t, target, title, abstract))
    return out


def test_how_often_the_abstract_names_a_region_from_its_own_table():
    """Pinned, not aspired to. The corpus figure is 14%; v18 shipped at 73%
    and `NAMES_ITS_OWN_REGION` holds this at v18's behaviour on purpose, so
    v19 measures a change of data and nothing else. The test exists so that
    moving the knob is a visible decision rather than a drift."""
    named, shared = 0, 0
    for t, _, _, abstract in _composed():
        if not abstract:
            continue
        rows = [l.split("|")[0].strip("#<^~0123456789: ")
                for l in t.grid.render().split("\n")]
        regions = [r for r in rows if len(r) > 6 and r[0].isalpha()]
        low = abstract.lower()
        if not any(w in low for w in ("gyrus", "cortex", "lobule", "nucleus",
                                      "cingulate", "insula", "amygdala")):
            continue
        named += 1
        if any(r.lower() in low for r in regions):
            shared += 1
    assert named > 150, named          # it names a region nearly every time
    assert 0.80 < shared / named, (shared, named)


def test_the_abstract_almost_never_states_a_coordinate_space():
    """1.2% of real ones do: the normalisation is stated in the Methods, and
    the Methods are not in the prompt. Generating more would teach that the
    abstract settles the space."""
    states = sum(1 for _, _, _, a in _composed()
                 if a and context.visible_space(a) is not None)
    assert states / 250 < 0.10, states


def test_a_title_is_a_title():
    """Drawn from the analyses rather than from the caption, whose words are
    boilerplate: `Local maxima of significant clusters` yields "local", and
    "the neural correlates of local" is not a title."""
    for _, _, title, _ in _composed(60):
        assert title and title[0].isupper(), title
        assert not title.lower().endswith((" of", " in", " the", " and")), title
        assert "local" not in title.lower().split(), title


def test_a_table_that_holds_nothing_still_has_a_paper_around_it():
    """Leaving the empty examples without an abstract made 84.7% of the rows
    lacking one answerable without reading the table."""
    for seed in range(40):
        rng = random.Random(seed)
        e = build_empty(seed=seed)
        title, abstract = context.title_and_abstract(
            rng, e.grid.render(), e.truth.as_target(), caption=e.caption)
        assert title and abstract, (seed, title, abstract)


def test_the_space_comes_from_the_visible_text_and_nothing_else():
    ex = Example(table="#Region | #<3:MNI coordinates\nInsula | 1 | 2 | 3",
                 target={"space": "TAL", "analyses": [{"name": "a",
                                                       "points": [[1, 2, 3]]}]})
    assert set_space_from_what_is_visible([ex])[0].target["space"] == "MNI"

    silent = Example(table="#Region | #x | #y | #z\nInsula | 1 | 2 | 3",
                     target={"space": "MNI", "analyses": [{"name": "a",
                                                           "points": [[1, 2, 3]]}]})
    assert set_space_from_what_is_visible([silent])[0].target["space"] is None


def test_both_parts_disagreeing_means_null_not_a_coin_flip():
    ex = Example(table="#<3:MNI coordinates", caption="Talairach space.",
                 target={"space": "MNI", "analyses": [{"name": "a", "points": []}]})
    assert set_space_from_what_is_visible([ex])[0].target["space"] is None


def test_an_empty_target_keeps_a_null_space():
    """A table stating no coordinates states no space for them. The
    template-comparison negatives are headed `MNI-305`, so the visible rule
    would otherwise put MNI on the very tables that teach answering nothing."""
    ex = Example(table="#Parameter | #MNI-305 | #ICBM-152\nLength | 1 | 2",
                 target={"space": None, "analyses": []})
    assert set_space_from_what_is_visible([ex])[0].target == {"space": None,
                                                              "analyses": []}


def test_a_real_article_supplies_its_own_title_before_anything_is_composed():
    ex = Example(table="#Region | #x\nInsula | 1", target={"space": None,
                                                           "analyses": []},
                 notes={"article_id": "slug-1"})
    got = add_context([ex], metadata={"slug-1": ("A real title", "A real abstract.")})
    assert got[0].title == "A real title"
    assert got[0].abstract == "A real abstract."


def test_every_example_ends_up_with_a_title():
    from nspond_tables.synth.trainset import build_trainset

    for ex in build_trainset(n=40, seed=5):
        assert ex.title, ex.notes
