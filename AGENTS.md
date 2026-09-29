# AGENTS.md

## The Grid is the contract

Real sources and the synthetic generator both build a `Grid`, and `render()` is
the only thing that emits serialised text. Never write serialised text directly:
that is what made the previous generator drift from the real serialiser.

A change to the rendered format changes both halves at once, and invalidates
every trained checkpoint. Say so in the PR.

## Difficulty weights are deliberate

`synth/weights.py` holds rates that are intentionally above what papers do, so
that a shortcut which is 90% accurate meets enough counterexamples to break.
Before changing one, decide which of three it is:

- miscalibration — fix it
- deliberate overweighting — leave it, and check the comment says why
- an artefact of how the curated labels were made — do not reproduce it

## Measurements

Numbers quoted in docstrings name the script that produced them and the sample
size. A measured claim with no script is a guess; delete it or re-measure.
