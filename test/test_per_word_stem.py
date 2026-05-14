"""Per-word stem tests retired 2026-05-14.

The per-word stem (``_forward_stem_per_word``) was the AR-mode stem
that pushed one C-tier idea onto ``ConceptualSpace.stm`` per
perceptual slot, so the body's chart-at-C had something to read.
With the IR-only refactor (no per-cursor walk, single-shot masked-LM
forward) the stem inlines into ``_forward_per_stage`` as
``InputSpace.forward`` + ``PerceptualSpace.forward`` and STM is left
empty -- the chart's ``valid_mask`` handles the no-op cleanly.

All tests in this file depended on the per-word STM contract that no
longer exists.  Retired wholesale; the file is kept as a marker so
the deletion shows up in ``git log`` and future bisects find the
explanation here.
"""
"""IR-only refactor retired per-word stem on 2026-05-14."""
