# Additive-only control: what this run establishes

The four stored answers give a checkerboard contrast
`y(hello world) + y(loving there) - y(hello there) - y(loving world)` of
**-0.850160598755**. A sum of independent word contributions followed by an
affine answer map would give zero, for any learned word codes. Thus the actual
end-to-end path in this run is not that restricted additive model. The observed
4/4 classifications fail the planned one-half control, while MSE .08882 still
fails the class gate's unchanged .05 bar.

Source inspection offers a specific hypothesis, not a measured localization:
the mixing path stages OBJECT references for the eight real word occurrences
(`test_grammar_object_leaves.py`), but also pushes the separating space. In
`stage_cs_lang`, a position without an object row keeps its perceived idea.
The supplied-answer loss now reaches that perception through its own pullback.
An input-dependent space contribution would violate the independent-word
premise even with only `sum` in the grammar. Item 6.8 owns the space/word-whole
question; no configuration or runtime repair has been made on this hypothesis.

This run did not retain trained per-position contributions, so it cannot tell
how much of the contrast came from that path. No second control training was
run to select or replace this outcome. The exact stored answers and all four
reconstructions are in `measurement.json`; the explicit verdict is `FAIL`.
