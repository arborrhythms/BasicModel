# R10 source audit — 2026-10-21

Base: published BasicModel `a514a71e0`, whose production source is
`3a9db635e`. The sole production hunk replaces the conditional empty value
with the site's content in the own-object record. Anonymous and pending
flags, kinds, score/choice rules, term registration, prepare, commit and
arrival refinement are unchanged. No parameters, detach, stored working
vector, grammar rule or lexical classification is added.

An AST comparison against the published reference-term test confirms that
every existing function, including every assertion, is unchanged. The only
added function is `test_named_token_keeps_its_content_before_and_after_the_close`.
It supplies reading order one through `reference_orders`, with no resolver
or grammar forcing and no seed. All other production, test and config files
are unchanged. The exact patch is retained in `source-diff.patch`.

The source is frozen before the one authorized five-file run. The run
records its Git hash in `frozen-source.json` and captures resolver plans,
features, probabilities and initial states from the outset. No fresh
pending-premise series and no sweep are authorized for this repair.
