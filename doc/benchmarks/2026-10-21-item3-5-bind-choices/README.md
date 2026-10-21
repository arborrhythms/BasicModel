# Step 1b after review nine: bind choices (§19)

Base: `d29480774a93dc7fba630c78ee1e08b750ce2632`. Source freeze and
its gates follow the development record below. No result from an earlier
source is promoted. Publication requires the new source gates and Claude's
review; no second full sweep or learning claim.

Closing resolver costs are diagnostics only. Bind decisions use the same
sentence R + A score-function surrogate as other departures, retaining the
resolver's single optimizer owner and known-referent supervision. The menu
contains a fixed zero-logit dummy, which wins ties. A departure takes the
next-ranked choice. Noun sites automatically open an object when unlinked;
gaps remain open without a pool entry. Arrival decisions use this same menu.
The probe records both trials' menus, features, logits, probabilities and
selected/greedy/departure indices alongside R, A and the keep decision.

Refinement merges the pending token into an already pooled antecedent,
preserving the antecedent key, kind and earlier uses. The retained review
reproduction now reads its token through `token_of`; its four semantic
assertions are unchanged. Native arrivals and unpooled arrivals retain their
accepted paths. No working vectors, new MTM read detaches or pool reads in
the compiled body were added.

The §14 pending-premise gate uses `ForcedGrammar(open_names=('x',))`, without
`named_bindings`. Each addressed x must bind to three or keep its named open
referent. Both orders retain the live departure assertion and record bound
and open counts. The previous strict assertions are copied verbatim into
`test_live_pending_premise_closes_in_both_orders`, registered under item 0's
`pending_premise_learning` phase with no supplied artifact. Premises still
receive no answer target. The decomposition fixture is unchanged.

## Development runs (not frozen-source gates)

- `bootstrap-trace-01`, `bootstrap-content-02`, `bootstrap-word-row-03`
  and `bootstrap-context-05` retain the exact initial RNG and the zero-R
  failure. Both unsuccessful content-reader patches were reverted; their
  diffs and complete failed runs remain here.
- `bootstrap-old-carrier-04` is a diagnostic reversion of the captured ID
  lane, not production: it passes. The first word's context norm changes
  from 1/2 to 1/3. The new carrier exposes empty embedded subjects as extra
  occurrences; the conjunction inverse then returns the lower compatible
  magnitude. Neither the inverse nor the reconstruction objective was changed.
- `bootstrap-reference-lane-06` separates captured content and reference IDs
  and passes, confirming the reader fault. It is superseded by the narrower
  repair: keep the accepted OBJECT `concept_ids` unchanged and consult the
  WORD carrier before interpreting an objectless word in `ClauseJournal`
  and `resolve_lexical_references`. Mixing values keep their captured bank
  content; references still reach the close and seat plan.
- `bootstrap-word-carrier-07`: nine cases pass, including all three updates
  of the exact zero-R replay, the unchanged eight-seat carrier certificate,
  six new bind-choice cases and the repaired gap case.
- `focused-dev-01` retains a collection failure from a misspelled selector.
  `focused-dev-02`: 116 pass, seven fail. G1–G3, the episode, both merge
  certificates, all fourteen Part-closing and four occurrence-view cases
  pass. One failure exposed the gap carrier bug, repaired and checked in
  the next run. Six unchanged arrival cases remain red under the dummy
  behavior: three expect a zero-score candidate to bind; three expect count
  two when the arrival's new noun now opens a second object before refinement.
  These failures are retained for review, not waived or relabeled as passes.

The only existing test assertion changes are E's replacement certificate and
B's `token_of` port (which changes no semantic check). The `SentenceCredit`
caller test changes its supplied term from the removed `reference` argument
to `answer`; all of its assertions stay unchanged. The strict learning
certificate adds a declared artifact prerequisite rather than running a
cold-start learning test as a mechanism gate.

## Remaining publication work

All frozen-source gates, their source audits, review and publication remain
to be recorded by hash. Publication includes both mains, the WikiOracle
submodule bump, the October 20 documents and the decided naming sweep:
old LTM1 → LTM0, old LTM2 → LTM1, stale LTM3 retired. No prerequisite skip
or development pass substitutes for a requested gate.
