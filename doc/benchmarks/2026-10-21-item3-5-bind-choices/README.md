# Step 1b after review nine: bind choices (§19)

Base: `d29480774a93dc7fba630c78ee1e08b750ce2632`. Frozen source: `575cab06cb7d2ecb94a0c02cf64cb71bc6de33b0`.
All requested gates have run on this hash. Six unchanged arrival tests
remain red on that source. Claude completed the review; §20 specifies the
next source and authorizes their six ports. No result
from an earlier source is promoted; no second full sweep or learning claim.

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
  two where B's repaired merge preserves all three occurrences. Claude's
  review corrected the initial attribution of the count failures to D:
  the arriving noun already opened its own object before this round.
  These failures are retained for review, not waived or relabeled as passes.

The only existing test assertion changes are E's replacement certificate and
B's `token_of` port (which changes no semantic check). The `SentenceCredit`
caller test changes its supplied term from the removed `reference` argument
to `answer`; all of its assertions stay unchanged. The strict learning
certificate adds a declared artifact prerequisite rather than running a
cold-start learning test as a mechanism gate.

## Frozen-source gates

[Source audit](source-audit.json): all 21 run manifests match the 827
validated source files at `575cab06c`, including the carrier gate added
after Claude's review and before any source change.
[Full ledger](gate-summary.json) and [twelve-draw receipt](fresh-final.md)
retain failures, skips and runner details.

| Gate | Frozen-source result |
|---|---|
| Twelve first draws, six per order | 12 pass; forward 0 bound / 16 open each, reverse 0 / 32 each |
| Eight-file set | 113 pass, 3 declared prerequisite skips |
| Additional MTM / arrival / choice / Part-closing / occurrence cases | 53 pass, 6 fail |
| Decomposition, zero and enabled gains | Both lengths pass in both arms; original snapshots and seeds |
| Eight-seat MM, rng-119 and the same captured fresh draw | Both pass; 0 seat requests, 0 preflight errors |
| C: exact bootstrap-diagnostic-01 RNG | Pass; R = 0 in both trials at all three updates |

G1–G3, the episode, both unchanged merge certificates, reference close and
mentions, all fourteen Part-closing cases, four occurrence views, B's
retained reproduction and all six new D/answer cases pass. The eight-file
skips are the existing copular prerequisite and the two newly registered
strict learning cases under item 0. They are not gate passes.

[Initialization fingerprints](fresh-initializations.json) confirm twelve
distinct fresh parameter initializations; the three specified replay RNG
states match their original captures exactly.

The [six retained failures](regression-failures.md) have their assertions
unchanged. Three expect binding to a zero-score candidate; three expect
count two where B's repaired merge preserves three occurrences. Alec
subsequently authorized the six ports described below; these frozen-source
failures remain failures. No incorrect fresh x binding occurred. The keep probes retain
[97 bind departures](bind-departures.json), each selecting its recorded
next option, with both trials' R/A and keep decisions. All 97 tie on R + A
and keep greedy. All checked x sites
remain open, so no trained-closure claim follows from the twelve passes.

[C's paired evidence](C-pair.json) includes the exact RNG hashes. The
original decomposition snapshots are unchanged (SHA256 `dbd7315f…367d34`
and `f1d141ab…63b8f`), at seeds 4199722147 and 2428440506, budgets 512 and
768, with zero premise budget. Both arms restore every surviving parameter
from those snapshots; their complete constructor receipts and companion
question certificates are retained.

Two runner events are explicit. Forward draw six's pytest worker wrote a
passing result, certificate and complete keep probe; its outer runner
stopped before updating the final summary. The original running record
and [recovery receipt](fresh-final-forward-6-recovery.json) remain; no draw
was replaced. The first eight-file invocation failed importing its
observer before collection. `native-final` retains that failure;
`native-final-02` uses the unchanged observer at the corrected import path.
[Harness recovery](harness-recovery.json) also records relocation of only
this run's two new decomposition metric files from the prior observer's
output directory. No historical result was changed.

## Review and publication

The [completed review](claude-review.md) says not to land yet. It accepts
the credit path, the first merge, isolation, and the bootstrap repair, and
identifies the later rebind's whole-owner mutation and the nested-role
answer view as mechanism defects. The subsequent §20 decision adopts the kind fallback only when the
antecedent has no kind, while preserving every non-null antecedent kind. Publication remains
pending repair, gates on the next frozen source, and review: both mains,
WikiOracle's submodule bump, the October 20 documents, and the decided
naming sweep (old LTM1 → LTM0, old LTM2 → LTM1, stale LTM3 retired).
No second full sweep.

## Subsequent decisions and rebinding research (2026-10-21)

Alec authorized the three count checks from 2 to 3 and a positive candidate
score in the three zero-score fixtures, with the other semantic assertions
unchanged. He also authorized leaving x's resolver live in both pending
premise certificates while retaining `open_names=('x',)` at formation;
the decomposition fixture's `what` seam stays. These are authorizations
for the next source, not changes to the frozen results above.

On rebinding, Alec suggested reconsidering all affected binding choices,
then requested a literature check before deciding the mechanism. The
following is a research recommendation, not an adopted amendment to
§10.3 or §19. No rebind production code changed during this investigation.

- Haug, [*Partial Dynamic Semantics for Anaphora*](https://academic.oup.com/jos/article/31/4/457/2937213)
  (2014, §2.4), separates the enduring discourse content from revisable
  anaphoric resolution. His example requires a later sentence to change
  an earlier pronoun's antecedent. This supports preserving the occurrence
  evidence while revising its interpretation; it does not prescribe an
  implementation of revision.
- Recanati, [*Mental Files: Replies to my Critics*](https://www.researchgate.net/publication/281157492_Mental_Files_Replies_to_my_Critics)
  (2013, §4, pp. 225–226), discusses links that let information participate
  jointly in reasoning, and separating a composite file again after a
  mistaken identity judgment. Preserved information channels make the
  separation possible. This is a philosophical account; linking versus
  merging is debated, as his later [2021 lecture](https://www.college-de-france.fr/en/agenda/lecture/mental-files-continued/linking-or-merging)
  also makes explicit.
- Lee, He and Zettlemoyer, [*Higher-Order Coreference Resolution with
  Coarse-to-Fine Inference*](https://aclanthology.org/N18-2108/)
  (2018, §§2–3), represent individual antecedent choices and recover
  clusters through their connections. They refine choices using other
  mentions' information. Their zero-score dummy is also relevant to D.
  This is a document inference model, not an online MTM rebind algorithm.
- Doyle, [*Truth Maintenance Systems for Problem Solving*](https://www.ijcai.org/Proceedings/77-1/Papers/035.pdf)
  (1977), explicitly requires reconsidering beliefs dependent on a
  changed belief, using recorded justifications. This is the clearest
  algorithmic precedent for Alec's affected-choices suggestion, although
  applying it to learned bind choices is our design inference.
- Tauer et al., [*An incremental graph-partitioning algorithm for entity
  resolution*](https://experts.illinois.edu/en/publications/an-incremental-graph-partitioning-algorithm-for-entity-resolution/)
  (2019), allow new references and revised pairwise evidence to change
  earlier partitions. This establishes a computational precedent for
  incremental revision; it does not establish the right scoring or
  revision budget for this model.

The recommendation is to preserve individual binding choices and their
dependencies, then reconsider the affected choices when an arrival changes
their evidence. Example: `x → B`, `y → x`, and `z → B`. Revising x toward C
requires reconsidering y's interpretation; z's independent support for B
may remain valid. Mere membership in B does not distinguish these cases.
The current pool stores that membership but loses the individual links
when entries merge. A selected-antecedent pointer alone may also be
insufficient: a choice can change when a competing candidate's pooled
content changes. The affected set must account for those scoring inputs.

This suggests a bounded record of binding decisions outside the immutable
rows, with token membership maintained from the retained decisions. Exact
dependency scope, eviction, revision scheduling, count semantics, and
native/unpooled outcomes still need a concrete contract. The existing
frame-access guard, chooser and dummy, score-function credit, and kept-close
write boundary remain constraints; the literature does not authorize new
objectives, forcing, or stored working vectors.
