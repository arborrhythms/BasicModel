# Reasoning and thinking

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Expectation is a **gain field** `c = m⊙a⊙o` with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.


The current contract is [item 6.2](specs/2026-10-07-thinking.md), following the
accepted 6.5 mechanism. The 6.2 mechanism and repairs are accepted and landed;
the [repair receipt](benchmarks/2026-10-07-item6-2-repair/README.md) records that
acceptance. Learned chaining is the separate
[MM_math_chain measurement](benchmarks/2026-10-07-math-chain/README.md), stopped
for review after all thirty attempts failed before completing an epoch.
Its sweep and mechanism gates pass, but no held-out learning result is
available. Configured-run completion does not establish learning.
The [second math-chain repair](benchmarks/2026-10-08-math-chain-repair-2/README.md)
is in development under §14; declaration awaits its complete epoch certificate.

A free variable is an open referent or relation. `(0,0)` is ignorance, not
an evidence slot. Wh-words and punctuation are evidence for the chooser.
A closing with a free variable, or an interrogative region without evidence,
enters `run_selected_thought`; nested `ask` shares the root's `attentionBudget`.
`conclude` requires bound variables and evidence, unless work is spent or
`thoughtSearchExhaustion` completed queries have found no candidate. At an
exhausted search, a source-supported forward name is provisionally minted;
a question's free variable remains open on its stored row. A later copula
can bind directly to that row's addressed column and fill it.

| Symbolic face | Conceptual face |
| --- | --- |
| `isTrue(P)`: ended row's pair, trust and witnesses | `exist(P)`: conceptual presence and its pair |
| `isPart(a,b)`: LTM and taxonomy parthood | `part(a,b)`: containment content and pair |
| `isEqual(a,b)`: DEF rows and references | `equal(a,b)`: identity of codes and pair |
| `isImplied(P,Q)`: implication with antecedent evidence | `implies(P,Q)`: containment of regions and pair |

`query(pattern)` returns the best matching row. `ask(row)` attempts to fill
an open reference, using queries and nested questions. `not` exchanges poles
of a pair or meaning. Gain changes the next sentence's global expectation.
No thought operation returns a scalar. Neither trust nor lack of support is
silently substituted for an evidence pole.

Every result becomes a serial slot with its content, pair, witnesses and
producing operation. Chains bind those slots: the `a < b`, `b < c` fixture
performs two queries and concludes `a < c` with both source references.
Modus ponens requires an antecedent witness as well as the implication.
Conclusions are LTM `inference` rows, addressed by document/turn, ordinal and
content. No second semantic store or planner stack is introduced.

A bound declarative opens no episode. Its uncancelled expectation image can
prompt `not X`, an absence inference with image confidence against. The
four-corner refinement policy remains future work.

The existing grammar scorer makes the choices. Compose's paired rule,
`K · R · p(a_dep) · ΔC`, compares reconstruction plus the supplied answer.
The thought walk also pays its own metered work; that cost does not enter
the sentence comparison. Expectation trains its predictors through their
owner-step registry. Exact ties move no policy weight. The comparison
reader scores the answer term; the presented answer is the filled row.
The REINFORCE/EMA path and `selectedThoughtPolicyWeight` are removed.

Thought `what` raises with `ask`; LTM `what`/`lookup` raise with `query`.
`chunk` becomes `synthesize`, inverse `analyze`. `quantize` is removed;
symbolize/conceptualize remain future work. `arma`/`expect` are removed from
thought in favour of `<sentenceExpectation>` and gain. `true` raises with
`isTrue`. `thinkingBudget` and `selectedThoughtBudget` fail at load; use
`attentionBudget`.

Boundary admission, typed capability views and work limits are documented in
[QueryContracts](QueryContracts.md). Replay, nesting and occurrence retention
are in [ThoughtHistory](ThoughtHistory.md). Numerical proof utilities in
`TruthGroundedReasoner` remain diagnostics; production reasoning uses the
ordinary grammar controller and native row evidence.

## Retention utility (Item 5)

[Forgetting](specs/2026-09-16-forgetting.md) uses one-step deducibility as a
retention score. A positive part edge can be omitted when two same-scope
positive edges supply its transitive proof, with at least its source trust
and positive evidence. Modus ponens needs both the implication and its actual
antecedent. Negative or intrinsically both-valued evidence does not supply
these deletion certificates. A removed conclusion pins a complete surviving
proof; losing a proof raises an unselected candidate's utility. This pass does
not recursively prove claims or change stored evidence.

Idea utility uses the recorded expectation discrepancy, treating missing
surprise as unknown/high utility. Cheap dissonance compares opposing sources
in matching scope and order, penalizing the weaker source; heterogeneous
content within a single both-valued row is not a disagreement between sources.
The optional coverage term measures loss of positive catuṣkoṭi coverage on
removal. Live thought results and pending expectation sources protect their
addressed evidence throughout the transaction.
