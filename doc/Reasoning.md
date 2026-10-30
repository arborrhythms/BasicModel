# Reasoning and thinking

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Expectation is a **gain field** `c = m⊙a⊙o` with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec; [thinking spec §§16, 22](plans/2026-10-27-item2-5-thinking.md), [2.7 record §42](plans/2026-10-27-item2-7-corpus.md)): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are not built.


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
| `infer(premise, rule)`: the rule chunk's consequent from the premise chunk by identity, membership, equality or instantiation, with the path-minimum pair, or its antecedent opened as a subgoal (item 2.5 round A, 2026-10-29) | `implies(P,Q)`: containment of regions and pair |

`bind` (a selected reference site, evidence `reference`) and `inferBackward`
(`infer`'s converse spelling, the subgoal face) complete the catalogue;
`intuit` is declared, its executor deferred and its menu entry masked until
item 2.5 round B.

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

*Not in the code at `3718d9d0dc` (audit 2026-10-30): `Forgetting.retention_values`
uses the recorded surprise, the same-scope dissonance flag and the optional
coverage term; no deducibility pass or deletion certificate exists. Whether
to build it or retire the statement is Alec's call.*

Idea utility uses the recorded expectation discrepancy, treating missing
surprise as unknown/high utility. Cheap dissonance compares opposing sources
in matching scope and order, penalizing the weaker source; heterogeneous
content within a single both-valued row is not a disagreement between sources.
The optional coverage term measures loss of positive catuṣkoṭi coverage on
removal. Live thought results and pending expectation sources protect their
addressed evidence throughout the transaction.

## Intuition, understanding and thought (Alec, 2026-10-30; item 2.5 round B's design, not yet built)

Alec: "Intuition is beyond understanding, it is conditioned by desired
output, and thoughts are a distillation of intuition."

- *Understanding* is the conceptual definitions, trained by
  reconstruction: they explain the input.
- *Expectation* is bottom-up. `ActivationPredictor` predicts the next
  concept state from the current field and STM, and that prediction is
  also the spread of activation that primes attention, the chooser and
  retrieval. It learns only from its own error against the detached next
  state.
- *Intuition* is top-down. `intuit()` is a Sigma over Pi with negation at
  the literals and its own learned weights. It reads the knowing but is
  trained only by its use in output, so it is biased toward doing
  something with what is understood (problem solving). The relations and
  ideas that fired are pulled into an STM slot as candidates.
- *Thought* is `infer`'s selection from those candidates in the serial
  workspace (the eight STM chunks), its trust pair computed from evidence,
  never learned; an intuition is provisional until `infer` selects it.

In one line: understanding explains the input, intuition is understanding
put to use, and thought is intuition that has been checked.

*Where it fits.*

- *Bottom-up expectation as priming.* `ActivationPredictor`'s spread is
  spreading activation (Collins and Loftus) read as prediction, as in
  predictive processing (Clark, Friston): priming as anticipation of the
  next state.
- *Intuition beyond understanding, shaped by goals.* Expert intuition is
  recognition of what to do (Simon; Klein's recognition-primed decisions),
  learned from feedback on outcomes rather than from explaining the input;
  Polanyi's tacit knowing ("we know more than we can tell") is the
  "beyond understanding" part.
- *Two kinds of fast thinking.* Glöckner and Witteman separate associative
  intuition from constructive, goal-driven intuition; that is the split
  between the predictor and `intuit`.
- *Thought distilled from intuition.* Poincaré and Hadamard describe
  discovery as unconscious combination offering candidates, a sense of fit
  picking one, and conscious work verifying it; global workspace theory
  (Baars, Dehaene) has the same shape, many parallel processes and a
  narrow serial workspace. The eight STM chunks and `infer`'s pick from
  the candidate menu are that workspace.

*Where it departs.*

- *Rationalization.* Intuition shaped by desired output is also how
  motivated reasoning works: in Haidt's social intuitionist model
  intuition decides and reasoning follows as justification. Here `intuit`
  proposes but an accepted answer's trust comes only from `infer`, recall
  or pervasion, computed from evidence (§3.6, §3.7). The design is
  normative rather than descriptive: thought distills intuition and then
  checks it, the step people tend to skip. Keep that.
- *Not every thought comes from intuition.* Reasoning also proceeds step
  by step from recall with no hunch first; the menu allows it through
  `query` and `infer` over recalled rules without `intuit`. "Thoughts
  distill intuition" is one route, not all of them.
- *Intuition is only as good as its feedback.* Intuition is reliable
  where feedback is accurate and prompt, a kind environment, and
  misleading where it is not (Hogarth). Because only output trains
  `intuit`, its quality tracks the supplied answers': the supplied-answer
  lessons are its kind environment, and wrong or noisy answers would
  train confident bad hunches.

Record: the thinking spec, [plans/2026-10-27-item2-5-thinking.md](plans/2026-10-27-item2-5-thinking.md)
§22.3 (the decision) and §22.1.
