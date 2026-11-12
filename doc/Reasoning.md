# Reasoning and thinking

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are built (accepted as implemented, Alec, 2026-11-01).


Item 6.2 closed on 2026-10-09 with its mechanism complete and decomposition
demonstrated; whether the model learns to chain is measured under item 0
(`MM_math_chain`).

A question, like any sentence, specifies a region in conceptual space: the
roles it fixes and its free variables. A free variable is an open referent or
relation. A declarative carries its source's evidence; a question carries none
and stands at `(0,0)`, which is ignorance, not an evidence slot, so an
episode's goal is to bind the free variables and find or derive evidence.
Truth comes two ways. Direct truth is a content match: `query` returns the best
row in the region with its own pair and trust, and the match binds the free
variable. Derived truth applies a premise to a rule (an implication, an
equality or a parthood), and the conclusion's pair is the minimum along the
path in each lane. The two lanes are never netted, so a region reached both
ways keeps both and prompts refinement.

Every completed field is offered one root thought episode at its close
(`ThoughtClosing.close`). For a field with no open slot the chooser's first
menu holds the catalogue candidates and `None`; choosing `None` is not
thinking: it spends no work, writes no record and returns the bound field
unchanged. A field with a free variable has no `None` in its first menu while
work remains, so it is always thought about. Whether to think about a closed
sentence is thereby learned from the parse's form; wh-words and punctuation
are evidence for the chooser only through the parse. Nested `ask` shares the
root's `attentionBudget`.

Openness is read from the row's shape as well as its annotation
(`ThoughtReferences.open_slots`): a relational meaning with an unfilled operand
role is open without an `_open_references` entry, and a pending row returned by
`query` keeps its open reference. A meaning with an unfilled role never reaches
the answer readers as a bound view. An elided participant also opens a role,
detected from the verb's history rather than from a word class: when a sentence
closes with a predicate whose subject or object role is empty, and MTM's event
histories show that predicate with a participant in that role at least the
verbs' recurrence threshold of times (four), the journal opens a referent there
(`ReferenceClose.expected_roles`). Its expected content is the coordinatewise
maximum of the descriptions of the participants that filled the role before
(`expected_content`). The open role then enters thought like a question's;
`bind` may fill it from the situation or leave it open.

While the episode's root has an open slot, the thought menu offers only `ask`,
`query`, `intuit`, `infer`, `not`, `isTrue`, `bind` and `exist`, and an `equal`
whose two operands a query has already supplied (`ThoughtStream.candidates`).
Every path to an answer is therefore a sequence of sound steps, and the chooser
decides which premises and rules to use. `conclude` requires every free
variable bound, unless work is spent, `thoughtSearchExhaustion` completed
queries have found no candidate, or the menu holds only the self-inverse `not`.
At an exhausted search, a source-supported forward name is provisionally
minted; a question's free variable remains open on its stored row. A later
copula can bind directly to that row's addressed column and fill it.

| Symbolic face | Conceptual face |
| --- | --- |
| `isTrue(P)`: ended row's pair, trust and witnesses | `exist(P)`: conceptual presence and its pair |
| `isPart(a,b)`: LTM and taxonomy parthood | `part(a,b)`: containment content and pair |
| `isEqual(a,b)`: LTM1 equality relations (membership in a singleton) and references | `equal(a,b)`: identity of codes and pair |
| `infer(premise, rule)`: the rule chunk's consequent from the premise chunk by identity, membership, equality or instantiation, with the path-minimum pair, or its antecedent opened as a subgoal (2026-10-29) | `implies(P,Q)`: containment of regions and pair |

`bind` (a selected reference site, evidence `reference`) and `inferBackward`
(`infer`'s converse spelling, the subgoal face) complete the catalogue with
`intuit`, the order-zero proposal described under
[Intuition, understanding and thought](#intuition-understanding-and-thought-alec-2026-10-30):
`intuit` proposes and `infer` selects.

`infer(premise, rule)` reads two held chunks: a premise and a recalled rule (an
implication, a part or kind row, or an equality). Forward, the premise matches
the rule's antecedent by identity, by membership through a bounded taxonomy
proof, by equality through an `isEqual` proof, or by instantiation of one role
of a premise with several; the match kind is recorded as the record's `reason`.
The instantiated consequent takes the pair
$(\min(p^+, m^+, r^+), \min(p^+, m^+, r^-))$, where $p$ is the premise's pair,
$r$ the rule's lanes and $m^+$ the positive support of the membership or
equality proof ($m^+ = 1$ for a direct identity), so a weak edge caps both
lanes. If a chunk with the same references is already held, its pair is raised
by the per-lane maximum and no chunk is added. Backward, the goal matches the
consequent and the antecedent is written as a question chunk with an open
referent, pair $(0, 0)$ and evidence kind `subgoal`; `query` or `ask` fills it,
and a forward `infer` over the filled chunk concludes with its pair. A question
claims nothing. `ThoughtInference.replay_pair` recomputes the pair from the
recorded supports.

No operation duplicates what calling an existing one gives. `not` applied to
the premise before `infer` gives the contrapositive (the rule is read from
consequent to antecedent and the lanes exchange), so there is no `refute`; an
equality row used as the rule substitutes one term for the other, so there is
no `substitute`; and an `isImplied` would be `infer` forward with the rule
fixed, so there is none. Transitivity over part or kind rows is two forward
steps. A rule over relations matches by kind and by its two references'
identity or membership; there is no further unification.

Time adds no relation kind: a relation's readings come only from the types of
its two referents (conditional, attribution, definition, subsumption, equality,
part; `LTM1.readings`). Before and after are read from the events themselves,
an identity's timeline or the rows' stamps, and cause is a conditional between
two situations whose stamps run from antecedent to consequent:
`relation_evidence` skips an implication whose antecedent was stamped after its
consequent. Tense adds no time either. `.when` is model time, the moment a
sentence was read; a phase question is answered by content entailment over the
event's recorded case, never by a stamp (`TimelineRead.case_coverage`), and a
fragment of the requested content proves neither completion nor its negation.

Equality can also be witnessed by evaluation. When `query` or `ask` fills an
equality goal whose two values MTM recorded as event outputs,
`IdeaEvaluation.isomorphic` checks at read time whether one declared fold kernel
and a common context explain both outputs: the kernel's inverse proposes a
context from each recorded before and after value, the meet and join of the two
contexts are tried as hypotheses, and a hypothesis counts only if both outputs
recompose within the native decoder's tolerance. A lexical value with no
recorded event is searched over its whole activation interval by a coarse scan
and golden-section refinement. Every inverse and forward check is charged to
the work meter, and nothing is stored.

**Two routes to a truth.** Whether a property pervades an object needs no
operator of its own. A part or property question (`part`, or the taxonomy's
`isPart`) can be answered implicitly in order-zero space or explicitly by the
chain, and both may run within the budget. The implicit route is pervasion
(`Intuition.pervasion`, which `intuit` tries before its rules): the knowing is
narrowed to the subject's parts, read from its Part edges and recorded
constituents and kept only where the subject's content contains the part's,
each weighted by its edge's evidence and joined with the wholes that contain the
subject and with recalled rules fired at unit exponents, so learned proposal
strength never becomes truth. For each part the property's coverage is read in
each lane, and the answer's pair is the minimum over the parts in each lane
(`Intuition.pervade`): pervasion is complete only when every part carries the
property, and a mixed object (furry outside, not inside) keeps both lanes
rather than averaging them. The result carries evidence kind `pervasion`. The
explicit route is the chain: `query` recalls rows from LTM1 and the taxonomy,
`infer` follows them, and the pair is the minimum along the path. At the close
`Intuition.judge` joins an explicit proof and a pervasion proof of the same
proposition by the per-lane maximum: agreement fills one lane, and disagreement
fills both, the corner that prompts refinement.

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

Before a field with no open slot is offered its episode, the close may conceive
one absence: when the closing expectation image's omission of a predicted
symbol exceeds `expectationOmissionThreshold` (0.5), it writes `not X`, an
absence inference with that confidence against (`ThoughtClosing.absence`). The
four-corner refinement policy remains future work.

The grammar's chooser makes the choices. Thinking is part of each whole path
([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)):
an explore path that departs at a thought decision restores the exploit
episode's state there, takes another action and completes its suffix, and each
stream keeps its cheapest complete path in reconstruction plus supplied-answer
cost (R + A). A departure's credit is the proposal-corrected score function
$p(a)\,\Delta C$ on that difference; exact ties move no policy weight. Work
bounds the walk but is not a cost in the comparison. Expectation trains only
its own predictor. The presented answer is the filled row; there is no separate
comparison reader. The REINFORCE/EMA path and `selectedThoughtPolicyWeight` are
removed.

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

Forgetting values ideas and chain rows by their own worth, row by row in each
store (`Forgetting.retention_values` for LTM0, `forget_parts` for LTM1): the larger
evidence pole as trust, the recorded surprise as utility (missing surprise
counts as unknown, so high), a third term, and age in model time since the
row's stamp. For an LTM0 row the third term is the dissonance flag or the
optional coverage term; for an LTM1 row it is its access count. Cheap
dissonance compares opposing sources in matching scope and order, penalizing
the weaker source; heterogeneous content within a single both-valued row is not
a disagreement between sources. The optional coverage term measures loss of
positive catuskoti coverage on removal. No pass changes stored evidence, and
none checks whether a row is deducible from others (Alec, 2026-11-01:
"Forgetting had been checking if rows were easily deductible; let's delete
that requirement.").

A row protects everything it references, transitively
(`LTM1.protected_references`), and the dependency set is frozen before a
forgetting pass: removing a parent releases its operands only for the following
pass. Live thought results and pending expectation sources protect their
addressed evidence throughout the transaction. Source wording kept beside a row
is the first thing forgotten. Cleanup compares tagged references, so equal
integers in different reference domains never remove unrelated entries.

## Intuition, understanding and thought (Alec, 2026-10-30)

In the dual-process terms of 2026-11-09, expectation and `intuit` are System 1,
and `infer`'s selection in the serial workspace, with `query`'s recall, is
System 2
([Architecture](Architecture.md#dual-process-theory-as-a-design-consideration)).
A `query` by time alone reinstates the order-zero concepts kept for that moment
as the attended field, so thought can reason over a past sentence as an
experience lived before
([Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11)).

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

*The calculus.* At thought time `intuit` reads the knowing $K$: the question's
order-zero content, the working values MTM prepends for the stream and, within
a third of the remaining work, the order-zero concepts active in the last
observed knowing field. Its rules are the LTM1 rows that reference the concepts
the predicted field holds active, taken in descending order of predicted
activation, with nonzero evidence and within the work meter
(`ThoughtLTMCapability.intuition_rules`). For a rule with lanes $(t^+, t^-)$,
antecedent literals $\ell_i$ (a negated literal exchanges its lanes before the
meet) and consequent $c$, the coverage is
$\kappa = \sum \max_i \min(K, \ell_i)^{\pi_i} / \sum \max_i \ell_i$, the firing
is $f = \min(\kappa, (t^+, t^-))$, and the term's field is
$\sigma(\max(\min(c, f^+), \min(\bar c, f^-)))$, where $\bar c$ exchanges the
lanes and $\sigma(x) = 1 - (1 - x)^{s}$. The slot's field is the maximum over
the fired terms. The exponents $\pi_i$ (one per rule literal) and $s$ (one per
rule) are output's parameters in two sparse banks of `intuitionCapacity`
addresses, initialized to one, where the map is exactly the max-min disjunctive
normal form; an overrun fails before a partial term is reserved. Each fired rule
becomes one `infer` candidate carrying its firing pair. Each consequent concept
receives the knowing deposit $\sum \min(\text{field}, \text{code}) / \sum \text{code}$
per lane, and `write_knowing` re-evaluates the selected proof live and joins it
into the field by maximum at the existing brackets. When `infer` selects a
firing, the pair is recomputed at unit exponents, $\min(\kappa_1, (t^+, t^-))$,
so learned weights shape what is proposed and never what is accepted. An
`intuit` cut off by the budget returns no content and leaves the goal and STM
unchanged.

`intuit` maps order zero to order zero. Its knowing is the order-zero field, a
rule's two sides enter as their order-zero content, and the consequent's
content is joined back into that field. Higher orders enter only through their
order-zero constituents: a higher-order concept as the join of its
constituents' content, an individual as its working value. So `intuit` cannot
tell two individuals apart except by their order-zero differences; individuation
is the serial grammar's work.

*Proposal and pick.* The breadth of the parallel pass and the narrowness of the
serial pick are two learners. Which fired candidate matters most for the focus
is not computed inside `intuit`: the grammar's operation scorer, the one compose
uses (`OperationSelectionLayer.thought_logits`), scores each `infer` candidate
from the focus and the candidate together, the candidate's firing pair among its
inputs, to one logit. The Sigma and Pi charts belong to combination, which must
stay monotone and bounded in the cube; a choice needs an unbounded logit, which
the softmax turns into the probability that the score function credits.

*What learns.* Memory is evidence, not a parameter. A row's lanes are what its
closes wrote: no answer loss, gradient or chooser credit changes a row or its
lanes, and a contrary statement observed later adds its own witness through an
ordinary close. An STM chunk has no parameters either; it holds references and
a detached working value, so what learns is what the references point at.
`intuit` improves as the things around it improve, each by its own owner:
memory (which rules exist and how true they are, written at kept closes),
meaning (the conceptual definitions and the taxonomy's memberships, which decide
how well an antecedent matches the knowing), place (which rules the predicted
field brings into reach), choice (the chooser, which learns when to call
`intuit`, which candidate `infer` takes, which premises to keep and when to
conclude: the path, never the truth of a step), and `intuit`'s own
output-owned weights. A wrong rule is not corrected in memory; it stops
mattering as the concepts it is stated over learn, as the chooser learns not to
take it, and as forgetting removes it when it goes unused.

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
  or pervasion, computed from evidence. The design is
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
