# Sentence understanding, thinking and concept-to-concept prediction

> **Status:** consolidated normative target and execution plan, revised
> **2026-09-15; unfinished**. This is the canonical document combining the
> September 11 sentence-boundary thinking specification and the revised
> September 15 next-sentence production-objective plan. The older
> [thinking filename](../specs/2026-09-11-sentence-boundary-thinking.md) retains
> section redirects. No implementation or learned-reasoning completion is
> implied by consolidation.
>
> Supersedes the shared-controller rule in
> [What §8](../specs/2026-07-27-teaching-modes-and-next-iteration.md#8-grammar-and-the-mind-boundary)
> and [WhatSpacetimeDesign §8](../WhatSpacetimeDesign.md#8-context-for-the-grammar-chooser),
> and the lexical ANSWER/OPEN policy of the
> [September 9 thinking spec](../specs/2026-09-09-mathematical-thinking.md),
> where they conflict below. Internal thinking uses ordinary thoughts with
> context levels and serial LIFO return, replacing Q/A pairing and parity
> as its target contract. Preserve target isolation, reconstruction/output
> separation, append-only history, bounded episodes and declared credit.
> Existing parity-based runtime behavior remains legacy until migration is
> verified. The [Symbol Firewall](../SymbolFirewall.md) remains governing.
>
> **September 19 correction:** the per-model boundary action list is the
> explicit lower-case `<thought>` grammar section, physically between
> `<compose>` and `<generate>`. Matching faces share one canonical operator
> identity but retain their own signatures and capability contexts. Earlier
> references to a `<Queries>` catalogue are historical unless this plan has
> been updated in place; the legacy spelling is rejected rather than retained
> as a parallel source of authority.

Read §§1–5 for the thinking contract, §6 for reconstruction requirements,
§7 for the earlier source review, §8 for the consolidated production objective,
§9 for conflicts and open decisions, and §10 for implementation order.
The approved default-on expectation/observation cycle is specified in §8.7.
The consolidation's source baseline is basicmodel
`d541c94a78c46756f074ab6b9266fd9c81b54de4`. The September 15 joint-learning
revision implements §8.4 and the training-gate/context corrections identified
there; it leaves the broader thinking and production rollout unfinished.
Code citations outside the historical reviews have been updated for this
revision. Historical reviews name their own baseline; use the cited function
names to locate code after later edits.

## 1. Two phases, two controllers

| Phase / path | Responsibility | Grammar |
|---|---|---|
| Within-sentence `forward()` | Construct conceptual meaning from input | `<compose>` |
| Within-sentence `reverseReconstruct()` | Reconstruct the input by reversing its compose derivation | `<compose>` and its inverse operations, not a free `<generate>` route |
| Within-sentence `reverseOutput()` | Express a resolved idea as output, including a linguistic subgoal | `<generate>` |
| Between-sentence thinking | Select the next thought action, refinement, nested context or conclusion | explicit `<thought>` plus level/termination control |

The **grammar MLP** controls sentence composition/realization. A separate
**thought/query MLP** selects boundary actions; it is a chooser, not an
unrestricted next-state generator. Its context includes the completed idea,
active/root questions, semantic STM/LTM reads, candidate operands, temporal and
spatial scope, context level and closure pressure. Scalar memory summaries alone
are insufficient. Retain `transformChooserHidden` / `transformChooserDepth`
and `whatThinkingHidden` / `whatThinkingDepth` as independent capacity settings;
document actual input dimensions and checkpoint changes when implemented.

Use the **completed structured idea plus bounded attended context**, not a
variable-length flattening of raw STM code IDs. Codes address symbol-attached
semantic payloads; their arbitrary numeric values are not semantic features.
For a relational idea, preserve `[NP1, VP, NP2]` and role-presence masks, mode,
bindings and scope. A root representation may accompany it, but cannot replace
it: the historical consolidated root reduction yields NP1 alone
([consolidated root reduction](../../bin/Layers.py#L10524)). Essential idea,
active/root-question and candidate-argument roles must not compete with
incidental context for top-k inclusion.

Specify the input shapes symbolically: relational slots `[B, 3, D]`, attended
context `[B, K, Dctx]` with validity/source/role metadata, and typed candidate
features. The wider retrieval pool `R >= K`, attended field width `K`, payload
widths and STM capacity are distinct settings. Variable live counts use masks
or a shared role-aware item encoder; document the adapter's actual MLP input
width, mandatory-role overhead and parameter count when implemented. Reuse the
field-width retrieval/attention work for optional context; do not silently
equate its widths with STM depth. The proposed retrieve-16/attend-8 operating
point is not a normative default.

The runtime MUST mask query execution inside sentence composition,
reconstruction and generation. A boundary is a committed conceptual sentence
or completed internal thought, not every word/fold or punctuation token;
internal thoughts need not be spoken. Boundary readiness is per batch row.
The runtime order is understanding with its enabled input reconstruction,
then reasoning, then response generation (§7.1). Resolve queries before
entering `reverseOutput()`; its setup and generate derivation must not select
or execute another thought query. The identified compose parse supervises
input reconstruction only. The output owns its generate derivation.

## 2. One grammatical deep structure, multiple surface forms

A linguistic question and an internally selected query MUST construct the
**same grammatical deep structure**, not separate sentence and tool-call
representations. A binary relational idea has canonical conceptual slots
`[NP1, VP, NP2]`: two operands with the specialized relational VP in the
**second (middle) slot**. For example, `isPart` identifies a predefined VP
whose conceptual encoding occupies that slot; it is not merely an operation
name attached to an otherwise unstructured request.

Grammatical interrogative/assertive meaning, polarity, open variables, support
and scope accompany this structure without replacing it. No administrative
Q/A record label is required. Asking a question does not assert its
relation as true. An answer to a relational question retains the relation and
its operands, with resolved bindings and actual truth/support; even a surface
"yes" or "no" must have that recoverable deep structure. Unary and
value-returning query signatures declare their absent, open and result roles
explicitly, without fabricating operand concepts or discarding the VP.

Extend/adapt `WhatQuestion` / `QuerySpec` as interfaces to this structure, not
as an independent semantic record or another planner/frame hierarchy. An
executor's operation and operands MUST be derived from the structured idea.
Its attached context carries:

- bindings/open variables and expected result kind (truth, concept, set or code);
- relation domain and semantic where/when scope;
- episode/context level and occurrence/source provenance, plus a surface
  derivation when present.

Semantic references point to ideas and occurrences in existing memory; they
are not a separate dependency graph or mandatory parent/return pointers.
The active root and suspended contexts are recoverable from the levelled
history defined in §3. Internal requests carry no answer target and cannot
manufacture dataset addresses. Existing `WhatQuestion.where` / `offset` keep
their presentation-index meanings; semantic scope is explicit, separate metadata.

**Linguistic route:** `forward()` uses `<compose>` to construct the operands
and the specialized VP from the sentence. Interrogative structure marks the
idea as a question; no keyword/regex dispatch may substitute for comprehension.
The grammar/lexicon MUST support surface realization and recognition of these
VPs, including paraphrases and declared argument permutations. For example:

- "The wheel is a part of the bicycle" uses `[wheel, isPart, bicycle]`.
- "The bicycle has a wheel", in the part-whole sense of "has", uses the same
  canonical structure by reversing the surface argument order.
- Their interrogative forms preserve that structure and mark it as a question.

Canonical operand order is semantic, not necessarily surface word order. The
compose derivation retains the surface ordering needed by
`reverseReconstruct()`; `reverseOutput()` can select a valid realization
through `<generate>`. Other senses of "has" (ownership, experience, etc.)
must not silently become parthood. Wording alone grants no extra authority.

**Internal query route:** the thought chooser selects a specialized VP and
grounded operands/open roles, constructing the same typed grammatical idea.
`isPart(wheel_ref, bicycle_ref)` is only shorthand for forming the question
`[wheel, isPart, bicycle]`, never permission to bypass its representation or
LTM recording. Surface realization is a required capability, but execution
does not require a verbalize/reparse round trip.

**Representation is not execution:** a query-capable VP can be composed,
reconstructed or generated within a sentence without invoking its executor.
Only a completed, selected question at a boundary activates the corresponding
explicit `<thought>` declaration. Its grammatical VP identity and its boundary execution
signature must be linked explicitly. Assertions using that VP remain
assertions, not implicit query calls. The current grammar's "build no
structure" description must not mean queries lack grammatical meaning: the
executor is inactive during parsing, but its associated VP is representable.

There is **one semantic VP per relation/domain, with grammar faces as
declared**: structural compose/generate faces and an optional checked
`<thought>` face. There is no second, independently meaningful VP that merely
names a tool. A model uses one canonical spelling consistently across its
faces; a declared `whole` converse carries its explicit argument permutation.
Concrete rule/executor implementations may differ, but role identity must
agree. Omitting an operator from `<thought>` keeps it out of boundary
execution while leaving any declared structural face pure; `query="false"`
is retired and cannot express that policy.
Interrogative versus assertive mode belongs to the completed idea, not a
duplicate relation VP.

Not every question reduces to one primitive. Add **`what(Q)`** to `<thought>`
as the general conceptual-subgoal request. It preserves and evaluates
the full deep structure of question `Q` through this same boundary loop; later
choices may select more subgoals or explicit tools. It is not a hidden solver
or a call to `Data.what()`.
Keep the existing `query(X,Y)` lookup distinct; do not silently redefine it.

Meronomy and knowledge taxonomy are both queryable, but their evidence is not
interchangeable. Signatures/scopes MUST distinguish structural part/whole
containment, taxonomic subsumption and asserted world relations. A word's
presence in a sentence, a fold-tree edge or geometric similarity does not prove
a world assertion. The registry must expose supported domains and reject
unsupported combinations, not silently treat every inclusion as `isPart`.

**Query evidence sources (Alec, September 16).** `PartOf` consults the
**conceptual taxonomy** for the described conceptual relationship. Its
boundary aliases, including `isPart`, must declare this domain explicitly.
A perceptual-mereonomy traversal is a separately typed request; neither the
input parse tree nor concept-vector overlap can substitute for taxonomic
evidence. Asserted world relations remain facts to check in LTM, rather than
being certified by either structural hierarchy.

**Truth and reference (Alec's clarification, September 16): a concept is true
if its referent exists; this is moderated by degree of truth.** `Exist` uses
LTM facts to assess the existence of the **full conceptual description's
referent**, returning degree of truth and its supporting evidence rather than
reducing the result to a binary hit/miss. Preserve its occupied NP/VP roles,
bindings and scope when matching;
finding the subject's word, a concept row, or an NP1 match alone is insufficient.
Return the matching fact's provenance and evidential status. Partial, negative
and conflicting support must retain their degrees. A missing match is
unknown, not proof of falsity; contradictory or explicitly negative evidence
must retain its status. Questions and predicted estimates cannot certify their
own existence as facts about the described world. The legacy `exist -> isTrue`
alias must not bypass this lookup through a generic truth/activation shortcut.

Tense queries and tense-based reasoning are **deferred**. No new tense executor
or automatic temporal claim is required by this implementation. Retaining
tense-bearing words and the identified grammar derivation for input fidelity
does not assert that grammatical tense establishes time in the world. A
future temporal design must distinguish linguistic description from observed
event time and specify any connection before enabling such queries.

The full-description `Exist` evidence foundation checks accepted facts with
graded support, conflict and occurrence provenance (§15). Public `PartOf`
now reads conceptual-taxonomy references, preserving native proof sources and
rejecting unsupported perceptual domains (§16). The checked shared-VP adapter
forms and dispatches typed meanings through explicit APIs (§17). Normal
linguistic/internal meaning agreement and the ordinary levelled controller
remain open; these foundations do not complete those migrations.
[Exist lookup](../../bin/reasoning.py#L156),
[taxonomy query](../../bin/reasoning.py#L376).

### 2.1 Nested clauses and phrases

> **Superseded in part (2026-09-16):** embedded clauses collapse unless they
> contain a relative truth, which forces a reference. See
> [Two truths in LTM: ideas collapse, relations refer](../specs/2026-09-16-two-truths-ideas-and-relations.md)
> §2; the `p0`/`p1`/`p2` example below describes the relative case only.

The **1–3-slot bound is local to a grammatical node**, not a bound on the
information in an entire sentence. Each thought retains a structured root,
directly or by a stable reference. An occupied slot may reference a
typed compound phrase or clause whose own roles remain addressable. Thus a
relational node still has `[NP1, VP, NP2]`, with the VP in the middle; an
NP-position argument may denote an entity, event or proposition without
erasing those semantic types. Compound VPs likewise retain their derivation,
not merely the name of a primitive tool.

For "he said that she said P":

```text
p0 = P
p1 = [she, said, reference(p0)]
p2 = [he,  said, reference(p1)]
thought root = reference(p2)
```

Retain stable node/occurrence references, ordered role-labelled constituent
references, bindings and grammatical scope. Semantic vectors support
illumination and retrieval; they are not the sole record from which arbitrary
nesting must be inverted. A compound occurrence need not become a newly
admitted lexical concept. Reuse/extend the existing conceptual and occurrence
reference storage, not an independent semantic store. This builds on
[Architecture's relation-table contract](../Architecture.md#relation-table-entry-contract),
[Spaces' typed reference ownership](../Spaces.md#symbolspace-and-languagespace-reference-grammar-and-scheduling),
and the [fold-ladder plan](2026-09-10-meronomy-fold-ladder.md)
contracts 1, 6 and 7 on constituent witnesses, occurrence identity and phrases;
those mechanisms alone do not establish this nested-LTM guarantee.

Scope and evidence attach to the occurrence: support for `p2` does not assert
`p1` or `p0`. Preserve quotation/reporting, negation, variable binding and
where/when scope. A quoted or embedded question is content, not an instruction
to execute it. Creating a subordinate clause does not enter a thinking level;
syntactic nesting and the execution context level are distinct. The existing
[What interaction stack](../STM.md#13-interaction-ltm-and-the-what-stack)
is the legacy Q/A mechanism being replaced, not the definition of hierarchy.
Only a selected boundary action changes the execution level. Record semantic
constituent references when committing each external or internal composition;
neither speculative candidates nor ordinary backward references push or pop.

Specify node-count, depth and traversal limits separately from field width
and thought-stack depth. The bounded attended field exposes selected nodes
without deleting their stored structure. Preserve reachable constituents
across save/load and memory trimming; detect cycles or unavailable references
and report incomplete structure explicitly rather than silently flattening or
rebinding it. Reconstruction follows the input's retained derivation; output
generation realizes the answer's own structure and cannot borrow a hidden
input witness. These references and traversals remain subject to the
[Symbol Firewall](../SymbolFirewall.md#principle) and declared work limits.

## 3. Execution, memory and credit

Each `<thought>` declaration MUST have a checked signature, executor, result type,
read/write scope and evidence semantics. Predicate queries retain truth,
support and conflict; retrieval returns grounded references/sets; `arma`
returns a prediction, not an observed truth. Merely listing signatures is not
implementation. Missing executors fail explicitly. Candidate construction and
scoring are side-effect-free; only a selected, validated call executes.

At each boundary, the controller may form a concluding/refining thought or
invoke one registered query, including `what(Q)`, in the current context or a
selected subordinate context. A query does not automatically create a new
level. Completed results become available before the next choice. Do not
re-observe the original external sentence or advance its discourse history
on each internal step; internal thoughts have explicit internal provenance.

### Depth-labelled thought history

LTM retains ordinary structured thoughts in chronological order, not required
Q/A pairs. Each thought has its 1–3-slot local content (or structured root
reference), an integer **context level**, and its semantic scope/evidence and
provenance. The level is execution metadata beside the conceptual slots, not
a fourth conceptual slot or a return address. Preserve interrogative meaning
within the grammar: a fully bound question can still ask about truth, and
filled operands or an available response do not themselves establish resolution.

The execution contract is **strictly nested and sequential**, per batch row
and root episode:

- Begin at level `0`, with the root request/context recorded once.
- Further thoughts at level `d` continue in that same context; neither a new
  thought nor a missing answer implies a push.
- A selected descent enters level `d + 1`, preserving the suspended context
  at `d` and recording the new context's initiating meaning.
- A selected return from `d > 0` resumes the context at `d - 1`, making the
  child's result and actual support/uncertainty available there.
- At level `0`, a concluding thought plus an explicit episode-finish event
  ends the attempt. Level zero alone is not completion: it is also the start
  and may contain many intermediate thoughts.

For example, `0 → 1 → 1 → 2 → 1 → 0` contains continued work at level 1,
not two independently suspended contexts at that level. Once level 1 has
returned to 0, a later descent to 1 starts a fresh context; it must not reuse
the previous child's pending bindings/state. Negative levels, skipped
descent/return levels and non-LIFO resumption are invalid. At most one context
at each level is live. No per-thought parent/return reference or independent
dependency store is required under these restrictions.

**A level integer alone is insufficient memory.** Preserve the initiating
meaning, pending operands/bindings and scope needed to resume each live
context in the ordered history. Log every descent, return and episode finish
unambiguously in the existing execution trace, including a return followed
immediately by another descent when no parent-level thought is emitted. Two
adjacent level-1 thoughts must not conceal `1 → 0 → 1`. Pure transition events
need not invent conceptual content. This transition evidence is scheduling
metadata, not a Q/A label or pointer to another context.

Replay of the episode's thoughts and transitions must recover the current
level and each suspended context; "latest thought at the parent level" is
not sufficient unless it retains all required continuation state. A bounded
cache of recovered contexts is allowed, but it is derived from this history,
not a second authoritative thought store. Preserve episode boundaries, row
isolation and all records/references needed by active contexts through memory
trimming and checkpoint restore. Semantic occurrence references remain useful
for clauses, anaphora and evidence; merely following one does not change level.

Every relational thought, including one that answers an earlier thought,
retains its VP and operands, directly or through resolvable constituent
references. A boolean, scalar or result code alone is insufficient when it
loses the proposition being evaluated. A short response may refer to earlier
propositional content; its role as an answer comes from its meaning in that
context, not an administrative tag. Storage does not turn a question or an
unsupported claim into an accepted true relation.

Use explicit adapters where current STM/legacy LTM physical layouts differ
from canonical `[NP1, VP, NP2]` order. The consolidated store's infix layout
alone does not establish this thought-history contract. `LTMSlot` input/output
halves and `OPEN/COMPLETE/CLOSE` parity are legacy storage/control semantics;
external `WhatQuestion` / `WhatAnswer` interfaces may remain, but internal
execution must not infer nesting or completion from missing halves. Document
the history/checkpoint migration rather than claiming current parity tests
verify the new design.

Record actual model-produced thoughts and evidence, never gold answers.
Every response emitted by `reverseOutput()` must be linked to the thought it
realizes. Relevant intermediate results MUST causally influence subsequent
choices and the root result; a trace-only attachment is insufficient. Retain
both the asking and responding thoughts without requiring them to be paired
or filling an earlier record in place.

### Shared work budget and bounded return

Each root episode owns **one shared work budget and pressure accumulator**.
Every ordinary boundary step consumes budget, including same-level refinement,
repeated queries and descents; constant depth cannot conceal unbounded work.
Nested tool internals must be bounded, traced and included in declared cost
accounting. A child does not start another episode, optimizer step or allowance.
Local executor limits may tighten, never replenish, the budget. Closure
pressure increases during continued pursuit and is not reset by descent or
return; it encourages termination but does not guarantee it.

At budget cutoff, stop ALL ordinary work and freeze the active level `d_cut`.
Allow one separately counted drain of **at most `d_cut` LIFO returns plus one
root finish**. No new queries, descents or same-level refinement are allowed;
only bounded concluding/output realization and termination transitions. At
level 0 cutoff there are zero returns and at most one finish. Returning to a
parent cannot restart work or allocate another drain.

Return and root finish may carry supported results or explicitly unresolved,
budget-exhausted outcomes. Emit scoreable best-effort responses with actual
support/uncertainty and forced-termination provenance. Ending pursuit, reaching
zero, pressure and balanced control flow never establish truth or successful
resolution. No empty answer slot or Q/A parity is needed to enforce termination.

Train one bounded episode with one optimizer step. Under episode detachment,
continuous query operands, memory reads and intermediate results retain their
gradient paths until that step. Hard choices require explicit policy credit,
not differentiation through argmax; hard tool results are not made
differentiable by relabelling them. Keep answer, reconstruction,
concept-prediction and policy losses distinguishable. Corpus expectation in §8 detaches observed targets and durable history,
while retaining source gradients within the current optimizer step. It does
not detach this episode's live thought computations. No target, oracle decomposition or gold subgoal enters
the chooser context.

Every committed step records its goal, operation, operands/bindings, sources,
result/support, semantic delta and level/termination transition. Persist only
symbol-attached state. The answer must follow this actual execution path, with no unrecorded
latent shortcut. These are enforceable audit boundaries, not a proof that
every goal or true answer is safe.

### 3.1 What an internal question contributes

Persist internal asking and responding thoughts, without Q/A tags or pairing,
but **do not assume that asking adds knowledge**.
A question can establish a subgoal, bind a scope, redirect retrieval or make a
previously unavailable consequence computationally accessible. Its result may
add evidence, resolve a binding or expose uncertainty; the mere existence of
another question or record does none of these. A step's recorded semantic
delta may legitimately be empty. Useful verification or repetition is allowed;
re-reading the same source or one's own answer is not independent corroboration.

Measure separately the utility of **question semantics/control**, **executed
reasoning work**, and **later access to the stored question**. Retaining these
thoughts for scope, identity, replay and audit remains required even if historical
question content has no measurable benefit beyond the answers. That null
result must be reported, not disguised by counting records or activations.
No reward for question count or novelty substitutes for supported answer
quality and the declared computation cost. Direct answering remains available;
the model need not invent a subquestion when the answer is already supported.

The behavioral gate is an empirical comparison, not a requirement that every
question improve an answer. Scripted examples, graph connectivity and longer
traces cannot establish learned utility. Keep targets and oracle subgoals out
of runtime context in every comparison.
Claim learned questioning is useful only after reproducible held-out gains
in supported-answer quality at matched resources, or less work at matched
quality, with the causal controls below. Predeclare tasks, metrics and pass
thresholds; report variation across seeds. If this fails, mark learned utility
unproven even when mechanisms pass. A null historical-question readback benefit
alone does not invalidate useful question selection or reasoning work.

## 4. Acceptance tests

1. **Phase/path isolation:** query spies never fire inside the three sentence
   paths; compose-inverse and generate dispatch remain distinct. Mixed/packed
   rows expose only their own completed boundaries; unchosen candidates have
   no effects.
2. **One deep structure:** linguistic and internally constructed questions
   agree on `[NP1, VP, NP2]`, bindings, scope and results. Store/read back both
   asking and responding thoughts and verify the specialized VP remains
   recoverable in the middle, including short yes/no responses that reference
   earlier content. Exercise every supported storage-layout adapter;
   neither questions nor unsupported answers become true premises.
   Compose and registry faces resolve to the same relation identity; converse
   interfaces preserve the declared operand permutation.
3. **Surface/role agreement:** compose and realize "is a part of" and the
   part-whole sense of "has", in question and assertion forms. Verify the
   declared argument reversal, preservation of each reconstruction derivation,
   and canonical equivalence after output recomposition. Other "has" senses
   must not dispatch `isPart`. Swapping canonical operands or changing scope
   must change directional requests; no raw-text dispatcher supplies the answer.
4. **General subgoals:** an unseen composed `what(Q)` can enter nested contexts
   through the same controller, without a text round trip or word-only menu.
   Registry tests cover every declared signature and reject type/domain errors.
5. **Evidence separation:** structural containment alone cannot certify an
   asserted relation; prediction is not observation; false, unsupported and
   conflicting evidence remain distinguishable.
   `PartOf` must change when the relevant conceptual-taxonomy edge changes,
   but not when only a parse-tree/perceptual edge changes. `Exist` must respond
   to matching LTM facts and distinguish descriptions with identical NP1 but
   different VP, NP2, bindings or scope. Missing, negative and conflicting
   matches remain distinct; query/estimate records cannot satisfy the fact
   lookup. Deferred tense queries must not return fabricated temporal evidence.
6. **Causal multistep reasoning:** learn general conditional chains through
   syntax/LTM, then test renamed concepts and unseen chains. Removing a needed
   premise or corrupting a relevant intermediate result must affect the emitted
   answer. Compare matched checkpoints across thought budgets and memory ablations.
7. **Levelled lifecycle/learning:** test multiple thoughts at one level;
   nested descent/return; sibling return/re-entry at the same level, including
   no intervening parent thought; exact restoration of parent operands/scope;
   and no leaked child state. Reject underflow, skipped levels and non-LIFO
   resumption. Test root-level continuation separately from explicit finish.
   Replay and checkpoint restore must recover active/suspended contexts from
   immutable thought/transition history with row and episode isolation, without
   Q/A pairing or per-thought return pointers. Include active-context memory
   trimming and target isolation. Constant-depth work and child entry must
   consume the shared budget; descent/return cannot refresh it or pressure.
   At cutoff `d_cut`, permit at most `d_cut` query-free returns and one root
   finish, including cutoff at zero and explicitly unresolved outcomes.
   Demonstrate earlier continuous-state gradients and policy learning separately; scripted traces
   and finite gradients alone do not establish learned reasoning.
8. **Semantic chooser inputs:** with root/NP1 fixed, changing VP or NP2 must
   remain distinguishable to the chooser. Padding cannot affect selection;
   attention preserves provenance and cannot prune mandatory question/argument
   roles. Test retrieval-pool and attended-width changes independently of STM
   capacity, and verify the documented adapter/MLP shapes.
9. **Nested structural retention:** round-trip nested NP/VP phrases and
   reported clauses in asking/responding thoughts, save/load and supported memory
   trimming. Include shared constituents, repeated occurrences and unseen
   nesting depths. Check role order, binding/scope, input reconstruction and
   answer-side realization independently. Changing an embedded clause must
   remain distinguishable; reporting a claim cannot certify the claim, and
   embedding a question cannot execute it. At capacity/depth limits or cyclic
   references, report the defined incomplete outcome without dangling or
   silently rebound references; syntactic depth and semantic backward references
   must not alter execution level or trigger a return.
10. **Internal-question utility:** use three controlled comparisons:

    - Hold the root task, evidence, candidate menu and budget fixed; vary the
      selected internal question's relevant bindings/content, with paraphrase
      controls. Measure subsequent operation/operand choices and final answers.
    - Hold questions and scheduling fixed; remove, corrupt or replace
      intermediate results with no-ops to isolate the value of executed work.
    - Replay the same committed operations and answers, masking only historical
      internal-question semantics from later reads. Preserve root/active
      questions, level-transition history, necessary continuation state and
      result provenance; retain the underlying ordinary thought records.
      Compare with answer-history masking and both histories intact to isolate
      the question's marginal persistence benefit.

    Match checkpoints, randomness and actual computation/attention opportunities,
    not merely maximum budgets. Add held-out learned-policy trials against
    direct-answer and equal-compute no-subgoal baselines; replay alone tests
    mechanisms. Include redundant questions with unchanged evidence and tasks
    already answerable without thinking. Report supported-answer quality,
    actual work and uncertainty, including zero or negative marginal gains;
    repetition must not inflate evidential support.

## 5. Documentation required with implementation

- [Language](../Language.md): the three grammar paths, query registry and
  specialized VP formation/realization, sense selection and argument mappings;
  distinguish composing a query-capable VP from executing it, and remove the
  generic reconstruction-equals-generate claim.
  Align [complete.grammar](../../data/complete.grammar) comments/declarations
  and grammar tests with that representation/effect distinction; do not add
  effectful query executors as ordinary compose rules. Document nested phrase/
  clause types, constituent roles, scope and their two realization paths;
  distinguish semantic references from execution-level transitions.
- [Architecture](../Architecture.md) and [Params](../Params.md): both controllers,
  semantic input schemas, capacity/budget knobs, checkpoint migration, and ARMA
  as a boundary tool rather than the sole next-thought mechanism. Distinguish
  local slot width, execution level, reference-graph size/depth and traversal
  limits; document the shared ordinary-step budget and `d_cut + 1` drain bound.
- [STM](../STM.md), [Reasoning](../Reasoning.md) and [Training](../Training.md):
  ordinary thought records with context levels, replayable descent/return/finish,
  recoverable suspended contexts, and migration away from two-half interactions
  and parity. Cover physical-layout adapters, grammatical meaning/evidence,
  causal memory use, episode credit and measured gates, including separate
  question/work/history utility results. Reconcile [Spaces](../Spaces.md) and
  the fold-ladder plan's reference/witness contracts with nested-LTM ownership,
  retention and readback;
  do not treat syntactic child references as execution-level changes. Keep
  legacy parity tests explicitly legacy and add the new lifecycle tests.
- [Mereology](../Mereology.md), [Logic](../Logic.md) and
  [Symbol Firewall](../SymbolFirewall.md): relation domains, operation contracts
  and actual audit coverage; distinguish geometric scores from asserted evidence.
- Reconcile the older What/thinking specs linked above and the README index.
  Until then, preserve current-runtime descriptions and label this design as
  the target; do not mark implementation or behavioral gates complete in advance.

## 6. Code review: compiled reverse loops (2026-09-12)

This records the Codex review and Alec's resulting decisions on the
[compiled reverse-loops plan](2026-09-12-compiled-reverse-loops.md).
These decisions supersede conflicting proposals in that plan; they are target
requirements, not a claim that the loops have been implemented or benchmarked.

1. **Reconstruct the completed sentence, not just its local folds.**
   `reverseReconstruct()` starts from the completed, sealed structured
   representation and its permitted compose-derivation/constituent evidence
   (§2.1). Immediate per-word recovery cannot establish that later folds and
   seals retained the earlier input. It must not replace end-to-end sentence
   reconstruction or justify leaving seals outside the fidelity checks.
   Input targets are available only for scoring, not as reconstruction inputs.
   Test one-word sentences, deferred/no-fold words, unary changes and final
   seals, including cases where local recovery succeeds but completed-state
   information has been lost.

   Surface fidelity includes word termination: a candidate matching the
   input word's prefix but adding suffix bytes must not receive a perfect
   reconstruction score. Padding after the word end is not additional input.
   The target covers the complete spelling even when a word or multi-byte
   prefix has been promoted to one percept; percept compression cannot remove
   bytes from the objective.

2. **Reconstruction need not be an exact inverse; share its weights.**
   The objective is input reconstruction, not recovery of arbitrary original
   operands to floating-point rounding. Use the compose path's shared learned
   transforms, with LDU-based inversion of their invertible linear components;
   do not introduce an independently parameterized reconstruction decoder to
   compensate for ambiguous folds. LDU inversion does not make a lossy merge
   bijective. Balanced splits can recompose to a parent without recovering its
   original children; retained structure supplies permitted disambiguating
   evidence. Measure sentence reconstruction fidelity separately from linear
   inverse/recomposition accuracy. Tests must verify parameter sharing and
   declared gradient paths, without demanding exact child recovery or treating
   arbitrary identity fallbacks as faithful inverses.

3. **Run input reconstruction once per completed sentence.**
   When reconstruction is enabled, run one bounded compiled traversal after
   that sentence's forward composition and seals have completed, not a reverse
   at each forward word index. A traversal may contain multiple constituent
   and word-level reverse steps; "once" does not mean one primitive operation.
   Packed rows retain separate sentence boundaries and loss accounting.
   Recurrent thinking must not reconstruct the original external input again
   at every thought boundary. Follow the retained compose derivation rather
   than speculatively evaluating all reverse operators at every word. Bound
   basis-search candidates and traversal work explicitly; compilation alone
   does not remove their cost.

4. **Segregate forward, input reconstruction and output generation.**
   Keep three explicit logical paths: `forward()` uses `<compose>`;
   `reverseReconstruct()` follows that input's compose derivation with tied
   inverse transforms; `reverseOutput()` realizes an already-resolved idea
   through `<generate>`. Output owns its answer-side derivation, traversal
   state and termination, and cannot borrow input-reconstruction witnesses or
   use `score_binary(parent, parent)` as a substitute for a generate policy.
   Compatible numerical kernels may be shared; this does not merge the
   policies, seeds, state or losses. Resolve boundary queries before output
   realization. Output completes only when its per-row pending constituents
   are exhausted under the declared end-of-sentence rule, or reports bounded
   truncation; emitting the first leaf is not sufficient. Test mixed output
   lengths, multiple words, output longer than input, and invariance to changes
   in reconstruction-only state. A shared compiled wrapper is permissible
   only if it preserves these contracts and actual data dependencies.

5. **Correct the training baseline and declare the migration.**
   At review time, [BasicModel.xml](../../data/BasicModel.xml) enables
   `teacherReconstruction` and `detachedReverse`.
   [Models.py](../../bin/Models.py) trains the idea-only reverse student through
   `_detached_reverse_construction_loss()` and skips duplicate training calls
   to `reverseReconstruct()`; the trace-unfold route described as the plan's
   training baseline is an evaluation path. Replacing the detached student
   with tied reconstruction is a learning-contract change, not merely moving
   eager work into a compiled loop. Document the objective, parameter ownership,
   gradient/stop-gradient boundaries and checkpoint migration before retiring
   that path. Hard selection and detached helpers need explicit credit rules
   and tensor implementations; wrapping them does not restore gradients.

6. **Make performance approval evidence-based.**
   Replace the unverified 13.2-second comparison with a reproducible baseline
   naming the commit, configuration overrides, device/backend, executed losses
   and workload. Measure warmed full training steps, including backward and
   optimizer work, alongside compile time, recompilations and peak memory;
   vary sentence/output lengths, batch size and basis size. Compare fidelity
   and the declared parameter gradients as well as speed. The referenced
   `backend="eager"` graph-capture test is not a production-backend throughput
   test. Dictionary values can change during ordinary contextual updates:
   use a consistent per-invocation snapshot/version with active-row masks,
   distinguish value updates from storage/shape changes, and preserve the
   values needed by backward. Do not assume the basis is immutable until Reset.

With implementation, reconcile the compiled reverse-loops plan and update
[Architecture](../Architecture.md) / [Language](../Language.md) for loop and
parameter ownership, [Training](../Training.md) / [Params](../Params.md) for
objectives, credit and migration, and the
[throughput report](../benchmarks/2026-09-11-fold-ladder-throughput.md) with the
measured configuration and results. Extend §4's isolation and structural tests
with the sentence-level fidelity, tied-weight, gradient, packed-row and
production-backend gates above before claiming completion.

## 7. Implementation review (2026-09-15)

Reviewed against basicmodel `bad0ae689471fb09b8c322b3eef0895b6a4151de`.
Source line references below are for that revision; function names identify
the relevant code when lines move. This is a source/documentation review,
not a new test run or throughput measurement.

**Verdict: retain this document as the open thinking specification.** The
[answer-path ownership plan, line 3](2026-09-14-answer-path-ownership-and-training.md#L3)
records completion of its seven fixes. That closes those defects, not the
boundary-controller, thought-history or learned-reasoning requirements here.

### 7.1 Confirmed interpretation

Alec's clarification is the execution contract for §§1, 3 and 6:

1. **Understand the input.** `forward()` composes the surface into its local
   1–3-idea form, with larger structures retained through linked constituents
   in memory (§2.1). `reverseReconstruct()` checks that representation by
   recovering the input surface along the **identified** compose derivation.
   That derivation is not a known-correct parse; reconstruction fidelity does
   not certify its semantic correctness. Run the enabled reconstruction once
   per completed external sentence, as required by §6.3.
2. **Reason over the understanding.** The boundary controller selects
   grounded queries, refinements and subgoals, retains their conceptual results
   and support, and concludes an answer or an explicitly unresolved outcome
   (§3). Internal thoughts need not be verbalized or parsed again.
3. **Generate the response.** `reverseOutput()` receives the concluded
   conceptual result and expresses it through its own `<generate>` derivation
   (§6.4). Its entire invocation must respect that boundary, including setup
   before the first generate step. It must not initiate the reasoning episode
   or choose the next subgoal.

The single answer conditioner adapts the conceptual answer to the question.
The two controllers in §1 choose grammatical operations and thought/query
actions respectively. These are distinct responsibilities; the decision to
use one conditioner leaves the two-controller requirement intact.

### 7.2 Implemented foundations and remaining gaps

1. **Owned concepts and one conditioner are implemented foundations.**
   `AnswerProgram` captures separate WORD and interpreted-concept rows,
   leaves, actions and end state, cloning its tensors
   ([Understanding.py:23](../../bin/Understanding.py#L23)).
   `AnswerDerivation` retains a cloned conceptual answer and conditioning
   context ([Output.py:70](../../bin/Output.py#L70)).
   `_materialize_answer_idea` applies `_condition_answer_on_question` once;
   that conditioner operates at the answer's conceptual width and updates its
   root ([Models.py:11483](../../bin/Models.py#L11483),
   [Models.py:8566](../../bin/Models.py#L8566)). WORD surface bytes remain
   separate from changeable OBJECT interpretation
   ([Spaces.py:19949](../../bin/Spaces.py#L19949)). Preserve these contracts
   when introducing reasoned or predicted answers with no input parse to replay.

2. **Reasoning still occurs inside the output path.** With answer synthesis
   enabled, `what()` calls `reverseOutput()`
   ([Models.py:9719](../../bin/Models.py#L9719)); `reverseOutput()` calls
   `_resolve_answer()` ([Models.py:9033](../../bin/Models.py#L9033)), which can
   call `_resolve_step()` ([Models.py:8097](../../bin/Models.py#L8097)).
   `think()` repeats `what()` while retaining the first execution
   ([Models.py:9884](../../bin/Models.py#L9884),
   [Models.py:9913](../../bin/Models.py#L9913)). Reusing the original forward
   result is useful, but this call chain still interleaves thought selection
   and realization. Move resolution/control before the output invocation and
   make the concluded conceptual result an explicit input to realization.

3. **The current subgoal menu and history remain the legacy design.**
   `_enumerate_step_candidates()` offers ANSWER and OPEN for presented words
   ([Models.py:8352](../../bin/Models.py#L8352)). The interaction input now
   retains all captured idea slots, but its output half records the produced
   head response ([Models.py:9724](../../bin/Models.py#L9724),
   [Models.py:9757](../../bin/Models.py#L9757)). Completion and forced closure
   still use Q/A parity ([Models.py:9903](../../bin/Models.py#L9903),
   [Models.py:9925](../../bin/Models.py#L9925)). This does not satisfy general
   structured `what(Q)` selection or §3's ordinary conceptual thoughts with
   explicit descent, return and finish. The migration must retain conceptual
   intermediate answers, scope and evidence without requiring a surface/head
   response at every internal step.

4. **A reasoning trace is not a resolved answer.** The optional
   `answer_query(prompt)` call in `_resolve_answer()` records returned
   posture, confidence and support in the trace; that call site does not
   install a returned conceptual answer as the output seed
   ([Models.py:8050](../../bin/Models.py#L8050),
   [Models.py:8109](../../bin/Models.py#L8109)). The replacement must consume
   actual conceptual query results. Corrupting a necessary intermediate result
   must affect the final answer, as required by §4.6; trace attachment alone
   cannot pass that gate.

5. **Production defaults still require an explicit migration.**
   `BasicModel.xml` enables `teacherReconstruction` and `detachedReverse`
   ([BasicModel.xml:176](../../data/BasicModel.xml#L176)); `runBatch()` invokes
   the detached reconstruction objective and deduplicates the additional
   reverse call ([Models.py:13357](../../bin/Models.py#L13357),
   [Models.py:13451](../../bin/Models.py#L13451)). The independent generate
   catalog exists ([Language.py:14155](../../bin/Language.py#L14155)), while
   `outputInLoop` defaults to false
   ([Models.py:2503](../../bin/Models.py#L2503)). Keep §6's target and its
   September 12 baseline distinct from current opt-in implementations and
   production defaults; the existence of the three method names does not
   establish the required execution order or training contract.

### 7.3 Inter-sentence prediction and answer supervision

The production plan was revised after the review above. Its current direction
is concepts-to-concepts learning over externally observed sentences, with
surface prediction deferred and supplied-answer eligibility retained. The
former review's description of immediate next-text answer training and FUTURE
realization is superseded by §8. That section also records the root/full-meaning
limitation, actual configuration, training gate and memory-adapter constraints.

### 7.4 Work required before closing this specification

The detailed thinking gates remain in §4, the reconstruction migration gates
in §6, and the combined implementation/measurement order in §10. No new test
run, benchmark or behavioral completion is claimed by this consolidation.

## 8. Production learning over encoded sentences

This section incorporates the revised September 15 production-objective plan.
Its concepts-to-concepts scope supersedes that plan's earlier proposal to
train generated surfaces from next-sentence text.

Alec (2026-09-15): "Production should train by reconstruction and next
sentence prediction whenever the next sentence is related to the last
(much of the current fineweb data set). That will train reconstruction,
prediction (when the text is a monologue), and Q/A (when the text is a
dialogue). That change should dovetail with the inter sentence
prediction module." On the representation: "this is concepts to concepts
prediction; it is possible that we want words to words and percepts to
percepts (visualization), but that is future work."

<a id="1-the-objective-and-where-it-lives"></a>

### 8.1 Objectives and phase boundaries

| Objective | Prediction / response | Target and credit |
|---|---|---|
| Input reconstruction | Recover the completed input through its identified compose derivation (§6) | The external input surface; keep the reconstruction objective and its declared gradient paths |
| Inter-sentence concept prediction | Predict the arriving sentence's encoded meaning from the preceding LTM context in the same document (§8.7) | That input's observed encoding, with stop-gradient on the target; retain root-only prediction as a comparison for the complete-meaning target |
| Supplied-answer learning | Reason over the understood question, then realize the concluded answer (§§1–3) | The separately supplied desired answer, available only to scoring; retain distinct answer and hard-choice credit |

The next-sentence target is grounded in a subsequent **external observation**.
Its encoding is an eligible concept-level target even though the encoder
produces that encoding. A predicted thought, generated answer or retrieved
belief does not become an observed next sentence or its own desired target.
Internal thought steps must not advance the external sentence sequence.
Each arriving input can therefore supervise its own prior estimate in the same
processing call; no separately supplied future-input example is required (§8.7).

Monologues and dialogues use the same concept-prediction objective. A dialogue
continuation can supply the next answer's idea; this does not establish that
every adjacent sentence answers a question or that lower prediction error
demonstrates multistep reasoning. Keep §4's causal and learned-utility gates.

Prediction supplies conceptual content before realization. It does not replace
the thought/query controller or authorize query execution during `forward()`,
`reverseReconstruct()` or `reverseOutput()`. The predictor may become a checked
boundary tool, with prediction provenance and work accounting, under §3.
The current corpus-prediction hook learns at completed sentence boundaries.
Moving query control into prediction is the separate migration in §8.10.

<a id="2-what-the-current-production-configuration-does-with-it"></a>

### 8.2 Verified implementation and production baseline

Baseline verified at basicmodel `d541c94a78c46756f074ab6b9266fd9c81b54de4`.
The table records the September 15 baseline and joint-learning revision.
The September 16 local-role/sequence implementation is described in §8.3.

| Mechanism | September 15 implementation and evidence |
|---|---|
| Predictor owner | `InterSentenceLayer`, not `WhatInteractionMemory`; the latter is the live What interaction memory, whose dual ownership is retired by §11.3 ([Layers.py:9202](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9202), [Layers.py:9360](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9360)) |
| Context window | `_inter_chain_window = max(1, min(ltm_capacity, 8))`; `interChainWindow` is not a separately wired configuration setting ([Layers.py:9408](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9408)) |
| Prediction | `predict_next_end_state()` reduces history to roots, predicts one root, copies the latest depth and broadcasts that root across its slots ([Layers.py:10037](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L10037), [Layers.py:10110](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L10110)) |
| Target and stored context | Observation detaches the actual target and durable payloads; a bounded per-row prediction view now retains current-step source encoder gradients ([Layers.py:9795](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9795), [Layers.py:9832](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9832), [Layers.py:8732](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L8732)) |
| Predict/observe order | `predict_and_observe_stm_end_state()` predicts from the old chain before scoring and appending the arriving sentence ([Layers.py:9861](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9861)) |
| Loss | MSE accumulates only in training with gradients enabled and positive weight; `consume_inter_loss()` returns the mean over scored sentences. Optional InfoNCE uses previous roots as negatives ([Layers.py:10125](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L10125), [Layers.py:10154](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L10154), [Layers.py:9818](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Layers.py#L9818)) |
| Optimizer ownership | `<sentencePrediction>` constructs the discourse layer; its parameters are appended to SymbolSpace's explicit `params` ([Language.py:10978](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Language.py#L10978), [Language.py:11032](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Language.py#L11032)) |
| Training total | A consumed `inter_loss` is added with `inter_loss_weight` independently of Teacher's legacy gate and is included in the shared downstream gradient budget ([Models.py:13535](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L13535), [Models.py:13642](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L13642)) |

The September 15 canonical configuration explicitly sets `sentencePrediction=false`,
`interLossWeight=0.0`, `armaScale=0.0` and `interContrastiveWeight=0.0`
([BasicModel.xml:188 at d161a7a](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/data/BasicModel.xml#L188)). The Python fallback
for `interLossWeight` is 0.1, but the explicit XML zero overrides it
([Models.py:2479](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L2479)). Production also enables
`teacherReconstruction` and `detachedReverse`
([BasicModel.xml:176 at d161a7a](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/data/BasicModel.xml#L176)); `reconstructInLoop`
defaults to false ([Models.py:2499](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L2499)). Therefore the
earlier statement that production already uses tied `reconstructInLoop`
and leaves `interLossWeight` unspecified is incorrect for this revision.
These are the verified code defaults, not the approved target defaults:
§8.7 requires expectation on by default with an explicit off switch. The
default-on rollout remains gated by §10. The joint-learning revision below
removes the Teacher gate for inter-sentence MSE/contrastive learning.

**Joint-learning revision:** `Teacher.legacy_prediction_enabled` still
controls the older ARMA/intra objectives. Inter-sentence MSE and contrastive
consumption now work alongside Teacher reconstruction. Current-step context
retains encoder gradients in a bounded per-row prediction view, while targets
and durable LTM remain detached. The shipped expectation flags above are not
changed by this gradient-contract revision; the default-on rollout still
requires the complete §10 sequence/quality gates.

The older pooled-sentence ARMA loss remains a separate term
([Models.py:13498](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L13498)). Keep `armaScale=0` and
`interContrastiveWeight=0` for the first concept-prediction measurement.
Retain the measured reconstruction baseline until §6's tied-reconstruction
migration is implemented and verified; the prediction change must name which
reconstruction path it actually ran.

### 8.3 Sequence identity, memory and complete meaning

Use consecutive external sentences from the **same row/stream and document**
as the first relatedness rule. Document continuity is a corpus scheduling
proxy, not semantic evidence. A document boundary invalidates that stream's
pending prediction and prediction context. Preserve durable LTM knowledge
and other rows' contexts; do not erase all memory to obtain this isolation.
Never use internal thoughts, provisioned facts or another stream's recent
entries as observations in that external sequence.

Both physical memory paths must be checked. Legacy durable history uses a
per-row deque; consolidated `get_stm_chain` still exposes global store recency
for its existing callers. Prediction now uses the separate bounded occurrence
view described in §8.4, populated only by external observation hooks. That view
preserves row isolation and can retain current-step encoder graphs with either
memory adapter. Resetting its selected row preserves durable LTM and other
streams. The data scheduler supplies true document boundaries from source
addresses; complete nested-meaning and episode replay acceptance remains in §10.

**September 16 local-role implementation.** The model selects
`sentenceExpectationScope=structured`. `SentenceExpectation` reads all
occupied local NP1/VP/NP2 vectors and masks in chronological order, and predicts
independent role vectors and occupancy logits. Its objective is occupied-role
MSE plus mean presence binary cross entropy. Current-step source encodings
remain live, while targets and durable observations are detached
([Layers.py:9499](../../bin/Layers.py#L9499),
[Layers.py:10096](../../bin/Layers.py#L10096)). This does not yet implement the
retained compound-reference prediction required by §§2.1 and 10.5.

Packed draining uses the existing sealed three-slot/depth outputs and final
seal state. Explicit STM/infix adapters keep the same local roles in both
memory modes. Cursor source addresses identify each packed sentence's document;
the first document transition is applied before priming, and later transitions
at their individual boundaries. The single-sentence cursor preserves a
continuation within the same addressed document. Provisioning runs under a
suspended external-observation scope: its parsed truths still enter durable
LTM, while external rows, pending predictions and scored losses survive even a
temporary batch reshape
([Models.py:12726](../../bin/Models.py#L12726),
[Models.py:12837](../../bin/Models.py#L12837),
[data.py:526](../../bin/data.py#L526),
[Layers.py:9999](../../bin/Layers.py#L9999)).

Restoring weights starts transient prediction context cold. A root-predictor
checkpoint migrates explicitly to a freshly initialized structured head;
unrelated weights remain intact and optimizer moments follow the existing
name-based mapping. The root architecture remains selectable for baseline
comparisons ([Layers.py:10161](../../bin/Layers.py#L10161)). Regression evidence
lives in [test_sentence_expectation.py](../../test/test_sentence_expectation.py)
and the real packed training tests in
[test_reconstruction_priority.py:126](../../test/test_reconstruction_priority.py#L126).

Pre-review validation on September 16: the failing structured-role, document-boundary,
checkpoint and real provisioning probes pass. Affected runs included 133
passing tests, 113 passing / 1 skipped, and 103 passing / 1 skipped. The full
background suite before the §11 changes completed with **4,119 passed, 54 skipped, 7 xfailed and
4 subtests passed in 2,039.99 seconds**. The optional schema test skips without
`lxml`; runtime scope validation passes. This verifies the local-role milestone,
not the remaining default-on, nested-meaning or reasoning requirements.

The predictor's full encoded-sentence target and the historical root baseline
must be distinguished. Here **root-only means one slot, not all three**.
In consolidated `[NP1, VP, NP2]` order it is the first slot, NP1, at Python
index `0` ([consolidated reduction](../../bin/Layers.py#L10524)). The legacy
newest-first adapter instead selects its last occupied slot
([newest-first reduction](../../bin/Layers.py#L10526)); "root" does not name one
universal physical index. The root benchmark copies its one predicted
vector into every returned slot ([Layers.py:10636](../../bin/Layers.py#L10636)).
Before the September 16 change, packed prediction
draining supplied a single root with depth one
([Models.py:12420 at d161a7a](https://github.com/arborrhythms/BasicModel/blob/d161a7af6e74f689f700be547c33576253b7b45b/bin/Models.py#L12420)). Repeating a root does not
recover distinct VP/NP2 content, role masks or nested references. Preserve
the complete structured-idea contract in §§1–2.1. A first root-only benchmark
is a limited measurement, not completion of full sentence prediction or
the semantic-controller requirements. The scope decision is recorded in §9.2.

At packed boundaries, stage context before the target sentence is appended;
exclude padding, unseen future slots and cross-document continuations.
Keep valid accumulated losses when one stream resets, and consume each
eligible prediction/observation pair once at the declared optimizer step.
Global and row resets differ today
([Layers.py:10809](../../bin/Layers.py#L10809)); their effect on already
scored losses needs explicit tests, not an assumption that every reset
clears the same state.

<a id="4-the-one-decision-does-prediction-reach-the-encoder"></a>

<a id="84-gradient-boundaries-and-learning-evidence"></a>

### 8.4 Joint representation learning and gradient balance

**Latest decision (2026-09-15).** Alec clarified that reconstruction and
answering each permit a variety of representations: learn a representation
that does both well. Prediction error must be allowed to improve the internal
representation's usefulness. This supersedes the earlier interpretation that
prediction must never train the encoder. Separate module ownership does not
imply a stop-gradient boundary at their inputs.

The learned NP/VP representation must retain word meanings and syntactic
structure, reconstruct the completed input through its identified compose
derivation, and support prediction/answering using LTM, perceptual mereonomy
and conceptual taxonomy. Keep the tied forward/reconstruction transforms
within comprehension (§6). Prediction, thinking and supplied-answer feedback
may train the representation-building parameters through their actual live
computation paths. Independent predictor/generator parameters also learn.

#### Executable balancing rule

`reconstructionPriority=true` applies one budget to **all non-reconstruction
losses together**, including concept prediction, output, thinking and other
auxiliaries reaching shared parameters. A batch does not need supplied answers
for the rule to apply. `runBatch` identifies its weighted reconstruction
reference `R`; the downstream objective is the trained total minus `R`.
The reference includes active reconstruction terms and their actual weights
and truth modulation. A reporting-only metric is not a training objective.

For each protected parameter tensor let `r = grad(R)` and
`d = grad(total - R)`. The implementation keeps `r` unchanged. The projection
reference `a` is `r` when `abs(R) > reconstructionLossTolerance`; otherwise it
is zero. Missing reconstruction gradient is also a zero reference. With
`alpha = outputGradientRatio`:

1. If `a` is nonzero, remove only opposition:
   `q = d - min(0, dot(d,a)) / dot(a,a) * a`. Otherwise `q = d`.
2. Set `budget = alpha * (norm(a) + norm(q))` and shrink `q` only if its
   norm exceeds this budget. A zero `q` remains zero.
3. Use `r + q` on protected weights and ordinary total-loss gradients on
   independent heads. Take one optimizer step at one parameter version.

The existing ratio name is retained, but its meaning changes from a cap
relative solely to reconstruction to a fraction of the **combined reference
and compatible downstream gradient scale**. Its default remains `0.5`, with
`0 <= alpha < 1`. At zero reconstruction gradient, half the downstream
contribution survives at the default setting; a faithful representation can
therefore continue becoming useful. Ratio zero withholds downstream credit
from shared weights while independent heads can still learn.

`reconstructionLossTolerance` defaults to `1e-8`, is finite/nonnegative, and
uses the absolute value of the **unscaled, weighted reconstruction loss**.
Below this numerical fidelity tolerance the projection reference is zero;
reconstruction's own gradient is still applied. This avoids pinning an
otherwise exact inverse to roundoff-sized residuals. AMP must make the
tolerance comparison before scaling the losses, and scale both branches
identically. This tolerance is not a claim that any dataset's semantic or
surface acceptance criterion is satisfied by a particular scalar loss.

Existing checkpoints keep their learned tensors and optimizer state. The
saved `outputGradientRatio` value uses the revised combined-scale meaning;
checkpoints/configurations without `reconstructionLossTolerance` use `1e-8`.
This is an intentional learning-rule change, not a parameter-shape migration.

The protected set contains optimizer-owned PartSpace, WholeSpace and
ConceptualSpace representation parameters plus grammar transforms from the
actual host registry. Include shared grammar parameters even when their
registration owner is SymbolSpace. Deduplicate by parameter identity and
exclude independent synthesis/output heads. Protect a shared parameter
regardless of whether prediction/output reaches it through a live input,
a memory read or a reused operator. Splitting catalogs (§8.9) cannot by itself
prove that an answer gradient never reaches comprehension.

The numerical rule lives in `reconstruction_priority_gradient` and
`backward_reconstruction_priority` ([Optimizer.py:113](../../bin/Optimizer.py#L113));
loss partition and ownership live in `_backward_training_loss` and
`_reconstruction_priority_parameters` ([Models.py:2708](../../bin/Models.py#L2708),
[Models.py:2746](../../bin/Models.py#L2746)).
This is a local gradient rule, not a guarantee of monotonic loss under Adam,
momentum or finite steps. Both tasks may adapt during learning; measure the
resulting reconstruction, prediction and answer quality. A permanent priority
for whichever encoding happened to reconstruct first is not the objective.

#### Context and target gradient boundaries

Hold each observed target encoding fixed for its prediction comparison.
Retain live preceding context **within the current training step** so the
predictor can train its source encoder. The bounded per-row `_inter_context`
view contains only observed external predecessors; consolidated durable LTM
remains a detached snapshot and is not replaced by this transient view.
Prediction uses this scoped chronological view, rather than global store
recency that can contain another row or an unrelated provisioned fact.
The live clone and detached durable write are adjacent in
[Layers.py:10149](../../bin/Layers.py#L10149).

Consume the prediction losses once, then detach the context view and clear
pending predictions before a subsequent optimizer version is used. Also
clean up at brick entry. Document/row resets clear the selected prediction
view and pending estimate without deleting durable LTM. Callers must deliver
the actual document boundary; row identity alone is not document identity.
Compiled and eager sentence products retain the same permitted graph;
detached targets and durable records must not be used as live-context
substitutes. Full nested-structure prediction remains a separate milestone; local-role
prediction is described in §8.3.
Context cleanup is implemented in
[`detach_prediction_context`](../../bin/Layers.py#L10689); packed chronology is in
[`_drain_packed_stm_end_states`](../../bin/Models.py#L12837), with the
[single-drain guard in `_end_step`](../../bin/Models.py#L12665).

§3 still requires live continuous query operands and intermediate results
inside the bounded reasoning episode, and explicit credit for hard choices.
A separately constructed predictor can send gradients through its inputs
without sharing its own Pi/Sigma parameters with the encoder. Supplied-answer
loss may train prediction/thinking when the answer actually depends on them;
constructing that response connection remains explicit work (§8.5).

The detached reverse-student baseline does **not** provide an encoder
reconstruction gradient. Enabling downstream feedback does not reconnect it
or complete §6's tied-reconstruction migration. Report which reconstruction
path is active and test the live tied path separately.

#### Learning evidence

Tests must cover opposing/aligned/orthogonal gradients; zero, missing and
near-zero reconstruction; combined prediction/thinking credit without answer
labels; actual grammar ownership; sparse gradients; common AMP scaling;
independent heads; and repeated optimizer steps without stale graphs.
A learning test must select a more predictive encoding from a family of
equally reconstructable encodings, preserving reconstruction while reducing
prediction error. A nonzero gradient alone is insufficient evidence.

No evaluation optimizer step or training-loss accumulation is permitted.
Measure held-out prediction and reconstruction, plus supervised answer quality
when labels exist. Compare ordered, shuffled and context-free controls at
matched resources. A shuffled model can learn marginal regularities; require
an advantage attributable to context. Monitor representation discrimination
and variation: stopping target gradients does not prove freedom from collapse,
and improving next-sentence MSE does not establish learned subgoal use (§4).

#### September 15 revision evidence

The initial reviewer probes failed before the fix on eight cases: absent,
zero and near-zero reconstruction freezing downstream learning; inability to
learn a more predictive invertible encoding; auxiliary objectives bypassing
the balance; omitted grammar ownership; detached source encodings; and
missing encoder credit across steps. Additional failing probes exposed
evaluation accumulation, omitted eager packed boundaries, duplicate teardown
observations and residual Adam updates when the effective prediction weight
was zero. Each defect received a regression check.

- The [invertible-family learning test](../../test/test_reconstruction_priority.py#L26)
  keeps reconstruction below `1e-25` while reducing prediction error to less
  than 2% of its initial value over 60 SGD steps. This establishes the intended
  behavior on a controlled family, not corpus learning or semantic quality.
- The [real packed training test](../../test/test_reconstruction_priority.py#L126)
  exercises both detached-student and tied reconstruction, two Adam updates,
  and an effective zero-weight step with the layer gate still enabled. It
  checks live encoder/predictor gradients, unchanged predictor weights when
  disabled, and detached context after consumption. The explicit compiled
  state handoff runs eagerly in this test; the affected compiler tests cover
  actual compiled loops.
- The [context tests](../../test/test_inter_sentence_prediction_shape.py#L51)
  cover source/target gradient separation, graph lifetime, row isolation,
  partial reset, consolidated observations and evaluation with ambient
  autograd enabled.
- Affected test run: **280 passed, 6 skipped**. After the teardown and
  effective-zero-weight corrections, the five affected files passed again:
  **139 passed, 4 skipped**. The
  [sparsity integration file](../../test/test_concept_readout_l1.py#L138),
  updated to expect the revised gradient cap while retaining its separate
  L1 checks, passed **17 tests**.
- Full suite: `.venv/bin/python -m pytest test -q -p no:cacheprovider`
  completed with **4,100 passed, 53 skipped, 7 expected failures and
  4 subtests passed** in **33m 57s**. Source files remained fixed during
  this background run.

This September 15 evidence preceded default-on local-role expectation (§11.6).
Supervised throughput, the complete tied-reconstruction migration and learned
query/subgoal behavior remain separate acceptance gates in §10.

<a id="5-what-this-does-not-train-on-purpose"></a>

<a id="6-future-work-recorded-not-planned"></a>

### 8.5 Answer learning and deferred extensions

Keep supplied-answer training for the answer conditioner, generate chooser
and dedicated synthesis modules. The current answer-target gate and its
call site enforce this separation
([Models.py:9419](../../bin/Models.py#L9419),
[Models.py:13698](../../bin/Models.py#L13698)). The concepts-to-concepts
objective adds no next-text answer target and no reconstruction-parse teacher
for output. Past/future realized-answer metrics remain evaluation metrics.

- **Words to words / surface prediction:** deferred. A future design may
  realize a predicted idea and score the true next sentence's surface. It
  must define eligible corpus targets, desired-target isolation and ownership
  of a predicted idea without a compose program. The old candidate check
  `answer.available and answer.provenance == "data" and
  answer.source_where != question.where` is insufficient by itself: exclude
  the target from the entire presented pack/context, preserve document scope
  and supplied-label behavior, and never train an output identity on PRESENT.
- **Prediction-to-output integration:** remains unfinished even for inference.
  The indexed FUTURE branch still selects no answer program
  ([Models.py:8236](../../bin/Models.py#L8236)). Shared inverse transforms do
  not establish a working FUTURE realization or answer quality. When this
  integration is taken up, capture a target-free conceptual value and
  provenance; do not fabricate input derivation witnesses or move query
  selection into `reverseOutput()`.
- **Percepts to percepts / visualization:** deferred and not designed here.

<a id="86-proposed-separation-of-understanding-and-response-production"></a>

### 8.6 Understanding, prediction and response production

**Decided, clarified 2026-09-15.** Keep explicit responsibilities and module
ownership while allowing their permitted gradients to shape one useful
representation under §8.4. Public method names do not change:

1. **Understanding:** `forward()` composes the complete structured idea;
   `reverseReconstruct()` checks input fidelity through the identified compose
   derivation and tied transforms in §6.
2. **Prediction and reasoning:** the predictor consumes understood ideas and
   memory, uses queries as tools (§8.10), and predicts/resolves a subsequent
   thought or answer idea before surface realization.
3. **Realization:** `reverseOutput()` expresses the concluded idea through its
   own generate derivation. Output training requires supplied desired answers;
   input reconstruction targets are not answer labels.

Retain tied forward/reconstruction weights within comprehension. The separate
comprehension/generation catalogs in §8.9 remain the target, with live data
handoffs and gradient permissions stated separately. Independent predictor
parameters do not require detaching the representation used for prediction.

Answer supervision may train the predictor when the answer depends on its
prediction through a live path or explicit hard-choice credit. Current indexed
FUTURE answers still lack an owned output program; do not claim that connection
exists merely because the gradient balancer admits it. The local-role sentence
predictor owns a separate `SentenceExpectation`
([Layers.py:9499](../../bin/Layers.py#L9499)); its parameters are independent
of the comprehension path. Concept-target prediction works without generating text, and corpus
surface prediction remains deferred (§8.5).

### 8.7 Expectation on every input (default on)

**Alec's decision (2026-09-15): internalize prediction into ordinary input
processing. Expectation is on by default and can be explicitly turned off.**
Predict the expected incoming idea or requested conceptual state from the
preceding LTM context, then use the actual input as its observation and training
target. "Future" describes its relationship to that prior context; the caller
does not need to supply a second, future input to train the first prediction.

This builds the predictor through corpus continuation at encoded-sentence
granularity: each eligible observation supplies supervision for the estimate
made from its preceding context. The analogy to LLM corpus learning concerns
that causal supervision pattern. Here the prediction object is a structured
sentence idea (§8.3), and corpus surface prediction remains deferred (§8.5).
Generated estimates and internal thoughts cannot supply their own observation
targets. A corpus stream supplies the observations; separate future labels,
supplied answers and a special What-task curriculum are not prerequisites.

#### Per-input cycle

For each eligible completed external sentence, independently per row/stream:

1. **Anticipate.** Capture the permitted LTM view before this observation and
   predict the expected idea for that occurrence. An already known request
   and its scope may condition the estimate; the arriving content, desired
   answer and unseen packed sentences may not. The object is the expected new
   idea or requested state, not a prediction of every physical memory slot.
2. **Understand.** Run `forward()` on the actual input and its enabled
   `reverseReconstruct()` using the identified compose derivation. Retain the
   complete observed idea. Expectation does not replace understanding or make
   the estimate the reconstruction target.
3. **Compare and learn.** Compare aligned predicted and observed meaning to
   obtain a residual. During training, accumulate the eligible concept loss
   against the detached observed encoding, once, for the declared optimizer
   step. Apply §8.4's update permissions and reconstruction-priority budget.
4. **Retain and reason.** Append the observation once, retain its relationship
   to the original estimate and residual, and let the reasoning controller use
   the mismatch to direct further work. Conclude an idea before surface
   generation. This observation becomes available to subsequent predictions.

Anticipation reads preceding state outside the three sentence paths. Under
§8.10 it may use queries over that preceding state; the current arriving
sentence cannot supply its own predictive context. Understanding/reconstruction
of an observed input precedes reasoning about it and surface generation. Prediction may be prepared
at the preceding boundary, or computed later from an isolated pre-observation
view, provided no target information can enter it through staging, memory
writes, dictionary changes or another row. For packed input, honor chronological
sentence boundaries instead of exposing the entire pack as prior context.

The existing helper already predicts before observing the arriving end state
([Layers.py:10362](../../bin/Layers.py#L10362)); the pending and packed drains call
it before their consolidated-store append
([Models.py:12777](../../bin/Models.py#L12777),
[Models.py:12837](../../bin/Models.py#L12837)). This supplies a starting point,
not proof of isolation throughout `forward()` or of the complete cycle above.
The ordinary input path must perform this cycle without a `Data.what(FUTURE)`
request or an available FUTURE realization program (§8.5).

#### Estimates in LTM and residual use

An estimate written to LTM retains complete structured meaning, role masks,
bindings and scope, plus explicit **estimate provenance**. Record the source
context/occurrences, stream/document and intended occurrence or request so the
estimate can be paired with the right observation. Preserve confidence or
uncertainty separately from its status as a prediction: confidence cannot turn
an estimate into observed evidence. Persist it through the symbol-attached
memory contract in §3, without a second authoritative semantic store.

Keep the original estimate and subsequent observation distinguishable in the
append-only history. Later corrections link to them instead of rewriting what
was predicted after seeing the target. An estimate is not an additional
observed sentence, a new training target, or independent corroboration of its
own source. Keep the external sequence view in §8.3 separate from the ability
to retrieve estimates as predictions. Replay, trimming and checkpoint restore
must preserve the required occurrence links and provenance.

The residual compares corresponding semantic roles and their full-width
features, with masks, bindings and scope respected. Subtracting arbitrary
concept IDs is not a semantic residual. Retain the actual observed meaning and
the estimate needed to interpret the difference; a bare delta cannot replace
either. Use mismatch to guide attention, retrieval, refinement and learning
under the existing chooser and shared work budget. Large surprise does not
itself establish a new fact or require a new subgoal. Small surprise does not
remove the input-reconstruction objective. Start by allocating further work
after full encoding; reduced initial encoding cost requires a separate design
and measurement.

An estimate may be consulted without a new observation, retaining its estimate
status. No observation residual or supervised prediction loss is then available.
When no usable prior context exists, record an unavailable/cold-start estimate,
understand and retain the input normally, and skip the undefined comparison.
Default-on operation does not make a placeholder a valid forecast. Document
boundaries and partial resets retain §8.3's scope and durable-memory rules.

#### Defaults, off switch and acceptance

Use the existing `sentenceExpectation` setting as the explicit switch, with a
target default of **true**. Training must give the eligible `inter` objective a
positive weight; use explicit `interLossWeight=0.1` for the initial measurement.
Apply the cycle at every eligible observed sentence boundary, rather than only
on selected question types. Keep the older `armaScale` and optional contrastive
weight zero for that measurement (§8.2). Enabling expectation does not enable
all legacy prediction objectives. Predictor learning can continue on detached
history; encoder feedback uses live current-step context and §8.4's shared budget.

Explicit `sentenceExpectation=false` bypasses this automatic estimate/residual
cycle and its predictor updates. Understanding, reconstruction and ordinary
memory/reasoning continue. Re-enabling starts from a fresh eligible context;
do not score a stale pending estimate or observe an input twice. At inference
and evaluation, the default still computes expectations and residuals when
observations arrive, but performs no training accumulation or optimizer update.
Held-out metrics remain available under §8.4.

In addition to §§4 and 10, require evidence that:

- Omitted configuration enables the predictor, and ordinary consecutive corpus
  inputs train it without explicit future examples or supplied-answer labels.
  Explicit off disables its effects and updates while preserving input learning.
- With prior state and known request fixed, substituting a controlled,
  semantically different observation leaves its original estimate unchanged
  and changes the measured residual or loss.
  Streaming single-input calls and equivalent packed input produce the same
  eligible pairs and order, counting each observation once.
- Stored estimates remain distinguishable from observations through replay and
  restore, do not enter the observed sequence or inflate evidential support,
  and retain the full-meaning scope selected in §9.2. Cold start, document/row
  resets and disable/re-enable cannot score stale or duplicate pairs.
- Prediction-only and joint updates obey §8.4, including batches with no answer.
  Residual-guided reasoning demonstrates useful held-out quality or reduced
  work at matched quality under §4's controls; the mechanism alone establishes
  neither improved reasoning nor a processing-speed gain.

### 8.8 Gradient interaction among the four pieces

The maintained implementation reference is
[Gradient flow across the architecture](../GradientFlow.md). It maps each
objective to its live inputs, stop-gradient boundaries and optimizer owners,
and distinguishes implemented credit paths from the remaining migrations.
The shared rule prevents aggregate downstream opposition to reconstruction;
it does not independently resolve conflicts among prediction, thinking and
output before those downstream gradients are aggregated.

**Latest decisions (2026-09-15), superseding the earlier disjoint-gradient
interpretation in this document:**

1. Comprehension and input reconstruction share compose transforms (§6).
   Generation owns its generate catalog (§8.9); this separates numerical
   parameter ownership without blocking useful feedback through live inputs.
2. Queries are tools of prediction/thinking (§8.10), before surface generation.
3. Jointly learn a representation that supports reconstruction, prediction
   and answering. Reconstruction constrains fidelity; downstream objectives
   help select useful encodings among those that satisfy it. They may train
   the encoder under §8.4's combined budget and fidelity tolerance.

#### Earlier measured baseline

Before this revision, the ladder probe with `sentenceExpectation=true` found
187 optimizer parameters, 127 protected parameters, and 28 grammar fold/inverse
parameters outside protection. All 16 binary compose/generate modules were
shared objects. These are historical measurements, not counts promised for
future configurations. The registry's SymbolSpace registration had caused
representation transforms to be omitted by the P/W/concept class filter.
The revised selector includes optimizer-owned grammar transforms by identity;
regression tests exercise both ownership and live training gradients.

#### Intended ownership and learning

| Component | Direct objectives | Feedback into shared representation |
|---|---|---|
| Comprehension and reconstruction | Input reconstruction through identified compose derivation | Reconstruction reference; jointly adapts with downstream learning |
| Prediction | Observed concept target; supplied-answer credit when a response path actually uses prediction | Live preceding context under the combined budget; observed target detached |
| Thinking/query control | Prediction residual or appropriate supplied-answer credit; explicit policy estimator for hard choices | Continuous episode paths use the same budget; hard actions need their own credit |
| Output generation | Supplied desired answer, through an independent generate derivation | Live conceptual inputs/shared transforms use the same budget; independent heads learn ordinarily |

Separate parameter sets do not establish gradient isolation. Test actual
reach through tensor handoffs, dictionary reads and operators. Parameters
that are persistent non-grad buffers remain under their existing update
protocol; this change does not silently convert those stores into Adam
parameters.

On each step aggregate downstream contributions before projecting/capping;
separate allowances for prediction and output would not bound their sum.
Use actual loss weights and the single optimizer version. Below the declared
reconstruction tolerance, permit further predictive refinement rather than
forcing the representation to remain fixed. Neither an absent encoder
gradient from the detached reverse student nor a scalar low loss is evidence
that the complete intended fidelity contract has been implemented.

When a new observation is absent there is no new observation-based prediction
target. Internal estimates remain estimates, and cannot supervise themselves.
The concept MSE predictor has no policy baseline; a query chooser's
score-function estimator owns its baseline. Future query credit must specify
its scalar reward, work cost and delayed-observation attribution explicitly.

Loss aggregation remains a **mean over eligible observed pairs**. Increasing
sentence packing changes available within-step contexts and gradient reach;
it does not automatically multiply the reported loss or its configured weight.
Keep discrimination/collapse checks and held-out causal controls even with
projection and detached targets. Comprehension learning may legitimately move
encodings; do not demand byte-identical representations across joint updates.

### 8.9 Separate comprehension and generation catalogs (decided)

The target remains separate compose/generate parameter catalogs. Comprehension
owns forward composition and its tied reconstruction inverses; generation
owns the operators selected by `<generate>`. Use the same operator classes,
copy comprehension's values once at construction/checkpoint migration, and
then allow the catalogs to learn independently. Preserve rule-meaning-based
chooser row migration and optimizer ownership.

The current host registry still supplies shared operators to both catalogs.
Until the split lands, §8.4 protects those shared transforms. After the split,
a live representation input may still carry output gradients into its
comprehension producer; those paths are intentional and must remain balanced.

Test parameter identity separately from gradient reach. With detached input
codes, reconstruction must train comprehension's operators and generation must
train its own. With live inputs, verify the permitted downstream encoder
gradient and combined budget. Catalog separation alone cannot demonstrate that
only the shared reverse perceptual chain remains an overlap.

### 8.10 Queries as tools at inter-sentence prediction (decided)

Queries let prediction/thinking retrieve context and pursue subgoals before
surface generation. For an arriving input's anticipation, the chooser may use
only preceding observations, already known request metadata and permitted LTM
reads. It must not inspect the arriving content or mutate its fixed target.
After that input has been understood, it may inform subsequent thoughts,
answers and the expectation for the next observation. §3's budget, episode
credit and append-only provenance remain in force.

A later observed prediction residual can supply query-policy credit on an
unlabelled corpus. This is the intended migration from answer-only query
reward, not a claim that the current controller already does it. Specify
residual-to-scalar reward, work penalty, baseline and delayed trajectory
attribution before implementation. Ordinary differentiable MSE has no policy
baseline to share; each actual score-function estimator must be accounted for.

The predictor and continuous query paths may train comprehension through live
source representations under §8.4. Detached targets prevent a comparison from
moving its own target directly; they do not prove that representation collapse,
shortcut learning or useless queries are impossible. Require held-out utility,
reconstruction and discrimination evidence with causal query ablations.
Query execution reads evidence; internal thought/estimate history remains
separately identified and cannot become an observation or its own target.

## 9. Reconciliation and decisions

### 9.1 Conflicts resolved in the document

| Issue in the source documents | Consolidated treatment |
|---|---|
| The earlier §7 review described next-text answer training and FUTURE realization as the new plan's immediate work | Superseded by the revised September 15 decision: concept-to-concept production learning now; surface/percept prediction and FUTURE realization are deferred (§§8.1, 8.5) |
| The production plan described `reconstructInLoop` as current and `interLossWeight` as unspecified | Corrected against the checked-in configuration: detached reconstruction and explicit zero inter weight (§8.2); tied reconstruction remains the §6 migration target |
| The production plan said enabling two prediction settings was enough | Added the reconstruction-dependent loss gate to the implementation work (§8.2); retain reconstruction and supplied-answer isolation |
| The production plan assumed every prediction chain is row/document-local and reset makes it cold | Kept row/document isolation as a requirement and recorded the consolidated global-recency mismatch (§8.3); durable memory must survive while the prediction view is scoped |
| The plan named `WhatInteractionMemory` as the predictor owner and `interChainWindow` as a knob | Corrected to `InterSentenceLayer` and the current capacity-derived window (§8.2) |
| "Nothing is scored in evaluation" conflicted with held-out learning evidence | Separate read-only validation metrics from training accumulators and optimizer updates (§8.4) |
| A shuffled control was required never to improve; reconstruction was said to guard collapse automatically | Require measured sequence-context benefit and representation/fidelity controls (§8.4) |
| Whether prediction may update the encoder | Latest clarification 2026-09-15: yes. Jointly learn a representation that reconstructs and predicts/answers well, with a combined downstream gradient budget and reconstruction fidelity tolerance (§§8.4, 8.8) |
| The compose and generate catalogs were treated as one shared set of operators | Separate catalogs remain the target (§8.9). Shared operators are protected until migration; live handoffs may still send balanced downstream gradients into comprehension after the split |
| Query and subgoal control was credited only by supplied-answer error | Queries are tools of prediction; migrate credit to observed residuals with explicit attribution and causal tests (§8.10). Detached targets and gradient balance do not prove freedom from shortcuts |
| Prediction was opt-in and could be read as requiring a separately supplied future example | Alec has specified default-on expectation for each eligible arriving input, with an explicit off switch; that input supervises its prior estimate in the same call, and estimate/residual provenance remains explicit (§8.7) |

These are documentation corrections and implementation requirements, not
claims that the code gaps have been fixed by consolidation.

<a id="7-questions-for-alec"></a>

### 9.2 Decisions still open

1. **Extent of concept prediction (resolved by full-spec implementation
   request, 2026-09-16).** The intended object is the encoded
   sentence in LTM, while the existing module predicts a single root and
   the thinking controller requires complete structured meaning (§8.3).
   Preserve the full-meaning target. Root-only prediction is retained as a
   benchmark; production must predict distinct occupied roles and their
   structural references. It cannot substitute for the full inputs to the
   thought chooser. Local-role prediction and retained nested meaning remain
   separately testable parts of that implementation.
2. **Closed, clarified 2026-09-15: joint representation learning.**
   Prediction and supplied-answer feedback may train the encoder; learn a
   representation that supports all objectives under §8.4. Separate module
   ownership and the catalog migration do not impose disjoint gradients.
   Tolerance/ratio values must be reported with quality and throughput evidence.
   Query reward/attribution and the actual prediction-to-answer connection
   remain implementation design work (§§8.5, 8.10).
3. **Validation display.** Held-out results are required evidence; whether
   to add `inter` to the validation epoch line immediately remains the
   production plan's reporting choice. It is not a blocker to computing and
   recording those results for the measurement.

<a id="3-turning-it-on"></a>

## 10. Consolidated implementation and verification order

**September 17 OS checkpoint:** implementation of this spec is unfinished.
Alec requested a pushed checkpoint, deferring the full-suite-green gate for this
checkpoint only. Completing this session and this spec is the next task in
[todo.md](../../todo.md), before the separately reserved September 16 work.
The latest full run timed out; [Testing](../Testing.md#validation) records the
passing focused evidence and incomplete full validation. Resume the item order
below and its ordinary test/commit/push gates; do not treat this checkpoint as
completion or a general relaxation of those gates.

This is implementation dependency order for the remaining work. The requested
joint-gradient revision (item 7) is implemented against the current shared
catalogs, with supporting loss-gate/context fixes from items 1–2; it does not
depend on the later catalog split or default-on rollout. Runtime phase order remains
understanding/reconstruction, then reasoning, then response generation (§1).
The seven fixes in the September 14 ownership plan are already recorded as
complete; this list does not repeat them. Each future implementation item
retains the requested workflow: failing reviewer probe first, fix, affected
test files, full suite green in the background, one basicmodel commit with
the required co-author trailer, push, parent submodule bump and push. Never
include the protected user documents in those commits.
Read the [ownership plan](2026-09-14-answer-path-ownership-and-training.md)
in full before code changes and re-read its
[standing invariants](2026-09-14-answer-path-ownership-and-training.md#2-standing-invariants-re-read-after-every-compaction)
after each context compaction. They include the exact co-author trailer,
protected paths and prohibition on editing `bin/*.py` while pytest runs.
The September 17 [bounded test workflow](../Testing.md) supplies the full-suite
commit gate: one suite per user, fresh sequential workers, memory limits,
worker/overall deadlines and persistent coverage/exit receipts. Use affected
file or node selections during development; use the full selected default suite
before each implementation commit. An incomplete or resource-terminated run
cannot satisfy the gate. Long convergence, memorization and performance checks
run under `RUN_SLOW=1`; default regression completion and optional learning-gate
results must be reported separately. The September 17 user instruction also
requires removing superseded legacy code together with its obsolete tests. Unused reasoning methods
require review and explicit user approval before removal: the planned grammar
queries may introduce their use. Preserve those methods until that decision.

1. **Establish the actual baseline and loss gate.** Reproduce the production
   settings in §8.2. Add a model-level failing probe showing that a positive
   concept-prediction weight alongside reconstruction actually reaches the
   trained total and updates predictor parameters. Preserve the supplied-answer
   gate and prove zero weight yields no predictor update, including after a
   previous Adam step. Existing layer-level tests are useful foundations
   ([test_inter_sentence_prediction_shape.py:264](../../test/test_inter_sentence_prediction_shape.py#L264));
   they do not substitute for the production training-call test.
2. **Verify sequence and representation contracts.** Exercise both supported
   memory adapters, mixed batch rows, packed sentences, cold starts, two
   documents in one stream, partial resets and durable-memory restore.
   Perturb another row, another document, padding or a future target and prove
   the earlier prediction is unchanged. Count each eligible pair once; internal
   thoughts and estimates do not count as corpus observations. Compare streaming
   single-input calls with equivalent packed input (§8.7). Verify the chosen
   root/full scope explicitly, including VP/NP2 sensitivity for complete ideas (§9.2).
3. **Enable expectation by default and measure joint learning.**
   Subject to those gates, set the `sentenceExpectation` default to true and
   explicit `interLossWeight=0.1` for the initial comparison; keep ARMA and
   contrastive weights zero. Verify omitted, explicitly enabled and explicitly
   disabled settings, inference behavior, and disable/re-enable under §8.7.
   Preserve and identify the reconstruction path. Include backward/optimizer
   work and check the predictor's parameters are stepped, with observed targets and
   durable history detached, live current-step context under §8.4, and no
   fabricated answer supervision from `inter`.
4. **Establish learned benefit and throughput.** Use synthetic sequences and
   held-out documents with shuffled/context-free controls (§8.4), then a short
   FineWeb run. Report reconstruction, `inter`, eligible-pair counts,
   document-boundary fraction, input sentences/s, predicted targets/s,
   compilation/warm-up, backend/device and memory. Report answer throughput
   separately when supplied-answer learning runs. The September 11 throughput
   report is historical, not a current production or supervised benchmark.
5. **Complete the thinking and reconstruction migrations.** Implement §6's
   tied sentence-reconstruction contract with its migration evidence. Move
   resolution and subgoal control before `reverseOutput()`; implement structured
   queries, ordinary levelled thoughts, retained nested meaning and causal
   intermediate results under §§1–3. Add owned estimate occurrences, observation
   links and bounded residual-guided reasoning under §8.7, with target-isolation
   and provenance tests. Keep those changes in separate commits
   with §4's phase, lifecycle, semantic, gradient and learned-utility tests.
   Completing the concept-prediction milestone does not close these gates.
6. **Split the comprehension and generation catalogs (§8.9).** Probe first:
   the generate ops and the compose ops currently share 16 modules and 28
   parameters. Resolve `<generate>` rules against a generation-owned
   registry, initialize those instances from comprehension's at construction
   and at checkpoint migration, and verify the generate policy's
   rule-meaning row migration still holds. Test catalog identity and
   input-mediated gradient reach separately, and protect all permitted
   downstream paths into representation parameters (§8.9).
7. **Joint representation learning (§8.4, implemented and verified).** Replace the
   no-encoder-gradient rule. Verify live predictor-to-encoder feedback,
   detached targets, all-downstream aggregation without supplied answers,
   actual shared-grammar ownership, zero/near-zero reconstruction refinement,
   AMP, sparse tensors, and graph cleanup across optimizer steps. Demonstrate
   a more predictive encoding while retaining reconstruction fidelity. Record
   the root/full-meaning milestone separately (§9.2 decision 1).
8. **Move queries to prediction and credit them by the residual (§8.10).**
   Keep §3's episode budget, credit contract and append-only history. Prove
   the chooser trains on an unlabelled corpus, that anticipation reads no
   arriving content, desired answer or unseen packed sentence, and that the
   observed encodings retain reconstruction and discrimination while the
   components jointly learn. Define the chooser's reward and policy baseline;
   ordinary MSE has no baseline to share. Report credit variance and causal
   utility with each component enabled and disabled. Surface and
   percept prediction stay deferred until separately designed and selected.

Update the linked Architecture, Language, Training, Params, STM and Reasoning
documentation as each implementation lands (§5). This consolidated document
remains **unfinished** until its relevant implementation and behavioral gates
have evidence. This revision implements §8.4's joint-gradient behavior and
related training-gate/context fixes; its evidence is recorded there. The
broader gates in this list remain open.

## 11. Code Review (2026-09-16): local-role expectation implementation

Review of the uncommitted September 16 working tree (base `d161a7a`) by
Claude, decisions by Alec. The §8.3 implementation matches its description:
structured NP1/VP/NP2 estimate with presence logits, explicit STM/infix
adapters that agree with the consolidated root path and the non-packed LTM
sink, detached targets with live current-step sources, per-row document
boundaries from cursor addresses, provisioning outside the observation
stream, and a declared root-to-structured checkpoint migration. The two
intended behaviour changes are recorded as accurate: single-sentence FineWeb
training hard-resets only at real document boundaries
([data.py:555](../../bin/data.py#L555)), and the packed drain writes depth-3
sentences to durable LTM as relations, matching the non-packed sink
([Models.py:12527](../../bin/Models.py#L12527)).

Codex is to address the following items in one change. Items are ordered by
importance. Tests are named so the other agent's suite can gate them.

### 11.1 Soft reset must keep the expectation stream (decided)

The packed loop soft-resets each row after a brick with
`space.Reset(batch=b, hard=False)`
([Models.py:14297](../../bin/Models.py#L14297)). SymbolSubSpace cascades that
call to every owned layer ([Language.py:13675](../../bin/Language.py#L13675)),
and the discourse layer's `Reset` discards `hard`
([Layers.py:10490](../../bin/Layers.py#L10490)), clearing the row's
observation view, its document key, the pending estimate and the ARMA rings.
Expectation therefore scores only between sentences inside one brick; the
first sentence of every brick is cold, and the §8.3 sequence contract
("row/document/time order", "continuation within the same addressed
document") does not hold across bricks. The single-sentence path already
avoids this because `soft_reset` leaves discourse history alone.

Required:

- `InterSentenceLayer.Reset` honours `hard`. Soft keeps the observation
  view, the document key, the ARMA rings and durable state; hard clears the
  transient stream as today. Align its default with the Space contract
  (`hard=True`), so a bare `Reset()` remains a document boundary.
- Test: two packed bricks in the same document score a loss on the second
  brick's first sentence (per document, scored pairs = observed sentences
  minus one, across bricks); a hard EOS between bricks still starts cold;
  a document change inside a brick still resets only that row.
- Update the Architecture.md sentence-representation text and the STM.md
  structured-path paragraph, which currently describe the boundary contract
  without the brick case.

### 11.2 Call it expectation, keep it on, never abort on addressing (decided)

Alec's direction: the model forms an **expectation** of the next sentence;
the training signal is the discrepancy between expectation and observation.
Expectation can be turned off, but it normally stays on so discrepancies
stay visible. Two consequences:

- **Naming.** The September 16 surface uses "prediction" throughout. Rename
  it to expectation: `SentenceMeaningPredictor` → `SentenceExpectation`,
  `MeaningPrediction` → `MeaningExpectation`, `predict_next_meaning` →
  `expect_next_meaning`, `_stage_prediction_documents` /
  `_prediction_documents` / `_prediction_documents_for_slot` →
  `_stage_expectation_documents` / `_expectation_documents` /
  `_expectation_documents_for_slot`, `migrate_prediction_checkpoint` →
  `migrate_expectation_checkpoint`, `suspend_external_observations` stays,
  and the configuration element `sentencePredictionScope` →
  `sentenceExpectationScope` (schema, model.xml, Language.py read, test).
  The older knobs `sentencePrediction`, `interLossWeight` and the ARMA
  vocabulary predate this work; rename `sentencePrediction` →
  `sentenceExpectation` in the same change if the blast radius is only the
  XML files, the schema, the one Language.py read and the tests that set it;
  otherwise leave them and record the deferral here. Docs (Architecture,
  STM, Training, Params, this plan's §8.3) follow the code names.
- **Default and robustness.** `model.xml` turns expectation on
  (`sentenceExpectation=true`, structured scope, `interLossWeight` at its
  documented value). `BasicModel.xml` follows once the §10 gates the running
  suite exercises are green; record the flip here when it lands. Document
  staging must never abort a run: today `_stage_prediction_documents` runs
  on every step even with expectation off
  ([Models.py:12911](../../bin/Models.py#L12911)) and raises when a source
  row lies outside the split's address table
  ([Models.py:12369](../../bin/Models.py#L12369)), while the question path
  deliberately folds such rows because some drivers emit them
  ([Models.py:12768](../../bin/Models.py#L12768)). Treat an out-of-range,
  negative or unaddressed row as `None` (one stream per row until reset);
  keep a missing `"document"` key fail-loud, since every loader writes it.
  Test: a driver emitting rows beyond the split extent trains without error
  and forms one stream per row.

### 11.3 One owner for the What interaction memory (decided)

`WhatInteractionMemory` is live thinking state (What-episode slots, parity,
closure pressure, episode credit; `Model.think()` reads it through
`Model._what_memory()`), not legacy. What is legacy is its **dual
ownership**: SymbolSubSpace builds one under `<whatThinkingMemory>` when
expectation is off ([Language.py:10974](../../bin/Language.py#L10974)),
the discourse layer composes another when expectation is on
([Layers.py:9380](../../bin/Layers.py#L9380)), and the discourse layer
carries a pass-through delegate API so old callers still find the slots on
it ([Layers.py:9680](../../bin/Layers.py#L9680)). The new suspension
context then has to snapshot `vars(what_memory)` shallowly
([Layers.py:9748](../../bin/Layers.py#L9748)); with a batch-1 caller the
resets during provisioning clear the same deques the snapshot points to,
so the docstring's preservation claim is false for the What slots.

Required (Alec's rule: the codebase does not keep legacy paths in source):

- SymbolSubSpace owns exactly one `WhatInteractionMemory`, built always.
  Retire the `<whatThinkingMemory>` element (schema, model.xml, Params.md
  row, Language.py and Models.py reads). `Model._what_memory()` reads the
  one owner. The `<whatThinkingDetach>` knob stays.
- Delete the discourse layer's composed copy and every delegate
  (`_what_slots`, `_what_closure_pressure`, `detach_mode`,
  `append_what_slot`, `get_what_slots`, `open_what_slots`,
  `what_open_depth`, `what_at_parity`, `what_context`,
  `begin_what_episode`, `in_episode`, `end_what_episode`, and the
  staticmethod aliases). Move callers and tests to the owner
  (`test_what_episode_memory.py`, `test_output_*.py`, `test_ltm_consolidation.py`).
- `suspend_external_observations` then covers only the expectation stream
  (batch, rings, observation view, document keys, pending estimate, scored
  losses). Provisioning's hard Reset of the What slots is the existing
  contract and is not part of the suspension. Test: a provisioning call in
  the middle of an open What episode behaves exactly as before this change.
- Update STM.md §"What interaction slots" (currently "two homes",
  [STM.md:1002](../STM.md#L1002)) and the Params.md rows to the single owner;
  correct this plan's §8.2 table row that calls the member "legacy".

### 11.4 Intermediate packed slots: assert the seal layout (decided)

The packed drain reshapes `_tensor_sentence_roots_live[b, t]` to `[3, D]`
and adapts it with the same newest-at-0 STM permutation as the final seal
from `_final_end_state` ([Models.py:12497](../../bin/Models.py#L12497)).
The parity test builds both banks from one tensor, so it cannot detect a
layout mismatch between the intermediate bank and the seal. Add one
real-brick assertion in `test_reconstruction_priority.py`: when a brick ends
exactly on sentence `t`, the intermediate slot `t` and the final seal yield
the same canonical `[NP1, VP, NP2]` and occupancy for that row.

### 11.5 Stale references (decided)

- §8.4 "Context cleanup is implemented in Layers.py:10141" points at a
  comment inside `predict_and_observe_stm_end_state`; it should cite
  `detach_prediction_context` ([Layers.py:10412](../../bin/Layers.py#L10412)).
- §8.4 "packed chronology and the single-drain guard are in Models.py:12386"
  points at `_prediction_documents_for_slot`; cite the drain
  ([Models.py:12447](../../bin/Models.py#L12447)) and the guard at its site.
- Architecture.md and STM.md cite `Layers.py:9763` as "InterSentenceLayer";
  that line is `begin_document`. Cite the class or the method by name.
- §8.3 cites `Layers.py:10215` for two different claims (consolidated root
  reduction and the legacy newest-first adapter); both resolve to
  `_reduce_end_state_to_root`. Cite the two code sites separately.
- After the 11.2 renames, re-verify every line reference in §8 and in
  Architecture.md, STM.md, Training.md, Params.md and Reasoning.md, and
  prefer function names to bare line numbers where the surrounding prose
  allows it.


### 11.6 Implementation and validation (September 16)

The review fixes implement the local-role sequence milestone and §10.3's
switch/default/inference behavior together. The historical citations in
§§11.1–11.5 describe the reviewed uncommitted tree; the following links name
the current implementation.

- [`InterSentenceLayer.Reset`](../../bin/Layers.py#L10809) preserves the stream
  on soft resets. Hard reset and [`begin_document`](../../bin/Layers.py#L10027)
  start the affected row cold. The real two-brick regression scores three
  pairs from four observations in one document.
- [`_stage_expectation_documents`](../../bin/Models.py#L12726) treats invalid
  source-row positions as unaddressed streams, while malformed addresses
  missing their document key fail explicitly. Existing cursor addresses still
  enforce document boundaries inside packed rows.
- `SentenceExpectation`, `MeaningExpectation` and `sentenceExpectationScope`
  are the new names. The old `sentencePrediction` configuration element is
  renamed to `sentenceExpectation` throughout runtime reads, schema and XML.
  Both `model.xml` and `BasicModel.xml` enable expectation with `inter=0.1`,
  ARMA zero and contrastive zero. Named experiment files retain their explicit
  overrides. [`set_sentence_expectation`](../../bin/Models.py#L12700) supports
  off/on at runtime, including first construction after an off start; its
  parameters join the optimizer once and disabled Adam steps leave them fixed.
  Head construction preserves the caller's random stream, allowing matched
  initialization of comprehension when comparing expectation on/off.
- [`SymbolSubSpace`](../../bin/Language.py#L10961) owns one always-present
  `WhatInteractionMemory`. The `whatThinkingMemory` switch, discourse copy and
  delegates are removed. Provisioning resets the interaction episode while
  [`suspend_external_observations`](../../bin/Layers.py#L9999) protects only the
  external expectation stream. Temporal prediction reads the discourse owner
  directly ([`_temporal_answer_rep_row`](../../bin/Models.py#L8817)).
- [`expectation_metrics`](../../bin/Layers.py#L10714) reports observation/pair
  counts, cold starts, document transitions, feature MSE and presence BCE.
  [`last_expectation_comparison`](../../bin/Layers.py#L10727) returns owned,
  detached prior/observation/residual values. These inspection values do not
  yet establish the durable occurrence-link contract in §8.7.
- [`runBatch`](../../bin/Models.py#L13193) applies its declared train/evaluation
  mode and grad gate. Real runtime calls report comparisons without training
  accumulation or updates. A controlled future-sentence/other-row perturbation
  leaves the earlier estimate unchanged. Fully masked observations are skipped.
- The packed LTM sink also runs when no expectation head exists, and ignores
  masked storage slots. A real off-start input previously stored only its final
  sentence; the regression now requires both observations.
- The real intermediate/final-seal regression checks canonical roles and
  occupancy from separately executed packed bricks, rather than constructing
  both banks from one test tensor.

The final affected run passed **345 tests, with 4 skips and 2 expected
failures, in 752.37 seconds**. Native `xmllint` validates `model.xml` and
`BasicModel.xml` against the schema. New probes reproduced the temporal-owner,
packed off-switch storage, masked-slot storage and random-stream defects before
their fixes; the original input-variation thresholds remain unchanged.
The checkpoint-key, AMP dispatch, staging and memory-owner compatibility checks
passed another **41 tests in 22.33 seconds**, preserving the original
non-expectation checkpoint-key hashes. The full background suite passed
**4,143 tests, with 51 skips, 7 expected failures and 4 subtests passed, in
2,114.95 seconds (35m 14s)**. Source files stayed fixed throughout that run.
The earlier 4,119-pass run in §8.3 is only pre-review evidence.

Current links in §8, Architecture, Language, STM, Training and Params have been
reviewed. `doc/Reasoning.md` contains no numeric code links and remains unchanged
under Alec's explicit protection of his uncommitted documents. The throughput,
tied reconstruction, complete nested meaning, levelled thinking and learned
query-utility gates in §10 remain open.

## 12. Learning and throughput measurement (September 16)

The [item 4 report](../benchmarks/2026-09-16-sentence-expectation.md) records
the preregistered protocol, exact configuration/runtime hashes, raw results,
failed memory runs and reviewer probes. Measurements and full-suite verification
are complete for this item.
The first full run terminated under host memory pressure at 81%; the report
retains that diagnostic and records the test-cache cleanup before the rerun.
The complete rerun passed **4,157 tests, with 51 skips, 7 expected failures and
4 subtests passed, in 2,466.59 seconds (41m 6s)**. Runtime and test files stayed
fixed throughout; the protected user documents remained unchanged.

- The actual local-role expectation head passes the fixed-meaning context
  gate in seeds 0, 1 and 2: held-out MSE is about 97.6% below both matched
  shuffled and context-free controls. This is predictor learning, not joint
  corpus encoder evidence.
- Native FineWeb W256/B1 training processes **0.260 input sentences/s** and
  **0.256 eligible prediction targets/s** after two warmup steps. Seven actual
  optimizer calls observe 85 inputs and score 83 targets. The short held-out
  corpus result is negative for vector prediction: MSE rises from 0.008798 to
  0.023064, despite improved role presence. Discrimination/collapse protection
  and learned querying are not established by this run.
- Separately supplied three-word numeric answer training processes **0.145
  sentences/s at B1** and **0.286 at B2**. Answer parameters update in both
  runs; all observations are cold starts, so no continuation target is invented.
- The native runs use MPS/eager capture with FP32 and the current detached
  reconstruction student. They are not Inductor throughput measurements or
  evidence completing §6's tied-reconstruction migration.

The memory probe exposed evaluation of every binary operator while replaying
one recorded compose choice. [`forward_binary_step`](../../bin/Language.py#L14615)
now executes only selected operators. The supplied-answer probe also exposed
dropped compiled/unpacked observations. The existing 21-value return now
feeds the host boundary explicitly through
[`_publish_compiled_sentence_state`](../../bin/Models.py#L7278) and
[`_drain_pending_stm_end_state`](../../bin/Models.py#L12777), preserving depth and
padding masks. The report identifies the pre-fix failures separately from the
final measurements.

The next ordered work is §10.5. Alec explicitly reserved
`doc/specs/2026-09-16-two-truths-ideas-and-relations.md` for work after this
session; this implementation does not adopt that separate fusion/collapse
contract. The broader thinking, reconstruction and learned-utility gates in
this integrated specification remain unfinished.

## 13. Tied input-reconstruction migration (verified)

The native measurement matrix and background full-suite verification are
complete. The final run passes **4,220 tests, with 51 skipped, 7 expected
failures and 4 subtests passed, in 4066.68 s**. All 554 frozen runtime/test/config
files and the 12 native measurement fingerprints are unchanged. Two earlier
full runs exposed stale test setup/hooks; their entire affected files pass
without runtime changes. The broader thinking and learned-utility migrations
remain open. An earlier
packed FineWeb AOT run completed validation but failed before its first optimizer update when a cached backward
could not support retained reads for gradient balance. The compiler fix disables
buffer donation within reconstruction compilation and normalizes the disabled
metadata; a probe now passes ordinary-first and subsequent retained reads with
the global setting preserved. A separate reviewer probe exposed the byte-only
scorer's inability to distinguish a word from a longer candidate sharing its
prefix. The fixed objective scores the existing NUL byte (`0`) as the word
terminator, including it once and ignoring later padding. It uses the existing
256-byte alphabet and adds no new symbol. Retain the completed
byte-only timing runs as the cache comparison, and remeasure the final objective.
See [retained failure and measurements](../benchmarks/2026-09-16-tied-input-reconstruction.md).

BasicModel selects a separate fullgraph reconstruction using
`reconstructionPlacement=compiled`. The `eager` forward backend promotes this
call to `aot_eager` so its backward program is cached. The original in-graph
baseline rebuilt 422 higher-order backward fragments per probe call; its
completed native run measured 0.00820 supervised input sentences/s. The new
cache probe passes. The final native B=1/B=2 supplied-answer runs measure
0.08720/0.17229 input sentences/s, versus 0.14370 for the matched B=1 detached
student. Packed FineWeb completes seven optimizer steps at 0.14236 input
sentences/s and 0.14010 prediction targets/s after warmup.
Its held-out byte cost improves, but prediction feature error worsens; this short
run does not demonstrate predictive benefit. The final 16-word workload completes
at 0.06519/0.06508 sentences/s for basis limits 8/16, with improved validation
byte, idea and event errors. All five candidate measurements include the
output-readout correction below;
these single runs do not establish a robust ranking or worst-case search cost.
The corrected B=2 output run completes fifteen optimizer steps, measuring
0.16762 supervised input sentences/s after warmup. Its generated lengths are
1, 2, 3 and 6 words from three-word inputs, with no truncation. This restricted
sum/stop workload establishes execution and cost, not learned language quality.
Full-suite verification passes. See the
[measurement report](../benchmarks/2026-09-16-tied-input-reconstruction.md)
and [implementation](../../bin/Models.py#L11257).

These learning and checkpoint declarations accompany the verified BasicModel
default change required by §6. The measurements expose its cost and fidelity
limits; they do not establish learned predictive utility. The measured
detached-student baseline is preserved in §12 and its
raw reports at commit `aa5e018b67bf3be946c0b75c5baf33c9cc84ab4b`.

The intended production settings are `teacherReconstruction=true`,
`detachedReverse=false`, `reconstructInLoop=true`. Explicit legacy experiments
may retain the detached student; the two reconstruction modes remain mutually
exclusive. Training and evaluation consume the same completed, owned input
reconstruction and the same byte objective once. They must not add a second
D3 or event-reconstruction loss. The optional event score returned when a caller
supplies a target to `reverseReconstruct` remains a diagnostic of that target;
changing the target cannot change the already reconstructed surface.

The objective is cross entropy over each word's bytes through its first NUL
terminator, averaged per active word, then per completed sentence, then per batch
row. Eager staging expands each input percept ID to its full bytes for scoring;
whole-word and prefix promotions retain the same target spelling. Padding after
termination is not scored. Candidate spellings are stored WORD surfaces,
with OBJECT rows following their current WORD associations. The null candidate
retains a uniform `log(256)` cost when there is no known spelling but a scoreable
target. NUL follows the existing [token-buffer contract](../../bin/Spaces.py#L1617).
Idea MSE,
continuous input-event error, truncation and actual reconstructed lengths are
reported separately. Cosine-based byte assignment does not constrain the
amplitude of an otherwise identical concept direction; this objective must
not be described as exact continuous-state reconstruction.

Gradients pass from the byte objective through the recovered ideas, selected
compose inverses and live sealed input representation. The compose path owns
the learned transforms; reconstruction adds no decoder parameters. Targets,
dictionary snapshots and occurrence-specific constituent witnesses are
detached. Reverse trace indices are constants. The forward chooser retains its
existing straight-through soft approximation, reachable through the completed
representation ([Language.py:7874](../../bin/Language.py#L7874),
[Language.py:14029](../../bin/Language.py#L14029)); the separately weighted local
grammar-choice objective remains explicit. Reverse replay adds no new selection
estimator or gradient through the recorded integer indices.
The joint-gradient rule in §8.4 remains in force for all downstream losses.

Sigma and Pi invert their learned affine map and subtract a known operand in
the corresponding chart; an unknown split is balanced. Verb reversal uses its
actual spectral transform with a known verb. Adverb reversal uses eight bounded
fixed-point corrections with its shared edit transform. Lossy/missing-operand
reconstruction uses the selected compose kernel over a masked invocation-owned
basis snapshot: at most `reconstructionBasisLimit` candidates per side (default
16), hence at most its square in pairs. This budget is independent of word,
STM and field capacities. Candidate reconstruction is approximate and must
have separate child/sentence fidelity measurements. Unavailable inverses or
exhausted traversal bounds report incompleteness; a duplicated parent is not
evidence of a faithful inverse. Output receives no input-reconstruction basis
or constituent witness; its unavailable requested operations stay pending and
report truncation under its own budget.

Checkpoint migration retains the compose parameters and their optimizer state
by name, drops detached reverse-student parameters and their optimizer entries,
and initializes no replacement reconstruction decoder. Restoring a tied model
into an explicitly selected legacy student mode creates a fresh student.
Before the output-readout correction, the affected group passed 211 tests (6 skipped),
including strict real checkpoint/optimizer migration and packed joint-gradient
training. The output-length native run then exposed a 1,156 GiB square-factor
allocation in the final answer readout. New adapters now store only the active
rectangular LDU factors, preserving the same forward function and gradients;
legacy checkpoints retain their layout. A further 27-test group passes, including
compact/legacy checkpoint and Adam continuation, the real `auto` compiler probe,
and explicit tied/legacy L1 modes. All three output synthesis/supervision/walk
files also pass, with 70 tests. This readout is not a reconstruction decoder
([Layers.py:1637](../../bin/Layers.py#L1637),
[Spaces.py:30277](../../bin/Spaces.py#L30277)). Matched native
measurements (including failed runs), documentation reconciliation and the
green background full suite complete this reconstruction migration's validation.
[Full-suite result](../benchmarks/2026-09-16-tied-reconstruction-data/full-suite-green.log).


## 14. Prepared-answer boundary (verified)

`resolveAnswer(understanding, questions)` now prepares the owned conceptual
answer before `reverseOutput(understanding, derivation)` expresses it. The
output API requires an `AnswerDerivation`; passing unresolved questions is an
error. Target-free presentation metadata belongs to that value, and repeated
realization consumes the same value without running resolution or queries.
`what` explicitly performs the handoff. The owned conceptual clone retains
current-step gradients, and standalone compiled realization initializes its
own gradient anchors even when understanding was captured under `no_grad`.
[Preparation](../../bin/Models.py),
[realization](../../bin/Models.py),
[owned value](../../bin/Output.py).

The initial reviewer probes fail in all seven cases on `b5518a1`: the held
answer calls re-enter the forbidden resolver, and the public preparation API
is absent. The first implementation also exposes a missing gradient anchor in
standalone compiled realization; that path must work without an earlier
training forward. All seven boundary probes pass (85.78 s), and all seven affected files pass
**109 tests, with 16 warnings, in 381.84 s**. The full suite also passes: **4227 passed, 51 skipped, 7 xfailed, 185 warnings, 4 subtests passed in 5568.01s (1:32:48)**, with exit status zero.
[Full-suite green](../benchmarks/2026-09-16-prepared-answer-data/full-suite-green.log),
[frozen source manifest](../benchmarks/2026-09-16-prepared-answer-data/full-suite-source-manifest.json).
Tests use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to avoid CPU oversubscription.
[Affected green](../benchmarks/2026-09-16-prepared-answer-data/affected-green.log).
[Original red](../benchmarks/2026-09-16-prepared-answer-data/boundary-red.log),
[standalone-gradient red](../benchmarks/2026-09-16-prepared-answer-data/standalone-gradient-red.log),
[boundary green](../benchmarks/2026-09-16-prepared-answer-data/boundary-green.log).
The output-length probe uses real
captured one-unit inputs and a controlled sum/stop policy; it tests independent
execution and termination, not learned linguistic quality.

This change established the preparation/realization API. The September 20
controller replacement now executes selected grammatical queries and nested
meanings in one ordinary episode before realization. Natural-wording learning
and residual credit remain open; see [SelectedMeaning](../SelectedMeaning.md).
Explicitly unresolved temporal answers remain unresolved; output does not
substitute the current input.

Full-suite command, from `basicmodel/` (runtime, tests and configuration frozen throughout):

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```

## 15. Complete-description existence evidence (verified September 16)

The receipt counts below describe the September 16 source. Controller
references reflect the September 20 replacement and its review corrections.

`ConceptualMeaning` retains canonical occupied roles, grammatical mode,
polarity, bindings, scope and constituent references. The existing truth
store owns durable facts/observations and their stable occurrence IDs.
`Exist` matches the entire description against assertive facts, keeps
positive/negative degrees separately, reports provenance and incomplete
metadata, and never substitutes concept activation for referent evidence.
The normal controller preserves both support degrees in its typed final result.
[Meaning](../../bin/Meaning.py),
[store](../../bin/Layers.py),
[lookup](../../bin/reasoning.py),
[controller and typed history](../../bin/Thoughts.py).

All three observation writers share the explicit physical-to-canonical
adapter, retaining depth-two VPs. Explicit TruthSet admission accepts facts;
source tags alone do not. Questions, estimates and unspecified observations
cannot satisfy Exist. Legacy relation writers without a grammatical VP
remain unverified for that lookup. Prediction results retain complete
descriptions and source identity as estimates, with no fact authority.
[Observation write](../../bin/Models.py),
[admission](../../bin/Layers.py),
[typed results](../../bin/Thoughts.py).

The existing structural sidecar stores context and source text, bound to the
tensor occurrence by a content fingerprint. Partial semantic checkpoints and
missing, replaced or swapped context are rejected; a tensor-only restore
quarantines evidence whose required metadata is absent. Legacy checkpoints
have an explicit migration. No learned parameters or loss were added; live
meaning clones preserve gradients and durable writes detach them.
[Sidecar](../../bin/Models.py),
[restore](../../bin/Layers.py),
[legacy migration](../../bin/Layers.py).

The repository's **51 new probes pass in 3.11 s**. The ten affected existing
files pass **281 tests, with one skip and three warnings, in 89.56 s**.
The full suite passes with exit status zero: **4279 passed, 51 skipped, 7 xfailed, 185 warnings, 4 subtests passed in 4175.57s (1:09:35)**. All 561
runtime/test/configuration files remained frozen.
[Full-suite green](../benchmarks/2026-09-16-existence-evidence-data/full-suite-green.log),
[frozen source manifest](../benchmarks/2026-09-16-existence-evidence-data/full-suite-source-manifest.json).
[Focused green](../benchmarks/2026-09-16-existence-evidence-data/focused-green.log),
[affected green](../benchmarks/2026-09-16-existence-evidence-data/affected-green.log).
Original and follow-up failing probes were run before their corresponding
fixes. The isolated candidate was checked separately and does not substitute
for the repository gate. See [Existence evidence](../ExistenceEvidence.md)
for contracts, limits and validation artifacts.

This is the existence evidence foundation; taxonomy evidence follows in §16.
The typed VP registry, ordinary controller and selected nested execution are
implemented. Anticipation read isolation, residual policy credit and
learned-utility gates remain open. The September 16 two-truths spec remains
separate work.

Full-suite command from `basicmodel/` (`OMP_NUM_THREADS` and `MKL_NUM_THREADS` unset):

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```

## 16. Conceptual-taxonomy PartOf evidence (verified September 16)

The receipt counts below are historical; current execution and retirement
contracts are in [TaxonomyQueries](../TaxonomyQueries.md).

`PartOf` now reads native conceptual reference records, including reified
relations without endpoint mutation. Typed concept handles remain addresses;
perceptual edges, vector overlap, sparse weights and LTM world rows do not
establish taxonomic inclusion. Proofs preserve their source owners and roles.
Known paths support inclusion; missing paths are unknown, including at a zero
posture threshold. Negated requests retain the opposite support channel.
[Capture](../../bin/Taxonomy.py),
[query](../../bin/reasoning.py).

The public model entries use this source through the sole normal controller.
Native premises support reader proofs; nested `what` returns carry their typed
child result and causal sources. Unrelated true children and unaccepted
predictions cannot certify inclusion. Results preserve unknown and
incomplete-read diagnostics. Successful proofs do not write world facts.
[Model entry](../../bin/Models.py),
[controller](../../bin/Models.py),
[write boundary](../SelectedMeaning.md).

Capture and traversal have explicit node/record/edge limits. The derived
view adds no authoritative memory, checkpoint schema, trainable parameter or
gradient objective. Existing structural checkpoint ownership preserves source
records; hard path selection has no ordinary derivative. The frame curriculum,
geometric/world-row proposal helpers and their separate policy objectives are
removed. The normal chooser's supplied-answer credit does not establish useful
learned decomposition. [Current policy](../SelectedMeaning.md),
[retirement dispositions](../KernelRetirement.md).

The repository's **37 new probes pass in 3.33 s**. The sixteen affected
files pass **277 tests, with one skip, two expected failures and seven
warnings, in 173.39 s**. The full suite passes with exit status zero: **4317 passed, 51 skipped, 7 xfailed, 184 warnings, 4 subtests passed in 5604.66s (1:33:24)**. All 569
runtime/test/configuration files remained frozen.
[Focused green](../benchmarks/2026-09-16-taxonomy-query-data/focused-green.log),
[affected green](../benchmarks/2026-09-16-taxonomy-query-data/affected-green.log),
[full green](../benchmarks/2026-09-16-taxonomy-query-data/full-green.log),
[frozen source manifest](../benchmarks/2026-09-16-taxonomy-query-data/full-validation-manifest.json).

Full-suite command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```
Original public-routing probes failed before implementation, as did the
additional bound/provenance/entry/policy/testimony/helper/unknown probes before
their respective fixes.
See [Taxonomy queries](../TaxonomyQueries.md) for contracts and evidence.

The explicit checked shared-VP API follows in §17. Surface/sense agreement,
causal answer adaptation, ordinary levelled control, nested retention, anticipation isolation,
residual policy credit and learned-utility gates remain open. The separate
two-truths spec remains reserved for the next session.

## 17. Checked query contracts and shared grammatical VP API (verified September 16)

Static grammar loading validates executor availability, typed signatures and
role/domain contracts before mutating the grammar. Complete and production
ladder grammars declare `what(Q)` alongside the distinct two-operand LTM lookup.
The real loader preserves whole query and anchor strings. Every checked entry
has an executor, result type, read/write scope and evidence semantics; tense
execution remains deferred.
[Loader](../../bin/Language.py),
[contracts](../../bin/Queries.py).

An explicit setup adapter binds one native named conceptual VP per relation
and domain. Compose/query aliases and converse forms share that identity;
mode belongs to the complete idea. Pure formation retains canonical roles,
open/absent masks, references, polarity, bindings and scope. Selected execution
resolves its interface from the middle VP and occupancy and retains the full
proposition with its evidence. Assertions, undeclared calls, unavailable
referents and wrong widths cannot execute.
[Binding](../../bin/Queries.py),
[formation](../../bin/Queries.py),
[dispatch](../../bin/Queries.py).

Description-valued arguments use existing occurrence references and bounded
full-description reads. The compound NP's illumination summary is not the
structural record; the executor resolves all occupied roles and context.
No candidate may silently commit a description or invent an occurrence.
Captured leaf programs also retain forward's selected native OBJECT/WORD IDs,
with legacy unknown IDs remaining unknown. Native bindings use the existing
conceptual checkpoint owners; no parallel semantic memory or parameter table
is added.
[Occurrence reader](../../bin/Queries.py),
[leaf identity](../../bin/Models.py).

The 53 new reviewer cases cover declarations, effects, loader/copy behavior,
native leaf identities, shared VP/checkpoint identity and width/occurrence bounds.
The initial focused run passed 57 tests; the 36-file affected run passed
353 tests with five skips in 366.32 seconds, including output, reconstruction
and compiled word paths. After the final direct-call width guard, 18 affected
files passed 138 tests with one skip in 13.31 seconds. Each managed run exited
zero. The full suite completed: **4371 passed, 51 skipped, 7 xfailed, 184 warnings, 4 subtests passed in 7222.95s (2:00:22)**. All 579
runtime/test/configuration files remained frozen.

[Initial focused](../benchmarks/2026-09-16-checked-query-data/focused-green.log),
[affected](../benchmarks/2026-09-16-checked-query-data/affected-green.log),
[final affected](../benchmarks/2026-09-16-checked-query-data/final-affected-green.log),
[declaration red](../benchmarks/2026-09-16-checked-query-data/declarations-red.log),
[VP red](../benchmarks/2026-09-16-checked-query-data/grammatical-vp-red.log),
[width red](../benchmarks/2026-09-16-checked-query-data/owner-width-red.log),
[full green](../benchmarks/2026-09-16-checked-query-data/full-green.log),
[frozen source manifest](../benchmarks/2026-09-16-checked-query-data/full-validation-manifest.json).

Full-suite command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```
See [Query contracts](../QueryContracts.md) for source links and limitations.

These APIs do not yet replace the normal raw-text legacy resolution path.
Selected linguistic derivation capture through all observation writers,
surface/sense agreement, causal answer construction,
ordinary levelled history and shared budget, nested retention, live episode
occurrence reads, anticipation isolation, residual policy credit, separate
compose/generate catalogs and learned utility remain open. The two-truths spec
remains reserved for the next session.

## 18. Ordinary thought-history and live occurrence APIs (September 17)

`WhatInteractionMemory` now owns ordinary structured thought records and
explicit begin/descend/return/cutoff/finish transitions in its existing
row-local deque. Replay recovers active and suspended contexts without a
second authoritative stack; same-level continuation and root finish are
distinct, and sibling re-entry starts a fresh context. One root budget spans
all ordinary work. Cutoff freezes the active depth and permits only a bounded
LIFO drain plus one root finish. Capacity cannot evict active or referenced
records.

Thought occurrence references are stable, row-local references to complete
meanings. They may be read live through the existing owner during an episode,
but do not create facts, execute queries, or change execution level. Checkpoint
sidecars contain detached copies; restore validates every row and preserves the
remaining budget and pressure. Explicit finish precedes credit release.

This is a storage/lifecycle foundation, not normal-controller integration or
learned utility. The reviewer set covers replay, row/sibling isolation, bounded
drains, retention, checkpoint restore, live query reads, and legacy credit
boundaries. See [ordinary thought history](../ThoughtHistory.md) and
[gradient flow](../GradientFlow.md).

## 19. Sentence masks and completed-row query permission (September 18)

`resolveAnswer()` now requires its owned tied reconstruction when that mode is
enabled and opens checked query execution only for nonempty captured-program
rows. Missing/padded rows cannot execute a query; held understandings keep
their own readiness after later staging. Permission restores after success or
failure, and nested resolution can narrow but not widen an outer set of rows.

Input forward paths, supplied executors, input reconstruction and output
realization mask checked query execution. Signature and shared-VP dispatch
guard before occurrence reads or effects; legacy query entry points reject
sentence-time use. Compiled tracing rejects a query directly while the host
wrapper covers compiled numerical calls and eager islands. The 21-value forward
contract, parameter set, loss set and gradient ownership are unchanged.

The September 18 reviewer set passed 23/23, including real fullgraph forward
and backward, and the broader affected selection passed 280/280 in bounded
workers. See [Query phases](../QueryPhases.md) for receipts and limits.

This is execution permission only. Normal selected linguistic meaning, the
ordinary controller, nested retention, shared executor work, anticipation
isolation, residual policy credit and learned utility remain open. The
separately queued two-truths and forgetting work remains out of scope.

## 20. Bounded nested-occurrence retention (September 18)

The existing durable truth owner now derives bounded structural views with
ordered role edges, repeated references and independent node/depth/record
limits. Missing, cyclic and unavailable structure is explicit. Each occurrence
retains its own scope and evidence; support for a reporting clause does not
transfer to an embedded question or claim.

Origin clearing retains the closure of surviving records and thought-owned
roots while withdrawing request evidential authority. Full checkpoint restore
restores semantic and thought owners before pruning; tensor-only restore keeps
zero-trust content when ownership metadata is absent. Durable views detach;
live episode gradients retain their existing boundary. See
[nested retention](../NestedRetention.md).

The preserved reviewer probes were red first; the focused 68-case and broader
309-case bounded affected gates are green. Shared query work remains next. This
does not complete typed linguistic observation, the normal controller, actual
shared executor accounting, expectations/residual credit, generation ownership
or learned utility. The separate two-truths and forgetting work remains
reserved.

## 21. Shared selected-query work accounting (September 18)

QueryWorkBudget now carries one transient allowance through selected VP and
operand preparation, validated executor invocation, occurrence/fact reads,
taxonomy capture and proof traversal, codebook candidates, prediction-context
reads, and nested selected callbacks. Each action charges before its native
read. Local node, record, depth, and expansion limits can tighten the meter but
cannot renew it; taxonomy capture reserves bounded proof work. Exhaustion is
explicit work_budget incompleteness and preserves any partial signed evidence.

The meter adds no semantic owner, checkpoint state, parameter, loss, or
ordinary derivative. It preserves existing live payload and thought-read
gradients while durable LTM reads remain detached. Hard reference/path choices
remain nondifferentiable and still require policy credit.

The reviewer run first failed before the module existed; the focused 13-case
selection and a non-overlapping 372-case affected selection are green in
bounded workers. See [shared query work](../QueryWork.md) for receipts and
limits.

This is accounting only. The normal controller must still create one meter from
its episode allowance, propagate it through every selected callback, and
record actual cost once alongside causal result use and policy credit. Selected
linguistic meaning, expectations/residual learning, generation ownership, and
learned utility remain open. The separately queued two-truths and forgetting
work remains out of scope.
