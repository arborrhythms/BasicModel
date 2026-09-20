# Reasoning System

> **2026-05-29 delta:** the chart / signal-router reverse path now
> passes the space-role-local Basis (`subspace.what`) to binary GrammarLayer
> reverses as `basis=space_role_basis`. The mereology-guided recommender
> (`Ops._binary_op_recommend`) walks the Codebook's `W` rows to find
> operand pairs $(x_1, x_2)$ such that $\mathrm{op}(x_1, x_2) \approx \mathit{parent}$. Under
> `<codebook>none</codebook>` on WholeSpace the recommender has no
> rows to walk and falls back to the lossy `(parent, parent)`
> pseudo-inverse — degrading the reasoning loop's structural recovery
> on multi-stage chart parses.

Truth-aware model methods plus the query-reasoning helpers in
`bin/reasoning.py`: `QuerySpec`, `TruthGroundedReasoner`, and the retained
numerical prediction/loss experiments. `BasicModel.run_selected_thought` is
the one query controller. Builds on the TruthLayer infrastructure
([Logic.md](Logic.md)) and grammar composition ([Language.md](Language.md)).

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

Reasoning is the point where BasicModel uses explicit structure instead of
asking an LLM-style prior to improvise an answer. Formal Concept Analysis
contributes the ordered concept support that makes grounding and entailment
auditable. DisCoCat contributes the typed composition path that turns phrases
and sentences into candidate propositions. The reasoner then checks those
propositions against the TruthLayer rather than treating fluent continuation as
evidence.

## Partitioned Symbol Space

> **Terminology (percept / concept / symbol).** Throughout this doc
> "symbol"/"symbolic" denotes the genuine SymbolSpace space-role — the 0-D,
> non-dimensionally-embedded references emitted as `symbolSum` — not the
> ConceptualSpace part$\leftrightarrow$whole relation table (those are *concepts*) and not
> the dimensionally-embedded perceptual content of PartSpace/WholeSpace
> (those are *percepts*: part-percepts and whole-percepts).

The symbol dimension is statically partitioned across conceptual orders
using geometric decay. Each order writes only to its slice of `symbolSum`,
while reading the full vector as feedback.

```
order 0:  [0,      D//2)       <- 1/2 of symbol_dim
order 1:  [D//2,   3D//4)      <- 1/4
order 2:  [3D//4,  7D//8)      <- 1/8
...
last order: remainder of D
```

Makes the symbol partition **self-describing**: position reveals conceptual
order. Truth methods use `_activation_order()` to determine a query's order
by finding the partition with the highest energy. Partition boundaries are
precomputed once at model creation via `BasicModel._order_partitions`.

## Reasoning Methods

### `isConsistent() -> dict`

Analyzes the TruthSet for internal consistency by folding all stored truths
into a single summary via successive `Ops.disjunction`. In bitonic mode,
conflicting +/- assertions on the same dimension cancel to zero. Returns
`{'consistent': bool, 'score': float, 'sites': tensor, 'union_vector': tensor}`.

### `ground(activation, threshold=0.6) -> dict`

Finds the minimal subset of the TruthSet entailing a query activation. Uses
`_activation_order()` to filter truths by partition. Falls back to
`TruthLayer.derive()` for indirect derivation. Returns
`{'grounded': bool, 'basis': [indices], 'trace': [...], 'confidence': float}`.

### `isTrue(activation) -> float`

Grounds a proposition and returns a scalar Degree of Truth in [-1, 1].
Positive = true, negative = false, zero = unknown. Delegates to `ground()`.

### `extrapolate(seed_indices, max_new, attenuation) -> dict`

Generalizes `TruthLayer.derive()` to all two-argument grammar methods (union,
intersection, `isEqual`, part). For each pair of stored truths, applies every
eligible method and accepts results that preserve or increase luminosity.
Accepted truths recorded at `attenuation * min(DoT_i, DoT_j)`. Returns
`{'added': [indices], 'rejected': [(i, j, rule, delta_lum), ...]}`.

> **Meronomy reconciliation (2026-06-11).** The gate's role is
> unchanged, but the gated quantity is now the MeronomySpec §3 rev-b
> measure: `TruthLayer.luminosity` = the catuṣkoṭi coverage measure
> over the codes, `mean_k[(T_k − F_k) − min(T_k, F_k)]` — signed area
> minus conflict — order-independent, no decode pullback. The same
> applies to the multiplicative luminosity modulation under
> "TruthLoss" below.

## TruthLoss

Additive loss penalty for false propositions, via `<TruthLoss>` in model.xml
(default 0.0 = disabled).

Measures the **union norm reduction** when a proposition is included in the
TruthSet union via `Ops.disjunction`:

```
truth_union = disjunction(all stored truths)
extended    = disjunction(truth_union, new_proposition)
penalty     = max(0, ||truth_union|| - ||extended||)
```

| Case | Effect |
|------|--------|
| Agreeing proposition | Preserves/extends union dims $\to$ no penalty |
| Unknown proposition (zero dims) | Passes through $\to$ no penalty |
| Contradicting proposition | Cancels conflicting dims $\to$ positive penalty |

DoT weighting is implicit: stored vectors carry DoT in magnitude, so
contradicting a high-DoT truth causes a larger norm drop.

TruthLoss is **additive** and coexists with the **multiplicative**
modulation applied by `SymbolSubSpace.truth_modulated_loss`, which carries
both the luminosity and the universality term:
$\mathrm{totalLoss} \cdot (1 + w_{lum}(1 - \mathrm{lum}) + w_{univ}(1 - u))$.

## Bidirectional Reasoning Loop

`BasicModel.reason(givens, target, direction, max_steps)`:

- **Forward** (givens $\to$ conclusion): Encode givens into TruthSet, extrapolate
  new truths each step, check `isTrue(target)` until DoT exceeds threshold
  or `max_steps` is reached.
- **Reverse** (target $\to$ grounding): Encode target, call `ground()` to find
  minimal basis, extrapolate if insufficient.

Luminosity non-decrease is the validity certificate.

## Selected meaning and the normal controller

A completed `AnswerProgram` owns its word leaves, native identities, WORD-row
provenance and grammatical actions. `LanguageSpace.program_meaning` recovers
its complete `ConceptualMeaning` without executing a thought or writing memory.
Nested descriptions preserve their child role triples; the existing LTM owner
binds local constituent references when the completed observation is recorded.
A containing claim cannot certify its embedded claim or question.

`BasicModel.run_selected_thought` is the only normal thought controller. It
chooses from the model's `<thought>` catalogue, executes checked native VPs,
and records ordinary thoughts in `SymbolSpace.what_memory`. `what(Q)` descends
into a child context in that same controller. Children can make repeated
choices, and their typed returned evidence is a causal source of the parent's
conclusion. One shared meter pays for choices, native payloads, evidence reads,
traversal and children. Cutoff permits only the bounded return drain and finish.

`reason_about`, `think_about` and `answer_query` enter `_query_boundary_scope`.
The first two accept completed meanings or explicit typed `QuerySpec` requests.
`answer_query` also accepts a captured `Understanding`, or understands text
once before testing its selected meaning. Natural words do not dispatch a
reader. Serving reuses that one understanding and summarizes its actual trace.
`think()` is a single presentation wrapper around the same answer boundary.

The old frame controller, addressee/testimony system, next-op head and
recurrent neural-tool facade are deleted. There is no facade or
second What parity selector on the answer path. `bin/thinking.py` retains only
the numerical `TruthInterval` value and status names. Old parity/next-op policy
weights are discarded on checkpoint migration, never relabelled as the new
policy's logits.

Truth, prediction, set, code and subgoal results have typed answer adapters.
Sets retain every checked member; codes retain their checked atom and reference;
subgoals unwrap the typed child. `resolveAnswer` prepares these owned values,
and `reverseOutput` realizes them without executing another reader. Missing
payloads remain unavailable rather than becoming an invented answer.

## Learning and credit

Natural operator associations are learned from owned word payloads. No natural
word, including “has,” is anchored to `part` or `whole`. The technical
`partOf` / `wholeOf` / `isEqual` corpus tokens remain explicit grammar provenance.
The optional `LanguageSpace.meaning_codec` learns operation, mode and canonical
operand alignment plus conditional word generation. Its targets are read only
by the loss after output; model optimizer adoption and checkpoint restoration
include these language parameters. A captured hard meaning is stable across
later parameter updates. See [selected linguistic meaning](SelectedMeaning.md)
for the supervision API, held-out learning evidence and current bounds.

The normal `SelectedThoughtChooser` sees full, separately masked root, active
and candidate role triples plus level, pressure and bounded evidence. Its input
width is `9D + 17`, including two action-kind flags. Root and active payloads
remain live within the optimizer episode; hard candidates are detached. Native
IDs and word spellings never supply numerical features. Capacity uses
`whatThinkingHidden` and `whatThinkingDepth`; changing it is not evidence of
learned reasoning.

There is one hard thought-choice credit objective:
`selectedThoughtPolicyWeight` multiplies REINFORCE credit from the later
supplied-answer loss and `0.01 * actual_shared_work`, with one EMA baseline.
`thinkingLossWeight` and `whatThinkingPolicyWeight` are migration aliases;
the maximum of the three values enables this same objective once. Checked
reader results and rewards remain detached; retained ordinary values detach at
the optimizer boundary. The detailed contract is in [GradientFlow](GradientFlow.md).

The independent `policy_answer_loss` bridge experiment remains available for
reviewed callers, with `answerLossWeight` defaulting to zero. It preserves
its soft query loss over detached candidates and performs no recurrent query
control. The separate next-idea blend, scorer and model prediction route are
removed; nonzero `predictNextLossWeight` is rejected.

The learning probe trains parthood wording on six noun pairs and then maps
“a bicycle has a wheel,” its converse and a paraphrase to the same canonical
operator and roles. Thought checks that identity and generation emits the
held-out sentence. Possessive controls remain unknown. This closes item 1's
specific learning gate; residual-based query credit, held-out multistep utility,
the separate generation-catalogue migration and throughput gates remain open.
See [Testing](Testing.md#selected-meaning-and-one-controller-september-20).

## Parser And Conceptual Order

Grammar mode is derived from the loaded grammar block. Default-only unary
`pi` / `sigma` rules take the fast path; non-default operator rules enable
grammar-directed parser dispatch.

`subsymbolicOrder` controls the number of P$\to$C$\to$S stages and the
symbol partition geometry. Higher-order symbols write to later partitions,
so truth grounding, consistency, and extrapolation can respect conceptual
order.

The parser backend is no longer selectable. Stage 3 of the substrate
refactor (2026-05-27) retired the CKY chart and STM shift-reduce parsers;
the signal router (`LanguageLayer`) is the single canonical parser. The
former `SymbolSpace.parserBackend` / `routerKind` knobs (along with
`chartTau`, `chartTopK`, `chartNoiseEps`) are RETIRED — setting any of them
in a config raises a loud `ValueError` at load time (see
`Language._assert_retired_chart_knobs_absent`).

Explicit ordered grammar is preferred:

```
S4 = lift(NP3, VP1)
S5 = lift(NP4, MP1)
```

Here all NPs share base category `NP`; the suffix gives the conceptual
order. Lift and lower are the only syntactic operations that change
argument/return order.

## Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `<TruthLoss>` | `<training>` | 0.0 | Additive truth-loss weight |
| `<subsymbolicOrder>` | `<architecture>` | 1 | Percept$\to$Concept$\to$Symbol iterations |
| `<reasoningIterations>` | `<architecture>` | 1 | Shared work allowance for explicit `reason_about` / `answer_query`; `0` disables those APIs. |
| `<queryReasoning>` | `<architecture>` | false | Deprecated alias; `true` maps to ten work units when `reasoningIterations` is unset. |
| `<parserBackend>` | `<SymbolSpace>` | — | **RETIRED** (Stage 3, 2026-05-27): the chart and STM parsers are gone; the signal router (`LanguageLayer`) is the only parser. Setting this (or `routerKind` / `chartTau` / `chartTopK` / `chartNoiseEps`) raises a loud `ValueError` at config load. |
| `truthCriterion` | `<architecture>` / `<ConceptualSpace>` / `<WholeSpace>` | 1.0 | Single continuous truth bar (0 $=$ all, 1 $=$ none; **default 1.0 $=$ off**, opt-in by lowering) governing BOTH WholeSpace truth **recording** (record a cell iff its clamped magnitude $\ge$ `truthCriterion`; fires in training + `store_truths` gold ingestion) AND learned relative-sentence **acceptance** (accept iff learn-score $\ge$ `truthCriterion`). Replaces the retired binary `<accumulateTruth>` / `<truthMinMagnitude>` switches. See [STM.md Section 9](STM.md#9-relative-vs-absolute-end-states). |
| `answerLossWeight` | `<training>` | 0.0 | Policy answer loss weight (NLL on the $[0,1]$ proof score; trains the soft query route, hard proof mask detached). |
| `predictNextLossWeight` | `<training>` | 0.0 | Retired; nonzero values are rejected. Thought selection uses the normal controller. |
| `intraLossWeight` | `<training>` | 0.1 | In-STM next-idea loss $\mathcal{L}_\text{intra}$ weight (`IntraSentenceLayer`). See [STM.md Section 6](STM.md#6-intrasentencelayer). |
| `interLossWeight` | `<training>` | 0.1 | Inter-sentence next-end-state loss $\mathcal{L}_\text{inter}$ weight. See [STM.md Section 11](STM.md#11-inter-sentence-prediction). |
| `routerWireSerial` | `<architecture>` | both | Per-word router-fire gating on the serial path (`per-word` / `boundary` / `both` / `off`). See [STM.md Section 7](STM.md#7-per-word-router-firing). |
| `ltmCapacity` | `<SymbolSpace>` | 1024 | LTM chain capacity (`InterSentenceLayer` deque of STM end-states). See [STM.md Section 10](STM.md#10-ltm-as-the-chain-of-stm-end-states). |

The relative-vs-absolute end-state machinery, the content-aware
learn-score gate, and the tetralemma trust 4-tuple carried on accepted
relative META edges are documented in
[STM.md Section 9](STM.md#9-relative-vs-absolute-end-states).

## Contemplative Awareness Methods

Four methods on `BaseModel` characterizing stages of contemplative awareness
as spatial/computational properties. `Contiguous()`, `Continuous()`, and
`Peaceful()` are implemented (each returns a measure in `[-1, +1]`; see
`bin/Mereology.py`). `Peaceful()` reads the TruthLayer and returns
`valence-symmetry × luminosity-uniformity` (balanced affirming/denying pole
masses × uniformly-held per-proposition magnitude; `0.0` when no truths are
stored). `Done()` remains a stub that raises `NotImplementedError` --- it
defines the target characterization, not the implementation.

| Method | Stage | Characterization |
|--------|-------|------------------|
| `Contiguous()` | One-Pointedness (Shamatha / FA) | Single connected, convex region in PartSpace; contiguous span in WholeSpace |
| `Continuous()` | Simplicity (Continuity / OA) | Concept states flow continuously; Jacobian of forward map is bounded |
| `Peaceful()` | One Taste (Emotional Symmetry) | TruthLayer luminosity uniformly high across stored propositions |
| `Done()` | Buddhahood (Non-Meditation) | Model is a fixed point of forward-reverse; reconstruction loss zero |

Shamatha Speech is the target grammar mode for `Contiguous()`: complete DNF
object grammar plus spatiotemporal contiguity. Every `conjunction` /
`disjunction` over object parts must keep `where()` support connected and
`when()` support continuous. Differs from serial mode --- may reduce over all
active percepts at once; rejects scattered object fields, not multi-percept
fields. See
[Philosophy.md](Philosophy.md#shamatha-speech-and-single-pointedness).

## Testing

Unit tests in `basicmodel/test/test_reasoning.py` cover all methods without
requiring a trained model. English-level tests (syllogisms, contrapositives,
semantic equivalence) are `@pytest.mark.xfail` until word identity is
learned through training.

[Kernel test migration](KernelRetirement.md) maps every retired test and
defines the replacement for conflicting, mixed and bounded-unknown statuses.
