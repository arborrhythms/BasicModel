# Short-Term Memory

> **2026-06-02 update (subsymbolic analyzer).** Operators no longer enter
> the STM idea space. They are kept in the SS **codebook**
> (`WholeSpace.insert_operations`, wired into `SymbolicSubSpace.__init__`)
> and resolved as a soft superposition over the operator-prefixed parse
> tree; the STM idea slots hold only **combined meanings** -- an operator
> defines *how* meanings combine, contributing none of its own. See
> [doc/old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md](old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md).

> **Status (2026-05-30):** new chapter for the STM serial / parallel
> modes work
> ([doc/old/2026-05-29-stm-serial-parallel-modes.md](old/2026-05-29-stm-serial-parallel-modes.md)).
> Documents the per-batch STM buffer, the predict-then-perceive cadence
> (serial and parallel), the attentional-filtering regime (serial runs
> **with** attention by design — the old serial-vs-attention guard was
> lifted), the routing-parser SS-analysis / CS-execution split, the
> in-STM `IntraSentenceLayer` AR predictor, masked-word reconstruction
> via priming, the relative-vs-absolute end-state preservation with its
> content-aware learn-score gate, and the LTM chain of end-states feeding
> inter-sentence prediction. Where a piece is a deliberate scaffold or a
> deferred wiring, this chapter says so.

## Overview

ConceptualSpace is, post-substrate-refactor, an STM container plus a
grammatical CPU — it owns no atomic forward fold (see
[Spaces.md](Spaces.md#shorttermmemory)). The **short-term memory (STM)**
is the structure CS manages: a per-batch stack of unquantized CS
"ideas" that the model accumulates across a sentence and reduces at the
sentence boundary. This chapter is the single reference for what the STM
is, how it fills, what reads it, and how its end-states chain into
long-term memory.

Two cadences write the STM, selected by `<symbolicOrder>` (see
[Architecture.md](Architecture.md#modes-of-operation)):

```
SERIAL    one idea per word; predict the free slot, then perceive (push)
PARALLEL  one whole-slab pass; predict the slab, then perceive (set all slots)
```

Both follow a **predict-then-perceive** discipline: before the freshly
materialized event is written into the STM, the in-STM predictor reads
the retained context and stakes a prediction; the just-written event is
then its supervision target. This is the concept-space analogue of a
language model's next-token objective, scaled to the in-sentence working
set.

The trainable target configuration is `MentalModel.xml` (serial,
`<data><dataType>embedding</dataType>`, attention on, sentence prediction
on, FineWeb data); the comparison framing against a transformer LM is
[Section 12](#12-nanochat-comparison-framing).

---

## 1. STM data model

`ShortTermMemory` ([Layers.py:8982](../bin/Layers.py)) is a `Layer`
(not a `Space`): it carries no SubSpace and no forward / reverse
tensor-map contract. ConceptualSpace builds one at construction as
`self.stm` and treats it as the primary structure it manages.

**Capacity.** Default `8`, set via
`<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>`
(`DEFAULT_CAPACITY = 8`); the legacy `<wMax>` alias is retired. Eight sits
inside Miller's $7 \pm 2$ band — the working-set size
psycholinguistics ascribes to human short-term memory. The capacity is
the rolling-window length: at steady state the STM holds the last `cap`
ideas and the oldest falls off as new ones arrive.

**Buffer.** The data is a single per-batch tensor
`[B, cap, concept_dim]` plus a `[B]` long depth-pointer vector recording
how many slots each row has filled (saturating at `cap`). The buffer is
a plain registered buffer with `persistent=False`: STM contents are
runtime working state, not learned weights, and never enter the
checkpoint. `concept_dim` is the CS feature width $D$ — the `what`
columns only; the positional / temporal (`where` / `when`) columns are
trimmed at the PS$\to$CS boundary so the slab the grammar dispatch runs
over is a clean concept-dim block (`_stm_payload_dim = concept_dim`).

**SymbolicSubSpace as the data carrier.** After the 2026-05-21 STM-Layer
refactor the `ShortTermMemory` object is itself data-free: its `_buffer`
/ `_depth` properties proxy to the owning `SymbolicSubSpace`'s `_idea_*`
buffers via `attach_word_subspace`. A standalone fallback buffer exists
for bare-construction unit tests, but in the live model the bytes live on
SymbolicSubSpace and `ShortTermMemory` is a thin accessor over them.

**API.** `push(b, idea)`, `pop(b)`, `peek(b, n=0)`
(`peek(b, 0)` = most recent), `snapshot(detach=False)`, `size(b)`,
`is_full(b)`, `is_empty(b)`, `clear(b=None)`, `ensure_batch(batch)`. The
signal router consumes `stm.snapshot()` as its slab input.

**Lifecycle.** Cleared on hard `Reset` (sentence boundary) — the
per-batch idea stack drops everything from the just-finished sentence and
the next sentence starts empty. Soft reset leaves the STM intact (see
[Spaces.md](Spaces.md#reset-cascade--hard-vs-soft)).

---

## 2. Serial sequencing

In SERIAL / GRAMMATICAL mode each word traverses a per-word path: MPHF
surface lookup $\to$ `PartSpace.forward` ($\pi(x) + \sigma(x)$, no
outer $\tanh$) $\to$ `ConceptualSpace.forward`, which does the STM
bookkeeping. The whole pass is **predict-then-perceive per word**,
implemented in `ConceptualSpace.forward`
([Spaces.py:14134](../bin/Spaces.py)):

1. **Snapshot + predict the free (newest) slot.** Before writing the new
   event, `_stm_predict_then_perceive_serial(idea)`
   ([Spaces.py:12290](../bin/Spaces.py)) takes `stm.snapshot()` and runs
   the in-STM predictor from the **retained context** — the snapshot with
   the oldest slot dropped (`snap[:, :-1]` when depth $\ge 2$; the whole
   snapshot when depth $= 1$). The prediction is stashed on
   `self._stm_predicted_idea`. On the first word (empty STM) the
   prediction degenerates to zeros and **no** loss accumulates (there is
   no prior context to predict from). The serial predictor folds the
   context with an order-invariant **sum** over the slot axis, so dropping
   the oldest (regardless of which physical slot holds it) yields the same
   prediction.
2. **Perceive (overwrite via `_stm_shift_and_push`).**
   `_stm_shift_and_push(idea)` ([Spaces.py:12056](../bin/Spaces.py)) is
   the perceive step. At capacity, slots $0\ldots(\text{cap}-2)$ shift
   into $1\ldots(\text{cap}-1)$ and the new idea lands in slot $0$ (the
   newest slot); the oldest idea (the last slot $\text{cap}-1$) falls off.
   Below capacity it is a plain push: shift the occupants right and write
   slot $0$.
3. **Accumulate the intra-loss.** The held prediction is scored against
   the just-perceived `idea` as $\mathcal{L}_\text{intra} =
   \mathrm{MSE}(\hat{c}_t, c_t)$ (see [Section 6](#6-intrasentencelayer)).
4. **Language dispatch.** The signal router dispatches grammar ops over
   the STM contents (read-only via CS, write-required via SS).

> **Convention pin — newest-at-slot-0, shift-RIGHT.** The **free / newest**
> slot is **slot $0$** and the rolling window shifts **right**, dropping
> the oldest (the last occupied slot, $\text{cap}-1$ at capacity).
> `peek(n)` counts $n$ back from the newest, so it reads slot $n$
> directly; `snapshot()` returns the live slab **newest-first**. The
> "predict the free slot" step predicts slot $0$, and the retained
> context that conditions it is `snap[:, :-1]` (everything but the
> soon-to-be-evicted oldest slot). A named primitive
> `_stm_shift_for_predict` ([Spaces.py:12121](../bin/Spaces.py)) exists
> for the literal "rotate so the free slot is free" step, but the hot
> serial path deliberately does **not** call it — `_stm_shift_and_push`
> already does shift-right-then-write in one pass, and a separate physical
> shift would double-shift the buffer. The helper is kept off the forward
> path and exists for independent unit-testing of the named step.
>
> *(History: this convention was flipped from the original newest-at-top /
> shift-LEFT layout; the flip preserves every semantic — `peek` still
> returns the most-recent idea, the reduce still collapses to the same
> root, and the relative end-state still yields the same
> predicate/idea1/idea2 — only the physical slot order changed.)*

---

## 3. Parallel sequencing

In PARALLEL mode the per-stage forward sees the whole sentence at once:
the slab `[B, N, D]` **is** the STM, each of the $N$ positions its own
slot. The same predict-then-perceive discipline applies whole-slab,
implemented in `ConceptualSpace.forward` via the `folded.shape[1] > 1`
branch:

1. **Predict $\hat{C}$ from the previous slab.**
   `_stm_predict_then_perceive_parallel(folded)`
   ([Spaces.py:12325](../bin/Spaces.py)) snapshots the previous STM and
   runs the predictor **per slot** (no cross-slot collapse), producing a
   `[B, prev_N, D]` slab stashed on `self._stm_predicted_slab`.
2. **Perceive via `_stm_set_all_slots`.**
   `_stm_set_all_slots(slab)` ([Spaces.py:11986](../bin/Spaces.py))
   writes all $N$ positions directly as the slot stack (no shift, no
   mean-reduction). The slab is position-ordered (position $0$ oldest,
   $N-1$ newest); under the newest-at-slot-0 convention it is **flipped**
   along the position axis so the slab's newest lands at slot $0$.
   Capacity-clip: if $N >$ cap it keeps the last `cap` positions (drop
   oldest, mirroring the serial rolling window) then flips; if $N <$ cap
   it zeroes the unfilled (older) tail so a prior pass's state cannot leak
   through. The matching `_stm_predict_then_perceive_parallel` flips the
   `folded` target the same way before scoring $\mathcal{L}_\text{intra}$,
   so the per-slot prediction (computed over the newest-first previous
   STM) and its target stay slot-aligned — byte-identical to the old
   oldest-first alignment.
3. **Loss only when shapes align.** $\mathcal{L}_\text{intra}$
   accumulates only when the predicted slab shape matches `folded`
   exactly. On the first pass (empty previous STM) the prediction is
   zeros and loss is skipped; when the previous slot count differs from
   $N$ the per-slot prediction does not align with the target and the
   pinned decision is to skip loss for that pass rather than fabricate an
   overlap. Loss resumes once the STM steady-state slot count matches the
   slab width.

The earlier "mean-reduce to a single idea, then shift-push" pattern was a
bug in parallel mode — it destroyed per-slot identity and made the
reverse pipeline emit the same recon vector in every slot. The
slot-preserving `_stm_set_all_slots` is the fix.

---

## 4. Attentional filtering

**Serial mode IS the attentional-filtering regime.** The old
serial-vs-attention guard was **lifted**: `MentalModel.xml` runs serial
*with* attention by design (`<symbolicOrder>1</symbolicOrder>`
with `<attention>primer</attention>` on ConceptualSpace; the legacy
`<hasAttention>` boolean is deprecated and inert, superseded by the
`<attention>` element — off/primer/second-order/low-rank). Serial
sequencing and attentional filtering are the same regime, not mutually
exclusive options.

**CS$\to$PS meronymic Gaussian mask.** On the per-word serial path the
input to each word is not a raw slice but a Gaussian-windowed contextual
trace, built by `gaussian_window_word(full_seq, center_k)`
([Models.py:2323](../bin/Models.py)). For the processed word at center
$k$ a single envelope over the whole per-sentence percept sequence
$[B, T, D]$ is formed:

$$
w_i = \exp\!\left(-\frac{(i - k)^2}{2\sigma^2}\right), \qquad
\sigma = \text{maskRate} \cdot T,
$$

normalized so the center weight $w_k \approx 1$. The peak sits at the
current word; words far from $k$ are attenuated toward $0$ by the
Gaussian tail. The windowed percepts are mapped into conceptual space and
**summed**, so word $k$'s representation carries a faint meronymic trace
of its neighbours (the local context *is* part of the embedding signal).
This **replaces** the prior BERT-style hide-a-token `create_ir_mask` on
the per-word grammar path; it is not target-hiding — the center word is
preserved. The centring is **hardcoded**: peak at the current word.

**CS$\to$SS taxonymic mask.** On the symbolic side the inverse
recommender zeros **non-admissible** codebook rows before a lift / lower
(union / intersection) reduction so an operand can only resolve to a row
of the right grammar category and conceptual order. The admissible row
set is built by `priming_kwargs_for_slots`
([Language.py:7340](../bin/Language.py)) — intersecting `refs_by_category`
with `refs_by_order` per slot — and applied in the recommender's
`_row_mask` ([Layers.py:10344](../bin/Layers.py)), which admits only the
category/order-matched rows plus the $\bot$ / $\top$ sentinels. A parallel
`_row_weights` path multiplies a taxonymic **priming** mask over the
admitted rows (the `left_priming` / `right_priming` weights from
`taxonomy.priming_mask`).

**Guidance-signal contract.** The mask is a *guidance signal*: it biases
which codebook rows / sequence positions the reduction may consult, given
the current word and grammar state. The centring policy (CS$\to$PS) is the
swappable seam — the present text-reading variant centres on the current
word; a future image-reading variant would swap that policy (centre on a
fixated region) without changing the rest of the pipeline.

**Out of scope.** Learning the mask is out of scope for this work — both
masks are **hardcoded** (Gaussian centring fixed at the current word;
taxonymic admissibility read structurally from the codebook's category /
order metadata). A learned attention policy is a documented future
direction, not built here.

---

## 5. Routing parser: SS-analysis vs CS-execution

The grammar runs through the signal router (`LanguageLayer`,
[Language.py:3493](../bin/Language.py)); `SymbolicSubSpace` owns it as
`self.languageLayer`. Conceptually the work splits in two:

- **SS-analysis** — `SymbolicSubSpace.compose`
  ([Language.py:7693](../bin/Language.py)) is the analysis stage: a soft
  superposition over the taxonymic codebook that selects, per-space, a
  **hard rule dict** `current_rules = {space-role: [rule_id, ...]}`. It chooses
  *which* reductions fire.
- **CS-execution** — actually applying the chosen reductions (lift,
  lower, union, intersection, swap, quantize, not) to the concept tensors
  runs CS-side in `ConceptualSpace.forward` and the WholeSpace
  stack-route path, with the per-space `SyntacticLayer` cursors
  ([Language.py:5453](../bin/Language.py)) executing the unary $\pi$ /
  $\sigma$ folds on reverse. Only lift / lower / union / intersection
  consult the codebook (inverse-recommended); swap / quantize / not are
  tensor-only.

> **Honesty — the split is a clean code boundary only on the
> default-only path.** When the grammar is *default-only* (every rule is
> the unary $\pi$ / $\sigma$ fold), `compose` emits `current_rules` from
> the grammar XML and runs **no** tensor reduction, and the per-space
> `SyntacticLayer.forward` / `reverse` cursors do the CS-side execution —
> a genuinely clean analysis/execution separation. On the **full-router**
> path, however, `LanguageLayer.compose` does **both**: it selects the
> rules *and* folds the slab tensorially through the op modules
> (`BinaryStructuredReductionLayer.forward` $\to$ `op(left, right)`),
> caching the root state. The per-space `SyntacticLayer` cursors are then
> **deliberately bypassed** on that path — guarded by
> `not _grammar_is_default_only`
> ([Language.py:5731](../bin/Language.py)) — precisely so the reduction is
> not double-applied. So in the full-router case the SS-analysis and
> CS-execution stages are **co-located** inside `LanguageLayer.compose`
> rather than separated across modules. The audit found this; it is a
> documented task-5 follow-up, not a finished boundary.

`_grammar_is_default_only` is computed from the configured grammar at
`SymbolicSubSpace.__init__` ([Language.py:6522](../bin/Language.py)) and gates
both `compose` and `generate`.

---

## 6. IntraSentenceLayer

`IntraSentenceLayer` ([Layers.py:6632](../bin/Layers.py)) is the in-STM
autoregressive predictor — the layer that stakes the prediction in
predict-then-perceive. It is owned by ConceptualSpace (`self.intraSentenceLayer`).

**Architecture: combined PI-then-Sigma, no intermediate $\tanh$.**

- `self.pi`: `PiLayer(concept_dim $\to$ working_dim)`, `invertible=True`,
  `nonlinear=True` — the log-domain multiplicative boundary fold lifts
  each STM slot into the working width and bounds it to $[-1, 1]$.
- `self.sigma`: `SigmaLayer(working_dim $\to$ concept_dim)`,
  `invertible=True`, **`nonlinear=False`** — a raw linear $W x + b$ that
  collapses the lifted slots into the predicted idea.

The defining requirement is that there is **no extra activation
interposed** between the PI body and the Sigma body: `sigma` is built
`nonlinear=False` (a raw $W x + b$), so no additional $\tanh$ sits
between PI's output and Sigma's input. This is **not** a linear fusion —
`pi` is `nonlinear=True` (its symmetric log-domain $(1+x)/(1-x)$
embedding is an intrinsic nonlinearity), so the composite is a nonlinear
PI followed by a raw-linear Sigma, not two fusable linear cores.
`working_dim` defaults to `concept_dim`, keeping both sublayers square
isomorphisms so the parallel per-slot round-trip
$\text{reverse}(\text{forward}(x)) \approx x$ is exact up to the LDU
inverse tolerance.

**Forward signature.** `forward(prior_slots, routing=None, parallel=False)`
with `prior_slots: [B, K, D]`:

- **Serial collapse** (`parallel=False`, the primary regime): PI-lift
  every slot, **sum-fold over the slot axis**, then Sigma-collapse to one
  idea $\to$ `[B, D]`.
- **Parallel per-slot** (`parallel=True`): PI$\to$Sigma per slot, no
  cross-slot mixing $\to$ `[B, N, D]`.

The serial collapse is many-to-one (sum over $K$ slots), so its
`reverse` is necessarily approximate (it divides the recovered fold
equally across the $k = \text{stm\_capacity} - 1$ slots); the parallel
per-slot path is width-preserving and exactly invertible.

**Loss.** $\mathcal{L}_\text{intra} = \mathrm{MSE}(\hat{c}_t, c_t)$ via
`intra_loss(pred, target)`. ConceptualSpace accumulates it per step
(`_accumulate_intra_loss`, [Spaces.py:12248](../bin/Spaces.py)) into a
live grad-carrying running sum and `consume_intra_loss`
([Spaces.py:12271](../bin/Spaces.py)) returns the per-step mean once per
batch (`runBatch` consumes it post-body / pre-backward, mirroring the
ARMA term). The weight is `<intraLossWeight>` (default `0.1`); the term
is gated off when grad is disabled or the weight is non-positive.

---

## 7. Per-word router firing

The `<routerWireSerial>` knob ([model.xsd:76](../data/model.xsd)) gates
when the router fires on the serial path:

| Value | Behaviour |
|---|---|
| `per-word` | fire per word; boundary fire off |
| `boundary` | fire only at the sentence boundary |
| `both` | **(default)** per-word AND boundary both fire |
| `off` | neither fires |

The **per-word fire** runs `symbolSpace.compose` over the current STM
snapshot *before* `cs.forward` for the next word, populating
`symbolSpace.current_rules` for the SS dispatch
([Models.py:6053](../bin/Models.py)). It lives in the host-side loop,
outside the per-iteration captured graph, so even a full `languageLayer`
path that introduces a host sync cannot break the per-word capture gate.
The **boundary fire** ([Layers.py / Models.py boundary hook]) runs iff
`router_wire_serial in ('boundary', 'both')`.

> **Honesty — routing conditioning of the predictor is DEFERRED.**
> `current_rules` is a **host-side rule dict** `{'S': [rule_id, ...]}`,
> **not** a `[B, concept_dim]` tensor. `IntraSentenceLayer.forward`
> accepts a `routing` tensor and projects it into a Sigma bias, but
> ConceptualSpace's `_intra_routing_for_predict`
> ([Spaces.py:12166](../bin/Spaces.py)) only forwards `routing` when it is
> a CS-width tensor; given a dict it returns `None`. So the per-word fire
> wires the SS dispatch context but does **not** yet flow a routing
> tensor into the intra predictor — the predictor runs **without** the
> routing bias for now. Rule-aware predictor conditioning awaits a future
> rule-dict $\to$ CS-width projection (deliberately not built here:
> rule-distribution width $\ne$ concept_dim, and a naive bridge risks
> capture-gate breakage). This is the current state, not a finished
> feature.

---

## 8. Masked-word reconstruction via priming

Masked words enter the pipeline as **all-zeros** percept slots. On
reverse, the router fills the blank via a best-fit codebook walk biased
by the taxonymic prior, POS selection, and accumulated prediction — it
reconstructs the most plausible word for the slot given the grammar
context and the codebook geometry.

**Priming machinery.** The bias enters through `left_priming` /
`right_priming` in `Basis.lift` / `Basis.lower`
([Spaces.py:1202](../bin/Spaces.py)), forwarded down to the inverse
recommender's `_row_weights` ([Layers.py:10364](../bin/Layers.py)) where
each admitted codebook row is scaled by its taxonymic priming weight (the
$\bot$ / $\top$ sentinels pinned to $1.0$). Primed rows are preferred in
the argmax that selects the operand. `test/test_primed_reverse_hard_mask.py`
exercises this path.

**Tests.** The reverse roundtrip and reconstruction tests are
`test/test_stm_reverse_roundtrip_lift_lower.py`,
`test/test_stm_reverse_roundtrip_union_intersection.py`, and
`test/test_stm_recon_from_cleared_cache.py`.

> **Honesty — the recon-from-cleared-cache test is an honest `xfail`.**
> `test/test_stm_recon_from_cleared_cache.py` marks the top-$k$ word
> recovery assertion `xfail` (not relaxed to a trivially-true bound)
> because of **two pre-existing upstream bugs**, both out of scope here
> and flagged as separate follow-ups:
> - **Finding A** — on the untrained config the per-word forward fills
>   the CS STM with NaN: `conceptualSpace.stm.snapshot()` is
>   non-finite, so the reduced single-$S$ seed is already NaN before
>   reverse runs.
> - **Finding B** — even with a *finite* seed, the reverse perceptual leg
>   (`PartSpace.reverse`) turns it NaN.
>
> The non-`xfail` assertions in that file pin everything that *does* hold
> today (the per-op reverses reconstruct `[B, N, D_c]`; the decode uses
> the real perceptual codebook, no reimplementation). When the two
> upstream bugs are fixed the `xfail` will xpass. Documenting the
> contract honestly: the priming-biased recon path is built and tested,
> but end-to-end word recovery on an untrained model is blocked upstream.

---

## 9. Relative vs absolute end-states

At the sentence boundary the STM is reduced to its end-state. The shape
of that end-state depends on whether the sentence asserts an **absolute**
or a **relative** truth.

- **Absolute** sentences reduce to a **single idea** (depth 1) — the root
  $S$ the start-symbol reduction wrote. This is the dominant path and the
  one the IR loss consumes.
- **Relative** sentences (the `part` / `isEqual` predicate family, which
  produce the `REL_T` relative start symbol) are **preserved at depth 3**
  as the `[predicate, idea1, idea2]` end-state. Under the newest-at-slot-0
  convention these are stored newest-first, so the predicate (oldest
  constituent) sits at the **last** slot ($\text{depth}-1$), idea1 at
  $\text{depth}-2$, and idea2 (the folded-newest-rest) at slot $0$.
  `learn_relations_from_stm` reads them from those slots so
  `_maybe_learn_relation` receives the identical predicate / idea1 / idea2
  it did under the old oldest-first (`slots 0/1/2`) layout.

The mechanism is a per-row `protect_depth` gate threaded into
`_stm_reduce_to_single_S` ([Models.py:5470](../bin/Models.py)). Each
masked micro-step (`_stm_bounded_reduce_step`,
[Models.py:5272](../bin/Models.py)) folds only while
$\text{depth} > \text{protect\_depth}$: absolute rows pass
$\text{protect\_depth} = 1$ (collapse all the way), relative rows pass
$\text{protect\_depth} = 3$ (stop at the depth-3 end-state).

**Conservative detection.** `_sentence_relative_mask`
([Models.py:5387](../bin/Models.py)) decides per row from the grammar —
it scans `current_rules`' $S$-space-role for a `TheGrammar.is_relative_rule`
rule_id (`REL_T` LHS, or an `isEqual` / `queryPart` / `assertPart` /
`part` op). It **defaults to collapse on any uncertainty**: a false
positive would stop an absolute sentence's collapse and break the
dominant path and the IR loss, so missing / empty rules, a grammar with
no relative rule, or any ambiguous shape all return all-False. The
absolute path stays **byte-identical** in every fall-through case.

**Learn-score acceptance gate.** Concept-codebook insertion of a learned
relation is gated by a content-aware **learn-score**
(`_compute_learn_score`, [Spaces.py:13312](../bin/Spaces.py)):

> **Terminology (2026-06-21 convention).** The CS part↔whole relation
> table is the **Concept codebook** — each entry is a *concept* tying one
> part-percept to one whole-percept by reference. Earlier text called this
> the "symbol table"; the de-overloaded name is *concept* (a *symbol* is
> the 0-D SymbolSpace reference *to* a concept, not the relation itself).
> The taxonymic / perceptual codebooks consulted for lift/lower operands
> (Sections 4, 8) are distinct and keep their names.

$$
\text{learn\_score} = \text{children\_in\_codebook} \times
\text{is\_truth\_obvious} \times \text{resolves\_contradiction},
$$

each factor in $[0, 1]$. A relation is accepted **iff**
$\text{learn\_score} \ge$ `<truthCriterion>` **and** $\text{truthCriterion} < 1$.
The **default is `1.0`** — truth-learning is OFF by default (nothing
learned or recorded); opt in by lowering `truthCriterion` toward $0$.
Because the factors multiply, a **low `is_truth_obvious` does not on its
own block**: lies and uncertain relations can still be learned if the
other two factors are high. At `truthCriterion = 1` nothing is learned;
at `0` everything is.

Accepted insertions carry a **tetralemma trust 4-tuple** $(t, f, b, n)$
(TRUE / FALSE / BOTH / NEITHER, summing to $1$) computed by
`_tetralemma_trust` ([Spaces.py:13343](../bin/Spaces.py)) from the
TruthSet posture via `assess()`, and bound onto the relation's META node
by `insert_relation` ([Spaces.py:16033](../bin/Spaces.py)) with the
predicate as the parent and the two ideas as its taxonomy children.

> **Honesty — the learn-score factors read the GLOBAL truth layer.**
> The `is_truth_obvious` and `resolves_contradiction` factors currently
> read the **global** `truth_layer.assess()` — `support` and `conflict`
> respectively (`_learn_score_is_truth_obvious` /
> `_learn_score_resolves_contradiction`,
> [Spaces.py:13279](../bin/Spaces.py)) — behind a swappable seam (each
> factor is an independently overridable method, the plan's required test
> seam). A **per-relation projection** is the documented refinement; the
> global read is a first cut, not the final formula.

Truth **recording** into the TruthLayer is governed by the *same*
continuous `<truthCriterion>` bar (a per-cell activation is recorded when
its clamped magnitude clears `truthCriterion`), so a single knob governs
both recording and learned-relation acceptance — there is no separate gold
path and no binary switch. The user-provided `<truth>` gold set is ingested
by `store_truths`, which drops `truthCriterion` to `0` for the ingestion
epoch (capturing every provided gold truth), then restores it; the same
recording block fires during normal training per the configured
`truthCriterion`. The binary `accumulateTruth` / `truthMinMagnitude`
switches are **retired**. See [Params.md](Params.md) and
[Logic.md](Logic.md).

> **Behavioural note.** Unlike the retired binary gate (which kept
> recording off during training), truths are now accumulated continuously
> during training per `truthCriterion`; raise it toward `1` to suppress
> recording.

---

## 10. LTM as the chain of STM end-states

Long-term memory (LTM) is the **full chain of STM end-states**. It lives
on `InterSentenceLayer` ([Layers.py:5744](../bin/Layers.py)) as a per-row
bounded `collections.deque` `_stm_end_states` of size `<ltmCapacity>`
(default `1024`). Each entry is a time-ordered tuple

```
(depth: int, payload: [depth, D] tensor, tetralemma: tuple | None)
```

where `depth` is $1$ for an absolute end-state and $3$ for a relative
`[predicate, idea1, idea2]` end-state (so the payloads are ragged — a
fixed register-buffer tensor does not fit, hence the per-row deque). The
chain is transient host-side state (`persistent=False` semantics, not in
`state_dict`).

`observe_stm_end_state(depths, payloads, tetralemmas=None)`
([Layers.py:6162](../bin/Layers.py)) records **every** sentence's
end-state — it is **not** gated by `truthCriterion`. The distinction is
load-bearing: `truthCriterion` gates only the separate Concept-codebook
insertion of *learned relations* (Section 9), whereas LTM is the AR sequence the
inter-sentence predictor consumes and must see the full history. A
non-finite payload **raises** (fail-loud on numerical divergence); the
deque `maxlen` evicts the oldest entry once full.
`get_stm_chain(n=None, b=0)` ([Layers.py:6262](../bin/Layers.py)) returns
the last $n$ (or all) end-states for a row, oldest-first.

LTM is the **full chain**; the **TruthSet** is the accepted-belief subset
(the relations that cleared the learn-score gate and were inserted into
the Concept codebook). LTM keeps everything that happened; the TruthSet
keeps what was believed.

---

## 11. Inter-sentence prediction

A **lifted `IntraSentenceLayer` instance** (`_inter_predictor`,
[Layers.py:5946](../bin/Layers.py)) predicts the next end-state over the
LTM chain — the same predictor class as the in-STM one, instantiated at
the inter-sentence level. Its chain window is
$K = \min(\text{ltmCapacity}, 8)$ (`_inter_chain_window`): the AR signal
that predicts the next end-state lives in the last handful of sentences,
so a small bounded window is used rather than the full `ltmCapacity`.

`predict_next_end_state(b=0)` ([Layers.py:6326](../bin/Layers.py))
produces the next end-state **shape** $(\hat{d}, \hat{p}[\hat{d}, D])$:

- **Chain reduction (ragged $\to$ fixed).** Each chain entry is reduced to
  its **root** by `_reduce_end_state_to_root`
  ([Layers.py:6316](../bin/Layers.py)). Under the newest-at-slot-0
  convention the end-state is stored newest-first, so the root lives at the
  **last** slot ($\text{depth}-1$): for an absolute end-state (depth 1)
  that is slot $0$ (the collapsed idea); for a relative end-state (depth 3)
  it is slot $2$, the predicate (the head the relative structure hangs
  off). The last $K$ roots form a `[1, K, D]` context, left-padded with
  zeros so the most recent sits at the tail (newest-at-$-1$, like the ARMA
  ring).
- **Root prediction.** `_inter_predictor.forward(context, routing=None,
  parallel=False)` $\to$ `[1, D]`, the predicted root.
- **Loss.** $\mathcal{L}_\text{inter} = \mathrm{MSE}(\hat{p}, p)$ on the
  roots, accumulated by `_accumulate_inter_loss`
  ([Layers.py:6413](../bin/Layers.py)) and drained by
  `consume_inter_loss`, weight `<interLossWeight>` (default `0.1`). The
  actual root is detached so the loss trains `_inter_predictor`, not the
  perception path. `observe_stm_end_state` scores the prediction made for
  a row against the end-state that actually arrived.

The prediction is consumed by `generate_sentence` via the `_c_prior`
`[depth, D]` staging path: a predicted next-end-state shape is staged
across the first `depth` STM slots as a sentence-level conditioning bias
(`ConceptualSpace.forward`'s slotwise `_c_prior` branch,
[Spaces.py:11201](../bin/Spaces.py)).

> **Honesty — $\hat{d}$ is a copy-last AR prior; reduction is slot-0 only.**
> The predicted **depth** $\hat{d}$ is a simple AR prior: the depth of the
> **most recent** end-state in the chain (a relative sentence tends to be
> followed by structure of the same shape; an absolute by an absolute).
> This delivers `depth in {1, 3}` without a separate learned head, but it
> is a **scaffold** — a tiny `concept_dim $\to$ 2` argmax head is the
> documented upgrade path. Likewise the chain reduction collapses each
> end-state to its **slot-0 root only** (mean-over-depth is the obvious
> alternative; the root carries the sentence-level signal, mirroring the
> ARMA `_pool_sentence_rep`). A non-finite predicted root raises.

---

## 12. nanochat comparison framing

The trainable target is `MentalModel.xml`:
`<symbolicOrder>1</symbolicOrder>`,
`<data><dataType>embedding</dataType>`, `<attention>primer</attention>`
(ConceptualSpace; the legacy `<hasAttention>` boolean is deprecated and
inert), `<sentencePrediction>true</sentencePrediction>`,
FineWeb data (`<shardDir>data/fineweb</shardDir>`). This is the
configuration that exercises the full STM stack — serial sequencing with
attentional filtering, the in-STM and inter-sentence predictors, and the
LTM chain.

The comparison loss is

$$
\mathcal{L} = \mathcal{L}_\text{IR} + \mathcal{L}_\text{intra}
+ \mathcal{L}_\text{inter},
$$

the masked-LM information-reconstruction term at the subsymbolic (PS) (see
[Spaces.md](Spaces.md#within-sentence-ar-retirement-2026-05-14)) plus the
in-STM next-idea term ([Section 6](#6-intrasentencelayer)) plus the
inter-sentence next-end-state term ([Section 11](#11-inter-sentence-prediction)).
This is analogous to a transformer language model's next-token
cross-entropy, but computed **in concept space** rather than over a token
vocabulary: $\mathcal{L}_\text{intra}$ predicts the next idea within a
sentence, $\mathcal{L}_\text{inter}$ the next sentence's end-state shape,
and $\mathcal{L}_\text{IR}$ reconstructs masked content — the three
together are the concept-space counterpart of the autoregressive LM
objective a system like nanochat trains.

> **Out of scope.** The comparison harness itself — running this against a
> transformer baseline and reporting the numbers — is a **separate plan**.
> This chapter documents the loss that *would* be compared and the
> configuration that produces it; it does not build the benchmark.

---

## See also

- [Spaces.md](Spaces.md) — ConceptualSpace as STM container; the
  `ShortTermMemory` API; Sigma / Pi ownership.
- [Architecture.md](Architecture.md) — modes of operation (serial /
  parallel); the per-word operational flow; `InterSentenceLayer` ARMA
  predictor.
- [Language.md](Language.md) — the signal router, `compose` / `generate`,
  GrammarLayer reductions.
- [Mereology.md](Mereology.md) — parthood as clipped-cosine projection;
  the relations the relative predicates draw on.
- [Reasoning.md](Reasoning.md), [Logic.md](Logic.md) — the truth surfaces
  and tetralemma trust.
- [Params.md](Params.md) — `<stmCapacity>`, `<intraLossWeight>`,
  `<interLossWeight>`, `<routerWireSerial>`, `<ltmCapacity>`,
  `<truthCriterion>` (the single continuous truth bar).
