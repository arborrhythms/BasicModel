# Short-Term Memory

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

STM is the local working set that replaces the anonymous residual-state view of
many LLM descriptions. It holds unquantized ideas while the parser decides which
typed DisCoCat-like reductions should apply. Those ideas are still grounded in
the Formal Concept Analysis side of the model: their slots point back to
part/whole support, concept order, and codebook rows rather than floating as
untyped context vectors.

> **2026-06-02 update (subsymbolic analyzer).** Operators no longer enter
> the STM idea space. They are kept in the SS **codebook**
> (`WholeSpace.insert_operations`, wired into `SymbolSubSpace.__init__`)
> and resolved as a soft superposition over the operator-prefixed parse
> tree; the STM idea slots hold only **combined meanings** -- an operator
> defines *how* meanings combine, contributing none of its own.

> **Status (2026-05-30):** new chapter for the STM serial / parallel
> modes work.
> Documents the per-batch STM buffer, the predict-then-perceive cadence
> (serial and parallel), the attentional-filtering regime (serial runs
> **with** attention by design — the old serial-vs-attention guard was
> lifted), the routing-parser SS-analysis / CS-execution split, the
> in-STM `IntraSentenceLayer` AR predictor, masked-word reconstruction
> via priming, the relative-vs-absolute end-state preservation with its
> grammatical clause closing, and the LTM chain of end-states feeding
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

Two cadences write the STM, selected by `<serial>` (legacy configs derive
it from `symbolicOrder > 0`; see
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
`<data><dataType>embedding</dataType>`, `<attention>` unset (resolves to
`"off"` — see [Section 4](#4-attentional-filtering)), sentence prediction
on, FineWeb data); the comparison framing against a transformer LM is
[Section 12](#12-nanochat-comparison-framing).

---

## 1. STM data model

`ShortTermMemory` ([Layers.py](../bin/Layers.py)) is a `Layer`
(not a `Space`): it carries no SubSpace and no forward / reverse
tensor-map contract. ConceptualSpace builds one at construction as
`self.stm` and treats it as the primary structure it manages.

**Capacity.** Default `8`, set via
`<ConceptualSpace><stmCapacity>N</stmCapacity></ConceptualSpace>`
(`DEFAULT_CAPACITY = 8`); the legacy `<wMax>` alias is retired. Eight sits
inside Miller's $7 \pm 2$ band — the working-set size
psycholinguistics ascribes to human short-term memory. The capacity is
the rolling-window length: at steady state the STM holds the last `cap`
ideas and the general push API drops the oldest when full. Item 7.5 compose
reserves a free slot through the online reduction deadline before the next
push. An arrival at a full stack raises an assertion instead of dropping a word.

**Buffer.** The data is a single per-batch tensor
`[B, cap, concept_dim]` plus a `[B]` long depth-pointer vector recording
how many slots each row has filled (saturating at `cap`). STM contents are
runtime working state, not learned weights. `concept_dim` is the full CS output
width reserved for the event payload; positional/temporal columns are preserved
when a config gives CS nonzero `nWhere` / `nWhen`.

**Spatial index (Alec, 2026-10-09).** The current conceptual eight-space
has its own range in the model's shared `.where` allocation.
`ConceptualSpace.where` exposes the slots' sinusoidal indices, using the
same encoding as the perceptual codebooks. A slot keeps its location when
ideas are pushed, reduced or cleared. The content's codebook address stays
separate. Closing stores the corresponding one- or three-slot semantic
content in LTM; a future episodic bank will retain located fields
([Future Work §7](FutureWork.md#7-episodic-memory-a-few-active-indices-beside-each-row)).
The seven learned output regions are superseded by
[the support-mask direction (§7.9–§7.10)](plans/2026-10-08-stream-state.md#79-alec-on-the-corrections-2026-10-09).

**Candidate support (item 6.1 candidate, 2026-10-09).**
`CandidateAttention(Layer)` scores each unread supported candidate with an
ordinary MLP. One hard choice supplies one word read and one grammar step.
The candidate's native paired evidence is its support mask, including
negative evidence; zero means no evidence. `hetTolerance` acts on these
lanes, with 1 a no-op. There is no learned region or boundary derivative.
The existing paired-cost departure learner trains the scorer.

During a sentence, `SentenceField` carries source-support indices beside
the eight STM slots: pushes move the previous placements, unary operations
retain their witnesses, and binary operations join their witnesses. The
indices refer to the sentence's native evidence bank and its original
`.where`/`.when` stamps. The scorer reads these stamps and all occupied
slots; it never places the earlier reads again. The field has its own
allowance and charges one `serial-word-loop` unit per read. The answer and
memory readers retain their landing paths; retrieval into a slot is later
work. Learning gates remain under validation in the
[6.1 receipt](benchmarks/2026-10-09-item6-1/README.md).

**`ShortTermMemory` owns its own live buffers (A5).** The idea stack is
NOT proxied off `SymbolSubSpace`. `_buffer` / `_depth` / `_max_depth_host`
are properties over plain (non-`nn.Module`-buffer) attributes
`_live_buffer` / `_live_depth` / `_live_max_depth_host`, set with
`object.__setattr__` so STM state stays out of the traced graph's inputs.
Only `capacity` and `concept_dim` still proxy: when `attach_word_subspace`
has wired an owning `SymbolSubSpace`, they read that subspace's
`_idea_capacity` / `_stm_payload_dim` (so an externally-grown idea-stack
capacity sizes the next `begin_forward` correctly); otherwise they fall
back to the constructor-supplied `_init_capacity` / `_init_concept_dim`.
This landed with the 2026-05-21 STM-Layer refactor; a later fix (the "A5"
comment on `ensure_batch`) retired the old `ss is None` branch (a second,
SymbolSubSpace-attached buffer) entirely — there is now a single live
store regardless of attachment.

**API.** Idea-stack: `push(b, idea)`, `pop(b)`, `peek(b, n=0)`
(`peek(b, 0)` = most recent), `snapshot(detach=False)`, `size(b)`,
`is_full(b)`, `is_empty(b)`, `clear(b=None)`, `ensure_batch(batch)`,
`ensure_capacity(capacity)` (grow-only). Batch push primitives (masked /
whole-slab writers used by the per-word and parallel forwards):
`push_step(ideas)`, `push_step_masked(ideas, gate_b_1)`,
`push_window_batch(ideas)`. Slot-kind provenance (word-bearing-fold
filtering; `None` = recording off, the default): `kinds_enable(batch,
depths=None, kind="other")`, `note_push_masked(gate_rows, kind)`,
`note_push_all(kind)`. STM shift/reduce scorer surface (formerly
`stm_driver.STMDriver` / `stm_trainer`, now living directly on
`ShortTermMemory`): `init_scorer(rule_signatures, payload_dim,
hidden_dim=None)`, `shift(word_subspace, b, payload, *, category, order,
ref_id)`, `reduce_step(word_subspace, b)`, `reduce_step_soft(word_subspace,
b)`, `train_scorer_step(word_subspace, input_vectors, target_rule_ids, *,
snap_fn, optimizer=None)`. The signal router consumes `stm.snapshot()` as
its slab input.

**Lifecycle.** Cleared on hard `Reset` (sentence boundary) — the
per-batch idea stack drops everything from the just-finished sentence and
the next sentence starts empty. Soft reset leaves the STM intact (see
[Spaces.md](Spaces.md#reset-cascade-hard-vs-soft)).

---

## 2. Serial sequencing

In SERIAL / GRAMMATICAL mode each word traverses a per-word path: MPHF
surface lookup $\to$ `PartSpace.forward` (synthesis front end + `self.sigma`)
$\to$ `ConceptualSpace.forward`, which does the STM
bookkeeping. The whole pass is **predict-then-perceive per word**,
implemented in `ConceptualSpace.forward`
([Spaces.py](../bin/Spaces.py)):

1. **Snapshot + predict the free (newest) slot.** Before writing the new
   event, `_stm_predict_then_perceive_serial(idea)`
   ([Spaces.py](../bin/Spaces.py)) takes `stm.snapshot()` and runs
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
   `_stm_shift_and_push(idea)` ([Spaces.py](../bin/Spaces.py)) is
   the perceive step. At capacity, slots $0\ldots(\text{cap}-2)$ shift
   into $1\ldots(\text{cap}-1)$ and the new idea lands in slot $0$ (the
   newest slot); the oldest idea (the last slot $\text{cap}-1$) falls off.
   Below capacity it is a plain push: shift the occupants right and write
   slot $0$. Under `<serialObjectMeta>` (default off; doc/specs/mereological-
   order-raising.md "Serial-mode word-at-a-time loop"), `p` indexes the
   independent sentence-word slab. Inside that iteration PartSpace gathers
   **all** of word $p$'s radix/chunk constituents and applies the configured
   sigma set-fold; the batch-padded raw width may exceed the eight-wide PS
   field. The live PS and WS folds bind strictly by location; the resulting
   concept records its actual subsymbolic order and full ordered fold support.
   Its current-word location is selected for
   `stm.push_step_masked(idea_bd, commit_b_1, orders=..., grammar_orders=...)`.
   That commits once using
   the word-active column
   `inputSpace._word_last_slot_mask[:, p:p+1]`
   (`BasicModel._per_word_body_step`, [Models.py](../bin/Models.py)).  The local
   Raw staging resets for word $p+1$; it is not an STM or PS-capacity axis.
   Flag off (or a legacy input without the word-loop tensors) falls
   back to the original per-slot path.
3. **Accumulate the intra-loss.** The held prediction is scored against
   the just-perceived `idea` as $\mathcal{L}_\text{intra} =
   \mathrm{MSE}(\hat{c}_t, c_t)$ (see [Section 6](#6-intrasentencelayer)).
4. **Language dispatch.** The signal router dispatches grammar ops over
   the STM contents (read-only via CS, write-required via SS).
5. **One operation per round.** `LanguageSpace.choose_operation` presents
   the newest two occupied slots to the shared `OperationSelectionLayer`.
   Binary candidates and unary candidates at both positions compete in one
   softmax. Three fixed rounds follow each word, with early STOP; the sentence
   closing runs up to twice capacity. A binary choice removes one slot, while a
   unary choice changes its selected slot without changing occupancy. STOP is
   forbidden until the complete STM fits the destination row (one absolute
   slot or up to three relative slots) and the phase allowance. Online rounds
   allow `K - 1` occupied slots; endings allow one absolute or three relative slots.
   A fixed binary-logit prior grows with whole-STM occupancy and required
   reductions per remaining round. Its `reducePressure` weight defaults to `1`.
   When remaining rounds equal required reductions, unary and STOP candidates
   are masked. Unary operations remain available while there is slack.
   Default budgets therefore reserve the next word's slot and finish each closing.
   An initially infeasible budget may still leave an incomplete sentence;
   an attempted push into a full stack is an assertion failure.
   The trace records the actual operator and position in each round. Both
   derivations use hard execution with a probability-weighted straight-through
   gradient from the untempered model softmax. At every sentence closing, each row
   trains both derivations from cached word vectors, retains the strictly
   lower sentence loss (ties retain exploit), and commits before the next
   sentence starts. Batch-end answer loss is outside this comparison.
   Explore uses the shared
   `composeTemperature` and forces one alternative round. Evaluation runs
   exploit alone with deterministic logit argmax.
   See [Language](Language.md#one-operation-per-round-item-75).

> **Convention pin — newest-at-slot-0, shift-RIGHT.** The **free / newest**
> slot is **slot $0$** and the rolling window shifts **right**, dropping
> the oldest (the last occupied slot, $\text{cap}-1$ at capacity).
> `peek(n)` counts $n$ back from the newest, so it reads slot $n$
> directly; `snapshot()` returns the live slab **newest-first**. The
> "predict the free slot" step predicts slot $0$, and the retained
> context that conditions it is `snap[:, :-1]` (everything but the
> soon-to-be-evicted oldest slot). A named primitive
> `_stm_shift_for_predict` ([Spaces.py](../bin/Spaces.py)) exists
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
   ([Spaces.py](../bin/Spaces.py)) snapshots the previous STM and
   runs the predictor **per slot** (no cross-slot collapse), producing a
   `[B, prev_N, D]` slab stashed on `self._stm_predicted_slab`.
2. **Perceive via `_stm_set_all_slots`.**
   `_stm_set_all_slots(slab)` ([Spaces.py](../bin/Spaces.py))
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
serial-vs-attention guard was **lifted**: serial sequencing and
attentional filtering are the same regime, not mutually exclusive
options — `<attention>` (off/primer/second-order/low-rank, read per-Space
with default `off`) composes with `<symbolicOrder>` rather than being
gated by it. The legacy `<hasAttention>` boolean is deprecated and inert,
superseded by the `<attention>` element.

> **Correction — `MentalModel.xml` does not set `<attention>`.**
> `data/MentalModel.xml` has no `<attention>` element at all, so the knob
> resolves to its default, `"off"` (`TheXMLConfig.space(section,
> "attention", default="off")`, [Spaces.py](../bin/Spaces.py)). The
> trainable target config therefore runs the Gaussian / word-span window
> below (CS$\to$PS) and the taxonymic mask (CS$\to$SS), but **not** any
> `<attention>` primer/second-order/low-rank retrieval mode. An earlier
> draft of this chapter claimed `<attention>primer</attention>` was set;
> that was wrong. See also [Section 12](#12-nanochat-comparison-framing).

**CS$\to$PS word input.** The word-major radix path and legacy serial inputs use
different representations.

- **Word-loop mereology (`<serialObjectMeta>` on).** The outer sequence is
  `[B,W,D]`, with one position per word and $W$ bounded independently by
  `<serialWordCapacity>`. BasicModel stages W=256 but its compiled
  `torch.while_loop` stops at the final live column in the batch. InputSpace
  owns iteration $w$ and presents that word's local state to PartSpace; only
  then does PartSpace gather its complete discrete ids `[B,P_raw]` and apply
  its sigma set-fold inside the loop. $P_raw$ is the longest complete spelling
  in the current batch; it is not bounded by `PartSpace.nOutput`. PS and WS
  retain their eight-wide live fields in BasicModel, CS binds all three PS and
  three WS folds by equal location, and one concept carrying its actual order
  enters the configured STM (default capacity 8). Grammar reductions free STM
  slots while the outer loop continues.

  With packed training, consecutive complete sentences share the W=256 slab
  but not conceptual state: CSLang commits a row-local soft boundary between
  them. Full resolved concepts remain only in the CS-owned `[B,8,D_c]` STM.
  SymbolSpace receives a scalar `[B,W,1]` activation/reference slab and
  quantizes that value; it does not retain a dense `[B,W,D_c]` concept
  history.

Legacy serial inputs retain the following window policies:

- **Gaussian window (default).** `gaussian_window_word(full_seq,
  center_k)` ([Models.py](../bin/Models.py)) forms a single envelope
  centred on the processed word at $k$:

  $$
  w_i = \exp\!\left(-\frac{(i - k)^2}{2\sigma^2}\right), \qquad
  \sigma = \text{maskRate} \cdot T,
  $$

  normalized so the center weight $w_k \approx 1$. The peak sits at the
  current word; words far from $k$ are attenuated toward $0$ by the
  Gaussian tail. The windowed percepts are mapped into conceptual space
  and **summed**, so word $k$'s representation carries a faint meronymic
  trace of its neighbours (the local context *is* part of the embedding
  signal). This **replaces** the prior BERT-style hide-a-token
  `create_ir_mask` on the per-word grammar path; it is not
  target-hiding — the center word is preserved. The centring is
  **hardcoded**: peak at the current word.
- **`word_span_window` (legacy hard same-word fallback).**
  `word_span_window(full_seq, center_k, word_idx)`
  ([Models.py](../bin/Models.py)) replaces the soft Gaussian tail with a
  **hard** same-word mask: the masked **sum** over exactly the slots
  sharing word $k$'s id (`word_idx == word_idx[:, k]`), so PS processes
  the active word's own span only — no part of a neighbouring word leaks
  in via the tail. Falls back to the single slot `full_seq[:, k:k+1, :]`
  when no per-slot word index is available (e.g. byte mode). Whole-
  sentence context still re-enters serial processing via the read prelude
  gist/intent, not this window.

**CS$\to$SS taxonymic mask.** On the symbolic side the inverse
recommender zeros **non-admissible** codebook rows before a lift / lower
(union / intersection) reduction so an operand can only resolve to a row
of the right grammar category and conceptual order. The admissible row
set is built by `priming_kwargs_for_slots`
([Language.py](../bin/Language.py)) — intersecting `refs_by_category`
with `refs_by_order` per slot — and applied in the recommender's
`_row_mask` closure ([Layers.py](../bin/Layers.py), nested in
`Ops._binary_op_recommend`), which admits only the category/order-matched
rows plus the $\bot$ / $\top$ sentinels. A parallel `_row_weights` closure
multiplies a taxonymic **priming** mask over the admitted rows (the
`left_priming` / `right_priming` weights from `taxonomy.priming_mask`).

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

## 5. Routing parser: SS-analysis vs CS-execution {#routing-parser}

The grammar runs through the signal router (`LanguageLayer`,
[Language.py](../bin/Language.py)); `SymbolSubSpace` owns it as
`self.languageLayer`. Conceptually the work splits in two:

- **SS-analysis** — `SymbolSubSpace.compose`
  ([Language.py](../bin/Language.py)) is the analysis stage: a soft
  superposition over the taxonymic codebook that selects, per-space, a
  **hard rule dict** `current_rules = {space_role: list[list[int]]}` — an
  outer per-space_role entry holding one inner rule-id list **per batch
  row** on the full-router path, or a single batch-shared inner list on
  the default-only path (`_default_compose_rules`, `_flatten_selected_rules`
  tolerate both shapes, plus the legacy flat `list[int]` per-space_role
  form). It chooses *which* reductions fire.
- **CS-execution** — actually applying the chosen reductions (lift,
  lower, union, intersection, swap, quantize, not) to the concept tensors
  runs CS-side in `ConceptualSpace.forward` and the WholeSpace
  stack-route path, with the per-space `SyntacticLayer` cursors
  ([Language.py](../bin/Language.py)) executing the unary $\pi$ /
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
> (`OperationSelectionLayer.forward` $\to$ `op(left, right)`),
> caching the root state. The per-space `SyntacticLayer` cursors are then
> **deliberately bypassed** on that path — guarded by
> `not _grammar_is_default_only`
> ([Language.py](../bin/Language.py), the `SyntacticLayer.forward`/
> `.reverse` guard) — precisely so the reduction is
> not double-applied. So in the full-router case the SS-analysis and
> CS-execution stages are **co-located** inside `LanguageLayer.compose`
> rather than separated across modules. The audit found this; it is a
> documented task-5 follow-up, not a finished boundary.

`_grammar_is_default_only` is computed from the configured grammar at
`SymbolSubSpace.__init__` ([Language.py](../bin/Language.py)) and gates
both `compose` and `generate`.

---

## 6. IntraSentenceLayer

`IntraSentenceLayer` ([Layers.py](../bin/Layers.py)) is the in-STM
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

**Loss.** $\mathcal{L}_\text{intra} = \mathrm{MSE}(\hat{c}_t, c_t)$.
`IntraSentenceLayer.intra_loss(pred, target)` ([Layers.py](../bin/Layers.py))
is a plain `F.mse_loss` helper documented as the training-path wiring
point, but the live path does **not** call it: `ConceptualSpace.
_accumulate_intra_loss` ([Spaces.py](../bin/Spaces.py)) inlines the same
math (`(prediction - target).square()`) directly. `intra_loss` is
exercised only by `test/test_intra_sentence_layer.py`, not by any forward
call site — an unused-but-documented helper, not dead code to delete.

`_accumulate_intra_loss` keeps each per-word step's scalar loss in a
**list** rather than folding it into a running `accum + step_loss` sum
(2026-07-08 fix): the old running-sum pattern built a per-word add-chain
*inside* the compiled forward, which Inductor inlines transitively into
one kernel with a buffer argument per step — over Metal's 31-buffer
kernel-arg limit on wide configs (e.g. `N=64` produced a 34-arg
scalar-sum kernel, a fullgraph blocker immune to fusion caps). As a list,
each step's scalar stays an independent graph output, and
`consume_intra_loss` ([Spaces.py](../bin/Spaces.py)) chunk-sums the list
**eagerly** (post-body, pre-backward, mirroring the ARMA term) and
returns the per-step mean, resetting the accumulator. The weight is
`<intraLossWeight>` (default `0.1`); the term is gated off when grad is
disabled or the weight is non-positive.

---

## 7. Per-word router firing

The `<routerWireSerial>` knob ([model.xsd](../data/model.xsd)) gates
when the router fires on the serial path:

| Value | Behaviour |
|---|---|
| `per-word` | fire per word; boundary fire off |
| `boundary` | fire only at the sentence boundary |
| `both` | **(default)** per-word AND boundary both fire |
| `off` | neither fires |

The **per-word fire** is the C stage of the peer pipeline:
`LanguageSpace.compose` runs over the B-stage STM snapshot and its timestamped
result becomes visible to B two symbolic indices later. On the normal MPS
fullgraph it is part of the tensor `while_loop` and the serial grammar route
remains capture-clean. With `BASICMODEL_MPS_WORD_LOOP_FULLGRAPH=0`, the same C
stage runs in the static host scheduler. The **boundary fire**
(`BasicModel._chart_compose_at_C`, [Models.py](../bin/Models.py)) runs
iff `router_wire_serial in ('boundary', 'both')`.

> **Routing conditioning of the intra predictor is LANDED.** Every
> `SymbolSubSpace.compose` call (per-word or boundary, default-only or
> full-router) builds a first-class `RoutingState`
> ([Language.py](../bin/Language.py)) ADDITIVE to the unchanged
> `current_rules` host-side rule dict: `_synthesize_rule_probs`
> ([Language.py](../bin/Language.py)) turns the fired rule_ids into a
> dense `rule_probs: [B, n_rules]` distribution — a gradient-bearing
> soft-marginal aggregation (`_synthesize_rule_probs_soft`) whenever the
> router ran tensorially, or a DETACHED hard scatter
> (`_synthesize_rule_probs_hard`: unit mass onto the fired rule-ids,
> L1-normalized per row) on the default-only fast path — and stashes it
> on `symbolSpace.routing_state.rule_probs`.
> `ConceptualSpace._intra_routing_for_predict`
> ([Spaces.py](../bin/Spaces.py)) reads that tensor, returns it only when
> its last dim matches `n_rules == len(TheGrammar.rule_table)`
> (`IntraSentenceLayer.routing_proj`'s expected width) and aligns its
> batch dim to the predictor's STM-snapshot context (broadcast when one
> side is `B=1`; `None` on an unreconcilable mismatch, fail-loud on a
> non-finite tensor). Both `_stm_predict_then_perceive_serial` and
> `_stm_predict_then_perceive_parallel` ([Spaces.py](../bin/Spaces.py))
> pass the resolved routing tensor into `intraSentenceLayer.forward`,
> which projects it `[B, n_rules] \to [B, concept_dim]` via
> `routing_proj` and adds it as a bias to the Sigma output
> ([Section 6](#6-intrasentencelayer)) — so the per-word fire now DOES
> make the in-STM predictor rule-aware, not just the SS dispatch context.
> `routing=None` (no reachable `symbolSpace`, wrong width, or an
> unreconcilable batch mismatch) degrades to the un-biased predictor,
> byte-identical to the pre-wiring behaviour.

---

## 8. Masked-word reconstruction via priming

Masked words enter the pipeline as **all-zeros** percept slots. On
reverse, the router fills the blank via a best-fit codebook walk biased
by the taxonymic prior, POS selection, and accumulated prediction — it
reconstructs the most plausible word for the slot given the grammar
context and the codebook geometry.

**Priming machinery.** The bias enters through `left_priming` /
`right_priming` in `Basis.lift` / `Basis.lower`
([Spaces.py](../bin/Spaces.py)), forwarded down to the inverse
recommender's `_row_weights` closure ([Layers.py](../bin/Layers.py),
nested in `Ops._binary_op_recommend`) where
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

The [two-truths contract](specs/2026-09-16-two-truths-ideas-and-relations.md)
assigns a row to each grammatical S. Its selected derivation decides the
one-slot or three-slot allowance. An absolute S fuses NP and VP into one
idea point. A generic subject or an operand that references a relation makes
the clause relative: its three roles are `[left, predicate, right]`, with
native row references and no fused point. The committed physical STM stores
those roles newest-first as `[right, left, predicate]`.

`NP → S` keeps an embedded absolute point; `NP → REF(S)` carries a relative
clause reference. The tensor clause journal records these endings during each
hard path. References remain trial-local until the per-sentence winner is
known. Its durable transaction registers children before the enclosing row;
embedded content receives no assertion authority merely from registration.
An absolute enclosing clause can fuse only when all operands have points.

Item 7.5 still supplies the fixed operation budget and early stop. Every
round chooses one binary or unary operation, with STOP eligible only when
the current sequence fits its allowance. Reduction pressure grows with
occupancy and urgency; at the deadline only the required binary reductions
remain eligible. `_stm_reduce_to_single_S` runs at most `2K` rounds for STM
capacity K, capped by a positive `syntacticOrder`. The default feasible
budget completes the reduction. A manually undersized cap can still leave
an incomplete forest, which is tagged incomplete and cannot publish a
program or memory row. The unchanged trained depth-three campaign remains
an empirical test of the selected grammar, not a guaranteed learned result.

The clause closing is the sole production relation writer. It admits `part`,
`implies` and `operator` rows into the shared store; there is no catch-all
relation kind, WholeSpace META insertion, reducible/ineffable routing or
multiplicative learn-score gate. Independent `(c_plus, c_minus)` poles retain
both and neither distinctly. They record identification evidence. Source
provenance supplies a separate scalar `trust` for the event as a whole;
neither changing that scalar nor withdrawing its authority changes the pair.
The row also keeps the ended field's `order`, so an abstract field can descend
through the sigma inverses. Sentence content never selects its own trust.
Luminosity reads idea rows only. See [Reasoning](Reasoning.md) for row-indexed
relation readers and the declared `true` thought operator.

---

## 10. LTM as the chain of STM end-states

`symbolSpace.ltm_store` is the shared `TernaryTruthStore` for all completed
clauses. `<ltmConsolidation>` is an ignored compatibility input and selects
no alternative backing store. Storage alone does not certify that a
described referent exists. Production observations must provide their owned
end state. Occupancy determines idea or relation; the REL identity determines
part, implies or operator. The writer neither needs nor stores a program.
[Clause admission](../bin/ClauseRow.py),
[observation writer](../bin/Models.py).

Each row retains its native `row_ids` address, three `refs`, relation kind,
independent positive and negative evidence poles, scalar source trust, order
stamp, role presence, grammatical
mode, polarity, evidence kind, writer origin and stable occurrence identity.
An idea uses one point slot; its cached NP and VP references are taken just
before fusion. Its numerical NP, V and modifier target exists only during
reading, for prediction training. A relation uses three infix roles and
references. Reading a row into words generates from its stored field and the
grammar; no derivation is retained or replayed. An operand that is itself a relation has no point,
so its vector slot is null and readers follow the row address.

The September 29 amendment adds `REL_DEF`, a fourth relation kind written
only by `interpret`: `word DEF object`. The two operands name conceptual
identities in `refs`, with null operand vectors and a fixed DEF atom in the
middle slot. A new word costs one inventory row, which becomes the object's,
and one row of this store. The definition has its own fixed `.when` and a
refreshable recency timestamp, and it is eligible for forgetting. Its
provenance and scalar trust do not come from the sentence containing the word.
One derived index supplies form/unit → word, word → objects and object → words;
load and compaction rebuild it. See
[the definition contract](specs/2026-09-16-two-truths-ideas-and-relations.md#17-definitions-word-def-object-decided-alec-2026-09-29).

The ended field's encoded `.where` and `.when` are stored separately from
its opaque concept point. They come from the clause's owned leaf field,
not coordinatewise averaging. `.when` retains the input field timestamp
under item 9b; the chronological store timestamp is separate. Embedded clauses
register without assertion. External TruthSet admission can assert the outer
row through provenance, while a conversation observation, question or estimate
cannot certify its own referents. Reasserting a relation joins its two evidence
poles independently rather than appending a duplicate claim.

Sentence expectation consumes its bounded row/document observation view,
including owned references. A different stream, internal thought or provisioned
fact cannot become an external predecessor through global recency alone.
The legacy `get_stm_chain` adapter is a read view of the shared store, not a
separate writer or configurable persistence path.
[Prediction context](../bin/Layers.py), [meaning roles](../bin/Meaning.py).

Tensor columns ride `state_dict`; bindings, semantic scope, constituent
references and source text ride the versioned `truth_semantics` sidecar.
Version 4 contains no clause or derivation field. Older derivations are checked
against their old fingerprint, then dropped with a warning. Restore validates occurrence identities and fingerprints. Missing
required metadata makes evidence unavailable rather than silently rebinding it.
Old catch-all conversation rows and their unavailable dependents are dropped
with a warning; old provisioned rows are re-provisioned from XML. WholeSpace
META state is retired and the native concept index is rebuilt from its rows
and bindings. Native word/object META bindings migrate to DEF rows in every
reading, including the serial aligned path; their META identities and fold
are retired. [Checkpoint migration](../bin/Layers.py).

Admission preflights clause dependencies and native concept capacity. A full
store returns `-1` for a new clause without publishing a child prefix; it may
still join evidence into an existing relation. Optional estimates yield to
observations when capacity is tight. Stable addresses survive row compaction,
and referenced prior states participate in the existing retention rules.

`Exist` compares the full occupied description, bindings, scope and references
against assertive fact records, retaining positive/negative degrees and
provenance separately. Conversation observations, questions, estimates and
unverified legacy rows are ineligible. The degree belongs to evidence about
the referent; model activation alone cannot establish it.
See [Existence evidence](ExistenceEvidence.md) and
[the implemented lookup](../bin/reasoning.py).

---
## 11. Inter-sentence prediction

### Clause endings and shared LTM

The [two-truths contract](specs/2026-09-16-two-truths-ideas-and-relations.md)
ends every grammatical S. `NP → S` retains the absolute clause's fused
point; `NP → REF(S)` carries a relative clause by reference. The allowed
row width is one or three, selected by the derivation. Relativity propagates
through a compound VP and enclosing S. The 7.5 reduction deadline still
requires completion within the fixed round budget.

Each derivation owns its clause references until the per-sentence choice.
Both hard paths train; only the lower-loss path is committed, with exact
ties going to exploit. Its nested rows register before the enclosing row,
without granting the embedded content assertion authority. The clause closing
is the writer of asserted relations; `interpret` owns the separate DEF
transaction above. The closing retains independent evidence poles, field
coordinates and the one-slot or three-slot end state. The temporary operation
record is discarded; generation reads the row without replay. WholeSpace
taxonomy insertion and the reducible/ineffable routing are retired.

The expectation target reads the NP, V and modifier roles before fusion,
while the sentence is open. These roles are not a stored derivation. A separate
kind logit predicts idea versus relation. Its identity candidates are held
in the bounded situation: live constituents, recent clause frames and frames
brought into STM by `what`. The predictor carries an imputed identity;
composition does not search LTM for one. Clearing the situation removes that
cross-sentence carrier. Word provenance remains the current sentence's own
reconstruction record when its subject refers to an earlier occurrence.


### Structured production path

`sentenceExpectationScope=structured` is the model default when sentence
prediction is enabled. `SentenceExpectation` reads the bounded
chronological observation view as `[K, 3, D]` values plus `[K, 3]` occupancy.
The three positions are NP1, VP and NP2. Padding is masked before the network;
role and chronological positions remain distinct. It predicts three separate
vectors, three presence logits and one clause-kind logit. The loss combines
role MSE with binary cross entropy for presence and clause kind. Targets are detached;
current-step source representations remain live under the objective-local
boundaries in [GradientFlow](GradientFlow.md). For individual-reference
grammars, item 6.5 re-encodes the latest source through the current identity
columns before each trial. Older context and the observation target detach.
Once columns exist, role prediction error is measured in their coordinates;
presence and clause-kind losses keep their existing meanings. This preserves
sentence optimizer boundaries without keeping a preceding compose graph
across an update. See [Layers.py](../bin/Layers.py) and
[IndependentComponents.py](../bin/IndependentComponents.py).

The independence population comes from retained external observation rows,
excluding definitions, estimates and asserted facts. It is snapshotted before
reading, independently of the batch partition. Accumulated innovations use
identified object coordinates and retain occurrence addresses. A has at most
two active noun columns per ordinary frame; B has at most one change column.
The same expectation objective owns both dictionaries. Recurrence admission
and witness counts happen only after the kept closing. These are mechanism
contracts under review, not a finding of learned identity or verb reuse.

The packed observer uses the existing ended end-slot/depth outputs for each
sentence, with an explicit STM-to-infix permutation. Corpus source addresses
identify each sentence's document, including changes within a packed row.
`begin_document` clears only the selected prediction stream; already-scored
losses, other rows and durable history survive. Soft packed-brick resets keep
the same document stream; only hard resets or document changes make it cold.
Weight restore starts the
observation view cold. These are sequence and local-role contracts; retained
compound references remain part of the separate nesting migration in the
[integrated spec](plans/2026-09-15-next-sentence-as-the-production-objective.md).
See [Layers.py](../bin/Layers.py) and
[Models.py](../bin/Models.py).

### Historical root baseline (`sentenceExpectationScope=root`)

A **lifted `IntraSentenceLayer` instance** (`_inter_predictor`,
[Layers.py](../bin/Layers.py)) predicts the next end-state over the
external observation sequence — the same predictor class as the in-STM one,
instantiated at the inter-sentence level. Its transient `_inter_context`
view is bounded and per-row; durable LTM retains detached observations.
Current-step source context keeps its encoder graph. Consuming the prediction
loss and entering the next brick detach that view without deleting history.
Document resets clear the selected view and pending estimate, preserving
other rows and durable LTM ([Layers.py](../bin/Layers.py),
[Layers.py](../bin/Layers.py),
[Layers.py](../bin/Layers.py)). Its chain window is
$K = \min(\text{ltmCapacity}, 8)$ (`_inter_chain_window`): the AR signal
that predicts the next end-state lives in the last handful of sentences,
so a small bounded window is used rather than the full `ltmCapacity`.

`predict_next_end_state(b=0)` ([Layers.py](../bin/Layers.py))
produces the next end-state **shape** $(\hat{d}, \hat{p}[\hat{d}, D])$:

- **Chain reduction (ragged $\to$ fixed), mode-dependent.**
  Each chain entry is reduced to its **root** by
  `_reduce_end_state_to_root` ([Layers.py](../bin/Layers.py)) — but WHICH
  slot is the root depends on the LTM mode ([Section 10](#10-ltm-as-the-chain-of-stm-end-states)):
    - **Legacy mode.** The end-state is stored newest-first, so the root
      lives at the **last** slot ($\text{depth}-1$): for an absolute
      end-state (depth 1) that is slot $0$ (the collapsed idea); for a
      relative end-state (depth 3) it is slot $2$, the predicate (the head
      the relative structure hangs off).
    - **Consolidated mode.** The payload is the store's native INFIX
      `[idea1, predicate, idea2]` order, and the root is always **slot
      $0$** = idea1 — the subject/topic, present even when there is no
      predicate (an absolute row has no `predicate`/`idea2`, only
      `idea1`), so it is the one slot every row shares regardless of
      depth.

  The last $K$ roots form a `[1, K, D]` context, left-padded with
  zeros so the most recent sits at the tail (newest-at-$-1$, like the ARMA
  ring).
- **Root prediction.** `_inter_predictor.forward(context, routing=None,
  parallel=False)` $\to$ `[1, D]`, the predicted root.
- **Loss.** $\mathcal{L}_\text{inter} = \mathrm{MSE}(\hat{p}, p)$ on the
  roots, accumulated by `_accumulate_inter_loss`
  ([Layers.py](../bin/Layers.py)) and drained by
  `consume_inter_loss`, weight `<interLossWeight>` (default `0.1`). The
  actual root is detached for this comparison, while live preceding context
  can train the encoder as well as `_inter_predictor`. Teacher reconstruction
  does not disable this term. `observe_stm_end_state` scores each arriving
  observation once. Evaluation does not accumulate training losses. Joint
  gradients use the [objective-local gradient contract](GradientFlow.md).
- **InfoNCE next-idea contrastive term (optional, additive).** When
  `<interContrastiveWeight>` is positive (default `0.0`, off),
  `observe_stm_end_state` also ranks the actual next root above the
  chain's past roots (negatives) under $\cos(\hat{p}, \cdot) /
  \text{temp}$ (`<interContrastiveTemp>`, default `0.1`) via
  `_accumulate_inter_contrastive` $\to$ `consume_inter_contrastive_loss`
  ([Layers.py](../bin/Layers.py)) — a `torch.nn.functional.cross_entropy`
  over `[pos_root; neg_roots]` with the positive at index 0. Best-effort:
  a short/odd chain simply yields fewer negatives (the accumulator no-ops
  with none; the MSE term above still runs regardless). Fail-loud on a
  non-finite step.

The prediction is subtracted only at the closing, as described in
[ExpectationRetention](ExpectationRetention.md). The observation remains in STM
and LTM unchanged; the conceived roles are derived evidence for the chooser.
`generate_sentence` sends a positive predicted idea directly to `<generate>`.
There is no comprehension-time prediction injection.

The following depth-copy behavior describes the explicit **root benchmark**;
production structured expectation predicts three distinct vectors and presence.

> **Honesty — $\hat{d}$ is a copy-last AR prior.**
> The predicted **depth** $\hat{d}$ is a simple AR prior: the depth of the
> **most recent** end-state in the chain (a relative sentence tends to be
> followed by structure of the same shape; an absolute by an absolute).
> This delivers `depth in {1, 3}` without a separate learned head, but it
> is a **scaffold** — a tiny `concept_dim $\to$ 2` argmax head is the
> documented upgrade path. The chain reduction always collapses each
> end-state to ONE root vector rather than mean-pooling over depth (the
> root carries the sentence-level signal, mirroring the ARMA
> `_pool_sentence_rep`) — but which physical slot is "the root" is
> mode-dependent (see the "Chain reduction" bullet above): it is
> "slot-0" only in consolidated INFIX mode, where idea1 is always at
> slot 0; in legacy mode the root is the OLDEST slot
> ($\text{depth}-1$), which is slot 0 only for a depth-1 absolute
> end-state. A non-finite predicted root raises.

---

## 12. nanochat comparison framing

The trainable target is `MentalModel.xml`:
`<symbolicOrder>1</symbolicOrder>`,
`<data><dataType>embedding</dataType>`,
`<sentenceExpectation>true</sentenceExpectation>`,
FineWeb data (`<shardDir>data/fineweb</shardDir>`). This is the
configuration that exercises the full STM stack — serial sequencing, the
CS$\to$PS windowing and CS$\to$SS taxonymic mask
([Section 4](#4-attentional-filtering)), the in-STM and inter-sentence
predictors, and the LTM chain. It does **not** set `<attention>`, so that
knob resolves to its default `"off"` — see the correction in
[Section 4](#4-attentional-filtering); the legacy `<hasAttention>` boolean
is deprecated and inert regardless.

The comparison loss, as trained by `runBatch`, is

$$
\mathcal{L} = \mathcal{L}_\text{IR} + \mathcal{L}_\text{intra}
+ \mathcal{L}_\text{inter} + \mathcal{L}_\text{ARMA}
+ \mathcal{L}_\text{inter-contrastive},
$$

the masked-LM information-reconstruction term at the subsymbolic (PS) (see
[Spaces.md](Spaces.md#within-sentence-ar-retirement-2026-05-14)) plus the
in-STM next-idea term ([Section 6](#6-intrasentencelayer)) plus the
inter-sentence next-end-state term ([Section 11](#11-inter-sentence-prediction))
plus the discourse ARMA($p$, $q$) term (`BasicModel._discourse_arma_loss`,
[Models.py](../bin/Models.py); `InterSentenceLayer.observe`,
[Architecture.md](Architecture.md) — a separate ring-based sentence-rep
predictor on the SAME `InterSentenceLayer`, distinct from the STM-chain
`_inter_predictor` of Section 11) plus the optional InfoNCE next-idea
contrastive term ([Section 11](#11-inter-sentence-prediction), off by
default via `<interContrastiveWeight>`). This is analogous to a
transformer language model's next-token cross-entropy, but computed **in
concept space** rather than over a token vocabulary:
$\mathcal{L}_\text{intra}$ predicts the next idea within a sentence,
$\mathcal{L}_\text{inter}$ the next sentence's end-state shape,
$\mathcal{L}_\text{ARMA}$ the next sentence-level representation over the
discourse AR/MA rings, and $\mathcal{L}_\text{IR}$ reconstructs masked
content — together the concept-space counterpart of the autoregressive LM
objective a system like nanochat trains.

> **Out of scope.** The comparison harness itself — running this against a
> transformer baseline and reporting the numbers — is a **separate plan**.
> This chapter documents the loss that *would* be compared and the
> configuration that produces it; it does not build the benchmark.

---

## 13. Interaction LTM and the What stack

Beside the chain of STM end-states (Section 10), `InterSentenceLayer` keeps
a per-row chronological sequence of **interaction slots** for the What
loop ([What spec Section 6](specs/2026-07-27-teaching-modes-and-next-iteration.md#6-ltm-interaction-slots)).
Each slot (`What.LTMSlot`) has two independently optional halves:

| Slot | Operation | Stack effect |
|---|---|---|
| `(input, —)` | OPEN | pushes an unanswered question |
| `(input, output)` | COMPLETE | no change |
| `(—, output)` | CLOSE | pops the most recent unanswered input (LIFO) |

The stack is the *imbalance* of the sequence: **parity** means no unmatched
input-only slot. There is no frame object and no in-place edit of an
opening slot; a later output-only slot closes it. The API on the layer
(`bin/Layers.py`):

- `append_what_slot(slot, b)` — validates (an empty slot and an unmatched
  close are rejected; closure pressure must not decrease while questions
  are open), detaches both halves from the live graph, appends the
  chronological record with its grammar trace, and trims only *balanced*
  prefixes when `ltmCapacity` is exceeded (an open question is never
  evicted).
- `get_what_slots` / `open_what_slots` / `what_open_depth` /
  `what_at_parity` — read-only views, oldest first.
- `what_context(question, b)` — the target-free chooser context: the
  question's coordinates, the input / output presence masks and
  representations, the open depth, parity, and the current closure
  pressure. `Model._what_grammar_context` folds it into the 29-dim vector
  the grammar chooser's `what_projection` consumes.
- Row hard resets clear the row's slots and its pressure; soft resets do
  not.

The output half always records the response the model actually produced,
never the desired `Data` answer.

SymbolSpace always owns exactly one `Layers.WhatInteractionMemory`, available
as `SymbolSubSpace.what_memory`. `Model._what_memory()` returns this owner.
Expectation has no interaction-memory copy or delegate API, and the
`whatThinkingMemory` configuration switch is retired. Provisioning resets the
What episode at its hard interaction boundary; suspending external expectation
does not preserve that episode. The **episode credit boundary** remains explicit:
`begin_what_episode(b)` / `end_what_episode(b, detach=True)`; under
`<whatThinkingDetach>episode` the values appended inside an episode stay
live on the autograd graph until `end` (called after the optimizer step),
so a root answer loss can reach states created at earlier iterations;
under the default `slot` every value is detached at append (the
established behaviour). `what_context()` also exposes `open_question` (the
newest unanswered input) and `latest_output` for the resolve step.

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
  `<interLossWeight>`, `<routerWireSerial>` and `<ltmCapacity>`.
- [What spec](specs/2026-07-27-teaching-modes-and-next-iteration.md) and
  the [mathematical thinking specification](specs/2026-09-09-mathematical-thinking.md)
  — the interaction slots' role in thinking (Section 13).

## Slot provenance for the fold ladder (2026-09-10)

Beside the slot kinds, the STM keeps one fixed-shape provenance slab
`_wholes` `[B, capacity, 3]` of longs, newest at slot 0 like the buffer:
per slot the index of the coarser analysis whole (the word) the unit
belongs to, the unit's loop position and the clause whole's index; `-1`
means none (a fold across wholes has no whole and is no single unit).
The slab is CS-owned sentence state exactly like the reference slabs: the
compiled word loop carries it through `torch.while_loop` beside the six
STM tensors (fullgraph), and the only mutation paths are the pure
primitives `ShortTermMemory.functional_wholes_push` (masked slot-0 push),
`functional_wholes_reduce` (a top-2 fold keeps a whole or clause only when
both operands shared it) and `functional_wholes_reset` (a sentence
boundary); the eager methods `note_whole_masked`, `note_reduce_wholes`
and `same_whole_rows` wrap them.  `same_whole` / `shared_whole` /
`newest_units` are the tensor reads.

The reduce step (eager and compiled alike) turns the slab into the
grammar's `chunk` licensing: `_chunk_structural_prior` is a `[B, 1, R]`
additive logit, the learned `chunk_prior` on a pair the tiling places in
one whole (word or clause) and `-1e4` elsewhere, handed to the Language
chooser as `op_prior`.  After the fold, `_chunk_reduce_provenance`
appends a chosen chunk of two units with concept ids to the
ConceptualSpace's fixed proposal slab (`[B, 32, 3]` plus a count, also
loop-carried), looks the pair up in the admitted phrase-row table
(`chunk_row_table`, `[64, 3]`) and, when admitted, references the phrase's
row at the folded slot (`apply_phrase_rows`).  `ConceptualSpace.Reset`
drains the proposal slab into the host counts that admit phrases
(doc/plans/2026-09-10-meronomy-fold-ladder.md, Phase 2b, contract 7).

Committing live state under the compiler (2026-09-11): every STM
property setter (`_buffer`, `_depth`, the order and reference slabs,
`_wholes`) goes through `ShortTermMemory._assign_live`, which copies in
place when compiling and the shapes match. With a `torch.while_loop` in
the graph, dynamo (torch 2.14 nightly) drops a later attribute
*assignment* of an attribute assigned earlier in the same graph (the
per-forward seed in `_per_word_prelude`), so the sentence reduce and the
reconstruction loss after the word loop read the empty seed; an in-place
copy is tracked and keeps the gradient. Eager execution keeps the plain
assignment. Anything an eager consumer reads after a compiled call is an
explicit output of `_forward_with_compiled_sentence_state`, never an
attribute escape.

## Taxonomy evidence and memory ownership (September 16)

Conceptual-taxonomy queries capture a bounded view of the existing native
reference records. The view owns no durable state and is rebuilt after restore.
Proof sources identify native edges; successful paths cannot append world-fact
lemmas. The single normal controller records nested questions and typed returns
on `WhatInteractionMemory`. Its attended context uses row-owned STM/history,
the recent discourse chain and frames brought into STM by cued `what`, under
the same work allowance. It never reads a recent slice of the LTM store.
The closing writer owns the per-role leaf-code index ([AccessibleMind](AccessibleMind.md)).
See [Taxonomy](../bin/Taxonomy.py), [Thoughts](../bin/Thoughts.py),
[SelectedMeaning](SelectedMeaning.md) and [TaxonomyQueries](TaxonomyQueries.md).

## Shared grammatical query references (September 16)

The explicit query adapter uses ConceptualSpace's existing named-concept
owner for specialized VPs. Checkpoints retain that identity and its payload
through the ordinary conceptual tensor state and structural sidecar. Complete
description operands reference existing LTM occurrences, retaining the inner
VP, NP2, bindings and scope. Candidate formation cannot create occurrences;
unknown namespaces or missing metadata fail without row substitution.
[VP owner](../bin/Queries.py),
[occurrence resolution](../bin/Queries.py).

Durable LTM occurrence reads detach their content; row-owned ordinary thought
references preserve live episode meanings. Completed selected programs use
ordinary levelled history, nested retention and one shared work meter. Complete
child meanings retain their structure; unsupported physical folds stay
unavailable. Natural-wording learning and residual policy credit remain open.
[Query contracts](QueryContracts.md) describes the execution boundary.

## Ordinary levelled thoughts (September 17)

The existing interaction deque now admits ordinary `ThoughtRecord` values with
explicit level and termination transitions. Replay recovers current and
suspended contexts, including scope, bindings and returned results; root level
is not completion. Only a closed legacy prefix may precede ordinary records.
Stable row-local `thought` references resolve complete live episode meanings;
capacity retains active contexts and referenced occurrences. The structural
sidecar persists detached copies and validates replay on restore. The normal
selected linguistic controller now creates its shared meter, records exact
executor/controller work, and runs `what(Q)` children through the same episode.
Residual credit and learned utility remain open. See [ordinary thought
history](ThoughtHistory.md).

## Completed-row query permission

Boundary readiness derives from the requested `Understanding`'s owned sentence
programs, not mutable STM staging. Missing/padded rows cannot enable checked
queries, and nested resolution can narrow but not widen row permission. A
sentence path masks even a parent boundary until it returns. This is separate
from ordinary controller work; see [Query phases](QueryPhases.md).

## Retained LTM content after origin withdrawal

Origin clearing retains records reached through surviving occurrences or the
ordinary thought history. A retained row keeps its stable ID, content and
source text, but a withdrawn accepted fact becomes unverified with zero trust;
questions and estimates retain their own provenance. The truth view therefore
does not surface a retained request row as accepted evidence. See
[nested retention](NestedRetention.md).
