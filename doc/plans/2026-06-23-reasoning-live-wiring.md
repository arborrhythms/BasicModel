# Turning Reasoning On — `<reasoningIterations>` live-wiring

> Companion to `doc/plans/2026-06-23-truth-grounded-reasoning.md` (the
> architecture). That plan landed the substrate gated-dark (Phases 0–7,
> `bin/reasoning.py`, 50 tests, suite 2773). This plan **turns it on**: a single
> `<reasoningIterations>` knob drives a recurrent tool-use loop that produces a
> chain of N reasoning ideas and **decodes them to N sentences**, ranked by
> subsymbolic relevancy. Each phase stays full-suite-gated; absent flag ⇒
> byte-identical.

## Status (2026-06-23)

### Predictor workstream — ARMA-as-tool + reasoner-as-next-sentence-predictor

Alec's direction: ARMA becomes a TOOL the reasoner uses (added to the grammar
`<queries>` as `arma(X)` → `TruthGroundedReasoner.arma()`); next-sentence
prediction is the reasoner's job, trained in the gradient path; train
reconstruction on some trials and pure prediction on others.

- **Decode round-trip G1 ✅** — lift/lower reverse route the `.what` split through
  the nearest-prototype recommender (`Ops.liftReverseAll`/`lowerReverseAll`) when
  a codebook basis is present (distinct constituents; closes "the cat"→"t t");
  balanced-split fallback preserved. `test/test_lift_lower_recommender.py`.
- **`arma(X)` in `<queries>` ✅** — `complete.grammar` + fixture + the reasoner
  tool (calls `symbolSpace.discourse.predict_next_end_state`).
- **Step 1 — trial-split + idea-space InfoNCE predictor ✅** — `InterSentenceLayer`
  contrastive accumulator (ranks the actual next root above the chain's past
  roots; trains `_inter_predictor`); runBatch `trial_mode` (predict trials zero
  the recon/aux terms so the next-idea signal is the sole gradient); the
  `sequences` dataset + `MM_sequence_predict.xml`. **Root-cause found + fixed:**
  the discourse chain only spans sentences when (a) the BYTE cursor keeps the
  document in one stream AND (b) each document EXCEEDS the byte slab width
  (`InputSpace nIdeas`, ~1024) so it is walked over multiple ticks — a short doc
  collapses to one tick / one end-state and the next-idea loss never fires. Tests:
  `TestInterContrastive` (unit) + `test_inter_contrastive_predict.py` (end-to-end:
  predictor trains).
- **Step 2 — `arma` in the tool-policy ✅** — `NeuralToolUser.reason_predict_next`
  blends `{arma, retrieval, deduction}` into one differentiable `e_hat` (in-graph
  cosine-softmax over the generator query head + a `NextIdeaScorer`; candidates
  detached; absent tools get −inf weight); `Models.reason_predict_next` +
  `_predict_next_loss` (`1−cos(e_hat, observed-next-root)`) + the runBatch hook
  (gated `predictNextLossWeight`). `TestReasonPredictNext`.
- **Step 3 — `reasoningIterations` default 0→1 ✅** — reasoning ON depth-1 by
  default; loss-identical for training (answer/predict weights default 0), not
  inference-identical (query routing changes). Set `0` for the old off behavior.

All three steps default-OFF for the new weights ⇒ existing configs are
training-byte-identical; the default flip changes only inference/serve routing.

- **Phase A — `<reasoningIterations>` knob. ✅ landed** (uncommitted). `model.xsd`
  + `Models.py` read; one-knob switch (absent ⇒ off/byte-identical; positive ⇒
  on, 10 canonical); `<queryReasoning>` folded in as a deprecated alias.
- **Phase B — the live driver. ✅ landed** (uncommitted). `NeuralToolUser` +
  `ReasoningResult` in `reasoning.py`; `BaseModel.reason_about` /
  `_reasoning_spaces` / lazy `_reasoning_tooluser` in `Models.py`; 6 tests in
  `test_truth_grounded_reasoning.py`. Full suite **2787 passed, 0 failed**; off
  ⇒ byte-identical (`reason_about` returns `None`).
- **Phase C — answer-policy loss. ✅ mechanism landed** (uncommitted).
  `reasoning.policy_answer_loss` (differentiable: the generator query head's
  cosine-softmax `α` over the truth-space, gated by a DETACHED hard bridge mask
  — keys/verdict never differentiated, §0) + `policy_examples_from_store`
  (self-supervised transitive pairs); `BaseModel._answer_policy_loss` +
  the runBatch hook (gated `answerLossWeight>0`) + eager generator build in
  `create_from_config` (so the query head joins the optimizer). Unit test proves
  the gradient reaches the generator head. *Subtlety found: `GlobalAttention`
  detaches its query, so the policy loss computes the query→key attention
  directly rather than via GA.*
- **Phase D — N-sentence realize. ✅ mechanism landed** (uncommitted).
  `_realize_vec` / `_realize_idea` / `_realize_ideas` (Tier-1 nearest-codebook
  lexicon, relevance-ordered). Tier-2 grammatical decode deliberately not used
  (inert on compact configs). Degrades to '' when no word vocab.
- **Phase E — serve/infer dispatcher. ✅ mechanism landed** (uncommitted).
  `_detect_query` + `answer_query` (Models.py) + the gated `serve.py` branch +
  the `queries` QA dataset + `data/MM_query_reasoning.xml`. Off ⇒ `answer_query`
  returns `None` before any detection (byte-identical).
- **Adversarial review (2026-06-23) — 11 confirmed findings, fixed.** Notably
  one CRITICAL: the eager generator registered in `model.parameters()` but NOT
  in the optimizer (`getOptimizer` walks `self.spaces` + named modules, not
  `parameters()`), so the query head got gradients but was never stepped — Phase
  C couldn't actually train. Fixed by adding `_intervening_generator` /
  `_reasoning_ga` to the `getOptimizer` module list; `test_answer_loss_actually_
  trains_the_head` now asserts the head weight changes after `optimizer.step()`.
  Also fixed: device placement of the soft components (MPS/CUDA mismatch), a
  hot-loop try/except guard, the `B=None` binary-query crash, the
  "does X exist?" mis-route, an `iterations` off-by-one, and dead realize code.
- **Validation status.** Mechanisms are unit-tested (`test_truth_grounded_
  reasoning.py`: gradient-flow, examples, realize, NeuralToolUser) +
  model-integration-tested (`test_reasoning_cde_model.py`: eager generator joins
  optimizer, training step runs the hook without crashing, graceful
  degradation, honest UNKNOWN). **Substrate coupling (the remaining gap):** the
  real-config end-to-end demo (a populated truthSet store + word-level operand
  extraction + N decoded sentences) is blocked by the **byte/radix grain** of
  the current configs — `complete.grammar` can't parse the truth texts into
  NP/VP/NP relations and there is no word-grain vocab to resolve/realize
  operands. This is the SAME word/byte-grain wall as the decode round-trip
  (Track 1); a word lexer / Track-1 progress unblocks the live demo. The
  reasoning wiring itself is complete and safe (off ⇒ byte-identical).

## Decisions (Alec, 2026-06-23)

1. **One knob.** `<reasoningIterations>` IS the on/off switch — there is no
   separate boolean. Absent ⇒ reasoning off (byte-identical; all existing
   configs unchanged). Present + positive ⇒ on. **10** is the canonical value
   ("10 ideas about the response / a 10-step chain leading to the answer"). The
   prior `<queryReasoning>` boolean is folded into this knob (`query_reasoning`
   becomes `reasoning_iterations > 0`).
   - *Byte-identical reconciliation:* "default 10" is the **on-value**, not the
     schema default. The schema default (element absent) is **off**. Adding
     `<reasoningIterations>10</reasoningIterations>` turns it on with a 10-step
     chain; `<reasoningIterations>0</reasoningIterations>` is explicitly off.

2. **Output = N decoded sentences, guided by subsymbolic relevancy.** The answer
   is not a bare posture — it is the N reasoning ideas `M₁…M_N` **realized to
   sentences**, ordered/selected by the GlobalAttention α relevance over the
   typed truth-space (the "subsymbolic relevancy"). Posture + trust ride along as
   metadata. This couples reasoning to the idea-decode realizer (two-tier, §D).

## Substrate (built, dark — `reasoning.py` imported by zero source files)

| Piece | Anchor |
|---|---|
| Hard tools (`part`/`equal`/`exist`/`wholes`/`parts`/`query`/`quantize`), beam chain `is_part`, `materialize`, `evaluate`/posture | `bin/reasoning.py` `TruthGroundedReasoner` |
| Soft policy: MLP query-head generator + `where_read` recurrence (`prev_r`) | `reasoning.py:515,303,538` |
| Answer loss / proof-score primitives | `reasoning.py:558,567` |
| Typed truth-space (INPUT/STM/LTM/PART/WHOLE/SYMBOL) the generator reads | `_addressable_spaces` `Models.py:8089` |
| GlobalAttention parked read + trainable `consume` hook | `Models.py:8207, 8893` |
| Gates read but inert | `query_reasoning` `Models.py:803`; `answer_loss_weight` `Models.py:5250` |
| IR decode (nearest-codebook lexicon realize) reusable for sentences | `_infer_ir` `Models.py:2547`; `codebook.wv.most_similar` `Models.py:2589` |
| Grammatical idea-decode realizer (tall-WS; INERT/identity on compact) | `_idea_decode_drive` `Models.py:8497`; `decode_to_concept` `Spaces.py:17663` |

The architecture plan's §6 lists the gaps as: **the policy that calls the tools**
(the recurrent driver), **the top-K beam wired to the generator**, **the answer
(policy) loss hook**, and **the serve/infer dispatcher**. This plan builds those
four plus the **sentence-decode of the chain**.

## Phases (each gated, full-suite-green before the next)

### A — the `<reasoningIterations>` knob (one switch)
- `data/model.xsd`: add `<reasoningIterations>` (`xs:nonNegativeInteger`,
  `minOccurs="0"`) under `<architecture>` near `queryReasoning` (xsd:197). Doc:
  "N intervening-idea iterations = N reasoning steps = max chain depth leading to
  the answer; absent ⇒ reasoning off (byte-identical); 10 = recommended on-value."
- `Models.py` (~:803): `self.reasoning_iterations =
  int(TheXMLConfig.get("architecture.reasoningIterations", default=0) or 0)`;
  set `self.query_reasoning = self.reasoning_iterations > 0` (fold the old gate).
  Keep `<queryReasoning>` parsing as a deprecated alias that, if present, sets
  iterations to 10 — so old configs/tests don't break.
- Tests: config round-trip (absent⇒0/off, `10`⇒on); gate-off byte-identity.

### B — the live reasoning driver (the NeuralToolUser policy)
`BaseModel.reason_about(query_spec) -> ReasoningResult`:
- Lazily own a persistent `InterveningIdeaGenerator(dim=Dc)` (Dc = concept
  content width) + `TruthGroundedReasoner(self)`; built only when
  `reasoning_iterations > 0`.
- Loop `for t in range(self.reasoning_iterations)`:
  1. `spaces, _ = self._addressable_spaces(prevCS_forSS, ps_stage0)` (reuse).
  2. `res = generator.propose(A, B, spaces, ga=self.global_attention,
     prev_r=r, top_k=beam)` → soft idea `Mₜ` + top-K candidate beam (each with
     α = subsymbolic relevance) + typed `.where` provenance.
  3. **Hard verify** the hop via `reasoner` (`part(A,Mₜ)≥θ ∧ part(Mₜ,B)≥θ`, or a
     stored relation fire). Accept ⇒ push `(Mₜ, α, where, trust)` onto the chain
     and advance the frontier `A←Mₜ`; reject ⇒ keep `Mₜ` as a low-relevance
     "idea about the response" but do not advance.
  4. `reasoner.materialize` the verified hop to LTM (lemma write-back).
  5. Recur `r←res['idea']`; early-exit when `part(frontier,B)≥θ`.
- Return `{posture, confidence, ideas:[(Mₜ,α,where,trust)…], chain, trace}` via
  `reasoner.evaluate` for the posture and the accumulated ideas for the output.
- Tests (synthetic typed spaces, mirror `test_truth_grounded_reasoning.py`):
  generator top-K contains the right intermediate above distractors; recurrence
  carries `r`; beam cap honored; 10-iteration loop terminates on a cyclic store;
  Socrates syllogism resolves through the LIVE loop (xfail until word identity).

### C — train the soft route (answer loss)
- Add an `answer_loss(predicted_signed, gold)` term to the loss assembly,
  weighted by `answer_loss_weight` (Models.py:5250). `predicted_signed` carries
  gradient through the generator head + the GlobalAttention α only (the
  `<globalAttentionConsume>` slot, Models.py:8893, is the trainable hook); the
  hard verdict only *gates which route scored* and is never differentiated (§0
  of the architecture plan).
- Tests: weight 0 ⇒ identical; weight>0 ⇒ loss decreases on a toy QA set; no
  gradient path touches a hard verdict.

### D — sentence decode (the N-sentence output, subsymbolic-relevance ranked)
Realize the chain's ideas to text. **Two tiers** (Tier 1 unblocks now; Tier 2 is
the Track-1 follow-on):
- **Tier 1 — stored/known ideas → surface lookup (works today).** Most chain
  ideas are stored rows (relation endpoints / codebook prototypes recalled via
  `query()`/`where_read`). Realize each via the IR decode path
  (`codebook.wv.most_similar`, `Models.py:2589`) over the idea's operands →
  "np1 vp np2" surface. Rank the N ideas by their α (subsymbolic relevancy) and
  emit the top-N as sentences. Gives real sentences for the syllogism corpus.
- **Tier 2 — novel synthesized ideas → grammatical generate (Track-1 dep).**
  For ideas with no near stored surface (the `synthesize_over_set` step-3 path),
  route through the grammatical idea-decoder (`decode_to_concept`) on a tall-WS
  config. This depends on the decode round-trip (partition-aware σ/π split +
  symbol→concept expander) from the other track; until then it falls back to
  Tier 1 nearest-idea realize. **Flag, don't silently truncate.**
- Tests: syllogism chain renders ≥1 faithful sentence per stored hop; α ranking
  orders the N ideas by relevance; Tier-2 fallback path is exercised + logged.

### E — serve/infer dispatcher + exemplar config
- `serve.py` `chat_completions` (:184) / a Models.py `infer`-side branch: when
  `reasoning_iterations > 0` and the input parses as a `query="true"` rule, build
  a `QuerySpec` from the chart operands (Phase-0 framing exists,
  `QuerySpec.from_surface`; the chart currently discards the operand vectors —
  package them). Route `QuerySpec → reason_about → N sentences (+ posture)`;
  attach to the response (serve already surfaces `_last_truth_assessment` :269).
  Non-query input ⇒ generative path unchanged (byte-identical).
- `data/MM_query_reasoning.xml`: small config — `<reasoningIterations>10`,
  `<ltmConsolidation>true`, `<globalAttention>true`, `<globalAttentionConsume>true`,
  `<answerLossWeight>`>0, `<truthSet>` with the syllogism axioms (`man ⊑ mortal`,
  `Socrates : man`).
- Validation run: axiom ⇒ `isTrue`; `Socrates ⊑ mortal` via the live chain;
  materialized lemma ⇒ 2nd query is a DIRECT hit; the 10 ideas decode to
  sentences; answer loss decreases on the toy QA set.

### Deferred (out of scope)
- `verify_relation` episodic grounding (Workstream G — no source/owner).
- Tier-2 fluent novel-sentence decode beyond what Track 1 (decode round-trip)
  delivers.

## Risks
- **Don't soften the deduction** (§0): gradients only through the policy α/head.
- **Byte-identical off**: absent `<reasoningIterations>` must change nothing —
  full suite green at every phase.
- **Decode coupling**: the N-sentence output is only as fluent as the idea
  decoder; Tier 1 (surface lookup) keeps it from blocking on Track 1.
- **Termination**: beam + visited-set + `max_steps = reasoning_iterations` +
  the `_creates_cycle` antisymmetry guard.
