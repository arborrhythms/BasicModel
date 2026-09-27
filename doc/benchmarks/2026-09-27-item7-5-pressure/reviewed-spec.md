# 7.5. One operation per round

Spec by Claude, supplied by Alec on September 26, 2026. Code by Codex; Claude
reviews before commit under the repository publish rule.

The parallel compose path is the same object as the serial STM reducer, thought
chooser and generate policy: one softmax over every candidate operation at every
location of the current sequence. Every binary operator is offered at every
adjacent pair; every unary operator at every position. Exactly one operation
fires in an active round. Binary shortens the sequence by one; unary does not.

There is no no-operation candidate while the sequence is longer than its LTM
row's slots. STOP becomes eligible once the sentence fits its row. Rounds use a
fixed budget with early stop, preserving a static compiled shape.

Every sentence yields two hard derivations. Exploit chooses from the tempered
softmax, mainly argmax with item 8's structural rule on exact ties. It is committed
to the program, STM and LTM. Explore is one sampled path that must differ: choose
a round uniformly from those exploit used and mask exploit's choice there; sample
freely at all other rounds. Explore trains and is not recorded. It replaces the
flattened-temperature second pass and approximates the full superposition over
all operators and locations.

Both derivations keep the chooser in the gradient path by the existing
straight-through pattern, weighting the chosen candidate by its softmax
probability. Compose has no advantage or policy term and no DP of any kind.
Delete the forward/backward and Viterbi tiling routines, tile compaction and
length-DP bookkeeping, DP-prior regularizer, flattened-temperature pass and
separate per-position unary layer. The STM reducer becomes this layer with a
two-slot window. Do not leave gated legacy alternatives.

Reconstruction parity numbers will move: record them without tuning. Run the
XOR_grammar and MM gates unchanged and keep all outcomes visible.

Alec's reasoning, for the record: training by softmax lets one gradient do all
the optimization, and DP is the pre-backprop symbolic solution that should not
be mixed into a working MLP.

Implementation choices made explicit for review: exploit is 90% argmax and 10%
sampling; an exhausted budget that still exceeds row capacity trains as an
incomplete forest and is not committed to memory. Parallel rounds use at least
2N; serial has three rounds per word and at most 2K at the seal (K is STM
capacity), capped by a positive syntacticOrder. Degenerate catalogs that cannot
supply a distinct legal exploration choice fail explicitly. These choices have
not been selected against reconstruction or learning scores.

## Review decisions (Alec, 2026-09-27) — these supersede the "Implementation choices" paragraph above

1. **Two optimizer steps per sentence stand.** Exploit runs forward, loss,
   backward and step; explore then runs its own forward, loss, backward and
   step on the updated parameters. The batch clock and training-step counter
   advance once.
2. **Commit the better derivation, per sentence, at its seal** (Alec,
   2026-09-27: per sentence is the better long-term choice, so it is done
   now, not after a cost breakdown). "Outperforms" means a strictly lower
   training loss on that sentence under the same objective; no label is
   involved. The transaction is per sentence, at the eager sentence
   boundary between compiled word bricks: exploit derives sentence *k* and
   seals; its cost for *k* is computed there (the reconstruction traversal
   for that sentence and its prediction term); the runtime state is then
   restored to the committed context after *k − 1*; explore re-derives
   sentence *k* from the cached word vectors — perception runs once, only
   the compose rounds differ — and its cost for *k* is computed the same
   way; the winner's program, STM, LTM row and observations are committed,
   and sentence *k + 1* proceeds from that committed context. Ties go to
   exploit. Within a packed row this makes every later sentence consistent
   with what was actually committed (expectation image, recency buffer,
   minted identities); choosing per batch, or mixing winners inside a row
   under whole-row trials, is excluded. The committed derivation is what
   the discourse chain holds and what the inter-sentence predictor trains
   on. Consequences Codex must implement: the per-sentence reconstruction
   traversal moves to the seal (the in-loop placement the reverse-loop plan
   already provides), the inter-sentence prediction term is kept per row
   and per sentence rather than as one batch scalar, and the two optimizer
   steps of decision 1 happen at each seal for that sentence index across
   rows; terms that exist only at batch end (the teacher's answer loss)
   keep their existing backward and do not enter the comparison.
3. **Evaluation runs exploit only.** No explore derivation, no optimizer, and a
   deterministic choice: argmax with item 8's structural tie rule.
4. **Temperature is a parameter.** One `model.xml` element (name indicative:
   `composeTemperature`, ≥ 0, default 0) tempers the *sampling* of exploit's
   hard choice; 0 is argmax. The 90% / 10% mixture is removed. The temperature
   never enters the probabilities used for the straight-through weight or for
   credit: those stay the model's own softmax, otherwise temperature 0 would
   saturate the softmax and the chooser would receive no gradient. **Explore
   uses the same temperature as exploit** (Alec, 2026-09-27). At 0 the explore
   is therefore exploit's nearest counterfactual: identical up to the
   uniformly chosen forced round, the best alternative there, then argmax
   from the changed state onward. Because the committed derivation is the
   better of the two (decision 2), this neighbour wins the comparison far
   more often than a free sample would, and when it wins, committing it
   moves the argmax to the better neighbour — a hill climb in derivation
   space under the same gradient. Over sentences the uniform forced round
   covers every first-order neighbour; deeper changes become reachable once a
   first change is adopted. A free sample at temperature 1 (Claude's earlier
   recommendation, withdrawn) diverges early into a low-probability path
   that rarely wins and whose probability weight makes its gradient small.
   Raising the shared temperature widens the neighbourhood if it ever proves
   too narrow.
5. **The tie rule applies to logits, not probabilities** (item 8 contract: no
   epsilon overturns a better score). The test that lets a logit better by
   1e-8 lose to float32 rounding of the probabilities is withdrawn.
6. **Throughput.** Two derivations are the minimum for a guaranteed
   alternative; the deep whole-batch snapshot and the second full batch
   forward are not inherent to them. The restructuring is decided now, as
   part of decision 2: explore re-derives each sentence from the cached
   pre-compose word vectors, and the per-sentence transaction writes to
   scratch STM/trace copies so the commit is a swap. The receipt still
   reports the per-batch cost split (exploit compose, explore compose, the
   backwards, snapshot and restore) so the effect of the restructuring is
   visible; it no longer gates anything.

For the record: the deleted `<learning>` two-pass driver was pass A over the
tiling DP plus pass B as a flattened *soft mixture*; the exploit/explore pair is
that two-pass idea rebuilt from two hard sampled derivations, with no DP and no
soft blend.

## Review findings for Codex (Claude, 2026-09-27) — resolve before commit

Sweep on the reviewed source: 4,945 cases, 4,622 passed, 321 skipped, one
failed (the depth-3 relative campaign, kept red), one expected failure.

1. **Defect: inference is stochastic.** `OperationSelectionLayer.forward` mixes
   sampling into exploit's hard choice with no training gate; the serial caller
   passes `exploit=not sample` and the parallel caller has no greedy path, so
   evaluation, the committed program and the LTM row are partly random. This is
   the most probable cause of the lost packed/single parity (.7878 vs .6836).
   Fix per decisions 3 and 4: evaluation runs exploit only, argmax with the
   item 8 tie rule; training samples exploit at `composeTemperature` (default
   0); the 90/10 mixture is removed; the temperature never enters the
   straight-through or credit probabilities; explore uses the same temperature.
2. **Defect: the item 8 tie rule is applied to probabilities.** Float32 rounding
   equates distinct logits, and `test_compose_exact_probability_tie_prefers_structure`
   enshrines a logit better by 1e-8 losing to the structural face. Apply
   `structural_argmax` to the logits; withdraw that test's claim.
3. **Implement decision 2 as amended: per sentence, at the seal.** The
   transaction moves from the batch to the eager sentence boundary; explore
   re-derives from cached word vectors; the reconstruction traversal for the
   sentence runs at its seal; the prediction term is kept per row and per
   sentence; the winner is committed before the next sentence begins. Never
   choose per batch, and never mix winners inside a row under whole-row
   trials.
4. **Mechanism tests to add:** (a) at temperature 0, explore's actions equal
   exploit's before the forced round and differ at it; (b) a lower explore loss
   commits explore's program and a higher one does not; (c) evaluation with
   the same input yields the same program twice; (d) in a packed row, the
   second sentence's derivation is conditioned on the committed first
   sentence, whichever trial won it.
5. **Receipt additions:** the per-batch cost split (exploit compose, explore
   compose, the backwards, snapshot, restore) per decision 6, informational;
   the parity and
   serial reconstruction measurements re-issued on the corrected source; the
   schema element and Params.md / Language.md / README updated where they
   describe the 90/10 mix.
6. **Then** one source-matched full sweep and stop for review. Keep the depth-3
   campaign, both XOR_grammar CLI failures and the historical MM result
   visible as they are.

## Review round 2 (Claude, 2026-09-27, on the sentence-seal corrections)

The September 27 corrections implement decisions 1–6 as amended: the shared
temperature governs sampling only and defaults to 0; the tie rule is on the
logits; evaluation is exploit-only and deterministic; the transaction is per
sentence at the eager seal with cached perception, scratch STM/trace swap and
commit before the next perception; the prediction term is per row and per
sentence; two optimizer steps run at each seal; the batch-end answer loss stays
separate. The four requested mechanism tests exist and pass. Packed/single
parity is exact again (.6496902332 both layouts); serial reconstruction
improves over training again (.1065 → .1001); warmed throughput is .919
sentences/s, above the item 8 baseline. The sweep is **not green**: ten
failures, of which eight passed on the committed baseline `206a0146`.

**Finding A — the budget can be spent without progress (structural).** Unary
operations are eligible in the seal rounds, and nothing compels a reduction.
Probe on the `test_expectation_off_keeps_every_packed_observation_in_ltm`
fixture: the first packed sentence used its entire 34-round budget on 4 binary
and 30 unary operations, ended at depth −1 (incomplete) and wrote no LTM row;
the second sentence completed. This is the common cause of the three
`test_expectation_defaults.py` failures, very likely of the output
question-conditioner failure (a multi-slot answer), and it bears on the
depth-3 campaign. It also conflicts with item 7's contract that one S writes
one row: under the current eligibility, completion is left to a chooser that
has not yet learned to reduce.
*Proposed (awaiting Alec's yes/no):* **seal rounds are reduction-only.** After
the online rounds of the last word, the only eligible candidates are binary
operations at pairs, and STOP once the sequence fits its row; unary
operations are eligible only in the online rounds after each word. The seal's
purpose is to bring the sentence to its row, and a unary rewrite cannot change
the depth. With a seal width of 2K ≥ N − 1 this guarantees every sentence
completes, so every S writes a row. It is still one softmax over the eligible
candidates: no no-operation candidate, no DP.

**Finding B — the operator-gradient report lost its reconstruction term
(reporting boundary).** Reconstruction now trains at the seal, so the
batch-end objective-agreement report sees no reconstruction graph and
`test_gradient_factorization.py::test_normal_batch_logs_named_shared_operator_gradients`
fails with every `reconstruction_norm` zero. The dissonance measurement
(per-operator gradient cosine) must aggregate the per-seal reconstruction
gradients into the report rather than read a batch-end graph that no longer
exists. Fix the report; do not waive the assertion.

**Finding C — four obsolete fixtures.** The `test_item9b_followup.py`
interleave tests assert a whole-batch explore call during evaluation, which
decision 3 removed. Update their expected call lists to the spec (native
context pass where needed, one serial exploit pass). This is a contract change
by decision, not a test tuned to pass.

**Then:** rerun the three observation tests, the conditioner test and the
depth-3 campaign on the corrected source; the campaign stays red if it stays
red. One source-matched full sweep; stop for review; keep both XOR_grammar CLI
failures and the historical MM result visible.

## Decision 7 — reduction pressure and deadlines (Alec, 2026-09-27; supersedes round 2's "reduction-only seal" proposal)

The retired design had a soft reduction pressure that rose as the number of
unreduced terms approached STM length, and a full stack of eight words forced a
reduction before a new word could be added. Item 7.5 dropped it with
`stmReduceTau` and marks an overflow incomplete instead. Restore it in the
one-softmax form, and lower the allowance toward the row's slots as the
sentence ends: every grammar reduces to NP, NP VP or S REL S, so one to three
slots depending on the derivation.

* **Allowance `a`.** During the sentence, `a = K − 1`: the next word must be
  admitted. At the seal, `a` is the row's slot count as the derivation
  decides it: an absolute S is **one** slot — NP VP fuses to one point,
  because the primitives of our reality are spacetime events and the
  sentence names one (Alec, 2026-09-27; two truths §1) — and a relative
  S REL S keeps **three**. An unfused NP VP at depth two is a transient state
  on the way to the row, never an allowance of its own.
* **Required reductions and slack.** With current depth `d`, `n = max(0, d − a)`
  reductions are still required; with `r` rounds remaining in the phase (the
  online rounds before the next word, or the seal rounds before the row is
  written), the slack is `s = r − n`.
* **Hard deadline.** When `s ≤ 0`, only binary candidates are eligible; unary
  and STOP are masked. This guarantees no overflow (three online rounds cover
  the one reduction a new word can require) and guarantees completion at the
  seal (seal width `2K ≥ K − 1`), so every S writes a row, which item 7
  assumes. STOP's existing eligibility rule (only when `d ≤ slots`) stands.
* **Soft pressure.** A fixed additive term on every binary candidate's logit,
  monotone increasing in occupancy `d / a` and in `n / r`, zero on an empty
  stack. Its weight is a `model.xml` parameter (name indicative:
  `reducePressure`). No learned parameters: it is a working-memory-load prior,
  not an operator preference. Like the existing category prior it is part of
  the model's own distribution, so it enters the credit probabilities; the
  sampling temperature does not.
* **Overflow.** Under the deadline a word never arrives at a full stack; the
  incomplete-on-overflow path remains only as an assertion, and an incomplete
  forest can arise only from an exhausted budget with `n > r` at phase start,
  which the widths above exclude.

Round 2's Finding A is resolved by this decision; Findings B and C stand.
Codex declares the pressure's monotone form and the parameter's default before
measuring, and records them in Params.md. *Claude's formalization of Alec's
description. Alec confirmed the fused NP VP reading (2026-09-27). The
"enters credit" choice stands unless Alec objects.*

**Mechanism tests for decision 7:** (a) with one online round left and the
stack at `K − 1`, unary and STOP are masked and a binary operation fires, so the
next word is admitted without overflow; (b) a sentence whose chooser prefers
unary operations still completes at the seal, and every sentence in a packed
row writes a row; (c) the pressure term is zero on an empty stack, increases
monotonically with occupancy, and changes the credit probabilities while the
sampling temperature does not; (d) the three `test_expectation_defaults.py`
cases pass without fixture changes. The allowance at the seal is one slot for
an absolute S and three for a relative S; a test asserts there is no two-slot
allowance.
