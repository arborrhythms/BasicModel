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
2. **Commit the better derivation.** "Outperforms" means a strictly lower
   training loss on the same sentence under the same objective; no label is
   involved. When explore's loss is strictly lower, its program, STM, LTM row
   and observations are committed in place of exploit's; otherwise exploit's
   are. Ties go to exploit. The committed derivation is what the discourse
   chain holds and what the inter-sentence predictor trains on.
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
6. **Throughput.** The receipt must break the per-batch cost into exploit
   forward, explore forward, the two backwards, snapshot and restore. Two
   derivations are the minimum for a guaranteed alternative; the deep runtime
   snapshot and the second full batch forward are not inherent to them.
   Candidate restructuring (decision after the breakdown): explore re-derives
   from the cached pre-compose word vectors, and writes to scratch STM/trace
   copies so the commit is a swap.

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
3. **Implement decision 2**, commit the better derivation: keep both trials'
   end states until their same-sentence training losses are compared; commit
   explore's program, STM, LTM row and observations only when its loss is
   strictly lower; ties go to exploit.
4. **Mechanism tests to add:** (a) at temperature 0, explore's actions equal
   exploit's before the forced round and differ at it; (b) a lower explore loss
   commits explore's program and a higher one does not; (c) evaluation with
   the same input yields the same program twice.
5. **Receipt additions:** the per-batch cost split (exploit forward, explore
   forward, two backwards, snapshot, restore) per decision 6; the parity and
   serial reconstruction measurements re-issued on the corrected source; the
   schema element and Params.md / Language.md / README updated where they
   describe the 90/10 mix.
6. **Then** one source-matched full sweep and stop for review. Keep the depth-3
   campaign, both XOR_grammar CLI failures and the historical MM result
   visible as they are.
