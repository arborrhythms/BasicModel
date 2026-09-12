# Compiled reverse loops: reconstruct inside the forward loop, output as the second loop

Status: plan, 2026-09-12. Alec's direction: the reconstruct must be a
compiled loop, and since it shares the loop index with the forward it should
run *inside* the forward loop; `reverseOutput` emits a sentence of its own
length, so it is the second compiled loop; forward, `reverseReconstruct` and
`reverseOutput` should form one compiled segment, the latter two independent
of each other.

## Where we are

- The forward is one `torch.while_loop` per word bucket
  (`TensorPeerWhilePipeline.run_cs_lanes_banked`, called from
  `_run_tensor_peer_word_pipeline`), traced fullgraph into one graph with the
  final seal and the head; its backward is the loop's autograd.
- `reverseReconstruct` runs eagerly in `runBatch` after the compiled call:
  it un-folds the published root by replaying the recorded reduction trace
  backward, one recorded op at a time, through each op's basis-threaded
  `reverse` (`ChunkLayer.peel`, `Ops.disjunctionReverse` /
  `conjunctionReverse` / `lowerReverseAll` / `liftReverseAll`, identity
  stubs for the rest), then walks the stages in reverse
  (`_reverse_body`: `ConceptualSpace.reverse` -> `reverse_stack` ->
  `LanguageSpace.reverse`; PartSpace and InputSpace reverses) and scores the
  reverse event against the input (`_reverse_event_loss`). The grammar
  reverse entries are `@torch.compiler.disable` islands. Roughly one eager
  reverse op per reduction; host Python between them.
- `reverseOutput` (with `answerSynthesis` and questions) is the same kind of
  eager walk seeded by the resolved answer.
- The compiled forward already carries per-word loss accumulators in the
  loop (`capture_intra`: `FunctionalPeerSTM.predict` adds a prediction loss
  per word, summed into the CSLang bank and consumed after the loop).

## Contracts

1. **Reconstruct at the shared index.** At CSLang step `w` (the word the
   post-deposit binary just folded), the loop reverses *that* fold: from the
   post-fold state and the recorded choice it recovers the two operands
   (`reverse_binary_choice`), the right operand being the reconstruction of
   word `w`'s idea, the left the reconstruction of the running parent. The
   word reconstruction then descends the perceptual and input reverses
   (idea -> part codes -> atoms -> bytes, the balanced split down the rungs
   and the byte snap) and is scored against the staged input word at index
   `w`. The per-word costs accumulate in the CSLang bank like the intra
   loss and leave the loop as explicit outputs. No un-fold of the root after
   the loop: the sentence's reconstruction *is* the sum of its per-word
   reconstructions at the index they were consumed.
2. **Functional reverses.** Every binary op the chooser can select gets a
   pure `reverse(parent, basis) -> (left, right)` of fixed shape on `[B, D]`
   (the existing tensor helpers wrapped; identity stubs stay identity). The
   loop computes all `R` reverses on the folded parent and selects by the
   chosen op's one-hot, exactly as the forward stacks `R` candidates
   (`_stacked_reduced`) and selects by `op_weights`. The concept dictionary
   (the `basis` of the PEEL and set reverses) enters the loop as a constant
   tensor; it changes only at `Reset`.
3. **Unary reverses.** The post-binary unary rewrite is reversed the same way
   (`reverse_unary_choice`) before the binary reverse, so the word-level
   reverse sees the state the binary produced.
4. **Seals.** The intermediate-end seals fold the sentence to its root inside
   the loop; their reverses are not needed for reconstruction (every word
   was reconstructed at its own index), so the seals stay forward-only.
5. **Output is the second loop.** `reverseOutput` becomes a second
   `torch.while_loop` over an output width: seeded by the resolved answer,
   each trip reverses one choice (the chooser's *reverse* decision: which
   op, from the recorded or a learned policy) and emits one word at the
   loop index; it ends at the output width or when every row has emitted a
   leaf. Its trace and losses are explicit outputs. It does not share the
   forward's index because its length differs.
6. **One compiled segment.** `_forward_with_compiled_sentence_state` runs the
   forward loop and then, when questions are present, the output loop; the
   two loops are independent (the output loop reads only the published root
   and the answer), so the compiler may schedule them apart. Everything an
   eager consumer needs is an explicit output (the existing contract; an
   attribute escape is dead state under torch 2.14).
7. **Eager paths retire to Legacy.** The trace-driven un-fold
   (`_reverse_reduce_unfold`), the stage-walk reverse for training, and the
   reverse islands remain reachable through Legacy for the parity gate, then
   go.

## Mechanics

- `LanguageSpace.reverse_binary_choice(state, choice, basis)`: stacks the
  `R` reverses of the folded slot-0 parent (`[B, R, 2, D]`), selects by
  `choice.local_op` one-hot, returns `(left, right, valid)` with `valid =
  choice.applied`. Ops' reverses used: `LiftLayer` / `LowerLayer` ->
  `_sigma.generate` / `_pi.generate` (balanced split, the chart's bounded
  backward applies), `ChunkLayer` -> `peel` against the basis (best row,
  exact remainder), `UnionLayer` / `IntersectionLayer` ->
  `Ops.disjunctionReverse` / `conjunctionReverse` with the basis, `NotLayer`
  / `NonLayer` -> their own reverses, the rest identity `(parent, parent)`.
- `ConceptualSpace.apply_binary_reverse(state, left, right, applied)`: the
  functional inverse of `apply_binary_language_choice` (slot 0 becomes
  `right`, slot 1 `left`, depth + 1) so a later step can reverse further if
  ever needed; not required for contract 1 but keeps the algebra honest.
- `PartSpace.reverse_word(idea)` and `InputSpace.reverse_word(parts)`: the
  fixed-shape per-word descent (rung codes -> atoms -> byte logits over the
  word's byte window `[B, P, 256]`), from the existing `reverse` /
  `generate` math without the host snap; the byte loss is a cross-entropy
  against the staged word bytes (`_ar_word_part_ids` / offsets give the
  window). The exact byte snap stays an eval-time decode.
- Loss plumbing: two new CSLang bank entries (`recon_loss_sum [B]`,
  `recon_loss_weight [B]`) updated per word; `runBatch` reads them as
  `lossIn` instead of calling `reverseReconstruct` when the loop ran them
  (`<reconstructInLoop>` knob, default on once the gate passes).
- Output loop: `TensorPeerWhilePipeline.run_output_loop(seed, width, ...)`
  with the same banked shape discipline; the reverse chooser is the
  existing forward chooser's scores read on the parent (`score_binary` on
  `(parent, parent)`) until a learned reverse policy exists.
- Compilation constraints already learned on this path: every loop constant
  has a fixed shape (pad to the bucket width), every slab the loop needs is
  allocated eagerly on the loop's device before the first compile, live
  state commits in place under the compiler, results are explicit outputs,
  and the charts' backward slopes are bounded.

## Gates

- Parity: on a fixed model and sentence, the in-loop per-word reverse of a
  lift/lower fold reproduces the operands to float rounding; the in-loop
  reconstruction loss equals the eager path's loss computed on the same
  per-word reverses (tensor test on the ladder fixture).
- One graph: the complete forward with in-loop reconstruction traces to one
  unique graph across two runtime lengths (extend
  `test_tensor_peer_complete_forward_is_one_graph_across_runtime_lengths`).
- Throughput: `data/BasicModel.xml` bricks with reconstruction in the loop
  are no slower than today's forward-plus-eager-reconstruct brick (13.2 s at
  B=24, 400 documents), and the eager reverse islands are gone from the
  profile.
- Output loop: `reverseOutput` on `MM_ladder`'s successor corpus produces the
  same answers as the eager path on a fixed model, in one graph.

## Slices

1. Functional reverses and the in-loop per-word reconstruction at the idea
   level (contracts 1-3 without the perceptual descent): `reverse_binary_choice`,
   `reverse_unary_choice`, the loss on the recovered word idea against the
   pushed idea, the knob, the parity and one-graph tests.
2. The perceptual and input descent inside the loop (contract 1 complete):
   byte-level loss, `runBatch` reads the loop's reconstruction, Legacy holds
   the eager un-fold.
3. The output loop (contracts 5-6) and the single compiled segment.
4. Legacy removal after the gates; benchmark table update.
