# Compiled reverse loops: reconstruct the completed sentence in one bounded compiled traversal, output as the second loop

Status: plan, 2026-09-12, revised the same day after Codex's review and
Alec's decisions (recorded in the
[sentence-boundary thinking specification, section 6](../specs/2026-09-11-sentence-boundary-thinking.md#6-code-review-compiled-reverse-loops-2026-09-12));
those decisions are the requirements below. Nothing here is implemented or
benchmarked yet.

Alec's direction: the reconstruct must be a compiled loop, and
`reverseOutput` emits a sentence of its own length, so it is the second
compiled loop; forward, `reverseReconstruct` and `reverseOutput` may form one
compiled segment, the latter two independent of each other. The first draft
of this plan proposed reversing each fold at the forward's word index; that is
withdrawn (decision 1 below): immediate per-word recovery cannot establish
that later folds and seals retained the earlier input.

## Where we are (verified 2026-09-12)

- The forward is one `torch.while_loop` per word bucket
  (`TensorPeerWhilePipeline.run_cs_lanes_banked`, called from
  `_run_tensor_peer_word_pipeline`), traced fullgraph into one graph with the
  final seal and the head; its backward is the loop's autograd.
- Training does **not** run `reverseReconstruct`. With `<detachedReverse>`
  (on in `data/BasicModel.xml`, with `<teacherReconstruction>`), `lossIn` is
  `_detached_reverse_construction_loss()`: the idea-only student
  `ReverseConstructionChooser` (its own parameters: an idea projection, a
  per-step slot embedding, a kind head and a rule head) predicts the
  ReconstructionStack's *detached* arity / rule / leaf targets from a
  *detached* root idea. No gradient reaches the forward through it, and the
  training call to `reverseReconstruct` is deduplicated away
  (`_rev_dedupe`). The trace-driven un-fold (`_reverse_reduce_unfold`, the
  stage walk `_reverse_body`, the `@torch.compiler.disable` reverse islands)
  is an evaluation path.
- `reverseOutput` (with `answerSynthesis` and questions) is an eager walk
  seeded by the resolved answer.
- The compiled forward already carries per-word loss accumulators in the
  loop (`capture_intra`), publishes its results as explicit outputs, and
  obeys the compile discipline learned on the fold-ladder path: fixed shapes
  for loop constants, eager allocation on the loop's device before the first
  compile, in-place commits of live state, bounded chart backward.

## Requirements (Alec's decisions, 2026-09-12)

1. **Reconstruct the completed sentence.** `reverseReconstruct` starts from
   the completed, sealed structured representation and its permitted
   evidence: the retained compose derivation and the ordered, role-labelled
   constituent references of specification section 2.1. It does not replace
   end-to-end sentence reconstruction with local per-word recovery, and the
   seals are inside the fidelity check. Input targets are used for scoring
   only, never as reconstruction inputs.
2. **Not an exact inverse; tied weights.** The objective is input
   reconstruction, not recovery of the original operands to rounding. The
   reverse uses the compose path's shared learned transforms, with LDU-based
   inversion of their invertible linear components (`compute_Winverse_current`
   already exists on the inner layers), and no independently parameterised
   decoder. This is the existing `invertible=True` forward/reverse pairing
   (Alec, 2026-09-12), not new inverse math: `lift` and `lower` reverse
   through `SigmaLayer.generate` / `PiLayer.generate` with the same `W`,
   `chunk` through its exact residual, `not` and `non` through their own
   inverses, while `intersection`, `union` and `product` are declared lossy
   and reverse through the set helpers or identity. What changes on that
   path is only that the loss always expects the reconstruction: the cost
   is scored against the input every time the traversal runs, replacing the
   detached student's target loss. LDU inversion does not make a lossy merge
   bijective; a balanced split recomposes to the parent without recovering
   its children, and the retained structure supplies the disambiguating
   evidence. Sentence fidelity is measured separately from linear inverse
   accuracy.
3. **Once per completed sentence, bounded.** One bounded compiled traversal
   per sentence, after that sentence's composition and seals have completed:
   inside the word loop at an intermediate end for packed rows, after the
   loop for a row's final sentence, each with its own boundary and loss
   accounting. The traversal follows the retained derivation (the recorded
   choice per step), so it evaluates one reverse operator per step, not all
   `R` at every word; basis searches (the PEEL's best row) are bounded to a
   declared candidate count; the traversal has a declared step limit.
   Recurrent thinking does not reconstruct the external input again at
   every thought boundary.
4. **Three segregated paths.** `forward()` composes; `reverseReconstruct()`
   follows that input's derivation with tied inverse transforms;
   `reverseOutput()` realises an already-resolved idea through the grammar's
   generate rules with its own policy, traversal state and termination
   (per-row pending constituents exhausted under the declared
   end-of-sentence rule, else a reported bounded truncation). Output cannot
   read input-reconstruction witnesses, and the forward chooser's
   `score_binary` on `(parent, parent)` is not a generate policy. Boundary
   queries resolve before output realisation. Kernels may be shared;
   policies, seeds, state and losses are not.
5. **Declared migration of the learning contract.** Replacing the detached
   student with tied reconstruction changes what is learned. Before the
   detached path retires, document the objective, parameter ownership, the
   gradient and stop-gradient boundaries (which forward parameters the
   reconstruction loss may move, and where the forward is detached) and
   the checkpoint migration (the student's parameters leave the state
   dict; loading an older checkpoint must be defined). Hard selections and
   detached helpers get explicit credit rules with tensor implementations.
6. **Evidence-based performance.** A reproducible baseline names the commit,
   configuration overrides, device and backend, executed losses and
   workload; measures warmed full training steps (forward, backward,
   optimizer) with compile time, recompilations and peak memory; varies
   sentence and output lengths, batch size and basis size; and reports
   fidelity and the declared parameter gradients with the speed. The
   `backend="eager"` graph-capture tests establish one-graph structure,
   not throughput. The concept dictionary can change during contextual
   updates: a traversal reads one snapshot with an active-row mask and a
   version, distinguishes value updates from storage or shape changes, and
   keeps the values its backward needs.

## Contracts

1. **Derivation as tensors.** The forward loop already records, per step,
   the chosen rule, arity and validity (`_choice_rule_ids`,
   `_choice_arities`, `_choice_mask`, three slots per word plus the seal
   slots). Reconstruction adds, per recorded step, the operand identity it
   needs to follow the derivation: which STM slots the fold consumed and
   the constituent reference each slot carried (the whole/unit/clause slab
   and the concept-row slab already ride the loop). Nothing is retained
   that the forward did not compute.
2. **Sentence traversal.** `LanguageSpace.reconstruct_sentence(root, trace,
   basis_snapshot, budget)` is a bounded fixed-shape traversal of the
   recorded derivation from the sealed root backward: at each recorded step
   the recorded op's tied inverse (`generate` for lift and lower through the
   LDU inverse; the PEEL with a bounded candidate set for chunk; the set
   reverses for union and intersection; the explicit inverse for not and
   non; identity where the op is declared non-invertible) yields the
   operands, guided by the retained constituent references where the
   inverse is ambiguous. The traversal writes recovered word ideas into a
   `[B, W, D]` slab at the recorded word positions; unvisited positions stay
   masked. Steps beyond the budget report truncation in a per-row flag.
3. **Descent and score.** `PartSpace.reverse_words` and
   `InputSpace.reverse_words` descend the recovered word ideas in fixed
   shape (rung codes to atoms to byte logits over each word's byte window,
   from the existing balanced split and byte snap without the host loop).
   The sentence fidelity cost is a byte cross-entropy against the staged
   input bytes; the idea-level cost against the pushed word ideas is a
   diagnostic, reported separately (requirement 2).
4. **Placement.** For packed rows the traversal runs inside the word loop at
   each intermediate end on the sealed root of that sentence (the loop
   already knows the boundary); for the final sentence it runs after the
   loop on the published root. Both are the same function; both write into
   the same per-row accumulators (`recon_loss_sum`, `recon_loss_weight`,
   `recon_truncated`) that leave the compiled segment as explicit outputs.
5. **Output loop.** `TensorPeerWhilePipeline.run_output_loop(seed, width,
   policy, budget)` realises the resolved answer's structure: its state is
   the answer-side pending-constituent stack, its policy is the generate
   policy (a learned chooser over the grammar's generate rules, initialised
   from the resolved derivation when one exists), and it terminates when
   every row's pending constituents are exhausted or the width is reached
   (reported). It reads nothing of the input reconstruction. Its emitted
   words, trace and costs are explicit outputs.
6. **One compiled segment.** `_forward_with_compiled_sentence_state` runs
   the forward loop (with in-loop reconstructions at intermediate ends),
   the final-sentence reconstruction, and, when questions are present, the
   output loop, in one compiled callable. The two reverse paths depend only
   on the published root and the answer, so the compiler may schedule them
   apart; actual data dependencies are preserved as is.
7. **Migration.** `<detachedReverse>` keeps selecting the student until the
   gates pass; `<reconstructInLoop>` selects the tied traversal; both
   cannot be on. The student's parameters are dropped from checkpoints
   under the tied contract, and loading an older checkpoint ignores them.
   The gradient boundary of the tied contract: the reconstruction cost
   trains the tied transforms and the forward's fold parameters, and is
   stopped at the recorded discrete choices (credited through the existing
   policy credit) and at the constituent references (indices).
8. **Legacy.** The trace-driven eager un-fold and the reverse islands remain
   reachable for the parity gate, then go.

## Gates

- Fidelity: on the ladder and idiom fixtures, sentence reconstruction
  through the traversal scores no worse than the eager un-fold on the same
  model, including one-word sentences, words that were never folded before
  the seal, unary rewrites and the final seal, and a constructed case where
  local recovery succeeds but the completed state has lost the word.
- Tying: a test asserts the traversal's parameters are exactly the compose
  path's (no new parameter besides the generate policy of the output loop)
  and that the declared gradient paths and stop points hold.
- Structure: the complete forward with in-loop reconstruction traces to one
  unique graph across two runtime lengths (extend the existing one-graph
  test); packed rows keep separate sentence accounting.
- Output: mixed output lengths, several words, output longer than the input,
  and invariance to changes in reconstruction-only state; the resolved
  answers of the successor corpus match the eager path on a fixed model.
- Performance: the protocol of requirement 6 on `data/BasicModel.xml` with
  the production backend, reported in the throughput document beside the
  pre-ladder baseline.

## Slices

1. Derivation tensors and the tied sentence traversal at the idea level
   (contracts 1, 2, 7 without the descent), run after the loop on the
   published root; fidelity and tying tests against the eager un-fold.
2. The descent and the byte cost (contract 3); placement inside the loop at
   intermediate ends (contract 4); the one-graph gate; the migration
   document and checkpoint handling (requirement 5).
3. The output loop and its generate policy (contract 5); the single compiled
   segment (contract 6); output gates.
4. Performance protocol and report (requirement 6); Legacy removal.

With implementation: reconcile this plan, update Architecture and Language
for loop and parameter ownership, Training and Params for objectives,
credit and migration, and the throughput report with the measured
configuration; extend the specification's section 4 isolation and
structural tests with the fidelity, tying, gradient, packed-row and
production-backend gates before claiming completion.
