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

Status 2026-09-12: slice 1 landed (`<reconstructInLoop>`, the traversal
as a second `torch.while_loop` after the final seal inside the one
compiled sentence graph, idea-level cost, truncation flag, tests in
`test/test_reverse_traversal.py`); a three-brick training smoke on the
ladder text config with `<reconstructInLoop>` (CPU, tensor word pipeline)
trained through the traversal's cost with gradient (0.124, 0.021, 0.014),
the detached student off. Slice 2 landed the same day: the byte-level
cost through the retained references (soft assignment to the sentence's
word rows, cross-entropy on the words' bytes), one traversal per packed
sentence from the live sealed roots with per-sentence costs, and the
checkpoint handling of the retired student. Slice 3 landed the same day
as the generate walk (`<outputInLoop>`, `_output_generate_walk`, the
second compiled call in `reverseOutput`) with the output gates of mixed
per-row lengths, truncation, fullgraph parity and invariance to
reconstruction-only state (`test/test_output_walk.py`); the learned
generate policy and the single compiled segment of contract 6 remain open
with slice 4. The first production-width run of the traversal ran out of
accelerator memory: the loop's autograd stacks every carried tensor once
per step, so the traversal now scores each popped word on the spot and
carries only running sums (the recovered-idea slab is a test diagnostic),
the tied inverses are computed once per traversal, and the pushed ideas
are no longer a carried bank (the retained references, a loop constant,
are the idea-level target). One traversal per sentence over a row-width
schedule cost a sentence count times the forward's loop steps; the
reconstruction is now two passes sharing the forward's word index: the
seal un-folds per sentence slot (a `torch.while_loop` with a tensor
bound; a host sentence count specialised the graph per brick), then one
`torch.while_loop` over the words (latest first) that switches to a
sentence's pre-seal stack at the word ending it. Measured on
`data/BasicModel.xml` at B = 4 the reconstruction adds about 8 % to a
training batch (Training, "Reconstruction objectives");
`BASICMODEL_RECON_PLACEMENT` compares the in-graph placement with a
separate compiled call and eager execution.

Generate policy (2026-09-13): `LanguageSpace.generate_policy` (contract 5)
decides unstamped tops in the walk and is credited by imitation of the
stamped rules (`<outputPolicyWeight>`, recorded as `output_policy`); tests
in `test/test_output_walk.py`. The learned policy is no longer open.

Tying gate (2026-09-13): `test_tied_traversal_trains_the_fold_parameters_and_owns_none`
asserts the reconstruction cost's gradient reaches the lift/lower inner
layers and that no reverse-student parameter exists. Writing it exposed
a `torch.while_loop` autograd defect (present in the 2.15 nightly too):
carries entering a loop without grad cut the gradient chain across trips,
in the forward word loop as much as in the traversal. Every loop's
carries now pass through `Models._carries_with_grad`
(`test/test_while_loop_gradients.py`; Training, "Loop gradients"), and
the loop's gradients equal a Python loop's on every parameter.

Fidelity gate against the eager un-fold (2026-09-13): not definable on
the ladder fixtures. The eager un-fold (`_reverse_reduce_unfold`) raises
`NotImplementedError` on the ops it declares to have no faithful reverse
(`part`, and the other set ops), which the recorded seals choose on
"12 plus 1"; the traversal follows requirement 2 and reverses those ops
through identity while scoring the result. Parity therefore holds only on
the invertible ops (the lift/lower round-trip and chunk-residual tests),
and the byte cost is the sentence fidelity measure. Alec retired the eager
un-fold (2026-09-13): `_reverse_reduce_unfold` and the exact leaves
teacher are gone; the traversal now starts from the sentence's end state
(the top three STM slots and depth, the three LTM slots of a relative
sentence) and serves evaluation too (`_recovered_word_ideas`).

Output loop operand, resolved (2026-09-14): the answer-materialisation
boundary `_materialize_answer_idea` (Training, "The
answer-materialisation boundary") builds the resolved answer as its own
conceptual idea, the operand of the walk; the walk runs on opaque
concept slots, follows and imitates the teacher actions of the idea's
own sentence's derivation, and the words are realised through the
reverse chain. Open there: a concept-level predictor for `future`
answers (the discourse predictor works on pooled reps).

Output loop operand (2026-09-13, was open): the walk un-reduces with the CS
grammar ops' tied inverses, which act on the muxed concept width (1032 in
`data/BasicModel.xml`), while the resolved answer (`_resolve_answer`) is a
symbol-space event (136 wide) that `ConceptualSpace.synthesize` maps to
concepts by the WholeSpace inverse, never through the grammar. The ladder
fixtures have equal widths, which hid this. `reverseOutput` now skips the
walk when the widths differ; which event the output loop should realise
in production (the answer's concept-level idea, or the symbol through
`SymbolSpace.generate`'s un-reduce) is Alec's call.

Codex's second review (2026-09-14) and the fixes: (1) the byte cost was
degenerate for one-word sentences (one candidate scores zero for any
idea) and (5) packed neighbours entered a sentence's candidates: a null
candidate and a per-sentence scope. (2) The tied contract still ran the
legacy per-word reverse and the training `reverseReconstruct`: both are
skipped. (3) The seal un-fold put the reference on the wrong operand for
seals after the first: the older word is the left operand, `lo + k`;
the third review found the cursor still advancing over unrecorded seal
levels, fixed by advancing only on applied steps (a synthetic chunk
chain now unwinds exactly). (4)
The walk never popped completed constituents: it now emits them and
continues until no slot is pending; the production width mismatch stays
open (above). (6) The checkpoint release touched every loop node: it now
walks only the previous brick's own graphs. (7) Timings re-measured on
the corrected tree (Benchmarks). The reviewer's probes also confirm what
the traversal cannot do: where a derivation used ops declared lossy or a
balanced split, the recovered ideas are not the words.

Codex's fourth review (2026-09-14, at 0ef5ff4) and the answers. (1)
"Answer materialisation substitutes the current input state": for the
present relation the resolved answer IS the sentence just understood,
and its conceptual idea is that sentence's end state, keyed on the
derivation's SOURCE (the current end state, the recalled sentence's end
state `k` back, no concept predictor for `future`); a recalled row takes
the recalled sentence's idea and its derivation, not the current
input's. The reviewer's probe (editing the resolved symbol tensor
without moving the idea) stands, and the reason is the shared index
(Alec, 2026-09-14; Architecture, "a × the row-aligned identity row"):
the symbol table and the concept table share row indices, one symbol
per concept, so the symbol-to-concept inverse is the INDEX (a slot that
is a symbol maps to the concept dictionary row at its row; an edited or
novel symbol snaps to the nearest row of the row-aligned symbol table),
while a composite slot (the folded root, row -1) has no row and is
materialised by its derivation over row-aligned leaves, which is the end
state. Materialisation is therefore index-driven (Alec, "Go", 2026-09-14): the
leaves are the dictionary rows at the answer's symbol rows scaled by the
symbols' activations, folded by the recorded derivation through the
grammar's forward ops (`_replay_program`); the replay reproduces the
forward's end state exactly and follows an exchanged row
(`test_materialised_idea_follows_the_symbol_rows`); no symbol vector is
ever snapped, because every symbol carries its row. Two defects surfaced
on the way: the reconstruction's retained references had been the
unscaled WORD atoms while the forward folds each word's OBJECT atom
scaled by its activation (the references are now the pushed leaves,
`_pushed_word_slab`), and the operand-row routing had matched concept
ids instead of symbol rows (`_word_symbol_rows`); the byte snap now
ignores the sign of a leaf (a symbol's value is its signed activation).
Note that on the production geometry the understanding's conceptual
state is None and the resolved symbol is the SS activation view, which
the forward's own comments describe as the same for every sentence at
initialisation. (2) The question conditioner is now one module per answer
width (`question_conditioners`, kept in the state dict), so switching
between the symbol width and the concept width discards nothing
(`test_question_conditioners_persist_per_answer_width`). (3) The walk's
chooser on unstamped conceptual slots is credited by the teacher actions
of the idea's own derivation (`_derivation_targets`: the recorded seals
last applied first, then per word its unary, post-binary, pop and
pre-binary), followed under teacher forcing and scored by cross-entropy;
the walk's operand is the idea with its live slots reversed (the walk's
top is its last live slot) on a stack of the STM capacity, with a static
budget of every word's pop and three folds plus the seals
(`test_generate_policy_is_credited_by_the_derivation_on_conceptual_slots`).
(4) Compound operands: the trace now records each binary fold's operand
concept rows (left = STM slot 1, right = slot 0, before the reduce moves
the stack; compiled bank entries 15 and 16, eager `record_choice`), and
both reconstruction passes route the residual reverse to the operand
that is a word of the sentence, whichever side it is on; a fold without
rows keeps the former fallback (`(a+b)+c` unwinds to `[a, b, c]`,
`test_seal_chain_of_chunks_unwinds_to_the_words`). (5) The snapshot's
bytes are staged after the brick's concept rows, so the byte decoder is
active from the first brick (`test_snapshot_bytes_are_staged_on_the_first_brick`);
its candidates are the dictionary rows, never the input's staged bytes.

Codex's fifth review (2026-09-14, at cf0daf7) was executed by Codex and
reviewed (basicmodel 8ea1e12, 2026-09-15): [answer path: ownership,
training and independence](2026-09-14-answer-path-ownership-and-training.md)
(the chooser trains from supplied answer error; the materialisation reads
the derivation's owned program; no output teacher in any mode; seal
operands recorded before the reduction; per-width conditioners reload;
the snapshot's bytes come from WORD-owned surfaces; the answer path runs
at native widths). Answer modules now train only from supplied answers.

Contract 6 reconciled (2026-09-13): the forward loop and both
reconstruction passes are one compiled segment; the output walk is the
second compiled call, because the answer it realises is resolved after
the forward (`_resolve_answer` reads the understanding and the question
eagerly), so no data for the walk exists inside the forward's graph. This
is the two-call structure Alec described ("Output() ... the second
compiled call"); a single segment would require the answer resolution to
become a tensor step of the forward, which is not in this plan.

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
