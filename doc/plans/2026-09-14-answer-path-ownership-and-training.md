# Answer path: ownership, training and independence (Codex round 5)

Status: IN PROGRESS (2026-09-15): items 4, 5, 1, 3, 6 and 2 complete and verified. Remaining
item: 7 under Alec's decisions
in section 6. Written 2026-09-14 at basicmodel `cf0daf7` for execution in
a fresh session. Alec's framing: are Codex's findings specification
issues or code that does not operate as specified? Answer, item by item,
below; in short: five of the six are code defects against a clear
specification, one (answer ownership) is a code defect plus one design
decision Alec must make. The seventh item is a finding of this session
that Codex has not yet reported.

Read this whole document before touching code. Then read the standing
invariants (section 2) again after every context compaction.

## 1. Where we are

Landed and pushed (basicmodel `cf0daf7`, WikiOracle `8289314`; the
review history is in
[the compiled reverse-loops plan](2026-09-12-compiled-reverse-loops.md),
"Codex's fourth review"):

* The forward is one `torch.while_loop` per word bucket
  (`TensorPeerWhilePipeline.run_cs_lanes_banked`, `Models.py`
  `_run_tensor_peer_word_pipeline`). Two reconstruction passes share its
  word index (`_reconstruct_sentences`): pass A un-seals each sentence
  from its end state, pass B undoes each word's unary, post-binary and
  pre-binary folds with the tied inverses and scores the popped word
  (idea distance to its retained reference, byte cross-entropy through
  the dictionary snapshot). Training and Architecture describe it
  ([Training, "Reconstruction objectives"](../Training.md)).
* The output loop (`<outputInLoop>`): `reverseOutput` materialises the
  resolved answer as its own conceptual idea (`_materialize_answer_idea`)
  and un-folds it with the generate walk (`_output_generate_walk`, the
  second compiled call), realising the emitted words through the reverse
  chain. The walk's chooser is `LanguageSpace.generate_policy`.
* Materialisation is index-driven (Alec, 2026-09-14): the symbol table
  and the concept table share row indices, so an answer is materialised
  from its SYMBOLS' rows and its DERIVATION: the dictionary rows at the
  symbol rows, scaled by the symbols' activations, folded by the recorded
  derivation through the grammar's forward ops (`_replay_program` over
  `_derivation_program`). The replay reproduces the forward's end state
  exactly (`test_materialised_idea_follows_the_symbol_rows`).

Codex, reviewing `cf0daf7`, reports six remaining issues. Each was
verified against the code on 2026-09-14 (section 3).

## 2. Standing invariants (re-read after every compaction)

These are facts of the design and of the code as of `cf0daf7`. Claims in
docs or review replies that contradict one of them are wrong until the
code is re-read. Cite the line when you rely on one.

1. **Concepts are opaque; percepts and symbols are located.** A concept
   is one full-width code (`canonical_shape("ConceptualSpace") = (0, 0)`,
   `bin/architecture.py`); percepts and symbols keep `.where`/`.when`.
   The CS grammar ops act on the muxed concept width
   ([Architecture, "Concepts are opaque"](../Architecture.md)).
2. **The symbol table and the concept table share row indices.** One
   symbol per concept; a symbol is a signed activation times the
   row-aligned identity row (`SymbolSpace.forward_concept_to_symbol`,
   `Language.py`; Architecture, "the once-built SS leg is a × the
   row-aligned identity row"). A completed word crosses into SymbolSpace
   as its row plus one activation (`SymbolSpace.commit_word_reference_slab`,
   the slabs `_word_reference_rows` / `_word_reference_activations`).
   The symbol-to-concept inverse is the index. Never snap a symbol
   vector.
3. **The row of a word's symbol is its OBJECT concept's row where the
   word has one, else its word concept's row** (`Models._word_symbol_rows`:
   `isp._ar_word_object_rows` else `isp._ar_word_concept_rows`).
   `isp._ar_word_concept_ids` are concept IDS (the allocator's `A`), not
   dictionary rows; do not route by them.
4. **The leaf the forward pushes is the symbol's activation times the
   dictionary atom of that row.** The loop pushes the word idea and then
   `FunctionalPeerSTM.resolve_top_reference` replaces the top with
   `activation × object_atom` (content) plus the word's band
   (`_run_tensor_peer_word_pipeline`, `stage_cs_lang`). The retained
   references of the reconstruction are exactly these leaves
   (`_pushed_word_slab`, from the staged atoms and the loop's activation
   slab `final[8]`).
5. **A composite (folded) STM slot carries row −1.** Rows live at the
   leaves; structure lives in the recorded derivation
   (`ReconstructionStack`: `_choice_rule_ids`, `_choice_arities`,
   `_choice_mask`, `_choice_left_rows`, `_choice_right_rows`).
6. **Trace layout.** Word `w` records pre-binary at slot `3w`, post-binary
   at `3w+1`, unary at `3w+2`. The row's LAST sentence's seals are at
   `3W + k`; an intermediate packed sentence ending at word `hi` records
   its seals at `3W + hi·seal_width + k`, `seal_width = capacity − 1`
   (pass A `body_a`, `_derivation_program`). Forward order per word:
   pre, push, post, unary; seals after the sentence's last word, `k`
   ascending. The reverse (the walk's teacher) is seals last-first, then
   per word latest-first unary, post, pop, pre.
7. **Operand orientation.** A binary fold's left operand is STM slot 1
   (older), its right operand slot 0 (newest); the reducer's window is
   `stack((left, right))` (`_FunctionalLanguageChooser.choose_binary`,
   `_stm_bounded_reduce_step`). The recorded operand rows follow this.
8. **Compiled tuple.** The published compiled result has 21 values; the
   CSLang bank indices are 0 symbol activations, 1–2 loss sums, 3
   prediction, 4–8 trace state, 9 detached roots, 10 wholes, 11–12 chunk
   proposals, 13 end-state slots `[B, slots, 3D]`, 14 end depth, 15–16
   left/right operand rows (`_publish_compiled_sentence_state`).
9. **`torch.while_loop` autograd defects and workarounds.** Carries
   entering without grad cut the chain (`_carries_with_grad`,
   `_ensure_grad_anchors`); the backward node keeps `fw_outputs`
   (`_release_loop_checkpoints` at brick entry); loop outputs must be
   cloned before the next loop; a host-int loop bound recompiles per
   distinct value (use tensor bounds); attribute-only escapes from a
   compiled graph are dropped (publish explicit outputs).
10. **Losses.** `record_loss` writes the REPORT registry only
    (`Models.record_loss` → `self.errors.add`); the trained total is
    assembled explicitly in `runBatch` (`totalLoss = ...`, about
    `Models.py:13315–13395`). A recorded loss that is not added there
    trains nothing.
11. **Optimizer.** `getOptimizer` walks `self.spaces` and takes each
    space's explicit `params` list (`Space.getParameters` returns
    `self.params`), plus a few model-level modules adopted through
    `_collect_fresh_synthesis_modules` (`add_param_group` once). An
    `nn.Module` attribute that is on no `params` list and not adopted is
    never stepped. `languageSpace` is an attribute of `symbolSpace`, not a
    member of `self.spaces`.
12. **Conventions.** Run the suite from `basicmodel/`
    (`.venv/bin/python -m pytest test -q -p no:cacheprovider`, about 30
    minutes; run it in the background and poll the log); never edit
    `bin/*.py` while pytest is in flight (inspect-based tests); commit in
    basicmodel, push, then bump the WikiOracle submodule pointer
    ("Bump basicmodel to <sha> (...)"), push; trailer
    `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never
    commit Alec's uncommitted files (`README.md`, `doc/Reasoning.md`,
    `doc/WhatSpacetimeDesign.md`, `doc/specs/2026-07-27-*`,
    `doc/specs/2026-09-09-*`, `doc/specs/2026-09-11-*`). zsh does not
    word-split an unquoted `$var`: pass pytest file lists literally.
    Fixtures: `test/test_output_walk.py::_model()` (ladder variant with
    `<outputInLoop>`), `test_meronomy_ladder._build_ladder_variant`,
    `test_compiled_word_chunk._stage_fullgraph_tensor_peer`; set
    `m._tensor_peer_while_eager = True` and
    `m._chart_compose_per_word = lambda: None` to run the loop body
    eagerly in tests.

## 3. The findings, classified

| # | Codex's finding | Kind | Verified cause |
|---|---|---|---|
| 1 | The generation chooser does not train through `runBatch` | code | the policy cost is recorded, never added to `totalLoss`; `generate_policy` is on no `params` list |
| 2 | Materialisation reads the staged input, not answer-owned structure | code + one design decision | `_materialize_answer_idea` reads live model staging, not the derivation; two conditioners condition two different objects |
| 3 | Output is teacher-forced even in evaluation | code | the walk follows `targets` whenever present |
| 4 | Intermediate packed seals record the post-reduction operands | code | `_tensor_record_operands` is called after `apply_binary_language_choice` |
| 5 | The per-width conditioners do not reload from a checkpoint | code | the loader builds only the singular `question_conditioner` |
| 6 | The byte snapshot copies the input's bytes | code (with a data gap) | candidates' bytes come from `_ar_word_part_ids`, the input; rows keep no surface |
| 7 | (this session) the production answer symbol is degenerate | design | `conceptual_state` is None in production; `symbolic_state` is the SS activation view |

None of the six is a specification problem: the specification (the
sentence-boundary thinking spec's requirements, the reverse-loops plan's
contracts, Alec's decisions of 2026-09-14) is unambiguous on each; the
code does not yet do what it says. Item 2 carries one decision Alec must
make (3.2). Item 7 is a design question.

### 3.1 Item 1: the generation chooser never trains

Evidence. `reverseOutput` (`Models.py` ≈8890–8940) computes
`policy_cost` and calls `record_loss("output_policy", ...)`, which is the
report registry (invariant 10); `runBatch` never reads
`_output_policy_cost` except to clear it (`Models.py:12563`).
`LanguageSpace.__init__` creates `self.generate_policy` (`Language.py`
≈14177) without appending its parameters to any space's `params`
(invariant 11), so `getOptimizer` never sees it. Codex's probe: with
`outputPolicyWeight=1`, standalone gradients exist, parameter updates are
zero. Both halves are code defects; the specification (Training, "The
output walk's generate policy": "the imitation credit trains the policy")
is clear.

Fix.
* In `runBatch`, after `reverseOutput` ran in training, add
  `output_policy_weight × _output_policy_cost.mean()` to `totalLoss`
  next to the other assembled terms (keep the `record_loss` for the
  report). Read the cost before the clearing list at `Models.py:12563`.
* Register the policy: append `generate_policy.parameters()` to the
  SymbolSpace's `params` where `LanguageSpace` is built (the
  `self.params.append(p)` idiom in `Language.py`), or adopt it through
  `_collect_fresh_synthesis_modules` (the once-only `add_param_group`
  idiom). Prefer the `params` list: it exists at optimizer construction.
* Tests: `test_output_walk.py`: build the fixture with
  `outputPolicyWeight` 1, run one training `runBatch` with a What
  question, assert the policy's weight changed and `totalLoss` includes
  the term (compare against the same batch with weight 0).

### 3.2 Item 2: answer ownership

Evidence. `_materialize_answer_idea` (`Models.py` ≈11229) reads
`self._word_symbol_rows()`, `self._tensor_pushed_ideas`, the
reconstruction stack and `symbolSpace._word_reference_activations`: the
model's LIVE staging. Holding an `Understanding` and its
`AnswerDerivation` fixed and staging another sentence changes the
materialised idea (Codex's probe), because nothing on the derivation
carries the answer's rows or program. Conversely the symbol-width
conditioner (`_condition_answer_on_question` inside `_resolve_answer` /
answer construction) moves the resolved answer symbol, while the
materialisation conditions the idea with the concept-width conditioner:
two modules conditioning two objects, so a change in one is invisible in
the other. Shared indices do not fix this; ownership does.

Fix (code).
* `_capture_understanding` (`Models.py` ≈7825) captures the sentence's
  program once, as a frozen record on the `Understanding`
  (`bin/Understanding.py`; add a field `answer_program`): symbol rows
  and activations of the last sentence, its compact leaves `[n, D]`
  (`_answer_leaf_slab` at the rows), its forward actions `[L, 3]`, the
  walk's targets `[T]`, and the end state `[3, D]`. Per packed slot the
  same capture feeds `_recall_program_history` at observation time
  (`_observe_discourse(slot=t)`), so recall entries are Understanding
  products too.
* `AnswerDerivation` (`bin/Output.py:48`) gains a `program` field.
  `_resolve_answer` fills it per row: identity/reasoning from
  `understanding.answer_program`, recall from the history entry `k`
  back, prediction `None` (unresolved).
* `_materialize_answer_idea(understanding, derivation, question)` reads
  ONLY `derivation.program` and the question. Delete its reads of model
  staging (`_word_symbol_rows`, `_tensor_pushed_ideas`, the trace).
* Test: hold `u`, `derivation`; stage another brick; materialise; assert
  equality with the first materialisation. Second test: a recalled
  answer materialises from the recalled sentence's program even after
  the current staging changed.

Decision for Alec (design). Which conditioner conditions the answer?
Today `question_conditioner` (symbol width, What spec Step 7, trained by
`answer_construction`) conditions the resolved symbol, and the
concept-width one conditions the materialised idea. Under the
index-driven design the idea is a function of rows and derivation; a
vector added to the symbol cannot move rows. Options: (a) one conditioner
at the concept width, applied in the materialisation, and the
symbol-width one retired (the answer-construction objective then trains
through the materialised idea's realisation); (b) keep both, and record
the question context on the derivation so both read the same context.
Recommendation: (a). Codex's "a learned conditioner changed the resolved
answer without changing its materialisation" is answered by (a) by
construction.

### 3.3 Item 3: teacher forcing in evaluation

Evidence. `_output_generate_walk` (`Models.py` ≈11646) computes
`forced = unstamped & (target >= 0)` and takes the target's action
whenever a teacher exists, in training and in evaluation. Opposing
chooser settings therefore produce identical output while targets are
present (Codex's probe). The specification wants the teacher as the
learning signal, not as the generator: "the output loop should realize
the resolved answer's own structure" through its chooser.

**Superseded by Alec's section 6 clarification:** the following original
recommendation assumed input-parse imitation. Item 3 now removes output
teacher forcing in both training and evaluation and uses `<generate>`.

Original recommendation. A model knob `<outputTeacherForcing>` (`data/model.xsd`,
`doc/Params.md`): `train` (default; force only when `self.training`),
`always`, `never`. In evaluation the chooser decides every unstamped
top; the credit is still computed against the teacher where one exists
and reported (an evaluation metric of the chooser). `reverseOutput`
passes `targets` to the walk only when forcing applies; the credit path
takes them regardless. Tests: in evaluation, opposing biases on
`generate_policy` (the `_prefer` / `_stop` helpers in
`test_output_walk.py`) produce different emitted sequences with targets
present; in training with forcing they produce the same sequence and
different credits.

### 3.4 Item 4: intermediate seal operands

Evidence. In `stage_cs_lang` (`Models.py` ≈20100–20130) the seal loop
reassigns `sealed_stm` with the result of
`cs.apply_binary_language_choice(sealed_stm, seal_choice)` and only then
calls `_tensor_record_operands(..., sealed_stm, seal_valid)`: the rows
recorded are those AFTER the reduction (the composite's −1 and the slot
below it). The per-word sites are right (they pass `pre_state` and
`resolved`, the states before their folds, `Models.py` ≈20071–20074),
and the eager path is right (`_stm_bounded_reduce_step` reads
`concept_rows` before reducing). Codex's `(a+b)+c` in an intermediate
packed sentence reconstructs as `[a−b, b, b+c]` without truncation.

Fix. Keep `pre_seal = sealed_stm` before `choose_sentence_seal_binary`
and record from `pre_seal`. Test (real, not synthetic): a packed
two-sentence brick run with `_tensor_peer_while_eager`; extend
`_replay_program` (or a small sibling) to track each slot's row while
replaying (push → the word's row, fold → −1), and assert for every
recorded binary slot (pre, post, seals of every sentence) that the
recorded left/right rows equal the tracked rows of the two tops before
that fold. This checks all three record sites at once.

### 3.5 Item 5: conditioner checkpoints

Evidence. `_materialize_answer_path_from_checkpoint` (`Models.py`
≈8775–8810) builds modules for a fixed set of keys before
`load_state_dict`; for the conditioner it handles only
`question_conditioner.weight`. A checkpoint saved with
`question_conditioners.136` and `.1032` produces a shape mismatch (the
singular module is built from the first key it finds) and an unexpected
key. `_collect_fresh_synthesis_modules` and `synthesis_parameters`
also read only the singular attribute.

Fix. In the loader, for every key matching
`question_conditioners.<width>.weight`, build `nn.Linear(in, width,
bias=False)` into the `ModuleDict` before loading; set the singular
attribute to the module of the symbol width for old callers. Iterate
the `ModuleDict` in `_collect_fresh_synthesis_modules` and
`synthesis_parameters`. Test: save a model that has used both widths,
load into a fresh model of the same configuration with `strict=True`,
assert both weights equal and no missing/unexpected keys; and that both
are stepped by the optimizer.

### 3.6 Item 6: the snapshot's bytes

Evidence. `_stage_snapshot_bytes` (`Models.py` ≈10638) takes each
candidate row's bytes from `isp._ar_word_part_ids` through
`ps._pid_byte_table`: the CURRENT INPUT's bytes placed at the word rows
and mirrored at the object rows. Changing the staged input bytes while
keeping rows changes the decoding (Codex's probe). The specification is
"the tied inverse of the concept lookup is the snap to a row and a row's
surface is its bytes" (Training, "byte-level fidelity"): the surface
must belong to the ROW, not to the input.

Data gap. Dictionary rows keep no surface today: `create_word_object_meta`
(`Spaces.py` ≈19804) is keyed by a surface string at creation
(idempotent per key) but stores no row → surface map;
`WholeSpace._row_bytes` is the property rows' byte predicates, a
different table. `_stage_serial_concept_rows` has `word_texts` and the
resolved rows per word at staging time.

Fix. A row-owned surface store on the concept owner (`_row_surfaces`:
row → bytes, written at `_stage_serial_concept_rows` when a word's row
is resolved or minted; the object row shares its word's surface),
persisted with the vocab extras (`_collect_vocab_extras` /
`_restore_vocab_extras`). `_stage_snapshot_bytes` builds `_ar_bank_bytes`
by reading the store at the bank's rows (a host gather at the eager
boundary, once per brick), never from `_ar_word_part_ids`. Tests: change
the staged input bytes keeping rows → decoding unchanged; change a
row's stored surface → decoding changes; a row whose surface is unknown
is absent from the candidates (the null candidate covers it).

### 3.7 Item 7: the production answer symbol (found here, not yet reported)

Evidence. On `data/BasicModel.xml` the understanding's
`conceptual_state` is None and `symbolic_state` is the SS activation
view, which `_capture_understanding`'s own comment describes as "the
same for every sentence" at initialisation. The ladder fixtures have
equal widths and an `answer_seed`, which hides this. Under the
index-driven design the answer path no longer needs a dense symbol
event at all: the answer is rows plus a derivation (item 2). Once item 2
lands, `_resolve_answer` can stop depending on `symbolic_state` for
identity and recall; prediction still needs a predictor over rows.

Decision for Alec: whether the symbol-space answer event
(`answer_symbol`, `AnswerConstruction`) stays as the object the
What-spec answer losses train on, or whether those losses move to the
materialised idea's realisation.

## 4. Execution order

One item per commit, suite green before each, push and bump after each
(invariant 12). Order: 4 (small, isolated), 5 (small), 1 (small, unlocks
training of everything after), 3 (independent generate), 6 (store + staging), 2 (the
ownership refactor; after Alec's decision on the conditioner), then 7
if Alec decides.

For each item: write the failing test first from Codex's probe, fix,
run the file, run the affected files
(`test_output_walk.py`, `test_reverse_traversal.py`,
`test_compiled_word_chunk.py`, `test_what_training.py`,
`test_output_path_supervised.py`, `test_while_loop_gradients.py`,
`test_reconstruction_roundtrip.py`, `test_word_store.py`), then the
suite. Update Training.md / Params.md / this plan's status in the same
commit. Answer Codex with the test names.

## 5. Verification gates (Codex's probes as tests)

Execution evidence, item 4:
`test_reverse_traversal.py::test_packed_trace_records_pre_fold_operand_rows_at_every_binary`
first failed at intermediate seal slot 159: recorded `(5, -1)` against
the replayed pre-fold rows `(-1, -1)`. The fix retains `pre_seal` for
operand recording. The test replays all live binary and unary choices
in two packed rows, checks every binary trace slot, and requires both
per-word folds and final/intermediate seals, including a leaf beside a
composite. Verification: the file passed (15 tests); the eight affected
files passed (112 passed, 6 skipped); the full suite passed (4047 passed,
53 skipped, 7 xfailed, 4 subtests passed, 169 warnings) in 1705.26 seconds.
The suite ran in the background with `DEVELOPER_DIR` selecting the installed
Command Line Tools, avoiding the unrelated Xcode license prompt. No
`bin/*.py` files changed while pytest was running.

Execution evidence, item 5:
`test_output_walk.py::test_question_conditioner_checkpoint_reloads_both_widths_strictly`
first failed with a `[1032, 29]` singular alias against the fresh
`[136, 29]` module and an unexpected `question_conditioners.1032.weight`.
`test_question_conditioner_optimizer_steps_both_widths` first failed
because the live `runBatch` optimizer omitted the 136-wide module. The
loader now materialises all saved widths and normalises the singular
alias; parameter collection visits all widths once. Both probes pass,
along with a strict-load regression for singular-only legacy checkpoints.
The output-walk and synthesis files pass (41 tests); the eight affected
files pass (115 passed, 6 skipped). The background full suite passed:
4050 passed, 53 skipped, 7 xfailed, 4 subtests passed, 170 warnings in
1700.19 seconds. No `bin/*.py` files changed during pytest.

Execution evidence, item 1:
`test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight`
first reproduced positive standalone policy gradients with no total-loss
gradients at weight 1, and missing optimizer ownership at weight 0.
`test_runbatch_does_not_train_generate_policy_without_supplied_answers`
first showed that the preceding supervised batch never updated the policy.
`test_runbatch_generate_policy_masks_rows_without_supplied_answers` exposed
the unlabelled row receiving credit 0.5. All four parametrized cases now
pass: the policy belongs to SymbolSpace's explicit parameter list and
`runBatch` trains it from supplied answer error, under the section 6
correction, rather than connecting the old input-imitation objective.
The tests observe the real `record_loss` and backward without replacing
losses, verify the exact answer-error multiplier, and cover zero weight,
mixed rows, missing labels after an Adam update, present/input questions,
and clearing the cost when the next batch skips output generation.
The full output-walk file passed (19 tests); the eight affected files passed
(119 passed, 6 skipped, 16 warnings in 260.97 seconds). The background full
suite passed: 4054 passed, 53 skipped, 7 xfailed, 4 subtests passed,
174 warnings in 1772.07 seconds. No `bin/*.py` files changed during pytest.
Training chooses independent actions now;
the remaining evaluation teacher and compose-rule inventory are item 3.

Execution evidence, item 3:
`test_reverseoutput_evaluation_uses_its_policy_with_input_trace_present`
first reproduced identical output under opposing evaluation policy biases.
`test_output_rule_inventory_comes_from_generate_even_without_compose_rule`
first found 22 choices for a grammar declaring only sum.reverse plus stop.
Both regressions now pass. The walk ignores the input teacher in both
modes, chooses from its own generate catalog, and retains reconstruction's
identified compose trace. Tests also execute a generate-only sum, compile
an empty unary/binary catalog, compare sampling with/without input targets,
and exercise saved chooser/Adam rows across catalog changes.
The output-walk file passed (24 tests), then the additional new-action
moment case passed with the eight affected files: 125 passed, 6 skipped,
16 warnings in 265.71 seconds. The background full suite passed:
4060 passed, 53 skipped, 7 xfailed, 4 subtests passed, 174 warnings in
1765.20 seconds. No `bin/*.py` files changed during pytest.

Execution evidence, item 6:
`test_dictionary_surface_snapshot_is_independent_of_staged_input_bytes`
first reproduced changed candidate bytes after replacing only staged input
byte IDs (both snapshots use the same fixed P shape). The other three
probes first failed because no `_row_surfaces` store existed. WORD rows
now own UTF-8 bytes; OBJECT rows resolve their current WORD association,
without owning copied surfaces. Tests check the actual byte loss as well
as the snapshot, changed associations, strict checkpoint persistence, and
uniform-null cost for missing surfaces or snapshots. The checkpoint probe
also exposed an existing lazy `chunk_prior` missing on a fresh model; the
loader now materialises a saved prior before its strict key audit, and the
regression restores a nonzero prior without staging any input first.
The word-store file passed (33 tests in 22.11 seconds), followed by the
eight affected files (129 passed, 6 skipped, 16 warnings in 269.46 seconds).
The background full suite passed: 4064 passed, 53 skipped, 7 xfailed,
4 subtests passed, 174 warnings in 1753.83 seconds. No `bin/*.py` files
changed during pytest.

Execution evidence, item 2:
`test_held_answer_idea_ignores_later_staging_and_memory_context` reproduced
272 changed elements in a held idea, with nonzero conditioner weights so
live memory reads could not hide behind zero initialization.
`test_materialised_answer_uses_one_conditioning_application` counted two
calls instead of one. `test_compiled_understanding_captures_explicit_sentence_products`
found no materialised idea before the caller published compiled outputs.
After correcting the new recall fixture's duplicate setting and disabling
unrelated forward priming, `test_resolved_recall_keeps_its_program_after_memory_advances`
reproduced 216 changed elements after advancing actual discourse memory.
A packed-slot regression covers separate sentence records, absent lanes,
and observation from a held understanding after different staging.
The fix captures immutable programs and target-free conditioning context,
selects recall once, and applies one conceptual conditioner. The six focused
checks passed (31.84 seconds), then the complete output-walk file passed
(30 tests, 6 warnings in 85.68 seconds). The dense production-symbol
dependency remains item 7. The first eight-file
run found a serial memorization regression (7/8 labels). Moving the single
conceptual conditioner before the dedicated conceptual answer operator
restored the unchanged 8/8 learning gate (39.19 seconds). The eight affected
files then passed (134 passed, 6 skipped, 16 warnings in 287.94 seconds),
and the synthesis/understanding/thinking compatibility checks passed
(41 tests, 2 warnings in 7.60 seconds). The compatibility observation spy
also forwards the new understanding argument. The first background full
suite found four failures in lightweight What adapters without InputSpace
(4065 passed, 53 skipped, 7 xfailed, 174 warnings, 4 subtests passed in
1899.71 seconds). Capture now treats the absent input owner as no row
program; all 22 spacetime tests pass (2.62 seconds). The eight affected
files passed again (134 passed, 6 skipped, 16 warnings in 286.18 seconds).
The repeated background full suite passed: 4069 passed, 53 skipped,
7 xfailed, 174 warnings, 4 subtests passed in 1910.95 seconds. No `bin/*.py`
files changed while pytest was active.

1. Policy training: weights change under `runBatch` with
   `outputPolicyWeight` 1; unchanged with 0.
2. Ownership: same `Understanding` + derivation, different staging, same
   idea; recall materialises from the recalled program.
3. Independence: in evaluation the chooser's setting changes the output
   with targets present.
4. Operands: recorded rows equal the tracked pre-fold rows at every
   binary slot of a packed brick.
5. Checkpoint: both conditioner widths reload strictly and train.
6. Surfaces: decoding invariant to input bytes, sensitive to the row's
   stored surface.

## 6. Open questions for Alec

2026-09-15 decisions and clarification during execution:

* Alec selected one conditioner; use the concept-width conditioner for
  the materialised answer. His subsequent role clarification places the
  answer loss on the realised output against a separately supplied desired
  answer (item 7), not on input reconstruction or a dense symbol proxy.
* Alec clarified that WORD and OBJECT are entirely different concepts;
  interpreting a word can replace its meaning. This supersedes item 6's
  proposed automatic copy of a word's surface to its object row. Bytes
  belong to the word. The existing `word_concept_of_object()` boundary
  (`bin/Spaces.py:19944`) already translates an object through its META
  association to a word. Preserve that boundary for decoding; never store
  a copied word surface as an object's own surface.
* Alec clarified that there is no known correct parse, only the parse
  identified by the forward. Reconstruction follows that identified
  compose derivation. `reverseOutput` uses a different surface and its own
  derivation through `<generate>`. This supersedes item 3's recommendation
  to force the input derivation during output training, and item 1 must
  not treat imitation of that input parse as justified output supervision.
* Alec restated the roles: `forward()` builds a 1-3 idea form using
  syntactic operations over known words; `reverseReconstruct()` tests that
  representation by recovering the input surface; `reverseOutput()`
  constructs a question-dependent answer with a potentially entirely
  different surface. Output training requires supervised desired answers.
  Input reconstruction targets or compose choices must not substitute for
  answer supervision. Add a no-output-update gate for unsupervised batches.
  Item 1 must therefore credit independently chosen output actions from
  supplied answer error rather than connect the old input-imitation term.
  Item 3 removes the remaining output teacher dependency and ensures the
  output rule inventory comes from `<generate>`. The proposed
  `<outputTeacherForcing>` knob is superseded; reconstruction retains the
  identified compose derivation.

Vocab extras remain the surface store's persistence boundary. The decisions
needed for items 2 and 7 are now recorded above.
