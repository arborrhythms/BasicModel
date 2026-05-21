# Dissolve SentenceState; Reframe WordSpace as WordSubSpace

**Status**: LANDED 2026-05-21.

## Context

The static-per-word-loop refactor (doc/plans/2026-05-20-static-per-word-loop-impl.md)
introduced `SentenceState` (formerly `WorkingState`) as a per-sentence
carrier passed via `work=` through every `Space.forward` / `Space.reverse`.
The original justification was "no Space should hold state, so pipelining
can occur" — a clean rule for the data Spaces
(Input / Perceptual / Conceptual / Symbolic / Output).

WordSpace is structurally not one of those Spaces: it is the grammar /
serial-processing carrier that travels *alongside* the data tensors, not a
pipeline stage that produces them. Field-by-field investigation showed
that **every live field of `SentenceState` either belonged on WordSpace
(serial / grammatical) or was a forward-loop temporary that should never
have been persisted at all**. The standalone class added no value and
obscured the design.

Goals (all met):

1. **Deleted `SentenceState`.** Partitioned its surviving state between
   WordSpace and per-iteration locals in the per-word loop.
2. **Reframed WordSpace as `WordSubSpace`**, a plain `nn.Module` subclass
   (no longer `Space`). The class is the third argument to the pipeline,
   travelling alongside the data SubSpaces via `subspace.wordSpace`.
3. **Stopped caching CS→PS / CS→SS feedback through any carrier.** The
   persistent `ConceptualSpace._subspaceForPS` / `_subspaceForSS`
   SubSpaces (mutated in place by `CS.forward`) are the sole storage;
   the per-word body reads them directly. The reverse path's contract
   is to *produce* estimates of every intermediate SubSpace, so it
   never reads forward bookkeeping — confirmed by audit.

## Field-by-field disposition of `SentenceState`

| Field | Disposition |
|---|---|
| `cursor` (int64 [n_tiers=3]) | Moved to `WordSubSpace.cursor`; allocated in `soft_reset` on the `_stm_fired` device, zeroed at each `compose`. |
| `recur_pass` (Python int) | Moved to `WordSubSpace.recur_pass`; written by `_forward_body` / `_per_word_prelude`, read by `PerceptualSpace.forward` via `self.subspace.wordSpace`. |
| `cs_for_ps` (SubSpace ref) | Removed. `BasicModel._prev_cs_for_ps` (Python attr) tracks the cross-iteration carry; the per-word body switches the pointer to `cs._subspaceForPS` (persistent CS-tier storage) after iteration 0's `cs.forward`, so iters 1+ see in-place updates. |
| `cs_for_ss` (SubSpace ref) | Same as `cs_for_ps` — local Python attr on BasicModel. |
| `gen` (int64 []) | Deleted. Never read. |
| `valid_mask` (Tensor) | Deleted. Lives on `SubSpace.valid_mask`. |
| `errors` (object) | Deleted. Lives on `SubSpace.errors`. |
| `op_sel`, `op_operands` | Deleted along with the dead Phase 2B tensor-driven S-executor branch in `SymbolicSpace.forward` and the `_arity1_ops` static op table, `_populate_op_sel_from_default_rules`, `_default_op_sel_rule_indices`, `selection_from_current_rules` helpers in `WordSubSpace`. The eager cursor path is the sole S-tier dispatch in production. Phase 2B chart soft-superposition (if/when reintroduced) gets a purpose-built carrier on SymbolicSpace, not co-located with sentence state. |

## Calling-convention change

* Every `Space.forward` / `Space.reverse` signature dropped the
  `work=None` kwarg:
  - `InputSpace.forward(inputData)`, `.reverse(subspace)`
  - `PerceptualSpace.forward(IS_subspace, CS_subspaceForPS=None)`, `.reverse(subspace)`
  - `ConceptualSpace.forward(PS_subspace, SS_subspace=None)`, `.reverse(subspace)`
  - `SymbolicSpace.forward(CS_subspaceForSS)`, `.reverse(subspace)`
  - `ModalSpace.forward(subspace)`, `.reverse(subspace)`
  - `OutputSpace.forward(subspace)`, `.reverse(subspace)`
* Grammar / serial-processing state is reached via
  `self.subspace.wordSpace` (the back-reference set by
  `copy_context`). No new explicit kwarg.

## Files changed

* `bin/Spaces.py` — deleted `SentenceState` class; dropped `work=None`
  from every `forward`/`reverse` signature; rewrote
  `ConceptualSpace.forward` to publish C→P/C→S feedback purely through
  `self._subspaceForPS` / `self._subspaceForSS`; PerceptualSpace.forward
  reads `recur_pass` off `self.subspace.wordSpace`; simplified
  `SymbolicSpace.forward` S-tier dispatch (eager cursor path only).
* `bin/Language.py` — renamed `class WordSpace(Space):` →
  `class WordSubSpace(nn.Module):`; added `self.cursor` / `self.recur_pass`
  fields; soft_reset allocates them on `_stm_fired.device`; inlined
  Space-contract methods (`set_sigma`, `paramUpdate`, `getParameters`,
  `Start`, `End`, `Reset`) directly; deleted `_arity1_ops`,
  `_default_op_sel_rule_indices`, `_populate_op_sel_from_default_rules`,
  `selection_from_current_rules`; back-compat alias `WordSpace = WordSubSpace`
  at module scope.
* `bin/Models.py` — per-stage `_forward_body` and per-word
  `_per_word_prelude` / `_per_word_body_step` thread
  `prevCS_forPS` / `prevCS_forSS` via `self._prev_cs_for_ps/ss`
  (initially `_empty_seed_ps/ss`, then `cs._subspaceForPS/SS`);
  dropped the dead `_LEGACY_INLINE_BODY` block.
* `test/test_phase2a_labor_division.py` — deleted (gated dormant
  op_sel/op_operands fields).
* `test/test_perceptual_loopback.py`, `test/test_phase2_pipeline_primitives.py`,
  `test/test_per_word_capture_gate.py`, `test/test_parallel_backend.py`,
  `test/test_activation_priming_multiply.py`,
  `test/test_per_word_capture_gate_cuda.py` — updated arity asserts
  and stale docstring/comment references.

## Verification

* `grep -r SentenceState bin/ test/` — only historical-reference
  comments remain (describing the dissolution itself).
* `grep -r 'work=' bin/Spaces.py bin/Models.py bin/Language.py` —
  no SentenceState-carrier matches.
* `grep -r '_work' bin/ test/` — no live references.
* Reverse-path audit: `_subspaceForPS` / `_subspaceForSS` are read
  ONLY inside forward methods (`__init__`, `forward`,
  `_stack_route_forward`, `_forward_body`, `_per_word_body_step`).
  Zero reverse-path readers — the reconstruction contract is clean.

## Out of scope

* Phase 2B `op_sel` / `op_operands` reintroduction. If chart
  soft-superposition lands, it gets a purpose-built carrier on
  SymbolicSpace or a dedicated `ChartSelection` object, not on the
  sentence state.
* Compile-stability re-validation (`subspace.wordSpace` is a plain
  Python attribute set via `object.__setattr__`, not registered as
  an `nn.Module` child, so no Dynamo regression expected).
