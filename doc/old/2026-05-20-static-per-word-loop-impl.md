# Static Per-Word Loop & OOV Reserve — Implementation Plan

Executes `doc/plans/2026-05-20-static-per-word-loop-oov-reserve.md`.

## Context

`MM_5M` compilation is unstable under `torch.compile(mode="max-autotune")`:
`TORCH_LOGS=recompiles` on `metalbaby` shows repeated whole-forward
recompiles driven by the `_valid_len_host` guard. The data-dependent
`while next_word() is not None` loop in `_forward_body_per_word`
(`bin/Models.py:5179-5199`) causes Dynamo to specialize on each
observed sentence length, and any OOV row arriving via
`Embedding.insert()` (`bin/Spaces.py:2655-2727`) reassigns
`nn.Parameter` and calls `_rebuild_optimizer()` mid-training —
guaranteeing further recompiles and unstable optimizer state.

Goal: make per-batch sentence length and per-batch vocabulary
invisible to the compiled graph. Trip count becomes the static
`InputSpace.outputShape[0]`, padding positions are tensor-masked
no-ops, and OOV rows are staged into preallocated reserve slots
outside compiled compute. Codebook parameter shape is fixed for the
duration of a training run.

## Decisions (from clarification)

- **Single bundled PR** — static loop + OOV reserve land together.
- **OOV-full fallback** = the existing PerceptualSpace byte path
  (`bin/bpe_gpu.py`, `bin/mphf_gpu.py`); increment counter; sample
  capture for offline rebuild.
- **Adam state for newly activated rows** = zero `exp_avg`,
  `exp_avg_sq`, `step` at the activated indices under `no_grad`.
  Do NOT call `_rebuild_optimizer()`.
- **Padding semantics = grammatical no-op, not masked-blend.** Add an
  identity rule `S → S` (LHS == RHS, `method_name=None`) to the
  grammar. At padding columns the rule cursor selects this rule;
  PS/SS/CS forwards still run (static graph), but per-iteration
  side-effects (STM push, carrier event update, priming application)
  are gated by `rule_is_real = (selected_rule != id_SS)`. The gate's
  mask source is the rule id, not a separate `word_active` tensor.
- **S → S is learnable**, not hard-overridden. The model treats it
  like any other rule and is free to fire it mid-sentence as a
  "wait / no-commit" step. word_active only biases the *initial*
  cursor padding in `WordSpace.compose`.
- **Forward/reverse padding is asymmetric.** Forward: `WordSpace.compose`
  pads the rule sequence to length `N = outputShape[0]` by appending
  `S → S` rule ids past the active prefix. Reverse: NO leading no-op
  padding — `reverse()` decodes the real rules immediately, then the
  generated surface representation is LEFT-SHIFTED so it begins with
  non-NULL content. Trailing zeros after the shift are the residual
  output padding.

## Files to modify (with anchor line numbers)

| Concern | File | Anchor |
|---|---|---|
| Add `S → S` to grammar config | `data/grammar2.cfg` (and XML start sym) | — |
| Dispatcher short-circuit for `method_name=None` | `bin/Language.py` | 5100-5124 (`SyntacticLayer.execute`) |
| Pad rule cursor to N with `id_SS` (forward) | `bin/Language.py` | `WordSpace.compose` + `_next_rule_name` (5012-5075) |
| Reverse path: no S→S padding + left-shift output | `bin/Language.py` | `LanguageLayer.reverse` / `reverse_stack` (1879-2066) |
| Per-word while → static range loop | `bin/Models.py` | 5179-5199 |
| Per-word body (rule-gate writes) | `bin/Models.py` | 4963-5084 |
| Concept output preallocation | `bin/Models.py` | 5139, 5195-5196, 5367-5381 |
| Discourse cache dtype staging | `bin/Models.py` | 2568-2572 |
| `word_at(p)` static feeder | `bin/Spaces.py` | 6520-6572 (replace `next_word`) |
| `word_active` mask construction | `bin/Spaces.py` | 6418-6452 (alongside `_valid_len_host` eager block) |
| Priming bias rule-gate | `bin/Spaces.py` | 9556-9581 (ConceptualSpace.forward) |
| Carrier event masked-blend by rule-id | `bin/Spaces.py` | 9590-9670 (ConceptualSpace.forward write of cs_for_ps/ss) |
| STM `push_step` masked variant | `bin/Spaces.py` | 8944-8966 |
| WorkingState carrier stability | `bin/Spaces.py` | 3414-3469 |
| `WordSpace.soft_reset` pre-allocation | `bin/Language.py` | 7111-7176 |
| Embedding OOV staging (move out of forward) | `bin/Spaces.py` | 3071-3079, 2655-2727 |
| Reserve-full → byte fallback hook | `bin/Spaces.py` (Embedding) + `bin/bpe_gpu.py` | 2675-2684, 132-160 |
| Adam moment zeroing (no rebuild) | `bin/Spaces.py` | 2634-2639 (replace `_rebuild_optimizer`) |

## Implementation steps

### 0. S → S identity rule + dispatcher

0.1 Add `S → S` (identity) rule to `data/grammar2.cfg`. The Grammar
loader at `bin/Language.py:633-638` already recognizes `rhs == lhs`
as a unary identity with `method_name=None`. Cache its rule id on
the grammar object after configuration: `TheGrammar.id_SS = <id>`.

0.2 `SyntacticLayer.execute` (`bin/Language.py:5100-5124`) currently
looks up a layer by `method_name`. For `method_name=None` it must
short-circuit and return the operand unchanged (no layer lookup,
no parameter touch). Add an explicit branch at the top of `execute`:

```python
if method_name is None:
    return left            # arity == 1, identity passthrough
```

0.3 The reverse path auto-derives the inverse of an identity forward
rule as `projectReverse` (`bin/Language.py:766-806`, lines 798-800),
which is itself identity (Phase 7 contract, lines 1813-1817). No
extra code is needed for reverse-side dispatch — the existing
projectReverse handler already returns its input.

0.4 The rule is **learnable**: do NOT bias logits toward `id_SS`
during forward selection. It enters the rule space on equal footing.

### 1. Forward rule-sequence padding to N

1.0 In `WordSpace.compose` (the function that builds the
cursor-indexed rule sequence consumed by `_next_rule_name` at
`bin/Language.py:5012-5075`), after the real rule sequence is
constructed, append `id_SS` rule ids until total length is
`N = InputSpace.outputShape[0]`. This is the ONLY place the
asymmetry between forward and reverse is encoded — the forward
sequence is length-N; the reverse sequence is length-(real-count).

1.1 The per-word loop reads the cursor uniformly with no special
case. The cursor advances one rule per iteration; the last few
iterations naturally pull `id_SS`.

### 2. Static per-word loop

2.1 In `InputSpace` (`bin/Spaces.py`), add `word_at(p) -> Tensor[B, 1, D]`
that returns `_ar_embedded[:, p:p+1, :]` for `0 <= p < outputShape[0]`,
and `word_active(p) -> Tensor[B, 1]` derived from `_bpe_word_mask` (or
`abs().sum(-1) > 0`) — same source as today's `_valid_len_host`
computation but **kept as a tensor**, never `.item()`'d. Construct the
full `[B, N]` activity tensor once in the same eager block at
`bin/Spaces.py:6418-6452` and stash it as `_word_active_mask`.

2.2 Keep `_valid_len_host` as an eager diagnostic only; remove all
code paths that gate Python control flow on it. `next_word()` is
deleted; the loop no longer calls a stateful cursor.

2.3 Replace the while loop at `bin/Models.py:5179-5199` with:

```python
N = self.inputSpace.outputShape[0]
id_SS = TheGrammar.id_SS
out_slot = self._per_word_concept_buf               # [B, N, D_c], preallocated, zeroed each step
for p in range(N):
    w = self.inputSpace.word_at(p)                  # [B, 1, D]
    CS_sub, idea_bd, selected_rule = self._per_word_body_step(w, p, out_slot)
    # selected_rule: [B] int64; rule_is_real := (selected_rule != id_SS) is the gate
    if CS_sub is not None:
        last_cs = CS_sub
```

Trip count is a Python int constant equal to `N = outputShape[0]`,
so Dynamo unrolls / specializes once.

2.4 Rework `_per_word_body_step` (`bin/Models.py:4963-5084`) so that
**rule-id selection drives all per-iteration gates.** The static loop
runs all N iterations; PS/SS/CS forwards are invoked unconditionally;
the dispatcher returns the operand unchanged when the selected rule is
`S → S` (step 0.2). Per-iteration side-effects (STM push, carrier
event update, priming bias) are gated by
`rule_is_real := (selected_rule != id_SS)` as a `[B, 1]` (or `[B]`)
tensor mask:

```python
prev_ps_event = prevCS_forPS._event       # snapshot (carrier obj stable)
prev_ss_event = prevCS_forSS._event
PS_sub = self.perceptualSpace.forward(word_sub, prevCS_forPS, work=_work)
SS_sub = ss.forward(prevCS_forSS, work=_work)
CS_sub = cs.forward(PS_sub, SS_sub, work=_work)

selected_rule = _work.op_sel_at(p)                  # [B] long
rule_is_real = (selected_rule != id_SS).view(B, 1, 1)
CS_sub._event = torch.where(rule_is_real, CS_sub._event, prev_cs_event_snapshot)
# same blend on PS_sub._event / SS_sub._event
# STM push gated by rule_is_real (step 2.6)
# Per-word concept buffer scatter gated by rule_is_real (step 2.5)
```

Inside ConceptualSpace.forward (`bin/Spaces.py:9556-9581`), the
priming bias `y = y + bias.unsqueeze(1)` becomes
`y = torch.where(rule_is_real, y + bias.unsqueeze(1), y)` — bias
fires only at real-rule steps. The current `valid_mask` zeroing of
bias stays (it's an orthogonal K-axis concern for the static-slab
path); the new gate composes with it.

The carrier OBJECTS (`_work.cs_for_ps`, `_work.cs_for_ss`) stay
identity-stable across iterations; only the inner `_event` tensors
are masked-blended.

**Why this is structurally cleaner than a raw `word_active` mask:**
the gate has a single source-of-truth (which rule the cursor picked)
that downstream code can introspect, log, and learn from. It also
preserves the model's freedom to fire S → S mid-sentence as a
no-commit "wait" (per the learnable decision).

**Wasted FLOPs are accepted.** PS/SS/CS run for inactive rows; their
outputs get masked away at commit time. `word_at(p)` returns zero
slices for those rows so the wasted compute is on zero-inputs.

**What PerceptualSpace actually emits at padding (verified):** the
forward path produces a zero `PS_sub._event` (or, with `quantize=True`,
a snap to codebook row 0, the `"\x00"` NULL row — `learnable_codebook=True`
gates EMA off at `bin/Layers.py:9738`, so the snap is read-only).
MPHF / BPE chunking sets `word_active=0`, `what_indices=null_idx`,
`word_vectors=0` (`bin/mphf_gpu.py:116-120`,
`bin/Spaces.py:7631-7637`). No `nn.Parameter` writes, no buffer
mutations.

**However: ConceptualSpace has an averaging step**
(`bin/Spaces.py:9495-9524`) that does `x = (primary + sym) / 2` when
PS and SS event shapes match. At padding, `primary=0` but `sym` is
the carrier-derived SS event, which is generally non-zero. Without
the rule-gate commit-time blend, `cs_for_ps` would be silently set
to `SS / 2` at every padding column — a real leak. The
`CS_sub._event = torch.where(rule_is_real, CS_sub._event, prev_cs_event_snapshot)`
blend at the body's commit point is what stops this. **The rule-gate
is load-bearing, not redundant with the zero input.**

2.5 Replace the growing `per_word_concepts = []` (`bin/Models.py:5139,5195-5196`)
and post-loop stack/pad (`bin/Models.py:5367-5381`) with a
preallocated `out_slot = self._per_word_concept_buf` of shape `[B, N, D_c]`.
Allocate it once on first call, `zero_()` at the top of each forward
(via `_per_word_prelude`). Write at position `p` with masked scatter
gated by `rule_is_real`:

```python
out_slot[:, p, :] = torch.where(rule_is_real_2d, idea_bd, out_slot[:, p, :])
```

After the loop, the buffer IS the stacked `[B, N, D_c]` (S→S
positions already zero), so `cs_sub.set_event(out_slot)` directly.
Eliminates Python list growth and `torch.stack` inside the captured
region.

2.6 Add `ShortTermMemory.push_step_masked(ideas, rule_is_real)` at
`bin/Spaces.py:8944-8966`:

```python
def push_step_masked(self, ideas, rule_is_real):  # rule_is_real: [B] bool
    B, D = ideas.shape
    device = self._buffer.device
    row_idx = torch.arange(B, device=device)
    depths = self._depth
    self._buffer[row_idx, depths] = torch.where(
        rule_is_real.view(B, 1), ideas, self._buffer[row_idx, depths])
    self._depth = depths + rule_is_real.long()
```

`_max_depth_host` mirror update happens in the eager wrapper around
the loop (not inside the captured body) so it can do
`self._max_depth_host = self._max_depth_host + int(any_real_in_batch)`
without a DtoH inside compiled code. Source for `any_real_in_batch`:
the rule-cursor sequence is known at the eager-prelude stage (built
by `WordSpace.compose`), so the host can count real-rule positions
per batch and add to the mirror once per forward.

2.7 Move the discourse cache cast to autocast dtype out of compiled
code. In `_begin_step` (`bin/Models.py:2568-2572`), after
`disc.stage_prediction()` returns its staged tuple, cast all
floating-point tensors in that tuple to the AMP dtype reported by
`amp_context()`. This kills the dtype-driven recompile path.

### 2R. Reverse / reconstruction loop (asymmetric)

2R.1 The reverse path (`LanguageLayer.reverse` /
`reverse_stack`, `bin/Language.py:1879-2066`) **does NOT pad with
`S → S`**. The reverse rule sequence is the inverse of only the REAL
forward rules — length equal to the active prefix, not N. The auto-
derived `projectReverse` for the identity rule remains in the
grammar's reverse catalog (`bin/Language.py:766-806`) but is not
exercised by the reverse cursor unless the model deliberately
generates `S → S` mid-stream.

2R.2 After the reverse loop produces the per-position generated
surface representation `gen [B, N, D]`, apply a **left-shift** so
the output begins with non-NULL content. Implementation:

```python
# active per position from the reverse cursor's non-identity rules
# build [B, N] active mask: True where generated rule != id_SS
real_pos = (rev_selected_rules != id_SS)             # [B, N] bool
# left-shift gen so real_pos[:, 0] == True for all rows
# (gather: for each row b, take positions where real_pos is True
#  packed into prefix [0, n_real(b)); zeros elsewhere)
gen_shifted = _left_shift_by_mask(gen, real_pos)
```

`_left_shift_by_mask` is a small helper using `torch.cumsum`-based
index gather; it must stay tensor-only (no Python loops, no
`.item()`) so it lives inside the compiled region without inducing
recompiles. Reference pattern: the existing batched stack
compaction at `bin/Language.py:1698-1722` uses the same
arange+gather style.

2R.3 The post-reverse `valid_mask` for the generated output is the
left-shifted equivalent of `real_pos` (a contiguous `[True]*K`
prefix). Downstream loss/eval code already consumes `valid_mask`
shape `[B, N]` (`bin/Spaces.py:11387`), so no further changes are
needed on the consumer side.

### 3. WorkingState / carrier object stability

3.1 In `WordSpace.soft_reset` (`bin/Language.py:7111-7176`), keep
`self._work` allocated once and reset fields in place (already done
for tensors, but ensure `_work` itself is never reassigned to a new
object — only its fields). Replace `self._work = new_working_state(...)`
with an in-place reset method `_work.reset_for_sentence(...)`.

3.2 `cs_for_ps` / `cs_for_ss` are already stable SubSpace objects
threaded through `ConceptualSpace.forward` (`bin/Spaces.py:9590-9647`).
Confirm `_per_word_prelude` (`bin/Models.py:4953-4955`) seeds them to
the SAME `self._empty_seed_ps` / `self._empty_seed_ss` objects every
call (it already does — verify nothing in `set_event` reallocates the
inner event tensor when shape matches).

### 4. OOV reserve staging

4.1 Add `Embedding.stage_oov(keys: list[str], vectors: Tensor) -> StagedResult`
to `bin/Spaces.py`. Steps:

a. Compute `n_free = lexicon_capacity - len(index_to_key)`.
b. Take the first `min(n_free, len(keys))` keys; the rest go to the
   fallback list.
c. Under `torch.no_grad()`: `self.wv._vectors.data[idx:idx+n, :] = new_vectors`
   (in-place write into preallocated tail rows — `nn.Parameter` shape
   already includes the reserve from
   `bin/Spaces.py:2577-2580`).
d. Update `wv.index_to_key`, `wv.key_to_index`, `wv.counts` for the
   activated rows.
e. Zero Adam moments at the activated row indices: get the optimizer
   state for `wv._vectors`, slice `state['exp_avg'][idx:idx+n] = 0`,
   `state['exp_avg_sq'][idx:idx+n] = 0`, `state['step']` is scalar so
   no per-row work. **Do NOT call `_rebuild_optimizer`**.
f. For each key in the fallback list, increment
   `self._oov_fallback_count += 1` and append to a bounded
   `self._oov_fallback_sample` (e.g., capacity 1024, ring buffer).

4.2 Move OOV collection out of `Embedding.forward`
(`bin/Spaces.py:3071-3079`). The current loop that calls
`self.insert(word)` per OOV is replaced by a single
`self.stage_oov(oov_words, oov_vectors)` call. This must run BEFORE
the compiled forward — call it from `Model._begin_step`
(`bin/Models.py:2568-2572`), alongside `disc.stage_prediction()`, so
the codebook is up to date by the time compiled compute reads it.

4.3 Confirm `wv._vectors` is allocated with the full
`lexicon_capacity` rows from the start. Today
`bin/Spaces.py:2555-2556` calls `super().create(...)` with
`vocab_size` rows, and the docstring at 2577-2580 implies reserve is
just a capacity bound. **Action**: change initial allocation to
`max(vocab_size, lexicon_capacity)` rows, leave the unused tail
rows zero-initialized (they're masked from gradient updates because
`key_to_index` doesn't reference them and the lexical loss only
backprops through referenced indices).

4.4 Reserve-full fallback path. In the eager OOV staging
(`stage_oov`), keys that didn't get a slot are stored in
`self._oov_fallback_keys` for this batch. Update
`PerceptualSpace.forward`'s embedding routing to consult that set:
for fallback keys, route through the existing `byte_mode` /
MPHF byte path (`bin/bpe_gpu.py:132-160`,
`bin/mphf_gpu.py:116-120` for the NULL row). Both already exist —
this is wiring, not new logic.

4.5 Replace `Embedding._rebuild_optimizer`
(`bin/Spaces.py:2634-2639`) with a no-op (or delete it and remove
callers). The persistent optimizer now owns the full `[capacity, D]`
parameter for its lifetime.

### 5. Verification (tests + manual)

Add the following test files under `test/`:

- `test/test_grammar_identity_rule.py`
  - `TheGrammar.id_SS` exists after configuration; `SyntacticLayer.execute`
    with rule_id == id_SS returns the operand unchanged; the auto-
    derived reverse rule for S→S is projectReverse (identity).
- `test/test_compile_static_loop.py`
  - parametrize observed-word counts `[5, 8, 15]`; assert no
    recompiles via `torch._dynamo.utils.counters` and no
    `_valid_len_host` guard hits.
- `test/test_per_word_ss_padding_noop.py`
  - sentence of length L followed by `N-L` S→S rules: STM depth
    increments by L (not N), `_per_word_concept_buf[:, :L, :]`
    matches the standalone-L run, columns `[L:]` are zero.
- `test/test_rule_gate_isolates_side_effects.py`
  - force `selected_rule[b, p] == id_SS` for a known position;
    assert `cs_for_ps._event`, `cs_for_ss._event`, STM `_buffer`,
    and priming bias are bit-identical to the pre-iteration snapshot
    for that row at that step.
- `test/test_reverse_left_shift.py`
  - reverse path with `K < N` real rules produces `gen_shifted`
    whose first `K` positions match the dense reverse output and
    whose tail is zero; `valid_mask` is `[True]*K + [False]*(N-K)`.
- `test/test_oov_reserve_no_resize.py`
  - `stage_oov` does not change `wv._vectors.shape` nor parameter
    identity; key lookup returns the activated row before compiled
    forward runs.
- `test/test_oov_reserve_full_fallback.py`
  - exhaust reserve; assert `_oov_fallback_count` rises and codebook
    shape is unchanged; assert PerceptualSpace routes those keys
    through the byte path.
- `test/test_no_rebuild_optimizer.py`
  - capture `id(optimizer.state[wv._vectors])` before/after a
    `stage_oov` call with one new row; identity must match;
    `exp_avg`/`exp_avg_sq` at the new row index must be zero.
- `test/test_discourse_cache_dtype.py`
  - staged cache tensors arrive at compiled forward in active AMP
    dtype; toggle MODEL_AMP between fp16/bf16 and assert no
    recompiles from dtype guard.

Targeted run pattern (per `feedback_targeted_tests`):

```sh
pytest -xvs test/test_compile_static_loop.py::test_no_recompile_on_length_change
pytest -xvs test/test_oov_reserve_no_resize.py
```

Manual acceptance on `metalbaby` (per spec):

```sh
TORCH_LOGS=recompiles python bin/Mereology.py \
  --batches 5 --compile-mode max-autotune
```

Expect zero `RecompileReason: _valid_len_host` lines.

### 6. Out of scope (explicitly)

- Bucketing the per-word loop (spec rejects it).
- Growing codebook parameters at runtime (spec rejects it).
- Promoting SymbolicSpace to a 1M-row table — keep it at the current
  65536 rows. 1M lexical coverage stays in PerceptualSpace / MPHF /
  BPE only.
- Evicting reserve rows (spec rejects it).

## Critical files to read at start of execution

1. `bin/Models.py:4900-5400` — per-word prelude, body, loop, stacking.
2. `bin/Spaces.py:6034-6572` — InputSpace init, valid_len cache,
   next_word.
3. `bin/Spaces.py:2374-2730` — Embedding class, insert, reserve.
4. `bin/Spaces.py:8818-8967` — ShortTermMemory.
5. `bin/Language.py:7111-7176` — WordSpace.soft_reset.
6. `bin/bpe_gpu.py:55-160` — byte fallback path.

## Done when

- `TORCH_LOGS=recompiles` clean across `--batches 5` on metalbaby
  with sentences of differing real-rule counts.
- `TheGrammar.id_SS` resolves; dispatcher passes-through;
  reverse `projectReverse` is identity.
- Forward `WordSpace.compose` produces length-N rule sequences;
  reverse produces length-K sequences and left-shifts the output.
- All nine new tests green.
- Existing `test/test_per_word_capture_gate.py` still green.
- No call sites of `Embedding.insert` or `_rebuild_optimizer` remain.
- `next_word` removed; only `word_at(p)` and the rule cursor feed
  the per-word loop.
