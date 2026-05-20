# Phase-5 whole-forward CUDA capture closeout

**Status:** open — single offender remains for end-to-end
`max-autotune` training. See `doc/plans/2026-05-18-two-loop-pipeline-architecture.md`
D8 section for the broader context and what's already landed.

## What's already done (read first)

The Phase-5 work landed in the 2026-05-19 session got the per-word
body's CPU fullgraph gate green (38/38 targeted tests, including
the 4-test capture-gate suite) **and** the per-word-body-in-
isolation CUDA fullgraph compile green. Then the production
whole-forward compile (`bin/train.py --compile-mode max-autotune`)
exposed four whole-forward-only offenders; three are fixed, one
is deferred to this plan.

**Fixed (in `bin/Spaces.py`, `bin/Models.py`, `bin/Language.py`):**

1. `next_word()` lazy compute `.item()` — replaced by eager
   compute in `InputSpace.forward()` under `torch.no_grad`.
2. `WorkingState.recur_pass` is now a plain Python int (was a
   0-d tensor; Inductor produced unbacked SymInt
   `Min(2, Max(0, u0))` when used as a `ModuleList` index in
   `PerceptualSpace.forward`).
3. `_stm_reducer`'s lazy `BinaryStructuredReductionLayer` build
   (`nn.Parameter(...)` ctor inside the trace) is now
   pre-warmed in `enable_compiled_step`.

**The single remaining offender** is in
`bin/Language.py:1531-1546` — `binary_tiling_viterbi` runs a
per-batch-row Python loop with `.item()` extractions for
sequential Viterbi backtrace. Dynamo traces it because
`util.py:621` sets `_dyn.config.allow_unspec_int_on_nn_module = True`
to avoid cursor-advance recompiles, so the guard
`if stm._max_depth_host >= stm.capacity:` in
`_per_word_body_step` ([Models.py:5019](../../bin/Models.py))
doesn't short-circuit at trace time — both branches get traced,
and the True branch calls the Viterbi.

## The single remaining offender, in detail

### Site

[`bin/Language.py:1531-1546`](../../bin/Language.py) inside
`binary_tiling_viterbi`:

```python
for b in range(B):
    t = N
    while t > 0:
        kind = int(back_kind[b, t].item())      # DtoH
        op   = int(back_op[b, t].item())        # DtoH
        if kind == 0:
            copy_mask[b, t - 1, op] = 1.0
            t -= 1
        elif kind == 1:
            reduce_mask[b, t - 2, op] = 1.0
            t -= 2
        else:
            raise RuntimeError(...)
```

This is the **Viterbi backtrace** — sequential by construction
(each step's choice depends on the previous step's argmax via
`back_kind`/`back_op` chains). The forward DP pass above it
(lines ~1480-1530) is already vectorized; only the backtrace is
per-row Python.

### Why the guard doesn't help

The call site looks like:

```python
# Models.py, _per_word_body_step
if stm._max_depth_host >= stm.capacity:
    self._stm_bounded_reduce_step()      # → reducer(window)
                                          #   → binary_tiling_viterbi
    stm._max_depth_host = stm.capacity - 1
stm.push_step(idea_bd)
```

`stm._max_depth_host` is a plain Python int on the
`ShortTermMemory` nn.Module. Dynamo would normally specialize
this at the value it sees at trace time (0 on the first
forward, < capacity) and trace ONLY the False branch. But
[util.py:611-621](../../bin/util.py) explicitly opts in to
`_dyn.config.allow_unspec_int_on_nn_module = True` — a
deliberate decision to avoid recompiles when grammar
cursor/generation ints advance every batch. That config makes
nn.Module int attrs **dynamic SymInts**, so Dynamo can't tell
the guard's value at trace time and traces BOTH branches.

## Three resolution paths

### (a) Mask-based unconditional reduce — SMALLEST

Restructure the back-pressure so the Python `if` is gone.
Always call `_stm_bounded_reduce_step` per word; the function
already masks out rows where `depth < 2` via tensor ops (it's
designed to be a no-op for those rows). The Python compare
becomes a tensor mask.

- **Pro:** smallest code change; doesn't touch the Viterbi at
  all (because the reducer is unconditional, so the True
  branch is always taken — but it's the SAME branch each call,
  so Dynamo specializes once and is happy).
- **Con:** STILL hits the Viterbi. (a) ALONE doesn't fix the
  offender — it just removes the data-dependent branch. The
  Viterbi `.item()` per-row loop remains. **(a) must combine
  with (b) or (c).**

### (b) Vectorized Viterbi backtrace — CORRECT FIX

Replace the per-row Python loop with a tensor-vectorized
backtrace. Viterbi backtrace has a structural recurrence
(`t` decreases by 1 or 2 based on `kind[t]`), but it's
deterministic given the DP table — so it can be expressed
as a fixed-trip-count `for _ in range(MAX_STEPS):` over a
known maximum chain length (which is N, the window length).

- **Approach sketch:**
  - Replace the `while t > 0:` per-row loop with a single
    pass `for step in range(N):` (fixed trip count), tracking
    an active mask per row.
  - At each step, gather `back_kind[rows, t]` and
    `back_op[rows, t]` as tensors (no `.item()`).
  - Update masks via `torch.where` based on `kind == 0` /
    `kind == 1` (tensor compare).
  - Advance `t` per-row: `t = t - 1 - (kind == 1).long()`
    (subtract 1 normally, 2 on reduce).
  - Loop terminates when no row has `t > 0` — but for
    fullgraph we want a fixed trip count; the answer is to
    run N steps unconditionally and let the active mask
    zero out steps past termination.
- **Pro:** correct fix; makes the reducer fullgraph-clean,
  unblocks the entire whole-forward compile.
- **Con:** medium-effort algorithmic refactor; needs
  equivalence verification against the current
  implementation (Viterbi backtrace must produce
  byte-identical `copy_mask`/`reduce_mask` on a fixed input).

### (c) Sub-region split — STRUCTURAL

Move the back-pressure reduce OUT of the per-word body into
a separate captured region that fires only when STM overflows
(rare). Concretely:

- Keep `_per_word_body_step` as SHIFT-only (no reduce).
- After the per-word loop's body call, the eager Python
  scope checks `stm._max_depth_host >= stm.capacity` and
  fires `_stm_bounded_reduce_step()` outside the captured
  region.

This is the D8 spec's "OS-region" idea applied surgically:
the back-pressure reduce is an OS-graph operation, not a
middle-graph operation.

- **Pro:** removes the offender from the captured body
  entirely; the Viterbi never runs in a traced region.
- **Con:** changes the per-word loop structure (the Python
  loop in `_forward_body_per_word` now has more eager
  logic between iterations); needs a careful semantic
  check that the SHIFT-then-reduce timing is preserved.

### Recommendation

**(a) + (b)** is the right combination if you want the
production whole-forward compile fully green. (c) is the
right answer if you decide the bounded reduce is rare enough
that paying eager wrap cost is fine (it probably is — only
sentences with > stmCapacity=7 words trigger it).

For a single-session closeout, **(c) is the smallest
delivery**: deliberately move the reduce out of the captured
region. (b) is the principled fix that also unblocks
fullgraph capture of the reducer for OTHER consumers (e.g.,
the OS-graph if it ever uses the reducer directly).

## Code pointers (read these in order)

| Topic | File:line | Notes |
|---|---|---|
| The Phase-5 D8 design | `doc/plans/2026-05-18-two-loop-pipeline-architecture.md` § D8 | The three-graph capture strategy + the full landing record |
| Per-word body extraction | [`bin/Models.py:4924`](../../bin/Models.py) `_per_word_body_step` | One-arg method called per word in the loop |
| Per-word prelude (boundary setup) | [`bin/Models.py:4806`](../../bin/Models.py) `_per_word_prelude` | MPHF pre-warm, STM resize, WorkingState invariants |
| Per-word loop driver | [`bin/Models.py:5025`](../../bin/Models.py) `_forward_body_per_word` | The variable Python `while` loop |
| The offender call site | [`bin/Models.py:5019`](../../bin/Models.py) `_per_word_body_step` | `if stm._max_depth_host >= stm.capacity:` |
| The forced reduce | [`bin/Models.py:4631`](../../bin/Models.py) `_stm_bounded_reduce_step` | Designed for fixed-shape masked ops |
| The reducer ctor | [`bin/Models.py:4554`](../../bin/Models.py) `_stm_reducer` | Lazy-built; pre-warmed in `enable_compiled_step` now |
| **The Viterbi backtrace (offender)** | [`bin/Language.py:1531-1546`](../../bin/Language.py) `binary_tiling_viterbi` | Per-row Python `.item()` loop |
| The Viterbi forward DP (already vectorized) | [`bin/Language.py:1480-1530`](../../bin/Language.py) | Reference for the vectorization style |
| Reducer forward consumer | [`bin/Language.py:1862-1946`](../../bin/Language.py) `BinaryStructuredReductionLayer.forward` | Uses `hard["copy_mask"]`, `hard["reduce_mask"]` |
| Dynamo unspec config | [`bin/util.py:611-621`](../../bin/util.py) | Why the guard doesn't short-circuit |
| WorkingState (no `__slots__`) | [`bin/Spaces.py:3408`](../../bin/Spaces.py) | Plain `__dict__` attrs (Phase-5 change) |
| Phase 2B selection-tensor | [`bin/Language.py:5546`](../../bin/Language.py) `_populate_op_sel_from_default_rules` | Already landed; reference for Phase 2B style |
| The compile entry point | [`bin/Models.py:2122`](../../bin/Models.py) `enable_compiled_step` | Where the eager pre-warm lives |
| `enable_compiled_step` STM pre-warm | [`bin/Models.py:2159-2168`](../../bin/Models.py) | Reference pattern for hoisting lazy builds |

## Local test gates (must remain green)

```bash
cd basicmodel
.venv/bin/python -m pytest \
  test/test_per_word_capture_gate.py \
  test/test_input_word_cursor.py \
  test/test_idempotent_loop.py \
  test/test_per_word_stem.py \
  test/test_phase2a_labor_division.py \
  test/test_lift_lower_factorization.py -q
```

Expected: **38 passed**. The capture-gate suite includes the
4 fullgraph tests for `_per_word_body_step` in isolation.

If you change the reducer's behavior, also re-run the wider
suite for sanity:

```bash
.venv/bin/python -m pytest test/ -q --deselect \
  test/test_space_equiv_selfcheck.py
```

(`test_space_equiv_selfcheck` has a pre-existing autograd
"backward through the graph a second time" issue — not your
problem.)

## Metalbaby protocol (bounded runs only — OOM/reboot history)

### 1. Sync the working tree

From the WikiOracle root (parent of `basicmodel/`):

```bash
cd /Users/arogers/Library/Mobile\ Documents/com~apple~CloudDocs/bits/projects/WikiOracle
make sync HOST=mb
```

The parent `Makefile` has a `sync` target that rsyncs (excluding
`.venv`, `.git`, `__pycache__`, large data, etc.) over SSH to
`admin@metalbaby.local:~/WikiOracle/`. Time: a few seconds.

### 2. Clean the stale ckpt if needed

The model XML config and the saved checkpoint dimensions
diverge after architectural changes. If you see a "Weight
file mismatch -- cannot load" error, delete the stale ckpt
(the user pre-authorized this):

```bash
ssh admin@metalbaby.local "rm -f /home/admin/WikiOracle/basicmodel/data/MM_5M.ckpt"
```

### 3. Run the production training (one bounded run)

The user's exact invocation:

```bash
ssh admin@metalbaby.local \
  "cd /home/admin/WikiOracle/basicmodel && \
   timeout 540 .venv/bin/python bin/train.py \
     --model data/MM_5M.xml --data text \
     --num-epochs 1 --batches 100 \
     --compile-mode max-autotune --log 2>&1 | tail -50"
```

- `timeout 540` (9 minutes) is the hard ceiling. Per
  `bin/util.py` bench notes, `max-autotune` does **mean=30.9s
  min=24.9s max=42.0s** for 8 batches — so 100 batches is
  ~5-7 minutes well within budget.
- `| tail -50` keeps the output bounded; full logs are in
  `output/logs/train_<timestamp>.log` on the host.
- If the run succeeds, you'll see `[train] end:` with elapsed
  time; if it fails, you'll see a `torch._dynamo.exc.*` or
  `Unsupported` trace.

### 4. Run the CUDA capture-gate tests

A leaner check that doesn't need a full training run:

```bash
ssh admin@metalbaby.local \
  "cd /home/admin/WikiOracle/basicmodel && \
   rm -f data/MM_5M.ckpt && \
   timeout 480 .venv/bin/python -m pytest \
     test/test_per_word_capture_gate_cuda.py -q -s --tb=short 2>&1 | tail -40"
```

Three tests:
- `test_per_word_step_compiles_and_replays_under_cudagraphs` —
  expected: PASS (already verified in the 2026-05-19 session).
- `test_per_word_step_actually_uses_cudagraphs` — currently
  fails because `mode="reduce-overhead"` silently doesn't
  activate CUDAGraph capture. See task #34.
- `test_per_word_step_emits_no_dtoh_under_profiler` — currently
  fails (52 DtoH events); once CUDAGraphs activate (task #34),
  this should pass.

## Acceptance criteria for this plan

1. `binary_tiling_viterbi` either (a)+(b) becomes
   fullgraph-clean OR (c) is no longer reached from the
   per-word body's captured region.
2. Production `train.py --compile-mode max-autotune` runs end-
   to-end for **at least 100 batches** without `Unsupported`
   or `UserError` from Dynamo.
3. Local 38/38 targeted suite stays green.
4. No regression in `test_per_word_capture_gate_cuda.py::
   test_per_word_step_compiles_and_replays_under_cudagraphs`
   (the primary CUDA gate; already green).

## Stretch acceptance (task #34 territory)

5. `mode="reduce-overhead"` (or `max-autotune`) actually
   activates CUDAGraph capture — verifiable via
   `torch._dynamo.utils.counters["inductor"]` containing
   `cudagraph_*` keys with nonzero values.
6. `torch.profiler` reports **0 `Memcpy DtoH (Device →
   Pinned)` events** inside a single compiled replay.

## Debug tips

When Dynamo's `Unsupported` error fires, the helpful env vars
are:

```bash
TORCHDYNAMO_VERBOSE=1 \
TORCH_LOGS="+dynamo" \
TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u0" \
ssh admin@metalbaby.local "..."
```

`u0`, `u1`, etc. are unbacked SymInt names; setting the env
var prints the creation stack for that specific symbol.

For specifically debugging the Viterbi:

```bash
TORCH_LOGS="dynamic,recompiles" \
ssh admin@metalbaby.local "..."
```

The `dynamic` log shows when Dynamo creates unbacked SymInts;
`recompiles` shows guard failures.

## Standing constraints (carry forward)

- **Never write to git.** Read-only `git status` / `diff` /
  `log` are fine, but no `add` / `commit` / `rm` / `checkout`
  / `stash` / `push` / `worktree`. User does all git writes.
- **Metalbaby runs are bounded/single** (OOM/reboot history).
  Cap each run with `timeout 540` or shorter; never poll-loop
  metalbaby.
- **Fail loud on numerical divergence.** Don't add tolerances
  to existing exact-equivalence asserts to make tests pass;
  if the new (a)/(b) implementation diverges from Viterbi,
  the divergence is the bug, fix it at source.
- **No `getattr`/`hasattr`/`setattr` on the per-word path.**
  This was a 2026-05-19 cleanup; preserve it. If a default
  is needed, initialize the attribute in `__init__` and use
  direct access.
- **MM_5M.ckpt is regenerated by training runs.** Delete it
  freely if the architecture/saved widths diverge.

## What this plan is NOT solving

- The Phase 2B chart→`op_sel` encoder for non-default-only
  grammars (`MM_5M.xml` is default-only so it doesn't hit
  this; idempotent.xml is non-grammar so it skips the
  per-word body entirely).
- Per-op `reverse()` math for syntactic transforms
  (owner-pending; D4 in the master plan).
- The CUDAGraph capture fallback investigation
  (task #34 — separate concern).
- `test_space_equiv_selfcheck` autograd issue (pre-existing,
  unrelated to D8).
