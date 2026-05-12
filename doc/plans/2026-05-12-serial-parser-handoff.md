# Serial / Shift-Reduce Parser — Handoff Spec

## Motivation

The chart parser today is batched CKY at S-tier — it fills all spans of a sentence in one shot, with rule selection driven by per-cell rule probabilities. The architecture has evolved to a place where this no longer fits:

- `ConceptualSpace.stm` now exists as a per-batch stack of unquantized C-tier ideas, designed for the parser's working memory but currently unused.
- The C→P subsymbolic loopback runs once per forward call (cross-sentence cadence), but the architecture demands **per-word** subsymbolic processing: each word needs its own round trip P → C → S → C → stm.push before the parser can reduce it.
- Lift/Lower have been refactored as gated substrate operations at C-tier, so the chart's grammar dispatch naturally pulls toward C.
- Linguistic 7±2 STM cap is a real forcing function on reduce decisions, which the serial parser respects naturally.

This pass replaces the batched CKY with a shift-reduce parser driven word-by-word, moves the chart's launch site from S to C, and adds a FIFO dispatch buffer from C to S for the operator-as-symbol relations (`query`, `equals`, `part`). S becomes the calculator: it responds to evaluation requests from C and accumulates results in the persistent TruthLayer.

## Pre-existing state to inherit

This work builds on a series of refactors that have already landed in the worktree (uncommitted). Critical state:

| Item | Where | What it gives the parser |
|---|---|---|
| C→P sigma-input loopback | `PerceptualSpace._sourced_input` ([Spaces.py:7307](../../bin/Spaces.py)) | Substrate iteration with mask gating; bivector-gated. The parser drives this per word. |
| SubsymbolicSpace removed | (class gone from Spaces.py) | P is the subsymbolic substrate; one less indirection. |
| Lexicon API on S | `S.vocabulary` + 7 orthographic methods, delegating to P's physical Embedding | Per-word codebook snap returns `(word_id, POS)` simultaneously via `category_ids` / `category_logits` on the bivector codebook. |
| MereologicalTree retirement | Class deleted; `PartLayer` / `EqualsLayer` / `QueryLayer` rewritten on codebook geometry ([Layers.py:3156-3489](../../bin/Layers.py)) | Pure-geometric relational ops; no sidecar to maintain. |
| Subsymbolic/symbolic split | Grammar XML lists S-tier rules; per-tier SyntacticLayer at P/C is retained as backward-compat no-op | Today's grammar is S-tier; this pass moves the active firing site to C. |
| Lift/Lower factorization | `LiftLayer.forward(VP, NP) = sigma(VP_c * NP_c)`; `LowerLayer.forward(VP, NP) = pi(VP_c * NP_c)` ([Layers.py:2455](../../bin/Layers.py)) | The chart fires these as binary C-tier rules; substrate sigma/pi handle the math; no `raw_gate` parameter. |
| ShortTermMemory on C | `ConceptualSpace.stm` ([Spaces.py:8126](../../bin/Spaces.py)) | Per-batch idea stack; capacity 9 default (7±2); cleared on hard Reset. **No consumer yet — this pass is the first.** |

Approximately 700+ tests pass at handoff. The only pre-existing failures (`test_invertibility::TestPerceptualSpaceReverseRangeCheck::test_roundtrip_output_in_range` and `test_mm_boolean::test_encode_decode_by_best_fit`) are test-ordering pollution and training-quality flakes respectively — both unrelated to this work.

## The cadence asymmetry — why per-word processing is the blocker

The two feedback loops have different iteration cadences in today's code, and **this is a blocker for the serial parser as-is**:

| Loop | Iterations per forward call | Governed by |
|---|---|---|
| Symbolic (S→C) | `T = conceptualOrder` (per stage) | `<conceptualOrder>T</conceptualOrder>` |
| Subsymbolic (C→P, new) | **1** (stem P reads `conceptualSpaces[-1]` from the *prior* call) | Cross-forward only |

For a single word to be identified as a single symbol on the STM, each word must be lexed (BPE), projected to C, snapped at S (codebook hit = word identity + POS), and brought back to C as an unquantized idea — *independently of other words in the sentence*. Today's batched forward processes all positions in the window simultaneously; the serial parser will drive a per-word forward path with its own iteration count (see Q6 below) that does NOT inherit from `conceptualOrder`.

## Operational flow (per word, target architecture)

```
byte stream
  │
  ▼
P (BPE lex via ChunkLayer; per-percept features; SigmaLayer if activated)
  │
  ▼
C (project the lexed word onto concept space; STM is NOT pushed yet)
  │
  ▼
S (codebook snap = word identity via Codebook.forward(project=True);
   POS rides the hit via category_ids / category_logits — single op,
   no separate POS tagger)
  │
  ▼
C (S→C reverse via Codebook.reverse(project=True) — the unquantized "idea"
   form, recovered from the bivector)
  │
  ▼
ConceptualSpace.stm.push(idea)   ← idea is now on the parser's stack
  │
  ▼
Chart-at-C decides: REDUCE adjacent STM items (grammar-greedy + STM-cap
pressure) OR SHIFT (consume next word)
  │
  ▼ (when REDUCE)
  ├─ If rule is a C-tier op (union, intersection, lift, lower, ...):
  │     parser pops operands, runs GrammarLayer.forward, pushes result idea
  │
  └─ If rule is an S-tier operator-as-symbol op (query, equals, part):
        parser pops operands, pushes op record onto C→S dispatch FIFO;
        S drains and evaluates; truth value returns as METADATA on the
        operand idea(s) in STM (does NOT push a new idea)
```

The parser's `shift` consumes one BPE chunk and runs the round trip above. The parser's `reduce` pops adjacent STM items and dispatches the appropriate rule. The cycle ends when the BPE chunker is exhausted OR a `<start>` reduction fires (existing `WordSpace._sentence_completed` signal); hard Reset clears STM on sentence boundary.

## Resolved design decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Reduce-decision policy | **Grammar-greedy with STM-cap pressure override.** Reduce eagerly when STM has a stack-adjacent pair the grammar can type-compatibly reduce (RHS-typed pair, including POS from the codebook hit). At `capacity - 1` with no clean reduce, force a reduce on the best-scored candidate. Otherwise shift. | Classical shift-reduce + STM cap as safety valve. Honors "the cap is a forcing function" without giving up grammar-driven structure. |
| 2 | Dispatch-buffer ordering | **Per-step batched drain; asynchronous with 1-step latency.** All dispatch records queued during step `t`'s chart reductions drain together at step `t+1`'s start as one batched call. Within a step the records are unordered (no FIFO semantics needed — they evaluate in parallel). Results arrive one step later than the queue. See "GPU dispatch wire format" below for the tensor shape. | Aligns with the GPU's preferred shape: one batched kernel per step rather than per-record sync. The 1-step latency is structurally similar to a pipelined CPU and bounded by the sentence-length outer loop (O(N) max dependency depth). |
| 3 | Truth-value attachment | **Metadata on the operand idea(s).** A per-idea field stored alongside the C-shape activation on STM. The chart's continuation reads the tag; reductions can branch on it. Truth tags do NOT push new ideas. | Avoids stack pollution. Mirrors TruthLayer's per-idea pattern but in-flight on STM. Avoids the "is yes/no an idea?" semantic question — it's a *property* of ideas. |
| 4 | Lift/Lower invocation | **Stay as grammar rules, fired by the chart-at-C.** Parser pops two STM-adjacent ideas, runs `LiftLayer.forward` / `LowerLayer.forward` (already factor through substrate sigma/pi via the codebook-gate mechanism — they use codebook *geometry* but NOT codebook *lookup*), pushes the result idea back to STM. No S-dispatch for lift/lower. | Lift/Lower compose ideas; they don't need codebook lookup to identify a specific symbol. Geometric use of the codebook (SVD pinv for the bivector lift) is not the same as a symbol identification. |
| 5 | Backward compatibility | **Flag, then flip.** Introduce `<chart><serialParser>true</serialParser></chart>` in the model XML. Default initially `false` (preserves batched CKY for migration). Tests for the serial path use a config that flips the flag. After the serial path proves itself across the regression suite, flip the default to `true` and retire the batched CKY in a final cleanup pass. | Two-step migration: prove correctness side-by-side; then collapse to one code path. The chart-launch-site move (S → C) and the parser switch both ride this flag. |
| 6 | Per-word P→C iteration count | **Fixed N iterations per word, configurable via `<serialParser><iterationsPerWord>N</iterationsPerWord>`; default N=1.** Each iteration runs one P→C cycle with the C→P loopback active (reading the prior iteration's terminal C). | Convergence detection adds complexity and runtime cost. Until-codebook-hit needs a confidence threshold and edge handling for OOV. Fixed N with N=1 matches the existing per-stage cadence; the knob lets us tune upward if needed. `<conceptualOrder>` remains a separate legacy knob for batched-stage configs. |
| 7 | Operation-by-tier classification | **At C (chart-fired, no codebook lookup):** `lift`, `lower`, `union`, `intersection`, `conjunction`, `disjunction`, `not`, `non`, `swap`, `copy`, `true`, `false`. **At S (calculator dispatch, operator-as-symbol):** `query`, `equals`, `part`. | The distinction is whether the operation needs **codebook lookup** (identifying a specific symbol by ID) — not whether it touches codebook geometry. Lift/Lower use codebook *geometry* but don't *look up* a specific symbol → C. Query / equals / part *do* depend on which specific symbols are involved → S. |

## Architecture after the pass

```
        BPE chunker (one word per shift)
                  │
                  ▼
P (subsymbolic substrate) ←──── C event (C→P loopback per iteration)
                  │
                  ▼
C (host: STM + chart + parser) ──→ S calculator (operator-as-symbol ops)
                  │                       │
                  ▼                       ▼
              STM stack                 S evaluates,
              of ideas                  truth returns as metadata
                                        on the operand idea
```

The chart-at-C operates over STM-adjacent ideas. Each `shift` triggers one per-word round trip (P → C → S codebook snap → C reverse-lift → stm.push). Each `reduce` either fires a C-tier rule (consuming two STM items, producing one) or dispatches an S-tier operator-as-symbol op (returning a truth tag that attaches to operand ideas on STM). The persistent TruthLayer on S accumulates the log of evaluations (LTM).

## GPU loop structure & dispatch wire format

The parser is a fixed-depth loop bounded by sentence length, with parallel work per step:

```
for t in range(sentence_length):              # outer loop: depth-N (N = sentence length)
    # parallel within the step:
    #   subsymbolic phase: queue lookup dispatch for word_t (arity 1)
    #                      (P → C project happens in this phase too)
    #   symbolic drain:    process all dispatch records queued at step t-1
    #                      (one batched kernel call)
    # both phases finish before step t+1 begins
    #
    # chart-at-C decides reductions based on STM after both phases;
    # new dispatches (lookups + ops) queue for step t+1
```

Each step is one synchronization point. Cross-step dependencies (a chart decision that needs an S result) carry a 1-step latency. Total dependency depth is O(sentence_length); structurally similar to Transformer training (fixed-depth, parallel per-layer).

**Dispatch buffer wire format.** A single uniform-shape tensor per step, zero-fill in unused slots:

```
Shape: [B, M, stmCapacity, D]

  B            = training batch dim
  M            = number of dispatch records queued at this step
                 (variable per step; K_ops + K_lookups in the buffer)
  stmCapacity  = the STM's capacity from <ConceptualSpace><stmCapacity>N</stmCapacity>
                 (default 9; the same parameter that sizes ShortTermMemory).
                 Slot 0 is the operator symbol (or zero sentinel for lookup mode);
                 slots 1..(stmCapacity-1) are operands or vectors-to-quantize.
  D            = concept_dim (the C-tier per-vector width)
```

Per record (each row in `M`):

| Operation | Slot 0 | Slot 1 | Slot 2 | Slots 3..(stmCapacity-1) |
|---|---|---|---|---|
| Word lookup | `0` (sentinel) | word vector | `0` | `0` |
| Batched lookup | `0` (sentinel) | word_t | word_t+1 | additional words... |
| Unary op (e.g. `not`) | op symbol | operand | `0` | `0` |
| Binary op (e.g. `query`) | op symbol | A | B | `0` |
| Higher-arity (future) | op symbol | A | B | C, D, ... |

S's per-step kernel:
1. Inspect slot 0 per record. If zero → lookup mode: snap each non-zero slot to the codebook, return `(word_id, POS)` per slot via the metadata channel.
2. If non-zero → op mode: snap slot 0 against the op-symbol codebook to get op identity; invoke the corresponding GrammarLayer (`NotLayer`, `QueryLayer`, etc.) with the non-zero operand slots.
3. Either way, results return via the out-of-band metadata channel (operand_refs / result_slot bookkeeping).

The `stmCapacity` choice gives a fixed upper bound that aligns with the linguistic STM cap (default 9, room for the classical 7±2). Future ops with higher arity fit without re-shaping the buffer. Zero-fill works mathematically because zero is identity in atanh-space — unused operand slots drop out of any sigma/pi fold.

**Why uniform shape over per-arity sub-buffers.** Earlier drafts proposed three sub-buffers (lookup, unary, binary). The uniform `[B, M, stmCapacity, D]` form trades ~50% bandwidth waste on the common small-arity dispatches for a much simpler kernel structure: one tensor, one kernel call, no Python-side dispatch table. For GPU dispatch this trade is correct — kernel-launch overhead and Python bookkeeping dominate the tiny bandwidth savings.

## Implementation outline

### Step 1: Add the `<serialParser>` XML knobs and runtime flag

- `<chart><serialParser>true|false</serialParser></chart>` — enables the serial parser path; default `false` initially.
- `<serialParser><iterationsPerWord>N</iterationsPerWord></serialParser>` — default `1`.
- Add a runtime flag on `Chart` / `WordSpace` that the model forward checks.

### Step 2: Build the per-word forward driver

- Introduce a per-word entrypoint in `Model.forward` (or `_forward_per_stage`) that's gated by the serial-parser flag.
- Loop over BPE chunks from the input; for each chunk, run the per-word flow: P (lex+features) → C (project) → S (codebook snap) → C (reverse-lift) → `ConceptualSpace.stm.push`.
- Reuse the existing `serial_mode` warm-cache machinery on PerceptualSpace ([Spaces.py:7300+](../../bin/Spaces.py)) for the per-position sliding.

### Step 3: Build the chart-at-C dispatcher

- Add a parser driver (call it `SerialParser`?) inside `WordSpace` or as a sibling. Its API:
  - `shift(b)` — drive one per-word round trip; STM gets a new top item.
  - `reduce(b)` — pop STM-adjacent items; fire the rule; push result OR attach truth-tag.
  - `step(b)` — pick shift vs reduce per Q1's grammar-greedy + STM-cap policy.
- Rule classification per Q7: C-tier rules fire directly; S-tier rules push to the dispatch buffer.

### Step 4: Add the C→S dispatch buffer (per-step batched, the calculator)

- New per-batch buffer on `WordSpace` (or `ConceptualSpace`) sized `[B, M, stmCapacity, D]` per the wire format above. `M` grows as records queue during step `t`'s chart reductions; the whole buffer drains at step `t+1`'s start.
- `SymbolicSpace.evaluate_batch(buffer)` is the per-step batched kernel: snaps slot 0 of each record to determine mode (lookup vs op), invokes the matching path, returns a results tensor that the parser fans out to the operand idea refs (out-of-band metadata channel).
- Per-record bookkeeping: `(operand_idea_refs, result_slot)` for the metadata channel — tells the parser where to attach the lookup `(word_id, POS)` or op truth-tag once results return.
- The 1-step latency is built into the buffer's lifecycle: step `t` queues; step `t+1` drains and writes results back to STM idea metadata.

### Step 5: Wire sentence-boundary

- Existing `WordSpace._sentence_completed` signals when a `<start>` reduction fires.
- On signal: hard-Reset the per-row STM (already in place at [Spaces.py:8296](../../bin/Spaces.py)); drain any pending dispatch ops.

### Step 6: Migration / parallel-path correctness

- The batched CKY remains the default path while `<serialParser>false</serialParser>` (default). Tests with the new flag exercise the serial path.
- New tests:
  - `test_serial_parser_per_word_round_trip` — verify each word lands as one idea on STM with the right `(word_id, POS)`.
  - `test_serial_parser_grammar_greedy` — verify reduce vs shift decisions match Q1.
  - `test_serial_parser_dispatch_buffer` — verify S receives ops in FIFO order; truth tags land on operand ideas.
  - `test_serial_parser_stm_cap_pressure` — verify forced-reduce at `capacity - 1`.
  - `test_serial_parser_sentence_boundary` — STM cleared on hard Reset.

### Step 7: Default flip + cleanup

- Once the serial path is green across regression + new tests, flip `<serialParser>` default to `true`.
- Retire the batched-CKY internals in `Chart` (the `_chart_inside` / `_chart_outside` machinery and the per-tier rule probability gate for the batched path). Keep `Chart` as a thin shell that the serial parser uses for rule-lookup metadata (grammar table, rule probabilities for selection).

## Files & line numbers to read in Phase 1 (next-conversation exploration)

| File | Lines / range | What's there |
|---|---|---|
| [Language.py](../../bin/Language.py) | `Chart` ~2015; `WordSpace.compose` ~4500s; `SyntacticLayer.forward` 4002; `_attach_per_space_syntactic_layer` 5380; the natural-fold rule machinery 5057-5142 | Today's batched-CKY parser, the per-tier rule probability gate, the grammar-dispatch glue. |
| [Models.py](../../bin/Models.py) | `_forward_stem` 4817; `_forward_body` 4844; `_forward_per_stage` 4930 | The per-stage forward pipeline that the serial parser will replace (under the flag). |
| [Spaces.py](../../bin/Spaces.py) | `ShortTermMemory` ~7853; `ConceptualSpace.stm` wired at ~8126; `ConceptualSpace.Reset` STM clear at ~8296; `serial_mode` warm cache ~7300+ | The structural slots the parser will consume. |
| [Layers.py](../../bin/Layers.py) | `LiftLayer` / `LowerLayer` 2455-2545; `PartLayer` / `EqualsLayer` / `QueryLayer` 3156-3489 | The grammar layers the parser dispatches into. |

## Known limitations / future follow-ups

- **OOV handling.** When a word doesn't have a confident codebook hit (`best_sim < threshold` in `_apply_codebook_pos_seed`), the nearest match still fires but with low POS confidence. The parser should either soft-fail (skip the word and continue) or use a special UNK token. Decide during implementation.
- **Look-ahead.** The Q1 policy is greedy + cap. A 1-token look-ahead (peek next BPE chunk's POS before deciding shift vs reduce) is a natural future enhancement, but not in scope here.
- **Lift/Lower reverse path.** Today lossy `(parent, parent)` pseudo-inverse (gate is non-invertible without VP). If reconstruction-loss training needs the exact inverse, the path is to pass VP to `LiftLayer.reverse` as a hint and divide it out before invoking the LDU reverse. Not in scope here.
- **Multi-word reductions at C-tier.** Today's grammar rules are unary or binary. Lift/Lower take two operands. Some natural-language constructs may want >2-ary composition (e.g., a 3-way coordination). Stay binary on first pass; chain reductions for higher-arity constructs.
- **K (AR-time microbatching) × `[B, ...]` semantic state.** The existing batched pipeline uses an AR microbatch dimension `K` to pipeline multiple within-document positions in one forward call (the `FlattenKWrapper` machinery). The semantic state under the serial parser — `ConceptualSpace.stm`, the dispatch buffer, the per-idea truth-tag — is indexed by `[B, ...]`, NOT `[B, K, ...]`. Concretely:
  - One STM instance per training-batch row `b`; it accumulates ideas *across* the K positions of one sentence.
  - K parallel AR positions for the same `b` would all push to the same STM, creating a recurrence dependency that can't be K-parallelized (each k depends on the STM state from k-1).
  - **The serial parser is fundamentally K=1 within a forward call.** The "outer time" of the parser IS the AR position; the parser advances one word per forward. We give up the AR microbatch's pipelining for serial-parser correctness.
  - Alternative we could pursue later if K>1 matters for training throughput: extend STM to `[B, K, capacity, D]` with an explicit recurrence (each k reads STM from k-1 and produces STM for k). This is a Transformer-like causal-mask pattern over K; possible but complicates the per-step dispatch buffer (which would also need a K-axis). Out of scope for the first parser pass.
  - The dispatch buffer's `M` axis (number of records per step, in the wire format) is independent of K and unaffected — it's parallel ops within ONE step for ONE training-batch row, not across K AR positions.

## Worktree state to inherit

- Branch: detached HEAD at basicmodel commit `556949b` ("use two-input form of ConceptualSpace") with the diffs from the prior conversation on top.
- Nothing committed yet (per the git-policy memory).
- Working tree at: `WikiOracle/.claude/worktrees/brave-visvesvaraya-6eed21/basicmodel`.
- Use `basicmodel/.venv/bin/python` (Python 3.12 + torch 2.10) for tests. Run via `BASICMODEL_DEVICE=cpu PYTHONPATH=bin <venv>/bin/python -m pytest test/<file>`.
- For training, sync to `metalbaby.local`: `make sync HOST=mb` then `make test HOST=mb`.

## References

- The prior-pass plan file at `~/.claude/plans/please-read-the-memory-zesty-leaf.md` has the full implementation history and the design discussion that produced these decisions.
- `doc/Architecture.md` describes the architectural state as implemented (including the "Per-word operational flow (planned)" section that references this handoff).
- `doc/Language.md` has the grammar / chart / parser details and points at this handoff via the deferred-parser callout.
- `doc/Spaces.md` documents `ShortTermMemory` and the Lift/Lower factorization that the parser will consume.
