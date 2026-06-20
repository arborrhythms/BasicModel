# 2026-05-13 — Sigma/Pi Rebalance + Byte-Direct Chunking handoff

## What landed

### Phase A — LiftLayer / LowerLayer become rule-id annotators

[bin/Layers.py:2303-2435](../../bin/Layers.py). The previous
"gated-substrate" pattern (LiftLayer borrowed `perceptualSpace.sigma`,
LowerLayer borrowed `conceptualSpace.pi`) is retired. Both layers now
compute a parameter-free static lattice op (`Ops._lower_kernel` for
lift, `Ops._lift_kernel` for lower) and the chart records the
`rule_id` on the surrounding parse cell. The lift-vs-lower distinction
("the boy runs" vs "the running boy") lives at the parse-tree /
rule_id level, not in a different substrate. Cognitive rationale and
tier table are documented in
[doc/Spaces.md §"Sigma / Pi ownership (2026-05-13 rebalance)"](../Spaces.md)
and
[doc/Language.md §"Lift / lower — rule-id annotators"](../Language.md).

`_gated_sigma` / `_gated_pi` helpers removed. Constructor signatures
keep `symbolicSpace` / `perceptualSpace` / `conceptualSpace`
parameters for API compatibility but ignore them.
[test/test_lift_lower_factorization.py](../../test/test_lift_lower_factorization.py)
updated: the "falls back to static kernel" tests renamed to "uses
static lattice kernel" — same behaviour, clearer semantics.

### Phase D — Byte-direct chunking (`<chunking>none</chunking>`)

[bin/Spaces.py:7432-7530](../../bin/Spaces.py) adds
`PerceptualSpace._embed_byte`. When the XML opts in via
`<chunking>none</chunking>`:

- Bytes from `InputSpace.subspace.what` are clamped to `[0, 255]`,
  remapped via two's-complement for int8 negatives, and used directly
  as codebook indices.
- The 256-entry byte codebook is the same `Embedding` instance used by
  the BPE path, but the cold-start path's `byte_value == codebook_index`
  alignment for the 0..255 range is now load-bearing (no merges grow
  past 256).
- `_bpe_word_mask` is derived as `(byte_indices != 0)` — a single
  tensor op, no Python BPE walker, no `torch.dynamo` graph break at
  the chunking boundary.
- `\0` (byte 0) doubles as the sentence-end / pad sentinel and lands
  at codebook index 0 by construction.

XSD allows `none` as a third value for `<chunking>` at
[data/model.xsd:343](../../data/model.xsd).
[InputSpace.forward](../../bin/Spaces.py) routes the new mode through
`peer._embed_byte(...)` in the same dispatcher that handles `bpe` /
`lexicon`. Downstream AR-unfold + mask reapply code accept the new
mode under the existing `_bpe_word_mask` contract; no consumer needs
to know which chunker fed the mask.

Phase D fall-through behaviour: when `chunking=none` and `nVectors <
256` or `modelType != embedding`, the constructor raises with a
specific error message rather than silently degrading.

## What did *not* land (deferred, per pacing agreement)

### Phase B — `ConceptualSpace.sigma_percept`

Replace `ConceptualSpace.self.pi` (square-iso percept_dim → percept_dim)
with `self.sigma_percept` (non-square percept_dim → concept_dim, the
canonical forward C-tier fold). Touches the per-word stem hot path in
4 sites in `bin/Models.py` and the C-tier forward at
`bin/Spaces.py:8622`. Dim change ripples through every downstream
consumer of `C.subspace.what`. Estimate: 2-3 hours plus a regression
cycle.

### Phase C — `PerceptualSpace.pi_input` + `pi_concept`

Restructure `PerceptualSpace.forward` so two distinct `PiLayer`s fire
unconditionally — `pi_input` on the IS argument
(input_dim → percept_dim), `pi_concept` on the C feedback
(concept_dim → percept_dim) — and their outputs are **summed** (no /2
averaging — the current `(primary + c_event) / 2` in `_sourced_input`
is retired). The current `PerceptualSpace.sigma` at concept_dim (which
Phase A made vestigial when LiftLayer stopped borrowing it) gets
deleted. Estimate: 2-3 hours.

### Phase B + C are coupled

`sigma_percept` and `pi_input`/`pi_concept` together produce the
canonical composition `C = sigma_percept(pi_input(IS) + pi_concept(C_prev))`.
Landing B without C (or vice versa) leaves the model dim-inconsistent.
Schedule them in the same session.

## Verification on metalbaby (GB10, 121 GiB unified)

### Architectural verification: is K gone in MM_5M?

**No — K is still present in the data flow.** Traced with
instrumented `InputSpace.forward` on MM_5M_bivector + chunking=none:

```
[InputSpace.forward] event=torch.Size([2, 32, 1024, 10]) k_axis=True
[PerceptualSpace.forward] in event=torch.Size([64, 1024, 10])
```

InputSpace still produces `[B, K=32, N=1024, D=10]` AR cursor windows,
and PerceptualSpace.forward sees the flattened `[B*K=64, N, D]` body
view. The serial stem (`perWordStemSerial=true`) walks one cursor at a
time in its accumulator but stages `[B*K, K, D_c]` snapshots back into
`STM` so the body can still process all K windows in parallel. The
`[B, V, D]` causal target requires retiring the AR unfold in
`InputSpace.forward` — see "Recommended optimization" below.

### Sentences/sec at the new code

| Configuration | bs | Steady per-batch | Sent/sec | Loss @ batch 10 |
|---|---|---|---|---|
| **MM_5M_bivector + BPE + serial stem + max-autotune** | 128 | ~3.5s | **~37** | 0.5675 |
| **MM_5M_bivector + byte-direct + serial stem + max-autotune** | 128 | ~22s | **~5.8** | 0.3641 |

Byte-direct is ~6× slower per batch at the same `B`. The reason is the
K-axis: a typical English sentence is ~60 bytes vs ~16 BPE chunks, so
the pow-2-bucketed K balloons from 32 to 128, the body sees 4× more
B*K rows, and the serial stem's K-loop runs 4× more iterations. The
graph break we eliminated (the BPE Python walker) was sub-millisecond
amortized; it was never the dominant cost at the K=32 bucket.

The per-step loss tells a different story — byte-direct reaches 0.36
in 10 batches where BPE was at 0.57 at the same point. The model
appears to learn faster per step from the wider input sequence, even
though wall-clock per step is higher. The right comparison is sent/sec
× loss-per-sec, not raw sent/sec.

## Recommended optimization (single item, per the handoff template)

**Retire the AR cursor unfold in `InputSpace.forward` when the serial
stem is enabled.** Make MM_5M run as `[B, V, D]` causally with no K
axis.

Current state:

```python
# InputSpace.forward, line 6175-6182
pad = torch.zeros(B, N, D, device=embedded.device, dtype=embedded.dtype)
padded = torch.cat([pad, embedded], dim=1)        # [B, T+N, D]
unfolded = padded.unfold(1, N, 1).permute(0, 1, 3, 2).contiguous()
# unfolded shape: [B, T+1, N, D]
```

This produces K cursor windows whose only purpose is to give the body
a `[B*K, N, D]` parallel view of the prefix at each cursor. Under the
serial-stem path the prefix is already walked sequentially in the
stem's K-loop; the body's parallel view is a redundant re-materialization
of the same prefix data K times.

The fix:

1. **Opt-in via XML flag**: `<inputUnfold>false</inputUnfold>` on
   `architecture` (default `true` for backward compat).
2. **When `inputUnfold=false`**: `InputSpace.forward` emits
   `[B, T, D]` (no unfold, no K axis, `k_axis=False`).
3. **Body `_forward_body` collapses to B**: `_flatten_k` returns
   `(B, 1)`; no `[B*K, …]` reshape; the stages process B rows directly.
4. **Per-cursor predictions**: the head fires inside the serial
   stem's K-loop (cursor-by-cursor), producing K predictions at
   `[B, K, output_dim]` — the same shape the AR-unfold path produces
   today, just built one step at a time.

Expected impact at bs=128 / K~128:

- **Memory**: the `[B, K, N, D]` tensor (`128 × 128 × 1024 × 10 × 2
  bytes = 320 MB`) and every B*K downstream tensor (STM snapshots
  scale at `B*K × K × D_c`) is replaced by `[B, T, D]` (~150 KB) and
  per-step B-shaped tensors. **~2000× reduction in the dominant
  activation slab.**
- **Wall-clock**: each serial-K step does `B × work_per_step` instead
  of `B*K × work_per_step / K`. Inductor's K-loop unrolling already
  collapses launch overhead within the stem (verified earlier in
  `bin/util.py` notes on `MODEL_COMPILE_MODE=max-autotune`); the body's
  3 stage passes are the unmoved cost. Net: should beat the BPE
  baseline (~37 sent/sec) by removing all the unfold-related
  redundant work.
- **Cognitive alignment**: this is what "[B, V, D] causally" means.
  STM becomes the sentence-lifetime accumulator (B rows, not B*K),
  and the chart at C consumes a real prefix-evolving state rather
  than K independent reconstructions.

The risk is the head's per-cursor firing: today the head sees one
[B*K, …] slab and emits K predictions in parallel; under the new
shape it has to fire K times serially inside the K-loop. Each call is
cheap (B × output_dim matmul); under torch.compile + max-autotune the
loop unrolls so the K calls fold into a single CUDAGraph.

Estimated effort: ~1 day of focused work. The key files are
`bin/Spaces.py` (`InputSpace.forward` unfold), `bin/Models.py`
(`_flatten_k` / `_restore_k` collapse to identity when K=1, head call
site in the stem loop), and the XML flag plumbing.

## How to reproduce the benchmark numbers

```bash
ssh admin@metalbaby.local
cd ~/WikiOracle/basicmodel

# BPE baseline (37 sent/sec)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_COMPILE_MODE=max-autotune \
PYTHONPATH=bin .venv/bin/python bin/train.py \
    --model data/MM_5M_bivector.xml --data text --num-epochs 1 --batches 10

# Byte-direct (5.8 sent/sec, loss decays faster)
python3 -c "
with open('data/MM_5M_bivector.xml') as f: c = f.read()
c = c.replace('<chunking>bpe</chunking>', '<chunking>none</chunking>')
c = c.replace('<batchSize>16</batchSize>', '<batchSize>128</batchSize>')
open('/tmp/MM_5M_bv_byte.xml', 'w').write(c)
"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_COMPILE_MODE=max-autotune \
PYTHONPATH=bin .venv/bin/python bin/train.py \
    --model /tmp/MM_5M_bv_byte.xml --data text --num-epochs 1 --batches 10
```

## Files touched

- [bin/Layers.py](../../bin/Layers.py) — LiftLayer / LowerLayer rewrites
- [bin/Spaces.py](../../bin/Spaces.py) — chunking=none validation,
  `_embed_byte`, dispatcher updates, `_bpe_word_mask` semantics under
  byte mode, post-VQ mask reapply for `none`
- [data/model.xsd](../../data/model.xsd) — `<chunking>` enum accepts `none`
- [doc/Spaces.md](../Spaces.md) — sigma/pi ownership section, chunking
  modes discussion
- [doc/Architecture.md](../Architecture.md) — spaces table update,
  composition formula
- [doc/Language.md](../Language.md) — lift/lower as rule-id annotators
- [test/test_lift_lower_factorization.py](../../test/test_lift_lower_factorization.py)
  — "uses static kernel" tests
