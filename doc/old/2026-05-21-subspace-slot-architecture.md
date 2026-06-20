# SubSpace Slot Architecture

**Status:** spec — describes the intended architecture, including the
target end-state for per-batch content storage. The current code is
partially in this design and partially on a transitional band-aid
(`_active_payload`); see "Current state vs target" at the bottom.

---

## Slots

A `SubSpace` owns five Basis slots and one selection / activation
buffer. Each slot has a fixed role:

| Slot          | Role                                                                                                                | Storage                                                                                |
|---------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `.what`       | "Content" dictionary (codebook prototype matrix, embedding table) for the what-axis content; OR plain per-batch buffer when no codebook is configured. | `Codebook` / `Embedding` / `ProjectionBasis` with `[V, nWhat]` Parameter; or plain `Tensor`. |
| `.where`      | Positional encoding basis. Usually a deterministic encoding (sinusoidal / byte-offset).                              | `Tensor` with `nWhere` width per slot.                                                  |
| `.when`       | Temporal encoding basis. Usually deterministic.                                                                      | `Tensor` with `nWhen` width per slot.                                                   |
| `.event`      | Muxed-event slot for the full `[nWhat + nWhere + nWhen]` row. For muxed codebook-bearing spaces this slot itself holds the codebook (the muxed prototype matrix); otherwise a plain `Tensor` cache of the muxed per-batch event. | `Codebook` (muxed mode) or `Tensor`.                                                    |
| `.activation` | Per-position selection / weight: indices, soft weights, or signed Degree-of-Truth scalars produced by the codebook snap. | `Tensor` with `activeEncoding.nDim` width per position (often `1` or `2`).              |

Plus `_active` (an integer index tensor `[B, N, M]` written by
`set_forward_content` when the inputs are already indices rather than
vectors).

### Codebook placement (the muxed / unmuxed split)

Two valid configurations for where the codebook prototype matrix lives:

* **Unmuxed** (e.g. `SymbolicSpace`): codebook on `.what`. The `[V, nWhat]`
  prototype matrix is `.what.W`. `.event` is a plain `Tensor` used only
  as a muxed-view cache.
* **Muxed** (e.g. `PerceptualSpace` for codebook-bearing configs like
  MM_xor / MM_5M): codebook on `.event`. The `[V, muxedSize]` prototype
  matrix is `.event.W`. `.what` may still carry an `Embedding` (lexicon
  lookup) but is not the codebook; the muxed event is the entity that
  gets quantized.
* **Pure-event** (e.g. `ConceptualSpace`): no codebook. `.event` is a
  plain `Tensor` holding the per-batch muxed content directly. `.what`
  / `.where` / `.when` are empty.

The rule of thumb: **exactly one slot per SubSpace holds the codebook
prototype matrix** (if any); the other slots are deterministic
encodings or plain buffers.

Survey of live configs:

| Space                              | `.event`   | `.what`    | Codebook lives on |
|------------------------------------|------------|------------|-------------------|
| `InputSpace`                       | `Tensor`   | `Tensor`   | none              |
| `PerceptualSpace` (MM_xor, MM_5M)  | `Codebook` | `Embedding`| `.event` (muxed)  |
| `PerceptualSpace` (MM_grammar)     | `Tensor`   | `Embedding`| `.what` (unmuxed) |
| `ConceptualSpace`                  | `Tensor`   | `Tensor`   | none (pure event) |
| `SymbolicSpace`                    | `Tensor`   | `Codebook` | `.what` (unmuxed) |
| `OutputSpace`                      | `Tensor`   | `Tensor`   | none              |

---

## Per-batch content: the target design

**The per-batch event tensor `[B, N, D]` is NOT stored.** It is the
return value of `materialize(...)`, reconstructed on demand from:

1. The codebook prototype matrix (`[V, D]` on `.what.W` or `.event.W`
   depending on placement), AND
2. The per-batch selection (`[B, N]` indices, or `[B, N, M]` index
   tuples, or `[B, N]` soft weights) stored on `.activation` (or, for
   index inputs, on `_active`).

```
materialize() -> [B, N, D]   ==   codebook.W[selection]
```

This means:

* `setW(per_batch_tensor)` on a Parameter-bearing slot is **architecturally
  wrong** in the target design — the per-batch content is not the
  prototype matrix, it's a function of selection + prototype. The
  correct write is to populate `.activation` (or `_active`) with the
  selection.
* `getW()` on a Parameter-bearing slot **always** returns the prototype
  matrix `[V, D]`. Per-batch shapes never appear on `.W`.
* `materialize(...)` is the single public read API for per-batch content.
  It handles the lookup; callers don't disambiguate by `ndim`.
* `materialize(mode='what')` / `materialize(mode='where')` /
  `materialize(mode='when')` slice the muxed view.
  `materialize(mode='event')` is the full muxed tensor.
  `materialize(mode='active')` (default) applies the per-position
  activation gate as well.
* Some operations only need the selection itself (e.g., a parse step
  reading "which symbol is here?" doesn't need the vector content).
  Those read `.activation` directly and **skip materialize** — the
  whole point of separating selection from content.

### Why this is cleaner

* No shape disambiguation downstream. Every read goes through
  `materialize`; every write goes through a setter that knows whether
  the target is "selection" (`.activation` / `_active`) or "raw event"
  (`.event` for pure-event subspaces).
* The codebook prototype matrix is unconfounded. `getW()` is a single
  semantic: "give me the prototype".
* Checkpointing is automatic. Only `.W` Parameters survive
  `state_dict()`; per-batch payloads are reconstructed on load.
* Bit-identical Dynamo capture: no `ndim`-branching means no
  shape-dependent control flow inside the traced region.

### Forward / reverse / chart contracts

* **Forward (codebook-bearing slot):** input is a per-batch vector
  `[B, N, D]`. The codebook snap (`Codebook.forward` / `ProjectionBasis.forward`)
  takes the vector, computes the selection (indices / soft weights /
  signed DoT), and writes the selection to `.activation` (or `_active`).
  The forward's *return value* is the snapped per-batch tensor (for
  callers that need it inline); but the SubSpace persists only the
  selection.
* **Reverse (codebook-bearing slot):** input is selection content
  (or, for the legacy "snap a raw event" path, a per-batch vector).
  `Codebook.reverse(y)` snaps `y` to the nearest prototype and returns
  the snapped vectors. The snap target is `self.W` (the prototype
  matrix); the legacy `getW()` read (which fell through to
  `_active_payload`) was the bug class fixed 2026-05-21.
* **Chart / parse ops:** read selection directly off `.activation`.
  No materialization. Selection is the atomic unit of meaning at
  this layer.
* **Tail consumers (loss, OutputSpace):** call `materialize(...)` once
  to get the per-batch tensor.

### Setter API

| Setter                                 | What it writes                                                                            |
|----------------------------------------|--------------------------------------------------------------------------------------------|
| `set_forward_content(indices, ...)`    | Selection indices `[B, N, M]` on `_active`. Use when forward stages have computed indices already (e.g., from text lex). |
| `set_activation(scalar)`               | Selection / weight `[B, N]` on `.activation`. Use when the snap produced a signed Degree-of-Truth or soft weight per position. |
| `set_event(per_batch_muxed)`           | **Pure-event subspaces only** (`.event` is plain `Tensor`). The muxed per-batch `[B, N, D]` lands on `.event.W`. Illegal for muxed codebook-bearing slots (use a forward snap to populate `.activation` instead). |
| `set_what(per_batch_what)`             | **Unmuxed Tensor-backed `.what` only**. The per-batch what slice `[B, N, nWhat]` lands on `.what.W`. Illegal for codebook-backed `.what` (use a forward snap to populate `.activation`). |
| `set_where(...)` / `set_when(...)`     | Per-batch encoding slices. Usually deterministic, written by `WhereEncoding.encode` / `WhenEncoding.encode`. |
| `set_demuxed(what, where, when)`       | Split-and-store helper. Same routing rules as the individual setters apply. |
| `set_muxed(event)`                     | Pure-event muxed write. Same constraints as `set_event`. |

### Reader API

| Reader                                       | What it returns                                                                          |
|----------------------------------------------|-------------------------------------------------------------------------------------------|
| `materialize(mode='event'\|'active')`        | Per-batch muxed event `[B, N, D]` reconstructed from selection + codebook (or read from `.event.W` for pure-event subspaces). `'active'` additionally applies the activation gate. |
| `materialize(mode='what'\|'where'\|'when')`  | Per-modality slice of the muxed event.                                                    |
| `materialize(mode='activation')`             | The stored activation / selection presence (not the per-batch vector content).            |
| `resolve()` / `resolve_pure()`               | Derived signed Degree-of-Truth scalar (`pos - neg`).                                       |
| `.activation.getW()`                         | Direct read of the selection / weight tensor. Used by parse / chart ops that need the selection itself, not the looked-up content. |
| `.what.getW()` / `.event.getW()`             | **Codebook prototype matrix `[V, D]`** (when the slot is codebook-backed). NOT per-batch content. Used by `Codebook.reverse`, SVD factorization, similarity probes, etc. |
| `.where.getW()` / `.when.getW()`             | Static encoding bases (rarely consumed directly).                                          |

The Basis `setW` / `getW` surface is internal to the SubSpace's
storage layer; downstream code should treat the SubSpace as the
public API.

---

## Current state vs target

The current code retains an `_active_payload` field on
`Tensor` / `Codebook` / `ProjectionBasis`. When `setW(per_batch_tensor)`
arrives at a Parameter-bearing slot, the tensor is routed to
`_active_payload` so the Parameter is not overwritten. `getW()`
returns `_active_payload` when set, else the Parameter — the single
slot multiplexes prototype matrix and per-batch payload.

This is a **band-aid** that lets the current forward paths
(`Codebook.forward` writing into `subspace.event.setW(x)` via
`set_event`, `PerceptualSpace._embed_*` paths, ModalSpace routing,
demux from a muxed event, …) work without crashing the Parameter
or the checkpoint. The bug that motivated the spec — `Codebook.reverse`
reading `getW()` and getting back a stale 3-D `_active_payload`
instead of the 2-D prototype — was fixed by having `Codebook.reverse`
read `self.W` directly (the Parameter / plain backing tensor,
never the shadow) and by removing the `self.setW(x)` self-poke at the
tail of `Codebook.forward` (which would re-poison the slot for the
next reverse). Both are landed 2026-05-21.

### Migration to remove `_active_payload`

The cleanup that retires `_active_payload` entirely requires:

1. **Audit every `setW(per_batch_tensor)` call site.** Each one is
   either:
   * a legitimate prototype write (replace with a clear API call,
     e.g., `addVectors` / `replace`),
   * a legitimate "set the cached muxed event on a pure-event SubSpace"
     write (keep — `.event` is `Tensor`, no Parameter to shadow), or
   * a per-batch write to a codebook-bearing slot that should become:
     - a forward snap → `set_activation(selection)` or
       `set_forward_content(indices)`, OR
     - removed entirely (the data was being cached but the
       `materialize` path can reconstruct it).
2. **Audit every `getW()` reader expecting per-batch content.**
   Migrate to `materialize(...)`. After this audit, `getW()` on a
   codebook-backed slot returns ONLY the prototype matrix.
3. **Drop `_active_payload`** from `Tensor` / `Codebook` /
   `ProjectionBasis`.
4. **Assert in `setW`** that plain-tensor writes to a Parameter-bearing
   slot are an error with a clear message pointing at the SubSpace
   setter API (`set_activation`, `set_forward_content`, …).

The migration is staged so each step keeps the suite green; the
strict assertion lands last (after all writers have been retargeted).
The 2026-05-21 attempt at a one-shot migration over-reached and was
reverted; it surfaced the muxed-vs-unmuxed codebook placement
distinction and confirmed `.activation` as the canonical selection
slot, both of which are baked into this spec.

---

## Glossary

* **Prototype matrix**: the `[V, D]` codebook on `.what.W` (unmuxed)
  or `.event.W` (muxed). The learned dictionary.
* **Selection**: the per-batch `[B, N]` (or `[B, N, M]`) tensor on
  `.activation` (or `_active`) that picks rows / weights from the
  prototype matrix.
* **Per-batch content / payload**: `[B, N, D]` per-position vectors.
  Reconstructed on demand by `materialize` as `prototype[selection]`;
  not stored as a separate field.
* **Snap**: the forward operation on a `Codebook` / `ProjectionBasis`
  that turns per-batch vectors into selections (indices / weights /
  signed DoT).
* **Muxed event**: the full `[B, N, nWhat + nWhere + nWhen]` per-batch
  tensor that interleaves the three modality contents.
* **Pure-event subspace**: a SubSpace whose `.what` / `.where` / `.when`
  are empty (no per-modality slot use); all per-batch content goes
  through `set_event` / `materialize(mode='event')`. `ConceptualSpace`
  is the canonical example.
