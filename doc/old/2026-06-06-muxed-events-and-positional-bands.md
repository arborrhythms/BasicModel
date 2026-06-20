# Muxed Events, Positional Bands, and the Pi/Sigma Transform

Date: 2026-06-06
Status: design (approved; implemented by `doc/plans/2026-06-06-postreview-refactor.md` Action C/D)

This pins down how the `.where`/`.when` positional band relates to the
`Pi`/`Sigma`/`ConceptualCombine` transform. It RETAINS the uniform `(2,2)`-band
convention of `bin/architecture.py` (the band is preserved as two-where/two-when
on every interior tier) and resolves the open question from
`doc/specs/2026-06-05-dimensional-governance.md` sec 2 ("is the band IN the
transform or riding along?") in favour of **option B: the band is muxed INTO the
transform** on the per-position operators.

ASCII only (`->`, `<->`, Pi, Sigma) -- no Unicode glyphs, so `make doc` / xelatex
is safe.

## 1. The muxed event IS the slab

When an event is MUXED, `.what`, `.where`, and `.when` are packed into ONE `nDim`
vector: `[ .what (content) | .where | .when ]`. The `sigmaPi` transform (Pi at PS,
Sigma at SS, and the per-stage `ConceptualCombine`) operates on the FULL muxed
event -- content AND the where/when tail TOGETHER -- NOT on demuxed content with
the band "riding along". The whole point of muxing where/when in is that they
PARTICIPATE in the transform. (If they only rode along untouched, the mux would
be pointless.)

When NOT muxed, the three are separate channels `.what` / `.where` / `.when` and
keep their own meaning.

## 2. Bands are uniform `(2,2)` on every interior tier -- never declared 0

`bin/architecture.py` `_CANONICAL_SHAPE` gives EVERY interior tier the SAME
`(nWhere=2, nWhen=2)` band:

- **IS, PS, ModalSpace, CS, SS, WordSpace**: `(2, 2)`.
- **OS** (terminal answer): `(0, 0)` -- the ONLY principled exception; a
  scalar/answer has no `.where/.when` to mux, and the loss would otherwise slice
  empty where/when segments and NaN.

`nWhere`/`nWhen` are NEVER declared `0` on an interior tier (not even on the deep
CS hub). The band is preserved as two-where + two-when EVERYWHERE and is kept
MUXED -- it rides INSIDE the transformed event, not as an inert tail. This
supersedes an earlier exploration that made the deep CS hub band-free `(0,0)`:
that collided with the `SS.nWhat == CS.nWhat` handoff check and forced configs to
re-dimension, and it is NOT the chosen design. The deep hub keeps its `(2,2)`
band like every other interior tier; the band simply participates in the
conceptual transform (sec 4).

## 3. .where / .when are learned positional encodings (RoPE-like), under gradient

- `.where` is a sin/cos positional encoding of the row's sequential index,
  stamped into the tail slots (`_WhereEncoding.forward`, `bin/Spaces.py`).
  `.when` likewise.
- They are RECONSTRUCTED under gradient: the per-slot loss adds
  `where_scale * MSE(pred_where, target_where)` (and `when_scale * ...`) when
  `nWhere > 0` (`ModelLoss.compute` / `compute_masked` / `compute_piecewise`,
  `bin/Layers.py`). **Keep this pressure.**
- Because where/when are CONSUMED (the reconstruction term + downstream readers),
  the gradient teaches the invertible `Pi`/`Sigma`/combine to keep those tail
  dims LEGIBLE through the dense transform -- they are just dimensions the matrix
  learns to respect. And since the operator is invertible over the FULL event,
  where/when round-trip for free (the inverse recovers content + band).

### 3.1 Decision: keep .where in the transform; do NOT revive the free neural line

There was a retired mechanism (the "free neural line", Phase 4, 2026-06-04) where
`nWhere != 0` was a sentinel that `.where` held an arbitrary input-space position
the content was SCATTERED to -- freeing the content row from a fixed position,
useful for reconstructing an input LARGER than perception. `WhereEncoding.recover`
(the `.where -> integer position` inverse) was removed; `.where` is now pinned to
the sequential row index and codebook identity is the row index (CS->SS reverse
decode is content-match / nearest row).

That free-pointer idea PULLS AGAINST muxing `.where` into the dense transform: a
dense invertible matrix that mixes `.where` into content scrambles a hard pointer,
and only the reconstruction gradient keeps `.where` legible (so it CAN be wrong,
the opposite of the free pointer's "can't be wrong"). You can have `.where` IN the
transform (reconstructed, soft) OR a free un-transformed pointer, not both.

**DECISION (2026-06-06): keep `.where` reconstructed and in the muxed transform
(soft / RoPE-like, current pressure). The free-neural-line /
`WhereEncoding.recover` / `.where`-as-arbitrary-position mechanism stays
RETIRED.** Reconstruction of an input larger than perception relies on the
content (each wide constituent's content) plus the sequential positional
encoding, not a `.where` scatter.

## 4. WHERE the band is muxed in -- and the one place it is not

The band PARTICIPATES in the transform on the PER-POSITION operators, which is
where "muxed" is meaningful:

- **PS.Pi / SS.Sigma in `butterfly` mode (the default `<sigmaPi>`)**: the layer
  is square at the FULL event width (`percept_dim` / `sigma_dim`, which INCLUDE
  the `(2,2)` band) and the butterfly cascade runs over `N_positions * full_event`.
  So the band is already woven through the cross-position transform.
  (`bin/Spaces.py`, the `_butterfly` branch.)
- **PS.Pi / SS.Sigma in `last` mode**: per-slot square fold over the full event
  width -- the band is in the slot.
- **The per-stage `ConceptualCombine`**: sized at `cs.muxedSize` (= `nWhat +
  nWhere + nWhen`), so each of its three streams `[PS_t || SS_t || CS_t]` is a
  FULL muxed event and the `.where/.when` tail is transformed with the content
  (sec 6). This is the C-tier conceptual advance; the band advances WITH the
  concept.

The ONE place the band is NOT folded in is the **`full`-mode dense wide<->deep
bridge** (`<sigmaPi>full` on PS.Pi / SS.Sigma): that flattens
`[N_positions * content]` into ONE LDU, and the per-position `(2,2)` band passes
through UNCHANGED. Reason: a per-position band cannot be reshaped across a
wide<->deep POSITION-COUNT change (folding `N_wide` bands into `N_deep` content
columns would scramble positional tails into unrelated content). The band rides
through this bridge as a clean identity precisely because it is uniform `(2,2)` on
both sides. The default mode is `butterfly` (fully muxed), so this content-only
bridge is the exception, not the rule.

## 5. The flat-slab invariant stays content-based

The constant slab across the wide<->deep bridge is `wide_positions * wide_content
== deep_positions * deep_content` (CONTENT widths, band excluded -- the band is
the per-position identity pass-through of sec 4). `bin/Models.py`
`ModelFactory.validate_config` (C1) checks this on the content widths; the
`reverseBegin` regroup and the bridge reshape are content-based. Configs do NOT
re-dimension to absorb the band -- the band is never reshaped across a
position-count change.

## 6. Worked examples

- **MM_20M** (parallel; PS and CS both `[8, *]`, same position count): PS and CS
  events are both `muxedSize = 1024` (1020 what + 2 where + 2 when). The per-stage
  `ConceptualCombine` is sized at `D = cs.muxedSize = 1024`, so it transforms the
  full `[8, 1024]` event -- the 4 band dims are mixed with the 1020 content dims
  by the combine. **CS `nDim` / band UNCHANGED (`(2,2)`, 1024)**; nothing
  re-dimensions. The same-position handoff means Pi is a per-position transform;
  the band participates and round-trips.
- **MM_5M_grammar** (serial deep<->wide; `<sigmaPi>butterfly`, conceptualOrder 3):
  the per-position butterfly Pi/Sigma carry the band through the cross-position
  cascade, and each stage's `ConceptualCombine` (sized at `cs.muxedSize`) advances
  the full muxed event. The wide<->deep regroup is content-based (sec 5); the
  uniform `(2,2)` band is the per-position identity tail across the bridge.

There is no band-per-position collision: the band is never RESHAPED across a
position-count change; on the per-position operators it is muxed into the
transform, and across the wide<->deep bridge it is a uniform identity tail.

## 7. Code implications (Action C of the post-review refactor)

- `bin/architecture.py`: `_CANONICAL_SHAPE` UNCHANGED -- uniform `(2,2)` on every
  interior tier, `(0,0)` only on OS. (No band-free CS.)
- `bin/Layers.py` `ConceptualCombine`: unchanged math; it is fed the FULL muxed
  event width (`content_dim = cs.muxedSize`) so `.where/.when` participate. The
  arg name `content_dim` is historical -- it is the whole muxed-event width now.
- `bin/Models.py`:
  - The per-stage combine is sized at `cs.muxedSize` (not `cs.nWhat`) and is
    HELD BY its `ConceptualSpace` -- registered in `cs.layers` (so
    `paramUpdate`/`set_sigma` cascade) and `cs.params` (so the `self.spaces`
    getParameters() walk optimises it); exposed as `cs.combine`. There is NO
    model-level `conceptual_combines` list and no separate param/paramUpdate/
    set_sigma loops (cf. PS owns `pi`, SS owns `sigma`).
  - The forward/reverse body demux each stream AT `D = cs.muxedSize`, so the demux
    returns the whole event with an EMPTY band -- the band flows through the
    combine.
  - The dead `_symbolic_sigma_step` (the content-only SS.sigma advance the combine
    replaced) is REMOVED; `test/test_cs_reentrancy.py` asserts `ss.sigma`
    invertibility directly.
- `bin/Spaces.py`: PS.`pi` (butterfly/last) and SS.`sigma` (butterfly/last) are
  already square at the full event width (band included). Only the `full`-mode
  dense bridge stays content-only (band passes through; sec 4).
- Keep the `.where`/`.when` reconstruction loss terms (sec 3) -- they are the
  gradient that keeps the muxed positional tail legible through the transform.
- (Action D, separate) eliminate `_peer_perceptual`: IS -> PS passes a 1-dim
  content (byte value) + (2+2) band; PS uses the byte as a codebook index.
