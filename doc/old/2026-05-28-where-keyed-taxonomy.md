# `.where`-Keyed Taxonomy + Per-Space `nWhere` Defaults

> Builds on the META taxonomy work from
> [`doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md`](2026-05-27-perceptstore-meta-taxonomy-reentrancy.md)
> (Stages 7-10 landed) plus the post-plan refactors:
>
> - SymbolizeLayer (renamed from MetaLayer)
> - Auto-bind on PerceptStore promotion (currently on ConceptualSpace per Task G)
> - LBG-style codebook splitting on SymbolicSpace
> - PerceptStore → RadixLayer relocation into bin/Layers.py
>
> This plan replaces the **signed-integer cross-codebook taxonomy refs** (positive
> = PS row, negative = `-(SS row + 1)`) with a unified **`.where`-keyed**
> taxonomy. The `.where` field becomes the canonical positional identifier across
> InputSpace, PerceptualSpace, and SymbolicSpace.

## Context

The signed-int taxonomy works but has three asymmetries:

1. **Two namespaces, sign-glued together.** The convention `positive → PS,
   negative → -(SS_row + 1)` is a hack that disambiguates two unrelated
   codebooks. It leaks across every call site that reads taxonomy refs;
   helper methods (`_ps_signed`, `_ss_signed`, `_ps_row_of`, `_ss_row_of`)
   exist solely to hide the seam.
2. **Row indices are extrinsic.** A row index is an artifact of *when* an
   entry was allocated, not *what* it is. Codebook growth (LBG split, grow_to)
   doesn't break the existing rows but does mean two "isomorphic" symbols
   can land at different indices depending on training order. Persistence
   format has to carry the exact integer assignments.
3. **No spatial reasoning.** Row indices are points; there is no notion that
   "this row is near that row." Downstream grammar dispatch and chart compose
   could benefit from positional decay, but the current encoding offers
   nothing to consume.

The fix is to make `.where` the canonical positional identifier. Each
allocated entry (input slot, percept row, symbol row, META row) gets a
unique integer position from a shared global counter, materialized into the
`.where` field as a **quadrature-sinusoidal vector**
(`(sin(pos·ω), cos(pos·ω))` pairs). The integer is recoverable from any
single quadrature pair via `pos = atan2(sin, cos) / ω` (mod `2π/ω`); the
sinusoidal vector is the storage form, the integer is a view.

## Architectural decisions (locked 2026-05-28)

1. **`.where` is the canonical positional identifier.** Used for taxonomy
   refs, LBG split positions, and any future positional dispatch. The
   existing `WhereEncoding` machinery (quadrature sin/cos) is the
   materialization form.

2. **Quadrature sinusoidal, not integer-scalar.** Quadrature pairs have
   invariant norm (`sin² + cos² = 1` per pair, `√k` for k pairs), so
   adding a `.where` component to any vector preserves global norm scaling.
   Integer-as-scalar would grow the norm linearly with position and break
   normalized similarity / VQ quantization downstream. This is the locked
   choice.

3. **Integer position is recoverable from the encoding.** `atan2(sin, cos)
   / ω_0 = pos` (mod `2π/ω_0`) using the lowest-frequency pair. Higher
   frequencies are redundant for identity but useful for fine-grained
   spatial similarity. This means there is **no parallel integer store** —
   the sinusoidal vector IS the position, the integer falls out on demand.

4. **Unified integer counter across IS/PS/SS.** A single monotonic counter
   on the model (or on a coordinator like `SymbolicSpace` since it owns the
   META taxonomy) allocates positions for every new entry — input slot,
   percept row, symbol row, META node. The counter never resets; positions
   are stable across the model's lifetime.

5. **Per-Space `nWhere` defaults move to model.xml.** Configs no longer
   declare `<nWhere>` / `<nWhen>` per file. `model.xml` carries the per-Space
   defaults: `<nWhere>2</nWhere>` on InputSpace, PerceptualSpace, and
   SymbolicSpace (the spaces whose entries participate in the taxonomy).
   Other spaces (ConceptualSpace, OutputSpace) keep the existing global
   default of 0 unless they need positional info.

6. **Taxonomy keying.** `meta_pair_to_idx[(ps_pos, ss_pos)] → meta_pos`
   stays a Python dict for O(1) idempotency lookup, but the keys and value
   are now **integer positions recovered from `.where`**, not row indices.
   The taxonomy `dict[int, list[int]]` and `taxonomy_parent_map` also
   migrate to integer positions.

7. **LBG split mechanics.** Split allocates `new_pos = next_int()`, encodes
   it via `WhereEncoding(new_pos)`, writes it to the new SS row's `.where`,
   and records a META edge `(child_pos, parent_pos) → split_pos`. The
   sinusoidal vector inherits norm `√k` automatically; no projection
   needed.

8. **The "frozen zeros anchor" reinterpretation.** Position 0 is reserved
   and never allocated to a content row. Its sinusoidal encoding is
   `[0, 1, 0, 1, …]` (sin(0)=0, cos(0)=1 for every frequency) — a specific
   unit-norm vector, not the literal zero vector. The "anchor" property
   comes from the integer index being held constant and untrained, not
   from the vector being literally zero.

9. **Sign-convention helpers retire.** `_ps_signed`, `_ss_signed`,
   `_ps_row_of`, `_ss_row_of` are deleted. Their call sites switch to
   integer positions throughout.

## Design

### `WhereEncoding` recovery API

Current `WhereEncoding` (`bin/Spaces.py:287`) takes a position and emits a
sinusoidal vector. Add the inverse:

```python
class WhereEncoding:
    def encode(self, pos: int) -> torch.Tensor:
        # Existing: returns [sin(pos·ω_0), cos(pos·ω_0), sin(pos·ω_1), cos(pos·ω_1), …]
        # Norm = √(nDim/2)  (since nDim is 2k for k frequency pairs).
        ...

    def recover(self, vec: torch.Tensor) -> int:
        """Recover the integer position from a sinusoidal vector.

        Uses the LOWEST frequency pair (ω_0 = 2π / maxP) for unambiguous
        identity within [0, maxP). Higher pairs are redundant for identity.

            sin_0, cos_0 = vec[0], vec[1]
            pos = atan2(sin_0, cos_0) / ω_0  (mod 2π/ω_0)
            return round(pos)  # snap to integer

        Raises ValueError if the recovered position is outside [0, maxP)
        or the encoded vector is not consistent across all frequency
        pairs (cross-validation against higher frequencies).
        """
        ...
```

The cross-validation against higher pairs catches drift / corruption: if
`atan2(sin_1, cos_1) / ω_1` doesn't agree with the recovered `pos` (modulo
its own period), the vector isn't a valid encoding and `recover` raises.

### Position counter

Allocated on `SymbolicSpace` (which already owns the taxonomy storage):

```python
class SymbolicSpace:
    def __init__(self, …):
        …
        # Position 0 reserved as the frozen-zeros anchor; counter starts at 1.
        self._next_position: int = 1

    def allocate_position(self) -> int:
        pos = self._next_position
        self._next_position += 1
        return pos
```

Persisted via `vocab_extras` so positions are stable across checkpoint
roundtrip.

### `.where` field as taxonomy storage

When `insert_percept(bytes)` runs:
1. Delegate to `RadixLayer.insert(bytes)` to allocate the PS-side codebook
   row (existing behavior).
2. Call `pos = ss.allocate_position()`.
3. Write `WhereEncoding(pos)` into the PS row's `.where` field.
4. Return `pos` (integer position, replaces the old positive signed-int).

When `insert_symbol(init_vec)` runs:
1. Allocate the SS codebook row (existing behavior).
2. Call `pos = ss.allocate_position()`.
3. Write `WhereEncoding(pos)` into the SS row's `.where` field.
4. Return `pos` (integer position, replaces the old negative signed-int).

When `insert_meta(ps_pos, ss_pos, fused_vec)` runs:
1. Idempotency check: `meta_pair_to_idx.get((ps_pos, ss_pos))` → existing
   `meta_pos` if present (return it + EMA-update fused vec).
2. Else, allocate a new META row + a new position. Write
   `WhereEncoding(meta_pos)` into the META row's `.where` field.
3. Store `taxonomy[meta_pos] = [ps_pos, ss_pos]` and
   `taxonomy_parent_map[ps_pos] = taxonomy_parent_map[ss_pos] = meta_pos`.
4. Return `meta_pos`.

**Notes:**
- Taxonomy refs are now **unsigned positive integers** drawn from one shared
  pool. There is no "PS = positive, SS = negative" convention; you ask the
  taxonomy what something is.
- The `.what` content of each row stays separate (codebook row vector). The
  `.where` field carries position. Together they answer "what is this and
  where does it sit in the positional namespace."

### Reverse decode under `.where` keys

`RadixLayer.reverse(vec, *, symbolic_space=None)`:
1. Find the nearest SS row to `vec` (argmin against `SS.codebook`).
2. Read the row's `.where` → `pos = WhereEncoding.recover(.where[row])`.
3. Look up `taxonomy_children(pos)` — if it's a META, walk to find the
   child whose position resolves to a PS row.
4. PS row → bytes via `RadixLayer.bytes_for`.

The parent-walk fallback (nearest match was a META's child, not the META
itself) is unchanged: `taxonomy_parent(pos)` gives the parent position;
if it's a META, walk it.

### LBG split under `.where` keys

`SymbolicSpace.maybe_split_lbg(pos)`:
1. Check the per-position variance accumulator
   (`_lbg_disp_sum[pos]`, `_lbg_count[pos]`, `_lbg_disp_sum_sq[pos]`).
2. If `variance.max() < threshold` or `count < min_count`, return None.
3. Allocate `new_pos = self.allocate_position()`.
4. Allocate a new SS codebook row at the next physical index; write
   `WhereEncoding(new_pos)` to its `.where`.
5. Perturb the new row's `.what` by `-ε · direction`; perturb the original
   row's `.what` by `+ε · direction`.
6. Inherit META binding from the parent: if `taxonomy_parent(pos)` is a META
   M with positive PS child `ps_pos`, write
   `insert_meta(ps_pos, new_pos, fused_vec=W[new_row])`.
7. Reset accumulators for `pos`. Return `new_pos`.

The `.where` perturbation is left out of LBG split intentionally — the
position is an identity, not a learned vector. Split halves get **different
positions** (different integer IDs), and their `.where` vectors are the
sinusoidal encodings of those distinct integers. No drift, no projection.

### Persistence

`SymbolicSpace.vocab_extras()` adds:
- `next_position`: the counter (so future allocations resume correctly).
- Existing taxonomy / parent_map / meta_pair_to_idx persist as before but
  keyed by integer positions.

The PS-side `RadixLayer.vocab_extras()` unchanged for now; positions live
on the SS side (they're allocated by `SymbolicSpace.allocate_position`).

If `RadixLayer.codebook` rows need their own positions (e.g., for direct
`.where`-keyed PS-side lookup without the SS detour), add a parallel
`positions: list[int]` mirroring `inverse_table`.

## Stages

### Stage 1: XML cleanup + per-Space `nWhere` defaults

**Goal.** Remove `<nWhere>` and `<nWhen>` from all non-default XML configs.
Move `nWhere` defaults into model.xml as per-Space values.

**Files modified.**

- `data/model.xml`:
  - Remove the global architecture-level `<nWhere>0</nWhere>` and
    `<nWhen>0</nWhen>` (lines 66-67).
  - Add `<nWhere>2</nWhere>` inside each of `<InputSpace>`,
    `<PerceptualSpace>`, `<SymbolicSpace>`.
  - `<ConceptualSpace>` and `<OutputSpace>` either omit `<nWhere>` (defaults
    to 0) or explicitly set `<nWhere>0</nWhere>` for clarity.
  - `<nWhen>` defaults stay at 0 everywhere (no temporal encoding by
    default; configs that need it must opt in).
- `data/BasicModel.xml`, `data/LM_5M.xml`, `data/LM_5M_IR.xml`,
  `data/MM_400M.xml`, `data/MM_5M.xml`, `data/MM_5M_AR.xml`,
  `data/MM_5M_IR.xml`, `data/MM_shamatha.xml`,
  `data/MM_xor_step{1,2,3,4}.xml`, `data/MentalModel.xml`,
  `data/RamsifiedModel.xml`, `data/XOR_pos.xml`, `data/stream_smoke.xml`
  (the 17 XMLs currently declaring `<nWhere>` or `<nWhen>`): **remove the
  explicit declarations**. They'll inherit the per-Space defaults from
  model.xml.
- `data/model.xsd`: update the schema annotation for `<nWhere>` /
  `<nWhen>` to document that per-Space defaults live in model.xml; remove
  any architecture-level enforcement that requires both to be declared
  globally.

**Tests.**

- Smoke: `make xor` and any other targeted config that previously declared
  `<nWhere>` must continue to construct without raising. The XML loader
  needs to fall back to per-Space defaults cleanly.
- New `test/test_where_defaults.py`:
  - With model.xml defaults: `InputSpace.nWhere == 2`,
    `PerceptualSpace.nWhere == 2`, `SymbolicSpace.nWhere == 2`,
    `ConceptualSpace.nWhere == 0`.
  - When a config does declare `<nWhere>` explicitly at a Space level, that
    overrides the default.

**Acceptance.**

- `grep -rE "<nWhere>|<nWhen>" data/*.xml` returns only `data/model.xml`
  matches plus opt-in configs that intentionally override.
- All existing targeted tests still pass.

### Stage 2: `WhereEncoding.recover()` + position counter

**Goal.** Add the integer-recovery primitive to `WhereEncoding`. Wire the
position counter onto `SymbolicSpace`. No taxonomy migration yet — Stages
3-4 do that.

**Files modified.**

- `bin/Spaces.py`:
  - `WhereEncoding.recover(vec)`: implementation per the API spec above.
    Cross-validates against higher frequency pairs; raises `ValueError`
    on inconsistency or out-of-range positions.
  - `SymbolicSpace.__init__`: add `self._next_position = 1`.
  - `SymbolicSpace.allocate_position() -> int`: monotonic int allocator.
  - `SymbolicSpace.vocab_extras()` / `load_vocab_extras()`: persist
    `next_position`.

**Tests.**

- New `test/test_where_encoding_recovery.py`:
  - Roundtrip: `recover(encode(pos)) == pos` for `pos` in `[1, maxP)`.
  - Boundary: `recover(encode(0)) == 0`.
  - Aliasing: `recover(encode(maxP + 5))` raises (cross-validation fails OR
    documented modular behavior).
  - Cross-validation: a forged vector that satisfies the lowest pair but not
    higher pairs raises.
  - Position counter: `allocate_position()` returns monotonically increasing
    ints starting at 1; survives `vocab_extras` roundtrip.

**Acceptance.**

- `recover` is exact within `[0, maxP)` and fails loud outside.
- Position counter is stable across checkpoint load.

### Stage 3: Migrate `insert_percept` / `insert_symbol` / `insert_meta` + taxonomy storage

**Goal.** Switch the API from signed-int row indices to unsigned integer
positions. Materialize `.where` on every insert. Retire the sign-convention
helpers.

**Files modified.**

- `bin/Spaces.py`:
  - `SymbolicSpace.insert_percept(bytes) -> int`: allocate position, write
    `.where` on the PS row, return position.
  - `SymbolicSpace.insert_symbol(init_vec=None) -> int`: allocate position,
    write `.where` on the new SS row, return position.
  - `SymbolicSpace.insert_meta(ps_pos, ss_pos, fused_vec=None) -> int`:
    allocate position for the META row, write `.where`, store
    `taxonomy[meta_pos] = [ps_pos, ss_pos]` and parent map, return position.
    Both `ps_pos` and `ss_pos` MUST be positive integers (no more sign
    convention). `meta_pair_to_idx` is keyed by `(ps_pos, ss_pos)`.
  - Helpers: `taxonomy_children`, `taxonomy_parent`, `is_meta` all consume
    integer positions.
  - Retire `_ps_signed`, `_ss_signed`, `_ps_row_of`, `_ss_row_of`.
- `bin/Layers.py`:
  - `RadixLayer.reverse(vec, *, symbolic_space=None)`: switch the SS-walk
    to read `.where` from the nearest SS row, recover the integer position,
    walk `taxonomy_children(pos)`, look up the PS-side child by position,
    map back to `RadixLayer.bytes_for(ps_row)` via a new
    `RadixLayer.row_for_position(pos)` helper (parallel `positions`
    list on the layer).
  - `RadixLayer.insert(bytes, *, init_vector=None, position=None) -> int`:
    accept the externally-allocated position to write into the
    `positions` list. If `position` is None, **do not** allocate one
    here — that responsibility is on `SymbolicSpace`. Raises if neither
    is supplied (forces the caller to be explicit during migration).
- `bin/Models.py`:
  - `_reverse_decode_one`: unchanged surface (thin wrapper around
    `RadixLayer.reverse`); the internals just consume integer positions
    now.
- `bin/Spaces.py` `ConceptualSpace._maybe_autobind_meta`: switches to call
  the new positive-int APIs.

**Tests.**

- Update `test/test_two_codebook_meta_taxonomy.py`: all assertions
  on signed-int returns become positive-int assertions; `_ps_signed` /
  `_ss_signed` references removed.
- Update `test/test_autobind_from_cs.py`: same.
- Update `test/test_radix_layer_reverse.py`: same.
- Update `test/test_symbolize_layer.py`: `SymbolizeLayer.forward` resolves
  positions, not signed-int row refs; the test fixtures need the
  positional convention.
- New `test/test_where_keyed_taxonomy.py`: end-to-end test that
  `insert_percept` + `insert_symbol` + `insert_meta` populates `.where`
  fields correctly; `recover` returns the same position used at allocation;
  `meta_pair_to_idx` is keyed by positions and is idempotent.

**Acceptance.**

- `grep -nE "_ps_signed|_ss_signed|_ps_row_of|_ss_row_of" bin/ test/`
  returns no live references (only retired-doc breadcrumbs OK).
- MM_xor smoke run produces non-blank reconstruction (already true; this
  is regression-only).
- Taxonomy storage is positive-int keyed throughout.

### Stage 4: Migrate LBG split to `.where`-position semantics

**Goal.** LBG split allocates a new position via `allocate_position()`,
writes the new SS row's `.where` from that position, and inherits the META
binding by integer position (not signed-int).

**Files modified.**

- `bin/Spaces.py` `SymbolicSpace.maybe_split_lbg(pos)`: accept and return
  positive integer positions; allocate a new position for the split-off
  row; write `.where`.
- `bin/Spaces.py` `SymbolicSpace.record_lbg_pull(pos, vec)`: accept integer
  position.
- Call sites in `ConceptualSpace._maybe_autobind_meta` and
  `SymbolizeLayer.forward`: pass integer positions.

**Tests.**

- New `test/test_lbg_where_keyed.py`:
  - Repeat the existing LBG split test with positive-int positions.
  - Verify the split's new row has a fresh position whose `.where` encoding
    is consistent with `WhereEncoding.recover`.
  - Verify the inherited META binding uses the new position.
- Update existing LBG tests to drop sign convention.

**Acceptance.**

- LBG split produces a new positive-int position; the new SS row's `.where`
  decodes back to that position.
- Taxonomy state is fully `.where`-keyed.

## Critical files

| File | Stage | Change type |
|---|---|---|
| `data/model.xml` | 1 | Per-Space `<nWhere>` defaults |
| Other `data/*.xml` (17 files) | 1 | Remove `<nWhere>` / `<nWhen>` |
| `data/model.xsd` | 1 | Doc-comment updates |
| `bin/Spaces.py` `WhereEncoding` | 2 | Add `recover` |
| `bin/Spaces.py` `SymbolicSpace` | 2, 3, 4 | Position counter; insert_* signatures; LBG migration |
| `bin/Layers.py` `RadixLayer` | 3 | `positions` list; row lookup by position |
| `bin/Models.py` `_reverse_decode_one` | 3 | Thin wrapper, no logic change |
| Tests (multiple) | 1-4 | Drop sign convention; add position assertions |

## Cross-cutting concerns

### What survives unchanged

- The PS/SS codebooks themselves (still independent `nn.Parameter` stores).
- The radix path (`RadixLayer.forward` / `reverse`).
- `SymbolizeLayer` (formerly MetaLayer) signature: forward/reverse still
  consume tensors; the integer positions live in the taxonomy it dispatches
  to, not in its own API.
- Auto-bind on CS (Task G stays).
- LBG split mechanics (Stage 4 just changes how positions are minted).

### What retires

- The signed-integer sign convention (positive=PS, negative=SS) and its
  helpers.
- The `meta_pair_to_idx[(positive_ps, negative_ss)] → negative_meta`
  signature; replaced by `meta_pair_to_idx[(pos_a, pos_b)] → pos_meta`
  where all three are positive integers.
- Global architecture-level `<nWhere>` / `<nWhen>` declarations in
  per-config XMLs.

### No-top-down-expectation premise (still holds)

The `.where`-keyed taxonomy doesn't introduce a top-down prediction
generator. The forward/reverse equations stay free of an interference
term. The same documented assumption from the prior plan applies.

### Documentation updates

- `doc/Spaces.md`: update the SS / taxonomy section to describe `.where`-
  keyed refs. Note that PS / SS / META all live in one positional
  namespace.
- `doc/Architecture.md`: brief note on the unified positional axis.
- Plan doc itself: mark Stage 1-4 as landed as they go.

## Verification

Whole-plan acceptance:

1. **`make xor` with `<chunking>radix</...>` converges and reconstructs**:
   same property as before, but now reverse decode walks `.where`-keyed
   taxonomy.
2. **Targeted tests pass** for every stage's gate file plus the existing
   META taxonomy + RadixLayer regression set.
3. **`grep -nE "_ps_signed|_ss_signed|_ps_row_of|_ss_row_of" bin/`**
   returns only retired-doc breadcrumbs.
4. **`grep -rE "<nWhere>|<nWhen>" data/*.xml`** returns only the model.xml
   defaults plus configs that intentionally override.
5. **`WhereEncoding.recover(WhereEncoding.encode(pos)) == pos`** for `pos`
   in `[0, maxP)`.
6. **`SymbolicSpace.allocate_position()`** returns monotonically increasing
   positive ints; survives checkpoint roundtrip.
7. **Taxonomy storage is positive-int keyed throughout**; no sign
   convention anywhere live.
8. **No git commits, no stashes** in the working tree across all stages.

## Risks / open items

1. **`maxP` configuration.** The unified position counter grows monotonically
   across the model's lifetime. If `maxP` (the period of the lowest frequency
   pair in `WhereEncoding`) is too small, allocated positions eventually wrap
   and `recover` either lies or raises. Mitigation: pick `maxP` generously
   (e.g., `10⁶`) at model construction; document the upper bound on
   allocations. Alternative: dynamically grow `maxP` (and re-encode existing
   `.where` rows) when the counter approaches the limit — but that's a
   migration each time, with O(V) cost.

2. **Recovery on quantized / drifted `.where` rows.** If the `.where` field
   participates in any operation that perturbs it (e.g., it's added to
   `.what` for an event tensor, then re-quantized via a codebook), the
   stored `.where` row might drift away from the exact encoded value. The
   spec assumes `.where` is **frozen** after `insert_*` writes it — no
   training updates, no perturbation. If downstream training ever touches
   `.where` directly, the recovery becomes lossy and idempotency breaks.

3. **Migration order.** Stages 3 and 4 touch shared call sites
   (`SymbolicSpace.insert_meta`, `SymbolizeLayer.forward`,
   `ConceptualSpace._maybe_autobind_meta`). Recommend doing them in one
   landing rather than two so the codebase doesn't briefly have a mix of
   signed-int and positive-int APIs.

4. **Backwards compat for existing checkpoints.** Checkpoints saved under
   the signed-int convention won't load cleanly under the new positive-int
   storage. Mitigation: a one-time migration helper in
   `SymbolicSpace.load_vocab_extras` that detects the old format (presence
   of negative ints in taxonomy keys) and rekeys on load. Or: punt and
   declare a clean break — checkpoints from before Stage 3 are unsupported.

5. **`nWhere=2` means norm=1, not `√2`.** With `nWhere=2` we get *one*
   frequency pair, so the encoded `.where` vector has norm 1.
   `nWhere=4` would give two pairs (norm `√2`). The user chose 2 — fine for
   identity, marginal for cross-validation. If we want stronger
   cross-validation against forged vectors, `nWhere=4` is the cheap upgrade.

6. **Tests that hand-construct sign-convention indices.** Anywhere a test
   says `assert ss_idx < 0` or constructs a META with `(positive, negative)`
   pair needs the migration. The Stage 3 file list covers the known ones;
   if a fresh `grep` turns up more, fold them in.

7. **`recover()` on the frozen-zeros anchor.** Position 0 encodes to
   `[0, 1, 0, 1, …]`. `atan2(0, 1) = 0`, so recovery is exact. But the
   anchor's `.where` vector also matches a fake input that's exactly
   `[0, 1, 0, 1, …]` — there's no way to distinguish "this is the
   anchor" from "this is a clean encoding of pos=0." Reserving pos=0 as
   non-allocatable means content rows never have this encoding, so
   collisions are by design impossible — but downstream code that reads
   `.where` should treat pos=0 specially (it's the anchor, not a content
   slot).

## Resolved decisions (2026-05-28, this plan)

0. `.where` is the canonical positional identifier; quadrature sinusoidal
   is the materialization (norm-invariant).
00. Integer position is a *view* on `.where`, recovered via `atan2`. No
    parallel integer store.
1. Unified counter on `SymbolicSpace`; positions are monotonic across IS/PS/
   SS/META.
2. Per-Space `<nWhere>` defaults live in model.xml. Other configs strip
   the explicit declarations.
3. Taxonomy keys are positive integer positions. Sign convention retires.
4. LBG split allocates a fresh position; `.where` is written from that
   position; no perturbation of `.where`.
5. Position 0 is the reserved frozen-zeros anchor.

## Implementation strategy

The four stages are mostly sequential:

- **Stage 1 first** (XML cleanup). Self-contained config + defaults change.
- **Stage 2 next** (`recover()` API + counter). Self-contained; no behavior
  change to existing flows.
- **Stages 3 and 4 together** in one landing. The signed-int → positive-int
  rekey is invasive enough that splitting it across two PRs creates a
  briefly-inconsistent API. Do them as a single coordinated change.

Each stage gets the standard subagent dispatch cycle
(`superpowers:subagent-driven-development`): implementer → spec reviewer →
code quality reviewer → mark complete. All work in the existing main
branch / working tree per the user's git policy (no commits, no stashes by
subagents; user commits at checkpoints).
