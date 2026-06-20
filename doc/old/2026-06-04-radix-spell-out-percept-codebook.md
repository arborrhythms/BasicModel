# Radix percept codebook on `.what` + spell-out emission

Date: 2026-06-04

## Model (per design discussion)

PerceptualSpace should NOT own a `<codebook>` option, and should NOT
populate `subspace.event` with a Codebook -- `.event` is only the muxed /
materialized event carrier. The percept prototypes live on `subspace.what`
(a `Basis`), so the SubSpace can store a `.active` selection and
materialize percepts **without copying** their vectors.

Radix emission = **"send the next-largest percept, up to the size of the
word; that percept may be the percept of a single byte."** Unfamiliar words
are *spelled out* as a run of byte / prefix percepts (each a single `.what`
row); as a byte-sequence recurs it is *concatenated* (promoted) into one
larger percept. Frequency drives concatenation. The radix trie's
byte-prefix structure is the (overlapping) part/whole structure on the PS
side. (Terminology: parent/child = **taxonomy**; part/whole = **meronomy**.
The explicit meronomy -- `part_parents` + `Basis.part` clipped-cosine --
lives on the **SymbolicSpace** codebook, not PS.)

## Done + verified

* **Part A** -- PerceptualSpace `<codebook>` retired:
  `PerceptualSpace._build_object_basis` always returns a passthrough
  `Tensor`, so `.event` is never a Codebook. Removes the lossy VQ snap and
  the `.what`(Embedding)/`.event`(Codebook) duplication. (`bin/Spaces.py`.)
  Verified: `.event` is Tensor for all configs; build-all 33/33; XOR gate
  still green.
* **Part B Steps 1 + 3** -- `RadixLayer` vector storage is a `Codebook`
  *Basis*, and in radix mode it is the **same** Codebook as
  `subspace.what` (one percept codebook, no duplicate table):
  * `RadixLayer.__init__(..., basis=None)`: uses a passed Basis
    (`object.__setattr__`, no double-registration) or owns a private
    `Codebook`; `codebook` is now a property -> `self._basis.getW()`;
    `_grow_to` delegates to `Codebook.grow_to`. (`bin/Layers.py`.)
  * `PerceptualSpace._build_what_basis`: in radix mode builds a `Codebook`
    on `.what` (not the lexicon Embedding); `percept_store` is constructed
    with `basis=self.subspace.what`. (`bin/Spaces.py`.)
  * Verified: MM_xor `.what` is a Codebook, `percept_store._basis IS
    subspace.what`, insert writes into the shared `.what`, round-trips;
    standalone radix tests (32) + build-all (33) green.
* **`RadixLayer.spell_out(chunk) -> List[int]`** -- the emission primitive:
  longest known prefix-percept at each position, falling to single-byte
  percepts (seeded on demand), so every id indexes one `.what` row.
  Tested in `test/test_radix_spell_out.py` (5 cases): cold word spells to
  bytes; longest-match after promoting a prefix; single percept after
  promoting the whole word; every pid a valid row; empty -> `[]`.

## 2026-06-04 (cont.): `_embed_radix` wired + Part B radix-lexing regressions

`_embed_radix` is now wired to the spell-out emission (per-chunk
`observe_chunk` then `spell_out`, event gathered from the shared `.what`
Codebook). Mechanically green: build-all 33/33 + radix tests 38.

**Digging into the 6 reported batch failures** showed 5 of 6 are NOT
regressions -- they pass in isolation AND in an 86-test perception/
chunking/grammar/config batch, so they are pre-existing order-fragility
(global-state leak from a test outside the perception theme, e.g.
`test_basicmodel`): `test_analyse_chunking_forward` (x2),
`test_perceptualspace_bpe_forward`, `test_xor_spaces::TestNullEOS` (x2).
The 6th, `test_mm_xor::test_mm_grammar_learns_xor_signal`, uses
`MM_grammar.xml` which is NOT radix (default chunking) -- so `_embed_radix`
cannot affect it -- and its $0.25 \not< 0.15$ convergence miss is the
known WIP flagged in the commit message ("MM_grammar.xml is slow and not
learning correctly"); convergence tuning is the human's domain.

**`test_learns_xor_signal` (the MM_xor.xml radix convergence test, NOT in
the reported 6) is a real regression from Part B** and exposed a cascade:
making radix `.what` a Codebook (was an Embedding) broke every consumer
that keyed on `.what` being an Embedding.

1. **`_peer_perceptual` not wired** -- `BasicModel.from_config`
   ([Models.py](../../bin/Models.py)) only wired the lexer peer when
   `isinstance(_vocab, Embedding) or isinstance(_what, Embedding)`. Radix
   `_vocab` is a PerceptStore and `_what` is now a Codebook $\to$ never
   wired $\to$ `InputSpace._lex_batch` asserts. **FIXED**: also wire when
   `perceptualSpace.percept_store is not None`.
2. **Orthographic tokenizer gone** -- `_lex_batch` resolved
   `_token_stream` from `_what` (the Embedding); a Codebook has none, and
   the PerceptStore fallback also has none $\to$ `AttributeError`. The
   tokenizer is config-driven (byte/word), not a property of the vector
   store. **FIXED**: `InputSpace._radix_token_stream` lexes base tokens
   from the InputSpace's own `byte_mode`/`lexer` (`parse(lex='bytes')` or
   `parse(lex='words')` + `\x00` sentinel; mirrors `Embedding._token_stream`);
   `_lex_batch` falls to it when no Embedding tokenizer is present.
3. **Codebook W lifecycle -- FIXED** -- the radix `.what` Codebook is
   created with `customVQ=False`, whose `addVectors` allocates `W` as a
   *plain tensor*. `_clear_runtime_basis` calls `setW(None)`, which
   preserves a Parameter `W` but *clears* a plain-tensor `W`; so the
   permanent percept codebook was wiped mid-forward and
   `RadixLayer.insert`/`lookup` hit `getW() is None`. The old RadixLayer
   owned an `nn.Parameter` codebook (learnable, never cleared). **Fix**:
   `PerceptualSpace._build_what_basis` radix branch now registers the
   freshly-created `W` as an `nn.Parameter` on the Codebook (the `.what`
   owner) -- a Parameter is permanent (`setW(None)` preserves it),
   learnable, and counted once (the RadixLayer adopts it via
   `object.__setattr__`, no double-registration). No further consumers
   crashed: the bounded probe forward-passes, gradients flow, and a
   40-step train loop converges $0.25 \to 0.002$ (threshold 0.15).
   **`test_learns_xor_signal` (MM_xor.xml radix) now PASSES.**

### Result of the 3 fixes

Comprehensive sweep (build-all + XOR_exact gate + perception + ownership
+ radix convergence): **84 passed, build-all 33/33 subtests, 4 xfailed**.
The only failures are `TestNullEOS::test_reconstruct_buffer_stops_at_null_token`
and `::test_trailing_null_produces_clean_reconstruction` -- and those are
**pre-existing order-fragility, not regressions**: they pass in isolation
(4/4), pass after `test_modality_configs` alone (5), pass after
`test_explicit_dimensions` alone (6), and fail ONLY when *both*
build-heavy tests accumulate global state (TheXMLConfig / TheData /
module-level lexicon caches) before TestNullEOS's nearest-neighbour
`reconstruct_buffer` lookup. Hardening that lookup against cumulative
state pollution is a separate test-infra task (the suite is known
order-fragile -- see memory `reference_test_runner.md`).

## Remaining: wire `spell_out` into `_embed_radix` (DONE 2026-06-04 -- see above; sub-items 3-5 below still open)

`bin/Spaces.py PerceptualSpace._embed_radix` still uses the old
one-slot-per-word + byte-fallback path (`ps.lookup_with_id` + `set_event`).
To realize the model:

1. **Seed byte-percepts** (or rely on `spell_out`'s on-demand byte
   insertion). Note: MM_xor PS `nVectors=200 < 256`; the store grows by
   doubling, so byte-percepts force a grow to 512 -- fine, but confirm the
   `.what` grow path (it goes through `Codebook.grow_to`).
2. **Emission**: per word, `pids = ps.spell_out(word_bytes)`; concatenate
   across words into a per-row pid sequence; **a word now spans multiple
   slots** until promoted. Build the per-slot event by gathering
   `subspace.what.getW()[pid_grid]` (shared Codebook).
3. **Concatenation / promotion**: track per-chunk frequency; when a word's
   count reaches `chunkPromotionThreshold` and `len >= chunkPromotionMinLength`,
   `ps.insert(word_bytes)` so the trie returns it as one percept next time.
4. **Reconstruction (re-join)**: a word spans several slots, so the decode
   must re-join consecutive byte/prefix percepts into words (track a
   per-slot word index, or emit a separator percept on word boundaries).
   Today's decode assumes one word per slot.
5. **No-copy `.active`** (optimization, currently BLOCKED): use
   `subspace.set_forward_content(pid_grid)` so `materialize` reconstructs
   from `.what[pid]` with no gather. Blocker: `.what` is `nDim`-wide but the
   muxed event is `nDim + nWhere + nWhen` (PS canonical_shape `(2,2)`), and
   `materialize`'s muxed-codebook path is gated on `proto width ==
   muxedSize`. Needs either a `materialize` extension that pads / muxes
   `.where`/`.when` for an `nDim`-wide `.what`, or a `.what` that carries
   the muxed width.

## Validation needed (owner: human)

The spell-out emission **changes MM_xor's slot structure** (variable slots,
words spelled out pre-promotion), so MM_xor's convergence must be
re-validated and likely LR/epoch/promotion-knob-tuned -- the same
forward-run-and-tune loop used to restore XOR_exact. The tiny-forward
build-all gate only catches crashes, not convergence.
