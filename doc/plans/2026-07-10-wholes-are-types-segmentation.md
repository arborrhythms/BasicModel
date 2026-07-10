# Wholes Are Types — property-basis cutover + type-run segmentation

> Decisions: Alec, 2026-07-10. Supersedes the "left-of-X / right-of-X
> separator dichotomy" framing of the word analyzer (Meronomy.word_spans
> docstring) and resolves the `Codebook.property_basis` todo item.

## Philosophy (the correction)

**Wholes are types.** Were they not types, it would be impossible to
bifurcate a whole into two clearly-identified regions. "Type" and
"property" name the same thing: a WholeSpace `.what` row is a binary
proposition over the field; materializing it bifurcates the input into
the has-type region (`+1`) and its complement (`-1`).

Types apply to **runs of characters, not single characters**. The four
obvious character types are:

    isLetter, isSpace, isDigit, isPunctuation

— exactly the existing `LETTER / WHITESPACE / DIGIT / PUNCT` classes of
`Layers.char_class_region`. Each is one codebook code.

**A whole is a maximal constant-type run.** Words are runs of letters,
punct, or digits; serial mode delimits word-wholes by run-length over
type. Boundary dichotomies ("left of space", "right of punctuation")
are rejected as the primitive: they force a boundary on both sides of
the token of interest and add nothing — a boundary is *derivable* from
properties when needed, not primitive.

## Decided behavior deltas (vs. the inline-LUT cut)

1. **`abc123` = two wholes** (letter-run + digit-run). Previously one
   word (word = maximal not-space-not-punct run).
2. **A punctuation run is one whole** (`"..."` = one punct-whole).
   Previously each punct char was its own one-char span.
3. **Within-whole division into concepts.** A type-run whole may divide
   into multiple *concepts* — `").."` region example: `")."` is one
   punct-whole but two concepts `)` + `.` — provided the parts are
   independently attested (appear standalone). Longest-match tiling
   prefers an attested whole (`"..."` as ellipsis, if attested) over
   its parts. Whole ≠ concept: the whole is the type-run; concepts are
   the attested parts that tile it.

## Tasks

- **T0 — docs.** This plan; `doc/Spaces.md` leads with wholes-are-types
  (the four binary character types are the canonical properties; the
  sinusoid stays as the continuous worked example); `Meronomy.word_spans`
  docstring drops the separator-dichotomy framing and points here.
- **T1 — remove `Codebook.property_basis`.** The `mode="property"`
  argument of `SubSpace.materialize` is already the explicit per-call
  opt-in; the class-attribute second gate is deleted. No production
  caller exists, so this is byte-identical. Tests updated.
- **T2 — type-run segmentation.** `stage_analysis_spans` cuts on
  per-position TYPE (a 256-entry type LUT built from the same four
  class ranges the codebook rows carry): wholes = maximal
  constant-type runs; space-type runs (incl. the `\0` pad sentinel)
  are discarded. Deltas 1 and 2 land here. Stays vectorized (run cut =
  type-change positions; the `_word_punct_spans` machinery generalizes).
  Bytes ≥ 127 (DEL, non-ASCII) keep their current word-char behavior by
  mapping to the letter type.
  **Landed (rows are the source of truth now).** The four binary
  char-class propositions (isSpace / isLetter / isDigit / isPunctuation)
  ARE codebook codes: they live as the four tagged rows of the frozen
  `.what` codebook of `WholeSpace.type_subspace` — a dedicated WS-owned
  `SubSpace` (Alec 2026-07-10: the type codebook is the `.what` of a
  subspace owned by WholeSpace; calculations in Spaces, DATA in
  SubSpaces, and "properties are WholeSpace.what" made literal). The
  subspace follows the `ConceptualSpace._subspaceForPS` auxiliary-
  subspace idiom: minimal `(4, nDim)` shape, default zero-band encodings
  (no fabricated `.where`/`.when`), `what=` handed in verbatim
  (`codebook_slot='what'`). That codebook is a `Codebook`, which now
  DERIVES FROM `Tensor` (`bin/Spaces.py`: `class Codebook(Tensor)` —
  the dense-`W` storage contract lives on `Tensor`; `Codebook` overrides
  only the VQ / SVD-cache / property side that genuinely differs and keeps
  `self.W = None` until `create`). Its `W` is a plain (non-Parameter)
  tensor, so it is FROZEN BY CONSTRUCTION — never snapped, never trained,
  invisible to any optimizer, and the subspace adds no `state_dict` keys
  to the WS (its word/pos buffers are `persistent=False`) — so it needs
  no VQ / EMA / LBG / adopt defenses. It is FORWARD-CONNECTED:
  `stage_analysis_spans` derives the byte→type LUT from THIS subspace's
  `.what.property_kind` (via `_derive_type_lut` / cached
  `_analysis_type_lut`, invalidated by `Codebook._property_kind_version`),
  replacing the direct `_LUT_ANALYSIS_TYPE` read. The derived LUT is
  byte-identical to the frozen module constant — default = letter
  (unassigned / bytes ≥ 127 keep word-char behavior), and byte 0 (the
  `\0` pad sentinel, not a WHITESPACE class byte) is forced to the space
  type. Built deterministically + idempotently at WholeSpace construction
  (only for the analysis modes that run the cut — word / grammatical /
  meronomy; byte / raw / sentence build none and fall back to the module
  LUT). Persistence rides `vocab_extras` (`type_property_kinds`), rebuilt
  on `load_vocab_extras`, so the rows + tags + cut survive a checkpoint
  roundtrip. The standalone `MeronymicAnalyzer` mirror reads the
  equivalent source — `Meronomy`'s `class_segments` / `_byte_class` over
  the same `Layers._CHAR_CLASS_RANGES` the type rows are tagged from.
  Tests: `test/test_type_run_spans.py` ("the four TYPE rows ARE a
  WS-owned subspace's .what" section).
- **T3 — within-whole division.** At the word→lexicon seam, a type-run
  that is not itself attested divides by longest-match tiling over
  attested standalone concepts (mechanism exists: `Meronomy.tile` /
  `match_greedy`). Tests: `")."` → two concepts when `)` and `.` are
  attested; an attested `"..."` stays one concept.
  **Landed (LIVE, default on).** Mechanism:
  `Spaces._divide_spans_into_attested` (host-eager, beside
  `_type_run_spans`), called from `WholeSpace.stage_analysis_spans` in
  the eager stem — the staged `[B, K, 2]` spans themselves divide, so
  each attested part fills its own stage-0 symbol slot
  (`_stage0_carrier`: span k → concept k) instead of the unattested
  whole adopting one virgin row. Attestation = the peer PS
  `RadixLayer` store (`perceptualSpace_ref.percept_store`); the tiling
  walk is `RadixTrie.longest_match` (the `spell_out` walk minus the
  on-demand byte seeding — division never mints). "Appears standalone"
  is enforced: a multi-byte part attests by being a store entry
  (promotion = standalone recurrence as a tile); a single-byte part
  attests only if that byte has occurred as its own type-run (a
  bounded ≤256-entry byte set the WS accrues from its staged wholes) —
  `spell_out`'s lazily-seeded byte percepts alone do NOT attest, or
  every unpromoted word would re-divide per byte. Attested wholes stay
  one span; incomplete tilings keep the span unchanged (the pre-T3
  fallback); attestation is one-step-stale (spans stage before this
  batch's spell-out), the `adopt_symbolic_evidence` contract. Knob:
  `<WholeSpace><divideWithinWhole>` (model.xsd), default TRUE when
  absent (live wiring — Alec 2026-07-10); explicit `false` opts out.
  The standalone `MeronymicAnalyzer` mirror (`granularity="type"` +
  `divide_within_whole`, bin/perceptual_analyzer.py) reads the same
  default. Tests: test/test_within_whole_division.py.
- **T4 — suite.** Full pytest gate; tests that pinned punct-per-char or
  mixed-alnum single tokens are re-pinned deliberately (the deltas are
  the point, not regressions) and listed in the execution notes.

## Non-goals

- The learned analyzer (`MeronymicRouter` Viterbi merge) stays the
  Phase 5-6 follow-on; this plan only moves the fixed cut from an
  inline rule to codebook-typed runs.
- No change to the WS symbol/truth prototype rows or the snap contract:
  type rows are a *read* (property materialization) of WS `.what`; the
  vector reads (snap / GA / taxonomy) are untouched.
