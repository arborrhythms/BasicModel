# Post-Review Refactor: forward/reverse unification, no cross-space refs, muxed transform

Date: 2026-06-06
Status: executing (directly, this session -- no subagents)
Context: follow-up to the dimensional-governance completion work, from the user's
holistic-review directives.

ASCII arrows (`->`) and operator names (Pi, Sigma) only -- no Unicode glyphs (so
`make doc` / xelatex does not choke).

## Load-bearing principles (from the user)

1. **Export must not bloat the model.** Unify `forward()` / `forward_core()`; isolate
   the speculative MLX-export adapter in `bin/export_mlx.py`. ("The optimization
   concern of export should have minimal impact on code clarity, especially since we
   don't know how effective it will be.")
2. **No cross-space referencing.** Eliminate `_peer_perceptual` (InputSpace -> PS
   codebook back-ref). IS passes a 1-dim content (the byte value) + (2+2) band to PS;
   PS uses the byte as a codebook index.
3. **Pi/Sigma/sigmaPi/ConceptualCombine operate on the FULL MUXED `nDim` slab**
   (content + .where + .when together), NOT demuxed content with the band riding
   along (option B). The transform preserves where/when AS PART OF the slab on the
   PER-POSITION operators (butterfly/last Pi/Sigma + the per-stage combine); the
   `full`-mode dense wide<->deep bridge stays content-only (the uniform band is a
   per-position identity pass-through). The band stays UNIFORM `(2,2)` on every
   interior tier -- never declared 0 (user: "preserve those as two, but keep them
   muxed" / "as = 2"). Unmuxed -> keep `.what/.where/.when` separate. This
   SUPERSEDES the 2026-06-05 spec sec 2 "content = nDim - band, Pi/Sigma square over
   the content slab" framing.
4. **The `ConceptualCombine` is a `Layer` held by its `ConceptualSpace`** (user:
   "Conceptual combine is best done as a Layer in Layers.py and held by CS"), cf. PS
   owns `pi`, SS owns `sigma`. No model-level `conceptual_combines` list.

## Sequence (dependency-aware)

A. Unify forward (clean, independent)
B. `BasicModel.reverse()` (clean, independent)
C. Muxed transform = option B (foundational slab change)
D. Eliminate `_peer_perceptual` via IS->PS 1-dim byte + 2+2 (builds on C; highest risk)

Each action must reach a GREEN test state before the next. D is last so an incomplete
D can stop at a green post-C state.

## A. Unify `forward()` / `forward_core()`

Goal: ONE forward path in `BasicModel`; the export adapter lives in `bin/export_mlx.py`.

- Remove from `bin/Models.py`: `forward_core` (3070), `stage_for_core` (3024),
  `export_core_module` (3126), `_ForwardCoreModule` (1871), `_stem_subspace_from_tensor`
  (3110).
- Keep the MINIMAL export hook: `_forward_per_stage(..., in_sub_override=None)` (7328)
  -- the one functional injection point `torch.export` needs (staged tensor as an
  ARGUMENT, not read from `self`). Document it as export-only.
- Add to `bin/export_mlx.py`: an `nn.Module` `ExportCore(model)` whose `forward(staged)`
  rebinds `staged` into the model's stem shell and calls
  `model._forward_per_stage(None, in_sub_override=...)`, plus a `stage_for_core(model, x)`
  helper (host lex+embed -> staged tensor, reusing `inputSpace.forward`). The export
  concern is entirely in the (opt-in, speculative) script.
- Update `test/test_mlx_export.py`: build the adapter from `export_mlx.py`, not model
  methods.
- GATE: `test/test_mlx_export.py -q` (3 pass / 2 skip); `BasicModel.forward()` green.

## B. `BasicModel.reverse()`

- Rename `_run_pipeline_rev_from_concepts` (7112) -> `reverse`; update the caller (3599)
  + comments. Post-C3 this is the single reconstruction reverse.
- Keep `_reverse_from_S` (serial, takes the single S) and `_run_pipeline_rev` (the
  off-dispatch head primitive) with accurate docstrings.
- GATE: `test_ir_only_refactor.py` (concepts-unconditional) + `test_dimensional_governance.py`
  (deep-CS reverse) green.

## C. Muxed transform (option B, resolved) -- DONE 2026-06-06

Design: `doc/specs/2026-06-06-muxed-events-and-positional-bands.md` (read it). The
band stays UNIFORM `(2,2)` on every interior tier (user: "preserve those as two,
but keep them muxed" / "as = 2") -- nWhere/nWhen are NEVER declared 0 on an
interior tier. The transform operates on the FULL muxed event on the PER-POSITION
operators (butterfly/last Pi/Sigma + the per-stage `ConceptualCombine`), so
`.where`/`.when` participate. The ONE exception is the `full`-mode dense
wide<->deep bridge, which stays content-only (the uniform band is a per-position
identity pass-through -- a band cannot be reshaped across a position-count
change). `.where`/`.when` stay RECONSTRUCTED (RoPE-like; keep the loss pressure);
the free-neural-line / `WhereEncoding.recover` stays retired. (This SUPERSEDES the
earlier band-free-CS exploration, which collided with the `SS.nWhat == CS.nWhat`
handoff and forced config re-dimensioning -- not the chosen design.)

The `ConceptualCombine` is also moved to be HELD BY its `ConceptualSpace` (user:
"Conceptual combine is best done as a Layer in Layers.py and held by CS"), cf. PS
owns `pi`, SS owns `sigma`.

- `bin/architecture.py` `_CANONICAL_SHAPE`: UNCHANGED -- uniform `(2,2)` on every
  interior tier, `(0,0)` only on OS. `test/test_canonical_shape.py` unchanged.
- `bin/Models.py` per-stage combine: size at `cs.muxedSize` (full muxed event, not
  `cs.nWhat`) so the band participates. HOLD it on the cs: append to `cs.layers`
  (paramUpdate/set_sigma cascade) + add to `cs.params` (getParameters walk);
  expose as `cs.combine` (via `object.__setattr__`, cs.layers is the real
  registration). Forward/reverse body read `cs.combine` and demux each stream AT
  `D = cs.muxedSize` (whole event, empty band). REMOVE the model-level
  `conceptual_combines` ModuleList and the three model-level loops (param collect /
  paramUpdate / set_sigma) -- the space walk now reaches the combine via the cs.
  REMOVE the dead `_symbolic_sigma_step` (content-only SS.sigma advance the combine
  replaced); repoint `test/test_cs_reentrancy.py` at `ss.sigma` directly.
- `bin/Layers.py` `ConceptualCombine`: unchanged math; doc the `content_dim` arg as
  the full muxed-event width.
- `bin/Spaces.py`: PS.`pi` / SS.`sigma` in butterfly/last already square at the full
  event width (band included) -- no change. Fix the stale SS `full`-mode comment
  that claimed SS is `(0,0)` (it is `(2,2)`; the band passes through the dense
  bridge like PS).
- Flat-slab / `validate_config` / `reverseBegin` stay CONTENT-based; configs do NOT
  re-dimension (uniform band; the combine just sizes to `muxedSize`).
- KEEP the `.where`/`.when` reconstruction loss terms (`ModelLoss.compute*`).
- GATE (all green 2026-06-06): `test_canonical_shape` (3), `test_cs_reentrancy` (7),
  `test_conceptual_recurrence` (11), `test_dimensional_governance`+`test_invertibility`
  (87), `test_modality_configs` (1+34 subtests), `test_mlx_export` (3 pass/2 skip),
  `test_partition_reconstruction_stack` (3), `test_ir_fullgraph_compile` (2).
  Ownership verified on MM_20M: `cs.combine.content_dim == cs.muxedSize == 1024`,
  registered in cs.layers + cs.params, reaches the optimizer; no model-level list.

## D. IS-always-RAW + PS owns all lexing + remove `_peer_perceptual`

Status 2026-06-07: IN PROGRESS (user-directed redesign). The user chose: "remove the
lexer [variation], force [IS] to emit RAW in all cases, remove the peer, and add
lexing for each bpe/radix/lexicon within PS as methods. Then we can work to eliminate
the graph breaks." So graph breaks (fullgraph gate) are DEFERRED to a follow-up; this
step lands the clean structure and gates on the EAGER paths.

### Validated coupling map (what the embed methods actually need)

- `<lexer>` (`word|sentence|byte|raw`) is on **InputSpace** (`Spaces.py` ~6867,
  `self.lexer`/`self.byte_mode`). `<chunking>` (`radix|bpe|analyse|none|mphf|lexicon`)
  is on **PerceptualSpace** (~7833). They compose; `raw` already means "IS emits the
  unanalyzed `[B,1,N]` byte buffer; PS owns tokenization".
- PS `_embed_*` already read the RAW byte buffer (`upstream.materialize(mode="what")`)
  + a host-surface (`_host_tokens` / `decode_tokens`). Self-lexing TODAY:
  - `analyse` (`_embed_analyse`): reconstruct surface -> `parse(lex="words")` (+ learned
    merges) -> `_embed`. SELF-LEXES. ✓
  - `bpe` (`_embed_bpe`/`_gpu`/`_trie`): reads raw `byte_indices`, longest-match + chunk
    + segment. SELF-LEXES. ✓
  - `radix` (`_embed_radix`): whitespace-splits the surface -> PerceptStore lookup.
    SELF-LEXES. ✓
  - `lexicon` (`_embed`): expects `_host_tokens` ALREADY word-tokenized by IS. NEEDS a
    parse-words front-end (= `_embed_analyse` minus merges). ✗
  - `none`/`mphf`: DEAD (no config). Remove.

### Stages (each gates on the EAGER paths; fullgraph deferred)

1. **Lexicon front-end in PS** (additive, safe): give `lexicon` its own parse step so
   `_embed` self-lexes the surface (route `lexicon` -> parse-words -> resolve, mirroring
   `_embed_analyse` without merges). Gate: XOR_exact, LM_5M_IR build+forward.
2. **IS always RAW**: `InputSpace._lex_and_embed`/`_lex_batch` always emit the raw byte
   buffer + whole-line surface `_host_tokens`, independent of `<lexer>` (which becomes
   inert / removable). IS stops borrowing PS's tokenizer. Gate: modality_configs +
   text configs (MM_20M analyse, MM_xor radix, MM_bpe bpe, XOR_exact lexicon).
3. **Remove `_peer_perceptual`**: init (`Spaces.py` ~6901), wiring (`Models.py`
   ~4645-4662), `_lex_batch` tokenizer use (~7090-7109), numeric-bookkeeping refs
   (~7356/7387/7405/7463 -> just use None), `Models.py` ~2186 (use `self.perceptualSpace`).
   Drop dead `_embed_byte`/`_embed_mphf` + dispatch branches. Update tests that build
   `_peer_perceptual` (`test_basicmodel.py` ~86, `test_input_word_cursor.py`,
   `test_lexicon_ownership.py` ~349, `test_unified_lexicon_codebook.py`).
4. **DEFERRED (separate)**: eliminate the fullgraph graph breaks — the host-side
   tokenization (regex `parse`, BPE trie) now runs in PS.forward inside the compiled
   `_forward_body`. Options: hoist the embed to an eager pre-stage in `_begin_step`
   (model-orchestrated), or make the per-mode lex tensor-only. Tracked separately;
   `test_ir_fullgraph_compile` stays RED until then.

### Original-premise correction (kept for context)

Goal (unchanged): IS emits `[byte(1) + where(2) + when(2)]`; PS does the embedding
(codebook index by byte); the `InputSpace._peer_perceptual` back-ref to PS is removed,
making IS->PS a clean forward handoff instead of IS reaching INTO PS.

### What `_peer_perceptual` actually is (corrected map -- read before implementing)

The original premise ("`_lex_batch` looks up the PS codebook via the peer") is WRONG.
`InputSpace._lex_batch` (`bin/Spaces.py` ~7073) is ALREADY a pure lexer (its own
docstring: "no codebook access ... those live on PerceptualSpace"); it emits
`what_buf` (UTF-8 bytes), `where_idx`, `when_idx`. The peer is used in FOUR distinct
places, only one of which is the tokenizer:

1. **Tokenizer borrow** (`_lex_batch` ~7102-7109): `peer.subspace.what._token_stream`
   / `peer.vocabulary._token_stream`. IS already has its own `byte_mode`
   (`parse(text, lex="bytes")`) and `_radix_token_stream`; word-mode would call
   `Embedding._token_stream` (a host method) directly. LOW risk to relocate.
2. **Embedding DISPATCH** (`InputSpace.forward`/`_lex_and_embed` ~7301-7331): IS DRIVES
   PS's embedding -- calls `peer._embed_bpe` / `_embed_byte` / `_embed_mphf` /
   `_embed_radix` / `_embed_analyse` / `_embed` keyed on `peer.chunking_mode`, then
   reads back `peer._embedded_input`. THIS is the real inversion: those `_embed_*`
   methods must run FROM `PerceptualSpace.forward(in_sub)` (the pipeline already calls
   it at `bin/Models.py` ~5405/6217) on the raw IS byte event, instead of being pulled
   from IS. HIGH risk (5 modes, OOV-insert write paths, BPE fuse).
3. **Valid-len + mask + width** (`_lex_and_embed` ~7356-7408): IS reads
   `peer._bpe_word_mask`, `peer.chunking_mode`, `peer.outputShape[0]` to compute
   `inputSpace._valid_len_host` and the per-position mask, and to hand off the PEER's
   word-reduced width (e.g. MM_20M 8192 chars -> 8 words). These are properties of the
   EMBEDDED output (PS produces it) but are cached ON IS because IS drives the embed.
4. **IR diagnostic** (`bin/Models.py` ~2186): nearest-neighbor decode for masked
   predictions -- ALREADY falls back to `self.perceptualSpace`; trivially de-peered.

### Why HIGH risk (the blocker)

`inputSpace._ar_embedded` (the `[B,T,D]` embedded sequence) and
`inputSpace._valid_len_host` are consumed by ~15 sites: the AR-streaming cursor +
`next_word` (`bin/Spaces.py` ~6930-6955), the CUDA-graph **static-N word loop**
(`bin/Models.py` ~6419-6449, `K_host = inputSpace._valid_len_host`), IR/AR training
targets (~7082/7349), masked-prediction, and `generate_sentence`. They live on IS
because IS produces the embedded output today. Inverting (PS embeds) forces relocating
that ownership to PS (or IS reads it back -- re-introducing a cross-ref) across the
most CUDA-graph-/host-sync-fragile subsystem (`test_ir_fullgraph_compile`,
AR-streaming brick capture). That is why it cannot land cleanly in one session
alongside C.

### Staged implementation strategy (for the future session)

1. **Tokenizer to IS** (item 1): give InputSpace a self-owned `_tokenize` (byte/word/
   radix) with NO peer. Gate: `test_modality_configs`, lex tests. Small, safe first step.
2. **Move embed ownership into `PerceptualSpace.forward`** (items 2-3): `PS.forward(in_sub)`
   detects a raw IS byte event and runs the right `_embed_*` on it, producing the embedded
   event + `_bpe_word_mask` + word-reduced width ON PS. IS.forward becomes lex-only and
   returns the raw `[byte|where|when]` event. Relocate `_ar_embedded` / `_valid_len_host`
   to PS (or a shared handle the AR cursor reads from PS). Gate per chunking mode:
   MM_20M (word/bpe), LM_* (byte/none), MM_grammar; plus `test_ir_fullgraph_compile`,
   AR-streaming + brick-capture tests, `test_input_word_cursor`, `test_basicmodel`.
3. **Remove the peer**: delete `InputSpace._peer_perceptual` init (`bin/Spaces.py` ~6901)
   + the wiring (`bin/Models.py` ~4645-4662); de-peer `Models.py` ~2186; update the tests
   that construct `_peer_perceptual` (`test_basicmodel.py` ~86, `test_input_word_cursor.py`
   ~91/226, `test_lexicon_ownership.py` ~349, `test_unified_lexicon_codebook.py` refs).
4. **Configs**: confirm whether IS `nDim` is the thin `1+2+2` muxed form or stays the
   `nWhat`-byte buffer (the user's "1-dim byte value" suggests byte-mode 5-wide; word/bpe
   modes lex multi-byte tokens -- resolve this with the user, it changes IS.nDim).

NOTE (open design question for the user): the original "IS passes a 1-dim byte" fits
byte-mode (one byte/slot). Word/BPE/MPHF/radix modes lex MULTI-byte tokens; under those,
"IS emits bytes, PS chunks" means PS does the BPE/word fusion on the byte stream it
receives. Confirm the intended granularity before item 2.

- RISK: HIGH (re-architects the lex/embed boundary + AR-streaming ownership). The
  off-ramp ("stop at green post-C") was taken 2026-06-06.

## Test runner

`KMP_DUPLICATE_LIB_OK=TRUE BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python -m pytest <nodeid> -q`
(targeted node ids; `test_basicmodel.py` fails ~24 on clean HEAD; order-fragile).
Git is the user's -- no git writes.
