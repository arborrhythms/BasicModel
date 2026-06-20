# PerceptualSpace BPE Chunking: Config Consolidation + Forward Path Wiring

## Context

Today BPE on `PerceptualSpace` is half-wired across two parallel config knobs and a dead code path:

1. **Two overlapping XML switches.** `<chunking>bpe|lexicon</chunking>` ([Spaces.py:5098-5102](../../bin/Spaces.py)) sets `self.chunking_mode`. `<chunkBPE>true|false</chunkBPE>` ([Spaces.py:5110-5113](../../bin/Spaces.py)) sets `self.chunkBPE`. They encode the same decision.
2. **Two overlapping vocab-size fields.** `<nVectors>` on `PerceptualSpace` is the codebook capacity ([Spaces.py:4027](../../bin/Spaces.py) — `spaceShape[0]`). `<chunkTargetVocabSize>` is the BPE merge-table target ([Spaces.py:5114-5118](../../bin/Spaces.py)). With the word-level MAX-fusion design below, each BPE token has one codebook slot, so these are the same number.
3. **BPE forward path unwired.** `PerceptualSpace.forward` at the text-mode branch raises `NotImplementedError` for `chunking='bpe'` ([Spaces.py:5438-5440](../../bin/Spaces.py)). The `ChunkLayer` built at [Spaces.py:5136-5141](../../bin/Spaces.py) is never called from `forward`. The only existing BPE test ([test_chunk_layer_bpe.py:151-175](../../test/test_chunk_layer_bpe.py)) constructs the layer directly without routing through `PerceptualSpace`.
4. **Schema drift.** `model.xsd` allows `<chunking>raw|bpe|lexicon</chunking>` ([model.xsd:294-296](../../data/model.xsd)), but the runtime dispatch expects `lexicon|cached|bpe` ([Spaces.py:5442-5443](../../bin/Spaces.py)). `raw` in XML raises at runtime.

The goal: make `<chunking>bpe</chunking>` on `MM_5M.xml` actually run BPE end-to-end, and consolidate the config surface so there is one way to say "turn on BPE."

**User directional choices (confirmed during brainstorming):**
- Clean break: remove legacy fields in a single change, no deprecation window.
- Unified vocab field: `<nVectors>` is the single knob for sub-token codebook size (which equals BPE vocab size under word-level MAX fusion).
- Word boundary: whitespace. Within a whitespace-bounded word, BPE segments into sub-tokens; sub-token vectors are MAX-fused into one word vector.
- Fuser: **MAX** (elementwise `max` in perceptual `[0,1]^d` space). Chosen for orthogonalization: orthogonal sub-token active-dim sets union under MAX, preserving constituent distinguishability.
- `raw`/`cached` mode is dropped — a cached/identity pass-through is "not sufficiently consistent in value to be trainable."

## Target Config Surface

On `PerceptualSpace` only:

```xml
<chunking>bpe|lexicon</chunking>
<nVectors>4096</nVectors>                    <!-- sub-token codebook, >=256 required when chunking=bpe -->
<chunkingFrequency>2</chunkingFrequency>     <!-- BPE merge-learning min pair frequency -->
```

**Removed from XSD, `Spaces.py`, and every XML file in `basicmodel/data/`:**
- `<chunkBPE>` element
- `<chunkTargetVocabSize>` element
- `<chunkMinPairFrequency>` element (renamed to `<chunkingFrequency>`)
- `raw` enum value on the `<chunking>` simpleType

**Validation at `PerceptualSpace.__init__`:**
- If `chunking == 'bpe'` and `nVectors < 256`: raise `ValueError` (can't seed the byte range).
- If `chunking == 'bpe'` and `model_type != 'embedding'`: raise `ValueError` (BPE only meaningful when an `Embedding` basis exists).

## Data Flow: `chunking='bpe'` in `PerceptualSpace.forward`

Replaces the `NotImplementedError` at [Spaces.py:5438-5440](../../bin/Spaces.py).

Input: byte buffer from `InputSpace.subspace.what.getW()` — shape `[B, N_bytes]`, with whitespace boundaries preserved in the byte stream.

1. **Whitespace word split.** Using `ChunkLayer.BOUNDARY_BYTES` ([Layers.py:3856](../../bin/Layers.py)), partition each batch row into a list of word-byte-ranges. Whitespace bytes themselves are not emitted — they only mark boundaries.
2. **BPE tokenize each word.** Call `chunk_layer.forward(word_byte_indices)` (existing, [Layers.py:3878-3934](../../bin/Layers.py)) — greedy longest-match against `self.vocab`. Byte-level fallback guarantees every word decomposes into ≥1 in-vocab sub-tokens.
3. **Embed each sub-token.** Look up each `chunk_id` in the `PerceptualSpace` codebook `W: [nVectors, nDim]` via `self.subspace.what` (the `Embedding` basis). One `[nDim]` vector per sub-token.
4. **MAX-fuse within word.** For each whitespace-bounded word containing sub-tokens `s_1, ..., s_k`: `word_vec = elementwise_max(emb[s_1], ..., emb[s_k])`. Output one `[nDim]` vector per word.
5. **Emit `[B, nOutput, nDim]`.** One row per batch item, one position per word. Pad/truncate to `nOutput` following the same convention as the lexicon path (NULL sentinel at positions beyond word count).

**Training-time merge learning.** When `self.training` is True, `ChunkLayer.train_step()` ([Layers.py:3936-3981](../../bin/Layers.py)) runs once per batch before step 2, counting adjacent-pair frequencies across this batch's tokenization and promoting the top pair if its count ≥ `chunking_frequency`. `train_step` stops adding merges once `len(self.vocab) == nVectors`. This is the existing pattern used by `hard_merge_spans` ([Layers.py:4004-4007](../../bin/Layers.py)).

**No cross-word merges.** `ChunkLayer.forward` already segments on per-word byte ranges when called with pre-split words, so whitespace-bounded isolation falls out of the word-split step.

## Implementation Layers

### `ChunkLayer` ([Layers.py:3767](../../bin/Layers.py))

`ChunkLayer` becomes a pure symbolic tokenizer. The `self.merge` / `self.split` / `self.threshold` 256-prototype table was auxiliary to an unused pair-scoring path and is dropped.

**Keep:**
- State: `self.vocab`, `self.merges`, `self.id_to_bytes`, `self._next_id`, `self._max_merge_len`, `self.bpe`.
- Methods: `forward(byte_indices)`, `train_step(byte_indices, k_merges)`, `is_word_boundary`, `BOUNDARY_BYTES`.

**Remove:**
- `self.split`, `self.merge`, `self.threshold` parameters (nn.Parameter-backed).
- Methods: `score_pair`, `encode`, `decode`, `should_merge`.
- Constructor arg `nChunks` (was 256, only used to shape the dropped prototype tables).

**Rename constructor args:**
- `target_vocab_size` → `n_vectors`
- `min_pair_frequency` → `chunking_frequency`

**Callers of removed methods:** verify via grep that nothing else depends on `score_pair`/`encode`/`decode`/`should_merge` before removal. If anything does, adjust scope.

**Existing consumers of `hard_merge_spans`** ([Layers.py:3985-4033](../../bin/Layers.py)) — legacy whitespace-mean aggregation — are unaffected; they run only when `self.bpe=False` and are unrelated to the new `PerceptualSpace.forward` BPE path.

### `PerceptualSpace.__init__` ([Spaces.py:5098-5141](../../bin/Spaces.py))

Replace the current three config blocks (chunking / chunkBPE / chunkTargetVocabSize / chunkMinPairFrequency) with:

```python
try:
    self.chunking_mode = str(TheXMLConfig.space(section, "chunking") or "lexicon")
except KeyError:
    self.chunking_mode = "lexicon"
if self.chunking_mode not in ("bpe", "lexicon"):
    raise ValueError(f"PerceptualSpace.chunking must be bpe|lexicon, got {self.chunking_mode!r}")
try:
    self.chunking_frequency = int(TheXMLConfig.space(section, "chunkingFrequency") or 2)
except (KeyError, TypeError, ValueError):
    self.chunking_frequency = 2
if self.chunking_mode == "bpe":
    if self.nVectors < 256:
        raise ValueError(
            f"PerceptualSpace.chunking='bpe' requires nVectors>=256 (byte seeding); got {self.nVectors}")
    if self.model_type != "embedding":
        raise ValueError(
            "PerceptualSpace.chunking='bpe' requires modelType='embedding'")
if isinstance(lexical_basis, Embedding):
    lexical_basis.chunking_mode = self.chunking_mode
    lexical_basis.lexer_mode = self.lexer
self.chunk_layer = ChunkLayer(
    self.nDim,
    bpe=(self.chunking_mode == "bpe"),
    n_vectors=self.nVectors,
    chunking_frequency=self.chunking_frequency,
)
```

### `PerceptualSpace.forward` ([Spaces.py:5431-5443](../../bin/Spaces.py))

Replace the text-mode dispatch:

```python
if isinstance(self.subspace.what, Embedding) and not vspace.stem_embedded:
    mode = self.chunking_mode
    if mode == "lexicon":
        vspace = self._embed(vspace)
    elif mode == "bpe":
        vspace = self._embed_bpe(vspace)
    else:
        raise ValueError(f"PerceptualSpace chunking must be bpe|lexicon, got {mode!r}")
```

New method `_embed_bpe(vspace)` implements the five-step data flow above. Structure mirrors `_embed`: input from `vspace.what.getW()`, output assigned to `self.subspace.event` via `set_forward_content`, `[B, nOutput, nDim]` shape.

### Schema ([model.xsd:288-299](../../data/model.xsd))

- Remove `<xs:element name="chunkBPE">`.
- Remove `<xs:element name="chunkTargetVocabSize">`.
- Remove `<xs:element name="chunkMinPairFrequency">`.
- Add `<xs:element name="chunkingFrequency" type="xs:positiveInteger" minOccurs="0"/>`.
- `<chunking>` `xs:restriction`: drop `raw`; keep only `bpe` and `lexicon`.

### Drop `raw` from every runtime site

- **`PerceptualSpace.chunk_static`** ([Spaces.py:5242-5262](../../bin/Spaces.py)): remove the `if mode == "raw"` branch and update the docstring / error message to list `bpe|lexicon` only.
- **`Embedding._token_stream`** ([Spaces.py:1809-1820](../../bin/Spaces.py)): replace `if mode in ('raw', 'bpe'): return self._char_stream(text)` with `if mode == 'bpe': return self._char_stream(text)`. (BPE still falls back to char stream at this layer — ChunkLayer handles byte-level segmentation downstream.)

## XML File Migration

| File | Change |
|------|--------|
| `basicmodel/data/MM_5M.xml` | Remove `<chunkTargetVocabSize>4096</chunkTargetVocabSize>`. Rename `<chunkMinPairFrequency>2</chunkMinPairFrequency>` → `<chunkingFrequency>2</chunkingFrequency>`. Set `<nVectors>4096</nVectors>` (down from 8192 — matches the prior BPE target and aligns slot count with vocab). |
| `basicmodel/data/MM_bpe.xml` | Replace `<chunkBPE>true</chunkBPE>` + `<chunkTargetVocabSize>512</chunkTargetVocabSize>` with `<chunking>bpe</chunking>` + `<nVectors>512</nVectors>`. Rename frequency tag. |
| `basicmodel/data/model.xml` | Remove `<chunkBPE>false</chunkBPE>` and `<chunkTargetVocabSize>1024</chunkTargetVocabSize>`. Keep `<chunking>lexicon</chunking>`. Rename `<chunkMinPairFrequency>` → `<chunkingFrequency>` if present. |

## Tests

### Updates to existing tests

- **`basicmodel/test/test_chunk_layer_bpe.py`** ([test/test_chunk_layer_bpe.py:151-175](../../test/test_chunk_layer_bpe.py)): Replace `cfg.space("PerceptualSpace", "chunkBPE")` / `"chunkTargetVocabSize"` / `"chunkMinPairFrequency"` with `"chunking"` (assert `== "bpe"`), `"nVectors"`, `"chunkingFrequency"`. Update `ChunkLayer(...)` call to the renamed constructor args.
- **`basicmodel/test/test_perceptual_chunking.py`**:
  - Remove `test_chunking_mode_raw_returns_bytes` ([line 5-9](../../test/test_perceptual_chunking.py)) and `test_embedding_token_stream_honors_raw_chunking_mode` ([line 45-50](../../test/test_perceptual_chunking.py)) — the `raw` mode is gone.
  - Keep `test_chunking_mode_bpe_returns_learned_segments`, `test_chunking_mode_lexicon_splits_on_spaces`, `test_chunking_invalid_mode_raises`, `test_perceptual_space_exposes_chunking_mode_attribute`, and `test_embedding_token_stream_honors_bpe_fallback_mode`. These stay compatible with the new `bpe|lexicon` surface.

### New tests (add to `test_perceptual_chunking.py` or a new `test_perceptualspace_bpe.py`)

1. **MAX fusion correctness.** Seed a `PerceptualSpace` with a known codebook (manually set `W`). Feed a byte buffer containing one word whose BPE tokenization is two known sub-tokens `s_1`, `s_2`. Assert the output at that word's position equals `torch.max(W[s_1], W[s_2])` elementwise.
2. **OOV byte-level fallback.** With no learned merges (fresh `ChunkLayer`), feed a word of several bytes. Assert output is `torch.max(W[b_1], ..., W[b_k])` — non-zero, correctly fused.
3. **Vocab cap.** Construct with `nVectors=260`. Run `train_step` repeatedly over a varied corpus. Assert `len(chunk_layer.vocab) <= 260` after many calls.
4. **Construction validation.** `PerceptualSpace` with `chunking='bpe'` and `nVectors=128` raises `ValueError`. With `nVectors=256` it succeeds.
5. **MM_5M smoke test.** Load `basicmodel/data/MM_5M.xml` via `init_config`, build the model, run one forward pass on a small ASCII byte buffer (e.g., `b"hello world foo"`), assert output shape is `[1, nOutput, nDim]` and is finite.

## Out of Scope

- Higher-level `Fusion()` / mereological composition at `ConceptualSpace` / `SymbolicSpace` — MAX here is the perceptual-tokenization fuser only. Higher spaces keep their current fusion operators.
- Pre-training the BPE merge table offline. Merges learn on-the-fly via `train_step` during training, exactly as the existing `hard_merge_spans` path does.
- Wiring the `cached` / `raw` mode. The user has decided identity pass-through lacks training signal and is dropping the mode outright.
- Non-PerceptualSpace consumers of `<chunking>` (e.g., `ConceptualSpace` uses `<chunking>lexicon</chunking>` in MM_5M.xml). Unchanged by this work.

## Risks

- **Removing `self.merge`/`self.split` may have external consumers.** Grep under `basicmodel/bin` and `basicmodel/test` for `chunk_layer.merge`, `chunk_layer.split`, `.score_pair(`, `.should_merge(` before deletion. Any hit is either removed with the caller or the deletion is narrowed.
- **Padding convention for `nOutput`.** The lexicon path's NULL-sentinel handling for "fewer words than `nOutput`" must match what downstream `ConceptualSpace` / attention expects. The implementation should reuse whatever lexicon uses (likely zero-padding at tail positions); not invent a new convention.
- **Word-vector content differs from lexicon even at identical shape.** Both `lexicon` and `bpe` emit `[B, nOutput, nDim]` with one position per whitespace-bounded word. But `lexicon` does a single codebook lookup per word-string (OOV → unknown token); `bpe` composes the word vector as MAX of its sub-token embeddings, so OOV handling is graceful (fall to byte-level sub-tokens) and identical surface words always produce identical vectors. Downstream code that depends on codebook row identity rather than vector content may need audit.
