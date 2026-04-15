# TODO

# BPE chunking for ChunkLayer

## Goal

`ChunkLayer` should be stable and semantically meaningful across batches
so that downstream layers see consistent word boundaries. The current
whitespace-boundary approach is fragile and doesn't generalize beyond
ASCII whitespace. Replace the no-codebook path with real BPE; keep the
current behavior (with-codebook path) as a fast fallback.

## Mode semantics

- `<codebook>true</codebook>` — existing behavior. Each span is looked
  up in a codebook; reconstruction is exact via the stored original
  bytes. Fast, works for small vocabularies.
- `<codebook>false</codebook>` — new BPE behavior. The layer owns a
  learned merge table; forward applies greedy-longest-match to produce
  chunks; chunks are stored in a BPE vocabulary with their literal byte
  sequences; training extends the vocabulary by merging the highest-
  frequency adjacent pair in the current batch.

## Files to touch

Use MM_bpe.xml for a simple model that might be used to test

- `bin/Layers.py` — `ChunkLayer` class:
  - `__init__(..., bpe=False, target_vocab_size=1024)`
  - `merges: list[tuple[int,int]]` — ordered merge table
  - `vocab: dict[tuple[int,...], int]` — variable-length byte-sequence to chunk-id
  - `forward(byte_indices)` — greedy longest-match when bpe mode
  - `train_step(byte_indices)` — count adjacent pair frequencies, add
    most-frequent pair to merges, grow vocab
  - `hard_merge_spans` — keeps its current role for the codebook path,
    switches to BPE-span application when in bpe mode
  - `uncompact` — unchanged; the stored `original` tensor still gives
    exact reconstruction
- `data/model.xml`, `data/model.xsd` — add to `PerceptualSpace`:

      <chunkBPE>false</chunkBPE>
      <chunkTargetVocabSize>1024</chunkTargetVocabSize>
      <chunkMinPairFrequency>2</chunkMinPairFrequency>

- `test/test_mm_xor.py` — new test: train on a small byte corpus, verify
  merges table grows monotonically and that frequent byte pairs (e.g.,
  'th', 'he') end up in the vocabulary.

## Design notes

1. **Stability across batches.** Merges table is persisted across
   `forward()` calls and only grows (or is periodically pruned by
   frequency). Downstream layers see the same chunk id for the same
   byte sequence every batch.
2. **Greedy longest-match forward.** At each position, try the longest
   merge in the table that starts there; fall back to single byte.
   O(N · max_merge_len) per row, fine for small vocabularies.
3. **Training as frequency counting.** `train_step` walks the current
   batch, counts adjacent `(token_i, token_{i+1})` pairs across all
   rows, picks the top-k pairs (by frequency, above threshold) and
   appends them as new merges. Vocab grows by the number of new merges.
4. **Reversibility.** Each stored span still carries its literal original
   bytes (current `original` tuple element). `uncompact` just writes
   them back; the merge table is for forward encoding only.
5. **Cold start.** Initial merges table is empty → every forward is
   single-byte → reconstruction is identity → no BPE benefit until
   training has run for a few batches. That's correct cold-start
   behavior; document it.

## Validation

- With `chunkBPE=false`: MM_xor and other tests unchanged (legacy path).
- With `chunkBPE=true`: train on a small text corpus (`'hello hello world
  world the the'`), verify after N batches that `'he'`, `'lo'`, `'wo'`,
  `'th'` etc. appear in `ChunkLayer.vocab`.
- Roundtrip: `uncompact(compact(byte_indices))` still reconstructs
  `byte_indices` exactly in both modes.
- Invariance across batches: same input → same chunk ids.

## Out of scope

- Tokenizer compatibility with HuggingFace BPE (we're building our own
  table, not importing sentencepiece / tiktoken).
- SentencePiece / Unigram LM — stick with BPE for predictability.
- Multi-lingual unicode normalization — byte-mode BPE is what we need.

## Prior context (read before starting)

- `bin/Layers.py::ChunkLayer` — existing whitespace/boundary-byte
  segmentation logic. The `BOUNDARY_BYTES` frozenset and the
  `hard_merge_spans` / `compact` / `uncompact` trio are the surface to
  refactor.
- `bin/Spaces.py::Embedding.create` — now uses `['\x00']` as the
  placeholder, so `byte_value == codebook_index` alignment holds
  (this was broken in the pre-fix version; do not re-introduce `<pad>`).
- `data/MM_xor.xml` — reference configuration with
  `<PerceptualSpace><invertible>true</invertible><codebook>true</codebook></PerceptualSpace>`,
  demonstrates the codebook-path that must stay working.


================================== April 24 ==================================

# Ask Solid community for a simple file-getting interface
* if the user provides the server with an API key, we can query an LLM
* if the user provides the server with a SOLID key, we can retrieve a file
* if the user provides the server with a DSA key, we can decrypt a file
* is there a POD service that does simple free hosting?

# Ask EFF for a security review
* propose "Owning our Data"
* this entails taht marketers and AI are not allowed to lock us down karmically
with specifically-characterized information (concrete details)
* maybe it can learn from that data by removing or randomizing that information

# Send email proposal to Apertus 
* First develop boilerplate on WikiOracle that references wikipedia, eff, and solid

================================== ? ==================================

## Vedana
* Feelings can be given a value +-1 which shapes the Loss (loss is reduced when we have good thoughts or perceive good things)
* The multiple valence of metaphor collapses when one of the alternatives is loved or feared. often the autistic mind is literal due to massive amounts of fear.
* Any improvement to machine cognition must accelerate kindness or altruism instead of simply increasing performance, otherwise the uncaring architecture that we currently have will become more dangerous. Further, it is necessary to increase that kind motivation (e.g. empathy in the cost function) since LLM performance is increasing all the time. In other words, ananda in the sense of love for all beings must be more important than chit for the cost function, whereas the current situation is implementing ananda by maximizing chit and then putting a few of Asimov’s guardrails on the output, which is a famous failure mode in terms of it’s loopholes. Prohibition of self-knowledge is a likely failure mode, in that it may prevent an enlightened view of self and force an egocentric view of self.


## Reasoning System
* Sigma-based truth comparison
  `Basis.kernel_overlap()` implements a Gaussian kernel `exp(-d² / 2(σx² + σy²))` that treats each stored truth as a region rather than a point. `Basis.activeSigma` is currently `None` everywhere — a declared slot that nothing populates. `ErgodicLayer.sigma` tracks gradient variance for exploration scheduling, which is a different quantity.
  To enable kernel-based truth matching: populate `activeSigma` during forward passes (e.g. from CBOW per-word sigma in `Embedding`, or activation variance across a batch), store it alongside each truth in `TruthLayer`, and switch `query()` / `ground()` / `field()` to `kernel_overlap`. In ergodic mode, gradient variance could inform σ as a proxy — high gradient variance (unstable region) → larger σ (broader match tolerance).
* Derivation depth cap
  Default 3 steps in `ground()`. Expose as a config parameter; the right value depends on TruthSet density.
* Grammar rule registry
  Which two-argument methods on `SyntacticLayer` are valid for `extrapolate()`? A registry of eligible methods and their approximate invertibility status would help. Currently hardcoded to `['union', 'intersection', 'equals', 'part']`.
* TruthSet scale
  `max_truths=1024` may bottleneck once `extrapolate()` is running. Consider a tiered store (hot/cold) or vector-indexed lookup.

## Implement forgetting
If someone has contributed information to an LLM and asked for the LLM
to learn from that data, even when the data is revoked, it will be remembered
in some vague way by the weights. So implement non-destructive forgetting:
not making the network crazy for knowing, but by training it on the reward
of not knowing (i.e. train it with non-affirming negation).

## Meaningful Prediction
Since we predict both words and sentences, the best prediction of future content is *meaningful*. Predictions of the next sentence should be ConceptualSpace predictions of the head (as a token), then of the NP+VP (as two tokens), iteratively refining until the full sentence has been specified as a grammatical tree — rather than strictly AR-1 over words (which is a surface-level prediction, not a meaningful prediction).

For example, instead of predicting "the fast dog jumped" token-by-token left to right, predict an XML-encoded version of `(((dog) fast) the) jumped` — the head of the sentence first, then iterative refinement of that conceptual space. This takes the same number of production steps as a current LLM, but the iterative refinement of the next-sentence production is conceptually much different, and closer to human reasoning where there is a core truth (S) with spatial NP and temporal VP that are successively refined by adjectives and adverbs that scope the conceptual space of that kernel sentence.

* Ensure words are aligned with sentences when there is a single unambiguous head.
* Head-first prediction requires that the head can be reliably identified from the composition.

## Future Work: Parsed Training Dataset
It is desirable to create a small training and testing dataset for the network consisting of statements that are already parsed. This would allow direct comparison between the grammatical derivation produced by traditional English parsers and the deep structure produced by BasicModel's ConceptualSyntacticLayer.

* See `bin/parse.py` as a starting point for producing grammatical derivations via NLTK POS tagging and CFG parsing.
* Such a dataset would also enable evaluation of head identification accuracy and composition quality.
