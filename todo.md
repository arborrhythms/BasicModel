# TODO

## Immutable Source Buffer + Span-Table Architecture

### Problem

Data currently lives in multiple forms: `Data()` stores tensors per split
(train_input, test_input, etc.), but text data passes through `LanguageModel`
as Python lists of strings.  `tokenizeList()` splits strings on spaces and
builds a Python-level vocabulary list.  `tokenize()` decodes byte tensors
back to strings, splits on words, and returns `List[List[str]]`.  This
means every batch re-tokenizes from scratch, there is no shared flat
representation that downstream spaces can slice into, and only one
tokenization can exist at a time.

### Proposal

#### 1. Immutable source buffer in Data()

Store the raw text once as an immutable `uint8` tensor per split:

```
data.train_source  : Tensor [total_bytes]  dtype=uint8   # raw UTF-8 bytes
data.train_input   : Tensor [N, D]                       # numeric (MNIST, XOR)
```

The source buffer is written once at load time and never mutated.  For
numeric datasets (simple, ergodic, xor), nothing changes — `train_input`
is already a flat `[N, D]` tensor and `toDevice()` pre-shapes it to
`[N, D, 1]`.  The source buffer only exists for text datasets.

#### 2. InputSpace creates span tables and encodes tokens as vectors

`InputSpace` receives the source buffer from `Data()` and delegates
tokenization to `Lex`, which produces a span table:

```
spans : Tensor [num_spans, 3]   # (start, end, type)
```

- `start`, `end` — byte offsets into the source buffer.
- `type` — integer token ID (vocabulary index).

Each span is then encoded as a vector with two meaningful components:

- **nWhat** — the content of the token, looked up via VectorSet codebook
  from the `type` field.  When the VectorSet is loaded from word2vec or
  similar pretrained embeddings, this quantization step naturally corrects
  misspellings: a misspelled word snaps to the nearest codebook entry
  (e.g. "teh" → "the") via the existing VQ snap-distance mechanism.
- **nWhere** — positional encoding derived from the span's `start` byte
  offset in the source buffer.

A whole sentence is sent at once as a batch of [nWhat + nWhere] vectors
to PerceptualSpace.  ObjectEncoding packages these via its existing
`forward(objects, what, where, when, pad)` method.

#### 3. Numeric path is unaffected

For numeric datasets (simple, ergodic, xor, MNIST), no source buffer,
span table, or Lex tokenization exists.  The numeric path is unchanged:

- `Data.toDevice()` pre-shapes `[N, D]` → `[N, D, 1]` as today.
- ObjectEncoding does NOT encode nWhat, nWhere, or nWhen when these are
  not provided — the objectSize contribution from unused dimensions is
  zero, and the tensor passes through without modification.
- `setSymbolDim()` must be set to 0 for models that use a passthrough
  symbolic space (no discrete symbol layer), ensuring symbolDim does not
  inflate the embedding size for models that don't use it.
- All numeric-only configs (simple.xml, ergodic-only.xml, xor.xml) are
  completely unaffected.

#### 4. Lex.py — the tokenizer module

New file: `basicmodel/bin/Lex.py`

Lex replaces `LanguageModel.tokenize()`, `tokenizeList()`, and the inline
`sentence.split(" ")` calls.  It produces span tables over the source
buffer, not copied strings.  InputSpace delegates span table creation
to Lex.

```python
class Lex:
    def __init__(self, granularity="word"):
        """granularity: "word" | "sentence" — selectable per XML config."""

    def build_vocab(self, source: Tensor) -> None:
        """Scan source buffer, build vocab, assign token IDs."""

    def encode(self, source: Tensor, example_offsets: Tensor) -> Tensor:
        """Return span table [num_spans, 3] over source buffer.
        Each span is (start_byte, end_byte, token_id)."""

    def decode(self, source: Tensor, spans: Tensor) -> List[str]:
        """Reconstruct surface strings by slicing source at span offsets."""
```

Granularity is controlled by XML config:
`<tokenization>word</tokenization>` or `<tokenization>sentence</tokenization>`.

#### 5. InputSpace batching from span tables

Instead of re-tokenizing per batch, InputSpace slices the pre-computed
span table:

```python
def get_example_spans(self, idx):
    """Return the spans belonging to example idx."""
    mask = (self.spans[:, 0] >= self.example_offsets[idx, 0]) & \
           (self.spans[:, 1] <= self.example_offsets[idx, 1])
    return self.spans[mask]
```

For batching, `prepInput()` gathers spans per example, pads to the longest
sequence, encodes each span as [nWhat + nWhere] via VectorSet +
ObjectEncoding, and produces a dense `[B, seq, embeddingSize]` tensor.

For numeric data (simple/ergodic), `prepInput()` is unchanged — the
tensor is already `[B, D, 1]` from `toDevice()` and needs no span logic.

#### 6. InputSpace.reverse() — round-trip reconstruction

InputSpace.reverse() must reconstruct the source buffer from the derived
latent state.  The round-trip (forward through all spaces, then reverse
back) already works for numeric data in BasicModel.

For text, reverse() decodes each vector's nWhat back to a token ID via
reverse VectorSet lookup, and uses nWhere to place the token at its
original byte offset in a reconstructed buffer.

**This round-trip must be perfected before moving on to OutputSpace
token reconstruction.**  The ability to go forward → latent → reverse →
original input is the foundation: if reverse() can faithfully recover the
input, the same mechanism can be applied in OutputSpace to generate new
text.

#### 7. OutputSpace — destination buffer from symbolic vectors

OutputSpace receives a set of vectors from SymbolicSpace, each carrying
nWhat + nWhere, and constructs a unified destination buffer:

- **nWhat** is decoded via reverse VectorSet lookup to recover a token ID,
  then decoded to bytes via Lex.decode().
- **nWhere**, if present, specifies a byte offset in the destination
  buffer — allowing positioned writes.
- If **nWhere** is absent, tokens are written consecutively and in-order,
  reducing to **successor-token (next-token prediction) semantics**.

```python
class OutputSpace:
    def forward(self, symbols):
        """Convert symbolic vectors to output buffer.

        1. Extract nWhat and nWhere from each objectEncoded vector.
        2. Reverse VectorSet lookup: nWhat → token ID.
        3. If nWhere present, write token at that offset.
           Otherwise, write consecutively.
        4. Decode token IDs to bytes via Lex.decode().
        """
```

For numeric data, OutputSpace continues to treat the output as a plain
`[B, nOutput]` tensor with no buffer logic.

#### 8. parse.py — future integration (not yet used)

parse.py produces constituent span tables (NP, VP, PP, etc.) over the
same source buffer.  It is NOT used in the current design because no
convention yet exists for representing syntactic constituent information
in ObjectEncoding at both InputSpace and OutputSpace.

If a convention is established — for example, using a new objectEncoding
dimension (nSyntax or extending nSymbols) to carry the constituent type —
then:

1. InputSpace could encode both word-level spans (from Lex) and
   constituent spans (from parse.py) into objectEncoded vectors, where
   nWhat carries token content and the syntactic dimension carries the
   constituent role.
2. OutputSpace could use a generative grammar (inverse of parse.py's
   analytical grammar) to expand constituent-tagged symbols into
   structured token sequences in the destination buffer.

This would require parse.py to be available on both the input and output
sides with a shared grammar.  Document this possibility as comments in
InputSpace code for future reference.

#### 9. Migration path

1. ~~**Add Lex.py**~~ Done. Word-level tokenization producing `[L, 3]`
   span tables over the source buffer.
2. ~~**Add source buffer to Data.load()**~~ Done. `train_source` /
   `test_source` (uint8 tensors) for text datasets. Numeric unchanged.
3. ~~**Ensure numeric path is unaffected**~~ Done. `symbolDim=0` for
   passthrough symbolic spaces; objectEncoding contributes zero when
   dimensions are unused.
4. ~~**Add span-to-vector encoding in InputSpace**~~ Done. Lex creates
   span tables; each span encoded as [nWhat + nWhere] with byte-offset
   positional encoding.
5. ~~**Perfect InputSpace.reverse() round-trip**~~ Done. Cosine-similarity
   snap to nearest codebook entry recovers words; nWhere recovers positions.
6. ~~**Wire OutputSpace**~~ Done. `reconstruct_text()` and
   `reconstruct_buffer()` decode symbolic vectors via reverse codebook
   lookup + nWhere positioning; consecutive writes when nWhere absent.
7. ~~**Conditionalize tokenizer via XML config**~~ Done.
   `<tokenizer>traditional</tokenizer>` (word2vec/ReversibleDictionary)
   or `<tokenizer>grammatical</tokenizer>` (Lex span tables). Both paths
   coexist for comparison.
8. ~~**Document parse.py integration**~~ Done. Comments in InputSpace
   docstring describe future syntactic encoding convention.
9. **Add sentence-level granularity** to Lex, controlled by XML config.
   (Not yet implemented.)

#### 10. What stays the same

- `Data.toDevice()` pre-shaping for numeric data.
- `VectorSet` / codebook lookup for embedding (InputSpace uses VectorSet
  to map token IDs → nWhat vectors; reverse lookup recovers token IDs).
- `ObjectEncoding` continues to handle nWhere and nWhen as it does today,
  and contributes zero objectSize when dimensions are unused.
- The dense `[batch, seq, embeddingSize]` tensor that enters the model
  stream is the same shape regardless of whether it came from span-based
  text or direct numeric input.
- The reverse pass through all spaces, which already works for numeric
  data, is the foundation for text round-trip reconstruction.
