# Lexicon

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The lexicon is the language-facing entry point shared with LLM practice:
surface forms become vectors learned from corpus statistics. BasicModel then
splits that vector work across two additional structures. Formal Concept
Analysis appears when word/object rows are bound into the part/whole concept
order; DisCoCat appears when those lexical vectors participate in typed grammar
reductions rather than only in distributional similarity.

PartSpace owns the orthographic lexicon and its embedding APIs. ConceptualSpace
owns word-concepts, object-concepts and their native reference index.
WholeSpace is only the learned property basis: it owns no word dictionary,
META dictionary or grammar decoder. Grammar composition and generation share
the operations declared in SymbolSpace.

The Lexicon's distance math is described in
[Spaces — Lexicon (Projective Unit Ball)](Spaces.md#lexicon-projective-unit-ball).
PartSpace's VQ snap is disabled; its orthographic embedding remains trainable.
After an optimizer step, its normalization projects the rows onto the unit
ball.

> **BPE as the option-flipped lexicon mode.** `<synthesis>bpe</synthesis>`
> on PartSpace makes the chunker produce byte-aligned BPE units
> instead of whitespace-split words, reusing the same Embedding for
> storage. The byte round-trip is fully invertible via the chunker's
> `id_to_bytes` table; see
> [test_chunk_layer_bpe.py::test_hard_merge_spans_bpe_roundtrip](../test/test_chunk_layer_bpe.py).

## Word forms and concept orders

A form or fused unit resolves to its word concept through the index derived
from the definition rows (`DefinitionIndex`): LTM1 rows whose two references
are an object and a word, read as definitions from their referents' types. The same index resolves word → objects and object → words;
no lookup scans or compares learned vectors. It rebuilds from the common
store on load and after compaction. There is no fixed order per spelling
and no permanent attended row identity.

Order counts symbolizations: order 0 is an event in the attentive field;
order 1 is a particular, named (*Felix*) or unnamed (*the cat*), whose
identity across events is object permanence; order 2 is the kind (*cat*).
Higher kinds can occupy further orders. Formation and testimony supply
these concepts; lexical resolution never invents a missing one. Grounding is
textual: a new word's object starts nearly empty and learns its parts and
kinds from what the text states
([Language](Language.md#interpret-word-concept-to-object-concept-2026-09-25)).

The operator that performs this resolution is `interpret`
([Language](Language.md#interpret-word-concept-to-object-concept-2026-09-25)),
the per-word step, forced in reading mode under every binding, from the
word-concept PartSpace looked up to its associated object-concept. An
existing association wins at its existing order, including a kind. If
several objects are associated with the word, grammar can resolve the
ambiguity. If none exists, the word is admitted pending, and a kept defining
clause or the recurrence of its mentions later gives it its object. A new
word costs two identities, one inventory row and one DEF row. The word retains its parts and wholes; the object's row carries the
learned content. Neither is a part of the other merely because they are
linked. A missing particular is not a reason to add one beside a known kind.

`InterpretLayer` reserves capacity before the transaction and is the only
writer of definitions (`Interpret.define`). A definition names its word and its
object as symbols, never by their codes, so what training does to either code
never redirects it. Reading identical canonical parts is a lookup,
even when their property evidence has changed. Where the field discovers
objects by recurrence, interpretation uses the case the field admits.
The definition's `.when` stays fixed and its recency timestamp refreshes.
If forgetting deletes the DEF row, both lookup directions lose that pair;
reading the word again interprets it afresh. No spelling heuristic or
surface-to-operator anchor decides reference order.

A compose rule names its operands positionally (`argin1`, `argin2`,
`argout1`) and declares only structure: its head operand (`head="argin2"`),
the operand that carries reference (`reference="argin2"`) and, for a
determiner, its referent (`referent="mint"`, `"bind"` or `"kind"`). No
operand is named as a kind of word (pronoun, particular, kind), and there is
no name-to-order table or maximum order: the requested order is what the
composition denotes (`ReferenceContext.binding_spec`, `denoted_order`;
[Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21)).
Which words fill which position is learned by participation, so proper-name
and generic readings resolve through the selected grammar, without
capitalization rules or word lists. Inner reference phrases retain their
choice when an outer rule requests an order. A unique association at that
order is selected; a carried referent can disambiguate it. A missing
particular never causes a downcast from a known kind. A newly requested
singleton keeps its source point during the trial and is admitted by the
kept closing; the read never allocates (`ReferenceContext.resolve_order`).

Sub-word structure is admitted one pattern at a time, only inside a named
lesson. `WordIdentity.nested_lesson(marker)` permits one suffix pattern while
the lesson runs, and `observe_nested` carves a witnessed word that ends with
that marker into its stem and the marker once the stem itself has been
witnessed ("walked" into "walk" + "ed" once "walk" has been seen; a stem
witnessed later licenses an earlier marked word). With no marker named, every
word stays whole, and an irregular form ("went") never nests. There is no
vocabulary-wide search for shared letter runs, no prefix or infix admission
and no token-count threshold: one generalization is admitted at a time.
The permission is not a learned identity or a checkpoint setting, but parts
already witnessed persist after the lesson ends. The word remains the unit of
attention. A marker, free or nested inside a word, is one kind of leaf: which
selection it calls, and with what object, is learned by exposure and never
declared, and no language is named or special-cased in code, configuration,
fixtures or tests (Alec, 2026-10-19).

A candidate's declared noun references are resolved before its numerical
operation runs. Particular and pronoun choices are limited to earlier live
constituents and the bounded situation of the symbol predictor
(`SymbolExpectation`, whose order-zero read, `WordExpectation`, predicts
activation over the candidate words;
[Expectation](ExpectationRetention.md)); pronouns never select kinds. The hard choice retains the softmax gradient. A completed row stores
its end-state values and references, not the operations or lexical leaves.
The temporary reading record keeps the original reconstruction provenance
and the selected operand values until the sentence ends, then is discarded.
Reading a row into words uses the shared generation policy. Retrieval terms
come from unfolding the row, limited by the symbolic activation already used
for semantic priming; no sentence word list or activation snapshot is stored.

## Quick reference

Perceptual lexicons reserve physical `nVectors` at construction. The active
prefix grows in place and search reads only that prefix; admission preserves
the Parameter, existing vectors and optimizer moments. A full allocation
raises an error naming `nVectors`. It does not silently omit a new word or
resize the codebook. Checkpoints preserve capacity and occupancy separately.

The Lexicon ([`bin/Layers.py`](../bin/Layers.py)) is a learnable
vocabulary embedding on the **projective unit ball** $B^D / (x \sim
-x) \cong \mathbb{RP}^D$ --- the closed unit ball with the
**negation identification** $w \sim -w$ (the $\pm$-quotient). Note
that $-w$ is the *negation* of $w$, not its **antipode**: the antipode
of a point is the furthest point in the manifold (used by SBOW as a
repulsion target), and on $\mathbb{RP}^D$ it is the orthogonal
hyperplane, not a unique point. Distance and lookup:

$$
d_{\mathbb{RP}}^2(a, b) \;=\; \min(\|a-b\|_2^2,\ \|a+b\|_2^2)
\;=\; \|a\|_2^2 + \|b\|_2^2 - 2\,|\langle a, b\rangle|,
$$

$$
\operatorname{score}(x, w_i) \;=\; |\langle x, w_i\rangle| - \tfrac{1}{2}\|w_i\|_2^2.
$$

Top-k lookup is one matmul, one elementwise abs, one broadcast subtract.

```python
lexicon = Lexicon(V, D)              # default: projective unit ball
lexicon.project_unit_ball_()         # call after optimizer.step()
W_index, W_norm2 = lexicon.lookup_index()
idx, dist_sq, scores = Lexicon.topk_rp(x, W_index, W_norm2, k=32)
```

For the full derivation, SBOW pode/antipode gradient consequences,
chunked-lookup helpers, and the legacy torus primitives kept for
backward compatibility, see
[Spaces.md --- Lexicon (Projective Unit Ball)](Spaces.md#lexicon-projective-unit-ball).

## See also

- [`bin/Layers.py`](../bin/Layers.py) --- `Lexicon` class and `topk_rp` /
  `topk_rp_chunked` helpers.
- [`bin/embed.py`](../bin/embed.py) --- SBOW training loop.
- [Spaces.md](Spaces.md) --- full per-space geometry discussion, including
  the contrast between PartSpace's projective Lexicon and
  ConceptualSpace's unit-direction codebook.
