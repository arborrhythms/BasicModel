# Lexicon

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The lexicon is the language-facing entry point shared with LLM practice:
surface forms become vectors learned from corpus statistics. BasicModel then
splits that vector work across two additional structures. Formal Concept
Analysis appears when word/object rows are bound into the part/whole concept
order; DisCoCat appears when those lexical vectors participate in typed grammar
reductions rather than only in distributional similarity.

> **2026-05-29 deltas:**
>
> - **Two-codebook split (Stage 8, 2026-05-27).** PartSpace owns
>   the orthographic Lexicon Embedding (`PS.subspace.what`, learned
>   surface-keyed); WholeSpace owns a separate VQ-quantized
>   prototype Codebook (`SS.subspace.what`). The two are bound
>   per-row via the META cross-codebook taxonomy
>   (`(ps_row, ss_row) -> meta_row`), populated by
>   `ConceptualSpace._maybe_autobind_meta` at stage 0.
> - **LBG-style splitting on SS codebook (2026-05-29 Task C).** The
>   SS Codebook is updated under Gray (1990) EMA with per-row
>   running-variance tracking. When a row's variance exceeds a
>   threshold, it splits along the top-variance eigendirection; new
>   prototypes are seeded around the parent
>   $\pm \delta \cdot \text{variance\_axis}$. EMA continues independently on each
>   child. This replaces the previous random-direction VQ init,
>   which landed too many prototypes in degenerate basins on
>   small-codebook configs.
> - **Embedding unit-ball normalization in train loop.** PS's
>   Embedding is unit-ball-projected via `Lexicon.normalize()` after
>   each `optimizer.step()` so joint training doesn't drift the rows
>   off the ball.
> - **PS `<codebook>none</codebook>` mode.** `MM_xor.xml` and
>   `XOR_exact.xml` turn off the PS-side VQ snap so the butterfly
>   cascade weights actually train. The orthographic Lexicon
>   Embedding remains live; only the VQ snap on top is retired.
> - See [doc/old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

The Lexicon's narrative description and distance math have moved into
[Spaces.md --- Lexicon (Projective Unit Ball)](Spaces.md#lexicon-projective-unit-ball),
co-located with the Codebook Similarity Metric and the per-space
geometry discussion (PartSpace, ConceptualSpace, WholeSpace).

> **Lexicon ownership: API on WholeSpace, physical Embedding on
> PartSpace.** Post-2026-05-12 the orthographic-lexicon **API**
> (the `vocabulary` property, `train_embeddings`, `sbow_loss`,
> `reconstruct_data`, `reconstruct_to_buffer`, `get_recovered_word`,
> `_snapshot_embeddings`, `set_embedding_sigma`) lives on
> `WholeSpace`. The Embedding *tensor* still lives on
> `PartSpace.subspace.what` because `InputSpace`'s
> `_peer_perceptual.vocabulary` wiring at the lexer is too deeply
> integrated to relocate without a separate refactor. S accesses the
> Embedding via its `perceptualSpace_ref` back-reference. The full
> codebook-as-lexicon physical unification is a deferred follow-up.

> **BPE as the option-flipped lexicon mode.** `<synthesis>bpe</synthesis>`
> on PartSpace makes the chunker produce byte-aligned BPE units
> instead of whitespace-split words, reusing the same Embedding for
> storage. The byte round-trip is fully invertible via the chunker's
> `id_to_bytes` table; see
> [test_chunk_layer_bpe.py::test_hard_merge_spans_bpe_roundtrip](../test/test_chunk_layer_bpe.py).

## Quick reference

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
  the contrast between the projective Lexicon (PartSpace,
  WholeSpace) and ConceptualSpace's unit-direction codebook.
- [test/bench_codebook_lookup.py](../test/bench_codebook_lookup.py) ---
  performance comparison of the broadcast, matmul, pole-aligned, and
  chunked-wrap forms.
