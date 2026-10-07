# Order-0 meaning by sentence membership: the fixed point before round 4a (Claude, 2026-10-07)

`sim.py` (numpy). A stored sentence row gets a fixed sparse identity code in
the meaning complement (`s` ones of `K`, seeded from the row's content key —
no RNG, no parameter). A word's order-0 meaning is the mean of the codes of
the sentence rows containing it (random indexing; Kanerva 2000, Sahlgren
2005), which is the existing context mean with the rows contributing their
identity instead of their (empty) composed meaning. Composition on the
meaning block: conjunction = min, disjunction = max (zero is false — a
membership field), sum = mean.

| check | result |
| --- | --- |
| XOR corpus (4 sentences, K=64, s=3): meanings distinct | yes |
| XOR corpus: semantic certificate — extent of `a∧b` = sentences with both, `a∨b` = with either, by code membership | 0 errors of 24 checks each |
| XOR meaning roots by min / max: affine fit, centered singular values | ~1e-32, [.866 .866 .866] for both — equally conditioned |
| XOR meaning roots by mean (sum control) | MSE .25, rank 2 — the floor holds |
| synthetic 400 sentences, code membership errors (∧ / ∨ of 80,000) | K=64: 276 / 7,828; K=256: 17 / 1,237; K=1024: 1 / 31 |
| Spearman(meaning cosine, normalized shared sentences), K=1024 | .45 |

The code is a sketch of the extent (a Bloom filter): exact membership at
the gates' scale, approximate at corpus scale, so membership is answered
through the index (postings) and the code carries similarity and
composition. Both operators' meaning roots read equally well, so the XOR
class gate cannot choose between them on meanings either; the operator's
semantics is certified statically by the membership certificate.

## The centroid with adjacent-word wholes (`sim4b_adjacent_ceiling.py`)

On the repository's own documents (21,290 sentences, 8,153 words), with 3a's
forms: a word's wholes are the spans containing it, the narrowest being its
adjacent-word pairs; a whole's form is the join of its constituents' forms;
the ceiling is the meet over the word's adjacent pairs,
`U(w) = L(w) ∨ ⋀ L(neighbour)`; the centroid `c = L + α(U − L)` with
`α = W_U / (W_P + W_U)` (adjacent occurrences against pair atoms).

| check | result |
| --- | --- |
| words whose centroid differs from `L` | 3,001 (36.8%), ~4.5 coordinates raised on average |
| identity: `L` recoverable as `[c = 1]` | all words |
| mean pairwise cosine, forms → centroids | .468 → .475 (no collapse) |
| words with a single repeated neighbour (52): cosine gain toward it | +.118 |
| containment violations caused by the centroid → after the projection | 218 → 0, never below `L` |

Unlike the letter-defined has-a wholes (§28), adjacent-word wholes are
sparse, so the ceiling is informative and the evidence-weighted centroid
does not collapse. Words seen once are pulled hardest toward their one
neighbour; the meet tightens as contexts accumulate.
