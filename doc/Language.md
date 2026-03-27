# Language

## Overview

The language system spans two spaces in the pipeline — **SymbolicSpace** and
**SyntacticSpace** — together with a grammar (`TheGrammar`) and a word encoding
(`WordEncoding`) that represent binary deep structure as derivation trees.

```
ConceptualSpace  →  SymbolicSpace  →  SyntacticSpace  →  OutputSpace
  [B, nConcepts]    [B, nSymbols]    [B, nSymbols] +      [B, nOutput]
  activation         activation       word list
                     + codebook       (derivation tree)
```

The key insight: **symbols are discrete, one-dimensional entities**. They carry
no content vectors ("what") of their own. Instead, the activation pattern over
a set of symbols *is* the representation, and the codebook provides a bridge
back to dense vectors when downstream spaces require them.

---

## Grammar

`TheGrammar` is a singleton instance of the `Grammar` class. It defines a
context-free grammar in **Chomsky Normal Form** (CNF) — every rule is either
binary (two non-terminal children) or terminal.

| Rule | Production | Arity | Group |
|------|-----------|-------|-------|
| 0 | S | 0 (leaf) | — |
| 1 | S → W | 0 (terminal) | conceptual |
| 2 | S → S AND S | 2 | conceptual |
| 3 | S → S OR S | 2 | conceptual |
| 4 | S → NOT S | 1 (unary) | conceptual |
| 5 | S → NON S | 1 (unary) | perceptual |
| 6 | S → S PART S | 2 | conceptual |
| 7 | S → S UNION S | 2 | perceptual |
| 8 | S → S INTERSECTION S | 2 | perceptual |
| 9 | S → S EQUALS S | 2 | perceptual |

Rules are partitioned into two groups:

- **`Grammar.conceptual()`** → `[1, 2, 3, 4, 6]` — operations that manipulate
  abstract concepts (word reference, conjunction, disjunction, negation, parthood).
- **`Grammar.perceptual()`** → `[5, 7, 8, 9]` — operations on perceptual fields
  (non-affirming negation, set union, set intersection, equality).

This partition mirrors the two-layer logic system described in [Logic.md](Logic.md):
conceptual rules operate on the symbolic/scalar layer, perceptual rules operate
on the subsymbolic/vector layer.

### CNF and Unary Rules

Rules 4 (NOT S) and 5 (NON S) are unary — they have one non-terminal child,
which breaks strict Chomsky Normal Form. Two design options remain open:

1. **Rewrite as binary** with an implicit identity/epsilon child:
   $S \to \mathrm{NOT}\ S\ \varepsilon$. This preserves strict CNF at the cost of introducing
   null nodes.
2. **Allow unary rules** as a first-class concept — unary operators transform
   a single child's representation without branching. The tree remains binary
   in the sense that all branching nodes have exactly two children; unary nodes
   are pass-through transformations.

---

## Word Encoding

Each word is a 3-tuple `(batch, vector, rule)`:

- **batch** — index into the batch dimension `[0, B)`
- **vector** — index into the activation vector `[0, N)`
- **rule** — grammar rule ID from `TheGrammar` `[0, 10)`

Words are stored as a Python list of tuples on the SubSpace (not muxed into the
event tensor). The list constitutes a **derivation tree** in pre-order: the
first entry is the root, and binary rules expand left-first.

The `WordEncoding` class validates tuple entries on construction. The `SubSpace`
provides accessors: `add_word(batch, vector, rule)`, `set_words(list)`,
`get_words()`.

---

## SymbolicSpace

### Role

Converts continuous concept activations into a discrete set of active symbols.
This is the information bottleneck of the pipeline.

### Forward Path

1. **Activation mapping.** Extract concept activation `[B, nConcepts]` from the
   input subspace. Map through an `InvertibleLinearLayer(nConcepts, nSymbols)` to
   produce `[B, nSymbols]` — a continuous activation in symbol space.

2. **Codebook quantization** (when `quantized=True`). The activation vector is
   reshaped to `[B, nSymbols, 1]` (each symbol is a 1-dimensional scalar) and
   passed through the codebook. The codebook quantizes, computes top-k by
   similarity, and produces:
   - A one-hot-ish activation over codebook entries (similarity-weighted)
   - Dense vectors for downstream materialization

   When `quantized=False`, the activation is stored directly on the subspace
   and the input vectors pass through unchanged.

3. **Output.** A SubSpace with symbol activation and (when quantized) codebook
   vectors. The output is a **one-hot encoding** over the codebook.

### Reverse Path

1. Extract symbol activation `[B, nSymbols]` from the input subspace.
2. Apply the **exact inverse** of the invertible layer to recover `[B, nConcepts]`.
3. Store the recovered concept activation on the output subspace, allowing
   ConceptualSpace to reconstruct its dense vectors.

### Key Properties

- Symbols are **zero-dimensional** — they are pure activation scalars, not vectors.
- The invertible layer allows **different-length** activation spaces ($nConcepts \neq nSymbols$).
- When quantized, the codebook size equals nSymbols — each symbol *is* a codebook entry.
- The `discretize()` method (sigmoid + straight-through round to {0,1}) is available
  but currently unused; the codebook's VQ serves as the discretizer.

---

## SyntacticLayer

### Architecture: Recursive Hybrid with Depth Embedding

`SyntacticLayer` is a weight-tied (recursive) derivation stack. A **single**
shared derivation layer and rule head are applied repeatedly at each depth,
with a learned depth embedding added to the hidden state so the shared weights
can specialize by tree level.

$$
\text{activation}\ [B, n_{\text{Symbols}}]
\xrightarrow{\text{input\_proj}}
h \in \mathbb{R}^{B \times d_{\text{hidden}}}
$$

For each depth $d = 0, 1, \ldots, D$, using **shared weights**:

$$
h \leftarrow h + \text{depth\_embed}[d], \quad
h \leftarrow \text{GELU}(\text{derivation\_layer}(h)), \quad
\text{logits}_d \leftarrow \text{rule\_head}(h)
$$

**Parameters:**

| Component | Shape | Count (hidden_dim=256, num_rules=10) |
|-----------|-------|-----|
| `input_proj` | nSymbols × hidden_dim | varies |
| `derivation_layer` | hidden_dim × hidden_dim | 65K |
| `rule_head` | hidden_dim × num_rules | 2.5K |
| `depth_embed` | max_depth × hidden_dim | 256 per depth level |
| **Total (excluding input_proj)** | | **~70K + 256 × max_depth** |

The learned weights (`derivation_layer`, `rule_head`) are constant regardless
of `max_depth`. Only the depth embedding table grows with depth (256 bytes per
level). This makes deep unrolling essentially free in parameter count.

### Why Recursive?

The same grammar rules apply at every level of syntactic embedding — "the cat
that the dog chased" uses NP expansion recursively. Weight sharing captures
this: the layer learns a *single* rule-prediction function conditioned on
derivation context (the hidden state `h`) and position (the depth embedding).

The depth embedding allows depth-specific behavior without depth-specific
weights. Depth 0 can learn "always predict S → NP VP" while deeper levels
learn context-dependent rule selection — all through the same weight matrix
receiving different input.

### Differentiable Rule Selection

Rule selection uses **Gumbel-softmax** with temperature `tau`:

- **Training:** `F.gumbel_softmax(logits, tau=tau, hard=False)` produces soft
  one-hot vectors. Gradients flow through the reparametrization trick.
- **Eval:** `F.softmax(logits, dim=-1)` gives a deterministic distribution;
  `argmax` selects discrete rule IDs.
- **Temperature annealing:** `set_tau(tau)` controls the sharpness. Start at
  `tau=1.0` (soft), anneal toward `tau=0.1` (near-discrete) over training.

---

## SyntacticSpace

### Role

Generates a binary derivation tree (deep structure) from the set of active
symbols produced by SymbolicSpace, using the learned `SyntacticLayer`.

### Forward Path

1. **Extract activation.** Read the activation vector from the input subspace.
   If no activation exists (all symbols active), create a full-ones vector.
2. **Predict rules.** Pass activation to `SyntacticLayer.forward()`, which
   unrolls the recursive derivation stack and returns rule distributions at
   each depth plus assembled word tuples.
3. **Store derivation.** Word tuples are set on the output subspace. Vectors
   and activation pass through unchanged.

### Reverse Path

1. **Decode derivation.** `SyntacticLayer.reverse()` walks the word list:
   every `(batch, vector, rule)` entry marks that position as active. This
   deterministically recovers the activation vector from the derivation alone.
2. **Store result.** Set the recovered activation on the output subspace
   (without recomputing from vector norms). Clear the word list.

### Key Properties

- The derivation is a pre-order list of word tuples.
- Reverse is **deterministic**: given the same word list, the same activation
  vector is always recovered.
- The round-trip `forward → (delete activation) → reverse` preserves the
  original active positions exactly.
- Default `max_depth = nSymbols - 1` (the maximum tree size for N leaves).
  Since the layer is recursive (shared weights), the only per-depth cost is
  one depth embedding vector — the parameter count is independent of depth.

---

## Open Question: Differentiating the Rules Themselves

The grammar rules are currently labels — AND, OR, PART, etc. do not define
distinct operations on their children's representations. The SyntacticLayer
learns *which* rule to predict at each depth, but not *what* each rule does.

For the rules to be **meaningful**, each must define a distinct computation
that transforms child representations into a parent representation:

- **Rule-specific layers.** Each rule maps to a learned transformation — e.g.,
  AND might use a SigmaLayer (additive), UNION a max operation, etc. The
  `Grammar.perceptual()` and `Grammar.conceptual()` groupings suggest a natural
  partition: perceptual rules use PiLayer-style multiplicative operations,
  conceptual rules use SigmaLayer-style additive operations.
- **Shared layer with rule embeddings.** A single composition function
  conditioned on a learned rule embedding vector. Fewer parameters but less
  expressive per-rule behavior.

This remains the primary open design question for the language system.
