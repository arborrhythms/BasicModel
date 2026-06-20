# The training curriculum — five staged objectives

*Spec authored 2026-06-19. The staged objectives a WikiOracle-class model trains
through, from substrate-building reconstruction to deliberate question
answering, and how each maps onto the architecture's own machinery.*

> **Status: design / roadmap.** This document records the *order* in which the
> objectives should be trained and *why that order is forced*, not a finished
> training loop. Stages 1–3 are largely reconstruction/prediction (the substrate
> builders); stages 4–5 are question answering (the deliberate layer). Cross-refs:
> [orders.md](orders.md) (the three pumps), [reading-attention.md](reading-attention.md)
> ((A) reading, (B) global attention, (C) idea decode), and the grammar-inverse
> handoff `doc/plans/2026-06-19-grammar-inverses-handoff.md` (which inverses
> stages 1–2 require).

## Why a curriculum at all — the two learning rules

The architecture carries **two distinct learning mechanisms**, and the
curriculum is just those two mechanisms in their natural developmental order:

- **EMA / occurrence** moves the codebooks — concepts form by being *seen*, with
  no gradient and no goal. This is *prediction as substrate*: "France",
  "capital", "Paris" become codebook rows because they recur. Unconscious,
  statistical — **System 1**.
- **Gradient on a task error** moves the attention readouts and the relation
  store — the model learns *where to look* and *what relates to what* by being
  *tested*. Directed, credit-assigned — **System 2**.

So "bootstrap with prediction, then do question answering" is not a workaround;
it is the EMA substrate-builder running first and the gradient deliberate-layer
running on top. It is also exactly what the §6c sentence protocol already says:
*prime subsymbolically from what you see, then pump symbolically.* You cannot ask
"why did the empire fall?" until *empire* and *fall* are codebook rows — and
those are EMA-built by exposure. The stages below climb that ladder.

The arc: **stages 1–2 reconstruct** (build the encode↔decode autoencoder and the
idea↔words bridge), **stage 3 predicts** (the discourse substrate), **stages 4–5
answer** (reason, then search). Each stage's substrate is the next stage's
prerequisite.

---

## Stage 1 — reconstruct a sentence, keeping the syntax, predicting the missing words

**The task.** The syntactic structure (the parse tree) is **retained**; only the
lexical slots are masked. The model fills the missing words from the structure +
the surrounding context. It is the gentlest reconstruction — the scaffold is
given, only the leaves are blank.

- **What it trains.** The **lexical inverse** only: the word↔slot mapping at the
  leaves. The codebook (EMA-built concepts) is queried for the row that fits the
  structural slot; the nearest-word codebook reverse renders it. The *structural*
  operators are NOT inverted here — the tree is supplied.
- **Machinery.** `Embedding.decode_reverse_meta` (nearest-word codebook reverse);
  the leaf `Lift`/`Lower` at the terminal tier; the existing reconstruction (IR)
  loss restricted to the masked leaf slots. PartSpace `radix` reverse for spelled
  words.
- **Loss.** Reconstruction of the masked words (cross-entropy / nearest-row over
  the codebook at each masked leaf).
- **Prerequisite.** A populated codebook — i.e. *Stage 0*, raw prediction/exposure
  (the EMA fill). Without concepts there is nothing to retrieve.
- **Order knobs.** `subsymbolicOrder` (the mereological pump that builds the
  percept→word); `symbolicOrder`/`syntacticOrder` inert (no structure to compose
  — it is given).
- **System / rule.** System-1-leaning; primarily exercises the EMA codebook
  through a thin gradient on the leaf readout.
- **Status.** The reconstruction loss + codebook decode **exist**; the
  syntax-retained masking setup + the precise leaf-only inverse are the build.
  The grammar-inverse handoff enumerates exactly which inverses Stage 1 needs
  (the lexical/leaf set, NOT the structural set).

## Stage 2 — reconstruct a sentence without keeping the syntax or the words

**The task.** Both the parse tree **and** the words are deleted; regenerate the
whole sentence from **just the idea** (the C-tier STM content). This is
deliverable **(C)** — the parse-tree-deleted reverse.

- **What it trains.** The **full generative inverse**: every grammar operation
  (`Lift`/`Lower`/`Sigma`/`Pi`, `LanguageLayer`, `SyntacticLayer`) run in
  reverse, **driven by the primed symbolic space rather than a stored
  `generate_rules`**. It also pressures **abstraction** — the single idea must
  carry enough structure to reproduce the sentence, so order-raising (the
  mereological climb) has to have packed it well.
- **Machinery.** The whole `reverse()`/`generate()` path
  ([reading-attention.md §4](reading-attention.md)) **decoupled** from the chart:
  a `parseFree` reverse, a **symbol-driven** relative mask (infer relative-ness
  from the STM symbol repertoire, not the grammar rules), and a STM that pre-assigns
  symbol IDs so "primed symbolic space" is well-defined without a chart.
- **Loss.** Reconstruct the original words from the idea alone — the existing
  reconstruction loss through the parse-tree bottleneck, with the bottleneck's
  parse tree *deleted* (an autoencoder over that deletion).
- **Prerequisite.** Stage 1 (the lexical inverse is a sub-step of the full
  inverse) + the abstraction/order-raising machinery.
- **Order knobs.** All three: `subsymbolicOrder` (refine the regenerated parts),
  `symbolicOrder` (the serial σ/π that the generative split unwinds),
  `syntacticOrder` (the parse-tree depth the generation may compose back to).
- **System / rule.** Straddles both — the *decode* is gradient-trained, but it
  reads off an idea whose content is EMA-built.
- **Status.** The reverse/generate path **exists but is parse-tree-DEPENDENT**
  (driven by `generate_rules` read off the chart). Stage 2 needs it **decoupled**
  — see the grammar-inverse handoff for the per-operation list of inverses to
  *build* vs *decouple*.

## Stage 3 — predict the next sentence

**The task.** Given the discourse so far, predict the **next idea / sentence**.
Causal / autoregressive at the idea level.

- **What it trains.** The **inter-sentence predictor** (the discourse AR ring) —
  the idea→idea transition, the gist that lets you anticipate what comes next.
- **Machinery.** The discourse AR / `InterSentenceLayer`; the `interSentence`
  CS_{-1} seed (A6) that primes the next forward; `_chart_generate_from_stm` to
  render the predicted idea.
- **Loss.** Next-idea prediction (the existing inter-sentence prediction error).
- **Prerequisite.** Stages 1–2 (you predict *ideas*, which the reconstruction
  stages taught you to form and render).
- **System / rule.** Bridges System 1 → System 2: local in form, but predicting
  the next sentence well requires the *gist*, which is already a deliberate
  abstraction.
- **Status.** The inter-sentence prediction machinery **exists**
  (`<prediction>interSentence`); it slots in once ideas are stable.

> **Note on stages 1–3 vs the literature.** Plain next-token/next-sentence
> prediction under-trains *this* architecture's retrieval and relation machinery,
> because the local window usually suffices and the separate, discrete global
> attention gets no reward (see [reading-attention.md §3](reading-attention.md)).
> That is *why* prediction is the **substrate** here, not the goal — it builds the
> codebook and local structure, then the QA stages train the deliberate layer.

## Stage 4 — answer a question based on reasoning

**The task.** Question answering where the answer is **derived** from relations
and entailment over what is already in STM/LTM — **no external search**. The
question primes; the model reasons over its stored relations to *deduce* the
answer. ("Answer from what you can derive.")

- **What it trains.** The **relational pump** (`symbolicOrder`) and the **truth
  stores** — the relation corpus (`RelativeTruthStore` / `TernaryTruthStore`,
  `partOf`/`implies`/trust), modus ponens (`consequents()`), and the catuṣkoṭi
  (true / false / both=contradiction / neither=unknown) routing.
- **Machinery.** The question becomes an intent (`set_intent`); the relational
  pump composes relations-of-relations; `consequents()` derives entailed facts;
  the tetralemma/`trust` adjudicates. The answer is produced by the same idea →
  words decode as Stage 2.
- **Loss.** Answer correctness (+ a **consistency** term: a derived/asserted fact
  that contradicts a stored *trusted* one is penalised — the truth machinery's
  native signal, which prediction never touches).
- **Prerequisite.** Stored relations (accumulated while *reading* during stages
  1–3) and the relational machinery; Stage 2's decode to render the answer.
- **Order knobs.** `symbolicOrder ≥ 1` (the relational pump *is* this stage);
  `syntacticOrder` bounds the reasoning depth.
- **System / rule.** System 2, gradient-trained — this is the deliberate layer.
- **Status.** The relation stores + `consequents()` **exist**; the QA framing
  (question → reason → answer) and the loss wiring are the build.

## Stage 5 — answer a question based on LTM search

**The task.** Open-domain question answering: the answer is **not** derivable
from what is currently in STM, so **global attention must search** the LTM /
codebook / corpus, retrieve the relevant fact, and then answer. ("Answer from
what you can find.")

- **What it trains.** **Global attention** (B) — the typed `.where` over {input
  window, STM, LTM, symbol codebook}, the **soft-read fed back into the answer**
  (the consumer that closes B's loop), and the **stochastic element** that lets
  the search *explore* (a search problem with only a distal reward cannot break
  symmetry without exploration). Retrieval that yields a correct answer is
  rewarded by backprop through the read (REALM-style).
- **Machinery.** `GlobalAttention` + `_addressable_spaces` (built, dark); the
  **consumer** (feed `_global_attention_obs["content"]` into the answer
  prediction — currently *parked, not fed back*); the two-pass
  `exploreTemperature` (built) as the explorer; for a *book*-scale corpus, the
  out-of-core hierarchical `.where` paging (deferred).
- **Loss.** Answer correctness, backpropped **through the retrieval** so the
  scorer learns *where to look*.
- **Prerequisite.** Stage 4's QA head + B's addressing + the consumer; true
  open-domain over all of Wikipedia additionally needs the deferred book paging.
- **Order knobs.** `symbolicOrder` (reason over what's retrieved) + the global
  `.where` (retrieve); the explorer rides the two-pass.
- **System / rule.** System 2, gradient-trained — and the place the stochastic
  exploration finally *earns its keep*.
- **Status.** B's addressing + explorer are **built/dark**; the **consumer**
  (feed the read back + the retrieval-trained answer loss) and **book paging**
  are the deferred next slices ([reading-attention.md §7](reading-attention.md)
  item 4).

---

## The ladder at a glance

| stage | objective | trains | dominant rule | status |
|---|---|---|---|---|
| 1 | reconstruct, keep syntax, fill words | lexical/leaf inverse + codebook | EMA (System 1) | loss+decode exist; masking + leaf inverse to build |
| 2 | reconstruct, no syntax/words | full generative inverse (C) + abstraction | mixed | reverse path exists but parse-tree-dependent → decouple |
| 3 | predict next sentence | inter-sentence predictor | EMA→gradient | exists (`interSentence`) |
| 4 | answer by reasoning | relations, modus ponens, tetralemma | gradient (System 2) | stores+`consequents` exist; QA framing to build |
| 5 | answer by LTM search | global attention + consumer + explorer | gradient (System 2) | B built/dark; consumer + paging deferred |

**Reconstruction (1–2) is the gate.** Stages 4–5 render their answers with the
same idea→words decode that Stage 2 builds, and a question can only be *answered*
once the model can turn a retrieved/derived idea back into words. So the two
reconstruction stages — and the grammar-operation inverses they require — are the
load-bearing prerequisite for the whole curriculum. Those inverses are enumerated
in `doc/plans/2026-06-19-grammar-inverses-handoff.md`.
