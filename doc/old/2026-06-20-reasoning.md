# Reasoning — the deliberate (System-2) layer and the query operators it drives

*Spec authored 2026-06-20. Captures the reasoning work, and records WHY the
"genuine stub" grammar reverses (`isEqual`/`isPart`/`part`/`assertPart`/the
`query` family/`bind`) are deferred: they are **introspection/query operators that
reasoning consumes**, and reasoning is not yet wired end-to-end. Cross-refs:
`training-stages.md` (stages 4–5), `reading-attention.md` §3 (global attention +
consumer), `2026-06-20-idea-decoder.md` (renders the answer), `data/full.grammar`
(the `<queries>` ops).*

> **Status: design / roadmap.** Much of the substrate EXISTS (truth store, modus
> ponens, tetralemma/trust, LTM retrieval, the query-op truth computations); the
> missing pieces are the **QA framing, the answer losses, and the
> global-attention consumer**. Default off → byte-identical.

## Why this is a separate layer (and why the stubs wait for it)

The architecture has two learning rules (`training-stages.md`): **EMA/occurrence**
(System 1 — concepts form by exposure) and **gradient on a task error** (System 2
— the deliberate layer learns *where to look* and *what relates to what*).
Reasoning is the System-2 layer. It runs on top of the reconstruction substrate:
you cannot answer "why did the empire fall?" until *empire* and *fall* are
codebook rows and *fall(empire)* is a stored relation.

The grammar's **query/relation operators are reasoning's verbs, not the decoder's.**
Their *forward* computes a truth value reasoning operates on; their *reverse* only
matters when reasoning must **generate/explain** a relation — which is a later,
reasoning-driven step. So their bare-`(parent,parent)` reverses are **correctly
deferred**: nothing in the reconstruction decoder needs them. (`data/full.grammar`
already files `exist`/`equal`/`part`/`query`/`queryPart`/`queryEqual` under
`<queries>` as **NOP-for-parse introspection**.)

## What already exists (the reasoning substrate)

| piece | where | role |
|---|---|---|
| `ltm_store` (unified truth store) | `Layers.py` (`append_relation`/`append_idea`, rows `(NP1, VP, NP2)` + per-row **trust**) | the persistent relation/idea corpus |
| `learn_relations_from_stm` / LTM consolidation | `Models.py:7607+` (gated `<ltmConsolidation>`) | writes relations from STM at the sentence boundary |
| `provision_ltm()` | `Models.py:2378` | seeds `ltm_store` from a `<truthSet>` (trusted axioms) |
| `consequents(state, vp, threshold)` | `Layers.py:7394` | **modus ponens** — entailed facts from a state |
| `ConceptualSpace.reason(query, …)` | `Spaces.py:14193` | forward modus-ponens inference over the store (stage 4) |
| `BasicModel.reason(givens, target, direction, max_steps=8)` | `Models.py:9156` | the reasoning entry (forward/backward, depth-capped) |
| `verify_relation(...)` | `Spaces.py:14275` | check a relation against episodes |
| `stm_end_state_trust` + catuṣkoṭi | `Spaces.py:14100`, `Layers.py:5537`, `Models.py:5199` | **tetralemma** (true / false / both=contradiction / neither=unknown) + the trust scalar |
| query-op truth computations | `Language.py` (`_parthood_geometric`, `_truth_bivector_like`, `QueryEqual`=mutual parthood) | the **forward** of the query ops = the truth value reasoning reads |
| global attention over `{input, STM, LTM, codebook}` + explorer | `Spaces.GlobalAttention`, `exploreTemperature` | LTM/codebook **retrieval** (stage 5); the explorer breaks symmetry |

## What is deferred / to build

**The genuine-stub reverses** — `isEqual`/`isPart`/`part`/`assertPart`/`query`/
`queryPart`/`queryEqual`/`bind` — are deferred to this layer. When reasoning needs
to *generate* a relation (explain an answer, assert a derived fact), route their
reverse through the working recommender (`isEqual` is a `max` → `disjunctionReverse`;
`isPart` = whole + a part-search), driven by the truth-store/codebook as the basis.
Not needed for reconstruction.

**Stage 4 — answer by reasoning** (`training-stages.md`): the QA framing
(question → `set_intent` → `reason()`/`consequents()` over the store → answer),
plus the **answer loss**: correctness **+ a consistency term** (a derived/asserted
fact that contradicts a *trusted* stored one is penalised — the catuṣkoṭi's native
signal). The relational pump (`symbolicOrder ≥ 1`) composes relations-of-relations;
`syntacticOrder` bounds reasoning depth. The answer is rendered by the **idea
decoder** (`2026-06-20-idea-decoder.md`).

**Stage 5 — answer by LTM search** (`reading-attention.md` §3, §7-item-4): the
**global-attention consumer** — feed the parked soft-read (`_global_attention_obs`)
back into the head and **train the retrieval by the downstream answer error**
(REALM-style), currently dark (`<globalAttentionConsume>`). The **explorer**
(two-pass `exploreTemperature`) earns its keep here. Book-scale corpora need the
deferred out-of-core `.where` paging.

## Truth framing

- **Two homes for truth** (see the truth/ideas memory): an *intuitive* truth is a
  reducible WS-META scalar; an *explicit* truth is an idea-relation in the store.
- **Trust** is a per-row scalar in `[−1, 1]` (the catuṣkoṭi `t − f`); authority
  (a user `<truthSet>`) seeds it via `provision_ltm`.
- **Catuṣkoṭi routing**: true / false / both (contradiction → flag) / neither
  (unknown → may trigger a stage-5 search).

## Build order (reasoning)

1. **QA framing** — question → intent → `reason()`/`consequents()` → answer
   (wire the existing `reason`/`consequents`/`verify_relation` into a question
   loop); render the answer via the decoder.
2. **Answer loss** — correctness + the consistency penalty against trusted rows.
3. **Global-attention consumer** (stage 5) — feed the LTM soft-read back, train
   retrieval by the answer error; let the explorer break symmetry.
4. **Query-op reverses** — only when reasoning must *generate* a relation (route
   through the recommender + the truth store as basis).
5. **`exist` as a live LTM query** — make `ExistLayer` query `ltm_store` + trust
   (today it is an identity wrapper; `data/full.grammar` already defines `exist`
   as "found in LTM and trusted").

All gated; off → byte-identical. Prerequisite: the reconstruction stages (1–2)
and a populated `ltm_store`.
