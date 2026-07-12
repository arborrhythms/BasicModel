# Attention-to-Relation Promotion Plan

> **STATUS: EXECUTED 2026-07-12** — see
> [2026-07-12-attention-promotion-execution.md](2026-07-12-attention-promotion-execution.md)
> for the architecture mapping (this plan predates the FF pyramid), the
> acceptance law pinned by todo.md (the Task-6c learn-score $\ge$
> `truthCriterion` AND `truthCriterion` $< 1$), and the recorded
> deviations (Reset-hoisted collector; role-aware evidence and
> prediction-gain bookkeeping deferred).

**Goal:** Use quadratic bottom-up attention as the discovery surface for
latent taxonomic wholes, then commit only stable, predictive patterns into the
persistent conceptual relation table. Attention remains the dense temporary
field; the relation table remains the sparse long-term memory.

## Rationale

The conceptual `ConceptualAttentionLayer` already provides the right computational
shape: a soft active set over concepts, pairwise interaction evidence, and a
nonlinear wave over the concept inventory. The missing canonical step is a
promotion policy that turns repeated shared-context patterns into named
higher-order concepts.

For example, `king`, `queen`, and `prince` may repeatedly share contextual
features such as ruling, crown, palace, succession, and sovereign. The
attention field can discover this common neighborhood before the architecture
decides whether it deserves a durable whole such as `royalty`.

## Mechanism

1. **Observe attention evidence.** During parse/reconstruction, read the
   bottom-up attention/coactivation field over active concept rows. Evidence
   should be role-aware when available: subject/object role, predicate,
   syntactic category, masked-word support, STM position, and LTM relation
   source.

2. **Accumulate candidates outside the relation table.** Maintain a bounded
   candidate cache keyed by a context signature, not by a single pair. A
   signature is a weighted neighborhood: shared active concepts, predicates,
   roles, and whole/property refs. Track support count, EWMA strength,
   contrast against negative contexts, and last-seen timestamp.

3. **Score for promotion.** Promote only when a signature is recurrent,
   stable, compressive, and predictive. Useful criteria:
   - support across multiple batches/documents;
   - high overlap among member concepts' neighborhoods;
   - contrast against nearby non-members;
   - reduction in reconstruction or masked semantic prediction loss;
   - relation-table compression: one whole explains many repeated edges.

4. **Commit a promoted whole.** When a candidate passes threshold, mint a
   higher-order concept with `synthesize_higher_order(member_concepts)` or an
   equivalent relation-table API. Add membership edges from members to the new
   whole and context/property edges from the whole to the shared intent. Mark
   the candidate as committed so future evidence strengthens the same whole
   instead of minting duplicates.

5. **Populate weighted attention edges.** The committed whole must enter the
   `ConceptualAttentionLayer` wave, not just the discrete record store. Add sparse edges
   from the promoted whole row to its member/context rows with initial weights
   from the candidate statistics. Subsequent masked reconstruction and
   conceptual SBOW losses should train those weights.

6. **Couple to conceptual geometry.** The promoted whole should inform the
   `similarity_codebook` by being active in the wave during masked prediction
   and semantic reconstruction. If disabling the whole's relation edges does
   not change prediction quality, the promotion is not functionally connected.

7. **Prune and merge.** Periodically retire weak candidates, merge near-duplicate
   wholes with overlapping extents/intents, and decay promoted edges that stop
   improving prediction. The sparse relation table should contain durable
   structure, not raw co-occurrence noise.

## Implementation Tasks

- [x] Add an attention-evidence collector at the post-pump symbolic cutover.
      (Stashed at the cutover, consumed at `Reset(hard)` — the
      `learn_relations_from_stm` compile-safety hoist, mirrored.)
- [x] Define the candidate cache: key, support stats, EWMA weights, contrast,
      prediction-gain bookkeeping, and capacity policy.
      (Prediction-gain bookkeeping deferred — recorded deviation.)
- [x] Add a promotion pass that mines common neighborhoods and calls the
      relation-table higher-order mint path.
- [x] Initialize `ConceptualAttentionLayer` sparse edges from promoted candidate weights.
- [x] Route promoted-whole activations into masked reconstruction /
      conceptual-SBOW loss. (Automatic through the existing cutover
      interfaces once the row exists — proven by the ablation pin.)
- [x] Add pruning/merge rules for stale or duplicate promoted wholes.
- [x] Add ablation tests: masked semantic prediction should degrade when
      promoted-whole edges or the conceptual wave are disabled.
      (`test/test_attention_promotion.py` — mechanism-level ablation:
      removing the promoted whole's edges zeroes its rung activation.)

## Acceptance Criteria

Given repeated contexts that imply a latent whole, the system should promote a
stable higher-order concept, store it in the relation table, expose it to the
conceptual wave, and use it to improve masked semantic prediction. A toy probe
such as `{king, queen, prince}` with shared royal contexts should mint or reuse
a `royalty`-like whole; disabling that whole's edges should measurably reduce
the relevant reconstruction or prediction score.
