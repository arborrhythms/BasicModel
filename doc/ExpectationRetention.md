# Expectation retention

`TernaryTruthStore` is the single durable owner of an external sentence,
its preceding estimate, and their occurrence relationship. The structured
sentence predictor remains a boundary computation in `InterSentenceLayer`; it
does not get a second LTM or a writable prediction cache.

For a warm, eligible external boundary, the host performs this order:

1. The predictor reads only the row-local, preceding external-observation
   view and forms a `MeaningExpectation`.
2. The input is understood into its actual `ConceptualMeaning`; its roles are
   the detached prediction target.
3. When the first prerequisite observation has a durable occurrence, the
   store appends an `estimate` row, then an `observation` (or interrogative
   `question`) row, and links their immutable occurrence identities.
4. Only the observation is appended to the next predictor context. An
   estimate is retrievable as a prediction but never becomes another external
   sequence item, a fact, or corroboration of itself.

Provisioned or request-ingested source text intentionally uses the same parse
and ordinary LTM-write machinery, but runs with external observations
suspended. It therefore creates neither an estimate nor an occurrence binding:
those rows are source material, not a continuation of the caller's prediction
stream. Generic LTM recurrence and global-attention/reasoning reads likewise
exclude `estimate` rows. Until a typed forecast capability is deliberately
added, `expectation_pair()` is the only durable reader that exposes one.

The estimate row contains its predicted full-width role vectors, predicted
role mask, and three detached presence logits. Its sidecar provenance contains
the ordered source occurrences, the external stream/document key, and the
intended observation occurrence. The observation sidecar contains the inverse
estimate occurrence. `expectation_pair()` reconstructs the role and presence
residual from those retained rows; it deliberately does not maintain a third,
mutable delta record.

The truth semantic sidecar is version 2. Version 1 remains readable for old
rows without expectation relationships. Version 2 fingerprints each link with
the row's existing semantic context and source text, validates both directions
on restore, and participates in origin-compaction reachability. A tensor-only
restore therefore leaves a linked row semantically unavailable until its
sidecar is restored, rather than silently dropping the relationship.

Durable capacity favors the actual external observation: when one slot remains,
the store records the observation and reports no retained estimate. This is an
honest incomplete provenance result, not a replacement of input data by a
forecast.

## Gradient and current boundary

The live preceding context can still receive the normal inter-sentence MSE/BCE
gradient in its optimizer step. The arriving meaning is detached for that loss.
Every stored estimate role, confidence logit, observation, source occurrence,
and reconstructed residual is detached hard evidence; no lookup or later
residual read opens a gradient path.

This establishes occurrence ownership and checkpoint fidelity, not residual
policy learning. The current predictor does not synthesize bindings or scope
for a novel estimate. It retains its predicted roles/mask/confidence and the
actual source/stream/document/target provenance; the understood observation
retains its own complete bindings and scope. Metadata prediction, residual
guided query credit, its separate baseline, parameter-version-safe trajectory,
and learned-utility evidence remain open under integrated-plan item 2.
