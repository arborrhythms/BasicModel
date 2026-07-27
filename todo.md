# TODO

## Deferred Teacher-to-LTM persistence

- Keep Teacher v1 writes limited to detached, row-local NP-VP transition
  records. Do not commit reconstructions or predictions directly to the
  persistent/global truth store.
- Define an admission policy for student-generated memories with provenance,
  confidence, `asserted_at`, represented/target-event support, revision, and
  contradiction handling.
- Prevent Teacher-only clean targets and future information from entering
  student-visible LTM before evaluation.
- Add leakage, replay, correction, and document-boundary tests before enabling
  persistent writes.

## Objective-address conditioning

- Follow the staged teaching-mode and contextual-grammar implementation spec:
  `doc/specs/2026-07-27-teaching-modes-and-next-iteration.md`.
- Keep corpus/snapshot/document/sentence/span coordinates on the Teacher query
  seam; never reuse or overwrite the model's subjective `.where`/`.when`.
- Split target observation/event time from source snapshot validity; retain the
  latter as provenance rather than overloading objective `when`.
- After the clean Teacher throughput gate, add a student-side encoder that
  embeds corpus/snapshot/split IDs categorically and document/sentence/span
  positions as ordered coordinates. Raw hash magnitude must have no meaning.
- Prove that an addressed clean input round-trips through
  `Teacher.What(where, when)`, then measure whether partial and blank
  reconstruction actually use the address rather than leaked clean content.
- Treat DOI and source date as optional aliases/provenance. FineWeb has neither;
  retain shard SHA-256 and corpus release metadata instead of fabricating them.
