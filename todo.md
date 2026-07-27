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
