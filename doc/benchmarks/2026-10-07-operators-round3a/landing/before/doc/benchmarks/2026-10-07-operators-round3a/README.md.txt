# Operators round 3a — identity by construction

Status: the delivered-source sweep is green (5,289/5,289 cases completed); the thirty measured trainings are in progress. No commit or push.

Start: the accepted frozen round-2e source, manifest `3fd07f0cd516da3102867207614e11a9153638d37057bcc82a8c7ff821689106`. Git remains at the round-1 commit because the accepted round-2 source was never committed. The frozen archive is preserved in `before/`.

The [construction review](construction-review.md) is preserved verbatim. This candidate applies Claude's plan §26 amendment: 64 pair coordinates, three initial bits per pair/mint, a separate 32-coordinate thermometer, checked positional mints, and separate native/witness containment reports. The [measurement protocol](measurement-protocol.json) declares exactly thirty unseeded gate trainings, no retry or replacement, and review before any commit.

Frozen candidate: `84edc57e0ef1c95478a0e17a78a44b23fbafb8bc441031591c5f70bb27d580dc`, 707 source files and 188 hashed measurement helpers. Complete old/new text is saved for 30 test ports; no test seed calls changed.

The frozen form audit reports zero collisions after minting, zero containment violations and zero indexed byte errors on each configuration. It separates native vocabulary from witnesses. The full local BasicModel corpus census covers 67,391 words and requires 69,566 atom/word rows in the separate audit bank; the production bank remains at its existing 32,768-row capacity. This is a storage limit, not a full-corpus training result.

The repeated seed-zero MM bisection confirms the historical RNG-only path. Construction, first output and after-forward parameter/RNG digests match round 2e exactly.
