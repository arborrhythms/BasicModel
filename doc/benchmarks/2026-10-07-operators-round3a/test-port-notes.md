# Round 3a test ports

The [complete old/new texts](delivered-source/test-ports.json) contain 29 test
files (including the new round-3a tests) and the observer probe. The
[seed audit](delivered-source/seed-port-audit.json) reports no changed test
seed calls. The class, reconstruction and MM gate files are byte-identical
to round 2e, as checked in [runtime-contract-check.json](runtime-contract-check.json).
No gate threshold, epoch budget or pass criterion was changed.

| Contract changed by the hand-off | Ports |
|---|---|
| Native sparse form is 96 coordinates plus an 8-coordinate event band; meaning complements retain their width | Dimensional governance, round-2 native-width assertion, review-13 subspaces, output-path width/checkpoint fixtures, output-walk conditioners, property migration, BasicModel word-loop configuration assertions |
| Words are identified on first sight by pair/length parts, with a word row distinct from its part rows | Word admission, positive-integer taxonomy, two-codebook taxonomy, tied-operator admission count, property parser's word-row lookup |
| Raw ordered byte witnesses remain available separately from the unordered form parts | Meronomy ladder and UTF-8 tests read `native_indices` / `native_part_spans`; word-form tests read the pair/length join |
| Binding and journal leaves use projected directions; indexed forms stay sparse | Grammar object leaves, the cleared-cache top-k dictionary, the observer's native geometry assertions |
| Rung-zero bytes are exact indexed reads with zero reconstruction gradient | Word-store reassociation, tied-operator byte gradients, cleared-cache surface shape/byte assertions |
| Signed ingestion trust is the row's pair; the independent scalar is retired | Identification evidence, runtime ingestion, query evidence, truth criterion/routing, truth-store serialization, LTM checkpoint assertions |
| Fixed construction, checked mints, reconstruction, projection and trust need direct coverage | New `test_operators_round3a.py`: native forms, witnesses, exact lengths 1–32, private RNG, stable row addresses, checkpoint restoration, optimizer invariance, ambiguous pair assembly, bit extension, quadruple fallback, pending/completed definition rekeying, numeric MM geometry, nonzero gate roots, rung-zero audit, trust and negation |

Two float32 comparisons now allow the observed projection/GEMM accumulation
rounding: `test_live_references.py` uses relative tolerance 1e-6 and absolute
tolerance 2e-7 for the projected word leaf; `test_one_attention.py` uses
2e-6 and 2e-7 for the padded matrix evaluation. The resolved live reference,
operation identity, and closing comparisons retain their original assertions.

The output-walk fixture that compared two permutations now compares different
word multisets. With the unchanged commutative kernels and fixed word forms,
a permutation is not required to change the root. The test still checks that
a held answer survives later staging and a different source changes the answer.

The small compiled-word fixture's part bank grows from 64 to 256 rows to hold
the new atom inventory. Explicit capacity-exhaustion tests retain their own
limits. The production BasicModel bank is unchanged.

The historical cleared-cache overlap threshold remains 0.8 and its existing
non-strict xfail marker remains. Its decode dictionary now uses the same
projected directions as the recovered leaves; it XPASSes in the frozen sweep.
The sweep completed 5,289 cases: 5,003 passed, 285 skipped by existing markers,
one XPASS, and no failures. Development failures and all repair checks remain
under `development/`; the delivered-source sweep is in `full-sweep/`.
