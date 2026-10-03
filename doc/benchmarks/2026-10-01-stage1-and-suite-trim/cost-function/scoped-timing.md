# Automatic reconstruction: first-batch timing

One unseeded run per arm per configuration, numerical execution with graph capture disabled. Before restores the XML false setting; after uses automatic reconstruction. The source copy and numerical parameters match. This measures a first batch, not a full epoch or compilation. All worker ceilings remain 8 GiB.

| Configuration | Before training s | After training s | Difference s | Outcome |
|---|---:|---:|---:|---|
| `data/BasicModel_answers_benchmark.xml` | — | — | — | memory / memory |
| `data/BasicModel_expectation_benchmark.xml` | — | — | — | exit / exit |
| `data/MM_20M_fineweb.xml` | — | — | — | memory / memory |
| `data/MM_20M_grammar.xml` | — | — | — | memory / memory |
| `data/MM_ladder.xml` | 5.264 | 5.109 | -0.155 | complete |
| `data/MM_ladder_idiom.xml` | 8.679 | 8.049 | -0.630 | complete |
| `data/MM_ladder_text.xml` | — | — | — | memory / memory |
| `data/MM_ladder_textpacked.xml` | — | — | — | memory / memory |
| `data/MM_nanochat_grammar_gate.xml` | 9.533 | 9.823 | 0.290 | complete |
| `data/MM_nanochat_grammar_pilot.xml` | — | — | — | memory / memory |
| `data/matrix/MM_20M_grammar_reading.xml` | — | — | — | memory / memory |
| `data/matrix/MM_20M_grammar_wordstore.xml` | — | — | — | memory / memory |

Memory stops and the expectation benchmark’s complete-sentence capacity error happened in both arms. They give no successful timing difference; no successful value was substituted and no data was clipped. Process-tree peaks, elapsed times and complete errors are preserved in the JSON and per-arm logs. The native production stage-1 receipt separately uses its expressly authorized 24 GiB ceiling; this measurement did not extend that exception to these configurations.
