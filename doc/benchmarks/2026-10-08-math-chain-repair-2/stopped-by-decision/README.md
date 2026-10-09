# MM_math_chain stopped by decision — §14.10–§14.11

Stopped by Alec’s decision at **two of thirty trainings**: answer plus expectation completed **8 epochs**, expectation-only completed **9**. The partial next epochs are retained too. The remaining 28 never started. No attempt is retried or replaced. **No learning claim or held-out evaluation.**

The reason is the §14.10 reading: no correct binding in the first start; query/conclude drift as the store fills while the menu cannot bind a found candidate; the easiest corpus problem already needs coordinated steps; weak answer-cost separation for near-misses. The learning measurement is deferred to item 0 after runtime optimization. The three future corrections are recorded beside the untouched `protocol.json` in `protocol-corrections-deferred.json`; they are not implemented here.

All existing partial outcomes, logs and archives stay byte-for-byte as found. `retained-files-sha256.json` records the stop boundary. The campaign controller was already absent at that boundary; the two live workers were stopped explicitly. The old controller progress file is preserved, including its stale timestamp.

| Condition | Completed epochs | Observed questions, including partial next epoch | Correct bindings | First-epoch greedy what openings |
|---|---:|---:|---:|---:|
| answer_and_expectation | 8 | 977 | 0 | 0/117 |
| expectation_only | 9 | 1167 | 0 | 0/117 |

These are **unforced observations**, not a bar. `summary.json` completes the per-start question/opening reports from saved logs; `episodes-by-start-epoch-kind.json` reports counts, share, work and mean work by epoch and sentence kind, marking partial epochs. Original files are not rewritten. Final chooser movement is unavailable because no final parameter checkpoint had been saved.

6.2 now closes on separately labelled **forced decomposition demonstrations** through the real driver, frozen observer and live explore suffix. The standing thirty remain as already measured. Stop for Claude’s review before any commit.
