# Item 6.9 — one training, both bars (§22 amendment)

Consolidation complete; stopped for review, nothing committed. **No retest
required after consolidation**, per Alec's follow-up. The additional campaign
already started was interrupted immediately; no measurement processes remain.
The [previous receipt](../2026-10-03-item6-9-free-readback/README.md) remains the
measurement baseline. This receipt applies Alec's October 3 amendment to the
[§§21–22 candidate](../2026-10-03-item6-9-free-readback/README.md). The production
model, configurations, operators, objectives and bars are unchanged. The prior
receipt's unresolved regressions and incomplete clause-journal scope remain
recorded there; this amendment does not claim their repair.

`TestXorGrammarLearnsXor` and `TestXorGrammarReconstruction` now consume one
module-scoped trained model and retain every assertion. A failing first bar
does not suppress the second. The bounded and weekly runners keep the selected
fixture consumers in one process, including at batch size one and across a
requested recycling boundary. Memory and time guards stay unchanged.

The pre-change probe in `probes/sharing-before/` executes the real gate bodies
with a counted stand-in and observes two training calls. `sharing-before.json`
records that failure. After the change there is one call; deliberately failing
the class bar, reconstruction bar, or both preserves the independent failures.
The observer hook now surrounds fixture setup, so the single real training also
supplies the audit. These fixture probes train no models and use no RNG.
The two gate bodies are AST-identical to their originals after substituting
only the model assignment ([assertion check](assertion-integrity.json)). All
35 bounded-runner and recycling checks passed. Complete old/new files and
test bodies are in [ports.json](ports.json), with the five-file change list
in [port-summary.json](port-summary.json).

The updated runner schedules ten XOR_grammar trainings, both bars and joint success in each
row; first training reused for both table rows; ten sum controls; the remaining
named table once, including one MM_20M exact round trip; ten MM_grammar runs;
the production native stage-1 test and audit; moved and MNIST cases; one full
source-matched sweep on ten workers. Attribution runs ten trainings per arm,
both bars per model, only if class is below 8/10 or reconstruction is at most
3/10. CPU, the 8 GiB ordinary ceiling, the previously decided 24 GiB native
slow-test ceiling, epoch budgets and unseeded initialization are unchanged.

The cancelled campaign is **not a completed ten-run receipt**. Before Alec's
follow-up, the native test passed (778.70 seconds, 21.79 GiB, zero ownership
conflicts) and nine shared XOR trainings finished. Their first outcomes are
retained: class 8/9, reconstruction 3/9, both 2/9. Run 10 and the first two
sum controls were interrupted. The remaining named table, MM_grammar,
attribution, moved cases and full sweep were not started. These partial counts
do not replace the baseline or trigger attribution. See the
[cancellation record](measurement-cancellation.json) and
[retained partial outcomes](partial-outcomes.json). Both already completed
audits, including geometry, displacement and stability, are in
[audits.md](audits.md); no additional training was used to produce that report.

Before-source hashes and complete files are in `before.json` and `before.zip`.
The candidate is preserved in `source-final.json` and
`source-final.zip`; all measurement outcomes are retained without retries.
The [integrity record](receipt-integrity.json) confirms unchanged source during
the partial measurements, unchanged environment and HEAD, and complete port
pairs. No HEAD measurements or commits.
