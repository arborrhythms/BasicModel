# Plan §13 review repairs

All six requested repairs and the listed minor corrections are implemented. Nothing is committed. The complete old/new source and all changed or removed test bodies are in `changes.json`, against `before.zip`; `orphan-use-audit.json` and `stale-comment-repairs.json` retain the use checks and comment edits. These are review repairs, not a new baseline measurement.

The packed two-row probe saved 16 stray applied operations before the row gate repair. The repair also preserves inactive predictions and symbol columns, and the regression compares all 28 language and six STM fields on inactive rows. Raw UTF-8 bytes now survive both word admission and the mixing reconstruction bank; word identities remain unpublished. Both grammar files retain their original absence assertions. Device dispatch uses explicit device marks, with CPU the default even for slow-only cases. The ineffectual admitted-word terminal test is recorded as retired. The prototype IdeaSubSpace and orphan spelling/invalidation APIs and their tests are removed.

`final-focused/` passed 38/38 cases, in 22.8 worker seconds, peak 3.34 GiB, on matching source. Earlier red receipts and the two intermediate failures are preserved; the non-ASCII integration fixture was shortened to one complete admitted word so truncation and unrelated admission cannot obscure the byte contract. The transitional grammar intentionally retains its POS tokens: its original guard forbids marker helpers and copy/swap, while complete.grammar additionally forbids the retired production tokens.

The compiled K=2 test again uses inductor; its expensive actual execution belongs to the final full sweep. The shared ladder fixture uses monkeypatch. Coverage-map corrections distinguish retired marker replay from related live behavior, and inactive trace gaps share frames 0–2 but never write them. Mixing reconstruction staging is demand-driven. Parked inline suites join the weekly run, and demo make targets pass --demo.

## Journal-width diagnostic

`journal_recompile_probe.py` captured a real fullgraph word loop with a stable counting backend, fixed batch/word dimensions, and two longest-sentence lengths. Changing the local record width from 12 to 15 caused one additional stage_cs_lang capture (25.68 seconds). Reusing both widths caused no further captures (0.070 and 0.097 seconds). The first forward captured all three bricks in 26.59 seconds. Peak process-tree memory was 1,229,555,512 bytes. This measures graph capture, not inductor lowering. `journal-recompiles.json` and `.log` retain the guard evidence. Two exploratory diagnostic attempts remain separately labeled: one failed to enable capture, and one changed backend identity and therefore induced unrelated recompiles. The final diagnostic establishes a shape warm-up cost, not continual recompilation; no record-width or guard change was made.

## First complete weekly run

Started on the immutable plain-file source copy `tmp/slow-tests/20261002T012227Z-32db47/source`, not a separate Git working tree. This preserves source matching while part 4 proceeds in the one working tree. The selected inventory is 302 pytest cases plus 26 inline unit tests. All 26 inline tests passed before the bounded ordinary tier started. Ordinary workers retain 8 GiB; the production native stage-1 workers use the previously decided 24 GiB ceiling, with a 24 GiB aggregate limit and one worker. Completion and failures are recorded by the runner in that directory; this section does not claim the run is finished.


The first ordinary segment stopped at worker 032's unchanged 8 GiB ceiling.
The native tier then continued. `weekly-first-stops/` preserves that stop and
the frozen-copy subprocess interpreter failure. The runner now supplies a link
to the unchanged runtime and continues only unattempted cases after a bounded
stop. The receipt-local `resume_first_weekly.py` finishes the first selection
on the same source copy after the native tier; it never retries a failed case.
The record reports attempted coverage separately from completed pytest reports.
Three focused runner tests verify continuation without retries, stopping on
collection failure, and source-copy/runtime-link separation.

The weekly native tier completed both arms on that frozen snapshot: step 5a
1,156.6 seconds / 21.63 GiB; trial answer cut 1,254.2 seconds / 20.78 GiB.
Both passed. `weekly-native-complete.json` retains their process results; the
ordinary continuation is independent and its resource failures remain red.

At the review-progress capture (2026-10-02 05:25:59 UTC), 156 of 328 selected
cases had finished an attempt: 151 completed reports and five process stops.
Those reports contain 92 passes, 51 failures, five skips and three expected
failures. One case was active and 171 had not started. The source copy still
matched its frozen hashes. `weekly-progress-at-review.json` records that
partial state and every failure; it is not a completed weekly receipt. The
background runner continues and will write its final `record.json` and
`tmp/slow-tests/latest.json` when the remaining cases have been attempted.
