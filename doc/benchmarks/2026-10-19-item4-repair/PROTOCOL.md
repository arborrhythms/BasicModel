# Item 4 repair round — 2026-10-19

Baseline: BasicModel 9207f5e03, WikiOracle 929e4a8, plus the reviewed candidate working tree. Work is on codex/item4-repair in both repositories. This directory owns the repair-round diagnostics and gate tables. The previous failed learning run is immutable and is not regraded.

Repair order is A–K from Alec's review instructions. All canonical P/S/R/D gains are zero. The seed-23 graded diagnostic uses the original six-batch native workload and optimizer, with per-batch departure scale, mask, cost difference, chooser loss, gradient and parameter-change instrumentation. It is a cause investigation, not extra training appended to the failed run. Execution remains unpassed until the cause and repair are written here.

After repairs, measure zero-gain consumer parity, native fully expected content/answer/store at positive and unit gain, omission, percept identity, and the fixed graded/shuffled comparison. No seed or extra updates may be chosen to make a gate pass. Mechanism, execution, and learning findings remain separate. Carried Item 0 findings and the independent Item 5.5 learning gates remain open.

The final native validation from the previous candidate was incomplete: 134 selected, 125 completed, 120 passed, one skipped, four failed, nine unfinished. Its timeout is not a pass. The four failing cases and the three requested math-chain reruns are listed separately in this round's tables; a missing prerequisite is not a completed rerun.

Landing order: candidate commit, Claude review, one full sweep, final commit/push, WikiOracle pointer bump. No commit on main.

Execution note: the predeclared measurements are complete. See README.md for results and the remaining false gates; this protocol is not an acceptance claim.
