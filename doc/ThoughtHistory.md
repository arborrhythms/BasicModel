# Ordinary thought history

Implementation reference, September 17.

`SymbolSpace.what_memory` remains the only owner of interaction history. It
stores legacy `LTMSlot` values and ordinary `ThoughtRecord` values in the same
row-local chronological deque. Ordinary execution records explicit `begin`,
same-level `thought`, `descend`, `return`, `cutoff`, and `finish` transitions;
replay derives current and suspended contexts from those records rather than
maintaining a second planner stack.

Each root episode declares one shared work budget. All ordinary work charges
that budget across levels. At cutoff the active depth is frozen; the drain can
perform at most that many LIFO returns and one root finish. Retention reserves
space for the drain and rejects evicting active or semantically referenced
occurrences. Stable `thought` references are row-local and resolve complete
meanings through the same owner; they are not facts, answer labels, or evidence.

In episode mode, ordinary meanings stay live through the caller's one optimizer
step. The episode must finish before `end_what_episode` releases that credit.
Checkpoints contain detached copies and replay/validate every row atomically on
restore; new computations after restore can be live, but restoration never
replenishes budget or pressure.

The implementation adds no learned parameter, objective, or policy reward.
Normal controller integration, selected linguistic meaning, actual executor
costs, residual policy credit, and learned utility remain separate gates in
[the integrated production specification](plans/2026-09-15-next-sentence-as-the-production-objective.md).

## Evidence

The reviewer probes first failed against the pre-installation state in
`output/tests/20260917-101448-ea1b90`. The focused contract set then passed
9/9 in `20260917-101736-c6a3eb`, its expanded lifecycle/checkpoint set passed
33/33 in `20260917-102018-1d12d3`, and the affected integration selection
passed 69/69 in `20260917-110047-33f23a`. These prove the storage, replay,
credit and occurrence-read foundation only; they do not prove normal thought
selection, learned utility, or end-to-end answer quality.

## Sentence-runtime integration

The completed-row query guard now constrains when a selected query may execute;
it does not select an ordinary thought, change its level, replenish its work
budget, or alter history replay/credit lifetime. See [Query phases](QueryPhases.md).

## LTM roots retained by ordinary history

The existing owner derives LTM roots from every retained ordinary record's role
references, bindings, scope and recorded sources. It reads that data on demand,
without a second reference index. Those roots protect required content during
request-origin replacement even after the content's evidential authority is
withdrawn. See [nested retention](NestedRetention.md).
