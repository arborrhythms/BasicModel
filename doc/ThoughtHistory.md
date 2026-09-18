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

The ordinary controller now owns the selected direct-relation boundary path.
It is deliberately separate from the legacy `LTMSlot` parity loop: a completed
interrogative compose program is adapted into canonical `[NP1, VP, NP2]`, then
`run_selected_thought()` opens one ordinary episode for that row. Composition
itself remains pure. The adapter retains the signed live leaves, native
references, mode and `not`/`non` polarity from the owned action program; a
physical nested fold is not flattened into an invented operand.

The controller's hard `query` / `finish` choice sees three separately masked
role schemas — root request, active context and candidate — plus level,
pressure and bounded actual-evidence flags. For width `D`, that controller
context is `9D + 15`; the two action-kind features make the MLP input `9D +
17`. Native IDs, row numbers, addresses and surface tokens remain metadata,
not numerical features. The chooser is width-owned, lazy/checkpointed and uses
the existing `whatThinkingHidden` / `whatThinkingDepth` capacity settings.

`selectedThoughtPolicyWeight` is a separate, default-off REINFORCE objective:
later answer loss less actual controller-choice cost, with its own EMA
baseline. Its log-probability path can reach the chooser and its live role
payloads; executor results, references, meter state and reward are hard or
detached. This is supplied-answer controller credit, **not** residual credit,
and it does not establish learned utility.

## Evidence

The reviewer probes first failed against the pre-installation state in
`output/tests/20260917-101448-ea1b90`. The focused contract set then passed
9/9 in `20260917-101736-c6a3eb`, its expanded lifecycle/checkpoint set passed
33/33 in `20260917-102018-1d12d3`, and the affected integration selection
passed 69/69 in `20260917-110047-33f23a`. These prove the storage, replay,
credit and occurrence-read foundation only; they do not prove normal thought
selection, learned utility, or end-to-end answer quality.

## Sentence-runtime integration

`resolveAnswer()` opens only its completed rows, then runs an interrogative
owned program through the ordinary controller before legacy thinking or
`reverseOutput()`. Assertions and unsupported/nested physical programs remain
observations; they cannot execute a checked VP. The boundary guard is checked
before registry, native or occurrence reads. A standalone/evaluation
resolution ends its finished ordinary episode immediately; training retains it
through its one optimizer step and closes it with the existing episode teardown.
See [Query phases](QueryPhases.md).

## Selected-query meter

Each controller episode creates one `QueryWorkBudget` from
`selectedThoughtBudget` (default 32). It charges a controller unit before each
chosen query, same-level conclusion, descent, return and root finish, and
passes that exact meter to registry preparation, execution and `what(Q)`
callbacks. Each ordinary transition records the actual delta; nested work does
not start another allowance or optimizer episode. At cutoff only the existing
query-free drain is legal. Live thought-occurrence reads preserve their live
meaning; durable reads retain their detached boundary. See [shared query
work](QueryWork.md).

The controller probes first failed on the absent context/lifecycle path, then
passed 25/25 focused semantics/controller cases in
`output/tests/20260918-033710-5f144b`; the non-overlapping normal/chooser
regression passed 39/39 in `output/tests/20260918-032344-fee2c1`. The two
polarity cases passed in `output/tests/20260918-034810-cd9c1c`. These are
mechanism and lifecycle evidence, not a learned-utility or residual-policy
result.

## LTM roots retained by ordinary history

The existing owner derives LTM roots from every retained ordinary record's role
references, bindings, scope and recorded sources. It reads that data on demand,
without a second reference index. Those roots protect required content during
request-origin replacement even after the content's evidential authority is
withdrawn. See [nested retention](NestedRetention.md).
