# Closing reads: causes, repairs and rejected attempts

These are mechanism repairs on the uncommitted step-1b branch. They do
not change R + A, the forced decomposition scope, certificate budgets or
the original initializations. Each run keeps its own outcome in
[runs.md](runs.md).

## 1. Can a definition be mistaken for an equality answer?

| Observation | Result |
|---|---|
| Concrete retained edge | In `closing-55.store.jsonl`, object 33 points to word 12 through Part 34 |
| Collision | Word 12 is a singleton; the old Part reading classified the same edge as both definition and equality |
| Consequence | An identity query could return the defining word as the object's equal |
| Repair | A definition reading excludes the equality reading; a genuine singleton object on the right still supplies equality |
| Before repair | New native regression fails in native-56 |
| After repair | Passes in native-57 and native-70 |

Native-50's c/e question ended bound at a random initialization that was
not saved. The definition/equality collision is independently reproduced;
it is **not claimed as an exact replay or proved sole cause of native-50**.
That historical failure and the limit on its diagnosis remain visible.

## 2. Why did the selected declarative closing fail to start its episode?

| Observation | Result |
|---|---|
| Saved native failure | Native-61: only one of the two required c/e episodes starts |
| Exact replay | Native-63 restores `closing-61.initial-rng.pt` and fails the same assertion |
| Read-only trace | The raw open copular occurrence at row 2 uses row 2 itself as its forward-reference witness, returning its own complement, symbol 25 |
| Consequence | The closing reader treats the selected open formation as already bound and suppresses its required episode |
| Rejected repair | Excluding a row's own witness from every working read |
| Counterexample to that repair | Native-67 passes c/e and forward premise order but fails reverse premise order, where an ordinary copular read legitimately uses its complement |
| Final repair | `ThoughtClosing.closing_question` reads the raw occurrence and calls the common reference reader with only the currently closing occurrence excluded |
| Ordinary working reads | Retain the public witnessed copular interpretation; raw occurrences are not rewritten |
| Current evidence | Whole §14 completes in native-68 and native-70: 12 passed, one artifact prerequisite skip, with c/e initialization 61 and both saved premise initializations 32 |

The [exact replay trace](closing-reference-63.jsonl) records the self
witness before repair. The initial broad run with the rejected global
guard, native-65, was interrupted after native-67 exposed its reverse-order
regression: **180/185 completed, not acceptance**. It is not a timeout or
a successful full run.

The final unit regression checks both boundaries:
`test_closing_excludes_its_own_witness_without_restricting_copular_reads`.
The selected open close remains open; ordinary copular interpretation and
an earlier matching question can read the witness. The original stored
occurrence remains open. Cyclic identities are not prohibited.

## 3. Why did ten old thinking fixtures fail after §14 passed?

| Failure group in native-68 | Port or production repair |
|---|---|
| Four later-binding variants call retired `fill_forward_references` | Append a witnessed copular occurrence and read the earlier question without upserting it |
| Modus ponens and three event-time fixtures use retired `REL_IMPLIES` | Supply two native situation references and their conditional Part |
| Two-query chain assumes physical inference row 2 | Assert the returned native Part evidence, arguments and proof witnesses |
| Chooser-credit contrast assumes forward conceptual part lookup has no shared-graph proof | Use the reverse part query as the wrong alternative; retain the chain, credit direction and reconstruction/answer comparison assertions |
| Conditional event-time lookup omitted by the Part conversion | Restore reads of the two endpoint situations' `.when`, with bounded read work and explicit unknown/reversed rejection |

The conditional repair is production behavior, not just a fixture update.
A rule's insertion time cannot stand for the endpoint times. Composed
native situations supply those times; each endpoint read consumes the
existing record budget. Unknown endpoints remain incomplete, and reversed
endpoints do not produce a conclusion. Atomic situation symbols retain
their atemporal conditional reading.

Native-70 then completes **206 passed, four failed, one skipped**. All
four remaining failures are the later-binding variants: the question is
in stream 0, but the new copular witness had been appended with no stream.
The shared reader correctly refused that witness. Giving the fixture's
later occurrence `stream=0` preserves the intended same-stream setup.
[Native-72](native-72/report.html) completes the whole thinking file:
**26 passed**, including the original no-open-slot, correct-reference,
witness, source-unchanged, chain-credit and temporal assertions.

## 4. What remains a failed gate?

Both unchanged decomposition lengths still fail in both gain arms.
The [eight question tables](decomposition-71.md) and
[symbol-type question](../../plans/2026-10-19-bind-thought-operation.md#69-implementing-68-a-missing-symbol-type-source-codex-2026-10-20)
record the separate expression-reader blocker. The passing §14 file and
thinking ports do not replace those proofs or the pending full sweep.
