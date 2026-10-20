# Read boundaries exposed by the Part conversion

**Current disposition:** Alec/Claude answered both questions in bind-plan
§6.8. The bootstrap is implemented and its exact/four-meaning gate,
saved reconstruction replay and random certificate pass; see
[the follow-up receipt](bootstrap.md). Implementing the expression rule
exposed a missing symbol-type source, recorded with native evidence in
§6.9. The earlier questions and failed measurements below remain the
historical record.

The latest [attempt 71](decomposition-71.md) completes all four original
decomposition certificates in both arms; all four fail, with zero
correct proofs among eight questions. [Current gate status](status.md)
records the passing saved/random bootstrap checks and the remaining
step-1b obligation. The pending wording below describes the earlier
review handoff, before §6.8 answered the two questions.

These are implementation findings on `codex/item3-5-first-pass`, not new
architecture decisions or learning results. The two questions below were
sent to Alec before changing either read contract. The §6.6 decisions,
original decomposition scopes, budgets, seeds and assertions remain in force.

## 1. How does a Part graph supply the expression's semantic operands?

**Review status, 2026-10-20:** Alec asked for this question to be written
down for review with Claude. No architectural choice was made. The
standalone explanation is in
[bind plan §6.7](../../plans/2026-10-19-bind-thought-operation.md#67-for-claudes-review-reading-an-expression-after-stm1-dissolves-codex-2026-10-20).

In plain terms, closing `y = x + 1` must leave enough structure for
thought to find `x` in the expression and ask what it denotes. The close
now retains every composing edge, including surface attachments. The
first argument of the outermost edge can therefore be an attachment,
although the unchanged decomposition scope needs the expression's
subject. **Which surviving data and read rule identify that subject
after STM1 dissolves?**

| Question | Observed result |
|---|---|
| Original one-successor initialization | Seed `4199722147`; original parameter snapshot |
| Original two-successor initialization | Seed `2428440506`; original parameter snapshot |
| Initial failure | The candidate bank contained the intended referent, but an implicit reference retained the WORD handle |
| Cause | The menu read the old form grouping instead of Part definitions, and deferred an implicit order-one use without leaving a closing request |
| Repair | Rebuild the word-to-object candidate pairs from the definition index; offer these known candidates at their actual concept orders |
| Small regression | Both order-zero and order-three candidate cases pass in `native-40`; the earlier explicit closing contracts also pass |
| One-successor result after repair | Both question rows target their intended retained referents; the recursive episode is reached |
| Remaining failure | The component reader exposes the outer surface edge's first argument as the expression's subject |
| Two-successor result | Also fails its unchanged proof assertion in `bind-decomposition-zero-stm1-13` |
| Enabled arm | Not run in attempt 13: the zero arm failed |

In the first document's actual kept graph, the question now names `39`.
The defining equality is `Part(39, 50)`. The expression under `50` includes:

| Result symbol | Part arguments | Producing operation, from the diagnostic journal |
|---:|---|---|
| 50 | 19, 49 | surface |
| 19 | 13, 14 | surface |
| 49 | 48, 47 | verb |
| 47 | 46, 45 | lift |
| 46 | 13, 17 | surface |

`17` is the retained subject needed by the original recursive scope. The
current reader returns `19`, the outer surface attachment, as its first
operand. The producing operations above are diagnostic data from the
temporary derivation; they are **not columns stored on Part**.

The path to the required subject is `50 → 49 → 47 → 46 → 17`. The current
reader instead exposes `(19, 49)` as symbol `50`'s semantic operands.
It has not shown how to recover the required view from existing symbol
types, inventory positions and Part edges. This does not establish that
the decided representation lacks the information.

The earlier question proposed retaining grammatical argument-role/type
information in minted intermediate symbols. That was a proposed repair,
not a requirement already established by the failure. Claude's review
should first specify how the existing representation is meant to supply
the semantic view; if additional information is necessary, specify its
minimal content and owner. No metadata has been added for this proposal,
and the forced scope has not been changed to select a convenient descendant.

Evidence: [menus and source readings](bind-decomposition-zero-stm1-13-reports/test_forced_native_decomposition_through_training_driver[1]/decomposition-1/progress.json),
[kept graph plans](bind-decomposition-zero-stm1-13-binding/closings.jsonl),
[both failed certificates](bind-decomposition-zero-stm1-13/report.html).

## 2. Which durable source supplies the word-context bootstrap?

| Question | Observed result |
|---|---|
| Native four-sentence bootstrap | Completes four LTM1 writes in `storage-41` |
| Part content | All four composing pairs remain: `(1,2)`, `(1,3)`, `(4,2)`, `(4,3)` |
| Existing LTM1 inverse postings | Only physical word row `0` is recovered, in two occurrences |
| Certificate | Exactness over that incomplete set is true; distinct meanings is `1`, required `4` |
| Other observed initialization | `native-38` retained completed relative roots in Part, leaving only one LTM1 idea |
| Proposed read | Use the explicit retained Part derivation for word membership; preserve exact-membership and four-distinct-meanings assertions |
| Decision | Pending; no new bootstrap source has been implemented |

The narrowed LTM1 point is not the discarded wide working object. A read
that requires it to invert to all original leaves would restore a gate
which §6.6 explicitly withdrew. Part retains the composing arguments,
but the current bootstrap reads only LTM1 postings.

Evidence: [storage diagnostics](storage-diagnostic-41.jsonl) and
[failed bootstrap](storage-41/report.html). The proposed change has not
been substituted for the failed result.

## Work held for these decisions

The decomposition component read and proof integration, the corresponding
observer port, and the word-context bootstrap remain open. Both decomposition
lengths must pass in both gain arms on the repaired source. A full landing
sweep, commit, push and both-main publication still follow those gates.
Step 2 has not started. Artifact regeneration, the LTM1 column cull, item 0
findings and the separate 5.5 learning gates remain in their declared tracks.

## Independent verification at the review handoff

`native-42` completed all 198 selected cases: **186 passed, 8 failed,
4 skipped**. It is a scoped run, not the full landing sweep. The failures
remain open:

- §14: the addressed copular read, forced c/e closing, and reverse-order
  pending premise.
- The real-model forgetting boundary.
- The paired bootstrap reconstruction invariant, with reported R values
  of `0.1` versus `0` in the first three paired rows.
- The compiled fixture's retired `symbolicOrder` override, its symbolic
  iteration-count assertion, and the peer-leg profiler's cube assertion.

The [run report](native-42/report.html) preserves the actual failures.
No fixture port or proposed repair is substituted for a passing result.
The cube/inverse, bind and reference groups passed in this run, but those
passes do not close the remaining gates or authorize publication.

The subsequent [regression receipt](regressions.md) records `native-43`:
the compiled file and addressed copular port pass; the paired reconstruction
and forgetting cases still fail. Its saved initialization and named costs
connect the reconstruction failure to the incomplete word-context bootstrap
in question 2. That question remains pending.
The later c/e port passes in `native-44`, and forgetting passes in
`native-45` after accounting for conceived-absence rows while preserving
the original absolute retention thresholds. Reverse-order premise
resolution remains failed. None of these scoped repairs closes either
architecture question or the full landing gate.
