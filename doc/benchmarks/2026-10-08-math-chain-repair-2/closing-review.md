# Item 6.2 — forced decomposition and stopped learning campaign

The MM_math_chain campaign remains **stopped by decision**, at two of thirty
trainings: eight complete answer-plus-expectation epochs and nine
expectation-only epochs, plus their partial next epochs. The other 28 trainings
never started. No training is retried or replaced, and there is no learning or
held-out-binding claim. The stopped files and original protocol are intact.

The minimal authorized binding/return repair now demonstrates one- and
two-successor decomposition through the real training driver. This candidate
is for Claude's review; it is not an acceptance record.

The reason and complete saved-run reports are in
[the stop receipt](stopped-by-decision/README.md). The future learning corrections
remain beside `protocol.json`. The user's subsequent authorization permits the
minimal general binding/return repair for this mechanism demonstration;
`binding-repair-authorization.json` records that scope. Curriculum and
intermediate-answer credit changes remain deferred to item 0.

## Mechanism and limits

The ordinary closing previously copied the subject of a retrieved equality
into the question's open subject. It could answer `what is y ?` with `y`.
Binding now aligns the fixed region with the matching side of the equality
and fills the variable with its other side. Unrelated positive rows cannot
fill it. Further substitutions require equalities already read into the
episode's serial results and matching structural operators.

`ask` enters a nested episode when its requested region is unresolved. Its
return carries the child's resolved meaning, pair and witnesses as a serial
result. A direct child lookup does not create an extra inference. A derived
child is committed before its parent, and the parent's provenance uses that
inference's native occurrence. Reader scratch values are restored at the
episode boundary; reader credit remains in its owning loss records.

A referenced composed description keeps its actual completed operand as an
ordinary native row. Its two-address cache carries the producing operation's
provenance, so identical operands composed by different operations cannot be
treated as equal. Ordinary absolute sentences retain their existing storage
contract. No pre-fusion numerical target or action journal is saved, and no
arithmetic operation, arithmetic rule or numerical decoding head is added.

The forced fixtures use `MathChainTraining.present(split='train', optimizer=…)`,
the unchanged observer and chain verifier, and the live compose and thought
explore suffixes. Only grammar operations, reference selections and checked
thought-request preferences are forced. The costs, keep rule, results,
bindings and writes are not replaced. No seed is supplied.
The fixture names complete checked premise patterns, including both operands;
the ordinary executor must retrieve them and justify the substitution. Bare
word binding preferences do not apply to an already completed phrase.

The single-step problem uses `x is three.`, `y is x plus one.`,
`three plus one is four.`, then `what is y ?`. The two-step problem makes the
decomposition explicit through `z is x plus one.` and `y is z plus one.`,
with both counting facts. It demonstrates nested substitution; learning a
recursive policy for the opaque phrase `plus two` remains deferred.

Premise ingestion uses zero thought budget; the question fixtures allow 512
and 768 work units respectively because native bounded reads count toward
work. These are forced mechanism fixtures, not the stopped campaign's
attentionBudget-32 learning protocol. Every companion-row outcome is kept;
the certificate is an architecture demonstration, not an accuracy bar.

## Forced outcomes on the closing source

Both certificate tests pass in [closing-certificates-4.log](closing-certificates-4.log)
(59.22 seconds). All four document-row outcomes are below; the companion failure
is part of the result. The current unseeded existence test is not a guarantee
that every future forced run will keep the same compose reading.

| Successors | Document | Bound answer | Frozen chain verifier | Inference rows | Episode work |
|---|---:|---|---|---:|---:|
| One | 0 | four | pass | 1 | 208 |
| One | 1 | four | pass | 1 | 211 |
| Two | 0 | incorrect | fail | 1 | 531 |
| Two | 1 | five | pass | 2 | 525 |

The one-successor trace is `query(y) → ask(x) → query(x) → conclude/return →
query(counting fact) → conclude`. Its inference witnesses the premise for `y`,
`x is three.`, and `three plus one is four.`. The two-successor proof descends
through `z` into `x`. Its first inference witnesses the premise for `z`, the
`x` fact and the first counting fact; its second witnesses the premise for `y`,
the first inference's native occurrence and `four plus one is five.`. All
witnesses are recognized native references. The unchanged verifier recognizes
`four` and then `five` by identity.

In the failed companion row, the committed reading of the first counting fact
has the bare `three` reference as its left operand, rather than the completed
`three plus one` phrase. It supplies no valid structural substitution, and the
verifier rejects the chain. The successful companion and this failure were
both retained; neither is a learning accuracy measurement.

The [one-successor trace](closing-certificates-4/test_forced_native_decompositi0/decomposition-1/certificate.json)
and [two-successor trace](closing-certificates-4/test_forced_native_decompositi1/decomposition-2/certificate.json)
include the full episode state diffs, source readings, selected grammar
choices and live departures. Every episode changes no unrelated state: only
the original question row, new inference rows and the credit/history trail
remain. Successful rows add exactly one or two inference rows respectively.

## Unforced stopped-run observations

| Condition | Complete epochs | Questions including partial next epoch | Correct bindings | First-epoch greedy `what` openings |
|---|---:|---:|---:|---:|
| Answer and expectation | 8 | 977 | 0 | 0/117 |
| Expectation only | 9 | 1,167 | 0 | 0/117 |

In the first epoch, the answer condition left 112 references open and opened
104 question episodes; expectation-only left 108 open and opened 108 episodes.
Both observed 117 questions and zero correct bindings. The per-epoch/per-kind
episode counts and mean work, including partial epochs, are preserved in
`stopped-by-decision/episodes-by-start-epoch-kind.json`. No final parameter
checkpoint was saved, so final chooser movement cannot be reported.

## Validation status

The [closing summary](closing-summary.json) is **ready for Claude's review**.
The closing source is frozen in [closing-source-4](closing-source-4/freeze.json),
with source-manifest SHA-256
`21320d471189ff2e5668cbd95394c8a50da4224338befde3ef8829f590924373`.
Its archive includes the certificate helpers and the patch against the stopped
campaign source. The sweep uses the identical source manifest from revision 3;
revision 4 changes only the separate forced grammar helper, whose two slow
certificate cases run explicitly. Thinking is **57/57**, unseeded, in
[closing-thinking-4](closing-thinking-4/result.json). The
[closing sweep](closing-sweep-3/result.json) is **green: 5,359 passed, 286 skipped,
one XPASS; all 5,646 cases completed**, in 639.98 seconds, with peak aggregate
worker memory 4.90 GiB. The focused storage, binding and retrieval check passed
18/18. `git diff --check` is clean.

All development and validation failures remain. The first closing sweep
found three clause-storage regressions and was interrupted after 1,426 cases.
Operand retention was then restricted to referenced descriptions. The second
sweep completed and found one stale lookup fixture, ported from `ask` to
`query` without removing assertions. Certificate revisions 2 and 3 failed the
two-successor existence assertion; the checked-query fixture and its bare-word
binding preferences were corrected before revision 4. These are development
checks, never new or replacement campaign attempts.

The final integrity check confirms 59 stopped files, nine stop-receipt files,
222 original measurement helpers, and the earlier receipts' 2,393 and 1,967
files unchanged. The [BindingAnswers diff](binding-answers-frozen.diff),
[verifier check](binding-answers-verifier.json) and
[unchanged matches() text](binding-answers-matches.py.txt) remain available:
only the previously authorized cost change differs from the original frozen
file; `matches()` and `references()` are byte-identical.

The standing thirty are reused exactly as requested: sum 10/10; XOR class and
reconstruction 10/10; raw MM 8/10. The two MM misses and their exact landing
reproductions remain in the original report. No new standing attempt is run.
The learning gates for 6.5 and 9 remain pending item 0's checkpoint alongside
the deferred MM_math_chain learning measurement.

Stop for Claude's review before any commit, push or bump.
