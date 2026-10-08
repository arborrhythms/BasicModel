# Item 6.2: thinking and operator renames

Status: **review hold; not ready to land**. The default sweep is green, but the explicit thinking gate failed. **No commit, push or parent bump.**

The [review blockers](review-blockers.md) give the exact failures and the
construction-only diagnosis. No failed training or gate was retried.
The base remains the accepted 6.5 mechanism landing,
`631d9e8e44b7c8034263e22b36dc74fa5df4eb75`.

The [specification](../../specs/2026-10-07-thinking.md),
[Alec's decisions](../../plans/2026-10-07-thought-loop.md), and
[predeclared protocol](protocol.json) govern this step.
The 6.5 learning gates — held-out anaphora, verb reuse, prediction control,
shuffled order, renamed vocabulary and determiner control, seeds 0/1/2 —
**remain pending the million-sentence checkpoint and are not claimed**.

## Delivered behavior

`ask` fills open references; `query` returns the best matching LTM row.
`synthesize` has inverse `analyze`; the retired names diagnose their replacements.
`quantize`, `arma` and `expect` leave thought; expectation and gain are global.
`isTrue` and `exist` leave pairs. Retired budget names and the old thought
policy weight fail at load. Every public or closing thought entry uses
`attentionBudget`.

Questions carry open referent, relation or evidence slots. The binder's null
choice survives the native grammar journal; a bound row opens no episode
whatever its surface. Nested asks share the work meter. Conclude is illegal
while an open reference and work remain; exhaustion preserves a question row.
Later matching content can fill it within its document.

Every checked result becomes a serial slot with content, a pair, witnesses and
its producing operation. Both faces are installed: symbolic faces read native
rows and taxonomy references; conceptual faces compute on codes. The parthood
fixture uses two queries and one conclusion, retaining both premises in an
occurrence-addressed inference. Modus ponens additionally requires an antecedent.
Temporal implication compares endpoint `.when` order within one document.
Nested thought writes are restricted to inference and unresolved question rows
and preflight their capacity. An absence inference retains the expectation's
source occurrences and its confidence against.

Thought uses the existing grammar scorer. Greedy and one uniform departure
receive compose's `K·R·p(a_dep)·ΔC` credit from available answer, next-sentence
expectation and work. The presented reader trains on the kept binding; the
comparison reader sees both eligible trials. The later expectation target
credits the held decisions without rewriting an already published conclusion.
Exact ties move no policy weight. The REINFORCE/EMA path is retired.

## Measurements on the delivered source

| Check | Outcome |
| --- | --- |
| Full default sweep | 5557/5557 completed; 5,270 passed, 286 existing skips, 1 non-strict XPASS; exit 0 |
| Standing XOR class gate, same ten trainings | 10/10 |
| Standing XOR reconstruction gate, same ten trainings | 10/10 |
| Both XOR gates | 10/10 |
| Standing MM_xor convergence | 10/10 |
| Standing sum negative control | 10/10 |
| Thought episodes in all thirty standing trainings | 0 |
| Thinking mechanisms, including 11c and chaining with credit | {'passed': 41, 'failed': 6}; exit 1 |
| Unforced configured MM_query_reasoning run | failed; 2 of 300 configured training epochs completed |

The unforced configured run's process result is `IndexError('tuple index out of range')`.
Its [outcome](mm-query-configured/outcome.json) and [log](mm-query-configured/run.log)
are distinct from the supplied-reading optimizer smoke in the mechanism suite.
No convergence or learned parsing claim is inferred from a smoke test.

The standing thirty use the unchanged four gate/config files, with their hashes
checked against the 6.5 landing. The two XOR bars share each of ten trainings;
there are ten additional sum controls and ten MM_xor trainings. The control is
read before XOR starts. Every outcome is retained; there is no seed, retry,
replacement or gate relaxation. The 11c measurement suppresses and records its
historical local `torch.manual_seed` calls, preserving an unseeded run while
leaving its source assertions intact.

The [thinking plan](thinking-gate-plan.json) lists the nineteen historical 11c
nodes verbatim, the MM_query_reasoning optimizer smoke, the native per-row
answer integration and the new thinking certificates. Supplied-operation and
supplied-reading fixtures establish mechanisms; they do not measure learned
English anaphora or unforced chain selection. Credit observations are retained
in `thinking-observations/`.

The supplied toy is unchanged: archived answer-credit success is **20/20** and
expectation-only success **17/20**. The specification's “every seed” sentence
does not describe the latter result. No toy rerun or seed selection was made.

The explicit thinking result breaks down as 25/25 new certificates, 1/1
native answer integration, 1/2 MM configuration/smoke checks, and 14/19
11c checks. These failures are retained in [the review blockers](review-blockers.md).

## Review artifacts

- [Machine-readable summary](review-summary.json) and [standing measurements](measurements/summary.json).
- [Final complete sweep](final-sweep/result.json) and [thinking mechanism sweep](thinking-gate/result.json).
- [Source and protected-file hashes](measured-source/freeze.json), [source map](measured-source/source.json), complete source archive and helper archive in `measured-source/`.
- The complete 6.5 baseline is preserved in `baseline-source.zip`; [test-name migrations](test-contract-migrations.json) identify retired or renamed contracts.
- Numbered development logs, `development-sweep/`, `affected-sweep/` and `development-sweep-2/` preserve diagnostic failures. The last candidate sweep had three stale-contract assertions, repaired before the final sweep. No training measurement had begun then.
- Current QueryContracts, AccessibleMind, Reasoning, ThoughtHistory, the September 18 spec status, catalogue §§3.1 and 3.5–3.7 and §6, and `todo.md` carry the new contract.

Stop here for Claude's review before any commit.
