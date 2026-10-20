# One-successor answer-cost cause, seed 4199722147

The native diagnostic reuses the original constructor parameters, two
documents, optimizer, forced grammar scope and 512-unit question budget.
`answer-trace-before.jsonl` records both sentence readers before each path
score and the subsequent episode. No assertion or winner is overridden.

| Question-batch component | Greedy, both rows | Explore row 0 | Explore row 1 |
|---|---:|---:|---:|
| Surface `answer.what` | 2.31085944 | 1.07095504 | 1.00866330 |
| Surface `answer.where` | 1.00903296 | 1.00903296 | 1.00914717 |
| Surface `answer.when` | 1.02497137 | 1.02497137 | 1.03047597 |
| Weighted reader total | 1.92190528 | 1.05397224 | 1.01094127 |
| Closing reference credit | 0 | 0 | .10000000 |
| Previous A used to keep | 1.92190528 | 1.05397224 | 1.11094129 |
| R | .18397716 | .19397715 | .20397714 |

The presented and comparison readers are still identical on this first
answer batch. Their individual terms are relative surface errors, combined
by the existing loss weights; they are not binding error components.

`_sentence_path_cost` calls `_sentence_answer_error` before
`_commit_sentence`. Only the kept close then calls `ThoughtClosing.close`.
Consequently, the greedy trial had not run a proof when the recorded 1.92
was computed. It lost because the output reader reconstructed its whole
question less accurately than the wrong explore reading. The proposed old
evidence-slot explanation does not account for this trace.

The repair uses `BindingAnswers.cost` for a binding task's sentence A, as
the completed thought scorer already does. Both unfilled sentence trials
now cost 1; row 1's separate closing reference cost remains .1. R favours
the intact reading, which then runs its ordinary checked proof. The output
readers still train on exactly the same surfaces: the raw reader numbers
above remain identical in `answer-scoring-repair.jsonl`. There is no new
keep rule, target-conditioned operation or forced winner.

The completed diagnostic reports both correct bindings, with one inference
and the required query/ask/descent/return chain per document. A regression
also presents a correct binding to a deliberately worse surface reader:
referent A is 0, incorrect identity .5, and missing referent 1, while the
surface reader's gradient remains intact. The unchanged native certificates
and final landing sweep are recorded separately in this round's README.

The completed one-successor replay's episode trace records the actual
post-thought judge as well: for both documents, the proved greedy binding
has **A=0**, while the live thought departure has **A=.66483945**. The
correct proof is kept without a work surcharge.
