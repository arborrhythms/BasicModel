# Forced ordinary-path diagnostic — certificate not passed

This is development evidence, not a learning attempt or a passing §14.11
certificate. Two real documents run through `MathChainTraining.present()` with
an optimizer, the unchanged frozen observer, and the live compose and thought
explore suffixes. Grammar choices and the first query's operands/operation are
forced and labelled. No executor result, binding, evidence, writer, cost or keep
decision is replaced.

Both documents present `x is three.`, `three plus one is four.`,
`y is x plus one.`, and `what is y ?`. The first forced query uses the ordinary
question's native `y` reference to retrieve its premise. In both rows the result
is `query`, `conclude`, with the question's role 0 bound to **y**, not four.
The frozen observer records two incorrect answers and two failed chains.
`diagnostic.json`, `questions.jsonl`, the ordinary trace and the copied fixture
sources retain the result.

`ThoughtStream.integrate()` immediately passes an ordinary query result to
`ThoughtReferences.fill()`. That fills the goal's open role from the same role
of the returned premise (`y`), rather than retaining a still-open answer goal
while a selected decomposition resolves the premise's value. Later fills cannot
replace that already-bound role. Separately, `ThoughtStream.commit()` writes
every bound subquestion return as an inference; a premise lookup return would
therefore add a row beyond the successor count required by the frozen verifier.
The latter is a source-code finding, not exercised by this first-query trace.

No production fix has been applied. A scope clarification is pending because
Alec's current instruction explicitly defers implementing the found-candidate
binding correction while also requiring this forced decomposition certificate.
The minimal proposal is a general selected binding/return repair sufficient for
the certificate; the curriculum, intermediate-answer scoring and learning
measurement remain deferred. No arithmetic executor or hand-written arithmetic
rule is proposed. Until that decision, no passing certificate, mechanism closure
or new learning result is claimed.
