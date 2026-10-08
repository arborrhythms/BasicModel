# Reasoning and thinking

The current contract is [item 6.2](specs/2026-10-07-thinking.md), following the
accepted 6.5 mechanism. The implementation is a review candidate; the receipt
is [here](benchmarks/2026-10-07-item6-2/README.md).

A question is a row with an open reference: referent, relation or evidence
pair. Wh-words and punctuation can suggest an opening, but a bound row is not
a question. Every open closing enters `run_selected_thought`; nested `ask`
shares the root's `attentionBudget`. While work remains, `conclude` requires
all references bound. Exhaustion stores the unfilled row as a question.

| Symbolic face | Conceptual face |
| --- | --- |
| `isTrue(P)`: ended row's pair, trust and witnesses | `exist(P)`: conceptual presence and its pair |
| `isPart(a,b)`: LTM and taxonomy parthood | `part(a,b)`: containment content and pair |
| `isEqual(a,b)`: DEF rows and references | `equal(a,b)`: identity of codes and pair |
| `isImplied(P,Q)`: implication with antecedent evidence | `implies(P,Q)`: containment of regions and pair |

`query(pattern)` returns the best matching row. `ask(row)` attempts to fill
an open reference, using queries and nested questions. `not` exchanges poles
of a pair or meaning. Gain changes the next sentence's global expectation.
No thought operation returns a scalar. Neither trust nor lack of support is
silently substituted for an evidence pole.

Every result becomes a serial slot with its content, pair, witnesses and
producing operation. Chains bind those slots: the `a < b`, `b < c` fixture
performs two queries and concludes `a < c` with both source references.
Modus ponens requires an antecedent witness as well as the implication.
Conclusions are LTM `inference` rows, addressed by document/turn, ordinal and
content. No second semantic store or planner stack is introduced.

A bound declarative opens no episode. Its uncancelled expectation image can
prompt `not X`, an absence inference with image confidence against. The
four-corner refinement policy remains future work.

The existing grammar scorer makes the choices. Compose's paired rule,
`K · R · p(a_dep) · ΔC`, credits the supplied answer, next-sentence expectation
error and spent budget. Exact ties move no policy weight. The comparison
reader scores the answer term; the presented answer is the filled row.
The REINFORCE/EMA path and `selectedThoughtPolicyWeight` are removed.

Thought `what` raises with `ask`; LTM `what`/`lookup` raise with `query`.
`chunk` becomes `synthesize`, inverse `analyze`. `quantize` is removed;
symbolize/conceptualize remain future work. `arma`/`expect` are removed from
thought in favour of `<sentenceExpectation>` and gain. `true` raises with
`isTrue`. `thinkingBudget` and `selectedThoughtBudget` fail at load; use
`attentionBudget`.

Boundary admission, typed capability views and work limits are documented in
[QueryContracts](QueryContracts.md). Replay, nesting and occurrence retention
are in [ThoughtHistory](ThoughtHistory.md). Numerical proof utilities in
`TruthGroundedReasoner` remain diagnostics; production reasoning uses the
ordinary grammar controller and native row evidence.
