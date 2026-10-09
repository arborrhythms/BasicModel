# Reasoning and thinking

The current contract is [item 6.2](specs/2026-10-07-thinking.md), following the
accepted 6.5 mechanism. The 6.2 mechanism and repairs are accepted and landed;
the [repair receipt](benchmarks/2026-10-07-item6-2-repair/README.md) records that
acceptance. Learned chaining is the separate
[MM_math_chain measurement](benchmarks/2026-10-07-math-chain/README.md), stopped
for review after all thirty attempts failed before completing an epoch.
Its sweep and mechanism gates pass, but no held-out learning result is
available. Configured-run completion does not establish learning.
The [second math-chain repair](benchmarks/2026-10-08-math-chain-repair-2/README.md)
is in development under §14; declaration awaits its complete epoch certificate.

A free variable is an open referent or relation. `(0,0)` is ignorance, not
an evidence slot. Wh-words and punctuation are evidence for the chooser.
A closing with a free variable, or an interrogative region without evidence,
enters `run_selected_thought`; nested `ask` shares the root's `attentionBudget`.
`conclude` requires bound variables and evidence, unless work is spent or
`thoughtSearchExhaustion` completed queries have found no candidate. At an
exhausted search, a source-supported forward name is provisionally minted;
a question's free variable remains open on its stored row. A later copula
can bind directly to that row's addressed column and fill it.

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
`K · R · p(a_dep) · ΔC`, compares reconstruction plus the supplied answer.
The thought walk also pays its own metered work; that cost does not enter
the sentence comparison. Expectation trains its predictors through their
owner-step registry. Exact ties move no policy weight. The comparison
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
