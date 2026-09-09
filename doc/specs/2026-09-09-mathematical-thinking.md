# Mathematical thinking: bounded specification for iterative `what()` evaluation

> **Status:** specification, 2026-09-09, formalized from the Codex execution
> draft (archived verbatim as
> [2026-09-09-mathematical-thinking-codex-draft](../plans/2026-09-09-mathematical-thinking-codex-draft.md)).
> Governs the thinking requirements of the
> [What specification](2026-07-27-teaching-modes-and-next-iteration.md)
> sections 7–9.5 and defines the exact-arithmetic testbed used to measure
> them. The driveable plan is
> [2026-09-09-mathematical-thinking](../plans/2026-09-09-mathematical-thinking.md).
> Open questions are collected in section 12; none is silently resolved
> here. Nothing in this document claims learned behaviour; section 11 names
> the mechanism invariants and section 10 the learning gates.

> **Decision (Alec, 2026-09-09):** no logical machinery for mathematical
> reasoning in the runtime. Math is a simple syntax that tests the
> universal grammar and is expected to demonstrate reasoning more easily
> than English. `plus` is a transitive VERB with the grammar's existing
> verb definition (it maps its two noun arguments to a third concept,
> their sum); numerals are nouns the lexicon learns; intermediate thoughts
> are LTM slots of successive `what()` queries that establish truth in new
> regions of conceptual space. Section 5's primitives therefore live ONLY
> on the data / evaluation side (`bin/exact.py`: generation, the oracle,
> the verifier for scripted traces); the runtime resolve step (6.3)
> chooses only between answering and opening a subquestion about a
> presented word, and the root answer is conditioned on the row's LTM
> outputs by a learned attention. Addition is iterated SUCCESSION: a
> successor VP (`next(one) = two`) and counting through the thinking
> loop; `VerbLayer` is sufficient for the successor by construction
> (`test_verb_successor`), and its readout's dead zone at zero init is
> fixed so verbs can be learned by gradient. The learning gates are open
> (strict xfails) until the successor run learns the counting facts.

## 1. Objective

Make `Model` choose useful subquestions through recurrent `what()`
evaluation, retain their results in LTM, and construct a better root answer
than a single evaluation gives. Exact mathematics is the testbed because
correctness, progress, and the value of each intermediate result are
decidable, so *progressive illumination* of the relevant conceptual state
can be measured rather than asserted.

| Stage | Example | Capability under test |
|---|---|---|
| 0 Direct arithmetic | `25 + 6` → `31` | answer a direct problem (and reconstruct its input) with no variables and no subquestions; the prerequisite for everything below |
| 1 Dependency arithmetic | `a = 3 ; b = a + 4 ; c = 2 * b ; what is c ?` → `14` | resolve dependencies in a useful order, retain `b = 7`, use it for the parent |
| 2 Simultaneous constraints | `x + y = 12 ; y = 2 * x ; what is x ?` → `4` | combine constraints on the same variables until the answer is determined |

Bounded integers, unique solutions. Exact lookup, arithmetic and
substitution are supplied as named operations; the model selects
operations, operands and subquestions. An independent verifier checks
emitted steps. The complete solver exists only in dataset generation and
scoring. Learning the primitives themselves is a later experiment.

## 2. Relation to the existing architecture

- `Model.think()` (`bin/Models.py`) is the iterative evaluation loop of
  What spec §7.2: it appends `LTMSlot`s until parity and forces best-effort
  closure at the limit (§7.3). This specification adds the *content* of an
  iteration (a resolve step with a learned hard choice), the credit
  boundary, and the exact primitives. It does not add a recursive frame
  hierarchy, a query planner, or a second model.
- `Model.reverseOutput()` and `AnswerDerivation` (`bin/Output.py`) remain
  the only route from an answer symbol to `OutputSpace`. A computed integer
  enters the model as an answer symbol and descends through conceptual and
  perceptual synthesis; forced answers take the same route.
- The grammar chooser (`Language.MLPTransformChooser`) keeps its role over
  parse structure. The new `WhatStepChooser` decides *what to do with the
  question*: answer it, open a subquestion, or execute a primitive. Both
  are hard choices credited by policy objectives (What spec §9.5), never by
  differentiating an argmax.
- The Thinking Kernel (`bin/thinking.py`, `<thinkingBudget>`) is the
  truth-interval tool behind `answer_query`. It is reachable from a resolve
  step as an external tool (section 5.4) and is otherwise unchanged. The
  two loops are distinct: the kernel answers *isTrue/isPart/isEqual* over
  stored truth with a frame stack it owns; `Model.think()` answers *what*
  over conceptual state with the stack encoded in LTM slot parity. See
  [Reasoning](../Reasoning.md#two-thinking-loops).

## 3. Mathematical objects

All objects are typed; `R` is the configured integer range.

```text
Num        = int in [0, R)                       # bounded, exact
Var        = identifier word (a lexicon surface form)
Expr      ::= Num | Var | Expr + Expr | Expr - Expr | Num * Var | Var * Num
Equation   = (lhs: Var, rhs: Expr)               # stage 1: solved form
           | (lhs: Expr, rhs: Expr)              # stage 2: linear constraint
Problem    = (equations: E_1..E_m, distractors: D_1..D_d, query: Var,
              answer: Num, depth: int, structure: hash)
Bindings   β : Var ⇀ Num                          # partial map
Candidates C : Var → 𝒫([0, R))                    # stage 2 only; oracle side
ExactState σ = (E, β)                             # model-owned scratchpad
```

`depth(P)` is the length of the longest dependency chain from a constant to
`query`. `structure(P)` is the isomorphism class of the dependency graph
with variable names and constants erased; it is the split key
(section 4.3).

Illumination of the query after step `t`, defined on the oracle side only:

```text
I_t = 1 − log|C_t(query)| / log R        ∈ [0, 1]
C_t = the set of values of query consistent with the constraints the model
      has applied through step t (stage 1: [0, R) until query is bound,
      then a singleton)
```

`I_t` is the mathematical analogue of the truth-interval *luminosity* used
by the kernel; it never enters model context.

## 4. Data

### 4.1 Presentation

Stage 0 presents the bare expression (`a + b`; `sub` / `mul` under
`<mathOperators>`), its value as the one-hot answer, and holds out unseen
operand PAIRS (`exact.split_by_surface`); it is a stochastic sample with
replacement, so `<mathProblems>` should be large (thousands). A
stage-0 row IS lexed: a bare expression is the question, presented as
the single solved-form premise `_ = a + b` with query `_`, so the same
primitives answer it (`evaluate(0)`, `bind(_, value)`, ANSWER). The
numeral code is one-hot inside the root slot when the symbol width is at
least `R` (a linear adapter then realizes the one-hot answer directly; a
binary code is not linearly decodable to one-hot), else the binary code.

For stages 1 and 2, one presentation (one `where`) is one problem: the
premises, distractors and question rendered as a single sentence with `;`
clause separators and a terminal `what is q ?`. The desired answer is the one-hot vector
`onehot(answer, R)`; `Data.what(What.supervised(where))` returns it.
Premise order is shuffled per presentation; positions are assigned
independently of answers.

`Data.loadMath()` (`bin/data.py`, `dataset="math"`) draws problems from
`exact.MathProblemGenerator(seed, R, depths, distractors, stage)` and calls
`processLM()` with numeric labels, so `has_supervised_outputs` is true and
`source_addresses` are per-row as for every non-FineWeb dataset.

### 4.2 Generator invariants

- Unique solution; every equation is consistent; every constant and
  intermediate value stays in `[0, R)`.
- Fresh variable names and constants per problem; distractors are
  well-formed equations over variables not on the query's dependency chain.
- The generator records `depth`, `structure`, and the solver's binding
  order (used only by the verifier and the reports).

### 4.3 Splits

`train` / `validation` / `test` partition problems by `structure`; the
held-out depth band (4–6) appears only in `test`. Datasets, seeds and
gates are frozen before evaluation (section 10).

## 5. Exact primitives (data and evaluation side only)

> Per the decision above, nothing in this section executes in the model.
> The primitives define the generator, the oracle answer, and the
> verifier's replay of a *scripted* trace; the "resolve step" that
> chooses them is the evaluation harness's, not the model's.

### 5.1 Lexical exposure

`exact.ExactLexer.lex(surface) -> (E, query)` is a deterministic,
parameter-free clause lexer, part of the lexicon level of perception (the
symbolic-mind boundary permits the lexicon to *present* content). It runs
inside `Model.understand()` when `<whatThinkingPrimitives>` is positive and
attaches its result to the derivation context, not to the `Understanding`
value (which stays free of answer-side state). Premise index `i` is an
explicit operand and is the minimal intra-datum `.where` rung deferred in
[WhatSpacetimeDesign](../WhatSpacetimeDesign.md) §2.

### 5.2 Operations

Each primitive is a total function of `(σ, operands)`; each execution costs
one unit of the per-iteration primitive budget.

```text
lookup(v)           -> β(v) | ⊥
evaluate(i)         -> Num | Unbound(vars)      # rhs of E_i under β
bind(v, n)          -> σ' with β' = β[v ↦ n]    # n clamped into [0, R)
substitute(i, j)    -> E_i rewritten with the variable E_j defines
                       (solved form v = expr) replaced by expr; the
                       rewritten E_i replaces the old one in σ   (stage 2)
constrain(i)        -> the linear constraint E_i applied under β; returns
                       Bound(v, n) when exactly one variable is unbound and
                       the integral solution lies in [0, R), else
                       Unbound(vars)                              (stage 2)
```

`substitute` is algebraic elimination (the draft's "substitution"); with
`x + y = 12 ; y = 2 * x` nothing is bound, so `substitute(0, 1)` gives
`x + 2 * x = 12`, `constrain(0)` gives `Bound(x, 4)`, `bind(x, 4)`, and
`constrain(1)` then gives `Bound(y, 8)`. Implementation: `bin/exact.py`
`ExactState`; justification of a `bind` (an earlier `evaluate` /
`constrain` producing that value for that variable) is checked by the
verifier (section 9), not by the scratchpad.

They are declared as `<Queries>` introspection ops in `data/math.grammar`
(parse-NOPs, like `isTrue`), so they are named grammatical operations with
explicit operands and results and never chart rules.

### 5.3 Trace record

Every execution appends to the derivation trace:

```text
{"operation": "exact:<op>", "operands": (...), "result": ...,
 "iteration": t, "references": (token positions of the operand referents
                                 in the presented surface)}
```

The references are the intra-datum `.where` rung of section 5.1 (token
positions), not concept ids: binding a referent to its concept row is a
follow-on once the word registry exposes a surface lookup (section 12,
Q3). Every step *choice* is also traced
(`{"operation": "step:<kind>", "choice", "candidates", "index", "role",
"referent", "iteration", "closure_pressure"}`), so the replay has both the
alternatives offered and the one taken.

### 5.4 External tools

A resolve step may also call the Thinking Kernel (`think_about`) or
`answer_query` for prompted truth questions, as today; the result enters
the trace with `source="reasoning"`.

## 6. The thinking episode

### 6.1 Episode object

```text
WhatEpisode:
    understanding: Understanding         # one per episode (decision D1)
    execution                            # the established forward tuple
    root: WhatQuestion
    σ: ExactState                        # empty when primitives are off
    slots: [LTMSlot]                     # appended this episode
    pressures: [float]
    budget: {iterations, primitives_per_iteration}
    trace: [...]
```

One `WhatEpisode` per batch row. The stack is *not* in the episode; it is
the parity of the row's LTM slot sequence (What spec §6.2).

### 6.2 Loop

```text
episode = begin(question, input)                     # forward() once
answer  = Model.what(question, input, iteration=0)   # iteration 0 as today
while any row not at parity and iterations < L:
    pressure = schedule(iterations, L)               # monotone, 6.4
    answer = Model.what(question, execution=episode.execution,
                        iteration=iterations, closure_pressure=pressure)
    iterations += 1
force LIFO closure for rows still open (What spec 7.3)
end(episode)                                         # section 8
```

`what()` at iteration ≥ 1 skips `forward()` (the existing `execution`
seam), recaptures the same `Understanding`, and runs `reverseOutput()` with
the enlarged LTM context. Rows already at parity are masked from further
slots.

### 6.3 Resolve step

> Runtime form (Alec 2026-09-09): `_resolve_step` offers ANSWER and
> OPEN(w) for each presented word `w` (the lexicon's segmentation of the
> input; `Meronomy.word_spans`), with lexical / mnemonic candidate
> features only (kind, surface position, already answered in this row's
> LTM). ANSWER conditions the root slot on the row's LTM output
> representations through a zero-initialised attention (`ltm_attention`);
> OPEN installs the referent's perceptual slot as the QUERY symbol. No
> EXECUTE kind, no bindings, no numeral or referent codes. The text
> below records the earlier exact-route form for the history of the
> pilot; it is not the runtime.

`_resolve_answer(understanding, question, ltm_context, pressure, σ)` runs
the `WhatStepChooser` once per iteration and returns an `AnswerDerivation`
whose new field `step` is one of:

```text
ANSWER(symbol)                  -> slot (input, output), or (—, output)
                                   when closing an open subquestion
OPEN(subquestion v)             -> slot (input=QUERY(v), —)
EXECUTE(op, operands) then ...  -> up to primitives_per_iteration
                                   primitives, then ANSWER or OPEN
```

`QUERY(v)` is the model's conceptual representation of the interrogative
over `v`: the referent's concept code installed in the root slot with the
interrogative role, the same `_install_root_slot` mechanism recall uses.
The subquestion carries no dataset coordinate.

A computed `Num` sets the answer symbol through `exact.numeral_code(n)`,
the fixed binary-digit code in the root slot (the lexicon-row variant of
section 12 Q3 is not implemented: the word registry has no surface lookup
today). A deferred question's QUERY symbol uses `exact.referent_code(v)`
(a stable hash of the name, disjoint from every numeral code). The symbol
then descends through `ConceptualSpace.synthesize` →
`PerceptualSpace.synthesize` → `OutputSpace.from_percepts` exactly as any
answer. Implementation: `Model._resolve_step`, candidates from
`_enumerate_step_candidates` (ANSWER first, then OPEN(v) for unbound
variables not in play, then the applicable EXECUTE primitives, bounded at
32), the chooser `Language.WhatStepChooser`, the slot rules in the default
`choose_what_slot`. Which question a row answers is derived, not stored:
the subquestion posed at the previous iteration (*pending*), else the
newest open LTM input's referent (read from the slot's trace), else the
root.

### 6.4 Closure pressure and forcing

`<whatThinkingPressure>` selects the schedule; all are monotone in the
iteration index with `p(L−1) = 1`:

```text
linear    p_i = i / (L − 1)                       (today's schedule)
quadratic p_i = (i / (L − 1))^2
step      p_i = 0 for i < L/2, else 1
```

At `L`, forced output-only slots close every open input from the top of
the stack, each carrying a concrete symbol chosen by `_best_effort_what`
and realized through `reverseOutput()`. `unknown` / `unresolved` / failure
are not answers.

## 7. Memory

### 7.1 Interaction memory

`Layers.WhatInteractionMemory` owns the per-row slot deques, closure
pressures, `append_what_slot`, `get_what_slots`, `open_what_slots`,
`what_open_depth`, `what_at_parity`, `what_context`, capacity trimming and
resets, extracted verbatim from `InterSentenceLayer`, which composes it.
`Model._what_memory()` returns the discourse layer's memory when
`sentencePrediction` is on, else the standalone memory built under
`<whatThinkingMemory>true`. Slot semantics (push / complete / pop, LIFO,
immutable openings, monotone pressure, balanced-prefix eviction) are
unchanged and remain covered by `test/test_what_spacetime.py`.

### 7.2 Slot content for thinking

- `(QUERY(v), —)` opens a subquestion (input-only).
- `(QUERY(v), value_symbol)` records a subquestion answered in one step.
- `(—, value_symbol)` closes the most recent open question.

The output half is always the model's constructed symbol, never a dataset
value.

### 7.3 Context feed

`what_context()` gains `open_question` (the newest open slot's input
representation) and `latest_output` (the newest output representation) as
tensors; `_resolve_answer` consumes them directly, so completed subanswers
transform the parent's evaluation (What spec §7.2). The chooser context
keeps its 29-dim summary form; the widening is on the resolve side.

## 8. Training and credit

### 8.1 One episode, one step

When `<whatThinkingIterations>` > 1 and `<answerSynthesis>` is on,
`runBatch` drives `think()` instead of a single `what()`. The root
construction after parity is `_last_answer_construction`;
`answer_construction` is scored on it against `Data.what()` (What spec
§9.5). Intermediate subanswers are not compared with the root target.
Exactly one optimizer step follows (reconstruction priority unchanged).
Implementation: `Model._what_or_think` (the `runBatch` seam; exploration
trials never think); a row's root answer symbol is recorded when it
answers and carried through the later iterations' constructions, so the
final construction holds every row's root.

### 8.2 Credit boundary

`begin_what_episode()` marks the row's slot sequence; values appended
during the episode stay live (not detached) so the root loss reaches
earlier continuous states. `end_what_episode(detach=True)` detaches them
after the optimizer step. `<whatThinkingDetach>`: `episode` (default) or
`slot` (detach at append, today's behaviour). The active boundary is
reported.

### 8.3 Hard choices

The `WhatStepChooser` samples during training and takes the argmax during
evaluation. Its credit is a policy objective, weighted by
`<whatThinkingPolicyWeight>`:

```text
G_row  = −L_answer(root) + c_bind · |β_row| − c_step · iterations
         − c_forced · forced_closures
A_t    = G_row − b        (b: running mean baseline per model)
L_pol  = −mean_t A_t · log π(step_t | context_t)
```

(Runtime form: the root answer is the only reward -- `c_bind` is gone
with the scratchpad.) Historical note on the exact route: `c_bind ·
|β_row|` was dense credit for every variable the row's scratchpad bound
during the episode. Without it the depth-1 chain (four primitives before any
answer reward) is not found by sampling; with it the policy discovers
open → evaluate → bind → answer (pilot report, "Stage 1"). The candidate
features the chooser reads are content, not position: kind, primitive,
active-referent, bound, applied, position, closure pressure, READY (an
`evaluate` whose operands are all bound) and DEPENDENCY (an `open` /
`bind` whose referent the active question's premise needs); the last two
are what lets a policy trained on three-premise problems answer
two-premise ones. `lookup` remains a primitive but is not on the action
menu; an `evaluate` whose result is already justified is offered as its
`bind`.

reported under `what_report()["policy"]["thinking"]` separately from the
grammar chooser's `forwardGrammarWeight` credit and from the continuous
`answer_construction`. Continuous operands (the answer symbol, synthesis
layers, the question conditioner) train through the answer loss as today.
Implementation (`_what_step_policy_loss`, weight
`<whatThinkingPolicyWeight>`): the reward uses the batch's
`answer_construction` value, `c_step = 0.01`, `c_forced = 0.1`, an
exponential-moving-average baseline (`0.9 / 0.1`), and the mean over the
episode's recorded choices; a positive weight also makes the chooser
*sample* during training (the untrained head is uniform over candidates,
so exploration starts wide). Behaviour cloning on the model's own
verifier-accepted traces (the `next_op_loss` pattern) is NOT implemented
(section 12, Q2); which estimators are active is visible in the report.

### 8.4 Per-iteration bound

`<whatThinkingPrimitives>` bounds primitive executions per iteration; the
iteration limit bounds the episode; together they bound compute and
memory.

## 9. Verification and measurement (evaluation side only)

- `exact.ExactVerifier.check(trace, problem)` replays every `exact:*` step
  with the primitives under its own bindings: a step is *accepted* iff its
  preconditions held (operands bound) and its recorded result equals the
  replay; the derivation is *valid* iff the emitted answer equals the
  replay's `β(query)`. Rejected steps and forced closures are counted
  separately.
- Illumination `I_t` (section 3) is computed from the accepted steps.
- The *illumination probe* is a linear readout trained on detached
  conceptual states of evaluation problems to predict the candidate mask
  `C_t(query)`; it measures whether useful steps narrow candidates while
  retaining the true value. Probe targets never enter model context.
- `bin/eval_math_thinking.py` runs a checkpoint at budgets 1 / 4 / 8 / 16
  with equal primitive allowance per iteration and writes the report of
  section 10.

## 10. Acceptance

Status 2026-09-09 (pilot report, "Stage 1"): the trained-depth gates are
met on the exact route -- unseen structures at the trained depths 100 %,
the never-trained next depth 100 % (depths 1–2 → 3) with the content
features -- on the 2048-problem, 64-wide configuration, and the small
configuration is a default-suite regression
(`test_stage_one_dependency_chains_learn_through_the_exact_route`).
Depths 4–6 and stage 2 are not yet evaluated.

Pilot gates (frozen before evaluation; three seeds):

- ≥ 95 % exact-answer accuracy on unseen structures at trained depths 1–3;
- ≥ 80 % at depths 4–6;
- ≥ +10 points on the multi-step subset versus the one-iteration budget;
- every accepted derivation step verifies; rejected steps and forced
  closures reported;
- removing intermediate memory (`<whatThinkingDetach>slot` plus a
  zero-capacity memory) measurably harms multi-step accuracy.

Report by depth and budget: accuracy, derivation validity, memory
ablation, candidate reduction, mean iterations, forced-closure rate, both
primary losses, primitive counts, latency, and the B24 clean-corpus
throughput band (the What spec 12.15 15 % gate) for the canonical config
with thinking off.

## 11. Invariants (each has a named test in the plan)

1. Byte-identical defaults: `whatThinkingIterations=1`, memory off,
   primitives off ⇒ established `what()` / `runBatch` unchanged.
2. One `forward()` per episode; iteration ≥ 1 never re-encodes the input.
3. The desired `Data` answer is unreachable from `think()`, the chooser,
   the primitives, `σ`, and LTM before loss.
4. Parity, LIFO order, immutable openings, monotone pressure, capacity
   eviction: unchanged and per row.
5. Forced closure always yields a scoreable root that traversed
   `reverseOutput()`.
6. Changing a necessary intermediate `bind` changes the emitted answer;
   editing only the trace does not.
7. Root loss reaches states created at earlier iterations under `episode`
   detachment and does not under `slot`.
8. Primitive executions per iteration ≤ budget; iterations ≤ limit.
9. Every primitive execution and every step choice is in the replayable
   trace with operands and result.
10. The verifier accepts every step of a solver-order trace and rejects a
    step whose result is edited.

## 12. Open questions

- **Q1** One understanding per episode (D1) versus re-running `forward()`
  each iteration as `think()` does today. Recommendation: D1.
- **Q2** Chooser credit: REINFORCE with root reward (8.3) only, or also
  behaviour cloning on the model's verifier-accepted traces.
  Recommendation: both, reported separately; cloning from the *solver's*
  order is excluded by the "solver only in generation and scoring" rule.
- **Q3** Numeral representation: lexicon-row code with binary fallback
  (6.3) versus a dedicated numeral sub-band of the symbol. Recommendation:
  the lexicon-row code for stage 1.
- **Q4** Exact clause lexing as lexicon-level perception (5.1) versus
  requiring the learned grammar to parse equations before any primitive
  runs. Recommendation: the lexer now; grammar parsing is the "learn the
  primitives later" experiment.
- **Q5** Are the numeric pilot gates merge-blocking for the phases that
  land mechanism and tests, or reported only? Recommendation: mechanism
  phases gate on section 11; the learning gates gate the final report.
- **Q6** Should `thought_free` (Shamatha) force `whatThinkingIterations=1`
  at serve time? Recommendation: yes (no internal dialogue).
- **Q7** Range `R` and premise count for the pilot (proposal: `R = 64`,
  `m ≤ 6`, `d ≤ 2`).
