# Thinking Kernel — Execution Plan + Notes

> **Executes:** [thinking_kernel_spec.md](thinking_kernel_spec.md) (the
> lookup/part/think/query/answer kernel over truth intervals + trust, STM
> frame stack, runtime-enforced execution loop). It EXTENDS the live
> reasoner per [2026-06-23-reasoning-live-wiring.md](2026-06-23-reasoning-live-wiring.md)
> (serve.py `answer_query` → `Models._detect_query`/`reason_about` →
> `NeuralToolUser` over `reasoning.py`'s `TruthGroundedReasoner`); the
> tools build NO parse structure and live in `<Queries>`.

## 0. Architecture mapping (spec term → existing surface)

| spec term | current surface |
|---|---|
| conceptual location | an idea vector (or a `QuerySpec` naming a proposition over idea vectors) |
| LTM | the unified reasoning store — `TruthGroundedReasoner.reasoning_store()` (TernaryTruthStore under `<ltmConsolidation>`, else RelativeTruthStore) |
| `lookup()` | LTM-direct evidence only: `exist` (isTrue), `is_part_direct` (geometric + stored row), `equal` — **no chaining** (chaining is thinking) |
| `part()` meronomy | `reasoner.wholes()`/`parts()` over REL_PARTOF rows |
| `part()` taxonomy | the SAME REL_PARTOF rows today (the syllogism corpus stores subsumption as parthood; the 2026-06-16 design keeps taxonomy symbolic in the truth store, so the two modes share rows until the stores split — the mode rides the provenance) |
| `think()` frame | a kernel-internal `Frame` (dataclass stack), NOT the tensor `ShortTermMemory` — the spec's STM frames hold symbolic scratch (bindings/trace/budget), a different animal from the sequence STM |
| `query()` | an addressee registry (name → callable); `arma` is the built-in addressee (the discourse predictor is outside the kernel's memory); results are `Testimony`, never truth |
| grounded LTM write | ONLY the runtime writes: `materialize` (trusted derivation, §14.2) and `incorporate` (testimony with source×channel trust ≥ floor); the policy cannot write |
| soft op-ordering | `InterveningIdeaGenerator.propose`/`where_read` α ranks which whole to explore first (soft-propose / hard-verify, unchanged §0 discipline) |
| reward / traces | compiled BY the kernel from the frame trace (§12.4 — the mind never writes reward); exported as `(state, op)` examples for next-op training (the training loop itself is the follow-on seam) |

## 1. Gate + knob (live-wiring directive)

- `<architecture><thinkingBudget>` — `xs:nonNegativeInteger`, one knob in
  the `<reasoningIterations>` pattern: **absent/0 ⇒ kernel off,
  byte-identical**; positive N ⇒ kernel on, N = the op budget of a
  top-level frame (every executed op costs 1; child frames draw from the
  parent's remaining budget).
- When on, `Models.answer_query` ADDITIONALLY runs the kernel and attaches
  its result under the payload's `kernel` key — the existing
  `reason_about` fields are untouched, and serve.py needs no change (the
  payload dict already rides `response["reasoning"]`).
- Policy constants (thresholds, step cost, child-budget split) are CLASS
  ATTRIBUTES on `ThinkingKernel` (test seams), not XML knobs.

## 2. Module: `bin/thinking.py`

### 2.1 `TruthInterval`
`lower, upper ∈ [-1,1], trust, provenance` + derived:
- `luminosity = max(|lower|, |upper|)` (the spec §15 sketch);
- `status` (τ = the kernel's `tau`): `unknown` when `luminosity < τ`;
  `true` when `lower > 0` one-sided and luminous; `false` when `upper < 0`
  one-sided and luminous; `conflicting` when BOTH `lower ≤ -τ` and
  `upper ≥ +τ`; `mixed` for a luminous straddle that is not two-sided-strong.
  Closure requires `status ∈ {true,false}` — a conflicting interval never
  closes as a truth (luminosity alone is not enough).
- Built from an evidence list of signed values+trusts: `lower/upper` =
  min/max of the signed values, `trust` = max |evidence trust|; empty ⇒
  `[0,0]`, trust 0, unknown.

### 2.2 `Frame` / stack
`target, purpose, bindings, trace, budget, status, depth, children,
result`. Pushed by `think()`, popped by `answer()`; a closed child returns
ONLY its `ChildResult` (target, value, interval, trust, trace, provenance,
relevance_to_parent) — scratch is discarded (§2.3, §14.4).

### 2.3 `ThinkingKernel` (the runtime)
Owns the reasoner + optional (generator, ga, spaces) + the addressee
registry. `run(target) → ChildResult` drives §10.1:

```
while not frame.closed:
    op = policy.next_operation(frame, kernel)   # the mind proposes
    kernel.execute(frame, op)                   # the runtime enforces
```

`execute` validates the op name, charges the budget, runs the semantics,
appends to the trace, and — budget exhausted — force-closes with
`bounded_unknown` (§9.2 rule 4). The policy has no store handle: LTM
writes happen only inside `execute` on the two grounded paths.

### 2.4 `KernelPolicy` (deterministic baseline + soft ordering)
- step 1: `lookup(target)`; luminous+trusted one-sided ⇒ `answer`.
- binary isPart(A,B): `part(A, up)` → unvisited wholes, ordered by the
  soft α (`where_read` on `[A;B]`) when the soft half is present, else by
  hop trust; direct `lookup(isPart(M,B))` hit ⇒ answer(min trust); else
  `think(isPart(M,B))` subgoal (§7.4), child true ⇒ answer(min of child
  trust and hop trust); no candidates ⇒ answer(unknown) (bounded search
  failed, §9.2 rule 3).
- isTrue/isEqual: lookup-only leaves; unknown stays unknown (§14.6) unless
  a registered addressee is consulted by an explicit caller.

### 2.5 Rewards + traces (§12)
`compile_rewards(frame)`: per-op `relevance · trust · Δluminosity(target)
− step_cost`, terminal bonus on a §9.2-1/2 close, `bounded_unknown` valid
terminal, unsupported assertion = failure. `trace_examples(frame)`:
`[(state_descriptor, op_name), …]` for next-operation supervision. Data
producers only — wiring them into the optimizer is the documented seam.

## 3. Integration (`Models.py`, `model.xsd`)

- XSD: `<thinkingBudget>` beside `<reasoningIterations>` (:265).
- `BaseModel.__init__`: `self.thinking_budget` (absent ⇒ 0), parsed next
  to `reasoning_iterations` (~:840).
- `BaseModel.think_about(query_spec)`: builds the kernel from `self`
  (reasoner, `_reasoning_tooluser` soft parts, `_reasoning_spaces`,
  budget, materialize gated by `<ltmConsolidation>` as in `reason_about`);
  `None` when off.
- `answer_query`: after the existing `reason_about` payload, attach
  `payload["kernel"]` (value/interval/trust/trace/ops) when the budget is
  positive. Off ⇒ byte-identical.

## 4. Tests (`test/test_thinking_kernel.py`)

Interval math + statuses; frame push/pop + scratch discard; depth-0
(lookup→answer), depth-1 (part→lookup→answer), depth-2 (nested think
syllogism: socrates ⊑ man ⊑ mortal ⇒ true, trust = min-hop, lemma
materialized so the SECOND run is a depth-0 direct hit); budget exhaustion
⇒ `bounded_unknown`; anti-hallucination (an unknown run leaves the store
row-count unchanged; the policy cannot reach a write); testimony
(registered addressee → Testimony; NOT truth until `incorporate`, which
gates on source×channel trust); rewards (success trace earns positive
terminal, step costs charged); trace examples shape; config (absent ⇒ off
⇒ `answer_query` payload has no `kernel` key — byte-identical); Models
integration on a truthSet config.

## 5. Follow-ons LANDED (same day, second pass)

- **§12.6/12.7 next-op learning**: `NextOpPolicy` (a 6-feature MLP over the
  trace-state descriptor scoring the five ops) + `next_op_loss` (CE) +
  `traces_from_store` (self-supervised: the store's 2-hop transitive
  targets run through the deterministic teacher, materialize=False — trace
  generation never writes LTM). Gated `<training><thinkingLossWeight>`
  (0 ⇒ head not built, byte-identical); eager build for optimizer
  membership; the runBatch hook mirrors `answerLossWeight`. At inference
  `KernelPolicy(next_op=…)` consults the head ONLY at explore-vs-stop
  choice points over the LEGAL option menu — a bad head can waste budget
  but never assert. **Neutral at init** (regression found in-suite): the
  output layer is ZERO-initialized so an untrained head ties every op and
  `choose()` keeps the explore option — exactly the baseline; random init
  had made a fresh head randomly prefer "answer", killing climbs at
  inference before any training.
- **§7.1/§14.2 testimony in the loop**: the leaf policy consults each
  registered addressee once before conceding unknown; NUMERIC testimony
  folds into the frame interval as `asserted × source_trust` (flimsy
  testimony cannot manufacture luminosity); tensor testimony is content,
  never truth. Per-addressee reliability at registration. The premise use
  is frame-local; durable writes stay the explicit `incorporate` path.
  The full §13 `think(next_sentence(ctx))` route still pends the decode
  round-trip.
- **Provisioning fix**: the observe-site conversation push lives ONLY in
  the SERIAL per-word body — MM_query_reasoning ran the parallel body and
  `provision_ltm` landed 0 rows. `<serial>true` fixes it (2 REL_PARTOF
  rows at the XML trust). Endpoint fidelity (the depth-3 NP/VP/NP split)
  still tracks parse quality: the untrained parse collapses each text to a
  depth-1 absolute row, so the provisioned chain is not yet climbable.
- **`_detect_query` post-relocation fix**: the query-capability signal now
  accepts a `<Queries>` declaration (`TheGrammar.query_ops`) — since the
  2026-07-05 relocation complete.grammar carries NO `query="true"` rules
  and detection had gone dead.
- **Grammar (b) RESOLVED**: default/shamatha relocated to the operator
  form (relations part/equal as assertive compose rules; is* in
  `<Queries>`; interrogative `query="true"` twins retired). Re-measured
  GREEN against the same MM_grammar XOR bar the 2026-07-05 full removal
  broke (test_mm_grammar_learns_xor_signal; baseline and post-relocation
  both converge on the early seed, ~24s).

## 6. Remaining seams

- The full §13 next-sentence route (`think(next_sentence(context))` with
  grammatical decode) — pends the decode round-trip (Track 1).
- Relation-endpoint parse fidelity for provisioned truths (the depth-3
  split) — pends the word-grain track; the kernel is ready for it.
