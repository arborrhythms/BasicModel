# Truth-Grounded Reasoning For Queries

## Goal

Given a query, find a path from known internal knowledge or LTM knowledge to an
end state that either proves or disproves the query.

The output should be:

- `TRUE`: a trusted path supports the query.
- `FALSE`: a trusted path supports the negation or contradiction of the query.
- `BOTH`: trusted support exists for both true and false.
- `NEITHER`: no sufficient path is found.

## Core Idea

Reasoning is graph search over typed idea states.

```text
known True/False states
        |
        v
reasoning expansions
        |
        v
candidate derived states
        |
        v
match against query target
        |
        v
TRUE / FALSE / BOTH / NEITHER
```

## State Model

Represent each reasoning node as:

```python
ReasonState:
    idea: Tensor
    relation: Optional[predicate, left, right]
    polarity: true | false
    trust: float
    source: internal | ltm | derived
    trace: list[ReasonStep]
```

A state is admissible only if it comes from:

- an internal truth known above threshold
- an LTM relation with positive or negative trust
- a derived state whose trace is built from admissible states

## Query Model

Parse each query into a `QuerySpec`:

```python
QuerySpec:
    predicate: exist | isPart | isEqual | part | equal | query
    left: Optional[idea]
    right: Optional[idea]
    variables: list
    desired_polarity: true
```

Examples:

```text
"is A part of B?"      -> isPart(A, B)
"is A not part of B?"  -> NOT isPart(A, B)
"what is part of B?"   -> isPart(X, B)
"does X exist?"        -> exist(X)
```

## Knowledge Sources

### Internal Knowledge

Use local/implicit stores first:

- WS META taxonomy for reducible relations
- absolute TruthLayer for known proposition-like truths
- concept codebook / category commitments
- current STM context as temporary givens

### LTM Knowledge

Use persistent relation rows:

```text
(predicate, idea1, idea2, trust)
```

Positive trust means support. Negative trust means disproof or counter-support.
Rows below the trust threshold are ignored except as weak candidates.

## Reasoning Search

### Forward Search

Use when givens are supplied.

```text
givens -> consequents() -> derived states -> query match
```

For each known state:

1. Find LTM/internal rules whose antecedent matches it.
2. Apply the consequent.
3. Compose trust:

```text
derived_trust = premise_trust * relation_trust * match_score
```

4. Stop if the derived state matches the query target.

### Backward Search

Use when the query gives a target but no givens.

```text
query target -> possible supports -> known True/False starts
```

For the target:

1. Search LTM for relations that could imply it.
2. Work backward to required antecedents.
3. Check whether those antecedents are known true or false.
4. If found, return a proof path.

### Bidirectional Search

Preferred default.

```text
known frontier <--> target frontier
```

- Forward frontier starts from known true/false internal and LTM states.
- Backward frontier starts from the query target.
- Success occurs when the two frontiers meet by parthood/equality match.

## Matching

A reasoning state proves a query when:

```text
predicate matches
left/right match by equality or parthood
polarity matches
trust >= threshold
```

A reasoning state disproves a query when:

```text
predicate matches
left/right match
polarity contradicts query
abs(trust) >= threshold
```

For open-variable queries, matching binds variables:

```text
isPart(X, B) -> X = candidate_left
```

## Truth Evaluation

Collect all terminal matches:

```python
support_true = max positive proof trust
support_false = max negative/counter proof trust
```

Then classify:

```text
TRUE     if support_true  >= threshold and support_false < threshold
FALSE    if support_false >= threshold and support_true  < threshold
BOTH     if both exceed threshold
NEITHER  if neither exceeds threshold
```

## Returned Answer

Return both the answer and the proof trace:

```python
ReasonResult:
    posture: TRUE | FALSE | BOTH | NEITHER
    confidence: float
    answer_bindings: dict
    proof_trace: list[ReasonStep]
    counter_trace: optional list[ReasonStep]
```

Each `ReasonStep` should record:

```python
ReasonStep:
    rule_or_relation_id
    input_state_ids
    output_state
    match_score
    trust_before
    trust_after
    explanation
```

## Implementation Phases

### Phase 1: Query Framing

Build `QuerySpec` from parsed grammar/query operators.

Initial supported query types:

- `exist(X)`
- `isPart(A, B)`
- `isEqual(A, B)`
- `part(A, B)`
- open slot: `isPart(X, B)`

### Phase 2: Known-State Loader

Create a unified iterator over known true/false states from:

- absolute TruthLayer
- RelativeTruthStore
- unified `ltm_store`
- WS META taxonomy
- optional STM givens

### Phase 3: Proof Search

Implement `prove(query, givens=None, max_steps=N)`:

- direct lookup first
- forward search from givens / known states
- backward search from target
- bidirectional meet when possible

### Phase 4: Disproof Search

Run the same search against:

- negated query
- contradictory stored relation
- negative-trust matching relation

### Phase 5: Trace Rendering

Return a compact explanation path:

```text
A is known true.
A implies B with trust 0.82.
B matches query target with score 0.91.
Therefore query is TRUE with confidence 0.75.
```

### Phase 6: Training Hook

Add answer loss:

```text
answer correctness loss
+ contradiction penalty against trusted truths
+ path sparsity / shorter-proof preference
```

## Success Criteria

- A direct known true relation proves the query.
- A direct known false relation disproves the query.
- A multi-step LTM chain can prove a query.
- A multi-step counter-chain can disprove a query.
- Unknown queries return `NEITHER`, not hallucinated answers.
- Conflicting trusted paths return `BOTH` with both traces.
