# Shamatha Speech Contiguity Handoff

## Goal

Implement Shamatha Speech as a first-class XML mode for one-pointed
object speech.

The mode should produce a complete DNF description of a single object
while enforcing Dakpo Tashi Namgyel's single-pointedness requirement as a
computable spatiotemporal contiguity constraint:

- logical completeness comes from DNF
- objecthood comes from connected `where()` and continuous `when()`
- sentencehood comes from rendering the DNF as narrow but valid English

This differs from serial mode.  Serial mode constrains the runtime cursor
and next-token task.  Shamatha Speech may conjoin and disjoin over all
active percepts in the current percept field at once; it only rejects
logical merges that would make the object spatially scattered or
temporally discontinuous.

## XML Mode

The code currently has:

- `WordSpace.useGrammar in {"all", "thoughtFree", "none"}`
- a per-request `thought_free` flag in `serve.py`
- comments describing "ShamathaSpeech mode"

Target behavior:

```xml
<WordSpace>
  <useGrammar>shamathaSpeech</useGrammar>
  <language>
    <grammarCfg>data/grammar_shamatha.cfg</grammarCfg>
  </language>
</WordSpace>
```

Implementation options:

1. Extend `useGrammar` with a fourth value: `shamathaSpeech`.
2. Keep `useGrammar` for architecture selection and add
   `<speechMode>shamathaSpeech</speechMode>`.

Prefer option 1 if the mode needs to alter the model path; prefer option
2 if it is only a grammar/generation policy layered on the ordinary
Sigma-Pi path.  In either case keep `thoughtFree` as a legacy alias and
make the docs explicit that the target name is Shamatha Speech.

Required config edits if option 1 is chosen:

- `bin/util.py`: add `"shamathaSpeech"` to `_VALID_USE_GRAMMAR`
- `data/model.xsd`: add the enum value under `useGrammarType`
- `data/model.xml`: default remains `none`; no default shape change
- tests: update `test/test_usegrammar_tristate.py`

Do not reuse `useGrammar="all"`: the full constituency grammar is too
broad for one-pointed speech.

## Narrow Grammar

Shamatha grammar should be only the DNF object grammar:

```ebnf
literal  := polarity? concept
polarity := "" | "non-" | "not"
term     := literal ("and" literal)*
object   := term ("or" term)*
sentence := object
```

Allowed operations:

- `project(concept)` / codebook lookup
- polarity selection: affirmation, `non-`, `not`
- `conjunction(A, B)`
- `disjunction(A, B)`
- `where(A)`, `when(A)` for contiguity validation
- `absorb(A, B)` only for punctuation/whitespace sugar around the DNF

Excluded operations:

- full typed constituency rules (`NP`, `VP`, `VO`, etc.)
- `lift` / `lower` as sentence-level grammar moves
- `swap`
- `equals`, `part`, `query` as surface grammar productions
- `identity`
- `chunk` as a grammar op

The generated English can be ugly:

```text
red and round and not blue or red and non-moving and round
```

That is acceptable.  The goal is a faithful DNF surface over the object
set, not idiomatic prose.

## DNF as Mereological Object Spec

Read the DNF as a complete object specification relative to the
SymbolicSpace codebook:

```text
O =
  (literal_1 and literal_2 and ...)
  or
  (literal_3 and literal_4 and ...)
```

Interpretation:

- literal = commitment about a candidate part/property
- conjunction = complete local part-description
- disjunction = fusion / extension over admitted local descriptions
- canonical DNF = complete specification relative to the chosen codebook

This is only a single object if the accepted descriptions are connected.
Without contiguity, the DNF denotes a class, collection, or scattered
aggregate.

## Contiguity Contract

Each logical operator must validate the operands' spatiotemporal support
before producing a merged object.

For every accepted merge `M = op(A, B)`:

1. Spatial support:
   - compute `WA = where(A)` and `WB = where(B)`
   - accept if regions overlap, touch, or are joined by an allowed
     adjacency edge
   - reject or downweight if they are disjoint

2. Temporal support:
   - compute `TA = when(A)` and `TB = when(B)`
   - accept if intervals overlap or are adjacent within tolerance
   - reject or downweight if they form discontinuous tracks

3. Object graph:
   - add `A` and `B` as nodes under object `M`
   - add an edge recording adjacency/continuity evidence
   - maintain connected-component id for the object

4. Result support:
   - `where(M) = union_region(WA, WB)` when spatially connected
   - `when(M) = hull_or_track(TA, TB)` when temporally continuous

At sentence completion, validate:

```python
single_object = (
    one_connected_component(where_graph)
    and one_continuous_track(when_graph)
    and no_disallowed_gaps(object_parts)
)
```

## Operator Semantics

### Conjunction

`A and B` means both commitments are part of the same object.

It is legal only if `A` and `B` are spatiotemporally compatible.  The
output inherits the fused support region.

### Disjunction

`A or B` in ordinary logic can mean alternatives.  In Shamatha Speech it
also functions as DNF fusion over admitted local descriptions.  The OR is
legal only if the terms it joins belong to the same connected object
field or are alternatives over the same object support.

Implementation should distinguish:

- **fusion OR**: joins adjacent local DNF terms into one object's
  extension
- **alternative OR**: keeps alternative complete descriptions of the
  same support

Both are allowed.  OR over unrelated supports is rejected.

### Negation and Non-Affirmation

Use `NegationLayer(ternary=True)` as the polarity model:

```python
[x, non(x), -x]
```

Surface mapping:

- `x` -> no marker, e.g. `red`
- `non(x)` -> prefix marker, e.g. `non-red`
- `-x` -> word marker, e.g. `not red`

Only one polarity channel should be true for a crisp concept at a time.
Soft training values are allowed but should be regularized toward the
one-of-three partition.

## Suggested Data Structures

Add an object-support record carried alongside activations during
Shamatha composition:

```python
@dataclass
class ObjectSupport:
    where_region: Tensor
    when_region: Tensor
    component_id: int
    part_ids: tuple[int, ...]
    polarity: Optional[int] = None
```

Do not force this into the tensor if the existing `where` / `when`
columns already carry the coordinates.  A sidecar structure is easier to
test and keeps the Sigma-Pi stack unchanged.

## Generation

Generation should produce a DNF sentence from the object graph:

1. gather connected object parts
2. group literals into conjunction terms
3. order terms by `where()` adjacency, then `when()` continuity
4. emit:
   - `literal`
   - `literal and literal`
   - `term or term`
5. absorb punctuation/whitespace as syntactic sugar only

Example rendering:

```text
red and round and not blue or red and round and non-moving
```

This is deliberately constrained.  Do not invoke SVO, VP, NP, or learned
head-emission grammar unless a later surface-realization pass is added
outside Shamatha Speech.

## Implementation Steps

1. Add XML mode / alias.
   - Update parser, XSD, default config, and tests.

2. Add `data/grammar_shamatha.cfg`.
   - Only DNF object rules and sugar absorption.
   - No full constituency grammar.

3. Add a `ShamathaComposer` or `ShamathaPolicy`.
   - It can live in `Language.py` if it is grammar-adjacent.
   - It can live in a new module if the sidecar object graph grows.

4. Add contiguity checks.
   - Implement `where_connected(A, B)` and `when_continuous(A, B)`.
   - Start with tensor interval / bounding-box checks.
   - Later replace with learned adjacency over the symbolic codebook.

5. Wire composition.
   - Before applying conjunction/disjunction, call the contiguity policy.
   - Illegal merges should either be masked out before rule selection or
     receive a large negative logit.

6. Wire generation.
   - Build DNF from accepted object graph.
   - Emit polarity markers from the ternary channels.

7. Keep serial mode separate.
   - Do not route through AR cursor-only logic.
   - Shamatha Speech sees the whole current percept field.

## Tests

Add tests for:

1. XML accepts `shamathaSpeech` and legacy `thoughtFree`.
2. `shamathaSpeech` rejects full grammar ops such as `swap`, `part`,
   `equals`, `query`, `lift`, and `lower` in its grammar cfg.
3. Conjunction of adjacent `where()` supports succeeds.
4. Conjunction of disjoint `where()` supports is masked/rejected.
5. Disjunction over same connected object support succeeds.
6. Disjunction over unrelated supports is masked/rejected.
7. Temporally adjacent supports within tolerance succeed.
8. Temporally discontinuous supports fail.
9. A complete DNF object emits a sentence containing literals joined by
   `and` / `or`.
10. The mode does not use serial cursor constraints and can compose over
    all active percepts.

## Acceptance Criteria

- A model configured for Shamatha Speech can form a DNF sentence from a
  connected object field.
- The same logical content is rejected or downweighted when its parts are
  spatially disconnected or temporally discontinuous.
- The mode does not invoke the full constituency grammar.
- The mode does not collapse to serial next-token behavior.
- Generated sentences are faithful DNF English over the object set.

## Risks

- If contiguity is enforced too hard during early training, valid object
  parts may never merge.  Start with a differentiable penalty or rule
  logit mask controlled by a temperature.
- OR is ambiguous between object fusion and logical alternatives.  Track
  this in the object graph rather than trying to infer it from text alone.
- Existing `thoughtFree` behavior is not enough.  Treat it as a legacy
  alias only, not as the completed Shamatha Speech implementation.
