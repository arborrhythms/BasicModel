# Ternary Negation and Symbolic Syntax Handoff

## Goal

Implement a grammar-free concrete syntax over Percepts by making
affirmation, non-affirming negation, and not-negation first-class
polarity choices for each SymbolicSpace codebook concept.

The local layer primitive is already started:

- `bin/Layers.py::NegationLayer`
- default bivalent form: `L2(x) = [x, -x]`
- ternary form: `L3(x) = [x, non(x), -x]`

`non(x)` must be the existing ternary residual:

```python
Ops.non(x, monotonic=False) == 1.0 - abs(clamp(x, -1.0, 1.0))
```

For crisp signed concept values `x in {-1, 0, 1}`, exactly one channel is
positive:

| Value | Affirmation | Non-affirming | Not-negation |
|-------|-------------|----------------|--------------|
| `+1`  | `x`         |                |              |
| `0`   |             | `non(x)`       |              |
| `-1`  |             |                | `-x`         |

For soft values this is a differentiable relaxation. A later loss or
codebook constraint should harden the one-of-three partition if exact
exclusivity is required during training.

## Linguistic Interpretation

For each concept encoded in the SymbolicSpace codebook, pair the concept
with one of three surface forms:

| Polarity channel | Operation on concept | Surface syntax |
|------------------|----------------------|----------------|
| affirmation      | `x`                  | `""`           |
| non-affirming    | `non(x)`             | `"non-"`       |
| not-negation     | `-x`                 | `"not"`        |

The empty affirmation is absence of a polarity token, not necessarily a
literal empty-token vocabulary entry. Use a metadata flag or enum if the
lexer/reconstructor needs to distinguish "no prefix" from "unknown".

This gives the model a concrete Percept-level syntax:

- consume `concept` as affirmation
- consume `non-concept` as non-affirming negation
- consume `not concept` as not-negation
- emit the same forms during reconstruction/generation

The important point is that polarity is not a grammar production. It is
a codebook-percept relation: a concept plus a polarity channel.

## Recommended Implementation

### 1. Add a compositional wrapper

Add a small layer wrapper, for example `DNFConceptLayer`, rather than
placing bare `NegationLayer` directly in `ConceptualSpace`.

The wrapper should own:

- `self.neg = NegationLayer(input_dim, ternary=...)`
- `self.pi = PiLayer(multiplier * input_dim, output_dim, ...)`

Forward:

```python
literals = self.neg(x)
return self.pi(literals)
```

Reverse, when available:

```python
x = self.pi.reverse(y)
return self.neg.reverse(x)
```

This keeps the dimension expansion local and lets `ConceptualSpace`
continue to treat its encoder as one layer with `forward`, `reverse`, and
`getParameters`.

### 2. Gate it by config

Add config fields under `ConceptualSpace`:

```xml
<useNegationLayer>false</useNegationLayer>
<ternaryNegation>false</ternaryNegation>
```

Suggested behavior:

- `useNegationLayer=false`: current path, unchanged.
- `useNegationLayer=true`, `ternaryNegation=false`: Pi input width doubles.
- `useNegationLayer=true`, `ternaryNegation=true`: Pi input width triples.

Update `data/model.xsd` and default `data/model.xml`. Keep defaults false
so existing models and checkpoints do not silently change shape.

### 3. Wire ConceptualSpace

In `bin/Spaces.py::ConceptualSpace.__init__`, when no butterfly `layer`
is provided and `useNegationLayer` is true:

- build `DNFConceptLayer` instead of a bare `PiLayer`
- preserve `invertible`, `reversible`, `nonlinear`, `naive`, `ergodic`,
  and `monotonic` behavior as much as the wrapper supports
- ensure `self.layers` contains the wrapper and parameters include the
  inner PiLayer parameters

Do not wire this into butterfly mode until there is a separate shape plan
for pair packing.

### 4. Add polarity metadata for the symbolic codebook

Represent each codebook concept as a base concept plus a polarity enum:

```python
AFFIRM = 0
NON = 1
NOT = 2
```

Avoid physically tripling every stored vector unless the current codebook
APIs require it. Prefer metadata or views over duplication:

- base concept id
- polarity id
- surface form
- optional token span used to consume/emit the polarity marker

### 5. Consume polarity from Percepts

Extend the text/percept path to recognize polarity markers:

- `foo` -> `(concept=foo, polarity=AFFIRM)`
- `non-foo` -> `(concept=foo, polarity=NON)`
- `not foo` -> `(concept=foo, polarity=NOT)`

The `not` form is a separate token/word-level operator; `non-` is a
prefixal marker. Keep both forms distinct because their semantics differ:

- `not` asserts the opposite.
- `non-` withdraws assertion or affirms indeterminacy.

### 6. Emit polarity to Percepts

During reconstruction/generation from a symbolic concept:

- AFFIRM emits no polarity marker
- NON emits `non-` plus the concept surface
- NOT emits `not` plus the concept surface

The emission decision should come from the active polarity channel, not a
grammar rule.

## Tests

Add or update tests for:

1. `NegationLayer(ternary=False)` returns `[x, -x]` and reverses exactly
   on its own output.
2. `NegationLayer(ternary=True)` returns `[x, Ops.non(x), -x]`.
3. Crisp ternary inputs `[-1, 0, 1]` have exactly one positive channel.
4. `ConceptualSpace` with `useNegationLayer=true` constructs the expected
   Pi input width (`2 * input_dim` or `3 * input_dim`).
5. `conceptualOrder=1`, `useGrammar=none`, and the negation wrapper run a
   forward pass without invoking `SyntacticLayer`.
6. A toy Percept/codebook roundtrip:
   - `foo` -> AFFIRM -> `foo`
   - `non-foo` -> NON -> `non-foo`
   - `not foo` -> NOT -> `not foo`

## Non-goals

- Do not reintroduce `chunk()` as a grammar op.
- Do not make `identity()` a grammar rule.
- Do not encode `non-` as `not not`; it is a distinct non-affirming
  polarity.
- Do not change default model shapes unless the new config flag is set.

## Risks

- `non(x)` is lossy. `NegationLayer.reverse()` should reconstruct from
  the affirmation/not pair and ignore the non channel.
- Soft values are not exactly one-hot across the three channels. If exact
  exclusivity is needed, add a loss or codebook projection.
- Existing checkpoints will not load into a widened PiLayer. Keep the
  feature off by default and require new checkpoints for the DNF stack.
