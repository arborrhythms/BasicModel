# Language

This document is synchronized to the code as of 2026-05-20. The source
of truth is `bin/Language.py`, `bin/embed.py`, `bin/typed_stack.py`,
`bin/stm_driver.py`, and `bin/parse_state.py`.

## Current Parser Surface

`WordSpace.compose()` and `WordSpace.generate()` are the public parser
entry points. `WordSpace.parser_backend` selects the implementation:

| backend | status | notes |
|---|---|---|
| `chart` | default | Compatibility path. Runs the existing chart / Viterbi machinery. |
| `stm` | active | Shift/reduce over `ConceptualSpace.stm_typed`; requires an attached `KnowledgeView`. |
| `parallel` | bridge | Builds the STM driver, then runs the chart path authoritatively. |

`WordSpace.routerKind` is separate from `parserBackend` and only affects
the chart backend:

| routerKind | status |
|---|---|
| `chart` | Default chart scorer / inside pass. |
| `signal` | Signal-router grammar path used by configs such as `XOR_grammar.xml`. |

The chart parser has not been deleted. It remains the default because
legacy configs can instantiate without a knowledge artifact, while STM
cannot.

## Retired XML Knobs

These XML fields are no longer read by the runtime and should not appear
in `data/*.xml`:

- `WordSpace.useGrammar`
- `WordSpace.chartCompose`
- `WordSpace.softChartCompose`
- `bivectorOutput`

Grammar mode is derived from the loaded grammar: default-only unary
`pi` / `sigma` rules derive `useGrammar == "none"` internally; any
non-default operator rule derives `useGrammar == "all"`.

## Grammar

`TheGrammar` is the singleton `Grammar` instance. Rules are loaded from
XML `<WordSpace><language><grammar>` blocks or from a configured grammar
CFG. A `RuleDef` stores:

```text
(tier, canonical, arity, method_name, lhs, rhs_symbols)
```

The grammar accepts explicit conceptual-order suffixes:

```text
NP3
NP4
VP1
MP1
S4
S5
```

The intended current style is explicit ordered rules, for example:

```text
S4 = lift(NP3, VP1)
S5 = lift(NP4, MP1)
```

Here all NPs share base category `NP`, but participate at different
conceptual orders. The parser carries stack frames as `(category, order)`,
so `NP3` and `NP4` both have `category == "NP"` but orders `3` and `4`.

Kleene order forms such as `NP*` are still parsed for compatibility, but
they are not required by the current grammar style.

## Knowledge Artifacts

`embed.build_knowledge_section(grammar)` creates the parser knowledge
section:

- `taxonomy`: base category refs plus explicit ordered refs.
- `reference_codebook`: scalar prototypes and `order` per ref.
- `typed_indexes`: `refs_by_category` and `refs_by_order`.
- `grammar.rule_order_signatures`: serialized rule typing.

For explicit ordered grammar, the taxonomy stores ordered refs as
children of their base category:

```text
NP
├── NP3
└── NP4
```

`KnowledgeView.category_of_ref(ref_id)` returns the base category
(`NP`), while `KnowledgeView.order_of_ref(ref_id)` returns the
conceptual order (`3`, `4`, etc.).

`WordSpace.category_codebook` has been retired. The live category
embedding is:

```python
WordSpace.category_embedding: nn.Embedding
```

## STM Shift/Reduce

The STM backend uses:

- `ConceptualSpace.stm_typed`: a `TypedStack` carrying payload,
  category, order, and ref id.
- `STMDriver`: shift/reduce coordinator.
- `RuleScorer`: MLP over top stack payloads.
- `embed.admissibility_mask`: hard category/order mask over grammar
  rule signatures.

SHIFT snaps each input vector to the nearest live reference, then pushes:

```text
payload, category, order, ref_id
```

REDUCE computes a typed admissibility mask, applies it to rule logits,
softmaxes over admissible rules, chooses the argmax for the current
greedy path, and records the chosen score/probability.

Typed admissibility is exact for explicit rules:

```text
S4 = lift(NP3, VP1)
```

fires only when the stack top matches:

```text
left:  category NP, order 3
right: category VP, order 1
```

## ParseState

STM writes parser-neutral state to `WordSpace.parse_state`:

- `frames`: leaves and reduce parents with payload, category, order, and span.
- `actions`: REDUCE operations with rule id, operand frame indices, parent frame,
  score, and probability.
- `row_traces`: per-row selected derivations.
- `current_rules` / `generate_rules`: per-tier rule selections for existing
  `SyntacticLayer` consumers.

New consumers should prefer `ParseState` or the parser-neutral accessors:

```python
wordSpace.parse_rules_for_tier(tier)
wordSpace.parse_derivation_trace()
```

Chart-only fields (`_chart_score`, `_chart_vec`, `_derivation_trace`) are
compatibility internals. Do not add new consumers of them.

## Syntax And SVO

`BasicModel.write_syntax_tree()` reads `ParseState` when available and
falls back to chart traces. STM also extracts SVO from `ParseState` for
the canonical pattern:

```text
S  = lift(NP, VP)
VP = intersection(V, O)
```

This avoids requiring `_chart_vec` for STM diagnostics.
