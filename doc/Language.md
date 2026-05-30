# Language

This document is synchronized to the code as of 2026-05-29. The source
of truth is `bin/Language.py` and `bin/embed.py`. (The earlier
`bin/typed_stack.py`, `bin/stm_driver.py`, and `bin/parse_state.py`
modules were retired in the 2026-05-21 / 2026-05-29 refactors; their
functionality folded into `bin/Layers.py` and `bin/Language.py`.)

> **2026-05-29 deltas:**
>
> - `unreduce()` passes the tier-local Basis (Codebook) to binary
>   GrammarLayer reverses as `basis=tier_basis` (replacing the prior
>   raw-`W` form). `UnionLayer.reverse` / `IntersectionLayer.reverse`
>   extract `W = basis.getW()` internally and dispatch to
>   `Ops.disjunctionReverse` / `Ops.conjunctionReverse`. Layers that
>   don't accept the `basis` kwarg yet are handled by a `TypeError`
>   fallback. No back-ref is stored on the layer.
> - `MetaLayer` was renamed to `SymbolizeLayer` (no semantic change).
> - Word-mode parse appends a `\x00` null sentinel after the words
>   slab for explicit end-of-sequence on the forward path.
> - See [doc/plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

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

> **SS-analysis vs CS-execution.** `WordSubSpace.compose` is the
> SS-side *analysis* stage (it selects the per-tier hard rule dict
> `current_rules`); the CS-side *execution* (applying lift / lower /
> union / intersection / swap / quantize / not to the concept tensors)
> runs in `ConceptualSpace.forward` and the per-tier `SyntacticLayer`
> cursors. This split is a clean code boundary **only on the
> default-only path**: on the full-router path
> `LanguageLayer.compose` does both selection and tensor reduction, and
> the per-tier cursors are deliberately bypassed
> (`not _grammar_is_default_only`). See
> [STM.md Section 5](STM.md#5-routing-parser-ss-analysis-vs-cs-execution)
> for the accurate, audited account.

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
|-- NP3
`-- NP4
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

## Syntax And SVO

`BasicModel.write_syntax_tree()` (in `bin/Models.py`) emits an XML
syntax-tree dump per batch row when configured with a syntax output
path. It reads the chart's per-row derivation trace and the symbolic
codebook's per-atom category tags; see the function docstring for the
exact format. SVO extraction is performed by the STM path on the
canonical subject lift over a transitive verb-phrase derivation; the
object role is the NP operand inside that verb-phrase derivation, not a
separate semantic category.
