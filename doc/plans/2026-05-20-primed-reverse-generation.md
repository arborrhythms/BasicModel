# Primed reverse generation with weighted activation masks

## Summary

Reverse grammatical operations should remain formally constrained by grammar,
word category, conceptual order, and mereological feasibility, but they should
also gain a psychologically realistic priming signal. The implementation
should build a weighted mask on `.activation`: a hard admissibility mask from
grammar/category/order multiplied by a soft part/whole priming mask.

Priming must bias selection only within the formally admissible candidate set.
It must not license a candidate that the grammar or order typing rejects.

The central rule is:

```text
final_activation_mask =
    hard_grammar_category_order_mask
    * priming_part_whole_mask
```

## Hard admissibility mask

The hard mask is formal. A candidate receives `0` unless it satisfies all
constraints required by the active reverse operation:

- grammar category compatibility for the target slot;
- conceptual order compatibility from the order-typed grammar rule;
- word/category membership when the reverse path predicts lexical items;
- operation compatibility for the active grammatical operation;
- existing formal mereological feasibility constraints.

This mask is binary or effectively binary. It should be applied before
candidate selection so inadmissible candidates cannot be rescued by priming or
by a learned predictor.

For lift/lower, the mask must use the grammar-derived order transition:

```text
LIFT  -> target/parent order follows the rule's +1 typing
LOWER -> target/parent order follows the rule's -1 typing
```

For all other grammatical operations, order is preserved unless the operation
registry explicitly marks the operation as order-changing.

## Part/whole priming mask

The priming mask is soft and transient. It represents spreading activation
over the sparse mereological graph of recently active symbols.

Maintain a per-order priming tensor shaped like the active reference
population at that order:

```text
priming_mask[k]: [V_ref_at_order_k]
```

At each order/iteration:

1. Recently active symbols are set to `1.0`.
2. Activation propagates upward to immediate wholes.
3. Activation propagates downward to immediate parts.
4. Propagated activation is multiplied by a decay factor.
5. Propagation repeats only for the configured priming depth.

Siblings are not directly primed. They can become active only as a
second-order effect through part/whole propagation, for example:

```text
A -> whole(A) -> sibling_part
```

There should be no explicit sibling edge in the priming graph.

Do not introduce a dense `V x V` connectionist matrix. Store propagation as
sparse part/whole adjacency, preferably CSR or top-k adjacency derived from
the codebook's mereological structure.

## Reverse operation flow

For each reverse grammatical operation:

1. Determine the target slot being predicted: left child, right child, or
   lexical realization.
2. Build the hard mask for that target slot from the active rule's category
   and order typing.
3. Read or update the priming mask for the target conceptual order.
4. Multiply the hard mask and priming mask into `.activation`.
5. Run existing candidate selection, codebook snapping, inverse
   recommendation, or reverse generation over the constrained activation.
6. Return the selected candidate or top-k candidates according to the caller.

The invariant is:

```text
hard_mask == 0  =>  final_activation_mask == 0
```

Priming changes ranking inside the admissible set. It does not change the
set of admissible candidates.

## Optional learned predictor

A later version may add an operation-conditioned MLP reranker over the
constrained candidate slice. This is the maximal version of the design, not
the first implementation step.

The MLP should score only candidates that survive the hard mask. Useful
features include:

```text
parent activation
operation id
target slot
candidate prototype
candidate category
candidate order
priming score
mereological features
STM context summary
```

For binary reverse, avoid full `O(V^2)` pair scoring initially. Use a
factorized top-k strategy:

```text
score left candidates
score right candidates
pair-rerank left_top_k x right_top_k
```

This preserves formal admissibility while allowing learned word prediction
inside the constrained region of the codebook.

## Tests

### Hard mask

- Inadmissible category candidates remain zero even when highly primed.
- Inadmissible order candidates remain zero even when highly primed.
- `LIFT` and `LOWER` reverse paths expose only grammar-compatible orders.
- With priming disabled or all-ones priming, reverse generation matches the
  current formal behavior.

### Priming propagation

- Active symbols receive priming score `1.0`.
- Immediate parts and wholes receive decayed nonzero activation.
- Siblings receive no activation after one propagation step.
- Siblings may receive activation only after two or more propagation steps
  through a shared part/whole path.
- Unrelated symbols remain zero or near zero after the configured propagation
  depth.

### Combined mask

- `final_mask = hard_mask * priming_mask`.
- Priming changes candidate ranking inside the admissible set.
- Priming never introduces candidates outside the admissible set.
- Existing inverse recommender tests continue to pass when no priming mask is
  supplied.

## Assumptions

- Grammar and word category define formal admissibility.
- Conceptual order typing comes from the grammar and operation registry.
- Mereological part/whole relations are sparse enough for CSR or top-k
  propagation.
- `.activation` is the correct place to apply the final weighted candidate
  mask.
- Priming is transient working-memory state, not a persistent learned
  taxonomy.
- Learned MLP prediction is optional and should be added only after sparse
  priming works.
