# Spec: SubSpace.what STM + Symbolic SignalRouter Refactor

This is a handoff spec for refactoring BasicModel's syntactic runtime so the
syntactic tree is the tensor passed through `Space.forward()`, not a side chart,
`WordSpace.current_rules`, or a CS-owned STM buffer.

Read this entire file before editing. The key architectural constraint is:

```text
Do not add new stack fields.
The live STM stack is the forwarded SubSpace.what tensor.
```

The target design reuses existing `SubSpace` modalities:

```text
SubSpace.what        live STM frontier payloads
SubSpace.where       codebook locations for the payloads
SubSpace.activation  structural occupancy mask for live stack slots
SubSpace.event       compatibility/materialized view, not parser state
```

The reduction loop lives in `SymbolicSpace`, because reductions are grammatical
operations. `SymbolicSpace` owns the `SignalRouter`, the S-tier
`SyntacticLayer`, the terminal symbol codebook, and a new grammatical operation
codebook. `WordSpace` becomes a static grammar/lexicon registry, not the live
parser state carrier.

## Current Problem

The current grammar runtime is split across several surfaces:

- `WordSpace` owns chart/router surfaces and stores `current_rules` /
  `generate_rules`.
- `SyntacticLayer` reads those rule lists through a cursor.
- `SignalRouter` exists as a chart alternative owned by `Chart`.
- The per-word path has a separate `ConceptualSpace.stm` side buffer and
  `BasicModel._stm_bounded_reduce_step()`.

That gives the model multiple representations of "the parse":

```text
WordSpace.current_rules
Chart / SignalRouter cached routing
ConceptualSpace.stm._buffer
SubSpace.event / .what / .activation
SyntacticLayer cursors
```

The target design collapses the live parser state into one object:

```text
the SubSpace passed through .forward()
```

That subspace's `.what` tensor is the bounded STM stack. The unrolled CS/SS loop
is therefore the syntactic tree itself: each `SHIFT` or `REDUCE` rewrites rows
of `.what`, `.where`, and `.activation`.

## Target Data Contract

For stack-mode subspaces:

```text
subspace.what:       [B, K, D]
subspace.where:      [B, K, W] or [B, K, 1]
subspace.activation: [B, K]
```

Where:

- `B` is batch.
- `K` is fixed maximum STM capacity.
- `D` is the payload width used for terminals and reduced ideas.
- `W` is the existing location encoding width.

Interpretation:

```text
.what[b, k]        payload for stack slot k
.where[b, k]       codebook location for that payload
.activation[b, k]  occupancy: 1/live, 0/empty, optionally soft in the future
```

Do not create `stack_c`, `stack_s`, `stack_depth`, `stack_valid`, or similar new
fields. If code needs helpers, add methods or utility functions that read and
write the existing `.what`, `.where`, and `.activation` slots.

### Terminals

A terminal is created by quantizing a continuous concept against the
`SymbolicSpace` terminal symbol codebook.

```text
terminal.what   = snapped or STE-snapped symbol vector
terminal.where  = codebook location for the matched terminal symbol
terminal.active = 1
```

Terminals are long-term matches: they correspond to rows in the symbol codebook.

### Reductions

A reduction combines live STM slots by a grammatical rule.

```text
left  = subspace.what[:, i, :]
right = subspace.what[:, j, :]
rule  = selected grammar operation

parent.what  = rule.operation(left, right)
parent.where = codebook location for the selected grammar rule
parent.active = 1

consumed slot .what = 0
consumed slot .where = 0
consumed slot .activation = 0
```

Reductions create short-term ideas. The parent `.what` is not copied from a
long-term codebook. It is computed by the operation on child arguments. The
associated `.where` records which grammatical operation produced it.

## Important Semantic Changes

### `.what` Becomes The STM Stack

In stack-mode, `.what` is not merely "demuxed content" in the old sense. It is
the live bounded syntactic frontier.

This is acceptable because `SubSpace` already treats `.what` as the content
modality, and `WordSubSpace` already uses inherited `.what/.where/.when` storage
as a stack-like buffer. The new design generalizes that idea to the forwarded
working subspace.

### `.where` Is Only Codebook Location

Do not use `.where` for byte offsets, spans, source positions, or derivation
step counters in this stack-mode runtime.

Use `.where` only as a codebook-location annotation:

```text
terminal slot  -> location in terminal symbol codebook
reduced slot   -> location in grammatical operation codebook
empty slot     -> zero location
```

Because terminals and rules live in different codebooks, there must be a stable
namespace rule. Recommended:

```text
0                           empty
1..V_sym                    terminal symbol locations
V_sym+1..V_sym+R_rule       grammar rule locations
```

If the implementation uses vector encodings rather than integer IDs in
`.where`, keep the same namespace logically and provide encode/decode helpers.
The live runtime should not require Python-side decoding to perform the current
step; the router already knows the selected rule for that step. Decoding `.where`
is for reverse, diagnostics, and tests.

### `.activation` Is Occupancy

In stack-mode, `.activation` is structural occupancy, not semantic truth,
confidence, or catuskoti state.

```text
activation[b, k] = 1 means stack slot k is live
activation[b, k] = 0 means stack slot k is empty/padding
```

This dovetails with existing materialization behavior: materialized views are
gated by activation, so dead stack slots zero out.

Semantic symbol activation remains owned by `SymbolicSpace` and its codebooks.
Do not read a stack-mode subspace's `.activation` as Degree of Truth or symbol
belief.

### `.event` Is Not The Parser State

In this spec, `.what` is the STM. `.event` must not become a second parser
state.

For compatibility, `.event` may remain a cached mux/materialized view. It may
also be set from `.what/.where` for old call sites. But the canonical parser
state is:

```text
subspace.what + subspace.where + subspace.activation
```

If existing `SubSpace` behavior invalidates `.event` when `.what` changes, that
is fine. Do not rely on `.event` for the syntactic frontier.

## Ownership Rules

Spaces hold layers. Layers do not own spaces.

Target ownership:

```text
SymbolicSpace
  owns terminal symbol codebook
  owns grammar operation codebook
  owns S-tier SyntacticLayer executor
  owns SignalRouter

ConceptualSpace
  owns concept transforms/layers
  does not own live parser STM side state

WordSpace
  owns or exposes static grammar registry data only
  does not own live parser state
```

`SignalRouter` may receive non-owning references or callable executor handles
from `SymbolicSpace`, but it must not register `SymbolicSpace` or
`ConceptualSpace` as child modules. Avoid module cycles.

## WordSpace Refactor

### Current Role

`WordSpace` currently does too much:

- grammar XML parsing
- chart/router ownership
- `current_rules` and `generate_rules`
- `WordSubSpace` stack buffer
- category/reconstruction stacks
- host-layer registry
- cursor-driven dispatch support

### Target Role

Refactor `WordSpace` into a static grammar/lexicon service.

Keep:

```text
TheGrammar / RuleDef access
rule_id -> method_name
rule_id -> arity
rule_id -> tier
rule_id -> canonical
rule_id -> operation-codebook location
terminal vocabulary helpers if still needed
compatibility wrappers during migration
```

Remove from the live parser path:

```text
current_rules as parser state
generate_rules as parser state
SyntacticLayer cursor source
Chart as active parser
SignalRouter ownership
WordSubSpace as active STM
category_stack as active parser state
reconstruction_stack as active parser state
```

Compatibility can remain temporarily. The live forward path should not depend on
`WordSpace.current_rules` or `_compose_generation`.

### Suggested New Interface

Create a small registry-style API on `WordSpace` or a new helper object:

```python
class GrammarRegistry:
    def num_rules(self) -> int: ...
    def rule(self, rule_id: int) -> RuleDef: ...
    def rules_for_tier(self, tier: str, arity: int | None = None) -> list[int]: ...
    def method_name(self, rule_id: int) -> str | None: ...
    def arity(self, rule_id: int) -> int: ...
    def where_id_for_rule(self, rule_id: int) -> int: ...
    def where_id_for_symbol(self, symbol_id: int) -> int: ...
```

Do not make this registry the live parser. It is a lookup table.

## SyntacticLayer Refactor

### Current Role

`SyntacticLayer` currently acts like a cursor-driven dispatcher:

```text
read word_space.current_rules[tier]
advance cursor
look up layer by method name
apply one unary fold if possible
```

That is the wrong shape for the target runtime. The selected rule is known at
the time of reduction, and the stack is directly rewritten.

### Target Role

`SyntacticLayer` becomes a pure executor.

Hard execution:

```python
def execute(self, rule_id: int, left: torch.Tensor,
            right: torch.Tensor | None = None) -> torch.Tensor:
    ...
```

Superposed execution:

```python
def execute_superposed(self, rule_weights: torch.Tensor,
                       left: torch.Tensor,
                       right: torch.Tensor | None = None,
                       rule_ids: torch.Tensor | list[int] | None = None
                       ) -> torch.Tensor:
    ...
```

The superposed version must follow this rule:

```text
Each candidate op computes independently on its own slot.
Combine once by weighted sum over the fixed op axis.
Do not use a shared in-place accumulator that one op mutates before the next.
```

Pseudo-code:

```python
outs = []
for local_idx, rule_id in enumerate(rule_ids):
    out = self.execute(rule_id, left, right)
    outs.append(out)
stacked = torch.stack(outs, dim=-2)  # [..., R, D]
parent = (stacked * rule_weights.unsqueeze(-1)).sum(dim=-2)
```

Implementation can optimize this later, but preserve the independent
contribution semantics.

### Where The Operation Lives

For this refactor, reductions happen in `SymbolicSpace`. The S-tier
`SyntacticLayer` should execute the grammatical operations used by the router.

If any old C-tier rule remains necessary, do not resurrect WordSpace cursors.
Either:

1. move the rule into the S-tier grammar for this runtime, or
2. expose a narrow C executor callable and let the SymbolicSpace-owned router
   call it without owning the C space.

Prefer option 1 for the first implementation unless a test proves it is
insufficient.

## SignalRouter Refactor

### Current Role

`SignalRouter` is currently a `Chart` alternative. It can run unary and binary
structured layers and return rule ids in a `current_rules`-style dictionary.

That should become a compatibility/test path, not the live parser design.

### Target Role

`SignalRouter` becomes a SymbolicSpace-owned routing layer over the forwarded
subspace's STM tensor.

Target call shape:

```python
subspace = self.signalRouter.forward(
    subspace=subspace,
    syntactic_layer=self.syntacticLayer,
    grammar=self.grammar_registry,
    terminal_codebook=self.subspace.what,
    rule_codebook=self.rule_codebook,
)
```

The exact arguments can vary, but the ownership should not:

```text
SymbolicSpace owns SignalRouter.
SignalRouter owns scoring parameters.
SignalRouter does not own runtime STM state.
Runtime STM state is subspace.what / where / activation.
```

### Router Responsibilities

The router does the following:

```text
1. Read stack payloads from subspace.what.
2. Read occupancy from subspace.activation.
3. Score legal SHIFT / REDUCE / COPY transitions.
4. Select or softly select a grammatical operation.
5. Call SyntacticLayer to compute parent.what.
6. Write parent.what into the surviving slot.
7. Write rule location into parent.where.
8. Zero consumed slots.
9. Update occupancy.
10. Return the same subspace object or a correctly context-copied subspace.
```

The router should reuse as much as possible from:

- `BinaryStructuredReductionLayer`
- `binary_tiling_soft_dp`
- `binary_tiling_viterbi`
- existing straight-through hard/soft routing patterns

But the final write target is no longer `ConceptualSpace.stm._buffer`. It is:

```text
subspace.what
subspace.where
subspace.activation
```

### SHIFT

`SHIFT` turns a continuous incoming concept into a terminal stack item.

Pseudo-flow:

```text
input concept candidate
  -> quantize / nearest / STE against SymbolicSpace terminal symbol codebook
  -> terminal.what
  -> terminal.where = symbol codebook location
  -> write into next empty stack slot
  -> activation = 1
```

For the first implementation, the source of "input concept candidate" can be
the same tensor currently passed into `SymbolicSpace.forward(CS_subspaceForSS)`.
Do not introduce a separate producer object.

### REDUCE

`REDUCE` combines two live stack slots.

Hard pseudo-code:

```python
what = subspace.materialize(mode="what")       # [B, K, D]
occ = subspace.materialize(mode="activation")  # [B, K]
where = subspace.materialize(mode="where")     # [B, K, W]

left = what[:, i, :]
right = what[:, j, :]

parent = syntactic_layer.execute(rule_id, left, right)
rule_where = rule_codebook.location(rule_id)

what[:, i, :] = parent
where[:, i, :] = rule_where
occ[:, i] = 1

what[:, j, :] = 0
where[:, j, :] = 0
occ[:, j] = 0

subspace.set_what(what)
subspace.set_where(where)
subspace.set_activation(occ)
```

Avoid Python `.item()` decisions in the long-term design. A small eager bridge
is acceptable for a first correctness patch, but isolate it and add a TODO/test
so it cannot become the final architecture.

### NULL Seal

At end of sentence, reduce until one live slot remains.

The final root is:

```text
root.what  = subspace.what[:, root_slot, :]
root.where = grammatical operation location for the root-producing rule
```

This root is the sentence idea. It is not a separate `Chart` result.

## SymbolicSpace Refactor

`SymbolicSpace` becomes the owner of syntactic reduction.

It needs:

```text
self.syntacticLayer   executor over grammar ops
self.signalRouter     routing/rewrite layer
self.rule_codebook    grammatical operation codebook
```

### Terminal Symbol Codebook

Use the existing SymbolicSpace symbol codebook for terminals.

Terminals are long-term codebook matches:

```text
concept -> nearest/STE symbol row -> terminal.what
terminal.where -> symbol location
```

### Grammar Operation Codebook

Add a second codebook for grammatical operations.

This codebook is not responsible for parent vectors. Parent vectors are
computed by operations on child arguments.

The rule codebook holds rule identity/location information:

```text
rule_id -> rule location in .where
rule_id -> optional rule embedding for scoring
rule_id -> optional rule .what identity vector for diagnostics
```

Important:

```text
parent.what != rule_codebook[rule_id].what
parent.what = op(left.what, right.what)
parent.where = rule_codebook.location(rule_id)
```

This matches the owner's requirement: rule vectors are determined operations on
child arguments, while the codebook provides the grammatical operation identity
and location.

## ConceptualSpace Refactor

`ConceptualSpace` should no longer own live parser STM side state.

Deprecate:

```text
ConceptualSpace.stm as active parser buffer
BasicModel._stm_bounded_reduce_step as active parser rewrite
BasicModel._stm_reduce_to_single_S as active parser finalize
```

Keep temporarily:

```text
ConceptualSpace.stm compatibility tests
old probes
diagnostic snapshots
```

ConceptualSpace still performs conceptual transforms. It can produce continuous
concept candidates that SymbolicSpace then terminalizes and shifts into
`subspace.what`.

## SubSpace Helper API

Do not add new runtime fields, but add helper methods if needed. Suggested
helpers:

```python
def stack_payload(self) -> torch.Tensor:
    return self.materialize(mode="what")

def stack_locations(self) -> torch.Tensor:
    return self.materialize(mode="where")

def stack_occupancy(self) -> torch.Tensor:
    return self.materialize(mode="activation")

def set_stack_payload(self, x: torch.Tensor) -> None:
    self.set_what(x)

def set_stack_locations(self, loc: torch.Tensor) -> None:
    self.set_where(loc)

def set_stack_occupancy(self, occ: torch.Tensor) -> None:
    self.set_activation(occ)
```

If implemented, keep these as thin wrappers around existing fields.

Also add an explicit comment/docstring for stack-mode:

```text
In stack-mode, .what is the STM payload, .where is codebook location, and
.activation is occupancy. Do not read .activation as semantic truth here.
```

## Migration Plan

### Phase 0: Guardrails And Tests

Before changing behavior, add tests that pin the intended contracts.

Tests should cover:

```text
SubSpace stack-mode uses .what/.where/.activation only.
No new stack fields are introduced.
.activation gates dead stack slots.
.where can encode terminal locations and rule locations.
Rule location namespace is stable.
```

Suggested test file:

```text
basicmodel/test/test_subspace_what_stm_contract.py
```

### Phase 1: Grammar Registry Extraction

Add a static lookup surface to `WordSpace` without removing old behavior yet.

Acceptance:

```text
rule_id -> method_name works
rule_id -> arity works
rule_id -> tier works
rule_id -> where location works
existing tests still pass
```

No forward-path rewrite yet.

### Phase 2: SyntacticLayer Executor API

Add `execute()` and `execute_superposed()` to `SyntacticLayer`.

Do not delete cursor behavior yet. Keep old `forward()` for compatibility.

Acceptance:

```text
execute(rule_id, left, right) dispatches the same operation as old layer lookup
arity-1 and arity-2 rules are handled
superposed execution computes independent op results then weighted sum
no WordSpace.current_rules needed for execute()
```

### Phase 3: SymbolicSpace Rule Codebook

Add the grammatical operation codebook to `SymbolicSpace`.

Acceptance:

```text
every live grammar rule has a stable rule codebook location
terminal and rule location namespaces do not collide
empty location is zero
rule codebook does not determine parent.what
```

### Phase 4: SignalRouter Stack Rewrite Path

Add a new `SignalRouter.forward_stack()` or refactor `forward()` to operate on:

```text
subspace.what
subspace.where
subspace.activation
```

Keep old `compose()` compatibility until this is stable.

Acceptance:

```text
SHIFT writes terminal.what and terminal.where into an empty stack slot
REDUCE writes parent.what and rule.where into surviving slot
REDUCE zeros consumed slot
occupancy updates correctly
gradients reach op parameters and child payloads
```

### Phase 5: Integrate Into SymbolicSpace.forward

Call the router from `SymbolicSpace.forward`.

Do not call full `SymbolicSpace.forward()` from inside `SignalRouter`; that would
re-enter the loop.

Acceptance:

```text
SymbolicSpace owns and calls SignalRouter
SignalRouter calls SyntacticLayer executor, not cursor dispatch
live forward path does not need WordSpace.current_rules
old compatibility path still available behind tests
```

### Phase 6: Retire Active WordSpace Parser State

Once the stack rewrite path is correct, remove or bypass live uses of:

```text
WordSpace.current_rules in forward
WordSpace.generate_rules in forward
SyntacticLayer cursor in forward
Chart.compose as live parser
ConceptualSpace.stm as live parser buffer
WordSubSpace as active parser STM
```

Keep compatibility stubs if external tests or diagnostics need them.

### Phase 7: Reverse And Reconstruction

Reverse can use `.where` to decode whether a slot came from a terminal symbol or
a grammar rule.

For a reduced slot:

```text
decode .where -> rule_id
use rule-specific reverse/generate if implemented
otherwise identity/pass-through stub
```

Do not block the forward refactor on complete reverse math. Preserve existing
identity-stub behavior where rule inverses are not implemented.

## Acceptance Criteria

The refactor is successful when:

```text
1. The live STM stack is subspace.what.
2. No new stack fields are added.
3. subspace.where stores only codebook locations.
4. subspace.activation is occupancy in stack-mode.
5. Reductions happen in SymbolicSpace.
6. SymbolicSpace owns SignalRouter.
7. SyntacticLayer has a cursor-free executor API.
8. WordSpace is no longer live parser state.
9. Parent vectors for reductions are computed by grammar ops, not looked up
   from the rule codebook.
10. Rule codebook locations are written into .where for reduced slots.
11. Terminal symbol codebook locations are written into .where for terminal
    slots.
12. Empty slots are zero in .what/.where and inactive in .activation.
13. Gradients flow through reduced parent.what into child.what and op params.
```

## Explicit Non-Goals

Do not do these in the first implementation:

```text
Do not add stack_c / stack_s / stack_depth fields.
Do not create a new parser object that owns runtime stack state.
Do not keep Chart as the live parser.
Do not use WordSpace.current_rules as the live rule transport.
Do not treat stack-mode activation as semantic truth.
Do not make rule codebook rows the parent vectors.
Do not call full SymbolicSpace.forward recursively from SignalRouter.
Do not make SignalRouter own SymbolicSpace or ConceptualSpace as child modules.
```

## File Guide

Relevant current files:

```text
basicmodel/bin/Spaces.py
  SubSpace contract
  WordSubSpace stack-like use of .what/.where/.when
  ConceptualSpace.forward
  SymbolicSpace.forward

basicmodel/bin/Language.py
  SyntacticLayer
  SignalRouter
  BinaryStructuredReductionLayer
  WordSpace
  Grammar / RuleDef

basicmodel/bin/Models.py
  _forward_body
  _forward_body_per_word
  _stm_bounded_reduce_step
  _stm_reduce_to_single_S
```

The highest-value code to reuse is:

```text
BinaryStructuredReductionLayer
binary_tiling_soft_dp
binary_tiling_viterbi
UnaryStructuredLayer if unary reductions remain needed
existing GrammarLayer implementations in Layers.py
```

The highest-risk code to remove from the live path is:

```text
WordSpace.current_rules / generate_rules
SyntacticLayer._next_rule_name cursor dispatch
ConceptualSpace.stm side buffer
Chart.compose as forward parser side channel
```

## Implementation Notes For Claude

Be conservative. Do not attempt the entire deletion in one patch.

Recommended sequence:

```text
1. Add registry helpers.
2. Add executor API.
3. Add rule codebook/location namespace.
4. Add stack rewrite router path under a new method.
5. Test it directly.
6. Wire SymbolicSpace.forward to use it behind a config flag or narrow gate.
7. Remove old live state only after tests prove the new path.
```

Keep the old path alive until the new path has direct tests. The architecture is
intended to simplify the runtime, but a big-bang removal will make failures hard
to localize.

Most important: preserve the distinction between runtime stack payload and
long-term codebook memory.

```text
subspace.what                 runtime STM payload
SymbolicSpace symbol codebook  long-term terminal memory
SymbolicSpace rule codebook    long-term grammar operation identity
subspace.where                codebook location annotation
```

That distinction is the core of this refactor.
