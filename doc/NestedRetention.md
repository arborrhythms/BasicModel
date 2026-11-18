# Nested occurrence retention

`LTM0` is the occurrence store and `LTM1` its relation stratum. An `LTM1` row
is two tagged references, each to an `LTM0` occurrence (a row named by its
time), a symbol or another `LTM1` row, and its kind is read from its
referents. An `LTM0` row's `referent` may also name another row by its time.
There is no sentence row: a sentence that is not a relation is one `LTM0`
idea whatever its operands, and what it leaves unresolved (an open slot, a
pending name, a forward reference) is held in MTM while it is pending, never
stored as a chain. A sentence that asserts a relation (`part`, `whole`,
`implies`) is an `LTM1` pair, and a relation over a relation is a chain of
pairs whose top row is the sentence. Nesting is therefore a chain of
references, and retention must keep every row that a retained row reaches.

`LTM0` is the durable owner of ideas: each row is named by its time, which
rises strictly, and keeps its own idea, trust, referent and entry.
`LTM0.read_structure()` follows the actual chain
references of `LTM1` (an idea stores no derivation) and derives a bounded
detached view from those records; it does not create another semantic store,
learned parameter, or reference-count table. Ordered role edges retain
repetition, while a shared child is visited once. Each row keeps its own
trust: support for a reporting clause does not certify an embedded question
or claim, which is written at zero trust.

The node, structural-depth and physical-record limits are independent. Limit,
unavailable-reference and missing-metadata cases return explicit `incomplete`
reasons rather than flattening structure, and a repeated reference is read
once. Appends reject a missing reference before any buffer changes, a
relation's child must exist before the relation, and a checkpoint load rejects
a dangling reference, a duplicate pair or a cycle before importing the store.
What a row left pending is held in MTM, not beside the row, and adds no edge
to the structure.

The bounded reader (`read_structure`), append validation
(`_validate_local_references`, `LTM1.add`) and checkpoint validation
(`LTM0.validate_checkpoint`, the `LTM1` load) are in [Memory](../bin/Memory.py).

## Live roots, and no withdrawal

The forgetting pass keeps, whatever their activity, the rows an `LTM1` row
references and everything those reach, the rows another row names as its
referent, and the live roots: the rows held by the last understanding, the
ordinary thought history and the expectation context
([STM](STM.md#10-ltm-as-the-chain-of-stm-end-states)). The relation pass keeps
a relation that another relation references. Nothing is withdrawn: a row has
no origin, an idea's trust is not changed after it is written, a list of truths
sent again is written again, and a restore withdraws nothing. Which rows are
truths is read from their trust: the materialized TruthLayer view takes the
latest `truthMaxEntries` rows of nonzero trust, each as its content times its
trust.

Thought roots are derived on demand from the existing per-row ordinary history,
including role references, bindings, scopes and recorded sources. Discovery
reads addresses and metadata only; it does not detach a live episode or create
a second owner.

The retention closure (`forget`, `forget_parts`, with its roots from
`model_boundary`) is in [Forgetting](../bin/Forgetting.py), the truth view is
`TruthLayer` in [Layers](../bin/Layers.py), and TruthSet admission is
`BasicModel.store_truths` in [Models](../bin/Models.py).

## Restore ordering and gradients

A restore withdraws nothing: the rows return with their trust and activity.
Every store's columns are validated against the construction before any store
is imported. An older layout is converted only where it maps exactly: a
retained row's selected reference becomes its referent and what its bindings
held pending moves to MTM; a row holding both trust lanes, or rows without
unique times, are rejected before loading. Invalid complete structure fails
explicitly.

Durable structure views are detached owned copies. Existing live episode paths
remain live through their explicit optimizer boundary; root discovery and
retention touch no content tensor. Checkpoints retain detached data and cannot
reconnect an earlier optimizer graph. Retention adds no objective, optimizer
parameter or policy reward.

The structural restore order is `BaseModel._restore_structural_extras` in
[Models](../bin/Models.py), and the import of an older layout is in
[MemoryCheckpoint](../bin/MemoryCheckpoint.py).

## Writers

Typed linguistic observation writes, nested and converse meaning, the normal
controller and the shared executor cost use this owner. Local child
references are validated and capacity-preflighted before a postorder append;
embedded claims are written at zero trust. Eager, pending and packed
writers use the same owner ([SelectedMeaning](SelectedMeaning.md)).
