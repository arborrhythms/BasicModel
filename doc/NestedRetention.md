# Nested occurrence retention

`LTM0` is the occurrence store and `LTM1` its relation stratum. An `LTM1` row
is two tagged references, each to an `LTM0` occurrence, a symbol or another
`LTM1` row, and its kind is read from its referents. There is no sentence row:
a sentence whose words are all known concepts compresses to one `LTM0` idea,
and a sentence that cannot is an `LTM1` chain whose top row is the sentence.
Nesting is therefore a chain of references, and retention must keep every row
that a retained row reaches.

`LTM0` remains the durable owner of ideas, stable occurrence IDs, semantic
context and evidence. `LTM0.read_structure()` follows the actual chain
references of `LTM1` (an idea stores no derivation) and derives a bounded
detached view from those records; it does not create another semantic store,
learned parameter, or reference-count table. Ordered role edges retain
repetition, while a shared child is visited once. Each occurrence keeps its own
polarity, bindings, scope and evidence: support for a reporting clause does not
certify an embedded question or claim.

The node, structural-depth and physical-record limits are independent. Cycle,
limit, foreign-owner, unavailable-reference and missing-metadata cases return
explicit `incomplete` reasons rather than flattening structure. Appends reject
missing local references before buffer mutation, and sidecar restore validates
the complete local constituent graph before publishing replacement metadata.
Bindings and scopes retain their addressed records but are not hidden syntax
edges and therefore do not change structural depth.

The bounded reader (`read_structure`), append validation
(`bind_constituents`) and the atomic sidecar restore (`load_semantic_extras`)
are in [Memory](../bin/Memory.py).

## Withdrawal and live roots

Clearing an origin computes the transitive closure from rows of other origins
and roots supplied by retained ordinary thought records. Unreferenced rows are
compacted; referenced request rows retain their occurrence IDs, content,
context and source text. Withdrawal removes fact authority instead: accepted
facts become `unverified`, every selected row receives zero trust, and existing
questions or estimates retain their explicit provenance. The materialized
TruthLayer view selects accepted facts only.

Thought roots are derived on demand from the existing per-row ordinary history,
including role references, bindings, scopes and recorded sources. Discovery
reads addresses and metadata only; it does not detach a live episode or create
a second owner.

Withdrawal (`withdraw_origin`) and the retention closure (`clear_origin`) are
in [Memory](../bin/Memory.py), the truth view is `TruthLayer` in
[Layers](../bin/Layers.py), and TruthSet replacement is
`BasicModel.store_truths` in [Models](../bin/Models.py).

## Restore ordering and gradients

For a stateless tensor-only restore, request-scoped rows lose authority
immediately but cannot be physically pruned while semantic metadata is absent.
A complete checkpoint restores semantic rows and thought owners first, then
prunes only unreachable request content. Invalid complete structure fails
explicitly; absent metadata is not evidence that no reader exists.

Durable structure views are detached owned copies. Existing live episode paths
remain live through their explicit optimizer boundary; root discovery and
retention touch no content tensor. Checkpoints retain detached data and cannot
reconnect an earlier optimizer graph. Retention adds no objective, optimizer
parameter or policy reward.

The structural restore order is `BaseModel._restore_structural_extras` in
[Models](../bin/Models.py).

## Writers

Typed linguistic observation writes, nested and converse meaning, the normal
controller and the shared executor cost use this owner. Local child
references are validated and capacity-preflighted before a postorder append;
embedded claims are unverified at zero trust. Eager, pending and packed
writers use the same owner ([SelectedMeaning](SelectedMeaning.md)).
