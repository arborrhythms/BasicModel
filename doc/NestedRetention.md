# Nested occurrence retention

Implementation reference, September 18.

`TernaryTruthStore` remains the durable owner of full role payloads, stable
occurrence IDs, semantic context and evidence. `read_structure()` derives a
bounded detached view from those records; it does not create another semantic
store, learned parameter, or reference-count table. Ordered role edges retain
repetition, while a shared child is visited once. Each occurrence keeps its own
mode, polarity, bindings, scope and evidence: support for a reporting clause
does not certify an embedded question or claim.

The node, structural-depth and physical-record limits are independent. Cycle,
limit, foreign-owner, unavailable-reference and missing-metadata cases return
explicit `incomplete` reasons rather than flattening structure. Appends reject
missing local references before buffer mutation, and sidecar restore validates
the complete local constituent graph before publishing replacement metadata.
Bindings and scopes retain their addressed records but are not hidden syntax
edges and therefore do not change structural depth.

[Bounded reader](../bin/Layers.py#L8904),
[append validation](../bin/Layers.py#L8889), and
[atomic sidecar validation](../bin/Layers.py#L9016).

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

[Withdrawal](../bin/Layers.py#L9322),
[retention closure](../bin/Layers.py#L9338),
[truth view](../bin/Layers.py#L7227),
[thought roots](../bin/Thoughts.py#L302), and
[normal TruthSet replacement](../bin/Models.py#L6453).

## Restore ordering and gradients

For a stateless tensor-only restore, request-scoped rows lose authority
immediately but cannot be physically pruned while semantic metadata is absent.
A complete checkpoint restores semantic rows and thought owners first, then
prunes only unreachable request content. Invalid complete structure fails
explicitly; absent metadata is not evidence that no reader exists.

Durable structure views are detached owned copies. Existing live episode paths
remain live through their explicit optimizer boundary; root discovery and
retention touch no content tensor. Checkpoints retain detached data and cannot
reconnect an earlier optimizer graph. This change adds no objective, optimizer
parameter or policy reward.

[Tensor restore hook](../bin/Language.py#L13700),
[final pruning](../bin/Language.py#L13732), and
[structural restore order](../bin/Models.py#L4718).

## Evidence and open gates

The rebased reviewer probes first failed in
`output/tests/20260918-014842-f679b1` (22/22 expected failures). The focused
green selection then passed 68/68 cases in 76.1 seconds with a 0.50 GiB peak in
`output/tests/20260918-015808-1dce43`. The broader affected selection passed
309/309 cases in two fresh bounded workers (73.5 and 28.9 seconds; 0.50 GiB
peak) in `output/tests/20260918-020312-c526c2`. These cover retention closure,
independent limits, cycles, atomic rejection, live thought credit, checkpoint
ordering, semantic-sidecar, query-occurrence and consolidation behavior.

This is a retention and lifecycle foundation only. Typed linguistic observation
writes, the normal selected-meaning/controller path, actual shared executor
cost, expectation/residual learning, generation ownership and learned utility
remain open. The separately queued two-truths and forgetting implementations
remain out of scope.
