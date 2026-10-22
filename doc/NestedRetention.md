# Nested occurrence retention

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.


Implementation reference, September 18.

`LTM0` remains the durable owner of full role payloads, stable
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

Item 1 now connects typed linguistic observation writes, nested/converse
meaning, the normal controller and actual shared executor cost. Local child
references are validated and capacity-preflighted before a postorder append;
embedded claims are unverified at zero trust. Eager, pending and packed writers
use the same owner. [SelectedMeaning](SelectedMeaning.md) records the integration
and learned paraphrase evidence. Expectation/residual learning, the separate
generation-catalogue migration and held-out utility remain open. The separately queued two-truths and forgetting implementations
remain out of scope.
