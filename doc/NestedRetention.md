# Nested occurrence retention

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23; recorded 2026-10-30):** an LTM1 row is two tagged references (each to an LTM0 occurrence, a symbol or another LTM1 row, its tag saying which) with its kind read from the referents, not three. The unused `LTM0` columns are removed and the row has this layout (Alec, 2026-11-01).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.


Implementation reference, September 18.

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
(`bind_constituents`) and the atomic sidecar restore (`set_extra_state`) are in
[Memory](../bin/Memory.py).

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
reconnect an earlier optimizer graph. This change adds no objective, optimizer
parameter or policy reward.

The structural restore order is `BaseModel._restore_structural_extras` in
[Models](../bin/Models.py).

## Writers and open gates

Typed linguistic observation writes, nested and converse meaning, the normal
controller and the shared executor cost use this owner. Local child
references are validated and capacity-preflighted before a postorder append;
embedded claims are unverified at zero trust. Eager, pending and packed
writers use the same owner ([SelectedMeaning](SelectedMeaning.md)).
Expectation/residual learning and held-out utility remain open.
