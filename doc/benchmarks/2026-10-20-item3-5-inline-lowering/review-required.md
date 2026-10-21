# Questions after §10.2 and the scoped set

Implementation candidate: `f9faae610f35d5ca660d0ac49249637cb23e7adf`.
Follow-up carrier/participant repair: `67c5b203f34e4299703d4c7801b34e06c1045eb6`.
Publication is held. No assertion beyond D2 and the three declared ports
has changed; no seat reservation, pool masking, fixture resizing or new
objective was introduced.

## 1. Additional representation ports need a declared scope

The instruction permits three named reference-close ports and prohibits
other assertion changes. The scoped set exposes additional expectations
of the retired implicit symbol mint. These cannot be made true by the
new occurrence-address contract. Recommended ports are below; none has
been applied. In particular, G1–G3's expected antecedent, access boundary,
truth/evidence, emitted content and forced scope must stay unchanged.

| Cases | Existing expectation | Proposed exact port |
|---|---|---|
| `test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[0..3]` | Every candidate is an int | Every candidate is an available native occurrence address; keep the complete later partition and content assertions |
| `test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised` | One symbol individual after the first ordinary mint | One retained token occurrence, zero implicit symbol individuals; retain no-access, no-new-mint and immutable-old-row checks |
| `test_revision_merges_three_accessible_groups_transitively_across_cued_documents` | Final individual bank count is four | Count the four distinct initial token occurrences, excluding the inaccessible one from working groups; the transitive merge assertion itself is unchanged and still fails |
| `test_kept_journal_supplies_np_content_predicate_and_positional_provenance[0,2]` | Audit reference is an order-1 bank individual with a symbol witness | Audit reference is the written order-1 occurrence with its retained derivation; keep content, cue, predicate, role, source leaves and immutable reload assertions |
| G1: 12 cases; G2: 12 cases; accessible G3: four cases; episode bind: one case (`test_bind_thought.py`) | `('sym', expected)` even though `expected` is now an occurrence tuple | Compare to the exact native occurrence reference. Do not change which antecedent is expected; a wrong occurrence remains a failure |
| Five native-identity tests (`test_identity_rows.py`: individual axes; native content/witnesses; forgetting; signed loader ×2) | Their ordinary-mint fixture returns a bank symbol | Construct an explicitly minted native concept for tests about bank content, axes and strict loader rejection. Keep the existing bank, gradient, witness, immutable-store and rejection assertions; ordinary tokens continue to allocate nothing |
| `test_selected_grammar_context_selects_order[particular-1,name-1,pronoun-1]`, `test_capture_owns_grammar_resolution_without_changing_reconstruction`, and `test_determiner_kept_closing_is_the_only_mint_and_lowers_once` | Reference staging cuts the known kind to -1 | Assert the preserved source-kind ID and unchanged order-1 request, leaves, close and reconstruction; this is the cut the native capacity repair removes |

**Question:** may these additional ports be made under exactly those
preserved semantic assertions, beyond the three already declared?

The first scoped set is recorded honestly as **70 passed, 43 failed, one
copular prerequisite skip** on `f9faae610`; see [every case](scoped-final.md).
The follow-up targeted set is **37 passed, five failed** on `67c5b203f`;
the five failures are the old -1 staging expectations in the last row.
The four new participant-address regressions pass, but are not substituted
for the existing G1–G3 certificates.

## 2. Working identity groups still lose an already chosen binding

This is a mechanism failure, not a representation assertion to weaken.
`working-groups-08` on `67c5b203f` retains both failures and actual read
inputs/outputs. All closing choices are right: later bind items select the
earlier token occurrence. At the next working read, each occurrence starts
its own group because frame activations intentionally carry `identity=None`.
`individuals()`'s current split/merge logic was written for native individual
symbols; its merge pass skips occurrence keys. The retained predicate lookup
is now repaired, but that alone does not restore the groups.

| Certificate | Correct closing choices | Subsequent working read | Required unchanged assertion |
|---|---|---|---|
| `test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them` | First two mint distinct addresses; third binds first, fourth binds second | Four separate occurrence groups | All four reconcile into one group after common later evidence |
| `test_revision_merges_three_accessible_groups_transitively_across_cued_documents` | Each of three documents mints one address; its later bind selects that address | Six separate accessible groups; outside occurrence remains excluded | Three groups reconcile transitively into one, without rewriting rows |

The full [row/read evidence](working-groups-08-evidence/) and
[compact table](working-groups-08.md) distinguish the selected binding from
its later interpretation. There is no proposal to store those working
identity labels back into LTM, revive automatic individual flags, use a
seat as a hidden identity, or merge all occurrences of a shared kind.
The new two-same-description/two-distinct-addresses regression must remain
green.

**Question for Alec/Claude:** which retained fact should the read use to
join these already-bound occurrence uses into an initial working group,
while keeping two independent mints of the same kind distinct? Part holds
the kind/fold symbol, not the chosen occurrence address; the frame must
hold `None` for an unminted individual. The required read rule must also
cover inline-lowered tokens sharing one kind symbol, rather than recover
identity by treating an anonymous fold-cache seat as an individual.

In ordinary terms: two introductions of the same kind must remain two
tokens. A later bind can select the first token correctly in STM. After
the close keeps the description and its kind-level Part arguments, the
next read must still be able to interpret that later use with the first
token, and then revise the grouping when more evidence arrives. The
question is what surviving evidence licenses that grouping; keeping the
last working identity as another permanent column would contradict §6.6.

## Evidence and remaining order

- D2's two saved reverse states pass on `eb75de56a`.
- Both D1 states and all twelve fresh initializations pass on `f9faae610`;
  D1 comparisons now tie and keep greedy, with no objective change.
- Both decomposition lengths in both arms pass on `f9faae610`: eight
  companion bindings and proof chains correct; paired zero-cost R passes.
- Native capacity is repaired on `67c5b203f`: rng-119 and the same captured
  fresh draw pass at eight seats, original budgets, zero seats and no lower
  edge. Every failed run and diagnostic replay remains retained.
- The four occurrence-participant regressions and all 14 PartClosing
  regressions pass in `address-read-07` on `67c5b203f`.
- The earlier twelve-fresh/scoped results are **not promoted** to the
  follow-up source. After resolving the two questions and freezing source,
  the requested final arrival and scoped/decomposition evidence must cover
  that source by commit hash. There has been no second full sweep.
- Both-main publication, the WikiOracle bump and the next 3.5 step remain
  pending. Item 0's learning findings and the separate 5.5 gates retain
  their existing ledger entries.
