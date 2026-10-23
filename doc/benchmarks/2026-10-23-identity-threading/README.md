# Identity threading — 2026-10-23

Source-07 `59b7c8b87` completed with 1,442 passes, twelve skips and one
retained c/e failure. Its read-only Claude diagnosis and the completed
repair are recorded below. Source-08 was interrupted for the incoming MTM/concept clarification.
Source-09 is prepared for freezing; gates, final review and publication
remain pending. No earlier result is promoted.
This is a mechanism receipt, not learning evidence. Item 3.0 is not reached.

The governing [identity spec](../../plans/2026-10-23-identity-threading.md)
records Alec's decisions in §1, the code review before implementation in
§5, the read-ahead alignment with LTM0/LTM1 and STM in §6, and every
implementation repair and port in §§7–22. Step-2 §39 authorizes continuing
with documented repairs and ports unless a decision in §1 must reverse.
No such reversal is made.

## Source and scope

The published storage source remains the basis. Its derivation layout,
readers, close-time minting and three-slot rows stay. The next storage
round is moved to item 3.0 by the latest §72 amendment: LTM0 ideas, LTM1 binary chains, then shared STM chunks. Its no-word-bag and immutable-chain decisions are read but not implemented here. Step 2 continues with 2c on today's storage.

MTM now keeps current values, mention readings and verb/role event
histories. Compose bind enumerates frame-accessible and trial-local
locations plus the zero-logit dummy in the existing operation chooser.
Determiners retain their folds and lose their reference variants. Trial
values are numerical state in the compiled loop; a bound VP reads the
identity and publishes its result before the next clause. Description
mentions join; VP results replace. Only the kept close publishes detached
values to the durable pool. Role evidence lasts with its MTM token.
Reference reconstruction reads the candidate snapshot at the bind.
Definitions reject a word as another word's object.

The c/e boundary repair prevents a child query from answering itself from
its enclosing assertion or from treating an unconstrained name as a match
to any retained row. A nested query inherits the close's authority and
mode, and an open child is stored as a question. This preserves the
certificate's existing mint, question, raw-row and witness assertions.

[`source-audit.json`](final/source-09/source-audit.json) hashes every source,
test and grammar/config file and lists changes from `b38889eb7`, the
pre-implementation review. [`test-ports.patch`](final/source-09/test-ports.patch)
is the exact complete test diff, including new certificates and both forced decomposition fixture files. [`assertion-audit.json`](final/source-09/assertion-audit.json) separately records changed existing Python `assert` statements by test; the patch also includes tensor and exception checks.

## Every existing-test port

No test is retired. The following list includes renames, fixture rewrites,
every changed existing assertion and new assertions in existing fixtures.
All assertions not listed remain unchanged.

| File and test | Exact disposition |
| --- | --- |
| `test_reference_close.py`, `form_exposure` helper | Replace the retired `expose_form(owner, cue, content)` call with an explicit `mint(owner, content, form=word)` to supply the fixture's already-known cue region. Each caller supplies it once. Every existing assertion stays verbatim. |
| `test_medium_term_memory.py`, native individual mint/bind | Rename `test_native_individual_mint_and_bind_make_no_pool_entry` to `test_native_individual_mint_and_bind_share_an_occurrence_keyed_value`. Replace `assert not ...mtm.entries` with `token == store.occurrence_of(row)`, native identity and `count == 2`, and exact current value `a[0]`. Selected native antecedent assertions stay. |
| `test_reference_arrivals.py`, named native bind | Rename `test_named_bind_to_a_native_individual_creates_no_pool_entry` to `test_named_bind_to_a_native_individual_keeps_an_occurrence_keyed_value`. Replace empty pool with own occurrence token/native identity and exact value `a[0]`. Bound native reference and no open/forward assertions stay. |
| `test_reference_arrivals.py`, native arrival, two cases | Read the owner with `pool.token_of(token)`. Kind remains the native symbol; count changes from `2` to `3 if retained_occurrence else 2`; replace last-member equality with membership of both the pending token and selected occurrence; replace `{token}` with `{owner}` for the entry keys. Retained antecedent equality, derivation symbol and closed working read stay. |
| `test_reference_arrival_review.py`, native arrival | Rename `test_native_arrival_without_occurrence_does_not_materialize_a_working_vector` to `test_native_arrival_without_synthetic_occurrence_joins_its_working_entry`. Read owner through `token_of`; occurrences change from `(token,)` to `(owner, token)`, count stays `2`, kind stays the native symbol; audit reference changes from token to owner. No synthetic companion and unchanged LTM/bank checks stay. |
| `test_bind_thought.py`, participant evidence | Rename `test_participant_cache_rebuilds_from_retained_identity_rows` to `test_participant_evidence_lasts_with_the_identity_pool`. Restore MTM's saved state instead of clearing a Part cache; forget the tokens instead of deleting Part rows. All role and equality assertions stay. |
| `test_identity_rows.py`, current value | Rename `test_native_row_is_the_only_content_and_witnesses_are_live_queries` to `test_mtm_holds_current_content_and_native_witnesses_are_live_queries`. Candidate content changes from exact `a[0]+a[1]` to exact `a[0]`; the separately changed bank row remains `a[1]`. Witness and no-chart checks stay. |
| `test_step2b_storage.py`, definition guard | Rename `test_word_bank_form_survives_use_as_another_words_object` to `test_definition_rejects_a_word_object_and_keeps_both_forms`. Both word-form `is None` assertions become `== b'12'`; defining the other word to this word now must raise `ValueError` matching `word as its object`; `definitions.words(word)` changes from `(other,)` to `()`. Add the other word's own form `== b'plus'`. Native object and byte-identity checks stay. |
| `test_step2b_storage.py`, provisioning, two cases | Add a caller MTM to the minimal model fixture. Add assertions that provisioning uses a different pool and finally restores the same caller pool, including on failure. Existing source-isolation assertions stay. |
| `test_math_chain_section14.py`, c/e | Before presentation admit x and what if absent and set their readings to order one via `reference_orders`, under reconstruction spec §13. No assertion, forced choice, budget or keep rule changes. The pending-premise and item-0 learning fixtures are unchanged. |
| `test_item6_2_thinking.py`, open closing, four surfaces | The direct boundary fixture bypasses compose/ReferenceClose. Supply its kept anonymous token with `pool.open`, then its selected refinement with `pool.bind` after the existing equality append. All native-reference, source-address, raw-open, witness, question-kind and reuse assertions stay. Equality alone does not bind. |
| `fixtures/when-readers-round4a0.json` | Add `MediumTermMemory.state` and `.restore` to the reader inventory with attribute `when`, as checkpoint readers. All inventory assertions stay. |
| `test_live_references.py`, public binding/capture | Replace the lift reference variant with a declared S (`clause_form='S', head_role=1, order_delta=0`) and an explicit compose bind, forced only in this unit fixture. The held occurrence and alpha reading are order one; stage its real MTM value through a `SituationAddress` frame. The operand equality changes from exact `prior` to exact `maximum(entry.leaves[0], prior)`. Replace `entry.operation_refs[actions][-1, 0] == cid` with `entry.identity_actions[actions][-1, 1] == 1`; replace `clause.composition.children[0].reference == cid` with `clause.children[0].identity['handle'] == 1`. All other capture, reconstruction and gradient assertions stay. |
| `test_thought_operation_catalog.py`, live router structural context | Change only the unary test stub from `value + 1.0` to `1.0 - value`, so every selected application stays in the identity cube. Both assertions that the unary and binary faces receive the identical owner context remain verbatim. |
| `test_math_chain_repair.py`, pending relation, both orders | Retain the appended arrival index, then explicitly `mtm.open` the pending occurrence and `mtm.bind` its selected native symbol after both rows exist. All assertions remain verbatim: no open/forward slot, native symbol 1, unchanged occurrence address, document, position and raw meaning/roles. This supplies the kept bind that the storage-only fixture bypasses; equality never selects it. |
| `decomposition_certificate.py` and `forced_decomposition_grammar.py` | Port the existing mint/reference/open fixture choices to determination and compose bind; declare the fixture's noun readings via `reference_orders` (x/y/what for one successor, plus z for two; constants retain their original readings). Preserve the preferred source references and the question-only open `what`, with no closure forcing. Read MTM mention labels and normalize its token handles at this fixture seam. Retire the stale unused `store.refs` read, keeping its +6 occurrence rank. Correct unary action indexing for the top operand at depth two, and tolerate an actually open preferred source. Guard these fixture NP actions with the bare pushed-word producing-action column, preserving the original scope. Attach trailing markers by transpose then surface, whose declared marker position preserves the bound noun head, instead of using verb. All certificate assertions, original snapshots/seeds, budgets, objective and live keep remain unchanged. The exact fixture diff is in the final test-port patch. |

## New certificates and development corrections

`test_identity_threading.py` covers the second VP reading the first VP's
result; pathwise derivatives in eager and fullgraph execution; replacement
rather than historical union; two aliases and description join; both keep
outcomes; stream isolation; checkpoint/reclamation ownership; the actual
global chooser's location menu, zero dummy and score-function credit;
observed exclusions; a real journal/write/commit of two VPs with immutable
occurrences; malformed checkpoint rejection; and unresolved-object
propagation; a later bind retaining its earlier introduction under a wrapper; and an unchanged dummy preserving a carried OBJECT with a different bank value. `test_query_executors.py` adds the no-witness result for an
entirely open name even when a row is held.

Same-round fixture corrections are retained with their failures: the
compiled fixture's Python bank moves to the host; its unused-gradient
assertion accepts `None` or exactly zero because AOT materializes zeros;
the journal fixture uses its declared `sum` container and an initial
determination followed by an alias. Its semantic assertions stay, including
two completed events. No production choice is forced by those fixtures.

## Gates by source

The required selections and runner are in `final/source-09`: the eight
regression files, the native eight-file set, all reference files, bind and
thought files, direct changed-module consumers, and both decomposition
arms. The distinct-file scope is 131 files, including the interleaving, lexical-order and copied-index consumers. This is not step 2's full sweep.
Initializations, journal states, failures and keep probes are retained.

Both decomposition arms restore their original snapshots and original
measurement seeds. `identity_decomposition.py` preserves the established
retired-predictor name mapping and maps the six expanded action/category
parameter arrays by grammar identity. Every surviving parameter row is
asserted bitwise equal to the original snapshot; only the new bind rows
retain their initialization at that original seed. Per-arm ledgers record
the input hashes, seed and row mappings. No production checkpoint loader
or original test assertion is changed.

Source-07 outcomes remain in `final/source-07/gate-summary.json`. Source-09
runs the complete scoped set, with its 114 module files partitioned between
two independent process groups and each selection run once.
No development pass or earlier frozen result substitutes for that source. The long module group and the other groups may run in separate processes on that same unchanged hash; each selection runs once.

## Retained unsuccessful attempts

Every development directory contains the source hash/diff, file hashes,
log, initial states and captured failure when execution reached the test.
The dated spec sections give the diagnosis and fix for each attempt:

- `foundation-01`, `foundation-02`: native working-entry port; minimal
  fixture rule metadata.
- `compose-01`: interrupted capture setup, no claimed result;
  `compose-02`: root-copy event correspondence, with keep probe.
- `reference-01` through `reference-07`: native entry ports and arrival
  ownership, commit-local reading initialization, thought mention-witness
  filtering, minimal plan event guard, and native word spelling.
- `integration-01` through `integration-05`: same-round compiled fixture,
  AOT unused gradient, declared container, new-site arrival guard and
  duplicate event adapter; all source captures retained.
- `c-e-probe-01`: original failure reproduced from captured initialization,
  both trial costs retained; original private presentation-generator state
  unavailable, so no claim of identical source addresses.
- `ports-01`, `c-e-probe-02`, `ports-02`, `ports-03`: nested query
  self-support, unconstrained query, source authority and open-row kind.
- `ports-04`: c/e passes; four stale equality-only boundary fixtures need
  explicit kept MTM state. `ports-05` passes after that fixture rewrite.
- `final/source-01`, `9623fa959`: 85 regression passes; native set 121
  passes, seven existing skips and two failures. Pending-premise orders
  pass unchanged; c/e and historical-value failures are not relabeled.
- `final/source-03`, `10a36531c`: 85 regression passes; native set 122
  passes, seven existing skips and one inference-kind failure. C/e and both
  pending-premise orders pass. `ports-06` passes the 25 targeted cases after
  correcting the distinction between a retained annotation and an open slot.
- `final/source-02`, withdrawn `2331c11b0`: 85 regression passes; native
  group interrupted for lossless artifact packaging. No gate promoted.
  The original freeze is retained in `withdrawn-freeze.bundle`, with
  prerequisite `9623fa959`.

- `final/source-04`, `731cb5791`: all 1,160 module cases completed; 13 failures retained. Six completed-root kinds, two forced relation cases, the time-reader inventory, the lost introduction, the stale lift-binding fixture, the eight-seat dummy wrapper and copied-owner attachment are diagnosed in spec §11. No other groups promoted.
- `scoped-01`: 65 focused passes after the repairs. `scoped-02`: collection error from a mistyped selector. `scoped-03` and `scoped-04`: the ported fixture's held object order and declared S/overlay catalogue correspondence; fixed without further assertion edits. `scoped-05`: the public binding, inverse and gradient certificate passes.

- `scoped-compiled-01` and `scoped-compiled-02`: the copied store still points at the original owner. The read-only close probe confirms its edge exists in the original Part table, not the copied one. `scoped-06` passes 33 identity/index/c/e cases after reattachment; no parity assertion changes.
- `scoped-07`: the new wrapper fixture used an undeclared `null` face; correct it to the declared transparent `not` face without changing assertions. `scoped-08` passes both wrapper/introduction certificates. Event deduplication keys on the recorded operation, so a transparent wrapper cannot duplicate its S occurrence.

- `final/source-05`, `4763623e6`: 85 regression passes; native 122 passes, seven existing skips and one recursive-alias failure. Both pending-premise orders and c/e pass unchanged. Spec §14 diagnoses preview/admission site ordering from the retained journal and plan. No result is promoted.
- `site-order-01`: 22 focused passes after all callers share chronological `sites()` order. The new wrapped-introduction regression now also repeats production's admission zip before commit; no existing assertion changes.
- `scoped-compiled-03`: the original compiled parity certificate passes after the copied-owner repair, with every parity assertion unchanged.

- `final/source-06`, `6da7938e6`: every requested group completes: 1,423 passes, twelve skips, nine failures (one native, four module, four decomposition). Full results and keep probes are retained. §§15–16 diagnose the lost nested determination, cube-invalid router stub, two stale equality-only storage fixtures, nested question eligibility, and the decomposition fixture seam. Nothing is promoted.
- `boundary-01`: 54 focused passes after the boundary repairs and fixture ports. `retained-menu-01`: the live initial-menu case passes from source-06's retained initial RNG.
- `decomposition-01`: the first port used the dummy for former mints, leaving genuinely open preferred source slots. `decomposition-02`: the explicit determination exposed the fixture's stale unary position index at depth two. Both failed states, original parameter mappings and keep probes remain retained; no proof assertion changes.

- `decomposition-03` and `decomposition-04`: the fixture's explicit NP operation must retain its original bare-operand guard; the remaining capture exposes a direct pending child losing its root relation context. Both proof assertions fail; both trials and keep probes remain.
- `direct-role-01`: 55 focused identity, arrival and math-repair cases pass after propagating only the direct child's pending name to its actual parent role, including two new subject/complement regression cases.
- `decomposition-05`: explicit determination of constants delayed lift beyond the unchanged three operations per word. `decomposition-06` stamps only the fixture's named references; it still fails the unchanged proof assertion. Its retained questions have both referents open; this is a mechanism/fixture diagnosis in progress, not a pass or a learning claim.
- `capture-sidecars-01`: the two direct-role cases pass with the expanded capture format. `capture-check.json` verifies that semantic sidecars and Part state are present; earlier captures are not relabeled as containing them.
- `claude-decomposition-diagnosis`: actual read-only Claude diagnosis completed on the failing source and exact retained diff. Its four findings and response are in spec §18; this is not the final frozen-source review.

- `decomposition-07`: the questions have only `what` open and premise relations are stored, but the forced punctuation attachment changes the bound noun to a predicate. The unchanged proof assertion fails; both trials and keep probes are retained.
- `decomposition-08`: after the documented fixture marker-position port, both zero-arm cases pass and all four companion questions are correct. This is diagnostic only, not a substitute for either frozen arm.
- `resolved-01`: 101 passes, four prerequisite skips, one existing R10 alias failure: direct-role propagation incorrectly treated fused structural children as relative roles. Restrict it to present participant roles of relative clauses. `resolved-02` passes all 102 cases, four skips, every R10 assertion unchanged.

New resolution certificates distinguish a completed unminted value from an
anonymous pending name without using content magnitude or a seat; cover
working reads, staged resolution, versioned/atomic checkpoint migration,
refinement and merge; and verify that resolved direct compose binds retain
the current Part relation and selected identity. Checkpoint v1 migration
preserves its old kind-based convention; v2 records explicit pending state.

- `final/source-07`, `59b7c8b87`: 1,442 passes, twelve skips, one c/e failure; both pending-premise orders and both original-snapshot decomposition arms pass. The actual focused Claude report identifies the lost unresolved-operand annotation; it is diagnosis, not a landing verdict.
- `unresolved-01`: 58 passes, one retained c/e failure. The answer guard reveals a repeated equality answering the free question with its own fixed side.
- `unresolved-02`: eleven passes and the new repeated-witness regression fails, reproducing that proof defect without training.
- `unresolved-03`: 61 passes; c/e stays open but its unchanged operation count exposes retrieval returning an unresolved operand as an answer witness and causing a second query.
- `unresolved-04`: all 72 focused cases pass after query applies the same answer-resolution check. No existing assertion changes.

The repair records `_unresolved_operands` once on the existing parent's
semantic sidecar, alongside its existing reference. It adds no row, column,
working vector or stored parse. The answer boundaries refuse that endpoint;
the unmarked reverse endpoint and explicit reflexive witnesses remain
valid. The no-progress guard also handles repeated equality witnesses.
The annotation is excluded from chooser metadata. New tests cover both
query/equality directions, ordinary fill, unchanged features, direct and
resolved counterparts, immutable rows/bank and checkpoint round-trip.
The full source-08 test patch records these additions; no existing test is
retired or changed for this repair.

- `final/source-08`, `248f8f8d7`: deliberately interrupted when Claude's incoming §2 clarification identified the live form-exposure bank write. Partial result: 517 passes, one skip, no failed assertion; the other cases and decomposition arms are incomplete. Its interruption ledger and every partial capture remain. Nothing is promoted.
- `bank-isolation-01`: 123 passes, four existing prerequisite skips after retiring the live exposure calls and functions. Four new cases prove kept bind training and delayed mention credit cannot allocate a cue-region concept or alter the bank, with/without a prior region. The delayed loss still requires gradients.

Claude's concurrent §2 concept/current-value clarification and grounding
text are preserved verbatim. Philosophy's first companion edit is commit
`738e51e8d`; its further clarification is preserved with the next freeze.
Current values live in MTM. Legacy cue regions remain readable; bind and
delayed mention credit no longer write them. Promotion/rebuilding remains
2c's work. The source-08 diagnostic source was incomplete for this
clarification and cannot support publication.

Large JSON and tensor captures are losslessly gzipped. `packed-captures.json` records
each original and compressed SHA-256; new captures write gzip directly
when larger than 2 MiB; tensor files larger than 16 MiB have per-capture packing ledgers. `torch.load(gzip.open(path, "rb"), weights_only=False)` reads a packed tensor capture. No state is dropped. The source-02 freeze correction changes packaging
only, not production or tests.

## Review and publication

Claude's actual review of the frozen source must follow the complete
scoped results. Publication means the BasicModel main push, then the
WikiOracle gitlink bump and main push. Neither has happened for this line.
Next is 2c with the two R10 arrival notes and step 2's one full sweep. The layout-and-removal spec and shared STM chunks now belong to item 3.0.
