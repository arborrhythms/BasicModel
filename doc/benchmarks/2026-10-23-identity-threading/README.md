# Identity threading — 2026-10-23

**Ready for Alec's mediated review.** Frozen source-13 `65ab7fbe9` passes
all requested scoped groups: **1,627 passed, twelve existing skips, zero
failures**; all 1,639 selected cases complete across 132 files.
[Review hand-off](final/source-13/review-handoff.md). No direct Claude
review is requested for this source, and publication remains pending.
Every earlier review, failed source and capture is retained; none is
promoted. This is mechanism evidence, not a learning claim. Item 3.0 has
not been reached.

The governing [identity spec](../../plans/2026-10-23-identity-threading.md)
records Alec's decisions in §1, the code review before implementation in
§5, the read-ahead alignment with LTM0/LTM1 and STM in §6, and every
implementation repair and port in §§7–33. Step-2 §39 authorizes continuing
with documented repairs and ports unless a decision in §1 must reverse.
No such reversal is made.

The later layout spec was re-read after Claude's incoming time-recall
amendment: only the final sentence entry will carry the stamp, and a
blank-content query may recall by `.when` alone. That is item 3's layout
work, alongside LTM0 ideas, LTM1 chains and shared STM chunks; today's row
stamping and identity event order are unchanged here. Claude's Architecture,
FutureWork and layout-spec edits are preserved unchanged for the landing.

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

[`source-audit.json`](final/source-13/source-audit.json) hashes every source,
test and grammar/config file and lists changes from `b38889eb7`, the
pre-implementation review. [`test-ports.patch`](final/source-13/test-ports.patch)
is the exact complete test diff, including new certificates and both forced decomposition fixture files. [`assertion-audit.json`](final/source-13/assertion-audit.json) separately records changed existing Python `assert` statements by test; the patch also includes tensor and exception checks.

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
| `test_live_references.py`, kind point | Parametrize the existing test over absent/present identity state. Keep the point and reference assertions; add `result['unary_ops'] == (0,)`. |
| `test_live_references.py`, public binding/capture | Replace the lift reference variant with a declared S (`clause_form='S', head_role=1, order_delta=0`) and an explicit compose bind, forced only in this unit fixture. The held occurrence and alpha reading are order one; stage its real MTM value through a `SituationAddress` frame. The operand equality changes from exact `prior` to exact `maximum(entry.leaves[0], prior)`. Replace `entry.operation_refs[actions][-1, 0] == cid` with `entry.identity_actions[actions][-1, 1] == 1`; replace `clause.composition.children[0].reference == cid` with `clause.children[0].identity['handle'] == 1`. Add `tuple(entry.identity_origins[0, :2].tolist()) == (2, occurrence[-1])`. All other capture, reconstruction and gradient assertions stay. |
| `test_thought_operation_catalog.py`, live router structural context | Change only the unary test stub from `value + 1.0` to `1.0 - value`, so every selected application stays in the identity cube. Both assertions that the unary and binary faces receive the identical owner context remain verbatim. |
| `test_math_chain_repair.py`, pending relation, both orders | Retain the appended arrival index, then explicitly `mtm.open` the pending occurrence and `mtm.bind` its selected native symbol after both rows exist. All assertions remain verbatim: no open/forward slot, native symbol 1, unchanged occurrence address, document, position and raw meaning/roles. This supplies the kept bind that the storage-only fixture bypasses; equality never selects it. |
| `decomposition_certificate.py` and `forced_decomposition_grammar.py` | Port the existing mint/reference/open fixture choices to determination and compose bind; declare the fixture's noun readings via `reference_orders` (x/y/what for one successor, plus z for two; constants retain their original readings). Preserve the preferred source references and the question-only open `what`, with no closure forcing. Read MTM mention labels and normalize its token handles at this fixture seam. Retire the stale unused `store.refs` read, keeping its +6 occurrence rank. Correct unary action indexing for the top operand at depth two, and tolerate an actually open preferred source. Guard these fixture NP actions with the bare pushed-word producing-action column, preserving the original scope. Attach trailing markers by transpose then surface, whose declared marker position preserves the bound noun head, instead of using verb. All certificate assertions, original snapshots/seeds, budgets, objective and live keep remain unchanged. The exact fixture diff is in the final test-port patch. |
| `test_review11_contracts.py`, free decoder policy | Uniformly scale event `[1,1]` to `[.5,.5]` and bank `[[1,0],[0,1],[1,2]]` to `[[.5,0],[0,.5],[.5,1]]`. This keeps the Difference operand inside the unchanged conceptual cube. Shapes, operations and every assertion remain verbatim. |

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

The required selections and runner are in `final/source-11`: the eight
regression files, the native eight-file set, all reference files, bind and
thought files, direct changed-module consumers, and both decomposition
arms. The distinct-file scope is 132 files, including the interleaving, lexical-order and copied-index consumers. This is not step 2's full sweep.
Initializations, journal states, failures and keep probes are retained.

Both decomposition arms restore their original snapshots and original
measurement seeds. `identity_decomposition.py` preserves the established
retired-predictor name mapping and maps the six expanded action/category
parameter arrays by grammar identity. Every surviving parameter row is
asserted bitwise equal to the original snapshot; only the new bind rows
retain their initialization at that original seed. Per-arm ledgers record
the input hashes, seed and row mappings. No production checkpoint loader
or original test assertion is changed.

Source-07 outcomes remain in `final/source-07/gate-summary.json`. Source-11
runs the complete scoped set, with its module files partitioned between
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
- `final/source-09`, `81ba14861`: 1,582 passes, twelve skips, three failures. All groups complete. The regression/native sets and both decomposition arms pass; the expanded module scope exposes a lost determined-relative introduction, an order-0 preview/admission mismatch, and a decoder fixture outside the cube. All failures and initial states stay; the three `diagnosis` replays reproduce them before changes. The first close fails before a keep exists; the second retains both R/A readings and greedy decisions; the third is a decoder unit without keep. The actual focused Claude diagnosis is retained and accepts the repairs under §39, not a landing verdict.


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

### Source-10 boundary evidence and ports

- `boundary-repair-01`: the retained-RNG math replay passes the original
  introduction boundary and fails in the thought writer. `boundary-repair-02`:
  all 51 selected identity/order/admission/decoder cases pass.
- `thought-boundary-01`: the new failure is reproduced with both trial
  readings, actual greedy tie and the concluded occurrence reference.
  `thought-boundary-02`: the unforced presentation passes after the writer
  accepts that existing reference. Private corpus RNG remains a declared
  diagnostic limitation, not a pinned gate or promoted source.
- `boundary-certificates-01`: 66 pass, five fail in newly added tests. Two
  dummy wrappers need their recorded sites retained; the order-zero check
  must exclude the word itself from object ambiguity. The two new writer
  fixtures unpack the returned symbol handle and correct their newly
  written raw-role check to today's exact derivation reference. This new
  assertion changes from `raw.role_refs[role] == address` to
  `raw.role_refs == (('sym', result), None, None)`; no pre-existing assertion
  changes. Both asserted pair/evidence and immutable-row checks remain.
- `boundary-certificates-02`: all 71 cases pass; `boundary-guard-01`: the
  missing-introduction event raises as required. These are development
  results; the next full scoped pass owns the frozen-source evidence.

New cases cover relation-backed determination, two ordered equality updates,
later binds, dummy reuse, separate determinations, direct/readmitted close
correspondence, order-zero fallback/ambiguity in both phases with unchanged
allocation/bank, concluded occurrence arguments in either role, and loud
loss of an event's introducing owner. No test is retired. The later layout
spec's corrected appended STM time stamp is read and preserved, not implemented.
It retains LTM0's time column; the earlier in-band draft is superseded
(spec §27). This does not change the frozen identity source.

### Source-11 stream-boundary repair

- `final/source-10`, `d55544d98`: 1,608 passed, twelve existing skips,
  one production failure; every requested case completes. Both pending
  orders pass unchanged (forward 0 bound / 16 open, reverse 0 / 32), as do
  c/e and both decomposition arms. Their original snapshots/seeds and
  surviving parameter equality are in `decomposition-snapshots.json`.
- `stream-boundary-01`: retained initial RNG, full capture and keep probe
  reproduce stream 1 indexing a one-column store scoped to stream 0.
  The failed close stops at exploit preview, before its second trial or
  keep. The earlier completed greedy tie is captured. Private corpus RNG
  is not restored; the run is diagnostic only.
- `stream-guard-01`: six passes, one new-fixture setup failure. Supply its
  omitted production zero `cost` field, preserving every assertion.
  `stream-guard-02`: all seven arrival-review cases pass.
- `stream-boundary-02`: the unchanged 1/2/3/8-stream driver passes from
  the retained initial RNG after both reader bounds checks.

Only `ReferenceClose.pending_names` and `_arrival_plan` gain the bound
check. No row is resized or rewritten. New tests check scoped rows are
excluded from not-yet-created streams while unscoped rows retain their
availability; pool, rows and bank stay unchanged. No pre-existing fixture
or assertion changes, and no test retires. The next frozen source owns all
its gate evidence independently.

Skip classification: the twelve existing skips on source-10 comprise nine
absent phase artifacts and three opt-in slow MPS/Inductor cases. Earlier
uses of “prerequisite skips” in this receipt include those three platform
cases; none is counted as a pass. The source-11 summary lists each reason.

## Completed source-11 gates

Frozen source: `bf44bbc47804899e470d4150f261496631b28a64`.
The [source-specific summary](final/source-11/gate-summary.json) verifies
all requested selections complete on this unchanged source, with no
compile-cache retry. The [source audit](final/source-11/source-audit.json)
contains 873 file hashes. No full sweep or extra fresh series is claimed.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| Regression files, including both new stream guards | 87 | 0 |
| Native eight-file set | 123 | 7 |
| Changed-module files, first partition | 707 | 5 |
| Changed-module files, second partition | 690 | 0 |
| Original decomposition, zero gain | 2 | 0 |
| Original decomposition, enabled gain | 2 | 0 |

The pending-premise certificate is unchanged: forward 0 bound / 16 open;
reverse 0 / 32; no closure forcing, zero attention budget and no wrong
binding. This is no claim that cold-start parsing has learned to close
those references. G1–G3 and c/e remain unchanged and pass. The unchanged
unforced presentation, optimizer-gradient log and 1/2/3/8-stream driver
also pass on this source.

The [decomposition ledger](final/source-11/decomposition-snapshots.json)
verifies both original snapshots and seeds (4199722147 and 2428440506),
with every surviving parameter bitwise equal. Four snapshot copies are
losslessly packed with original-byte hashes. All failures from prior
sources, their captures and pre-change keep probes remain retained.

The twelve skips are nine absent phase artifacts (copular identity,
pending-premise learning, MTM re-identification, fineweb and stage3) and
three opt-in slow MPS/Inductor cases. Each exact skip reason is in the
summary; none is counted as a pass. No learning claim or GPU execution
claim follows from the skipped cases.

## Source-12 repair after Claude's landing review

[Claude's complete source-11 review](final/source-11/claude-review.md)
returns DO NOT LAND for one blocker: non-order-one operands bypassed
`resolve_order` in the identity path. The repair restores that lookup
without reference alternatives, preserves order-one values and implicit
literals, and changes neither keep nor objective. The unchanged kind
assertions now run on both paths; seven new cases check the exceptions.
`development/typed-operand-01` passes sixteen tests. No existing assertion
is weakened and no test retires.

The public capture fixture adds the held occurrence's exact origin check.
Source-12's decomposition adapter also maps unary parameter rows by names
from the original grammar, independently of parameter equality checks,
and verifies the binary catalogue unchanged. The original snapshots and
seeds stay. No gate result is inherited from source-11.

Claude's two nonblocking history notes are carried to FutureWork: event
re-close ownership and role evidence for unfactored leaf verbs. They are
not claimed resolved by this source. The incoming future layout now puts
time in the percept portion of the paired vector; this supersedes §27's
read-ahead draft, remains item 3's work, and changes no current storage.

## Completed source-12 gates

Frozen source: `9ccc4911c5a6792cbe97de8e80de373b03cb41ce`.
The [source-specific summary](final/source-12/gate-summary.json) confirms
all 1,631 selected cases complete, 1,619 passed and twelve skipped, with
zero failures and no compile-cache retry. Its audit holds 880 source,
configuration, test and measurement-file hashes. No source or measurement
file changed after freeze. No full sweep or extra fresh series ran.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| Regression files | 87 | 0 |
| Native eight-file set | 123 | 7 |
| Changed-module files, first partition | 707 | 5 |
| Changed-module files, second partition | 698 | 0 |
| Original decomposition, zero gain | 2 | 0 |
| Original decomposition, enabled gain | 2 | 0 |

The [snapshot ledger](final/source-12/decomposition-snapshots.json) verifies
both original seeds and snapshots, the independent action-name map and
every surviving parameter in both arms. The four snapshot copies are
losslessly packed with original/compressed byte hashes. The unchanged
pending certificate records forward 0 bound / 16 open and reverse 0 / 32,
no wrong binding, no closure forcing and zero attention budget. G1–G3,
c/e, the unforced presentation, optimizer-gradient log and multi-stream
driver all pass. The added live-reference cases pass on this frozen hash.

Nine skips require absent phase artifacts; three are opt-in slow
MPS/Inductor cases. Their exact reasons are in the summary. No skipped
case counts as a pass or a learning result; no GPU execution is claimed.
Claude reviews this source before publication of main and the WikiOracle
bump. No additional gate or sweep is scheduled after that review.

## Source-13 capture-reference repair and mediated review

The [source-12 review](final/source-12/claude-review.md) is retained in full.
Its necessary repair is in `ClauseJournal`: identity programs retain the
different native reference already used by a known, non-order-one
implicit operand, then bypass all legacy formation/open/mint handling.
Kind/generic and order-one contracts remain unchanged.

`development/typed-journal-01` retains all eight pre-repair failures at
orders zero/two for lift/part/whole/equal. The new fixture feeds the actual
prepared operands into `finish_clause`; its assertions require reference
7 in the main, factored and composition views and the exact typed point.
No trial or keep exists in these deterministic unit calls. No existing
assertion changes and no test retires. The decomposition measurement adds
the direct unary-sequence check requested by review, without changing
snapshots, seeds or the numerical map.

The literal exemption is declared accurately: identity programs retain a
literal's raw value and reference; the ordinary legacy path excluded it
from binding only after typed lookup. The source-12 review accepts that
identity-path choice; it is not claimed as identical legacy behavior.

Alec now mediates the review with Claude. Source-13 is prepared for that
handoff; no direct review or publication follows its scoped gates.

`development/typed-journal-02`: 78 passed across live references, sentence
journal and identity threading after the repair. The eight pre-repair
failures keep all their assertions. No development result substitutes for
the source-13 gates.

## Completed source-13 gates; waiting for mediated review

Frozen source: `65ab7fbe9d5c67271b0bea7e4853c8ba2b690363`.
The [summary](final/source-13/gate-summary.json) confirms all 1,639
selected cases complete: 1,627 passed, twelve existing skips and zero
failures, with no compile-cache retry. The source audit verifies 886
production, configuration, test and measurement files. No source, test
or measurement file changed after freeze. No full sweep or extra fresh
series ran. No earlier result substitutes for this source.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| Regression files | 87 | 0 |
| Native eight-file set | 123 | 7 |
| Changed-module files, first partition | 707 | 5 |
| Changed-module files, second partition | 706 | 0 |
| Original decomposition, zero gain | 2 | 0 |
| Original decomposition, enabled gain | 2 | 0 |

The [snapshot ledger](final/source-13/decomposition-snapshots.json) checks
the two original snapshots and seeds (4199722147 and 2428440506), the
direct unary-name sequence and every surviving parameter in both arms.
All four snapshot copies are losslessly packed with original/compressed
byte hashes. The unchanged pending-premise certificate records forward
0 bound / 16 open and reverse 0 / 32: no wrong binding, no closure
forcing and zero attention budget. G1–G3, c/e, ordinary presentation,
optimizer-gradient logging and the multi-stream driver pass.

Nine skips require absent phase artifacts; three are opt-in slow
MPS/Inductor cases. Their exact reasons are in the summary. None is a
pass or learning evidence, and no GPU execution is claimed for them.

The [hand-off](final/source-13/review-handoff.md) is ready for Alec to pass
to Claude. No direct review of source-13 has been requested; no LAND
verdict or publication is claimed. BasicModel main remains local pending
review, with no WikiOracle bump. 2c follows the eventual identity landing;
item 3.0 is not reached.
