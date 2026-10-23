# Identity threading — 2026-10-23

The next source freeze on main runs its scoped gates in
[`final/source-04`](final/source-04). Review and publication are pending.
This is a mechanism receipt, not learning evidence. Item 3.0 is not reached.

The governing [identity spec](../../plans/2026-10-23-identity-threading.md)
records Alec's decisions in §1, the code review before implementation in
§5, the read-ahead alignment with LTM0/LTM1 and STM in §6, and every
implementation repair and port in §§7–9. Step-2 §39 authorizes continuing
with documented repairs and ports unless a decision in §1 must reverse.
No such reversal is made.

## Source and scope

The published storage source remains the basis. Its derivation layout,
readers, close-time minting and three-slot rows stay. The next storage
round owns §71's LTM0 ideas/LTM1 binary chains and §72's shared eight STM
chunks. The latest three open STM/layout questions in §72 are read and
remain for that spec; they are not implemented here.

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

[`source-audit.json`](final/source-04/source-audit.json) hashes every source,
test and grammar/config file and lists changes from `b38889eb7`, the
pre-implementation review. [`test-ports.patch`](final/source-04/test-ports.patch)
is the exact complete test diff, including new certificates.

## Every existing-test port

No test is retired. The following list includes renames, fixture rewrites,
every changed existing assertion and new assertions in existing fixtures.
All assertions not listed remain unchanged.

| File and test | Exact disposition |
| --- | --- |
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

## New certificates and development corrections

`test_identity_threading.py` covers the second VP reading the first VP's
result; pathwise derivatives in eager and fullgraph execution; replacement
rather than historical union; two aliases and description join; both keep
outcomes; stream isolation; checkpoint/reclamation ownership; the actual
global chooser's location menu, zero dummy and score-function credit;
observed exclusions; a real journal/write/commit of two VPs with immutable
occurrences; malformed checkpoint rejection; and unresolved-object
propagation. `test_query_executors.py` adds the no-witness result for an
entirely open name even when a row is held.

Same-round fixture corrections are retained with their failures: the
compiled fixture's Python bank moves to the host; its unused-gradient
assertion accepts `None` or exactly zero because AOT materializes zeros;
the journal fixture uses its declared `sum` container and an initial
determination followed by an alias. Its semantic assertions stay, including
two completed events. No production choice is forced by those fixtures.

## Gates by source

The required selections and runner are in `final/source-04`: the eight
regression files, the native eight-file set, all reference files, bind and
thought files, direct changed-module consumers, and both decomposition
arms. The distinct-file scope is 116 files. This is not step 2's full sweep.
Initializations, journal states, failures and keep probes are retained.

Both decomposition arms restore their original snapshots and original
measurement seeds. `identity_decomposition.py` preserves the established
retired-predictor name mapping and maps the six expanded action/category
parameter arrays by grammar identity. Every surviving parameter row is
asserted bitwise equal to the original snapshot; only the new bind rows
retain their initialization at that original seed. Per-arm ledgers record
the input hashes, seed and row mappings. No production checkpoint loader
or original test assertion is changed.

Final outcomes will be written in `final/source-04/gate-summary.json`.
No development pass or earlier frozen result substitutes for that source.

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

Large JSON captures are losslessly gzipped. `packed-captures.json` records
each original and compressed SHA-256; new captures write gzip directly
when larger than 2 MiB. The source-02 freeze correction changes packaging
only, not production or tests.

## Review and publication

Claude's actual review of the frozen source must follow the complete
scoped results. Publication means the BasicModel main push, then the
WikiOracle gitlink bump and main push. Neither has happened for this line.
The next layout-and-removal spec remains Claude's, followed by 2c with the
two R10 arrival notes and step 2's one full sweep.
