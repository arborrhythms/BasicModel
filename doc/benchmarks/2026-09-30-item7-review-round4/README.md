# Item 7, review round 4 — ready for review; sweep red

Stopped for Claude's review. Nothing is committed or accepted. The ordered
work is spec §20, with Alec's
§19.5 answers. HEAD is `1678ee1fb79c474ffaee5725fd035716de5f1913`.
The complete sweep is **4,774 passed, three failed, 322 skipped and one
expected failure**. All thirteen round-3 failures now pass; three newly
failing cases share the predicate-address mismatch detailed below.
The [review map](review-map.md) links each finding to its failing probe,
repair and verification in work-list order.

The baseline and every composition/closing step measure the complete §19.1
XOR table, including fifteen fresh unseeded MM_20M_xor exact round trips per
tree. All attempts remain in the record. The existing configuration-matrix
smoke fixtures retain their original seed argument; no seed is added or
changed to obtain a pass. No threshold, configuration or guard changes.

A failing probe is saved before each repair. Every test port retains its
old and new body. The memory guard remains 8 GiB; a case stopped by it gets
one unguarded diagnostic, recorded separately. Tests run against a frozen
source. Reconstruction's eight declared seeds are 0 through 7 on both trees.
MM_grammar uses ten full-budget unseeded attempts per tree. Item 7 tests 29
and 31 each run in twenty fresh processes.

Final-source results:

| Measurement | Result |
|---|---|
| All item 7 cases | 192/192 pass |
| Tests 29 and 31, twenty fresh processes per concrete case | 140/140 pass |
| Two-epoch graph-release gate | pass, 7.45 GiB under 8 GiB |
| Final explicit aggregate, including all named slow XOR proofs | 251 pass, 3 fail, 1 skip; all 255 cases accounted for |
| Final XOR table, HEAD | 44 pass, 5 fail; exact round trip 14/15 |
| Final XOR table, candidate | 46 pass, 3 fail; exact round trip 14/15; no memory stops |
| MM_grammar full runs | both trees complete 10/10; all ending errors recorded |
| Reconstruction, declared seeds 0–7 | both trees 8/8 complete; candidate peak 5.73 GiB, HEAD peak 6.96 GiB |
| Source-matched full sweep | 4,774 pass, 3 fail, 322 skip, 1 expected failure; all 5,100 cases completed once |

Candidate seeds 0 and 1 first stopped at the unchanged 1,200-second deadline;
both original attempts are retained. Each received one serial retry with the
same inputs, source, deadline and 8 GiB guard. Seed 0's retry completed at
5.65 GiB in 993 seconds and seed 1's at 5.72 GiB in 1,048 seconds. Both retries
match their original seed, configuration, initial atom prefix and all four
before-training reconstruction values. The [complete reconstruction table](final-reconstruction-table.md)
reports all sixteen measurements. Mean after-training error is .1112381346 at
HEAD and .1183237548 in the candidate; no re-baseline is made.
The [AH comparison](xor-ah-comparison.md) and [AI comparison](xor-ai-comparison.md)
are complete. The one [full sweep](full-sweep/run/result.json) is complete;
its [manifest](full-sweep/source-manifest.json) matches the explicit gates and
the frozen final source. Its existing limits remain three workers, 8 GiB per
worker, 24 GiB aggregate, 1,800 seconds per worker and 10,800 seconds overall.
It finishes in 9,282 seconds with no missing or duplicate cases, compile-cache
retries, memory stops or deadline stops. Peak worker memory is 7.24 GiB;
peak aggregate memory is 11.26 GiB. All thirteen prior failures pass
([case-by-case comparison](full-sweep/comparison.md)). There are 52 added
selectors and two removed selectors; the two removed names are the recorded
phrase-admission port, and every added selector passes.

The [three remaining failures](full-sweep/failures.md), all passes in round 3,
compare a stored relation's predicate address `4010033338316264258` with the
registry-based meaning's address `1`. Both operand references agree. They are
two observation-boundary cases (forward and packed) and the generic-grammar
case. These are unresolved failures; their assertions remain intact and their
later checks do not execute. The complete [coverage summary](full-sweep/summary.json)
and [receipt audit](final-receipt-audit.json) retain every result. No second
sweep or repair is made after this measurement.

The [final XOR comparison](xor-final-comparison.md) names every proof and
all fifteen exact repetitions on each tree. The candidate's failures are
XOR_grammar's two gates (0.0 class accuracy; zero of four reconstructions) and
one MM_20M_xor exact round trip at .5. These remain item 6.9's. Both XOR_exact
CLI gates pass on the candidate. The [explicit aggregate](explicit-final3/result.json)
includes the production-width owned-answer case; its source verification is
recorded alongside the aggregate.

The explicit skip is
`test_thinking_kernel.py::TestDepth3RelativeEndState::test_first_trained_read_reaches_depth3_end_state`:
its existing prerequisite is a checkpoint with at least 1,000,000 completed
FineWeb training sentences. The prior [depth-three campaign remains red](../2026-09-27-item7-5-landing/README.md).
The case-level [explicit summary](explicit-final3/summary.json) counts this skip
separately; an earlier progress count had included it among passes.

Item 6.9 is outside this pass: XOR_grammar, sentence-trial comparison, and
the intermittent MM_20M_xor exact-round-trip cause remain measured findings.
The [protected-contract audit](protected-contract-audit.json) confirms no
seed-setting call in any item 7 test, no round-4 configuration change, and
unchanged `NonLayer.forward` and `ConjunctionLayer.forward` bodies from HEAD.

## Before repair: complete XOR baseline

Every proof is individually named in the [HEAD table](baseline-head/table.md)
and [candidate table](baseline-candidate/table.md), including all slow cases.
Both receipts include source manifests and raw worker reports. The incoming
candidate is the 704-file snapshot in [incoming-source.json](incoming-source.json),
unchanged throughout these measurements.

HEAD: 45 passed and four failed across 49 case attempts. The two XOR_exact
CLI gates crash at `int(None)`; both XOR_grammar gates stop at capacity.
All fifteen MM_20M_xor exact round trips pass under the guard.

Candidate: 26 passed, two failed, and 21 attempts stopped by the memory guard.
The two XOR_exact CLI gates pass (MSE 0.0, four of four inputs reconstruct).
XOR_grammar remains red (class accuracy 0.0; zero of four reconstruct).
The 21 memory stops include both configuration-matrix cases, four other slow
reconstruction cases, and all fifteen exact round trips. Each received exactly
one unguarded diagnostic. The six shorter diagnostics pass; the exact repeats
give ten passes and five failures (three exact-match rates of .5 and two of .75).
These are diagnostics, not guarded passes. Their peaks reach about 14.6 GiB.
The intermittent exact-round-trip cause remains item 6.9's.

## Step 1: Z and the first half of AG — measured

The saved probes precede the runtime repair:
[closing identities, phrases and thought dependency](z-before/result.json),
[corrected MM_grammar fixture](z-mm-before/result.json), and
[atomic symbolization refusal](z-symbolization-before/result.json).
The first MM fixture omitted the production store adapter; its setup failure
is preserved in the first receipt. The corrected fixture fails on the native
conceptual-row capacity check, as does the separate symbolization probe.

The initial [repair patch](z-initial-repair.patch) gives closed rows their
LTM occurrence addresses, puts a referenced phrase's point on its LTM row,
and ties grammar predicates by stable slot identities without inventory
admission. Part and implication recovery no longer request thought execution
capability. A full inventory refuses the optional taxonomy transaction without
preventing the completed field from being stored.

The [first affected-file run](z-first-after/result.json) passed 44 cases and
exposed one standalone-writer defect: without the optional taxonomy adapter,
its open predicate reference was unresolved. That saved failure precedes the
[writer repair](z-writer-reference-repair.patch); the rerun of its file, the
five closing probes, and lexical-reference tests passes all 19 cases.
The complete [item 7 run](z-item7-all/result.json) then passes all 175 cases.

The [graph-release gate](z-graph-release/result.json) stops at 8.47 GiB.
Its one unguarded diagnostic runs both epochs and reaches the gate's assertions
at 17.50 GiB. It now fails the action-presence prerequisite: the closing has
cleared the action record. No assertion was changed. That prerequisite and the
memory growth remain for step 9; the clause-recovery exception is gone.

The [post-Z HEAD table](z-head/table.md) again has 45 passes and four
failures; all fifteen exact round trips pass. The [post-Z candidate table](z-candidate/table.md)
has 26 passes, two grammar failures, and the same 21 memory stops. Each stop
has one unguarded diagnostic; the fifteen exact diagnostics give thirteen
passes and two failures. They remain diagnostics, with peaks up to 14.60 GiB.
Every proof is named individually in both tables.

All ten full MM_grammar runs complete on each tree
([all measurements](z-mm-grammar-table.md)); median ending training MSE is
.039155416 at HEAD and .023251077 after Z. All seven concrete test-29/test-31
cases pass twenty fresh processes each: [140/140](z-repeats-29-31/result.json).
These measurements used the unchanged [post-Z source](z-source.json).

The phrase-capacity test now exercises the phrase's LTM owner; it still checks
atomic refusal and invalid width. Both bodies are in
[the port ledger](test-ports.json). No protected numerical or gradient
assertion is changed.

## Step 2: AB — verified in the isolated work copy

All three checkpoint probes fail before repair at the missing owner allocator
([before](ab-before/result.json)). Restoring allocators in a first pass is
necessary but did not suffice ([first rerun](ab-after/result.json)): the
reading publisher still stamped the terminal owner's field as space 0.
The [two-pass repair](ab-repair.patch) and [owner-stamp repair](ab-owner-label-repair.patch)
then pass the three original probes and the structural checkpoint file
([verification](ab-owner-after/result.json)). No checkpoint assertion is ported.

The source-equivalent [isolated copy](work-copy.json) allows later repairs
without changing the ongoing step-1 measurement source. Its source manifests
are retained with each run; changes were reconciled into the working tree
after the relevant measurements finished.

## Saved probes for subsequent steps

[AC, AE and AF](ac-ae-af-probes-before/result.json) reproduce the owned-answer
mismatch, the absent attended identity, and the relation's unresolved third
reference. AC's supplementary probe changes only its lookup to the definition
table and reaches the protected answer assertion. AE's [discovery diagnostic](ae-discovery-before/group-00/result.json)
finds twelve pending word identities, no completed definitions and an empty
native field; a lookup-only port is insufficient on that fixture. AD's two
original counts fail at eight versus two and seven versus three
([reduction](ad-reduction-before/result.json), [provisioning](ae-discovery-before/group-02/result.json)).
The mistaken reduction selector in the earlier batch is preserved as a
collection error; the corrected selector supplies the actual failing probe.

[All five AI probes](ai-before/result.json) fail. The additional
[compiler diagnostic](ai-diagnostics-before/group-00/result.json) names the
new `_clause_state` attribute as the guard that creates a second graph.
The normal-policy diagnostic finds one-row reference metadata beside a
six-row program. Its supplementary file also accidentally collected the
imported unittest class; those extra results, including its unavailable
fixture, are retained and are not counted as repair verification.
The [native predictor observation](ah-before/result.json) reproduces AH's
inventory lookup of an ended-clause identity.

## Steps 3 and 4: AC and AD — verified

The [definition-table reader repair](ac-repair.patch) passes the development
owned-answer test with every original numerical assertion
([verification](ac-after/result.json)). Its object has the order interpret
gives it; no minimum order is assumed.

The [sentence-kind count ports](ad-repair.patch) pass both complete affected
files ([verification](ad-after/result.json)). DEF rows are excluded from the
sentence count, and all later assertions run. All five ports so far retain
both bodies in [the ledger](test-ports.json). Verified AB–AD changes have been
[reconciled](reconciled-through-ad.json) from the isolated copy into the main
tree after the post-Z measurements completed.

## Step 5: AE — affected files and full XOR table measured

The failed probe showed pending words, not completed object definitions.
The [repair](ae-initial-repair.patch) limits deferral to fields with witnessed
object knowledge (or a reservation before native input is available). An empty
field uses the ordinary forced interpret admission. Existing lexical objects
are identified through the definition table, so they do not turn later word
admission into discovery. A field with native object knowledge still owns
its discovered cases.

The complete attended-field, word-admission, definition-integrity and grounded
XOR files all pass ([verification](ae-after/result.json)); this includes the
six test-33 cases and the six original grounded cases. The attended test's
lookup now names its object through the table; all activation and symbol
assertions remain unchanged. Both bodies are in the port ledger.
The complete post-AE XOR tables are measured on frozen source, including
fifteen exact round trips on each tree. [HEAD](ae-head/table.md) has 44 passes
and five failures: its two CLI crashes, two grammar failures, and one exact
round-trip failure (14 of 15 pass). The [candidate](ae-candidate/table.md) has
26 passes, two grammar failures and 21 memory stops. Its six shorter unguarded
diagnostics pass; the fifteen exact diagnostics give thirteen passes and two
failures. These remain diagnostic outcomes, not guarded passes.

## Step 6: AF — isolated, verified, and measured

The binary nested-clause probe fails with fewer than three references
([before](ac-ae-af-probes-before/group-02/result.json)). A second forced
three-slot forest exposes the same unindexed lexical predicate before the
writer: signature lookup is attempted without a predicate address
([before](af-before/group-00/result.json)). The original unseeded
partition-isolation case passes this particular pre-repair attempt; the forced
probes establish the defect independently of chooser selection.

The [repair](af-repair.patch) closes an unindexed lexical predicate as an
unasserted LTM point in either form. The reference needs no inventory row.
Both permanent mechanism tests, the clause-program and acceptance files,
and partition isolation pass ([verification](af-after/result.json)).
The frozen [AF source](af-source.json) resides in the first isolated work copy.
The [complete AF comparison](xor-af-comparison.md) records HEAD's 44 passes
and five failures, and the candidate's 26 passes, two grammar failures and
21 memory stops. Each stop has one separate diagnostic; thirteen of fifteen
exact-round-trip diagnostics pass. Historical diagnostics were serialized
to avoid competing large allocations.

## Step 7: AH — verified and measured

The [native before observation](ah-before/result.json) and the
[forced point-owner probe](ah-point-owner-before/result.json) both fail when
an ended identity is read from the inventory. The [repair](ah-repair.patch)
reads ended points through their LTM rows and ordinary concepts through the
inventory. Relative clauses remain relations, without an invented point.

The permanent point-owner test and the nested-meaning file pass
([first verification](ah-after/result.json)). That batch also preserves two
harness path errors, neither a runtime test result. The corrected
[verification](ah-native-after/result.json) passes the lexical-reference file
and completes the native predictor-context observation: three predictor rows,
zero definition rows, alongside four definitions in the twelve-row store
([raw observation](ah-native-after/native-definition-context.json)). The full
[XOR comparison](xor-ah-comparison.md) is measured on the frozen
[AH source](ah-source.json). HEAD has 45 passes and four known failures,
including fifteen guarded exact-round-trip passes. This historical candidate
has 26 passes, two XOR_grammar failures and 21 memory stops before the AG
repair. Each stop has one unguarded diagnostic: all six shorter cases pass;
the fifteen exact repeats give thirteen passes and two failures. These remain
diagnostics, with peaks up to 14.60 GiB, not guarded passes.

## Step 8: AI — all five original failures pass

All five original probes, plus the adjacent detached-student training-step
check, pass with their existing assertions ([verification](ai-after/result.json)).
The [repair](ai-initial-repair.patch) retains the current numerical root for
a raw differentiable forward, while committed history stays detached; publishes
the current batch's complete word-reference metadata; rebuilds ordinary
sentence boundaries for each batch; initializes clause state before entering
the compiled chunk; and hard-resets the temporary what episode after
provisioning. No test is ported. The boundary diagnostic
[before](ai-c-trace-v2-before/result.json) shows current active words beside
stale sentence IDs; its earlier observer error is retained separately.

The full [XOR comparison](xor-ai-comparison.md) is complete on the frozen
[AI source](ai-source.json). HEAD has 45 passes and four known failures;
all fifteen exact round trips pass under the guard. The historical candidate
has 26 passes, two XOR_grammar failures and 21 guard stops before the memory
repair. Each stop has one separate unguarded diagnostic: the six shorter
cases pass, and thirteen of fifteen exact round trips pass. Diagnostic peaks
reach 14.60 GiB. The second half of AG below measures saved tensors and model
tensors by owner before repairing that memory growth.

## Step 9: AG — ownership repairs and reconstruction campaign verified

The initial owner observer failed because its wrapper did not preserve the
batch method's signature; that harness error is retained
([first observer](ag-owner-before/result.json)). The corrected
[owner measurement](ag-owner-before-v2/result.json) stops under the 8 GiB
guard and completes its one unguarded diagnostic at 17.60 GiB, reaching the
old action-presence assertion. The
[per-closing observations](ag-owner-before-v2.jsonl) identify about 2 GiB
of retained event × property × byte values in `on_counts`, plus the
diagnostic membership cache retaining that graph across ticks.

The [saved-product probe](ag-property-before/result.json) fails, while its
value/gradient comparisons already pass. The
[diagnostic-cache probe](ag-cache-before/result.json) also fails. The
[property repair](ag-property-repair.patch) recomputes bounded reduction
workspaces in backward, preserving neutral-byte ties exactly, and detaches
the diagnostic cache at tick end. All six strict reduction tests (including
compiled execution), the cache probe, and three affected files pass
([verification](ag-property-after/result.json)). The native graph gate still
stops under the guard; its one diagnostic passes at 13.61 GiB.

The graph gate's [port](ag-graph-gate-port.patch) observes the chosen action
record before the closing discards it. Its original non-vacuity and
matching-attempt assertions remain, and new assertions check the final
record is empty. Both complete bodies are retained in the port ledger.

A [separate selected-property probe](ag-selected-property-before/result.json)
shows a read of three selected columns expanding all 513 first. Gathering
before the byte read preserves exact values and gradients
([repair](ag-selected-property-repair.patch),
[verification](ag-selected-property-after/result.json)). This does not resolve
the native graph gate: its one diagnostic still peaks at 13.65 GiB.

The subsequent [storage census](ag-owner-storages.jsonl) and
[allocation-owner report](ag-allocation-owners.json) locate the largest
temporaries in `WholeSpace._predicate_slab` and `_predicate_unit_spans`: a
`[4, 4096, 65792]` Boolean slab, with several 1.004 GiB allocations in the
boundary calculation. The profiler's own memory is overhead; its diagnostic
peak is not a model gate measurement
([profiler receipt](ag-forward-allocations/result.json),
[compressed raw trace](ag-forward-allocations.json.gz)).

The [predicate probe](ag-predicate-before/result.json) passes its dense
boundary-learning reference and fails the three column-storage assertions.
The [repair](ag-predicate-repair.patch) gathers precisely the columns present
in the input, keeps their original identities for boundary learning, and
preserves the global cold-start decision. All six predicate probes pass,
including their dense-reference comparisons, in the final-source
[item 7 run](final3-item7/result.json).

HEAD reconstruction's first three attempts reached the unchanged 1200-second
time limit during concurrent diagnostics. Their partial measurements and
process receipts are retained in `final-reconstruction-head`; no completed
learning outcome is inferred from them. The declared seeds remain 0–7.

The [HEAD baseline reuse check](reconstruction-head-reuse.json) confirms that
all eight completed round-3 HEAD runs have byte-identical runtime source and
measurement driver; each uses PyTorch 2.14.0 and the same 8 GiB guard. Their
maximum peak is 6.96 GiB. As §19 closes the reconstruction baseline, these
completed runs are the reference. The redundant round-4 HEAD rerun and its
queued timeout retries were cancelled, with all partial attempts preserved
separately. The candidate's eight fresh declared seeds also complete under
the unchanged guard, independently of this reuse; all sixteen measurements
are in the [final reconstruction table](final-reconstruction-table.md).

## Two additional AI defects found and repaired during final validation

The first final complete-forward gate fails before its numerical assertions:
AI's ordinary-boundary refresh calls the host layout routine from inside the
compiled forward (`Tensor.tolist()` is unsupported there). The
[failing gate](explicit-final/group-00/result.json) is retained. An earlier
progress message misread completion as a pass; it was corrected when the
failure report was inspected. The packed binary-reference gate passes.

The [repair](ai-compiled-boundary-repair.patch) refreshes at the host entry;
a compiled forward consumes the layout already prepared by its lexical stem.
Verification passes, with no assertion port. The
[superseded validation record](compile-boundary-validation-stop.json) names
all stopped candidate jobs. Their completed and partial outcomes remain in
their original directories and are excluded from final-source measurements.
HEAD measurements remain valid.

The repaired complete-forward gate passes its two backward passes and one-graph
assertion ([verification](ai-compiled-boundary-after/group-00/result.json)).
The provenance graph, identity-candidate and normal-policy checks also pass
([affected checks](ai-compiled-boundary-affected/result.json)). The earlier
affected-file run independently found the same host-only refresh defect in
its provenance case.

The corrected-source [two-epoch graph-release gate](final2-graph-candidate/result.json)
passes at **7.85 GiB** under the unchanged 8 GiB guard (48.51 seconds including
worker setup). Both epochs run, the original graph-free and non-vacuity
assertions pass, and the closing leaves the action record empty. The eight
fresh reconstruction measurements remain required. This intermediate
candidate's receipts use the `final2-` prefix, alongside `explicit-final2`;
both earlier attempts were later superseded by the corrected `final3-`
source below and are not counted as final evidence.

The [post-AH HEAD XOR table](ah-head/table.md) is complete: 45 passes,
four known failures, and fifteen guarded exact-round-trip passes. Every proof
is named. The matching [historical candidate table](ah-candidate/table.md)
is also complete, separately from the final corrected-source table.

The initial full meronomy-file run completed 31 cases before its unchanged
35-minute suite deadline, with the provenance graph failure above and the
remaining slow training cases incomplete. Type-run checks and its native
graph gate pass ([affected-file receipt](ag-predicate-after/result.json)).
The corrected provenance test passes in its fresh worker; the normal-policy
and identity checks also pass ([verification](ai-compiled-boundary-affected/result.json)).
The final full sweep accounts for every ordinary case; unfinished optional
slow-file cases from that earlier affected-file run are not reported as passes.

The [owner comparison](memory-owners.md) records what survives into the next
batch: saved storage falls from 2.605 GiB to 0.338 GiB after the support-read
and diagnostic-cache repair. The largest removed owner is the stage-0
property read (2.008 GiB before, 0.070 GiB after). These are diagnostic
ownership observations, separate from the final guarded peak above.

The next final item 7 run exposes a second AI regression: the unaligned
identity reader returns an all-`-1` slab where its API previously returned
`None` ([saved failing assertion](explicit-final2/group-35/result.json)).
The fixed pipeline still needs complete batch metadata, but that metadata is
not itself a native identity carrier. The
[reader repair](ai-native-id-repair.patch) restores `None` when no word or
object has a native row, retaining unknown entries within an existing native
carrier. The entire eight-case unindexed-relation file passes with every
assertion unchanged ([verification](ai-native-id-after/group-00/result.json)).

[Superseded candidate validation](native-id-validation-stop.json) retains its
partial measurements and the failed assertion. No completed result is
selected or discarded to obtain a pass. The complete item 7 set is rerun
before the expensive final measurements restart. Final receipts from the
corrected source use `final3-`; earlier source receipts remain separate.

All **192 item 7 cases pass** on the corrected source
([complete rerun](final3-item7/result.json)). The native-ID reader file and
normal-policy regression also pass ([verification](ai-native-id-after/result.json)).
The corrected-source graph-release gate passes again, now at **7.45 GiB**
([gate](final3-core-gates/group-00/result.json)); the complete-forward graph
gate passes too. Final explicit gates and repeated measurements are recorded
at the top of this receipt. The runtime stayed frozen through the full sweep.

A [harness probe](diagnostic-environment-before.json) found that historical
unguarded repeats inherited the parent thread environment, while guarded
workers set their thread limits. Those earlier diagnostics are retained and
are not repeated or promoted to gates. Future diagnostic repeats use the same
worker environment as the gate, with only its memory guard removed. The
actual guarded test inputs, model configurations, deadlines and 8 GiB limits
are unchanged ([dispatch record](diagnostic-environment-dispatch.json)).

The complete [post-AF candidate table](af-candidate/table.md) records 26
passes, two XOR_grammar failures and 21 guard stops. The fifteen exact
round-trip diagnostics give 13 passes and two failures. Every proof and every
attempt remains named separately from guarded results.

All eleven [core gate groups](final3-core-gates/result.json) complete without
failure, including three native packing-parity cases with their original
tolerances and ambient initializations. The missing-checkpoint skip is counted
separately in the explicit aggregate. The [production-width owned-answer case](final3-ac-production/result.json)
also passes; it is marked slow and therefore needed this explicit run in
addition to the earlier development-width check.

The [final HEAD XOR table](final-xor-head/table.md) is complete. Its fifteen
exact round trips give fourteen passes and one failure; the known CLI crashes
and XOR_grammar failures remain visible. The [final candidate table](final3-xor-candidate/table.md)
and the [eight candidate reconstruction measurements](final-reconstruction-table.md)
are also complete, including all failures and original deadline stops.

The final [MM_grammar campaign](final-mm-grammar-table.md) completes **10/10
on both trees**, with all twenty 900-epoch measurements recorded. Median ending
MSE is **0.0696157217 at HEAD** and **0.13429676 on the candidate**. This campaign
does not reproduce the candidate's earlier lower median. The capacity failures
are gone; no learning value is selected, tuned, or re-baselined.

The final [test-29/test-31 campaign](final3-repeats-29-31/result.json) passes
**140/140** concrete cases in fresh processes. The six current-round ports
retain both bodies, and [their audit](final-port-audit.json) verifies that every
recorded new body matches the current file. The [source audit](final-source-audit.json)
finds 21 changed runtime/test files relative to the incoming round-4 candidate
and **no configuration changes**.

Reconstruction's first two candidate attempts stop at the unchanged 1,200-second
deadline, with peaks of 5.02 and 5.66 GiB. Their partial measurements and process
results remain in [the primary receipt](final3-reconstruction-candidate/processes.json).
The declared campaign completed all eight seeds. Only the two incomplete
deadline stops received one serial retry, using the same seed, driver, deadline
and 8 GiB guard. Completed numerical results were not repeated or selected.

Seeds 2–5 subsequently complete under the guard, with peaks of 5.73, 5.35,
4.90 and 5.68 GiB. Historical AH/AI tables were released after the two-worker
primary campaign ended, so at most one historical table overlapped one serial
retry; the full sweep waited for all of them
([dispatch record](historical-xor-primary-release.json)). No running test was
stopped by that scheduling change.

## Review hand-off checks

The [final audit](final-receipt-audit.json) verifies fourteen complete XOR
tables (49 named attempts and fifteen exact repetitions in each), both
eight-seed reconstruction campaigns, twenty full unseeded MM_grammar runs,
140 fresh-process test-29/test-31 completions and the sweep's exact coverage.
The sweep and explicit gates share the frozen 711-file source and fixture
inputs. The new predicate-reference failures remain unresolved for review.

The final documentation-link check passes **158/158**, including `todo.md`
([receipt](final-doc-links)).
`git diff --check` reports only the existing extra blank line at the end of
Claude's item-7 spec, line 2371; that document is preserved. basicmodel remains
at `1678ee1f` and WikiOracle at `850798fc`. Nothing is staged, committed, pushed
or bumped by this pass; nanochat is untouched.
