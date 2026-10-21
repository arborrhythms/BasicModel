# Step 1b §10.3: medium-term token identity

**Current §21:** the participant-position repair and Alec’s seven check
dispositions are implemented for a new freeze. The captured forward-five
regression passes with the original parameters and RNGs; the preceding
stopped series remains historical. The [§21 receipt](../2026-10-21-item3-5-review11-final/README.md)
owns the new source audit and gates. No earlier evidence is promoted.
Claude review and publication remain pending.

**Current §20 source `f56fdf8ff` is stopped for Alec:** forward first draw
5 binds x to the captured space symbol instead of three. The
[wrong-binding receipt](../2026-10-21-item3-5-review10-observer/wrong-binding.md)
retains the resolver and keep evidence, with unchanged source and assertion.
Remaining gates, Claude review and publication are pending. The evidence
below retains its original source attribution; nothing is promoted.

**§18 / §10.4 update:** accepted MTM remains `eac2fe87b`. The
[third frozen source `3d453e6bb`](../2026-10-21-item3-5-anonymous-bind-v3/README.md)
completes all requested gates: 10/12 fresh first draws pass, 48 focused
regressions pass, and the eight-file set records 110 passes, three failures
and one existing skip. Both original decomposition arms and both eight-seat
MM draws pass. Reverse draws 2 and 3 exactly replay; their retained
order-0 complement is absent from the next bind menu. A nested occurrence
witness is repaired in intermediate source `a93f93c0e`; its
[review follow-up](../2026-10-21-item3-5-anonymous-bind-review-followup/README.md)
also repairs native-arrival row materialization, audit/copy isolation and
historical-site frame access. Pooled-candidate ownership and bootstrap
zero-R remain open. The first named-site reading loses on reference credit;
Alec deferred the forcing/objective decision to review time. Claude's
completed review of `3d453e6bb` says not to land. Every failed run is retained,
and no gate is promoted to the intermediate source. Resolving the review
findings and failures precedes the next freeze and publication. The §17
evidence below is historical.



Current: Claude's seventh review (bind plan §17) accepts the MTM
implementation and the original ports. Finding A is repaired: re-closing
an occurrence first detaches its previous membership in both `open` and
`bind`. Finding B's four occurrence-view assertions are ported exactly as
authorized. Source is frozen at
`eac2fe87b2e2f0d24ee16fe8250dec37e59abbc7`. Its fresh gate is stopped at
11/12 first draws: six forward passes, three reverse passes, two reverse
failures and one reverse draw unrun. Both failures reproduce exactly with
the keep probe. [Draw 5](reverse-5-failure.md) exposes the R-only arrival
reading concern in the reverse arm; Alec's contract/mechanism decision is
needed. No production or assertion change follows the freeze. The other
gates, Claude's review and publication remain pending; earlier-source
results are not promoted.

## Implementation

`ConceptualSpace.mtm` owns a bounded `MediumTermMemory` with default
`mtmCapacity=64`, also declared in the XML schema and Params. Each entry
retains only occurrence pointers, the introducing derivation kind, access
count and the model tick decoded from `RowTime.stamp`. A standalone
occurrence with no retained derivation keeps `kind=None`; no kind is
invented or seated for it. There are no parameters or retained vectors.

`ReferenceClose.commit` opens ordinary tokens and appends selected binds,
including same-close `('new', n)` aliases. Explicit native individuals do
not enter the pool. Reads key mentions by the pool's introduction, use its
row point as the baseline, and compute splits/merges without changing MTM.
Candidates still come only from the frames. The oldest access is evicted;
a bind to an evicted occurrence starts a new token at the new mention.
Row reclamation calls `forget`, removing a use pointer or the whole entry
when its introduction is reclaimed.

Architecture's STM paragraph and Params describe the pool. Recurrence
promotion, candidates outside the frame window, cross-document identity,
LTM layout, objectives, forward assertion and capacity gates are unchanged.

## Development evidence (not final-source certification)

| Run | Outcome | Interpretation |
|---|---|---|
| `grouping-01` | 4 passed, 7 failed | First MTM baseline read passed an occurrence tuple to `point_of_row`, which takes the address key; fixed at the read boundary. Original failure retained. |
| `grouping-02` | 10 passed, 1 failed | Both unchanged merge assertions passed; the transitive case then hit its retired four-bank-individual assertion. The authorized four-introduction-address port follows. |
| `ports-03` | 114 passed | MTM's first nine cases, reference-close, bind-thought G1–G3 and episode, native identity rows, lexical ports, 14 Part-closing cases and determiner closing. Exact expected antecedents and semantic checks preserved. |
| `forward-04` | 11 passed | MTM regressions including eager/compiled close isolation and the complete real forward body with pool methods made inaccessible; compiled capture is fullgraph. |
| `working-groups-05` | 4 passed | Both merge certificates, LRU over all retained rows, and a separate check that merging working groups never merges pool entries. |
| `occurrence-06` | 4 failed | Historical: all stopped at the old `identity is None` assertion, before §17 authorized its port. |
| `reclose-07-red` | 4 passed, 2 failed | Both new regressions reproduce stale membership in `bind` and `open`; the four authorized occurrence-view ports pass. |
| `reclose-08` | 20 passed | Reassignment and eviction repaired; the full MTM file, four occurrence-view cases and both merge certificates pass, including eager/compiled checks. |
| `reclose-09` | 2 passed | The two reassignment regressions additionally assert that A's occurrence list has already detached the use before eviction. |

[Replay tables and full reads](working-groups.md) show one group where
`working-groups-08` showed four and six. The outside occurrence remains
inaccessible. Every failed run, log and bounded source manifest is retained.

## Seventh-review repairs and port

Both `open` and `bind` call `forget(occurrence)` when membership changes,
before assigning the new token. Same-token touches preserve occurrence
order and increment the access count. The regression re-closes one
retained occurrence from A to B, then evicts A at capacity two; the B
mapping and occurrence list survive. Its companion covers re-closing a
bound use as a new introduction. Both preserve row content, derivation,
Part state and bank; each kept commit is checked independently for writes.

In all four `test_occurrence_candidates_use_retained_part_participants`
cases, only `context['identity'] is None` becomes
`context['identity'] == address`. Predicate, selected antecedent,
participant prediction, content and unchanged store/bank checks stay
verbatim. This is the additional port explicitly authorized in §17.

## Frozen-source evidence and stop

[Source manifest](frozen-source.json) pins production, test and data trees,
the unchanged arrival/MM certificate blobs and all measurement helpers.
The [fresh ledger](fresh-final.md) verifies 11 distinct parameter
fingerprints, complete first-draw state and both trials in both batches.
The [source audit](source-audit.json) verifies all 824 manifest files
against the frozen Git blobs in all 13 runs, including the two exact
replays, plus unchanged measurement helpers. All draws use the frozen hash;
the two saved failing replays are additional diagnostics, not fresh draws.

| Required gate on `eac2fe87b` | Result |
|---|---|
| Six fresh forward initializations | 6 passed, unchanged strict assertion |
| Six fresh reverse initializations | 3 passed, 2 failed, draw 6 unrun |
| Eight-file set | Not run; stopped |
| Both decomposition arms at original snapshots/seeds | Not run; stopped |
| Eight-seat MM, rng-119 | Not run; stopped |
| Eight-seat MM, same captured fresh draw | Not run; stopped |

[Reverse draw 2](reverse-2-failure.md) fails D2 at row 5/document 2:
explore's second-premise equality wins on R (.0324606411 versus
.0360673778; A=0), adds a second x complement, and the existing ambiguity
rule leaves x open. It was exactly replayed and diagnosed before draws
3–5 continued. [Reverse draw 5](reverse-5-failure.md) fails at row 1/document
0. Its first premise also keeps a differently headed `x is three` reading
on R (.0300561469 versus .0360673778; A=0); its second premise adds another
conflicting x equality. Its exact replay matches too. Complete readings,
R/E/A, keep decisions and retained state are linked from each diagnosis.

Both are semantic failures, not ports. Neither is a reproduction of the
forward assertion; all six forward runs passed. The R-selected arrival
in draw 5 nevertheless reaches the reading concern behind the requested
stop rule. Codex stops for Alec's decision, without changing the assertion,
forced scope, budget, objective, keep law or ambiguity rule. Every failure
is retained. The carried source has no new learning claim or second sweep.

## Remaining sequence

Resolve the surviving failures under contract/mechanism review before
finishing the source-specific gates. A source repair would require a new
freeze and evidence on that hash; no result here can be promoted to it.
Claude's review of the frozen source and receipts precedes publication to
both mains, the WikiOracle bump and the October 20 docs. Publication is
held; BasicModel's local receipt commit does not land step 1b.
