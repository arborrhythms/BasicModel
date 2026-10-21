# Step 1b §10.3: medium-term token identity

Current: Claude's seventh review (bind plan §17) accepts the MTM
implementation and the original ports. Finding A is repaired: re-closing
an occurrence first detaches its previous membership in both `open` and
`bind`. Finding B's four occurrence-view assertions are ported exactly as
authorized. The source is ready for freezing; final-source gates have not
begun and earlier-source results are not promoted.

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

## Remaining sequence

Freeze the source in a commit. On that hash collect twelve fresh arrivals (six per
order, all first draws and full captures), the eight-file set, both
decomposition arms, rng-119 and the same captured fresh eight-seat MM draw.
Any failure is diagnosed from its capture with the keep probe before a
change. The fresh runner stops on its first failure for diagnosis. An
R-tie-broken forward reading returns the contract question to Alec, as §17
requires; no assertion, scope, budget, objective or keep law changes. The copied measurement harnesses add MTM state to their captures;
they have not run final gates yet. No second full sweep.

Update the final receipts and status by source, then obtain Claude's review
of implementation, ports, gates and docs before publication to both mains.
The WikiOracle submodule bump and clean BasicModel landing remain pending.
