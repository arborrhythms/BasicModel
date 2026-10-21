# Closing reads: causes, repairs and rejected attempts

These are mechanism repairs on the uncommitted step-1b branch. They do
not change R + A, the forced decomposition scope, certificate budgets or
the original initializations. Each run keeps its own outcome in
[runs.md](runs.md).

## 1. Can a definition be mistaken for an equality answer?

| Observation | Result |
|---|---|
| Concrete retained edge | In `closing-55.store.jsonl`, object 33 points to word 12 through Part 34 |
| Collision | Word 12 is a singleton; the old Part reading classified the same edge as both definition and equality |
| Consequence | An identity query could return the defining word as the object's equal |
| Repair | A definition reading excludes the equality reading; a genuine singleton object on the right still supplies equality |
| Before repair | New native regression fails in native-56 |
| After repair | Passes in native-57 and native-70 |

Native-50's c/e question ended bound at a random initialization that was
not saved. The definition/equality collision is independently reproduced;
it is **not claimed as an exact replay or proved sole cause of native-50**.
That historical failure and the limit on its diagnosis remain visible.

## 2. Why did the selected declarative closing fail to start its episode?

| Observation | Result |
|---|---|
| Saved native failure | Native-61: only one of the two required c/e episodes starts |
| Exact replay | Native-63 restores `closing-61.initial-rng.pt` and fails the same assertion |
| Read-only trace | The raw open copular occurrence at row 2 uses row 2 itself as its forward-reference witness, returning its own complement, symbol 25 |
| Consequence | The closing reader treats the selected open formation as already bound and suppresses its required episode |
| Rejected repair | Excluding a row's own witness from every working read |
| Counterexample to that repair | Native-67 passes c/e and forward premise order but fails reverse premise order, where an ordinary copular read legitimately uses its complement |
| Final repair | `ThoughtClosing.closing_question` reads the raw occurrence and calls the common reference reader with only the currently closing occurrence excluded |
| Ordinary working reads | Retain the public witnessed copular interpretation; raw occurrences are not rewritten |
| Current evidence | Whole §14 completes in native-68 and native-70: 12 passed, one artifact prerequisite skip, with c/e initialization 61 and both saved premise initializations 32 |

The [exact replay trace](closing-reference-63.jsonl) records the self
witness before repair. The initial broad run with the rejected global
guard, native-65, was interrupted after native-67 exposed its reverse-order
regression: **180/185 completed, not acceptance**. It is not a timeout or
a successful full run.

The final unit regression checks both boundaries:
`test_closing_excludes_its_own_witness_without_restricting_copular_reads`.
The selected open close remains open; ordinary copular interpretation and
an earlier matching question can read the witness. The original stored
occurrence remains open. Cyclic identities are not prohibited.

## 3. Why did ten old thinking fixtures fail after §14 passed?

| Failure group in native-68 | Port or production repair |
|---|---|
| Four later-binding variants call retired `fill_forward_references` | Append a witnessed copular occurrence and read the earlier question without upserting it |
| Modus ponens and three event-time fixtures use retired `REL_IMPLIES` | Supply two native situation references and their conditional Part |
| Two-query chain assumes physical inference row 2 | Assert the returned native Part evidence, arguments and proof witnesses |
| Chooser-credit contrast assumes forward conceptual part lookup has no shared-graph proof | Use the reverse part query as the wrong alternative; retain the chain, credit direction and reconstruction/answer comparison assertions |
| Conditional event-time lookup omitted by the Part conversion | Restore reads of the two endpoint situations' `.when`, with bounded read work and explicit unknown/reversed rejection |

The conditional repair is production behavior, not just a fixture update.
A rule's insertion time cannot stand for the endpoint times. Composed
native situations supply those times; each endpoint read consumes the
existing record budget. Unknown endpoints remain incomplete, and reversed
endpoints do not produce a conclusion. Atomic situation symbols retain
their atemporal conditional reading.

Native-70 then completes **206 passed, four failed, one skipped**. All
four remaining failures are the later-binding variants: the question is
in stream 0, but the new copular witness had been appended with no stream.
The shared reader correctly refused that witness. Giving the fixture's
later occurrence `stream=0` preserves the intended same-stream setup.
[Native-72](native-72/report.html) completes the whole thinking file:
**26 passed**, including the original no-open-slot, correct-reference,
witness, source-unchanged, chain-credit and temporal assertions.

## 4. What remains a failed gate?

Both unchanged decomposition lengths still fail in both gain arms.
The [eight question tables](decomposition-71.md) and
[symbol-type question](../../plans/2026-10-19-bind-thought-operation.md#69-implementing-68-a-missing-symbol-type-source-codex-2026-10-20)
record the separate expression-reader blocker. The passing §14 file and
thinking ports do not replace those proofs or the pending full sweep.

## 5. §12 review: why did the ordinary introduction exhaust MM_grammar?

| Question | Evidence / answer |
|---|---|
| Native-119 failure | The close required a bank seat for the ordinary new referent |
| First repair attempt | `review3/capacity-02.jsonl`: the root lost its seat, but the description operand gained one; this moved the leak rather than removing it |
| Rejected enlargement | `review3/mm-03` at 8192 seats was interrupted, not passed; the fixture is restored to eight seats |
| Mechanism boundary | Description and determiner operations do not require new operand identities. A relation argument does; inability to seat it is an atomic loud failure |
| Original regression | `review3/mm-05`: native-119 initialization, eight seats, original 900-epoch/.20 certificate passes; all ten close regressions pass |

This does not introduce recurrence minting. An ordinary occurrence has its
own LTM1 address and point; neither its introduction nor its description
requires a new symbol. The chooser menu does not inspect capacity.

## 6. §12 review: what was missing from an RNG-only arrival replay?

| Capture | What it establishes |
|---|---|
| `review3/arrival-07` → `arrival-08`, forward | Same initial parameters, buffers, post-constructor Torch RNG, final parameters and Part pairs; same line-391 failure at row 6 |
| Limitation of that pair | Occurrence addresses differ. This is not yet an exact native-address replay |
| Missing owner | `MathChainCorpus` constructs `MathProblemGenerator(seed=None)`, whose private `random.Random(None)` is independent of the saved Python/NumPy/Torch process RNGs |
| Consequence | The independently shuffled corpus changes `source_manifest.content_sha256`, which participates in the staged document address even after the certificate stages its own eight documents |
| Expanded capture | `review3/arrival_capture.py` saves/restores each private generator state before presentation, alongside all three process RNGs and initialized parameter/buffer hashes |

The full capture starts at `arrival-11`. Earlier saved-state passes are
not promoted to coverage of the live failures. Runs using the explicitly
recorded `PartClosing_6903713c7.py` diagnostic override reproduce the
reviewed close; they are not acceptance runs of the new candidate.

### 6.1 Forward failure: a named intermediate identity was skipped

| Check | Evidence |
|---|---|
| Original reviewed close | `arrival-12`, fresh initialization, forward failure at line 391 / row 6 |
| Replay | `arrival-13`: initial parameters/buffers/RNG/source manifest, final parameters/RNG, all LTM1 rows and native addresses match |
| Bound reference | Symbol 101, through heads 100 → 99 → 24 |
| Retained definition | Part 102 = `(24, 91)`; word 91 is the supplied target `three` |
| Faulty answer read | `arrival-15/failure-expression.json`: the structural reader flattens all those heads to symbol 21, bypassing the already named symbol 24; no incomplete-read flag |
| Repair | The structural view retains its flattening, and exposes each direct declared head for the identity reader. The answer reader checks each identity before following the next head |
| Regression | `test_answer_read_checks_a_named_composite_before_descending_its_head`, passed in `unit-16` |

The new capacity candidate also exposed a separate missing definition: a
surface phrase minted as a relation argument could leave its description
unminted and lose that description's word link. The close now carries the
lexical head's origin to the phrase's mint. Its regression passes in
`unit-16`. Definition handles are assigned in derivation order, replacing
the iteration over a set of process-specific object IDs. In the baseline
replays that set permuted some DEF handles, although the pairs/evidence and
all LTM1 rows agreed; `exact-replay-comparison.json` records that limitation.

### 6.2 Reverse failure: the global assertion includes an unrelated variable

| Check | Evidence |
|---|---|
| Original reviewed close | `arrival-10`, reverse failure at line 398 / row 5 |
| Replay | `arrival-14`: same initialized model and native LTM1 rows/addresses, same assertion and unresolved reference |
| Failing row | Stream 2's unverified constituent, with role 2 open on word 14 (`is`) |
| Pending reference specified by the certificate | Word 12 (`x`), in the separate question at row 6 |
| Conflict | The live explore suffix may open another name. The forward arm explicitly excludes those unrelated variables; the reverse arm asserts over every non-estimate row |
| Decision (Alec, §13 / §14 round) | Keep the universal over every non-estimate row, but reject forward-name IDs for x only; also require every pending x row's referent slot to be filled. Row 5 / word 14 (`is`) is outside the certificate's own forward-arm scope. D2 is applied in the §14 round, with arrival-14 and fresh-both-reverse-5 replays next |

No assertion, forced scope, budget or initialization changed in this
diagnosis. The separate raw-graph count was subsequently ported to the
working read in §6.5 and accepted in plan §13. After accepting the D1
probe, Alec's §14 order applies D2 before §10.2. It checks the working
forward names in every row, and uses raw forward names only to identify
the rows whose pending x referent slots must now be closed. Publication
and the six-fresh-per-order acceptance run remain pending.

### 6.3 Fresh forward initialization 2: a changing name key and a surface read

| Check | Evidence / cause |
|---|---|
| Original and replay | `review3/fresh-forward-2` → `arrival-20`, source `2fa238049`; initialized model, all native LTM1/Part rows, final parameters/RNG and source manifest match |
| First failure | Row 2 names lexical word 15; between premises another kept reading admits object 29 with DEF `(29,15)`. The arriving row 35 then records 29 as its forward name, so the exact posting lookup for 15 cannot find it |
| Name repair | The journal's forward name is the admitted lexical word ID when present, not the word's changing interpreted object; nonlexical programs retain their concept identity |
| Regression | `test_pending_name_stays_the_word_after_an_object_is_admitted`; `unit-22`, 63 passed |
| Native replay after name repair | `arrival-23` fills row 2 from row 35 but fails line 391; the unchanged strict answer check exposes a second read defect |
| Second cause | Root 78 is the surface pair `(75,77)`, with direct `surface_argin1`/`surface_argin2` occupations. Node 75 is itself a surface composition containing a named object. The reader wrongly excludes the catalogue's surface rule unless every marker leaf is objectless |
| Surface repair | Keep the objectless-marker priority; otherwise admit an unambiguous operation from its recorded argument occupations, including surface. No edge metadata or fixture recognition |

### 6.4 Fresh forward initialization 5: the arrived complement remains open

| Check | Evidence |
|---|---|
| Exact replay | `review3/fresh-forward-5` → `arrival-21`, source `2fa238049`; every captured store field, initial parameters/buffers/RNG and manifest agrees |
| Pending row | Row 30, stream 7, forward word 15; after arrival it is bound to row 54's occurrence |
| Arriving witness | Row 55, equality with forward word 15 and complement at row 54 |
| Complement | Row 54 is a three-slot expression with references `(('sym',14), ('sym',4010033338316264258), None)` and forward word 59, rather than a completed one-slot referent |
| Consequence | Filling the pending name succeeds, but the strict supplied-referent identity check is false. A separate open variable cannot be silently treated as the supplied target |

This mode is retained for the certificate-scope review, alongside §6.2.
It has not been suppressed, regraded or replaced by a passing initialization.

### 6.5 Fresh reverse count failure: the certificate traverses raw storage

| Check | Evidence |
|---|---|
| Candidate | `1be79c9d52fc358f4c33f3313412141e2ca3b8a2` |
| New failure | `review3/fresh-both-reverse-1`, line 422; every working forward reference has resolved, but the raw-graph count is zero |
| Read mismatch | `carries_three` traverses `store.refs[child]`, bypassing `meaning_of(child)` and its witnessed binding |
| Architectural rule | Identity is computed on read and never written back to the source LTM occurrence; an open raw reference is expected to remain raw |
| Row evidence | `review3/raw-working-count-1.json`: the same eight sentence roots have no raw path and eight working paths to a retained supplied-answer identity. Every working value is from a read the native run already made |
| Required port | At an LTM occurrence, traverse the references from its working `meaning_of` value. Keep the same eight roots, Part edges, cycle boundary, `>= 2` assertion, forced scope and budgets |

This port does not settle the unrelated-variable assertion (§6.2) or the
unresolved explored complement (§6.4). Those remain separate decisions.

`review3/arrival-27` reproduces the count failure exactly at the saved fresh
initialization (`raw-count-replay.json`); `arrival-28` passes after the
working-read port, with all eight selected roots bound correctly. No
assertion, root selection, forced scope or budget changed.

### 6.6 D1: which arrival trial wins, and is an answer score missing?

Source **`266189edab25985ac1d000b80553ee398f873b69`**, unchanged production,
certificate, forced scope, budgets and keep law. The diagnostic wraps the
same `_sentence_path_cost` and `runBatch` boundaries as the step-0 keep
probe; it inspects already-produced readings and cost registries, without
an additional model read or random draw.

Both exact native replays **fail** the unchanged forward certificate.
`review3/d1-forward-4-before` takes 39.1 seconds in the worker;
`review3/d1-arrival-21-before` retains its own bounded-run report. The
[validation](review3/arrival-keep-validation.json) verifies identical
initial parameters, buffers, source manifest, thread count and complete
RNG states (including the private corpus generator), and identical final
LTM rows, Part rows, occupations, word identities, parameters and RNG.
Type collections are compared as sets. This is observation of the saved
failures, not replacement by different initializations.

The two arrival batches, both trials in all eight streams, are in
[arrival-keep.jsonl](review3/arrival-keep.jsonl): `readings` with refs/open,
clause/composition graphs, reference plans and the named cost registry;
`credit.costs`, `credit.components`, `keep_costs`, `keep_decision`,
`deciding`, `departure`. There is
[one table per document](review3/arrival-keep-tables.md). The full
per-run logs also retain the first premise batch.

**Question 1: what happened at `fresh-both-forward-4`, stream 6?**

| Trial | Arrival reading | R | A | R + A | Kept / deciding |
|---|---|---:|---:|---:|---|
| Greedy (`exploit`) | x is open; complement is a completed child whose declared head is `three` (word 72) | .0360673778 | 0 | .0360673778 | Discarded |
| Explore | x is open; complement is symbol 14 (`is`). `three` is attached under `adverb` as a modifier, **not** left open and not the complement's head | .0300561469 | 0 | .0300561469 | **Explore; reconstruction** |

Departure is compose round 13. The kept equality becomes row 51 and the
earlier pending row 26 is filled with symbol 14. Its `_forward_references`
are empty, but the unchanged identity assertion correctly fails. The fourth
review's description conflates this case with arrival-21: the three-slot
open-`three` complement is **not** present here. The full composition in
the probe preserves the distinction.

**Question 2: what happened at `arrival-21`, stream 7?**

| Trial | Arrival reading | R | A | R + A | Kept / deciding |
|---|---|---:|---:|---:|---|
| Greedy (`exploit`) | x is open; complement is a completed child whose declared head is `three` (word 59) | .0360673778 | 0 | .0360673778 | Discarded |
| Explore | x is open; complement is a relative child `(is, part, open three)` | .0300561469 | 0 | .0300561469 | **Explore; reconstruction** |

Departure is compose round 13. The kept relative child is row 54, with
`_forward_references ((2,59),)`; equality row 55 supplies its occurrence
to pending row 30. This is exactly §6.4's incomplete complement. A does
not charge it. The R difference is **.006011230871081352** in both cases;
the trained reconstruction term is `reconstruction.free_bytes`, and
`forward_discard` / `content` contribute zero. E differs but is excluded
from the comparison, as required.

**Question 3: is this the step-0 supplied-answer judge defect?**

| Read / contract | Evidence |
|---|---|
| Supplied targets | All 16 entries are `None` in each replay, including every arrival. `ChainDocument(str(i), sentences)` supplies neither `question` nor `answer`; `ChainDocument.target` therefore supplies no `ReferentTarget` |
| Sentence answer reader | `_sentence_reader_error` returns `None` because no supplied target exists; `_sentence_answer_error` returns before invoking the referent judge |
| Reference-closing credit | Both deciding trials have `reference_plans.items=[]`, cost 0. These journal-open operands are not order-1 `ReferenceClose` requests |
| Final A | `SentenceCredit.components` maps absent answer cost to zero, plus the zero reference-closing credit. Thus **A=0 is actually recorded**, not an unreported surface or binding loss |
| Previous step-0 cause | Its question had a supplied referent target but A used the wrong surface reader. The repair selected the existing binding judge for that supplied target; it did not introduce a target on unlabelled premises |
| Existing boundary | `Data.what` makes missing supervised targets unavailable. Thinking §14.2 explicitly records zero answer cost without an answer; the accepted step-0 review keys the judge on the supplied answer's kind |
| Required decision | A new cost for an unresolved input role, or a different arrival-certificate contract, must be declared. An open-role charge alone cannot repair `forward-4`, whose complement is complete but has the wrong head |

**Disposition: D1 stop, as requested when the existing R + A judge
legitimately prefers the alternative.** A is indeed zero on the open
complement; the probe establishes that no supplied answer was lost or
mis-scored. Under the current objective these are unlabelled premise
readings and lower R selects explore. Treating the certificate's expected
word as a new loss-side target, or adding an unsupervised open-role penalty,
would change that contract. No such change has been made. This conclusion
is not a claim that the kept semantic reading is correct or that the
certificate passes. The two failures stand.

D2 is decided but not applied because the requested order stops at D1.
The twelve `fresh-final` captures, the first eight-file set, §10.2's eight
changes and its own gates/receipt, the second eight-file/decomposition set,
and publication to both mains remain explicitly queued. No fresh run,
rescue training, second sweep, fixture change or assertion port followed
these probes.

### 6.7 D1 after the decided §10.2 round

Source `f9faae610f35d5ca660d0ac49249637cb23e7adf`. D2 was applied in
`eb75de56a`; both saved reverse replays passed. The eight §10.2 changes
then produced the candidate above. The forward assertion, forced scope,
budget, objective and keep rule remain unchanged.

Both formerly failing forward initializations now **pass**. The retained
[validation](../2026-10-20-item3-5-inline-lowering/arrival-keep-validation.json)
checks identical initial parameters, buffers, source manifest, thread
count, all three process RNGs and the private corpus generator. The
different closing mechanism changes the subsequent native trajectory;
final stores are not claimed identical to the old failing stores.
[Both trials](../2026-10-20-item3-5-inline-lowering/arrival-keep.jsonl)
and [one table per document](../2026-10-20-item3-5-inline-lowering/arrival-keep-tables.md)
are retained, including both premise batches in the individual run files.

**Question 1: fresh-both-forward-4, original failing stream 6.**

| Trial | Arrival reading on §10.2 | R | A | R + A | Keep |
|---|---|---:|---:|---:|---|
| Greedy | x open; completed complement headed by three (word 56), whose derivation reference survives the mint | .0360673778 | 0 | .0360673778 | Tie: greedy |
| Explore | Same completed complement in this stream | .0360673778 | 0 | .0360673778 | Discarded by the existing tie rule |

**Question 2: arrival-21, original failing stream 7.**

| Trial | Arrival reading on §10.2 | R | A | R + A | Keep |
|---|---|---:|---:|---:|---|
| Greedy | x open; completed complement headed by three (word 45), whose derivation reference survives the mint | .0360673778 | 0 | .0360673778 | Tie: greedy |
| Explore | Relative child with a bound occurrence and symbol 14 (`is`); no open three role in this new reading | .0360673778 | 0 | .0360673778 | Discarded by the existing tie rule |

No answer target was introduced and A remains zero. The old failures are
not regraded: the new source passes their exact initializations. Twelve
fresh initializations, six per order, are the separate next gate; these
saved passes do not replace it.
