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
| Decision pending | [Exact rows and proposed scope correction](review3/reverse-scope-question.md) |

No assertion, forced scope, budget or initialization has been changed.
The reverse raw-graph count also reads stored references instead of the
working identity read; its disposition remains open. Publication and the
six-fresh-initialization acceptance run remain pending.

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
