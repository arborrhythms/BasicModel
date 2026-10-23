# Step 2b storage: preserved bound-word terminal contract

2026-10-22. Current storage-only 2b receipt, governed by §36 and §§39–65.
Work stays on main. The [full review-repair receipt](../review-repair/README.md)
records the storage implementation and earlier attempts. All failed runs,
withdrawn sources and reviews remain retained. Nothing is promoted.

## N1: narrow word-form repair

[Claude's HOLD on `867452a79`](../word-form/claude-review.md) accepts B1–B3,
the §41 dispositions and the earlier ports, but correctly identifies that
§64 made all bound word rows spellable. That changed terminals and derived
form reads; the earlier claim that these were unchanged was incorrect.
Its 1,100 passes/nine existing skips remain evidence only for that withdrawn
source, which was not published.

§65 takes the recommended narrow repair under §39, preserving the existing
bound-word terminal contract without reversing an explicit Alec decision.
`word_surface_for_row` returns `None` for a word with an object, even if a
second definition names that word. Its native form still comes from its
own perceptual parts. Objectless words and ordinary objects retain their
prior spelling paths. No new terminal or bind menu rule is introduced.
The original failed RNG passes all three joint-objective presentations
with this repair ([diagnostic replay](../held-content/rung0-narrow-replay/run.json));
it is never used to initialize a gate.

**Exact assertion port in the same-round regression:** the loop's
`assert owner.word_surface_for_row(row) == b'12'` for word and object is
replaced by `assert owner.word_surface_for_row(owner._csw_row_of(word)) is
None` and `assert owner.word_surface_for_row(owner._csw_row_of(obj)) ==
b'12'`. The word is also checked to have no surface before the second
definition. Both exact code-decode checks and both definition-link checks
stay verbatim. The old assertion embodied the rejected broadening; this
is a disclosed contract port, not a setup correction claimed to preserve
all assertions. Development-18 passes. [Exact N1 patch](n1-repair.patch).
No other assertion, fixture or retirement changes in this repair.

**Review declarations:** interrogative completed Parts now retain their
live mode and are excluded from answer seeds. `closing_question` prefers
the supplied parse's mode, otherwise the clause mode. Invalid thought-shaped
LTM content receives no extra expectation holes under the signature guard;
invalid live thought records still fail loudly. An explicit eligibility
predicate is a possible later clarity improvement. The predicate ports
check each actual occurrence link and value, not global uniqueness; the
narrow-idea diagnostic checks a second sighting of one row. The observer
no longer gets ordinary word/object duplicates from the withdrawn §64
change. The five indirect-consumer files noted by Claude (`ltm_consolidation`,
`existence_ingestion`, `thought_binding`, `truth_ideas_routing`,
`query_work_budget`) remain in step 2's end full sweep; they are not claimed
as scoped gates here.

## Earlier held-content repair retained on this source

B1's held-description reader now follows only a valid thought signature.
An ordinary LTM description adds no expectation holes; the filled current
`ask` retains its empty mask. Live thought records retain strict signature
validation and their mode guard. Two new cases cover a fused idea and a
relation with an ordinary concept in its VP slot; both require one metered
read and an exactly empty mask. The registered direct/nested question and
live thought-memory budget regressions remain unchanged.

The first run of these tests (development-13) had one setup failure: its
new ordinary-relation fixture used `append_meaning`'s default fact kind
with the catalogue's live interrogative mode. §63 explicitly supplies
`kind='observation'`, matching production observation admission. Both
assertions remain verbatim; no production behavior was changed for this
setup correction. Development-14 reruns both new cases. All failures stay.

The prior frozen `9c3899f2f` completed 1,023 passes and nine existing
prerequisite skips, but the [plain-idea probe](../review-repair/held-idea.json)
found its missing guard. That source is withdrawn. It was not submitted
for a second Claude review or published. The earlier `3c5cc2a1d` source
has Claude's HOLD; the first `bfa9c94ee` source has its retained module
failure. No attempt is relabeled or used as evidence for this new hash.

## Full storage contract and §41 dispositions

The source removes persisted leaf postings, adjacency/nudge, stored mode
and unused LTM0 refs; rebuilds the derived index after complete restore;
keeps polarity on occurrences and reverses only trust-derived negative
Part evidence; retains every completed S beside its existing relation;
and stages forgetting before deletion. Derivation, its readers,
close-time minting and MTM are unchanged. Metadata authentication stays.

- **Forgetting:** §42's conservative stage barrier is the explicit §41
  alternative, with persisted `forget_pending` below high water and across
  restore. It avoids replacing the selected low-value child's structure
  advance with a costlier deletion. One boundary is one pass; no row
  advances twice and a released child survives until a later pass. Gist
  vectors stay unchanged. Claude accepted this rationale in the HOLD.
- **Fact admission:** stored question and estimate kinds are refused,
  before and after restore. An observation with evidence has no saved
  interrogative mode; incoming live TruthSet questions are still refused.
- **Mode consumers:** no active `LTM0.MODES` or stored-mode reader remains.
  The former column, raw read, validation, migration and admission uses
  are retired, and obsolete checkpoint keys are discarded. The embedded
  question fixture checks kind/evidence. `reasoning.py` drops the stored
  conjunct. `ThoughtFeatures.MODES` keeps live modes; stored reads receive
  the default. Closing questions and end-state answer fields preserve the
  current clause/parse mode. A held `ask` supplies current query intent,
  with the eligibility guard above. No replacement mode is stored.

## Every port, fixture and retirement

[Exact assertions and loop scopes](assertion-ports.md), [the complete
machine ledger](assertion-ledger.json), [all modified functions](fixture-ledger.json),
[whole test patch](test-ports.patch) and [observer-copy ports](copied-observer-ports.patch)
are relative to `bb0045d7a`, Claude's unchanged §38 commit. They include all
new regression and fixture-helper bodies. The audit covers every modified
production/configuration/test file. The removed packed `where` comparison
is an explicit assertion-scope entry.

§§59–61 list the count ports exactly: conflicting opposite-polarity
relative occurrences `0 -> 2` in both fixtures; selected relation `0 -> 1`
(two cases); concluded `not` `1 -> 2`; native chain conclusion `0 -> 1`
(three cases); nested taxonomy conclusion `0 -> 1`. Earlier count and
operand-scope ports remain listed in the ledger. Evidence, path ablation
and content checks are retained.

The four predicate-argument ports check each source occurrence's exact
link to its retained derivation root and original full operand value;
[both writers reproduce the old bank-seat failure](../review-repair/predicate-arguments.json).
The narrow-idea check excludes all LTM operand refs and [passes with both
writers](../review-repair/narrow-idea.json). The joint-objective test checks
absence of the retired switch and reads its bounded predictor context
before testing the unchanged no-gradient predicate; its gradient/loss and
update assertions remain. No detach is added. The §46 Part-read tests
already failed after §10.4, independently of 2b; they now certify that a
row alone does not select a bind. Actual refinement remains gated.

Only three tests are retired: P3's adjacency/nudge-only cases. P5 retires
none and keeps its two approved pair-corruption cases. No new skip is
introduced; P2, tagged P4, P6 and P7 remain deferred. Every failed run,
initial state, diagnosis and withdrawn freeze remains under the parent
receipt. Compressed captures retain original byte hashes.

## Gates, review and publication

Freeze and run all three scoped groups on one new hash: the eight
regression files, the complete eight-file native set and 76 module files.
This includes the four direct decoder consumers Claude named in N1:
`radix_layer_reverse`, `active_decode_candidates`, `echoic_decoder`, and
`tied_operator_reconstruction`, in addition to the six form-reader files
added previously. [Coverage](module-coverage.json) and
[expanded selectors](expanded-scope.json) name every file. The source audit
hashes all modified source, configuration and test files; the runner
checks those hashes before and after every group. No fresh series and no
full sweep. Step 2's one full sweep stays at its end.

Claude reviews this hash and its new evidence before publication: push
BasicModel main and bump WikiOracle main. Then write the requested §§34–35
code review; Claude's identity spec precedes implementation. No learning
claim is made. Item 3.0 has not been reached.
