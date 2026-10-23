# Step 2b storage: native word form ownership

2026-10-22. This is the current receipt for storage-only step 2b (§36,
continued under §§39–64). Work stays on main. The [full review-repair
receipt](../review-repair/README.md) records the implementation and all
previous attempts; [Claude's HOLD](../landing/claude-review.md) remains
retained. No earlier result is promoted to this source.

## Word-form repair and retained failed source

The frozen `1f53af50ce5119a8da6b1ea3f01e1d0fbc04d171` completed 1,024 passes,
nine existing skips and one production failure. The joint-objective test
raised the rung-zero identity audit on its second presentation. The first
draw and two reproducing replays are retained in [held-content](../held-content/README.md).
A selected object named `2` reached a word row that had also become the
referent of `plus`; the bank's surface reader gave that word the latter
form. This source is withdrawn, without changing the failing assertion.

§64 repairs native form ownership in `word_surface_for_row`: a word row
keeps its own one-form admitted description; a non-word object still
follows its unique definition. Associations and conceptual reads are
unchanged. The original captured initial RNG now completes all three
presentations; [that replay](../held-content/rung0-repaired-replay/run.json)
is diagnostic only. No gate is seeded from it.

A new regression checks the word and its named object's exact bytes and
native form after the word also becomes another word's referent, retaining
both definition links. Its first setup assumed the surface helper admitted
`12` as one word. Development-15 and -16 retain that setup failure;
explicit admission fixes it with all assertions unchanged. Development-17
passes. No existing assertion, fixture or test is changed by this repair.

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

The [expanded module coverage](module-coverage.json) includes all direct
clause writers and reading-commit callers. `run_gates.py` records the new
hash and checks audited source hashes before and after each group: eight
regression files, the full eight-file native set, and all expanded module
files, plus the six form/definition and reconstruction files listed in
`expanded-scope.json`. No fresh series or full sweep; step 2's full sweep remains at its
end. Results and Claude's review are recorded below when complete.

Publication follows a LAND verdict: push BasicModel main and bump
WikiOracle main. Then write the requested §§34–35 code review; Claude's
identity spec precedes identity implementation. No learning claim is made.
Item 3.0 has not been reached.
