# §6.10 closing and arrival repairs — development record

This round implements Alec's §6.10 decision on the existing branch. It is
not a landing or a learning claim. The earlier failed measurements remain
in [runs.md](runs.md); source snapshots are not combined into one pass.

## What did the later occurrence fail to supply?

`ClauseRows.read_reference_bindings` accepted only a complement with a
concept-bank point. A witnessed unminted LTM1 occurrence is a valid native
reference at its retained width, and a Part address can be valid without a
bank seat. Both were being rejected. The read now preserves the typed
reference and takes the already observed complement value; it does not
invert an edge, reconstruct missing content or rewrite the source row.

The occurrence-complement regression fails before the repair in native-76
and passes in native-77. The Part-without-bank-seat variant also passes in
native-87 and native-91. Exclusion of a closing's own witness is unchanged;
other occurrences in the same stream remain eligible witnesses.

Claude's `/tmp/review-1b-tests.log` retains the reverse-order assertion and
its occurrence complement, but no initialization state. Its pytest temporary
directory is gone. An exact replay of that unsaved run is therefore not
claimed. The representation defect is reproduced in the isolated regression.
Saved initializations 32 and 75 are retained and replayed. Initialization 75
was captured when first encountered, not selected for a pass.

Native-87 fills all of initialization 75's earlier pending references, but
its forward arm still fails `bound_correct`. Native-88 also fails that check
in the forward arm of saved initialization 32; both reverse arms pass.
These are failures, not completion of the arrival gate.

## What survives a kept close?

A minted terminal now retains its word-origin DEF edge and its admitted
formal occupations. The common PartsOfSpeech owner supplies those types.
Part itself still has only the six decided columns. The reader classifies
ordered arguments using those occupations, follows a declared head for an
attachment, and preserves a clause's subject. A later definition of another
object named by a marker does not retrospectively change the retained
marker's type.

A bound terminal also admits its word use: selecting an existing object is
not a reason to omit the new word's DEF edge. This is a native Part definition,
not an equality between the object and its spelling. The bound-terminal
regression passes in native-91 and native-94.

## Why did a licensing expression retain a different operand?

An ambiguous word could reach a predicate position as its WORD symbol,
while the preceding expression contained a defined object. The grammar
bank now exposes admitted word-derived candidates from its recent closings
for the current lexical cues. At a formal argument without a reference
contract, only that word's retained definitions are enumerated. Their
selection remains in the ordinary chooser. There is no nearest-code match,
normalization, word list or extra scorer.

A broader candidate filter tried in native-83 was rejected because it
regressed six reference-close cases. The first general menu expansion then
exceeded the 8 GiB bound in native-89, which completed 76/77 cases; it is not
a pass. The replacement bounds lexical alternatives by the staged definition
fan-out and retains the ordinary full menu only at reference arguments.
Native-92 completes with a 3.77 GiB worker peak, but the forward arrival
check still fails.

An objectless marker with an admitted surface occupation now remains a
literal during an implicit reference read. Explicit binding and minting
operations retain their contracts. This prevents a generic implicit
participant read from replacing the marker with an unrelated object before
the §6.10 reader sees it. No spelling identifies this type.

## Why did a completed phrase create an open row?

The old `cache_operands` path recovered an already complete verb phrase as
an additional LTM1 row. That synthetic recovery treated a participant as a
predicate, then expected another participant. It left the two-successor
premise dependent on a fabricated open occurrence. The Part derivation now
owns those operands; `cache_operands` and `_point_constructor` are removed.
The 48 storage/reference/clause cases in native-94 pass after this removal.

## What does the verifier read?

The historical October 7 observer remains unchanged. Its current test port
uses the common `store.location` address read for source texts and witness
existence, covering both LTM1 occurrences and Part symbols. The successor,
license, dependency, work, inference-count and binding checks are unchanged.
The thought writer also avoids publishing unreferenced working constituents;
its regression passes. The one-successor trial reaches the correct referent
in native-90/93, but those runs fail the unchanged side-effect assertion.
The final diagnostic in attempt 96 shows the cause: a `query` conclusion
was treated as a newly selected Part composition. Publishing its inferred
binding created a redundant Part edge and grammatical occupations. The
unchanged §14 footprint assertion rejected that mutation even though both
proof chains were correct. `query` and `bind` conclusions now take the
existing witnessed proof-occurrence path; explicit composing operations
still close through Part. The two new regressions fail in native-98 before
the repair and pass in native-99 afterward.

## Acceptance before the landing sweep

| Question | Result |
|---|---|
| Later occurrence with LTM1 or unseated Part complement | Native typed reference retained; both isolated regressions pass |
| Saved arrival initialization 32, both orders | 2/2 pass in native-97 |
| Saved arrival initialization 75, both orders | Both pass in native-95; all `_forward_references` filled on the working read |
| Saved c/e initialization 61 | Passes in native-95 and native-99 |
| Broad scoped selection | Native-95 completes 231/231: 230 passed, one retired copular-artifact prerequisite skip |
| Final proof-writer and thinking checks | Native-99 completes 46/46, all passed |
| One successor, gains 0 / .5 | Both pass in attempt 100; 247/512 work per question |
| Two successors, gains 0 / .5 | Both pass in attempt 100; 549/768 work per question |
| Companion questions | 8/8 correct, with the required proof chains and no remaining open slots |

[Attempt 100](decomposition-100.md) has one table per question. The broader
native-95 snapshot precedes the final proof-writer guard; native-99 and
attempt 100 exercise that guard. These results are not combined into a
full sweep. The one landing sweep and publication follow this record.
Step 2, Item 3, step 4, step 5 and the separate Item 0/5.5 tracks retain
their destinations in [status.md](status.md).

## Follow-up after the one full sweep

The sweep completed 5,965 cases with 219 failures. Its new arrival failure
is captured, reproduced in native-101, repaired in native-102, and reproduced
again by removing only the typed-DEF filter in native-104. Native-125 passes
at that same saved state on the final source. Generic structural reads must
exclude word-definition edges; sharing the physical Part store does not
make a spelling an identity operand. The [landing receipt](landing.md)
records this additional cause and all fixture migrations.

Saved arrival states 32/75 pass in both orders in native-122/123; saved c/e
state 61 passes in native-120. Saved paired state 43 and a fresh random case
tie exactly at zero R. Both unchanged decomposition lengths pass in both
arms in [attempt 124](decomposition-124.md), all eight companion questions
correct at 234/512 and 557/768 work. This closes the §6.10/§8 mechanism gates,
not publication: the unrelated but native pool-capacity failure remains
open for the decision in bind-plan §10.
