# Step 1b landing: sweep and repairs

Candidate: `dde89352756aba8cb0862fdcba0e60c94fff567b`, on
`codex/item3-5-first-pass`. The October 20 amendments are in that candidate.
Publication is held for the pool-capacity decision in [bind plan §10](../../plans/2026-10-19-bind-thought-operation.md#10-landing-follow-up-a-full-concept-row-pool-codex-2026-10-20) and the resulting native repair, in addition to final repaired-source gates. This receipt records
mechanisms and invariants; it makes no new learning claim.

## What did the one full sweep find?

[The full sweep](full/report.html) completed **5,965/5,965** unique cases:
**5,420 passed, 219 failed, 325 skipped and one non-strict XPASS**. Its
source was frozen, it completed without a timeout, and the unchanged random
paired-reconstruction case passed. Three duplicate worker reports from
recycling are counted once per case. The failed sweep is not relabeled green
and is not repeated. Later scoped runs repair its findings. The current failure index has 218 passing cases or declared fixture ports and one unresolved native MM_grammar capacity failure; an earlier pass at another initialization does not close that failure.

The broad affected follow-up [native-113](native-113/report.html) was
**interrupted** after 893/926 cases: 837 passed, 50 failed, five skipped and
one XPASS. Two behavior cases spent the remaining time compiling training
loops. The supervisor reaped the workers. Its 33 uncompleted cases were
selected again in native-117, using eager execution for behavior certificates;
explicit compiled-boundary tests still compile. Native-113 is not a pass.
All budgets, seeds and learning assertions in those behavior cases remain
unchanged. No training was added to rescue a failed learning run. Native-117
then timed out at 901 seconds in the original six-epoch cold-start boundary
learning certificate: **62/70 cases completed**, with six fixture/read
failures and eight uncompleted cases. A process sample showed numerical
Torch work, not compilation. The learning case is **incomplete**, not passed.
Its budget is not extended and it is not rerun to obtain a learning result.
It is explicitly carried to Item 0's learning evidence; it was not selected
by this round's default full sweep or the step-1b mechanism gates. The other
seven uncompleted cases complete in native-118; it completes 59/59 cases
with 56 passes and three failures: two packed reconstruction checks and
the native MM_grammar pool-capacity error. The error is not a convergence
verdict. Native-119 captures a second failing initialization and its
complete allocation plan. Required native and
compiled mechanism checks still have to complete before publication.

[All run results](runs.md) retain the failures and intermediate repairs.

## Why did the pending premise fail at a new initialization?

The full sweep captured a fresh failure in the forward arrival arm of the
same pending-reference certificate. The saved Python, NumPy and Torch states
are in `landing-diagnostics/`. Native-101 replays that failure. A shared
physical Part store had made DEF edges visible through generic part/whole
reads: a word’s objects could be interpreted as its composing constituents,
and an object’s spelling as its taxonomic parent. That unrelated identity
could win the later occurrence’s bind read.

`ConceptAllocator.records` now excludes edges whose argument types license
the definition reading from generic structural reads. DEF still lives in
Part and remains available through the definition reader. Native-102 passes
at exactly the failing initialization. Native-104 removes only this filter
and fails again at that state. This is a causal mechanism repair, not a seed
choice. Claude’s unsaved October 20 state was never available and is not
claimed as replayed. Saved arrival states 32 and 75 pass in both orders in native-122/123;
c/e state 61 passes in native-120. The full-sweep failure passes again in
native-125. [Arrival tables](arrival-replays.md) retain the individual reads.

## What else changed after the sweep?

- Word recognition normalizes byte/text form keys through the shared form
  key owner. An unknown percept ID does not call an out-of-range radix read.
  The native word-store regressions cover both reads.
- Native words acquire their existing word type and order at allocation,
  including terminals first encountered during a closing. The §6.10 DEF and
  occupation writer supplies the expression reader’s type source.
- A Part edge owns a native handle; it does not automatically seat a copy of
  dissolved STM content in the concept bank. Existing native but unseated
  arguments also remain unseated. The latter regression fails before the
  repair in native-115 and passes afterward in native-116.
- A self-edge of a word that also has an object use is equality, not its own
  definition. A shared typed reader supplies both definition admission and
  interpretation. This regression fails in native-114 and passes in native-116.
- Reference candidates come from the frame’s retained Part roots before any
  time-based fallback. Participant prediction caches belong to Part, use
  native identity points, and follow Verb’s participant/verb ordering.
- Concept protection and forgetting include native Part and DEF references.
  Unseated edge handles do not consume concept-bank seats. Close allocation
  is preflighted as a unit, with no partial prefix on refusal.
- Reconstruction now uses the held reading’s accepted-word mask. Clearing
  the live attention mask changed its cost from 2 to 0 in native-119; the
  new regression requires the immutable record to retain the same cost.
  Packed compilation also captures the exact inputs that were scored,
  before a later thought can bind their roots. Native-120 passes the held-support regression and compiled parity.
- Dense event readers retain their signed linear contract; native identity
  readers use cube coordinates. Ineligible decoder proposals are excluded
  before a forward identity map validates recomposition.
- The grammar teacher receives its own stream’s expectation prime. Packed
  streams no longer reuse another stream’s batch shape.
- TruthLoss now measures overlap with opposite evidence. The lattice join
  preserves both truth poles and therefore cannot detect contradiction by
  a reduction in union norm. Its default remains zero; the unchanged
  contradiction/gradient checks pass.
- Surface composition reserves finite chart headroom. With chart `C`,
  translation `d=W(C(marker))` and rail `r=C(1)`, it writes
  `C⁻¹((d+C(content))/(2(1+|d|/r)))`. Supplying the marker uniquely recovers
  content. The right-witness inverse solves the corresponding piecewise
  affine equation. The generative proposal interpolates toward zero
  translation until its residual is in-domain; no identity coordinate is
  clipped. The occurrence inverse supplies the marker. The original
  fixed-budget Surface certificate and inverse checks pass in native-112.

## Which fixtures changed, and what is not claimed?

| Fixture | Port and retained contract |
|---|---|
| Native signed identity examples | Use production cube coordinates. Signed dense-event examples keep their separate explicit contract. This does not relax the permanent signed-identity loader rejection |
| Reference-close and G1–G3 | Supply the actual native predicate symbol and `(participant, verb)` pair. The original assertions remain unchanged; all 43 bind cases pass in native-114 and native-116 |
| Four-word inventory in six/eight seats | Hold binding to legal existing/open choices with a test-only fixture. The tests measure word-field isolation and inventory size; they cannot also promise room for unbounded new individuals. Native-115 diagnostics show explicit mint choices exhausting the order-0 pool. Production capacity refusal is retained; no capacity fix is claimed |
| Free reconstruction versus compose journal | Free decoding does not replay the journal. Tests now require journal independence, honest missing-leaf reporting, packed ownership and eager/compiled parity. The withdrawn STM1 inversion gate is not reintroduced |
| Native lexical audit versus dense byte distribution | Native bank membership scores identified words and rejects unknown content. Continuous byte-mixture, null-candidate and gradient assertions run against the separate dense scoring boundary |
| Retired chunk traversal tests | Removed because chunk is absent from the grammar catalogue. Surviving tied inverse and shared-gradient certificates cover declared operators |
| Old all-fold gradient/free replay assumptions | Removed from the retired traversal contract. `test_tied_affine_inverse_uses_the_retained_operand` and prepared-answer executed-inverse gradient certificates retain shared-weight coverage |
| Prepared-answer gradients | Supply a nonempty concluded identity and its readable terminal bank; untrained input composition can erase a root. Remove the unused `supplied` parameter, which duplicated each test at a second random initialization. Both output-loop paths and every state/weight-gradient assertion remain; all four repaired fixture checks pass in native-121 |
| Native percept checkpoint | Restore the same WordIdentity owner the common constructor supplies. The bare Radix replacement had discarded typed identity metadata; it is not a model reload |
| Packed provenance and compilation | Replay only accepted grammar leaves; retained sentence banks may have different static widths. Each signature must compile as a full graph and reuse its graph on replay |
| Behavioral epoch/checkpoint tests | Use the existing eager fixture; seeds, fixed budgets and assertions are unchanged. Explicit compiled parity remains separately tested |

No fixture recognizes a decomposition seed or spelling. The decomposition
forced scopes, seeds, budgets, assertions and R+A keep law are unchanged.

## Carry destinations

LTM1 column cull, context wholes, recurrence minting and producing-artifact
regeneration stay in step 2. Strata, geometric order-0 diffusion, placement
and content-width cleanup stay in Item 3; STM1 has no inversion or width gate.
Verb groups stay in step 4; second difference and occurrence tense/carrier
work stay in step 5. Item 0 findings and independent 5.5 learning gates remain
separate, without regrade or rescue training. The cold-start boundary
learning timeout in native-117 is also carried to Item 0 and is not a pass.
See [the ledger](status.md).

## Repaired-source native check

Native-120 completes **316/316: 309 passed, three failed, four skipped**.
The three failures are a packed fixture targeting the final raw tile instead
of the final accepted leaf, and two prepared-answer fixtures lacking a
readable concluded idea. Native-121 passes all four repaired fixture cases.
The §14 file, G1–G3, reference-close, cube/inverse, random paired invariant,
word-store and compiled boundary checks pass in native-120; its skips are
the retired copular artifact and three MPS prerequisites. Saved-state replays
and both decomposition arms are recorded separately; results from these selections
are not described as a second full sweep.

## Final saved-state and decomposition evidence

| Question | Result |
|---|---|
| Saved arrival 32, both orders | Native-122: both pass |
| Saved arrival 75, both orders | Native-123: both pass |
| New full-sweep arrival failure | Native-125: passes at the captured Python/NumPy/Torch state |
| Saved paired-reconstruction failure 43 | Native-122: exact zero tie in every stream of all three batches |
| Fresh paired-reconstruction certificate | Native-120: exact zero tie in all prescribed batches |
| Saved c/e closing state 61 | Native-120: passes |
| One/two successors, all gains zero | Attempt 124: both pass, 4/4 companion questions correct |
| One/two successors, all gains .5 | Attempt 124: both pass, 4/4 companion questions correct |

[Each decomposition question](decomposition-124.md), [each saved arrival
read](arrival-replays.md), [each paired batch](paired-final.md) and the raw
files are in this receipt directory. The one-successor proof spends
234/512 work; the two-successor proof spends 557/768. The changed work count
reflects the repaired native graph; budgets, original parameter snapshots,
forced scopes and assertions are unchanged. These are mechanism certificates.

[Source validation](validation.json) confirms that native-121, all final
saved-state replays and both decomposition arms match the current validated
source. Native-120 differs only in the two repaired fixture files. All
production files match its tested source. The one unresolved native
MM_grammar capacity failure blocks publication; an earlier scoped pass at
another initialization does not close it. No full sweep is repeated and
no additional training is used to obtain a passing learning result.
