# Item 7.5 validation and review handoff

The implementation is ready for Claude review and remains uncommitted under
the repository publish rule. Validation covers all 4,945 selected cases; its
assertion and resource failures remain visible below. The
[specification](../../specs/2026-09-26-one-operation-per-round.md) is authoritative.

The [structured results](validation-summary.json),
[full coverage accounting](full-coverage.json),
[reconstruction comparison](reconstruction-comparison.json),
[runtime/configuration/test patch](changes.patch), and
[source archive](review-source.tar.gz) form the review packet.

Review starts with the joint selector and two-slot reducer in
[Language.py](../../../bin/Language.py), the paired training transaction and
fixed word/seal trace in [Models.py](../../../bin/Models.py), and operation
application in [Spaces.py](../../../bin/Spaces.py). The new mechanism tests are
[operations](../../../test/test_compose_operations.py),
[paired training](../../../test/test_compose_pair_driver.py), and
[record ownership](../../../test/test_compose_records.py).

## Behavior and review choices

`OperationSelectionLayer` replaces binary tiling and the separate unary layer.
One tempered softmax covers every binary operator at each adjacent pair, every
unary operator at each live position, and eligible STOP. Each active round
selects exactly one hard operation. The selected candidate keeps its
probability-weighted straight-through derivative. No compose DP, DP-prior,
advantage objective or flattened-temperature pass remains.

The implementation uses 90% argmax / 10% sampling for exploit, three joint
rounds per serial word, a 2K seal budget (capped by positive syntacticOrder), and
at least 2N parallel rounds. Exact softmax-probability ties use item 8's
structural preference. STOP is unavailable above the row's capacity. An
incomplete forest trains but is excluded from memory. Overflow is incomplete;
no constituent is evicted. These choices were stated before measurement and
were not tuned against reconstruction or learning scores.

`runBatch` owns the exploit/explore pair. Each packed sentence gets a uniformly
selected used exploit round whose chosen action is masked in exploration.
Other explore rounds sample freely; an earlier divergence or STOP already
makes the paths distinct. A catalog without a legal alternative raises.
Each derivation performs its own backward and optimizer update. The public
batch clock and training-step counter advance once. Raw `forward` remains an
individual-trial primitive. Explore uses the eager outer forward with the same
tensor word recurrence; the strict whole-model graph check covers exploit.

Only exploit's program, STM, LTM and observations survive. The runtime
transaction restores Teacher staging, trace, memory, radix tries, percept
inventory, taxonomy, category learners and allocators. Contextual dictionaries
retain their owned storage and external-reader aliases. Sparse topology and
optimizer moments are restored by surviving edge identity: shared edges retain
both trials' learning; exploit-only edges retain their committed values;
explore-only definitions disappear. Diagnostic warning deduplication survives
both trials.

Bounded prediction histories are cloned as deques, including their nested
mutable values. Sharing those queues across the transaction had erased owned
observations and prediction contexts. A further unseeded readiness attempt
also exposed a legitimate incomplete compose path. Checkpoint plumbing and
unlabelled thought-credit fixtures now explicitly supply completed compose
observations, preserving their original counter, ownership and thought-learning
assertions. The thought chooser is not controlled. Completion state is
cleared per derivation and read from the active serial or parallel path, so a
previous serial batch cannot supply a stale completion mask.

Packed seals have separate fixed trace groups: group zero holds the final
seal, and an intermediate sentence ending at word w uses group w+1. This
preserves a first sentence containing just one word. Forward recording, reverse
traversal, auxiliary reconstruction, exploration ownership and postfix programs
use the same mapping. The final seal also records the pre-operation operand
rows, as intermediate seals already did.
The same positive syntacticOrder cap now applies to intermediate and final
seals; the fixed trace slab remains 2K wide. A unary-only red probe recorded
16 intermediate rounds with a cap of one, and the corrected 20-case
record/budget group passes.

Relative-row capacity now reads the actual selected operation trace, including
inside the tensor word loop. It preserves the existing closed-class anchor
condition and scopes choices to their packed sentence. The final sentence's
same evidence reaches the existing memory-learning reader. No parser is rerun
to infer a second derivation for bookkeeping.

## Validation status

The first full sweep completed all **4,932 cases**: **4,589 passed, 321 skipped,
21 failed and one expected failure**. Its 2,627-second receipt is preserved in
`full-before-full-fixes`. No worker exceeded its original 1,800-second / 8 GiB
limit. Failures exposed Teacher restaging, sparse topology, owned-buffer
identity, and warn-once transaction bugs, plus fixtures expecting one training
update. The corrected transaction group passed **94/94** cases.

Cadence fixtures now check exploit followed by explore while retaining their
oracle isolation and answer-credit assertions. Storage and canonical-layout
fixtures explicitly select a completed derivation; a separate regression
requires incomplete ingestion to leave LTM unchanged. No learning gate's
threshold, seed or assertion was relaxed.

The unchanged depth-three relative campaign still reports only depth-one
end states. A read-only diagnostic first showed selected relative operations
missing from the old parser rule list. The trace-capacity mechanism is fixed
and its scoped/fullgraph checks pass, but STOP is eligible, not compulsory,
at depth three. The empirical campaign's failure is retained without tuning.
The final affected run completes all 142 cases: **109 passed and 33 opt-in
skips**, with no failures, in 997.87 seconds. Its peak worker memory is 3.97 GiB.
All four explicit compiler/trace checks and all 17 ownership/word-store checks
pass. These receipts and measurements used all 666 source files, SHA-256
`1ac4baac29471abdde6dcf5c8514d661cca56d3c1d4e3d8397640583aed27c50`.
Later sweeps exposed two synthetic fixtures needing the new trace layout:
the detached reconstruction fixture still placed a final seal in its former
packed group, and the structural-preference fixture supplied only a buffer
instead of a complete STM state. Their corrected files pass all **18 cases**,
including explicit operand-row assertions. All production/configuration/
measurement inputs remain identical; every source delta is recorded explicitly.
The intermediate 4,945-case sweep is preserved in
`full-before-trace-fixture-port`: only the incomplete fixture and the unchanged
relative-depth campaign fail. The submission source is
`c0e9097ae05b800b8de4622d410eacaf7fa775236f7602f6d953f120b5fa2e98`.
All **4,945 cases** are covered: **4,622 passed, 321 skipped, one failed and
one expected failure**. The unchanged relative-depth campaign is the only
assertion failure.

The final pool started expensive files first and preserved the complete case
set, within-file order and suite lock. It stopped after 868 completed cases
when a relative-STM test worker crossed its 8 GiB cap (observed peak 8.55 GiB).
The original exit 137 and raw receipt remain in `full`. Its three unfinished
cases pass in fresh workers in `full-memory-remainder`, peaking at 1.82 GiB;
`full-continuation` completes the other 4,074 cases, peaking at 4.48 GiB. The
868 completed cases were not repeated. Independent coverage accounting verifies
each completed case exactly once across the three source-identical segments.
Their combined execution time is 3,297.45 seconds. The original suite deadline,
three-worker / 24 GiB reservation and 1,800-second / 8 GiB worker limits were
preserved. Complete coverage does not erase the memory-stop failure.

Optional packed reverse probes exposed a stale deleted-unary API call and an
eager/compiled comparison that sampled different derivations. Those mechanism
fixtures now select the same complete path; the provenance probe retains a
lexical leaf beside an older unary-rewritten constituent until each seal.
The resulting red probe caught missing final-seal operand rows. Reconstruction
measurements and learning gates continue to use their unchanged chooser
protocols.

Earlier iterations, interruptions and failures remain in their named receipt
directories; names containing `final` in historical iterations are not claims
of final-source validation. `relative-diagnostic-process.json` records the
instrumented unchanged campaign and its source. The earlier transaction
preview used an isolated tree and is not a final-source receipt.

## Reconstruction and unchanged learning gates

The measurement driver
reuses the reviewed seed-42 protocol, budgets, optimizer and thresholds. Its
copied `parity.xml` only removes retired configuration elements; the historical
drivers remain unchanged. The corrected-source results are:

| Measurement | Reviewed control | Item 7.5 |
| --- | ---: | ---: |
| Serial before training | .10059066489338875 | .10405020788311958 |
| Serial during training | .09482414424419403 | .10417700409889222 |
| Serial after training | .09287650510668755 | .10392807051539421 |
| Packed sentence byte cost | .6839025616645813 | .787799209356308 |
| Single sentence byte cost | .6839025616645813 | .6835970133543015 |

Packed and single initial parameter/dictionary hashes match, but their sentence
artifacts and costs do not. Packed
rows reported truncation `[true, true]`; the single steps reported
`[false, true]` and `[true, false]`. Exact parity was lost. Warmed serial
throughput is .12717 sentences/second with concurrent CPU work and routing
reads; it is not an isolated speed comparison. All three measurement processes
finish within their original 1,200-second / 8 GiB limits. Earlier completed and
interrupted measurements remain in `measurements-before-*`; the completed
pre-transaction results match these reconstruction values.

Both unchanged XOR_grammar CLI gates fail at the six-row WholeSpace symbol
inventory during first-epoch reset/autobind. The unseeded 900-epoch
MM_grammar gate passes its unchanged <.20 assertion. Earlier failures remain
visible: the historical unseeded MM result is .21757 against <.20. No pass
replaces that result or establishes mature learned utility. The MM test calls
raw `forward` and therefore does not measure the paired training driver.
[Hashes](unchanged-contracts.json) verify the gate XMLs and test files are
unchanged. The final gate receipt records one pass and both capacity failures.

## Preserved limits and evidence

The slowest full-suite workers use small integration fixtures, not a
million-sentence training corpus. The operator-gradient and grammar-chooser
checks each train one two-sentence batch; the training interleave check makes
four calls on two short sentences. Their CPU execution time is retained in
the worker receipts and must not be mistaken for evidence from a long training
campaign.

The optional full word-store file reached the 8 GiB worker limit when several
models shared a process; unfinished cases then ran separately at the same cap.
Four missing inventory-row failures in `_populate_concept_weights` also fail
on the committed baseline (`baseline-word-store` and
`baseline-word-store-additional`). A fifth failure was the exploration trie
regression, which was corrected and passed both training epochs. The explicit
operator trace, real-model one-graph/backward check across runtime lengths,
packed operand provenance and eager/compiled packed traversal checks now pass
all four explicit cases on the identical production source.

The deterministic tie and STOP probes preserve their red results: distinct
float32 logits can produce equal probabilities, which must favor the structural
operator, and STOP must preserve the hard value exactly. The selector compares
probabilities and groups the zero-valued straight-through gradient term before
adding the hard value. The transaction probes preserve the original Teacher,
sparse topology and dictionary-identity failures.

Alec's rationale is retained in the specification: training by softmax lets one
gradient do all the optimization; DP is the pre-backprop symbolic solution and
should not be mixed into a working MLP. The explored path is a tractable
approximation, not an exact marginalization. No mature-checkpoint quality claim
follows from these mechanism checks or short reconstruction measurements.
