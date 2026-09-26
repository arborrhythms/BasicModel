# Item 9b follow-up: parallel context before serial reading

The follow-up is implemented, validated and ready for Claude's review.
It remains uncommitted and unpushed. [Claude's earlier review](review.md)
accepted the core implementation and requested this follow-up.
The original [9b receipt](../2026-09-25-item9b/README.md) preserves the accepted
core implementation; this receipt covers the changes requested afterward.

## Pass order and state

The cursor stages the next N complete sentences. Native parallel processing
reads that whole group first. Serial batches then read the same sentences,
using the shared inventory updated by the context pass. The shorter final
group is processed too. `ModeSchedule.resymbolize` and `_label_feedback` are
removed: the ordinary serial reading provides the symbolic processing.

`runEpoch` walks corpus order with serial batches no larger than the requested
batch size, splitting at group boundaries. Its context batch contains N rows.
Targets and source IDs stay attached to their serial presentations; the native
context pass receives no supplied answer, does not advance the external clock,
and does not append a second LTM observation. Direct `runBatch` calls can use
a complete group, including packed input, or provide `schedule_context` before
its first serial prefix. Mode flags, input staging and Teacher source staging
are restored even when context execution fails.

Checkpoints store the unread serial prefix and completed context-pass count.
The continuation test saves after the first serial sentence, restores, and
verifies that the second is read without repeating the native pass. Mid-epoch
resume requires the same grouping and serial batch size. Old checkpoints with
pending serial-first replay records cannot be resumed as unread sentences;
they fail with an explanation rather than processing the wrong direction.
Loading old weights at a completed group boundary remains supported.
Evaluation uses a temporary queue and restores an unfinished training prefix,
even on an exception. A new training epoch starts a new queue; a resumed epoch
keeps the saved one. This protects checkpoint continuation when evaluation runs
between training calls.

## Descriptive metric and archived experiment

The three-seed job is removed from the suite and preserved with its original
source and findings in [FutureWork](../../FutureWork.md#shared-mode-erosion-measurement-item-9b).
**Its interleave rows and every comparison using them are void for the new pass
order.** The raw values remain historical evidence; no replacement job or
ordering gate is introduced.

[CategoricalDiscrimination.py](../../../bin/CategoricalDiscrimination.py)
accepts captured readings for the [fixed probes](../../../data/categorical_discrimination_probes.json):
four XOR inputs and 68 FineWeb word forms. The FineWeb labels describe native
orthographic properties, not semantic categories. One JSON-ready logger field
reports between-minus-within L2 distance (CP), both component distances, and
pair counts. Every unordered pair is counted once; unknown or zero readings
stay in the result. The reducer does no model execution, training or read-back.

The [cost measurement](metric/metric.json) uses all 65,536 physical concept
rows and both poles, on one CPU thread. Five timed repetitions have a median
of **49.5 ms**, with 36 MiB of input readings. This measures the reducer on
synthetic captured data, not model quality or the cost of obtaining 72 probe
readings. Item 4 must budget that collection cost at its logging interval.
There is no numerical pass threshold or required ordering.

## New reconstruction baseline

The same seed-42, seven-update measurement gives:

| Phase | Earlier item 9 | Reviewed 9b | This follow-up |
|---|---:|---:|---:|
| Before training | .10061445273458958 | .10059066489338875 | .10059066489338875 |
| Mean during seven updates | .0879226878285408 | .0949082300066948 | .0949082300066948 |
| After seven updates | .08916187100112438 | .08824818022549152 | .08824818022549152 |

These reviewed 9b values are the baseline for later reconstruction comparisons.
The relevant training correction was in the derivative of fractional concept
feature weights. Previously an existing edge also received insertion credit.
Now insertion credit applies only to an unwritten edge; an existing edge uses
its actual exponent derivative. This changes the optimizer's trajectory. The
comparison includes the other 9b changes too; it is not an isolated ablation.
The mean training cost is about 8% higher than item 9 and the final cost about
1% lower. Neither difference is presented as an overall improvement.

[Raw serial measurements](measurements/baseline.json) and the complete
[execution manifest](measurements/manifest.json) are retained. The follow-up
reproduces all three reviewed 9b phases exactly. [Packed and single sentence
reconstruction](measurements/comparison.json) both remain
**.6839025616645813**, with zero differences in roots, captured program roots,
references, recovered values and byte costs.

## Documentation clarifications

**Superseded by Alec's September 26 corrections:** the following paragraphs
record the reviewed follow-up, not the current design. Context is now required
to take no gradient step, known-word interpretation must reuse its existing
object, and registry addresses must live in the shared bands themselves.
See the [current plan](../../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md#september-26-corrections-supersede-earlier-defaults).

`interpret` defaults to an individual object, the order-1 particular.
Grammar can request an order-2 kind, such as cats in general. An unknown
individual becomes a provisional individual; missing evidence does not turn
it into a kind. Architecture, Language and Lexicon now say this directly.

Registry locations are exact `int64` coordinates saved beside captured
programs. The periodic position signal mixed into event vectors is a separate
representation. Its default period of 8192 does not wrap or truncate registry
addresses. Params and Architecture distinguish the two.

The current PartSpace reserve is 32,768 admitted part rows. That small-run
setting is not a capacity estimate for a million sentences, and there is no
one-row-per-sentence rule. Item 3 now explicitly requires measuring part
admission, raising the reserve, and recording dictionary and optimizer memory
before the full corpus run. Exhaustion stops training; the table does not grow
under its live optimizer.

## Validation and fixture prerequisites

The initial failing probes reproduce serial-first ordering on the reviewed
runtime and identify the missing look-ahead cursor and fixed-probe reducer.
The follow-up integration checks cover
both input layouts, full and short groups, source/target preservation, document
boundaries, runtime clock behavior, checkpoint continuation, incompatible
resume settings, and restoration after context failure. Additional probes
cover evaluation during an interrupted group, epoch queue lifetime on success
and error, and byte capacity measured with the input encoder's actual ASCII
replacement policy.

The old nonzero-gradient schedule probe ran its first parallel pass after two
serial training observations. The new order removes that implicit prerequisite.
The test now makes those same two observations explicitly before switching to
interleaving; its nonzero and finite gradient assertions and shared-parameter
checks remain. A cold context pass produced zero gradients in that fixture;
that development result is preserved rather than described as a passing
learning check. No seed or tolerance was changed.

The new packed-order probe initially reserved eight slots before installing
the native unit tiler. Its two sentences actually require eleven units including
spaces, so it now declares sixteen slots. The pass-order and whole-sentence
assertions remain. The runtime-only context probe also exposed a real bug:
inference has no loss to publish, and the scheduler now records that explicitly.

A later audit reproduced a second boundary defect: a pending training group
blocked evaluation. That defect was captured red, then fixed by scoping epoch
queues. The earlier full sweep was deliberately interrupted at 2,246/4,899
cases for this fix; it is retained only as a diagnostic. The encoder-capacity
probe separately reproduced a false rejection of non-ASCII input that fits
its actual encoded byte buffer, and the guard now uses the same encoder policy.

The explicit slow selection repeats all three ambient-initialization
packed/single parity runs and the excluded-candidate value/gradient regression.
The earlier reconstruction-cache probe still belongs to item 1: its compose
gradient assertion failed before the cache check on both the committed and
patched item 9 runtime. This follow-up does not claim that backward-cache
behavior is verified.

Two full sweeps on the pre-cleanup source stopped at their memory cap after
3,162/4,905 cases. The first worker reached 13.90 GiB while entering
`TestWriteMask.test_partition_isolation`; isolating each file still reached
13.94 GiB at the same case. The case passes alone at 1.77 GiB, and its 21-case
file sometimes completes by recycling workers at a 7.19 GiB peak. The prior
reviewed 9b sweep also needed recycling in that batch. That timing did not
reliably protect the next allocation.

`test_reasoning.py` now collects the completed case's model callback cycles
after every test. All 21 cases then complete in one worker at 3.13 GiB.
Assertions and seeds are unchanged. This fixture cleanup is the only source
change after the earlier passing affected and reconstruction checks. Final
validation repeats those checks and the full suite on the resulting source;
the two stopped sweeps and isolation checks remain diagnostic evidence.

Final validation uses 8 GiB per worker. The final full suite has two workers
(16 GiB aggregate); affected and measurement jobs each use one worker (8 GiB).
Concurrent reservations total at most 24 GiB. Runtime, test and configuration source stays frozen
during every bounded run. The final source contains 662 files with aggregate
SHA-256 `d82a8d00c202e058c218e8a52c7219e08ad41752e4a164700125706d6d195df4`.

The [final validation summary](validation-summary.json) records:

| Selection | Results | Time | Peak per worker |
|---|---|---:|---:|
| Full suite | 4,580 passed, 324 skipped, 1 existing expected failure; all 4,905 cases completed | 1,094.6 s | 7.73 GiB |
| Affected, including fixture cleanup | 110 passed | 424.4 s | 3.14 GiB |
| Explicit reconstruction selection | 4 passed | 224.0 s | 4.46 GiB |

Final documentation-link verification also passes **93/93**.

All three selections, both measurement manifests and the
[review source map](review-source.json) match exactly. The full suite's peak
aggregate footprint is 10.12 GiB. No skip or expected-failure marker was added.
The exact tested source is in [the archive](review-source.tar.gz).
[The source diff](followup-source.patch) and [nine-file change map](source-delta.json)
isolate this follow-up from the accepted core implementation; prose changes
are described above.
