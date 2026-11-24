# Item 1.5 performance handoff — 23 November 2026

## Source and scope

The original worktree started at `d38029113103ce5e9cf1223eaa55c9181010d649`.
Work paused when Alec said 1.7 was not ready. After publication, the measured
base is **`ad8fafea927839c0d7f92d0aee35c913d02b2612`**, which contains that
handoff and matched local `main` and `origin/main`. No pull was needed.
Alec then requested work in the canonical `basicmodel` directory on `main`.
The earlier branch's changes were ported, not merged; its old measurements
are not used below. Branch cleanup belongs to the parallel conversation.

This round changes computation cost. The objectives, candidate decisions,
attention order, clocks, memory writes and existing acceptance assertions
remain in place. Architecture proposals require Alec's review. Item 1.5 as
a whole remains open; this is the measured performance return for integration
with 1.6, not a claim that the entire sentence is compiled or host-free.

## Changes

- `PerceptProperties.counts_in_spans` scatters directly into floating-point
  `[batch, byte, position]` storage and scans the contiguous position axis.
  This removes the int64 one-hot temporary and the strided scan. Values and
  observation gradients, including empty spans and repeated bytes, agree.
- `BasicPercepts._occurrence_what` gathers a sentence's distinct runs together.
  Padding is excluded from max ties. `lanes` reuses its already computed code;
  `lookup_rows` gathers the unique codes instead of adding a full output slab
  for each row.
- `BasicPercepts.prepare`, called at `ConceptualSpace.cs_lookup`, shares the
  opening's request-independent part/whole reads, outside evidence and location
  factors. Activity, conjunction evaluation, requests and causal forecasts
  still belong to each look. Attended reads retain the live registry-wire
  gradient; salience retains its original stored-wire read. Replacing a bank
  drops its prepared values. The cache expires with the sentence.
- `_EntryIndex.of` transfers its immutable compressed topology once per
  rebuild. Pending additions remain current, and callers still receive their
  own lists. The device index and learned weights are unchanged.
- `MemberRead` shares basic-percept values only during one read-only
  `MemoryLink.menu` call, keyed by owner and autograd mode. A `finally` block
  expires the scope before any selected link writes or a later update runs.
- `LookForecast` obtains the observation width from shape metadata instead
  of synchronizing `slots.max()` on every forecast.
- `STM0` constructs its `STM1` directly instead of allocating an `LTM1` and
  immediately replacing it. Store formats, capacities and ownership are unchanged.

The wide-open pass still precedes one whole-field reconstruction, then narrowed
looks and optional thinking. The published 1.7 lesson uses the page's observed
percept activity, with its own fixations; the retired paced-text interface was
not restored. Per-look timestamps and reconstruction were not implemented.

## Measurements

One-thread CPU, arm64 macOS, Python 3.12, PyTorch 2.15.0, float32, batch 4;
`data/XOR_grammar.xml` and `data/lessons/reading-order.json` are identical in
each pair. Seed 1505 is used only for paired measurement, not acceptance tests.
The four repeated training pages are `hello world hello`, `world hello world`,
`red blue red`, and `blue red blue`. Each timed `runBatch` includes forward,
backward and one optimizer update. Twelve updates run; the first three are
reported separately and the remaining nine form the steady sample.

| Measurement | Released base | This change |
| --- | ---: | ---: |
| Eager steady median / mean, seconds per update | 0.57465 / 0.58161 | 0.49415 / 0.49785 |
| Eager warm-up updates 1–3, seconds | 0.98293, 0.65954, 0.59458 | 0.83757, 0.54745, 0.48781 |
| Native Inductor steady median / mean, seconds | 0.55618 / 0.58824 | 0.47028 / 0.50411 |
| Native Inductor first update, including compile, seconds | 3.74457 | 3.43608 |
| Inductor warm-up updates 2–3, seconds | 0.58553, 0.59213 | 0.48825, 0.48253 |
| Full reading lesson, qualification and restored assessment, seconds | 99.201 | 75.768 |
| Lesson epochs / training updates | 3 / 54 | 3 / 54 |
| Restored held-out sequences, sampled / evaluation | 122/122 / 122/122 | 122/122 / 122/122 |

The update median improved **14.0% eager** and **15.4% through the native
Inductor entry**. Lesson wall time to the same held-out result improved
**23.6%**. Every validation result agreed at each of the three epochs.
The Inductor runs used separate empty cache directories and each captured one
graph on the first update, with no new graphs in subsequent updates. These
are CPU measurements of this lesson; no accelerator speed claim is made.

Instrumentation runs on separate updates after the timed sample. The profile
contains both the greedy and explore candidates; inclusive times overlap
between rows and must not be summed as independent phases.

| Profiled operation per update | Base seconds | After seconds |
| --- | ---: | ---: |
| Compose both paths, 2 calls | 0.28337 | 0.24132 |
| Score both paths, 2 calls | 0.11910 | 0.10851 |
| Byte-span counts, 6 calls | 0.07177 | 0.02067 |
| Memory-link menus, 12 calls | 0.03210 | 0.01409 |
| Bank reads, 9 calls | 0.00913 | 0.00425 |
| Saved-tensor packing hook | 0.08476 | 0.07764 |
| Sentence backward and optimizer step | 0.04543 | 0.04499 |

The separate census includes explicit `.tolist`, `.cpu`, `.item` and implicit
Python `int`, `bool`, and `float` tensor reads. Counts fell from **23,736 to
20,115 per update** and **2,409 to 1,517 inside the sentence look loops**.
Calls from percept-boundary reads to `Codebook.lookup_rows` fell from
**1,079 to 419**. These are call counts, not distinct percept identities.

## Verification

The eager and native-Inductor paired replays each match exactly across 595
tensors: 86 parameters, 292 buffers, 190 optimizer tensors, the RNG state,
and 26 live parameter gradients. Gradients are captured immediately before
the optimizer step on a separate full update; post-update gradients are
cleared and are not used as evidence. The only buffer regions excluded are
uninitialized tails of nonpersistent priming scratch arrays; their initialized
lengths and prefixes are compared. Unit checks separately cover live registry
wire gradients, capped attention, tie gradients, and cache invalidation.

Fullgraph certificates exercise the byte-span kernel, whole/part min/max reads,
and prepared bank lookup with backward. The latter two use AOTAutograd. This
does not certify fullgraph capture of the Python dictionary/topology staging
or of all the sentence's candidate walks.

Fresh-start verification: the main group passed **107 tests**. It includes
the unchanged 99% reading gate, numeric XOR convergence, raised-class gate,
single reconstruction, candidate rollback, memory, causal forecast and
gradient-ownership checks. A separate unchanged 400-epoch verbal XOR run
passed **4/4 ordered grammar inputs and 4/4 ordered inverse outputs** (one
test, 302.49 seconds). Two independently initialized readers, one under the
checkpoint test and one through the standalone lesson command, each restored
and scored **122/122 in both modes**, with age 0.00.

The additional look/place/sparse-layer group passed **43 tests**, with one
long reading diagnostic deselected. Across the final three test runs:
**151 passed, one diagnostic deselected, no failures**. The 14 new performance
certificates are included in those counts. No full-suite sweep was run.

The release's documented intermittent loss of order during subsequent grammar
training remains assigned to item 1.45; this passing fresh run does not claim
to fix it. The known item 1.8 omitted-word failure is outside this change.

## Port of the 2026-10-30 preparation branch

Inspected `02e371aa95`, `0b4e01ac38`, `b42704aa94` and the plan at `9151b6f10b`.
Their older memory layout was not reintroduced.

| Old change | Disposition on the released base |
| --- | --- |
| Derived forms and stream-independent form cache | Already represented by `FeatureTable` and `MereologicalCodes`, using current order-zero memberships. Existing derive certificates retained. |
| Old `BoundView`, field-store binding and order-zero membership folds | The old field-store interfaces were replaced by the current order-zero Pi/Sigma and percept bank. Port the gathering principle to the current bank; do not restore retired readers or their incompatible tests. |
| Incremental LTM1 index | Already in `Memory.LTM1._index_pair`; existing add/reuse/retention/checkpoint tests retained. The same topology-reuse principle removes repeated reads in `_EntryIndex.of`. |
| Address-order candidate generator and cosine-based familiar/MTM matching | Superseded by explicit/extension-recalled candidates and current exact/parthood matching. Restoring cosine or expanding the bounded candidate set would change behavior. |
| Witness/definition/consolidation folds | Current sources use revised witness eligibility and order-zero membership semantics. Keep those contracts; share repeated percept reads during current memory-link menu construction. |

## Remaining costs and integration with 1.6

Candidate suffixes still execute sequentially, sharing the immutable opening.
Putting whole paths on a batch axis would change scheduling and the ownership
of STM, thought stores, fields, clocks, activity, error registries, and output
scratch. That requires a concrete architecture proposal for Alec before
implementation. The same applies to a unified MTM/promotion table and changing
the boundaries where saved graphs are detached. The saved-tensor hook remains:
packed sentences can update parameters while an earlier perception graph is
still needed. It is not safe simply to remove it.

Other remaining work is host topology/dictionary staging and complete lookup
capture; local fullgraph certificates are not a substitute for capturing the
entire live route. Prepared bank storage grows with rows × streams ×
occurrences and retains up to six factor tensors for the sentence. Larger
corpus and accelerator memory/performance measurements are still needed.

Potential 1.6 overlaps are specific:

- `BasicPercepts.PerceptBank`, `prepare`, `_occurrence_what`, `lanes`,
  `lookup_rows`: number/felt percepts should retain live codes and invalidate
  the prepared bank when replacing its table. `_boundary_codes`, `felt_of`
  and value-bin definitions were not changed.
- `Spaces.ConceptualSpace.cs_lookup`: the only edit prepares the completed
  bank after `Places.meet`.
- `LookForecast._extent`/`observe`: only width bookkeeping changed; preserve
  1.7's current `OccurrenceCredit` when resolving edits from 1.6.
- `Memory.IdeaRows.__init__`/`STM0.__init__`: relation-type construction only;
  `MemoryLink.menu`/`MemberRead.read`: scope of shared basic-percept reads.
- `Salience.rows_read` and `PerceptProperties.counts_in_spans`: reuse and
  layout only. `Layers._EntryIndex` caches host topology, never parameters.

## Reproduction

`bin/bench_item15.py` records the source SHA, hashes of every Python source,
configuration and lesson, device, environment, graph counts, update counts,
timings, profile and host census. `bin/compare_item15.py` verifies paired
state and gradients. Receipts remain local under `output/item1-5`.

For a pristine base without creating or changing a branch:

```sh
mkdir -p /private/tmp/item15-base
git archive ad8fafea927839c0d7f92d0aee35c913d02b2612 | tar -x -C /private/tmp/item15-base
OMP_NUM_THREADS=1 BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONHASHSEED=0 \
  .venv/bin/python bin/bench_item15.py --source /private/tmp/item15-base \
  --source-revision ad8fafea927839c0d7f92d0aee35c913d02b2612 --output output/item15-base
OMP_NUM_THREADS=1 BASICMODEL_DEVICE=cpu MODEL_COMPILE=none PYTHONHASHSEED=0 \
  .venv/bin/python bin/bench_item15.py --output output/item15-after
.venv/bin/python bin/compare_item15.py output/item15-base output/item15-after
```

Use fresh output paths. Add `--lesson` to each measurement for the full lesson.
For the native compiled pair, add `--compiled-step`, set
`MODEL_COMPILE=inductor MODEL_COMPILE_MODE=default`, and provide a separate
empty `TORCHINDUCTOR_CACHE_DIR` for each arm. Acceptance tests use fresh starts
without the measurement seed. No existing assertions are weakened.
