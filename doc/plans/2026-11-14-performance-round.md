# Laptop performance round — perf-1

Base: `3926005f12cabfd8233fc444963eae96c994d7ba` (GitHub `main`).
Local checkout: `/Users/arogers/github/basicmodel-perf`; local branch `perf-1`.
This round is local only. Nothing was pushed.

## Boundary and environment

The grammar and sentence owner, `thought`, `expect`, `intuit`, LTM reads,
placement, numeric percepts, registry, trust and checkpoint code remain as at
the base. Item 1.5's forms, membership readers, lookups and percept pass also
remain unchanged. The sentence loop is not compiled by this measurement.
The integration boundaries are §27 and §30 of the
[item 1.8 plan](2026-11-11-item1-8-last-architecture-round.md) and item 1.5 in
[todo.md](../../todo.md).

Machine: Apple M2, eight CPU cores, 24 GiB RAM, arm64, macOS 27.2.1
(26B131). Python 3.12.13; torch 2.15.0, torchvision 0.29.2,
executorch 1.6.0, pytest 9.2.0, NumPy 2.5.4, PyArrow 26.0.1.
The unpinned environment was installed as [Installation](../Installation.md)
specifies, using `make install PYTHON_BOOTSTRAP=/opt/homebrew/bin/python3.12`;
the explicit path was needed by Make's shell. `pip check` passed.
`bin/prepare_data.py` downloaded and authenticated both MNIST CSVs and the
FineWeb shard; the initial network preparation took 6.52 s. Network download
time is not compared with a later already-present-file verification.

No learner seed was selected. The sentence comparisons restore recorded
ambient Python, NumPy and Torch streams, rather than search for an
initialization. Paired subprocesses also reuse the hash setting recorded by
the original bounded worker's launcher. Leaf inputs use `arange`, with no random draw. Existing XML
capacities, corpus contents and test fixture dimensions are unchanged.

## Changes

* The default harness still collects unreachable models after every test and
  resets compilation caches at module teardown. It no longer scans the same
  live module graph again at the next test's setup. Worker memory limits,
  recycling, deadlines and shared-training boundaries are unchanged.
* The bounded batcher keeps an adjacent module together when it fits the
  existing case limit. The original ordinary selection still needs 46 batches;
  module executions fall from 721 to 720, with no ordinary module split at a
  batch boundary. Oversized modules retain the cap. `make testp` uses xdist's
  `loadfile` scheduling for the same fixture-reuse reason.
* `SparseLayer.fold_presence` stages integer edge positions from its existing
  host topology instead of doing a tensor boolean selection at every call.
  The full range needs no selection. Ranges invalidate when `_indices` is
  rebuilt. Its unique-edge gather uses an exact `index_copy` pullback: the
  general `index_select`/`index_add` backward exposed an MPS compiled-gradient
  error in this stack. All fold arithmetic and reduction order are preserved.
* The legacy two-coordinate `WhereEncoding.decode` stages a float32 MPS divisor
  tensor by device and period. Inductor otherwise replaces scalar
  division with reciprocal multiplication; at a large address that changed
  the final rounded integer by one. The tensor input keeps eager division
  rounding. The scratch cache adds no state-dict entry. The registry and
  period-setting methods are unchanged. Half precision retains the original
  scalar arithmetic, including its MPS rounding; integral carriers follow
  `atan2`'s promotion to float32.
* Corpus preparation reuses the SHA-256 it has already computed for a pinned
  source. Plain external files download directly into the authenticated,
  adjacent staging file before atomic installation, avoiding a second copy
  and hash pass. Archive/member/header verification and failure cleanup stay
  intact. Corpus loading is measured without a new mutable-data cache.
  Assembled row bytes, splits and addresses must match; each run's manifest
  naturally records its own timestamp and implementation digest.

## Measurement protocol

[performance_round.py](../../test/tools/performance_round.py) records the
timings, values and random streams. The optional
[pytest observer](../../test/perf_observer.py) records setup, call, teardown,
fixture setup and GC time without changing test execution. Local raw evidence
is under `output/perf-1/` (ignored), including the environment freeze, logs,
worker receipts and tensor references. Matched runs used the exact original
source in temporary baseline checkouts, removed after their evidence was saved.

CPU microbenchmarks use one Torch thread. Leaf timings are medians of five
blocks of 1,000 CPU (100 MPS) warm forward calls at batch eight, with the same gradient-enabled
contract as the certificate. Capture/AOT startup is recorded separately; the
shared Inductor disk cache is retained, so this is not a clean-cache compiler
benchmark. Corpus timings use five repetitions (three for assembly), with
1,000 complete FineWeb documents and all their sentences, and a local `file:`
transport for the plain-file installation comparison.

Sentence runs load the exact requested XML files. Lessons use the existing
two-stage curriculum and its four epochs, batch one. Math measures one complete
presentation of the frozen grade-00 corpus, batch eight with native tails;
it does not claim a full 17-epoch training campaign. Both run the native
training/presentation path with the repository's eager `while_loop` equivalent.
`TORCH_LOGS=graph_breaks,recompiles` is enabled for matched sentence and leaf
runs. No graph event in an eager sentence run is evidence of compilation.

<!-- MEASURED_RESULTS -->

## Measured results

The tables use the controlled baseline runs and the final-source after runs. All raw timing samples remain in their JSON files. Differences in unchanged paths are controls for run-to-run variation, not claimed speedups.

### Native sentences: CPU, eager, one Torch thread

| Measurement | Before | After |
| --- | --- | --- |
| Lessons completed sentences | 21 | 21 |
| Lessons construction (s) | 0.168 | 0.160 |
| Lessons runBatch time / completed sentence (s) | 5.176400 | 5.256960 |
| Lessons presentation wall time (s) | 109.228 | 110.925 |
| Math chain completed sentences | 60 | 60 |
| Math chain construction (s) | 0.295 | 0.299 |
| Math chain runBatch time / completed sentence (s) | 2.878274 | 2.913542 |
| Math chain presentation wall time (s) | 173.157 | 175.275 |

The math presentation completes all 60 sentences in nine native batches. Lessons complete 21 sentences and then both runs fail on the same attempted 22nd sentence: `ambiguous identity reading for word 152 ('here',): denotations [160, 182]`. The lesson wall time includes that failing attempt; the per-sentence figure uses only completed calls. The later curriculum stages were not reached, and no fixture was shortened to produce a success.

All recorded losses, breakdowns and answer strings, final model tensors, optimizer state and Python/NumPy/Torch streams match at zero tolerance, including the failed lesson state. The after sentence timings are about 1–2% higher. With one paired presentation per corpus, these runs do not establish a throughput change; no sentence-loop speedup or learning improvement is claimed. Both sentence logs contain zero Dynamo graph-break/recompile events because the sentence owner stays eager.

### Harness: an identical 256-case worker

| Measurement | Before | After | Change |
| --- | --- | --- | --- |
| Capped process wall time (s) | 80.237 | 56.621 | -29.4% |
| Test bodies (s) | 23.083 | 22.792 | -1.3% |
| Outside test bodies (s) | 57.154 | 33.830 | -40.8% |
| Pytest setup (s) | 24.278 | 0.122 | -99.5% |
| Pytest teardown (s) | 25.185 | 25.851 | +2.6% |
| GC across all phases (s) | 50.042 | 26.583 | -46.9% |
| Peak process-tree memory (GiB) | 2.691 | 2.750 | +2.2% |

This is the first full 256-case worker from the original default sweep (`worker-001`), replayed with its unchanged selectors, initial random streams and recorded hash setting, with the default worker environment (all six native/compiler thread limits set to one). Their final Python, NumPy and Torch streams also match exactly. Both runs have 255 passes and one skip, under the same 8 GiB process-tree cap. No other benchmark or sweep runs during this pair. GC time overlaps the pytest phases; it is not an additional summand. Evidence: `before/harness-default/` and `after/harness-default/`. This final pair supersedes the earlier pair that inherited native thread defaults, and the incomplete-data diagnostic that skipped the MNIST case.

### Complete default sweep

| Measurement | Before | After |
| --- | --- | --- |
| Selected / completed cases | 7008 / 7008 | 7047 / 7047 |
| Passed / skipped / failed / xpassed | 6670 / 333 / 4 / 1 | 6705 / 336 / 5 / 1 |
| Wall time (s) | 4,166.849 | 3,563.929 |
| Collection process (s) | 17.998 | 17.844 |
| Sum of worker wall times (s) | 7,150.010 | 5,952.549 |
| Test-body time (s) | 3,684.651 | 3,348.780 |
| Outside test bodies (s) | 3,465.360 | 2,603.769 |
| Outside fraction | 48.5% | 43.7% |
| Setup (s) | 2,090.476 | 1,220.831 |
| Teardown (s) | 1,014.454 | 1,023.812 |
| GC across all phases (s) | 2,078.818 | 1,211.611 |
| Other worker time (s) | 360.430 | 359.126 |
| Peak aggregate process-tree memory (GiB) | 7.344 | 7.768 |
| Fresh workers, including recycling | 61 | 62 |

Counts are unique selected tests; unittest subtest reports are not extra selected cases.

The baseline on this laptop spends **48.5%** of worker time outside test bodies, rather than the previously reported 61%. Its `trained_xor_grammar` fixture, configured for 400 epochs, alone takes 1,174.140 s of setup and passes its reconstruction bar. The redundant pre-test GC fixture accounts for another 875.159 s of setup. These are different kinds of work: the training remains required, while the second GC scan is removed.

| Fixture setup only | Before count / seconds | After count / seconds |
| --- | --- | --- |
| `trained_xor_grammar` | 1 / 1,174.140 | 1 / 1,182.476 |
| `_release_unreachable_test_models` | 6678 / 875.159 | 6714 / 0.062 |
| `trained_ladder` | 1 / 6.320 | 1 / 6.338 |
| `truth_model` | 11 / 4.377 | 11 / 4.344 |
| `model` | 14 / 4.067 | 14 / 4.101 |
| `named_scalar_model` | 18 / 2.248 | 18 / 2.139 |
| `loop_model` | 14 / 1.418 | 14 / 1.383 |

Fixture times are observed setup work, not separate additions to setup. Model creation and configuration that a test performs inside its body remain in body time. The ordinary selection is unchanged; the bounded planner reduces planned module executions from 721 to 720 without changing the original 46 batches or the case/file caps. Resource-triggered recycling can still split a module. The optional `testp --dist loadfile` change is not given a timing claim here.

The full baseline overlapped some small certificate/diagnostic runs; the full sweeps also have fresh, unselected learner initializations and retain the shared compiler disk cache. Their wall times are observational checks, not an isolated causal speedup estimate. The matched worker above isolates the harness change. Both retain the default 16 GiB aggregate budget, two 8 GiB workers, deadlines and memory recycling. Weekly slow coverage is absent in this new clone and is reported by the runner.

Baseline failures carried without repair:

* `test/test_doc_links.py::test_relative_links_resolve[doc/Params.md]`
* `test/test_doc_links.py::test_relative_links_resolve[doc/Philosophy.md]`
* `test/test_identification_evidence.py::test_native_sentence_carries_paired_identification_to_its_row`
* `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`

The two document links refer to `../../doc/Ethics.md` outside this standalone clone. The sentence identification and feature-layout assertions belong to the explicitly excluded integration work. The report does not claim a green baseline.

The final sweep has five failures: those four plus `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`. The fresh after initialization reconstructs three of four word multisets (`loving world` becomes `loving there`); the fresh baseline initialization reconstructs all four. Replaying the recorded after worker streams and hash setting on both source versions reproduces exactly the same 3/4 failure. All 292 final model tensors compare at zero tolerance, as do the initial and final Python/NumPy/Torch streams and all cached reconstructions. This observed failure is initialization-sensitive and is reproduced on the base; it is not repaired by choosing a seed.

The diagnostic replay worker wall times are 1,232.498 s before and 1,231.174 s after. These two diagnostic workers ran concurrently with the default one-thread settings and 8 GiB caps, so their times are descriptive, not an isolated speedup measurement. Evidence: `before/xor-replay/`, `after/xor-replay/` and `xor-replay-audit.json`. An earlier interrupted replay with inherited native thread settings is excluded.

All 7,007 other original test outcomes match. The 39 added cases produce 36 passes and three explicit MPS skips in the CPU sweep; those three MPS regressions pass separately, alongside all 31 MPS leaf certificates. All 1,159 source files recorded by the completed after sweep still match their recorded SHA-256 digests. The full sweep is complete, but it is not green; the paired equality evidence is reported separately from its fresh-initialization outcomes.

### Retained leaves: matched fixed inputs

Forward medians below are in **microseconds**, written **before → after**. The last column is seconds for capture plus six eager/compiled forward and gradient comparisons. That startup measurement retains the shared disk cache and is descriptive; cache-dependent reductions are not claimed as compiler improvements. The baseline legacy MPS integer decoder fails eager parity, so it has no valid compiled timing to compare.

#### CPU

| Variant | Eager µs | Compiled µs | Capture + comparisons s |
| --- | --- | --- | --- |
| `householder` | 121.13 → 122.25 | 41.27 → 41.00 | 1.107 → 1.211 |
| `householder_action` | 97.07 → 98.22 | 35.30 → 34.99 | 0.123 → 0.124 |
| `householder_action_transpose` | 96.42 → 98.28 | 35.50 → 36.34 | 0.112 → 0.111 |
| `lane_neutral_forward` | 104.17 → 105.16 | 47.70 → 46.64 | 0.537 → 0.552 |
| `lane_neutral_inverse` | 104.53 → 108.73 | 48.71 → 49.43 | 0.351 → 0.345 |
| `lane_learned_forward` | 104.56 → 104.73 | 49.06 → 47.68 | 0.329 → 0.330 |
| `lane_learned_inverse` | 104.15 → 105.59 | 48.48 → 48.29 | 0.332 → 0.330 |
| `cube_chart_True` | 20.96 → 21.79 | 23.89 → 23.31 | 0.114 → 0.114 |
| `cube_chart_False` | 17.97 → 18.16 | 23.74 → 23.07 | 0.043 → 0.042 |
| `cube_unchart` | 16.07 → 16.03 | 22.86 → 22.42 | 0.042 → 0.040 |
| `meet_ratios` | 25.05 → 24.56 | 28.68 → 28.19 | 0.052 → 0.051 |
| `fold_False_False_False` | 88.40 → 74.18 | 53.64 → 34.82 | 2.487 → 0.140 |
| `fold_False_False_True` | 87.31 → 83.27 | 52.82 → 36.55 | 2.225 → 0.302 |
| `fold_own_False_False` | 105.93 → 99.32 | 64.28 → 42.76 | 2.677 → 0.388 |
| `fold_False_True_False` | 113.82 → 97.38 | 54.73 → 36.29 | 2.322 → 0.128 |
| `fold_False_True_True` | 112.27 → 105.86 | 54.40 → 36.92 | 2.148 → 0.329 |
| `fold_own_False_True` | 129.68 → 124.33 | 64.63 → 43.17 | 2.582 → 0.418 |
| `fold_True_False_False` | 110.33 → 95.63 | 55.41 → 35.98 | 0.911 → 0.110 |
| `fold_True_False_True` | 109.74 → 104.26 | 53.87 → 37.26 | 0.967 → 0.309 |
| `fold_own_True_False` | 127.94 → 122.84 | 64.90 → 43.73 | 1.157 → 0.386 |
| `fold_True_True_False` | 89.82 → 78.34 | 53.15 → 35.70 | 0.765 → 0.104 |
| `fold_True_True_True` | 89.38 → 85.23 | 53.24 → 36.43 | 0.807 → 0.283 |
| `fold_own_True_True` | 107.42 → 102.43 | 66.14 → 42.89 | 0.989 → 0.360 |
| `fold_empty` | 13.95 → 6.13 | 15.41 → 13.93 | 0.083 → 0.031 |
| `fold_unsorted` | 81.69 → 76.69 | 49.31 → 36.70 | 1.889 → 0.281 |
| `where_encode_2` | 7.96 → 8.08 | 14.80 → 14.08 | 0.140 → 0.135 |
| `where_decode_2` | 16.00 → 16.45 | 22.10 → 21.73 | 0.041 → 0.041 |
| `where_decode_index_2` | 18.43 → 18.78 | 14.14 → 13.61 | 0.034 → 0.033 |
| `where_encode_4` | 13.70 → 13.91 | 15.14 → 14.83 | 0.032 → 0.031 |
| `where_decode_4` | 29.67 → 30.39 | 22.41 → 22.14 | 0.047 → 0.047 |
| `where_decode_index_4` | 25.39 → 25.52 | 14.12 → 13.66 | 0.037 → 0.037 |

#### MPS

| Variant | Eager µs | Compiled µs | Capture + comparisons s |
| --- | --- | --- | --- |
| `householder` | 532.16 → 483.96 | 273.05 → 272.47 | 1.595 → 1.136 |
| `householder_action` | 355.33 → 331.58 | 117.59 → 120.80 | 0.518 → 0.263 |
| `householder_action_transpose` | 329.04 → 325.83 | 119.70 → 120.24 | 0.555 → 0.300 |
| `lane_neutral_forward` | 369.59 → 365.82 | 121.56 → 120.23 | 0.604 → 0.711 |
| `lane_neutral_inverse` | 370.11 → 369.89 | 121.08 → 123.23 | 0.474 → 0.484 |
| `lane_learned_forward` | 368.48 → 371.82 | 120.31 → 119.79 | 0.471 → 0.475 |
| `lane_learned_inverse` | 371.93 → 369.79 | 119.30 → 122.29 | 0.474 → 0.478 |
| `cube_chart_True` | 95.11 → 96.86 | 42.04 → 42.16 | 0.178 → 0.181 |
| `cube_chart_False` | 91.52 → 90.61 | 41.10 → 41.59 | 0.141 → 0.077 |
| `cube_unchart` | 64.52 → 64.99 | 39.58 → 40.39 | 0.118 → 0.070 |
| `meet_ratios` | 86.69 → 92.23 | 74.78 → 74.94 | 0.195 → 0.097 |
| `fold_False_False_False` | 1,124.85 → 276.09 | 914.69 → 105.97 | 1.360 → 0.248 |
| `fold_False_False_True` | 1,127.46 → 307.89 | 1,006.46 → 117.99 | 1.284 → 0.440 |
| `fold_own_False_False` | 1,690.45 → 356.44 | 1,279.69 → 134.95 | 1.515 → 0.591 |
| `fold_False_True_False` | 1,274.81 → 366.50 | 1,031.31 → 123.43 | 1.477 → 0.293 |
| `fold_False_True_True` | 1,285.47 → 389.99 | 1,036.44 → 128.56 | 1.530 → 0.513 |
| `fold_own_False_True` | 1,769.42 → 459.52 | 1,307.33 → 144.95 | 1.761 → 0.678 |
| `fold_True_False_False` | 1,270.23 → 361.26 | 1,027.11 → 119.47 | 1.250 → 0.291 |
| `fold_True_False_True` | 1,277.51 → 385.46 | 1,044.56 → 127.39 | 1.244 → 0.507 |
| `fold_own_True_False` | 1,769.82 → 479.35 | 1,344.93 → 144.15 | 1.357 → 0.669 |
| `fold_True_True_False` | 1,211.30 → 280.07 | 996.47 → 111.62 | 1.050 → 0.239 |
| `fold_True_True_True` | 1,209.30 → 314.11 | 1,017.22 → 119.41 | 1.083 → 0.440 |
| `fold_own_True_True` | 1,690.69 → 369.86 | 1,290.24 → 135.61 | 1.169 → 0.597 |
| `fold_empty` | 52.26 → 33.89 | 27.05 → 26.33 | 0.104 → 0.050 |
| `fold_unsorted` | 970.39 → 284.55 | 783.68 → 119.28 | 1.208 → 0.415 |
| `where_encode_2` | 65.58 → 49.18 | 28.05 → 27.20 | 0.055 → 0.058 |
| `where_decode_2` | 64.38 → 59.57 | 37.20 → 37.04 | 0.135 → 0.066 |
| `where_decode_index_2` | invalid baseline → 80.49 | invalid baseline → 27.53 | invalid baseline → 0.071 |
| `where_encode_4` | 83.30 → 85.86 | 31.29 → 29.76 | 0.067 → 0.068 |
| `where_decode_4` | 101.41 → 94.65 | 38.89 → 44.03 | 0.087 → 0.089 |
| `where_decode_index_4` | 90.29 → 86.75 | 28.12 → 27.61 | 0.095 → 0.101 |

All 31 final variants on each device produce one full graph, zero batch recompilations and no checked synchronization operations in forward or backward. Before and after logs emit zero graph-break/recompile events under the same explicit shape contract. The original 13 nonempty fold variants still run `aten::nonzero` despite that successful capture. The original MPS `where_decode_index_2` fails exact integer parity; the final float32 path passes. All final eager values and gradients match their baseline references at zero tolerance.

### Corpus preparation and loading

| Operation | Before (s) | After (s) | Change |
| --- | --- | --- | --- |
| Verify existing prepared data | 0.098110 | 0.103762 | +5.8% |
| Authenticate SHA-256-pinned source | 0.082722 | 0.041602 | -49.7% |
| Install plain file via local transport | 0.141533 | 0.076234 | -46.1% |
| Assemble 1,000 complete documents | 0.977438 | 0.943329 | -3.5% |
| Read and validate all corpus rows | 0.089705 | 0.089809 | +0.1% |
| Load curriculum splits and addresses | 0.154886 | 0.157324 | +1.6% |

The plain source is the 94,511,680-byte pinned FineWeb shard. The assembled corpus contains 39,289 sentences. Shard byte digests, row counts, the complete row sequence, loaded text splits and source addresses all match. Existing-data verification and loading use unchanged code paths; their small variations are not attributed to the preparation edits.

* Canonical row SHA-256: `7fe4b8cac580a0f44bcbdb8671692915996ab2a7419c55a52ff00feef49af213`.
* Loaded splits/addresses SHA-256: `8d2a28758c59bf0bfb8c474001344d9153246f0d5c2609b957e0461f4f7aee79`.

Evidence directories: `before/*-controlled/`, `after/*-final/`, `before/default/`, `after/default/`, their observer directories, and the corresponding top-level logs in `output/perf-1/`. Earlier exploratory runs are retained there; the tables use only the explicitly named matched runs.

## Leaf contract and equality

[performance_leaves.py](../../test/performance_leaves.py) certifies 31 fixed-input
variants: both Householder actions, neutral and learned LaneKernel in both
directions, both cube-chart modes, cube unchart, unequal-width meet ratios,
all conjunction/dual/range fold combinations, explicit own/edge knowledge,
empty topology, unsorted topology, and both two- and four-coordinate codecs at
capacity `2**26`. Forward values and all participating input/parameter
gradients are compared with eager; integer indices must be exactly equal.
Householder uses four 16-coordinate reflections and the action has two lanes;
LaneKernel and the cube maps use width 32, and meet ratios use widths 32/24.
Sparse fixtures have 16 source rows, eight output rows and seven edges, with
the ranged case covering rows `[1, 6)`. These fixed microbenchmarks measure
leaf overhead; the sentence measurements retain the full XML inventories.
Floating compiled results use `rtol=2e-5, atol=2e-6`. Before/after eager values
and gradients use zero tolerance.
The two-coordinate codec retains its original precision limits; integer
equality here means equality to eager, including at large addresses.

The compiled signature fixes widths, dtype, device, topology, mode flags and
layout. Positive batch sizes `3, 1, 2, 4, 8, 17` share one graph. Batch axes
are explicitly unbacked (`mark_unbacked(..., shape_id='batch')`, with strict
specialization checking), other
dimensions stay static, scalar configuration floats are specialized, and
shape duck-sizing is disabled. These settings are scoped to the leaf
certificate; they do not change sentence compilation defaults. Sparse topology
and the MPS divisor are staged before capture. Fullgraph capture, graph counts,
FX inspection and a runtime profiler jointly check for graph breaks,
recompilation and `nonzero`/`item`/`_local_scalar_dense` host synchronization.
The runtime check covers forward and backward. The baseline captured a full
graph while 13 nonempty sparse-fold variants still executed `aten::nonzero`;
these dynamic selections synchronized MPS. All 31 final variants have none of
the checked operations on either CPU or MPS. Unmodified leaves are timing
controls; small timing differences in them are not claimed as improvements.

An extra CPU zero-batch diagnostic is not included in the positive-batch claim:
ordinary contiguous `[N, 0]` tensors have stride `(1, 1)` rather than `(B, 1)`,
and Torch specializes that different layout in the sparse folds. It produces
a second graph at zero; the other leaves retain one graph. No fixture was
reshaped to conceal this boundary.

## Reproduction

From either source checkout, use that checkout's `bin` first on `PYTHONPATH`
and this branch's `test` directory for the certificate/driver. Use a fresh output
directory for every command. For example:

```sh
PYTHONPATH=bin:test BASICMODEL_DEVICE=cpu MODEL_COMPILE=none \
  TORCH_LOGS=graph_breaks,recompiles .venv/bin/python \
  test/tools/performance_round.py --output output/perf-1/new-leaves leaves
PYTHONPATH=bin:test BASICMODEL_DEVICE=mps MODEL_COMPILE=none \
  TORCH_LOGS=graph_breaks,recompiles .venv/bin/python \
  test/tools/performance_round.py --output output/perf-1/new-mps leaves --device mps
PYTHONPATH=bin:test BASICMODEL_DEVICE=cpu MODEL_COMPILE=none \
  .venv/bin/python test/tools/performance_round.py \
  --output output/perf-1/new-corpus corpus
PYTHONPATH=bin:test BASICMODEL_DEVICE=cpu MODEL_COMPILE=none \
  TORCH_LOGS=graph_breaks,recompiles .venv/bin/python \
  test/tools/performance_round.py --output output/perf-1/new-lesson \
  sentence lesson --replay output/perf-1/before/lesson-controlled
BASICMODEL_DEVICE=cpu PYTEST_PLUGINS=perf_observer \
  PERF_OBSERVER_DIR="$PWD/output/perf-1/new-observer" \
  make test TEST_ARGS="--run-dir $PWD/output/perf-1/new-default"
```

For baseline leaf audits, add `--audit-baseline` after `leaves` to record the
original synchronizing gathers. `--record-eager` writes the fixed reference;
the top-level `--compare` accepts its `leaf-reference.pt`. Sentence comparisons
accept the baseline `result.json`; corpus comparisons accept `corpus.json`.
For a sentence comparison, export the recorded `pythonhashseed` value as
`PYTHONHASHSEED` as well as using `--replay` for the learner's random streams.
Sentence comparisons check losses, breakdowns, answers, status/failure,
tensor/optimizer state and final random streams. Corpus comparisons check the complete
row sequence and loaded split/address sequence by SHA-256, not just counts.
