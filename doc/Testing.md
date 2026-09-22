# Bounded development tests

Run affected files or individual cases during development:

```sh
make test TEST_ARGS='test/test_query_registry.py --timeout 120 --suite-timeout 600'
make test TEST_ARGS='test/test_query_registry.py -k declaration'
```

Run the complete default suite in the background before a commit:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py
```

`make test` uses the same runner. `make test_all` additionally enables the existing
`RUN_SLOW=1` tests, assigning marked slow cases to an available GPU and ordinary cases to CPU.
`make testp` is deliberately a direct `pytest-xdist` iteration command (`TEST_JOBS=auto`
by default), not a bounded receipt. `make preflight` and `make preflight_full` use the
receipt runner. Select affected tests for short development cycles; the complete default
suite remains the commit gate. Compilation remains enabled for tests that exercise it.

## Fast regression and slow experiments

Long convergence, memorization and performance experiments use `@pytest.mark.slow`.
The central collection hook skips them before fixture setup unless `RUN_SLOW=1`.
`RUN_SLOW=0` stays fast. The older per-test `RUN_SLOW` gates remain supported.
A slow check keeps its assertions and any explicit expected-failure status;
skipping it in the default run is not evidence that its learning gate passed.

```sh
make test_all TEST_ARGS='test/test_output_path_supervised.py -k memorizes --timeout 1800'
```

The initial timing audit found a 119-second isolated compiler test, a 72-second
XOR training test and 130 seconds of repeated 20M-model blind-decoder training.
Two unmarked expected-failure arithmetic experiments also trained for 40 and
30 epochs. 245 long-running functions now use the central slow gate. These include
repeated corpus-learning experiments, six-batch phrase/checkpoint training,
multi-step Adam checks, full compiled reconstruction-order integration and
production-size 20M configuration/reconstruction
integration checks. The next bounded audit reached its 30-minute worker deadline
while tracing/backpropagating a three-batch tied-reconstruction integration. That
run failed the gate; its process receipt and stack sample identify the workload.
The L1 multi-batch case, full-model backward/cache checks and the large serial
output fixture now run explicitly with `RUN_SLOW=1`. The smaller compiler,
numerical training, ordinary forward/backward, ownership,
checkpoint and query regressions stay in the default suite. Run the slow checks
explicitly when their corresponding training paths change.

Native answer fixtures use 264-wide concepts and 136-wide percepts during
development. Their ownership, checkpoint and gradient checks retain distinct
widths. Seven corresponding 1032-wide cases remain available under `RUN_SLOW=1`.
Repeated packed training, expectation toggling and multi-brick optimizer
integrations also use the slow gate. A further default profile completed only
912 of 4,458 cases in 26.2 minutes before an explicit stop: repeated 20M legacy
builds caused 4–7 GiB footprints and frequent fresh-worker restarts. The same
production-size recurrence, dual-input/tower, symbolic-iteration, decode,
export and per-word compiler fixtures now run with `RUN_SLOW=1`, as do full CLI
training/reconstruction checks. Lightweight geometry, masking, query, dictionary,
checkpoint and compiled-kernel checks remain in the default suite.
The subsequent profile completed 1,246 of 4,458 cases in 25.8 minutes before
another explicit stop. A fixture audit found additional 16,000–65,000-row legacy
configurations in global attention, reading, masked semantics, word storage and
serial reconstruction. Their full-model checks now use the same slow gate;
small layer-level lifecycle, loss, projection and runtime-inference checks stay
in the default suite. All assertions and reasoning methods are retained.
The compact numerical readout and metadata tests continue covering larger
production shapes without materializing their full training graph.

The next attempt recorded 2,907 of 4,458 cases before a 30-minute worker timeout.
That batch completed its two full reconstruction graph-capture checks and its
tied-gradient check, but could not finish its remaining selection. The full-model sentence-traversal integration checks and a measured
280-second sequence-training check now use the slow gate. The first traversal
check alone subsequently measured 102 seconds in the affected run.
Their function bodies and assertions are unchanged. The direct binary recomposition/residual checks in that file and the
smaller operator, reconstruction-objective, byte-fidelity and role/gradient
regressions remain available in the default suite. This timed-out attempt is not a passing validation result.

Memory- and time-triggered recycling permits larger case batches (256 cases / 16 files)
without retaining a worker past its resource boundary. The fast runner uses a
small concurrent pool under one aggregate reservation rather than one serial
worker; it retains process isolation and hard caps while avoiding needless
Python/PyTorch import waits. Per-case timings in the durable receipt identify
further expensive checks; a timeout is a failure that needs investigation,
never an automatic slow mark.

## Limits and isolation

- One full receipt per user across worktrees, enforced by a nonblocking file
  lock. Selected-file, `-k`, and marker runs may proceed concurrently.
- Up to `cpu−4` concurrent, one-thread fresh workers by default: **10** on
  this 14-core machine, leaving four cores for SSH and interactive work. Each
  receives at most 256 selected cases from 16 files; a new process releases
  its model and compiler allocations at a boundary. `--workers` can lower the
  pool for diagnosis.
- The aggregate default is physical RAM minus **8 GiB**: **28 GiB** on this
  36 GiB machine. Each worker retains an **8 GiB** `taskpolicy -m ... -P kill`
  ceiling; the supervisor samples every active worker tree and stops the
  largest worker if their aggregate physical footprint crosses 28 GiB. This
  supports legitimate 4.8 GiB cases without allowing the pool to consume the
  8 GiB reserved for the machine. `--memory-gib` may lower the aggregate
  budget, but cannot leave less than 8 GiB for the machine.
- At 80% of a worker deadline, or at the earlier of 80% of its 8 GiB cap and
  that cap minus 64 MiB (never later than half its cap for a small cap), a
  worker finishes its current test and exits. A fresh worker runs remaining
  cases. Coverage records count one final completion per selected case; resource
  recycling does not repeat cases. The hard memory limit still applies within a
  test, and a killed worker is a failure.
- An ordinary pytest assertion, setup or teardown failure does **not** cancel
  the remaining workers. The receipt keeps its `test_failure` outcome and
  runs every selected case so a diagnostic full run reports all failures.
  Process-boundary failures (timeout, memory kill, unavailable accounting,
  collection/protocol failure, interrupted execution or invalid coverage)
  still stop the pool immediately to protect the machine and preserve an
  honest receipt.
- A typed Inductor `CppCompileError` reporting a modified input to a stale
  precompiled header gets **one** fresh-worker retry with
  `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0`. Compilation still runs;
  shared caches are not deleted. The receipt records `compile_cache_retries`
  and marks the original report `compile_cache_retry`; raw worker JSON and
  logs retain the original failure. The retry must pass normally. A second
  cache failure, an ordinary compiler error, or any simultaneous assertion,
  setup or teardown failure remains a failure. No other errors are retried.
- CPU, BLAS and compiler pools use one thread. The existing on-disk Inductor cache
  remains reusable between workers. macOS workers run at `nice -n 10`, not the
  throughput-throttling `taskpolicy -b` background class.
- Default deadline: 1,800 seconds per worker, including collection; 10,800 seconds
  for the whole suite. Configure finite limits with `--timeout` and
  `--suite-timeout`. Tests exceeding a limit fail the run; they are not skipped.
- `--batch-size 1` reduces accumulation within a batch when diagnosing a memory
  failure. Reducing batch size does not omit any selected cases.

## GPU training tests

Significant training belongs in the explicit slow run and uses the available
accelerator. With `RUN_SLOW=1` and no explicit device, marked `slow` cases run
on the available GPU and ordinary cases run on CPU. The pool permits ordinary
CPU workers alongside it but admits only one accelerator worker at a time, so
MPS/CUDA training never contends with a second shared-device test. On this
machine the GPU resolves to the Apple M4 Max's MPS device.
A GPU request fails explicitly when no accelerator is available. The quick
regression gate defaults to CPU. Older manual slow gates without a `slow`
marker still need the marker audit listed in the checkpoint; request MPS
explicitly for their significant training until that audit is complete.
An explicit `BASICMODEL_DEVICE=mps` or `cuda:0` is preserved. For a compatibility
test that requires CPU, select that test and request CPU explicitly; report it
separately from GPU training evidence.

```sh
RUN_SLOW=1 BASICMODEL_DEVICE=mps .venv/bin/python test/test_report.py test/test_inter_contrastive_predict.py
```

Worker receipts identify the resolved device. End-to-end GPU timing must include
compilation, forward, backward and optimizer work; no speedup is implied by
device selection. The same aggregate memory, thread and deadline limits apply.

macOS accounting uses `proc_pid_rusage` physical footprint, including compressed
memory. RSS alone can become misleading during compression. Each macOS worker
has an 8 GiB kernel limit through `taskpolicy -m ... -P kill`, while the
supervisor samples all active process groups and enforces the 28 GiB aggregate
reservation. Linux accounting uses RSS plus swap. The monitor samples each
process group and known children, including compiler subprocesses; sampling can
allow brief overshoot.
Timeout, memory exhaustion, unavailable accounting or interrupted execution
terminates the worker and its children and returns failure. Linux enforcement
is sampled, without the additional macOS kernel cap. Unsupported platforms
fail before running tests.

## Evidence and reports

Each invocation creates a new `output/tests/<timestamp>-<id>/` directory, or the
new directory named by `--run-dir`. It contains:

- `source-manifest.json`: validated hashes of code, tests, data/grammar and
  configuration, plus separately recorded documentation and `todo.md` hashes.
  Documentation changes do not void a code receipt; code/data/grammar changes
  still do.
- Collection output and the complete list of selected pytest node IDs.
- Per-worker logs, process exit status, elapsed time and peak measured memory.
- Per-worker JSON progress after collection and each test transition, including
  the active case and confirmed completed cases. `result.json` lists all active
  worker progress files; the aggregate completion list updates when each worker
  exits. A later timeout retains earlier confirmed completions.
- `result.json`: durable overall status, limits and exact selected/completed cases.
- `report.html`: escaped test diagnostics, including setup and teardown failures.

A run is green only after every selected case completed exactly once, every
worker exited successfully and the source snapshot still matches. Expected
failures and explicit skips retain their pytest outcomes. Empty selection,
changed source, missing coverage, collection errors and killed workers cannot
receive a success receipt. In-progress receipts remain incomplete even if all
workers completed so far have passed. Logs and receipts survive disconnection.

Do not edit files in the tested snapshot while a suite runs. Use an isolated
checkout for independent implementation work. Direct `python -m pytest` calls
(including `make testp`) are the fast iteration path and do not create a bounded
receipt. Use the bounded entry point for full commit gates.

## Validation

Resource probes exercise real hanging processes, resistant children, aggregate
child memory, native compressed-memory accounting, monitor failures, lock
exclusion, fresh-worker coverage, teardown failures, source mutation, empty
selection and interruption with persistent evidence. The checkpoint tooling selection passed **29 passed in 138.27s; peak aggregate footprint 0.56 GiB**
(20 resource/coverage cases, six device cases and three slow-switch cases).
Earlier affected-file runs and their tested source hashes are archived separately;
those earlier runs do not validate later runner revisions.

### Fast aggregate-runner evidence (September 19)

The reviewer probes were intentionally red in
`output/tests/20260919-000218-47b7c4`: the old runner had neither an
eight-GiB-reserved default budget nor a parallel-worker API. After the pool
implementation, the aggregate-budget/real-overlap probe passed in
`20260919-000715-e5f66d`. It proves two isolated workers overlap while their
independent caps remain subject to one aggregate reservation.

The complete resource, recycling and device selection then passed **28/28** in
`20260919-001157-622faf`. A first 80%-recycling attempt correctly failed its
128 MiB allocator fixture before the hard cap; the fixed 64 MiB headroom made
that same fixture recycle safely, while the time probe verified the new 80%
deadline boundary. The MPS-routing file passed **7/7** in
`20260919-001547-f7d2d6`, including the two-worker probe that confirms two
accelerator training workers do not overlap.

### Complete diagnostic receipts (September 19)

The reviewer probe `test_test_failure_does_not_cancel_remaining_worker_coverage`
was red under the old policy in `20260919-091149-3f647d`: a deliberate first
assertion failure left the inner receipt at its first selected case. The repaired
policy passed the focused probe in `20260919-091247-f6225b`: it retains exit
status `1` and `test_failure` for the deliberate red case while completing the
second and third selected cases. Resource, deadline, protocol and coverage
failures remain fail-fast; only ordinary pytest failures receive complete
diagnostic coverage. The complete bounded-runner file then passed **25/25** in
`20260919-091322-343583`.

### Selected direct-unary meaning (September 19)

The reviewer probe for a completed `what(quantize(x))` action program was red
in `20260919-044105-39096d`: the former binary-only adapter returned no
meaning. The direct concept-unary adapter passed its recovery/gradient probe in
`20260919-044334-1c1008`; its normal-controller `code` result probe passed
alongside it in `20260919-044414-7a428b`. The companion arbitrary-ID guard,
which proves that a direct leaf cannot manufacture the description occurrence
needed by `arma`, passed in `20260919-091719-52637f`. The source-matched full
default receipt then passed **4,664/4,664** in **195.2 s**, peaking at
**18.7 GiB** aggregate, in `20260919-091747-2d70c9`.

Claude's throughput review then supplied five concrete follow-ups. Their new
reviewer probes were red in `20260919-002323-cd7319`; the repaired policy passed
5/5 in `20260919-002553-8b53ae`: no `-b` throttle, `nice -n 10`, `cpu−4`
execution capacity, full-receipt-only lock, recorded-but-noninvalidating prose,
and direct xdist iteration. The aggregate-memory kill probe passed in
`20260919-002627-a87223`: two workers below their individual caps were stopped
when their combined footprint crossed 256 MiB. These focused receipts establish
the resource and routing mechanism only; the current source still requires a
full default receipt and measured end-to-end throughput before the gate closes.

After those policy changes, the full runner/recycling/device selection passed
**35/35** in `20260919-002831-c030ea`. It includes the restored direct-xdist
iteration target, lock/snapshot behavior, real child cleanup and recycling,
aggregate cap, ordinary CPU dispatch, explicit MPS routing, and the one-lane
accelerator concurrency probe.

The current fast **default** receipt passed **4,649/4,649** selected cases once
in `output/tests/20260919-003054-07304d`: 31 fresh workers, 10 execution slots,
8 GiB per-worker caps, a 20.20 GiB observed aggregate peak, and 196.80 seconds
end-to-end. The prior serial receipt took 4,164.65 seconds for 4,640 cases, so
this is a measured current throughput result, not an extrapolation from a
microbenchmark. Its source manifest validates 608 code/test/data/configuration
files and separately records 61 prose/todo hashes.

**September 18 default-suite coverage record.** The September 17 checkpoint
attempt remains an incomplete historical receipt: it exited 124 after 5,136
seconds with 2,907 of 4,458 selected cases completed. After the OS update,
Alec directed the runner to finish every case without re-running cases that had
already passed. The resulting composite receipt record covers all **4,502**
node IDs in the then-current default selection exactly once after deduplication:
the last remaining selection, `output/tests/20260918-004212-cf8aae`, passed
273/273 in 386.55 seconds. Earlier portions of the record are preserved in
`output/tests/20260917-131759-a07e8f` through
`output/tests/20260918-003129-a73d62`.

Two real failures found while completing that record were fixed and their
specific nodes were re-run green: the output-only walk in
`20260917-231508-6f272c` (30/30), and the peer language pipeline in
`20260917-234649-21de8b` (23/23). The latter revealed that a local
`LanguageSpace` without an output loop must not construct legacy output-rule
keys. The full default record therefore does not describe either failure as a
pass.

This is deliberately a **no-rerun composite coverage gate**, not one fresh
single-snapshot success receipt: source marker changes and the VQ tile-bound
fix occurred while the historical coverage was being completed. It is the
validation form explicitly requested for this session; future source changes
require their ordinary affected tests and a fresh bounded full receipt.

### Thought-operator catalogue record (September 18)

The catalog-unification snapshot was run through the bounded default selection
at `output/tests/20260918-101358-b952cf`: all **4,623** selected node IDs
completed once in 38 serial fresh workers. Exactly one node failed:
`test_signal_router_has_grammar_ops_attached` still assumed that unary grammar
kernels were unwrapped, while the common structural-face contract now wraps
both unary and binary kernels. The production code was not changed for that
failure; the reviewer assertion was corrected to inspect the wrapped layer and
its exact bounded re-run passed **1/1** in
`output/tests/20260918-112216-01823c` (19.8 seconds, 0.41 GiB peak).
The explicit guard that keeps deferred `true` out of the production catalogue
also passed **1/1** in `output/tests/20260918-112724-2a007a`.

The earlier item-0 catalog/contract regression selection passed **137/137** in
`output/tests/20260918-091955-3962bf`; direct context/catalog probes passed
**38/38** in `output/tests/20260918-091419-96aacc`. Per Alec's explicit
no-rerun instruction, the 4,622 nonfailing default nodes were not rerun after
the assertion-only repair. This is current composite coverage evidence, not a
fresh single-snapshot all-green receipt; no source behavior changed after the
full selection.

The audit retained every assertion. Full server construction, full traversal
and compiled `runBatch` integration checks are marked `slow`, while their
compact contract checks remain default. A real VQ failure showed that the
former 4 GiB default distance-tile budget could exceed the 8 GiB bounded-worker
cap through allocator retention. The default is now 512 MiB. Its two
large-flat OOM probes passed under the explicit slow CPU gate in
`20260918-004019-7f77ee` (2/2; about 29.4 seconds and 1.41 GiB peak per case).

The explicit current supervised MPS training check also passed in
`20260917-131359-7cb733`: requested and resolved device were MPS, the worker
took 170.806 seconds, and peak aggregate footprint was 2.36 GiB. This proves
routing and bounded execution, not a controlled MPS speedup or a held-out
utility result.

### Explicit `<thought>` catalogue (September 19)

The corrected catalogue began with red bounded reviewer probes in
`20260919-024405-84a3aa`: a `<thought>` selection was ignored, structural-only
forms leaked into the registry, and a conflicting role declaration was
accepted. A selected thought-only form then correctly exposed the missing
structural-family assumption in `20260919-025908-3f603a`, and an exact
`isPart` declaration exposed its missing checked descriptor in
`20260919-030338-bc3909`. The repaired loader retains thought declarations
separately, allows selected thought-only forms, joins matching structural
faces only after role validation, and fails closed when no executor exists.

The final affected selections passed **114/114** in
`output/tests/20260919-030521-44dd7c` and the broader grammar/controller
selection passed **311/311** in
`output/tests/20260919-030535-01e73f`. They cover explicit membership,
structural omission, thought-only forms, exact `isPart`, descriptor failure,
legacy `<Queries>` rejection, phase gates, query VP boundaries and lifecycle
contracts.

The first current default receipt,
`output/tests/20260919-030555-026513`, collected 4,655 nodes and completed
1,727 before one warm 194-case worker jumped to 12.96 GiB and was killed. Its
aggregate peak was only 13.97 GiB, below the 28 GiB suite reservation; this was
worker accumulation, not an aggregate-cap breach or a test assertion failure.
Per the explicit no-rerun instruction, the remaining node IDs were recovered
only once in fresh 16/32-case batches. Green receipts covered 1/1
(`20260919-030947-4e7358`), 700/700 (`20260919-031100-5f6a87`), 225/225
(`20260919-031141-b9d4f0`), 1,000/1,000
(`20260919-031304-62fdab`), 2/2 (`20260919-031727-2bd475`), and 46/46
(`20260919-032312-ae21f2`). A transient source-snapshot invalidation in
`20260919-031556-440285` occurred after 751 otherwise passing nodes; its
current 608-file source manifest again equals the frozen manifest, but it is
not presented as a green receipt. The stateful
`test_what_thinking_episode.py` module was consequently checked once in its
native collection order (**10/10**, `20260919-032221-078c5e`) after an
alphabetically reconstructed shard demonstrated that its module fixture cannot
be meaningfully split by node ID.

Together these receipts exercise every one of the 4,655 collected default
nodes on the unchanged source, with the corrected thought catalogue tests
green. This is deliberately composite coverage, not a single fresh green
default receipt: future source changes require their own affected tests and a
new frozen full receipt.

### Direct lexical meaning and normal-route quarantine (September 19)

The reviewer probe for an unreduced lexical `[NP1, VP, NP2]` relation was red
in `output/tests/20260919-033451-e57145` (**2/2 expected failures**): the
adapter returned no meaning when the middle word was an installed native VP.
The repaired direct adapter passed **2/2** in
`output/tests/20260919-033626-3a7262`, preserving the live signed noun leaves,
the canonical native VP, and the intended operand-only gradient route.

The normal-route reviewer probe was red in
`output/tests/20260919-033828-bd076e` (**1/1 expected failure**) because an
unselected surface prompt still called `answer_query()`. After the production
handoff was removed, its exact rerun passed **1/1** in
`output/tests/20260919-034105-e1c0e0`; the dense compatibility topology was
also protected by **2/2** in `output/tests/20260919-034349-fe62f1`.
The combined meaning/catalogue/controller/phase/work selection passed
**116/116** in `output/tests/20260919-034137-49803c`. This verifies the
narrow adapter and normal-path quarantine; it does not claim completion of
paraphrase, nested-reference, residual-credit, learned-utility, or full-suite
gates.

The affected dense output-synthesis module also passed **26/26** in
`output/tests/20260919-034617-30b357`. The first current default receipt then
collected 4,658 nodes and passed 2,812 before the old
`TestWriteMask.test_partition_isolation` smoke test exceeded its worker cap
after twelve earlier legacy-model constructions in that same process
(`20260919-034713-fef3fa`, 11.23 GiB). The still-uncompleted node passed alone
**1/1** in `20260919-035338-24e7c8` at 1.76 GiB, proving cumulative worker
state rather than an intrinsically oversized test. The exact remaining 1,845
node IDs then passed in fresh 8-case workers in
`20260919-035428-56e69c` (302.13 seconds; 12.24 GiB aggregate peak).

The three receipts have identical 608-file source manifests and together cover
each current default node exactly once: **2,812 + 1 + 1,845 = 4,658** passing
cases. This is current-source composite default coverage under the explicit
no-rerun policy, not a single green receipt; the first receipt's cap breach is
retained rather than hidden.

### Anchored lexical converse provenance (September 19)

The reviewer probe for an unreduced lexical `whole` converse was red in
`output/tests/20260919-041551-7522bb` (**1/1 expected failure**): recovery
selected the first shared-VP form (`part`) and reversed the canonical roles.
The repair passed **8/8** focused selected-meaning/native-ID cases in
`output/tests/20260919-041836-2d0155`. It retains a detached grammar-form
classification rather than a raw surface, row, or native-ID feature.

A second owner probe was red in `output/tests/20260919-042140-06b248`
(**1/1 expected failure**): it proved that form classification had incorrectly
read the raw InputSpace record rather than the PartSpace word segmentation.
That exact probe and the anchor-spelling/converse check passed **2/2** in
`output/tests/20260919-042345-c87b31`; the complete affected thought,
registry, phase, and selected-meaning set then passed **135/135** in
`output/tests/20260919-042411-0f7ebf`.

The first fresh default attempt exposed a clone-safety regression from using a
non-pickleable `MappingProxyType` for the private grammar snapshot:
`output/tests/20260919-042441-af77b2` stopped at **1,654/4,660**, with the two
deep-copy tests in `test_compiled_word_chunk.py` failing. The two failures
were reproduced red in `output/tests/20260919-042542-ba6855`, then the
clone-safe copied snapshot plus lexical probes passed **4/4** in
`output/tests/20260919-042614-e53e23`.

The new frozen full default receipt is
`output/tests/20260919-042649-57d821`: **4,660/4,660** cases passed in 195.49
seconds, at a 18.74 GiB aggregate peak under the 28 GiB cap. The failed first
attempt remains recorded above; it is not counted as a green receipt.

### Forward-owned lexical forms (September 19)

The review of `943a64f` identified a second text lookup at program capture:
`_word_lexical_forms` reclassified PartSpace's transient `word_texts` through
the anchor table instead of retaining the forward's decision. The row-dispatch,
missing-provenance, and ordinary-forward probes were red in
`output/tests/20260919-121232-1f0289`. That run also caught a packed-fixture
bucket mismatch; after using a fixed 16-word fixture, both ordinary and packed
forward probes failed for the intended missing-form reason in
`20260919-121308-c9f0aa`.

The eager word-row staging now resolves each WORD's form once and retains it
by WORD row. Capture neither reads text nor repeats the anchor lookup, and
unresolved words remain unclassified. The selected-meaning file passed
**22/22** selected cases in `20260919-121433-f7074c`. The final affected
selection, including additional soft/hard-reset and restaging assertions,
completed **172/172** selected cases with a green receipt in
`20260919-121554-8d81e1`. It covers packed and ordinary capture, distinct WORDs
sharing an OBJECT, unknown provenance, detached recall, native IDs, the normal
controller, structural checkpoints, compiled word loops, output ownership,
reconstruction, word storage, and batch isolation.

Two fresh full attempts stopped at the unchanged per-worker memory cap:
`20260919-121914-ef91a4` completed **2,521/4,667** cases with the default
256-case/16-file batches, and `20260919-122113-2130d6` completed
**2,725/4,667** with 32-case/4-file batches. Neither recorded a pytest failure;
neither is a passing full receipt. The second run's active reasoning case
passed alone in `20260919-122420-d322df` at **1.76 GiB**. Both affected worker
groups then completed **155/155** cases in `20260919-122509-f550d4` using
eight-case, one-file batches; the largest worker peaked at **5.78 GiB**.
This changes process lifetimes only, with no omitted tests, assertion changes,
or higher resource limits.

A subsequent eight-case full run, `20260919-122610-9cc2f0`, stopped at
**389/4,667** because an existing category fixture temporarily wrote an XML
file inside `data/`. The source hashes matched again after fixture cleanup,
but the run correctly rejected the transient mutation. The new fixture
isolation probe was red in `20260919-122953-8d39a1`. Eight temporary XML
creation sites across six test files now use the system temporary directory.
Their affected selection passed **44/44** cases in `20260919-123031-39a665`,
including the existing guard that rejects actual source mutations. Model
assertions and the runner's source validation are unchanged.

The next full run, `20260919-123135-1a2b27`, completed **4,668/4,668**
cases with two failures and no resource or source-validation error. Isolated
workers exposed a peer-pipeline probe that inherited its eager gradient-anchor
setup from an earlier test, and a routing-credit probe whose linear-sum loss
is invariant under its addition operator. The peer fixture now initializes
the required anchors itself; the routing probe uses squared values to give
different pairings different costs, with a fixed seed and the same nonzero
gradient assertions. Both files passed **27/27** cases, each in its own fresh
worker, in `20260919-124907-090f08`.

The final frozen default selection passed with **4,668/4,668** cases completed
in `output/tests/20260919-125200-894ef3` (1,039.56 seconds). It used
`DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python
test/test_report.py --batch-size 8 --max-files 1`, with the unchanged ten-worker
pool, 8 GiB worker cap, and 28 GiB aggregate cap. Peak worker footprint was
**7.19 GiB**, and aggregate peak was **11.33 GiB**. The receipt's source
manifest was compared with the final implementation and matched exactly.
The default 256-case/16-file grouping's memory instability remains a runner
residue in `todo.md`; smaller batches validate the same default selection.

This closes the item-1 anchoring review note only. General non-anchor and
nested-reference meaning, typed answer adapters, and controller unification
remain in `todo.md`.

### Query-phase evidence (September 18)

The rebased phase reviewer probes first failed in
`output/tests/20260918-010057-1c5b25` before the mask existed. The expanded
reviewer set then passed 23/23 in 166.9 seconds in
`20260918-011031-938ed0`; it includes real fullgraph forward/backward and
trace/eager-island negatives. The broader phase-affected selection passed
280/280 in `20260918-011401-fd5b1c`, using three bounded fresh workers with a
1.56 GiB peak. This is affected-file evidence for the changed phase code, not
a claim of a new single-snapshot global receipt under the no-rerun policy.
See [Query phases](QueryPhases.md).

### Nested-retention evidence (September 18)

The rebased nested-retention reviewer probes first failed in
`output/tests/20260918-014842-f679b1` (22/22 expected failures). The focused
selection passed 68/68 in 76.1 seconds at a 0.50 GiB peak in
`output/tests/20260918-015808-1dce43`. The broader LTM, sidecar,
thought-history, query-occurrence and checkpoint selection passed 309/309 in
`output/tests/20260918-020312-c526c2`, using two fresh workers (73.5 and 28.9
seconds; 0.50 GiB peak). These are affected-file evidence after the historical
no-rerun composite, not a newly rerun global snapshot. See
[nested retention](NestedRetention.md).

### Shared query-work evidence (September 18)

The rebased reviewer run was red in
`output/tests/20260918-021028-e3efa7` because the QueryWork module did not
exist. The focused reviewer selection then passed 13/13 in 20.1 seconds at a
0.40 GiB peak in `output/tests/20260918-021708-e98757`. The broader,
non-overlapping query/phase/taxonomy/expectation/history/retention selection
passed 372/372 in `output/tests/20260918-021926-07eb2a`, with fresh workers
at 1.05 GiB, 0.50 GiB, and 0.40 GiB peaks. This is affected-file evidence
after the historical no-rerun composite, not a new single-snapshot global
receipt. See [shared query work](QueryWork.md).

The actual MPS sequence-training check passed: 877.81 seconds in `runEpoch`
for two batches, 1.24 seconds for construction and a 2.54 GiB peak aggregate
footprint. All 137 parameter tensors, including the predictor, were on MPS.
This used the current `eager` backend and was slower than an earlier 280-second
CPU test call under different run conditions. It is not a controlled speedup
comparison, a held-out utility result or a current supervised-answer throughput
measurement. GPU performance tuning remains open.

### Expectation-retention evidence (September 18)

The retained-estimate reviewer probe first failed as intended in
`output/tests/20260918-204026-f5887b`: no observation occurrence could yet be
bound to the row-local expectation view. The implementation probe then passed
1/1 in `20260918-204810-91888a`. Its capacity and actual pending/packed writer
selection passed 10/10 in `20260918-205730-f7e900`; this includes the rule
that the last free LTM slot belongs to the understood external observation,
not its forecast. The first broad pass correctly exposed an old two-observation
count assertion after a new estimate row made the durable history three rows;
the assertion now checks the explicit pair and observation-only predictor
view instead.

The final affected selection passed **236 passed, 5 slow-skipped** in
`output/tests/20260918-205842-148281` (278.55 seconds, 1.06 GiB peak). It
covers structured prediction/lifecycle, all three observation writers,
consolidated LTM and checkpoint sidecars, origin compaction/nested retention,
typed prediction query consumers, and ordinary thought-history boundaries.
It validates retained occurrence ownership and detached checkpoint fidelity;
it does not establish metadata prediction, residual-policy credit, learned
utility, or throughput gates.

A subsequent full run reached 4,010/4,639 cases before the seeded depth-3
relative-end-state probe exposed a lifecycle regression
(`20260918-210435-23f39f`). The new writer had attempted to bind provisioning
rows while external observations were suspended; the provisioning wrapper
recovered the exception per text, leaving that parse lifecycle incomplete.
The reviewer probe was red in `20260918-223655-2ea861`; after gating retention
to actual external boundaries, the direct chain, slow global-LTM and depth-3
probes passed 4/4 in `20260918-223806-83083d`, and the affected LTM,
expectation, chain and thinking selection passed 192/192 in
`20260918-224214-d9227c`. Generic LTM recurrence and attention now exclude
estimate rows; this is still ownership containment, not residual-policy or
learned-utility evidence.

[Checkpoint summary](benchmarks/2026-09-17-bounded-test-data/checkpoint-summary.json),
[latest tooling receipt](benchmarks/2026-09-17-bounded-test-data/checkpoint-tooling-result.json),
[timed-out full receipt](benchmarks/2026-09-17-bounded-test-data/reconstruction-profile-result.json),
[MPS measurement](benchmarks/2026-09-17-bounded-test-data/mps-training-audit.json),
[partial timing profile](benchmarks/2026-09-17-bounded-test-data/partial-call-profile.json).

## Future work: bounded fixture reuse

Evaluate compatible preallocated fixture reuse with complete reset checks and
separate construction/compilation counters. Existing fixed-capacity contracts
remain in place. [Assessment and acceptance criteria](plans/2026-09-17-bounded-fixture-reuse.md)
record the follow-up; comparative speedups remain unmeasured.

## Selected meaning and one controller (September 20)

**Review correction:** `288b56b` passed regression but did not complete item 1.
Its standalone language codec violated the three-grammar architecture, and its
natural-wording experiment is not accepted learning evidence. The review
correction below removes that path. Grammar-owned wording remains open and
learned questioning utility remains unproven.

The original landing consolidates public and normal reasoning in `run_selected_thought`,
retires the other controllers and next-idea selector, carries typed
set/code/child answers, retains nested meanings through all three writers, and
adds a separate supervised codec, subsequently rejected and removed.
[SelectedMeaning](SelectedMeaning.md) specifies ownership, credit and limits;
[KernelRetirement](KernelRetirement.md) maps all 61 removed frame-kernel tests.

Development receipts were red before the relevant fixes:

| Receipt under `output/tests/` | Observed failure |
|---|---|
| `20260920-024809-ab08f1` | Public entry points still used alternate paths; hard candidate values retained gradients. |
| `20260920-025130-36147a` | Set, code and child results lacked owned typed answer adapters. |
| `20260920-030228-999e68` | Nested selected meanings did not survive compose and observation. |
| `20260920-031028-602bfe` | Retired controller hooks remained; policy cost counted choices instead of actual shared work. |
| `20260920-033908-1796db` | Tool-user facade survived and set generation spent a fresh word allowance per member. A separate training probe also had a stale staged-input fixture, subsequently corrected. |
| `20260920-035122-712aa9` | The real two-row training probe found internal thought references defaulting to row zero. |
| `20260920-040001-ef5ef4` | The broad affected run completed 545/545 cases but failed legacy narrow-conditioner migration. |

The final checkpoint correction passed `20260920-040354-15296e`: 7/7 selected
cases completed, exit 0. This includes native-width and legacy strict reloads,
the trained natural-language experiment and the real model's language
loss/optimizer/checkpoint integration. The broader affected run had already
passed the controller, row-local rewards, output, nested meaning, taxonomy and
retirement contracts. Neither receipt substitutes for the full landing gate.

The rejected language experiment trained 36 annotated examples at seed 19 for 240 Adam
steps, then checks held-out nouns, the parthood sense of “has,” “contains,”
converse roles, alternate-sense negatives, execution, generation and
recomposition. All relation wordings appeared in training; its programs were
synthetic, and its function vocabulary was restricted to `a`, `has`, `equals`.
These results do not demonstrate grammar-owned interpretation or generation.
The controller tests separately establish exact zero-init baseline
behavior, actual-work policy credit and a real optimizer update. This is not the
multi-seed learned reasoning utility study, which remains item 4; on FineWeb,
controller reward awaits item 2's residual credit.

The first full default receipt, `20260920-040537-5788f1`, completed
4,630/4,630 cases in 1,029 seconds with three failures. One expected inline
candidate descriptions to be rejected before retaining their complete children;
two expected the retired repeated-presentation/parity loop. The updated tests
check no-write candidate formation, refusal to execute before occurrence binding,
retention without fact authority, one presentation, and no invented closure.
Their preserved controller contracts are linked in the retirement audit.
The implementation source was unchanged by these final test migrations.

The corrected boundary/presentation tests and normal-controller regression
selection passed `20260920-042401-e18b3d`: 42/42 cases completed, exit 0.

The final **source-matched default receipt** is
[`20260920-042439-ea7ff2/result.json`](../output/tests/20260920-042439-ea7ff2/result.json):
**4,630/4,630 selected cases completed, exit 0**, in **1,026.3 seconds**.
Unique outcomes are 4,299 passed, 330 skipped and one expected failure; there
are no failures. The full selection used `--batch-size 8 --max-files 1`, ten
workers and unchanged limits (8 GiB per worker, 28 GiB aggregate, 1,800-second
worker and 10,800-second suite deadlines). Peak footprints were 7.19 GiB per
worker and 10.80 GiB aggregate. No selector or test exclusion was added.

All 615 entries in the receipt's `validated_source` manifest matched a fresh
`bounded_tests.source_snapshot()` of the `288b56b` landing source. Its SHA-256, over the
sorted compact JSON map, is
`6adaa745620cc69e44cccf773ae49b00769dbae10dc7235d05c32074e403e100`.
The selected and completed case sets are identical. Final receipt notes and
todo reconciliation were written after the run; they do not change that tested
source map. The user-owned thought-operator specification edit is excluded from
the implementation commit.

This establishes the original regression result only. The review reopens item
1; expectation/residual credit, the generation-catalogue migration and the
full learned-utility study remain items 2–4.

## Three-grammar review correction (September 20)

The separate `LinguisticMeaningCodec`, its fallback and pre-generate route,
`WhatStepChooser`, `TruthInterval`, nearest-word realisers and the soft bridge
policy are removed. The sole controller now receives mode, polarity,
alpha-renamed bindings/scope and bounded attended visible STM/LTM content.
Incomplete-context checkpoint policies and their optimizer moments reset;
current policies restore strictly. The chooser can select the active question
as a nested `what(Q)` through its already-owned occurrence.

The depth-one/two probe fits the real MLP to authored routing examples, then
runs its selections and actual episode credit without a scripted chooser.
That is mechanism evidence only, not useful learned decomposition. The
natural-wording replacement must run through compose/generate on actual
forward text and a normal generation vocabulary; that gate is still open.
The matched-compute, multi-seed utility comparison remains item 4 and gates
every learned-utility claim.

Development receipts under `output/tests/`:

| Receipt | Result and disposition |
|---|---|
| `20260920-050623-6107ad` | 5/5 completed, all failed: missing semantic context and forbidden extra codec/policy. |
| `20260920-051310-78166c` | 77/77 completed, 20 failed: deque slicing, fixture graph reuse and old context-width assumptions. |
| `20260920-051418-af928a` | 125/125 completed, three failures and three skips: nested fixture routing and a stale fixture variable. |
| `20260920-051535-54ea52` | 44/44 completed, depth-two fixture still selected an unwanted third descent. |
| `20260920-051632-3adeb1` | 9/9 completed, one failed auxiliary training-loss threshold. It was replaced by an assertion that every trained discrete routing choice matches its label; actual depth/support/credit checks stay strict. |
| `20260920-051736-eec5e0` | 9/9 passed with both actual chooser-selected depths. |
| `20260920-052549-f3e4a4` | 167/167 passed across the affected controller, context, adapters, queries and retirement tests. |
| `20260920-052710-743926` | Full default selection completed 4,599/4,599 in 1,025 s; two failures: five deleted-file links in the older plan and a test monkeypatch of the deleted bridge builder. Both are corrected without restoring a legacy path. |
| `20260920-054832-33af0f` | Final focused selection completed 112/112 with exit 0, covering review probes, all documentation links, public taxonomy entry points, numerical readers and the remaining math/configuration tests. |

All 19 previously unexplained test deletions have individual dispositions in
[KernelRetirement](KernelRetirement.md), along with the tests removed or
migrated in this correction. The taxonomy public-entry probe now asserts the
bridge builder is absent and still forbids the global vector-read route.

The final **source-matched full default receipt** is
[`20260920-054917-5d73b8/result.json`](../output/tests/20260920-054917-5d73b8/result.json):
**4,599/4,599 selected cases completed, exit 0**, in **1,029.4 seconds**.
Unique outcomes are **4,268 passed, 330 skipped and one expected failure**.
The selected and completed case sets are identical; no selector or exclusion
was added. The default selection includes the documented slow-test skips and
does not establish any open learning or throughput gate.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
4.46 GiB per worker and 12.56 GiB aggregate. All **614** entries in
`validated_source` match a fresh `bounded_tests.source_snapshot()` of the
landing source. The SHA-256 over its sorted compact JSON map is
`e34d256402a69525f256c9928c4ff6d23901d6f013edbd2ac1e67126d27051fa`.
This receipt note was added after the run; it does not change the validated
source map. The user-owned thought-operator specification edit remains
untouched and excluded from the correction commit.


## Grammar-owned wording architecture (September 20)

`d1d8b5b` is an architecture/wiring increment, not completion of item 1. Its
initial completion claim incorrectly moved the working natural-wording gate to
item 4. The user's clarification relaxed structural-preference and routing-share
measurements, not that original gate. At that commit language quality remained
unproven and item 1 remained open. See [SelectedMeaning](SelectedMeaning.md).
No additional LM, interpreter, realiser or auxiliary language loss is added.

The existing compose MLP now retains ordered operand and role context. A zero
input projection into its existing first hidden layer preserves old predictions
and initialization order; ordinary reconstruction/answer credit trains it.
BasicModel enables the declared generate walk and its existing supplied-answer
policy objective. The natural word `equals` is removed from the shipped
technical anchor tables. The ordinary forward program and output trace own the
actual choices; no hand-annotated `AnswerProgram` supplies the new text probe.

Development receipts under `output/tests/`:

| Receipt | Result and disposition |
|---|---|
| `20260920-065559-6d8750` | 2/2 completed, both failed: the mean-only chooser could not distinguish reversed pairs, and normal text training had no operand-order parameter. |
| `20260920-070111-86005c` | 4/4 passed: learned sensitivity to concept/role order, old/current checkpoint behavior, optimizer moments and full-graph execution. |
| `20260920-070203-0196da` | 1/1 completed, failed: BasicModel had not enabled its generate walk or output-policy objective. |
| `20260920-070250-9f3989` | 85/85 completed, exit 0: grammar chooser, output walk, checkpoint and reconstruction checks, including an ordinary text reconstruction batch using production generation configuration. |
| `20260920-070644-0bd526` | 3/3 completed, all failed: the three shipped grammars still assigned `equals` to the equality operator before learning. |
| `20260920-071333-6860c6` | Focused selection completed 110/110 with exit 0, covering the corrected anchors, order projection, selected-meaning recovery, chooser parity, grammar fixtures and documentation links. The previously passed 202-second text integration runs again in the full default selection. |
| `20260920-071513-b13772` | Interrupted explicitly after 692/4,607 completed, exit 130, to correct a GPU RNG compatibility issue found during review. This incomplete receipt is not a landing gate. No source was edited until its workers had exited. |
| `20260920-071829-c758ce` | 1/1 completed, failed on MPS: a CPU-only RNG fork did not prevent the new zero-initialized Linear from advancing the GPU stream. |
| `20260920-071913-1d8538` | 54/54 completed, exit 0: the projection now allocates its weight directly as zeros without a random draw; CPU and MPS RNG checks, chooser architecture, optimizer/checkpoint restoration and full-graph execution pass. |

The text probe presents “a bicycle has a wheel” and “a wheel belongs to a
bicycle” through the real input/forward/reconstruction path and `runBatch`.
It checks live compose-MLP updates and generate-path invocation, while missing
answer labels leave generate-policy weights unchanged. It does **not** assert
that those sentences have learned canonical meanings or fluent outputs. The
existing supplied-answer output tests separately check actual policy updates.
All prior tests are retained; the new probes add coverage rather than replacing
language-quality evidence or retiring any further tests.


The final **source-matched full default receipt** is
[`20260920-071956-ad78e0/result.json`](../output/tests/20260920-071956-ad78e0/result.json):
**4,608/4,608 selected cases completed, exit 0**, in **1,036.3 seconds**.
Unique outcomes are **4,277 passed, 330 skipped and one expected failure**.
Selected and completed case sets are identical. The full default selection
has no added selector or exclusion; its documented slow-test skips do not
establish the remaining language-quality, utility or throughput goals.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
7.19 GiB per worker and 10.80 GiB aggregate. All **615** entries in
`validated_source` match a fresh `bounded_tests.source_snapshot()` of the
landing source. The SHA-256 over its sorted compact JSON map is
`3338902b6483d4217b064f52b36abe083c64adb4fd58bddd27c7ce393f88581b`.
This receipt note was added after the run and leaves that source map unchanged.
The user-owned thought-operator specification edit is excluded from this commit.


## Working grammar wording gate (September 20)

This completion keeps the original item 1 gate with item 1. A selected
`surface` attachment preserves the complete right semantic subtree while
retaining the marker numerically. Both attachment and wait decisions are made
by the existing grammar MLP; a small marker prior is computation inside the
operator. The normal generate grammar emits word concepts before the lexical
inverse compares against every known spelling. No LM or fallback interpreter
is introduced. [SelectedMeaning](SelectedMeaning.md) records the design and
the scope of the result.

The explicit slow run uses seed 931, 1,599 actual forward-parsed training
sentences, 1,000 generate-learning steps and 8,000 compose-learning steps. It
holds input encodings fixed for this focused grammar study. Annotated operand
variables get teacher-state full-code augmentation. There are 32 development
cases and 24 additional cases with three previously unused complete marker
wordings and eight previously unused nouns, in both orders. Result: 56/56
comprehension/control cases, 53/53 natural relation outputs and 53/53 meaning
recompositions. The three alternate `have` cases do not invent part relations.
All 178 observed word spellings compete in lexical inversion. This is a small
supervised compositional result, not general English, unsupervised learning,
or useful learned questioning. The causal utility study remains item 4.

Development receipts under `output/tests/`:

| Receipt | Result and disposition |
| --- | --- |
| `20260920-082112-a31393` | Real text exposed mismatched concept-identity owners. Registry and input now use the first-stage allocator. |
| `20260920-083914-c585c3` | Compose-only 31/32; not accepted as a generation gate. |
| `20260920-084751-eedf09` | Generated words worked, but twelve parses failed; not accepted. |
| `20260920-085402-fa09bb` | 29/32 parses: the correct modifier operation lost to an isolated copy score. Neighbor context now reaches that decision inside the same MLP. |
| `20260920-085823-56d59a` | Resumed development checkpoint passed 32/32 and generation/recomposition. A fresh run was still required. |
| `20260920-090413-8194c8` | 16/16 checks passed, including fresh training, normal compiled reconstruction/`runBatch` updates, unlabelled-credit isolation, chooser checkpoint/optimizer/fullgraph behavior. |
| `20260920-091026-4437c5` | 109/109 passed: the expanded fresh 56-case language run plus affected selected-meaning, output, adapter, grammar-layer and bounded-runner tests. |

The shipped XML validates against `model.xsd`. This ordinary CLI smoke exited
0 after one training batch (reconstruction loss 1.2483), and saved its normal
checkpoint. Its untrained diagnostic output is **not** language-quality evidence:

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager BASIC_MAX_BATCHES=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python bin/Models.py data/MM_grammar_wording.xml
```

Reproduce the focused language gate with:

```sh
BASICMODEL_DEVICE=cpu RUN_SLOW=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py test/test_surface_grammar.py::test_real_text_has_a_complete_selected_meaning --batch-size 8 --max-files 1
```

The final explicit slow receipt is
[`20260920-095850-51ad00/result.json`](../output/tests/20260920-095850-51ad00/result.json):
**1/1 passed**, 154.5 seconds, 1.24 GiB peak on CPU. It reruns all 56 cases, 53 generated
relations and 53 recompositions from fresh training at seed 931 on the final source. Its 620
validated source entries match the full-suite snapshot. The default suite skips
this marked slow study; these are separate receipts. No existing tests are deleted.

The final **source-matched full default receipt** is
[`20260920-095850-082122/result.json`](../output/tests/20260920-095850-082122/result.json):
**4,616/4,616 selected cases completed, exit 0**, in **1,039.3 seconds**.
Unique outcomes are **4,284 passed, 331 skipped and one expected failure**.
Selected and completed case sets are identical. No selectors or exclusions
were added to the default suite; the explicit slow receipt above supplies the
separate language-quality evidence.

Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Ten CPU workers used the unchanged 8 GiB worker / 28 GiB aggregate limits and
1,800-second worker / 10,800-second suite deadlines. Peak footprints were
4.46 GiB per worker and 8.70 GiB aggregate. All **620** entries in
`validated_source` match a fresh `bounded_tests.source_snapshot()` and the
explicit slow receipt. The SHA-256 over the sorted compact JSON source map is
`e2c6a0c66dc5c1578c267f6a3e374ae2d49928742cb557ef9f692222ce088255`.
This receipt note was added afterward without changing that source map.
The user-owned thought-operator specification and figure edits are excluded
from the completion commit. Learned questioning utility remains unproven.

The first full run, `20260920-091556-2e1cc5`, completed all 4,615 cases
in 1,055.3 seconds with five failures. Two revealed that separate-stage
taxonomy configurations must retain their terminal registry owner; readers now
use the registry's actual owner. One exposed a structural taxonomy neighbor
without a numerical payload: the bounded reader now retains its unavailable
reference and reports incompleteness without allocating or aborting. One width
assertion still expected the symbol's content slice; it now checks the full
opaque concept width. The fifth required identical float32 top-k index order
across GEMM tilings. Seed 2385 reproduces a two-neighbor swap with distance
differences no greater than 2.4e-7. That test now checks unique selected rows and
distances against a float64 projective-distance oracle, with a 16-epsilon
absolute bound. No production ranking code changes. These are corrections,
not deleted tests or exclusions. The complete source is rerun after the fixes.

Focused correction receipt `20260920-093532-2bf724` passed 41/41 cases,
including both taxonomy ownership/checkpoint failures, actual per-row controller
credit, the new missing-payload case, opaque grammar width and lexicon geometry.
The next full run `20260920-093631-a0ea7f` was interrupted at 377/4,616 cases
(exit 130), with no observed failures, after a direct MPS check showed that the
new float64 oracle needed CPU storage. Its concurrent slow run was also stopped.
All workers exited before the test was edited. Only the reference calculation
moves to CPU; the lookup under test stays on its selected device. Neither
interrupted run is a landing receipt.

The portable lexicon checks passed **27/27 on MPS** in
`20260920-093838-4d5332`. The earlier slow receipt
`20260920-091556-ea2413` also passed before the full-suite corrections; the
explicit CPU receipt linked above supersedes it for this landing.
The intervening slow receipt `20260920-093930-b0e9b7` also passed before the
straight-through estimator correction below; the final receipt repeats it afterward.

The next full run, `20260920-093930-5223b0`, completed all 4,616 cases in
1,038.9 seconds: 4,283 passed, 331 skipped, one expected failure and one
failure. Its remaining failure exposed cancellation in the codebook's
straight-through estimator, `e + (q - e).detach()`: a small quantized value
can be rounded away even for unit-scale encoder values. A deterministic
probe reproduced it in `20260920-095755-75737b`. The implementation now forms
the zero-valued gradient carrier first, `q.detach() + (e - e.detach())`,
preserving the quantized forward value exactly and the identity encoder
gradient. The retained test now checks those values, gradients and the absence
of target gradients on fixed cancellation cases. All 11 affected checks passed
on CPU (`20260920-095823-3d7c1a`) and MPS (`20260920-095823-23fb9e`). All full-run
workers had exited before either source or test was changed. Both the full
default suite and the fresh slow language study are rerun after this correction.


## Gradient factorization (September 20)

Item 1b replaces the global reconstruction-priority projection with ordinary
shared gradients and a measured boundary: output receives a detached concluded
idea and detached contextual inputs, while the generation operator's own
parameters train. Expectation still reaches current-step preceding encodings;
its target and previous-step context detach. The thought chooser's credit
trains its policy only. [Contract](GradientFlow.md).

At the 1b landing, the production dictionary used the shared optimizer
(`conceptualContextLearningRate=0`); item 1d below restores rotation ownership.
That landing's normal-batch diagnostic verified nonzero codebook reconstruction credit
and actual operator reconstruction/output overlap, in addition to standalone
cosine arithmetic. Sparse measurements never densify the inventory. The run
harness logs weighted R/O/E norms, R/O and R/E cosines and names persistent
opposition. No per-operator guard was justified by these short observations.

The first boundary probe, `20260920-151056-40365e`, failed both output paths
on leaked conclusion-state credit. The pre-removal diagnostic receipt
`20260920-152014-c24ed9` passed 9/9 cases with the old projection still
present. Only then was the projection deleted. Recheck
`20260920-153716-11fde3` covered corrected boundaries, the executed tied
inverse and diagnostics. `20260920-154540-010d7e` covered the final shared
codebook configuration.

The final explicit slow receipt, `20260920-160301-0d078f`, passed both
real packed reconstruction/expectation studies with `RUN_SLOW=1` in 362.8 s.
The final **source-matched full default receipt** is
`20260920-161333-09e757`: exit 0, all 4,609 selected cases completed
(4,277 passed, 331 skipped, one expected failure), 1,055.7 s and 14.93 GiB
peak aggregate memory. Command:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

Both receipts match the final 622-file validated source map. Its SHA-256
over sorted compact JSON is
`ffc0bd8308926ab88a6370d131ceb1fce6a60368bec70a2bc0e8de35f55a9faa`.
The preceding full receipt, `20260920-155402-3e98c2`, completed the same
4,609 cases with one failure: the plan still linked to the renamed test file.
Those links were corrected before the final full run. Receipt notes and the
todo reconciliation were added after validation; neither changes tested source.

`test_reconstruction_priority.py` becomes `test_joint_objectives.py`. Its
real packed prediction, exact reconstruction-family learning, shared ownership,
independent heads, sparse updates, AMP, one-backward/one-step, truth scaling and
seal-layout checks remain. The retired tests' dispositions are:

| Old check | Disposition |
|---|---|
| prediction_can_refine_a_reconstructable_representation; model_balances_prediction_and_thinking_without_answer_labels | Projection-specific synthetic checks replaced by the actual joint sum and retained real packed training checks. |
| fidelity_tolerance_uses_unscaled_loss_under_amp; invalid_reconstruction_tolerance_fails; invalid_ratio_fails | Deleted knobs reject configuration; ordinary AMP scaling remains tested. |
| projection_and_combined_norm_budget; zero_and_missing_reconstruction_keep_bounded_output_credit; enormous_opposing_output_does_not_erase_reconstruction | Projection/budget behavior is retired; ordinary sums and observational opposition reporting replace it. |
| large_finite_gradients_do_not_overflow_projection; sparse_projection_matches_dense_without_capacity_allocation | Large finite and sparse/missing/dense cosine checks moved to `test_gradient_factorization.py`; no gradient modification remains. |

The immediate-stop generation fixture proves the state cut and generator
credit. A separate real `lower` inverse fixture executes the shared operator
and proves its nonzero gradient; an unused operator is never counted as evidence.


## Accessible-mind effects (September 20)

Item 1c replaces descriptor scope strings with ten checked subsystems and
explicit effect targets. The ordinary controller commits detached effects;
structural reads stay live owned snapshots. Order-zero `part` leaves its
vector residual, while higher-order parthood emits bounded symbolic support.
The existing LTM writer owns per-role leaf-code tensor columns and their
checkpoint/compaction lifecycle. Cued `what` retrieves an old matching frame
into ordinary thought history; the chooser's recent-store scan is gone.
[Current contract and limitations](AccessibleMind.md).

| Coverage | Evidence |
|---|---|
| Grammar permissions, structural capability refusal, detached thought effects | `test_accessible_mind.py`, `test_thought_operation_catalog.py` |
| Meronymic residual and higher-order symbolic value, including missing paths | `test_accessible_mind.py`, `test_query_vp_boundaries.py`, `test_grammatical_query_vps.py` |
| Old-row cue retrieval, priming, scope, fan, familiarity and contiguity | `test_accessible_mind.py`, `test_query_contract_boundaries.py` |
| Exact leaf postings, forward-forest indexing, checkpoint, compaction and actual codebook remap | `test_accessible_mind.py`, `test_nested_retention_owners.py` |
| Normal `what` effect, detached knowing, retained frame ownership and no implicit recent-LTM read | `test_accessible_mind.py`, `test_thought_review.py` |
| Budget cutoff, nested return/finish, typed result retention | `test_normal_thought_controller.py`, `test_nested_retention.py`, `test_thought_checkpoint_credit.py` |
| Measured recovery by depth, training since storage and distinct-code chain length | `test_mind_generativity.py`; rates and limits in [AccessibleMind](AccessibleMind.md#measured-limits) |

No existing tests were deleted. Taxonomy fixtures now use higher-order symbols;
order-zero `part` has its own geometric checks. Structural fixtures carry the
geometry capability and expect live cloned compose inputs. Description
occurrences use a charged direct address, so one record of allowance reaches
an old row; zero allowance still rejects it. `lookup` checks a previously held
frame and first proves that an unseen row is inaccessible. The controller
trace includes its ordinary explicit conclusion when reduced read work leaves
room for that action.

Development receipts under `output/tests/`:

| Receipt | Result |
|---|---|
| `20260920-154425-c452b7` | Initial leaf-index probe failed as intended before implementation. |
| `20260920-155402-05c5b9` | 45 index/retention checks completed green. |
| `20260920-161847-25c26a` | 70 affected cases completed green, including a real normal training batch. |
| `20260920-162453-258d6a` → `20260920-162719-910047` | The 89-case edge run found a nested cutoff append; fixed, then 33 selected cases completed green. |
| `20260920-163001-88f9de` | 67 checkpoint/retention cases completed green. |
| `20260920-163525-ebc533` | New failing probe: a held retrieved frame was missing from retention roots. Fixed by walking typed result evidence on the existing owner. |
| `20260920-163651-f187ca` → `20260920-163752-22e6f9` | 106 cases completed; only two link checks failed because the isolated checkout lacked real parent-doc and prior-receipt artifacts. Restored those files; all 66 link checks passed. |
| `20260920-163821-246b1d` | Explicitly interrupted at 455/4,629 cases to fix late codebook binding. All workers exited before source edits. Not a landing receipt. |
| `20260920-164036-ad7cf0` | New failing probe: an unbound writer treated allocator IDs as codebook rows. Missing terms now wait for real owner binding, preserving recorded leaves. |
| `20260920-164202-415076` → `20260920-164303-f05ff6` | The 101-case run exposed a small controller fixture without a generate owner; code-row binding is now independent of unfolding. The final 68-case controller/index/checkpoint recheck passed, including nested stream isolation and write-target enforcement. |
| `20260920-164354-f26608` | Explicitly interrupted at 351/4,632 cases to strengthen the chain probe from repeated to distinct codes. All workers exited before edits. Not a landing receipt. |

The final distinct-code measurement, `20260920-164554-616d87`, passed all
22 selected measurement/index cases. At 0, 1 and 8 updates, a single code
recovered exactly; chains of lengths 2, 3 and 5 recovered at rate zero.
These are reported null results, not successful learned generativity.

The first complete 1c default run, `20260920-164725-78e442`, completed all
4,632 cases with eleven failures. Old taxonomy fixtures used order-zero
points, two standalone fact fixtures supplied no index cues, and interface
checks still expected an empty write scope or an obligatory `what` child.
Those checks now cover the corresponding subsystem effects without deleting
cases. Public vector questions assert a meronymic result and no taxonomy
identity/proof; native taxonomy fixtures use higher-order symbols. Fact
support retains bounded occurrence/origin/text/trust provenance without
retaining the stored meanings as frames. The existing conflicting-source
check verifies both source texts and occurrence identities.

The first correction run, `20260920-170647-e26621`, completed 72 cases
and found one additional candidate-contract defect: a trained chooser could
select open `part` with an unnamed vector. Closed `part` accepts full-width
vectors, but grammar-open taxonomy enumeration now requires a native
reference consistently in formation, candidate construction and dispatch.
The new regression check asserts that no such invalid open candidate is offered.

The expanded correction recheck, `20260920-170916-4bf690`, passed all
140 selected cases, including the public model entry, taxonomy provenance,
checkpoint restore, index and normal controller contracts.

The final **source-matched full default receipt** is `20260920-171033-62d16c`:
exit 0, all 4,633 selected cases completed (4,296 passed, 336 skipped, one
expected failure), 1,041.8 s and 14.11 GiB peak aggregate memory.
Command, from the isolated BasicModel checkout:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

The receipt was produced at `/private/tmp/wikioracle-item1c` and copied to
`output/tests/20260920-171033-62d16c` beside the main checkout. Its entire
626-file validated source map equals the staged landing source, including
unchanged files. SHA-256 over sorted compact JSON:
`553d40622f177fa9b2fe2b68af30e03c947d60f5d47f47a392eb4bdadf504f8f`.
The full run repeats the distinct-code recovery measurement and reports the
same zero compound rates. Five optional `sentence.pt` embedding probes skip
in this isolated checkout because that locally built artifact is absent;
the main-checkout supplemental receipt below exercises the existing artifact.
This completion note and todo reconciliation were added after the full run;
they do not change the validated source map.


The supplemental main-checkout receipt, `20260920-172952-394937`, completed
all 17 `test_testpoint.py` cases with exit 0 (five passed, twelve skipped),
including all five embedding probes using the existing `sentence.pt`. Its validated source map is identical
to the full receipt above.


## Item 1d review corrections (September 21)

The baseline includes documentation commits `7d4c290` and `e13c4ea`. The
dictionary correction was made and measured first, as requested. All 11
configurations changed by `7c2fa5a` restore
`conceptualContextLearningRate=0.01`: the shared concept dictionary is a
non-grad buffer, with the existing sentence-local unit-sphere rotation updater
as its sole owner. Objectives train operators and code-to-operation
projections; dictionary positions and `codebook.*` entries are absent from
the gradient diagnostic. Situation context and its three proposed XML
variables remain under two-truths §3.5.

The prepared-answer boundary applies when `answerSynthesis=true`; the direct
head still trains the whole input state. Combining reconstruction and
supervised outputs without answer synthesis now warns at configuration, or
first training use if the data arrives later. The supplied-answer tied
benchmark now samples operator gradients every batch. Diagnostic exceptions
warn without aborting backward or the optimizer step; checkpointed opposition
streaks survive unused samples and reset on measured nonnegative agreement.

`QuerySignature` inherits checked `Subsystem` scopes from its canonical
thought descriptor. The allocator caches row-to-concept identities, restoring
the cache from its existing checkpoint owner. The leaf-code column grows
geometrically and stores its used prefix in the checkpoint tensor; reset,
compaction, remap and subsequent appends retain the existing index contract.
The current docs and later plan sections now describe ordinary gradients and
the state boundaries, rather than the deleted projection. `lxml` was added to
requirements only; no dependency was installed. The 11 XML files validate
under `xmllint`; pytest's optional `lxml` cases still skip in this environment.

### Fixed-seed reconstruction baseline

[Probe](benchmarks/2026-09-21-item1d/probe.py), seed 42, CPU, one Torch thread,
Torch 2.14.0, native eager tensor loop, `data/MM_ladder.xml`, batch size 2.
Each run uses four validation batches, seven ordinary training updates, then
the same four validation batches. The training mean and rate exclude two
warmup batches. Both runs begin from the same seeded initialization and data;
the dictionary update setting is the intended configuration difference.
Checkpoint writes and compilation are disabled. Raw results include complete
validated source maps and config hashes:
[optimizer-owned baseline](benchmarks/2026-09-21-item1d/before.json),
[rotation restored](benchmarks/2026-09-21-item1d/rotation.json).

| Measurement | Optimizer-owned dictionary | Rotation-owned dictionary |
|---|---:|---:|
| Initial validation reconstruction | 0.100593256 | 0.100593256 |
| Training reconstruction, last five batches | 0.092587703 | 0.092793070 |
| Validation reconstruction after seven updates | 0.092143942 | 0.092326729 |
| Training sentences/s, after warmup | 0.56807 | 0.56798 |

The post-update validation shift is **+0.000182787 (+0.198%)**. This records
the changed baseline; it is not evidence of improved learning.

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools PYTHONPATH=bin:test .venv/bin/python doc/benchmarks/2026-09-21-item1d/probe.py --out /tmp/item1d-reconstruction.json
```

### Thought-enabled throughput

The [fixed thought config](benchmarks/2026-09-21-item1d/thought.xml) declares
the complete grammar and enables the ordinary LTM owner. For each parsed
sentence, the probe executes one declared `quantize` request on a retained
leaf through the normal controller. This deliberately exercises thought
effects and retention; it is not learned question selection or a wording
gate. It measures native parsing, reading, appends, the thought call and the
epoch's post-tick compaction together. Two warmup batches precede ten measured
batches, comprising 20 sentences and 20 executed thought requests.

| Measurement | Before cache/capacity changes | After |
|---|---:|---:|
| Sentences/s with thought | 14.65773 | 14.64366 |
| Reconstruction | 0.102225167 | 0.102225167 |
| Executed thought requests | 20 | 20 |

The rate changes by −0.096%, a **null throughput result** on this small
workload. The amortized allocation and absence of a per-effect dictionary
scan are mechanism checks, not a demonstrated end-to-end speedup. Raw results:
[before](benchmarks/2026-09-21-item1d/thought-before.json),
[after](benchmarks/2026-09-21-item1d/thought-after.json).

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools PYTHONPATH=bin:test .venv/bin/python doc/benchmarks/2026-09-21-item1d/probe.py --config doc/benchmarks/2026-09-21-item1d/thought.xml --thought --eval-only --out /tmp/item1d-thought.json
```

### Regression evidence and dispositions

| Receipt | Result |
|---|---|
| `20260921-015653-a5d584` | The inverted real-batch dictionary-ownership probe failed on the optimizer-owned parameter before the correction. |
| `20260921-020340-0472ba` | All 18 new cases completed: 11 configuration checks passed after (b); seven probes exposed missing warning, diagnostic failure handling, opposition persistence, checked signature scopes, cached identities and geometric leaf growth. |
| `20260921-021212-11780c` | All 67 focused cases passed after the remaining corrections. |
| `20260921-021549-5b4875` | All 112 affected cases passed, including the actual normal-batch operator diagnostic, prepared-answer state boundaries and checkpoint/index lifecycle. |

That native training batch measured `operator.CS.surface` reconstruction norm
0.000749390, output norm 1.829513384 and R/O cosine 0.087219789. Expectation
was unused in that batch and has a null cosine. This is an actual shared
operator measurement, not evidence about persistent agreement or useful
learning.

The mistaken selector run `20260921-021439-d30882` exited at collection
because `test_query_contracts.py` does not exist. The corrected affected run
above selects `test_query_contract_boundaries.py` and `test_query_registry.py`;
the collection error is not a validation receipt.

No existing tests were deleted. The real-batch optimizer-ownership assertion
is inverted: the dictionary must be a non-grad buffer outside all optimizer
groups, while shared operators still receive actual reconstruction/output
credit. Generic sparse cosine tests retain their numerical coverage under
operator names. Added checks exercise all 11 configs, direct-head warnings,
nonfatal diagnostics, integrated opposition checkpoint restore, checked
signature scopes, geometric growth and cached thought effects. The existing
structural checkpoint test also verifies the reconstructed reverse cache.

The first full run, `20260921-022112-2e9f67`, was explicitly interrupted at
3,228/4,651 completed cases with no observed test failure. Review found that
PyTorch serializes a tensor view's entire backing allocation: saving the used
leaf-column slice still retained spare capacity. All workers exited before
edits. The strengthened serialization probe `20260921-023349-88e051` failed
as intended (4,096 stored bytes for 3,072 logical bytes). Checkpoint export now
clones the used prefix, and the test exercises an actual save/load round trip
before appending. The interrupted run is not a landing receipt. This
checkpoint-only correction does not change the timed workload, whose
checkpoint writes are disabled; its original source maps remain in the raw
measurements above.
The corrected checkpoint/index/retention receipt
`20260921-023423-81fc5f` passed all 31 cases before the final full run.

### Full source-matched landing receipt

The final default receipt is **`20260921-023448-ddf5f0`**: exit 0, all
**4,651/4,651** selected cases completed (**4,319 passed, 331 skipped, one
expected failure**), 1,045.1 seconds and 13.06 GiB peak aggregate memory.
Command from `basicmodel/`:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py --batch-size 8 --max-files 1
```

The entire **627-file** validated source map, including unchanged files,
matches the landing source. SHA-256 over sorted compact JSON:
`8ae5ceae6380886964b2529aee8347b7bfbd166d8755a6a7da2c18c5860352c7`.
The [complete result](benchmarks/2026-09-21-item1d/full-result.json.gz)
is archived as gzip-compressed JSON, alongside its
[source manifest](benchmarks/2026-09-21-item1d/full-source-manifest.json)
and the full run's
[measured operator gradients](benchmarks/2026-09-21-item1d/operator-gradients.json).
The archive retains every selected/completed case, worker outcome, resource
limit and captured report. Receipt prose and the todo reconciliation were
added after validation; they do not change the validated source map.
Optional slow learning studies are not part of this default receipt, and
the unavailable `lxml` schema cases remain skips. Nothing here closes the
remaining learning gates or starts item 2.

After adding the archived receipt and completion notes, documentation-link
receipt `20260921-025341-86b75b` passed all 66 cases with the same validated
source map.

## Negative-image expectation (September 21)

Item 2 implements pure composition, the per-role signed image at the seal,
all-role surprise, observation/estimate retention, grammatical object masks,
and residual credit on the existing controller. The design, configuration and
checkpoint migrations are in [ExpectationRetention](ExpectationRetention.md).
The situation context and its three configuration variables are untouched.

The initial failing probes exposed absent negative-image helpers, erased
residuals on empty roles and missing durable surprise. Native exploration then
exposed two integration defects: `what` could bind a declarative description,
and a subsequent declarative input attempted to append a legacy parity slot to
ordinary thought history. The catalogue now uses known question occurrences
for `what`; sealed observations stay with their existing observation owner.
The final generation smoke also caught raw text crossing the tensor input API.
Packed native exploration exposed the new object-mask lookup's one-record
limit: a later serial occurrence could not be read. It now scans the bounded
history on the shared meter and handles exhaustion through the controller's
normal cutoff. A separate delayed-contrastive probe found that replay had
covered MSE but not the optional contrastive objective; both now replay current
predictor parameters while retaining the original estimate as evidence.

Affected validation: **235 passed, 14 skipped** across the prediction,
retention, controller, output, migration, reset and retired-prior files. After
strengthening the absence probe and cleaning obsolete test descriptions,
**92 passed, 1 deselected** in the relevant focused selection. The native
unlabelled run includes three real `runBatch` calls, tied reconstruction,
backward and optimizer updates to the same chooser. No answer is supplied.
The generation correction passed **29 cases, 2 skipped, 1 deselected**.
After the final history and contrastive corrections, the focused prediction,
controller and work-boundary selection passed **71 cases, 1 skipped,
3 deselected**. The first full run (`20260921-051525-796d18`) was explicitly
stopped before editing source for these corrections; it is not a landing
receipt.
The final [history red probe](benchmarks/2026-09-21-item2/history-red.txt),
[contrastive red probe](benchmarks/2026-09-21-item2/contrastive-red.txt) and
[affected green result](benchmarks/2026-09-21-item2/affected-final.txt) are archived.
The next full run (`20260921-053243-5154bb`) completed all 4,665 cases and
found two [controller regressions](benchmarks/2026-09-21-item2/regression-failures.json).
Reads could consume the last work unit before history recorded it; exhaustion
now records the already-paid work and enters the ordinary zero-cost drain.
The explicit routing-calibration fixture now initializes only its selected
feature subspace, so ignored columns cannot change its initialization. Its
800-update limit, actual learned descent, depth and credit assertions are
unchanged. Cutoff coverage now runs both with and without a prior comparison.
The corrected [controller selection](benchmarks/2026-09-21-item2/controller-green.txt)
passed **80 cases, 4 deselected** before the final full run.

The [measurement report](benchmarks/2026-09-21-item2/README.md) separates learning
from these mechanisms. Three-seed frozen synthetic and parsed-English studies
beat shuffled/context-free prediction controls. The existing wording recipe
again parses all 56 held-out examples and generates/recomposes the expected
meanings. Related unexpected text leaves a smaller conceived remainder in
120/120 comparisons. Residual policy updates run without labels, with their own
baseline and measured return variance.

Six native runs cover unpacked and packed input, each with prediction plus
queries, prediction alone, and reconstruction alone. The packed runs each
complete 78 observations and 74 pairs in seven optimizer steps. None beats its
context-free prediction control; none demonstrates joint quality benefit.
Native reconstruction uses a 16-row basis and still reports truncated rows,
so it does not establish full-basis fidelity or discrimination. Commands,
counts, losses, warm-up, throughput, source fingerprints and limitations are
in the measurement report.

**Reasoning utility is null:** at gains 0 and 1, held-out questions take the
same two thought steps and 17 work units, and have identical unsuccessful
positive-assertion answers (Brier score 1.0). This does not pass the useful-query
or reasoning-work learning gate. The remaining joint/causal learning evidence
stays under item 2 in `todo.md`; it is not transferred to item 4.

Retired assertions and their replacements are dispositioned in
[ExpectationRetention](ExpectationRetention.md#migration-and-regression-dispositions).

Final source-matched default selection, `20260921-055251-c20fa5`:
**4,338 passed, 330 skipped, 1 expected failure, 1 compiler-cache failure**;
all **4,670 selected cases completed** in 371.338 s. The sole failure was
`test_normal_batch_trains_supplied_grammar_lessons`: clang rejected a generated
precompiled header whose recorded PyTorch header timestamp predates the
installed header. It was not an assertion failure. The old generated-header
directory was preserved as a backup so compilation can rebuild it. No package
or repository source was changed to address the environment error.
The affected case then [passed in 185.99 s](benchmarks/2026-09-21-item2/compiler-cache-green.txt)
against the same source after rebuilding the generated headers.

The user explicitly waived another full rerun: **“No need for the full rerun.”**
This is a completed full receipt with a documented exception, not a claim of a
green full run. The [complete result](benchmarks/2026-09-21-item2/full-result.json.gz),
[source manifest](benchmarks/2026-09-21-item2/full-source-manifest.json),
[receipt summary](benchmarks/2026-09-21-item2/receipt-info.json) and
[compiler failure](benchmarks/2026-09-21-item2/compiler-cache-failure.json)
are archived. The full **628-file** validated source map matches the landing
source; SHA-256 of sorted compact JSON:
`8fb4b1752effbf1bdcc57783fde48a95eea8c0927251ec929fc60806e73b879a`.
The runner used its default 256-case/16-file batches, 10 one-thread workers,
8 GiB per worker and 28 GiB aggregate cap; peak aggregate memory was 18.83 GiB.
Receipt prose, measurements and todo reconciliation do not change that source
map. Optional slow learning studies are separate; unavailable `lxml` checks
remain skips.

After adding the final receipt and measurement notes, documentation-link run
`20260921-060447-d4ce9c` passed **67/67 cases** with the same validated source.


## Item 12 review corrections (September 21)

Countdown item 12 corrects the expectation landing's retention score and pair
lookup. The retained scalar normalizes over the soft union of observed and
expected roles, so equally surprising idea and relation rows score equally;
the predictor still trains against the full all-role residual. Native purity
checks now include nonuniform priority and live reading scope. Indexed pair
reads are checked after restore and compaction. The harness performs one
visible fresh-worker retry for a typed stale precompiled-header failure, with
PCH reuse disabled and compilation still active; genuine test failures remain
failures. Operator reports now include both norm ratios beside cosine.
[Design and development evidence](benchmarks/2026-09-21-item12/README.md).

The affected run `20260921-063236-32ba94` completes **131/131** cases:
**128 passed, 3 skipped**, exit 0. Full run `20260921-063627-f5a587` completes
**4,683/4,683** cases: **4,352 passed, 330 skipped, 1 expected failure**,
exit **0**, in **1,059.92 seconds**, with **no waived failure and no cache retry**.
The previously failing grammar-lesson compiler case passes. These are unique
case counts; repeated passing phase reports are preserved separately in the
[receipt metadata](benchmarks/2026-09-21-item12/receipt-info.json).

The [full receipt](benchmarks/2026-09-21-item12/full-result.json.gz) and
[source manifest](benchmarks/2026-09-21-item12/full-source-manifest.json) match
**629 source files**, SHA-256 of the sorted compact validated-source map:
`ec2a3c999c5e3c5c26de2ca2034fbf2195cc55c82f5e47628fd1eb652f06f034`.
The run used `--batch-size 8 --max-files 1`, ten workers, the unchanged 8 GiB
per-worker / 28 GiB aggregate caps, and a 14.34 GiB peak aggregate footprint.

The [native gradient report](benchmarks/2026-09-21-item12/operator-gradients.json)
measures output/reconstruction norm ratio **2,441.3373** for
`operator.CS.surface`, with cosine **0.08721979**. This is one diagnostic batch,
not a learning gate. The earlier null useful-query result remains open under
countdown item 9. Item 11 is next; the remaining countdown numbers are unchanged.

## Item 11 generation ownership (September 21)

The September 17 generation candidate is rebased onto the shared-operator
contract. Both output modes retain a declared generation catalogue, ordinary
output inverse dispatch uses its model's scoped catalogue, and numerical hosts
keep one optimizer owner and their existing checkpoint names. Chooser weights
and Adam moments retain rule-meaning migration. The concluded idea remains a
detached operand for output; supplied output trains the active conditioner and
any shared maps actually executed. No independent operator copy, objective or
global gradient budget is installed.

[Design, preserved-probe dispositions and development evidence](benchmarks/2026-09-21-item11/README.md)
record the rebase and the non-loop fixture's absent shared-map credit. This
landing validates mechanisms and normal supervised updates; useful learned
language, expectation and reasoning remain subject to countdown items 9 and 8.

The broad affected run passes **212 cases, 18 skipped**; explicit slow
supervised integration passes **4/4** on MPS. After correcting the compatibility
and adapter fixtures, the final affected run `20260921-085306-14909c` completes
**76/76** cases: **60 passed, 16 skipped**, exit 0. The runtime is unchanged
between these successful runs; their individual source manifests are retained.

Full run `20260921-085748-b59239` completes **4,695/4,695** cases:
**4,364 passed, 330 skipped, 1 existing expected failure**, exit **0**, in
**1,059.34 seconds**, with **no waived failure and no cache retry**.
The [full receipt](benchmarks/2026-09-21-item11/full-result.json.gz) and
[source manifest](benchmarks/2026-09-21-item11/full-source-manifest.json) match
all **630 source files**, SHA-256 of the sorted compact validated-source map:
`215308c7bed6922c72e6599319f6fefd62e124bb0032b75d10e183d77f3c1cd0`.
The run uses `--batch-size 8 --max-files 1`, ten workers, the unchanged 8 GiB
per-worker / 28 GiB aggregate caps and a **14.73 GiB** peak aggregate footprint.
Unique case counts and repeated phase reports are separated in the
[metadata](benchmarks/2026-09-21-item11/receipt-info.json).

The receipt preserves the failed development attempts and their corrections.
An existing untrained MentalModel overflow reproduced at seed 3 with identical
inputs and scores on that landing's baseline and runtime. Item 11 pinned the
compatibility fixture to a successful seed and separately asserted rejection
of the failing initialization. **Item 10 removes that workaround:** consumed
operands were reused by recursive soft compaction. See the correction and
seed-selection audit below; the historical receipt remains unchanged.

Final documentation-link run `20260921-091700-f39470` passes **69/69** cases,
exit 0, with the same validated source; its receipt is recorded in the metadata.

## Item 10: reconstruction baseline and seed audit (September 21)

The [measurement and audit record](benchmarks/2026-09-21-item10/README.md)
contains the fixed-seed native baseline, packed/single reconstruction
comparison, MentalModel overflow diagnosis and test dispositions. A seed may
make a measurement reproducible; it cannot be selected to make a capability
assertion pass. Known failing initializations remain useful regression cases
beside unseeded compatibility assertions. Convergence failures remain failures,
with no replacement passing seed, relaxed threshold or new expected-failure
marker.

The corrected compaction matches exhaustive small tilings and enumerated
derivatives, including a fullgraph capture. The known failing MentalModel
seed 3 now forwards successfully beside an unseeded smoke test. The declared
32-seed measurement passes 32/32. The native item-1d reconstruction baseline
is exactly unchanged; the tied packed/single comparison is a null for parity,
with mean costs **.7866926491 / .6838697642**. The plan's §8.2 records both.

The bounded Pi/Sigma pair's hidden tanh is cancelled by the next atanh;
its old seed-selected XOR assertion is replaced by an interior algebra check.
The adjacent exponential-feature XOR learning gate retains its original bar.
The explicit slow audit exposes a grammar-XOR miss, zero free-derivation
recovery and two unsupported W=6 configurations. Those failures remain open
in the countdown. A default receipt does not close these learning gates.

The final source-matched full selection `20260921-102325-6e2fd3` completes
**4,706/4,706** cases: **4,375 passed, 330 skipped, 1 existing expected failure**,
exit **0**, in **1,064.73 seconds**, with **no waived failure and no cache retry**.
The [full receipt](benchmarks/2026-09-21-item10/full-result.json.gz),
[source manifest](benchmarks/2026-09-21-item10/full-source-manifest.json) and
[summary](benchmarks/2026-09-21-item10/full-summary.json) match all **630 source
files**, SHA-256 of the sorted compact validated-source map:
`0e47c3133f953bd5920e1a02f977509cefda0f6394ff3cf32b4bd02980a5e0d2`.
The three final reconstruction measurements match the same source map.
The run uses `--batch-size 8 --max-files 1`, ten workers, unchanged 8/28 GiB
caps and a **10.78 GiB** peak aggregate footprint. The interrupted development
selection and the unchanged binder assertion's correction remain in the
linked audit record; the corrected affected selection passes **48/48**.

Final documentation links pass **70/70**, `20260921-104438-87cac7`, with the
same validated source ([receipt](benchmarks/2026-09-21-item10/doc-links-final-result.json.gz),
[landing metadata](benchmarks/2026-09-21-item10/receipt-info.json)).

## Item 11: concept parts and provisional rows (September 22)

The [implementation and measurement record](benchmarks/2026-09-22-item11/README.md)
links the settled design and records the live union default, optional
conjunction/union passes, signed exponents, provisional row pool, context
weights, detached use gate, transpose reverse and checkpoint/optimizer
ownership. The host promotion dictionaries and additive conceptual hop are
removed. Object testimony and sequence removal remain item 7.

The layer retains presence .9999998808 through four orders with one of eight
parts present. Thirty-two weak parts at .001 give .0315085351, below the
stated .031 excess bound. Balanced transpose checks recover conjunction parts
at .9 from .81 and union alternatives at .6 from .84, within 1e-6. Context
matches assign a provisional pair immediately; use promotes it in place,
non-use permits recycling, and the optional co-present whole requires all
its parts. Both matrices and their state survive a fresh model checkpoint;
sparse growth/pruning preserves optimizer moments and pending credit by edge
allocation identity.

The [affected selection](benchmarks/2026-09-22-item11/affected-result.json.gz)
completes **140/140** cases, **133 passed and 7 skipped**, exit 0. The explicit
[unseeded XOR selection](benchmarks/2026-09-22-item11/xor-result.json.gz)
completes **3/3**, exit 1: the four-conjunction signed representation passes,
but four and eight zero-initialized conjunction concepts both finish 900 Adam
updates at **MSE .5**, against the unchanged **< .1** bar. There is no seed
selection, retry of an initialization, relaxed threshold or expected-failure
marker. Item 11 retains this learning residue.

The repeated `d4dc385` native reconstruction means are unchanged with the
switch off and on. The native workload is serial and does not exercise the
sparse pyramid; trained sparse-pyramid reconstruction remains unmeasured.
The packed/single payloads remain unchanged, including the .7866926491 /
.6838697642 mean tied byte costs. Diagnosing that gap remains item 9's first
landing.

The development record retains the initial mechanism failures, checkpoint
regression, interrupted selections and temporary-config source-hashing race.
Scratch XML fixtures now live outside the validated source tree, with a
regression checking the snapshot while a scratch file exists. The runner's
source guard, finite limits and coverage requirements are unchanged.

The source-matched full selection `item11-publish-full` completes
**4,717/4,717** cases: **4,384 passed, 332 skipped and one existing expected
failure**, exit **0**, in **1072.14 seconds**. There is no waived
failure and no compiler-cache retry. The
[full receipt](benchmarks/2026-09-22-item11/full-result.json.gz),
[source manifest](benchmarks/2026-09-22-item11/full-source-manifest.json),
[summary](benchmarks/2026-09-22-item11/full-summary.json) and
[landing metadata](benchmarks/2026-09-22-item11/receipt-info.json) match all
**631 source files**, SHA-256 of the sorted compact validated-source map:
`b289442fecf0af783383cebe5560e6d26d22f803ff7f078eca45034f8e478596`.
The affected and explicit XOR receipts and all six reconstruction records
match this source map. Peak aggregate footprint is **10.63 GiB** under
the 20 GiB cap, with 8 GiB per worker. The two new default slow skips are
explicitly measured XOR failures, not capability passes.

Final documentation links pass **71/71**, `item11-doc-links-final`, with the same
validated source ([receipt](benchmarks/2026-09-22-item11/doc-links-final-result.json.gz)).
