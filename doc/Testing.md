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
  cases. Coverage records prove that each selected case completed once; tests
  are not repeated or discarded. The hard memory limit still applies within a
  test, and a killed worker is a failure.
- An ordinary pytest assertion, setup or teardown failure does **not** cancel
  the remaining workers. The receipt keeps its `test_failure` outcome and
  runs every selected case so a diagnostic full run reports all failures.
  Process-boundary failures (timeout, memory kill, unavailable accounting,
  collection/protocol failure, interrupted execution or invalid coverage)
  still stop the pool immediately to protect the machine and preserve an
  honest receipt.
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
