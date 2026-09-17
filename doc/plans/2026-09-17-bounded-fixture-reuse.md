# Future work: reuse bounded fixed-capacity test fixtures

Status: evaluated direction; implementation and comparative measurements pending.
Requested by Alec, September 17, while profiling the default suite.

## Assessment

Keep fixed-capacity allocation and extend reuse where measurements justify it.
Vocabulary staging already inserts into a reserved table without replacing its
Parameter or rebuilding the optimizer. Frozen codebook refreshes copy into the
existing owner and reject shape changes. Existing regressions check Parameter,
storage and Adam-state identity. This work dates to `d1f89c3f` (May 21); it has
not been removed.
[OOV staging](../../bin/Spaces.py#L5989),
[frozen refresh](../../bin/Spaces.py#L3054),
[optimizer checks](../../test/test_no_rebuild_optimizer.py#L47),
[storage checks](../../test/test_oov_reserve_no_resize.py#L39).

Fixed staging shapes also prevent graph specialization on varying unit counts
and transient STM/chunk sizes. That regression was added in `ab148467`
(September 12). Preserve it while measuring construction and compilation
separately.
[Shape contract](../../test/test_meronomy_ladder.py#L518).

The current test profiles show costly full-model cases and frequent worker
recycling. Their helpers repeatedly call `from_config`; preallocation inside
one instance does not reuse an instance created by a later helper call.
The attribution to fresh construction is supported by the helper code, but the
profiles do not yet separate allocation, initialization, reset, compilation and
forward/backward time. Do not claim an allocation-only speedup from those totals.
[Full-model helper](../../test/test_global_attention.py#L149),
[native variant helper](../../test/test_meronomy_ladder.py#L467),
[profiling evidence](../benchmarks/2026-09-17-bounded-test-data/additional-large-profile.json),
[partial default call profile](../benchmarks/2026-09-17-bounded-test-data/partial-call-profile.json).
The latter counts passing call phases by test and file from the incomplete,
timed-out run; it is not a full-suite completion record. Its call timings do not
separate setup, import, allocation or compiler costs; the full receipt includes
their contribution to elapsed time. Retain essential answer/ownership checks
while targeting their shared expensive construction and execution paths.

A one-second sample of a long reconstruction worker showed TorchDynamo callbacks
and tensor metadata handling while its physical footprint was about 1 GiB.
This supports measuring graph construction as well as allocation; it does not
attribute the entire test duration or establish a fixture-reuse speedup.
[Stack sample](../benchmarks/2026-09-17-bounded-test-data/reconstruction-stack-sample.txt).

## Bounded follow-up

1. Instrument representative repeated builds with counts and timings for
   constructor/allocation, reset, compiler graph creation, forward, backward and
   optimizer work. Separate cold import/compile from steady state. Include peak
   physical footprint and eligible completed cases per second.
2. Use one compact, preallocated model per compatible fixture family within a
   bounded worker when a test does not require fresh construction. Key reuse by
   configuration contents, grammar, backend, dtype/device and declared capacity.
   Size the fixture for that family's contract; avoid allocating the largest
   production tables for every small regression. Keep the central hard memory
   cap, deadlines and worker recycling. Evict between incompatible families;
   do not create an unbounded global model cache.
3. Define and verify a complete reset: weights, buffers, gradients, optimizer
   state, RNG, dictionaries/allocators and row activation, grammar ownership,
   dataset/config globals, STM/LTM, thought history, expectation stream, traces,
   hooks and retained autograd graphs. Restore values in place where stable
   storage is the contract. Construction, ownership, checkpoint-migration and
   reset-isolation tests still need fresh models where their assertions require
   it. A shared fixture must not make tests order-dependent.
4. Audit backend selection and graph guards. `TheCompileBackend` is initialized
   at import; changing the environment alone later does not update that module
   value. Fixtures should select and restore backend state explicitly. Record
   compilation counts across allowed length buckets and reuse the existing
   disk cache; changing genuinely static configuration still needs compilation.
   [Backend initialization](../../bin/util.py#L501).
5. Run substantial training measurements on MPS on this machine (or the
   available accelerator elsewhere), recording actual parameter devices and
   unsupported paths. Separate cold compilation from warmed training steps;
   preserve finite process/allocator memory and timeout limits. Do not infer
   a speedup merely from GPU selection.
6. Compare fresh construction with bounded reuse over the same test selection,
   seeds, ordering permutations and backend. Include tests that deliberately
   mutate state. Require identical outcomes and preserved storage/optimizer
   identity where promised, no cross-test gradient or memory contamination,
   bounded memory, and a reproducible total-time improvement. Report null or
   negative results. Move suitable coverage back into the default gate only
   after its runtime and isolation are demonstrated.

This is a development/test optimization proposal. It does not remove reasoning
methods or change the specified model semantics. The existing slow gate retains
expensive assertions while this follow-up remains unimplemented.

## MPS measurement at the checkpoint

The sequence-training check measured 1.24 seconds for construction and 877.81
seconds for two training batches on MPS with the current `eager` backend. This
case therefore does not support construction time as the dominant cost. Record
warmed step timings and backend/dispatch costs before choosing the optimization.
[Measured phases and parameter devices](../benchmarks/2026-09-17-bounded-test-data/mps-training-audit.json).
