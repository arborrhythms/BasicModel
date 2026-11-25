# Item 1.5: PyTorch 2.15 MPS investigation

## Question and environment

Alec asked for version-specific research before attributing Metal's slower
sentence execution to host interrupts. That attribution remains unconfirmed.
The local environment is PyTorch **2.15.0**, source revision
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`, Python 3.12, macOS 27.2.1,
and an M4 Max Mac Studio with 36 GB unified memory. These experiments use
the physical MPS device, FP32, and the existing four independent lesson rows.

Separate three costs: Dynamo/Inductor compilation, backend graph/shader
construction and submission, and steady execution including synchronization.
The eager baseline uses `MODEL_COMPILE=none`; its throughput gap cannot be
attributed to Dynamo recompilation. It can still pay backend compilation costs.

## Sources read

- The [PyTorch 2.15 release article](https://pytorch.org/blog/pytorch-2-15-release-blog/)
  (October 28) describes further native Metal migrations and shape-specific
  gains for reductions, softmax, linear and grouped matrix multiplication.
  These results do not predict this model's small symbolic operations.
- The upstream contributors' [MPS in PyTorch 2.15](https://huggingface.co/blog/mps-pytorch-2-15)
  (October 20) distinguishes specialized kernels from removal of CPU offset
  reads. Its later section describes command-buffer submission work for 2.16;
  that section is not a list of features available in our installed release.
- The [2.14 release article](https://pytorch.org/blog/pytorch-2-14-release-blog/)
  explains that native index, reduction and other kernels replace MPSGraph
  paths and avoid their graph compilation/dispatch costs. Those changes are
  already included in 2.15, so older reports blaming every operation on
  MPSGraph should not be applied wholesale to this environment.
- [PyTorch PR 200181](https://github.com/pytorch/pytorch/pull/200181) identifies
  delayed submission when a chain contains only Metal kernels. It introduces
  periodic command-buffer commits, allowing CPU encoding and GPU execution
  to overlap. Its reported results distinguish this effect from scalar reads.
  Our installed `MPSStream.h` has no corresponding encoded-kernel counter;
  the [2.15 stream implementation](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/mps/MPSStream.mm)
  still commits through explicit, MPSGraph, or memory-pressure paths.
- The [2.15 compiler profiling guide](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
  recommends measuring compilation separately and examining GPU gaps for
  launch overhead. CUDA Graph advice does not establish an MPS remedy.
  The [compile API](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
  specifically describes `reduce-overhead` in terms of CUDA graphs.
- The [2.15 MPS environment reference](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html)
  documents profiling controls and `PYTORCH_MPS_PREFER_METAL` for matmul.
  Neither that setting nor fast math is a general fix for host dependencies.

## Local evidence so far

Receipts below share the prefix `output/item1-5/resume-20261125-`.
The before source is handoff `24a01c5597`; after includes fixed structural
seats in `ConceptSpread`/`Salience` and tensor-masked native forecast retention.
Each throughput measurement contains 12 complete updates, excludes three
warm-ups, synchronizes the device, and counts 36 actual sentences in the nine
timed updates. Candidate walks are not counted as additional sentences.

| MPS, explorations per row | Before sentences/s | After sentences/s |
| --- | ---: | ---: |
| 1 | 2.326 | 2.425 |
| 3 | 1.317 | 1.393 |

These are single paired runs, not confidence intervals. The explicit native
host-operation counts are unchanged: one exploration has 197 device-to-host
copies, 176 MPS scalar reads, 24 CPU scalar reads and 12 equality results in
the sentence. Nonzero calls decrease from 179 to 141; unique calls from 120
to 108. Generic `aten::index` tags include fixed-size integer indexing and
must not all be counted as synchronizations. The new separate diagnostic
table distinguishes boolean and integer indexing.

In `after-one-mps`, the dispatch-instrumented sentence took 2.151 seconds.
The measured copies, MPS scalar reads and equality calls consumed 0.122
seconds (5.7%). The timer includes queued device work waited for inside
these calls. It excludes implicit synchronization in other operators and
does not measure lost overlap caused by those waits. This is neither the
total synchronization cost nor an upper bound on the benefit of removing it.

The additional `native-wall-mps` diagnostic, after four rather than twelve
training updates, records 236,940 native dispatches in a 2.711-second sentence
entry, with 0.843 seconds inside the native calls. The remaining time includes
Python model work, dispatcher overhead and instrumentation. It must not be
reported as a direct measurement of uninstrumented Python overhead. Boolean
indexing, nonzero and unique together take about 0.103 seconds in this sample.

`metal-plain.trace` captures the real lesson without PyTorch signposts. The
union of Python's GPU Active intervals is 4.119 seconds between the first
and last GPU events, 24.414 seconds apart. During seconds 8–26, each one-second
bin contains approximately 0.13–0.26 seconds of GPU activity. This supports
investigating host-side supply of work. It includes construction, warm-up and
the runner's diagnostic updates, and Instruments slows the run; it is not
a steady-state utilization estimate. Instruments also warned that its chosen
GPU counter profile is unsupported on this device, so no hardware-counter
conclusion is available. GPU execution intervals were still exported.

CPU samples in trace seconds 8–27 include 0.311 seconds of sample weight in
MPSGraph's `compileWithDevice` stack and 0.369 in executable specialization
(overlapping categories). Compilation at the backend remains observable in
this eager workload. The samples also include Python, allocator and driver
work. `__kdebug_trace64` alone has 1.107 seconds of sample weight, showing
substantial tracing overhead. Sample weights are statistical, not measured
durations; this trace does not record sleeping threads, so it cannot quantify
all CPU wait time. The exported summary is `metal-plain-cpu-summary.json`.

The signposted companion is for operation attribution only. In 2.15,
enabling MPS signposts disables `commitAndContinue` in the stream constructor.
The installed `torch/mps/profiler.py` also warns that
`wait_until_completed=True` slows execution by waiting after every operation;
that option was not used. Always keep unprofiled timing separate.

## Next comparisons

Use the real sentence and keep FP32, seeds, row ownership, exploration,
learning and optimizer participation fixed. Record graph counts, compiler
seconds, cold-cache first calls, warm calls, and the exact source revision.
Removing host dependencies is still needed for whole-sentence capture, but
will be judged by matched runtime and learning receipts, not host-call counts
alone. Inspect CPU stacks and submission gaps to distinguish backend
compilation, repeated small operations, implicit waits and delayed submission.

## Later ExecuTorch MLX experiment

Alec suggested trying the MLX delegate at some point. The
[delegate overview](https://docs.pytorch.org/executorch/stable/backends/mlx/mlx-overview.html)
describes export through `torch.export`, persistent mutable buffers and MLX
execution. The [operator coverage page](https://docs.pytorch.org/executorch/stable/backends/mlx/mlx-op-support.html)
states that unsupported operators fall back to the portable CPU runtime.
Export and delegation coverage therefore need measuring before interpreting
speed. In particular, the current Python-consumed learned span endpoints
remain an export problem; changing backends does not remove them.

ExecuTorch is not categorically inference-only. Its experimental
[training extension](https://github.com/pytorch/executorch/blob/main/extension/training/README.md)
exports forward and backward graphs, exposes gradients as outputs, and runs
the optimizer separately. It says delegate integration is still in progress
and lists SGD and AdamW, with Adam planned. A later experiment should first
compare a captured forward region, then prove backward, exact optimizer
semantics (including absent gradients), state mutation and complete lesson
learning before claiming comparable training speed. The existing BasicModel
architecture and PyTorch environment have not been switched to this runtime.

Alec subsequently pointed out the existing export infrastructure and asked
to continue compilation first. On this branch it is in `bin/export_mlx.py`:
it already selects the MLX partitioner and excludes several gather/index
operators following earlier failures. Reuse and revalidate that path when
the experiment resumes; do not introduce a parallel exporter. Its historical
2.12/ExecuTorch 1.3.1 comments are not evidence about this 2.15 environment.
