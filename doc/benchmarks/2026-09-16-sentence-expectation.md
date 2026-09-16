# Sentence expectation: learning and full-training measurements

Status: measurements and full-suite verification complete, September 16, 2026.
This is item 4 of the
[integrated plan](../plans/2026-09-15-next-sentence-as-the-production-objective.md#10-consolidated-implementation-and-verification-order).
The runtime baseline is basicmodel `70eaa2d384552192fb021d2154852af66381c86a`,
plus the measured runtime fixes included with this report. Each raw record
contains runtime/configuration SHA-256 hashes; the base revision alone does
not identify the measured working tree.

The benchmark uses the actual [local-role expectation head](../../bin/Layers.py#L9189)
and [native training scheduler](../../bin/Models.py#L14603). The native workload
retains the current [detached reverse student](../../data/BasicModel.xml#L177);
tied sentence reconstruction is the next migration. The supplied-answer
fixture retains BasicModel's space widths and dictionary capacities, with the
explicit [workload](../../data/BasicModel_answers_benchmark.xml#L37) and
[loss-weight](../../data/BasicModel_answers_benchmark.xml#L51) changes below.

The executable is [bench_sentence_expectation.py](../../bin/bench_sentence_expectation.py).
The measurements below separate supplied-answer training, corpus continuation
and controlled predictor learning.

## Operating point and supplied-answer speed

[Hardware record](2026-09-16-expectation-data/hardware.json): Apple M4 Max,
36 GiB RAM, macOS 26.6.2, PyTorch 2.14.0.dev20260722. Native runs use MPS,
`backend=eager`, fullgraph word-loop capture, one PyTorch CPU thread and FP32 with
AMP off. This measures actual native training with the configured MPS backend;
it is not an Inductor code-generation measurement. Runs execute alone, with
no pytest or competing benchmark process.

The supplied-answer configuration has 90,583,950 parameters. Its inputs are
three-word addition questions with separately supplied numeric answers. Both
runs use seven real optimizer steps: two warmup steps, then five timed steps.
Time includes input staging, forward, losses, backward, gradient balancing,
optimizer, reset/compaction and measurement overhead. The common gradient
settings are `reconstructionPriority=true`, `outputGradientRatio=0.5` and
`reconstructionLossTolerance=1e-8`.

| Supplied-answer workload | Batch 1 | Batch 2 |
|---|---:|---:|
| Measured input sentences / supplied answers | 5 / 5 | 10 / 10 |
| Measured seconds | 34.573 | 34.932 |
| **Supervised sentences per second** | **0.1446** | **0.2863** |
| Median full step, seconds | 6.947 | 6.988 |
| Two-step warmup, seconds | 28.847 | 30.017 |
| Compiler `_compile.compile_inner`, seconds | 14.665 | 14.651 |
| Unique captured graphs / observed graph breaks | 1 / 0 | 1 / 0 |
| Measured reconstruction objective | 4.891713 | 4.829926 |
| Measured supplied-answer objective | 0.040068 | 0.045279 |
| Answer-parameter change, L2 norm | 0.152237 | 0.154292 |
| Sampled peak MPS tensor allocation, GiB | 6.32 | 6.39 |
| Sampled peak MPS driver allocation, GiB | 7.26 | 7.27 |
| Process peak RSS, GiB | 12.67 | 12.67 |

Raw records: [batch 1](2026-09-16-expectation-data/answers-b1.json),
[batch 2](2026-09-16-expectation-data/answers-b2.json). Each input has its own
document identity, so every observation is a cold start and there are zero
eligible continuation targets. The expectation head correctly does not update;
the answer parameters update in both runs. Full training observation counts
are 7 and 14, matching the input counts exactly. The steady cold-start fraction
is 1.0 for each run.

Validation takes no optimizer steps or training-loss accumulation. Batch 1
evaluates eight questions before/after; batch 2 exhausts the twelve-question
validation partition in six calls. Their respective supplied-answer losses
change from 0.134204 to 0.132228 and from 0.129302 to 0.128396. These are short
numeric-workload measurements, not evidence of learned natural-language answers
or multistep reasoning. The batches see different numbers of examples, so this
is a throughput operating-point comparison, not a matched-quality experiment.

## Controlled learned context benefit

The [raw three-seed experiment](2026-09-16-expectation-data/synthetic.json)
passes the preregistered 20% improvement gate in all three seeds. All three
main conditions train the actual 7,371-parameter `SentenceExpectation` for
300 Adam updates of 128 examples. There are 1,472 training pairs and 368
held-out pairs per seed, with disjoint documents. NP1 is fixed while VP and NP2
vary according to the declared transition process.

| Seed | Ordered held-out MSE | Shuffled held-out MSE | Context-free held-out MSE | Root-broadcast held-out MSE |
|---|---:|---:|---:|---:|
| 0 | 0.001242 | 0.053293 | 0.052959 | 0.078369 |
| 1 | 0.001127 | 0.047074 | 0.046630 | 0.072400 |
| 2 | 0.001186 | 0.049837 | 0.048985 | 0.074815 |

Ordered MSE is about 97.6% below both matched controls in every seed. The
360-parameter historical root predictor is an additional scope baseline,
not a capacity-matched control. The raw records retain per-role errors,
target/estimate variance, occupancy loss and checkpoint context ablations.
This uses fixed synthetic meanings and establishes predictor learning from
context. It does not train the sentence encoder or prove corpus representation
quality, freedom from collapse, or useful questioning.

## Native FineWeb training and held-out result

The [raw W256/B1 run](2026-09-16-expectation-data/fineweb-w256-b1.json)
uses 90,336,255 parameters and the canonical `BasicModel.xml` widths,
dictionaries and detached reverse student. The local shard fingerprint and
document-disjoint source manifest are in the record. The 24-document cap
admits 995 sentences: 826 training, 66 validation and 103 test; no splitter
record exceeds the 256-word admission limit. This short measurement takes
seven training calls and eight single-sentence validation calls before/after.

| Training measurement | Value |
|---|---:|
| All training inputs / observations / eligible targets | 85 / 85 / 83 |
| Measured inputs / eligible targets after warmup | 63 / 62 |
| Measured full-step seconds | 242.355 |
| **Input sentences per second** | **0.25995** |
| **Predicted targets per second** | **0.25582** |
| Two-step warmup, seconds | 107.469 |
| Median measured step, seconds | 48.957 |
| Compiler `_compile.compile_inner`, seconds | 14.383 |
| Unique captured graphs / observed graph breaks | 1 / 0 |
| Measured cold-start fraction | 1 / 63 = 0.01587 |
| All-training cold starts / document starts | 2 / 2 |
| Measured reconstruction objective | 4.692519 |
| All-training feature MSE / presence BCE | 0.024791 / 0.205145 |
| Expectation-head parameter change, L2 norm | 5.376659 |
| Sampled peak MPS tensor / driver allocation, GiB | 7.11 / 8.07 |
| Process peak RSS, GiB | 12.78 |

The seven packed word-unit counts are 228, 251, 255, 247, 255, 247 and 255.
Every call takes one optimizer step; scored concept losses retain gradients.
Targets are detached, and the transient context is detached at step completion.
There are no supplied answer rows. This rate is reconstruction plus corpus
expectation training, not supplied-answer throughput.

| Same held-out eight-input slice, seven eligible targets | Before | After |
|---|---:|---:|
| Ordered feature MSE | 0.008798 | 0.023064 |
| Shuffled-context MSE at the same checkpoint | 0.008826 | 0.023124 |
| Context-free MSE at the same checkpoint | 0.008646 | 0.008962 |
| Presence BCE | 0.690466 | 0.060687 |
| Evaluation reconstruction diagnostic | 0.106251 | 0.108602 |
| Observed root-feature variance | 0.006763 | 0.002199 |

**This short corpus run does not establish a learned context advantage.**
Presence prediction improves, while vector error worsens and context-free
prediction is better. Rescoring the trained head against the fixed pretraining
encodings gives MSE 0.009572, also worse than its initial 0.008798. Observed
feature variance declines; this does not establish preserved discrimination
or rule out collapse. This held-out slice contains one occupied root role per
input, so it supplies no VP/NP2 fidelity evidence. The three-role synthetic
experiment above tests that separate capability.

Training reconstruction is the detached student's rule/arity/leaf objective;
evaluation uses the current D3 reconstruction diagnostic. These scalars are
different objectives and cannot be directly compared. See
[`runBatch`'s reconstruction branch](../../bin/Models.py#L13464) and
[`_detached_reverse_construction_loss`](../../bin/Models.py#L22730).
The tied reconstruction migration must repeat the relevant measurements and
add its sentence fidelity/gradient evidence. Query utility and complete nested
meaning remain later gates in the integrated plan.

## Memory amendment declared before the capped run

The first canonical W256/B1 packed run completed eight held-out evaluations
and one training update, then failed during the next backward pass at the
default MPS allocation limit (16.85 GiB). It cannot supply the required five
steady training steps. The allocator defaults remain
[high 0.60 / low 0.50](../../bin/mps_memory.py#L6).

The next FineWeb operating point uses
[BasicModel_expectation_benchmark.xml](../../data/BasicModel_expectation_benchmark.xml):
W64 storage and packing, with whole splitter records longer than 64 words
excluded by the loader. It retains the native space widths, dictionaries,
expectation weight and reconstruction path. This is a separate memory-bounded
operating point; report its admitted examples and lengths with every rate.
The synthetic learning gate and its controls are unchanged. The supplied-answer
fixture continues to use W256 with unpacked, short arithmetic inputs.

The W64/B1 retry also exhausted memory, after two training steps. Its failure
trace reached answer materialization's replay of a recorded compose program.
The reviewer probe then showed that a recorded `sum` executed all 16 binary
operators. [`forward_binary_step`](../../bin/Language.py#L14430) now executes
only operators selected by live rows, using conditional branches in compiled
calls. The new tests preserve values and gradients for mixed rows and shared
operand storage. This removes unnecessary recorded-program evaluation; §6's
separate reverse-traversal migration remains open. The canonical W256 retry
then completed all seven training steps. Final measurements use the repaired
replay path.

The first supplied-answer run exposed a second defect: the compiled unpacked
path took optimizer steps but recorded no external observations. Its eager
evaluation counterpart recorded the inputs. The graph's attribute-only
boundary handoff was not sufficient; the repaired
[`_publish_compiled_sentence_state`](../../bin/Models.py#L7291) uses the existing
explicit sealed slots/depth, and the
[`pending boundary drain`](../../bin/Models.py#L12432) retains occupied roles and
skips masked rows. The 21-value graph return is unchanged. The
[regression](../../test/test_compiled_expectation_boundary.py#L6) runs real compiled
and eager forwards, verifies once-only observations and predictor gradients,
and deliberately clears the attribute-only handoff. Sentence throughput now
counts presented input masks independently of predictor observation eligibility.
The supplied-answer measurements reported above follow this correction.

Retained failure records:
[W256 memory failure](2026-09-16-expectation-data/fineweb-w256-before-replay-fix-oom.json),
[W64 memory failure](2026-09-16-expectation-data/fineweb-w64-before-replay-fix-oom.json),
and [unpacked observation defect](2026-09-16-expectation-data/answers-b1-before-boundary-fix.json).
The last record's zero input count and invalid document-start fraction are
defect evidence, not usable input-throughput measurements. Neither memory
failure completed the five required steady training steps.

## Reproduction

Run from `basicmodel/` with the local FineWeb-Edu shard already installed.
Execute each command alone. The native runs retain dictionary capacities and
space widths; the JSON records include the exact XML/default/runtime hashes,
shard fingerprint, document/source addresses, actual input/target counts,
optimizer steps, dtype, compiler intervals and sampled memory. Compiler
intervals overlap and must not be summed as independent costs. The MPS memory
sampler reports an observed high-water mark, not a guarantee of catching every
allocation spike.

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
.venv/bin/python bin/bench_sentence_expectation.py synthetic --threads 1 --out /tmp/expectation-synthetic.json
.venv/bin/python bin/bench_sentence_expectation.py fineweb --device mps --backend eager --batch-size 1 --docs 24 --train-steps 7 --eval-steps 8 --threads 1 --out /tmp/expectation-fineweb.json
.venv/bin/python bin/bench_sentence_expectation.py answers --config data/BasicModel_answers_benchmark.xml --device mps --backend eager --batch-size 1 --train-steps 7 --eval-steps 8 --threads 1 --out /tmp/expectation-answers-b1.json
.venv/bin/python bin/bench_sentence_expectation.py answers --config data/BasicModel_answers_benchmark.xml --device mps --backend eager --batch-size 2 --train-steps 7 --eval-steps 8 --threads 1 --out /tmp/expectation-answers-b2.json
```

## Verification

The recorded-compose reviewer probe failed because one selected `sum`
evaluated all 16 binary operators. The repaired eager and fullgraph paths
preserve selected values and mathematical gradients, including mixed rows
whose two operands share storage. The affected output/reconstruction run
passed **111 tests** in 352.10 seconds.
The gradient comparison treats an unused parameter's `None` gradient as zero;
it does not claim identical optimizer histories for branches now skipped.

The compiled-unpacked probe then reproduced zero training observations with
real explicit-state forwards. Its fix and independent throughput-accounting
probe passed **9 focused tests**; the broader affected run passed **81 tests,
with 4 skips**, in 215.46 seconds. Both new XML fixtures pass `xmllint` schema
validation.

The first full-suite run terminated with exit 137 at 81% under macOS memory
pressure, without a final test result. The filtered
[host diagnostic](2026-09-16-expectation-data/full-suite-resource-failure.json)
identifies pytest as the largest process. The test harness now
[releases compilation caches and collects reference cycles between modules](../../test/conftest.py#L49),
after their model fixtures finish. Collection, assertions and within-module
execution remain unchanged. The cleanup's affected run passed **12 tests in
30.13 seconds**. The complete rerun then passed **4,157 tests, with 51 skips,
7 expected failures and 4 subtests passed, in 2,466.59 seconds (41m 6s)**.
The interrupted run does not count as a green suite. All 47 runtime files and
395 test files retained their pre-run hashes; the six protected user documents
also remained unchanged. Model runtime files retain the measured hashes.

## Protocol declared before measurement

### Learned context gate
- Seeds 0, 1, 2; disjoint synthetic training/held-out documents, fresh initial state per document. Local meanings [NP1,VP,NP2], D=8, K=4. NP1 is fixed, VP follows a known role-sensitive rotation, NP2 depends on its preceding value and VP. Full vectors change across document instances. All targets detached. Never encode addresses as features.
- Train the actual SentenceExpectation head with the same optimizer, examples and update count for ordered, shuffled and context-free conditions. Use the actual root benchmark as an additional scope baseline, reporting its smaller parameter count and evaluating its broadcast prediction against the complete target. Shuffled training destroys adjacency within the training partition only. Context-free uses zeroed prior payloads with the same head and target distribution. No test-document leakage.
- Record pre/post held-out feature MSE, occupancy BCE, eligible pairs, separate role errors, target variation/discrimination, parameter count, update count and seed. The controlled synthetic representation is fixed; do not label this alone joint encoder learning. Existing invertible-family test is separate joint-gradient evidence.
- Predeclared positive context gate: ordered held-out feature MSE at least 20% below both shuffled and context-free in each seed. Report means and ranges, including failed seeds without retuning the threshold. Root-only fidelity is measured over complete roles as well as its own root objective.

### Native FineWeb / speed
- Use actual local FineWeb-Edu shard through source_addresses and the native corpus scheduler. Train/validation documents disjoint. Record shard fingerprint and exact XML/overrides, count all seen observations and eligible pairs (including cross-brick continuation).
- Run real warmed runBatch/runEpoch including backward, gradient balancing and optimizer; no inference-only timings presented as training throughput. Record inter feature/BCE, reconstruction path/loss, supplied-answer loss separately, live-predictor changes, detached targets/context after step, input sentences/s and predicted-targets/s. Include known-address boundary fraction.
- Canonical BasicModel.xml: current expectation on, structured scope, inter=.1, ARMA=contrastive=0, detached reverse student baseline until item5. B=1 or2, modest document cap as memory requires; keep native dimensionality. Record measured batch/word counts and any cap rather than claiming generic model speed. A compact topology can be an additional explicitly labelled experiment.
- First 2 batches are warmup; at least 5 subsequent full steps for steady timing. Synchronize device; report total and median timings, initial compile/warmup, actual selected backend, unique graph/recompile events and peak process/device memory. Run timed measurements without concurrent pytest or other benchmark workers.
- Start on this host's MPS/eager graph-capture operating point, recording AMP and effective compute dtype. This backend is the configured MPS default; it is not an Inductor code-generation result. Use CPU only as a separately labelled fallback or comparison, with any failure and reason retained.
- Read-only held-out reconstruction/inter before and after the short run; ordinary default inference produces metrics but no training accumulators/updates. Also evaluate matched context-free and shuffled history on the same held-out checkpoint. Report null/negative short-run corpus learning honestly; do not infer multistep reasoning from reduced inter MSE.
- Supplied-answer throughput is a separate native supervised workload with separately supplied desired answers and actual output loss/backward. Give dataset, batch, lengths, sample count and output mode. Do not call unlabeled FineWeb reconstruction supervised-answer throughput.

### Later tied reconstruction and reasoning
- Repeat matched full training measurements after tied reconstruction migration; preserve pre-migration reference.
- §4 learned query utility uses separate preregistered tasks/controls and explicit reward/baseline definitions; head learning or throughput is not its proof.

### Supplied-answer operating point (declared before measurement)
Use a derived BasicModel configuration with unchanged canonical PS/WS/concept widths and dictionary capacities. Change the workload to the existing math loader's stage-0 addition problems (range 16, 64 problems, fixed seed 42), set OutputSpace.nOutput=16 for its supplied one-hot labels, disable packing and the What curriculum, and use reconstructionScale=0.5 so both reconstruction and supplied-output losses have positive weight. Keep expectation on. Report this as a short numeric supplied-answer workload, not natural-language answer quality or a reasoning benchmark. No target or solver structure enters runtime question context. Run B=1 and B=2 if memory permits, each with at least seven full training calls. FineWeb comparisons use canonical BasicModel.xml separately.

The benchmark counts only SUPERVISED questions among actual eligible answer-mask rows; PRESENT and temporal validation targets must not be called supplied answers. Include actual active word-unit counts and effective loss weights. Reject insufficient measured steps, two-pass training without explicit attribution, and answer configurations whose reconstruction ratio zeroes answer credit. Record real head changes so an active-looking reported loss cannot masquerade as a trained objective.
