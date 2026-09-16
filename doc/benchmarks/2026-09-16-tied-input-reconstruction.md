# Tied input reconstruction: migration measurements (September 16)

Status: **native measurements and full-suite verification complete**.
This report accompanies the integrated specification's
[section 6](../plans/2026-09-15-next-sentence-as-the-production-objective.md#6-code-review-compiled-reverse-loops-2026-09-12).
The completed optimizer runs and fidelity checks below are accompanied by a
green background full suite: **4,220 passed, 51 skipped, 7 expected failures,
and 4 subtests passed in 4066.68 s**. Two earlier failures exposed stale test
setup/hooks; their affected files and the subsequent full run pass.

## Comparison and protocol

The preserved before baseline is basicmodel
`aa5e018b67bf3be946c0b75c5baf33c9cc84ab4b`, described in
[the sentence-expectation report](2026-09-16-sentence-expectation.md).
That version trained a detached reverse student. The tied migration trains
cross entropy over bytes through the existing NUL word terminator from the completed input's
recovered word ideas, once per word/sentence/row. Earlier byte-only runs are
labelled separately below. Those optimized scalar losses have different meanings and
are not interchangeable fidelity scores. The observer also reports continuous
input-event MSE and the existing band-aware event score, without rerunning
reconstruction. Event error is a common diagnostic where either path publishes
a reconstructed input event; byte/idea error is available only when published.

Use the same M4 Max / 36 GiB host, configured MPS device and `eager` fullgraph
forward backend, FP32 with AMP off, seed 42, one CPU thread, ordinary Adam and
native data/epoch scheduling. The candidate's separate reconstruction uses
`aot_eager` to capture its backward program as well; report both backends and
the placement explicitly. Neither backend uses Inductor code generation.
`eager` is the configured MPS default in
[backend selection](../../bin/util.py#L472); these runs measure full native
training, not just graph-capture correctness.
Keep `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.60` and the low ratio `0.50`;
retain failed allocation/compile runs. Never run a benchmark concurrently with
pytest or another benchmark.

The answer, FineWeb and long-input runs have two warmup and five measured full
optimizer steps. The output-length run has two warmup and thirteen measured
steps. Step
time includes staging, forward composition, reconstruction, downstream losses,
backward, gradient balance, optimizer, cleanup and observer overhead. Compiler
intervals overlap and are not summed. Memory reports distinguish OS peak RSS
from device high-water samples every 20 ms. Raw reports identify source/config
hashes, actual words, completed sentences, prediction targets and supplied
answer rows.

Apply the retained runtime patches to `aa5e018` to reproduce the uncommitted
measurement versions: [eager-backward baseline source](2026-09-16-tied-reconstruction-data/eager-backward-baseline-source.patch)
and [AOT candidate source](2026-09-16-tied-reconstruction-data/aot-candidate-source.patch).
The five changed baseline files were reconstructed and verified against every
corresponding SHA-256 in its raw report. Use that report's effective configuration;
the baseline predates the separate-placement setting in the current XML.
The [final candidate patch](2026-09-16-tied-reconstruction-data/final-candidate-source.patch)
and [SHA-256 manifest](2026-09-16-tied-reconstruction-data/final-candidate-manifest.json)
also include word termination, the retained-buffer fix, and all new benchmark
configurations. Final timing results must match those hashes.

The observer records compose-parameter ownership, the quadrature norm of hook
contributions before joint-gradient balancing, and final parameter deltas.
These are separate measurements: hook credit alone does not prove an optimizer
update, and a total update does not identify which objective supplied it.
Direct kernel and reconstruction-only training tests establish source-specific
gradient paths. Dictionary targets and snapshots stay detached.

## Operating points

- The unchanged historical `BasicModel_answers_benchmark.xml` supplies
  three-word arithmetic questions and separate 16-way numeric labels. The new
  `BasicModel_answers_tied_benchmark.xml` changes only reconstruction selection,
  placement and its decoder weight. Compare B=1 and B=2.
- Production `BasicModel.xml`: FineWeb-Edu, W=256, B=1, 24 documents, no supplied
  answers. Report eligible predecessor/target pairs separately from input rate.
- `BasicModel_long_tied_benchmark.xml`: controlled 16-word inputs with separate
  binary labels. Compare explicit reconstruction basis limits 8 and 16. This
  repeats a small vocabulary and is a workload/cost probe, not learned language
  or held-out generalization evidence.
- A numeric output tensor's dimensions are not a surface-word count. The
  observer reports actual emitted words only if the output walk executed.
  `BasicModel_output_tied_benchmark.xml` adds a controlled output-cost run:
  native widths and math inputs, unchanged compose rules, and a generate catalog
  restricted to the existing sum expansion plus policy stop. Training samples
  the policy with supplied-answer credit. Record each actual emitted length and
  truncation separately; this restricted-catalog run is not language-quality
  evidence. Its 15 B=2 optimizer steps allow more output-length observations
  after the same two warmup steps. The completed measurements are below.

## Final readout-corrected native measurements

The matched original baseline has completed using the final observer. Its five
measured B=1 steps process five supervised input sentences in 34.79496 s:
**0.143699 sentences/s**. Two warmup steps total 29.00260 s. Input-event MSE
on the same eight validation inputs falls from 0.00794243 to 0.00562872.
Peak sampled training MPS tensor/driver allocations are 6.75/7.80 GB; OS peak
RSS including setup is 13.60 GB. The answer head updates; these independent
questions have no eligible prediction pairs.
[Final matched baseline](2026-09-16-tied-reconstruction-data/answers-b1-legacy-observer.json),
[observer patch](2026-09-16-tied-reconstruction-data/legacy-observer-source.patch).

### Corrected tied reconstruction, B=1

The corrected B=1 candidate completes five measured steps in 57.33988 s:
**0.0871993 supervised input sentences/s**, 60.68% of the matched baseline rate.
Warmup totals 210.32442 s. All seven training steps retain four captured graphs.
Peak sampled training MPS tensor/driver allocations are 6.15/7.08 GB; OS peak
RSS including setup is 13.60 GB. The allocator cap remains unchanged.
[Final tied B=1 report](2026-09-16-tied-reconstruction-data/answers-b1-final.json).

| Mean over the same eight B=1 validation inputs | Before | After |
|---|---:|---:|
| Byte/NUL cross entropy | 0.976566 | 0.486366 |
| Recovered-idea MSE | 0.000662273 | 0.000342276 |
| Continuous input-event MSE | 0.000193558 | 0.000211832 |

No training or validation reconstruction is truncated. Answer-parameter delta
norm is 1.24508; there are no eligible prediction pairs. The post-training
fidelity values agree with the preserved pre-readout run to the reported
precision. Byte and idea error improve, while continuous event error worsens.
These short supervised runs do not establish held-out language generalization.
[Fidelity, counts and updates](2026-09-16-tied-reconstruction-data/answers-b1-final.json).

### Corrected tied reconstruction, B=2

Ten measured supervised input sentences take 58.04072 s:
**0.172293 sentences/s**. Warmup totals 210.35442 s; all seven training steps
retain four graphs. Peak sampled MPS tensor/driver allocations are 6.21/6.72 GB,
with 13.60 GB OS peak RSS. The twelve validation inputs have byte/NUL cost
0.891990 → 0.415938, recovered-idea MSE 0.000655356 → 0.000360117, and
continuous input-event MSE 0.000193449 → 0.000215769. No reconstruction is
truncated. Answer-parameter delta norm is 1.33016; no predictor pair is eligible.
B=1 and B=2 have different validation sample counts, so their means are not a
matched-sample quality comparison.
[Final tied B=2 report](2026-09-16-tied-reconstruction-data/answers-b2-final.json).

### Corrected packed FineWeb

All seven optimizer steps complete. The five measured steps process 63 input
sentences and 62 prediction targets in 442.54584 s: **0.142358 inputs/s** and
**0.140098 targets/s**. Warmup totals 360.57351 s; the graph count stays at five
throughout training. The measured document-boundary fraction is 1/63. Peak
sampled training MPS tensor/driver allocations are 8.09/9.64 GB; OS peak RSS
including setup is 13.73 GB. The entire training phase observes 85 sentences
and 83 eligible targets, with no supplied answers. The predictor's parameter
delta norm is 4.81945; the shared lift/lower and verb parameters also update.
Observed targets and durable context remain detached, and no reconstruction
is truncated. [Final FineWeb report](2026-09-16-tied-reconstruction-data/fineweb-b1-final.json).

| Mean over eight held-out inputs / seven prediction pairs | Before | After |
|---|---:|---:|
| Byte/NUL cross entropy | 0.610513 | 0.558510 |
| Recovered-idea MSE | 0.000209186 | 0.000335417 |
| Continuous input-event MSE | 0.00175806 | 0.00192063 |
| Prediction feature MSE | 0.0120727 | 0.0399858 |
| Role-presence BCE | 0.694184 | 0.461728 |

Byte error and role-presence error improve, while idea, event and prediction
feature errors worsen. Post-training feature MSE on fixed pretraining encodings
is 0.0156696, so encoder drift alone does not explain the regression. The final
ordered feature error (0.0399858) is also worse than shuffled (0.0391895) and
context-free (0.0396886) controls. These results verify executed joint training,
not learned predictive benefit. The small held-out set has no occupied VP/NP2
targets; NP1 target variance changes from 0.00853102 to 0.0186967. It cannot
establish empirical full-role quality or language generalization.
[Fidelity and causal controls](2026-09-16-tied-reconstruction-data/fineweb-b1-final.json).

### Corrected 16-word input, basis limit 8

Five measured supervised inputs take 76.69574 s: **0.0651927 sentences/s**.
Warmup totals 219.16476 s; all seven training steps retain four graphs. Peak
sampled training MPS tensor/driver allocations are 6.37/6.72 GB, and OS peak
RSS including setup is 13.60 GB. No training or validation reconstruction is
truncated. The eight validation inputs have byte/NUL cost 0.683980 → 0.118890,
idea MSE 0.000294057 → 0.0000387970, and event MSE 0.00119715 → 0.00105843.
The answer head updates (delta norm 1.32611); independent questions provide
no eligible prediction targets. This controlled small-vocabulary workload does
not establish language generalization or worst-case basis-search cost.
[Final 16-word / basis-8 report](2026-09-16-tied-reconstruction-data/long-b1-k8-final.json).

### Corrected 16-word input, basis limit 16

Five measured supervised inputs take 76.82774 s: **0.0650807 sentences/s**.
Warmup totals 218.41971 s; all seven training steps retain four graphs. Peak
sampled training MPS tensor/driver allocations are 6.37/6.71 GB, and OS peak
RSS including setup is 13.60 GB. No reconstruction is truncated. The eight
validation inputs have byte/NUL cost 0.589136 → 0.125864, idea MSE
0.000266865 → 0.0000266718, and event MSE 0.00118811 → 0.00105541. The answer
head updates (delta norm 1.32610); there are no eligible prediction targets.
[Final 16-word / basis-16 report](2026-09-16-tied-reconstruction-data/long-b1-k16-final.json).

| Basis limit | Measured seconds for five inputs | Input sentences/s | Warmup seconds |
|---|---:|---:|---:|
| 8 | 76.69574 | 0.0651927 | 219.16476 |
| 16 | 76.82774 | 0.0650807 | 218.41971 |

These single short runs have nearly equal rates and mixed relative fidelity;
they do not establish a reliable ranking. Both workloads use repeated words
from a small vocabulary. Snapshot entry counts include WORD and OBJECT entries
and must not be interpreted as distinct candidate concepts or as proof that
every bounded candidate was needed by the selected inverses.

### Corrected output generation, B=2

All fifteen optimizer steps complete. The thirteen measured steps process
26 supervised inputs in 155.10878 s: **0.167624 input sentences/s**. Two warmup
steps total 213.15711 s, and all training steps retain six graphs. Peak sampled
training MPS tensor/driver allocations are 6.48/7.87 GB; OS peak RSS including
setup is 13.60 GB. The answer parameter delta norm is 0.454615. No eligible
prediction pairs occur in these independent arithmetic questions.
[Final output report](2026-09-16-tied-reconstruction-data/output-b2-final.json).

| Generated words per row | Rows in the thirteen measured steps |
|---|---:|
| 1 | 23 |
| 2 | 1 |
| 3 | 1 |
| 6 | 1 |

Every input has three words, so the six-word response is a completed output
longer than its input. There is no output or reconstruction truncation in
training or validation. The two batches containing three- and six-word
responses take 12.55484 and 12.57643 s respectively; each includes another
one-word response. These are observed batch timings, not isolated per-word
costs. Evaluation selects one word in all twelve rows both before and after
training. The restricted sum/stop catalog probes execution cost; its outputs
are not evidence of learned linguistic quality.
[Lengths, truncation and step times](2026-09-16-tied-reconstruction-data/output-b2-final.json).

Across the same twelve validation inputs, byte/NUL cost improves from 0.372731
to 0.335348 and continuous event MSE from 0.000216558 to 0.000206819; recovered
idea MSE worsens from 0.000286443 to 0.000313540. Report these diagnostics
separately from answer-head updates and throughput.
[Validation fidelity](2026-09-16-tied-reconstruction-data/output-b2-final.json).

Full-suite verification passes; the complete result is recorded under
[Validation](#validation). The prior version's preserved measurements follow.

## NUL objective before the output-readout correction

These preserved measurements precede the output-readout allocation fix below.
Their source is retained in the [pre-readout patch](2026-09-16-tied-reconstruction-data/pre-readout-candidate-source.patch)
and [manifest](2026-09-16-tied-reconstruction-data/pre-readout-candidate-manifest.json).
The matched B=1 comparison is complete for that version. Both runs use the same observer,
seven optimizer steps, the same eight validation inputs before/after training,
and the protocol above. The final operating points above supersede these
pre-readout measurements.

| Reconstruction | Five measured steps | Supervised input sentences/s | Warmup seconds |
|---|---:|---:|---:|
| Original detached student at `aa5e018` | 35.1335 s | 0.142314 | 29.4134 |
| Completed tied reconstruction, NUL objective | 57.3601 s | 0.0871685 | 210.133 |

The corrected tied run reaches 61.25% of the matched student's sentence rate.
Its captured graph count stays at four throughout training. These are different
learning objectives and gradient paths, not interchangeable implementations of
one scalar loss. [Matched original report](2026-09-16-tied-reconstruction-data/pre-readout-answers-b1-legacy-observer.json),
[corrected tied report](2026-09-16-tied-reconstruction-data/pre-readout-answers-b1-final.json).

Peak sampled training MPS tensor/driver allocations are 6.74/7.80 GB for the
original student and 6.28/6.67 GB for tied reconstruction. OS peak RSS including
setup is 13.68/13.60 GB respectively. No allocation cap was raised.

| B=1 validation diagnostic, mean over eight batches | Before | After |
|---|---:|---:|
| Original student: input-event MSE | 0.00794243 | 0.00562872 |
| Tied reconstruction: input-event MSE | 0.000193558 | 0.000211832 |
| Tied reconstruction: byte/NUL cross entropy | 0.976566 | 0.486366 |
| Tied reconstruction: recovered-idea MSE | 0.000662273 | 0.000342276 |

The tied run has no reconstruction truncation. Its byte and idea errors improve,
while its continuous event error worsens. Cosine-based word identity scoring
does not enforce exact continuous-state recovery; the event diagnostic makes
that limitation visible. The original student publishes no comparable byte or
idea cost. Both answer heads update; neither predictor has an eligible target
pair in these independent arithmetic questions. This short run does not establish
held-out language generalization. [Original fidelity and updates](2026-09-16-tied-reconstruction-data/pre-readout-answers-b1-legacy-observer.json),
[tied fidelity and updates](2026-09-16-tied-reconstruction-data/pre-readout-answers-b1-final.json).

### Batch size 2

The corrected B=2 run processes ten measured input sentences in 58.0811 s:
**0.172173 supervised input sentences/s**. Warmup totals 211.919 s; all seven
training steps retain four captured graphs. Peak sampled training MPS tensor/
driver allocations are 6.42/6.69 GB, and OS peak RSS including setup is 13.60 GB.
[B=2 raw report](2026-09-16-tied-reconstruction-data/pre-readout-answers-b2-final.json).

The same twelve validation inputs are checked before/after this run. Byte/NUL
error falls from 0.891990 to 0.415938 and recovered-idea MSE from 0.000655356
to 0.000360117; event MSE rises from 0.000193449 to 0.000215769. There is no
reconstruction truncation. Answer-parameter delta norm is 1.33016; the predictor
has no eligible targets and remains unchanged. B=1 and B=2 validate different
sample counts, so their fidelity means are not a matched-sample comparison.
[Fidelity, counts and parameter changes](2026-09-16-tied-reconstruction-data/pre-readout-answers-b2-final.json).

### Packed FineWeb

The corrected FineWeb run completes all seven optimizer steps, including the
retained gradient reads that failed in the earlier candidate. Its five measured
steps process 63 sentences and 62 prediction targets in 444.293 s:
**0.141798 input sentences/s** and **0.139548 prediction targets/s**. The measured
document-boundary fraction is 1/63. All seven training steps retain five captured
graphs; warmup totals 357.966 s. Peak sampled MPS tensor/driver allocations are
8.11/9.70 GB and OS peak RSS including setup is 13.72 GB.
[Completed native report](2026-09-16-tied-reconstruction-data/pre-readout-fineweb-b1-final.json),
[run log](2026-09-16-tied-reconstruction-data/pre-readout-fineweb-b1-final.log).

Training observes 85 sentences with 83 eligible targets and no supplied answers.
The predictor's parameter delta norm is 4.81945. Shared lift/lower and verb
transforms also update; observed targets and durable prediction context remain
detached. There is no reconstruction truncation in training or validation.
[Counts, ownership and updates](2026-09-16-tied-reconstruction-data/pre-readout-fineweb-b1-final.json).

The eight held-out inputs provide only seven prediction targets. Mean batch
byte/NUL cost falls from 0.610513 to 0.558511, but recovered-idea MSE rises from
0.000209186 to 0.000335416 and input-event MSE from 0.00175806 to 0.00192063.
Prediction feature MSE worsens from 0.0120727 to 0.0399857, while presence BCE
improves from 0.694184 to 0.461749. Evaluating the trained predictor on fixed
pretraining encodings still gives worse feature MSE, 0.0156696, so encoder drift
alone does not explain the feature regression.
[Held-out fidelity and fixed-encoding control](2026-09-16-tied-reconstruction-data/pre-readout-fineweb-b1-final.json).

The final ordered-context feature MSE (0.0399857) is also worse than shuffled
(0.0391895) and context-free (0.0396886) controls. This short run verifies executed
joint training and the compiler fix; it does **not** demonstrate learned
predictive benefit or language generalization. The target's observed NP1 variance
rises from 0.00853102 to 0.0186966; these samples have no occupied VP/NP2 targets,
so they provide no empirical full-role prediction comparison.
[Controls and per-role statistics](2026-09-16-tied-reconstruction-data/pre-readout-fineweb-b1-final.json).

## Output readout allocation and checkpoint compatibility

The first native output-length run failed before its initial validation completed.
Generated percepts had shape `[2, 512, 1088]`; the old readout attempted a
557,056-by-557,056 LDU factor, requesting 1,156 GiB for that matrix. The allocator
limit was unchanged. A meta-device regression reproduces the parameter-count
problem without attempting that allocation.
[Failure report](2026-09-16-tied-reconstruction-data/output-adapter-allocation-failure.json),
[traceback](2026-09-16-tied-reconstruction-data/output-adapter-allocation-failure.log).

New answer readouts retain only the rectangular factor entries that contribute
to the existing LDU forward projection. All generated percept coordinates remain available to the learned projection;
there is no pooling, cropping or changed input reconstruction inverse. Value and
gradient comparisons cover shrinking, square and expanding maps. Checkpoints
identify the compact layout with `_readout_format`; older adapters restore their
original factors. The focused group passes 27 tests, including strict real-model restores of both
layouts, exact Adam-state equality and identical next updates. All native reruns
above complete, and the subsequent full suite passes.
[Focused green log](2026-09-16-tied-reconstruction-data/readout-checkpoint-l1-green.log).
The three complete output synthesis/supervision/walk files also pass: 70 tests,
12 warnings, 283.20 s.
[Output affected log](2026-09-16-tied-reconstruction-data/readout-output-affected-green.log).
[Readout](../../bin/Layers.py#L1633), [adapter](../../bin/Spaces.py#L30277),
[restore](../../bin/Models.py#L8967),
[regression](../../test/test_output_percept_readout.py#L10).

The initial compatibility probes also exposed `auto` being passed directly to
PyTorch's backend lookup. The separate reconstruction compiler now resolves that
application policy to its first configured backend, matching the ordinary
compiler's first choice. An actual CPU Inductor probe passes repeated gradient
reads. The L1 test fixture now constructs its explicit legacy student mode,
instead of turning on `detachedReverse` after constructing tied mode and leaving
its student absent. No L1 gradient or optimizer assertions were removed.
[Backend red](2026-09-16-tied-reconstruction-data/reconstruction-auto-red.log),
[mode red](2026-09-16-tied-reconstruction-data/l1-explicit-mode-red.log),
[green group](2026-09-16-tied-reconstruction-data/readout-checkpoint-l1-green.log),
[backend resolution](../../bin/Models.py#L11186),
[fixture](../../test/test_concept_readout_l1.py#L247).

## Failures retained

The first packed FineWeb AOT candidate completed eight validation batches,
then failed before its first optimizer update: its cached backward rejected
retained gradient reads. A focused probe reproduces the transition from an
ordinary first backward to retained reads on a later call; retained-first
probes alone do not expose it. The fix disables buffer donation during
reconstruction compilation and normalizes this PyTorch build's disabled
metadata (`None` to `[]`) before restoring the global setting. The cache-lifetime
probe passes; the final packed run above now completes all seven optimizer steps.
[Raw report](2026-09-16-tied-reconstruction-data/fineweb-aot-retained-backward-failure.json),
[traceback](2026-09-16-tied-reconstruction-data/fineweb-aot-retained-backward-failure.log).

The byte-only objective also has a confirmed word-termination gap: WORD
candidates "a" and "ab" both score 0.00004524 against target "a" when only its
valid byte positions are scored. The suffix probe reproduced that behavior and
passes when the terminator is scored. The final objective uses the existing
NUL byte (`0`) within the 256-byte alphabet and must be measured again. The
initial extra-category prototype was corrected before final native measurement.
The completed B=1/B=2 runs below remain the
byte-only cache comparison.

Alec's terminator clarification exposed the unnecessary extra category in the
first fix. The established token buffer uses NUL (`0`), so the final scorer
includes the first NUL within the existing 256-byte alphabet and ignores later
bytes, even when a supplied span includes them. The probe checks both value and
gradient invariance to trailing junk, and the uniform 256-byte null-candidate
cost. The affected termination/operator/dictionary group passes 66 tests in
44.39 seconds. [Failing probe](2026-09-16-tied-reconstruction-data/nul-termination-probe-red.log),
[passing group](2026-09-16-tied-reconstruction-data/nul-termination-probes-green.log).

This is whole-word spelling scoring: `a\0` has bytes `[97, 0]`, while `ab\0`
has `[97, 98, 0]`. At the second position the longer candidate supplies `b`
where the target requires NUL. Internal byte/morpheme segmentation does not
change that comparison. The scorer's null candidate is a uniform fallback for
an idea that matches no dictionary row; it is not a terminator. No additional
end-of-sentence category is introduced.
[Token layout](../../bin/Spaces.py#L1617),
[word cost](../../bin/Models.py#L11648).

A second probe exposed candidate clipping at the input window: when the target
fills that window, the longer candidate again appeared to end at the target's
last byte. Both one- and two-byte window probes failed. The snapshot now retains
one extra candidate byte and the scorer uses the target's own window, preserving
the longer spelling's continuation. Six targeted termination/null/masking checks
pass in 2.96 seconds.
[Failing boundary probes](2026-09-16-tied-reconstruction-data/full-window-probe-red.log),
[passing checks](2026-09-16-tied-reconstruction-data/word-end-probes-green.log).

The real packed joint-gradient test then exposed a dynamic-shape constraint in
the scorer: padding a fixed candidate window by `max(P + 1 - C, 0)` and slicing
it to the dynamic target width produced incompatible compiler guards. An
isolated probe reproduced the failure. Gathering through one masked sentinel
column preserves both shorter and longer target windows without constraining
`P`. The focused value/gradient probe, the packed optimizer test in both
reconstruction modes, and two explicit legacy-student compatibility checks
pass: five tests, three warnings, 281.33 seconds.
[Failing probe](2026-09-16-tied-reconstruction-data/dynamic-window-probe-red.log).

The first B=1 native run failed before validation completed because the eager
evaluation dispatcher did not produce the completed sentence tensors required
by tied reconstruction.
[Raw report](2026-09-16-tied-reconstruction-data/answers-b1-missing-state-failure.json),
[traceback](2026-09-16-tied-reconstruction-data/answers-b1-missing-state-failure.log).
The default-dispatch reviewer probe reproduced the same failure. Tied mode now
selects the canonical tensor loop in eager evaluation too; the objective and
checkpoint test file passed four tests after that change.

The target builder had another scoring gap: it recognized only single-byte
percepts. A whole-word percept for "alphabet" removed all eight target bytes;
a known prefix removed part of the spelling. The real radix-store probe
exercised these valid word-major tilings. Eager staging now expands every
input percept to its complete bytes for scoring, separately from the WORD
candidate snapshot. The two promotion probes and related window/dictionary/
checkpoint checks pass: 38 tests in 25.71 seconds. Earlier byte-cost changes
alone cannot establish learning benefit, because that older scorer could omit
promoted chunks. [Failing probes](2026-09-16-tied-reconstruction-data/promoted-target-probe-red.log).

## Historical byte-only measurements

### B=1: cached reconstruction backward

Seven optimizer steps completed with separate `aot_eager` reconstruction and
the unchanged `eager` forward backend. The five measured inputs took
58.2326 seconds: **0.0858626 supervised input sentences/s**. The two warmup
steps took 202.265 and 10.025 seconds; the five measured steps took
11.927, 12.006, 10.039, 12.181 and 12.079 seconds. The captured graph count
stayed at five across every training step.
[Raw report](2026-09-16-tied-reconstruction-data/answers-b1-aot.json),
[log](2026-09-16-tied-reconstruction-data/answers-b1-aot.log).

This is 10.47 times the diagnostic tied baseline's rate, and 59.37% of the
historical detached-student rate. It preserves the tied baseline's per-step
reconstruction scores and parameter deltas within rounding. The reconstruction
objective and encoder gradient reach differ from the detached student, so the
remaining slowdown is reported alongside that learning change. Peak sampled
training allocation was 6.34 GB in MPS tensors and 6.74 GB in the MPS driver;
OS peak RSS, including setup, was 13.60 GB. No allocation cap was raised.
[Timing, fidelity and parameter evidence](2026-09-16-tied-reconstruction-data/answers-b1-aot.json).

The 48 affected reconstruction/ownership/checkpoint/cache/benchmark probes
passed (one warning, 407.62 seconds) at this intermediate revision. The final
native measurements and validation record below supersede that partial gate.

### B=2: cached reconstruction backward

Seven optimizer steps completed; the five measured steps processed ten
sentences in 59.3985 seconds: **0.168355 supervised input sentences/s**.
Warmup took 201.260 and 13.008 seconds, and the graph count remained five
throughout training. Peak sampled MPS tensor/driver allocations were
6.41/6.69 GB; OS peak RSS including setup was 13.60 GB.
[Raw report](2026-09-16-tied-reconstruction-data/answers-b2-aot.json),
[log](2026-09-16-tied-reconstruction-data/answers-b2-aot.log).

All training and validation rows had zero reconstruction truncation. The
12 validation sentences' mean byte error fell from 1.06267 to 0.41593 and
idea MSE from 0.00065536 to 0.00036952. Event MSE worsened from 0.00019345 to
0.00021589. Answer parameters changed, with delta norm 1.32168; the predictor
had no eligible target pairs. B=1 validated eight sentences and B=2 validated
twelve, so their quality summaries are not comparisons on identical samples.
[Fidelity and update evidence](2026-09-16-tied-reconstruction-data/answers-b2-aot.json).

### Diagnostic B=1 baseline: eager backward

The first complete tied run used the graph placement and `eager` backend. Seven
full optimizer steps completed. After two warmup steps, five inputs took
609.572 seconds: **0.00820248 supervised input sentences/s**, versus the
preserved detached-student baseline's 0.144623 sentences/s. This is a performance
regression, not an approved operating point. A one-second macOS process sample
was taken at 09:10:58 PDT during this run to diagnose the slowdown; retain this
as a diagnostic baseline and rerun the eventual candidate without sampling.
[Raw report](2026-09-16-tied-reconstruction-data/answers-b1-eager-backward-baseline.json),
[log](2026-09-16-tied-reconstruction-data/answers-b1-eager-backward-baseline.log),
[historical comparison](2026-09-16-sentence-expectation.md).

The eight validation rows had no truncation before or after training. Mean
byte cross entropy fell from 1.15479 to 0.37009 and recovered-idea MSE from
0.00066227 to 0.00034929. Common input-event MSE worsened from 0.00019356 to
0.00021248; band-aware event score worsened from 0.00118766 to 0.00120174.
These are short-run checks, not established language generalization. The
supplied-answer heads and the selected lift/lower/verb compose parameters
changed. The sentence predictor remained unchanged because these independent
math questions contain no eligible predecessor pairs. Numeric answer tensors
do not establish generated surface length.
[Per-step fidelity and parameter changes](2026-09-16-tied-reconstruction-data/answers-b1-eager-backward-baseline.json).

The first optimizer step took 150.584 seconds; subsequent steps remained near
121–123 seconds without additional Dynamo graphs. Profiling points to repeated
construction of higher-order backward programs in the installed PyTorch loop
and conditional implementations. The completed AOT comparison above removes
that repeated compilation; final measurements with word termination and
retained-buffer support are separate. Partial compilation progress is not
throughput evidence.

## Validation

The final background command, run from `basicmodel/`, was:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python -m pytest test -q -x -p no:cacheprovider
```

It exits successfully with **4,220 passed, 51 skipped, 7 xfailed, 185 warnings,
and 4 subtests passed in 4066.68 s (1h 7m 46s)**. All 554 frozen runtime, test
and configuration files match the recorded SHA-256 manifest after the run.
The 12 native measurement source/configuration fingerprints and six originally
protected user-file fingerprints also remain unchanged.
[Full-suite green log](2026-09-16-tied-reconstruction-data/full-suite-green.log),
[frozen manifest](2026-09-16-tied-reconstruction-data/full-suite-frozen-manifest.json).

Native `xmllint --noout --schema data/model.xsd` validates `BasicModel.xml`
and all three new tied benchmark configurations (answers, long input and
output generation). This checks the schema changes even where the optional
Python `lxml` test is skipped.

The first full-suite run was interrupted after one failure, 514 passes and ten
skips (246.68 s), to diagnose the failure before another complete run. The aligned
geometry test assumed that forward left a one-word reshape cache. It now installs
that stale descriptor explicitly before testing root reversal, retaining the
eight-location shape, sparse-gradient and optimizer-update assertions. Its whole
file passes: **9 tests, 1 warning, 21.30 s**. No runtime or benchmark configuration
changed; the native source manifest still matches. The next run used `-x`
to stop at any further failure.
[Interrupted full-suite red](2026-09-16-tied-reconstruction-data/full-suite-interrupted-red.log),
[affected file green](2026-09-16-tied-reconstruction-data/aligned-fixture-green.log),
[explicit stale-cache setup](../../test/test_aligned_fold_binding.py#L273).

The second full run stops at an older IR refactor test after 1,401 passes,
25 skips and one expected failure (815.28 s). That test spied on `reverse`,
where input reconstruction now uses `_reverse_input_surface` to keep free
generation out of its path. The updated test inspects that boundary and checks
every actual seed against the owned conceptual state, retaining the finite
`reconstruction_reverse` diagnostic assertion for its legacy non-word fixture.
The whole file passes: **10 tests, 2 skipped, 2 warnings, 2.43 s**. Runtime and
benchmark configurations are still unchanged. The third full run passes.
[Second full-suite red](2026-09-16-tied-reconstruction-data/full-suite-v2-red.log),
[IR boundary green](2026-09-16-tied-reconstruction-data/ir-boundary-green.log),
[seed verification](../../test/test_ir_only_refactor.py#L266).

Before the readout/compiler compatibility follow-up, the affected group using
the existing NUL byte and complete percept spellings passed **211 tests, with
6 skipped and 15 warnings, in 1510.59 s**.
This includes real checkpoint/optimizer migration, cached repeated backward
reads, packed joint training, output isolation and explicit legacy modes.
All 552 frozen runtime/test/config files and the six originally protected user
files matched their recorded hashes after the run. The later 27-test and 70-test
groups above validate the readout/compiler follow-up. Native measurements and
the background full suite are complete.
[Complete affected-test log](2026-09-16-tied-reconstruction-data/affected-final-green.log).

The initial affected traversal group passed 48 tests (one warning, 485.43 s).
Additional operator checks passed 20 tests in 21.92 s: direct inverse gradients,
retained dictionary values through backward, selected-only operators, masked
candidates and exact 8/16 candidate pair bounds. These earlier groups are
historical evidence; the later affected runs and full-suite result determine
the migration's current verification status.

An intermediate affected run passed 58 tests in 734.62 seconds before an explicit
interrupt to address the full-window clipping probe. Its compiled packed and
cache-lifetime checks passed, but the interrupted run is not full affected-suite
approval. The complete final-source result is recorded above.
