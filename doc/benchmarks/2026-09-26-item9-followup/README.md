# Item 9: learning evaluations after substantial training

Alec selected **one million completed FineWeb training sentences** as the
minimum for quality gates. Small-model correctness checks still run normally.
Missing or insufficient training evidence skips a quality evaluation; it does
not pass the gate. Once a checkpoint qualifies, failures remain failures.

The unchanged wording diagnostic still misses 5 of 56 examples. Four failures
come from the learned chooser leaving `also match` uncombined; later folds
attach the wrong subject. The fifth picks `part` where a surface attachment is
needed. Retained lexical values are unchanged, and the errors reproduce when
scoring the intended parser states. These are useful development findings,
not evidence about a well-trained grammar. The old curriculum is available as
`run_wording_curriculum` in `test/test_surface_grammar.py`; its assertions,
seed and budgets are unchanged.

## Training prerequisite

Checkpoints now save `training_state.fineweb_training_progress`: completed
training sentence presentations, completed optimizer updates and source
manifests. The counter uses owned sentence programs and their training-source
addresses. It handles ragged packed inputs and excludes padding, inference,
validation, preflight and extra exploration/context passes. It reads host
metadata and tensor shapes, never accelerator values. AMP-skipped updates do
not count; a fused AMP optimizer's internal skip cannot be certified without
a device read, so those updates are conservatively uncounted. Repeated epochs
count as repeated presentations, not distinct sentences.

Old checkpoints without this record have unknown exposure. A large corpus,
a large epoch number or a batch count does not replace the missing evidence.
The minimum is configurable for explicitly different evaluation protocols.

The loaded-model wording test keeps the same 56 meaning, generation and
rereading assertions. Two additional artifact tests measure prediction's
benefit from context and predictive-thought work at matched or better answer
error. The former is a same-checkpoint ablation, not a substitute for the
original separately trained, equal-update controls. These tests do not train
a fresh tiny model before asserting quality.

Run the qualified evaluations with the checkpoint's matching configuration:

```sh
BASICMODEL_FINEWEB_CHECKPOINT=/path/to/trained.ckpt \
BASICMODEL_FINEWEB_CONFIG=/path/to/model.xml \
RUN_SLOW=1 BASICMODEL_DEVICE=cpu \
.venv/bin/python test/test_report.py --memory-gib 8 --workers 1 \
  test/test_fineweb_learning.py \
  test/test_surface_grammar.py::test_real_text_has_a_complete_selected_meaning
```

`BASICMODEL_FINEWEB_MIN_SENTENCES` defaults to `1000000`. No qualifying
checkpoint was supplied or evaluated in this follow-up. Unit tests use small
explicit thresholds only to verify the loading/evaluation plumbing; they do
not establish learning quality.

## Measurement corrections

The context-free control now removes role masks as well as vectors. The
previous native training path already did this, but its scoring helper did
not, creating a train/evaluation mismatch. Historical context-free scores
must not be used as a clean control comparison.

The old thought score assumed that presenting an assertion made it known
true, and the text probe had not presented its held-out assertions to memory.
Its Brier scores are therefore not valid answer-error evidence. The native
probe also tried to question meanings without a grammatical VP. The new
predictive-thought test asks an explicit `arma` question about an owned
observation and scores its forecast only when the next sentence arrives.
It requires a typed prediction; an unknown truth answer, another operation or
no result cannot masquerade as a successful answer. Both gains start from
separate restores of the same checkpoint. This measures predictive thought,
not general reasoning competence.

The first real restore probe also exposed an existing checkpoint-loader bug:
the outer shape audit rejected LTM's variable-length `leaf_codes` column
before its owner could resize it. The audit now permits that owner's valid
one-dimensional column to reach its existing loader. Malformed ranks and
fixed-column mismatches still fail. No weights are replaced to make a quality
test run.

See the [protocol](PROTOCOL.md) for the policy and evaluation details. The
[original item 9 receipt](../2026-09-26-item9/README.md) and
[9b source snapshot](../2026-09-26-item9b-corrections/README.md) remain historical.
All changes are uncommitted for Claude's review.

## Verification and review material

The 668-file runtime/test/config snapshot has aggregate SHA-256
`a0d8e87bc157c303061eba13798c59908545455d29fcb4a52688c2a2ce03cc20`.
The [source map](source-manifest.json), [source archive](review-source.tar.gz)
and [patch since the 9b corrections](changes-since-corrections.patch) identify
the ten source files changed in this follow-up. Earlier 9b changes remain in
the tree and in their own review receipt.

The [targeted selection](targeted/result.json.gz), with slow tests enabled,
passes **32 checks** and skips the **three quality evaluations** because no
qualified checkpoint was supplied. It takes 201.58 seconds, with a maximum
worker footprint of 1.73 GiB under the 8 GiB cap. The integration test performs
an actual optimizer update, saves/restores its ragged sentence count, restores
the complete model, reads a held-out fixture and executes typed predictions.
Its deliberately small threshold verifies the machinery, not learned quality.

The [final default sweep](full/result.json.gz) completes **4,938 unique cases:
4,611 passed, 326 skipped and one existing expected failure**, with no
unexpected failures. It takes 1,408.68 seconds. Three fresh workers each have
the unchanged 8 GiB cap, with one file and at most 32 cases per worker batch;
the largest worker peaks at 7.19 GiB. There are no compile-cache retries or
memory-limit stops. The two new slow artifact evaluations account for the
increase in default skips; the explicit targeted run above verifies their
missing-checkpoint prerequisite. The existing wording test was already slow.
The full and targeted runs and the serial baseline all match the source map
above. [Worker logs](full/workers.log.gz) and the
[full source manifest](full/source-manifest.json.gz) preserve the evidence.
Final documentation-link verification passes **98/98**.

The source-matched [serial baseline](baseline/baseline.json.gz) is unchanged:
reconstruction cost **.1505906619 before training, .1347484022 during training
and .1362805218 after training**. No learning improvement is claimed from
these seven updates.

The [wording diagnostic](diagnostics/wording/result.json.gz) was collected on
the earlier 663-file snapshot, before this follow-up changed test entry points.
Its preserved `wording_diagnostic.py` harness is for that source archive; use
`run_wording_curriculum` for the current diagnostic entry point. The failed
benchmark, provenance and checkpoint-restore probes are retained in the
[diagnostic index](diagnostics/index.json). The [validation summary](validation-summary.json) records
the final verification status.
