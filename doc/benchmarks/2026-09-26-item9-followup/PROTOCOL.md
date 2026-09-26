# Item 9 follow-up: qualified learning evaluations

Alec clarified the acceptance policy during this follow-up: the small models
are undertrained, significant success is not expected, and learning-quality
tests should require substantial FineWeb training. He selected **one million
completed training sentences** as the initial minimum. This replaces the
initial follow-up proposal to repeat twelve more short training arms. Those
additional arms were not run.

Correctness tests remain unconditional. Learning-quality tests require an
explicit checkpoint and its matching configuration. They inspect persisted
FineWeb training exposure before constructing the model. Unknown or insufficient
exposure is a skipped prerequisite, never a pass. Corrupt metadata or an
explicitly missing/mismatched artifact is an error. A qualified checkpoint
runs the assertions normally; poor performance is not converted to a skip.

## What is counted

The training boundary counts nonempty, owned completed sentence programs with
valid training-source addresses, after a successful main optimizer update.
It recognizes the loader's FineWeb corpus manifest. Ragged packed rows count
actual sentences, not batch size times slots. Validation, inference, padding,
preflight and exploration/context passes do not increment exposure. Counts and
source manifests survive checkpoint save/load. Historical checkpoints without
the counter have unknown exposure; epochs or batch counts cannot substitute.
The count is training presentations, including repeated epochs, not unique
sentences. The default minimum is configurable explicitly for other studies.
AMP-skipped updates do not count. Fused AMP optimizers can skip inside their
device kernel; without a device read their updates cannot be certified, so
the counter conservatively excludes them.

## Quality evaluations

- Wording: preserve the existing 56 held-out meaning, generation and rereading
  assertions, but run them on the qualified loaded model. The old 9000-update
  tiny curriculum remains a separately callable development diagnostic.
- Prediction: read 64 chronological held-out FineWeb sentences from the first
  1024 documents, with the loader's document-separated validation split. Compare
  the trained head's ordered inputs with shuffled inputs (three permutations)
  and zero context. Require a benefit in MSE plus presence BCE, without an
  arbitrary minimum percentage improvement. This is a checkpoint ablation,
  not the original separately trained, equal-update control experiment.
- Thought: restore the same qualified checkpoint separately at gains zero and
  one. After a held-out observation, ask the grammar-owned `arma` operation for
  the next meaning; score its typed prediction only when the next sentence in
  that document arrives. Compare actual work at no worse MSE/BCE (1e-6 numerical
  tolerance), with complete answer coverage. A different operation, no result,
  or an unknown truth answer is unanswered, never zero prediction error. This
  tests predictive thought, not general theorem proving.

## Measurement defects and preserved diagnostics

The old context-free scoring helper removed values but retained masks, whereas
native context-free training removed both. Both now remove both. The original
short-run receipt remains historical; its context-free scores must not support
a performance claim.

The old thought probe treated a sentence as known true merely because it was
presented. Observations do not establish truth; its text trial also omitted the
held-out assertions from memory. Native root-only meanings cannot be made
executable questions merely by changing their mode. Those assertion-error/work
rows are not valid utility evidence. The new forecast task uses an explicit,
executable operation and a future observation as its offline target.

The unchanged wording diagnostic reproduced 5/56 wrong meanings. Its captured
word values were stable; the learned chooser made the same errors on annotated
parser states. No seed, label, budget or threshold was tuned to remove them.
This remains a development result, not a failed million-sentence model gate.

## Verification

All model/test processes have an 8 GiB cap; concurrent reservations total at
most 24 GiB. Freeze runtime/test/config source during runs, retain red probes,
run affected tests and one final full source-matched receipt, and leave the
changes uncommitted for Claude's review. No qualifying model is claimed or
manufactured by this change.
