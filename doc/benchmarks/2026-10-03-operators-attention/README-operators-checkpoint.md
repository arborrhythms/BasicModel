# Decoder exploration and operators — incomplete review checkpoint

This work continues the one existing tree from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`. Nothing has been committed or
pushed. **The operators update is incomplete at this checkpoint; 6.8 has not started.** This is
an inspectable checkpoint, not a claim that the requested sequence is finished
or ready to land. Claude must review before any commit. There is no conference
freeze. [plan.json](plan.json) preserves the requested order and baseline.

The accepted 6.9 record is unchanged: class MSE **.1147481948**, reconstruction
**0/4**, zero ownership conflicts; the earlier class **9/10** and reconstruction
**5/10** remain prior measurements. MM_xor remains red pending 6.8, with its
unchanged loss < .20 / 200-epoch bar. No seeds, gate bars, row capacities or
resource guards have changed. These two new unseeded trainings are separate
single measurements at different source stages, not a repeated-seed campaign
or evidence of a causal comparison.

## Decoder exploration (§26.3)

The reconstruction decoder now evaluates a greedy path and one path that
replays its greedy prefix, departs at one eligible round, and then proceeds
greedily. Illegal inverses and a binary undo without stack capacity are masked
before selection. Both paths are scored under the same parameters, before
training. Only a strictly lower reconstruction cost selects exploration;
ties and a path without a legal departure retain greedy. The score is the
existing free-byte/log(256) plus antipode/log(2) objective; the common
reconstructionScale does not change the comparison. Row selection retains only
the chosen decoder graph. Both compose trials still receive their own paired
decoder comparison before compose's existing training sequence.

Evaluation is greedy. The numerical decoder still sees only the root, end
slots and primed bank, with no compose journal teacher. The observational audit
records both candidate action sequences, their costs, departure and winner.
Tests cover strict ties, absent alternatives, selected-graph gradients, greedy
prefix/suffix replay and eager/fullgraph decoding. Failed probes precede repairs
in [probes/](probes/), including the compiled trace-stride repair.

## Measurements

Each stage uses one model and one 400-epoch training for both unchanged
XOR_grammar gate functions, bounded at 8 GiB and 1,800 seconds on CPU with
MODEL_COMPILE=none. The two gate consumers have the same model identity.

| Source stage | Class MSE | Class labels | Reconstruction | Ownership conflicts | Decoder explore wins |
|---|---:|---:|---:|---:|---:|
| Accepted 6.9 | .1147481948 | 4/4 | 0/4 | 0 | none |
| Decoder exploration, before operators | .03125037 | 4/4 | 0/4 | 0 | 1,790/3,200 (55.9375%) |
| Current operators checkpoint | .00130919 | 4/4 | 0/4 | 0 | 1,094/3,200 (34.1875%) |

The two new class gates pass; reconstruction remains red. Both new trainings
have zero decoder selection-rule violations and zero ownership conflicts over
1,200 backwards (20 active and 55 inactive parameters). **Exploration has not
made the greedy decoder infer the missing binary undo:** evaluation still
returns `hello`, `hello`, `loving`, `loving`. The mechanism is implemented;
useful operation inference and a complete read-back are not demonstrated.
Code geometry is diagnostic, never an added acceptance bar.

Complete observations, kept/greedy decoder stability per sentence, compose
stability, geometry and parameter ownership are in
[decoder summary](decoder-xor/summary.json) and
[checkpoint summary](checkpoint-xor/summary.json). Each summary links its
source and process record; the ownership folders retain raw events and every
saved gradient/displacement array. [decoder-source.zip](decoder-source.zip)
is the first stage; [checkpoint-source.zip](checkpoint-source.zip) is the
current stage. Their source manifests and each measurement's complete.json
confirm no source edits during training. The gate process exits 1 because the
reconstruction assertion fails; it is not rerun to select a better result.

## Operator changes so far

- `sum` is a mean with its corresponding exact witness inverse; `chunk` stays
  additive. Adverb composition is a repeatable gain with an exact inverse given
  its modifier, using the existing chart and guards.
- `non` clears the expressed pole and has no faithful inverse. Its code form
  withdraws the concept. Clause evidence exclusion is separate from negation
  and never changes trust.
- `exist`, `true` and `lookup` leave the executable catalogue. `what` reads
  complete matching conceptual frames with provenance; `quantize` and `arma`
  are thought-only. `generic` gets its own identity base. Absolute starts no
  longer require an existence wrapper. Relation thought results carry content
  and evidence rather than a scalar result.
- Rule metadata declares predicate identity, relation kind, scope, polarity,
  mode and head roles. Clause scope/journal, relative-rule detection and inverse
  dispatch use those properties in the changed paths. Implementation aliases,
  operand declarations and explicit write declarations have initial load
  checks. **This does not yet complete every name-based branch, exhaustive
  effects, face invariance or field eligibility.**
- The new exclusion exposed a real output packing defect: occupied roles after
  a silent slot were dropped. The output stack now compacts live roles before
  reversing their order. The failing probe and repair are saved. Conceptual
  distinction and generation gradients recover; two native numeric-output
  distinction tests remain red and are unwaived.

## Intersection question at this checkpoint

[Catalogue §4.2–4.3](../../specs/2026-09-29-operator-catalogue.md#43-what-follows-for-the-update)
asks for exact idempotence, silent coordinates that preserve the other word,
and the same behavior for evidence against. It also describes all ones as
`everything`. Zero as a neutral silent code requires meet(0, 1) = 1, whereas
all ones as an unconditional neutral universe requires meet(0, 1) = 0. Those
cannot both be numeric identities on the same untagged carrier.

At this checkpoint intersection is explicitly **provisional**: it is a signed minimum
with zero skipped. It satisfies the positive idempotence/silence examples but
does not settle the signed symmetry and universe representation. The old
soft-kernel assertion remains red. [Examples](intersection-question.json)
preserve the concrete outputs. The question initially proposed an explicit universe marker. On rereading,
§4.2 is a measurement of the old implementation, while §4.3 supplies the
decision: zero is silent. Subsequent work follows that decision, treats all
ones as an ordinary code, and does not assume a user answer. This checkpoint
precedes the completed operator-contract tests.

## Validation and complete ports

The initial default run completed 5,040 selected cases and saved 70 failures in
[default-01](default-01/result.json). Reproducing those selectors at published
HEAD in the same tree yielded 30 failures and 40 passes; the source was restored
exactly afterwards. This is a reproduction subset, not a historical full-sweep
pass or a training-gate campaign. [Comparison](failure-comparison.json) and
[restore check](baseline-restored.json) keep that distinction explicit.

Focused port work has 131/136 passing in `catalogue-ports-after`; two fixture
mistakes were repaired in the subsequent **71-pass** run
[catalogue-ports-final](probes/catalogue-ports-final/process.json). Earlier
query/controller coverage passed 177 cases. The 5,042-case second default run had 40 failures. Five relative-rule fixture
ports subsequently pass in a 47-case affected run. The complete default result
for the final frozen review source is in [default-03](default-03/result.json); its
machine-readable summary and remaining failures are in
[checkpoint-validation.json](checkpoint-validation.json). No failing gate is
silenced, skipped or re-barred.

[test-ports.json](test-ports.json) contains the **entire old and new contents**
of all 35 changed/new test and fixture files, including imports, decorators,
helpers and assertions. It is not a truncated diff. Renamed or retired contracts
remain recoverable in full. The substantive ports replace scalar/retired-query
expectations with owned content and provenance, replace additive sum with mean,
remove the existence wrapper from grammar fixtures, and exercise declared
operator properties. [Seed audit](seed-port-audit.json) finds zero changed seed
calls. [before.zip](before.zip), [checkpoint-source.zip](checkpoint-source.zip)
and [review-source.zip](review-source.zip) preserve the old and new source;
[review.patch](review.patch) is the tracked delta. The review snapshot adds only
two mechanical test-file edits after the checkpoint measurement;
[the bridge](measurement-source-bridge.json) verifies production code and both
gate tests are unchanged. New
files absent from git diff are included in the archive and test ports.

## Remaining authorized work

Finish intersection's decided signed-minimum/silent-coordinate tests; finish
all declared-operator effects and name-independent dispatch; implement compound
case selection and the decided verb-object subtyping/head behavior; retire the
unused binary SymbolizeLayer; reconcile equality's question identity with its
two closing part rows; finish remaining operator tests, documentation and the
operator diagram. The present metadata is not a substitute for those behaviors.

Then implement all of 6.8-1, including §§7–9: the typed bracket table and one
budget, open read, field-versus-symbol eligibility, word whole and numeric-run
handling, word/sentence expectation, migration of every ReadingAttention and
GlobalAttention capability, removal/rejection of retired knobs, genuine
word-level MM_xor convergence, paired thought/output-generation trials and
their audit. Preserve the accepted XOR_grammar baseline throughout. Run the
required affected checks and source-matched measurements, update documentation,
and stop for Claude's review before committing. Nothing in this checkpoint
waives or marks any of that work done.
