# Next-sentence prediction as the production objective

Status: PLAN, revised 2026-09-15 (basicmodel `bad0ae6` and after) for
execution in a fresh session. Alec's decision: prediction is done over
the encoded sentences as they are stored in LTM, concepts to concepts.
The surface-level path this plan first proposed is future work.

Alec (2026-09-15): "Production should train by reconstruction and next
sentence prediction whenever the next sentence is related to the last
(much of the current fineweb data set). That will train reconstruction,
prediction (when the text is a monologue), and Q/A (when the text is a
dialogue). That change should dovetail with the inter sentence
prediction module." And, on the level: "this is concepts to concepts
prediction; it is possible that we want words to words and percepts to
percepts (visualization), but that is future work."

## 1. The objective, and where it lives

The objective is the inter-sentence prediction module, which exists and
is off in production.

| step | where |
|---|---|
| every sentence's end state (the three STM slots, newest at 0, and its depth) is appended to the row's LTM chain | `WhatInteractionMemory.observe_stm_end_state`, `bin/Layers.py` |
| the next end state's root is predicted from the last K stored end states (`K = interChainWindow`, at most 8) | `predict_next_end_state`, `bin/Layers.py` |
| when the next sentence's end state arrives, the prediction is scored against its root (MSE) and accumulated as `L_inter`; the target is detached | `observe_stm_end_state`, `_accumulate_inter_loss` |
| an optional InfoNCE term ranks the true next root above the chain's past roots | `interContrastiveWeight` (default 0), `interContrastiveTemp` |
| the per-sentence mean enters the training total with `interLossWeight` (default 0.1) and is reported as `inter` | `_discourse_inter_loss`, `runBatch` |
| the predictor exists only under `<sentencePrediction>`; its parameters are appended to SymbolSpace's explicit `params`, so the optimizer sees them | `SymbolSubSpace.__init__`, `bin/Language.py` about line 11001 to 11034 |

The relatedness test at this level is the document boundary. A hard
end of document resets the row's chain, the pending prediction and the
loss accumulator (`WhatInteractionMemory.Reset`), so no prediction is
scored across documents. Within a document the chain is consecutive
sentences by construction (row `b` at slot `t` is followed only by row
`b` at slot `t+1`). This is Alec's "whenever the next sentence is
related to the last" with no extra machinery.

Monologue trains prediction and dialogue trains question answering
through the same term, because in a dialogue the next sentence's end
state is the answer's idea.

## 2. What the current production configuration does with it

`data/BasicModel.xml` ships `<sentencePrediction>false</sentencePrediction>`,
so the predictor object is not built, and `<armaScale>0.0</armaScale>`
(the separate, older ARMA term on pooled sentence reps, `_discourse_arma_loss`;
it stays off). `interLossWeight` is not set (default 0.1).

Reconstruction (`<reconstructInLoop>`) is the other production objective
and stays as it is.

## 3. Turning it on

1. In `data/BasicModel.xml` `<training>`: `<sentencePrediction>true</sentencePrediction>`
   and an explicit `<interLossWeight>` (start at the default 0.1). Leave
   `armaScale` at 0 and `interContrastiveWeight` at 0 for the first
   measurement. `ltmCapacity` under `<SymbolSpace>` bounds the chain;
   the predictor's window is at most 8.
2. Prove the term trains, not merely reports (the round-5 lesson:
   `record_loss` is report-only and a module on no `params` list is never
   stepped). Test: one training batch on a two-sentence document changes
   the predictor's parameters; with `interLossWeight` 0 they do not
   change.
3. Prove the boundary. Test: two documents in one row, the second
   sentence of each is scored, the first sentence of the second document
   is not (the reset cleared the pending prediction), and nothing is
   scored in evaluation.
4. Prove it learns. Test: on a synthetic monologue whose sentences
   follow a fixed pattern, `L_inter` on held-out documents falls over a
   short run; on shuffled sentences it does not.
5. Measure on the real corpus: a short FineWeb run reporting
   reconstruction and `inter` per batch, the fraction of sentence
   boundaries that were document boundaries, and throughput against the
   current production number.

## 4. The one decision: does prediction reach the encoder?

Today both the target and the context are detached: the chain stores
detached end states, and the loss trains the predictor head only. So
prediction is a probe on top of reconstruction-trained encodings; the
fold operators and everything in `forward()` learn nothing from it.

If prediction is to shape the representations (predictive coding), the
context must stay live for the current brick's sentences: gradient
through the previous end states into the folds, stop-gradient on the
target (the JEPA arrangement). Reconstruction already anchors the end
states, so collapse is guarded either way. Recommendation: run section 3
with the head-only objective first and get the measurement, then try
the live context as a separate commit with the same tests, and compare
`inter` on held-out documents and reconstruction cost. Alec decides
after seeing both.

## 5. What this does not train, on purpose

The generate chooser, the question conditioner and the synthesis
operators learn only from supplied answers (round 5). Reconstruction
trains the tied inverses they share, so a predicted idea can still be
realised at inference. Past and future realised answers stay evaluation
metrics. The answer path's training gate stays as round 5 left it.

## 6. Future work (recorded, not planned)

* **Words to words.** Prediction at the surface level: realise the
  predicted idea through the answer path and score it against the true
  next sentence's text. Everything needed exists (FineWeb sentence-ordered
  addresses, pack-anchored questions, the document-scoped relatedness
  test in `Data.what`, surface scoring in input-event space, a quarter of
  batches asking temporal questions), and one rule change would enable
  it: a target may train the answer path when it is corpus ground truth
  the model did not see as input (`answer.available and
  answer.provenance == "data" and answer.source_where != question.where`),
  never a PRESENT target, which is reconstruction's. This would need a
  program-free `AnswerProgram` (an end state with nothing to replay) so
  a predicted idea can be resolved. Not now.
* **Percepts to percepts.** Prediction at the perceptual level, the
  visualization case. Not designed.

## 7. Questions for Alec

1. Section 4, after the measurement: head-only, or live context.
2. Whether `inter` on the validation split goes into the epoch line now
   that it is a training objective on the training split.
