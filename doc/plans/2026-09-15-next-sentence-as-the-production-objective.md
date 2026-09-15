# Next-sentence prediction as the production answer objective

Status: PLAN, written 2026-09-15 at basicmodel `fa5ee93` for execution in
a fresh session.

Alec (2026-09-15): "Production should train by reconstruction and next
sentence prediction whenever the next sentence is related to the last
(much of the current fineweb data set). That will train reconstruction,
prediction (when the text is a monologue), and Q/A (when the text is a
dialogue). That change should dovetail with the inter sentence
prediction module."

## 1. The finding

Almost all of this already exists and is correctly built. One gate,
landed yesterday in Codex round 5, turns it off in production.

The mechanism, end to end, on `data/BasicModel.xml` + FineWeb:

| step | where | state |
|---|---|---|
| FineWeb rows are one sentence each, appended in document order with `document` / `sentence` addresses | `data.py` FineWeb loader | works |
| a packed batch row's question anchors on the LAST sentence of its pack (`row[-1]`), so `future(+1)` lands after the pack, never inside it | `_questions_for_batch` | works |
| a quarter of training batches ask past / future / inference at distance 1 | `<whatCurriculum>full</whatCurriculum>`, ratio 0.25 | works |
| `Data.what(What.future(k, 1))` returns the actual next sentence's TEXT, and returns "unavailable" when the target leaves the document | `Data.what`, `_same_what_document` | works, and is exactly Alec's relatedness test |
| the answer is realised to a surface and scored against that text in input-event space | `_answer_surface_from_percepts`, `_reverse_event_loss` | works |
| the answer target is admitted in TRAINING | `_what_answer_target(..., supervised_only=...)` | **blocked** |
| a FUTURE question resolves to a conceptual answer | `_resolve_answer` | **absent** |

The block: round 5 set `supervised_only = train and answer_synthesis`,
and admits a row only when `data.has_supervised_outputs` AND the
question's relation is `SUPERVISED`. FineWeb sets
`has_supervised_outputs = False` (no labels), and the curriculum asks
`FUTURE` / `PAST`. So in production every answer row is masked out in
training, and `_output_action_credit` is gated the same way. That is the
"production trains no answer modules" note in the round-5 review
([plan section 7](2026-09-14-answer-path-ownership-and-training.md)).

Round 5 was right that the model's own state must never become its own
target, and right that a PRESENT answer must not stand in for answer
supervision. It drew the line in the wrong place: at "explicitly supplied
label" instead of at "target independent of the model".

## 2. The correct rule

A target may train the answer path when it is **ground truth the corpus
supplies and the model has not seen as its own input**:

| relation | target | verdict |
|---|---|---|
| `SUPERVISED` | the supplied label | trains (unchanged) |
| `FUTURE` +k, inside the document | the true next sentence's text | **trains** — this is Alec's production objective |
| `PAST` -k, inside the document | the true previous sentence's text | **trains**, but it is MEMORY, not prediction (section 5, question 1) |
| `PRESENT` | the current input sentence | never trains the answer path: that is reconstruction's objective, and letting the answer path learn identity on it is the failure round 5 correctly blocked |
| `INFERENCE` | none | no target (unchanged) |
| any relation whose target crosses a document boundary | `Data.what` already returns unavailable | masked (unchanged) |

`WhatAnswer` already carries `provenance="data"` and `source_where`, so
the test is `answer.available and answer.provenance == "data" and
answer.source_where != question.where`. That one expression replaces
`supervised_only` and states the rule directly: a real corpus row the
model was not shown.

Monologue then trains prediction and dialogue trains question answering
through the same path, with no special casing, because in a dialogue the
next sentence IS the answer to the last. That is Alec's point.

## 3. The dovetail with inter-sentence prediction

A FUTURE question currently has no conceptual answer: the indexed
resolution sets `entry, source = None, "prediction"` and reports the row
unresolved, because a predicted sentence has no row program to replay.
That is correct — a prediction is not a replay of anything — and it is
the standing open item ("a concept-level predictor for future answers").

The inter-sentence module is that predictor.
`WhatInteractionMemory.predict_next_end_state` (`bin/Layers.py:9996`)
already predicts the NEXT sentence's end state from the chain of previous
end states, at concept width, in the same `[depth, D]` three-slot form
the answer path materialises. So:

* **Idea level.** `observe_stm_end_state` scores the predicted root
  against the arriving end state and accumulates `L_inter`, consumed by
  `_discourse_inter_loss`. This trains the predictor cheaply at every
  sentence boundary, with no realisation.
* **Surface level.** The same predicted end state becomes the FUTURE
  question's conceptual answer, is conditioned once, unfolds through the
  generate walk, realises through the shared reverse chain, and is scored
  against the true next sentence's text.

Two levels of one objective on one predictor: the idea-level term shapes
the prediction, the surface-level term makes it expressible. The answer
path needs no second predictor, and the discourse module gains a
realisation it never had.

Consequence for the answer path: `_resolve_answer` must accept a
conceptual answer **without** a program. Today `AnswerProgram` carries
rows, leaves, actions and end state, and `_materialize_entries` replays
the actions. A predicted idea has only an end state. Extend
`AnswerProgram` so a record may carry `end_state` with empty
`rows`/`actions` (a predicted idea, nothing to replay), and have
`_materialize_entries` use the end state directly for those rows. The
frozen-record ownership property from round 5 must hold unchanged: the
prediction is captured at resolution and later staging cannot alter it.

## 4. The work

Production config (`data/BasicModel.xml`) currently ships
`<sentencePrediction>false</sentencePrediction>` and `<armaScale>0.0`,
so the predictor does not exist and its loss is off. Turning them on is
part of this change, with a weight for the inter term.

1. **Replace the `supervised_only` gate with the section-2 rule** in
   `_what_answer_target` and `_output_action_credit`. Tests: a FineWeb-like
   split with no supplied labels trains the answer path on a future
   question whose target is in the document; the same question at a
   document boundary is masked; a PRESENT question never produces an
   answer gradient or optimizer step, including after an Adam step.
2. **Resolve FUTURE conceptually** from `predict_next_end_state`:
   a program-free `AnswerProgram` carrying the predicted end state;
   `resolved=True` when the chain is warm, unresolved when cold. Tests:
   a future answer is resolved and full width; it is frozen against later
   staging and memory advance; a cold chain stays unresolved and trains
   nothing.
3. **Turn on the inter-sentence predictor in production** and give
   `L_inter` a weight; confirm `_discourse_inter_loss` is in `totalLoss`
   and its parameters are in the optimizer (the round-5 lesson:
   `record_loss` is report-only, and a module on no `params` list is
   never stepped). Test: one training batch changes the predictor's
   weights; the two levels agree in sign on a synthetic monologue.
4. **Measure on the real corpus.** A short FineWeb run reporting
   reconstruction and answer costs per batch, the fraction of curriculum
   rows whose future target is available (the document-boundary rate),
   and throughput against the current production number. This is the gate
   that says the objective actually trains.

## 5. Questions for Alec

1. **Does PAST train the same modules as FUTURE?** A past answer is
   materialised from the stored program of the sentence that actually
   occurred, so realising it and scoring it against that sentence's text
   trains memory plus realisation, not prediction. Legitimate, but a
   different skill. Recommend: yes, train it, at the same weight, and
   report the two families separately (`what_report` already splits by
   family).
2. **Curriculum ratio.** A quarter of batches ask temporal questions
   today. If next-sentence prediction is a primary production objective
   rather than a curriculum trial, the ratio should probably rise.
   Recommend: raise to 0.5 after item 4's measurement, not before.
3. **Held-out honesty.** Past and future are currently evaluation
   metrics; once they are training objectives the same numbers on the
   training split stop being evidence. Recommend: report them on the
   validation split in the epoch line.
4. **`outputInLoop` in production.** Realising a predicted idea needs the
   generate walk; without it the answer goes through the dedicated
   synthesis operators instead. Both paths run at native widths since
   round 5. Recommend: leave the walk off for item 4's measurement, then
   compare.
