# Saved objective measurements

Complete raw observations are in `events.jsonl`; values below are medians conditional on a term being recorded, not new forwards. Each sample count and range is in `summary.json`. Missing terms are not imputed as observations.

## Configured priorities

These include disabled or unexercised objectives; the term table separately shows what was actually costed. Term purpose and ownership are specified in the main receipt and GradientFlow.

| Configuration field | Value |
|---|---|
| grammarLessonWeight | 1 |
| reconstructionScale | 0.5 |
| whatScale | 0.7 |
| whereScale | 0.2 |
| whenScale | 0.1 |
| TruthLoss | 0 |
| trainEmbedding | NONE |
| embeddingScale | 0.25 |
| conceptualContextSituationWeight | 0 |
| conceptualContextExpectationWeight | 0 |
| sentenceExpectation | True |
| sentenceExpectationScope | structured |
| armaScale | 0 |
| expectationGain | 1 |
| expectationPolicyWeight | 0 |
| expectationQueryBudget | 64 |
| intraLossWeight | 0 |
| interLossWeight | 0.1 |
| interContrastiveWeight | 0 |
| selectedThoughtPolicyWeight | 0 |
| leafDistillWeight | 0 |

## Term magnitudes and weights

| Scope | Term | Weight | Context | Trained | Raw | Baseline | Relative / penalty | Weighted |
|---|---|---|---|---|---|---|---|---|
| evaluation.batch | answer.what | 0.7 | 1.0 | True | 0.372706 | 0.0625 | 5.96329 | 4.17431 |
| evaluation.batch | output | 1.0 | 1.0 | False | — | — | 0.260894 | 0.260894 |
| evaluation.batch | reconstruction | 1.0 | 1.0 | False | — | — | 1.72477 | 1.72477 |
| evaluation.batch | reconstruction.bytes | 0.5 | 1.0 | True | 1.72477 | 5.54518 | 0.311039 | 0.15552 |
| evaluation.batch | reconstruction_unavailable_sentences | 1.0 | 1.0 | False | — | — | 0 | 0 |
| evaluation.trial | reconstruction.bytes | 0.5 | 1.0 | True | 1.72477 | 5.54518 | 0.311039 | 0.15552 |
| train.batch | answer.what | 0.7 | 1.0 | True | 0.603444 | 0.0625 | 9.6551 | 6.75857 |
| train.batch | concept_readout_l1 | 0.01 | 1.0 | False | — | — | 0 | 0 |
| train.batch | output | 1.0 | 1.0 | False | — | — | 0.42241 | 0.42241 |
| train.batch | reconstruction | 1.0 | 1.0 | False | — | — | 0.16414 | 0.16414 |
| train.batch | reconstruction.bytes | 0.5 | 1.0 | True | 0.16414 | 5.54518 | 0.0296004 | 0.0148002 |
| train.batch | reconstruction_unavailable_sentences | 1.0 | 1.0 | False | — | — | 0 | 0 |
| train.trial | answer.what | 0.7 | 1.0 | True | 0.147728 | 0.0625 | 2.36365 | 1.65456 |
| train.trial | concept_readout_l1 | 0.01 | 1.0 | False | — | — | 1.08333 | 0.0108333 |
| train.trial | reconstruction.bytes | 0.5 | 1.0 | True | 0.180297 | 5.54518 | 0.0325142 | 0.0162571 |

## Actual optimizer ownership

| Owner | Parameters | Elements | Reached parameters | Reached elements |
|---|---|---|---|---|
| reconstruction | 82 | 27503612 | 30 | 18896746 |
| expectation | 50 | 54322180 | 0 | 0 |
| output | 19 | 25078641 | 14 | 24806176 |

An empty writer list is inactive, not a second owner. Full parameter names and writer lists are in `ownership.json`.

## Selection audit

| Objective | Comparisons | Kept worse | Mean excess when worse | Maximum excess |
|---|---|---|---|---|
| reconstruction | 28 | 0 | — | — |
| expectation | 0 | 0 | — | — |
| supplied_answer | 28 | 0 | — | — |
| total | 28 | 0 | — | — |

The rule constrains accepting explore; greedy can remain when explore improves only one side of the condition. Thus a kept-worse count alone is not a rule violation. Explore acceptance audit: {'explore_kept': 0, 'reconstruction_violations': 0, 'total_violations': 0}.

## Activated candidates

| Scope | Own-word occurrences | With activated competitor | Activated outranks own |
|---|---|---|---|
| train | 168 | 0 | 0 |
| evaluation | 48 | 0 | 0 |

## Same-state gradient snapshots

Norms are restricted to permitted optimizer writers. `None` cosines mean a zero/absent vector, not an observed angle. These diagnostic groups can overlap: an answer-only `perceptualSpace.synthesis_layer` adapter appears under both the perception container and the generate/reader groups. That does not make an input-perception parameter answer-owned; the per-parameter names and actual owner are recorded separately.

### trial, exploit, pair 0

Parameter version digest: `d2917b28b141eb4899da96081225fd461564dd70481af6723ad2146dd88fde28`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 4.81248e-25 | 0 | 0 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0.000111571 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0.024799 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 66.3896 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

### trial, explore, pair 0

Parameter version digest: `d2917b28b141eb4899da96081225fd461564dd70481af6723ad2146dd88fde28`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 5.73234e-25 | 0 | 0 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0.000100825 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0.0248429 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 66.2762 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

### batch, None, pair 1

Parameter version digest: `61b4371c5a4a05f6090b2c30110ffb2afc638c5959d89cfd50521112bb04f4cd`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 0 | 0 | 11.4103 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 25.4558 | — | — | — |
| reading_map | 0 | 0 | 154.4 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

## Endpoint costs

`outcome.json` retains the last training trials and evaluation batch; `summary.json` preserves both. The answer-free total below omits the answer at the same recorded state; it is not a second training arm.

| Phase | R | E | A | R+E (without A) | Trained total |
|---|---|---|---|---|---|
| last training | 0.0148002 | — | 6.75857 | 0.0148002 | 6.77337 |
| last evaluation | 0.15552 | — | 4.17431 | 0.15552 | 4.32983 |

The final column is the observer’s optional evaluation answer cost for each row, not an answer prediction. Training rows have no such extra evaluation read.

| Phase | Trial | R | E | A | Selection total R+A | R+E without A | Observer evaluation answer costs per row |
|---|---|---|---|---|---|---|---|
| training | exploit | 0.0148002 | — | 1.65384 | 1.66864 | 0.0148002 | [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None] |
| training | explore | 0.017714 | — | 1.65528 | 1.67299 | 0.017714 | [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None] |
| evaluation | exploit | 0.15552 | — | — | 0.15552 | 0.15552 | [2.0988380908966064, 1.3033076524734497, 1.1166216135025024, 1.815488576889038, 1.8150137662887573, 1.852658748626709, 2.090649366378784, 2.0901734828948975, 2.088834047317505, 1.6924362182617188, 1.331908941268921, 1.3830397129058838, 2.6091508865356445, 2.0922365188598633, 2.0940585136413574, 1.8491687774658203] |
