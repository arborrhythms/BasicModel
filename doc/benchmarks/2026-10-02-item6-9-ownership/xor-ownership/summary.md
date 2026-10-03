# Saved objective measurements

Complete raw observations are in `events.jsonl`; values below are medians conditional on a term being recorded, not new forwards. Each sample count and range is in `summary.json`. Missing terms are not imputed as observations.

## Configured priorities

These include disabled or unexercised objectives; the term table separately shows what was actually costed. Term purpose and ownership are specified in the main receipt and GradientFlow.

| Configuration field | Value |
|---|---|
| grammarLessonWeight | 1 |
| reconstructionScale | 0.1 |
| whatScale | 0.7 |
| whereScale | 0.2 |
| whenScale | 0.1 |
| TruthLoss | 0 |
| trainEmbedding | JOINT |
| embeddingScale | 0.05 |
| conceptualContextSituationWeight | 0 |
| conceptualContextExpectationWeight | 0 |
| sentenceExpectation | True |
| sentenceExpectationScope | structured |
| armaScale | 0 |
| expectationGain | 1 |
| expectationPolicyWeight | 0 |
| expectationQueryBudget | 64 |
| intraLossWeight | 0.1 |
| interLossWeight | 0.1 |
| interContrastiveWeight | 0 |
| selectedThoughtPolicyWeight | 0 |

## Term magnitudes and weights

| Scope | Term | Weight | Context | Trained | Raw | Baseline | Relative / penalty | Weighted |
|---|---|---|---|---|---|---|---|---|
| evaluation.batch | answer.what | 0.7 | 1.0 | True | 0.237026 | 0.5 | 0.474052 | 0.331836 |
| evaluation.batch | output | 1.0 | 1.0 | False | — | — | 0.165918 | 0.165918 |
| evaluation.batch | reconstruction | 1.0 | 1.0 | False | — | — | 1.06673 | 1.06673 |
| evaluation.batch | reconstruction.bytes | 0.1 | 1.0 | True | 1.06673 | 5.54518 | 0.19237 | 0.019237 |
| evaluation.batch | reconstruction_unavailable_sentences | 1.0 | 1.0 | False | — | — | 0 | 0 |
| evaluation.trial | reconstruction.bytes | 0.1 | 1.0 | True | 1.06673 | 5.54518 | 0.19237 | 0.019237 |
| train.batch | answer.what | 0.7 | 1.0 | True | 0.126373 | 0.5 | 0.252747 | 0.176923 |
| train.batch | output | 1.0 | 1.0 | False | — | — | 0.0884614 | 0.0884614 |
| train.batch | reconstruction | 1.0 | 1.0 | False | — | — | 0.586079 | 0.586079 |
| train.batch | reconstruction.bytes | 0.1 | 1.0 | True | 0.586079 | 5.54518 | 0.105692 | 0.0105692 |
| train.batch | reconstruction_unavailable_sentences | 1.0 | 1.0 | False | — | — | 0 | 0 |
| train.trial | answer.what | 0.7 | 1.0 | True | 0.250882 | 0.5 | 0.501763 | 0.351234 |
| train.trial | expectation.intra | 0.1 | 1.0 | True | 0.00126555 | 0.400051 | 0.00316347 | 0.000316347 |
| train.trial | reconstruction.bytes | 0.1 | 1.0 | True | 0.591229 | 5.54518 | 0.10662 | 0.010662 |

## Actual optimizer ownership

| Owner | Parameters | Elements | Reached parameters | Reached elements |
|---|---|---|---|---|
| reconstruction | 37 | 10647 | 5 | 612 |
| expectation | 30 | 22358 | 10 | 480 |
| output | 6 | 258 | 3 | 137 |

An empty writer list is inactive, not a second owner. Full parameter names and writer lists are in `ownership.json`.

## Selection audit

| Objective | Comparisons | Kept worse | Mean excess when worse | Maximum excess |
|---|---|---|---|---|
| reconstruction | 1600 | 14 | 0.0430936 | 0.06765 |
| expectation | 1600 | 253 | 0.00371125 | 0.15657 |
| supplied_answer | 1600 | 23 | 0.0871493 | 0.408308 |
| total | 1600 | 14 | 0.102642 | 0.348355 |

The rule constrains accepting explore; greedy can remain when explore improves only one side of the condition. Thus a kept-worse count alone is not a rule violation. Explore acceptance audit: {'explore_kept': 870, 'reconstruction_violations': 0, 'total_violations': 0}.

## Activated candidates

| Scope | Own-word occurrences | With activated competitor | Activated outranks own |
|---|---|---|---|
| train | 6400 | 0 | 0 |
| evaluation | 8 | 0 | 0 |

## Same-state gradient snapshots

Norms are restricted to permitted optimizer writers. `None` cosines mean a zero/absent vector, not an observed angle. These diagnostic groups can overlap: an answer-only `perceptualSpace.synthesis_layer` adapter appears under both the perception container and the generate/reader groups. That does not make an input-perception parameter answer-owned; the per-parameter names and actual owner are recorded separately.

### trial, exploit, pair 0

Parameter version digest: `027f0265c79a07761255988a91975f6d0ab0cf8324ac606a454e93ae0c263f1b`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 0 | 0 | 0 | — | — | — |
| codes | 0.00199975 | 0 | 0 | — | — | — |
| chooser | 0.00125673 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.094504 | — | — | — |
| expectation_predictor | 0 | 0.222624 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

### trial, explore, pair 0

Parameter version digest: `027f0265c79a07761255988a91975f6d0ab0cf8324ac606a454e93ae0c263f1b`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 0 | 0 | 0 | — | — | — |
| codes | 0.00206503 | 0 | 0 | — | — | — |
| chooser | 0.000746613 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.29375 | — | — | — |
| expectation_predictor | 0 | 0.160032 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

### batch, None, pair 1

Parameter version digest: `ef096b3c615f9bd07b47b5546ae2edb5577c87253ca797297f3725b8f7eafff7`. RNG unchanged: True.

| Group | R norm | E norm | A norm | R:E cosine | R:A cosine | E:A cosine |
|---|---|---|---|---|---|---|
| perception | 0 | 0 | 0 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.618371 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

## Endpoint costs

`outcome.json` retains the last training trials and evaluation batch; `summary.json` preserves both. The answer-free total below omits the answer at the same recorded state; it is not a second training arm.

| Phase | R | E | A | R+E (without A) | Trained total |
|---|---|---|---|---|---|
| last training | 0.0110131 | — | 0.189455 | 0.0110131 | 0.200468 |
| last evaluation | 0.019237 | — | 0.331836 | 0.019237 | 0.351073 |

The final column is the observer’s optional evaluation answer cost for each row, not an answer prediction. Training rows have no such extra evaluation read.

| Phase | Trial | R | E | A | Selection total R+A | R+E without A | Observer evaluation answer costs per row |
|---|---|---|---|---|---|---|---|
| training | exploit | 0.0110131 | 0.000223376 | 0.358657 | 0.36967 | 0.0112364 | [None, None, None, None] |
| training | explore | 0.0160023 | 0.000223376 | 0.240408 | 0.25641 | 0.0162256 | [None, None, None, None] |
| evaluation | exploit | 0.019237 | — | — | 0.019237 | 0.019237 | [0.505809485912323, 0.2016155868768692, 0.3741532862186432, 0.24576640129089355] |
