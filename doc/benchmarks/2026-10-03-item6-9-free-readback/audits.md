# Ownership, costs and geometry

Derived from the two saved first measurements. No model was rerun. The XOR audit reuses class run 1; the native audit reuses the production stage-1 test.

Gradient tables show raw graph reach before the owner restriction. Structural groups overlap (for example a perceptual synthesis module also belongs to generation). The per-parameter ownership files show the actual permitted/applied writer. A cosine is undefined when either gradient has zero norm. These zeros do not establish how an inactive objective would behave.

Term statistics are over logged event-level means, not a pooled estimator over unequal batches. Trial and batch totals are separately reported locations; they must not be added again. The endpoint is the saved evaluation batch after training; the last training batch is also retained in the JSON. Penalties and reporting/count entries are listed separately from trained relative errors by their `kind` and `trained` fields.

Term definitions, targets, uninformed baselines, purpose and owner are in [GradientFlow](../../GradientFlow.md). Reconstruction has only the free byte term, divided by log(256) and weighted by reconstructionScale. The supplied squared-error answer uses the detached target mean-square norm. Zero-target squared terms are penalties. concept_readout_l1 retains its own coefficient as a proximal penalty rather than a normalized objective.

## XOR_grammar

Declared parameter records: 73; active: 18; inactive: 55; conflicts: **0**; training backward calls: 1200.

[Every declared and observed writer](xor-ownership/ownership.json); [complete per-parameter gradients](xor-ownership/gradients.json); [configuration record](xor-ownership/configured-training.json).

Configured priorities can be inactive in the observed data. The term table below shows what was actually registered, in each location; an absent term is not a measured zero.

| Setting | Value | Purpose |
|---|---:|---|
| `reconstructionScale` | 0.1 | Weight of the free read-back relative error |
| `whatScale` | 0.7 | Supplied-answer what band |
| `whereScale` | 0.2 | Supplied-answer where band, when present |
| `whenScale` | 0.1 | Supplied-answer when band, when present |
| `intraLossWeight` | 0.1 | Within-sentence prediction |
| `interLossWeight` | 0.1 | Between-sentence prediction |
| `interContrastiveWeight` | 0 | Between-sentence contrastive prediction |
| `armaScale` | 0 | ARMA prediction |
| `grammarLessonWeight` | 1 | Annotated compose/generate chooser lessons |
| `embeddingScale` | 0.05 | Lexical embedding targets, when present |
| `expectationPolicyWeight` | 0 | Retired expectation policy route; stays zero |
| `selectedThoughtPolicyWeight` | 0 | Answer-owned selected-thought policy |
| `TruthLoss` | 0 | Stored-truth penalty strength |

### First-state gradients

| State | Group | Reconstruction norm | Expectation norm | Answer norm | R/E cosine | R/A cosine | E/A cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| trial.exploit | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | codes | 1.622099e-10 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 1.728268e-12 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 0.370079 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0.1987541 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 2.068116e-11 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 4.718643e-13 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 0.1744529 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0.1561337 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 0.3645124 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `1e45c81225a2d0a4b93cf8dc81268ae5529a03d3fe362f327167903b961bcac4`, `1e45c81225a2d0a4b93cf8dc81268ae5529a03d3fe362f327167903b961bcac4`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 1600; explore kept: 245; selection-rule violations: 0.

The kept trial has the lower reconstruction; equality keeps greedy. Answer and expectation can be worse because neither enters the comparison.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 1600 | 0 | undefined | undefined |
| expectation | 1600 | 244 | 0.007273517 | 0.1321068 |
| supplied_answer | 1600 | 240 | 0.02301091 | 0.5999289 |
| total | 1600 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 2.133405e-12 [2.133405e-12, 2.133405e-12] | 1.493383e-12 [1.493383e-12, 1.493383e-12] | 0.5 [0.5, 0.5] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 7.466916e-13 [7.466916e-13, 7.466916e-13] | 7.466916e-13 [7.466916e-13, 7.466916e-13] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 5.468639 [5.468639, 5.468639] | 5.468639 [5.468639, 5.468639] | — |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.9861974 [0.9861974, 0.9861974] | 0.09861974 [0.09861974, 0.09861974] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 5.0399e-05 [6.057377e-13, 0.4962853] | 3.52793e-05 [4.240164e-13, 0.3473997] | 0.5 [0.5, 0.5] |
| batch_end.train | `output` | penalty / [False] | [1.0] | 1.763965e-05 [2.120082e-13, 0.1736998] | 1.763965e-05 [2.120082e-13, 0.1736998] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 4.317347 [1.439116, 5.468639] | 4.317347 [1.439116, 5.468639] | — |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.7785769 [0.2595256, 0.9861974] | 0.07785769 [0.02595256, 0.09861974] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.9861974 [0.9861974, 0.9861974] | 0.09861974 [0.09861974, 0.09861974] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 4.165066e-05 [1.172396e-12, 0.5079206] | 2.915546e-05 [8.206769e-13, 0.3555444] | 0.5 [0.5, 0.5] |
| trial.exploit.train | `expectation.intra` | relative / [True] | [0.1] | 0.001231272 [0.000269031, 1.762468] | 0.0001231272 [2.69031e-05, 0.1762468] | 0.4001582 [0.4001582, 0.4001583] |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.9861974 [0.9861974, 0.9861974] | 0.09861974 [0.09861974, 0.09861974] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 0.8907125 [0.2620222, 1.716992] | 0.6234987 [0.1834155, 1.201895] | 0.5 [0.5, 0.5] |
| trial.explore.train | `expectation.intra` | relative / [True] | [0.1] | 0.002720873 [0.000269031, 1.560197] | 0.0002720873 [2.69031e-05, 0.1560197] | 0.4001582 [0.4001582, 0.4001583] |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 1.197525 [0.4671461, 1.824095] | 0.1197525 [0.04671461, 0.1824095] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.0986197367310524, "expectation": null, "output": 1.493383216567834e-12}`.
With the answer: 0.09861974; without it: 0.09861974.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R |
|---|---:|---:|---:|---:|
| exploit | 0.09861974 | 0.0002787705 | 9.412915e-13 | 0.09861974 |
| explore | 0.1512664 | 0.0004419347 | 0.3046382 | 0.1512664 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 6 × 10 | 1 / 1 / 1 | 0.1027022 / 0.09555364 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_start | 1 | 6 × 10 | 0.9999999 / 1 / 1 | -0.01280249 / 0.07961137 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 0 | 6 × 10 | 1 / 1 / 1 | 0.1027022 / 0.09555364 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 1 | 6 × 10 | 0.9999999 / 1 / 1 | -0.01280249 / 0.07961137 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

geometry_start roots: shape [4, 10]; singular values [1.1999900341033936, 1.134284496307373, 0.866492748260498, 0.7229194641113281]; centered singular values [1.195438265800476, 0.8848400115966797, 0.8404192328453064, 4.031085509836885e-08].


geometry_end roots: shape [4, 10]; singular values [1.1999900341033936, 1.134284496307373, 0.866492748260498, 0.7229194641113281]; centered singular values [1.195438265800476, 0.8848400115966797, 0.8404192328453064, 4.031085509836885e-08].


### Gradient displacement and derivation stability

Every observed optimizer step has coordinate arrays under `displacements/`; events map each array to its parameter and sparse row ids. The table aggregates step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later step differ from the negative current gradient.

| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |
|---|---:|---:|---:|---:|---:|
| `conceptualSpaces.1.layers.3.W` | 800 | 0 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.stop_anchor` | 800 | 0 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.reduce_anchor` | 800 | 0 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.apply_anchor` | 800 | 0 | 0 | 0 | undefined |

| Sentence word rows | Observed epochs | Modal fraction | Distinct derivations |
|---|---:|---:|---:|
| [0, 1] | 400 | 0.7075 | 3 |
| [0, 2] | 400 | 1 | 1 |
| [4, 1] | 400 | 1 | 1 |
| [4, 2] | 400 | 0.68 | 2 |

Native training has one configured epoch. Its modal fraction is descriptive, not evidence of stability across repeated epochs. If a native sentence repeats within an epoch, its last kept derivation is that epoch’s sample; all trial events remain saved.

Across recorded training trials and evaluation: 6408 word occurrences, 0 with an activated candidate, 0 outranked by an activated candidate. A zero candidate count supplies no evidence about competition with activated words.

Full pairwise tensors and row IDs are beside [xor-ownership/geometry-end.json](xor-ownership/geometry-end.json). Native reserve-wide pair moments cover all row pairs exactly; full matrices cover all observed primed rows. A missing cluster-size buffer in the indexed-only native allocation is reported as absent, not as a measured vector of zeroes.

## BasicModel_answers_tied_benchmark

Declared parameter records: 152; active: 41; inactive: 111; conflicts: **0**; training backward calls: 3.

[Every declared and observed writer](native-stage1/BasicModel_answers_tied_benchmark-ownership/ownership.json); [complete per-parameter gradients](native-stage1/BasicModel_answers_tied_benchmark-ownership/gradients.json); [configuration record](native-stage1/BasicModel_answers_tied_benchmark-ownership/plan.json).

Configured priorities can be inactive in the observed data. The term table below shows what was actually registered, in each location; an absent term is not a measured zero.

| Setting | Value | Purpose |
|---|---:|---|
| `reconstructionScale` | 0.5 | Weight of the free read-back relative error |
| `whatScale` | 0.7 | Supplied-answer what band |
| `whereScale` | 0.2 | Supplied-answer where band, when present |
| `whenScale` | 0.1 | Supplied-answer when band, when present |
| `intraLossWeight` | 0 | Within-sentence prediction |
| `interLossWeight` | 0.1 | Between-sentence prediction |
| `interContrastiveWeight` | 0 | Between-sentence contrastive prediction |
| `armaScale` | 0 | ARMA prediction |
| `grammarLessonWeight` | 1 | Annotated compose/generate chooser lessons |
| `embeddingScale` | 0.25 | Lexical embedding targets, when present |
| `expectationPolicyWeight` | 0 | Retired expectation policy route; stays zero |
| `selectedThoughtPolicyWeight` | 0 | Answer-owned selected-thought policy |
| `TruthLoss` | 0 | Stored-truth penalty strength |

### First-state gradients

| State | Group | Reconstruction norm | Expectation norm | Answer norm | R/E cosine | R/A cosine | E/A cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| trial.exploit | perception | 1.21565e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | codes | 0.7205128 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 0.0003133089 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 1.29677e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 22.19218 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 5.9991e-25 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 0.6468823 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 0.0003264112 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 7.841964e-25 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 21.59729 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 5.608893 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 12.47234 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 53.14097 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `a44e9584b494e47d16e2f482e8f8e629368a8b4a335f54516c3dd4c19309ca48`, `a44e9584b494e47d16e2f482e8f8e629368a8b4a335f54516c3dd4c19309ca48`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 28; explore kept: 2; selection-rule violations: 0.

The kept trial has the lower reconstruction; equality keeps greedy. Answer and expectation can be worse because neither enters the comparison.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 28 | 0 | undefined | undefined |
| expectation | 0 | 0 | undefined | undefined |
| supplied_answer | 28 | 3 | 1.253819 | 2.890855 |
| total | 28 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 6.042614 [6.042614, 6.042614] | 4.22983 [4.22983, 4.22983] | 0.0625 [0.0625, 0.0625] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.2643644 [0.2643644, 0.2643644] | 0.2643644 [0.2643644, 0.2643644] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 9.952058 [9.952058, 9.952058] | 9.952058 [9.952058, 9.952058] | — |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.794723 [1.794723, 1.794723] | 0.8973615 [0.8973615, 0.8973615] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 4.797213 [4.797213, 4.797213] | 3.358049 [3.358049, 3.358049] | 0.0625 [0.0625, 0.0625] |
| batch_end.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083331 [1.083331, 1.083331] | 0.01083331 [0.01083331, 0.01083331] | — |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.2098781 [0.2098781, 0.2098781] | 0.2098781 [0.2098781, 0.2098781] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 10.12421 [10.12421, 10.12421] | 10.12421 [10.12421, 10.12421] | — |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.825769 [1.825769, 1.825769] | 0.9128843 [0.9128843, 0.9128843] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.794723 [1.794723, 1.794723] | 0.8973616 [0.8973616, 0.8973616] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 1.92219 [1.92219, 1.92219] | 1.345533 [1.345533, 1.345533] | 0.0625 [0.0625, 0.0625] |
| trial.exploit.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.847911 [1.847911, 1.847911] | 0.9239555 [0.9239555, 0.9239555] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 1.988221 [1.988221, 1.988221] | 1.391755 [1.391755, 1.391755] | 0.0625 [0.0625, 0.0625] |
| trial.explore.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.900338 [1.900338, 1.900338] | 0.9501689 [0.9501689, 0.9501689] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.8973615169525146, "expectation": null, "output": 4.229829788208008}`.
With the answer: 5.127191; without it: 0.8973615.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R |
|---|---:|---:|---:|---:|
| exploit | 0.9239555 | undefined | 1.345533 | 0.9239555 |
| explore | 0.9501688 | undefined | 1.391755 | 0.9501688 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 65536 × 1032 | 0.9999996 / 1 / 1 | 1.178209e-08 / 0.0009690098 | False | None |
| geometry_end | 0 | 65536 × 1032 | 0.9999932 / 1 / 1.000095 | 1.177732e-08 / 0.0009690098 | False | None |

geometry_start roots: shape [28, 1032]; singular values [8.102242469787598, 4.586543560028076, 3.4966719150543213, 3.0512356758117676, 2.775428533554077, 2.0673959255218506, 1.8576520681381226, 1.6990407705307007, 1.3841193914413452, 1.267945647239685, 0.785152018070221, 0.6202095150947571, 0.5429512858390808, 0.4520306885242462, 0.38279709219932556, 9.679758932179539e-07, 2.9008856472501066e-07, 1.3628988426717115e-07, 1.1733800420188345e-07, 8.889025338021384e-08, 6.173337396830902e-08, 3.9660417172626694e-08, 1.7703984056538502e-08, 1.3868585391207944e-08, 5.476314068886268e-09, 2.736523785351608e-10, 1.3626105731823234e-15, 9.770009092671274e-19]; centered singular values [5.163276672363281, 3.5593156814575195, 3.389726400375366, 2.7865426540374756, 2.3818204402923584, 2.053831100463867, 1.699404001235962, 1.386357307434082, 1.3661681413650513, 0.7895296812057495, 0.6372889280319214, 0.6109821796417236, 0.4537200927734375, 0.3849126398563385, 5.671116696248646e-07, 2.876464293422032e-07, 1.427223850214432e-07, 1.3175308311019762e-07, 1.0604104261346947e-07, 9.901084752073075e-08, 7.484471353791378e-08, 2.1712343922786204e-08, 1.5558077492983102e-08, 7.4104495872973075e-09, 1.9163439723968168e-09, 7.638448033808753e-12, 3.826227023155578e-16, 2.783898519987631e-17].


geometry_end roots: shape [16, 1032]; singular values [6.421976089477539, 4.531103610992432, 2.3452796936035156, 1.9870270490646362, 1.6900817155838013, 1.5466779470443726, 1.384819507598877, 1.0112775564193726, 4.953681127517484e-07, 2.1814183526203124e-07, 7.819353697868792e-08, 6.853396516959265e-09, 4.1325373678624544e-10, 5.3391900982663834e-20, 2.493398890381098e-23, 1.3032918926576065e-26]; centered singular values [4.739805221557617, 2.8455429077148438, 1.992057204246521, 1.7774585485458374, 1.6305747032165527, 1.3946318626403809, 1.1967045068740845, 1.244949885403912e-06, 7.449519330293697e-07, 2.2209964356534329e-07, 6.630732940493544e-08, 3.080019794765576e-09, 7.404949917133585e-12, 7.675418597402347e-20, 2.273414771768451e-23, 9.162751918122226e-27].

The native root samples are different training/evaluation batches (28 and 16 rows); their spectra are not a before/after comparison of identical sentences. XOR uses the same four sentences at both endpoints.


### Gradient displacement and derivation stability

Every observed optimizer step has coordinate arrays under `displacements/`; events map each array to its parameter and sparse row ids. The table aggregates step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later step differ from the negative current gradient.

| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |
|---|---:|---:|---:|---:|---:|
| `conceptualSpaces.0.layers.3.W` | 2 | 0.6836997 | 0.0005037743 | 0.0007503143 | -0.999704 |

| Sentence word rows | Observed epochs | Modal fraction | Distinct derivations |
|---|---:|---:|---:|
| [7, 8, 9] | 1 | 1 | 1 |
| [10, 8, 10] | 1 | 1 | 1 |
| [7, 8, 11] | 1 | 1 | 1 |
| [12, 8, 9] | 1 | 1 | 1 |
| [9, 8, 12] | 1 | 1 | 1 |
| [12, 8, 11] | 1 | 1 | 1 |
| [11, 8, 7] | 1 | 1 | 1 |
| [12, 8, 10] | 1 | 1 | 1 |
| [13, 8, 14] | 1 | 1 | 1 |
| [12, 8, 15] | 1 | 1 | 1 |
| [16, 8, 11] | 1 | 1 | 1 |
| [10, 8, 16] | 1 | 1 | 1 |
| [9, 8, 10] | 1 | 1 | 1 |
| [15, 8, 11] | 1 | 1 | 1 |
| [16, 8, 12] | 1 | 1 | 1 |
| [9, 8, 17] | 1 | 1 | 1 |
| [18, 8, 10] | 1 | 1 | 1 |
| [12, 8, 12] | 1 | 1 | 1 |
| [9, 8, 9] | 1 | 1 | 1 |
| [9, 8, 11] | 1 | 1 | 1 |
| [11, 8, 13] | 1 | 1 | 1 |

Native training has one configured epoch. Its modal fraction is descriptive, not evidence of stability across repeated epochs. If a native sentence repeats within an epoch, its last kept derivation is that epoch’s sample; all trial events remain saved.

Across recorded training trials and evaluation: 216 word occurrences, 0 with an activated candidate, 0 outranked by an activated candidate. A zero candidate count supplies no evidence about competition with activated words.

Full pairwise tensors and row IDs are beside [native-stage1/BasicModel_answers_tied_benchmark-ownership/geometry-end.json](native-stage1/BasicModel_answers_tied_benchmark-ownership/geometry-end.json). Native reserve-wide pair moments cover all row pairs exactly; full matrices cover all observed primed rows. A missing cluster-size buffer in the indexed-only native allocation is reported as absent, not as a measured vector of zeroes.
