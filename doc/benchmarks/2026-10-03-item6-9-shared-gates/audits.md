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
| trial.exploit | codes | 7.101464e-08 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 1.463645e-15 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 0.01392181 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0.1595242 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 7.101464e-08 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 7.288188e-16 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 0.2985372 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0.1443219 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 0.1656652 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `1e45c81225a2d0a4b93cf8dc81268ae5529a03d3fe362f327167903b961bcac4`, `1e45c81225a2d0a4b93cf8dc81268ae5529a03d3fe362f327167903b961bcac4`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 1600; explore kept: 0; selection-rule violations: 0.

The kept trial has the lower reconstruction; equality keeps greedy. Answer and expectation can be worse because neither enters the comparison.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 1600 | 0 | undefined | undefined |
| expectation | 1600 | 77 | 0.0004229976 | 0.01444174 |
| supplied_answer | 1600 | 776 | 0.009022436 | 0.016803 |
| total | 1600 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 0.5000004 [0.5000004, 0.5000004] | 0.3500003 [0.3500003, 0.3500003] | 0.5 [0.5, 0.5] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.1750001 [0.1750001, 0.1750001] | 0.1750001 [0.1750001, 0.1750001] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 0] | 0 [0, 0] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 0.500002 [0.5, 0.5039951] | 0.3500014 [0.35, 0.3527966] | 0.5 [0.5, 0.5] |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.1750007 [0.175, 0.1763983] | 0.1750007 [0.175, 0.1763983] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 0 [0, 1.490117e-07] | 0 [0, 1.490117e-07] | — |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 2.68723e-08] | 0 [0, 2.68723e-09] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 0] | 0 [0, 0] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 0.5000018 [0.5, 0.5029316] | 0.3500012 [0.35, 0.3520521] | 0.5 [0.5, 0.5] |
| trial.exploit.train | `expectation.intra` | relative / [True] | [0.1] | 0.0009054445 [0.0003313145, 1.256032] | 9.054445e-05 [3.313145e-05, 0.1256032] | 0.4000993 [0.4000993, 0.4000993] |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 2.68723e-08] | 0 [0, 2.68723e-09] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 0.5034996 [0.4890541, 0.5129545] | 0.3524497 [0.3423379, 0.3590682] | 0.5 [0.5, 0.5] |
| trial.explore.train | `expectation.intra` | relative / [True] | [0.1] | 0.001874735 [0.0002620714, 1.219928] | 0.0001874735 [2.620714e-05, 0.1219928] | 0.4000993 [0.4000993, 0.4000993] |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.5709563 [0, 1.356948] | 0.05709563 [0, 0.1356948] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.0, "expectation": null, "output": 0.3500002324581146}`.
With the answer: 0.3500002; without it: 0.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R |
|---|---:|---:|---:|---:|
| exploit | 0 | 3.575054e-05 | 0.35 | 0 |
| explore | 0.08304821 | 4.894233e-05 | 0.3552471 | 0.08304821 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 6 × 10 | 0.9999999 / 1 / 1 | 0.03093585 / 0.1529259 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_start | 1 | 6 × 10 | 0.9999999 / 1 / 1 | -0.04151795 / 0.08214228 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 0 | 6 × 10 | 0.9999999 / 1 / 1 | 0.03093585 / 0.1529259 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 1 | 6 × 10 | 0.9999999 / 1 / 1 | -0.04151795 / 0.08214228 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

geometry_start roots: shape [4, 10]; singular values [0.8928763270378113, 0.8164832592010498, 0.48917874693870544, 4.035945266878116e-08]; centered singular values [0.8423495292663574, 0.514386773109436, 5.2457863830568385e-08, 2.589200143177095e-08].


geometry_end roots: shape [4, 10]; singular values [0.8928763270378113, 0.8164832592010498, 0.48917874693870544, 4.035945266878116e-08]; centered singular values [0.8423495292663574, 0.514386773109436, 5.2457863830568385e-08, 2.589200143177095e-08].


### Gradient displacement and derivation stability

Every observed optimizer step has coordinate arrays under `displacements/`; events map each array to its parameter and sparse row ids. The table aggregates step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later step differ from the negative current gradient.

| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |
|---|---:|---:|---:|---:|---:|
| `conceptualSpaces.1.layers.3.W` | 800 | 2.631649e-35 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.stop_anchor` | 800 | 0 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.reduce_anchor` | 800 | 0 | 0 | 0 | undefined |
| `symbolSpace.subspace.layers.0.operation_layer.apply_anchor` | 800 | 0 | 0 | 0 | undefined |

| Sentence word rows | Observed epochs | Modal fraction | Distinct derivations |
|---|---:|---:|---:|
| [0, 1] | 400 | 1 | 1 |
| [0, 2] | 400 | 1 | 1 |
| [4, 1] | 400 | 1 | 1 |
| [4, 2] | 400 | 1 | 1 |

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
| trial.exploit | perception | 9.99393e-25 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | codes | 0.434734 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 5.047901e-05 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 0.0001728136 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 27.59552 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 1.000261e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 0.4397227 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 5.57258e-05 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 0.001347973 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 26.6488 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 4.213016 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 10.56051 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 44.58728 | undefined | undefined | undefined |
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
| supplied_answer | 28 | 1 | 0.8583933 | 0.8583933 |
| total | 28 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 4.213189 [4.213189, 4.213189] | 2.949232 [2.949232, 2.949232] | 0.0625 [0.0625, 0.0625] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.184327 [0.184327, 0.184327] | 0.184327 [0.184327, 0.184327] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 8.516648 [8.516648, 8.516648] | 8.516648 [8.516648, 8.516648] | — |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.535866 [1.535866, 1.535866] | 0.7679329 [0.7679329, 0.7679329] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 4.180762 [4.180762, 4.180762] | 2.926533 [2.926533, 2.926533] | 0.0625 [0.0625, 0.0625] |
| batch_end.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083331 [1.083331, 1.083331] | 0.01083331 [0.01083331, 0.01083331] | — |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.1829083 [0.1829083, 0.1829083] | 0.1829083 [0.1829083, 0.1829083] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 8.40099 [8.40099, 8.40099] | 8.40099 [8.40099, 8.40099] | — |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.515008 [1.515008, 1.515008] | 0.7575042 [0.7575042, 0.7575042] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.535866 [1.535866, 1.535866] | 0.767933 [0.767933, 0.767933] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 1.974343 [1.974343, 1.974343] | 1.38204 [1.38204, 1.38204] | 0.0625 [0.0625, 0.0625] |
| trial.exploit.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.538223 [1.538223, 1.538223] | 0.7691113 [0.7691113, 0.7691113] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 1.978979 [1.978979, 1.978979] | 1.385285 [1.385285, 1.385285] | 0.0625 [0.0625, 0.0625] |
| trial.explore.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.515008 [1.515008, 1.515008] | 0.7575041 [0.7575041, 0.7575041] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.7679328918457031, "expectation": null, "output": 2.9492321014404297}`.
With the answer: 3.717165; without it: 0.7679329.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R |
|---|---:|---:|---:|---:|
| exploit | 0.7691114 | undefined | 1.38204 | 0.7691114 |
| explore | 0.7575041 | undefined | 1.385285 | 0.7575041 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 65536 × 1032 | 0.9999996 / 1 / 1 | -6.314817e-07 / 0.0009689612 | False | None |
| geometry_end | 0 | 65536 × 1032 | 0.9999945 / 1 / 1.000078 | -6.314801e-07 / 0.0009689612 | False | None |

geometry_start roots: shape [28, 1032]; singular values [7.782539367675781, 3.5992138385772705, 3.506134271621704, 2.572850227355957, 2.361125946044922, 1.9817675352096558, 1.7523704767227173, 1.4721051454544067, 1.2806090116500854, 1.2605030536651611, 1.130115270614624, 0.9377956986427307, 0.7251294851303101, 0.6613879799842834, 0.6019846200942993, 0.4683470129966736, 0.3796735405921936, 3.5844303170051717e-07, 3.414750153751811e-07, 1.6839021554915234e-07, 1.5042139978049818e-07, 8.271359774880693e-08, 4.312261125960504e-08, 1.9821063901304115e-08, 1.803826066293368e-08, 1.1839847324779385e-08, 8.207788226854973e-09, 8.185894114789115e-15]; centered singular values [5.246207237243652, 3.506218910217285, 2.670189380645752, 2.4359519481658936, 2.1584675312042236, 1.756584882736206, 1.4820736646652222, 1.4708269834518433, 1.2761955261230469, 1.2588162422180176, 0.9607177376747131, 0.7332451343536377, 0.6873828768730164, 0.6019929647445679, 0.4683673679828644, 0.4008380174636841, 6.440177457989193e-07, 3.7119642115612805e-07, 3.511610486839345e-07, 2.4833380507516267e-07, 1.8188471528901573e-07, 1.370311366599708e-07, 5.524797686007332e-08, 4.420630617119059e-08, 2.844868696172398e-08, 1.946924577111986e-08, 3.4434290974161286e-09, 8.743284145729807e-15].


geometry_end roots: shape [16, 1032]; singular values [5.568166732788086, 3.5228564739227295, 2.7215323448181152, 2.5149827003479004, 2.1287713050842285, 1.781286358833313, 1.6112457513809204, 1.2823774814605713, 0.9390164017677307, 0.7813537120819092, 1.3945032151241321e-06, 1.3884332474844996e-06, 7.336476670616321e-08, 3.730741227059298e-08, 2.5763269420392074e-16, 5.6376516515775894e-24]; centered singular values [4.7054972648620605, 2.90873122215271, 2.625598669052124, 2.2431023120880127, 1.78184974193573, 1.611306071281433, 1.4190120697021484, 0.9390260577201843, 0.7822648286819458, 6.200934876687825e-07, 5.2164233466101e-07, 3.623652844453318e-07, 1.2291594941871153e-07, 1.1614498696133069e-08, 9.334156033822746e-14, 7.088125209251099e-21].

The native root samples are different training/evaluation batches (28 and 16 rows); their spectra are not a before/after comparison of identical sentences. XOR uses the same four sentences at both endpoints.


### Gradient displacement and derivation stability

Every observed optimizer step has coordinate arrays under `displacements/`; events map each array to its parameter and sparse row ids. The table aggregates step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later step differ from the negative current gradient.

| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |
|---|---:|---:|---:|---:|---:|
| `conceptualSpaces.0.layers.3.W` | 2 | 0.4372298 | 0.0003162414 | 0.0007220175 | -0.9995952 |

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
