# Ownership, costs and geometry

Derived from the sole shared XOR training, with both unchanged bars. No model was rerun. No native measurement is requested or performed in the closing round.

Gradient tables show raw graph reach before the owner restriction. Structural groups overlap (for example a perceptual synthesis module also belongs to generation). The per-parameter ownership files show the actual permitted/applied writer. A cosine is undefined when either gradient has zero norm. These zeros do not establish how an inactive objective would behave.

Term statistics are over logged event-level means, not a pooled estimator over unequal batches. Trial and batch totals are separately reported locations; they must not be added again. The endpoint is the saved evaluation batch after training; the last training batch is also retained in the JSON. Penalties and reporting/count entries are listed separately from trained relative errors by their `kind` and `trained` fields.

Term definitions, targets, uninformed baselines, purpose and owner are in [GradientFlow](../../GradientFlow.md). Reconstruction has the free byte term divided by log(256) and the antipode term divided by log(2), both weighted by reconstructionScale. The supplied squared-error answer uses the detached target mean-square norm. Zero-target squared terms are penalties. concept_readout_l1 retains its own coefficient as a proximal penalty rather than a normalized objective.

## XOR_grammar

Declared parameter records: 75; active: 20; inactive: 55; conflicts: **0**; training backward calls: 1200.

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
| trial.exploit | codes | 0.2563451 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 0.001552728 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0.04471921 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 0.3347066 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0.2039439 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 0.1690238 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 0.003199663 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0.05617309 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 0.3044338 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0.1322127 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 0.3281622 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `99ddf384f74b53317e7291c8554549e81ef5493d79b8098f23ce4ed5f8b6d1fb`, `99ddf384f74b53317e7291c8554549e81ef5493d79b8098f23ce4ed5f8b6d1fb`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 1600; explore kept: 173; selection-rule violations: 0.

The kept trial has the lower reconstruction; equality keeps greedy. Answer and expectation can be worse because neither enters the comparison.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 1600 | 0 | undefined | undefined |
| expectation | 1600 | 101 | 0.02106268 | 0.2347022 |
| supplied_answer | 1600 | 177 | 0.2768367 | 4.944702 |
| total | 1600 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 0.2294964 [0.2294964, 0.2294964] | 0.1606475 [0.1606475, 0.1606475] | 0.5 [0.5, 0.5] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.08032373 [0.08032373, 0.08032373] | 0.08032373 [0.08032373, 0.08032373] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 0.7299317 [0.7299317, 0.7299317] | 0.7299317 [0.7299317, 0.7299317] | — |
| batch_end.eval | `reconstruction.antipode` | relative / [True] | [0.1] | 0.6797953 [0.6797953, 0.6797953] | 0.06797953 [0.06797953, 0.06797953] | 0.6931472 [0.6931472, 0.6931472] |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.1316336 [0.1316336, 0.1316336] | 0.01316336 [0.01316336, 0.01316336] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 0.1711201 [2.119481e-05, 1.571544] | 0.1197841 [1.483637e-05, 1.100081] | 0.5 [0.5, 0.5] |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.05989203 [7.418183e-06, 0.5500405] | 0.05989203 [7.418183e-06, 0.5500405] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 0.7137045 [0.4897702, 6.710762] | 0.7137045 [0.4897702, 6.710762] | — |
| batch_end.train | `reconstruction.antipode` | relative / [True] | [0.1] | 0.8012713 [0.2523355, 2.941344] | 0.08012713 [0.02523355, 0.2941344] | 0.6931472 [0.6931472, 0.6931472] |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.1287072 [0.08832362, 1.210198] | 0.01287072 [0.008832362, 0.1210198] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.antipode` | relative / [True] | [0.1] | 0.6797953 [0.6797953, 0.6797953] | 0.06797953 [0.06797953, 0.06797953] | 0.6931472 [0.6931472, 0.6931472] |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.1316336 [0.1316336, 0.1316336] | 0.01316336 [0.01316336, 0.01316336] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 0.1703186 [0.0001243934, 1.788486] | 0.119223 [8.70754e-05, 1.25194] | 0.5 [0.5, 0.5] |
| trial.exploit.train | `expectation.intra` | relative / [True] | [0.1] | 0.001438029 [0.0002555965, 1.583451] | 0.0001438029 [2.555965e-05, 0.1583451] | 0.4000362 [0.4000362, 0.4000362] |
| trial.exploit.train | `reconstruction.antipode` | relative / [True] | [0.1] | 0.8015946 [0.2523355, 4.756801] | 0.08015946 [0.02523355, 0.4756801] | 0.6931472 [0.6931472, 0.6931472] |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.1295332 [0.08832362, 1.22142] | 0.01295332 [0.008832362, 0.122142] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 0.5082538 [0.001132887, 2.828597] | 0.3557777 [0.0007930212, 1.980018] | 0.5 [0.5, 0.5] |
| trial.explore.train | `expectation.intra` | relative / [True] | [0.1] | 0.003020982 [0.0002793792, 2.200624] | 0.0003020982 [2.793792e-05, 0.2200624] | 0.4000362 [0.4000362, 0.4000362] |
| trial.explore.train | `reconstruction.antipode` | relative / [True] | [0.1] | 1.240933 [0.2523355, 3.748931] | 0.1240933 [0.02523355, 0.3748931] | 0.6931472 [0.6931472, 0.6931472] |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.1243895 [0.0883213, 1.508596] | 0.01243895 [0.00883213, 0.1508596] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.08114288747310638, "expectation": null, "output": 0.1606474667787552}`.
With the answer: 0.2417904; without it: 0.08114289.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R |
|---|---:|---:|---:|---:|
| exploit | 0.0670221 | 0.0003260154 | 0.1524133 | 0.0670221 |
| explore | 0.1024448 | 0.0004149062 | 0.1623808 | 0.1024448 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 6 × 10 | 1 / 1 / 1 | -0.04136143 / 0.1696337 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_start | 1 | 6 × 10 | 0.9999999 / 1 / 1 | 0.0461256 / 0.0718581 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 0 | 6 × 10 | 1 / 1 / 1 | -0.04136143 / 0.1696337 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 1 | 6 × 10 | 0.3361666 / 1.07053 / 1.894742 | 0.5788282 / 0.367688 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

geometry_start roots: shape [4, 10]; singular values [0.8874570727348328, 0.6843959093093872, 0.5780777931213379, 3.069297349611588e-08]; centered singular values [0.7447500824928284, 0.5935366153717041, 0.3971485495567322, 4.6777994811009194e-08].


geometry_end roots: shape [4, 10]; singular values [1.095977783203125, 0.5157907009124756, 0.17930617928504944, 0.14577431976795197]; centered singular values [0.5160010457038879, 0.18162882328033447, 0.15404513478279114, 2.7221460641158046e-08].


### Gradient displacement and derivation stability

Every observed optimizer step has coordinate arrays under `displacements/`; events map each array to its parameter and sparse row ids. The table aggregates step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later step differ from the negative current gradient.

| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |
|---|---:|---:|---:|---:|---:|
| `conceptualSpaces.1.layers.3.W` | 800 | 0.2269808 | 0.008286319 | 0.03948986 | -0.5500265 |
| `symbolSpace.subspace.layers.0.operation_layer.stop_anchor` | 800 | 0.000621083 | 5.824187e-05 | 0.08656277 | -0.6728297 |
| `symbolSpace.subspace.layers.0.operation_layer.reduce_anchor` | 800 | 0.0001965971 | 2.366416e-05 | 0.1152925 | -0.8296271 |
| `symbolSpace.subspace.layers.0.operation_layer.apply_anchor` | 800 | 0.0006051307 | 5.525161e-05 | 0.08372342 | -0.7523797 |
| `symbolSpace.languageSpace.generate_policy.weight` | 800 | 0.009725347 | 0.0007152666 | 0.07368871 | -0.8163928 |
| `symbolSpace.languageSpace.generate_policy.bias` | 800 | 0.01348676 | 0.001038552 | 0.07833923 | -0.9338907 |

| Sentence word rows | Observed epochs | Modal fraction | Distinct derivations |
|---|---:|---:|---:|
| [0, 1] | 400 | 0.885 | 7 |
| [0, 2] | 400 | 0.8275 | 6 |
| [4, 1] | 400 | 0.4375 | 6 |
| [4, 2] | 400 | 0.44 | 6 |

The modal fractions describe the 400 epochs of this one XOR training; all trial events remain saved.

Across recorded training trials and evaluation: 6408 word occurrences, 0 with an activated candidate, 0 outranked by an activated candidate. A zero candidate count supplies no evidence about competition with activated words.

Full pairwise tensors and row IDs are beside [xor-ownership/geometry-end.json](xor-ownership/geometry-end.json). Both six-row dictionaries have complete saved cosine matrices and cluster-size vectors.
