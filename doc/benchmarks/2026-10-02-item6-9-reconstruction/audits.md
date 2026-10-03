# Ownership, costs and geometry

Derived from the two saved first measurements. No model was rerun. The XOR audit reuses class run 1; the native audit reuses the production stage-1 test.

Gradient tables show raw graph reach before the owner restriction. Structural groups overlap (for example a perceptual synthesis module also belongs to generation). The per-parameter ownership files show the actual permitted/applied writer. A cosine is undefined when either gradient has zero norm. These zeros do not establish how an inactive objective would behave.

Term statistics are over logged event-level means, not a pooled estimator over unequal batches. Trial and batch totals are separately reported locations; they must not be added again. The endpoint is the saved evaluation batch after training; the last training batch is also retained in the JSON. Penalties and reporting/count entries are listed separately from trained relative errors by their `kind` and `trained` fields.

Term definitions, targets, uninformed baselines, purpose and owner are in [GradientFlow](../../GradientFlow.md). Reconstruction has independent witnessed and free byte terms, each divided by log(256) and each weighted by reconstructionScale. The supplied squared-error answer uses the detached target mean-square norm. Zero-target squared terms are penalties. concept_readout_l1 retains its own coefficient as a proximal penalty rather than a normalized objective.

## XOR_grammar

Declared parameter records: 73; active: 18; inactive: 55; conflicts: **0**; training backward calls: 1200.

[Every declared and observed writer](xor-ownership/ownership.json); [complete per-parameter gradients](xor-ownership/gradients.json); [configuration record](xor-ownership/configured-training.json).

Configured priorities can be inactive in the observed data. The term table below shows what was actually registered, in each location; an absent term is not a measured zero.

| Setting | Value | Purpose |
|---|---:|---|
| `reconstructionScale` | 0.1 | Weight of each of the two relative reconstruction terms |
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
| trial.exploit | codes | 5.111615e-05 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 3.436972e-07 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 0.0611638 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0.1469594 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 0.03356786 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 0.0008248413 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 0.4787931 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0.1402253 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 0.5979033 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `027f0265c79a07761255988a91975f6d0ab0cf8324ac606a454e93ae0c263f1b`, `027f0265c79a07761255988a91975f6d0ab0cf8324ac606a454e93ae0c263f1b`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 1600; explore kept: 519; selection-rule violations: 0.

A kept greedy trial can have worse reconstruction than explore when explore does not also lower reconstruction-plus-answer. The rule constrains accepting explore; it does not always pick the smaller reconstruction or the smaller total in isolation.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 1600 | 27 | 0.1298449 | 0.2105865 |
| expectation | 1600 | 143 | 0.005850992 | 0.08835245 |
| supplied_answer | 1600 | 204 | 0.1512162 | 1.031955 |
| total | 1600 | 69 | 0.1776005 | 0.8658582 |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 0.292129 [0.292129, 0.292129] | 0.2044903 [0.2044903, 0.2044903] | 0.5 [0.5, 0.5] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.1022451 [0.1022451, 0.1022451] | 0.1022451 [0.1022451, 0.1022451] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 0.06372189 [0.06372189, 0.06372189] | 0.06372189 [0.06372189, 0.06372189] | — |
| batch_end.eval | `reconstruction.bytes` | relative / [True] | [0.1] | 0.01149141 [0.01149141, 0.01149141] | 0.001149141 [0.001149141, 0.001149141] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 0] | 0 [0, 0] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 0.2519537 [0.04765703, 0.6655148] | 0.1763676 [0.03335992, 0.4658604] | 0.5 [0.5, 0.5] |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.08818379 [0.01667996, 0.2329302] | 0.08818379 [0.01667996, 0.2329302] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 2.918561 [0.01513421, 9.846773] | 2.918561 [0.01513421, 9.846773] | — |
| batch_end.train | `reconstruction.bytes` | relative / [True] | [0.1] | 0.01897053 [8.008845e-05, 0.6201538] | 0.001897053 [8.008845e-06, 0.06201538] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.5190513 [0, 1.512664] | 0.05190513 [0, 0.1512664] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.bytes` | relative / [True] | [0.1] | 0.01149141 [0.01149141, 0.01149141] | 0.001149141 [0.001149141, 0.001149141] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0 [0, 0] | 0 [0, 0] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 0.360774 [0.1175638, 1.122547] | 0.2525418 [0.08229469, 0.7857827] | 0.5 [0.5, 0.5] |
| trial.exploit.train | `expectation.intra` | relative / [True] | [0.1] | 0.001874012 [0.0002686952, 1.79124] | 0.0001874012 [2.686952e-05, 0.179124] | 0.4002155 [0.4002155, 0.4002155] |
| trial.exploit.train | `reconstruction.bytes` | relative / [True] | [0.1] | 0.01919771 [8.008845e-05, 0.6201538] | 0.001919771 [8.008845e-06, 0.06201538] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.5190513 [0, 1.772189] | 0.05190513 [0, 0.1772189] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 0.5782977 [0.1202959, 1.659863] | 0.4048084 [0.08420711, 1.161904] | 0.5 [0.5, 0.5] |
| trial.explore.train | `expectation.intra` | relative / [True] | [0.1] | 0.0050724 [0.0002686952, 1.79124] | 0.00050724 [2.686952e-05, 0.179124] | 0.4002155 [0.4002155, 0.4002155] |
| trial.explore.train | `reconstruction.bytes` | relative / [True] | [0.1] | 0.03502823 [0.001200183, 0.6201538] | 0.003502823 [0.0001200183, 0.06201538] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.1] | 0.9821635 [0, 2.039247] | 0.09821635 [0, 0.2039247] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.0011491406476125121, "expectation": null, "output": 0.2044902890920639}`.
With the answer: 0.2056394; without it: 0.001149141.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R+A |
|---|---:|---:|---:|---:|
| exploit | 0.001163078 | 0.0001153014 | 0.1745447 | 0.1757078 |
| explore | 0.05380971 | 0.0001153014 | 0.3532269 | 0.4070366 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 6 × 10 | 0.9999999 / 1 / 1 | -0.04135598 / 0.07457291 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_start | 1 | 6 × 10 | 1 / 1 / 1 | -0.04505096 / 0.09414768 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 0 | 6 × 10 | 0.9999999 / 1 / 1 | -0.04135598 / 0.07457291 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| geometry_end | 1 | 6 × 10 | 1 / 1.372137 / 2.351079 | -0.1164866 / 0.1878895 | False | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

geometry_start roots: shape [4, 10]; singular values [1.7048332691192627, 0.6673901677131653, 0.5027346014976501, 0.12145832180976868]; centered singular values [0.7148867845535278, 0.5460012555122375, 0.12301744520664215, 4.255820229559504e-08].


geometry_end roots: shape [4, 10]; singular values [2.436300039291382, 1.1879216432571411, 0.5959881544113159, 0.4028618335723877]; centered singular values [1.188253402709961, 0.5969142913818359, 0.48787739872932434, 6.515878681057075e-08].

Across recorded training trials and evaluation: 6408 word occurrences, 0 with an activated candidate, 0 outranked by an activated candidate. A zero candidate count supplies no evidence about competition with activated words.

Full pairwise tensors and row IDs are beside [xor-ownership/geometry-end.json](xor-ownership/geometry-end.json). Native reserve-wide pair moments cover all row pairs exactly; full matrices cover all observed primed rows. A missing cluster-size buffer in the indexed-only native allocation is reported as absent, not as a measured vector of zeroes.

## BasicModel_answers_tied_benchmark

Declared parameter records: 153; active: 45; inactive: 108; conflicts: **0**; training backward calls: 3.

[Every declared and observed writer](native-stage1/BasicModel_answers_tied_benchmark-ownership/ownership.json); [complete per-parameter gradients](native-stage1/BasicModel_answers_tied_benchmark-ownership/gradients.json); [configuration record](native-stage1/BasicModel_answers_tied_benchmark-ownership/plan.json).

Configured priorities can be inactive in the observed data. The term table below shows what was actually registered, in each location; an absent term is not a measured zero.

| Setting | Value | Purpose |
|---|---:|---|
| `reconstructionScale` | 0.5 | Weight of each of the two relative reconstruction terms |
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
| trial.exploit | perception | 2.867226e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | codes | 0.06818158 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | chooser | 0.0009817934 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | operators_and_tied_inverses | 2.882885e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | reading_map | 0 | 0 | 16.50283 | undefined | undefined | undefined |
| trial.exploit | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.exploit | other | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | perception | 2.866628e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | codes | 0.06932234 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | chooser | 0.0009818885 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | operators_and_tied_inverses | 2.882298e-24 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | generate | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | reading_map | 0 | 0 | 16.43319 | undefined | undefined | undefined |
| trial.explore | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| trial.explore | other | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | perception | 0 | 0 | 5.504513 | undefined | undefined | undefined |
| batch | codes | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | chooser | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | operators_and_tied_inverses | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | generate | 0 | 0 | 13.15347 | undefined | undefined | undefined |
| batch | reading_map | 0 | 0 | 67.38666 | undefined | undefined | undefined |
| batch | expectation_predictor | 0 | 0 | 0 | undefined | undefined | undefined |
| batch | other | 0 | 0 | 0 | undefined | undefined | undefined |

First trial parameter hashes: `dd1466af6ffd73865c35c31e92036484505ce046b41b8abb8b35047280ce54f9`, `dd1466af6ffd73865c35c31e92036484505ce046b41b8abb8b35047280ce54f9`.
RNG unchanged by each gradient observation: [True, True, True].

### Trial selection

Active training rows: 28; explore kept: 1; selection-rule violations: 0.

A kept greedy trial can have worse reconstruction than explore when explore does not also lower reconstruction-plus-answer. The rule constrains accepting explore; it does not always pick the smaller reconstruction or the smaller total in isolation.

| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |
|---|---:|---:|---:|---:|
| reconstruction | 28 | 0 | undefined | undefined |
| expectation | 0 | 0 | undefined | undefined |
| supplied_answer | 28 | 2 | 0.009076625 | 0.0092749 |
| total | 28 | 0 | undefined | undefined |

### Every recorded cost term

Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.

| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |
|---|---|---|---:|---|---|---|
| batch_end.eval | `answer.what` | relative / [True] | [0.7] | 2.497356 [2.497356, 2.497356] | 1.748149 [1.748149, 1.748149] | 0.0625 [0.0625, 0.0625] |
| batch_end.eval | `output` | penalty / [False] | [1.0] | 0.1092593 [0.1092593, 0.1092593] | 0.1092593 [0.1092593, 0.1092593] | — |
| batch_end.eval | `reconstruction` | penalty / [False] | [1.0] | 5.129617 [5.129617, 5.129617] | 5.129617 [5.129617, 5.129617] | — |
| batch_end.eval | `reconstruction.bytes` | relative / [True] | [0.5] | 0.1960631 [0.1960631, 0.1960631] | 0.09803153 [0.09803153, 0.09803153] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 0.7289961 [0.7289961, 0.7289961] | 0.364498 [0.364498, 0.364498] | 5.545177 [5.545177, 5.545177] |
| batch_end.eval | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `answer.what` | relative / [True] | [0.7] | 4.516988 [4.516988, 4.516988] | 3.161891 [3.161891, 3.161891] | 0.0625 [0.0625, 0.0625] |
| batch_end.train | `concept_readout_l1` | penalty / [False] | [0.01] | 0 [0, 0] | 0 [0, 0] | — |
| batch_end.train | `output` | penalty / [False] | [1.0] | 0.1976182 [0.1976182, 0.1976182] | 0.1976182 [0.1976182, 0.1976182] | — |
| batch_end.train | `reconstruction` | penalty / [False] | [1.0] | 8.93264 [8.93264, 8.93264] | 8.93264 [8.93264, 8.93264] | — |
| batch_end.train | `reconstruction.bytes` | relative / [True] | [0.5] | 0.2447793 [0.2447793, 0.2447793] | 0.1223897 [0.1223897, 0.1223897] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.366105 [1.366105, 1.366105] | 0.6830525 [0.6830525, 0.6830525] | 5.545177 [5.545177, 5.545177] |
| batch_end.train | `reconstruction_unavailable_sentences` | penalty / [False] | [1.0] | 0 [0, 0] | 0 [0, 0] | — |
| trial.exploit.eval | `reconstruction.bytes` | relative / [True] | [0.5] | 0.196063 [0.196063, 0.196063] | 0.09803152 [0.09803152, 0.09803152] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.eval | `reconstruction.free_bytes` | relative / [True] | [0.5] | 0.7289962 [0.7289962, 0.7289962] | 0.3644981 [0.3644981, 0.3644981] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `answer.what` | relative / [True] | [0.7] | 1.001387 [1.001387, 1.001387] | 0.7009712 [0.7009712, 0.7009712] | 0.0625 [0.0625, 0.0625] |
| trial.exploit.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.exploit.train | `reconstruction.bytes` | relative / [True] | [0.5] | 0.2403282 [0.2403282, 0.2403282] | 0.1201641 [0.1201641, 0.1201641] | 5.545177 [5.545177, 5.545177] |
| trial.exploit.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.378709 [1.378709, 1.378709] | 0.6893545 [0.6893545, 0.6893545] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `answer.what` | relative / [True] | [0.7] | 1.000461 [1.000461, 1.000461] | 0.7003229 [0.7003229, 0.7003229] | 0.0625 [0.0625, 0.0625] |
| trial.explore.train | `concept_readout_l1` | penalty / [False] | [0.01] | 1.083333 [1.083333, 1.083333] | 0.01083333 [0.01083333, 0.01083333] | — |
| trial.explore.train | `reconstruction.bytes` | relative / [True] | [0.5] | 0.2670189 [0.2670189, 0.2670189] | 0.1335094 [0.1335094, 0.1335094] | 5.545177 [5.545177, 5.545177] |
| trial.explore.train | `reconstruction.free_bytes` | relative / [True] | [0.5] | 1.368331 [1.368331, 1.368331] | 0.6841654 [0.6841654, 0.6841654] | 5.545177 [5.545177, 5.545177] |

Endpoint objective costs: `{"reconstruction": 0.46252956986427307, "expectation": null, "output": 1.7481495141983032}`.
With the answer: 2.210679; without it: 0.4625296.

Expectation is evaluated in the training trial, not recomputed in the evaluation forward. Its final recorded value is therefore given below with the other final training-trial costs; `undefined` at evaluation means absent, not zero.

| Last training trial | R | E | A | Comparison R+A |
|---|---:|---:|---:|---:|
| exploit | 0.8095187 | undefined | 0.7009712 | 1.51049 |
| explore | 0.8176748 | undefined | 0.7003229 | 1.517998 |

### Code geometry and activated competitors

| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |
|---|---:|---|---|---|---|---|
| geometry_start | 0 | 65536 × 1032 | 0.9999996 / 1 / 1 | -2.186654e-07 / 0.0009689574 | False | None |
| geometry_end | 0 | 65536 × 1032 | 0.9946571 / 1 / 1.012325 | -2.175316e-07 / 0.0009689584 | False | None |

geometry_start roots: shape [28, 1032]; singular values [4.614182949066162, 3.6187262535095215, 3.2102293968200684, 2.634153127670288, 2.4002668857574463, 1.9643213748931885, 1.8136740922927856, 1.4765844345092773, 1.1908005475997925, 0.9077839255332947, 0.8452531695365906, 0.6610108613967896, 0.6321401000022888, 0.49003323912620544, 0.42434459924697876, 0.33866679668426514, 0.27671554684638977, 0.20872052013874054, 0.12267161160707474, 3.5070239391643554e-05, 2.1735141331191699e-07, 1.1428421231585162e-07, 1.0134030503650138e-07, 8.00578874304847e-08, 5.555624582598284e-08, 4.556400057253995e-08, 1.81439361313096e-08, 1.3748764349230669e-09]; centered singular values [3.6239731311798096, 3.4828851222991943, 3.2086501121520996, 2.413766384124756, 2.1216228008270264, 1.8480020761489868, 1.6124160289764404, 1.3853851556777954, 1.1370731592178345, 0.8955324292182922, 0.7307859063148499, 0.6609726548194885, 0.6112619042396545, 0.4267676770687103, 0.41200095415115356, 0.29639750719070435, 0.21310749650001526, 0.19677850604057312, 0.07216017693281174, 2.6135156076634303e-05, 3.825765304554807e-07, 2.0920676035984798e-07, 1.2600568766174547e-07, 5.451098772368823e-08, 4.8003833796883555e-08, 3.936062853426847e-08, 1.79392234400666e-08, 1.1359171381286615e-08].


geometry_end roots: shape [16, 1032]; singular values [5.723824501037598, 2.8607711791992188, 2.0061445236206055, 1.7453874349594116, 1.390321969985962, 1.1737395524978638, 0.7006210684776306, 0.3274006247520447, 4.5126691361474514e-07, 2.816155415530375e-07, 2.2498115015423537e-07, 1.233049715665402e-07, 1.65806257612644e-09, 4.183589577739545e-14, 1.1667180707990992e-17, 7.023070250704042e-26]; centered singular values [2.966834545135498, 2.2166450023651123, 1.9271583557128906, 1.7453879117965698, 1.384560227394104, 0.7006254196166992, 0.3842439651489258, 4.778577817887708e-07, 1.9428028963375255e-07, 1.636109203673186e-07, 1.2427918250068615e-07, 1.0942791561774357e-07, 3.701826067903369e-10, 2.9513363798759455e-15, 9.239080528167123e-18, 3.023491054482955e-24].

The native root samples are different training/evaluation batches (28 and 16 rows); their spectra are not a before/after comparison of identical sentences. XOR uses the same four sentences at both endpoints.

Across recorded training trials and evaluation: 216 word occurrences, 0 with an activated candidate, 0 outranked by an activated candidate. A zero candidate count supplies no evidence about competition with activated words.

Full pairwise tensors and row IDs are beside [native-stage1/BasicModel_answers_tied_benchmark-ownership/geometry-end.json](native-stage1/BasicModel_answers_tied_benchmark-ownership/geometry-end.json). Native reserve-wide pair moments cover all row pairs exactly; full matrices cover all observed primed rows. A missing cluster-size buffer in the indexed-only native allocation is reported as absent, not as a measured vector of zeroes.
