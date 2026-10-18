# Measured tables

Description failed its prerequisite gate. These are observations of the kept readings, not target labels. Full precision, all first-pair menus and the saved programs are in [measurements.json](measurements.json).

## Word classes and counts

Before is the restored passed definite artifact; after is the description artifact saved before evaluation. Unobserved words retain the uniform floor and the tie-break label `adverb_argin1`; this is not learned class evidence.

| Word | Before: label; counts | After: label; counts |
| --- | --- | --- |
| a | `lower_argin1`; `definite_argin1`=2, `lower_argin1`=4 | `lower_argin1`; `definite_argin1`=2, `lower_argin1`=4 |
| bird | `definite_argin2`; `definite_argin2`=1, `definite_argout1`=1, `lower_argin2`=1, `lower_argout1`=1 | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `intersection_argin2`=1, `intersection_argout1`=1, `lower_argin2`=6, `lower_argout1`=6 |
| black | unobserved; {} | `lower_argin1`; `definite_argin1`=1, `lower_argin1`=5 |
| cat | `definite_argin2`; `definite_argin2`=2, `definite_argout1`=2, `lower_argin1`=1, `lower_argin2`=1, `lower_argout1`=1 | `lower_argin2`; `definite_argin2`=3, `definite_argout1`=3, `intersection_argin2`=1, `intersection_argout1`=1, `lower_argin1`=2, `lower_argin2`=4, `lower_argout1`=4 |
| chases | unobserved; {} | unobserved; {} |
| dog | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `lower_argin1`=1, `lower_argin2`=2, `lower_argout1`=2 | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `lower_argin1`=4, `lower_argin2`=5, `lower_argout1`=5 |
| horse | `definite_argin2`; `definite_argin2`=1, `definite_argout1`=1, `lower_argin2`=1, `lower_argout1`=1 | `lower_argin2`; `definite_argin2`=2, `definite_argout1`=2, `intersection_argin2`=1, `intersection_argout1`=1, `lower_argin2`=5, `lower_argout1`=5 |
| is | unobserved; {} | unobserved; {} |
| large | unobserved; {} | `lower_argin1`; `intersection_argin1`=1, `intersection_argout1`=1, `lower_argin1`=4, `lower_argin2`=1, `lower_argout1`=1 |
| near | unobserved; {} | unobserved; {} |
| on | unobserved; {} | unobserved; {} |
| runs | unobserved; {} | unobserved; {} |
| sees | unobserved; {} | unobserved; {} |
| sleeps | unobserved; {} | unobserved; {} |
| small | unobserved; {} | `intersection_argin1`; `intersection_argin1`=2, `intersection_argout1`=2, `lower_argin1`=2, `lower_argin2`=2, `lower_argout1`=2 |
| the | `definite_argin1`; `definite_argin1`=3, `lower_argin1`=1, `lower_argin2`=2, `lower_argout1`=2 | `definite_argin1`; `definite_argin1`=3, `lower_argin1`=1, `lower_argin2`=2, `lower_argout1`=2 |
| white | unobserved; {} | `lower_argin1`; `definite_argin1`=1, `lower_argin1`=4, `lower_argin2`=1, `lower_argout1`=1 |

## Argument occupant cosines

The [complete collapse table](collapse.json) includes all 4,186 argument pairs before and after. Nonzero after-values are below. A **flow** mark identifies input-to-output routing imposed by §2b, not learned output typing. Symmetric output pools both inputs; its cosine need not be one with either input alone.

| Argument A | Argument B | Before | After | Construction |
| --- | --- | ---: | ---: | --- |
| `definite_argin1` | `lower_argin1` | 0.699913 | 0.521641 |  |
| `definite_argin1` | `lower_argin2` | 0.501745 | 0.170783 |  |
| `definite_argin1` | `lower_argout1` | 0.501745 | 0.170783 |  |
| `definite_argin2` | `definite_argout1` | 1.000000 | 1.000000 | flow |
| `definite_argin2` | `intersection_argin2` | 0.000000 | 0.894427 |  |
| `definite_argin2` | `intersection_argout1` | 0.000000 | 0.547723 |  |
| `definite_argin2` | `lower_argin1` | 0.260133 | 0.260820 |  |
| `definite_argin2` | `lower_argin2` | 0.683763 | 0.805118 |  |
| `definite_argin2` | `lower_argout1` | 0.683763 | 0.805118 |  |
| `definite_argout1` | `intersection_argin2` | 0.000000 | 0.894427 |  |
| `definite_argout1` | `intersection_argout1` | 0.000000 | 0.547723 |  |
| `definite_argout1` | `lower_argin1` | 0.260133 | 0.260820 |  |
| `definite_argout1` | `lower_argin2` | 0.683763 | 0.805118 |  |
| `definite_argout1` | `lower_argout1` | 0.683763 | 0.805118 |  |
| `intersection_argin1` | `intersection_argout1` | 0.000000 | 0.790569 | flow |
| `intersection_argin1` | `lower_argin1` | 0.000000 | 0.361403 |  |
| `intersection_argin1` | `lower_argin2` | 0.000000 | 0.211289 |  |
| `intersection_argin1` | `lower_argout1` | 0.000000 | 0.211289 |  |
| `intersection_argin2` | `intersection_argout1` | 0.000000 | 0.612372 | flow |
| `intersection_argin2` | `lower_argin1` | 0.000000 | 0.116642 |  |
| `intersection_argin2` | `lower_argin2` | 0.000000 | 0.818317 |  |
| `intersection_argin2` | `lower_argout1` | 0.000000 | 0.818317 |  |
| `intersection_argout1` | `lower_argin1` | 0.000000 | 0.357143 |  |
| `intersection_argout1` | `lower_argin2` | 0.000000 | 0.668153 |  |
| `intersection_argout1` | `lower_argout1` | 0.000000 | 0.668153 |  |
| `lower_argin1` | `lower_argin2` | 0.345857 | 0.400892 |  |
| `lower_argin1` | `lower_argout1` | 0.345857 | 0.400892 |  |
| `lower_argin2` | `lower_argout1` | 1.000000 | 1.000000 | flow |

Intersection was unobserved before. Its three kept applications produced the following counts. Output equals the combined inputs exactly, by the flow rule; no type merge or learned equality is claimed.

| Word | argin1 | argin2 | argout1 |
| --- | ---: | ---: | ---: |
| bird | 0 | 1 | 1 |
| cat | 0 | 1 | 1 |
| horse | 0 | 1 | 1 |
| large | 1 | 0 | 1 |
| small | 2 | 0 | 2 |

## Recorded noun prior

All direct ties to individual columns are absent before and after. The chooser does not read this table; an absent tie does not establish that a word is not a noun.

| Word | Concepts before | Concepts after | Individual columns before → after |
| --- | --- | --- | --- |
| a | [239] | [239] | [] → [] |
| bird | [40] | [40] | [] → [] |
| black | [44] | [44] | [] → [] |
| cat | [36] | [36] | [] → [] |
| chases | [] | [] | [] → [] |
| dog | [38] | [38] | [] → [] |
| horse | [42] | [42] | [] → [] |
| is | [] | [] | [] → [] |
| large | [50] | [50] | [] → [] |
| near | [] | [] | [] → [] |
| on | [] | [] | [] → [] |
| runs | [52] | [52] | [] → [] |
| sees | [] | [] | [] → [] |
| sleeps | [54] | [54] | [] → [] |
| small | [48] | [48] | [] → [] |
| the | [245] | [245] | [] → [] |
| white | [46] | [46] | [] → [] |

## Trajectory

Only six training batches ran, so the eight-batch reporting block ends at batch six. CE is observed pre-update CE on reachable student trial states; reused prefixes count in each lesson call. These are changing states, not a matched-state learning comparison.

| Batches | Lesson calls | Fold targets / call | Wait CE (reachable / unreachable) | Fold CE (reachable / unreachable) |
| --- | ---: | ---: | ---: | ---: |
| 1–6 | 48 | 43/48 = 0.895833 | 0.107229 (149 / 0) | 1.755699 (43 / 0) |

First-pair accuracy on the kept description readings: **3/24 (12.5%)**, with four unreached pairs counted as incorrect. Those four kept readings
visited the second word first; their source pair was (1, 0), not (0, 1).
“Correct” means the kept action was intersection, not that intersection had
the highest chooser logit. No other family was presented in this pass.

| Batches | Sentences | Leaves | Binary | Unary | Transpose | All steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–6 | 24 | 48 | 24 | 0 | 0 | 72 |

All 24 kept training compose steps, and the four held-out compose steps, pay `WHAT_STEP_COST` in the keep comparison.

## First-pair scores

Means over the 20 reached first pairs in kept training trials; the four unreached pairs have no score observation. Maximum logit means the largest per-variant total logit, including occupation. Probability is summed across the operation’s variants. These are observations, not a causal attribution of the failed gate.

| Operation | Mean maximum logit | Mean occupation term | Mean probability | Declared / legal variants |
| --- | ---: | ---: | ---: | --- |
| STOP | 2.639151 | 0.000000 | 0.105075 | [1] / [1] |
| lower | 4.539691 | 0.436975 | 0.516503 | [19] / [1] |
| definite | 3.142771 | 0.373348 | 0.095232 | [19] / [1] |
| intersection | 4.167450 | 0.128106 | 0.283191 | [1] / [1] |

| Batch | Case | Correct | Intersection: logit / occupation | Lower: logit / occupation | Definite: logit / occupation | STOP logit |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `np:description:3:3` | True | 4.555327 / 0.010870 | 0.930098 / 0.260870 | 3.664028 / 0.260870 | -0.810784 |
| 1 | `np:description:1:0` | False | 5.069927 / 0.010870 | 0.569903 / 0.153727 | 4.111772 / 0.296584 | -0.641115 |
| 1 | `np:description:1:1` | unreached | — | — | — | — |
| 1 | `np:description:0:3` | False | 4.142515 / 0.010870 | 0.758841 / 0.260870 | 4.009934 / 0.260870 | -1.870868 |
| 2 | `np:description:2:1` | unreached | — | — | — | — |
| 2 | `np:description:3:1` | unreached | — | — | — | — |
| 2 | `np:description:2:0` | True | 5.959620 / 0.010870 | 0.450153 / 0.121981 | 3.992121 / 0.344203 | 1.799480 |
| 2 | `np:description:2:2` | True | 6.130702 / 0.010870 | 0.496798 / 0.260870 | 1.991903 / 0.260870 | 1.541174 |
| 3 | `np:description:0:0` | False | 1.388451 / 0.090909 | 4.018169 / 0.090909 | 1.683921 / 1.272727 | 1.959125 |
| 3 | `np:description:1:3` | False | 4.375588 / 0.125000 | 7.298420 / 0.125000 | 3.762609 / 0.583333 | 5.346648 |
| 3 | `np:description:3:2` | False | 2.566499 / 0.416667 | 4.571166 / 0.166667 | 1.196864 / 0.166667 | 5.202833 |
| 3 | `np:description:0:2` | False | 2.050296 / 0.166667 | 4.547630 / 0.166667 | 2.465759 / 1.166667 | 3.530395 |
| 4 | `np:description:3:1` | False | 5.277465 / 0.200000 | 6.662119 / 0.400000 | 2.584977 / 0.100000 | 3.355274 |
| 4 | `np:description:2:0` | unreached | — | — | — | — |
| 4 | `np:description:2:2` | False | 5.520591 / 0.433333 | 5.776724 / 0.300000 | 2.503413 / 0.100000 | 4.966531 |
| 4 | `np:description:0:3` | False | 4.547133 / 0.100000 | 9.383318 / 0.866667 | 5.908523 / 0.533333 | 4.205593 |
| 5 | `np:description:0:2` | False | 4.404519 / 0.083333 | 6.428757 / 1.083333 | 3.602470 / 0.333333 | 2.187497 |
| 5 | `np:description:3:3` | False | 5.819809 / 0.250000 | 7.384185 / 0.583333 | 4.846230 / 0.166667 | 4.382906 |
| 5 | `np:description:1:0` | False | 1.564375 / 0.071429 | 3.833333 / 0.392857 | 1.406653 / 0.464286 | 2.306143 |
| 5 | `np:description:2:1` | False | 5.353137 / 0.222222 | 6.449681 / 0.361111 | 2.590320 / 0.083333 | 4.700104 |
| 6 | `np:description:1:3` | False | 3.758071 / 0.071429 | 5.675735 / 0.685714 | 4.020182 / 0.342857 | 4.515758 |
| 6 | `np:description:0:0` | False | 3.968946 / 0.062500 | 5.933574 / 0.987500 | 3.352364 / 0.387500 | -0.160576 |
| 6 | `np:description:1:1` | False | 2.396875 / 0.000000 | 4.676721 / 0.685714 | 1.844133 / 0.271429 | 3.524462 |
| 6 | `np:description:3:2` | False | 4.499149 / 0.214286 | 4.948502 / 0.785714 | 3.317233 / 0.071429 | 2.742438 |

## Meaning and keep observations

The three kept intersections had closed meaning norms 0, .1154700667 and 0.
The two definites had norm .2000000328; the 19 lowerings ranged from
.2000000328 to .4285714328. Both programs cost one compose step. The
per-trial greedy/explore keep costs were not retained, so the reason a trial
won cannot be established from these norms or the chooser logits alone.
