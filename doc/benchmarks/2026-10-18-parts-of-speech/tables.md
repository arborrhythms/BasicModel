# Measured tables

All tables are observations of kept readings. The nested adjective lesson failed; these are not target labels. Full precision and first-pair logit/occupation terms are in [measurements.json](measurements.json).

## Word classes and counts

An unobserved word has a uniform distribution and the tie-break label `adverb_argin1`; that label is not learned evidence. Counts below use the full formal argument names.

| Word | Before: label; counts | After: label; counts |
| --- | --- | --- |
| a | unobserved; {} | `lower_argin1`; `definite_argin1`=3, `intersection_argin1`=1, `intersection_argin2`=3, `lower_argin1`=23 |
| bird | unobserved; {} | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `intersection_argin1`=1, `lower_argin2`=6, `lower_argout1`=6 |
| black | unobserved; {} | `lower_argin1`; `intersection_argin1`=1, `intersection_argin2`=1, `lower_argin1`=4, `lower_argin2`=4, `lower_argout1`=4 |
| cat | unobserved; {} | `lower_argin2`; `definite_argin2`=3, `definite_argout1`=3, `lower_argin1`=3, `lower_argin2`=6, `lower_argout1`=6 |
| chases | unobserved; {} | unobserved; {} |
| dog | unobserved; {} | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `intersection_argin2`=2, `lower_argin1`=1, `lower_argin2`=7, `lower_argout1`=7 |
| horse | unobserved; {} | `lower_argin2`; `definite_argin2`=1, `definite_argout1`=1, `intersection_argin2`=2, `lower_argin2`=6, `lower_argout1`=6 |
| is | unobserved; {} | unobserved; {} |
| large | unobserved; {} | `lower_argin2`; `intersection_argin1`=3, `lower_argin1`=2, `lower_argin2`=5, `lower_argout1`=5 |
| near | unobserved; {} | unobserved; {} |
| on | unobserved; {} | unobserved; {} |
| runs | unobserved; {} | unobserved; {} |
| sees | unobserved; {} | unobserved; {} |
| sleeps | unobserved; {} | unobserved; {} |
| small | unobserved; {} | `lower_argin2`; `intersection_argin1`=2, `lower_argin1`=3, `lower_argin2`=5, `lower_argout1`=5 |
| the | unobserved; {} | `definite_argin1`; `definite_argin1`=3, `lower_argin1`=1, `lower_argin2`=2, `lower_argout1`=2 |
| white | unobserved; {} | `lower_argin1`; `lower_argin1`=5, `lower_argin2`=5, `lower_argout1`=5 |

## Argument occupant cosines

The [complete collapse table](collapse.json) contains every argument pair. Only nonzero entries are shown here. Coincidences marked **head** follow the same head being counted on an input and output; they do not establish learned output typing.

| Argument A | Argument B | Cosine | Construction |
| --- | --- | ---: | --- |
| `definite_argin1` | `intersection_argin1` | 0.176777 |  |
| `definite_argin1` | `intersection_argin2` | 0.500000 |  |
| `definite_argin1` | `lower_argin1` | 0.696311 |  |
| `definite_argin1` | `lower_argin2` | 0.089087 |  |
| `definite_argin1` | `lower_argout1` | 0.089087 |  |
| `definite_argin2` | `definite_argout1` | 1.000000 | head |
| `definite_argin2` | `intersection_argin1` | 0.072169 |  |
| `definite_argin2` | `intersection_argin2` | 0.272166 |  |
| `definite_argin2` | `lower_argin1` | 0.118445 |  |
| `definite_argin2` | `lower_argin2` | 0.672838 |  |
| `definite_argin2` | `lower_argout1` | 0.672838 |  |
| `definite_argout1` | `intersection_argin1` | 0.072169 |  |
| `definite_argout1` | `intersection_argin2` | 0.272166 |  |
| `definite_argout1` | `lower_argin1` | 0.118445 |  |
| `definite_argout1` | `lower_argin2` | 0.672838 |  |
| `definite_argout1` | `lower_argout1` | 0.672838 |  |
| `intersection_argin1` | `intersection_argin2` | 0.235702 |  |
| `intersection_argin1` | `lower_argin1` | 0.400047 |  |
| `intersection_argin1` | `lower_argin2` | 0.551198 |  |
| `intersection_argin1` | `lower_argout1` | 0.551198 |  |
| `intersection_argin2` | `lower_argin1` | 0.725324 |  |
| `intersection_argin2` | `lower_argin2` | 0.445435 |  |
| `intersection_argin2` | `lower_argout1` | 0.445435 |  |
| `lower_argin1` | `lower_argin2` | 0.240375 |  |
| `lower_argin1` | `lower_argout1` | 0.240375 |  |
| `lower_argin2` | `lower_argout1` | 1.000000 | head |

## Recorded noun prior

These are direct concept-to-individual-column ties, not conceptual similarity. All are absent before and after this pass; the chooser never reads this table. Empty means no tie, not a proof that a word is not a noun.

| Word | Before concept IDs | After concept IDs | Individual columns (before → after) |
| --- | --- | --- | --- |
| a | [] | [239] | [] → [] |
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
| the | [] | [245] | [] → [] |
| white | [46] | [46] | [] → [] |

## Trajectory

Global blocks of eight training batches, with a final two-batch remainder. CE is observed pre-update CE on reachable student states; changing states and lesson mix prevent a matched-state learning claim. Reused prefixes count in each loss call.

| Batches | Calls | Fold targets / call | Wait CE (reachable / unreachable) | Fold CE (reachable / unreachable) |
| --- | ---: | ---: | ---: | ---: |
| 1–8 | 56 | 22/56 = 0.392857 | 0.749248 (266 / 0) | 2.016231 (22 / 0) |
| 9–10 | 16 | 12/16 = 0.750000 | 0.506534 (84 / 0) | 1.480456 (12 / 0) |

First-pair accuracy on kept readings. Missing pairs count as incorrect.

| Batches | Lower bare NP | Definite bare NP | Nested adjective NP |
| --- | ---: | ---: | ---: |
| 1–8 | 3/6 (0 unreached) | 3/6 (2 unreached) | 0/16 (2 unreached) |
| 9–10 | — | — | 3/8 (2 unreached) |

Every binary, unary-face or transpose compose step pays `WHAT_STEP_COST` in the keep comparison.

| Batches | Sentences | Leaves | Binary | Unary | Transpose | All steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–8 | 28 | 72 | 44 | 0 | 0 | 116 |
| 9–10 | 8 | 24 | 16 | 0 | 0 | 40 |
