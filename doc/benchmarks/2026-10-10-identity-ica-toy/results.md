# Identity-by-unmixing toy: full tables (written by sim.py)

## Q1 factorial vs confounded
| codes | corpus | learner | recovery | columns | one-atom columns | identification | o0 / p0 | o0+p0 column | merged |
|---|---|---|---|---|---|---|---|---|---|
| dense | A | code: mint=value, top-k, k=3 | 0.88 ± 0.01 | 456 | 0.02 ± 0.01 | 0.02 ± 0.01 | 0.83 / 0.86 | 0.90 ± 0.08 | 2/5 |
| dense | A | variant: mint=residual, top-k, k=3 | 0.95 ± 0.00 | 177 | 0.22 ± 0.03 | 0.32 ± 0.03 | 0.96 / 0.95 | 0.95 ± 0.04 | 0/5 |
| dense | A | variant: mint=residual, greedy, k=3 | 0.98 ± 0.00 | 66 | 0.40 ± 0.12 | 0.95 ± 0.03 | 0.98 / 0.98 | 0.71 ± 0.03 | 0/5 |
| dense | A | variant: mint=residual, top-k, k=4 | 0.98 ± 0.00 | 33 | 0.76 ± 0.14 | 0.97 ± 0.04 | 0.98 / 0.98 | 0.69 ± 0.05 | 0/5 |
| dense | A | ref: FastICA (22 given) | 0.70 ± 0.09 | 22 | 0.36 ± 0.23 | 0.31 ± 0.20 | 0.67 / 0.76 | 0.57 ± 0.10 | 0/5 |
| dense | A | ref: DictLearning (22 given) | 0.92 ± 0.02 | 22 | 0.87 ± 0.08 | 0.84 ± 0.08 | 0.85 / 0.93 | 0.71 ± 0.04 | 0/5 |
| dense | B1 | code: mint=value, top-k, k=3 | 0.86 ± 0.01 | 419 | 0.02 ± 0.00 | 0.02 ± 0.01 | 0.77 / 0.74 | 0.96 ± 0.04 | 3/5 |
| dense | B1 | variant: mint=residual, top-k, k=3 | 0.94 ± 0.00 | 177 | 0.18 ± 0.01 | 0.29 ± 0.03 | 0.81 / 0.82 | 0.98 ± 0.00 | 5/5 |
| dense | B1 | variant: mint=residual, greedy, k=3 | 0.96 ± 0.00 | 45 | 0.48 ± 0.09 | 0.90 ± 0.03 | 0.75 / 0.84 | 0.99 ± 0.00 | 3/5 |
| dense | B1 | variant: mint=residual, top-k, k=4 | 0.96 ± 0.01 | 55 | 0.48 ± 0.10 | 0.80 ± 0.13 | 0.84 / 0.84 | 0.98 ± 0.00 | 4/5 |
| dense | B1 | ref: FastICA (22 given) | 0.83 ± 0.01 | 22 | 0.55 ± 0.07 | 0.48 ± 0.06 | 0.73 / 0.72 | 0.93 ± 0.01 | 5/5 |
| dense | B1 | ref: DictLearning (22 given) | 0.90 ± 0.02 | 22 | 0.81 ± 0.07 | 0.77 ± 0.09 | 0.67 / 0.65 | 0.97 ± 0.01 | 5/5 |
| dense | B2 | code: mint=value, top-k, k=3 | 0.88 ± 0.01 | 430 | 0.03 ± 0.00 | 0.01 ± 0.00 | 0.89 / 0.81 | 0.95 ± 0.02 | 3/5 |
| dense | B2 | variant: mint=residual, top-k, k=3 | 0.96 ± 0.00 | 175 | 0.23 ± 0.02 | 0.31 ± 0.02 | 0.97 / 0.90 | 0.95 ± 0.01 | 0/5 |
| dense | B2 | variant: mint=residual, greedy, k=3 | 0.97 ± 0.00 | 59 | 0.41 ± 0.11 | 0.92 ± 0.05 | 0.98 / 0.70 | 0.99 ± 0.00 | 0/5 |
| dense | B2 | variant: mint=residual, top-k, k=4 | 0.97 ± 0.01 | 29 | 0.75 ± 0.10 | 0.96 ± 0.02 | 0.98 / 0.77 | 0.98 ± 0.00 | 0/5 |
| dense | B2 | ref: FastICA (22 given) | 0.85 ± 0.02 | 22 | 0.62 ± 0.11 | 0.54 ± 0.09 | 0.93 / 0.65 | 0.93 ± 0.01 | 0/5 |
| dense | B2 | ref: DictLearning (22 given) | 0.92 ± 0.01 | 22 | 0.85 ± 0.04 | 0.83 ± 0.05 | 0.87 / 0.58 | 0.96 ± 0.01 | 3/5 |
| sparse | A | code: mint=value, top-k, k=3 | 0.90 ± 0.02 | 452 | 0.03 ± 0.01 | 0.02 ± 0.01 | 0.92 / 0.92 | 0.96 ± 0.02 | 1/5 |
| sparse | A | variant: mint=residual, top-k, k=3 | 0.95 ± 0.00 | 181 | 0.20 ± 0.01 | 0.29 ± 0.02 | 0.97 / 0.96 | 0.86 ± 0.10 | 0/5 |
| sparse | A | variant: mint=residual, greedy, k=3 | 0.98 ± 0.00 | 63 | 0.39 ± 0.13 | 0.96 ± 0.02 | 0.98 / 0.98 | 0.73 ± 0.03 | 0/5 |
| sparse | A | variant: mint=residual, top-k, k=4 | 0.97 ± 0.01 | 46 | 0.63 ± 0.18 | 0.90 ± 0.15 | 0.98 / 0.97 | 0.74 ± 0.02 | 0/5 |
| sparse | B1 | code: mint=value, top-k, k=3 | 0.87 ± 0.02 | 404 | 0.02 ± 0.01 | 0.01 ± 0.01 | 0.77 / 0.77 | 0.98 ± 0.03 | 5/5 |
| sparse | B1 | variant: mint=residual, top-k, k=3 | 0.94 ± 0.00 | 184 | 0.16 ± 0.02 | 0.30 ± 0.02 | 0.78 / 0.81 | 0.98 ± 0.01 | 5/5 |
| sparse | B1 | variant: mint=residual, greedy, k=3 | 0.96 ± 0.01 | 88 | 0.26 ± 0.05 | 0.85 ± 0.04 | 0.81 / 0.76 | 0.99 ± 0.00 | 4/5 |
| sparse | B1 | variant: mint=residual, top-k, k=4 | 0.96 ± 0.00 | 34 | 0.60 ± 0.05 | 0.91 ± 0.01 | 0.83 / 0.81 | 0.99 ± 0.00 | 5/5 |
| sparse | B2 | code: mint=value, top-k, k=3 | 0.88 ± 0.00 | 431 | 0.02 ± 0.00 | 0.01 ± 0.00 | 0.90 / 0.80 | 0.96 ± 0.04 | 3/5 |
| sparse | B2 | variant: mint=residual, top-k, k=3 | 0.95 ± 0.01 | 182 | 0.20 ± 0.01 | 0.29 ± 0.03 | 0.96 / 0.88 | 0.92 ± 0.04 | 0/5 |
| sparse | B2 | variant: mint=residual, greedy, k=3 | 0.96 ± 0.02 | 73 | 0.35 ± 0.16 | 0.91 ± 0.03 | 0.97 / 0.78 | 0.99 ± 0.00 | 0/5 |
| sparse | B2 | variant: mint=residual, top-k, k=4 | 0.97 ± 0.00 | 43 | 0.61 ± 0.14 | 0.91 ± 0.08 | 0.98 / 0.77 | 0.99 ± 0.00 | 0/5 |

FastICA mean recovery: one per role (Q1 corpus A) 0.70 ± 0.09, independent presence 1.00 ± 0.00

## Q2 support curriculum (identification / recovery (columns))
| curriculum | learner | n=250 | n=500 | n=1000 | n=2000 | n=3000 |
|---|---|---|---|---|---|---|
| staged 1->2->3 | code: mint=value, top-k, k=3 | 0.99±0.01 / 1.00 (22) | 0.99±0.00 / 1.00 (22) | 0.98±0.02 / 1.00 (23) | 0.99±0.01 / 1.00 (23) | 0.94±0.11 / 1.00 (27) |
| staged 1->2->3 | variant: mint=residual, top-k, k=3 | 0.99±0.00 / 1.00 (22) | 0.99±0.00 / 1.00 (22) | 0.99±0.00 / 1.00 (22) | 0.99±0.00 / 1.00 (22) | 0.82±0.19 / 1.00 (42) |
| staged 1->2->3 | variant: mint=residual, greedy, k=3 | 0.99±0.02 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) |
| staged 1->2->3 | variant: mint=residual, top-k, k=4 | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) | 1.00±0.00 / 1.00 (22) |
| mixed 1-3 | code: mint=value, top-k, k=3 | 0.62±0.06 / 0.85 (19) | 0.83±0.06 / 1.00 (32) | 0.71±0.08 / 1.00 (49) | 0.44±0.08 / 1.00 (115) | 0.27±0.05 / 1.00 (201) |
| mixed 1-3 | variant: mint=residual, top-k, k=3 | 0.80±0.11 / 0.93 (20) | 0.97±0.02 / 1.00 (24) | 0.96±0.05 / 1.00 (24) | 0.83±0.17 / 1.00 (42) | 0.66±0.18 / 0.99 (92) |
| mixed 1-3 | variant: mint=residual, greedy, k=3 | 0.81±0.09 / 0.93 (20) | 1.00±0.01 / 1.00 (24) | 1.00±0.00 / 1.00 (24) | 1.00±0.00 / 1.00 (24) | 1.00±0.00 / 1.00 (24) |
| mixed 1-3 | variant: mint=residual, top-k, k=4 | 0.71±0.13 / 0.89 (20) | 1.00±0.00 / 1.00 (24) | 1.00±0.00 / 1.00 (24) | 1.00±0.00 / 1.00 (24) | 1.00±0.00 / 1.00 (24) |
| three only | code: mint=value, top-k, k=3 | 0.00±0.00 / 0.09 (0) | 0.06±0.07 / 0.58 (9) | 0.70±0.08 / 0.99 (39) | 0.25±0.05 / 0.97 (169) | 0.07±0.01 / 0.95 (328) |
| three only | variant: mint=residual, top-k, k=3 | 0.00±0.00 / 0.09 (0) | 0.00±0.00 / 0.37 (3) | 0.12±0.08 / 0.65 (10) | 0.57±0.07 / 0.98 (86) | 0.40±0.03 / 0.98 (187) |
| three only | variant: mint=residual, greedy, k=3 | 0.00±0.00 / 0.09 (0) | 0.00±0.00 / 0.36 (3) | 0.11±0.08 / 0.63 (9) | 0.97±0.01 / 1.00 (32) | 0.97±0.01 / 1.00 (34) |
| three only | variant: mint=residual, top-k, k=4 | 0.00±0.00 / 0.09 (0) | 0.00±0.00 / 0.36 (3) | 0.05±0.06 / 0.62 (9) | 1.00±0.00 / 1.00 (25) | 1.00±0.00 / 1.00 (26) |

| support (sources per row) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| residual under the true atoms | 0.094 | 0.106 | 0.138 | 0.223 | 0.346 |
| rows above tau = .2 | 0.000 | 0.001 | 0.033 | 0.144 | 0.305 |

| staged corpus, then "the X V the Y ." | learner | columns (per seed) | one-atom columns | identification |
|---|---|---|---|---|
| verbs absent from stage 2 | code: mint=value, top-k, k=3 | 187 (191, 195, 181, 190, 179) | 0.15 ± 0.02 | 0.04 ± 0.01 |
| verbs absent from stage 2 | variant: mint=residual, greedy, k=3 | 32 (36, 30, 38, 26, 30) | 0.58 ± 0.07 | 0.99 ± 0.01 |
| verbs absent from stage 2 | variant: mint=residual, top-k, k=4 | 54 (64, 38, 59, 79, 28) | 0.69 ± 0.11 | 0.80 ± 0.14 |
| verbs kept in stage 2 | code: mint=value, top-k, k=3 | 84 (26, 144, 90, 30, 132) | 0.52 ± 0.34 | 0.44 ± 0.40 |
| verbs kept in stage 2 | variant: mint=residual, greedy, k=3 | 26 (26, 27, 26, 26, 26) | 0.97 ± 0.04 | 1.00 ± 0.00 |
| verbs kept in stage 2 | variant: mint=residual, top-k, k=4 | 32 (27, 27, 45, 26, 34) | 0.87 ± 0.12 | 0.96 ± 0.05 |

## Q3 pronoun binding credited by prediction
| credit | training data | held-out CB (strong / weak) | reversed recency | reversed subject | control: picks recent | control: picks subject | w content | w recent | w subject |
|---|---|---|---|---|---|---|---|---|---|
| prediction | counterbalanced | 1.00 / 0.88 | 1.00 / 0.90 | 1.00 / 0.88 | 0.49 ± 0.14 | 0.44 ± 0.27 | 9.89 ± 0.08 | -0.04 ± 0.73 | -0.17 ± 1.36 |
| prediction | recency-biased | 0.79 / 0.58 | 0.59 / 0.19 | 0.78 / 0.66 | 0.92 ± 0.14 | 0.40 ± 0.18 | 5.42 ± 1.53 | 3.30 ± 1.48 | -0.72 ± 1.29 |
| prediction | subject-biased | 0.86 / 0.65 | 0.81 / 0.54 | 0.59 / 0.18 | 0.62 ± 0.08 | 0.98 ± 0.02 | 4.58 ± 0.63 | 0.92 ± 0.55 | 3.23 ± 1.31 |
| prediction, lr .1 | counterbalanced | 1.00 / 0.93 | 1.00 / 0.93 | 1.00 / 0.96 | 0.55 ± 0.10 | 0.44 ± 0.08 | 6.06 ± 0.15 | 0.09 ± 0.23 | -0.14 ± 0.24 |
| reconstruction: learned | counterbalanced | 1.00 / 0.92 | 1.00 / 0.93 | 1.00 / 0.97 | 0.51 ± 0.13 | 0.41 ± 0.09 | 2.74 ± 0.41 | -0.00 ± 0.17 | -0.11 ± 0.12 |
| reconstruction: true atoms | counterbalanced | 0.57 / 0.51 | 0.40 / 0.37 | 0.80 / 0.66 | 0.66 ± 0.42 | 0.39 ± 0.20 | 0.12 ± 0.10 | 0.13 ± 0.21 | -0.05 ± 0.08 |

dictionary: 26 ± 1 columns, one-atom 0.99 ± 0.01; content margin antecedent - other: strong 0.79 ± 0.03, weak 0.26 ± 0.01; reconstruction credit higher for the antecedent: learned 0.87 ± 0.02, true atoms 0.50 ± 0.02

## Q4 same-kind individuals and determiners (accuracy against the true identity)
| chooser | ordinary | "a dog . a dog" (2) | "a dog . the dog" (1) | conflict: "a black dog . the white dog" (2) | new info: "a dog . the black dog" (1) | "a black dog . a black dog" (2) | w the | w residual | w exclusivity | w bias |
|---|---|---|---|---|---|---|---|---|---|---|
| content only: bind iff residual <= tau | 0.64 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | - | - | - | - |
| credit, det + residual | 0.89 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.09 ± 0.18 | 0.91 ± 0.17 | 1.00 ± 0.00 | 6.97 ± 0.68 | -3.02 ± 0.61 | 0.00 ± 0.00 | -2.18 ± 0.70 |
| labels (reference), det + residual | 0.91 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 3.98 ± 0.06 | -1.12 ± 0.07 | 0.00 ± 0.00 | -1.23 ± 0.08 |
| credit, det + residual + learned exclusivity | 0.92 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 6.11 ± 0.48 | -1.96 ± 0.68 | -4.17 ± 0.49 | -1.70 ± 0.23 |
| labels (reference), det + residual + learned exclusivity | 0.92 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.87 ± 0.12 | 1.00 ± 0.00 | 1.00 ± 0.00 | 3.80 ± 0.05 | -0.77 ± 0.08 | -2.12 ± 0.07 | -1.15 ± 0.08 |

| two dogs in STM, then "the ... dog V ." | bound to the right dog |
|---|---|
| named again | 1.00 ± 0.00 |
| not named again | 0.50 ± 0.01 |
| never named | 0.48 ± 0.01 |

second-sentence row explained by the kind column (no mint possible): 1.00 ± 0.00

(5 seeds, 219 s)
