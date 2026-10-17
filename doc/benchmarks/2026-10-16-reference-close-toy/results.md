# Results: reference at the close (toy), full tables

Seeds [0, 1, 2, 3, 4]; elapsed 53s; defaults `{"rho_c": 0.3, "rho_p": 0.5, "pronoun_can_mint": false, "lam_assoc": 10.0, "forget": 0.9993, "atom_floor": 0.3, "nu": 0.5, "exclusive": true, "excl_min": 8, "eta_w": 0.2, "beta": 1.0, "lam_next": 1.0, "kappa": 6.0, "alpha_credit": 1.0, "explore": 0.05, "soft_assoc": true, "beta_assoc": 3.0, "decay": 0.0001, "epochs": 2, "recurrence": 0, "close": true, "pred_in_credit": true, "learn_salience": true, "learn_detcue": true, "fix_det_a": null, "noise": 0.0, "det_flip": 0.0}`

## Q1 Binding accuracy by family (eval split; probe for probe families, the final `it` for determiner families, the two `the colour kind` mentions for same_kind)

| family | n | proposal | B1 code rule | B2 recency | B3 during compose |
|---|---|---|---|---|---|
| pronoun_one | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| pronoun | 32 | 1.00 | 0.50 | 0.25 | 0.50 |
| pronoun_three | 54 | 1.00 | 0.33 | 0.11 | 0.33 |
| document_context | 8 | 1.00 | 0.50 | 0.50 | 0.50 |
| neither_candidate | 32 | 0.50 | 0.50 | 0.25 | 0.50 |
| determiner (mentions) | 16 | 1.00 | 1.00 | 1.00 | 0.95 ± 0.10 |
| determiner_conflict (mentions) | 8 | 1.00 | 0.50 | 0.50 | 0.90 ± 0.20 |
| same_kind (mentions) | 24 | 0.67 | 0.67 | 0.83 | 0.67 |
| all probes | 130 | 0.88 | 0.45 | 0.23 | 0.45 |
| all graded mentions | 358 | 0.93 | 0.77 | 0.66 | 0.77 ± 0.01 |

By recency rank and role (informative probes only):

| group | n | proposal | B1 code rule | B2 recency | B3 during compose |
|---|---|---|---|---|---|
| candidates=1 | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| candidates=2 | 40 | 1.00 | 0.50 | 0.30 | 0.50 |
| candidates=3 | 54 | 1.00 | 0.33 | 0.11 | 0.33 |
| recency=0 | 42 | 1.00 | 1.00 | 0.52 | 0.82 ± 0.36 |
| recency=1 | 38 | 1.00 | 0.00 | 0.00 | 0.14 ± 0.27 |
| recency=2 | 18 | 1.00 | 0.00 | 0.00 | 0.13 ± 0.27 |
| role=subject | 64 | 1.00 | 0.44 | 0.22 | 0.42 ± 0.04 |
| role=object | 34 | 1.00 | 0.41 | 0.24 | 0.45 ± 0.07 |

## Q2 Row pairs (one vs two individuals), eval split

| family / expected rows | n | proposal | B1 code rule | B2 recency | B3 during compose | variant flat prior nu=1 | abl no exclusivity | proposal det flip .2 |
|---|---|---|---|---|---|---|---|---|
| determiner / 1 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 ± 0.20 | 1.00 | 1.00 |
| determiner / 2 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| determiner_conflict / 2 | 4 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| same_kind / 1 | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| same_kind / 2 | 4 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| all row pairs | 20 | 0.80 | 0.80 | 0.80 | 0.80 | 0.98 ± 0.04 | 0.60 | 0.80 |
| individuals per document | | 2.28 | 2.25 | 2.25 | 2.28 | 2.36 ± 0.01 | 2.23 | 2.30 ± 0.01 |

## Q3 Reversed recency

| model | reversed_eval acc | eval pronoun acc | learned recency w | log α | det_pro |
|---|---|---|---|---|---|
| proposal | 1.00 | 1.00 | -0.00 ± 0.01 | 1.17 ± 0.05 | 0.00 |
| branch balanced | 1.00 | 1.00 | -0.02 ± 0.01 | 1.08 ± 0.03 | 0.00 |
| branch biased | 1.00 | 1.00 | -0.05 ± 0.02 | 1.09 ± 0.04 | 0.00 |
| branch biased x4 | 1.00 | 1.00 | -0.09 ± 0.01 | 1.12 ± 0.03 | 0.00 |
| branch balanced (B3) | 0.10 ± 0.20 | 0.50 | 0.16 ± 0.10 | 0.99 ± 0.03 | 0.00 |
| branch biased (B3) | 0.00 | 0.50 | 1.54 ± 0.08 | 0.81 ± 0.04 | 0.00 |
| B1 code rule | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| B2 recency | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 |
| B3 during compose | 0.20 ± 0.40 | 0.50 | 0.32 ± 0.20 | 1.08 ± 0.04 | 0.00 |

## Q4 Learned weights (mean ± sd over seeds) and the recurrence gate

| run | compat | pred | recency | last_role | cur_role | log_alpha | det_a | det_the | det_pro |
|---|---|---|---|---|---|---|---|---|---|
| proposal | 3.94 ± 0.06 | 1.00 ± 0.01 | -0.00 ± 0.01 | -0.09 ± 0.02 | -0.36 ± 0.04 | 1.17 ± 0.05 | 1.25 ± 0.03 | -0.08 ± 0.06 | 0.00 |
| proposal 1 epoch | 3.46 ± 0.05 | 1.03 ± 0.03 | 0.01 ± 0.01 | -0.13 ± 0.05 | -0.28 ± 0.05 | 1.00 ± 0.10 | 1.31 ± 0.02 | -0.31 ± 0.10 | 0.00 |
| proposal 3 epochs | 4.18 ± 0.04 | 1.01 ± 0.01 | -0.01 ± 0.02 | -0.09 ± 0.02 | -0.41 ± 0.01 | 1.27 ± 0.04 | 1.20 ± 0.02 | 0.07 ± 0.05 | 0.00 |
| B3 during compose | 4.15 ± 0.04 | 0.00 | 0.32 ± 0.20 | -0.21 ± 0.11 | -0.31 ± 0.04 | 1.08 ± 0.04 | 1.23 ± 0.04 | -0.15 ± 0.06 | 0.00 |
| branch balanced | 3.91 ± 0.03 | 1.00 ± 0.01 | -0.02 ± 0.01 | -0.04 ± 0.03 | -0.31 ± 0.06 | 1.08 ± 0.03 | 1.24 ± 0.01 | -0.16 ± 0.03 | 0.00 |
| branch biased | 3.90 ± 0.04 | 1.03 ± 0.01 | -0.05 ± 0.02 | -0.06 ± 0.04 | -0.31 ± 0.08 | 1.09 ± 0.04 | 1.24 ± 0.01 | -0.15 ± 0.04 | 0.00 |
| branch biased x4 | 3.97 ± 0.04 | 1.01 ± 0.01 | -0.09 ± 0.01 | 0.06 ± 0.03 | -0.30 ± 0.05 | 1.12 ± 0.03 | 1.19 ± 0.01 | -0.07 ± 0.02 | 0.00 |
| abl no prediction | 3.96 ± 0.05 | 0.00 | -0.00 | -0.05 ± 0.02 | -0.33 ± 0.04 | 1.15 ± 0.04 | 1.20 ± 0.03 | -0.05 ± 0.06 | 0.00 |
| abl no salience | 3.92 ± 0.06 | 1.00 ± 0.01 | 0.00 | 0.00 | 0.00 | 1.30 ± 0.05 | 1.32 ± 0.03 | -0.02 ± 0.07 | 0.00 |
| abl no det cue | 4.51 ± 0.02 | 0.99 | 0.01 ± 0.01 | -0.18 ± 0.03 | -0.50 ± 0.04 | 1.67 ± 0.04 | 0.00 | 0.00 | 0.00 |
| abl no next-sentence credit | 3.90 ± 0.05 | 1.01 ± 0.01 | -0.00 ± 0.01 | -0.09 ± 0.02 | -0.36 ± 0.03 | 1.18 ± 0.04 | 1.20 ± 0.02 | -0.02 ± 0.06 | 0.00 |
| abl no predicate absorption | 3.97 ± 0.05 | 0.99 | -0.01 ± 0.01 | -0.19 ± 0.03 | -0.31 ± 0.03 | 1.13 ± 0.04 | 1.20 ± 0.03 | -0.07 ± 0.06 | 0.00 |
| abl no exclusivity | 4.22 ± 0.04 | 1.02 | 0.02 ± 0.01 | -0.09 ± 0.02 | -0.07 ± 0.01 | 1.25 ± 0.02 | 1.25 ± 0.02 | -0.01 ± 0.02 | 0.00 |
| abl hard association | 3.96 ± 0.05 | 1.00 ± 0.01 | -0.00 ± 0.01 | -0.09 ± 0.02 | -0.36 ± 0.04 | 1.18 ± 0.05 | 1.26 ± 0.04 | -0.08 ± 0.07 | 0.00 |
| abl pronoun may mint | 3.86 ± 0.06 | 1.02 ± 0.02 | -0.02 ± 0.01 | -0.05 ± 0.02 | -0.20 ± 0.03 | 0.88 ± 0.01 | 1.58 ± 0.02 | 0.24 ± 0.04 | -0.94 ± 0.02 |
| abl no exploration | 3.81 ± 0.03 | 1.02 ± 0.02 | -0.01 ± 0.01 | -0.19 ± 0.03 | -0.36 ± 0.02 | 1.17 ± 0.02 | 1.31 ± 0.03 | -0.14 ± 0.02 | 0.00 |
| diag det_a fixed 3 | 3.76 ± 0.08 | 1.01 ± 0.01 | -0.02 ± 0.01 | -0.00 ± 0.02 | -0.02 ± 0.03 | 0.18 ± 0.07 | 3.00 | 0.94 ± 0.10 | 0.00 |
| variant flat prior nu=1 | 2.98 ± 0.01 | 1.01 ± 0.01 | 0.04 ± 0.01 | -0.09 ± 0.01 | -0.68 ± 0.02 | 1.89 ± 0.02 | 1.25 ± 0.01 | 0.63 ± 0.02 | 0.00 |
| variant nu=1 no exclusivity | 3.51 ± 0.02 | 1.01 ± 0.01 | 0.04 ± 0.02 | -0.08 ± 0.01 | -0.62 ± 0.02 | 2.49 ± 0.02 | 1.88 ± 0.01 | 0.61 ± 0.01 | 0.00 |
| variant nu=1 no det cue | 3.15 | 1.00 | 0.02 ± 0.01 | -0.12 ± 0.01 | -0.89 ± 0.03 | 2.61 ± 0.02 | 0.00 | 0.00 | 0.00 |
| variant nu=1 no next-sentence credit | 2.93 ± 0.01 | 1.01 ± 0.01 | 0.03 ± 0.01 | -0.09 ± 0.01 | -0.69 ± 0.02 | 1.91 ± 0.02 | 1.25 ± 0.01 | 0.66 ± 0.01 | 0.00 |
| variant nu=1 det flip .2 | 3.19 ± 0.01 | 1.01 ± 0.02 | 0.02 ± 0.01 | -0.08 ± 0.02 | -0.57 ± 0.05 | 1.90 ± 0.02 | 1.00 ± 0.01 | 0.90 ± 0.02 | 0.00 |
| variant nu=1 noise 1.0 | 3.45 ± 0.05 | 0.99 | 0.02 ± 0.04 | -0.06 ± 0.03 | -0.63 ± 0.04 | 2.56 ± 0.06 | 1.86 ± 0.05 | 0.70 ± 0.05 | 0.00 |
| proposal det flip .2 | 4.42 ± 0.03 | 1.01 ± 0.02 | 0.01 ± 0.01 | -0.14 ± 0.03 | -0.29 ± 0.05 | 1.26 ± 0.02 | 0.85 ± 0.03 | 0.41 ± 0.04 | 0.00 |

| run | individuals/doc | rows (durable)/doc | probes strict | probes lenient | row pairs strict | row pairs lenient |
|---|---|---|---|---|---|---|
| proposal | 2.28 | 2.28 | 0.88 | 0.88 | 0.80 | 0.80 |
| proposal gate r=2 | 2.28 | 1.54 ± 0.04 | 0.26 ± 0.02 | 0.88 | 0.40 | 0.80 |
| proposal gate r=3 | 2.28 | 0.33 ± 0.04 | 0.00 | 0.88 | 0.40 | 0.80 |
| B1 code rule | 2.25 | 2.25 | 0.45 | 0.45 | 0.80 | 0.80 |
| B1 gate r=2 | 2.25 | 1.03 | 0.03 | 0.45 | 0.60 | 0.80 |

## Q5 Robustness

| condition | model | probes | pronoun | pronoun_three | document_context | row pairs | det. mentions |
|---|---|---|---|---|---|---|---|
| proposal | proposal | 0.88 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |
| B1 code rule | B1 | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 1.00 |
| proposal noise 0.3 | proposal | 0.87 ± 0.02 | 1.00 | 0.98 ± 0.04 | 1.00 | 0.80 | 1.00 |
| B1 noise 0.3 | B1 | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 1.00 |
| proposal noise 0.6 | proposal | 0.87 ± 0.02 | 1.00 | 0.98 ± 0.04 | 1.00 | 0.76 ± 0.08 | 1.00 |
| B1 noise 0.6 | B1 | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 1.00 |
| proposal noise 1.0 | proposal | 0.84 ± 0.02 | 0.94 ± 0.04 | 0.96 ± 0.04 | 0.95 ± 0.10 | 0.81 ± 0.02 | 1.00 |
| B1 noise 1.0 | B1 | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 1.00 |
| variant nu=1 noise 1.0 | proposal | 0.73 ± 0.04 | 0.78 ± 0.07 | 0.87 ± 0.03 | 0.93 ± 0.10 | 0.66 ± 0.07 | 0.64 ± 0.08 |
| proposal det flip .2 | proposal | 0.88 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 |
| B1 det flip .2 | B1 | 0.36 ± 0.02 | 0.46 ± 0.04 | 0.20 ± 0.02 | 0.33 ± 0.06 | 0.64 ± 0.07 | 0.86 ± 0.07 |
| B3 det flip .2 | proposal | 0.45 | 0.50 | 0.33 | 0.50 | 0.80 | 0.90 ± 0.12 |
| variant nu=1 det flip .2 | proposal | 0.88 | 1.00 | 1.00 | 1.00 | 0.78 ± 0.02 | 0.97 ± 0.03 |

## Q6 Ablations and epochs

| run | probes | pronoun | pronoun_three | document_context | neither | row pairs | same_kind rows | reversed_eval |
|---|---|---|---|---|---|---|---|---|
| proposal | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| proposal 1 epoch | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| proposal 3 epochs | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| B3 during compose | 0.45 | 0.50 | 0.33 | 0.50 | 0.50 | 0.80 | 0.50 | 0.20 ± 0.40 |
| abl no prediction | 0.45 | 0.50 | 0.33 | 0.50 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl no salience | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl no det cue | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl no next-sentence credit | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl no predicate absorption | 0.85 | 1.00 | 1.00 | 0.50 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl no exclusivity | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.60 | 0.50 | 1.00 |
| abl hard association | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| abl pronoun may mint | 0.80 ± 0.02 | 1.00 | 1.00 | 0.90 ± 0.09 | 0.20 ± 0.10 | 0.80 | 0.50 | 1.00 |
| abl no exploration | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| diag det_a fixed 3 | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 0.50 | 1.00 |
| variant flat prior nu=1 | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.98 ± 0.04 | 1.00 | 1.00 |
| variant nu=1 no exclusivity | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.98 ± 0.04 | 1.00 | 1.00 |
| variant nu=1 no det cue | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.76 ± 0.08 | 0.50 | 1.00 |
| variant nu=1 no next-sentence credit | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.80 | 1.00 | 1.00 |
| variant nu=1 det flip .2 | 0.88 | 1.00 | 1.00 | 1.00 | 0.50 | 0.78 ± 0.02 | 0.50 | 1.00 |
| variant nu=1 noise 1.0 | 0.73 ± 0.04 | 0.78 ± 0.07 | 0.87 ± 0.03 | 0.93 ± 0.10 | 0.36 ± 0.08 | 0.66 ± 0.07 | 0.62 ± 0.19 | 1.00 |
