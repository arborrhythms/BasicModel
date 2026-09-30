# MM_grammar: ten full 900-epoch runs per tree

Unseeded; unchanged configuration and 8 GiB guard. The error is the training error at the end of the budget.

| Trial | HEAD ending MSE | Candidate after Z ending MSE | HEAD GiB | Candidate GiB |
|---|---:|---:|---:|---:|
| 1 | 0.002242304 | 0.089635707 | 7.642 | 0.595 |
| 2 | 0.081080094 | 0.190566197 | 7.640 | 0.541 |
| 3 | 0.081999771 | 0.001827302 | 7.639 | 0.553 |
| 4 | 0.000101929 | 0.009807453 | 7.638 | 0.542 |
| 5 | 0.057719693 | 0.023795001 | 7.641 | 0.701 |
| 6 | 0.045061179 | 0.022707153 | 7.640 | 0.600 |
| 7 | 0.033048313 | 0.011003243 | 7.641 | 0.701 |
| 8 | 0.245104656 | 0.000000398 | 7.641 | 0.597 |
| 9 | 0.001478278 | 0.116843097 | 7.641 | 0.624 |
| 10 | 0.033249654 | 0.349332631 | 7.642 | 0.699 |

Both trees complete 10/10. Median ending MSE: HEAD 0.039155416; candidate 0.023251077.
