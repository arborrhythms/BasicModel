# MM_grammar: final ten full runs per tree

All ten unseeded trials are reported. Each runs the unchanged configuration for 900 epochs under the 8 GiB guard. Rows show run order; the two trees do not share matched initializations.

| Trial | HEAD ending MSE | Candidate ending MSE | HEAD after 900 updates | Candidate after 900 updates | HEAD GiB | Candidate GiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.928066761e-10 | 0.0424849689 | 3.280895555e-10 | 0.04228069633 | 7.646 | 0.552 |
| 2 | 0.1131030023 | 0.1433953047 | 0.1130752414 | 0.1432686895 | 7.642 | 0.581 |
| 3 | 4.630707053e-07 | 0.2445593774 | 4.482241422e-07 | 0.2448780984 | 7.642 | 0.545 |
| 4 | 0.0261284411 | 0.1251982152 | 0.02440025657 | 0.1249466687 | 7.641 | 0.571 |
| 5 | 0.3215630352 | 0.3362744749 | 0.3236429989 | 0.3349574506 | 7.642 | 0.628 |
| 6 | 4.572948953e-08 | 0.1937815845 | 4.421038113e-08 | 0.1937020719 | 7.641 | 0.642 |
| 7 | 0.2485499084 | 0.1866282374 | 0.2485503256 | 0.1862090081 | 7.642 | 0.612 |
| 8 | 0.1586389393 | 0.08990462124 | 0.158559218 | 0.0898129791 | 7.641 | 0.566 |
| 9 | 0.1872016788 | 0.06369277835 | 0.1867669374 | 0.06351113319 | 7.641 | 0.556 |
| 10 | 5.35901836e-06 | 0.002839832334 | 5.260485523e-06 | 0.002896032063 | 7.639 | 0.582 |

Both trees complete 10/10. Median ending MSE: HEAD 0.0696157217; candidate 0.13429676.

Ending MSE is the training error before the last update; the adjacent column measures after that update. No threshold or baseline is changed.

[HEAD raw trials](final-mm-grammar-head/processes.json); [candidate raw trials](final3-mm-grammar-candidate/processes.json).
