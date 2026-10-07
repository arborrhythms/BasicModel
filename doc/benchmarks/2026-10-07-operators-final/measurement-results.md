# Operators final: all thirty trainings

Each row is its one declared, unseeded run. No retry or replacement.

For sum, the Control column is the additive criterion. The separate required
¼ band passes 9/10: sum-08 misses at .4929095507. The full-sweep summary in
`results-validation.json` counts raw subtest reports; the receipt README gives
the 5,341 unique-case counts.

| Run | MSE (MM: best) | Class | Reconstruction | Control | Rows/capacity | Sentence/DEF rows | Witnesses per sentence after training |
|---|---:|---|---|---|---:|---:|---|
| sum-01 | 0.2500000894 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-02 | 0.25 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-03 | 0.2500010133 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-04 | 0.2500121295 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-05 | 0.2500004768 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-06 | 0.2500011325 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-07 | 0.2500040233 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-08 | 0.4929095507 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-09 | 0.2500000298 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| sum-10 | 0.25 | FAIL | pass | pass | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-01 | 0.005943967247 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-02 | 1.831630511e-06 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-03 | 2.048139436e-12 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-04 | 3.605310939e-09 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-05 | 4.107516993e-09 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-06 | 0.0008651362868 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-07 | 0.0001073877699 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-08 | 1.439035557e-10 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-09 | 0.001635263431 | pass | FAIL | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| xor-10 | 0.03307267176 | pass | pass | — | 8/1024 | 4/4 | 400, 400, 400, 400 |
| mm-01 | 0.1667948365 | — | — | pass | 4/1024 | 0/4 | — |
| mm-02 | 0.1499582678 | — | — | pass | 4/1024 | 0/4 | — |
| mm-03 | 0.1931855083 | — | — | pass | 4/1024 | 0/4 | — |
| mm-04 | 0.1866156757 | — | — | pass | 4/1024 | 0/4 | — |
| mm-05 | 0.1751848161 | — | — | pass | 4/1024 | 0/4 | — |
| mm-06 | 0.1992654502 | — | — | pass | 4/1024 | 0/4 | — |
| mm-07 | 0.1973546445 | — | — | pass | 4/1024 | 0/4 | — |
| mm-08 | 0.1982398331 | — | — | pass | 4/1024 | 0/4 | — |
| mm-09 | 0.1953415275 | — | — | pass | 4/1024 | 0/4 | — |
| mm-10 | 0.1974080503 | — | — | pass | 4/1024 | 0/4 | — |

Evaluation also re-witnesses the existing addresses. Full final witness counts, timestamps, content keys and addresses are in [validation](results-validation.json).

The unchanged raw MM control has no completed sentence closing and retains only its four DEF rows; no new closing was added for this receipt.
