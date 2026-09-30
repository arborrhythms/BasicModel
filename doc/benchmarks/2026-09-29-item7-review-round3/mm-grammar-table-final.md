# MM_grammar: ten fresh unseeded runs per tree

Every attempt has a 900-epoch budget. A failed process is shown as failed, with no 900-epoch ending measurement. For completed runs, the main error and predictions are the final training forward, before its update, matching the gate's observation point. The raw comparison also reports a separate evaluation after the 900th update. No stopping threshold, seed selection or configuration edit.

| Run | HEAD ending MSE | HEAD predictions | Candidate ending MSE | Candidate predictions | Candidate process |
|---|---:|---|---:|---|---|
| 0 | 0.05476863 | 0.23618, 0.76727, 0.769311, 0.236459 | 0.00000438 | 0.000344753, 0.9969, 0.997213, -0.000136733 | complete |
| 1 | 0.01548596 | 0.124512, 0.877478, 0.873211, 0.123909 | 0.00240416 | 0.0524838, 0.95308, 0.954549, 0.0509399 | complete |
| 2 | 0.59691989 | 0.778598, 0.228326, 0.231382, 0.771499 | unavailable | unavailable | failed (exit 1) |
| 3 | 0.00231121 | 0.0316952, 0.94039, 0.946357, 0.0425359 | 0.00011232 | 0.0145461, 1.002, 0.996887, 0.0149665 | complete |
| 4 | 0.00000113 | 0.00139526, 1.0006, 1.00116, 0.000919133 | 0.00769565 | 0.0879522, 0.912623, 0.912613, 0.0881807 | complete |
| 5 | 0.00000028 | -0.00042659, 0.999454, 0.999449, -0.000565708 | unavailable | unavailable | failed (exit 1) |
| 6 | 0.11029997 | 0.329102, 0.670781, 0.665181, 0.335264 | 0.21557093 | 0.463885, 0.536584, 0.533941, 0.463821 | complete |
| 7 | 0.00006918 | 0.00684676, 1.00925, 1.00835, 0.00863704 | 0.00026029 | 0.0161746, 0.983896, 0.98396, 0.0162146 | complete |
| 8 | 0.24936770 | 0.499263, 0.50079, 0.500535, 0.499531 | 0.00458829 | 0.0676877, 0.932217, 0.932472, 0.0679477 | complete |
| 9 | 0.31322652 | 0.575282, 0.448355, 0.43628, 0.5476 | 0.00537238 | 0.110123, 1.00926, 0.98605, 0.0952997 | complete |

## Failed attempts

Completed-run statistics exclude these unavailable ending measurements; all ten attempts per tree remain in the table and comparison.

- candidate, run 2: RuntimeError: native conceptual row capacity exhausted before clause admission; before the first 50-epoch report.
- candidate, run 5: RuntimeError: native conceptual row capacity exhausted before clause admission; before the first 50-epoch report.

All four targets, post-update predictions, memory use and process results are retained in [the complete comparison](mm-grammar-comparison-final.json).
