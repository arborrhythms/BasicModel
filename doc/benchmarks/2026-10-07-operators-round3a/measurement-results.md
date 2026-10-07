# Round-3a measurement tables

These tables read the saved runs only. Source, budgets, thresholds and runs are unchanged.

Standing gate passed: **True**. Thirty gate trainings; zero retries or replacements.

## Class and reconstruction

The answer is read from each final greedy committed root. Each row below is one shared training for both gates.

| Run | MSE | §20.5 band | Correct | Multisets | Class | Reconstruction | Final operator | Last epoch with any greedy disjunction |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| [1](measurements/xor-01/run.log) | 4.39248359e-06 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 19 |
| [2](measurements/xor-02/run.log) | 0.0207201172 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 26 |
| [3](measurements/xor-03/run.log) | 0.0160247575 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 32 |
| [4](measurements/xor-04/run.log) | 7.50732809e-13 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 38 |
| [5](measurements/xor-05/run.log) | 8.2600593e-14 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 20 |
| [6](measurements/xor-06/run.log) | 3.64993565e-06 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 7 |
| [7](measurements/xor-07/run.log) | 4.33253433e-12 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 40 |
| [8](measurements/xor-08/run.log) | 2.0250468e-13 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 28 |
| [9](measurements/xor-09/run.log) | 0.00370835332 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 12 |
| [10](measurements/xor-10/run.log) | 8.75934553e-07 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |

## Walk proposals

Every recorded proposal is checked against `sampling_scale = K · R_walk · W`. Counts below are sentence rows, from the same trainings.

| Run | Walk | W | R_walk | K | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 1 | 833 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 2 | 763 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 2 | 805 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 1 | 791 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 1 | 799 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 2 | 797 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 2 | 811 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 1 | 785 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 2 | 822 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 1 | 774 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 2 | 807 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 1 | 789 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 1 | 824 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 2 | 772 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 1 | 804 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 2 | 792 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 2 | 809 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 1 | 787 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 1 | 834 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 2 | 762 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | compose | 2 | 1 | 1 | 803 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 1 | 382 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 3 | 413 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-02](measurements/xor-02/run-audit.json) | compose | 2 | 1 | 1 | 827 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 1 | 372 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 3 | 398 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-03](measurements/xor-03/run-audit.json) | compose | 2 | 1 | 1 | 812 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 1 | 381 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 3 | 404 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-04](measurements/xor-04/run-audit.json) | compose | 2 | 1 | 1 | 798 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 1 | 391 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 3 | 409 |
| [xor-05](measurements/xor-05/run-audit.json) | compose | 2 | 1 | 1 | 829 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 3 | 411 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 1 | 359 |
| [xor-06](measurements/xor-06/run-audit.json) | compose | 2 | 1 | 1 | 801 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 1 | 420 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 3 | 379 |
| [xor-07](measurements/xor-07/run-audit.json) | compose | 2 | 1 | 1 | 797 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 3 | 398 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 1 | 403 |
| [xor-08](measurements/xor-08/run-audit.json) | compose | 2 | 1 | 1 | 784 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 3 | 395 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 1 | 420 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-09](measurements/xor-09/run-audit.json) | compose | 2 | 1 | 1 | 795 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 3 | 385 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 1 | 418 |
| [xor-10](measurements/xor-10/run-audit.json) | compose | 2 | 1 | 1 | 794 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 1 | 409 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 3 | 396 |

## Cost and policy audit by run

Negative advantage rewards the explore action; positive advantage rewards greedy. E is separately retained in every raw trial. “Against keep” counts the rows where the total rewards a different trial from strict reconstruction.

| Run | Walk/action | Departures | Nonzero | Reward explore / greedy | Keep explore | Answer against keep | Sum ΔR | Sum ΔE | Sum ΔA |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| [1](measurements/xor-01/run-audit.json) | compose/conjunction | 10 | 10 | 3 / 7 | 0 | 3 | 0 | 0 | 4.42816483 |
| [1](measurements/xor-01/run-audit.json) | compose/disjunction | 793 | 793 | 80 / 713 | 0 | 80 | 0 | 0 | 322.885887 |
| [1](measurements/xor-01/run-audit.json) | narrowing/and | 161 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/gloss | 382 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/not | 119 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/or | 134 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | compose/conjunction | 35 | 35 | 11 / 24 | 0 | 11 | 0 | 0 | 1.8057791 |
| [2](measurements/xor-02/run-audit.json) | compose/disjunction | 792 | 792 | 103 / 689 | 0 | 103 | 0 | 0 | 312.715683 |
| [2](measurements/xor-02/run-audit.json) | narrowing/and | 121 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/gloss | 372 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/not | 141 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/or | 137 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | compose/conjunction | 48 | 48 | 20 / 28 | 0 | 20 | 0 | 0 | -0.843553593 |
| [3](measurements/xor-03/run-audit.json) | compose/disjunction | 764 | 764 | 74 / 690 | 0 | 74 | 0 | 0 | 311.474332 |
| [3](measurements/xor-03/run-audit.json) | narrowing/and | 134 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/gloss | 381 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/not | 138 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/or | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | compose/conjunction | 37 | 37 | 13 / 24 | 0 | 13 | 0 | 0 | 7.42160128 |
| [4](measurements/xor-04/run-audit.json) | compose/disjunction | 761 | 761 | 44 / 717 | 0 | 44 | 0 | 0 | 326.314822 |
| [4](measurements/xor-04/run-audit.json) | narrowing/and | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/gloss | 391 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/not | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/or | 145 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | compose/conjunction | 20 | 20 | 9 / 11 | 0 | 9 | 0 | 0 | -0.554095763 |
| [5](measurements/xor-05/run-audit.json) | compose/disjunction | 809 | 809 | 66 / 743 | 0 | 66 | 0 | 0 | 316.870717 |
| [5](measurements/xor-05/run-audit.json) | narrowing/and | 137 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/gloss | 359 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/not | 124 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/or | 150 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | compose/conjunction | 15 | 15 | 12 / 3 | 0 | 12 | 0 | 0 | -1.15087896 |
| [6](measurements/xor-06/run-audit.json) | compose/disjunction | 786 | 786 | 83 / 703 | 0 | 83 | 0 | 0 | 313.383382 |
| [6](measurements/xor-06/run-audit.json) | narrowing/and | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/gloss | 420 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/not | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/or | 118 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | compose/conjunction | 48 | 48 | 30 / 18 | 0 | 30 | 0 | 0 | -2.04602588 |
| [7](measurements/xor-07/run-audit.json) | compose/disjunction | 749 | 749 | 69 / 680 | 0 | 69 | 0 | 0 | 298.324844 |
| [7](measurements/xor-07/run-audit.json) | narrowing/and | 136 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/gloss | 403 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/not | 134 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/or | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | compose/conjunction | 47 | 47 | 22 / 25 | 0 | 22 | 0 | 0 | -3.24511881 |
| [8](measurements/xor-08/run-audit.json) | compose/disjunction | 737 | 737 | 74 / 663 | 0 | 74 | 0 | 0 | 298.083027 |
| [8](measurements/xor-08/run-audit.json) | narrowing/and | 132 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/gloss | 420 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/not | 129 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/or | 134 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | compose/conjunction | 9 | 9 | 8 / 1 | 0 | 8 | 0 | 0 | -2.43579373 |
| [9](measurements/xor-09/run-audit.json) | compose/disjunction | 786 | 786 | 81 / 705 | 0 | 81 | 0 | 0 | 310.305774 |
| [9](measurements/xor-09/run-audit.json) | narrowing/and | 120 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/gloss | 418 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/not | 138 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/or | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | compose/disjunction | 794 | 794 | 62 / 732 | 0 | 62 | 0 | 0 | 309.775261 |
| [10](measurements/xor-10/run-audit.json) | narrowing/and | 147 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/gloss | 409 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/not | 120 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/or | 129 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |

### Final training commitments and selecting costs

The paired components are before either trial update. A is the comparison reader’s relative answer cost; it is separate from the presented reader’s raw MSE. The keep is R alone; the credit uses R+E+A. Evaluation has no supplied answer and its A term is zero. The complete final evaluation derivations and costs are in `measurements/summary.json`.

| Run / row | Kept operator | R greedy / explore | E greedy / explore | A greedy / explore | Keep | Advantage sign |
| --- | --- | --- | --- | --- | --- | ---: |
| 1 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.00353872729 / 0.477691382 | tie:greedy | 1 |
| 1 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.00137780572 / 0.258053154 | tie:greedy | 1 |
| 1 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.0157221053 / 0.426701397 | tie:greedy | 1 |
| 1 / 3 | conjunction | 0 / 0 | 0 / 0 | 8.48024285e-07 / 0.449903637 | tie:greedy | 1 |
| 2 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.110152856 / 0.577620268 | tie:greedy | 1 |
| 2 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.0749070495 / 0.191218123 | tie:greedy | 1 |
| 2 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.130724683 / 0.186329797 | tie:greedy | 1 |
| 2 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.109431691 / 0.109431691 | tie:greedy | 0 |
| 3 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.0141102392 / 0.0141102392 | tie:greedy | 0 |
| 3 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.018580161 / 0.271732152 | tie:greedy | 1 |
| 3 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.00548267365 / 0.299050599 | tie:greedy | 1 |
| 3 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.0256724339 / 0.468776107 | tie:greedy | 1 |
| 4 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.00529763103 / 0.00529763103 | tie:greedy | 0 |
| 4 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.00978753995 / 0.259029686 | tie:greedy | 1 |
| 4 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.000475339883 / 0.000475339883 | tie:greedy | 0 |
| 4 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.00491592428 / 0.00491592428 | tie:greedy | 0 |
| 5 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.0756995976 / 0.443548739 | tie:greedy | 1 |
| 5 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.0452367701 / 0.112413168 | tie:greedy | 1 |
| 5 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.0818092376 / 0.448063016 | tie:greedy | 1 |
| 5 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.0809143707 / 0.693089247 | tie:greedy | 1 |
| 6 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.00428737747 / 0.710255742 | tie:greedy | 1 |
| 6 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.00154767407 / 0.228008762 | tie:greedy | 1 |
| 6 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.00452320464 / 0.00452320464 | tie:greedy | 0 |
| 6 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.0180866718 / 0.460717827 | tie:greedy | 1 |
| 7 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.000459371367 / 0.000459371367 | tie:greedy | 0 |
| 7 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.0016691268 / 0.0016691268 | tie:greedy | 0 |
| 7 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.00222155545 / 0.835263908 | tie:greedy | 1 |
| 7 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.0110620083 / 0.191376001 | tie:greedy | 1 |
| 8 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.0985737592 / 0.682547808 | tie:greedy | 1 |
| 8 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.0862179026 / 0.215360552 | tie:greedy | 1 |
| 8 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.0920331478 / 0.0920331478 | tie:greedy | 0 |
| 8 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.145187706 / 0.563306332 | tie:greedy | 1 |
| 9 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.173770219 / 0.63046968 | tie:greedy | 1 |
| 9 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.0920112282 / 0.144361332 | tie:greedy | 1 |
| 9 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.131858975 / 0.131858975 | tie:greedy | 0 |
| 9 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.117007338 / 0.117007338 | tie:greedy | 0 |
| 10 / 0 | conjunction | 0 / 0 | 0 / 0 | 0.0989865139 / 0.0955710411 | tie:greedy | -1 |
| 10 / 1 | conjunction | 0 / 0 | 0 / 0 | 0.188045546 / 0.828496099 | tie:greedy | 1 |
| 10 / 2 | conjunction | 0 / 0 | 0 / 0 | 0.0879524425 / 0.78250891 | tie:greedy | 1 |
| 10 / 3 | conjunction | 0 / 0 | 0 / 0 | 0.0859951004 / 0.0859951004 | tie:greedy | 0 |

## Sum controls

| Run | MSE | Band | Checkerboard contrast | Max deviation from 0.5 | Floor |
| --- | ---: | --- | ---: | ---: | --- |
| [1](measurements/sum-01/run.log) | 0.25 | at 1/4 | 8.94069672e-08 | 6.2584877e-07 | pass |
| [2](measurements/sum-02/run.log) | 0.2500000596 | at 1/4 | 1.78813934e-07 | 6.55651093e-07 | pass |
| [3](measurements/sum-03/run.log) | 0.2500001192 | at 1/4 | 5.06639481e-07 | 1.1920929e-06 | pass |
| [4](measurements/sum-04/run.log) | 0.2499999404 | at 1/4 | -1.78813934e-07 | 5.06639481e-07 | pass |
| [5](measurements/sum-05/run.log) | 0.2500001192 | at 1/4 | 3.57627869e-07 | 0.000145673752 | pass |
| [6](measurements/sum-06/run.log) | 0.2499999553 | at 1/4 | -1.78813934e-07 | 5.36441803e-07 | pass |
| [7](measurements/sum-07/run.log) | 0.2500000298 | at 1/4 | 1.1920929e-07 | 7.74860382e-07 | pass |
| [8](measurements/sum-08/run.log) | 0.2500001192 | at 1/4 | 4.47034836e-07 | 1.9967556e-06 | pass |
| [9](measurements/sum-09/run.log) | 0.2500000596 | at 1/4 | 2.08616257e-07 | 6.55651093e-07 | pass |
| [10](measurements/sum-10/run.log) | 0.2499999553 | at 1/4 | -2.38418579e-07 | 1.7285347e-06 | pass |

## MM_xor

Raw MM counts remain visible. The repeated first-forward bisection establishes an RNG-only path, so §20 applies §5’s amendment and this count is not a regression test of the round. The separate paired trajectories are also retained.

| Run | Epochs | Best MSE | Final MSE | Gate |
| --- | ---: | ---: | ---: | --- |
| [1](measurements/mm-01/run.log) | 51 | 0.19575128 | 0.19575128 | pass |
| [2](measurements/mm-02/run.log) | 18 | 0.192050308 | 0.192050308 | pass |
| [3](measurements/mm-03/run.log) | 26 | 0.193357274 | 0.193357274 | pass |
| [4](measurements/mm-04/run.log) | 52 | 0.177393943 | 0.177393943 | pass |
| [5](measurements/mm-05/run.log) | 75 | 0.194779262 | 0.194779262 | pass |
| [6](measurements/mm-06/run.log) | 63 | 0.195753545 | 0.195753545 | pass |
| [7](measurements/mm-07/run.log) | 89 | 0.179967463 | 0.179967463 | pass |
| [8](measurements/mm-08/run.log) | 36 | 0.195012897 | 0.195012897 | pass |
| [9](measurements/mm-09/run.log) | 117 | 0.191194475 | 0.191194475 | pass |
| [10](measurements/mm-10/run.log) | 52 | 0.198101357 | 0.198101357 | pass |

## Reader steps and pole consumers

| Run | Updates per reader | Final presented Adam steps | Presented kept rows | Comparison mixed compose rows |
| --- | ---: | --- | ---: | ---: |
| [sum-01](measurements/sum-01/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-02](measurements/sum-02/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-03](measurements/sum-03/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-04](measurements/sum-04/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-05](measurements/sum-05/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-06](measurements/sum-06/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-07](measurements/sum-07/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-08](measurements/sum-08/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-09](measurements/sum-09/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [sum-10](measurements/sum-10/run-audit.json) | 400 | [400.0] | 1600 | 0 |
| [xor-01](measurements/xor-01/run-audit.json) | 400 | [400.0] | 1600 | 803 |
| [xor-02](measurements/xor-02/run-audit.json) | 400 | [400.0] | 1600 | 827 |
| [xor-03](measurements/xor-03/run-audit.json) | 400 | [400.0] | 1600 | 812 |
| [xor-04](measurements/xor-04/run-audit.json) | 400 | [400.0] | 1600 | 798 |
| [xor-05](measurements/xor-05/run-audit.json) | 400 | [400.0] | 1600 | 829 |
| [xor-06](measurements/xor-06/run-audit.json) | 400 | [400.0] | 1600 | 801 |
| [xor-07](measurements/xor-07/run-audit.json) | 400 | [400.0] | 1600 | 797 |
| [xor-08](measurements/xor-08/run-audit.json) | 400 | [400.0] | 1600 | 784 |
| [xor-09](measurements/xor-09/run-audit.json) | 400 | [400.0] | 1600 | 795 |
| [xor-10](measurements/xor-10/run-audit.json) | 400 | [400.0] | 1600 | 794 |

Each run also records per-epoch logit ranges, every cost and keep decision, both final operators, image widths and values, containment before/after, and sentence-path gradients. The tenth XOR run includes the ownership audit and analytic/finite-difference checks.

Table postprocessor SHA-256: `2883984a2f0836efc4d422959fc96f3c0b234e7e4c4628e0a102ed12594f6e38`.
