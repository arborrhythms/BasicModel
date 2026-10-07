# Round-2e measurement tables

These tables read the saved runs only. Source, budgets, thresholds and runs are unchanged.

Standing gate passed: **True**. Thirty gate trainings; zero retries or replacements.

## Class and reconstruction

The answer is read from each final greedy committed root. Each row below is one shared training for both gates.

| Run | MSE | §20.5 band | Correct | Multisets | Class | Reconstruction | Final operator | Last epoch with any greedy disjunction |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| [1](measurements/xor-01/run.log) | 3.72895048e-11 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 14 |
| [2](measurements/xor-02/run.log) | 0.00687594264 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 48 |
| [3](measurements/xor-03/run.log) | 0.0202075404 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [4](measurements/xor-04/run.log) | 0.000153554436 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 13 |
| [5](measurements/xor-05/run.log) | 0.000318584207 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 17 |
| [6](measurements/xor-06/run.log) | 0.0801115114 | between | 4/4 | 4/4 | fail | pass | conjunction | 139 |
| [7](measurements/xor-07/run.log) | 0.00146726917 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [8](measurements/xor-08/run.log) | 0.00652075451 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [9](measurements/xor-09/run.log) | 0.00110106565 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 32 |
| [10](measurements/xor-10/run.log) | 3.25428486e-07 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |

## Walk proposals

Every recorded proposal is checked against `sampling_scale = K · R_walk · W`. Counts below are sentence rows, from the same trainings.

| Run | Walk | W | R_walk | K | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 2 | 783 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 1 | 813 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 2 | 809 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 1 | 787 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 2 | 3 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 1 | 1 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 2 | 785 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 1 | 811 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 2 | 3 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 1 | 1 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 2 | 809 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 1 | 787 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 2 | 766 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 1 | 830 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 2 | 821 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 1 | 775 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 1 | 794 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 2 | 802 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 1 | 824 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 2 | 772 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 2 | 813 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 1 | 783 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 3 | 1 | 4 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 2 | 760 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 1 | 836 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | compose | 2 | 1 | 1 | 839 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 1 | 390 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 3 | 369 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-02](measurements/xor-02/run-audit.json) | compose | 2 | 1 | 1 | 807 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 3 | 382 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 1 | 409 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-03](measurements/xor-03/run-audit.json) | compose | 2 | 1 | 1 | 805 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 3 | 391 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 1 | 402 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-04](measurements/xor-04/run-audit.json) | compose | 2 | 1 | 1 | 844 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 3 | 388 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 1 | 367 |
| [xor-05](measurements/xor-05/run-audit.json) | compose | 2 | 1 | 1 | 772 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 1 | 401 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 3 | 425 |
| [xor-06](measurements/xor-06/run-audit.json) | compose | 2 | 1 | 1 | 798 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 3 | 406 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 1 | 396 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-07](measurements/xor-07/run-audit.json) | compose | 2 | 1 | 1 | 817 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 1 | 382 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 3 | 400 |
| [xor-08](measurements/xor-08/run-audit.json) | compose | 2 | 1 | 1 | 777 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 1 | 404 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 3 | 417 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 3 | 1 | 3 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-09](measurements/xor-09/run-audit.json) | compose | 2 | 1 | 1 | 800 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 3 | 394 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 1 | 402 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-10](measurements/xor-10/run-audit.json) | compose | 2 | 1 | 1 | 815 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 1 | 421 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 3 | 362 |

## Cost and policy audit by run

Negative advantage rewards the explore action; positive advantage rewards greedy. E is separately retained in every raw trial. “Against keep” counts the rows where the total rewards a different trial from strict reconstruction.

| Run | Walk/action | Departures | Nonzero | Reward explore / greedy | Keep explore | Answer against keep | Sum ΔR | Sum ΔE | Sum ΔA |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| [1](measurements/xor-01/run-audit.json) | compose/conjunction | 13 | 13 | 12 / 1 | 6 | 6 | -1.01655734 | 0 | -0.234386474 |
| [1](measurements/xor-01/run-audit.json) | compose/disjunction | 826 | 826 | 7 / 819 | 0 | 7 | 0.618534155 | 0 | 326.903874 |
| [1](measurements/xor-01/run-audit.json) | narrowing/and | 117 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/gloss | 390 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/not | 106 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/or | 147 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | compose/conjunction | 83 | 83 | 50 / 33 | 7 | 43 | -1.2580524 | 0 | -0.762180373 |
| [2](measurements/xor-02/run-audit.json) | compose/disjunction | 724 | 724 | 33 / 691 | 0 | 33 | 0 | 0 | 166.985611 |
| [2](measurements/xor-02/run-audit.json) | narrowing/and | 106 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/gloss | 409 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/not | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/or | 148 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | compose/disjunction | 805 | 805 | 226 / 579 | 0 | 226 | 0 | 0 | 135.45152 |
| [3](measurements/xor-03/run-audit.json) | narrowing/and | 140 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/gloss | 402 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/not | 134 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/or | 118 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | compose/conjunction | 17 | 17 | 13 / 4 | 0 | 13 | 0 | 0 | -0.0947261974 |
| [4](measurements/xor-04/run-audit.json) | compose/disjunction | 827 | 827 | 15 / 812 | 0 | 15 | 63.5476425 | 0 | 245.949163 |
| [4](measurements/xor-04/run-audit.json) | narrowing/and | 120 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/gloss | 367 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/not | 135 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/or | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | compose/conjunction | 23 | 23 | 15 / 8 | 8 | 7 | -2.26639111 | 0 | -0.24279815 |
| [5](measurements/xor-05/run-audit.json) | compose/disjunction | 749 | 749 | 143 / 606 | 0 | 143 | 83.2088176 | 0 | 193.8332 |
| [5](measurements/xor-05/run-audit.json) | narrowing/and | 157 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/gloss | 401 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/not | 138 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/or | 130 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | compose/conjunction | 244 | 244 | 114 / 130 | 0 | 114 | 0 | 0 | 2.03386331 |
| [6](measurements/xor-06/run-audit.json) | compose/disjunction | 554 | 554 | 46 / 508 | 0 | 46 | 0 | 0 | 92.5907314 |
| [6](measurements/xor-06/run-audit.json) | narrowing/and | 127 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/gloss | 396 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/not | 147 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/or | 132 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | compose/disjunction | 817 | 817 | 4 / 813 | 0 | 4 | 124.476065 | 0 | 210.713157 |
| [7](measurements/xor-07/run-audit.json) | narrowing/and | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/gloss | 382 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/not | 136 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/or | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | compose/disjunction | 777 | 777 | 2 / 775 | 0 | 2 | 44.4417631 | 0 | 179.652122 |
| [8](measurements/xor-08/run-audit.json) | narrowing/and | 151 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/gloss | 404 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/not | 136 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/or | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | compose/conjunction | 63 | 63 | 38 / 25 | 15 | 23 | -2.51046832 | 0 | -0.138659477 |
| [9](measurements/xor-09/run-audit.json) | compose/disjunction | 737 | 737 | 9 / 728 | 0 | 9 | 37.6581123 | 0 | 191.180812 |
| [9](measurements/xor-09/run-audit.json) | narrowing/and | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/gloss | 402 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/not | 148 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/or | 116 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | compose/disjunction | 815 | 815 | 60 / 755 | 0 | 60 | 1.23615769 | 0 | 278.150724 |
| [10](measurements/xor-10/run-audit.json) | narrowing/and | 104 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/gloss | 421 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/not | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/or | 125 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |

### Final training commitments and selecting costs

The paired components are before either trial update. A is the comparison reader’s relative answer cost; it is separate from the presented reader’s raw MSE. The keep is R alone; the credit uses R+E+A. Evaluation has no supplied answer and its A term is zero. The complete final evaluation derivations and costs are in `measurements/summary.json`.

| Run / row | Kept operator | R greedy / explore | E greedy / explore | A greedy / explore | Keep | Advantage sign |
| --- | --- | --- | --- | --- | --- | ---: |
| 1 / 0 | conjunction | 3.49130745e-13 / 3.49130745e-13 | 0 / 0 | 0.000337836856 / 0.000337836856 | tie:greedy | 0 |
| 1 / 1 | conjunction | 2.4132482e-14 / 2.4132482e-14 | 0 / 0 | 0.0111084161 / 0.0111084161 | tie:greedy | 0 |
| 1 / 2 | conjunction | 0.303038687 / 0.303038687 | 0 / 0 | 0.0522985943 / 0.374297291 | tie:greedy | 1 |
| 1 / 3 | conjunction | 4.61293432e-14 / 4.61293432e-14 | 0 / 0 | 7.75012668e-05 / 7.75012668e-05 | tie:greedy | 0 |
| 2 / 0 | conjunction | 1.69687819e-08 / 1.69687819e-08 | 0 / 0 | 0.218017414 / 0.218017414 | tie:greedy | 0 |
| 2 / 1 | conjunction | 0.000118470321 / 0.000118470321 | 0 / 0 | 8.88442719e-06 / 0.382675976 | tie:greedy | 1 |
| 2 / 2 | conjunction | 0.161996767 / 0.161996767 | 0 / 0 | 0.0252024028 / 0.0252024028 | tie:greedy | 0 |
| 2 / 3 | conjunction | 3.21164961e-08 / 3.21164961e-08 | 0 / 0 | 0.165781319 / 0.165781319 | tie:greedy | 0 |
| 3 / 0 | conjunction | 2.04927837e-05 / 2.04927837e-05 | 0 / 0 | 0.0877590403 / 0.0877590403 | tie:greedy | 0 |
| 3 / 1 | conjunction | 2.66738644e-13 / 2.66738644e-13 | 0 / 0 | 0.175323159 / 0.175323159 | tie:greedy | 0 |
| 3 / 2 | conjunction | 0.041440364 / 0.041440364 | 0 / 0 | 0.350898802 / 0.350898802 | tie:greedy | 0 |
| 3 / 3 | conjunction | 0.389256537 / 0.389256537 | 0 / 0 | 0.02343034 / 0.469604611 | tie:greedy | 1 |
| 4 / 0 | conjunction | 6.13067925e-07 / 6.13067925e-07 | 0 / 0 | 0.271951914 / 0.271951914 | tie:greedy | 0 |
| 4 / 1 | conjunction | 0.00114645239 / 0.00114645239 | 0 / 0 | 0.0177466087 / 0.147782207 | tie:greedy | 1 |
| 4 / 2 | conjunction | 0.265015066 / 0.265015066 | 0 / 0 | 0.0110197244 / 0.269595295 | tie:greedy | 1 |
| 4 / 3 | conjunction | 1.08125453e-09 / 0.299893945 | 0 / 0 | 0.131180108 / 1.13910508 | greedy | 1 |
| 5 / 0 | conjunction | 0.000372291135 / 0.000372291135 | 0 / 0 | 0.0809006691 / 0.34401238 | tie:greedy | 1 |
| 5 / 1 | conjunction | 0.000344528147 / 0.000344528147 | 0 / 0 | 0.0263689477 / 0.376217455 | tie:greedy | 1 |
| 5 / 2 | conjunction | 0.180338666 / 0.180338666 | 0 / 0 | 0.0861637369 / 0.148320645 | tie:greedy | 1 |
| 5 / 3 | conjunction | 4.32975135e-13 / 4.32975135e-13 | 0 / 0 | 0.0598302819 / 0.0598302819 | tie:greedy | 0 |
| 6 / 0 | conjunction | 2.88758739e-10 / 2.88758739e-10 | 0 / 0 | 0.183337569 / 0.183337569 | tie:greedy | 0 |
| 6 / 1 | conjunction | 2.11158138e-07 / 2.11158138e-07 | 0 / 0 | 0.172641069 / 0.229199067 | tie:greedy | 1 |
| 6 / 2 | conjunction | 0.33845073 / 0.33845073 | 0 / 0 | 0.0474714898 / 0.687800944 | tie:greedy | 1 |
| 6 / 3 | conjunction | 0.335241526 / 0.335241526 | 0 / 0 | 0.218461499 / 0.29193148 | tie:greedy | 1 |
| 7 / 0 | conjunction | 2.44771059e-07 / 2.44771059e-07 | 0 / 0 | 0.036983788 / 0.036983788 | tie:greedy | 0 |
| 7 / 1 | conjunction | 7.60077853e-07 / 7.60077853e-07 | 0 / 0 | 0.0246174373 / 0.0246174373 | tie:greedy | 0 |
| 7 / 2 | conjunction | 6.39382658e-10 / 0.258921266 | 0 / 0 | 0.0436351076 / 0.31583333 | greedy | 1 |
| 7 / 3 | conjunction | 6.06216412e-13 / 6.06216412e-13 | 0 / 0 | 0.0201199576 / 0.0201199576 | tie:greedy | 0 |
| 8 / 0 | conjunction | 3.0653743e-07 / 3.0653743e-07 | 0 / 0 | 0.0628204942 / 0.0628204942 | tie:greedy | 0 |
| 8 / 1 | conjunction | 1.45477963e-10 / 1.45477963e-10 | 0 / 0 | 0.0881675929 / 0.0881675929 | tie:greedy | 0 |
| 8 / 2 | conjunction | 1.06758291e-08 / 0.216009289 | 0 / 0 | 0.137057334 / 0.243656129 | greedy | 1 |
| 8 / 3 | conjunction | 0.408367395 / 0.408367395 | 0 / 0 | 0.0229623523 / 0.0229623523 | tie:greedy | 0 |
| 9 / 0 | conjunction | 8.79454589e-08 / 8.79454589e-08 | 0 / 0 | 0.0389518701 / 0.0389518701 | tie:greedy | 0 |
| 9 / 1 | conjunction | 0.000149594343 / 0.000149594343 | 0 / 0 | 0.0374901257 / 0.0374901257 | tie:greedy | 0 |
| 9 / 2 | conjunction | 1.22718857e-08 / 1.22718857e-08 | 0 / 0 | 0.0575012118 / 0.0575012118 | tie:greedy | 0 |
| 9 / 3 | conjunction | 1.31327674e-12 / 1.31327674e-12 | 0 / 0 | 0.018600788 / 0.018600788 | tie:greedy | 0 |
| 10 / 0 | conjunction | 4.74512929e-11 / 4.74512929e-11 | 0 / 0 | 0.00232638209 / 0.106715754 | tie:greedy | 1 |
| 10 / 1 | conjunction | 7.16312912e-11 / 7.16312912e-11 | 0 / 0 | 0.0504174531 / 0.0504174531 | tie:greedy | 0 |
| 10 / 2 | conjunction | 0.210676074 / 0.210676074 | 0 / 0 | 0.0482086726 / 0.0482086726 | tie:greedy | 0 |
| 10 / 3 | conjunction | 0.562432706 / 0.562432706 | 0 / 0 | 0.0031663191 / 0.355960459 | tie:greedy | 1 |

## Sum controls

| Run | MSE | Band | Checkerboard contrast | Max deviation from 0.5 | Floor |
| --- | ---: | --- | ---: | ---: | --- |
| [1](measurements/sum-01/run.log) | 0.2499999851 | at 1/4 | -2.98023224e-08 | 3.57627869e-07 | pass |
| [2](measurements/sum-02/run.log) | 0.25 | at 1/4 | 5.96046448e-08 | 1.78813934e-07 | pass |
| [3](measurements/sum-03/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 1.78813934e-07 | pass |
| [4](measurements/sum-04/run.log) | 0.2499999851 | at 1/4 | -2.98023224e-08 | 1.78813934e-07 | pass |
| [5](measurements/sum-05/run.log) | 0.25 | at 1/4 | 8.94069672e-08 | 1.78813934e-07 | pass |
| [6](measurements/sum-06/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 1.49011612e-07 | pass |
| [7](measurements/sum-07/run.log) | 0.2499999851 | at 1/4 | -5.96046448e-08 | 4.17232513e-07 | pass |
| [8](measurements/sum-08/run.log) | 0.2499999851 | at 1/4 | -5.96046448e-08 | 1.1920929e-07 | pass |
| [9](measurements/sum-09/run.log) | 0.25 | at 1/4 | 0 | 2.38418579e-07 | pass |
| [10](measurements/sum-10/run.log) | 0.25 | at 1/4 | 2.98023224e-08 | 2.98023224e-07 | pass |

## MM_xor

Raw MM counts remain visible. The repeated first-forward bisection establishes an RNG-only path, so §20 applies §5’s amendment and this count is not a regression test of the round. The separate paired trajectories are also retained.

| Run | Epochs | Best MSE | Final MSE | Gate |
| --- | ---: | ---: | ---: | --- |
| [1](measurements/mm-01/run.log) | 97 | 0.12374191 | 0.12374191 | pass |
| [2](measurements/mm-02/run.log) | 53 | 0.192104727 | 0.192104727 | pass |
| [3](measurements/mm-03/run.log) | 74 | 0.155932814 | 0.155932814 | pass |
| [4](measurements/mm-04/run.log) | 65 | 0.193828329 | 0.193828329 | pass |
| [5](measurements/mm-05/run.log) | 63 | 0.196188599 | 0.196188599 | pass |
| [6](measurements/mm-06/run.log) | 58 | 0.183453977 | 0.183453977 | pass |
| [7](measurements/mm-07/run.log) | 157 | 0.178155482 | 0.178155482 | pass |
| [8](measurements/mm-08/run.log) | 29 | 0.194205761 | 0.194205761 | pass |
| [9](measurements/mm-09/run.log) | 167 | 0.198815867 | 0.198815867 | pass |
| [10](measurements/mm-10/run.log) | 48 | 0.198696345 | 0.198696345 | pass |

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
| [xor-01](measurements/xor-01/run-audit.json) | 400 | [400.0] | 1600 | 839 |
| [xor-02](measurements/xor-02/run-audit.json) | 400 | [400.0] | 1600 | 807 |
| [xor-03](measurements/xor-03/run-audit.json) | 400 | [400.0] | 1600 | 805 |
| [xor-04](measurements/xor-04/run-audit.json) | 400 | [400.0] | 1600 | 844 |
| [xor-05](measurements/xor-05/run-audit.json) | 400 | [400.0] | 1600 | 772 |
| [xor-06](measurements/xor-06/run-audit.json) | 400 | [400.0] | 1600 | 798 |
| [xor-07](measurements/xor-07/run-audit.json) | 400 | [400.0] | 1600 | 817 |
| [xor-08](measurements/xor-08/run-audit.json) | 400 | [400.0] | 1600 | 777 |
| [xor-09](measurements/xor-09/run-audit.json) | 400 | [400.0] | 1600 | 800 |
| [xor-10](measurements/xor-10/run-audit.json) | 400 | [400.0] | 1600 | 815 |

Each run also records per-epoch logit ranges, every cost and keep decision, both final operators, image widths and values, containment before/after, and sentence-path gradients. The tenth XOR run includes the ownership audit and analytic/finite-difference checks.

Table postprocessor SHA-256: `212c75eee3351054bf7bffc957224a4139e7d70f9986a791aa98926200ca0b62`.
