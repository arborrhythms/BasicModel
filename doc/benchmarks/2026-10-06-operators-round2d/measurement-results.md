# Round-2d measurement tables

These tables read the saved runs only. Source, budgets, thresholds and runs are unchanged.

Standing gate passed: **False**. Thirty gate trainings; zero retries or replacements.

## Class and reconstruction

The answer is read from each final greedy committed root. Each row below is one shared training for both gates.

| Run | MSE | §20.5 band | Correct | Multisets | Class | Reconstruction | Final operator | Last epoch with any greedy disjunction |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| [1](measurements/xor-01/run.log) | 0.070232812 | between | 4/4 | 4/4 | fail | pass | conjunction | 0 |
| [2](measurements/xor-02/run.log) | 0.191986635 | between | 4/4 | 4/4 | fail | pass | conjunction | 0 |
| [3](measurements/xor-03/run.log) | 0.11801415 | between | 4/4 | 4/4 | fail | pass | conjunction | 21 |
| [4](measurements/xor-04/run.log) | 0.0639815362 | between | 4/4 | 4/4 | fail | pass | conjunction | 23 |
| [5](measurements/xor-05/run.log) | 0.129314278 | between | 4/4 | 4/4 | fail | pass | conjunction | 64 |
| [6](measurements/xor-06/run.log) | 0.0284346324 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 7 |
| [7](measurements/xor-07/run.log) | 0.144109057 | between | 4/4 | 4/4 | fail | pass | conjunction | 0 |
| [8](measurements/xor-08/run.log) | 0.0835429512 | between | 4/4 | 4/4 | fail | pass | conjunction | 21 |
| [9](measurements/xor-09/run.log) | 0.0483400879 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [10](measurements/xor-10/run.log) | 0.142018276 | between | 4/4 | 4/4 | fail | pass | conjunction | 105 |

## Walk proposals

Every recorded proposal is checked against `sampling_scale = K · R_walk · W`. Counts below are sentence rows, from the same trainings.

| Run | Walk | W | R_walk | K | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 2 | 800 |
| [sum-01](measurements/sum-01/run-audit.json) | narrowing | 1 | 2 | 1 | 796 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 1 | 778 |
| [sum-02](measurements/sum-02/run-audit.json) | narrowing | 1 | 2 | 2 | 818 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 2 | 807 |
| [sum-03](measurements/sum-03/run-audit.json) | narrowing | 1 | 2 | 1 | 789 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 2 | 804 |
| [sum-04](measurements/sum-04/run-audit.json) | narrowing | 1 | 2 | 1 | 792 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 2 | 816 |
| [sum-05](measurements/sum-05/run-audit.json) | narrowing | 1 | 2 | 1 | 780 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 2 | 799 |
| [sum-06](measurements/sum-06/run-audit.json) | narrowing | 1 | 2 | 1 | 797 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 1 | 810 |
| [sum-07](measurements/sum-07/run-audit.json) | narrowing | 1 | 2 | 2 | 786 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 1 | 816 |
| [sum-08](measurements/sum-08/run-audit.json) | narrowing | 1 | 2 | 2 | 780 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 3 | 2 | 1 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 3 | 1 | 3 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 1 | 797 |
| [sum-09](measurements/sum-09/run-audit.json) | narrowing | 1 | 2 | 2 | 799 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 3 | 1 | 2 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 3 | 2 | 2 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 1 | 801 |
| [sum-10](measurements/sum-10/run-audit.json) | narrowing | 1 | 2 | 2 | 795 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 3 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-01](measurements/xor-01/run-audit.json) | compose | 2 | 1 | 1 | 771 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 3 | 413 |
| [xor-01](measurements/xor-01/run-audit.json) | narrowing | 2 | 2 | 1 | 414 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-02](measurements/xor-02/run-audit.json) | compose | 2 | 1 | 1 | 795 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 1 | 405 |
| [xor-02](measurements/xor-02/run-audit.json) | narrowing | 2 | 2 | 3 | 398 |
| [xor-03](measurements/xor-03/run-audit.json) | compose | 2 | 1 | 1 | 806 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 3 | 419 |
| [xor-03](measurements/xor-03/run-audit.json) | narrowing | 2 | 2 | 1 | 375 |
| [xor-04](measurements/xor-04/run-audit.json) | compose | 2 | 1 | 1 | 804 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 1 | 407 |
| [xor-04](measurements/xor-04/run-audit.json) | narrowing | 2 | 2 | 3 | 387 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 3 | 1 | 3 |
| [xor-05](measurements/xor-05/run-audit.json) | compose | 2 | 1 | 1 | 807 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 3 | 383 |
| [xor-05](measurements/xor-05/run-audit.json) | narrowing | 2 | 2 | 1 | 407 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 3 | 1 | 2 |
| [xor-06](measurements/xor-06/run-audit.json) | compose | 2 | 1 | 1 | 795 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 1 | 413 |
| [xor-06](measurements/xor-06/run-audit.json) | narrowing | 2 | 2 | 3 | 390 |
| [xor-07](measurements/xor-07/run-audit.json) | compose | 2 | 1 | 1 | 781 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 3 | 395 |
| [xor-07](measurements/xor-07/run-audit.json) | narrowing | 2 | 2 | 1 | 423 |
| [xor-08](measurements/xor-08/run-audit.json) | compose | 2 | 1 | 1 | 797 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 1 | 378 |
| [xor-08](measurements/xor-08/run-audit.json) | narrowing | 2 | 2 | 3 | 424 |
| [xor-09](measurements/xor-09/run-audit.json) | compose | 2 | 1 | 1 | 798 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 3 | 1 | 3 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 1 | 403 |
| [xor-09](measurements/xor-09/run-audit.json) | narrowing | 2 | 2 | 3 | 396 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 3 | 1 | 1 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 3 | 3 | 2 |
| [xor-10](measurements/xor-10/run-audit.json) | compose | 2 | 1 | 1 | 824 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 3 | 386 |
| [xor-10](measurements/xor-10/run-audit.json) | narrowing | 2 | 2 | 1 | 387 |

## Cost and policy audit by run

Negative advantage rewards the explore action; positive advantage rewards greedy. E is separately retained in every raw trial. “Against keep” counts the rows where the total rewards a different trial from strict reconstruction.

| Run | Walk/action | Departures | Nonzero | Reward explore / greedy | Keep explore | Answer against keep | Sum ΔR | Sum ΔE | Sum ΔA |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| [1](measurements/xor-01/run-audit.json) | compose/disjunction | 771 | 771 | 128 / 643 | 0 | 128 | 24.162832 | 0 | 143.472075 |
| [1](measurements/xor-01/run-audit.json) | narrowing/and | 141 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/gloss | 414 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/not | 145 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/or | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | compose/disjunction | 795 | 795 | 252 / 543 | 0 | 252 | 0.0130824838 | 0 | 40.9391924 |
| [2](measurements/xor-02/run-audit.json) | narrowing/and | 141 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/gloss | 405 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/not | 130 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/or | 127 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | compose/conjunction | 42 | 42 | 26 / 16 | 8 | 18 | -0.791355509 | 0 | -0.271860838 |
| [3](measurements/xor-03/run-audit.json) | compose/disjunction | 764 | 764 | 34 / 730 | 0 | 34 | 27.6989644 | 0 | 94.7581841 |
| [3](measurements/xor-03/run-audit.json) | narrowing/and | 146 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/gloss | 375 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/not | 140 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/or | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | compose/conjunction | 16 | 16 | 7 / 9 | 5 | 2 | -0.50494124 | 0 | 0.230018675 |
| [4](measurements/xor-04/run-audit.json) | compose/disjunction | 788 | 788 | 10 / 778 | 0 | 10 | 24.5551225 | 0 | 152.070696 |
| [4](measurements/xor-04/run-audit.json) | narrowing/and | 119 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/gloss | 407 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/not | 122 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/or | 146 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | compose/conjunction | 82 | 82 | 36 / 46 | 1 | 35 | -0.0279947147 | 0 | -0.251206473 |
| [5](measurements/xor-05/run-audit.json) | compose/disjunction | 725 | 725 | 149 / 576 | 0 | 149 | 0 | 0 | 101.410393 |
| [5](measurements/xor-05/run-audit.json) | narrowing/and | 128 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/gloss | 407 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/not | 129 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/or | 126 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | compose/conjunction | 4 | 4 | 0 / 4 | 0 | 0 | 0 | 0 | 0.126192093 |
| [6](measurements/xor-06/run-audit.json) | compose/disjunction | 791 | 791 | 62 / 729 | 0 | 62 | 93.6398947 | 0 | 240.434105 |
| [6](measurements/xor-06/run-audit.json) | narrowing/and | 131 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/gloss | 413 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/not | 140 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/or | 119 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | compose/disjunction | 781 | 781 | 247 / 534 | 0 | 247 | 0 | 0 | 100.758155 |
| [7](measurements/xor-07/run-audit.json) | narrowing/and | 152 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/gloss | 423 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/not | 124 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/or | 119 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | compose/conjunction | 20 | 20 | 12 / 8 | 1 | 11 | -0.167015508 | 0 | 0.0681456178 |
| [8](measurements/xor-08/run-audit.json) | compose/disjunction | 777 | 777 | 183 / 594 | 0 | 183 | 61.974294 | 0 | 167.177193 |
| [8](measurements/xor-08/run-audit.json) | narrowing/and | 140 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/gloss | 378 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/not | 145 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/or | 139 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | compose/disjunction | 798 | 798 | 28 / 770 | 0 | 28 | 0 | 0 | 221.951907 |
| [9](measurements/xor-09/run-audit.json) | narrowing/and | 108 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/gloss | 403 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/not | 148 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/or | 140 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | compose/conjunction | 164 | 164 | 94 / 70 | 1 | 93 | -0.0194833502 | 0 | -0.792139873 |
| [10](measurements/xor-10/run-audit.json) | compose/disjunction | 660 | 660 | 123 / 537 | 0 | 123 | 0 | 0 | 66.4994561 |
| [10](measurements/xor-10/run-audit.json) | narrowing/and | 118 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/gloss | 387 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/not | 133 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/or | 137 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |

### Final training commitments and selecting costs

The paired components are before either trial update. The keep is R alone; the credit uses R+E+A. Evaluation has no supplied answer and its A term is zero. The complete final evaluation derivations and costs are in `measurements/summary.json`.

| Run / row | Kept operator | R greedy / explore | E greedy / explore | A greedy / explore | Keep | Advantage sign |
| --- | --- | --- | --- | --- | --- | ---: |
| 1 / 0 | conjunction | 0.000169978623 / 0.000169978623 | 0 / 0 | 0.129041955 / 0.142340481 | tie:greedy | 1 |
| 1 / 1 | conjunction | 3.86781709e-12 / 3.86781709e-12 | 0 / 0 | 0.0483697243 / 0.0483697243 | tie:greedy | 0 |
| 1 / 2 | conjunction | 3.11422991e-06 / 3.11422991e-06 | 0 / 0 | 0.181617886 / 0.181617886 | tie:greedy | 0 |
| 1 / 3 | conjunction | 0.23185356 / 0.23185356 | 0 / 0 | 0.0291812047 / 0.465261847 | tie:greedy | 1 |
| 2 / 0 | conjunction | 2.40063596e-08 / 2.40063596e-08 | 0 / 0 | 0.131287217 / 0.131287217 | tie:greedy | 0 |
| 2 / 1 | conjunction | 1.9424283e-06 / 1.9424283e-06 | 0 / 0 | 0.323591501 / 0.323591501 | tie:greedy | 0 |
| 2 / 2 | conjunction | 0.107892931 / 0.107892931 | 0 / 0 | 0.456648558 / 0.456648558 | tie:greedy | 0 |
| 2 / 3 | conjunction | 0.122639619 / 0.122639619 | 0 / 0 | 0.208926275 / 0.208926275 | tie:greedy | 0 |
| 3 / 0 | conjunction | 0.000101436417 / 0.000101436417 | 0 / 0 | 0.214207962 / 0.214207962 | tie:greedy | 0 |
| 3 / 1 | conjunction | 1.34689458e-07 / 1.34689458e-07 | 0 / 0 | 0.118971586 / 0.118971586 | tie:greedy | 0 |
| 3 / 2 | conjunction | 7.31800014e-07 / 7.31800014e-07 | 0 / 0 | 0.168062434 / 0.168062434 | tie:greedy | 0 |
| 3 / 3 | conjunction | 0.252515554 / 0.252515554 | 0 / 0 | 0.145870164 / 0.145870164 | tie:greedy | 0 |
| 4 / 0 | conjunction | 5.99270643e-05 / 5.99270643e-05 | 0 / 0 | 0.132738337 / 0.358905286 | tie:greedy | 1 |
| 4 / 1 | conjunction | 2.98615723e-05 / 2.98615723e-05 | 0 / 0 | 0.0986300409 / 0.309273988 | tie:greedy | 1 |
| 4 / 2 | conjunction | 3.09029861e-06 / 3.09029861e-06 | 0 / 0 | 0.0429474972 / 0.0429474972 | tie:greedy | 0 |
| 4 / 3 | conjunction | 0.332548708 / 0.332548708 | 0 / 0 | 0.0902947858 / 0.507427275 | tie:greedy | 1 |
| 5 / 0 | conjunction | 5.32587046e-06 / 5.32587046e-06 | 0 / 0 | 0.269725114 / 0.269725114 | tie:greedy | 0 |
| 5 / 1 | conjunction | 1.29651453e-05 / 1.29651453e-05 | 0 / 0 | 0.0894940495 / 0.0894940495 | tie:greedy | 0 |
| 5 / 2 | conjunction | 0.136378482 / 0.136378482 | 0 / 0 | 0.150941342 / 0.150941342 | tie:greedy | 0 |
| 5 / 3 | conjunction | 6.64127668e-08 / 6.64127668e-08 | 0 / 0 | 0.199857607 / 0.465106219 | tie:greedy | 1 |
| 6 / 0 | conjunction | 1.39482315e-10 / 1.39482315e-10 | 0 / 0 | 0.0077154031 / 0.222853318 | tie:greedy | 1 |
| 6 / 1 | conjunction | 1.97686987e-08 / 1.97686987e-08 | 0 / 0 | 0.0589537993 / 0.0589537993 | tie:greedy | 0 |
| 6 / 2 | conjunction | 0.138831094 / 0.138831094 | 0 / 0 | 0.134616479 / 0.515315294 | tie:greedy | 1 |
| 6 / 3 | conjunction | 8.68769236e-14 / 8.68769236e-14 | 0 / 0 | 0.000412288995 / 0.000412288995 | tie:greedy | 0 |
| 7 / 0 | conjunction | 0.000123834878 / 0.000123834878 | 0 / 0 | 0.267935455 / 0.393623233 | tie:greedy | 1 |
| 7 / 1 | conjunction | 1.49513362e-05 / 1.49513362e-05 | 0 / 0 | 0.0740784407 / 0.688938558 | tie:greedy | 1 |
| 7 / 2 | conjunction | 0.126638755 / 0.126638755 | 0 / 0 | 0.190551206 / 0.206803277 | tie:greedy | 1 |
| 7 / 3 | conjunction | 0.198700428 / 0.198700428 | 0 / 0 | 0.24507086 / 0.332051724 | tie:greedy | 1 |
| 8 / 0 | conjunction | 1.17327463e-05 / 1.17327463e-05 | 0 / 0 | 0.315938532 / 0.385674506 | tie:greedy | 1 |
| 8 / 1 | conjunction | 5.91635808e-06 / 5.91635808e-06 | 0 / 0 | 0.0151393898 / 0.332713813 | tie:greedy | 1 |
| 8 / 2 | conjunction | 0.104604423 / 0.104604423 | 0 / 0 | 0.0855795741 / 0.0302324928 | tie:greedy | -1 |
| 8 / 3 | conjunction | 1.72982725e-10 / 1.72982725e-10 | 0 / 0 | 0.106347911 / 0.106347911 | tie:greedy | 0 |
| 9 / 0 | conjunction | 5.13777643e-08 / 5.13777643e-08 | 0 / 0 | 0.0838582516 / 0.379840165 | tie:greedy | 1 |
| 9 / 1 | conjunction | 5.65221228e-07 / 5.65221228e-07 | 0 / 0 | 0.0136778429 / 0.43470186 | tie:greedy | 1 |
| 9 / 2 | conjunction | 0.23695755 / 0.23695755 | 0 / 0 | 0.01778071 / 0.296855658 | tie:greedy | 1 |
| 9 / 3 | conjunction | 0.349927753 / 0.349927753 | 0 / 0 | 0.050864622 / 0.673493564 | tie:greedy | 1 |
| 10 / 0 | conjunction | 4.3563424e-09 / 4.3563424e-09 | 0 / 0 | 0.184504658 / 0.184504658 | tie:greedy | 0 |
| 10 / 1 | conjunction | 1.27764565e-06 / 1.27764565e-06 | 0 / 0 | 0.213527024 / 0.325900406 | tie:greedy | 1 |
| 10 / 2 | conjunction | 0.213868976 / 0.213868976 | 0 / 0 | 0.182627723 / 0.182627723 | tie:greedy | 0 |
| 10 / 3 | conjunction | 0.186470911 / 0.186470911 | 0 / 0 | 0.215559393 / 0.419238091 | tie:greedy | 1 |

## Sum controls

| Run | MSE | Band | Checkerboard contrast | Max deviation from 0.5 | Floor |
| --- | ---: | --- | ---: | ---: | --- |
| [1](measurements/sum-01/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 2.38418579e-07 | pass |
| [2](measurements/sum-02/run.log) | 0.25 | at 1/4 | 0 | 2.98023224e-07 | pass |
| [3](measurements/sum-03/run.log) | 0.2499999851 | at 1/4 | -2.98023224e-08 | 2.98023224e-07 | pass |
| [4](measurements/sum-04/run.log) | 0.25 | at 1/4 | 5.96046448e-08 | 2.08616257e-07 | pass |
| [5](measurements/sum-05/run.log) | 0.25 | at 1/4 | 5.96046448e-08 | 1.1920929e-07 | pass |
| [6](measurements/sum-06/run.log) | 0.25 | at 1/4 | 0 | 3.27825546e-07 | pass |
| [7](measurements/sum-07/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 1.78813934e-07 | pass |
| [8](measurements/sum-08/run.log) | 0.25 | at 1/4 | -5.96046448e-08 | 1.78813934e-07 | pass |
| [9](measurements/sum-09/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 2.08616257e-07 | pass |
| [10](measurements/sum-10/run.log) | 0.25 | at 1/4 | 2.98023224e-08 | 3.27825546e-07 | pass |

## MM_xor

Raw MM counts remain visible. The repeated first-forward bisection establishes an RNG-only path, so §20 applies §5’s amendment and this count is not a regression test of the round. The separate paired trajectories are also retained.

| Run | Epochs | Best MSE | Final MSE | Gate |
| --- | ---: | ---: | ---: | --- |
| [1](measurements/mm-01/run.log) | 39 | 0.185359165 | 0.185359165 | pass |
| [2](measurements/mm-02/run.log) | 34 | 0.159015179 | 0.159015179 | pass |
| [3](measurements/mm-03/run.log) | 47 | 0.187595338 | 0.187595338 | pass |
| [4](measurements/mm-04/run.log) | 68 | 0.170888111 | 0.170888111 | pass |
| [5](measurements/mm-05/run.log) | 79 | 0.173547775 | 0.173547775 | pass |
| [6](measurements/mm-06/run.log) | 23 | 0.186212718 | 0.186212718 | pass |
| [7](measurements/mm-07/run.log) | 37 | 0.19826366 | 0.19826366 | pass |
| [8](measurements/mm-08/run.log) | 51 | 0.188463271 | 0.188463271 | pass |
| [9](measurements/mm-09/run.log) | 16 | 0.184351444 | 0.184351444 | pass |
| [10](measurements/mm-10/run.log) | 80 | 0.199263528 | 0.199263528 | pass |

## Reader steps and pole consumers

| Run | Reader updates | Final reader Adam steps | Narrowing rows (kept root) | Compose rows (mean roots) |
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
| [xor-01](measurements/xor-01/run-audit.json) | 400 | [400.0] | 829 | 771 |
| [xor-02](measurements/xor-02/run-audit.json) | 400 | [400.0] | 805 | 795 |
| [xor-03](measurements/xor-03/run-audit.json) | 400 | [400.0] | 794 | 806 |
| [xor-04](measurements/xor-04/run-audit.json) | 400 | [400.0] | 796 | 804 |
| [xor-05](measurements/xor-05/run-audit.json) | 400 | [400.0] | 793 | 807 |
| [xor-06](measurements/xor-06/run-audit.json) | 400 | [400.0] | 805 | 795 |
| [xor-07](measurements/xor-07/run-audit.json) | 400 | [400.0] | 819 | 781 |
| [xor-08](measurements/xor-08/run-audit.json) | 400 | [400.0] | 803 | 797 |
| [xor-09](measurements/xor-09/run-audit.json) | 400 | [400.0] | 802 | 798 |
| [xor-10](measurements/xor-10/run-audit.json) | 400 | [400.0] | 776 | 824 |

Each run also records per-epoch logit ranges, every cost and keep decision, both final operators, image widths and values, containment before/after, and sentence-path gradients. The tenth XOR run includes the ownership audit and analytic/finite-difference checks.

Table postprocessor SHA-256: `f2b803626981b4608c9c7d14c347915dfa160e9c89e9dc14209214b9ec3fc6dc`.
