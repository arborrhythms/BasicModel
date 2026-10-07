# Round-2c measurement tables

These tables read the saved runs only. Source, budgets, thresholds and runs are unchanged.

Standing gate passed: **False**. Thirty gate trainings; zero retries or replacements.

## Class and reconstruction

The answer is read from each final greedy committed root. Each row below is one shared training for both gates.

| Run | MSE | §20.5 band | Correct | Multisets | Class | Reconstruction | Final operator | Last epoch with any greedy disjunction |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| [1](measurements/xor-01/run.log) | 0.21850897 | between | 3/4 | 4/4 | fail | pass | conjunction | 308 |
| [2](measurements/xor-02/run.log) | 0.0515852351 | between | 4/4 | 4/4 | fail | pass | conjunction | 150 |
| [3](measurements/xor-03/run.log) | 0.01129213 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 20 |
| [4](measurements/xor-04/run.log) | 0.0712649584 | between | 4/4 | 4/4 | fail | pass | conjunction | 4 |
| [5](measurements/xor-05/run.log) | 0.0488325414 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [6](measurements/xor-06/run.log) | 0.047240047 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 121 |
| [7](measurements/xor-07/run.log) | 0.176236955 | between | 4/4 | 4/4 | fail | pass | conjunction | 290 |
| [8](measurements/xor-08/run.log) | 0.0478399336 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [9](measurements/xor-09/run.log) | 0.0361345913 | at 0 | 4/4 | 4/4 | pass | pass | conjunction | 0 |
| [10](measurements/xor-10/run.log) | 0.159531065 | between | 3/4 | 4/4 | fail | pass | conjunction | 331 |

## Cost and policy audit by run

Negative advantage rewards the explore action; positive advantage rewards greedy. E is separately retained in every raw trial. “Against keep” counts the rows where the total rewards a different trial from strict reconstruction.

| Run | Walk/action | Departures | Nonzero | Reward explore / greedy | Keep explore | Answer against keep | Sum ΔR | Sum ΔE | Sum ΔA |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| [1](measurements/xor-01/run-audit.json) | compose/conjunction | 420 | 420 | 218 / 202 | 0 | 218 | 0 | 0 | -0.907010764 |
| [1](measurements/xor-01/run-audit.json) | compose/disjunction | 140 | 140 | 49 / 91 | 0 | 49 | 0 | 0 | 6.02842025 |
| [1](measurements/xor-01/run-audit.json) | narrowing/and | 192 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/gloss | 501 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/not | 184 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [1](measurements/xor-01/run-audit.json) | narrowing/or | 160 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | compose/conjunction | 173 | 173 | 90 / 83 | 1 | 89 | -0.0290394351 | 0 | -0.29617013 |
| [2](measurements/xor-02/run-audit.json) | compose/disjunction | 343 | 343 | 38 / 305 | 0 | 38 | 0 | 0 | 79.7718367 |
| [2](measurements/xor-02/run-audit.json) | narrowing/and | 186 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/gloss | 546 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/not | 180 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [2](measurements/xor-02/run-audit.json) | narrowing/or | 170 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | compose/conjunction | 13 | 13 | 7 / 6 | 0 | 7 | 0 | 0 | 0.0298007578 |
| [3](measurements/xor-03/run-audit.json) | compose/disjunction | 492 | 492 | 13 / 479 | 0 | 13 | 16.8945959 | 0 | 190.602307 |
| [3](measurements/xor-03/run-audit.json) | narrowing/and | 185 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/gloss | 532 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/not | 189 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [3](measurements/xor-03/run-audit.json) | narrowing/or | 186 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | compose/disjunction | 524 | 524 | 41 / 483 | 0 | 41 | 1.68407643 | 0 | 95.8327252 |
| [4](measurements/xor-04/run-audit.json) | narrowing/and | 170 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/gloss | 529 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/not | 182 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [4](measurements/xor-04/run-audit.json) | narrowing/or | 192 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | compose/disjunction | 545 | 545 | 79 / 466 | 0 | 79 | 0 | 0 | 138.609766 |
| [5](measurements/xor-05/run-audit.json) | narrowing/and | 188 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/gloss | 530 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/not | 167 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [5](measurements/xor-05/run-audit.json) | narrowing/or | 169 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | compose/conjunction | 99 | 99 | 20 / 79 | 0 | 20 | 0 | 0 | 8.52114905 |
| [6](measurements/xor-06/run-audit.json) | compose/disjunction | 448 | 448 | 52 / 396 | 0 | 52 | 0 | 0 | 117.95059 |
| [6](measurements/xor-06/run-audit.json) | narrowing/and | 167 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/descend | 3 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/gloss | 492 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/not | 192 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [6](measurements/xor-06/run-audit.json) | narrowing/or | 199 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | compose/conjunction | 385 | 385 | 198 / 187 | 0 | 198 | 0 | 0 | -1.50449976 |
| [7](measurements/xor-07/run-audit.json) | compose/disjunction | 154 | 154 | 57 / 97 | 0 | 57 | 0 | 0 | 13.5172357 |
| [7](measurements/xor-07/run-audit.json) | narrowing/and | 164 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/gloss | 552 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/not | 165 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [7](measurements/xor-07/run-audit.json) | narrowing/or | 178 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | compose/disjunction | 550 | 550 | 9 / 541 | 0 | 9 | 0 | 0 | 114.824961 |
| [8](measurements/xor-08/run-audit.json) | narrowing/and | 170 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/descend | 1 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/gloss | 528 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/not | 187 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [8](measurements/xor-08/run-audit.json) | narrowing/or | 164 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | compose/disjunction | 546 | 546 | 95 / 451 | 0 | 95 | 0 | 0 | 152.913628 |
| [9](measurements/xor-09/run-audit.json) | narrowing/and | 173 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/gloss | 521 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/not | 179 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [9](measurements/xor-09/run-audit.json) | narrowing/or | 179 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | compose/conjunction | 393 | 393 | 194 / 199 | 0 | 194 | 0 | 0 | 1.08623397 |
| [10](measurements/xor-10/run-audit.json) | compose/disjunction | 125 | 125 | 25 / 100 | 0 | 25 | 0 | 0 | 24.7967891 |
| [10](measurements/xor-10/run-audit.json) | narrowing/and | 179 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/descend | 2 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/gloss | 540 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/not | 176 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| [10](measurements/xor-10/run-audit.json) | narrowing/or | 185 | 0 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |

### Final training commitments and selecting costs

The paired components are before either trial update. The keep is R alone; the credit uses R+E+A. Evaluation has no supplied answer and its A term is zero. The complete final evaluation derivations and costs are in `measurements/summary.json`.

| Run / row | Kept operator | R greedy / explore | E greedy / explore | A greedy / explore | Keep | Advantage sign |
| --- | --- | --- | --- | --- | --- | ---: |
| 1 / 0 | conjunction | 3.7857194e-06 / 3.7857194e-06 | 0 / 0 | 0.360839069 / 0.360839069 | tie:greedy | 0 |
| 1 / 1 | conjunction | 1.61306968e-08 / 1.61306968e-08 | 0 / 0 | 0.203867465 / 0.203867465 | tie:greedy | 0 |
| 1 / 2 | conjunction | 0.0925081372 / 0.0925081372 | 0 / 0 | 0.293365449 / 0.347818106 | tie:greedy | 1 |
| 1 / 3 | conjunction | 0.183579654 / 0.183579654 | 0 / 0 | 0.378991574 / 0.378991574 | tie:greedy | 0 |
| 2 / 0 | conjunction | 2.1706803e-09 / 2.1706803e-09 | 0 / 0 | 0.00849034544 / 0.241398796 | tie:greedy | 1 |
| 2 / 1 | conjunction | 2.95783058e-14 / 2.95783058e-14 | 0 / 0 | 0.155715629 / 0.155715629 | tie:greedy | 0 |
| 2 / 2 | conjunction | 0.28439793 / 0.28439793 | 0 / 0 | 0.253275126 / 0.386114568 | tie:greedy | 1 |
| 2 / 3 | conjunction | 0.445175081 / 0.445175081 | 0 / 0 | 0.000146120932 / 0.000146120932 | tie:greedy | 0 |
| 3 / 0 | conjunction | 1.28682718e-08 / 1.28682718e-08 | 0 / 0 | 0.0417881198 / 0.0417881198 | tie:greedy | 0 |
| 3 / 1 | conjunction | 2.8112147e-06 / 2.8112147e-06 | 0 / 0 | 0.00119978213 / 0.00119978213 | tie:greedy | 0 |
| 3 / 2 | conjunction | 7.75975877e-07 / 7.75975877e-07 | 0 / 0 | 0.00183447567 / 0.00183447567 | tie:greedy | 0 |
| 3 / 3 | conjunction | 0.553510427 / 0.553510427 | 0 / 0 | 0.0248032212 / 0.0248032212 | tie:greedy | 0 |
| 4 / 0 | conjunction | 3.64012753e-09 / 3.64012753e-09 | 0 / 0 | 0.017729748 / 0.017729748 | tie:greedy | 0 |
| 4 / 1 | conjunction | 2.52091536e-06 / 2.52091536e-06 | 0 / 0 | 0.269629359 / 0.619079173 | tie:greedy | 1 |
| 4 / 2 | conjunction | 0.168286577 / 0.168286577 | 0 / 0 | 0.140008524 / 0.968647659 | tie:greedy | 1 |
| 4 / 3 | conjunction | 0.188595146 / 0.188595146 | 0 / 0 | 0.0445777215 / 0.0445777215 | tie:greedy | 0 |
| 5 / 0 | conjunction | 4.03436707e-06 / 4.03436707e-06 | 0 / 0 | 0.197738543 / 0.197738543 | tie:greedy | 0 |
| 5 / 1 | conjunction | 1.67102741e-07 / 1.67102741e-07 | 0 / 0 | 0.000334709737 / 0.000334709737 | tie:greedy | 0 |
| 5 / 2 | conjunction | 0.150870368 / 0.150870368 | 0 / 0 | 0.0141652487 / 0.0141652487 | tie:greedy | 0 |
| 5 / 3 | conjunction | 3.90695046e-11 / 3.90695046e-11 | 0 / 0 | 0.0985644534 / 0.0985644534 | tie:greedy | 0 |
| 6 / 0 | conjunction | 1.32877219e-07 / 1.32877219e-07 | 0 / 0 | 0.130720451 / 0.130720451 | tie:greedy | 0 |
| 6 / 1 | conjunction | 1.82366655e-09 / 1.82366655e-09 | 0 / 0 | 0.0231347959 / 0.0231347959 | tie:greedy | 0 |
| 6 / 2 | conjunction | 0.141870424 / 0.141870424 | 0 / 0 | 0.0374709293 / 0.0731619969 | tie:greedy | 1 |
| 6 / 3 | conjunction | 0.337876648 / 0.337876648 | 0 / 0 | 0.0643097758 / 0.0643097758 | tie:greedy | 0 |
| 7 / 0 | conjunction | 1.38495961e-05 / 1.38495961e-05 | 0 / 0 | 0.280804098 / 0.280804098 | tie:greedy | 0 |
| 7 / 1 | conjunction | 1.23564939e-06 / 1.23564939e-06 | 0 / 0 | 0.139928043 / 0.139928043 | tie:greedy | 0 |
| 7 / 2 | conjunction | 0.0890009031 / 0.0890009031 | 0 / 0 | 0.260747105 / 0.260747105 | tie:greedy | 0 |
| 7 / 3 | conjunction | 3.40203627e-07 / 3.40203627e-07 | 0 / 0 | 0.31475991 / 0.31475991 | tie:greedy | 0 |
| 8 / 0 | conjunction | 2.78106891e-05 / 2.78106891e-05 | 0 / 0 | 0.0521906018 / 0.0521906018 | tie:greedy | 0 |
| 8 / 1 | conjunction | 1.24482946e-09 / 1.24482946e-09 | 0 / 0 | 0.0636967048 / 0.0636967048 | tie:greedy | 0 |
| 8 / 2 | conjunction | 0.194989875 / 0.194989875 | 0 / 0 | 0.129654706 / 0.129654706 | tie:greedy | 0 |
| 8 / 3 | conjunction | 8.99265905e-13 / 8.99265905e-13 | 0 / 0 | 0.0337839462 / 0.0337839462 | tie:greedy | 0 |
| 9 / 0 | conjunction | 4.44117404e-06 / 4.44117404e-06 | 0 / 0 | 0.0488318764 / 0.192003772 | tie:greedy | 1 |
| 9 / 1 | conjunction | 6.3961755e-14 / 6.3961755e-14 | 0 / 0 | 0.0221627466 / 0.0221627466 | tie:greedy | 0 |
| 9 / 2 | conjunction | 0.219057798 / 0.219057798 | 0 / 0 | 0.10814286 / 0.206559792 | tie:greedy | 1 |
| 9 / 3 | conjunction | 0.391005725 / 0.391005725 | 0 / 0 | 0.0164528154 / 0.0164528154 | tie:greedy | 0 |
| 10 / 0 | conjunction | 6.73210465e-09 / 6.73210465e-09 | 0 / 0 | 0.0563189089 / 0.0563189089 | tie:greedy | 0 |
| 10 / 1 | conjunction | 7.83899855e-12 / 7.83899855e-12 | 0 / 0 | 0.256263942 / 0.256263942 | tie:greedy | 0 |
| 10 / 2 | conjunction | 0.0765638873 / 0.0765638873 | 0 / 0 | 0.425880283 / 0.425880283 | tie:greedy | 0 |
| 10 / 3 | conjunction | 0.316027135 / 0.316027135 | 0 / 0 | 0.0386702418 / 0.400249183 | tie:greedy | 1 |

## Sum controls

| Run | MSE | Band | Checkerboard contrast | Max deviation from 0.5 | Floor |
| --- | ---: | --- | ---: | ---: | --- |
| [1](measurements/sum-01/run.log) | 0.2499999851 | at 1/4 | 0 | 1.49011612e-07 | pass |
| [2](measurements/sum-02/run.log) | 0.2499999851 | at 1/4 | -5.96046448e-08 | 1.78813934e-07 | pass |
| [3](measurements/sum-03/run.log) | 0.2499999851 | at 1/4 | -5.96046448e-08 | 2.08616257e-07 | pass |
| [4](measurements/sum-04/run.log) | 0.25 | at 1/4 | 2.98023224e-08 | 1.78813934e-07 | pass |
| [5](measurements/sum-05/run.log) | 0.25 | at 1/4 | 2.98023224e-08 | 1.78813934e-07 | pass |
| [6](measurements/sum-06/run.log) | 0.25 | at 1/4 | 0 | 6.55651093e-07 | pass |
| [7](measurements/sum-07/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 8.94069672e-08 | pass |
| [8](measurements/sum-08/run.log) | 0.25 | at 1/4 | -2.98023224e-08 | 2.68220901e-07 | pass |
| [9](measurements/sum-09/run.log) | 0.25 | at 1/4 | 0 | 2.38418579e-07 | pass |
| [10](measurements/sum-10/run.log) | 0.25 | at 1/4 | 5.96046448e-08 | 4.17232513e-07 | pass |

## MM_xor

Each run is live. The separate paired comparison has zero identical trajectories out of three; its first-forward path is the retired sampling draws followed by the Bernoulli reconstruction mask.

| Run | Epochs | Best MSE | Final MSE | Gate |
| --- | ---: | ---: | ---: | --- |
| [1](measurements/mm-01/run.log) | 200 | 0.200930029 | 0.252170831 | fail |
| [2](measurements/mm-02/run.log) | 82 | 0.133164018 | 0.133164018 | pass |
| [3](measurements/mm-03/run.log) | 45 | 0.187565893 | 0.187565893 | pass |
| [4](measurements/mm-04/run.log) | 49 | 0.186232403 | 0.186232403 | pass |
| [5](measurements/mm-05/run.log) | 24 | 0.19117257 | 0.19117257 | pass |
| [6](measurements/mm-06/run.log) | 59 | 0.183509126 | 0.183509126 | pass |
| [7](measurements/mm-07/run.log) | 55 | 0.191354915 | 0.191354915 | pass |
| [8](measurements/mm-08/run.log) | 97 | 0.194874167 | 0.194874167 | pass |
| [9](measurements/mm-09/run.log) | 38 | 0.198028788 | 0.198028788 | pass |
| [10](measurements/mm-10/run.log) | 115 | 0.176601827 | 0.176601827 | pass |

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
| [xor-01](measurements/xor-01/run-audit.json) | 400 | [400.0] | 1040 | 560 |
| [xor-02](measurements/xor-02/run-audit.json) | 400 | [400.0] | 1084 | 516 |
| [xor-03](measurements/xor-03/run-audit.json) | 400 | [400.0] | 1095 | 505 |
| [xor-04](measurements/xor-04/run-audit.json) | 400 | [400.0] | 1076 | 524 |
| [xor-05](measurements/xor-05/run-audit.json) | 400 | [400.0] | 1055 | 545 |
| [xor-06](measurements/xor-06/run-audit.json) | 400 | [400.0] | 1053 | 547 |
| [xor-07](measurements/xor-07/run-audit.json) | 400 | [400.0] | 1061 | 539 |
| [xor-08](measurements/xor-08/run-audit.json) | 400 | [400.0] | 1050 | 550 |
| [xor-09](measurements/xor-09/run-audit.json) | 400 | [400.0] | 1054 | 546 |
| [xor-10](measurements/xor-10/run-audit.json) | 400 | [400.0] | 1082 | 518 |

Each run also records per-epoch logit ranges, every cost and keep decision, both final operators, image widths and values, containment before/after, and sentence-path gradients. The tenth XOR run includes the ownership audit and analytic/finite-difference checks.

Table postprocessor SHA-256: `098c53ba322f3b15820ff7a3111dc008c72bfd262dbfae0132c109fa8b68242b`.
