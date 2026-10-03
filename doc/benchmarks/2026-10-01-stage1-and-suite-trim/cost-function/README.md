# Objective-conflicts stage 1 — measurement only

One fresh unseeded run per configuration and arm. The cut arm omits the trial answer term only; batch-end answer training remains. Initializations differ, so these are measured outcomes, not a controlled estimate of the causal effect of the cut. The cost-function work remains part 4 of item 6.9 and is not implemented at this measurement state.

Parameter groups are listed by physical parameter identity in each arm’s manifest. They may overlap (the native reading map includes generation); overlaps are saved explicitly. A zero gradient and an untrainable or non-parameter codebook are different states. Undefined cosines are shown as —.

The native test uses the production batch of 28 and a 24 GiB slow-worker ceiling; the sweep keeps its 8 GiB ceiling. Native results remain in the preceding stage1 receipt. XOR uses the receipt-local fixture whose only change is reconstructInLoop=true. Gradient groups use the corrected role ownership directly; no post-hoc relabelling is needed.

The native factory performs its unchanged 16-row endpoint evaluation; the observer saves predictions and omits only console rendering. Where the factory supplies no evaluation, it performs one final test pass at the production batch with no optimizer. Both arms retain all training and measurement outcomes.

## XOR_grammar

### step5a

Process: exit; exit 0; 80.40078 seconds; peak 0.6721968 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 4 | 0.1397129 | 0.1309025 | 1.017811e-05 | 0.2970363 |
| expectation | 4 | 0 | 0 | 0 | 0 |
| supplied_answer | 4 | 0.06435586 | 0.03540691 | 0.02244856 | 0.1641611 |
| raw_reconstruction | 4 | 1.397129 | 1.309025 | 0.0001017811 | 2.970362 |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0.1395805 | 0.0003594756 | 0.06296291 | 0.2029029 |
| explore | 0.1688316 | 0.0004438758 | 0.4559155 | 0.625191 |

XOR’s legacy intra-sentence expectation is evaluated only during training. Its final-evaluation E=0 means that branch is inactive, not that prediction is perfect; the last training values above preserve the actual expectation comparison. The final owned byte reconstruction cost is 1.39712858; additional lossRev is 0.0.

Answers: `[0.20396706461906433, 0.5948320627212524, 0.8501715660095215, 0.17091301083564758]`; MSE **0.064355859**; correct **4/4**; class bar **False**; reconstructed **2/4**.
Read-backs: `['world there', 'there there', 'loving world', 'there loving']`; unavailable: `[False, False, False, False]`; contrast: -1.07012355.

Kept minus other trial, in the objective’s weighted trial units. Positive means worse. Every active comparison is included; no tolerance discards a conflict.

| Objective | Compared | Kept worse | Mean worsening | Max worsening | Mean signed difference |
|---|---:|---:|---:|---:|---:|
| reconstruction | 1600 | 21 | 0.2821329 | 0.3245137 | -0.05692523 |
| expectation | 1600 | 207 | 0.002859489 | 0.1098557 | -0.00583034 |
| supplied_answer | 1600 | 88 | 0.107811 | 0.320221 | -0.2687299 |

Gradient snapshots use autograd reads with the exact cached-perception pullback. Both trials precede their first optimizer step and carry matching parameter-version digests. The first later nonzero expectation pair is also retained if the first pair has no expectation.

Observed nonzero reach at the saved states (absence means not observed at these states):

| Scope | Objective | Groups with nonzero gradient |
|---|---|---|
| trial | reconstruction | chooser, codes |
| trial | expectation | chooser, codes, expectation_predictor, perception |
| trial | supplied_answer | chooser, codes, reading_map |
| batch | reconstruction | none observed |
| batch | expectation | none observed |
| batch | supplied_answer | reading_map |
| batch_auxiliary | sbow | perception |

[Codebook parameter/buffer ownership](XOR_grammar-step5a/codebook-ownership.json) records contextual rotation separately from autograd.

trial, pair 0, trial exploit, parameter digest `c40f46e777e1cce2`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0.01871684 | 0 | — | — | — |
| codes | 0.01427685 | 0.03334859 | 0.0003603148 | -0.05551555 | 0.378269 | -0.04694275 |
| chooser | 0.003443148 | 0.001274169 | 0.001113713 | 0.2978423 | 0.9755378 | 0.2020178 |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.08251789 | — | — | — |
| expectation_predictor | 0 | 0.161423 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

trial, pair 0, trial explore, parameter digest `c40f46e777e1cce2`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0.01871684 | 0 | — | — | — |
| codes | 0.01307024 | 0.03334859 | 0.0005661914 | -0.05569401 | -0.3970527 | 0.02643646 |
| chooser | 0.001355544 | 0.001274169 | 0.0004094715 | -0.3652746 | -0.5369721 | -0.2987356 |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.1614557 | — | — | — |
| expectation_predictor | 0 | 0.161423 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

batch, pair 1, trial None, parameter digest `3f9181985582e410`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0 | 0 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.1712722 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

Additional batch objectives: `{'sbow': 0.07373522222042084}`. Their full reach and cosines are in `gradients.json`.

[Every cost term’s magnitude](XOR_grammar-step5a/term-magnitudes.md), [weights and formulas](XOR_grammar-step5a/weights.json), [parameter membership](XOR_grammar-step5a/parameter-groups.json).

### cut

Process: exit; exit 0; 79.90175 seconds; peak 0.67238 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 4 | 0.004475194 | 0.002837285 | 0.0009214242 | 0.01130478 |
| expectation | 4 | 0 | 0 | 0 | 0 |
| supplied_answer | 4 | 0.02142903 | 0.02062792 | 0.018537 | 0.02592329 |
| raw_reconstruction | 4 | 0.04475193 | 0.02837285 | 0.009214242 | 0.1130478 |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0.00447917 | 5.983746e-05 | — | 0.004539008 |
| explore | 0.03438368 | 5.9837e-05 | — | 0.03444351 |

XOR’s legacy intra-sentence expectation is evaluated only during training. Its final-evaluation E=0 means that branch is inactive, not that prediction is perfect; the last training values above preserve the actual expectation comparison. The final owned byte reconstruction cost is 0.0447519347; additional lossRev is 0.0.

Answers: `[0.14214110374450684, 0.8389928936958313, 0.8638493418693542, 0.1450921893119812]`; MSE **0.021429032**; correct **4/4**; class bar **True**; reconstructed **4/4**.
Read-backs: `['world hello', 'there hello', 'loving world', 'there loving']`; unavailable: `[False, False, False, False]`; contrast: -1.41560894.

