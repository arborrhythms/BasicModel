# Objective-conflicts stage 1 — measurement only

One fresh unseeded run per configuration and arm. The cut arm omits the trial answer term only; batch-end answer training remains. Initializations differ, so these are measured outcomes, not a controlled estimate of the causal effect of the cut. The cost-function work remains part 4 of item 6.9 and is not implemented at this measurement state.

Parameter groups are listed by physical parameter identity in each arm’s manifest. They may overlap (the native reading map includes generation); overlaps are saved explicitly. A zero gradient and an untrainable or non-parameter codebook are different states. Undefined cosines are shown as —.

The native test uses the production batch of 28 and a 24 GiB slow-worker ceiling; the sweep keeps its 8 GiB ceiling. The earlier 12 GiB sizing stops are preserved separately. XOR uses the receipt-local fixture whose only change is reconstructInLoop=true. Gradient groups use the corrected role ownership directly; no post-hoc relabelling is needed.

The native factory performs its unchanged 16-row endpoint evaluation; the observer saves predictions and omits only console rendering. Where the factory supplies no evaluation, it performs one final test pass at the production batch with no optimizer. Both arms retain all training and measurement outcomes.

## BasicModel_answers_tied_benchmark

### step5a

Process: exit; exit 0; 1276.782 seconds; peak 22.02584 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 16 | 0.8126668 | 0.8206292 | 7.828225e-05 | 1.514571 |
| expectation | 16 | 0 | 0 | 0 | 0 |
| supplied_answer | 16 | 0.06696155 | 0.06415036 | 0.05728065 | 0.07774051 |
| raw_reconstruction | 16 | 1.625334 | 1.641258 | 0.0001565645 | 3.029141 |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0.379961 | 0 | 0.1459474 | 0.5259084 |
| explore | 0.3435794 | 0 | 0.14363 | 0.4872093 |
Kept minus other trial, in the objective’s weighted trial units. Positive means worse. Every active comparison is included; no tolerance discards a conflict.

| Objective | Compared | Kept worse | Mean worsening | Max worsening | Mean signed difference |
|---|---:|---:|---:|---:|---:|
| reconstruction | 28 | 0 | — | — | -0.03638164 |
| expectation | 28 | 0 | — | — | 0 |
| supplied_answer | 28 | 0 | — | — | -0.002317434 |

Gradient snapshots use autograd reads with the exact cached-perception pullback. Both trials precede their first optimizer step and carry matching parameter-version digests. The first later nonzero expectation pair is also retained if the first pair has no expectation.

Observed nonzero reach at the saved states (absence means not observed at these states):

| Scope | Objective | Groups with nonzero gradient |
|---|---|---|
| trial | reconstruction | chooser, operators_and_tied_inverses, perception |
| trial | expectation | none observed |
| trial | supplied_answer | chooser, generate, operators_and_tied_inverses, perception, reading_map |
| batch | reconstruction | none observed |
| batch | expectation | none observed |
| batch | supplied_answer | generate, perception, reading_map |
| batch_auxiliary | readout_l1 | none observed |

[Codebook parameter/buffer ownership](BasicModel_answers_tied_benchmark-step5a/codebook-ownership.json) records contextual rotation separately from autograd.

trial, pair 0, trial exploit, parameter digest `d359d437641bfc86`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 3.394196e-26 | 0 | 0.2319637 | — | -1.671877e-24 | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 9.467548e-06 | 0 | 0.02301802 | — | -0.5274047 | — |
| operators_and_tied_inverses | 3.606578e-26 | 0 | 1.354092e-21 | — | 0.1627795 | — |
| generate | 0 | 0 | 0.4790322 | — | — | — |
| reading_map | 0 | 0 | 0.4790322 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

trial, pair 0, trial explore, parameter digest `d359d437641bfc86`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 3.680055e-26 | 0 | 0.2275786 | — | 1.643462e-24 | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 9.372213e-06 | 0 | 0.02055129 | — | -0.4547754 | — |
| operators_and_tied_inverses | 5.38416e-05 | 0 | 1.189891e-21 | — | 0.009572579 | — |
| generate | 0 | 0 | 0.4686637 | — | — | — |
| reading_map | 0 | 0 | 0.4686637 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

batch, pair 1, trial None, parameter digest `8bfb19651fb64e7f`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0 | 0.2827622 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0.6002251 | — | — | — |
| reading_map | 0 | 0 | 0.6002251 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

Additional batch objectives: `{'readout_l1': 0.010833333246409893}`. Their full reach and cosines are in `gradients.json`.

[Every cost term’s magnitude](BasicModel_answers_tied_benchmark-step5a/term-magnitudes.md), [weights and formulas](BasicModel_answers_tied_benchmark-step5a/weights.json), [parameter membership](BasicModel_answers_tied_benchmark-step5a/parameter-groups.json).

### cut

Process: exit; exit 0; 1277.31 seconds; peak 21.0425 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 16 | 0.1789082 | 0.002657943 | 0.001359395 | 0.4100563 |
| expectation | 16 | 0 | 0 | 0 | 0 |
| supplied_answer | 16 | 0.1814664 | 0.1835174 | 0.1587818 | 0.1909201 |
| raw_reconstruction | 16 | 0.3578164 | 0.005315887 | 0.00271879 | 0.8201126 |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0.231799 | 0 | — | 0.231799 |
| explore | 0.1896471 | 0 | — | 0.1896471 |
## XOR_grammar

### step5a

Process: exit; exit 1; 4.127944 seconds; peak 0.5399639 GiB.

Observer/run error: `{'type': 'RuntimeError', 'message': 'tied reconstruction bank requires staged word and sentence masks', 'seconds': 0.5913316669975757}`

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| unavailable | 0 | — | — | — | — |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

Kept minus other trial, in the objective’s weighted trial units. Positive means worse. Every active comparison is included; no tolerance discards a conflict.

| Objective | Compared | Kept worse | Mean worsening | Max worsening | Mean signed difference |
|---|---:|---:|---:|---:|---:|

Gradient snapshots use autograd reads with the exact cached-perception pullback. Both trials precede their first optimizer step and carry matching parameter-version digests. The first later nonzero expectation pair is also retained if the first pair has no expectation.

Observed nonzero reach at the saved states (absence means not observed at these states):

| Scope | Objective | Groups with nonzero gradient |
|---|---|---|

[Codebook parameter/buffer ownership](XOR_grammar-step5a/codebook-ownership.json) records contextual rotation separately from autograd.

[Every cost term’s magnitude](XOR_grammar-step5a/term-magnitudes.md), [weights and formulas](XOR_grammar-step5a/weights.json), [parameter membership](XOR_grammar-step5a/parameter-groups.json).

### cut

Process: exit; exit 1; 4.136367 seconds; peak 0.5412762 GiB.

Observer/run error: `{'type': 'RuntimeError', 'message': 'tied reconstruction bank requires staged word and sentence masks', 'seconds': 0.5946170420065755}`

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| unavailable | 0 | — | — | — | — |

The endpoint reconstruction is the trial’s configured objective; The separate batch reconstruction and reverse costs are in `last_batch` and the term table. Absent objectives have no fabricated value.

