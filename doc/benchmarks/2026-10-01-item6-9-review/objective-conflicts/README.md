# Objective-conflicts stage 1 — measurement only

One fresh unseeded run per configuration and arm. The cut arm omits the trial answer term only; batch-end answer training remains. Initializations differ, so these are measured outcomes, not a controlled estimate of the causal effect of the cut. No stage-2 or item-6.85 design is implemented.

Parameter groups are listed by physical parameter identity in each arm’s manifest. They may overlap (the native reading map includes generation); overlaps are saved explicitly. A zero gradient and an untrainable or non-parameter codebook are different states. Undefined cosines are shown as —.

The XOR tables use `gradients-by-role.json`: the original observer followed InputSpace’s registered OutputSpace back-reference, so head parameters appeared in perception and vocabulary parameters appeared in the reading map. `regroup_xor.py` corrects only these labels and assigns the intra-sentence predictor to expectation. It proves every recovered nonzero norm/cosine has exactly the same saved parameter support. The original files remain; no model was rerun. The native observer classifies those owners directly and also includes the model’s named shared transforms in operators/tied inverses.

Both native arms hit the unchanged 8 GiB worker guard before their first cost or gradient snapshot. Their reach, selection, magnitude and endpoint measurements are unavailable. The sampled peaks can exceed 8 GiB between guard polls; the limit was not raised.

## XOR_grammar

### step5a

Process: exit; exit 0; 68.9494 seconds; peak 0.6952835 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 4 | 0 | -0 | 0 | 0 |
| expectation | 4 | 0 | 0 | 0 | 0 |
| supplied_answer | 4 | 0.02812306 | 0.02399639 | 0.008521436 | 0.05597803 |
| raw_reconstruction | 4 | 0 | -0 | 0 | 0 |

The endpoint reconstruction is the trial’s configured objective; XOR’s separate batch D3 cost is in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0 | 0.0005115285 | 0.0279584 | 0.02846993 |
| explore | 0 | 0.0004930489 | 0.3557621 | 0.3562551 |

XOR’s legacy intra-sentence expectation is evaluated only during training. Its final-evaluation E=0 means that branch is inactive, not that prediction is perfect; the last training values above preserve the actual expectation comparison. The final batch D3 reconstruction is 0.713413835; it also appears as lossRev, each with the configured reconstruction weight.

Answers: `[0.23659676313400269, 0.9076883792877197, 0.833829402923584, 0.14275890588760376]`; MSE **0.028123059**; correct **4/4**; class bar **True**; reconstructed **0/4**.
Read-backs: `['world world', 'there world', 'there world', 'there hello']`; unavailable: `[False, False, False, False]`.

Kept minus other trial, in the objective’s weighted trial units. Positive means worse. Every active comparison is included; no tolerance discards a conflict.

| Objective | Compared | Kept worse | Mean worsening | Max worsening | Mean signed difference |
|---|---:|---:|---:|---:|---:|
| reconstruction | 1600 | 0 | — | — | 0 |
| expectation | 1600 | 297 | 0.00317355 | 0.1373406 | -0.003514367 |
| supplied_answer | 1600 | 12 | 0.03248312 | 0.07249915 | -0.2821565 |

Gradient snapshots use autograd reads with the exact cached-perception pullback. Both trials precede their first optimizer step and carry matching parameter-version digests. The first later nonzero expectation pair is also retained if the first pair has no expectation.

Observed nonzero reach at the saved states (absence means not observed at these states):

| Scope | Objective | Groups with nonzero gradient |
|---|---|---|
| trial | reconstruction | none observed |
| trial | expectation | chooser, codes, expectation_predictor, perception |
| trial | supplied_answer | chooser, codes, reading_map |
| batch | reconstruction | none observed |
| batch | expectation | none observed |
| batch | supplied_answer | reading_map |
| batch_auxiliary | sbow | perception |

[Codebook parameter/buffer ownership](XOR_grammar-step5a/codebook-ownership.json) records contextual rotation separately from autograd.

trial, pair 0, trial exploit, parameter digest `2999a5b0cc9e8e77`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0.01936252 | 0 | — | — | — |
| codes | 0 | 0.0284148 | 0.0009124148 | — | — | -0.11709 |
| chooser | 0 | 0.001532963 | 0.0009322573 | — | — | -0.02966448 |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.1895363 | — | — | — |
| expectation_predictor | 0 | 0.1574053 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

trial, pair 0, trial explore, parameter digest `2999a5b0cc9e8e77`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0.01867642 | 0 | — | — | — |
| codes | 0 | 0.02108363 | 0.0008990749 | — | — | -0.01641266 |
| chooser | 0 | 0.003056519 | 0.0008053818 | — | — | -0.6016575 |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.2455168 | — | — | — |
| expectation_predictor | 0 | 0.1310721 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

batch, pair 1, trial None, parameter digest `707ab309847e1faf`:

| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |
|---|---:|---:|---:|---:|---:|---:|
| perception | 0 | 0 | 0 | — | — | — |
| codes | 0 | 0 | 0 | — | — | — |
| chooser | 0 | 0 | 0 | — | — | — |
| operators_and_tied_inverses | 0 | 0 | 0 | — | — | — |
| generate | 0 | 0 | 0 | — | — | — |
| reading_map | 0 | 0 | 0.1152265 | — | — | — |
| expectation_predictor | 0 | 0 | 0 | — | — | — |
| other | 0 | 0 | 0 | — | — | — |

Additional batch objectives: `{'sbow': 0.08025392144918442}`. Their full reach and cosines are in `gradients.json`.

[Every cost term’s magnitude](XOR_grammar-step5a/term-magnitudes.md), [weights and formulas](XOR_grammar-step5a/weights.json), [parameter membership](XOR_grammar-step5a/parameter-groups-by-role.json).

### cut

Process: exit; exit 0; 68.40105 seconds; peak 0.7031112 GiB.

Final evaluation trial costs (answer evaluated at the same state in both arms; no optimizer step and no change to the kept trial):

| Objective | Rows | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| reconstruction | 4 | 0 | 0 | -0 | 0 |
| expectation | 4 | 0 | 0 | 0 | 0 |
| supplied_answer | 4 | 0.1948185 | 0.1943238 | 0.1097556 | 0.280871 |
| raw_reconstruction | 4 | 0 | 0 | -0 | 0 |

The endpoint reconstruction is the trial’s configured objective; XOR’s separate batch D3 cost is in `last_batch` and the term table. Absent objectives have no fabricated value.

Last training comparison, before that sentence’s two optimizer updates (weighted means over active rows; distinct from the final evaluation above):

| Trial | R | E | Supplied answer | Total |
|---|---:|---:|---:|---:|
| exploit | 0 | 0.00153423 | — | 0.00153423 |
| explore | 0 | 0.00153423 | — | 0.00153423 |

XOR’s legacy intra-sentence expectation is evaluated only during training. Its final-evaluation E=0 means that branch is inactive, not that prediction is perfect; the last training values above preserve the actual expectation comparison. The final batch D3 reconstruction is 0.300065219; it also appears as lossRev, each with the configured reconstruction weight.

Answers: `[0.350686252117157, 0.6687062382698059, 0.4700273871421814, 0.5154287219047546]`; MSE **0.19481854**; correct **2/4**; class bar **False**; reconstructed **0/4**.
Read-backs: `['world there', 'there there', 'world there', 'there there']`; unavailable: `[False, False, False, False]`.

## BasicModel_answers_tied_benchmark

Both requested fresh unseeded arms were attempted once. The step-5a arm stopped
at the 8 GiB memory guard after 7.499839 seconds (sampled peak 8.932488 GiB);
the cut arm stopped after 7.495509 seconds (8.931740 GiB). Both exit codes are 137.
Only the batch-open event was recorded. Neither arm reached its first trial cost
or gradient snapshot, and neither supplies a completed training outcome.

Gradient reach, norms, cosines, trial-selection conflicts, cost magnitudes and
endpoint costs are **unavailable** for this configuration. Empty raw tables do
not mean zero gradients or zero costs. This portion of stage 1 is incomplete.
No smaller fixture, altered configuration, additional run or larger guard was used.

The initialized-model observations that were obtained are preserved:

- Step 5a: [effective weights](BasicModel_answers_tied_benchmark-step5a/weights.json),
  [parameter groups](BasicModel_answers_tied_benchmark-step5a/parameter-groups.json),
  [codebook ownership](BasicModel_answers_tied_benchmark-step5a/codebook-ownership.json).
- Cut: [effective weights](BasicModel_answers_tied_benchmark-cut/weights.json),
  [parameter groups](BasicModel_answers_tied_benchmark-cut/parameter-groups.json),
  [codebook ownership](BasicModel_answers_tied_benchmark-cut/codebook-ownership.json).

The native codebooks are non-gradient buffers with contextual rotation ownership;
that is an observed storage/update property, not a measured gradient-reach result.
[Every coefficient and its purpose](weights-purpose.md) is available for both
configurations. The original source, plans, logs, manifests and guard results
remain in each arm's directory.
