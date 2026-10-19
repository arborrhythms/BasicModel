# Item 4 pre-handoff measurements

Declared before running, October 19, 2026. The starting commits are BasicModel
`9207f5e03` and WikiOracle `929e4a8`. The input is the **working-tree rev 5** of
[the expectation spec](../../specs/2026-10-08-expectation-at-every-level.md),
including its withdrawal of subtraction and cross-level inversion. Existing
5.5/POS/NP edits and nanochat changes are separate.

This pass runs the required toys 2, 7 and 8 and the diagnostic toys 5–6. It
does not train BasicModel, run the NP continuation, alter learning gates, or
claim an Item 4 mechanism landing. Toy 1 already has its
[receipt](../2026-10-08-viterbi-departure-toy/README.md); toys 3–4 moved to Item 3.
Production source and the updated spec are inputs, not edited by these toys.

## Fixed measurements

| Question | Experiment and controls | Report |
|---|---|---|
| 2: Does E change the departure's preference? | Two terminal options: explanation `(R,A,E)=(.1,0,1)`; predictable omission `(.4,0,0)`. Paired, single-departure credit for 256 updates at LR .1. Sixteen initial-logit draws, seeds 0–15, shared between R+A and R+E+A arms. Also call the landed `SentenceCredit.comparison` on explicit costs. | Probability of the explanation and selected option, by arm. Separately report the landed keep rule and credit rule; do not assume the spec's October 8 audit describes October 19. |
| 5: What can the minimal shallow statistic capture? | One field coordinate: .2 for 128 arrivals, .8 for 256, then .2 for 64. Prior mean .5, variance .25, EMA rate .05; read before update; confidence starts after 32 arrivals. Mask-placement proxy is confidence times saturated normalized residual, with a zero-statistic control. An additional alternating/broken-order probe has the same marginal mean and variance. | First violation, habituation, return, cold start and the order-blind counterexample. This is a salience-input proxy, not a trained attention model. |
| 6: Does variance normalization stop noisy-TV capture? | Calibrated Bernoulli(.5) noise in [0,1], 4096 arrivals for each seed 0–7. Compare raw absolute residual, normalized/clipped residual and exact Beta(1,1) posterior-to-prior KL per update. | Early/late means and all seeds; distinguish uncertainty normalization from information gain. The Bayesian arm is a diagnostic, not an approved replacement predictor. |
| 7: Gain, gradients and omission | 1024 arrivals per seed 0–7, with one constant .75 symbol and one Bernoulli(.5) symbol. One two-output predictor learns detached local MSE (128 full-batch steps, LR .5). Compare relative clipped, relative smooth and absolute clipped residual saturation, gains 0/.5/.9/1, plus subtraction. Separately train the same scalar predictor against a conflicting downstream brightness objective, with full, partial and no gain detachment. | Content bounds; floor; predictable/noisy signal; residual; omission; object-mask, confidence, region and two-evidence-lane witnesses; predictor drift and gradients. Detaching only the residual numerator is a distinct, intentionally unsafe control. |
| 8: Does scalar attenuation preserve identification? | Two [0,1] prototypes: whiskers `(0,0,1)` and cat `(.8,.8,.2)`, with the cat's first two coordinates predicted. Gains 0/.5/.9/1; coordinate gain vs pooled scalar. Call the production `VectorQuantize` in cosine and Euclidean modes with EMA off. Also test a perfectly predicted input at gain 1 and an identification-before-gain carrier. | Native selected identities, positive scalar invariance, zero-scalar tie, and Euclidean counterexample. No claim that a zero vector can carry its own identity. |

`epsilon=1e-6`, CPU, float64 for toy arithmetic. Native quantizer calls use
float32, as their production buffers do. Seeds make measurements reproducible;
none is selected to turn a failed assertion into a pass. All arms and failures
are reported; no BasicModel learning budget is spent. Algebraic checks validate
the measuring code, not the architectural learning claims. Proposed defaults
and integration choices follow the measurements, not a search for passing seeds.

## Acceptance boundary

The deliverable is executable evidence and a source-audited start plan. Full
architecture gates in spec §9 remain open: XOR/math controls, local predictor
learning, fully expected cat, omission, identity, and graded versus shuffled
free derivation. The item-0 carried findings and independent 5.5 prerequisites
retain their existing status. No full landing sweep is warranted before a
production candidate exists.

## Follow-up declared after the first toy readout

The first run's gradient control shows that complete detachment removes the
predictor's downstream gradient. It does not establish safe *discrete* chooser
credit. Add one fixed analytic witness to toy 7: observed .75, perfectly
predicted .75, gain .9, two reconstructions .75 and 0. Compare scoring against
the raw observation with scoring against attenuated content, for every detach
arm. Report both choices without training or tuning. Keep the first run as
`initial-results.json`; `results.json` includes this additional witness. This
follow-up does not repeat a BasicModel run or use a learning budget.
