# Item 4: gain-field toys, implementation and native measurement

**Implementation update:** the preparation below is preserved as its original
receipt. An uncommitted production candidate is under validation; Item 4 remains
incomplete. See the
[execution record](../../plans/2026-10-19-item4-start.md). The separate
[native measurement protocol](native_protocol.md) and [harness](native_experiment.py)
exercise the real paired sentence optimizer. Neither a toy nor a lower scalar
reconstruction cost establishes the spec's derivation-optimality gate.
The [native readout](native_readout.md) records a failed gate, not a learning
success; remaining compose/proposal integration and boundary timeouts are in
the execution record.

**Started from BasicModel `9207f5e03`, WikiOracle `929e4a8`, and the working-tree
rev 5 spec.** Required toys 2, 7 and 8 are complete. Toys 5–6 also ran as
diagnostics. This is preparation for the
[implementation handoff](../../plans/2026-10-19-item4-start.md), not a mechanism
landing or a BasicModel learning result. Production code, the user's spec
revision, and the separate 5.5 work were not changed.

The [protocol](PROTOCOL.md) fixes the budgets, arms and seeds. The
[script](toys.py) and [results](results.json) retain all arms. The
[first readout](initial-results.json) is preserved; the final readout adds
the explicitly declared discrete-choice witness below. No failed result was
removed or seed selected. Toy 1's prior Viterbi result remains in its
[original receipt](../2026-10-08-viterbi-departure-toy/README.md); withdrawn toys
3–4 belong to Item 3.

Reproduce from `basicmodel` (use a new output path to preserve this receipt):

```sh
PYTHONPATH=bin .venv/bin/python doc/benchmarks/2026-10-19-item4-expectation/toys.py --out /tmp/item4-toys.json
```

## 2. Which objective credits the departure?

Sixteen initializations, 256 paired updates each. Option 0 explains the
observation; option 1 is predictable omission. This is a two-option tabular
toy with fixed terminal costs, not the native chooser's learning rate or loss.

| Terminal judge | Mean probability of explanation | Greedy explanation |
|---|---:|---:|
| R + A | .999999412 | 16/16 |
| R + E + A | 7.83e-16 | 0/16 |

The **landed source already uses R + A for departure credit**:
`SentenceCredit.comparison` and the nested scorer in
`BasicModel._run_sentence_word_bricks`. E still trains its own predictor. The
October 8 audit in spec §2 is stale on this point. Keeping is different:
strictly lower **R** retains the explore path. The native comparison witness
has `(R,A)=(.3,.6)` versus `(.4,.1)`: credit favors explore (.5 vs .9), while
keep retains greedy (.3 vs .4). Changing E from `(99,0)` to `(0,999)` changes
neither decision. Rev 5's unified R+A judge would change **keeping**, not
introduce the existing E exclusion.

## 5. Minimal mean and variance: capture and its limit

The observable is a mask-input proxy, not a learned placement or admission.
Confidence is explicitly withheld for the first 32 observations; the zero
cold-start result is therefore an imposed control, not learned behavior.

| Point in the stream | Salience proxy |
|---|---:|
| Cold observations, maximum | 0 |
| Last .2 before first .8 | .019811 |
| First .8 | 1 |
| Last 32 of 256 repeated .8 observations, mean | .002207 |
| Return to .2 | 1 |

The minimal statistic detects a marginal change and habituates. It cannot
establish the full higher-order signature in toy 5: alternation and a broken
ordering with the same calibrated marginal moments give **identical**
normalized residuals. A richer transition rule remains proposed; it is not
silently added to the decided mean/variance predictor.

## 6. Normalization is not noise rejection

Eight streams, each 4096 Bernoulli(.5) observations, evaluated against their
exact mean .5 and variance .25. These are already correct predictions of
the distribution; there is no model error to fix.

| Quantity | Mean across seeds |
|---|---:|
| Absolute residual | .5 |
| Saturated normalized residual | .999998 |
| Beta posterior/prior KL, first 64 updates | .030777 |
| Beta posterior/prior KL, last 1024 updates | .00014054 |

Variance normalization removes scale; it does **not** make irreducible noise
low-gain. The stronger noisy-TV claim in spec §4.5 is unsupported even with
perfect moments. The Beta arm demonstrates declining information gain under
one stationary, accumulating-count model. It does not select a replacement
novelty objective or prove adaptation under change. The optional novelty
mechanism and the higher-order capture gate stay open.

## 7. Content, omission and gradient isolation

Eight mixed streams: constant .75 and unpredictable binary symbols. The
two-output local predictor is trained only against detached observations.
At `g=.9`, the constant passes .075 for all three saturations. Every gain
arm stays in `[0,o]`; zero gain reproduces `o` exactly. Subtraction has
negative coordinates (minimum across arms -.53515625; at `g=.9`, -.481640625).

| Saturation at g=.9 | Mean constant content | Mean noisy content |
|---|---:|---:|
| Relative, clipped at 1 (spec default) | .075 | .485376 |
| Relative, smooth r/(e+r+epsilon) | .075 | .275211 |
| Absolute, clipped at 1 | .075 | .275211 |

The relative clipped rule preserves the spec's saturation semantics. This
toy provides no learning-quality reason to select another saturation or a
production floor. An expected-but-absent `.8` gives content 0, positive
residual 0, negative residual .8. A matched observation with zero confidence
or an object exemption passes unchanged. Exact region exclusion gives zero
content. Opposing evidence lanes are compared separately: `(1,0)` versus
`(0,1)` is a full mismatch, not a zero net residual.

A separate predictor is asked to predict .75 locally while a downstream
objective rewards brighter content. All arms begin at .6 and use identical
updates:

| Gradient boundary | Initial downstream gradient to predictor | Final prediction | Local MSE |
|---|---:|---:|---:|
| No cut | 1.423826 | .371952 | .142920 |
| Residual numerator only detached | .284765 | .391457 | .128553 |
| Entire saturation detached | 0 | .75 | 8.40e-26 |
| Local loss only, control | 0 | .75 | 8.40e-26 |

Detach the **whole saturation**, including its expected-magnitude denominator.
All local-loss source gradients are absent. This prevents downstream gradient
gaming, but does not protect discrete chooser credit if its target is changed.
The added witness compares reconstructions `.75` and `0` against an observation
`.75`, perfectly predicted and attenuated to `.075`:

| Reconstruction target | Cost of explanation .75 | Cost of omission 0 | Preferred choice |
|---|---:|---:|---|
| Whole observation .75 | 0 | .5625 | Explanation |
| Attenuated content .075 | .455625 | .005625 | Omission |

All three detach arms make the same discrete choices. R must keep the whole
observation as its target and denominator; a stop-gradient alone does not
enforce that requirement. These are mechanism counterexamples, not estimates
of frequency in the trained model.

## 8. Identity must survive independently of brightness

This calls the production `VectorQuantize` with fixed prototypes and EMA off.
Both rows are in the percept cube. Index 0 is whiskers, index 1 is the cat.
The first two cat coordinates are predicted; the third is unexpected.

| Metric / g | Coordinate gain identifies | Pooled scalar identifies | Perfectly predicted scalar identifies |
|---|---|---|---|
| Cosine / 0 | Cat | Cat | Cat |
| Cosine / .5 | Cat | Cat | Cat |
| Cosine / .9 | Whiskers | Cat | Cat |
| Cosine / 1 | Whiskers | Cat | Whiskers (zero-vector tie) |
| Euclidean / 0 | Cat | Cat | Cat |
| Euclidean / .5 | Cat | Cat | Cat |
| Euclidean / .9 | Whiskers | Cat | Whiskers (positive scalar .1) |
| Euclidean / 1 | Whiskers | Cat | Whiskers (zero vector) |

Positive scalar scaling preserves direction and cosine identity. It does
not preserve arbitrary Euclidean nearest-neighbor identity, and no metric
can recover the original identity from zero alone. Identifying the original
observation first and retaining its reference preserves the cat in **all
eight** cases, including zero brightness. Scalar content gain then applies
to activation, never to the evidence used to identify the observation. This
is the recommended integration, rather than changing the codebook metric or
weakening the spec's identity gate. The raw observation and estimate must
remain separately accessible for the fully expected cat.

## Boundary checks and scope

Ten existing checks passed in the bounded runner: gain/object-mask boundaries,
source/predictor isolation, raw composition independence on the existing
stance path, normalized retention surprise, linked estimate/observation
restoration, atomic pair overflow, and exclusion of estimates from ordinary
LTM history. The local receipt is
`output/item4-start-boundaries-20261019/result.json` (10/10, 10.22 s).
These validate carried contracts on the current checkout; they do not validate
the proposed gain architecture. The checkout has concurrent 5.5 edits, so this
is not advertised as a pristine-checkout full baseline sweep.

Four changed/new Markdown link checks also passed in
`output/item4-start-links-20261019/result.json`. A further source audit found
that `_prediction_inputs` detaches older frames but intentionally keeps the
newest ordinary source live; `_observe_meanings` retains that graph during
training. The baseline's explicit full-role-loss certificate requires source
encoder credit. The spec's claim that sources are already detached everywhere
is therefore too broad: rev 5 must change this API contract when it migrates
the owner. Objective-level optimizer ownership remains a separate boundary.
That unchanged certificate passed 1/1 in
`output/item4-start-source-credit-20261019/result.json`; this confirms the
existing live-source behavior, not the new detached-source gate.

The required XOR/math learning comparisons, graded/shuffled BasicModel run,
and full landing sweep are **not run**. Item 0's paired `.0517` counterexample,
c's marker-headed `five`, and the other carried learning findings are not
reclassified by passing boundary checks. No NP, constituent, identity, VP,
order-2, FineWeb or mature-checkpoint learning gate changes status here.
