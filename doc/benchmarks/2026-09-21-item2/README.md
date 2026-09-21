# Expectation negative-image measurements — September 21

Base revision: `2de9672` (including the reviewed item-1d landing and the two
baseline documentation commits). The implementation, limitations and migration
are in [ExpectationRetention](../../ExpectationRetention.md).

The predictor passes the bounded representation-controlled learning comparisons.
**Useful anticipatory querying and a reasoning-work advantage remain unproven.**
These measurements are not a reason to close those item-2 learning gates or
move them to item 4.

## Synthetic control

Reproduce with the existing virtual environment, from `basicmodel/`:

```sh
PYTHONPATH=bin:test OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python bin/bench_sentence_expectation.py synthetic --threads 1 --updates 300 --out doc/benchmarks/2026-09-21-item2/synthetic.json
```

The existing fixed-meaning dynamics study uses 300 updates, 368 held-out pairs
per seed and a frozen representation. Its predeclared 20% improvement over both
trained controls passes for every seed. It does not test language learning or
joint encoder learning.

| Seed | Ordered MSE | Shuffled MSE | Context-free MSE |
|---|---:|---:|---:|
| 0 | 0.001242 | 0.053293 | 0.052959 |
| 1 | 0.001127 | 0.047074 | 0.046630 |
| 2 | 0.001186 | 0.049837 | 0.048985 |

Full results, root-only comparison, runtime fingerprints and command:
[synthetic.json](synthetic.json).

## Parsed English and conceived remainder

```sh
PYTHONPATH=bin:test OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python doc/benchmarks/2026-09-21-item2/text_probe.py
```

[text_probe.py](text_probe.py) first repeats item 1's existing grammar-owned
wording training (seed 931; 1,000 generate and 8,000 compose updates). All 56
held-out wordings parse correctly, their generated forms match, and generation
recomposes to the same meaning. It then freezes these encodings. This adds no
interpreter, language model or inference-time word-to-operator table.

The continuation corpus pairs actual forward-parsed paraphrases sharing a
meaning: 1,400 training pairs and 40 held-out pairs. Singletons have no pair.
Each of the three predictors receives 1,000 updates in the ordered, shuffled
and context-free conditions, with identical architecture and update count.
This is a constructed paraphrase-continuation task, not natural discourse.

| Seed | Ordered MSE | Shuffled MSE | Context-free MSE | Related `‖c‖` | Unrelated `‖c‖` |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000848 | 0.013529 | 0.012174 | 1.321959 | 1.780535 |
| 1 | 0.000781 | 0.013870 | 0.011901 | 1.317981 | 1.779291 |
| 2 | 0.000820 | 0.014140 | 0.012285 | 1.316575 | 1.779728 |

Related continuations change one noun; unrelated ones change both. All 80
additional strings are forward-parsed successfully. The related remainder is
smaller in **40/40 comparisons in each seed**, with the same prediction and
gain. This is the semantic portion of spec test 32.

The same ordinary chooser then receives 120 unlabelled presentations at each
gain, with predictor and encoder frozen to isolate residual action credit.
There are 119 credited episodes per condition, each bounded by 64 actual work
units, with its own residual EMA baseline. Across-episode return variance is
0.0528–0.0597. No supplied answer trains that policy. These are policy updates
in a controlled study; native joint training is checked separately.

On the same 40 held-out parsed statements posed as questions, both gains use
**two thought steps and 17 work units per question**, in all three seeds.
Answers are identical. The Brier score against the presented positive assertion
is **1.0 at both gains**: this probe does not obtain successful positive answers
from the relation reader. Thus the reasoning comparison is **null**, not an
accuracy-preserving speedup. It does not establish useful anticipatory queries,
a joint learned encoding advantage, or matched-compute causal utility.
[text.json](text.json) retains every trial, return, control and source digest.

## Native document training

[native.xml](native.xml) is a reduced-width BasicModel configuration using the
cached FineWeb shard, complete sentences, tied reconstruction, the rotation-owned
concept dictionary, structured expectation and residual query credit. The
whitespace sentence limit is eight; the lexical bucket is 16 because punctuation
can produce more lexical words. No sentence is clipped to fit. The initial W8
attempt rejected an 11-word lexical sentence before training and is not a result.

```sh
PYTHONPATH=bin:test OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python bin/bench_sentence_expectation.py fineweb --config doc/benchmarks/2026-09-21-item2/native.xml --docs 64 --batch-size 2 --train-steps 7 --eval-steps 8 --updates 300 --threads 1 --out doc/benchmarks/2026-09-21-item2/native.json
```

The report records actual completed observations, predicted targets, document
boundaries, reconstruction, expectation and backward/optimizer time. Optional
learning measurements and the full default regression receipt are separate.

The initial native run completed 7 optimizer steps on 14 sentences (12 predicted
pairs). After two warm-up batches, it processed 10 sentences/targets in 216.624 s
(0.04616/s); setup/loop compilation remains included in this CPU eager-backend
operating point. Peak process RSS was 2.30 GiB. Validation comprised 16 sentences
and 12 predictions before and after; the document-start fraction was 0.25.
Mean reconstruction byte cost rose from 1.44983 to 1.55145 (+7.01%). Held-out
feature MSE fell from 0.046531 to 0.029550, but context-free MSE was better
(0.028346), and the fixed-pretraining-encoding score was 0.045365. Target NP1
variance fell from 0.086355 to 0.055543. This is not evidence of a better joint
representation; raw prediction error alone would overstate the result.
These short native runs seal NP1-only observations; VP/NP2 target variance is
zero. They do not establish learned three-role semantics. Distinct-role
prediction is exercised by the separate controlled representation studies.

`no_queries.xml` keeps prediction training and disables residual policy;
`reconstruction_only.xml` also sets the prediction-loss weight to zero.
`packed.xml` packs whole sentences into W32 so prediction can train preceding
source encodings within the same optimizer step. These follow-up controls run
alongside the full regression suite; their timing is recorded for audit, not
used as a clean throughput comparison. The original unpacked run cannot by
itself establish within-step expectation-to-encoder learning.

The two unpacked controls completed with the same seed, input and update count:

| Training condition | Reconstruction before → after | Held-out MSE | MSE on fixed original encodings | Context-free MSE |
|---|---:|---:|---:|---:|
| Prediction and optional queries | 1.449833 → 1.551450 | 0.029550 | 0.045365 | 0.028346 |
| Prediction, no queries | 1.449833 → 1.653469 | 0.023534 | 0.045466 | 0.022597 |
| Reconstruction only | 1.449833 → 1.653469 | 0.024451 | 0.046531 | 0.023267 |

Reconstruction is identical in the last two rows: unpacked source encodings
cross an optimizer boundary and are detached. A learned predictor improves its
fixed-encoding score slightly; a changing encoder also reduces raw residual
without any predictor update. Optional queries alter the training trajectory,
but these runs establish neither useful query selection nor joint representation
benefit. All three lose to their context-free control. The comparison does not
attribute the reconstruction regression to expectation. See
[no_queries.json](no_queries.json) and [reconstruction_only.json](reconstruction_only.json).

The corrected packed run completed 78 observations and 74 eligible pairs in
seven optimizer steps; the two-warm-up/five-measured split contains 59 measured
sentences and 57 targets. Measured wall time was 248.658 s (0.2373 sentences/s,
0.2292 targets/s), including eager-loop overhead and contention with the suite.
Validation reconstruction rose from 1.449833 to 1.503289; prediction MSE fell
to 0.038134, versus shuffled 0.038790 and context-free 0.036565. The score on
fixed original encodings was 0.045078. The predictor changed (parameter delta
norm 0.315670), and the answer head did not change. See [packed.json](packed.json).

The first packed attempt's failure, rather than a learning result, is recorded
in [packed-initial-failure.json](packed-initial-failure.json): the new mask lookup
used a one-record serial read limit. The regression covers later owned records
and exhaustion. Packed query-disabled and reconstruction-only controls use the
same command above with `packed_no_queries` or `packed_reconstruction_only` in
both config and output paths.

Both packed controls also completed 78 observations, 74 eligible pairs and
seven optimizer steps. Their final held-out results are:

| Packed training condition | Reconstruction after | Held-out MSE | MSE on fixed original encodings | Context-free MSE |
|---|---:|---:|---:|---:|
| Prediction and optional queries | 1.503289 | 0.038134 | 0.045078 | 0.036565 |
| Prediction, no queries | 1.484372 | 0.036267 | 0.045198 | 0.034445 |
| Reconstruction only | 1.481877 | 0.034323 | 0.046531 | 0.032288 |

The query-disabled predictor changes (parameter delta norm 0.310015); the
reconstruction-only predictor does not. All answer-head deltas are zero. Unlike
the unpacked controls, the two query-disabled packed runs produce different
reconstruction trajectories, as prediction can train the current step's prior
encodings. This short run still shows **no joint quality advantage**: prediction
does not beat context-free scoring and reconstruction is slightly worse with
prediction than without it. Optional querying does not improve these results.
See [packed_no_queries.json](packed_no_queries.json) and
[packed_reconstruction_only.json](packed_reconstruction_only.json).

These native probes use `reconstructionBasisLimit=16`; they are tied-backward
measurements with a bounded reconstruction basis. All 16 validation rows report
basis truncation before training; 15 of 16 do after each packed run (11 after
the original unpacked run). Input sentences are complete, but the reconstruction
metric does **not** establish full-basis fidelity or discrimination. The seven
steps and one native seed do not establish convergence or a multi-seed causal
advantage. The learning gates remain open under item 2.

Each benchmark retains its own source fingerprints. Synthetic, text and the
initial native runs preceded the final generation and history-read corrections;
the packed run preceded the optional contrastive-replay correction (its
contrastive weight is zero). The default-suite receipt separately validates
the complete final source, including the subsequent controller budget-drain
correction. These are not interchangeable receipts.
