# Gradient flow across the architecture

The current contract is item 6.9 [plan §§21–23](plans/2026-09-29-item-6-9-xor-grammar.md#23-review-of-the-22-round-claude-2026-10-03),
building on [§15](plans/2026-09-29-item-6-9-xor-grammar.md#15-one-writer-for-each-weight-alec-2026-10-02),
October 3. Each optimizer parameter has one objective owner. The supplied
answer stops at the understanding, and expectation detaches both its evidence
and targets. Step 5a's uncut answer and the supplied-answer gradient projection
are retired. Forward representations and numerical operator hosts remain shared.

<a id="the-training-step-october-1"></a>

## The training step (October 3)

A sentence has a greedy and an explore derivation. Both are composed and
costed at identical parameters before either trains. Each trial then takes
its own restricted backward and optimizer step, including its own perception
pullback for reconstruction. Saved forward parameter values preserve the
second graph. A batch-end step runs only when its remaining registered total
has a derivative. No extra step is added for diagnostics.

Explore is kept only when its reconstruction is **strictly lower**. A tie
keeps greedy. The answer and expectation never enter the comparison.
Reconstruction and expectation train on both trials; the reader trains only
the kept rows of each trial. A trial with no kept rows gives the reader no
optimizer step, including no momentum-only step. The kept state is committed detached; no gradient crosses that
boundary into the next sentence.

| Parameter owner | Weights and permitted update |
|---|---|
| Reconstruction (R) | Perception, unconstrained object codes, compose operators and tied inverses, compose chooser, and the shared generate decoder (chooser and parameterized generate faces). Grammar lessons train only their chooser, never an operator. Momentum descent; codes have no EMA refresh or contextual rotation. |
| Expectation (E) | Within-sentence, between-sentence, ARMA and contrastive predictors; the predictive reading-attention module where enabled. All sources, routing evidence and targets are detached. |
| Answer (A) | Numeric head and affine root/end/echoic-bank reader; for answer synthesis, question conditioners and answer adapters/controllers. The concluded understanding is detached. The generate chooser belongs to reconstruction. |

[ObjectiveOwnership](../bin/ObjectiveOwnership.py) reads the trained named
terms from `Layers.Error` and obtains gradients only for their owner lists.
A differentiable numerical operator can transmit a generation derivative to
the conditioner while its own parameters receive no answer update. Regularizers
are unnormalized and follow the parameters they use. The diagnostic operand
term of a grammar lesson trains no operator. `expectationPolicyWeight` stays 0.
The chooser's negative prediction image remains detached forward evidence.

The concept dictionary is a trainable parameter everywhere, owned only by
reconstruction, without a norm constraint. `conceptualContextLearningRate`
and `conceptualSimilarityScale` are zero in the canonical configurations.
The concept VQ EMA refresh, contextual rotation, feature-derived row refresh,
promotion/phrase code replacement and post-step unit normalization are off or
retired. Admission assigns identities and relations without replacing codes;
lookup preserves their learned magnitudes. Initialization and checkpoint
restoration are not training updates. The leaf remains `code × signed activation`.
`XOR_exact`'s evidence coefficients belong to the answer, their only reader.
Ownership audits distinguish absent gradients from zero ones; not every
parameter is active in every batch.

## One understanding record, two consumers

[SentenceUnderstanding](../bin/SentenceUnderstanding.py) is built once per
trial. The tied reconstruction and the supplied answer receive that same
object. It carries the concluded root, three end slots and depth, packed
sentence roots/depths, per-word target values/rows/validity, sentence id,
and a primed-symbol snapshot. No rule IDs, arities, operand positions or
journal columns enter the record or the answer reader. It carries no
witness offsets or numerical operation frames. Temporary clause-closing
frames preserve the performed relation semantics and are discarded at the
closing; neither reconstruction nor the answer reads them. Addresses select evidence;
they are not semantic magnitudes.

The snapshot contains each row's own sentence symbols plus up to
`reconstructionBasisLimit` activated others from `Space.prime_seen`. It is
taken after the sentence's seen write and shared by both trials. The `[B,V]`
surface decays, diffuses with `primingSpread` over on-device sparse definition
edges and bumps the seen rows. Serial reading also diffuses; there is no
4,096-edge cap and no cross-row priming. A row at neutral sends no energy.
Symbols with surfaces are byte candidates; all valid rows enter answer context
with their priming weights.

The numeric head reads detached end slots plus an answer-owned, zero-initialized
linear reader of the root (D), three masked end slots (3D), and the echoic
priming-weighted code sum (D). Its extra width is 5D. There are no rule
histograms or original per-word features. The linear reader preserves additive
evidence for the sum-only control. Generated answers detach the same conceptual
state and bank before the question conditioner; no extra record reader rewrites
that answer. The focused test checks that both consumers receive the same object.

## State paths and shared parameters

`resolveAnswer()` prepares an owned `AnswerDerivation`; `reverseOutput()`
consumes it without repeating resolution. Both numeric and generated answers
cut the concluded state, all contextual operands and the record features.
Generation keeps the declared numerical hosts and their tied interfaces,
without a second optimizer owner or checkpoint copy. Grammar lessons cannot
move those hosts. Catalogue migration still follows rule meaning, preserving
unchanged chooser weights and optimizer moments.

Reconstruction and output share one **generate decoder**. Starting from the
root and occupied end slots, its generate chooser infers a binary undo, unary
undo or STOP from the current top. No recorded compose sequence or numerical
witness is supplied. Binary undo searches both children over the echoic
shortlist; unary undo calls the operator's generate face. The shortlist keeps
the sentence's own words and primed others, snapshotted after the seen write;
§24.4's proposed pre-write snapshot is withdrawn. Equal-residual pairs retain
bank order. A missing shortlist cannot manufacture a balanced binary split.
The gate spells the leaves of this same walk with the same scoring as output.

The decoder's hard choice determines stack topology. Its numerical transition
uses a straight-through softmax over candidate generate transitions at the
existing unit softmax scale: the forward value is the chosen transition, and
its derivative includes the chooser. It adds no policy reward or extra cost.
Both reconstruction terms train that graph; the answer's restricted backward
can train its conditioner through it but cannot update decoder parameters.
Compose-only grammars implicitly expose their operators' generate faces, so
numeric-answer configurations also learn a decoder without a configuration edit.

`reconstruction.free_bytes` scores emitted leaves against input bytes, including
NUL. Target positions only align scoring after decoding; they never select an
operation or determine when the walk stops. Missing words and excess emitted
leaves are scored. Packed sentences retain their own denominator.
Hard candidate addresses are detached; the selected candidate values
and all scored codes remain live. For recovered leaf `x` and candidate `c`,
read-back scores `a × cosine(x,c) × priming(c)`, where the signed activation
`a = dot(x,c) / dot(c,c)` is recovered without a unit-code assumption. A
zero code has no direction and scores zero. At the unchanged temperature,
the target byte's log probability is `logsumexp(emitting candidate logits,
-log(256)) - logsumexp(all candidate logits, 0)`. The second entry in each
sum represents the null candidate. Non-emitting candidates contribute only
to the normalizer. There is no probability clamp, so a wrong sharp winner
does not erase the true byte's gradient. CPU/CUDA byte scoring accumulates
float32 inputs in float64 through the log sums, then returns their original
dtype; MPS uses its supported input precision. This avoids amplified
reduction-rounding drift between eager and Inductor execution without
changing the temperature, normalization or tolerances. Both the true code
and its competitors receive reconstruction gradients. The gate
uses the same free inverse and scoring. Neither score normalizes a code
parameter or imposes a post-step constraint.

`reconstruction.antipode` aligns the selected shortlist word's code with each
emitted leaf and repels every other valid shortlist code plus the former
rotation's hashed negative rows. For cosine c the selected target contributes
softplus(-10c); each negative contributes softplus(10c), the existing SBOW
negative form. These binary errors are averaged over active comparisons and
leaves, divided by log 2, and weighted by reconstructionScale. Both operands
remain live. The old integer hash and conceptualContextNegatives default (4)
are reused, including the training-step component; no RNG is seeded or drawn.
There is no co-activation attraction, centroid target, norm constraint, EMA
refresh or rotation write. Batch end reports the detached kept-trial term.

Conjunction binds `x` and `y` as `norm(x) * norm(y) * unit(x*y)`; a
repeated native reference returns that reference. Equal vectors at distinct
addresses remain distinct references. Disjunction is `(x+y)/2`, and `not`
negates. The catalogue retains `min` and `max` under their own names and
all three faces, but current grammar files do not select them. Compose,
generate and reverse share each kernel; a free inverse searches through
that same compose kernel. These operations impose no code norm constraint.

Reconstruction uses gradient descent with momentum 0.9 at the configured
learning rate. Its compact row-local form retains a float32 momentum prefix,
updates only observed nonzero-gradient rows (or rows with an active proximal
penalty), and leaves unobserved rows unchanged. It does not divide a step by
gradient magnitude. Readers and expectation predictors keep Adam. The L1
proximal threshold for a reconstruction-owned coefficient is `lr * strength`;
answer-owned adaptive coefficients keep their existing Adam metric. Current
saved Adam checkpoints retain the readers' moments and explicitly discard
adaptive moments for parameters now owned by reconstruction's SGD.
The audit records every anchor/code/decoder-chooser gradient and displacement coordinate,
code geometry, VQ counts, and each sentence's modal-derivation fraction and
number of distinct recorded rule sequences.

The compose chooser retains its straight-through estimator. Target spellings,
priming weights and selection addresses are detached. Durable LTM writes,
finished episodes and retained expectations detach; they cannot reopen an old
graph. Query masks, reference matching and work accounting remain hard
bookkeeping. These mechanisms do not establish useful learned reasoning.

## Trained total and diagnostic objectives

`Layers.Error` owns the trained sum. Its targeted APIs register the error
and a detached uninformed baseline over the same active entries. Repeated
contributions to one named term combine before division: `sum(error) /
sum(baseline)`, a ratio of means, with no epsilon floor or running scale.
For squared errors the baseline is the target's squared norm about the
origin. For categorical errors it is `log K`, and for binary errors `log 2`.
A constant target therefore has a usable baseline. If every active squared
target is exactly zero, the term is an unnormalized penalty toward zero.
An empty active set contributes zero.

Every relative term has weight 1 unless the configuration supplies a
priority, including inherited, explicitly stated `model.xml` priorities.
`reconstructionScale` multiplies reconstruction terms independently.
`Error.total(kind='relative')` and `Error.total(kind='penalty')` report the
two sums separately; `total()` supplies their trained sum. A row registry
normalizes with the shared target sum, then returns row contributions whose
active-row mean is that same training objective. A small-target row does
not get a separate large inverse scale. `breakdown()` records raw error,
baseline, entries, weight, contextual multiplier, kind, ownership and row
selection values. `trained=False` metrics never enter the optimizer total.

All retained serial grammar readings reconstruct through their understanding.
The old reading modes, D3 path and detached student runtime are retired. A
reading without a grammar keeps perceptual reconstruction. A sentence with no
admitted surface candidate has no reconstruction term and is counted.
A zero-weight term does not require its source registry.

The inventory below gives each active term's definition. “Origin” means the
detached target mean square over exactly the error's active entries. R is the
input reading and tied inverse; A is the answer reading map; E is the
expectation predictor. Every A term stops at the understanding, in a trial and at batch end.
Every E term detaches both its source and target. Restricted backward also
prevents generation or lessons from writing an operator.

| Registered term | Comparison and uninformed baseline | Priority | Owner and gradient |
|---|---|---|---|
| `answer.what/where/when` | Numeric output or generated surface band vs supplied target; origin per band | corresponding event-band scale | Trial and batch; A |
| `reconstruction.free_bytes` | Shared generate walk spelling vs observed bytes including NUL; `log 256` | `reconstructionScale` | Trial; R, including decoder, true and competing codes. Detached chosen report at batch end |
| `reconstruction.antipode` | Selected-code alignment and all other shortlist/hashed-code repulsion, logistic cosine errors; `log 2` | `reconstructionScale` | Trial; R, decoded leaf and live codes. Detached chosen report at batch end |
| `reconstruction.what/where/when`, `reconstruction_reverse.what/where/when` | Grammar-free perceptual/masked or reverse event vs input target; origin per band | reconstruction scale × band scale | Batch; perception and the active input inverse |
| `leaf_distill` | Historical root decoder's predicted exact leaves vs retained leaves; origin | `leafDistillWeight` | Batch; decoder and live input root where enabled |
| `expectation.roles` | Predicted complete role payload vs arriving detached meaning; origin | `interLossWeight` | Trial or batch, once; E only; source and target detached |
| `expectation.presence` | Role-presence logits vs occupied-role bits; `log 2` | `interLossWeight` | Trial or batch, once; E |
| `expectation.kind` | Idea/relation logit vs arriving kind; `log 2` | `interLossWeight` | Trial or batch, once; E |
| `expectation.root` | Root predictor vs arriving root where structured roles are inactive; origin | `interLossWeight` | Trial or batch, once; E only |
| `expectation.contrast` | Next-idea candidate CE vs actual next idea; `log(candidate count)` | `interContrastiveWeight` | Trial or batch, once; E only |
| `expectation.intra` | Predict-then-perceive code vs arriving code; origin over active scalar entries | `intraLossWeight` | Trial or batch, once; within-sentence predictor only |
| `expectation.arma` | ARMA forecast vs detached arriving representation; origin | `armaScale` | Batch; ARMA predictor only |
| `grammar.compose.choice` | Joint compose probability vs annotated operator; `log(choice count)` | `grammarLessonWeight` | Trial; compose chooser; unlabelled waiting sites excluded |
| `grammar.generate.choice` | Generate action logits vs annotated action; `log(choice count)` | `grammarLessonWeight` | Trial; R generate chooser only |
| `grammar.generate.operands` | Generated children vs annotated detached codes; origin | `grammarLessonWeight` | Reporting only (`trained=False`); no operator gradient |
| `embedding.positive/negative` | Sampled lexical logits vs positive/negative membership; `log 2` separately | `embeddingScale` | Batch; joint perceptual embedding |
| `conceptual_sbow.positive/negative` | Cosine logits vs context/non-context membership; `log 2` separately | `conceptualSimilarityScale` × existing SBOW strength | Batch; conceptual codes on the enabled parallel path |
| `reading_attention` | Next unconsumed span NLL; `log(legal span count)` | 1 | Batch; expectation-owned reading attention |

Targetless terms retain their strength and are recorded apart from relative
errors. They have no fabricated target or normalization denominator:

| Term | Definition and strength | Owner and gradient |
|---|---|---|
| `definition_sparsity` | Rank-ordered soft-L0; `definitionSparsityScale` | Batch; definition coefficients |
| `gate_l1` | Absolute lift/lower raw gates; `gateL1Lambda` | Batch; operator gates |
| `concept_readout_l1` | Active sparse readout L1; configured `l1Lambda`, averaged over distinct observed concepts | Trial/batch report only; one proximal application per applicable optimizer step, never another autograd penalty |
| `truth.falsity`, `truth.balance` | Stored-truth incompatibility and forbidden-corner penalties; `truthLossWeight` and balance strength | Batch; live operands if present |
| `selected_thought_policy` | Answer/work return advantage × selected-controller log probability; `selectedThoughtPolicyWeight` | Batch; answer-owned selected controller |
| `expectation_policy` | Historical residual/work return advantage; `expectationPolicyWeight` stays 0 | Disabled: expectation must not train a thought controller |

Policy return baselines remain detached variance-reduction baselines; they
are not relative-error divisors. Raw `output`, reconstruction, reverse and
embedding totals, already-trained grammar-lesson summaries, and missing-candidate
counts are reporting-only entries. The old `ws_codebook_recon` and
`ws_semantic_arrangement` stash consumers have no production writer.
Standalone embedding pretraining is separate from these costs. Contextual
sphere rotation is retired; it is not an additional dictionary optimizer.

<a id="per-operator-agreement"></a>

## Ownership audit

`branchDiagnosticsEvery` now reports `[gradient-ownership]` rather than
per-operator opposition. The audit lists every trainable optimizer parameter,
its sole declared owner and the actual writers observed during backward.
Inactive parameters remain listed; they are not counted as reached. Duplicate
owners or multiple objective writers raise an error. The receipt aggregates
these reports across actual training steps on XOR_grammar and the production
native benchmark, alongside the stage-1 term and gradient measurements.
The receipt also saves dictionary pairwise cosines and XOR root singular
values before the first step and after training, plus the final VQ cluster
sizes. For the native reserve, exact all-pair moments avoid allocating a
65,536-square matrix; full matrices are saved for all observed primed rows.
The [§§21–22 audit](benchmarks/2026-10-03-item6-9-free-readback/audits.md)
records zero observed writer conflicts in both configurations. The native
indexed dictionary has no VQ cluster buffer; XOR's six cluster counts stay
at one. Neither run supplies an activated outside-word competitor, so its
zero outrank count does not establish competitive retrieval quality. In the
audited XOR run, all saved code and chooser-anchor displacements are exactly
zero at float32 precision, despite tiny nonzero reconstruction gradients at
some steps. Its answer-reader learning therefore does not establish learning
in those weights. The native run records nonzero code updates and fits its
existing 24 GiB slow-only ceiling; its single epoch does not establish
multi-epoch derivation stability. The
[closing receipt](benchmarks/2026-10-03-item6-9-free-readback/README.md)
retains every gate outcome and remaining failure; writer separation is not
an acceptance claim or a guarantee that the objectives improve together.
The [§20.3 catalog](plans/2026-09-29-item-6-9-xor-grammar.md#203-catalog-set-aside-now-to-return-once-reconstruction-and-xor-hold)
records the deferred sphere, distributional pressure and answer reach.

No gradient projection, opposition streak, norm rebalance or persistent
conflict counter remains in training or the checkpoint. The stateless gradient
math utilities remain available for read-only measurements. Historical per-operator receipts retain their
original interpretation; they are not evidence about the new ownership split.
Separate weights remove direct shared-weight competition, but reconstruction
can still change the forward inputs of the answer or expectation predictors.

## Verification and retired checks

[Ownership tests](../test/test_objective_ownership.py) check restricted writers,
expectation source/routing cuts, perception pullback ownership, a common trial
record and the answer cut. [Priming tests](../test/test_sentence_priming.py)
check row isolation, serial diffusion and the uncapped sparse edge path.
[Selection tests](../test/test_reconstruction_precedence.py) retain reconstruction
precedence; [sentence comparison tests](../test/test_sentence_comparison.py)
check equal-parameter costs before updates. The old uncut-answer, projection
and expectation-encoder training tests are retired with those behaviors; the
receipt preserves their old bodies and the complete bodies of each port.

## Expectation at the closing

The signed image and observed meaning produce detached negative evidence for
the compose chooser. Prediction separately minimizes all-role relative squared
error and presence/kind BCE against detached targets. Empty roles have zero
targets, not zero residuals. Context, arriving ideas and routing inputs are
all detached; only predictor weights train. Pending estimates survive an
update as detached evidence and are recomputed for the current predictor's
step. The configured zero expectation-policy weight preserves the one-owner
rule. Expectation quality and utility remain empirical questions.
