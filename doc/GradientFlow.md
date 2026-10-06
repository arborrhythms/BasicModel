# Gradient flow across the architecture

The current contract is item 6.9 [plan §§21–23](plans/2026-09-29-item-6-9-xor-grammar.md#23-review-of-the-22-round-claude-2026-10-03),
building on [§15](plans/2026-09-29-item-6-9-xor-grammar.md#15-one-writer-for-each-weight-alec-2026-10-02),
October 3, with sampled exploration, the decoder margin audit, affine numeric
reading and support-governed decoder eligibility under the uncommitted
[6.8 §§10–14 review and §14 addendum](plans/2026-09-27-item-6-8-one-attention.md).
Each optimizer parameter has one objective owner. The supplied
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

Input narrowing, compose, ordinary thought, anticipation and generation each
compare greedy with one departure. The departure round is sampled uniformly
from the greedy walk's rounds with a legal alternative. At that round, greedy's
action is excluded and the departure action is sampled from the remaining
eligible policy probabilities at unit softmax scale; it is not the next-ranked
action. The prefix is replayed and the suffix is greedy. The original policy
probabilities supply training credit, without the exclusion mask. Both walks
are costed at the same parameters; only a strictly lower owner cost keeps
exploration, and ties keep greedy. Anticipation holds both forecasts until
the arriving observation can cost them. Its controller keeps its existing
ownership. No learning rate, optimizer or budget changes accompany sampling.

| Parameter owner | Weights and permitted update |
|---|---|
| Reconstruction (R) | Perception (the sole writer of native PS/WS prototypes and 11b evidence), field dictionaries, compose operators and tied inverses, compose chooser, and the shared generate decoder (chooser and parameterized generate faces). Grammar lessons train only their chooser, never an operator. Momentum descent; codes have no EMA refresh or contextual rotation. |
| Expectation (E) | Within-sentence, between-sentence, ARMA and contrastive predictors; the predictive reading-attention module where enabled. All sources, routing evidence and targets are detached. |
| Answer (A) | Numeric head and affine root/end/echoic-bank reader; learned memory retrieval for thought and generation, question conditioners and answer adapters/controllers. The concluded understanding is detached. The generate chooser belongs to reconstruction. |

[ObjectiveOwnership](../bin/ObjectiveOwnership.py) reads the trained named
terms from `Layers.Error` and obtains gradients only for their owner lists.
A differentiable numerical operator can transmit a generation derivative to
the conditioner while its own parameters receive no answer update. Regularizers
are unnormalized and follow the parameters they use. The diagnostic operand
term of a grammar lesson trains no operator. `expectationPolicyWeight` stays 0.
The chooser's negative prediction image remains detached forward evidence.

Under 6.8 §14 and its addendum, two spaces share an index: a symbol's form
lives in perceptual space and its concept's meaning in conceptual space.
The index pairs the symbol with its concept. The implementation stores the
pair as `[form | meaning]` in a refreshed buffer, without a free word row or
a learned percept-to-concept map. Letters used to derive a form do not gain
conceptual rows merely by being percepts. XOR_grammar and MM_grammar restore
their concept capacities to 6 and 8; perception retains its native inventory.

Perception's codes have one writer: perception's reconstruction. Native PS/WS
prototypes and 11b evidence are detached when the sentence path derives a
symbol's form. Pair search and byte scoring also detach their candidate banks;
the recovered leaf and root stay live so sentence reconstruction trains the
choosers through the root. The affine answer has its own owner and cannot
write these sources.

For net evidence `d = relu(e_for - e_against)`, the content bounds are
`L = max_parts(d * part_code)` and
`U = min_property_wholes(1 - d * (1 - whole_code))` (default `U = 1`).
The form is `(sum(d_parts)*L + sum(d_wholes)*U) / sum(d_parts,d_wholes)`;
without property wholes it is `L`. The both corner belongs to attention and
never enters this derivation. Repeated addresses in a part group do not
multiply its evidence. The serial leaf remains `[form | meaning] × signed activation`.
Field-path dictionaries retain their existing ownership; `XOR_exact`'s
answer coefficients are unchanged.

After the existing owner step and [0,1] projection, the deterministic room
pass visits concept rows and coordinates in order. For `v = relu(L-U+m)`,
the maximal part moves down by `v/2` and the minimal property whole up by
`v/2`, clamped to [0,1]. Missing towers retain their fixed lattice boundary.
`ConceptualSpace.latticeMargin` defaults to zero. This is a projection, not
a new objective. Fractional evidence, clipping and floating-point arithmetic
can leave violations; the audit reports the exact positive count and largest
violation before and after the pass, without a tolerance hiding residuals.

The first block stores the symbol's perceptual position. Its content coordinates
carry the interval midpoint; its reserved location/time positions are zero for
a generally characterized type, not an occurrence stamp. The complement stores
the concept's conceptual position: at order zero, only the detached,
recency-weighted context mean of existing occurrence roots' meaning coordinates.
References and inverted
leaf postings supply occurrence membership; DEF rows are excluded. These
occurrence rows are context, not the property wholes that bound `U`. `1/(1+age)` supplies
recency, independently of the sentence's truth poles. Context is snapshotted
before a forward and shared by its greedy and explore walks. It never dilutes
the perceptual block and never carries a durable-row gradient.

**Explicitly deferred by Alec, October 4:** property/situation bootstrap
learning and its co-activation objective belong to the operators update.
There is no new context optimizer or loss. A zero complement cannot bootstrap
itself from zero occurrence roots. XOR_grammar and MM_grammar currently have
14-dimensional PS events and paired representations: six native form content
coordinates, eight unused occurrence positions, and an empty meaning complement.
The nonempty-complement mechanism check verifies isolated, detached reads;
these toy gates do not test learned distributional similarity.

The XOR table therefore measures perception's composition of forms (the binding
kernel this round), its inverse, the affine read at unit norm, and one owner.
The concepts in these gate configurations are empty: identical contexts and
no bootstrap. Same-context concepts coinciding in conceptual space is correct;
it is not a collapse to repair. Connectives over meanings are measured where
meanings exist, as in MM_xor's field path. Form and root geometry remain useful
measurements of distinguishability for reconstruction.

Identity comes from below through the fold of forms and meaning from above
through contexts at every order; at orders ≥ 1, composition of meanings also
joins from below. The form fold at all orders, Kleene meet/join on meanings,
bootstrap from conceptual wholes' locations, the `not` items, and expectation's
negative image restricted to the concept face (item 2) are carried to the
operators update. The present kernel still composes the paired vector.
The dictionary has no EMA refresh or contextual rotation. Ownership audits
distinguish absent gradients from zero ones and retain inactive weights.

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
edges and bumps the seen rows. Existing occurrence rows now also conduct:
a primed word sends energy to its rows, and those rows return it to their
constituent words. Both directions use the same native-reference/postings
incidence and existing `primingSpread`, without new rows or cross-batch flow.
The audit records activated competitors. Serial reading also diffuses; there is no
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

The learned `PrimedSymbolReader` is excluded from `_forward_head` for every
configuration, including `XOR_grammar.xml` and `MM_xor.xml`. Its scorer and
consume gate remain available to `reverseOutput` and to the row-owned STM/LTM
reads of selected thought. Generation consumes an owned operand without changing
the reconstruction carrier. Thought preserves its metadata and charges its
existing shared work meter. Keys remain detached. Neither numeric gate can
learn a nonlinear bag-of-words answer through this reader. The §10 class counts
must be read with their failed sum control; they do not establish composition.
The shipped configurations declaring `answerSynthesis=true` are `BasicModel`,
`BasicModel_answers_tied_benchmark`, `MM_add`, `MM_add_verb`, `MM_math`,
`MM_ladder`, `MM_ladder_idiom` and `MM_grammar_wording` (all under `data/`).
Their generated answers use this retrieval path; explicit `reverseOutput`
calls can use it in other configurations. `MM_global` and `MM_qa` keep their
numeric heads affine too; their retrieval probes now exercise generation.

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

Under §11.6, STOP versus a supported binary undo is eligibility. For each top,
the decoder compares the selected pair's existing squared recomposition residual
with the best one-code least-squares explanation (the signed activation used by
`readback_scores`). Both children must be nonzero valid shortlist codes, neither
may repeat the parent, and the pair must explain it at least as well as the
single code. Equal fits prefer the two readable parts, including when the bank
also holds a code for the larger chunk. If such a pair exists, STOP and unary
rewrites are masked; the chooser still selects between eligible binary operations.
Otherwise a unique nearest one-code explanation is singular and emits. A tie
between identities is not singular. There is no new residual cutoff. Missing
support or insufficient stack capacity leaves work pending rather than making
STOP eligible. These residual and progress tests are the implementation of the
review's readability rule and remain subject to Claude's review.

This mask replaces the policy-credit proposal in §11.3. No advantage loss,
learning-rate change or extra training step is added. The audit keeps raw
STOP-minus-undo logits, their actual backward gradients and fixed-parent update
deltas, and records the eligibility mask. A positive raw STOP margin no longer
implies STOP was eligible; a masked STOP logit receives zero choice gradient.
The gate spells the leaves of this same walk with the same scoring as output.

The decoder's hard choice determines stack topology. As of the operators
update round 1, its numerical transition has no straight-through policy path.
Reconstruction teaches undo, unary and STOP by cross-entropy on detached
states from the compose derivation. The free byte walk and the answer's
restricted backward cannot train the policy; selected numerical inverses
retain their ordinary parameter/operand gradients.
Compose-only grammars implicitly expose their operators' generate faces, so
numeric-answer configurations also learn a decoder without a configuration edit.

During reconstruction training the decoder costs both complete walks before
either can train, and selects the graph with strictly lower free-byte
relative error. At evaluation it makes one greedy walk. This decoder
selection is inside each sentence trial; the two outer compose trials still
take their existing steps after their own comparison.

`reconstruction.free_bytes` scores emitted leaves against input bytes, including
NUL. Target positions only align scoring after decoding; they never select an
operation or determine when the walk stops. Missing words and excess emitted
leaves are scored. Packed sentences retain their own denominator.
Pair search detaches candidate codes in both the residual and its soft blend.
The parent stays live, retaining the compose chooser's gradient through the
root. Residuals are divided by the parent's mean square before the unchanged
`.01` soft temperature; that quantity is dimensionless. An exactly zero parent
uses the existing zero-target squared-penalty convention (divisor one), with
no floor on positive parent scales. Byte scoring detaches the bank codes and
keeps the recovered leaf live. For recovered leaf `x` and candidate `c`,
serial derived-code read-back scores `abs(cosine(x_PS,c_PS)) × priming(c)`.
It reads only native perceptual content; magnitude and occurrence context do
not determine lexical identity. Absolute cosine preserves the identity of a
negative-activation leaf without treating that sign as another spelling.
A zero code scores zero. The generic field-code utility retains its prior
signed-activation times cosine behavior. At the unchanged temperature,
the target byte's log probability is `logsumexp(emitting candidate logits,
-log(256)) - logsumexp(all candidate logits, 0)`. The second entry in each
sum represents the null candidate. Non-emitting candidates contribute only
to the normalizer. There is no probability clamp, so a wrong sharp winner
does not erase the true byte's gradient. CPU/CUDA byte scoring accumulates
float32 inputs in float64 through the log sums, then returns their original
dtype; MPS uses its supported input precision. This avoids amplified
reduction-rounding drift between eager and Inductor execution without
changing the temperature, normalization or tolerances. Candidate codes receive
no sentence reconstruction gradient; the recovered leaf carries its gradient
to the compose chooser. The gate
uses the same free inverse and scoring. Neither score normalizes a code
parameter or imposes a post-step constraint.

The antipode objective, its reporting key, its diagnostic helper and its two
obsolete tests are removed in §14. Byte error is the complete sentence
reconstruction objective. No repulsion objective replaces it.

XOR_grammar's class reader reads the concluded root at unit norm through a
fixed, parameterless transform. Both affine reader paths use that same
normalization, and output alone owns their weights. The sum control retains
its existing unnormalized affine reader: normalizing an additive root would
break that control's affine zero-contrast property. Reader weight norms are
saved per epoch, including each contributing weight tensor.

Conjunction binds `x` and `y` as `norm(x) * norm(y) * unit(x*y)`; a
repeated native reference returns that reference. Equal vectors at distinct
addresses remain distinct references. Disjunction is `(norm(x)+norm(y)-norm(x)*norm(y))*unit(x+y-x*y)`;
`sum` is the mean control, and `not` negates. The catalogue retains `min` and `max` under their own names and
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

Under §12, every compose and decoder derivation in the audit names its rules
beside their IDs, including positions and arities for compose. Names come from
the model's held catalogue, not a later global grammar configuration. Each XOR
run's summary captures its final greedy compose derivations, one per sentence,
at the existing evaluation boundary before the temporary trace is discarded.
There is no additional forward or training run. The decoder's first-logit
audit also gives the binary rule names beside their indices and rule IDs.

The binding kernels take magnitude from the operand activations, not the
lengths of their form codes (§20). With `u=unit(x)`, `v=unit(y)` and activation
magnitudes `a,b`, conjunction returns `a*b*unit(u*v)` and disjunction returns
`(a+b-a*b)*unit(u+v-u*v)`. A present native word has activation one regardless
of its form norm; a composed root carries activation in its norm. The free
inverse searches through the same kernels. The mean remains `sum`, the additive
control; both gates retain the raw-root affine reader. These changes introduce no parameters,
loss, learning rate or optimizer change. The XOR table is the composition
mechanism gate under §12.1; its class and reconstruction bars remain unchanged.

The eager training audit additionally records the first decoder step's raw
logits and `STOP − undo` margin for each binary action, on both decoder paths
within each compose trial. Hooks record the gradient reaching those logits
only during the actual owned backward; diagnostic gradient queries do not
count as training. Each optimizer step records those gradients, their
`dL/dSTOP − dL/dundo` difference, and the margin before and after the update
with the same detached parent held fixed. This distinguishes policy motion
from changes to its input representation. Steps that do not reach decoder
logits are recorded with no walks. No RNG draw, extra backward or optimizer
update is added. Raw gradients and actual margin changes must both be read:
shared policy parameters and momentum can move a margin differently from
independent descent on its two logits. The [review receipt](benchmarks/2026-10-03-operators-attention/README.md)
also records the decoder's kept-path stability on the same final XOR training.

The saved [§11 receipt](benchmarks/2026-10-03-operators-attention/README-before-review12.md) measures
class **2/10**, reconstruction **6/10**, joint **2/10**, MM_xor **10/10** and
sum control **10/10**, once on the frozen model source. The class count is
below §22's 9/10 and remains for review. On the tenth run, ownership conflicts
are zero (23 active, 64 inactive parameters over 1,200 backwards). Every
audited first-step row/path masks STOP as compound. Its raw margin remains
near 2, but its gradient is zero; undo-gradient differences are at most
1.96e-9 and all fixed-parent margin changes are exactly zero. Changing raw
epoch means reflect changing parents, not an observed policy update.
Decoder kept-path stability is **.339487**, with zero explore wins and zero
strict-rule violations. The audit supplies these observations without assigning
a cause to the class failures. The candidate is uncommitted for Claude's review.

The [§12 composition mechanism receipt](benchmarks/2026-10-03-operators-attention/README.md)
measures class **0/10**, reconstruction **7/10**,
joint **0/10**, MM_xor **10/10**, and sum **10/10**.
All ten sum controls were read before the gates started. Comparison is the
saved §11 round (2/10, 6/10, 2/10, 10/10, 10/10); below-comparison counts:
**class_pass 0/10 versus 2/10, joint 0/10 versus 2/10**. The final run records **0 ownership conflicts**
and decoder kept-path stability **0.009387**. Its first-step eligibility
counts are `{"compound": 6400}`, and the largest absolute
fixed-parent margin change is **0.000302553177**. Full named paths, margins and
gradients remain in the receipt. No attribution training or retry was added.
§11 remains as measured: only its tenth run saved final compose operators,
all mean disjunction; the other runs' operators cannot be recovered from their
scores. The five-run mean hypothesis is unconfirmed. This candidate remains
uncommitted for Claude's review.


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
| `reconstruction.free_bytes` | Shared generate walk spelling vs observed bytes including NUL; `log 256` | `reconstructionScale` | Trial; R, including decoder and live root; candidate bank codes detached. Detached chosen report at batch end |
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
owners or multiple objective writers raise an error. Historical receipts
aggregate these reports on XOR_grammar and the production native benchmark,
alongside the stage-1 term and gradient measurements. The §14 audit reuses
the tenth XOR training, including the decoder's STOP-minus-undo margins,
actual logit gradients, fixed-parent margin changes and kept-path stability.
Every XOR run saves code and root pairwise cosines and centered singular
values at start/end, each word's exact perceptual support and minimum absolute
coordinate, and per-word code-versus-priming read-back decisions. Room reports
record before/after violations at the first and final projection. The reader's
weight norm is retained for every epoch. A separate sentence-path audit follows
the complete saved perception pullback and distinguishes absent, zero and
nonzero gradients at native prototypes and 11b evidence. Its requirement is
zero sentence-path gradient; owner labels alone do not prove that cut.
These observations add no forward, RNG draw or optimizer step to the gates.
Historical receipts also retain their final VQ cluster sizes. For the native reserve, exact all-pair moments avoid allocating a
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
records the historical deferred items. The sphere/norm-as-certainty proposal
is retired by §14; bootstrap co-activation learning and the operator split
remain deferred, along with the answer reach.

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

## Codes are perception's; the chooser is trained from the trial comparison (October 4)

Decided in the 6.8 §14–§15 rounds
([plan §14.3](plans/2026-09-27-item-6-8-one-attention.md#143-the-fix-the-part-codes-are-perceptions-alec-2026-10-04),
[§15](plans/2026-09-27-item-6-8-one-attention.md#15-review-of-the-14-measurement-claude-2026-10-04)).

- **Perception's codes have one writer, perception's reconstruction.** In the
  conceptual derivation the percept prototypes and the 11b evidence are
  detached; the sentence path's pair search, byte scorer and answer head
  reach no gradient to them (audited: zero gradient and zero optimizer
  displacement at the prototypes in the §14 measurement). The §13 collapse
  was the straight-through pair-search blend pulling every candidate code
  onto every read-back target with uniform weights; the §14 erosion was the
  shared wholes' term of the midpoint, moved by the room clamp. Both paths
  are closed: no code moves under the sentence path.
- **The sentence handoff detaches input attention's credit (§20).** The
  attention and compose walks share a scorer. Byte reconstruction previously
  reached it through the attention straight-through value, cached perception
  and the perception pullback even when the two compose costs tied. Detaching
  that credit closes this additional sentence-path writer; the original
  no-movement tie assertion passes. Round 1 adds attention's own paired-cost
  score-function objective; the sentence handoff remains detached.
- **The compose chooser's gradient is the score-function estimator (Alec,
  2026-10-05: "So we are doing SCG?"; supersedes the comparison step of
  2026-10-04 and the straight-through of the same morning).** The chooser's
  parameters enter the trial only through which operation is chosen, so the
  exact gradient of the expected reconstruction cost with respect to them is
  the score-function term at the sampled departure, with the greedy trial's
  cost as the paired baseline (self-critical sequence training): surrogate
  `K · R · p_θ(a_dep | state) · (C_explore − C_greedy)`, costs detached, one term per
  sentence with a departure, reconstruction-owned; `∇p` rather than `∇log p`
  because the departure is drawn uniformly over the value-distinct eligible
  alternatives (the coverage floor) and the importance weight cancels the
  `1/p`. Here K counts the value-distinct eligible alternatives at the sampled
  round and R counts eligible rounds in that sentence. Their product cancels
  the uniform proposal probability `1/(K·R)`, giving the sum of the
  baseline-subtracted gradients over those alternatives and rounds, before
  the existing active-row mean reduction. No differentiable decoder is needed;
  a tie teaches nothing and the greedy argmax supplies no chooser gradient.
  (6.8 plan §§16.3, 17.)
- **The pair search is hard.** Candidate codes detached, residual relative to
  the parent's mean square, the straight-through blend deleted: with codes
  perception's and operators parameterless, nothing on the compose side
  needs a gradient through the inverse, and the blend's gradient was a biased
  proxy (the true pair's margin) weighted by the decoder's uncertainty. The
  byte scorer's bank codes are detached. Selected numerical inverses retain
  operand gradients, but the free walk supplies no policy gradient.
- **The decomposition chooser learns the true pair (6.8 §§16.4, 17).** For
  the selected undo, a separate reconstruction-owned scorer reads each
  shortlisted pair's negative relative residual, both candidates' activation
  and both candidates' priming. Its softmax has a hard argmax value; fit weight
  1 and context weights 0 reproduce the previous residual argmin without a
  random initialization. Teacher-forced cross-entropy targets the input's
  resolved word identities at that composition's operand positions. A target
  absent from the shortlist is counted and contributes no term. Candidate
  codes and all scorer features are detached. The teacher term is added in
  reconstruction's owner step after both trials' byte costs and the keep
  decision; targets never enter free decoding. These parameters are separate
  from the forward chooser. Round 1 extends supervision to the walk policy
  and retains the §11.6 mask at free inference.
- **One affine reader for both gates.** XOR_grammar's class reader and the
  sum control read the raw root through the same output-owned affine head;
  the root's address bands are zero. A normalization of the root is a
  nonlinearity the control cannot hold and is not the affine read (the
  unit-norm reader of the §14 round is withdrawn).
- **Certainty is the activation.** Forms are at full presence; the leaf's
  activation, the projection coefficient, carries how sure the reader is.

## Attention's credit detached; the kernels' magnitude from the activation (October 5)

- **The input-attention walk shares the compose chooser's parameters** and its
  straight-through credit reached them through the perception pullback (the
  byte cost's cotangent at the forked leaves pushed back through the cached
  perception graph), a second, pathwise writer beside the score-function term
  that was below float32 resolution at the old code scale. Attention's credit
  remains detached at the sentence handoff. Round 1 trains the attention
  choices by the score-function estimator below. Reconstruction remains
  the shared chooser's sole gradient owner.
- **The binding kernels take magnitude from the activation**, codes entering
  as directions: conjunction `a·b·unit(u∘v)`, disjunction
  `(a + b − ab)·unit(u + v − u∘v)` with `u, v` unit codes and `a, b` the operand
  activations (1 for a present word). Roots carry unit certainty; identity is
  in the direction. The code's norm is no longer read as certainty (the
  reading retired 2026-10-04); at full presence the old `a + b − ab` on norms
  above 1 was non-monotone and the product leaked form length.
- **Full-presence admitted rows:** `RadixLayer.insert()` initializes an
  admitted percept row at full presence (6.8 plan §19–§22; unit L2 norm then
  the cube clamp from the last round on). The decoder's policy now learns
  from the compose teacher even where its free walk has one legal action.


## Operators update round 1 (October 5)

**Attention uses the same estimator as compose.** The input walk samples one
eligible round uniformly and one non-greedy legal action uniformly. With K
alternatives and R eligible rounds, its surrogate is
`K·R·p(a_departure)·detach(C_explore−C_greedy)`. Both native-percept byte costs
are measured before any owner step, with the greedy cost as baseline. Features,
keys and priming are detached; hard attention values have no straight-through
credit. The term is consumed once at reconstruction's first sentence owner
step (or the batch owner step for a field-only path). Exact ties attach no
gradient or momentum-only update. Compose retains
its own sentence comparison; both terms have the same reconstruction owner.
The tenth-run audit records costs, actions, K and R, advantages, probabilities
before/after the actual update, analytic gradients and finite differences.

**Exact pair recomposition precedes learned context.** The bounded shortlist
and candidate limits are unchanged. A squared relative residual at most
`(8·finfo(dtype).eps)²` is float-roundoff exact; if one exists, the residual
argmin wins independently of all context weights. Equal residuals retain bank
order. The pair scorer's CE still sees every eligible pair. Each operand's
projection coefficients are centered and divided by their population standard
deviation over its valid shortlist; constant features become zero and invalid
slots contribute neither mean nor variance. Codes and these features are
detached. Missing targets are still counted without a CE term.

**The walk policy is teacher-forced.** Each actual compose parent teaches its
declared undo or unary generate face; each input leaf teaches STOP. The states
are detached saved values. Cross-entropy averages nodes within a sentence and
then active rows; it is added as `reconstruction.walk_policy` after the paired
byte-cost comparison, alongside `reconstruction.decomposition`. No imitation
loss enters the forward chooser or trial selection. At inference the policy
chooses freely under the existing eligibility mask, with no compose journal.
Its straight-through transition blend is removed. The receipt audits the new
CE logits and confirms that the free decoder logits receive no gradients.

In the round's tenth measured run, all 1,600 attention advantages and all
1,600 compose advantages were zero. Those records confirm tied-cost behavior,
not a nonzero score-function learning signal; the positive/negative-advantage
analytic and finite-difference checks are separate focused tests. The 3,200
walk-teacher records match CE gradients within 7.45e−9 and show actual policy
logit updates. Code displacement, sentence-path perception gradients and
ownership conflicts remain zero. The round nevertheless fails its standing
gate because MM_xor is 9/10 rather than 10/10; the receipt records the miss.

[Round-1 source, tests, measurements and review receipt](benchmarks/2026-10-05-operators-update/README.md).
