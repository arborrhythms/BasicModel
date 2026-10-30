# Gradient flow across the architecture

> **Read first (2026-10-29): the simple story, then the alternatives under
> exploration.** Written for Alec as one picture of the current contract.
> Everything below this box is the historical record and is kept as it was.

## The simple story, as built today

**One forward, separate gradients.** A sentence runs forward once per
trial. Every reader sees live values: output reads the understanding in
STM, attention reads perception, the decoder reads the completed
understanding. The graph is differentiable end to end. What keeps the
objectives apart is the backward: `ObjectiveOwnership.backward_owned`
calls `torch.autograd.grad` once per objective, each time only with
respect to the parameters that objective owns, and an audit fails if any
parameter receives a second writer (item 6.9 §15, Alec 2026-10-02: "the
gradients separate over the weights"). So an error can be differentiable
through a parameter without ever training it. Per sentence, the greedy
trial's step writes reconstruction, attention and the lessons; the departure
trial's step adds the one output and one expectation update, weighted by the
kept rows; a batch-end step runs only if a registered term still
differentiates. Reconstruction and attention take two `autograd.grad`
traversals each (direct, then through the perception pullback), the other
owners one (audit 2026-10-30, `3718d9d0dc`).

**Who owns what** (`Models.objective_parameter_groups`):

| Owner | Parameters | Trained by |
|---|---|---|
| reconstruction | perception's codebooks (through the perception pullback), the conceptual definitions (sparse sigma and pi values), the grammar operators and their inverses, the compose chooser, the generate policy and the decoder's operators | the derivation's reconstruction losses (`reconstruction.*`), the decomposition and walk teachers (`reconstruction.decomposition`, `reconstruction.walk_policy`), the walk credit (`reconstruction.walk_credit`), the grammar lessons, and the chooser's paired-trial credit |
| expectation | one shared `ActivationPredictor` (`SymbolExpectation._inter_predictor`, also read as the word predictor) and the conceptual components (`IndependentComponents`, including the bind resolver `reference_close`); the percept predictor has its own local Adam and is not an owner group | expectation terms whose targets and sources are detached: a predictor never trains what it predicts; as built also `expectation.reference_mentions` and `expectation.bind_score_function` (the audit note below) |
| output | the reading map: the output space, the answer attention, the answer and comparison readers, the synthesis parameters; in configurations whose output names concepts, also the concept layer's sparse coefficients | the answer and output costs |
| attention | the candidate attention | the sentence's reconstruction and answer costs through the admission read (`AttentionCredit`, a cotangent to the candidate attention only), plus `attention.reading_lesson` and `attention.candidate_departure` |

**How the answer reaches the understanding: only by selection.** Each
sentence is read in two trials, the greedy one and one departure. Their
costs are compared as reconstruction plus the supplied answer's cost
(`SentenceCredit.components`; expectation is excluded from the
comparison; since item 2.5 round A the answer's cost is graded, the kept
answer idea's content coverage of the supplied word's code plus the text
cost, so a departure that binds better wins). The cheaper trial is kept, and the chooser is credited for
the departure by a score function on the detached cost difference
(`Models._compose_score_function_loss`), booked to reconstruction, the
chooser's owner. The answer's gradient never reaches the operators, the
definitions or perception. A supplied answer teaches which reading to
choose, not how to bend the weights of the reading that was chosen.
*Audit 2026-10-30 (`3718d9d0dc`): one exception as built. The bind resolver
(`IndependentComponents.reference_close`, expectation-owned) is supervised
directly by a supplied referent (`expectation.reference_mentions`,
`ReferenceClose.answer_loss`, features and cue detached), and its departure
credit is booked to expectation (`expectation.bind_score_function`), not
reconstruction. Whether to keep that is Alec's call.*

**Where reconstruction is a loss and where it is an audit.** The
derivation's reconstruction (the completed understanding inverted back to
the sentence's leaves) is a loss, owned by reconstruction. The free
decoder's byte identity (`reconstruction.free_bytes`) is reporting only under
the field objective and is the decoder's lesson: it decides which decoder
walk is kept and trains the generate policy. Its percept-residual identity
(`reconstruction.free_percept`, the decode against the leaves' residual) is a
trained reconstruction term at `reconstructionScale` with the trial root live
(corrected 2026-10-30). Built in item 2.7 (record §38.3): a leaf recovered
through a certified exact native inverse is detached from the field loss and
reported as `certified_inside`; the co-operand witness stays live and the
observed target stays detached; approximate maps keep their reconstruction
loss.

**Expectation and reconstruction (decided 2026-10-29).** Percept
expectation predicts each arrival from prior context and learns at that
arrival, inside the processing loop, from its own local loss (item 2.7
record §39; once per word, a local Adam at the main learning rate, before
either trial composes, shared by both trials). The forward path carries the full signal, `expected +
residual`. The inverse side is split by residual (record §42): the
completed understanding's inverse learns only `observed -
stop_gradient(expected)`, using the value frozen at arrival, so no
top-down gradient reaches expectation and expectation's loss reaches
nothing else. Perceptual layer in item 2.7's final round. At the conceptual
layer (item 2.5 round B) the expectation is `ActivationPredictor`, whose
predicted next state is also the spread that replaces location diffusion,
trained only by its own error against the detached next state; `intuit` is
the top-down component, a Sigma over Pi with its own weights trained only by
output. The conceptual split and the correction-output form (D7) are future
work (Alec, 2026-10-29/30; thinking spec §22).
This supersedes, at the percept level, the 2026-10-20 amendment below
that describes expectation as a gain field "never a subtraction": the
residual is now signed (an expected absence is negative), and the full
signal is recovered as `expected + residual`.

## Alternatives under exploration (Alec and Claude, 2026-10-29; not decided)

The question: should output be able to shape the understanding by
gradient, not only by selection, without the conflict 6.9 removed (output
and expectation pulling the codes away from what reconstruction needs)?

- **A. A residual band for output.** The understanding gets two parts: a
  base owned and trained by reconstruction, and a residual band owned by
  output. Output reads `stop_gradient(base) + residual`, so its error
  propagates through the operations back to the input, but only into the
  residual band's weights. Every parameter keeps one writer. The band's
  magnitude per question measures what the base lacks for answering.
- **B. Slack in reconstruction.** For approximate maps, the reconstruction
  loss gets a tolerance, `max(0, error - eps)`, with `eps` relative to its
  own error scale, and output reads the base fully. Inside the tolerance
  only output shapes the understanding; at the boundary reconstruction
  pushes back. This is the constrained problem "the best answer, subject to
  the understanding still explaining the input within `eps`". The base then
  has two writers, an extension of 6.9. Exact maps keep zero tolerance.
  *A note on random perturbation (Alec, 2026-10-29).* A zero-mean random
  perturbation of the reconstruction target leaves reconstruction's
  expected gradient unchanged, so on average it gives output no extra
  influence; it adds jitter only. Output's influence comes from the
  tolerance band (B) or from weighting (C): at perfect reconstruction the
  reconstruction gradient is zero, so output moves the understanding until
  reconstruction's restoring pull balances it. A randomized tolerance width
  has the average effect of a fixed one, plus exploration.
  Alec's intent (same day) is to keep reconstruction slightly wrong on
  purpose, so output always has a gradient. Forcing reconstruction or
  expectation to be wrong adds a gradient on reconstruction's side (a pull
  toward a wrong target, and an inverse that fits the injected error), not
  output's; output's gradient comes only from its own error, through the
  cut. With a leak or slack, perfect reconstruction leaves output the most
  room: reconstruction's gradient is zero there, so output shapes the
  understanding unopposed whenever its answer is wrong. For a force that
  persists when answers are right, a margin or confidence term on output is
  the principled form; for robustness, perturb reconstruction's input
  (denoising), not its target.
  Alec's settled intent: expectation stays exact (if it does all the work,
  so much the better), and the jitter goes on reconstruction. Jittering
  reconstruction's *strength* (randomly scaling or dropping its term per
  step) or its tolerance does give output influence: while reconstruction
  is weak, output moves the understanding unopposed, and on average this
  equals the leak (C) or slack (B), plus exploration. Jittering its target
  does not. Either way output's gradient must first be allowed to reach the
  understanding (A, B or C); under the current ownership cut it never is.
- **C. A leak.** Output reads the base through a gradient-scaling read,
  `stop_gradient(base) + a * (base - stop_gradient(base))`: the forward
  value is unchanged and output's gradient reaches the base scaled by `a`.
  Simpler than B, but it pushes against reconstruction everywhere.
- **D. One backward instead of several.** If every place one objective
  reads another's value is a stop-gradient in the forward (as A writes it),
  the objectives' losses can be summed and differentiated by a single
  backward with one optimizer; each parameter still receives only its
  intended gradient, and the ownership audit becomes a certificate that the
  single backward equals the per-owner gradients. B and C fit a single
  backward too. One caveat: perception's gradients are pulled through a
  separate traversal today (`pullback.gradients`), which must join the same
  graph or remain a second call.

- **E. One objective for syntax and semantics.** With hard choices the
  model is a stochastic computation graph, and the gradient of one expected
  cost, `E[reconstruction + answer]`, has two parts: the pathwise gradient
  through the values the choices produced (the signal gradient), and the
  score-function gradient, the cost times the gradient of the choice's log
  probability with the other trial's cost as baseline (the chooser's credit,
  already computed by `_compose_score_function_loss` from the paired
  trials). Both come from one surrogate loss and one backward:
  `total_cost + sum over choices of detach(cost - baseline) * log_prob(choice)`.
  Two things keep syntax and semantics apart today: the chooser reads
  detached features, so its credit never reaches the meaning representation
  that made a choice easy or hard; and the ownership cut keeps the answer
  out of the understanding. Lifting the first lets the meanings learn, in
  part, to make good parses likely, while the parses are learned from the
  meanings they produce. The risk is variance: the score-function term is
  noisy, and in shared parameters Adam's per-coordinate scaling can let that
  noise dominate, so it needs its own scale and a measurement first.

Alec's reading (2026-10-29): expectation and attention are well handled by
removing a band and stopping its gradient; output and reconstruction need
slack or a leak; a single gradient is the ideal, since a strongly
partitioned syntax and semantics is doubtful as a psychological model.
Claude's recommendation: D with A in a round of its own after item 2.7,
then measure what the band learns; add B only if the base itself must
move for answers; E as the next step once A and D are measured.

**Decided direction (Alec, 2026-10-29).** Stop-gradient for attention and
expectation; jitter of reconstruction's strength against output (a leak
with a random strength, so the understanding's parameters have two
writers); and the chooser's log-probabilities summed into the same
objective with live features (E), all differentiated by one backward (D).
Specified in item 2.5's spec, `plans/2026-10-27-item2-5-thinking.md` §16,
with the re-check in §16.1: each band is stopped at its output where it is
consumed, percept expectation keeps its in-loop update, the 6.9 drift risk
returns in weighted form and is measured. Not built. Also decided (Alec,
2026-10-30, thinking spec §22): `intuit`'s own Sigma and Pi weights are
output's, trained only by its use in output; `ActivationPredictor` is an
expectation band, its prediction also the conceptual spread.

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Expectation is a **gain field** `c = m⊙a⊙o` with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec; [thinking spec §§16, 22](plans/2026-10-27-item2-5-thinking.md), [2.7 record §42](plans/2026-10-27-item2-7-corpus.md)): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are not built.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23, [step 2 §71](plans/2026-10-21-step2-one-ltm.md); recorded 2026-10-30):** an LTM1 row is two tagged references with its kind read from the referents, not three. As built at `3718d9d0dc`, `LTM0` still carries `c_plus`/`c_minus`, `source_trust`, `order`, `row_ids`, `origin`, `record_kind`, `polarity` and `witness_count` columns; only `refs`, `grammatical_mode` and the index columns are retired (audit 2026-10-30, Alec to decide).
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.


The current contract is item 6.9 [plan §§21–23](plans/2026-09-29-item-6-9-xor-grammar.md#23-review-of-the-22-round-claude-2026-10-03),
building on [§15](plans/2026-09-29-item-6-9-xor-grammar.md#15-one-writer-for-each-weight-alec-2026-10-02),
October 3, with sampled exploration, the decoder margin audit, affine numeric
reading and support-governed decoder eligibility under the uncommitted
[6.8 §§10–14 review and §14 addendum](plans/2026-09-27-item-6-8-one-attention.md).
Each optimizer parameter has one objective owner. The supplied
answer stops at the understanding. Item 6.5's accepted mechanism adds
fresh current-source encodings through the live identity columns; older
context, routing evidence and observed targets remain detached. The
[implementation proposal](specs/2026-09-26-independent-components.md#9-implementation-proposal-october-7)
records this sentence-boundary choice and the sparse-coding energy, accepted
in spec §10 and by Alec on October 7. Step 5a's uncut answer and the supplied-answer gradient projection
are retired. Forward representations and numerical operator hosts remain shared.

<a id="the-training-step-october-1"></a>

## The training step (October 3)

A sentence has a greedy and an explore derivation. Both are composed and
costed at identical parameters before either trains. Each trial then takes
its own restricted backward and optimizer step, including its own perception
pullback for reconstruction. Saved forward parameter values preserve the
second graph. A batch-end step runs only when its remaining registered total
has a derivative. No extra step is added for diagnostics.

Under Item 4's October 19 rule, explore is kept only when `R + A` is
**strictly lower**. A tie keeps greedy. The chooser's detached advantage uses
that same owner-step total: registry reconstruction and relative answer error
when supplied. E trains the shared predictor independently on its local
residual; it enters neither keeping nor departure credit.
Each answer is read from that trial with the understanding detached, before
either optimizer step. Reconstruction and expectation train on both trials.
Round 2e gives the answer owner two independent readers of the same form.
The presented reader takes one step on the kept root only.
The comparison reader supplies A and takes one step on the mean of both root
losses for compose departures, or the kept root for narrowing departures.
Both readers see detached roots. The batch-end answer read uses the presented
reader and reports its error without another update. The kept state is committed detached;
no gradient crosses that boundary into the next sentence.

Input narrowing and compose share one departure per sentence. Round 2d draws
uniformly among the `W` walks with an eligible round, then among that walk's
`R_walk` eligible rounds, then among the selected round's `K` eligible
alternatives. The unmasked policy probability supplies the
`K·R_walk·W`-corrected score-function credit. A
narrowing departure continues through admission and compose greedily; a
compose departure uses greedy narrowing and replays its compose prefix.
Ordinary thought, anticipation and generation retain their existing paired
walks and owner boundaries. No learning rate, optimizer or budget changes
accompany the sentence comparison.

| Parameter owner | Weights and permitted update |
|---|---|
| Reconstruction (R) | Perception (the sole writer of native PS/WS prototypes and 11b evidence), field dictionaries, compose operators and tied inverses, compose chooser, and the shared generate decoder (chooser and parameterized generate faces). Grammar lessons train only their chooser, never an operator. Momentum descent; codes have no EMA refresh or contextual rotation. |
| Expectation (E) | Item 4's shared symbol activation predictor receives local residual gradients. Every source, including the latest one, and every target is detached; the entire gain saturation is detached. Percept moments update from external observations without gradients. Existing independent auxiliary objectives retain their own owners. ARMA and structured contrastive heads are retired. |
| Answer (A) | Numeric head and affine root/end/echoic-bank reader, plus its independent comparison copy on the sentence path; learned memory retrieval for thought and generation, question conditioners and answer adapters/controllers. Only the presented reader supplies outputs; the comparison copy supplies the chooser's answer cost. The concluded understanding is detached. The generate chooser belongs to reconstruction. |

[ObjectiveOwnership](../bin/ObjectiveOwnership.py) reads the trained named
terms from `Layers.Error` and obtains gradients only for their owner lists.
A differentiable numerical operator can transmit a generation derivative to
the conditioner while its own parameters receive no answer update. Regularizers
are unnormalized and follow the parameters they use. The diagnostic operand
term of a grammar lesson trains no operator. `expectationPolicyWeight` stays 0.
The chooser reads detached expected activations and attenuated content separately; excess and omission never substitute for content.

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

A native word's lower bound is `L = max_parts(part_code)` over present
parts. Its wholes are adjacent-word pairs witnessed in addressed sentence
rows. Their joins give `U(w) = L(w) ∨ ⋀ L(neighbour)`. The detached placement
is `c = L + α(U−L)`, with `α = W_U/(W_P+W_U)`: `W_P` counts native part
atoms (including length and collision atoms), and `W_U` counts distinct
`(sentence address, pair position)` witnesses. Re-reading does not increase it.

In decreasing part count, containment caps each word by all its containers,
never below `L`; equal part sets are capped together. The index recovers
`L = [c == 1]`; binding projects `c` through round 3a's fixed matrix. The
room pass is retired. Placement is a detached index computation with no
optimizer owner. The audit records movement, all-pair cosine, identity, and
containment before and after the cap.

The serial leaf is `[form × |activation| | meaning × |activation|]`, with
meaning poles exchanged for negative activation. Location bands remain with
the occurrence. Field-path dictionaries retain their existing ownership;
`XOR_exact`'s answer coefficients are unchanged.

The first block stores the symbol's perceptual position. Its content coordinates
carry the placed form `c`; its reserved location/time positions are zero for
a generally characterized type, not an occurrence stamp. The complement stores
the concept's conceptual position. In the final operators update, every stored sentence row
has a fixed sparse identity: `s` ones of `K`, generated privately from its
content key. Identified word bytes define the text sentence's key; occurrence
addresses remain provenance and never seed the code. Direct opaque rows use
native roles and metadata as a fallback content key. The content lookup retains all
distinct occurrences; repeated presentations re-witness one addressed row.

Under plan §44, each fixed identity direction has magnitude `n/(n+4)`, with
`n` the row's witness count. Order-zero meaning takes the detached,
recency-weighted mean of those identities, normalizes its nonzero direction,
and scales it by `n/(n+4)` for the number of containing rows. It remains
on the **for** pole, with zero on the **against** pole. The flat complement is
`[for_0 … for_K−1 | against_0 … against_K−1]`. References and inverted leaf
postings retain their membership; only REL_NONE rows contribute and DEF rows
are excluded. `1/(1+age)` supplies recency independently of truth evidence.
One snapshot of meanings and centroids precedes lexical staging, including
the eager staging used by compiled forwards. Composition, inverse search and
readback share it across both trials. It is not refreshed until committed
writes finish; a current closing becomes context on the next forward. Composed meanings remain
stored as `o`, but no longer supply their own bootstrap.

The bootstrap is resolved by an index, with no learning owner, parameter or
new objective. Grammar gates have 64 pairs and three bits per identity
(128 meaning floats after the unchanged 104-dimensional form event).
BasicModel keeps 896 meaning floats: 448 pairs, six identity bits.

Composition keeps the round-3a form kernels and their normalization. Meanings
compose independently: conjunction/intersection `(min⁺, max⁻)`,
disjunction/union `(max⁺, min⁻)`, sum the mean of each pole, and `not` the pole
exchange. No normalization joins the blocks. Operators without a meaning
write preserve that block. `(0,0)` is unknown, so `a ∧ not b` has no for
evidence and has b's extent against; it does not assert a closed-world
complement. Interpretation applies signless identification presence to form
and the independent evidence magnitudes to two copies of the context code:
`[form × presence | code × c⁺ | code × c⁻]`. `non` clears only the for lane.

The static semantic certificate checks all word pairs against sentence
postings, including negation. It is exact on the grammar vocabularies and a
false-membership diagnostic on BasicModel's corpus. The sparse code carries
composition and similarity; exact membership remains a posting query. Both
connectives can support the XOR read, so mixed final operators are valid.
Membership priming and the adjacent-word centroid are included in the combined step.
The dictionary has no EMA refresh or contextual rotation. Ownership audits
distinguish absent gradients from zero ones and retain inactive weights.

The inverse identifies each word by its signless form: one bank entry per
word, form-only shortlists, pair residuals and STOP eligibility. After word
selection it reads each evidence lane independently against that word's
frozen context code. For each lane, the min inverse chooses the least compatible operand magnitudes and the max inverse the greatest.
A mean uses a nonnegative two-code solve. An ambiguous inverse recovers a compatible
reading, without consulting the input derivation. Readback uses the form,
including when both lanes are full or empty.

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
with their priming weights. The degree-normalized energy ledger is retained.
Retrieval reads detached positive excess divided by its per-stream maximum
(clamped below by one), then scaled to `primingMaxBoost - 1`. Neutral remains
one; the default maximum is two. Signed inhibition below neutral is preserved.

The numeric head reads detached end slots plus an answer-owned, zero-initialized
linear reader of the root (D), three masked end slots (3D), and the echoic
priming-weighted code sum (D). Its extra width is 5D. There are no rule
histograms or original per-word features. The linear reader preserves additive
evidence for the sum-only control. With `readerBlockNormalization=true`,
each form and meaning block is divided by the maximum norm of that block
in the shared code snapshot, clamped below by one. The echoic bank uses its
own shared block scales. These scales are common to all trial roots and
streams; a composed root is never normalized by its own norm. Thus the
reader remains affine and the sum control remains additive. This scaling
is a reader operation and never changes composition or stored meanings.
Generated answers detach the same conceptual
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
with the best one-code least-squares explanation on the form block.
Both children must be nonzero valid shortlist forms, neither
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

*A superseded passage beginning "The eager training audit additionally records the first deco" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*


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

*Audit 2026-10-30 (`3718d9d0dc`): this table is the record, not the inventory.
Trained terms it does not list: `reconstruction.field`, `.content`,
`.coverage`, `.part`, `.iterations`, `.forward_discard`, `.reference`,
`.free_percept`, `.walk_residual`, `.decomposition`, `.walk_policy`,
`.walk_credit`, `.compose_score_function`; `expectation.equal`, `.components`,
`.bind_score_function`, `.reference_mentions`; `attention.reading_lesson`,
`.candidate_departure`; `<owner>.definition_fold`;
`<category>.thought_score_function`. The listed `reading_attention` and
`expectation_policy` have no writer. The current mapping is
`ObjectiveOwnership.registry_costs`.*

| Registered term | Comparison and uninformed baseline | Priority | Owner and gradient |
|---|---|---|---|
| `answer.what/where/when` | Numeric output or generated surface band vs supplied target; origin per band | corresponding event-band scale | Trial and batch; A |
| `reconstruction.free_bytes` | Shared generate walk spelling vs observed bytes including NUL; `log 256` | `reconstructionScale` | Trial; R, including decoder and live root; candidate bank codes detached. Detached chosen report at batch end |
| `reconstruction.what/where/when`, `reconstruction_reverse.what/where/when` | Grammar-free perceptual/masked or reverse event vs input target; origin per band | reconstruction scale × band scale | Batch; perception and the active input inverse |
| `leaf_distill` | Historical root decoder's predicted exact leaves vs retained leaves; origin | `leafDistillWeight` | Batch; decoder and live input root where enabled |
| `expectation.activation` | Predicted two-lane bank activation vs the next row's actual activation; bounded MSE | `sentenceExpectationLossWeight` | Trial or batch, once; shared predictor only, sources and targets detached |
| `expectation.word` | Pre-arrival word activation vs the admitted word's two lanes; bounded MSE | `wordExpectationLossWeight` | Trial or batch; shared predictor at the word read, sources and targets detached |
| `expectation.symbols` | Pre-compose bank activation vs the resulting symbolic field's two lanes; bounded MSE | `wordExpectationLossWeight` | Trial; shared predictor at the S read, sources and targets detached |
| `expectation.components` | Retained observation population plus current fused S; Laplace sparse-coding energy for A and one-sparse innovation energy for B | `independenceWeight` | Trial; E owns columns and relevance; the existing closing comparison credits the global grammar choice |
| `expectation.intra` | Predict-then-perceive code vs arriving code; origin over active scalar entries | `intraLossWeight` | Trial or batch, once; within-sentence predictor only |
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

`branchDiagnosticsEvery` now reports `[objective-ownership]` rather than
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
is retired by §14. Round 4a resolves the bootstrap and the operator split;
co-activation priming and the answer reach remain deferred.

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

*Moved to [DevelopmentHistory](DevelopmentHistory.md#expectation-at-the-closing) (2026-10-30): "Expectation at the closing".*
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
  `K · R_walk · W · p_θ(a_dep | state) · (C_explore − C_greedy)`, costs detached, one term per
  sentence with a departure, reconstruction-owned; `∇p` rather than `∇log p`
  because the departure is drawn uniformly over the value-distinct eligible
  alternatives (the coverage floor) and the importance weight cancels the
  `1/p`. Here K counts the value-distinct eligible alternatives at the sampled
  round, `R_walk` counts eligible rounds in the drawn walk, and `W` counts
  walks with an eligible round in the sentence. Round 2d draws walk first;
  their product cancels the proposal probability `1/(W·R_walk·K)`, giving the sum of the
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


*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-1-october-5) (2026-10-30): "Operators update round 1 (October 5)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#the-target-scheme-october-6-decided-scheduled-by-rounds) (2026-10-30): "The target scheme (October 6; decided, scheduled by rounds)".*
*A superseded passage beginning "<a id="operators-update-round-2-credit-october-6-uncommitted" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-2b-october-6-rejected-candidate) (2026-10-30): "Operators update round 2b (October 6; rejected candidate)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-2c-october-6-not-accepted) (2026-10-30): "Operators update round 2c (October 6; not accepted)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-2d-october-6-review-candidate) (2026-10-30): "Operators update round 2d (October 6; review candidate)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-2e-october-6-review-candidate) (2026-10-30): "Operators update round 2e (October 6; review candidate)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-3a-october-7-review-candidate) (2026-10-30): "Operators update round 3a (October 7; review candidate)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-4a-october-7-superseded-by-the-combined-step) (2026-10-30): "Operators update round 4a (October 7; superseded by the combined step)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#operators-update-round-4a-0-the-occurrences-address-october-7-isolated-historical-measurement) (2026-10-30): "Operators update round 4a-0: the occurrence's address (October 7; isolated historical measurement)".*
*Moved to [DevelopmentHistory](DevelopmentHistory.md#final-operators-update-october-7-uncommitted-review-candidate) (2026-10-30): "Final operators update (October 7; uncommitted review candidate)".*
## The lanes at the leaf (Alec, 2026-10-07)

The combined step's interpretation rule — the form takes `|a|`, the meaning
takes `|a|` with its poles exchanged when `a < 0` — and `pole_activation`'s
`net = pair⁺ − pair⁻` collapse the walk's pole pair into one signed scalar,
which cannot distinguish *both* from *neither* (two-truths §1.1). The repair
pass carries the pair to the leaf: `[form × presence | code × c⁺ | code ×
c⁻]`; the for lane receives `c⁺`, the against lane `c⁻`; no difference or
sign is formed anywhere between the walk and the stored row; the inverse
identifies a word by its signless form and reads both lanes with the
leaf's pair. The row's `(c⁺, c⁻)` is the spec's required-evidence read: min
over nonzero contributions on each pole independently.

## The generation residual: how it is used, and its repair (Alec, 2026-10-30)

*Decided in conversation on 2026-10-30 (thinking spec §§26.1, 26.2). Not
built: item 2.5's repair round builds it. Where this section and the
"generation exception" paragraph above disagree, the paragraph above
describes the code as it stands and this section the decided repair.*

**What the residual is.** Generation reads a concept back into words by a
walk over a stack: at each step the top code (the parent) is split by an
inverse binary operation, rewritten by an inverse unary operation, or
emitted as a word (STOP). For every legal proposal at the step, the menu
(`Generative.eligibility_menu`) composes the proposed pieces forward again
and measures how well that reproduces the parent, separately for the form
block and the two meaning lanes (for and against). That measurement is the
residual. It is used four ways:

1. *Chooser features.* Each proposal carries five detached numbers into the
   generate chooser (`InverseMenu.features`, scored by
   `LanguageSpace.generate_policy_logits`): the form residual, the two lane
   residuals, an exactness flag and the stack effect ($+1$ binary, $0$
   unary, $-1$ STOP).
2. *Exactness.* A proposal is exact when its residuals are at most
   $10^{-8}$; an exact binary pair or exact terminal pre-empts every
   approximate proposal at that step (the pair-first rule).
3. *Admissibility.* A proposal whose residual is not finite is removed from
   the menu.
4. *The loss.* The chosen proposal's residual at each step, summed over the
   walk, plus the energy left unresolved on the stack, is the reconstruction
   term `reconstruction.walk_residual`. On the gradient path it is recomputed
   through the chosen recomposition (`Generative.selected_residual`), so its
   gradient trains the grammar operators and their inverses.

**What was wrong.** Today every residual is a ratio: the step's error energy
divided by the parent's own energy, per lane, with the parent live
(`Generative._relative_energy`). The against lane is usually exactly zero,
where the code falls back to the raw error, or nearly zero, where the ratio
is the raw error times an arbitrarily large factor: the measure is
discontinuous at zero. That produced the loss of about $10^{29}$ and the
non-finite gradient of the one-backward arm (thinking spec §25). Within one
step every proposal shares the parent, so the division never changed the
ranking among proposals; it only rescaled the features by a step-dependent
factor, and through use 3 it could remove a correct proposal from the menu.

**The repair, decided (D-i).** The loss is registered through the error
registry like every other term, never as a finished scalar:

1. The per-step measure (`relative_residual`, renamed `absolute_residual`)
   returns, per lane, the step's absolute error in the code's own chart
   (below) and no longer divides.
2. `walk_residual` sums the chosen steps' errors over the walk and adds the
   unresolved stack energy, per lane, without dilution by the step count.
3. The sentence close registers three terms,
   `reconstruction.walk_residual.form`, `.for` and `.against`, through
   `Error.error(name, per_stream_sum, 1., mask=active,
   category='reconstruction')`, one entry per active stream, so the registry
   averages over streams and divides once per term. Three terms keep each
   lane in its own units (2.7 record §38); what goes is the per-step
   division.
4. No floor, cap, detachment of the error, or dropped term anywhere. The
   per-step live-denominator derivative has no site, so the overflowing
   terminal becomes a passing certificate.

**The measure, decided (D-ii; Alec: no ratios of error terms).**
The residual is an absolute error in the code's own chart, in all four uses:

- *Symbolic codes* are membership vectors in the unit cube (every admissible
  symbolic proposal is checked `in_cube`), so the mean squared error per
  lane is already bounded in $[0, 1]$ and in fixed units. The symbolic
  exact-pair test in `LanguageSpace.decoder_eligibility` already uses exactly
  this measure.
- *Numeric codes* (the math chain's walks, `cube_domain=False`) are not
  membership coordinates and span magnitudes. Their error is the squared
  difference in an $\operatorname{asinh}$ chart,
  $(\operatorname{asinh} r - \operatorname{asinh} p)^2$: absolute near zero,
  logarithmic for large magnitudes, continuous, and with no division.
- *Exactness* reads the same absolute error against the same tolerance.
- *Admissibility*: an absolute error of finite codes is finite, so the
  finiteness gate becomes an assertion that fails loud instead of a silent
  exclusion.
- *The loss* then registers with the registry's constant unit baseline, as
  `reconstruction.reference`, `reconstruction.free_percept` and the
  expectation terms already do. D-i as first written divided by the
  sentence's detached parent energy per lane; that is still a ratio, and for
  the against lane, usually zero or nearly so, it keeps the same
  discontinuity at sentence scale. With the absolute measure no
  data-dependent denominator remains anywhere in the generation residual.

The absolute measure changes the scale of $R$ against $A$ in the kept-trial
comparison ($R + A$) and, under `separateChooser` false, in the single
backward. No balancing weight is introduced to compensate; the three
lesson-13 arms measure the effect. Decided in full (Alec, 2026-10-30:
"Yes, let's try that"); thinking spec §26.3 holds the round's text.
