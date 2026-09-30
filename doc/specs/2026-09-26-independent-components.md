# Independent components: identity as columns, verbs as change

> **Status:** specification, 2026-09-26, written by Claude from Alec's
> decisions in conversation on 2026-09-26. Todo item **6.5**, to be done
> after item 7 (two truths) and before item 6. Codex implements; Claude
> reviews before commit under the repository publish rule. It governs how
> the identity of an object across consecutive sentences and the identity
> of a transformation across their differences are *learned*, and it adds
> the gradient pressure that §1 shows does not exist today. It refines
> [two truths §3.5](2026-09-16-two-truths-ideas-and-relations.md) (object
> permanence by reference) and
> [the accessible mind §2.6.4](2026-09-20-accessible-mind-subsystems.md)
> (surprise is what is learned). Everything marked **(decided)** is Alec's
> decision. Nothing here claims learned behaviour.

## 1. The problem (measured, 2026-09-26)

Alec's framing: identification of an object across observations means
applying the same noun phrases; prediction means applying the same verb
phrases. Two questions followed: what pressure exists within the gradient
to make this happen, and are LTM and inter-frame prediction used to
encourage it. The audit of `206a0146` answers both in the negative.

**Identity has no gradient behind it.** A noun phrase gets its object by
lookup. The interpret operator maps a word to an object by set logic over
its recorded associations: a new word mints a new object, a seen word
returns its existing object, and ambiguity is resolved only by a referent
carried from the reference chain, otherwise it stays unknown or raises.
`resolve_lexical_references` is rule code, excluded from compilation and
outside autograd. Identity is therefore *word* identity, a type, unless
the chain carries a token, and no loss anywhere depends on which binding
was made. Two mentions of "cat" are the same object whether or not they
are the same cat, and nothing can prefer "same individual" over "new
individual".

**Prediction trains the predictor, not the encoder.** The inter-sentence
layer predicts the next sentence's end state, three roles plus presence,
from the last eight end-state roots of the same stream row. Both the
target and the context are detached, so the loss shapes the predictor
only. §2.6.4 of the accessible-mind spec already requires the *source*
ideas to be live ("the gradient runs through the predictor and its live
source ideas"); the code does not do this. The expectation's negative
image enters the ended idea and the thought context detached. At their
defaults the relevant knobs are: inter-sentence prediction weight .1 with
target and context detached; within-sentence prediction weight .1;
expectation gain 1, detached; expectation policy weight 0; selected-thought
policy weight 0; inter-sentence contrastive weight 0; prediction-trial
ratio 0; embedding co-occurrence training off. Reconstruction is the only
pressure shaping representations.

**LTM is not used for this.** The predictor sees the per-row chain of
recent roots plus retrieved what-frames from thought records as further
detached context. LTM rows are never sources or targets of gradient.

## 2. Decisions (decided, Alec, 2026-09-26)

### 2.1 ICA, not SIGReg

SIGReg (LeJEPA's sketched isotropic Gaussian regularization) and ICA use
the same machinery — one-dimensional projections tested against a
Gaussian — as duals. SIGReg *minimizes* the deviation on random
projections so the marginal becomes isotropic Gaussian; that distribution
is rotation-invariant, which is why it prevents collapse and why it
cannot identify anything. ICA *maximizes* the deviation on learned
projections after whitening, because mixtures of independent sources look
more Gaussian than the sources. If the components we want are noun
phrases, independence is the objective that names them. **ICA is the
model. SIGReg is not adopted**; its anti-collapse role is covered by the
unit-variance constraint the independence objective already carries.

### 2.2 Gradient form only

The independence objective is the maximum-likelihood (Infomax) form: a
differentiable loss with a heavy-tailed prior on the components, trained
by the same gradient as everything else. No fixed-point iteration
(FastICA), no separate algorithm, no model-selection search. This is the
same position as item 7.5: one gradient does all the optimization.

### 2.3 The bridge: the symbol level is the classical mixing model

Classical ICA: observations are a fixed matrix times independent,
non-Gaussian source series, and the matrix is the same at every time step.
At the symbol level a sentence already has that form, because a meaning
is a superposition of concept rows and the symbol layer represents a
sentence as pairs of row and signed activation:

```text
frame_t = A · a_t     frame_t: the sentence's activations over order-0 event rows
                      A: order-1 concept rows as columns — object signatures over those coordinates
                      a_t: sparse signed activations of the objects sentence t mentions
```

The sources are not the encodings; they are each object's activation
series across consecutive sentences. **An individual is a column, and
identity across frames is the fixedness of A over the LTM chain.**
Identity is not found by matching components frame to frame; it falls out
of estimating one matrix over the whole chain. This individuates within a
word: if the occurrences of "cat" split into two independent activation
series with distinguishable property or where bands, the estimator wants
two columns, two cats; if two words always co-occur, it wants one column,
one whole. Sparsity supplies the non-Gaussianity — an object is absent from
most sentences, so its series is heavy-tailed. Linear ICA applies here and
not at the byte level because perception's nonlinearity is already spent
by the time a sentence is composed.

Verbs are the columns of a second matrix over the temporal differences in
those coordinates — the surprise residual `r = o − ê` of §2.6.4, taken over
identified objects:

```text
Δ_t = B · e_t         B: verb signatures — fixed patterns of change over object coordinates
                      e_t: sparse occurrence series of event kinds
```

A verb is a fixed pattern of change: which object's presence flips, whose
where band moves, which property appears. Identity of a transformation is
again one column, the same every time it fires. In classical terms this is
ICA on the innovations of a temporal predictor. **The second decomposition
is taken in the coordinates the first defines**: nouns first, verbs second.
A sentence then reads as which columns of A are active and which column of
B fired, which is the `NP VP NP` row the two-truths spec writes.

### 2.4 Priors on the number of sources

Classical ICA needs the number of sources as a hyperparameter. Here it is
fixed at three levels, none tuned.

* **Per frame: the grammar, bounded by STM.** One S has three roles, so a
  frame mixes at most two noun columns and one verb column, plus the rows a
  relative S references; across the recency buffer the ceiling is the STM
  capacity (`<stmCapacity>`, 8 — Miller's number, not an estimate). One VP
  per S makes the event vector **one-sparse**: one-sparse coding is vector
  quantization, so the verb decomposition over the surprise stream is a
  learned codebook of change patterns.
* **Which few, of a large dictionary: the sparsity prior.** The dictionary
  is far larger than the concept dimension, so both matrices are
  overcomplete and the working form is **sparse coding**, not square ICA.
  The heavy-tailed prior supplies the non-Gaussianity and makes overcomplete
  recovery well posed: a few active columns are recoverable when their
  count is small relative to the dimension and the columns are mutually
  incoherent. The STM bound keeps the count small; the independence
  pressure keeps the columns incoherent.
* **In total: growth and pruning, not a prior.** Minting becomes
  residual-driven: a new column is minted when the surprise cannot be
  explained by existing columns, gated by item 9b's recurrence rule so a
  single unexplained frame does not mint. Pruning is item 5's forgetting,
  with an automatic-relevance scale per column that the gradient may drive
  to zero. Model order is decided by the same gradient.

Declared hyperparameters: the sparsity prior's strength, the mint threshold
on unexplained surprise, and the forgetting threshold. STM capacity and the
three roles are already declared.

### 2.5 Population, not batch

The batch is two sentences, so the independence statistics run over a
population the model already holds: the LTM chain of end states for nouns
and the accumulated surprise residuals for verbs. Memory holds the
independent components of the world so far.

### 2.6 Scope: which binding problems this solves

* **Across frames — which thing this is.** Yes: identity is a fixed column.
* **Within an object — which features belong together.** Yes, statistically:
  a column is a bundle of features that co-vary across frames (Barlow's
  suspicious coincidences); independence between columns keeps two
  objects' features apart.
* **Within a sentence — who did what to whom.** **No.** A superposition
  says which columns are present, not which role each fills; "cat chases
  dog" and "dog chases cat" activate the same three columns (the
  superposition catastrophe). Roles are bound where they already are: at
  the closing, by the three slots. What ICA adds is the *credit*: prediction
  runs in column coordinates, so the correct assignment is the one under
  which the next frame is predictable, and a wrong binding yields surprise
  that reaches the chooser's slot decisions through the closing.
* **Within a field — two red things at once.** Not ICA's either:
  co-variation cannot separate two things present in the same frame.
  Features that pervade the same extent belong to the same whole (the
  meronomy and item 11c's located concepts). ICA takes those located
  bundles as its observations.

The spec states this so that nobody reads it as a claim about role
assignment.

### 2.7 Tie to the grammatical derivation

Not a pipeline. Identification fixes the *semantic skeleton* of a frame —
which identity columns are present and which verb column fired — and
nothing inside the phrases. Spans, determiners, adjectives, morphology,
tense, prepositions, embedding and word order are the grammar's. The
coupling is by gradient through the closing in both directions:

* **Upward:** the parse fixes how many sources the frame mixes and which
  words fill which roles; the ended row is what identification decomposes.
  Grammar errors show up as smeared columns and are recorded as grammar
  limits.
* **Downward:** the column inventory defines the candidate referents. At
  each noun position the item 7.5 softmax gains candidates that bind the
  phrase to a column present in the recency buffer or in cued frames, or
  mint a new one. The independence and prediction losses are computed in
  column coordinates and their gradient reaches the compose operators and
  the chooser through the closing by the straight-through path 7.5 defines.

No word is anchored to an operator ([operators have no predefined
surface](../../todo.md)): a word whose row usually serves as an object column
tends to be routed as a noun by learned credit, not by lookup, and "run"
as noun or verb is decided per frame by which matrix its row best serves.
Psychological analogue: Pinker's semantic bootstrapping (things → nouns,
actions → verbs seed the syntactic categories) with Gleitman's syntactic
bootstrapping as the reverse direction; both live in the loop.

**The determiner is the lexical cue for the choice** *(Alec, 2026-09-28;
[accessible mind §2.0.1](2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention))*.
The bind-or-mint candidates of the downward coupling have words that ask
for them: "a" lowers the noun phrase and mints a column, "the" lowers it and
binds to a column already in the recency buffer or the cued frames, and
"every" does not lower at all — the sentence "remains a high-order
relation" between the concepts at their own order, and writes no identity
column. *Amended (Alec, 2026-09-30): "every lowers like all but has a
different plurality"; it lowers to the extension and chooses no member, so
it still mints and binds no column
([5.5 spec §6](2026-09-30-occurrence-tense-aspect.md#6-surface-form-and-markers)).* This is a cue and not an anchor: the word biases the softmax
through its learned row like any other, and surprise through the closing
still credits the choice.

*Which mint and which bind (Alec, 2026-09-28).* "Mint after 'a' and bind
after 'the' seems reasonable if you mean that the mint or bind are those
used by the identity tracking system." They are: the candidates of §3.2, a
column present in the recency buffer or the cued frames, or "mint". They are
not the minting of a concept or object row, which `interpret` does for an
unknown word and discovery does for an unexplained recurrence: "a cat"
introduces a new individual, not a new concept. The cue bears on the
*choice* in the softmax; whether a column is then allocated is still gated
by recurrence (§3.5). This is the novelty and familiarity of Heim (1982,
1983), in which an indefinite requires that no file exist for its referent
and a definite that one does, with the file card as the identity column.
The learning gate is in §5.

### 2.8 Where it acts

The content band only. The `.where` and `.when` bands carry geometric
meaning (brackets, sinusoids) that an independence prior would destroy,
and the prior applies in a chart where a heavy-tailed density makes sense,
not inside the unit percept cube.

**Identity is a concept, so the columns live in the conceptual space
(decided, Alec, 2026-09-26).** A WholeSpace row is a type — a property that
pervades an extent and is shared across individuals — so it cannot be an
identity; WholeSpace rows are the field's property axes, beneath the
concept ladder, and ICA does not read them directly. The frame's
coordinates are the order-0 rows (below). The object columns are ConceptualSpace
concept rows at order 1, the particulars whose symbolization two truths
already names as object permanence (11c entry 11); kinds sit above them in
the taxonomy and are not columns. The activation series are the LTM chain
of symbol `(row, activation)` pairs. Verb columns are likewise concept
rows, over the surprise stream. Claude's earlier proposal of WholeSpace
rows as host is withdrawn: it would have made identity a fact of percepts.

Nor order 0 (Alec, 2026-09-26). Occurrences — events located in the
field with their own `.where` and `.when` — are not rows. **An order-0 row
is an event generalization: it already generalizes over where and when.**
The word and event concepts a sentence activates are order-0 rows, so the
frame `frame_t` of §2.3 is a vector over order-0 coordinates. An identity is
what stays fixed across frames of many different event kinds — the same
thing sleeping, meowing, being black — so nouns first appear at order 1 as
the generalization over events, and an order-1 column is a conjunction of
co-varying order-0 rows (the pyramid's pi over sigma rows). Kinds appear at
order 2 as the generalization over particulars. The ladder is ordered by what is abstracted away, not by
set inclusion: order 0 drops where and when and keeps the event kind, order 1
drops the event kind and keeps the individual (permanence), order 2 drops the
individual and keeps the kind — which is why one word form can address a row
at every order (11c entry 11). Likewise the surprise
`r = o − ê` is measured over order-0 coordinates and the verb column is its
order-1 generalization: a recurring pattern of change, not one change.
The row a closing writes is one fused `NP VP` point because the primitives of
our reality are spacetime events (Alec, 2026-09-27); the order-0 rows a frame
is measured in are the generalizations of such events over where and when,
so what the closing writes and what ICA reads are the same kind of thing.

## 3. Mechanism

3.1 **Independence loss over the LTM population.** For the retained chain
of ended rows (per stream row, then pooled), the sparse-coding negative
log-likelihood of each frame's content band under the current object
columns with a heavy-tailed prior on the activations and unit-norm
columns; in the square case this is maximum-likelihood ICA. Gradient flows
to the columns (the order-1 concept rows, §2.8) and to the live source
ideas of the current step; retained rows are detached observations.

3.2 **Identity binding as a chooser candidate.** The interpret operator's
set logic becomes the *enumerator* of candidates, not the decider: at a
noun position the candidates are the columns present in the recency buffer
(the situation, §3.5) and in cued frames, plus "mint". The 7.5 softmax
scores them; exploit commits, explore samples; credit is the surprise in
column coordinates. The raise-on-ambiguity path is deleted, not gated.
Because order-0 rows have already abstracted where and when (§2.8), two
individuals with the same content cannot be told apart by the columns, and
must not be: the binding decision's *evidence* includes spatiotemporal
continuity — the `.where` / `.when` reference bands carried beside the
rows and the situation's anchors — while the column's *signature* stays
over content. Indexed by continuity, described by content: the object file
of Kahneman, Treisman and Gibbs.

3.3 **Verb columns over the surprise stream.** `r = o − ê` per role in
identified coordinates is the observation; one column fires per S, so the
verb matrix is a learned codebook of change patterns with the same mint
and prune rules. Its columns are the VP rows.

3.4 **Prediction with live sources.** The inter-frame predictor's current
source ideas carry gradient, as §2.6.4 requires; the target frame stays a
detached observation; context beyond the current step stays detached.
The recency buffer is read as the situation (anchors into LTM), not copies.

3.5 **Minting and pruning.** Mint when the residual under existing columns
exceeds the declared threshold on a recurrence (9b); prune through item 5's
value with the automatic-relevance scale as one term. Both are effects, not
return values.

3.6 **No separate algorithm.** One backward per step. No FastICA, no
whitening pass outside the loss, no model-order search.

## 4. Configuration

New `<training>` elements, documented in Params.md and validated by the
schema: the independence weight, the sparsity prior scale, and the mint
threshold on unexplained surprise. The existing inter-sentence prediction
weight keeps its name and gains the live-source behaviour of §3.4 without
a compatibility switch. Names are indicative; Codex proposes them at
landing and Claude reviews.

## 5. Acceptance tests

Mechanism tests are unconditional. No test pins a random seed in order to
pass; measurements declare seeds 0/1/2 and report all of them.

1. **Identity is a choice.** For a pronoun following its noun, the chooser
   enumerates the noun's column as a binding candidate, the choice is a
   softmax decision with live log-probability, and no dictionary lookup
   decides it.
2. **Fixed column.** A synthetic stream in which one individual recurs
   yields one column whose activation series is non-zero exactly at its
   mentions; the same stream shuffled destroys the temporal structure and
   the test records the degradation rather than asserting it away.
3. **Individuation within a word.** Two individuals sharing a word but with
   distinguishable property bands, each recurring, obtain two columns after
   the recurrence gate; before it, one.
4. **Role binding is the closing's.** "cat chases dog" and "dog chases cat"
   write different rows and identical independence loss. The test asserts
   the superposition catastrophe rather than hiding it.
5. **One verb per S.** Over a synthetic stream of one event kind, the surprise
   residual is explained by one change column per sentence; two event kinds
   yield two columns with independent occurrence series.
6. **Live source, detached target.** The inter-frame loss has non-zero
   gradient at the current source idea and none at the target frame or the
   retained context.
7. **Population statistics.** The independence loss over the LTM chain is
   identical for the same chain presented at batch size 1 and 2.
8. **Content band only.** The `.where` and `.when` bands receive zero
   gradient from the independence term.
9. **Mint and prune.** An unexplained residual above threshold mints only on
   recurrence; one below does not mint; a column whose relevance scale falls
   to zero is offered to forgetting and not deleted by this item.
10. **One backward.** The step performs no iterative solve; a counter on the
    independence path records exactly one gradient evaluation per step.

Learning gates follow item 9's million-sentence prerequisite and item 8's
protocol form: held-out anaphora resolution against the rule-based binding
of `206a0146` at matched exposure; verb reuse — the same event kind maps to
the same column across held-out streams; prediction error against the
detached-target control; shuffled-order and renamed-vocabulary controls;
seeds 0/1/2, declared before training, recorded and never tuned. Learned
identity and learned verb reuse stay explicitly unproven until these pass.
*Added in direction (Alec, 2026-09-28; §2.7):* the determiner cue — on
held-out text the identity system's mint candidate is preferred after "a"
and its bind candidate after "the", against a shuffled-determiner control
and under the same prerequisite and seeds. It measures the choice in the
softmax, not the allocation of a column.

## 6. Documentation required with implementation

`STM.md` §11 (inter-sentence prediction: live sources, population
statistics), `AccessibleMind.md` (§2.6.4 implemented; §2.7.3 situation as
binding candidates), `Language.md` (binding candidates in the compose
softmax), `Params.md` (§4 elements), `Philosophy.md` (the four binding
scopes of §2.6), `FutureWork.md` (SIGReg considered and not adopted, with
the duality argument), and this spec's status line.

## 7. Decisions record (2026-09-26)

* Identity and prediction "seem to need more work"; identification =
  applying the same NPs, prediction = applying the same VPs (Alec).
* SIGReg shares principles with ICA, but ICA is the better model to
  identify the independent components that constitute an environment
  (NPs), and the same process over temporal differences of the identified
  NPs identifies the VPs (Alec). Claude: SIGReg is the rotation-invariant
  half; not adopted.
* Item number 6.5, after item 7 (Alec).
* The bridge: at the symbol level a frame is the classical mixing model;
  identity is the fixedness of the matrix (Claude, accepted).
* Priors on the number of sources: grammar and STM per frame, sparsity for
  which, mint and prune for how many (Claude, accepted).
* Not a pipeline; ICA fixes the skeleton, grammar derives, coupled by
  gradient through the closing (Claude, accepted).
* ICA does not solve role binding within a sentence; the three slots do
  (Claude, accepted).
* Identity is a concept: the object columns are hosted in the conceptual
  space at order 1, not in WholeSpace, whose rows are shared types (Alec;
  Claude's WholeSpace proposal withdrawn). Not order 0 either: an order-0
  row is an event generalization that already generalizes over where and
  when; occurrences are field events, not rows; nouns are the order-1
  generalization over events (Alec, correcting Claude's "order 0 is one
  occurrence").

## 8. References

Attias, H. (1999). Independent factor analysis. *Neural Computation, 11*, 803–851.
Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and scalable self-supervised learning without the heuristics. *arXiv:2511.08544.*
Barlow, H. B. (1989). Unsupervised learning. *Neural Computation, 1*, 295–311.
Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. *Neural Computation, 7*, 1129–1159.
Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., & Moulines, E. (1997). A blind source separation technique using second-order statistics. *IEEE Transactions on Signal Processing, 45*, 434–444.
Choudrey, R. A., & Roberts, S. J. (2003). Variational mixture of Bayesian independent component analyzers. *Neural Computation, 15*, 213–252.
Comon, P. (1994). Independent component analysis, a new concept? *Signal Processing, 36*, 287–314.
Donoho, D. L. (2006). Compressed sensing. *IEEE Transactions on Information Theory, 52*, 1289–1306.
Gleitman, L. (1990). The structural sources of verb meanings. *Language Acquisition, 1*, 3–55.
Heim, I. (1982). *The semantics of definite and indefinite noun phrases* (Doctoral dissertation). University of Massachusetts, Amherst.
Heim, I. (1983). File change semantics and the familiarity theory of definiteness. In R. Bäuerle, C. Schwarze, & A. von Stechow (Eds.), *Meaning, use, and interpretation of language* (pp. 164–189). de Gruyter.
Hyvärinen, A., & Morioka, H. (2016). Unsupervised feature extraction by time-contrastive learning and nonlinear ICA. *Advances in Neural Information Processing Systems, 29*.
Hyvärinen, A., & Morioka, H. (2017). Nonlinear ICA of temporally dependent stationary sources. *Proceedings of Machine Learning Research, 54*, 460–469.
Kahneman, D., Treisman, A., & Gibbs, B. J. (1992). The reviewing of object files: Object-specific integration of information. *Cognitive Psychology, 24*, 175–219.
Hyvärinen, A., & Oja, E. (2000). Independent component analysis: Algorithms and applications. *Neural Networks, 13*, 411–430.
Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review, 63*, 81–97.
Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature, 381*, 607–609.
Pinker, S. (1984). *Language learnability and language development.* Harvard University Press.
Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks, 6*, 623–641.
Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence, 46*, 159–216.
Treisman, A. (1996). The binding problem. *Current Opinion in Neurobiology, 6*, 171–178.
von der Malsburg, C. (1999). The what and why of binding: The modeler's perspective. *Neuron, 24*, 95–104.
