# Per-stream state: audit and the ownership rule (Claude, 2026-10-08)

Alec, 2026-10-08, on the stale priming surface: "This requirement seems
necessary, but what other state needs to be owned per batch? Perceptual,
conceptual, and symbolic activations, LTM state (which may be threaded by
.where?). Please take this opportunity to look."

## 1. What the audit found (verified in the working tree)

A batch row is one document stream for the life of a batch group
(`MathChainTraining.document_batches`: documents grouped by sentence count,
all rows ending together, `B` changing only at group boundaries after a
global `Reset`). There is no registry of per-stream state; each module
resizes and resets its own, by six different routes. Only one owner,
`BracketExpectation` (the ARMA rings, situation frames and pending
estimates), is both resized on a `B` change and keyed by the document
(`begin_document` clears a row when its address changes).

| state | holds | on `B` change | keyed by |
| --- | --- | --- | --- |
| `_priming_boosts` [B,V] on concept, word and percept spaces | seen/desire energy ledger | resized only when written; readers (`stage_input`, the thought context, relevance priority, the tower projection, the retrieval prior) see the stale shape; a [1,V] surface left by any B=1 write **silently broadcasts** one row to every stream; **never reset** (zero reset references) | position |
| `BracketExpectation` rings, frames, estimates, comparisons, documents | expectation | `ensure_batch` reallocates | **document** |
| `what_memory` slots, closure pressure, episode flags | thought/question history | fresh via `ensure_batch` each forward | position (records carry addresses) |
| concept-space STM `_live_*`; symbol-space category/reconstruction stacks, `_last_svo`, `_sentence_completed` | idea stack, parse scratch | fresh per forward | per forward |
| symbol-space typed STM `_buffer`, `_category`, … | legacy stack (dormant) | grows only | position |
| taxonomy `_priming` [B·K, cap] | symbolic heat (off by default) | keeps overlapping rows on resize; cleared only by soft reset | position |
| attention meters, poles, words, reads | bracket reading, work budget | rebuilt per batch/forward | per forward; `_last_gist` pools **all rows** |
| `_sentence_sources`, `_document_position`, `_address_sources` | (document, sentence) per row | overwritten per batch | document |
| `_word_reference_*` [B,W] | word references | overwritten on commit; a `[b]` read without a shape check | position; never reset |
| `_closing_images`, `_open_thought_rows`, `_pending_thought_credit`, `_what_recall_history`, `_what_recall_sentence_history` | row → image / open question / held credit / past sentences | **not resized**; rows at or beyond the new `B` persist; the sentence history is read for past questions after a document reset | position (two with a document check) |
| IndependentComponents population; MereologicalCodes context; `when_time` | over all LTM rows; the clock | snapshot per forward; global | global |
| **LTM** `index_stream` | the row's batch position at write | — | **position**: `cued_rows` skips rows whose `index_stream` is not the reader's row and links "contiguous" rows by equal `index_stream`; so a later document at row `b` retrieves earlier documents' rows written from row `b`, and a document shown again at another row cannot see its own earlier rows; only one reader compares `document_keys`; `unfold_idea` uses row 0's priming for every stream |

## 2. Alec's response (2026-10-08) and the division

Alec: "Document owned state" is a high-level conversation first. The LTM's
truth is **shared**: write identifiers that separate rows (document key,
address, timestamp — 4a-0) and let any batch query across all batches. "In
general, the less we manage explicitly, the better": is explicit context
management necessary, or can context learning and batch differentiation be
forced as required learning stages?

The audit's items divide in two. **Mechanism**, explicit and small: which
document each batch row is reading (the loader's fact); the work meter; the
credit held between the two trials; per-forward scratch. **Context**, which
is content and should be learned rather than managed: which document I am
in, what is recent, what is primed, what is open, what is expected. The
only repairs the math-chain failure requires are the two defects that are
not management at all: the LTM filter that hides rows by the batch position
they were written from (retired — identifiers separate, queries span), and
the priming surface's stale shape and silent broadcast (a mismatch raises).

## 3. Required learning stages (the replacement for explicit context management)

A corpus whose questions cannot be answered unless context is kept, kept
apart, or recalled forces the model to learn context as content. Proposed
stages, each a gate corpus:

1. **One document at a time** — today's gates; context trivially right.
2. **Batch differentiation** — several documents in parallel sharing
   vocabulary (`x is three` in one, `x is seven` in another), each question
   answerable only from its own document. Forces the situation code in
   content (the 2026-10-03 decision for 5.5) to be used: streams separated
   by what the model knows, not by row.
3. **Return after interruption** — a document split across batches with
   another document between; its second half asks about its first. Forces
   the situation to be recalled from LTM by its code: object permanence at
   the document level ("identity is an expectation").
4. **Shared truth across documents** — a fact stated in document A needed
   in document B. Forces the model to learn which rows are situated (a
   variable's value, a referent) and which absolute (`two plus one is
   three`): the two truths decided by credit, not by a flag. Stages 2 and 4
   pull against each other — separate referents, share facts.
5. **Priming as a prior, not a ledger** — a word whose referent differs
   between two documents; retrieval must favour the current situation's.
   Forces what is primed to be what the situation predicts, retiring the
   managed per-row energy surface.
6. **Expectation from the right history** — the next-sentence predictor fed
   from the current document's rows, selected by the situation rather than
   a per-row ring; stage 3's corpus tests it.

Open for Alec: which stages are required, and their place in the sequence
(before item 0, as gates the full training must pass).

## 4. The attention this requires, read through QKV (Alec's question, 2026-10-08)

Alec: the stages need an attention that attends to the word being processed
and to the document it is found in — a nonlinear mask over all of `.where`,
multi-head generalized to a mask governing which parts, wholes and symbols
are admitted into conceptual space at one time. How close are we, and what
does the QKV analogy illuminate?

Mapping: `K` = our content keys, exact by construction (forms, meanings,
sentence identity codes) plus the address bands; `V` = rows and symbols
with two-laned evidence; `Q` = implicit and unlearned today (the pooled
bracket key, the question's content, the gist); the softmax mask = the
walk's chooser scorer, already a learned `QK` scorer but committing one
hard choice over one set; the heads = present by construction and not
unified (the walk over the input, priming over the store, the LTM query,
the expectation's frames); the residual stream = conceptual space, under a
budget.

Missing: one admission step per word with several queries — this bracket,
this sentence, this document's rows, the primed concepts, the open
reference's need — scored at once, admitted softly within the budget, the
written derivation still committed hard (a soft mask committed hard, as
2e's reader; attention is a mask, not a derivation). What the analogy adds:
keys with structure make some heads exact — containment on `.when` ("in
this document") and `.where` ("this word") need no learning — so the
nonlinear mask is the lattice of structural masks intersected with learned
relevance masks, lane by lane; only relevance (the `Q` maps) is learned.
Distance: the scorer, the candidate sets and the lanes exist; the
unification does not — a design round and an implementation round, after
the thinking gate shows a chain, with stages 2, 3 and 5 as its gates.

## 5. The mask (Alec, 2026-10-08)

Alec: attention masks on `.where` as a precursor to conceptual processing;
mask-training rounds join the learning stages; not QKV — a multiplicative
mask that zeros the output, so the loss gradient is the importance signal;
the machinery behind it is likely a two- or three-layer MLP; situationally
a mask over all percepts, symbols included.

**Stage 7, mask training**: corpora whose answer depends on admitting a
particular word and its document — the XOR pairs among filler words
(the corpus of the withdrawn filter, now in its place), the math chain's
`what is y ?` needing `y`'s premise and the document's facts, stage 2's
documents.

**Machinery.** Per candidate the mask sees the current need (the open
reference's slot, the expectation image, the gist), the candidate's key
(form, meaning) and its structural relations (in this word / sentence /
document by the bands; recency; priming energy as salience); two or three
layers over the concatenation suffice — Bahdanau's additive attention is a
one-hidden-layer scorer, dot-product QKV its bilinear special case. Several
needs give several masks, combined lane by lane; divisive normalization
over the pool keeps the budget; top-k under the budget commits hard, soft
weights train; `∂L/∂mᵢ` is importance. The candidate set is prefiltered by
the exact structural masks. Caution (filter toy): start from attend-all
with a small budget, or the mask collapses to nothing.

**Psychology**: Broadbent's early filter and Treisman's attenuation (a
graded multiplicative mask); Lavie's load theory (early when the field is
wide — the budget); biased competition (Desimone & Duncan) implemented as
multiplicative gain with divisive normalization (Reynolds & Heeger);
feature integration (Treisman & Gelade: features parallel, binding needs
the spotlight on location; illusory conjunctions = the superposition
catastrophe); the spotlight/zoom lens (Posner; Eriksen & St. James) = the
walk; one attention over external and internal representations (Chun,
Golomb & Turk-Browne; Gazzaley & Nobre) = percepts and symbols under one
mask; the global workspace (Baars; Dehaene) = what is admitted at once.

## 6. The architecture of the attention (Alec's question, 2026-10-09; the opening of item 6.1)

Alec: "it should operate on `.where`, and take as input symbolic/perceptual
activation, after the spreading activation. It probably needs access to
STM, which itself may not be well defined."

1. **The field.** Every located item — percepts with their `.where`
   brackets, symbols (identity rows at their occurrence; the sign in
   perceptual space, the concept in conceptual space, decided
   2026-10-04), LTM rows with `.where`/`.when` — with its activation
   **after the spreading activation** (priming diffused over the store's
   edges). That spread is bottom-up salience and is the mask's input.
2. **The needs.** Query vectors from the current state: the open
   reference's slot (the region with its free variable), the expectation
   image, the gist (the situation code). Several; the one piece that does
   not exist yet (today the only need is the walk's "read the next thing").
3. **Keys and structure.** An item's code plus its relations to the
   current position: same word / sentence / document by the bands,
   recency on `.when`, distance on `.where`; the band relations are exact
   masks, unlearned.
4. **The scorer.** Per item and need, a two- or three-layer MLP over
   (need, key, structure) — Bahdanau's additive form; the chooser's
   scorer already has this shape over compose candidates.
5. **Combination and normalization.** Several needs → several masks,
   combined lane by lane with the admitting need retained (it routes the
   item); divisive normalization over the pool keeps the budget; top-k
   commits hard in the forward, soft weights stay in the graph.
6. **Output = STM.** The admitted items with `mask × spread activation`.
   STM is ill-defined today because five holders exist — compose's K-slot
   window, the episode's serial results, the reference bank of recent
   frames, concept space's `_live_*` stack, the priming surface. Defined
   as the mask's output, they are its views: compose's window = the
   newest admitted items in `.where` order; the serial results = items
   admitted by the open reference's need; the reference bank = the
   admitted rows. Cowan's embedded-processes model: the field after
   spreading is activated memory, the admitted subset is the focus of
   attention (4 ± 1 = the budget); nothing else. Composed rows enter the
   field as new located items and are masked like everything else.
7. **Training.** The loss gradient with respect to an item's soft mask
   weight is its importance; no mixing of unadmitted items into the
   forward; the departure can land on the mask as the narrowing walk
   already allows. Start from attend-all with a small budget (the filter
   toy's collapse).

The 6.8 narrowing walk is this mask with one need, over percepts only,
serial; 6.1 keeps it as the perceptual need and adds the others over
symbols and rows in parallel — one mechanism, several needs (§4).

Open for the spec and its toy: how the gist is formed (5.5's situation
code, or the mean of admitted rows); whether an item can serve two needs
at once; one budget or one per need.

### 6.1 Seven slots, each a `.where` and a `.when` (Alec, 2026-10-09)

Alec: "there are seven attention outputs, each of which define a `.where`
and a `.when`: so attention is both a spatial and temporal filter. This
allows a memory traversal … although presumably we want to train the
mind to be mostly present and focused on one word at a time."

Psychologically well founded, on three counts: Pylyshyn's FINSTs — a
small fixed number of movable pointers to locations that attention
operates through; Oberauer's three levels — activated long-term memory
(the field after spreading), a region of direct access of a few bound
items (the slots; Miller's 7 ± 2 with chunking, Cowan's 4 ± 1 without),
and a focus of one item (the present word); and attention to memory —
internal attention uses external attention's selection (Chun, Golomb &
Turk-Browne), retrieval is attention directed at internal representations
top-down or bottom-up (Cabeza's AtoM), episodic memory is indexed by
where and when (Tulving; Eichenbaum's time cells), and remembering and
anticipating are one machinery pointed backward and forward (Schacter &
Addis).

What it does to §6: the attention's outputs are **K slots** (K a
parameter, 4–7), each a bracket in `.where` × `.when` — both already
endpoint-sum brackets on every item — with the items inside admitted at
their spread activation. The slots are STM; one is the **focus** on the
present word, advanced by the walk; the others hold the sentence's
recent items, the gist, and what the needs retrieve. A thought operation
is a slot movement (`query` moves a slot to the best match;
`descend`/`return` are the path; `conclude` returns the focus to the
present) — thinking is attention traversing memory and the trace is the
slots' path, which the `descend`/`return` records already are.
Expectation is a slot at `.when` = next. Retrieval by time and place joins
retrieval by content (the postings); across documents a slot's `.when`
needs the situation code (stage 3). "Mostly present" is the default and
the cheap policy: reading moves only the focus; traversal is charged as
work, so the budget keeps the mind present unless a need pays for a trip.

Cautions for the spec: a slot admits a region's items, not one item, so
there are two budgets (slots; items admitted) — the toy decides whether
the second is needed; and K must be allowed to be 4 as well as 7.

### 6.2 Fixed outputs and nonlinear hidden processing (Alec, 2026-10-09)

Alec, in the item 6.1 implementation dialogue:

> 7 .where/.when output pairs, and the inputs as described, determine the
> architecture. Except for the width and number of hidden layers. I would
> suggest using a nonlinear layer

This supersedes the variable slot count and proposed separate admitted-item
limit above. The architecture has seven spatial/temporal region outputs,
with the already described perceptual and symbolic activations after
spreading, current needs, keys and structural relations as inputs. The
regions determine the masks; their count is not a count of individual
items inside them. The existing work accounting for traversal remains.

Implementation starting point: one hidden layer of 64 tanh units and a
readout for the seven `.where`/`.when` pairs. Hidden width and depth remain
configuration choices to measure; seven output pairs is the fixed contract.
Alec clarified the layer choice in the same dialogue: an ordinary MLP is
intended; standard PyTorch layers may replace `SigmaLayer(nonlinear=True)`
where appropriate. Use `nn.Linear → nn.Tanh → nn.Linear` here, without
Sigma's reversible `atanh` entry chart.
The MLP is wrapped by `SpacetimeAttention(Layer)`, as Alec requested in the
same dialogue. It uses the repository's parameter collection and lifecycle
interface, with 28 scalar outputs arranged into seven spatial/temporal pairs.

### 6.3 Current conceptual space and future episodic fields (Alec, 2026-10-09)

Alec, on coordinates for the attention outputs:

> It makes sense to give conceptual space, as it exists in the current
> moment, a .where (in addition to the .where used by the codebook).
> Epispodic memory over LTM would carry a bank of .where, so allowing the
> full range would also fot long-term design (you can add this to future
> work in an episodic memory section).

Alec clarified the location unit in the same dialogue: conceptual space
is the **eight-space**, whose contents correspond to one or three slots
when stored in LTM. Its `.where` sinusoids uniquely index the eight live
positions in exactly the same way that the existing perceptual codebooks
are indexed. This is an address of a slot in the current conceptual space;
its contents can change while the address stays fixed. Concept identity
keeps its existing codebook address. There is no unresolved choice between
allocating locations by word mention and allocating them by concept.

The seven attention output pairs and the eight conceptual positions are
different counts. The outputs can range across the complete shared
`.where` allocation, including the conceptual eight-space. Episodic LTM
will carry a bank of these located fields, selected in space and time by
the same attention architecture. That bank is recorded in
[Future Work §7](../FutureWork.md#7-episodic-memory-a-few-active-indices-beside-each-row);
it is not implemented by assigning a new spatial location to each existing
semantic LTM row.

**Implementation status, 2026-10-09.** `SpacetimeAttention(Layer)` is wired
through `ModelSpacetime` to the native perceptual evidence, symbolic words,
conceptual slots, reference bank and bounded memory reads. The canonical
sentence transaction computes one set of regions from each trial's current
pre-word state, and its readers share that set. Exact membership preserves
both evidence lanes; overlapping regions admit a value once. Read-side
features are detached, while the soft boundary derivative reaches the MLP.
The initial ownership choice is reconstruction plus answer credit, with a
dedicated attention owner and a parameter-only answer cotangent; expectation
retains its existing owner. This choice awaits review, not a claim that the
new mask has learned context.

Batch repairs cover row-local gist, both recall histories, word-reference
lifecycle and shape checks, and taxonomy priming resize. `WhereRegistry`
includes the conceptual slot range, and `ConceptualSpace.where` derives its
sinusoidal indices from the shared encoding. The current MLP reads the full
three-role expectation image. The gist remains the row's open-field mean;
the region views do not themselves replace the chronological prediction
windows or establish a learned situation code.

The [6.1 handoff](../benchmarks/2026-10-09-item6-1/README.md) distinguishes
mechanism checks, source-matched validation and the unproven learning stages.
Item 6.1 remains open pending its exit criteria and Claude's review. Section
6.4 below remains a separate, unaccepted design discussion.

### 6.4 Is anything expected outside attention? (Alec's question, 2026-10-09; Claude's reading, pending Alec's decision)

Alec: "Do we predict/expect things that are outside of attention? Or is it
possible that they are the same mechanism: we have expectation only where we
are attending, so that the mask and prediction together cover all possible
attended space?"

Reading: for **content**, the same mechanism — a region (one of the seven
`.where` × `.when` outputs, §6.2) is the unit of both attention and
expectation: its bracket, the items it admits at their spread activation,
an expectation image for the bracket, and the surprise (the negative image
applied to what it admits). Nothing outside the regions carries an image or
produces surprise. Two things necessarily live outside, and neither is an
image: the **unlocated prior** — priming over concepts (`[B, V]`, the
spreading activation), content without a place — and the mask's own
**relevance logit**, one scalar per field item, parallel and one layer.
This is Treisman's feature integration applied to expectation: pre-attentive
features float free of location and attention binds them to a place; here
priming is expectation free of location, and a region is where expectation
acquires a `.where` × `.when` — and so, only there, can be wrong.

What exists agrees in *what* is predicted and disagrees in *from what*:
`word_distribution` predicts the next word of the sentence being read and
`SentenceExpectation` the next idea at the closing — both at the focus,
nothing unattended — but each conditions on a positional window (ARMA over
the last `p` words and `q` residuals; `context_window` lagged meanings).
Stage 6 (expectation from the right history) becomes: the predictors read
the regions, not a lag window.

Psychology, both sides: predictive coding with attention as the precision on
prediction errors (Feldman & Friston 2010) — predict everywhere, count errors
only where attended; here taken one step further: outside the regions the
error is not computed at all. Against total unification, the mismatch
negativity (Näätänen) — an unattended deviant still produces a mismatch
response — so a coarse pre-attentive prediction exists; here that is
priming, and a deviant is an unprimed concept arriving. Kok, Rahnev, Jehee,
Lau & de Lange 2012: prediction suppresses the unattended and sharpens the
attended — expectation and attention dissociable but multiplicative, which
is `mask × (observed − expected)`. Reading: E-Z Reader (Reichle, Pollatsek,
Fisher & Rayner) — serial attention one word at a time, parafoveal preview
of the next, predictable words skipped (never fixated; the prediction
stands in for perception). That is the sense in which mask and prediction
together cover the input: every item is either admitted (composed, surprise
computed) or assumed (the expectation stands, nothing composed). Skipping
as a cost saving belongs to item 1, not 6.1.

Consequences for 6.1 and 4.5 if accepted: 4.5's levels are regions by
extent (word = the focus; sentence; document = the gist; `.when` = next), so
"expectation at every level" is an image in every region; 6.1 gives a
region its bracket and admission and makes the predictors read the regions;
4.5 puts the image in it. The image reaches the mask through priming (4.5
§4.2): the expected is easier to admit — perceptual set, and confirmation
bias; the bottom-up route is what keeps it honest.

The risk is the filter toy's: expectation only where attending, with a
learned mask, means the excluded get no gradient. Guards: regions start
covering the field (`SpacetimeAttention`'s initial regions do); the focus is
always on the present word, and text is serial, so nothing is permanently
unseen — "outside attention" here means memory and the coarser levels;
departures on the mask. **Open (Alec):** whether novelty — an arriving
concept with low priming — is a salience term into the mask (abrupt onsets
capture attention, Yantis & Jonides; the unexpected gorilla goes unnoticed,
Mack & Rock, Simons & Chabris — both are real).

**Literature answer (2026-10-09; report
[doc/research/reports/Expectation outside attention.md](../research/reports/Expectation%20outside%20attention.md),
notes beside it; every citation verified).** Expectation gets *deep* at the
edge of attention rather than stopping there. (a) Low-level located priors
(where, when, form, first-order transition) run field-wide and their error
is registered without attention, scaled by attention as a gain with a floor
(Kok 2012; Ekman 2017; MMN under reading/Tetris, Paavilainen & Ilola 2024;
Bekinschtein 2009 local). (b) Cross-level identity/next-item expectation
leaves no sensory error signature outside attention (Richter & de Lange
2019, preregistered, BF10 0.18–0.25; Bekinschtein 2009 global; lexical
tracking only for the attended talker, Brodbeck 2018; nothing lexical from
n+2 in reading). (c) Orienting to surprise is a third system: fires when
(b)'s signature is absent, driven by violation of a learned prior, not
unfamiliarity (Vachon, Hughes & Jones 2012: no capture before a rule exists,
d < 0.11; capture at its first violation, d ≈ 0.9–1.4; habituates), set-match
dominates the switch (Most 2005; Simons & Chabris 1999), ~400 ms latency,
stochastic (30–50 % misses). (d) Regularities are learned only over selected
content, but exogenous selection suffices (Duncan, van Moorselaar & Theeuwes
2025). So: the region as the unit of expectation holds for every level that
carries comprehension; one cheap field-wide transition image lies beneath
the regions, its residual the salience candidate; novelty enters region
*placement*, as violation of the learned local prior (prediction gain, not
raw residual energy — the noisy-TV pathology), subordinate to priming and
set-match. Low priming alone is not a capture signal; low priming plus a
violated local prior is. Pending Alec's decision.

## 7. The training paradigm: isolation through the bottleneck, coverage at the floor (Alec, 2026-10-09)

The first native run collapsed
([receipt in progress](../benchmarks/2026-10-09-item6-1/README.md#current-blocking-training-defect)):
with the mask enabled, 0/12 eligible symbolic-word reads and 0/384
generation-field candidates admitted, MSE equal to the disabled control,
readbacks 0/4; once the field was empty the reconstruction and answer
objectives had exactly zero gradient to the MLP (the identity
reconstruction audit is hard; the downstream hard reading decisions remove
the missing content's numerical path). Codex put a binary choice — a
loss-side differentiable surrogate, or sampled region movements credited by
actual cost — and installed neither. Alec's answer is a change of
objective, not a choice between credit mechanisms.

### 7.1 The reason for attention is the importance of given objects

Alec: "If attention does not isolate its object, then conceptual
reconstruction will fail; the concepts will try to represent the entire
field." Attention's job here is **isolation**: the region's output is what
the concept is about, and the conceptual eight-space holds one object's
worth. A glimpse-model objective (reconstruct the field, pay per unit of
attention) misses this: under it attend-all is optimal unless the budget is
tight, and attention has no meaning beyond coverage. The limit on the
complexity of reconstruction is what gives the mask its meaning, and it is
two-sided: a region too wide asks the concept to represent the field and the
object's reconstruction fails, so the boundary gradient pushes items out; a
region too narrow leaves the object incomplete, so it pulls items in. The
fixed point is the object's extent.

### 7.2 The objective

Alec: "scoring the whole field is fine. But if there is a lesser penalty
for the unattended field, and a limit to the complexity of reconstruction,
then the mask plus concept pair will be the optimal at that output
construction." That is 4.5 spec §4.3's `g·(o + n)` read as a loss:

    L = Σ_i g_i · |o_i + n_i|  +  c · iterations
    g_i = 1 inside the regions;  g_i = attentionFloor outside

- **Inside a region** the concept must reconstruct what the region admits;
  its limited complexity caps how much it can hold — the isolating pressure
  (the existing differentiable reconstruction path over admitted content
  carries it; this is the path that drove the collapse when it was the
  only term).
- **Outside**, expectation stands in for perception and the residual still
  counts, at the floor — the coverage pressure, graded. The term is
  differentiable in the admission weight: an item's unattended cost
  `attentionFloor·|o_i + n_i|` pulls the boundary toward it through the
  straight-through edge, in proportion to its surprise — importance.
- **The floor is on the cost, not on the forward.** No unadmitted value
  enters compose; the tree stays a tree; the identity audit stays the
  reported score.
- **`c` per serial-word-loop iteration** is the budget on time (the work
  meter), and it rewards admitting the largest unit the concept can hold —
  a word, later a phrase. Words are the largest percepts (6.8); larger
  wholes are conceptual.

The floor is the dial the run lacked: at 0 exclusion is free (the collapse
observed); at 1 attention is coverage only, time order without importance;
between, the optimum attends the items whose floor-weighted surprise is
largest per unit of concept capacity and iterates while the marginal gain
beats `c`. Until 4.5's image lands, `n = 0`: the unattended cost is
`attentionFloor·|o_i|`, uniform pressure to cover with no importance yet,
which the first stages do not need.

### 7.3 Supervision names the part, never the attention

Alec leans to a supervised paradigm asking for reconstruction of parts of
the sentence: single-word testing; reconstruction of a sentence where words
are the largest percepts and each serial-word-loop iteration costs.
"Reconstruct the second word" labels the **target**; isolation of the region
is forced through the bottleneck. Attention is never labelled — context
stays content to be learned (§2).

Stages, each a gate, measured with the mask enabled against disabled under
the same unseeded single-run protocol as the receipt so far:

1. **A one-word field** — the capacity check; attend-all is already the word.
2. **Two words, reconstruct each in turn** — the first isolation lesson:
   the concept holds one, so the focus region must isolate one per
   iteration and move.
3. **The sentence, with `c` per iteration** — coverage under the time budget.
4. **Asked parts** — "the second word", then a phrase.

The focus region does this work; the other six wait for the question
stages (§3). Report per stage: eligible reads admitted, region extents per
iteration against word extents, readbacks, the iteration count.

### 7.4 Start wide and shrink; recovery

Under this objective the collapse reverses direction: from attend-all the
bottleneck pushes boundary items out one at a time with a dense gradient
the whole way, and the floor term holds the equilibrium at the object's
extent with the next object just outside, within `softness` of the edge.
The straight-through edge suffices; sampled region movements in the
departure learner are backup only, for a part the region does not touch,
and whether they are ever needed is reported, not assumed. The collapsed
run is re-initialised from attend-all under the new objective, not
recovered. `hetTolerance` (4.5 spec §4.4) is declared at 1, a no-op, where
the lanes enter the mask input.

### 7.5 Hand-off to Codex (interrupted work resumes here)

> **6.1 resumes with a change of objective, not a choice between your two
> credit mechanisms** (stream-state plan §7; Alec, 2026-10-09). Keep the
> seven-region MLP, the exact-containment forward, the straight-through
> edges, the batch repairs and the native wiring as they stand. Score the
> **whole field**: inside the regions the concept reconstructs what it
> admits through the existing differentiable path (its limited complexity
> is the isolating pressure); outside, every item's residual `|o_i + n_i|`
> is charged at a new model.xml parameter **`attentionFloor`** (0 < floor < 1;
> `n = 0` until 4.5's image lands), on the **cost only** — no unadmitted
> value enters the forward, and the identity audit stays the reported score
> — so an excluded item pulls the boundary toward it through the edge in
> proportion to its surprise. Charge the existing work meter **per
> serial-word-loop iteration**. Install no loss-side surrogate; leave the
> departure learner as backup for a part no region touches, and report
> whether it was ever needed rather than assuming it. Declare
> `hetTolerance = 1` (4.5 spec §4.4) at the lane read into the mask input,
> a no-op now. **Re-initialise from attend-all**; do not try to recover the
> collapsed parameters. Then run the staged gates of §7.3 — one-word field;
> two words reconstructed in turn; the sentence with the per-iteration cost;
> asked parts ("the second word", a phrase) — mask enabled against disabled,
> same unseeded single-run protocol as your receipt, reporting per stage the
> eligible reads admitted, region extents per iteration against word
> extents, readbacks and iteration count; supervision names the target part,
> never the attention. Append the outcomes to the 6.1 receipt with the
> collapse record retained as it is. Publish sequence unchanged: no commit
> before Claude's review of the source-matched receipt.

### 7.6 Executable objective and traversal (Alec accepted Codex's amendments, 2026-10-09)

The native identity audit has no useful reconstruction derivative, and the
previous driver supplied a preselected word at every fixed iteration. Alec
accepted making the admitted-content error explicit, allowing region-driven
progression/stopping, and adding an actual region movement to the departure
learner. This does not authorize a replacement reconstruction backward.

The candidate now costs the actual free decoder's numerical leaves against
immutable native observations. Emitted leaves are aligned **after** decoding;
missing admitted words and excess emitted leaves count. The outside term is
`attentionFloor * (1 - admission) * abs(observation)` until 4.5 supplies an
image. Admission is the exact union of the focus reads. The named training
term uses Error's existing relative normalization by the complete observation,
independent of admission. The work term uses the existing `WHAT_STEP_COST`
of .01, and both terms retain the configured reconstruction priority. The
hard identity audit remains a separate reported measure. Part lessons add
the actual decoded-word error against the supplied target words; targets
never select masks, bounds, actions, decoder length, or stopping.

Each focus placement sees the full native candidate field, with already read
items marked unavailable for another read. It can admit multiple words in
one read; the native grammar still processes those admitted words in their
source order. An empty read stops that row, the native sentence end closes
it, and the remaining shared work allowance bounds attempts. The meter
charges each executed field iteration, including a terminating empty read,
once for the kept trial. Bracket work remains charged separately. The
iteration count is **not** a claim that a multiword unit takes one primitive
grammar operation; the admitted-word records also expose those visits.

The focus uses the entire observed field's bounds, including excluded items,
and a soft-edge width of two percent of each field extent. Stored `.when`
continues to address sentences. The live read subdivides that sentence
interval into word extents so repeated words can be distinguished. The
other six region outputs retain the shared coordinate range for memory
reads. The ordinary MLP still has exactly 28 outputs and no additional stop
head. `hetTolerance=1` enters the paired symbolic-activation read as a no-op.

A blocked focus makes a region departure available in the second sentence
trial. It uniformly samples an unread observed item and proposes a focus
translation that reaches it, without consulting target words. The existing
paired-cost departure learner credits that action; the sampled action has
no fabricated pathwise derivative. The receipt reports blocked rows,
attempted movements and movements kept, separately. Fresh attend-all
initialization remains mandatory for the campaign.

The part-lesson prompt enters the open-need input through the existing
native form keys of its words. This is a content cue, not an implemented
ordinal parser. Learning what “second word” or “middle phrase” means is
therefore still a measured gate. The declared campaign uses one unseeded
model per mode, 64 epochs at each of the four sequential stages, no retries
or selected checkpoints. Later stages are diagnostics if a prerequisite
fails. A full source-matched receipt and Claude review still precede commit.

### 7.7 Review of the candidate (Claude, 2026-10-09) — not accepted

Reviewed: the uncommitted tree (35 modified files, 8 new modules, 8 new
test files), the [receipt](../benchmarks/2026-10-09-item6-1/README.md), its
JSON results, and the source-matched sweep (`output/item6-1-review-final-full`:
5,717 cases, 5,425 passed, 286 skipped, 5 failed, 1 non-strict XPASS). The
receipt is complete and honest: both collapses, every partial sweep, every
failure retained; nothing seeded, relaxed or bypassed. The verdict is not an
execution verdict — most of what fails is the design we handed over.

**A. Built as decided.** Seven `.where`/`.when` region pairs from one
`Linear → Tanh → Linear` MLP; exact-containment forward with straight-through
sigmoid edges (2 % of the field extent); the Boolean union counts an item
once; attend-all initialisation; `attentionFloor = .1` and `hetTolerance = 1`
in XML and schema; `tolerate_heterogeneity` is §4.4's op exactly
(`lanes − (1 − τ)·min`); `field_cost` is §7.2's objective with the floor on the
cost only and the identity audit reported separately; the conceptual
eight-space has fixed sinusoidal slot addresses (§6.3); the batch repairs
(gist, recall histories, word-reference shapes, taxonomy priming) are in
and tested; the departure samples unread items without consulting targets.

**B. Regressions of 6.2's closing certificates — blocking.**
`test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`
(certificate c) and `test_forced_ordinary_bound_declaratives_open_no_episode`
(d) pass at HEAD `ed0d031bd` with Codex's own `eager_reading` fixture copied
in (3 passed, 27 s) and fail deterministically on the candidate (2 failed,
1 passed, 30 s; the sweep agrees). `test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage`
(g) failed in the sweep and passed in isolation here — intermittent. The
receipt's own notes point at the cause: `What.past` answers now pass
through the region reader, and forced declaratives open episodes. The 6.2
demonstration is a standing certificate; it must pass before any landing.

**C. Order-dependent parity failure.**
`test_real_packed_ends_train_before_the_next_sentence[False]` failed in
the sweep (64/312 role entries differ, max .16) and passes at HEAD, alone
on the candidate (3/3) and with its module (7 passed) — state is leaking
between tests in a worker: the new per-sentence holders
(`_sentence_field`, `_last_sentence_field`, `_attention_part_lesson`,
the meters) or unreset priming are the suspects. Reproduce with the sweep's
worker order from the retained logs.

**D. Pre-existing flaky test.** `test_xor_router_gradients_reach_all_three_ops`:
file unchanged, standalone router, unseeded `randn` routing; 5/5 passes in
isolation. An op the random routing never selects gets no gradient, so the
assertion is flaky by construction; recorded under operators plan §20. Fix
the test by construction (inputs routed through every op), never by a seed.

**E. The learning result is the design's, not Codex's.** The run saturated
open (temporal focus 3–5× the word interval, 16/16 admitted, edge
derivatives exactly zero after stage 1). §7.6's accepted amendment — "a read
can admit multiple words; the grammar processes them in order" — removed
the complexity limit that §7.1–§7.2 rely on: the grammar reconstructs any
number of admitted words, so nothing is lost by admitting all of them. With
no bottleneck the objective has no interior optimum: for a reconstructed
item `∂L/∂m = −floor·|o|` (expand, with unbounded radii), for an
unreconstructed one `(1 − floor)·|o|` (exclude). Codex measured the optimum
of the objective as written; §7.5 asserted a bottleneck the driver does not
have. (Codex's "total derivative does not guarantee attraction to an
unreconstructed item" is the same point from the other side: the
straight-through linearisation sees `r = 0` for an unread item.)

**F. The stages do not yet measure attention.** Stage 2 as run is a
prompt-cued part lesson ("first word"/"second word" in separate rows), not
two reads in turn; all 8 words were admitted in one iteration in both
modes with inside cost 0 — the field was reconstructed perfectly, the
target was not isolated. Stage 3's decoder reconstructs 2 of 4 words in
*both* modes (inside cost 53–129): it measures the decoder, not attention.
Every row used one field iteration; the `serial-word-loop` meter counts
placements, not word reads; no row was blocked, so the departure never
fired. Stage 1 (4/4 both modes) is the only informative gate so far.

**G. Production limitation to repair.** Field iterations draw on the same
32-unit allowance as bracket work; a 16-word fixture spent it all on
brackets and got zero reads (the fixture was raised to 48; production is
unchanged). Reading must not be blocked by its own bracket work: the field
needs its own allowance, or the per-read cost stays in the loss only.

### 7.8 Design corrections (Claude's recommendation; Alec to decide)

1. **One read per placement.** `k = 1` at the word stage — §5's "top-k under
   the budget" with the budget at one object. The read is the admitted item
   nearest the focus centre (ties in source order); it is one grammar step
   and one meter unit. Items inside the focus but unread are charged, per
   iteration, as admitted and unreconstructed: the concept could not hold
   them. A two-word focus then pays `(1 − floor)·|o|` for the word it did
   not read, and shrinks; the isolating pressure of §7.1 becomes real.
2. **Clamp the straight-through derivative to the flippable direction.**
   An admitted item carries only the exclude direction (`∂L/∂m > 0`), an
   excluded item only the include direction (`∂L/∂m < 0`). A derivative
   asking for more admission of an item already admitted is an artefact of
   the surrogate; removing it removes the unbounded expansion.
3. **The focus's default centre is the next unread item in `.when` order**
   — the serial walk, structural, no learning (§4: structural masks are
   exact; §6.1: reading moves only the focus, "mostly present"). The MLP
   supplies extent and offset; a non-default move is a departure, credited
   by the paired cost through the existing learner — expected to fire now,
   and reported.
4. **Stages that measure attention.** Stage 2: two words, no prompt, two
   reads; passes when both are decoded, one per read, with the focus's
   extent at most one word at each read; control = source-order serial
   reads. Stage 3: the control must pass the sentence readback before the
   stage is read as an attention result — the four-word decoder failure is
   a separate defect to fix or the stage shortens to three words. Stage 4
   after 3. Report per read: the item read, the focus extent, the decoded
   word, the cost terms.
5. **Repairs before any landing:** B (certificates c, d, g green at the
   candidate), C (the leak found and reset), D (the test by construction),
   G (the allowance).

### 7.9 Alec on the corrections (2026-10-09)

**1. One read per placement — accepted, and it reshapes STM.** Alec: the
past placements are remembered, so STM is effectively seven; multi-head
attention was devised so that processing is aware of both a word and the
significant wholes that set its context, and that is still needed; "when we
process a conceptually-attended space, we see all of its parts and wholes
at the same time, so the seven previous just become the 8 slots that words
had been processed in."

Consequence: the MLP's learned output is the **focus alone** — one
`.where`/`.when` pair per read. The other seven regions are not re-placed
each step; they are the **previous placements, remembered** as the occupied
slots of the eight-space, each stamped with the `.where`/`.when` it was
read at. Conceptual processing operates over all eight at once — the
words just read and the wholes already composed from earlier ones — which
is multi-head attention's word-plus-context function obtained by the
window rather than by parallel heads. Retrieval is the focus placed into
the past, its content entering a slot as a read word does. This supersedes
§6.2's seven learned pairs (one learned pair; seven remembered); §6.3's
slot addresses stand.

**2. The derivative clamp — rejected.** Alec: clarify the problem; a
different solution will follow. The problem, stated:

The forward is hard — an item is in the focus or not. The boundary learns
through a surrogate: each item's weight is `hard + (soft − soft.detach())`,
`soft = σ(left/s)·σ(right/s)`, so the loss sees the hard value and the
boundary receives `∂L/∂w_i · ∂soft_i/∂edge`. Both factors mislead.

- *An admitted, well-reconstructed item.* `∂L/∂w_i = |o_i − r_i| − floor·|o_i|
  ≈ −floor·|o_i| < 0`: the loss asks for *more* admission of an item already
  fully admitted, because admission is what removed its floor charge. The
  surrogate has no way to say "already one"; `soft_i < 1` while the item is
  within a few `s` of an edge, so the edge moves away from it — and the
  radius is unbounded, so it keeps moving until the sigmoids saturate and
  the derivative is exactly zero. That is the measured saturation.
- *An excluded item.* `∂L/∂w_i` is evaluated with `r_i = 0` — nothing
  reconstructed it — so `= (1 − floor)·|o_i| > 0`: the loss asks for *less*
  admission of an item not admitted, because the surrogate cannot see that
  reading it would make `r_i ≈ o_i`. The counterfactual read is what the
  boundary needs and a pathwise derivative cannot supply.

What any solution must provide, then: a boundary signal that saturates when
the item is admitted (as the hard forward does), and that knows the
counterfactual for the excluded item — the actual cost difference between
reading it and leaving it at the floor. Under correction 1 the inward
signal exists (an unread item inside the focus is charged in full), so
the residue of the problem is the outward artefact along any axis with no
neighbour, and the excluded item's missing counterfactual.

**3–4.** Alec: resolved by 1. The focus reads one item; the remembered
placements are STM; the next read proceeds in `.when` order unless the
focus is placed elsewhere. One residue stays from 4 as a measurement
condition, not a design point: the disabled control must pass a stage's
readback before the stage is read as an attention result.

### 7.10 The mask is the concept's support (Alec, 2026-10-09; the solution to §7.9's problem 2)

Alec: "The way that an attended object appears to the mind is the negation
of the non-object. So an object has some positive and negative evidence
across parts and wholes; it is either a concept or a conceptual candidate.
And the parts that are not evidence whatsoever are zeroed out, leaving the
'significant' parts and wholes. So given that formulation, all concepts or
conceptual candidates have a mask, which is just where they have no
support."

Reading and consequences (Claude; pending Alec's confirmation):

- **The mask is derived, not learned as a boundary.** A candidate's mask
  over the field is its support: its two evidence lanes over the parts and
  wholes within its extent, exactly zero where it has no evidence either
  way. Negative evidence is support too — a part that counts against the
  candidate is significant and is admitted, which is biased competition
  between candidates. The extent is the candidate's whole, structural
  (6.8's word whole for a word; the composed whole for a phrase), stamped
  with the `.where`/`.when` it was found at; the significance within it is
  the lanes. `hetTolerance` acts on exactly these lanes.
- **Problem 2 dissolves.** There is no radius, no sigmoid edge, no
  straight-through surrogate; the mask's gradient is the lanes' gradient,
  ordinary, through reconstruction and the evidence objective. The outward
  artefact had a radius to run on; there is none. The excluded item's
  missing counterfactual becomes evidence learning: a part a candidate has
  no support for is perceived field-wide, costs its floor, and acquires
  support only by the candidate's lanes learning it — content-addressed,
  gated by the candidate having been entertained (the research's level (d)).
- **Placement becomes candidate selection.** Candidates arise bottom-up
  from the field (the symbolic activation after spreading: which concepts
  have support here; an unclaimed letter run is a conceptual candidate with
  provisional support) and are weighted top-down by need. One is entertained
  per read (§7.9 item 1); its support is the read's mask. This is §5's
  original per-candidate scorer — need, key, structural relations, hard
  commit, departures credited by the paired cost — not §6.2's region
  generator. The `SpacetimeAttention` MLP's inputs survive; its 28 outputs
  do not: it scores candidates. No legacy region path is kept.
- **STM.** The eight-space holds eight entertained candidates, each with its
  mask; conceptual processing sees all eight with their parts and wholes.
- **Unchanged:** the floor cost on the unsupported field; one read per
  placement; the serial-word loop as the default order; the stages.

### 7.11 Hand-off to Codex: the support mask (Alec confirmed §7.9–§7.10, 2026-10-09)

> **6.1 candidate: not accepted (stream-state plan §7.7); 6.1 is redirected
> (§7.9–§7.10, Alec, 2026-10-09).** The receipt is complete and honest and
> stays as it is; append to it. Four repairs come first, on the current
> source: (i) 6.2's closing certificates c and d
> (`test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`,
> `test_forced_ordinary_bound_declaratives_open_no_episode`) pass at HEAD
> `ed0d031bd` with your `eager_reading` fixture and fail deterministically on
> the candidate; g is intermittent — the answer and memory readers return to
> the landing's paths (retrieval as a candidate entertained into a slot is a
> later stage, not now); (ii) `test_real_packed_ends_train_before_the_next_sentence[False]`
> fails only in the sweep and passes alone and with its module — find the
> state leaking between tests (the per-sentence holders `_sentence_field`,
> `_last_sentence_field`, `_attention_part_lesson`, the meters, or unreset
> priming) and reset it; (iii) `test_xor_router_gradients_reach_all_three_ops`
> is flaky by construction (unseeded routing; 5/5 alone) — make it route
> through every op by construction, never by a seed, recorded under
> operators plan §20; (iv) field reads must not draw on the bracket-work
> allowance (the 16-word zero-read limitation): give them their own, with the
> per-read cost staying in the loss.
>
> **The mask is the candidate's support, not a learned region.** A read
> entertains one candidate — a concept, or a conceptual candidate (an
> unclaimed letter run, 6.8's word whole, with provisional support). Its
> mask over the field is its two evidence lanes over the parts and wholes
> within its structural extent, exactly zero where it has no evidence either
> way; negative evidence is admitted (it is significant); the extent is
> stamped with the `.where`/`.when` it was found at; `hetTolerance` acts on
> these lanes (no-op at 1). No radius, no sigmoid edge, no straight-through
> weight anywhere: the mask's only gradient is the lanes' ordinary gradient
> through reconstruction and the evidence objective. Retire
> `SpacetimeAttention`'s 28 region outputs, `read_regions`' soft edges and the
> region departure — no legacy path (no-legacy-code rule). The MLP keeps its
> inputs (activation after spreading, the needs, keys, structural relations)
> and becomes §5's per-candidate scorer: one score per candidate, hard
> commit, the alternative drawn by the existing paired-cost departure
> learner over candidates, no fabricated pathwise derivative through the
> choice. Candidates for a read are the unread ones with support in the
> field, bottom-up from the symbolic activation, weighted by need; the
> disabled control is source order.
>
> **One read per placement; STM is the eight-space.** Each read is one
> candidate, one grammar step, one `serial-word-loop` meter unit. The
> previous placements are not re-placed: they are the eight-space's occupied
> slots, each with its mask and stamp, and conceptual processing sees all
> eight — the words just read and the wholes composed from earlier ones.
> The objective keeps `field_cost`'s form: inside = the read candidate's
> reconstruction error over its support; outside = every never-read item at
> `attentionFloor` (`n = 0` until 4.5); work = `WHAT_STEP_COST` per read;
> floor on the cost only; the identity audit reported separately; decoder
> leaves aligned after decoding as now. Keep the batch repairs, the
> eight-space addresses (§6.3), the XML/schema parameters and
> `AttentionCredit`.
>
> **Stages, fresh attend-nothing initialisation of the scorer, enabled against
> disabled, the same unseeded single-run protocol:** 1, a one-word field;
> 2, two words, no prompt, two reads — passes when both are decoded, one per
> read, each read's support exactly one word; 3, the sentence — the disabled
> control must pass the readback before the stage is read as an attention
> result (diagnose the four-word decoder failure first; if it is a decoder
> limit, shorten to three words and record that); 4, asked parts, after 3.
> Report per read: the candidate entertained, its support size, the decoded
> word, the cost terms; per stage: departures attempted and kept. Reconcile
> `todo.md` 6.1 to §7.9–§7.10 (support mask; one read; eight-space STM;
> seven learned pairs superseded). Publish sequence unchanged: no commit
> before Claude's review of the source-matched receipt.

### 7.12 Review of the support-mask candidate (Claude, 2026-10-09) — not accepted; the mask worked, three other things failed

Reviewed: the receipt's redirect append, `support-stages.json`,
`support-reads.json`, the suite summary (5,706 cases: 5,415 passed, 286
skipped, 4 failed — c, d, two fixture adapters repaired after the freeze —
one non-strict XPASS), `CandidateAttention`, `ModelCandidateAttention`,
`AttentionTraversal.select`, `SentenceField`.

**What worked — the mechanism.** Every final read supports exactly one word
candidate in both modes; no saturation, no collapse; one read per placement
with one grammar step and one meter unit; the eight-space carries the
witnesses with their masks and stamps through pushes and operations; the
read allowance is independent of bracket work (15 reads at a budget fully
spent on brackets); `Queries.py` byte-identical to HEAD; the XOR gradient
certificate routes through every op by construction; g, both
pending-premise variants, the packed certificates (eager and compiled) and
the fullgraph check pass. The region machinery is gone from `bin/`.

**What failed.**

1. *Reading order was left to be learned, and the scorer learned content
   instead of position.* Stage 2 enabled, final reads (`support-reads.json`):
   in `red blue` it read `blue` first, in `green gold` it read `gold` first —
   the same words it read first in `blue red` and `gold green`. The
   objective did penalise the reversal (inside 16.66 per word against 0.00
   in the source-order row), so the signal was right; but 97 departures with
   26 kept over 64 epochs on four fields credited "blue first" in the rows
   where that happened to be source order, and the zero-initialised readout
   generalised it as a content preference. Reversed reads then composed
   reversed ideas, whose failed decodings trained the composer and decoder
   on garbage: row 3 read `gold green` in source order and still decoded
   `green`, `<unresolved>` (inside 25.0, 23.1). 1/4 against the control's
   4/4 (inside 0.00 everywhere). The structural feature is in the inputs
   (`time`), but nothing made the next unread item the default.
2. *The free decoder cannot unfold composed children.* The disabled control
   decodes exactly two leaves per row for three- and four-word sentences
   (`end_depth 1`, `decoded 2`, inside 25–84): "the free lexical binary
   inverse chooses its children from the primed word bank, which has no
   composed child states." The landing's reconstruction is the inversion
   along the actual derivation, not free decoding; the field cost chose free
   decoding and inherited production's gap (item 6). Stages 3 and 4
   therefore cannot measure attention in either mode.
3. *Certificates c and d still fail — and c fails with the scorer disabled
   too* (Codex's source-order diagnostic). So the traversal machinery itself,
   not the scorer, changed what the forced grammar receives: the per-read
   grammar step with masked payloads, the NULL-closing fork fix, or the
   zero-budget path. Blocking; 6.2's demonstration must pass unchanged.

**Recorded, not blocking:** the original region candidate's sweep-only
packed failure did not reproduce in eight unseeded replays of its exact
source and worker selection; the two fixture adapters (two lines, tests
only) were repaired after the freeze; one isolated repair run failed the
unseeded binding-distribution bounds (.946–1.107 against .9–1.1) and is
retained.

### 7.13 What to do next (Claude's recommendation; Alec to decide)

A. **Make the serial order structural.** The default read is the next
   unread candidate in `.when`; any other read is a *move*, charged as work
   (`moveCost`, a model.xml parameter) — the scorer must earn a departure
   from the present (§6.1: reading moves only the focus; traversal is
   charged). Stages 1–3 then hold in both modes by construction and test
   that the default is preserved; the first measurement of attention proper
   is stage 4 (asked parts) and the §3 question stages, where a move pays.
   Prediction for the toy: zero departures kept in stages 1–3.
B. **The field cost reconstructs along the actual derivation** — the
   landing's inversion given the co-operands (4.5 spec §3.1), per admitted
   word — not by free decoding. Free decoding stays production's path
   (item 6) and the identity audit. The four-word control then passes, and
   stages 3–4 can be read.
C. **Bisect c/d with the scorer disabled**: the forced grammar's input must
   be identical to the landing's in source order; candidates in order of
   suspicion — the masked per-read payloads, the closing-fork fix, the
   zero-budget path.
D. Then rerun stages 1–4 on the repaired source, same protocol; proceed to
   the question stages (§3) where attention has something to earn.

### 7.14 Alec's decisions on §7.13 (2026-10-09) and the hand-off

- **Order is taught, not wired** (a rejected): word order is not universal
  across languages; a structural default would leave a pattern attention
  keeps after training. The supervision is a **generated corpus** read in
  stages — identity and object permanence; identifying words; reading words
  in order; simple sentences of various kinds (identities, NP, NP + VP,
  relations) — now ordered with the later decided gates in
  [Training.md, "The generated supervised curriculum"](../Training.md#the-generated-supervised-curriculum-in-order-alec-2026-10-09).
  The order lesson (stage 3): the lesson names the part whose turn it is;
  cross-entropy on the candidate scorer toward that candidate; reads
  teacher-forced during the lesson; a content rule cannot fit the batch, so
  position is learned; afterwards free reading trains by the field cost and
  departures as now.
- **Invertibility is intact; free decoding is deferred, the regression is
  not permitted** (B accepted in substance): operations invert given one
  operand; along the actual derivation the co-operand is known and
  reconstruction is exact — the landing's `R`. Free decoding must recognise
  children against a bank that held only words, so a four-word root's
  phrase children had nothing to snap to. Primed words stay primed for the
  sentence (with decay), as Alec notes; what the free inverse lacks is STM's
  composed wholes, which the eight-space holds — the fix when item 6 takes
  free compound decoding up. The field cost returns to reconstruction along
  the derivation now.
- **c and d:** d follows attention (passes in source order); c fails with the
  scorer disabled and is bisected in the traversal.

> **6.1 support-mask candidate: not accepted (plan §7.12); next round
> (§7.13–§7.14; Alec, 2026-10-09).** Keep the support mask, one read per
> placement, the eight-space witnesses, the independent read allowance and
> the repairs; append to the receipt. Three changes. **(1) Reconstruction
> along the derivation.** The field cost's inside term is the landing's
> reconstruction of each admitted word by inversion along the actual
> derivation (4.5 spec §3.1), not the free decoder's leaves; the free decoder
> and the identity audit remain reported measures; free compound decoding is
> deferred to item 6 (its inverse will clean up against the primed words and
> STM's composed wholes). The four-word control must then reconstruct. **(2)
> Reading order is taught by a lesson, never wired.** No default order, no
> move cost. Stage 3 of the generated curriculum (Training.md): the lesson
> gives the words in turn; at each read the candidate scorer receives a
> cross-entropy toward the candidate the lesson names (its support identifies
> it; no position label, no attention action is supplied otherwise), and the
> read is teacher-forced to it during the lesson; outside lessons the hard
> choice, the paired-cost departures and the field cost train it as now.
> Stages: 1, identity and object permanence (one percept whole presented,
> re-presented, re-presented after an intervening item: same row, recalled
> after the gap); 2, identifying words (known and novel; a novel word minted
> once, recognised on its second presentation); 3, reading words in order
> (two, then more); 4, simple sentences of each kind — identities, NP,
> NP + VP, relations — reconstruction, asked parts, and a supplied answer to
> one question of each kind. Enabled against disabled (source order), one
> fresh unseeded run per mode, no retries; a stage is read as a result only
> when its control passes. Report per read as now, plus the lesson's
> cross-entropy per stage. **(3) Bisect c with the scorer disabled**: the
> forced grammar must receive exactly what the landing gives it in source
> order — suspect, in order, the masked per-read payloads, the closing-fork
> fix, the zero-budget path; d is expected to follow the order lesson, but is
> verified, not assumed. Then the source-matched sweep. Publish sequence
> unchanged: no commit before Claude's review.

*Training.md revised (Alec, 2026-10-09):* stage 4 is broken into 4a identities,
4b NP, 4c NP + VP, 4d relations — one more operation at each step — and a new
stage 5, **tying sentences together: object permanence by reference**
(pronoun and definite re-mention across sentences, with and without an
intervening sentence), comes *after* the simple sentences, since a reference
needs sentences to refer to and a row to persist. Stage 1 is identity of the
single item only. §7.14's hand-off stands as pasted: its stage 4 is 4a–4d.

### 7.15 Review of the curriculum candidate (Claude, 2026-10-10) — close; two repairs before landing

Reviewed: the receipt's "Curriculum and derivation reconstruction" append,
`curriculum/summary.json`, `reads.tsv`, `bisect-summary.json`, the sweep
(5,714 cases: 5,425 passed, 286 skipped, 1 XPASS, 2 failed), and the new
modules (`DerivationReconstruction`, `reading_lesson`, `CandidateAttention`
owner wiring).

**Disruption of other tests — one real, one flaky.**
- *c* (6.2's forced-answer certificate) fails, and Codex's paired diagnostic
  localises it well: the same traversal and the same inverse with the
  landing's `R` restored passes; with the field cost it fails, because the
  field cost keeps the explore trial in rows 0, 3, 5, 7 and one kept reading
  of `what is y ?` has lost its open referent. The likely reason is the
  undefined inverse: the question's operator is lossy (`inverse_kind =
  'search'`), the derivation inverse supplies no search, so the correct
  reading's words come back unrecovered and are charged in full — and a
  reading that avoids the operator wins. 4.5 spec §3.1 already states the
  rule: where the inverse is undefined there is no expectation, `κ = 0`, no
  charge. The inside term must exclude undefined coordinates (reported
  separately), not charge them. Verify with the same paired diagnostic.
- *b* (the unseeded initial binding distribution, bound .9–1.1) failed once
  in the sweep at 1.129 and passes 3/3 at HEAD and 3/3 on the candidate in
  isolation: a flaky statistical bound, not a regression. Recorded; the
  bound should be tightened by construction (more menus), never by a seed.
- d, g, the packed certificates, the constructed XOR gradient test pass.

**6.1's own gates.**
- Pass in both modes: identity (singleton rows, retained across the gap) and
  identifying words (a novel word minted once, recognised the second time).
  The support audit holds over 46,856 per-read rows: every active read
  supports exactly one word; all 6,144 lesson-named reads select the named
  support.
- Two-word reading: control 4/4; enabled 3/4 (`gold green` → `green green`,
  one reconstruction miss, order 4/4).
- **The order lesson did not teach.** Mean CE 0.34657 → 0.34599 at two
  words — the start is exactly `ln 2 / 2`, the zero-init tie, and the end is
  a logit gap of ≈ 0.002 after 64 teacher-forced presentations; four words
  0.7936 → 0.7832 likewise. The reported "order 4/4" is the tie-break (first
  eligible = source order), as the receipt itself notes; free four-word
  order is 2/4. Cause: the scorer's owner is momentum SGD at lr .01 and the
  two candidates look nearly alike to the hidden layer, so the CE gradient
  cannot move the readout in 64 steps. Repair: an adaptive optimizer for the
  scorer's owner (as the choosers have) or its own learning rate; the
  lesson's gate is its CE falling to ≈ 0 within the stage, and free order
  on permuted fields after it.
- Four-word word lists: the control reconstructs 1/4 while 4–5-word
  relations reconstruct 4/4 — not a length limit; a list of nouns composes
  through a lossy operator whose inverse is partial. Stage 3's gate should
  be read on the reads (one word each, in the lesson's order), not on the
  reconstruction of a word list; and undefined coordinates uncharged, as
  above.
- Enabled identities 2/4 with 9 kept reorderings: the same keep-decision
  defect as c, on `x is three`.
- All answer controls 0/4: the output path does not learn in 64
  presentations. That is 6.2's deferred learning (item 0), not 6.1's; the
  output stages are uninterpretable for attention, as the receipt says. A
  consequence to note: the scorer, credited by departures during those
  lessons (48/68 kept while the answers were noise), learned content
  preferences again — relations then read `cat mat is on`. The lesson was to
  prevent this; it will once it trains.

**Verdict.** The mechanism is in an acceptable state once two repairs land:
(1) the undefined-inverse charge removed, with c passing on the paired
diagnostic and in the sweep; (2) the scorer's optimizer, with stage 3
re-measured on reads and CE. Then land the mechanism as 6.2 and 6.5 did —
support mask, one read per placement, eight-space STM, derivation
reconstruction, the lesson — with the learning gates (answers; stage 3's
free order on permutations; §3's context stages) carried to item 0's
checkpoint, b recorded as flaky, and the rising per-presentation runtime
noted for item 1.

### 7.16 Alec, 2026-10-10: implement the two repairs and accept either way

Alec: "Let's implement them and accept either way, continuing any fixes as
necessary in conjunction with adding more features." 6.1 lands as a
mechanism landing (as 6.2 and 6.5 did), with the two repairs applied and
their outcomes recorded whichever way they fall; whatever is still open is
carried as work alongside the next items, not deferred silently.

> **6.1: two repairs, then land (stream-state plan §7.15–§7.16; Alec,
> 2026-10-10).** **(1) Undefined inverses are not charged.** Where the
> derivation inverse is unavailable — a lossy operator (`inverse_kind =
> 'search'`) or any coordinate the inverse does not define — the inside term
> excludes those coordinates (4.5 spec §3.1: `κ = 0`, "one expects only what
> the operation let through") and reports them separately per read
> (unrecovered words and coordinates); no word-bank search, no target. Verify
> on your captured-initialization paired diagnostic that the candidate now
> passes certificate c where the cost-only control did, then in the sweep.
> **(2) The scorer's optimizer.** Give the candidate scorer's owner an
> adaptive optimizer (Adam, as the choosers' adaptive group has; learning rate
> as configured) in place of momentum SGD, or its own learning rate if you
> find a reason to prefer that — record which. Re-measure stage 3 (two words,
> then four-word permutations): the lesson's gate is its cross-entropy
> falling to near zero within the 64 presentations, and free reads following
> the lesson's order on permuted fields afterwards; gate stage 3 on the reads
> (one word each, in the lesson's order), not on reconstructing a word list,
> and report the list reconstruction beside it. **Then land, either way.**
> Run the source-matched sweep; complete the receipt with both repairs'
> outcomes exactly as measured (c passing or still failing; CE and free
> order); record b as a flaky unseeded bound (3/3 at HEAD, 3/3 on the
> candidate in isolation; one sweep failure at 1.129) to be tightened by
> construction, never seeded; note the rising per-presentation runtime for
> item 1. Commit the source and the receipt on BasicModel with the co-author
> trailer, push, bump WikiOracle and push. Reconcile `todo.md`: 6.1 to Done
> as a mechanism landing — support mask, one read per placement, eight-space
> STM with witnesses, derivation reconstruction, the order lesson, the
> curriculum harness and its first two stages passing — with the learning
> gates (answers; free order on permutations; §3's context stages) carried to
> item 0's checkpoint, and anything still failing after the repairs (c, the
> four-word list inverse) listed as open work to continue alongside item 6,
> not deferred.
