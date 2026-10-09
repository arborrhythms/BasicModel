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
