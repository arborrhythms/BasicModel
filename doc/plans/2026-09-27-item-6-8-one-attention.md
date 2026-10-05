# Item 6.8: one attention — brackets, narrowing, and expectation at every bracket

**Status.** Decided in direction by Alec, 2026-09-27; plan by Claude.
Codex implements landing 6.8-1 after items 7.5, 7 and 6.9 and the
operators update; there is no conference freeze (Alec, 2026-10-03). Claude
reviews before commit.
Landing 6.8-2 (the dynamic stop) is recorded in
[FutureWork](../FutureWork.md#the-dynamic-stop-glossing-above-words-descending-at-novelty-item-68-2)
and is not scheduled. The todo entry is item 6.8.

## 1. The simplification

Attention is one mechanism: a **bracket** over the input, and a rule for
narrowing it. Everything the architecture now calls parallel mode, serial
mode, the subsymbolic loop, the symbolic loop and the boundary predictor
is this one mechanism seen at different bracket widths.

- **Open awareness** is the bracket set to the whole input. The field
  reads it: every active percept's presence and observed complement,
  pooled over the bracket (item 9b entry 4c).
- **The field's operations are the order-independent ones**, and only
  those: and, or, not, on pooled presences. This is the criterion, not a
  list: an operation belongs to the field exactly when it is commutative
  and associative over the bracket's contents, because a pooled reading
  has no order to offer. Everything order-dependent — lift, lower, verb,
  adverb, preposition, part, equal — is the grammar, and it acts
  **between** brackets, over the sequence of readings that narrowing
  produces. The 11c mode exclusion stops being a rule and becomes a
  theorem: field operations within a bracket, grammar operations across
  brackets.
- **Narrowing is the reading policy the four corners already give**
  (11c entries 5–6; 9b entry 4c): a bracket whose pair reads *true-only*
  or *false-only* is pure — move on; *both* — divide the bracket;
  *neither* — look closer: descend to the bracket's parts, and if nothing
  is there, `interpret` mints (9b entry 5). Serial reading is "narrow
  until each bracket's encoding is pure".
- **What narrowing does on text.** A known word is one fused part and
  reads purely at its word bracket, so narrowing stops there. An unknown
  word reads *neither* at its bracket, so narrowing continues to bytes and
  `interpret` mints a provisional object. A known multi-word unit reads
  purely at the wider bracket and is **glossed** as one symbol. This is
  speed reading that slows at novelty.
- **Parallel-first falls out.** The open read is the first bracket of any
  input; the context pass of 9b (7a) is not a schedule but the first
  thing attention does. `interleave:N` was the open bracket's width in
  sentences.
- **Expectation at every bracket (Alec, 2026-09-27: call it expectation,
  in line with mathematical expectation, not "prediction").** We expect
  something in a past, current or future frame, and the expectation is
  applied with the opposite sign so that what is processed is mostly
  surprise. There is an expectation for any subject of attention — a
  whole sentence, the next concept, a past frame recalled — so the one
  mechanism is the expectation of the reading at the current bracket: the next byte inside a word,
  the next word inside a sentence, the next sentence inside a document,
  the next row in the chain. Same ARMA machinery, same negative image,
  same surprise column ([expectation §2.6](../specs/2026-09-20-accessible-mind-subsystems.md#26-expectation)),
  with the level as an argument. Consequences: a training target at every
  bracket rather than one per sentence, which closes most of the
  supervision-density gap against a token model; and the pilot's next-word
  gate becomes this predictor at the word bracket, not a separate scorer
  ([LLMEquivalence](../LLMEquivalence.md), condition 2).

## 2. Why after 7.5 and 7, before 6.5

- **7.5 supplies the chooser.** One softmax over every candidate operation
  at every location, fixed rounds with an eligible STOP, exploit plus one
  explore path, straight-through learning
  ([spec](../specs/2026-09-26-one-operation-per-round.md)). Narrowing is
  not a second policy: **divide**, **descend**, **gloss** (stop at this
  bracket) are candidates in that softmax at the bracket's location,
  chosen and credited the same way as the grammar's operators. Built
  before 7.5, the policy would need a chooser 7.5 replaces.
- **7 supplies the terminal encoding.** The closing makes the sentence
  bracket's reading a row with its `.where` and `.when`
  ([two truths §3.1](../specs/2026-09-16-two-truths-ideas-and-relations.md#31-idea-rows-decided)),
  which the level-indexed predictor needs at the row level and which pins
  the order of readings across the open bracket.
- **6.5 continues it.** Identity binding — bind to a column in the
  recency buffer or cued frames, or mint — enters the same softmax as
  further candidates, and the inter-frame predictor's sources go live
  ([spec](../specs/2026-09-26-independent-components.md)). 6.8 puts the
  bracket candidates and the level-indexed predictor in place first, so
  6.5 adds to one chooser and one predictor rather than two.

## 3. Landing 6.8-1: the fast loop is kept

**The stop is pinned at words.** Narrowing always proceeds to the word
bracket and never below it unless the reading is *neither* (unknown word
→ bytes → mint) and never above it. The compiled per-word step therefore
keeps its shape: 7.5's one-softmax reducer over the two-slot window is
its inner body, and 6.8-1 adds to that layer the bracket candidates and,
before the sentence, the open read. Nothing else changes in the inner
loop. The open read is amortised over the group it covers.

**What is added.**

1. *Bracket candidates in the 7.5 softmax*: `divide` (split the current
   bracket at its heterogeneity, i.e. between the occurrences whose poles
   disagree), `descend` (read the bracket's parts), `gloss` (accept this
   bracket's pure reading as one symbol; STOP at this level). With the
   stop pinned, `gloss` above the word bracket is masked and `descend`
   below it is eligible only on *neither*. The candidates are declared in
   the grammar file like any operator and carry no predefined surface.
2. *One narrowing budget*: the number of bracket operations an input may
   cost, replacing `subsymbolicOrder` and `symbolicOrder`. It bounds the
   7.5 rounds spent on bracket candidates. Name it in Params; the old
   knobs are rejected by the schema.
3. *Expectation at every bracket* (the "level-indexed predictor" of
   earlier drafts): the existing inter-sentence expectation generalised by
   a bracket-level argument, with the **word** and **sentence**
   levels live in 6.8-1 (byte and row levels declared, off by default,
   measured in 6.8-2 and item 6.5 respectively). The word-level prediction
   is a discrete choice from the candidate bank, so it is a distribution,
   not a point (LLMEquivalence, condition 2). Its surprise feeds the same
   column and the same negative image at its level. The word level is the
   NanoChat gate's evaluator (item 4).
4. *The open read as the first bracket*: the 9b context pass, re-expressed
   as bracket width; no gradient step, as decided (9b (i)).

**Compiled shape (Alec's question, 2026-09-27: without the loop over
words, will this compile in a GPU-friendly way?).** Yes, on the same
terms 7.5 already meets, and 6.8-1 does not remove the word loop: with
the stop pinned, narrowing *is* the per-word `while_loop` with its static
bound (`serialWordCapacity`). The general form keeps three static
properties: (i) **brackets are a padded table**, `[B, K, 2]` intervals
with a validity mask, `K = attentionBudget`; `divide` writes two intervals
into free slots, `descend` writes the children from the fold ladder's
retained witness, `gloss` marks a slot done — all tensor writes, no
Python over rows; (ii) **rounds are a static budget with early STOP**, as
7.5's rounds are, inside the existing `while_loop` autograd pattern; (iii)
**per-bracket work is batched over the reading axis** the field already
has (`[percept, batch, reading, pole]`, containment by mask), so the open
read and every narrowed read are one batched field read. What would not
compile is the same as today: host-side Python over rows or edges (item
1's islands) and data-dependent recursion; the design has neither. The
grammar's sequence of pure readings is variable-length under a static
cap, which is what 7.5's rounds already consume. Batching across
sentences remains the throughput lever (item 1).

**What is deleted**, per the no-legacy rule, not gated:

- `modeSchedule` and `serial` as a mode: the bracket schedule with its
  pinned stop is the serial loop; `parallel` is the open bracket alone.
- The subsymbolic-versus-symbolic loop distinction and `subsymbolicLoop`:
  narrowing is the loop; encoding (symbolization) is what a pure bracket
  does.
- The two order budgets, replaced by the narrowing budget.
- The boundary-only sentence predictor as a separate object; it is the
  sentence level of the one predictor. Its knobs (`interLossWeight`,
  `armaScale`, `interContrastiveWeight`) become per-level.

**What stays.** The grammar catalogs and their faces; `interpret`; the
closing; STM; the tied reconstruction; the field's folds; the where registry
and the two-rung ladders; every 9b contract.

## 3a. The word whole (Alec, 2026-09-29; taken up with this item)

"We can predefine a word whole that is a cut of non-white space letters
if that would help our certainty with Lexing words." It was offered
during the review of item 7
([two truths §17.7](../specs/2026-09-16-two-truths-ideas-and-relations.md#177-a-word-is-a-concept-its-object-replaces-it-and-no-row-is-added))
and placed here the same day: "Waiting for 6.8 is fine, as long as it's
written down. That's next anyway." Item 7 builds none of it.

**What it is.** A whole, predefined in WholeSpace, whose extent is a
word by the common definition: a maximal run of letters. **Decided
(Alec, 2026-09-29):** "In the word whole, punctuation and digits also
separate words; use the common definition." White space, punctuation and
digits all separate; a word then has one whole that is its own extent.

**Why it belongs to this item.** 6.8-1 pins its stop at the word bracket
(§3). The word whole is the property whose run is that bracket, so the
bracket a word is read at and the whole a word has are one extent.

**What it is for.**

- *A whole that learning does not move.* The wholes a word has today are
  the character classes that hold on its surface. WholeSpace names eight
  of them, `letter`, `digit`, `whitespace`, `punctuation`, `capital`,
  `control`, `high_byte` and `pad`, and their memberships are learned, so
  a word's wholes can change while it is being learned ("hello" has
  `letter`; "abc123" has `letter` and `digit`). Measured in the item 7
  review: the property rows of one word read as `(1, 9)`, `(1, 8, 9)`,
  `(8, 9)`, `(9,)` and `()` over one training run.
- *One cut.* WholeSpace cuts its field where a learned property changes,
  and the reader cuts words by a fixed rule (`Meronomy.word_spans`). The
  word whole makes the two agree.

**To settle in this item** (Claude's notes; nothing here is decided).

1. *The reader's rule changes with it.* `Meronomy.word_spans` separates
   at white space and punctuation and not at digits, so today "abc123"
   is read as one word and "01" as a word. By the decision above
   "abc123" holds the word "abc" and a run of digits, and "01" is not a
   word. One rule is to serve the reader and WholeSpace both.
2. *The proofs of XOR read digits.* XOR_exact and the grounded XOR read
   "00", "01", "10" and "11". When digits separate words these are runs
   of digits and not words, so what reads a run of digits, and what
   concept it is given, has to be said before the word whole is built,
   and every proof of XOR is measured before and after it
   (two truths §15.1; they run in every receipt).
3. *Fixed or learned.* "Predefine" is read as fixed: the word whole's
   membership is not a learned coefficient. The other eight stay
   learned.
4. *It does not say where a word is.* Every word has the word whole, so
   it cannot by itself keep a word from being read where only its whole
   is. Item 7 requires that of the read: a word is evidenced only where
   its part is read (two truths §17.7, test 29).

**Tests, proposed.** On every sentence of the evaluation corpus the runs
of the word whole are the reader's word brackets; the word whole's
membership is the same after training as before it; and a word's
definition holds the word whole beside its other wholes.

## 4. Tests and measurements

Mechanism tests, unconditional: (1) the field admits only order-independent
operations — a test that a grammar file declaring `lift` as a field
operation is rejected and that `and`/`or`/`not` are rejected between
brackets; (2) a pure bracket is not divided, a *both* bracket is, a
*neither* bracket descends, and with the stop pinned a known word never
descends and a wider bracket never glosses; (3) an unknown word descends
to bytes and mints exactly once, and its second occurrence reads purely at
the word bracket; (4) the open read precedes the sentence and takes no
gradient; (5) the word-level predictor produces a distribution over the
candidate bank and its surprise reaches the column; (6) the deleted knobs
are rejected by the schema and absent from every shipped config; (7) the
XOR gate and exact-zero controls are unchanged; (8) the serial
reconstruction baseline is within the reviewed-9b tolerance of its
current values.

Measurements, recorded not gated: the open pass's cost against item 1's
per-word throughput baseline, taken before 6.8-1 starts; the both-rate and
categorical-discrimination index per level (item 4's fields), since the
policy turns on the field's *both*; and the word-level predictor's top-1
and reciprocal rank on the frozen NanoChat item manifest with the
shuffled-prefix control. Learning gates follow item 9's million-sentence
prerequisite: ordered word-level prediction against shuffled and
context-free controls at equal updates, three seeds, declared in advance,
re-declared from item 9 at the word bracket.

## 5. Documentation after landing

- [Architecture](../Architecture.md): the three-operations section becomes
  "one attention": bracket, field criterion, narrowing policy, level-indexed
  predictor; the mode-exclusion paragraph becomes the theorem; the pump
  section describes the open read and narrowing budget.
- [Params](../Params.md): the narrowing budget; the per-level predictor
  knobs; removed knobs listed as rejected.
- [Language](../Language.md): the bracket candidates in the operator
  catalog with their three faces (`divide`/`descend`/`gloss` have no
  generate face; their reverse is the closing's bracket).
- [Accessible mind spec](../specs/2026-09-20-accessible-mind-subsystems.md):
  §2.2–2.4 restated as bracket widths; §4's table gains the bracket
  candidates' row.
- [NanoChatGrammarPilot](../NanoChatGrammarPilot.md): the gate is scored by
  the word-level predictor.
- [LLMEquivalence](../LLMEquivalence.md): condition 2 notes the word-level
  predictor as the discrete distribution.
- [FutureWork](../FutureWork.md): 6.8-2, already recorded.
- [Testing](../Testing.md) and a receipt under `doc/benchmarks/`.

**The conceptual counterpart of the bracket (Alec, 2026-09-28).** "Words are
a formula for narrowing attention." The bracket narrows the input; the words
narrow the **domain of discourse**, Boole's universe of discourse, which the
situation seeds and each word restricts. They are two attentions that meet
in one chooser:
`divide`, `descend` and `gloss` choose the next bracket, and the
restrictors — nouns, adjectives, verbs and adverbs, one kind of operator,
each a projection onto a smaller subspace — choose the next domain, all as
candidates in the 7.5 softmax. The criterion of this plan, that the field's
operations are the order-independent ones, is derived from the
representation in
[accessible mind §2.0](../specs/2026-09-20-accessible-mind-subsystems.md#20-fields-codes-and-ideas),
and the language mechanics are in
[§2.0.1](../specs/2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention).
Expectation at every bracket has a counterpart at every index — the next
thing, the next time, the next alternative — of which only the first two
have machinery
([FutureWork](../FutureWork.md#modality-as-the-third-index-noted-2026-09-28)).
**`attentionBudget` does not meter the words** *(Alec, 2026-09-28)*: "words
do narrow attention, but I think the main use of the attention budget is
prior to grammatical analysis, so they would not figure in." The budget is
the number of bracket operations an input may cost (§3); the words'
narrowing of the domain is grammatical analysis and has no budget of
attention; 7.5's static round count is the shape of a compiled loop. Nor must a reading narrow the domain at all: "percepts are not
necessarily limiting because they may direct us to the symbol 'everything',
which has no narrowing effect" — the conceptual counterpart of the open
read.
**What the budget does cover** *(Alec, 2026-09-28)*: "we regard symbols as
percepts. So perceptual attention may exclude perception or higher order
thought; and there is good evidence that both of those do share a single
attentional budget." A bracket may therefore be over the input or over
symbols, and both draw on `attentionBudget`. "The conceptual narrowing is
different, and probably needs a Ground in order to make its Figure
meaningful": perceptual attention discards what it excludes, while the
words keep the domain they elect from
([accessible mind §2.0.1](../specs/2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention)).
**The budget is for percepts** *(Alec, 2026-09-28)*: "I think there is a
budget for percepts, not for conceptual space (although thoughts do shape
the latter)." *Claude's reading, to confirm:* the thought work budget
(accessible mind 2.8) and `attentionBudget` are then one meter, since a
thought operation attends to symbols and symbols are percepts. They are
separate today, the first built and the second not; this item is where they
would be joined.

## 6. Questions for Alec (none blocking 6.8-1) — 6a and 6b answered 2026-09-27

**6a, decided (Alec, amended 2026-09-27).** Some operators require a
**single prominent symbol** to operate on. **All Boolean operators** —
`and`, `or`, `not` — **operate on a field of concepts at one time**: a
plural bracket is aggregated by them directly (a union of the field's
concepts is a temporary whole — *symbolic*, not conceptual, since it is a
superposition of symbol values; corrected 2026-09-28,
[accessible mind §2.0](../specs/2026-09-20-accessible-mind-subsystems.md#20-fields-codes-and-ideas))
and no separate
multi-argument operator is declared for it. The non-Boolean
(order-dependent) operations of a grammar are operable only once a
single concept has been identified.
Applied to the parallel activation of the whole field they would produce
a field of concepts that would then have to be aggregated or attentionally
eliminated before anything could proceed. The decision is made as
follows:

- *Operand kinds are declared.* Every operator declares the kind of each
  operand: **field** (a pooled reading over a bracket) or **symbol** (one
  identified concept). The Boolean operations `and`/`or`/`not` and the
  bracket candidates `divide`/`descend`/`gloss` take field operands; every
  order-dependent operator takes symbol operands. The parser rejects a
  grammar file that gives an order-dependent operator a field operand.
  This is static, checked at load, and it is the whole of the enforcement.
- *A bracket yields a symbol when it is pure and singular.* Pure: exactly
  one pole of its pair is nonzero after pooling, with the minor pole at
  the existing exact-zero tolerance. Singular: exactly one concept
  identity has support in the bracket, so the projection to a code is
  unambiguous ([accessible mind §2.0](../specs/2026-09-20-accessible-mind-subsystems.md#20-fields-codes-and-ideas),
  a code is the nearest-row projection of a field). Both tests are crisp;
  `gloss` is the candidate that performs the projection, and it is
  eligible only when both hold.
- *Otherwise only field and bracket candidates are eligible.* A bracket
  that is impure (*both*) or plural (several identities) offers `and`/`or`
  (aggregate the field of concepts into a kind or a conjunction), `not`
  (attentional elimination, non-affirming exclusion), `divide` and
  `descend`. The 7.5 softmax masks the ineligible candidates at that
  location; everything graded — how impure, how plural — goes into the
  logits, so the chooser learns when to gloss a nearly pure bracket or
  divide a barely heterogeneous one, while the mask alone forbids a
  grammar operator on a non-symbol.

**6b, decided (Alec).** The narrowing budget is `attentionBudget`.

**6c, decided (2026-09-27; the conference freeze dropped 2026-10-03).**
Until 6.8-1 lands, the NanoChat gate is scored by the existing
`IntraSentenceLayer` held prediction, an untrained point predictor in idea
space used only for evaluation, stated as such; the trained, distributional
word-level expectation is part of 6.8-1 and does not reintroduce the retired
within-sentence cursor loop (one expectation read per word inside the
existing per-word step).

- **6a.** The field's `and`/`or`/`not`: keep them declared in `<compose>`
  as today, with the field-versus-grammar criterion enforced by the parser
  (recommended), or make them the field's built-ins outside the grammar
  file?
- **6b.** The narrowing budget's name: `attentionBudget` (recommended), or
  keep `symbolicOrder` with the new meaning?
- **6c.** For the conference checkpoint, is the word-level predictor pulled
  forward as evaluator only (recommended; no training-loop change before
  the freeze), or also trained at the word level before the freeze?

## 7. What reading attention and global attention carry into 6.8 (Alec, 2026-10-02)

Alec, on the configuration review: "Reading attention: if 6.8 replaces it,
great, but let's make sure we aren't deleting useful functionality."
"Global attention: same; let's make sure we are not dropping features with
the 6.8 integration."

Both modules stay until 6.8 lands, and they retire within it once every
capability below has its home. They are `ReadingAttention` and
`GlobalAttention` in `Spaces.py`, with the flags `readingAttention`,
`globalAttention` and `globalAttentionConsume`. Their four configurations
remain the reference until then: `MM_reading`, `MM_global`, `MM_qa` and
`matrix/MM_20M_grammar_reading`. Before then, two things make them usable:

- the 6.9 defect in the mixing leaf staging is fixed;
- `MM_qa`'s TruthSet input is repaired.

Their weekly tests crash on the first and the second keeps `MM_qa` from
building.

| Capability today | In this plan | To add |
|---|---|---|
| **Reading attention** | | |
| A learned choice of where to read next, from the previous pass's concept and the STM symbols | The narrowing candidates (`divide`, `descend`, `gloss`) in the 7.5 chooser; 6.8-1 pins the stop at words | — |
| **Priming steers reading.** A span scores high when its content lies near a prototype the intent has primed: `max over v of cos(key, row_v) · boost_v`, the codebook-retrieval prior | Not here | The priming surface enters the logits of the bracket candidates. This is the same surface that item 6.9's §15.3 uses for the answer's context. |
| Supervision on the next word: cross-entropy on the next span | The word-level expectation predicts the next word as a distribution over the candidate bank (§3, item 3) | — |
| Teacher forcing: the true next span in training, its own prediction at inference | Not stated | State it for the word-level expectation |
| A shift bootstrap: at initialization it reads exactly one word per pass | 6.8-1's stop pinned at words | — |
| The scope it writes drives the top-down mereology handoff (`_passback_scope_where`) | The bracket is the scope | The handoff reads the bracket table |
| **Global attention** | | |
| One registry of addressable spaces: the input window, STM, LTM and the part, whole and symbol codebooks (`_addressable_spaces`, which the reasoner also reads) | "A bracket may be over the input or over symbols" (§5); the spaces are not named | Name the spaces a bracket may cover, from the existing registry. Keep the registry. |
| One choice across all the spaces, with a learned prior for each. Recall and reading are one mechanism; the type tag says which. | One attentional budget for both (§5); no choice across spaces | A space choice among the 7.5 candidates, under `attentionBudget` |
| A typed `.where`: which space, and the interval within it | Brackets are intervals over the input | The bracket table carries a space tag |
| **A learned read trained by the answer.** The read enters the answer through a gate initialized at zero, so retrieval that lowers the answer's error is rewarded. The keys are detached. | Thought's hard reads, which have no parameters. Since 6.9 §15.3, the primed symbols are also in the answer's context. | A learned reader of the primed symbols, owned by the answer. 6.9 §15.3 may already provide it if the answer reads the bank through this module's scorer and consume gate. |
| Temperature on the explore pass, so that task error shapes where attention lands | 7.5's explore trial | — |

## 8. Acceptance test carried from item 6.9: XOR of two words (Alec, 2026-10-02)

Alec: "Yes, option B, we'll get it working in the next stage."

MM_xor's `test_convergence` must pass, with its bar unchanged (loss below
.20 within 200 epochs), while the words are read as words.

Until the 6.9 migration, this test passed by a shortcut. The radix reading
promoted chunks that cross the word boundary (`hello wo`, `hello th`,
`loving w`, `loving t`), which gave one percept per sentence, so XOR was a
lookup of four codes. Meronomy promotes words only. Now the test needs the
open read to compose two words nonlinearly before the answer reads them.
Measured in the 6.9 receipt of 2026-10-02 (6.9 plan §17), three things stand
in the way:

1. **No live recurrence.** Each parallel binding re-reads the original
   percepts, and the WholeSpace carrier between bindings returns a neutral
   field. So only the last of MM_xor's three bindings learns from the
   answer.
2. **A linear reading of words.** The numeric head reads the word slots
   linearly, which gives a sum of per-word contributions. A sum cannot be
   XOR, and its best is one half everywhere (6.9 plan §3.3).
3. **The first word drops out.** In failed runs the first word's six content
   numbers fall to zero.

The field's `and`/`or`/`not` over the open bracket (§1) is the
order-independent nonlinear operation this test needs. XOR_exact already
shows it on primitive memberships.

## 9. Carried from item 6.9 (Alec, 2026-10-03)

Item 6.9 closed on 2026-10-03 as the baseline of grammatical learning
([6.9 plan §25.3, §26](2026-09-29-item-6-9-xor-grammar.md#26-review-of-the-closing-round-claude-2026-10-03)).
Two of Alec's closing comments are this item's work.

**9.1 Exploit and explore for compose, think and generate.** "Compose,
think, and generate all need exploit and explore trials." Item 7.5 gave
compose a greedy derivation and an explore derivation that departs from it at
one round, both costed under the same parameters, the explore kept only on
strictly lower cost of the objective that owns the choice. 6.9's closing run
showed why generate needs the same: its walk, a hard argmax trained through
a straight-through surrogate, chose STOP at the first step in all 3,200
decodes and never discovered that undoing the binary operation would yield
the missing word, because the missing-word penalty does not depend on the
transition not taken. So:

- each of the three grammars' walks has an exploit path and one explore path;
- both are costed under the same parameters before either trains;
- the explore path is kept only on strictly lower cost of the walk's owner:
  reconstruction for generate as the decoder, the answer for generate as
  output (a choice, not a gradient, so ownership stands), the thought
  controller's return for think;
- the record keeps which path was kept, and the audit reports the fraction
  of walks where explore won and the derivation stability per sentence, as
  for compose.

This is the §3 chooser extended, not a second policy: `divide`, `descend`
and `gloss` already sit in that softmax, and the explore path is how any of
the three grammars finds an action the greedy path never takes.

**9.2 Codes need not converge all the way.** "Yes, but if they don't
converge all the way, that's fine." The codes may drift partly together
under reconstruction (6.9 §26.1: mean squared off-diagonal cosine .07 → .37
in the closing run). The antipode term keeps them from collapsing; the gates
read the answer and the read-back, not the geometry. Code geometry stays in
the audit as a diagnostic, not as a bar.

**Acceptance tests carried over** (with §8): MM_xor's word-level XOR; and
6.9's two XOR_grammar gates, which may only improve against the closing
record (the no-regression rule), with the decoder's inferred operations
expected to match the compose derivations once it explores.

## 10. Review of the first 6.8 round, with the decoder exploration and the operators update (Claude, 2026-10-04)

[Receipt](../benchmarks/2026-10-03-operators-attention/README.md). One
uncommitted round carried 6.9 §26.3 (the decoder's explore walk), the
operators update (the catalogue's §12 implementation) and 6.8-1 with §§7–9.
Not accepted yet; held for the measurement below, not for a found regression.

### 10.1 What landed

- The default suite is green: 4,829 passed, 0 failed, 285 skipped, 124 s.
- XOR_exact, grounded XOR and the 46 retained reading/global capability
  tests pass. The four reference configurations pass their mechanism checks.
- **MM_xor passed its unchanged word-level XOR bar (.183 against .20)**, §8's
  acceptance test, in one run.
- Operators: operand kinds, faces, the effect ledger; `sum` as a mean,
  intersection as the signed minimum, compound case selection, `non`
  without a faithful inverse; `exist`, `true`, `lookup` retired.
- Attention: the typed bracket table and one budget, the open read without
  a gradient step, divide/descend/gloss eligibility from the paired
  evidence, the fixed word whole (maximal letter runs), word and sentence
  expectation, one registry over input, STM, LTM and the three codebooks,
  the priming prior in the chooser, the answer-owned learned reader behind a
  zero gate. The old modes, orders and budgets are rejected.
- Exploit and explore for input narrowing, compose, thought, anticipation and
  the decoder (§9.1), with zero selection violations.

### 10.2 The two "blockers", read

**The class comparison (.321 against the closing record's .115).** One
unseeded run against one. This round's five single trainings gave .0013,
.031, .040, .321 and .875. By 6.9 §20.5 a run between the bands is a moving
representation, and five changes to the reading went in at once. The
comparison that can show a regression is the ten-run gate with both bars
against §22's 9/10 and 5/10.

**Reconstruction 0/4: the decoder emits one word.** The generate walk, not
attention. In the audit's last 128 decoder walks greedy chose STOP first in
91. The explore walk was kept in 37; in 32 of those it tried "undo the
binary, then stop" and cost .33 against greedy's 7.3: both words recovered.
In the other 91 the alternative was mostly `not`, cost about 16, and greedy
was kept. Two things follow:

1. Exploration is deterministic where it should sample: the departure
   "excludes greedy's choice and follows the best available suffix", so with
   a one-step greedy walk it is always the second-ranked action. When that is
   `not`, undo is never tried in that row that epoch.
2. After 400 epochs the policy still ranks STOP first despite 32 large wins
   for undo. Either the straight-through gradient barely moves the
   STOP-versus-undo margin under momentum SGD, or it moves it too slowly for
   the budget. The audit records actions and costs, not the margin, so it
   cannot say which; "more training" helps only in the second case.

Alec, 2026-10-04: the two-word walk "is an interesting finding that may
eventually speed everything up", though it "would have to incorporate a bit
of the grammatical tree over the two words". The walk is the tree run
backwards: one undo per node, the operation inferred from the root and the
pair searched in the shortlist. What the root does not yet carry is the
order of the pair (6.9 §24.5: product and mean are commutative), so a glossed
two-word unit decodes to the right words in an arbitrary order until order
is bound in (MAP's permutation, or a surface marker; catalogued). With that,
a known multi-word unit is what 6.8-2's dynamic stop wants: read as one
symbol above the word bracket and decoded exactly in one undo, the tree
inside the unit carried by the root rather than walked.

### 10.3 Also noted

- The forward is 2.6× slower with the open read (.141 s against .055 s whole
  forward). Alec: "the slowdown will have to be addressed." Item 1's
  territory; recorded there.
- The decoder's kept path flips between the one-word and two-word walks
  (stability .51), which feeds the moving target.
- The fresh-model NanoChat scoring filled the definition store at item 136:
  frozen-learning evaluation still mints definitions. Item 4's trained
  evaluator needs evaluation not to admit. No capacity change.

### 10.4 Asked of Codex (Alec agreed, 2026-10-04)

1. The ten-run XOR_grammar gate, one training per run with both bars and
   the §20.5 bands; MM_xor ×10 (the §8 acceptance); the sum control ×10.
2. The explore departure sampled among the eligible actions, not the
   second-best, for all walks.
3. In the audit, the decoder's STOP-versus-undo margin per step.
4. Evaluation does not admit definitions (for item 4; noted now).

## 11. Review of the §10 measurement (Claude, 2026-10-04)

[Receipt](../benchmarks/2026-10-03-operators-attention/README.md). The three
changes of §10.4 are in: sampled exploration for every walk, the decoder's
margin in the audit, frozen evaluation that admits nothing. The measurement
is complete: thirty trainings, no retries. Not accepted; two findings, one of
them about the gate itself.

### 11.1 The counts

| | This round | §22 |
|---|---|---|
| XOR_grammar class | 7/10: seven at 0 (four below 1e-3), one between (.065), two above ¼ (1.05, 3.04) | 9/10 |
| XOR_grammar reconstruction | 0/10 | 5/10 |
| MM_xor | 10/10 | red by 6.9 §17 |
| Sum control | **0/10** | 10/10 |

### 11.2 The control is red, so the class gate no longer measures composition

The sum-only grammar learned XOR: three runs at 0 (.0013, .0007, .017) and
checkerboard contrasts up to −2.0 in nine of ten. By 6.9 §20.5 an affine
reader over an additive composition cannot leave ¼; so the answer path is no
longer affine. Codex found the cause and the code confirms it: §7's "learned
read trained by the answer" was built as `PrimedSymbolReader`
(`Attention.py`), an MLP scorer over cosine features with a softmax read and
a learned consume gate, and `BasicModel` installs it in every numeric head's
path (`answer_attention`, applied in `_forward_head` whenever the output has
no concept ids). Its keys are the echoic shortlist, which holds the
sentence's own words. A nonlinear reader over the bag of words is a hidden
layer over word presence, and XOR of two words is learnable from it with no
composition at all. In §22 the shortlist entered the head as an affine
feature (the priming-weighted code sum) and the control held at 10/10.

So the class 7/10 is not evidence of grammatical learning, and MM_xor's
10/10 is suspect for the same reason unless its head bypasses the reader.
The control did its job.

**The rule, made explicit:** the answer's reader on the XOR gates is affine.
The understanding's context may include the shortlist, as Alec required, but
as affine features; the learned attention read (the migrated GlobalAttention
consumer) is retrieval for thought and generation, not part of the numeric
head. Where the control passes, the class gate measures composition; where
it does not, nothing the class gate says counts.

### 11.3 Reconstruction 0/10: the margin moves, too slowly for the budget

The audit answers §10.2's question. The decoder's first-step margin of STOP
over "undo the binary" falls from 1.99 to 1.43 over 400 epochs, −.0006 per
update, positive at every observed parent; greedy chose STOP in all 3,200
walks. At that rate the margin crosses zero after roughly 2,300 more updates,
three times the budget. Meanwhile the explore walk, now sampled, is kept in
1,030 of 3,200 decodes with cost .33 against greedy's 7.3: a large,
systematic advantage that the straight-through gradient transmits at
−.0006 per step, because that gradient flows through the chosen action's
value and not through the choice.

Alec's "more training" would work in principle and is the wrong fix: the
budget is the bar. The right one is the mechanism the thought and generate
policies already have, explicit credit: reconstruction credits the decoder
chooser with the kept walk's advantage (its log-probability times greedy
cost minus kept cost), owned by reconstruction like the rest of the decoder.
Compose never needed it because its choices change the root continuously;
STOP versus undo is a discrete structural choice whose cost gap the
surrogate cannot see.

### 11.4 Also

- Two class runs diverged (MSE 1.05, 3.04): the reader's gate and scorer
  moving under a moving representation. They should disappear with 11.2.
- Kept-path stability of the decoder .61 (was .51); compose .94; input
  narrowing .99.
- Zero ownership conflicts; 161 complete test ports; suite green in the
  previous round, not re-run here.

### 11.5 Asked of Codex

1. The numeric head's reader is affine (11.2): remove `answer_attention`
   from the numeric head's path; the shortlist enters as the affine
   priming-weighted code sum of §22. The learned reader stays for the
   generate walk and thought where retrieval over memory spaces is its job.
   State which configurations route the answer through it.
2. Decoder policy credit (11.3): reconstruction credits the decoder chooser
   with the kept walk's advantage; the audit records the margin as now.
3. Re-measure the three tens on the frozen source: the control must be
   10/10 before the class count means anything; comparison §22 (9/10, 5/10)
   and MM_xor 10/10.

### 11.6 The decoder's stop is §6a run backwards (Alec, 2026-10-04)

Alec: "So we need a mode that will force attention to break at spaces? The
best idea would be to have the grammar learn that it needs to further analyze
the compound if it consists of two meaningful wholes (which are preferable to
a larger chunk, assuming the meaning can be computed)."

No forced mode. Reading already breaks at words (6.8-1 pins the stop at the
word bracket; the audit's compose derivations are over two leaves). The
failure is the decoder's, and the rule is §6a's criterion applied to
decoding:

- The decoder **emits** (glosses) only when the current top is **singular**:
  one shortlist symbol accounts for it.
- When the undo yields two children that are both **meaningful**, two
  symbols with support in the shortlist, the top is a compound and the undo
  is the only eligible move. "The meaning can be computed" is "both children
  are readable", which the pair search already computes.
- The policy keeps what it should learn: which operation to undo, inferred
  from the root. STOP versus undo is no longer learned at −.0006 per update
  (§11.3); it is eligibility.

Consequences: 6.9 §17's shortcut is closed at its root, since a chunk whose
parts are meaningful is never glossed as one symbol; and 6.8-2's glossing is
confined to compounds whose parts do not compute (idioms such as
`MM_ladder_idiom`'s "kick the bucket"). Two meaningful wholes are preferred
to a larger chunk.

This replaces §11.3's policy-credit proposal. §11.5 item 2 becomes: the
decoder's eligibility mirrors §6a. Items 1 and 3 stand.

## 12. Review of the §11 measurement (Claude, 2026-10-04)

[Receipt](../benchmarks/2026-10-03-operators-attention/README.md). The three
asks of §11.5 with §11.6 are in: the numeric head is affine (no
`PrimedSymbolReader` in `_forward_head`; the learned reader stays with
generation and thought), the decoder's eligibility mirrors §6a (emit only
when singular; undo when two meaningful wholes), the three tens measured
once. Not accepted yet.

| | §11 round | §10 | §22 |
|---|---|---|---|
| Class | 2/10: two at 0, five at ¼, three between | 7/10 (contaminated) | 9/10 (frozen codes) |
| Reconstruction | 6/10 | 0/10 | 5/10 |
| Both bars in one run | 2/10, the first time | 0 | — |
| MM_xor | 10/10, affine head, so real: §8's acceptance | 10/10 (suspect) | red |
| Sum control | 10/10, every run at exactly ¼ | 0/10 | 10/10 |

**What worked.** STOP masked in all 6,400 first-step observations; the
decoder undoes to both words; reconstruction 0 → 6; two runs meet both bars;
MM_xor's word-level XOR passes ten of ten with an affine reader; zero
ownership conflicts.

**Why the class gate fell.** Five runs at exactly .2500 are the affine floor,
reached only over an additive composition. Mean is additive (6.9 §22.2), and
the compose chooser, trained by reconstruction alone, has no reason to prefer
conjunction: product and mean are both exactly invertible by search, so
about half the runs settle on the linear one. In §22 the operators were
`min` and `max`, both nonlinear, so any derivation was readable and §3.11's
property held ("any of the four operations and the same choices for every
sentence"). With mean as OR that property is gone and the gate contains a
coin flip. The three runs between the bands are the moving target: in the
audited run two sentences switched derivation once around epoch 200, and the
decoder's kept path is unstable (.34) under sampled exploration among the
eligible undos, so the codes' gradient changes direction. To confirm from
the saved observations: the final compose operator per run against its band
(the run-10 audit records rule ids without names).

**Recommendation.** Disjunction becomes the probabilistic sum, a + b − ab,
in conjunction's form (certainties combined, identity normalized): nonlinear
and invertible by search, so every derivation is XOR-readable and the class
gate measures stability, not the chooser's choice of operator. Mean stays in
the catalogue as `sum`, the control's operator. Then the three tens again.
If mean stays as OR, the gate's reading must change instead, since half its
runs sit on the floor by construction.

Also for the audit: rule names beside rule ids; per-run final compose
derivations.

### 12.1 The composition gate, not a grammar gate (Alec, 2026-10-04)

Alec: "The xor test, as we've written it, is kind of meaningless for grammar
(on an untrained model)." Agreed. On four sentences with no corpus nothing
says which operation a construction calls for, which is what grammar is;
under reconstruction-first the chooser is indifferent between equally
invertible operations, and the class gate then measures a coin.

What the fixture measures is the machinery: two words compose nonlinearly
into one understanding; the understanding decodes back to the words; an
affine reader reads a nonlinear function of the pair from it; one objective
owns each weight; none of it regresses. The sum control keeps that honest.
So the XOR table (XOR_exact, MM_xor, XOR_grammar's two bars, the sum
control) is the **composition gate**, read by 6.9 §20.5, under the standing
no-regression rule. To take the coin out: every binary operator nonlinear
(disjunction = probabilistic sum; mean stays catalogued as `sum`, the
control's operator), so any derivation is readable and the gate measures
stability and decoding.

**Grammatical learning** is which operation, learned from what the
composition is for. Its supervised form is measured now by the grammar
lessons in `MM_grammar_wording` (the compose-choice lesson; the bounded
wording gate of 2026-09-20). Its unsupervised form, operator choice shaped by
expectation over a corpus, is measured by item 9's evaluation, which is
complete and reviewed, on the checkpoint item 0's run produces: "item 9
should be complete: we are at 6.8 and descending." No four-sentence fixture
stands in for it. Item 6.9's "baseline of grammatical learning" is renamed
accordingly: the baseline of the composition mechanism.

## 13. Review of the §12 measurement (Claude, 2026-10-04)

[Receipt](../benchmarks/2026-10-03-operators-attention/README.md).
Disjunction is the probabilistic sum; `sum` keeps the mean; derivations
carry rule names; the thirty trainings ran once. Class **0/10** (nine
between, one at ¼), reconstruction **7/10**, both bars 0/10, MM_xor 10/10,
control 10/10. Not accepted.

### 13.1 Correction to §12

The retrospective confirmed the mean only for §11's run 10; the other
quarter-band runs have no saved derivation. And with OR nonlinear no run
reached 0. So the §12 reading, that the ¼ floor came from the mean, was
wrong or at most a minor cause. The saved roots say what did.

### 13.2 The second word has vanished from the understanding

Computed from the audited runs' saved roots (`geometry-end.json`):

| Run | hw·ht cosine | lw·lt cosine | centered singular values | affine weight norm for an exact fit |
|---|---:|---:|---|---:|
| §11 run 10, end | 1.000 | .884 | .727, .015, 0 | 8×10⁷ |
| §12 run 10, end | 1.000 | 1.000 | 2.607, .021, .013 | 77 |
| either run, start | −.15 / .71 | .59 / .72 | three nonzero | 2.5 / 17.5 |

Four roots have collapsed to two directions, one per first word, with
opposite labels inside each pair. Identical roots for labels 0 and 1 force
an affine reader to ½: the ¼ floor, with no linear operator needed. The
1 to 2 percent residue of the second word lets the reader hedge toward the
labels with enormous weights, which is "between" with all four labels right.

The roots coincide because `world` and `there` merged again, as in §24. The
mechanism is §24.3 and §25.4, still present: the echoic prior tells the
read-back which words were in the sentence, so reconstruction succeeds
whether or not the codes are distinct; the soft pair search blends the
competitors into the recovered leaf and trains the true code toward the
blend. The antipode term as implemented does not hold them apart.
Reconstruction's 7/10 is largely echoic: the words come from the prior, not
from conceptual space. The decoder's flip-flop (stability .009) is a likely
symptom: with merged codes both undos explain the parent equally. MM_xor's
10/10 is unaffected (the field path).

### 13.3 Two ways out

- **(B, recommended) Keep echoic memory; remove the merging channel.** The
  pair search selects hard; the recovered operand is the operator's exact
  inverse given the selected partner (division for the product, the
  corresponding inverse for the probabilistic sum); the gradient flows
  through the exact inverse and the byte cross-entropy, never through a
  softmin blend. Nothing then pulls `world` toward `there`; the antipode
  spreads the codes; product's invertibility, the reason it was chosen, does
  the work.
- **(A)** The prior from before the sentence's own write (§24.4, withdrawn):
  identity must then come from the codes.

Then the three tens again, with reconstruction's count annotated by how
often the read-back's winner was decided by priming rather than by code.

### 13.4 Perceptual space is the basis of zero-order conceptual space (Alec, 2026-10-04)

Alec: "The mereological towers are designed so that their parts and wholes
are the organizing principle within their space. If this is not currently the
case, it should be restored ... a poset of codes for each tower, parts
growing from nothing [0,0,0,…] and wholes shrinking from everything
[1,1,1,…]. Order zero concepts are a collection of positive evidence and
negative evidence, collected from these parts and wholes ... Concepts of
subsequent orders can be determined by using the connectionist strategy
already discussed, leveraging the symbol activations. ... collapse at the
higher-order stage is permissible ... 'world' and 'there' are separated by
their parts, even though they are united by their larger wholes (which, in
this test corpus, is in fact an identical context). Concepts, if they are of
a different size than percepts, might blend the conceptual locations of the
relevant percepts ... if a concept has a basis in percepts, and those basis
vectors have associated concepts, then the weights within that basis can be
applied to create a structured conceptual space which is not necessarily the
mereological space of percepts. ... perceptual space is the basis of
zero-order conceptual space."

**State verified.** The towers keep their poset (percepts in the unit cube;
properties as member sets). The field path computes an order-zero concept
from its 11b definition, evidence over parts and wholes, and does not merge
`world` and `there` (MM_xor 10/10). The serial path's leaf is the word's
dictionary row, `conceptualSpaces.1.layers.3.W`: a free parameter, randomly
initialized, trained by reconstruction since 6.9 §20; its definition sits
beside it and does not determine its location. That free row is the writer
that merged (§13.2).

**The rule.**

- Each basis element, a part percept or a whole property, has an associated
  conceptual location: its own row in conceptual space, learned, shared by
  every concept that uses it (under the shared index, the percept's
  concept).
- An order-zero concept's code is the evidence-weighted sum of those
  locations: parts and wholes with their evidence for; evidence against
  subtracting. Derived at each forward; no free word row.
- Conceptual space keeps its own dimension and metric; the percept basis
  structures it.
- Higher orders by co-activation (the catalogue's distributional pressure),
  where merging of identical contexts is permissible.

The precedent is sub-word compositional embedding (fastText, Bojanowski et
al. 2017: a word vector as the sum of its character n-grams' vectors), in
mereological form: evidence as weights, wholes in the basis.

**Consequences.** `world` and `there` share no letters, so their codes differ
by construction; merging them would merge letter rows every other word
shares. Reconstruction trains the letter and property rows and the evidence
weights. The echoic prior may stay: identity comes from parts. Shared wholes
supply the common component ("partly context, partly parts" by
construction).

**For XOR_grammar.** The association is a lookup, one row per percept, not a
linear map: its percepts carry two content numbers, and a linear image of a
two-dimensional fold cannot put four roots in general position; a table of
letter rows can. The inventory gains the letter concepts: a declared
capacity change.

Supersedes §13.3's (A) and (B). The fold ladder and the 11b evidence tables
compute the weights already; the serial leaf takes the derived code in place
of the dictionary row.

**The context half (Alec, 2026-10-04).** "Two words that mean the same thing
should have a difference in parts but an identity in context, and a
similarity space in the word2vec sense has much of its utility in virtue of
that latter metric. ... the context in the BOW models is lacking in
WholeSpace, but an obvious integration might do something like create wholes
out of parts, which is the mereological equivalent of an association
strength."

- In the derived code, parts give identity and wholes give similarity.
  Synonyms differ in their parts term and, where their contexts are wholes
  they belong to, coincide in their wholes term: different in parts,
  identical in context. The permissible higher-order collapse is the wholes
  term merging while the parts term does not. On the XOR corpus `hello` and
  `loving` share every whole and no letter.
- WholeSpace holds properties read off the input, not the contexts a word
  occurs in. The integration: a whole minted from parts that co-activate
  recurrently, membership weight as association strength. This is the sigma
  fold over the activation field (the set one order above its members,
  decided 2026-09-28), the rule chunk promotion already applies one level
  down (letters that recur together become a word). The store's weighted
  edges are the association structure; priming diffusion already runs over
  them.
- The catalogue's distributional pressure (6.9 §20.3) becomes membership
  weights in context wholes, not a pull between word codes: word2vec's
  pointwise mutual information is the association strength of a word in a
  context whole, and the word's code inherits the whole's location in
  proportion. Co-activation trains the context term, reconstruction the
  parts term; they write different things. Second-order similarity comes
  from overlapping whole sets.

This belongs to the operators update's distributional item; this round's
fix (the derived order-zero code) stands as written above.

**The wholes already exist (Alec, 2026-10-04).** "I'm wondering if we have
that higher order concepts? In which case, there may be a concept minted from
co-occurring parts that is active at the same time as its parts, whose
activation provides the distributional pressure necessary to provide BOW
implicitly." We do: item 7 writes one row per sentence at the closing, its
slots referencing its constituents' rows, and identity is its occurrences. A
word's context wholes are the rows it occurs in; their locations are their
roots; the two-way lookup gives them. No BOW minting is needed: the memory
is the bag of words, by reference. The gap: spreading activation runs over
the concept store's part edges and rows do not conduct, which is why the
primed bank found no activated competitor. Rows conducting (a word primes
its rows, a row its constituents) makes "the words those words activate"
true. Alec: the update in this round, if possible. Hand-off given
2026-10-04: the derived code (parts live, wholes from occurrence rows
detached), rows conducting, the three tens with root geometry and the
priming-versus-code annotation.

**Fallback (Alec, 2026-10-04: "if it's a big change, ok to wait for a future
round… I guess we would temporarily ignore the collapse?").** If the derived
code does not fit the round, the interim is §13.3 (B): hard pair selection
and the operator's exact inverse given the selected partner, so no gradient
flows through a blend; the derived code moves to the next round. Meanwhile
the class gate is recorded red with its known cause (§13.2) and the fix
dated, not tuned around; MM_xor, the control and reconstruction carry the
mechanism evidence, and the no-regression rule applies to those three until
the class gate is sound again.

**Perceptual space is a subspace of conceptual space (Alec, 2026-10-04;
restated the same day as two spaces, symbol in perceptual space and concept
in conceptual space, see "Reframing, DECIDED" below; the implementation is
unchanged).**
Alec's concern: order-0 concepts mereological, higher orders distributional,
"a bit at odds". Two forms were considered and set aside: bands filled only at
order 0 (higher orders have no form; inside a meaning band nothing keeps
vectors apart) and two scales in one vector, parts as the low-order bits
("sounds a bit kludgy"). Decided in direction: "two subspaces in a single
concept vector ... positing that perceptual space is a subspace of conceptual
space."

A concept vector is [perceptual coordinates | conceptual coordinates]; nothing
is scaled or weighted.

- **Order 0.** The perceptual coordinates are the evidence-weighted fold of
  the concept's parts' percept vectors, in percept space itself (item 11b's
  located fused parts, literally). The conceptual coordinates are the context
  term: the mean of its occurrence rows' roots restricted to that subspace,
  detached; zero for a new word.
- **Identity** has its room in the perceptual coordinates. *Corrected
  2026-10-04 on Codex's qualification ("the separate coordinates provide
  room for identity; the composition rule must actually preserve it"):* what
  the derived code guarantees is the removal of the writer that merged, the
  free row of §13.2; order-0 collapse would require merging letter rows other
  words share, so it is resisted, not impossible; at higher orders the
  normalized product preserves two partners' difference only on the shared
  operand's support (unit(x∘y) = unit(x∘y′) iff y′ ∝ y wherever x ≠ 0), so
  root distinctness is generic, not structural, and is measured (root
  geometry, code support) rather than assumed. The structural guarantee, the
  form band composed by the fold and the connectives on the conceptual
  coordinates, is the operators update's; it requires the gate configs to
  carry a nonempty complement, since with the fold on the form band and an
  empty complement the operators would not reach the root.
- **Similarity** is read on the conceptual subspace; readers choose the
  subspace a metric uses.
- **Higher orders.** Coordinate-wise composition keeps the subspaces
  separate: a root's perceptual coordinates are the composed forms (identity
  residue), its conceptual coordinates the composed meanings. Composition
  fills what a form band could not.

**The fixture.** XOR_grammar's percepts carry two content numbers (6.9
§3.13: "not the four its comment states"); with form confined to the
perceptual subspace, the roots of a context-identical corpus differ in two
coordinates, and four points in a plane are XOR-separable half the time. At
nDim 14, as the comment intended and MM_grammar has, the percepts carry six
and the roots are in general position. The stale-fixture correction goes with
the design. Production has 128 perceptual coordinates.

For Codex, item 1: perceptual space as a coordinate subspace of conceptual
space; order-0 perceptual coordinates = the fold of the parts (live),
conceptual coordinates = the occurrence rows' roots on that subspace
(detached); no free word row; XOR_grammar's nDim 14.

**Confirmed (Alec, 2026-10-04), with the bootstrap.** Concepts with
identical contexts are differentiated in the full vector by their perceptual
coordinates, identical on the conceptual subspace (the permissible collapse).
Higher-order concepts keep perceptual coordinates, the composed forms:
meaningless as a form, but the identity residue that keeps distinct
compositions distinct and what the free read-back inverts. The complement is
determined by context and wholes, and the wholes are load-bearing: context
alone cannot start, since a new word's conceptual coordinates are the mean
of its occurrence roots' conceptual coordinates, compositions of its
neighbours' conceptual coordinates, which begin at zero, and zero composed
with zero stays zero under product and the probabilistic sum. The seed is the
wholes: the conceptual locations of a word's properties and of its situation
(type, document, group), the complement's learned parameters. Words inherit
them, contexts average them. Those locations are trained by co-activation,
the catalogue's distributional item; reconstruction trains the perceptual
subspace and never touches them. In XOR_grammar every word shares every
whole, so the complement is identical across words and identity rides on the
six perceptual coordinates at nDim 14.

**Sphere, origin, bivalence (Alec's question, 2026-10-04; revised twice the
same day, see the two findings below).**

- *Lookup.* Cosine over the stored codes, one matmul, scale-free; the reader
  picks the subspace: identity on the perceptual coordinates (the
  read-back), similarity on the conceptual (priming), both for a whole match.
  Codes are not normalized to a sphere: the perceptual coordinates are the
  percept cube [0, 1] (nothing the origin, everything the all-ones corner, an
  absent letter a known absence), the conceptual coordinates bipolar
  [−1, 1] with zero as unknown. Certainty lives in the activation and the
  poles, not in a vector's norm; the 6.9 catalogue row "unit-sphere codes,
  magnitude = certainty" is retired rather than reinstated.
- *Uncertainty.* Zero, and it stays zero: a new word's conceptual
  coordinates, an absent leaf, the field's neither. It is the fixed point of
  `not` and of conjunction and disjunction with itself under the Kleene
  algebra below; false absorbs conjunction, true disjunction.
- *Bivalence.* The pair (c⁺, c⁻) is Belnap's FOUR over the Kleene algebra
  (Belnap 1977: the catuṣkoṭi's four corners, is / is not / both / neither,
  with the algebra of how compounds take their corner; a bilattice whose
  truth order gives ∧, ∨, `not` and whose knowledge order gives the
  gathering of evidence; Nāgārjuna's negation of all four corners is the
  spec's `non`, a step down the knowledge order toward neither, hence
  without inverse):
  ∧ = (min c⁺, max c⁻), ∨ = (max c⁺, min c⁻), `not` the swap; d = c⁺ − c⁻
  projects FOUR onto K3 with both landing on unknown, the division of labour
  already decided (m for attention, d for expectation). `not` on a location
  is −d, reflection through the origin, which at the symbol level is the
  exchange of the concept's two symbols. Unrelated concepts are far in
  cosine, not antipodal; the antipode term leaves reconstruction (identity
  needs no repulsion once it is in the perceptual coordinates by
  construction).

**The product kernel pools negation (Alec, 2026-10-04: "If 'the composed
location cannot say which constituent was negated', then we have done it
wrong").** He is right. `Ops._conjunction_kernel` is the normalized Hadamard
product of signed carriers: a bipolar Hadamard product is VSA binding (MAP),
self-inverse, its sign a phase and not a truth value, so (−x)∘(−y) = x∘y
("not hot and not wet" composes to "hot and wet") and (−x)∘y = x∘(−y) =
−(x∘y). Any bilinear binding does this; it is the "over symbols" pair of
catalogue §3.8, not a connective over codes, and it cannot host negation.
Two remedies were tried and withdrawn the same day: carrying the valence on
the slot reference (true only of the bilinear product), and Boole's product
in the presence chart p = (1 + d)/2 (a category error: it reads the unknown
as a probability ½, and Boole has no unknown, so 0 ∧ 0 went to −½).

**Zero as uncertainty makes the logic Kleene's (Alec, 2026-10-04: "If zero
is conceptual [un]certainty, we are using a fuzzy Kleene or Church ternary
logic").** Kleene's strong three-valued logic (Łukasiewicz shares its ∧, ∨,
¬), fuzzified: ∧ = min, ∨ = max, ¬ = −d on [−1, 1]. Unknown is the fixed
point of all three with itself; false absorbs ∧ and true absorbs ∨; De Morgan
is exact (−min(−x, −y) = max(x, y)); no pooling (min(−x, y) ≠ min(x, −y)), so
the location says which constituent was negated and double negation across a
conjunction does not cancel; negation stays reflection through the origin,
unchanged. Not an import: componentwise min and max are the meet and join of
the towers' poset (the cube under the componentwise order is a lattice,
nothing its bottom, everything its top; with −d as order-reversing involution
it is a Kleene algebra). `Ops.intersection` is already `torch.minimum` with a
silent coordinate as no restriction (the restrictor reading), `union` the
lattice join. For the operators update: §22's choice of product and
probabilistic sum as the grammar's conjunction and disjunction, made for
invertibility, is revisited; the inverses are codebook searches anyway, and
min inverts the same way (x = r where r < y, x ≥ y elsewhere, the search
deciding). When the connectives become the lattice pair, the identity
residue on the perceptual coordinates composes by the fold, as a word's form
does from its letters (a meet of two forms is their shared letters), and the
connectives act on the conceptual coordinates; the binding kernel does both
at once today, which is fine for this round, which has no `not`.

**One space across orders (Alec's question, 2026-10-04: "does every
conceptual order have a different (shared) conceptual space? ... does order
have any bearing" on the associationist distribution).** One conceptual
space, one shared index; order is a stamp on the row (the fold-provenance
`ramsification` record beside the codebook), not a partition of the space;
[perceptual | conceptual] at every order. Order's one bearing on the
placement is direction: identity from the order below, meaning from the
order above, by the same rule at every rung (order 0: perception's towers
below, rows and properties above; a row: its words below, documents and
situations above). Spreading activation runs across orders over the store's
edges; same-order similarity is the two-step walk word → row → word,
cross-order similarity the content-addressed LTM query of 5.5; one metric,
the reader choosing by purpose. Caveat: the perceptual subspace is form at
order 0 and identity residue above, so identity cosines across orders say
nothing; the meaning subspace is comparable everywhere. Order acts in
minting: the sigma fold makes the co-activating set one order above its
members, a narrowing stays at its order. Precedent: bipartite and spectral
embeddings (Laplacian eigenmaps, the successor representation), words and
contexts in one space, each node at the mean of its neighbours.

**Genera are not located (Alec's worry, 2026-10-04: a mereological subspace
in a concept's code "feels like the philosophical position that genera are
located in space and time, but we have removed their particularity in virtue
of abstracting over space and time").** The address is not in the code. A
percept has content (.what: qualities, the letters and their arrangement) and
an address (.where, .when: this occurrence); the concept's perceptual
coordinates are the interval midpoint over the parts' content only, the
address coordinates zero in the concept code (receipt: "with the type code's
WHERE/WHEN coordinates zero"). That zeroing is the abstraction over space and
time; what remains is a type, and the particular lives in the occurrence row
with its .where and .when (identity is its occurrences). The mereological
subspace of a genus is therefore its structure, not a location: typed parts
in an arrangement (cats have paws; "dog" has d-o-g in that order), Armstrong's
structural universal; the cube's order on content is feature inclusion among
types, while containment among particulars is the .where meronomy, order 0
only, in the address band and the rows. Lewis's objection (methane has
hydrogen four times; a universal cannot have the same part four times) is met
by the idempotent join (a part counted once) plus the located fold (the
arrangement, Armstrong's and Bigelow–Pargetter's relations): parts once,
arranged, no location. The universal is in every instance and at none. A
better name for the subspace inside a concept is *form*.

**Reframing, DECIDED (Alec, 2026-10-04; confirmed the same day): "concepts
exist in conceptual space, and their symbols exist in perceptual space. So
they have a separate mereological subspace and conceptual subspace in virtue
of having both percepts and concepts."** This is the architecture statement;
"one vector, two subspaces" above remains its implementation (the same
object), and the §14 hand-off stands. Saussure's sign: the signifier in perceptual space,
the signified in conceptual space, the shared index their pairing (it now
ties a form to a concept, not an identity row to a code). Consequences: no
form in a concept, so the genera worry above dissolves (the structure is the
symbol's, a percept; the concept has it through its symbol as a kind has a
name); perception composes forms by the fold (letters to words, words to the
cross-word chunks of 6.9 §17) and the grammar's connectives compose
meanings, two compositions in two spaces tied by reference, so the
fold/connective split is the architecture rather than the operators
update's patch; the read-back and byte decoding are perception's (form to
bytes), as implemented in the §13 round; expectation's negative image acts on
the concept face only ("sensation is never subtracted from", accessible mind
§2.6.2 item 1); lift and lower act on concepts and mint a symbol by index;
SymbolSpace is the index joining a form to a concept, without geometry of
its own. In the gate configurations the concepts are empty (identical
contexts, no bootstrap), so the XOR table measures perception's composition
of forms (the binding kernel today, the fold after the update), its inverse,
the affine read and one owner; the connectives over meanings are measured
where meanings exist (MM_xor's field path, whose 11b evidence is content;
corpora with contexts once the complement is seeded): §12.1 made literal.
The §14 hand-off stands: [form | meaning] with per-block operations is the
same object as two aligned vectors in two spaces, and in the gates the
meaning block is empty. The catalogue's footprint rule reads: form =
perceptual space, meaning = conceptual space, poles = the symbol's
activation.

*The conceptual collapse is valid (Alec, 2026-10-04: "We are basically
saying that the conceptual collapse is valid, then?").* Yes: a concept is its
contexts, and words with the same contexts (`hello`/`loving`, `world`/`there`)
coincide in meaning until the corpus separates them; what was invalid was the
symbols (forms, identity) collapsing under a gradient from the conceptual
path. XOR in the gates is therefore a fact about forms, since no word's
context separates the four sentences (each word occurs once with 0 and once
with 1); the sentence rows, one order up, can differ in meaning where the
answer is in their context. No pressure keeps same-context concepts apart:
orthogonality is the outcome for unrelated concepts, not a constraint.

*Confirmed (Alec, 2026-10-04).* An order-0 symbol's perceptual position is
the interval midpoint over its perceptual parts (letters, L) and perceptual
wholes (the WS types read off the input, U), computed by perception and
reached by no conceptual gradient; a higher-order symbol is the fold of its
constituents' forms. An order-0 concept's conceptual position is context
entirely: the rows it occurs in and the conceptual wholes it belongs to (sets
one order up, situation and document codes, properties as concepts), by
co-activation, zero before any, shared by concepts with the same contexts;
nothing enters from below, since below an order-0 concept there is only
perception (Saussure's arbitrariness of the sign). At higher orders meaning
has two sources, composition from below (the connectives over constituent
meanings) and context from above, while identity keeps one (the fold). The
two placements never consult each other; the index ties them.

## 14. Review of the §13 measurement (Claude, 2026-10-04)

Receipt: `doc/benchmarks/2026-10-03-operators-attention/README.md` (review13).
Thirty trainings, no retries, frozen source. **Not accepted**: reconstruction
regressed below its comparison.

| Gate | §12 | §13 |
|---|---:|---:|
| XOR class | 0/10 | 0/10 (9 at ¼, 1 between) |
| XOR reconstruction | 7/10 | **0/10** |
| joint | 0/10 | 0/10 |
| MM_xor | 10/10 | 10/10 |
| sum control | 10/10 | 10/10 |

### 14.1 What the saved tensors show

The tenth run's dictionary collapses onto one ray: `world`, `there` and
`loving` end pairwise proportional (cosines .99999999997, .99999999992,
.99999999999, recomputed in float64), `hello` at cosine .8717 to all three;
the third centered root singular value falls from 1.0e-3 to 2.0e-8. `there`
and `loving` share no letter and still coincide, so the collapse is in the
letter prototypes, which the max fold shares: removing the free word row
moved the collapse onto the fold's inputs. The receipt's own attribution
(shared basis; product erasing differences outside the operand's support) is
not the cause: every word's support is 6/6 at the end.

### 14.2 The writer that merges: the straight-through pair search

`LanguageSpace._bounded_binary_reconstruction` returns
`hard.detach() + (soft − soft.detach())`: the value is the least-residual
pair, the gradient is the soft blend `Σ wᵢ cᵢ` over all K² shortlisted pairs,
`wᵢ = softmax(−residualᵢ / .01)`, with the candidate codes live. The byte loss
pulls the recovered leaf toward the target word's identity, and through the
blend every candidate code receives that same pull, weighted by `wᵢ`. At the
cube codes' scale the weights are uniform: prototypes are initialized at
`embeddingScale` .05, root norms are .004 at start and .023–.047 at the end,
residuals are 1e-5–1e-3, and the absolute temperature .01 makes the softmax
flat. Recomputed from the saved codes over the 16 pairs: maximum weight .0630
at start and .0657 at the end against a uniform .0625, for all four roots.
So every shortlisted code is pulled onto whatever word is being read back,
at equal strength, every step, and the whole dictionary converges to one
direction. The hard pick itself is still right at the end (by the small
magnitude difference), but the read-back by cosine can no longer tell the
merged words apart and priming decides: 52 of 77 evaluation read-backs
decided by priming, 2/4 reconstructed per run. §12's 7/10 was priming
distinguishing two merged second words; the three-way merge defeats it. The
antipode term, removed this round at my instruction, had been the only
counter-force: its removal exposed the writer rather than causing it.

The derived code is not wrong; it was not the writer. This is §13.3 (B), now
located precisely.

### 14.3 The fix: the part codes are perception's (Alec, 2026-10-04)

Alec: "The part codes are different: the concept codes cannot merge." The
premise held at the start and the training broke it: the fold was live to
its inputs, so the sentence read-back's gradient reached the letter
prototypes and the evidence, and `there` and `loving`, sharing no letter,
merged because their parts did. The rule, held structurally:

- The derived code is a function of perception. In the conceptual
  derivation the PS prototypes and the 11b evidence are **detached**;
  perception's codes are trained by perception's reconstruction and by
  nothing the sentence path does. Distinct parts then give distinct concept
  codes by construction, and no gradient from the sentence read-back can
  move a word's coordinates toward another's.
- **The code is the midpoint of its lattice interval (Alec, 2026-10-04:
  "ideally, greater than any part, and less than any whole. So max the
  former, min the latter, and then take a centroid").** For a concept with
  parts P and wholes W (evidence for; against-evidence is not a part or a
  whole and stays in the negative pole), L = ∨ P, the coordinate-wise max of
  the part codes (the least upper bound), U = ∧ W, the coordinate-wise min of
  the whole codes (the greatest lower bound; U = 1, everything, with no
  wholes), and c = (L + U)/2: above every part, below every whole, when the
  interval is nonempty. This replaces the part-group max and the plain
  centroid considered the same day. Percept and concept width coincide in
  the gates (14 = 14), so c is the whole code; in production c occupies the
  perceptual 128 and the complement stays for context.
- **Weighted by evidence (Alec's question, 2026-10-04).** Evidence places a
  code along the ray from its pole: a part with net evidence d counts as
  d·c_p, from nothing up; a whole as 1 − d(1 − c_w), from everything down. So
  L = ∨_p d_p·c_p (the fuzzy union of evidence-scaled parts) and
  U = ∧_w (1 − d_w(1 − c_w)) (its De Morgan dual over the wholes), the dual
  fold the accessible-mind spec names for the two poles, applied to the
  bounds; an element with d = 0 constrains nothing. Negative evidence enters
  by the net, d = relu(e⁺ − e⁻): more against than for is no part and no
  whole. It does not otherwise shape the location (Kleene: what the concept
  is not lives in its negative pole); the both corner min(e⁺, e⁻) is the
  dissonance and stays attention's. Between the bounds the centroid is
  weighted by aggregate evidence, c = (W_P·L + W_W·U)/(W_P + W_W) with
  W_P = Σ d_p, W_W = Σ d_w: still inside [L, U], and a concept with no wholes
  yet is the join of its parts rather than halfway to everything. The room
  rule applies to the weighted bounds.
- **Order 0 only (Alec's two questions, 2026-10-04).** The interval rule
  places perceptual coordinates from perception's two towers, PS parts and WS
  property-wholes; occurrence rows are not in U, they are context and feed
  the conceptual complement. Higher-order concepts are placed by the
  operators (identity residue; composed meanings plus the context mean) and
  the room clamp touches only PS and WS codes: their values follow their
  constituents, no placement rule changes. A conjunction narrows, so its
  compound sits below its constituents (the meet), not above them. No
  symbol-based interval defines higher-order locations: higher orders have
  no form to bound, and the two symbol-defined locations in the design, the
  composed root and the word's mean over the rows referencing it, are not
  intervals. With Kleene connectives composition is itself the lattice
  placement (conjunction at the meet, disjunction at the join).
- **Room between the towers.** The parts' tower (from nothing) and the
  wholes' tower (from everything) are independent, so L > U can happen on a
  coordinate: no room for the concept. Rule: U − L ≥ m coordinate-wise, for a
  margin m. Enforced as the cube's second clamp, after the owner's step: for
  every (concept, coordinate) with v = relu(L − U + m) > 0, the maximal part
  on that coordinate moves down by v/2 and the minimal whole up by v/2,
  clamped to [0, 1]: parts toward nothing, wholes toward everything, "more
  densely into their corners" (Alec). Deterministic like the [0, 1]
  projection already on the codes, so no new writer; the alternative, a
  hinge term Σ relu(L − U + m) on perception's owner, needs a rate and is
  not taken. The receipt reports violations before and after (count, largest
  v) at start and end.
- **Known lossiness of the join.** If one letter's prototype is maximal on
  every coordinate among a word's letters, L is that letter's code, and words
  differing only in dominated letters share L. In six coordinates with five
  letters a letter is dominated everywhere with probability about (4/5)^6 ≈
  .26, so gate words share coordinates (the saved start codes show it); at
  128 coordinates it does not occur. The pair search reads exact residuals,
  so partial sharing costs nothing; a full coincidence would show in the
  geometry files. Denser parts leave fewer coordinates for one letter to
  dominate.
- The pair residual is relative, divided by the parent's mean square (the
  error registry's rule), so the .01 applies to a dimensionless quantity;
  this is scale, not collapse. From the saved start codes the true pair and
  its commuted twin then share the weight (.50 each), as a commutative
  operation should.

Consequence: XOR_grammar's class gate becomes a structural check, four
distinct roots in general position in six content coordinates read
affinely, which any labelling allows, rather than a learning result; the
learning the table witnesses is MM_xor's (the field path), as §12.1 already
said. The earlier form of this section, detaching codes only inside the pair
search and the byte scorer so that reconstruction reached a code through its
root, is subsumed.

### 14.4 Also

- **Kept against the hand-off.** The 256-percept reserve stands (CS inventory
  6 → 262, 8 → 264) as "native percept-concept addresses" with no parameters.
  If the shared index needs a concept row per percept (letters as concepts
  that are referenced and prime), one sentence in the receipt says so;
  otherwise it goes.
- **Legacy residue.** The antipode's reporting key ("an untrained, detached
  zero") and `embed.conceptual_antipode_loss_codes` ("a diagnostic API only")
  are kept to keep two tests alive (`test_echoic_decoder.py`). Delete the
  function, the key and both tests.
- **Dropped from the final probe.** The settled probe ran 14 files (3 failed:
  the antipode test and two output-walk fixtures, since ported); the final
  probe ran 10, dropping `test_decoder_exploration.py` and the
  review10/11/12 contract files without saying so. I ran the four: 34 passed.
  Next receipt: state the list and run the full sweep once, which was asked
  and not run.
- **The decoder has no decisions on this corpus.** 6,400/6,400 first steps
  have exactly one legal action, the policy's weight and bias never move, the
  decoder's explore is 0/3200: on two-word sentences the walk is forced and
  reconstruction is the pair search alone. Expected, not a defect; it means
  the pair search's gradient is reconstruction's entire signal to the codes
  here, which is why §14.2 was total.
- **Scale.** Codes at .05 in a cube whose poles are 0 and 1 are all near
  *nothing*; the cosine read-back is scale-free, the pair search was not.
  With §14.3 nothing in the decode path compares against an absolute number.
- Delivered as asked: nDim 14 in all four spaces; the blend deleted (form not
  attenuated by occurrence count); rows conduct priming (6,392 opportunities,
  0 activated candidates outranking own words); `|cos|` read-back on the
  perceptual coordinates; MM_xor and the control hold.

### 14.5 Forecast, and the decision asked of Alec

Reconstruction: expected 10/10 without learning, since the merging writer is
gone, codes are distinct interval midpoints, the hard pick was right even at
the end of the collapsed run, and the read-back of a hard-picked bank code is
exact. MM_xor and the control are untouched. Class: identity is fixed, scale
is not. Root norms are .004–.05, so fitting labels 0/1 needs reader weights
of order 1/‖Δroot‖ ≈ 10²–10³, and whether they grow within 400 epochs is
what decides ¼ against between; §20.5 would record ¼ as a plateau, but here
it would mean a slow descent. Proposed (pending Alec): the class reader
reads the root at **unit norm**, a fixed parameterless normalization with
the head still affine, since certainty is not in the norm (§13.4); weights
are then O(1) and ¼ can only mean a plateau. The sweep was not run this
round and detaching the derivation will meet tests expecting gradients at
the prototypes: ports. If reconstruction and class come in and the sweep is
green, 6.8 is acceptable as the composition baseline, carrying the bootstrap
and the form-fold/connective split to the operators update and the 2.6×
open-read slowdown to item 1; the MM_ladder cases recorded red since 6.9 §23
are not 6.8's.


Apply §14.3 (prototypes and evidence detached in the conceptual derivation;
relative residual), delete the antipode residue, then the same measurement
(sweep once, sum ×10, XOR ×10, MM_xor ×10, geometry and read-back
annotation), one Codex text. Decided in direction by Alec's statement above.

## 15. Review of the §14 measurement (Claude, 2026-10-04)

Receipt: `doc/benchmarks/2026-10-03-operators-attention/README.md` (review14).
Thirty trainings, no retries, frozen source; the full sweep once. **Not
accepted.**

| Gate | §12 | §13 | §14 |
|---|---:|---:|---:|
| XOR class | 0/10 | 0/10 | 1/10 (8 at ¼, 1 between) |
| XOR reconstruction | 7/10 | 0/10 | 8/10; 10/10 before learning |
| joint | 0/10 | 0/10 | 1/10 |
| MM_xor / sum control | 10/10 / 10/10 | same | same |
| full sweep | — | not run | 4,813 passed, 74 failed, 285 skipped |

### 15.1 What held

Reconstruction is exact in all ten runs at the first costed trial, before
any optimizer step: distinct codes, the right pair, the right read-back, as
§14.5 forecast. The percept prototypes never moved: zero gradient and zero
optimizer displacement in all 800 recorded reconstruction steps (read from
`xor-10/ownership/displacements`). Codes are perception's.

### 15.2 What eroded it: the shared wholes in the midpoint

The word codes converged anyway: pairwise cosines .89–.98 at the start,
.998–.9999 at the end; deviations from the mean code .015 → .004 (norm .07
→ .125); the roots' XOR interaction (r_hw − r_ht − r_lw + r_lt) 1.5e-3 →
1.8e-5; the third centered root singular value fell 100× in every run. With
no gradient at the prototypes, the mover is the derivation's inputs. `U` is
the same vector for all four words (same WS types); the room clamp pushed it
up early (19 violations → 7 → 0; the shared coordinate doubled for every
word, .055 → .12) and pushed parts down; and `W_W·U` is common mode. The
centroid blends the word with its type, and for words of one type the blend
carries no identity. Thinner pair-search gaps let priming flip read-backs in
two runs (8/10); flatter roots left the reader on a plateau in eight (the
reader norms sit at 5.5–7 with ~.01–.15 change over the last fifty epochs;
runs 6, 8, 9 moved to 18, 15, 10, and 8 passed).

**Proposed (Alec's decision; the centroid was his):** the symbol's position
is its form, `L = ∨ d_p·c_p`, the join of its parts. The wholes contain it,
the room rule `L + m ≤ U` acting on the wholes only (a type grows to hold its
members; a form does not shrink to fit), and do not enter the position.
"Greater than any part, less than any whole" still holds.

### 15.3 A second cause: the form's magnitude is the evidence, not the presence (corrected 2026-10-04)

The §14 codes are ~.06 at the start and grow to ~.125 by the end with zero
gradient and zero displacement at the prototypes. The knob I first named,
`embeddingScale`, is the SBOW error weight, not an init; the codebook init
is unit-norm Gaussian rows (`<initScale>` unset in the gates) clamped onto
the cube, so the prototypes already sit at full presence (entries ~.4, half
of them zero). The small scale and the drift are therefore the **evidence
weights**: `L = ∨ d_p·c_p` scales each part by its net evidence, and `d`
accumulates over occurrences. At that scale the probabilistic sum is the
plain sum (`x∘y ≪ x + y`: run 10's disjunction roots have norm .135 = .07 +
.07 − .005), so the disjunction is the control, and the conjunction's
direction is common mode with the XOR term of relative size `(δ/m)²`, about
1% of a root at the start before any erosion. Alec (2026-10-04): the scale
"is a replacement for the constraint that all vectors are unit length (it is
the projection of the incoming vector whose magnitude then embodies
uncertainty)". **Decided in direction:** the form is at full presence; `d > 0`
selects a part, it does not scale it; certainty is the leaf's activation, the
projection coefficient `(leaf·c)/(c·c)` the reader already computes. The 6.9
catalogue row "magnitude = certainty" returns in cube form (its retirement in
§13.4 is withdrawn). Simulated over the four XOR words: dense prototypes give
mean cos(L_i, L_j) ≈ .98 at scale .05 and at scale 1 alike, and a unit-root
XOR term of .03–.05 (conjunction) and .006 (disjunction at .05) or .04
(disjunction at full scale); sparse presences (a letter present on ~30% of
coordinates) give cos ≈ .83 and a XOR term of .25–.35 for both connectives.
Distinct letter sets give distinct joins; sparsity is what makes them far
apart, and it removes the dominance case (the max of presences is their
union). Codex reports `d`'s definition and its range over training.

### 15.4 The unit-norm reader is withdrawn

Codex kept it off the sum control because normalizing an additive root would
break the control's zero contrast: four points of a parallelogram become
four points of a sphere, affinely independent, XOR-separable. That is the
point: the gate and the control share one reader, and what the control cannot
hold is not the affine read. The reader is affine on the raw root for both;
the root's address bands are zero (verified), so the raw root is the form
composition. My proposal in §14.5 was a mistake of the same kind as §11's
nonlinear reader.

### 15.5 Two gradient holes, exposed by the sweep

- **The compose chooser has lost its gradient.**
  `test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser`
  fails at the current tree (passed in 6.9 §23): `operand_order.weight`
  moves, no `mlp.*` parameter does. The relative residual saturates the
  pair-search softmax (true pair ≈ 0, wrong pairs O(1), temperature .01), so
  the straight-through gradient to the root is zero and nothing trains the
  chooser; an explore trial can be kept but moves no parameter. The chooser's
  signal must not depend on the pair search being uncertain. **Decided
  (Alec, 2026-10-04: "Train the chooser"):** when the explore trial's
  reconstruction cost is strictly lower than the greedy's, one step on the
  chooser's operation logits toward the explore's operation (a preference
  term, reconstruction-owned: the chooser learns what decoded better);
  otherwise no step. Not policy credit: both costs are measured, nothing is
  estimated from samples, and the step records a comparison the trials
  already make. The pair search's straight-through blend has no trainee left
  and is deleted; the pair search returns the hard pick. The generate
  policy's own straight-through path in the walk is unchanged.
- **The output owner receives no gradient** in sixteen fixtures
  (`test_prepared_answer_boundary` ×7, `test_trial_policy_ownership` ×4,
  `test_generation_catalog` ×2, `test_output_path_supervised` ×2,
  `test_arithmetic_isolation` ×1: `assert any(p.grad ... for p in
  owners['output'])` false). The XOR reader did move, so the loss is
  path-specific (a zero or detached input to the answer head under §14 is
  the suspect). To be found, not ported.

### 15.6 The sweep and the measured source

74 failures at the frozen pre-addendum source (capacities 262/264): 38 are
`concept inventory exhausted before percept admission`, from the percept
allocation the addendum deleted (fixed in the focused files, unverified by a
rerun); the rest are to be classified as ports where the assertion encodes
the superseded design (a dictionary that is a `Parameter`; competitor-code
gradients; the soft symmetric reverse; a shared-owner `data_ptr`) or as
regressions (§15.5; a `size of tensor a (3) must match (6)` in
`test_compose_review`; a `torch.full` TypeError in the exploration test;
non-distinct leaves in `test_mind_generativity`; two-word decodes in
`test_grammar_separator`). The gates and the sweep were measured on 262/264;
the delivered source is 6/8 and has not been measured.

### 15.7 Asked of Alec

`L` for the form with the clamp on wholes only (§15.2, accepted "if it
ensures differences": it does when parts are presences, §15.3); the form at
full presence with certainty in the activation (§15.3, decided in direction);
the chooser's preference rule (§15.5: the pair-search softmax is one-hot
whenever the search is clean, at any scale, so the straight-through gradient
exists only under confusion; the preference rule records a measured
comparison, not a sampled estimate). Then one Codex text: those,
the output-owner diagnosis, the sweep classified and green on the delivered
source, the same thirty trainings.

## 16. Review of the §15 hold (Claude, 2026-10-05)

Receipt: `doc/benchmarks/2026-10-03-operators-attention/README.md` (review15,
held before measurement). Capacities 6/8. The §15 items are implemented: the
form `L` at full presence with `d` selecting; room on the wholes only; one
raw-root affine reader for both gates; hard-pick pair search with the blend
deleted; the preference term, reconstruction-owned. The output owner's
missing gradient had a cause: the answer path inferred occupied roles from
nonzero values, so a valid zero-valued answer lost its slot and its gradient,
and native numeric inverses went through the lexical bank; fixed with all
sixteen assertions unchanged. Other regressions repaired, ports saved. The
diagnostic sweep before repairs: 4,885 passed, 9 failed; the final 51-file
focused run: 504 passed, 1 failed. No gates run (the campaign refuses
without a green sweep), correctly.

**The one red test is my instruction's error.**
`test_normal_text_reconstruction_updates_the_grammar_chooser` asserts that
any reconstruction batch moves the chooser's MLP, which was true of the
blend's gradient and is not true of the preference rule: in the fixture's
batch (seed 613) the explore departures are STOP or an identity-equivalent
unary, both trials cost the same to the last digit, and nothing should move.
The assertion encodes the superseded design; it is a port, and "must pass as
written" (§15 text, item 4) is withdrawn. **Resolution:** keep the seed and
the batch; assert (a) on this batch the trials tie and the chooser's
parameters do not change, (b) on a constructed strict win the selected logit
moves toward the explore's operation and nothing else (Codex's controlled
check). Not seed-pinning: the tie is the seed's natural outcome and the win
is constructed.

**Structural line from the tie:** a departure that leaves the understanding
numerically unchanged is not an exploration and can only tie; value-identical
actions are excluded from the departure draw. Once every operation
reconstructs exactly the chooser has nothing to learn from reconstruction,
which is correct; it learns where operations differ in decodability.

Then: the green full sweep and the thirty trainings as specified in §15.

### 16.1 Training the chooser by comparison, decided (Alec, 2026-10-05)

Alec: "I think we want the old mechanism that learns on every trial; can you
present again the problem with that method?" The old gradient was the
straight-through mixture's first-order comparison, `∂L/∂logit_o ∝ ∂L/∂root ·
(op_o − root)`, legitimate in principle and degenerate here for structural
reasons: the decoder is a hard, scale-free pipeline (argmin pair, cosine
read-back), so `∂L/∂root` exists only through the pair search's soft weights,
hence only in proportion to the decoder's uncertainty, vanishing when
decoding is right; what it says is "make the root fit the pair I picked",
right or wrong, never "the other operation would have decoded better"; and
its strength is a temperature against a residual scale (uniform at .05, where
it also collapsed the codes; one-hot at full presence with the relative
residual). A temperature change re-enables only the first defect.

**Decided: (c), comparison of measured costs, with two samples.** Of (a) wins
only, (b) symmetric pairwise preference, (c) all alternatives at the explored
round trained toward the measured ranking with ties toward indifference,
Alec chose (c) and asked why two samples would not do: they do. The ranking
loss decomposes into pairwise terms (Plackett–Luce into Bradley–Terry); one
pair per sentence, the alternative drawn uniformly from the value-distinct
eligible actions, is an unbiased estimate (a draw from the chooser's own
softmax would need an importance weight). Per pair: cheaper → its logit up
and the other's down; tie → the two logits toward equality. Ties teach
indifference, so every trial teaches; the price of two over K is variance.
Precedent: ProxylessNAS (Cai, Zhu & Han 2019) trains architecture choices by
sampling two paths per step as a binary comparison; dueling bandits (Yue et
al. 2012) for the incumbent-versus-challenger structure. Value-identical
departures (STOP where both leaves already decode exactly, identity
unaries) are excluded from the draw: they cannot differ and are not
explorations.

**The single-network wish, and its condition.** Alec: "my preference would be
for the gradient to train a single network: is there any literature that
trains in a soft superposition that embeds the chooser ... even for one trial
per sentence, as long as it is randomly sampling?" Yes: DARTS (Liu, Simonyan
& Yang 2019) and Neural Programmer (Neelakantan, Le & Sutskever 2016) fully
soft; Gumbel-softmax / Concrete (Jang, Gu & Poole 2017; Maddison, Mnih & Teh
2017) with SNAS (Xie et al. 2019) and GDAS (Dong & Yang 2019) sampling one
operation per step and backpropagating through the relaxed sample; sparsely
gated mixtures of experts (Shazeer et al. 2017; Fedus, Zoph & Shazeer 2021)
with the gate probability multiplying the chosen branch; Schulman et al. 2015
for the general accounting, REBAR/RELAX for the estimators between. All
require the loss to be smooth in the mixture's output downstream of the
superposition. Ours is not: the hard pair pick makes the loss piecewise
constant in the root, and the cosine read-back is blind to the magnitude a
gate probability would multiply, so the MoE trick is inert. The documented
DARTS collapse (Zela et al. 2020; Chu et al. 2020), the mixture gradient
favouring whatever already dominates, is the mechanism that merged our codes
through the blend. The single-network form is available at the price of a
decoder made smooth in the root (expected byte cost under the pair-search
softmax, annealed relative temperature), which reintroduces the temperature
and weights the signal by the decoder's confusion; with parameterless
operators and perception's codes nothing continuous remains on the compose
side, so the measured comparison is the gradient of the discrete choice. It
returns if operators acquire parameters. The generate chooser's
straight-through in the walk computes every candidate transition (SNAS-shaped)
and stays until it shows the symptom.

For Codex: port the chooser test to the two cases (§16); exclude
value-identical actions from the departure draw; the preference term is
pairwise in both directions as above; then the green sweep and the thirty
trainings of §15.

### 16.2 Decided: the choosers are continuous (Alec, 2026-10-05)

Alec: "We have two choosers, one for words to use and one for derivation
choices. They need to be continuous." Supersedes §16.1's comparison rule and
§15 item 4.

- **The word chooser is already continuous:** a softmax over the primed bank
  of `|cos| × priming / τ`, the byte loss a log-likelihood over it. Its
  learnable part, when wanted, is the learned reader (`PrimedSymbolReader`),
  reconstruction-owned; nothing now.
- **The derivation chooser is continuous when the decoder is smooth in the
  root.** The straight-through mixture already supplies the first-order
  comparison `∂L/∂logit_o ∝ ∂L/∂root · (op_o − root)`; what failed in §13–§15
  was `∂L/∂root`, uniform at .05 (collapse) and one-hot at full presence
  (silence), the temperature meeting the wrong scale. Design: the pair search
  keeps the straight-through (hard value, soft gradient) with the candidate
  codes detached and weights `softmax(−relative residual / τ)` at `τ = 1`,
  the relative residual's own unit (zero for the true pair, order one for the
  wrong ones): never flat, never one-hot, a gradient on every trial, larger
  where pairs are close (a margin loss). Both trials backpropagate; the
  kept-trial rule decides only which derivation is the understanding. The
  comparison rule is dropped; the chooser test passes as written. The sampled
  departure is the counter to rich-get-richer. What the chooser learns: the
  operations whose roots are more uniquely invertible, the true pair's margin
  over the wrong pairs, the first-order form of "decodes better".
- **One network, later:** compose chooses an operation from two operands,
  generate chooses an undo from a root; one scorer `score(o | x, y)` serves
  both, applied backward to each operation's recovered operands. The
  operators update's (FutureWork).

*Codex's note (2026-10-05), answered.* Codex judged §15's strict-win update a
heuristic (correct; dropped) and offered two alternatives: straight-through
Gumbel-softmax on the compose chooser with "a differentiable backward rule"
at the pair-search boundary, codes detached, which is §16.2 as decided (no
Gumbel noise on the forward: the sampled departure is the exploration); and a
contextual gate over the two completed derivations (Jacobs et al. 1991),
declined for the limitations Codex itself lists (a reranker, not the
generating chooser; the minimum of two measured costs already selects).
Accepted as a mechanism check in the receipt: the straight-through gradient's
sign on the chooser's logits against the measured cost difference
`C_explore − C_greedy` per sentence, the agreement rate overall and on
non-ties; a biased estimator validated against the zeroth-order truth, no
training change following from it this round.

*The second trial's role (Alec's question, 2026-10-05: drop the two
alternatives for a hard-choice MoE over derivations and words, learning the
distribution over repeated trials?).* Yes for learning: one hard-chosen
derivation per trial with the soft backward is GDAS/SNAS, and the mixture
gradient already compares every operation at every round to first order,
since all outputs are computed for the superposition. Two clarifications: the
sparsely-gated MoE's router gradient (gate probability multiplying the
expert's output) is identically zero here because the decoder is scale-free,
so the hard-choice MoE must be the straight-through kind, which §16.2 is; and
sampling from the chooser's own distribution has no coverage floor, so its
order effects are the routing collapse MoE counters with load balancing. The
second trial is therefore not the learning mechanism but three other things:
coverage independent of the chooser's confidence (the departure drawn
uniformly over eligible alternatives, ε-greedy rather than Boltzmann); the
commit (the understanding written and answered from is the better of two
actual derivations); and the exact zeroth-order check against which the
biased gradient is validated. Its 2× cost is an efficiency lever for item 1
(the departure on a fraction of sentences, the ergodic schedule) without
changing what trains. Words: soft in training (a likelihood over the bank),
hard in production; nothing to change.

*Switch Transformers and stochastic computation graphs (Alec's question,
2026-10-05).* Switch (Fedus, Zoph & Shazeer) confirms hard top-1 routing with
a softly trained router at scale; its router gradient (gate probability
multiplying the expert's output) is zero for our scale-free readers and would
be output's path for the one reader that sees scale, so not ours. Its
pathologies are: routing collapse, cured there by a load-balancing term and
here structurally by the uniform departure (a coverage floor); router
instability, cured there by a z-loss bounding the logits and here by the
anchor-dot scorer's bounds given normalized inputs (one audit check).
Stochastic computation graphs (Schulman et al. 2015) are the formalism of our
trial: the pathwise estimator (differentiate through relaxed nodes; biased;
§16.2's straight-through with the τ = 1 boundary) and the score-function
estimator (∇log p(a)·(C − b) at the stochastic node; unbiased; needs no
differentiable decoder). With the departure drawn uniformly and the greedy
cost as baseline, the score-function estimator is the two-trial comparison
of §15 item 4 (self-critical sequence training, Rennie et al. 2017): not a
heuristic but the unbiased member of the family, dropped for its variance
and its silence on ties rather than for lack of standing. The choice is bias
against variance; both estimators come from the same two trials. Decided:
§16.2 stands (dense, every round, every trial); Codex's agreement check
measures the straight-through's bias; the score-function term with the greedy
baseline is the named correction if the agreement is poor (REBAR/RELAX the
principled combination if ever needed).

### 16.3 Decided: the chooser's gradient is the score-function estimator (Alec, 2026-10-05)

Alec: "So we are doing SCG?" ... "I'm happy that we are following the
literature more closely here." Supersedes §16.2's straight-through and §15
item 4's fixed step. In the trial's stochastic computation graph the
chooser's parameters enter only through which operation is chosen; the
root's value does not depend differentiably on the logits. The exact gradient
of the expected cost with respect to the chooser is therefore the
score-function term at the departure node, with the greedy trial's cost as
the paired baseline (self-critical sequence training, Rennie et al. 2017):
unbiased, trained by backprop on a surrogate loss, needing no differentiable
decoder and no temperature. Because the departure is drawn **uniformly** over
the value-distinct eligible alternatives (the coverage floor), the importance
weight against the policy turns `∇log p` into `∇p`: the surrogate is
`p_θ(a_dep | state) · (C_explore − C_greedy)` with both costs detached, one
term per sentence with a departure, `Σ_a b·∇p(a) = 0` keeping the baseline
unbiased. A cheaper explore raises `p(a_dep)`, a dearer one lowers it, a tie
teaches nothing (there is nothing to prefer); the greedy trial, being the
argmax, contributes no chooser gradient. Costs: credit at one round per
sentence (the departure's, uniformly drawn, so every round over sentences);
variance borne by the paired baseline; silence on ties, which in the XOR
gates means the chooser stays where it started, correctly. The pair search
stays hard (§15), the straight-through blend stays deleted, the agreement
check of the reply to Codex's note is withdrawn (no biased estimator remains
to validate), and the chooser test is ported to two cases (tie: no movement;
constructed non-tie: movement toward the cheaper derivation). The dense
pathwise signal, if ever wanted, enters as a control variate (REBAR/RELAX),
not as a sum. Forecast for the gates unchanged from §15.7/§16: reconstruction
10/10, MM_xor and control 10/10, class a majority at 0, ¼ with distinct roots
and a flat reader norm being the reader's convergence (output's), recorded.

*Hard inverse, hard forward (Alec's question, 2026-10-05).* The hard inverse
is the pair search returning one least-residual pair as the recovered
operands, no blend. The forward is hard too, verified in the operation layer:
one action per round (argmax or the sampled departure), the committed value
`chosen = candidates.gather(action)`, the chooser's probability recorded
beside it ("credit always uses the model's own distribution over legal
actions"); compose's soft superposition is the CKY-chart era retired by 7.5.
The one soft backward left is the generate walk's straight-through over its
candidate transitions, for the decoder's own policy. The asymmetry is in
parameters, not hardness: the forward choice has them (the chooser) and is
sampled, so its exact gradient is the score-function term, and the recorded
probability is the surrogate's `p(a_dep)`; the inverse pick has none (an
argmin over measured residuals against detached codes), so a soft backward
could only manufacture a gradient to the root, which nothing on the compose
side needs.

### 16.4 The inverse has a chooser (Alec, 2026-10-05; in this round)

Alec: "The inverse should have a chooser also." It chooses a decomposition:
which operation to undo and which candidate pair (or single word) the root
came from, replacing the argmin residual and the fixed `activation × cosine ×
priming` with a learned score over candidate decompositions whose features
are the relative residual under each operation's inverse (fit), the
candidates' activation and priming (echoic context) and the sentence's end
state; its softmax is the decoder's word chooser. The §11.6 eligibility rule
stays as the legal mask. **Trained by cross-entropy toward the truth:**
reconstruction's target is the input, so the true decomposition is known
(the input's words; the operation the compose trial applied), teacher forcing
with free inference; exact, dense, continuous, no sampling or baseline. The
forward chooser needs the score-function estimator because it has no target,
only cost; the inverse chooser has a target. Effects: the echoic prior gets a
learned weight (the §24 collapse was a fixed `× priming`); the two choosers
close a loop (the forward learns by cost which operations the inverse can
decode, the inverse learns by truth to decode them, explored operations
included). Codes detached; reconstruction-owned; output's backward cut at the
understanding. **Separate parameters from the forward chooser:** sharing
would make the inverse's cross-entropy toward compose's own choices train the
forward to prefer what it already does, with no cost in the loop (withdraws
the "one scorer" unification of §16.2 in that form; the duality holds in what
is learned, not in shared weights). Placement: recommended as the operators
update's first item, with the walk policy (undo / unary / STOP) trained the
same way toward the compose derivation's structure; the gates do not need it.

*Placement decided (Alec, 2026-10-05: "No? Because it's really a
reconstruction, not an exact inverse").* The operators are lossy, the pair
search is a search over a shortlist, and context is already in the candidate
set: an inference under uncertainty, where a learned chooser belongs. In this
round: the decomposition chooser over the shortlisted pairs for the selected
undo (features: fit, activation, priming), initialized to reproduce the argmin
(fit weight 1, context weights 0) so reconstruction stays exact at step 0,
trained by cross-entropy toward the true pair at each undo, teacher-forced,
reconstruction-owned, codes detached; absent target (true word not
shortlisted) counted, not trained. The walk policy keeps its straight-through
and mask this round; its cross-entropy toward the compose derivation is the
operators update's. Addendum text given to Codex.

*Sequence (Alec, 2026-10-05: "this round. Let's let Codex finish its tests,
and then make the change").* §16.3 first (score-function term, chooser test
port, repairs) through to the green full sweep; then §16.4 (the decomposition
chooser, items 9–12) and the sweep green again; then the one measurement on
that frozen source. One campaign.

## 17. Review of the §16.3 round (Claude, 2026-10-05)

Receipt: `doc/benchmarks/2026-10-03-operators-attention/README.md` (review16,
measurement held). Landed and checked: the score-function surrogate
`p(a_dep)·(C_explore − C_greedy)` with detached costs, uniform draws over
value-distinct eligible alternatives and rounds, ties contributing nothing,
the §15 strict-win term deleted; pair search hard, byte-scorer bank detached,
walk unchanged, zero sentence-path gradient at prototypes and evidence.
Gradient equals `ΔC·∇p` to 1.9e-9 and central finite differences to 1.2e-9;
logits bounded; zero ownership conflicts. The chooser test carries the two
controlled-cost cases; the natural batch, once ineffective departures are
excluded, no longer ties (both departures `non`, ΔC ±.3). Sweep 4,900 passed,
0 failed, 1 non-strict XPASS; the first sweep's two failures were the
superseded straight-through's own assertions, ported.

**The XPASS** (`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`,
`xfail(strict=False)`): verified unseeded, 5 of 8 runs pass and 3 fail at
overlap .5, so the mark still tells the truth; the campaign guard treating a
non-strict XPASS as a failure contradicts pytest's semantics and the mark's
own reason. Resolution: the guard counts non-strict XPASS as pass; the mark
stays; the .5 runs are the join's dominance lossiness at that fixture's small
content width (§14.3), the operators update's.

**The estimator's scale:** `p·ΔC` without the sampling factors (Codex said
so). Multiply by `K` (value-distinct eligible alternatives at the departure
round) and `R` (eligible rounds in the sentence) so the surrogate equals the
baseline-subtracted policy gradient exactly; a normalization, not a tuning.

Then §16.4 items 9–12 as sequenced, the green sweep, the one measurement.
