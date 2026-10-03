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
