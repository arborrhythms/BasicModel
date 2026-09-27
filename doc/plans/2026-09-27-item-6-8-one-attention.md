# Item 6.8: one attention — brackets, narrowing, and expectation at every bracket

**Status.** Decided in direction by Alec, 2026-09-27; plan by Claude.
Codex implements landing 6.8-1 after item 7.5 and item 7 have landed and
after the conference checkpoint is frozen; Claude reviews before commit.
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
- **7 supplies the terminal encoding.** The seal makes the sentence
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
seal; STM; the tied reconstruction; the field's folds; the where registry
and the two-rung ladders; every 9b contract.

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
  generate face; their reverse is the seal's bracket).
- [Accessible mind spec](../specs/2026-09-20-accessible-mind-subsystems.md):
  §2.2–2.4 restated as bracket widths; §4's table gains the bracket
  candidates' row.
- [NanoChatGrammarPilot](../NanoChatGrammarPilot.md): the gate is scored by
  the word-level predictor.
- [LLMEquivalence](../LLMEquivalence.md): condition 2 notes the word-level
  predictor as the discrete distribution.
- [FutureWork](../FutureWork.md): 6.8-2, already recorded.
- [Testing](../Testing.md) and a receipt under `doc/benchmarks/`.

## 6. Questions for Alec (none blocking 6.8-1) — 6a and 6b answered 2026-09-27

**6a, decided (Alec, amended 2026-09-27).** Some operators require a
**single prominent symbol** to operate on. **All Boolean operators** —
`and`, `or`, `not` — **operate on a field of concepts at one time**: a
plural bracket is aggregated by them directly (a union of the field's
concepts is a temporary whole in conceptual space) and no separate
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

**6c, decided (2026-09-27).** Before the conference freeze the NanoChat
gate is scored by the existing `IntraSentenceLayer` held prediction, an
untrained point predictor in idea space used only for evaluation, stated
as such; the trained, distributional word-level expectation is part of
6.8-1 after the conference and does not reintroduce the retired
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
