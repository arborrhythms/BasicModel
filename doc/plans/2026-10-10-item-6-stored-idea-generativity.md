# Item 6: stored-idea generativity

Codex builds; Claude reviews ([todo](../../todo.md) item 6). The item's
direction is in the todo text (unmixing by type family; Alec, 2026-10-09).
The identity-from-data amendment (Alec, 2026-10-10) is item 6's second part,
recorded in §2 below and linked from the todo item; the toy for it is in
[`doc/benchmarks/2026-10-10-identity-ica-toy/`](../benchmarks/2026-10-10-identity-ica-toy/README.md).

## 1. Review of the October 10 candidate (Claude, 2026-10-10) — not accepted as item 6's closing; a mechanism landing after three repairs

Reviewed: the uncommitted working tree, the
[receipt](../benchmarks/2026-10-10-item6/README.md), its sweep receipts and
[`gates/results.json`](../benchmarks/2026-10-10-item6/gates/results.json);
the 17 new tests were rerun here, and the three sweep failures were read
from the worker receipts.

### 1.1 What the candidate is

- **One inverse menu** (`bin/Generative.py: inverse_menu`) now serves both
  the tensor output walk and the stored read (`MemoryIndex.unfold_idea`).
  The stored read previously used the affine, reference-free inverse; it
  now does the bounded candidate search through the forward kernel the
  todo asked for. Every binary candidate is checked by recomposition.
- **Type families** (`primed_type_families`): the primed rows whose
  identities are in `components.nouns.ids` are family A, in
  `components.verbs.ids` family B, other higher-order primed rows are the
  property family. Membership is the learned dictionaries'; no word or POS
  table. The S inverse admits A/property on the left and B on the right,
  VP the reverse, adverb B/property; words are never restricted.
- **Live wholes** (`LiveConstituents`): during a sentence, a detached
  ring buffer of composed values (support ≥ 2) the size of the eight-space;
  created at sentence start, dropped at sentence end, never written to LTM.
  The stored read sees only the type families — structure only, as decided.
- **Completion contract**: STOP and pair completion require recomposition
  within the reader's relative 1e-4; otherwise the result is explicitly
  incomplete.
- **Determiners**: in the stored read, a point whose order stamp is one
  below a primed row at the same point gets a determiner expansion; the
  marker word is ranked by the compose chooser over every bank word (no word
  map); a stored filled referent licenses the definite alternative
  (`stored_binding`). The grammar gains a declared definite *reverse* rule
  (`complete.grammar` 238–239) with its own policy key.
- **Side work**: the `Taxonomy.concept_reference` retrieval error is
  repaired with a regression; b's coverage is expanded by construction (32
  documents, histories 1–4); a lift activation leaking from a stored read
  into an episode's state is repaired with a deterministic probe.

The measurement protocol is the one the todo asks for: forced (raw and
bounded) against free recovery, by chain length 1/2/3/5, unseeded, no
retries, predeclared stop conditions, forward artifacts only.

### 1.2 Measured (from `gates/results.json`)

| operator | words | candidates | forced raw | forced bounded | free |
| --- | ---: | --- | --- | --- | --- |
| lift | 1 | either | exact | exact | exact |
| lift | 2 | either | wrong | exact | exact |
| lift | 3, 5 | word bank only (a stored idea) | wrong | wrong | **incomplete, nothing emitted** |
| lift | 3, 5 | + live wholes (during a sentence) | wrong | exact | exact |
| lower | 1 | either | exact | exact | exact |
| lower | 2, 3, 5 | either | wrong | 2 exact, 3/5 wrong | **one word emitted, reported complete** |

The lift policy fits (CE .000997 at update 3,983); the lower policy
plateaus at CE .418. The A/B dictionaries are untrained in this
measurement, so the family mechanism contributes nothing to it; the receipt
says so.

### 1.3 Findings

**A. From a stored idea, compound recovery stops at two words.** With the
word bank alone, lengths 3 and 5 are incomplete with nothing emitted; the
exact recoveries at 3 and 5 need the live wholes, which only exist during
the sentence whose wholes they are. That is the sentence-time inverse
finding its own forward tree in the candidate set, not generativity from a
stored idea. The item's objective is therefore not yet met, and the receipt
is right not to claim it. What is missing is the item's own direction: the
*unmixing* — projecting the stored idea on the A and B columns to propose
its NP and VP — is not implemented; the families only filter a candidate
menu drawn from priming heat, and with untrained dictionaries the menu is
the word bank. The amendment's lessons are what train those dictionaries, so
this measurement belongs to part 2.

**B. Dropped-phrase readings are still reported complete.** For `lower`
the free decoder emits the head word and declares the compound complete at
2, 3 and 5 words (`free.complete = true, exact = false`). The completion
contract cannot catch a projecting operator, whose output *is* its head
operand within tolerance. This is exactly the 6.2 observation in the todo
(`three plus one is four.` kept with the bare `three`); the receipt's
"emits the final word in every compound case" understates it, and it
concedes the keep cost does not charge it. The order-stamp expansion exists
only in the stored read (`unfold_idea`), not in the tensor walk.

**C. A regression, called unchanged.** The cleared-cache owned inverse
returning no indexed word first appears in the sweep that introduced the
lexical STOP mask (the development table goes from 2 failures to 3 at that
step) and persists in the final sweep. The receipt lists it among "the
unchanged failures"; it is new with this candidate. Likely cause: after the
cache is cleared the primed rows have no spelling, so `terminal_valid`
masks every STOP.

**D. A new test passes in the sweep by accident.**
`test_unnamed_primed_concept_cannot_complete_as_a_word` fails here under
the project's default backend (`MODEL_COMPILE` unset → eager capture):
`torch.while_loop` rejects input-to-input aliasing between two captured
tensors. With `MODEL_COMPILE=none` it passes. In the sweep it took 0.01 s,
i.e. ran with capture already disabled by an earlier module in its worker.
The fixture builds the bank, root and constituents as views of one tensor;
production buffers are separate (`LiveConstituents` copies), so I read it as
a test-isolation defect — use the `eager_reading` fixture or distinct
tensors — but Codex should confirm no production pair of walk inputs can
alias.

**E. The order lesson still does not move.** The initial-source curriculum
reports the enabled lesson CE at .3465735912 and .7945134640 — identical
to 6.1's landing values and identical first → last across 64 epochs. The
gradient is not reaching the scorer (zero, not small — Adam cannot help);
"fails its unchanged threshold" understates this. Open since 6.1 §7.15; not
this candidate's regression, but still unrepaired.

**F. Unmeasured exit criteria.** The MM_20M grammar recovery assertion is
capacity-blocked (LTM 1,024 exhausted in epoch 2) and gives no rate; the
list curriculum's free recovery is 0/4 in every mode and length and was
measured only on the initial source. The compiled retention failure seen
once (64 of 312 elements, max .2556) has no cause.

**G. Smaller.** `LanguageSpace.reverse_inverses` has no production caller
left (test stubs only) — delete it. `test_output_walk`'s materialisation
test now accepts "emitted a word or truncated", which no longer tests
realisation. The definite reverse rule adds a second *declared* referent
mode; under the amendment these modes are retirement targets, so nothing
further should be built on them.

Verified sound: no seed pinning anywhere; the live snapshot is detached,
bounded, invocation-scoped and never stored; stored reads carry no trace;
`preserve_operator_activations` restores diagnostics after candidate
evaluation; b's expansion is by construction; the receipt's honesty about
its own nulls.

### 1.4 Recommendation (Alec to decide)

Not item 6's closing. The shared bounded inverse, the family filter, the
live snapshot and the stored-read determiner are sound mechanism and can
land as *item 6 part 1* once three repairs are measured in one pass:

1. **B** — a reading that drops a constituent must not be complete:
   either the order stamp (or the eight-space's support count) marks a
   projecting operator's output as not yet terminal in the tensor walk as it
   does in the stored read, or the kept derivation is charged for the
   dropped constituent. Measure: `lower` at 2/3/5 reports incomplete or
   exact, never complete-wrong.
2. **C** — the cleared-cache inverse recovers a word again, or the mask's
   rule is changed so an unspelled-but-indexed row can still end a walk.
3. **D** — the test runs under the default backend.

Plus **G**'s deletion. **A** is not a repair: it is part 2 — the unmixing
through trained A/B columns, trained by the amendment's lessons, measured
by the same fixed-artifact protocol with trained dictionaries. **E** and
**F** stay on the list of open correctness work alongside item 6.

### 1.5 Alec's decisions (2026-10-10)

1. **Part 1 completes directly after the repairs.** No further review
   round: the three repairs are measured in one pass and the landing
   follows (commit, push, bump WikiOracle, todo Done for part 1).
2. **What is minted** (Alec asked: the residual with respect to the
   expectation, or the ICA residual?). Both, composed — the spec's
   "unexplained surprise" ([6.5 §2.4](../specs/2026-09-26-independent-components.md):
   "a new column is minted when the surprise cannot be explained by
   existing columns"): the part of the expectation residual that the
   existing columns of that dictionary cannot explain. Witnessing (binding)
   still reads the whole row's explanation, so `the cat` re-witnesses the
   cat column; minting reads only the unexplained remainder of the surprise,
   so a known cat in a new sentence cannot seed a second cat column. Where
   there is no prediction yet, the surprise is the row. Today
   `IndependentComponents.observe` mints `F.normalize(value)`, the whole
   observed value (line 162), and `encode` selects a fixed top-k at the
   ceiling (line 99), which forces k columns and distorts the remainder;
   the toy's runaway (Q1: .02 identification, ~450 columns) came from the
   two together, and minting the remainder with greedy selection — add a
   column only while it lowers the remainder — removed it (.95). This is a
   6.5 mechanism change and goes into part 1's pass.
3. **No declared constituent order.** The family-to-side map in
   `inverse_menu` (S: A/property left, B right; VP: B left, A right;
   adverb: B, property) is English SVO, not UG, and is retired. The
   grammar's declared operands stay (an S rule has a noun-family and a
   verb-family operand); which operand is left or right in the eight-space
   is learned from the corpus, as reading order was in 6.1 (§7.13–§7.14:
   taught by lesson, never wired). The candidate search admits any family
   on either side; the asymmetric forward kernel's recomposition check and
   the generate policy's credit decide the orientation, and stage 4 of the
   curriculum ("order tells the roles") teaches it. If pruning is ever
   needed, a learned per-rule side preference initialized uniform — never a
   declared one.

### 1.6 Hand-off to Codex

> Item 6 candidate reviewed (plan §1): not accepted as the closing; it
> lands as **part 1** directly after one measured repair pass (Alec,
> 2026-10-10). Four repairs, one sweep, one receipt:
>
> 1. **Dropped constituents are never complete.** A free reading that emits
>    the head of `lower` — or of any operator whose output equals an operand
>    within tolerance — must not report complete: carry the stored read's
>    order-stamp condition into the tensor walk, or charge the dropped
>    constituent in the keep cost. Remeasure `gates/` so `lower` at 2/3/5 is
>    incomplete or exact, never complete-wrong.
> 2. **The cleared-cache inverse regression** (introduced with the lexical
>    STOP mask; sweep failures went 2 → 3 at that step): recover a word
>    there without relaxing the assertion.
> 3. **`test_unnamed_primed_concept_cannot_complete_as_a_word`** fails under
>    the default backend (`torch.while_loop` input-to-input aliasing); it
>    passed in the sweep only because an earlier module in its worker had
>    disabled capture. Fix the test's isolation and confirm no production
>    pair of walk inputs can alias.
> 4. **No declared constituent order.** Retire the family-to-side map in
>    `inverse_menu` (S/VP/adverb). Families remain candidate types; any
>    family is admitted on either side; orientation is decided by
>    recomposition and the generate policy, and taught by the corpus. No
>    English order anywhere in code.
>
> Also in the same pass, the 6.5 mechanism change decided in §1.5(2):
> `IndependentComponents` mints the normalized *unexplained remainder of
> the surprise* (expectation residual minus its explanation by the
> dictionary's existing columns; the row itself when there is no
> prediction), not the whole observed value; witnessing keeps reading the
> whole row's explanation; `encode` selects greedily — a column is added
> only while it lowers the remainder — instead of a fixed top-k at the
> ceiling. Keep the recurrence gate. Add a regression: a known column
> re-witnessed in a new sentence does not seed a pending prototype.
>
> Delete `LanguageSpace.reverse_inverses` (no production caller). Do not
> extend the declared referent modes; they are retirement targets under
> the identity-from-data amendment (todo item 6, part 2). Report the sweep
> against the accepted 6.1 baseline, naming any failure that is new with
> the candidate as new. Then land: commit, push, bump WikiOracle, todo Done
> for part 1 with the open findings (stored-idea recovery beyond two words,
> order lesson CE, MM grammar recovery, compiled retention) carried to
> part 2 and the alongside-item-6 list.

## 2. Identity-from-data amendment (Alec, 2026-10-10)

*One mechanism for noun phrases and identity.* Identity should be learned
from the data wherever the data can teach it, with rules kept only where a
lesson shows they are needed. [6.5's](../specs/2026-09-26-independent-components.md)
unmixing is the one mechanism for both noun phrases and identity:

- **Noun phrases.** A constituent is a noun phrase to the extent the noun
  dictionary explains it sparsely: one `A` column plus the property family.
  This is §2.7's credit-not-lookup routing, raised from words to constituents.
- **Identification.** An occurrence is identified as the column that explains
  it. Binding is the reuse of a column that explains the occurrence within the
  sparsity budget; minting is a recurring unexplained residual (§3.5).
- **Candidates** are the eight-space's contents plus the cued frames; nothing
  positional.
- **Pronouns.** A pronoun carries almost no content, so its choice among
  candidates is credited by what follows: the binding under which the next
  sentence is better predicted is kept (the downward coupling through the
  closing).
- **Determiners** are learned cues on that choice (2026-09-28), not declared
  modes.
- **Same-kind individuals.** Two individuals of one kind with the same content
  cannot be told apart by content; the determiner and later distinguishing
  properties are the only evidence.

*Teaching data designed for the learner.* The learner is linear ICA with a
sparsity prior, so [Training.md](../Training.md)'s stage 1 and stage 5
lessons are built to its recovery conditions:

1. Vary independently whatever must stay separate: every property with several
   objects, every object with several properties, every verb with several
   subjects. A confound becomes one column.
2. Grow the support: one object per sentence, then two, then three.
3. Pronoun lessons have two candidates with recency, grammatical role and order
   counterbalanced, so only content the follow-up predicts decides
   (`the cat saw the dog . it barked .` / `… it purred .`, each candidate in
   each position equally often).
4. Determiners are reliable but imperfect cues (`a` with a first mention, `the`
   with a re-mention); a held-out set has the determiner conflict with the
   content.
5. Same-kind pairs (`a dog runs . a dog sleeps .`) include lessons where later
   properties distinguish the two individuals.
6. Document context varies the distributions (stage 7): the same words with
   different referents across documents.

Two controls: a follow-up predicted by neither candidate leaves binding at
chance; training on a recency-biased set and testing on the reversed set shows
what a positional shortcut looks like.

*Rules the data may replace.* Each rule below is retired only when its lesson
passes without it; a rule whose lesson fails without it stays, with the
failure reported. A retired rule is removed, not kept beside its learned
replacement. Source locations below identify the reviewed candidate.

- The declared referent modes on the determiner rules (`referent="mint|bind"`,
  `data/complete.grammar` lines 170–171 and 238–239). Bind versus mint becomes
  the unmixing's outcome; `kind` stays as an order difference (no lowering).
- The pronoun's separate `bind` rule and reference kind (`I2:pronoun`, line 177).
- `ContextualBindLayer`'s positional path (bind to the nearest constituent on
  the left), live in compose's eager forward (`bin/Language.py:7426`).
- `resolve_operand`'s mode masks and cosine argmax
  (`bin/ReferenceContext.py:195`); candidates are scored by the dictionary's
  explanation plus prediction credit.
- `IndependentComponents.commit`'s admission keyed on the selected mode (only a
  selected mint advances recurrence; a kind scope cancels the individual);
  admission is by recurring residual alone.

*Order and measurement.* This is item 6's second part, after the stored-idea
inverse: one landing, one measurement. Build the corpus and its measurement
before changing any mechanism: binding accuracy by candidate count, recency
and role; one row versus two for stage 5's examples; the determiner-conflict
items; the two controls. Claude runs a numpy toy of four questions (factorial
versus confounded recovery; growing the support; pronoun binding by
prediction; same-kind individuals) in
[the toy directory](../benchmarks/2026-10-10-identity-ica-toy/README.md);
rule retirement waits for it.


## 3. Part-1 repair response (Codex, 2026-10-10)

The [one-pass receipt](../benchmarks/2026-10-10-item6/part1/README.md) records
§1.6's implementation, affected/default/compiled checks, fixed-artifact gates
and the single frozen-source full sweep. Completion is audited after decoding:
missing support is charged in the actual keep objective, including when the
free-byte audit is reporting-only. The raw walk's termination remains separately
reported, so a lower head alone does not masquerade as recovery of the whole.
Self-reproducing candidate pairs are excluded before ranking, captured walk
inputs have independent storage, and family-to-side masks are removed. The
6.5 repair mints unexplained surprise and selects support greedily. Recurrence
and the declared identity rules remain for §2's controlled retirement lessons.
`reverse_inverses` is deleted. The materialization fixture requires exact text
and completion for a representable answer.

The historical review above is preserved, including its diagnosis and the
later decision replacing the initial three-repair recommendation. The cache
diagnostic retained lexical spellings; the repaired obstruction was candidate
self-pairs. Its earlier empty inverse is explicitly classified as new relative
to accepted 6.1. Stored recovery beyond two words and the order-lesson, MM and
compiled-retention findings continue under part 2 and the alongside-item-6 list.
