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

### 1.7 Code review of the part-1 landing (Claude, 2026-10-10; `324b84d67` + `0b052d5ef`)

Reviewed the landed source against the reviewed candidate's archive
(`doc/benchmarks/2026-10-10-item6/source.tar.gz`) and the
[part-1 receipt](../benchmarks/2026-10-10-item6/part1/README.md); reran the
stored-idea, component, cleared-cache and scope tests here under the default
backend (`MODEL_COMPILE` unset): 61 passed, 1 xfailed, 1 failed (the new
scope failure below).

**All five hand-off items are in the code as decided.**
1. Dropped support is charged: `Generative.reconstruction_coverage` audits
   emitted against expected leaf *counts* after the walk, marks the
   reconstruction truncated and adds `reconstruction.coverage` to the keep
   cost (`Models.py` ~12247, ~12299, ~20081). `lower` at 2/3/5 now reads
   incomplete with costs .5/.667/.8. The walk itself still terminates on the
   head (`walk_complete`); the charge is post-hoc, which the hand-off's "or"
   allowed.
2. The cleared-cache empty inverse was a self-pair, not the spelling mask:
   `_bounded_binary_reconstruction(require_progress=free)` now rejects any
   pair in which a child equals the parent within tolerance before ranking
   (`Language.py` ~13847). The drivability test passes; the receipt corrects
   its earlier "unchanged" claim.
3. The walk clones every captured bank, mask and depth (`own()`,
   `Models.py` ~13047), so overlapping views cannot alias across
   `torch.while_loop`; the test enables capture explicitly. Passes here.
4. The S/VP/adverb family-to-side map is gone; families are carried as
   candidate types only; a 16-pair, both-orientation regression exists.
5. `SparseDictionary.encode` is greedy (a column is kept only while it
   lowers the remainder); `observe` mints the normalized remainder of
   `value − prediction` under the existing columns, witnesses from the whole
   row; `commit` passes the prediction's noun frame through the same chart.
   `reverse_inverses` and its stubs are deleted; the materialisation test
   now requires a supported word, exact text and no truncation.

**Findings (follow-ups, not a re-review):**

- **H. New regression from repair 1 — fix now.** The coverage charge is
  added outside the candidate-availability mask (`Models.py` 12276:
  `cost = cost + (missing + excess) / n_target`), so a packed sentence with
  no candidates is charged 1.0 and dilutes the owned sentence's cost —
  `test_missing_packed_sentence_does_not_dilute_the_owned_reconstruction`
  (new failure vs 6.1). One line: gate the term by `eligible`
  (`torch.where(eligible, …, 0.)`); such a sentence is already flagged `bad`.
- **I. Regression 6.1 → item 6, behind an xfail marker.**
  `test_topk_recovered_words_overlap_input` XPASSed at accepted 6.1 (overlap
  ≥ .8) and observes **0.000** in every item-6 sweep from the first candidate
  on: after a cache clear the inverse emits a word (drivability passes) but
  not the sentence's words. The receipt reports it as adverse; it needs a
  bisect across the candidate's changes (shared menu, terminal mask,
  progress rule) rather than carrying.
- **J.** The audit is count-based: a reading that drops one modifier and
  emits one spurious leaf has the expected count and is "covered"; only the
  lexical score then catches it. Acceptable for part 1; content coverage
  belongs with the derivation reconstruction in part 2.
- **K.** With the progress rule, a projecting operator can never be a free
  pair in the tensor walk (one child always equals the parent), so
  determiners are recoverable only through the stored read's order-stamp
  path. Consistent with the measurement; note it so part 2 carries the
  order stamp into the walk rather than rediscovering this.
- **L.** `constituent_families` is computed and threaded through
  `SentenceUnderstanding` and the walk but now has no consumer. It is part
  2's input; it should not stay dark longer than that.
- b at 1.101915 against the 1.1 bound is the documented flaky finding.

Disposition: the landing matches the authorization. H is a one-line
follow-up commit; I is the one open regression to bisect before part 2's
measurements are read against this baseline.

### 1.8 Follow-ups H and I (Codex, 2026-10-10)

H now applies `eligible` to the coverage charge as well as lexical cost.
The unchanged missing-packed-sentence regression and all 24 scope/stored-read
checks pass. The [follow-up receipt](../benchmarks/2026-10-10-item6/followup/README.md)
freezes the one-file production change separately from part 2's corpus work.

I is causally bisected on matching initial tensors and the identical forward
root. The first candidate adds live constituents to pair search: a self-pair
wins and is then rejected, concealing the lexical split. The terminal mask
comes later. Part 1's progress filter repairs that obstruction, but a live
near-copy of word 0 wins by a roundoff-sized residual advantage; the exact-fit
override bypasses learned pair scores and the symmetric pair is emitted in
reverse order. Both words are recovered, so overlap still reads 0.000.
Removing the live constituents yields 1.000 on the same artifact at both
stages. Full paired results and frozen tensors are in the receipt.

This reports I's cause without weakening its .800 bar or declaring order
learning repaired. Part 2 must measure the exact-fit override and orientation
lesson, including the newly distinguished reversed-output case. J's content
coverage, K's order stamp and L's unused family metadata remain open, together
with stored recovery beyond two words, order CE, MM grammar and compiled
retention. No identity rule or declared referent mode changed in this follow-up.

## 4. Part-2 corpus and measurement preparation (Codex, 2026-10-10)

The [preparation receipt](../benchmarks/2026-10-10-item6/identity-corpus/README.md)
freezes the first 740-document corpus, label sidecars, grader and native
observation driver on the reviewed H/I source. It covers the amendment's
factorial/support curriculum, counterbalanced pronouns, imperfect determiners
and held-out conflicts, same-kind distinctions, document context and both
controls. The text supplied to the learner contains no candidate or identity
labels. Thirteen corpus/grader checks pass, including the corrected distinction
between grammatical roles and passive agent/patient labels, and a positive
observer check against the real row writer. The observer follows the owned
NP's reference when its head and order also appear on the enclosing sentence.

Native reports preserve failures and unreached cases. The known-policy
controls verify the grader; they do not stand in for learned controls. In
particular, the cold biased preflight is not matched to a trained
counterbalanced branch. The passive-role prototype was stopped after 420 training documents
to correct its grammatical labels, with no evaluation reached; it is retained
as an incomplete diagnostic. Revision 2's baseline starts cold. A learned
recency comparison requires the same warm-up state and presentation budget on
both branches. Read the native outcomes in the receipt. The prototype's
native trace/admission underflow is a separate open finding, not repaired by
changing the lesson wording.

On the corrected observer and corpus, the cold stream completes 132 of 150
documents before a four-document batch raises the trace underflow; fourteen
documents are unreached. No completed binding or row pair resolves. The biased
preflight completes 16 training and 16 reversed-evaluation documents, also
with no resolved binding. Both reports match the frozen source and corpus.
These are baseline findings, not passing learning or rule-retirement gates.

This is preparation for review, not item 6's closing. No production code,
declared mode, pronoun rule, positional binding, cosine resolver or recurrence
admission changes. Each retirement still requires its lesson to pass without
the rule. The inverse, order-credit, MM and compiled-retention findings and
review J–L remain open alongside this work.

## 5. Review of the follow-up landing and the part-2 preparation (Claude, 2026-10-10)

### 5.1 Follow-up H/I (`9cc5f6e32` + `c1d1ea03c`) — fine, one mechanism fix to carry

H is the one-line gate; the sweep has no new failing node against 6.1
(5,451 passed, 1 failed = c). I's bisection is sound and worth the trouble:
the first candidate's live constituents won as a self-pair; part 1 removed
that; what remains is that `_bounded_binary_reconstruction` takes the raw
minimum residual among pairs that are *all* within the exact-fit threshold
(3.7e-15 vs 7.0e-15 against a 9.1e-13 threshold), so a roundoff-sized
difference overrides the learned decomposition scores and the symmetric pair
comes out reversed. That is not an orientation-lesson problem: among exact
fits the learned score must break the tie, or the order lesson cannot reach
this path at all. Fix (mechanism, not a rule): rank pairs within the
exact-fit threshold by the chooser's score; residual decides only outside
it. Do it with part 2's order lesson, measured by this same probe.

### 5.2 Part-2 preparation (uncommitted; receipt `identity-corpus/`)

What is there: a 740-document generated corpus in five streams with labels
in sidecars the learner never sees; a grader with oracle and recency sanity
controls (16/16, 0/16, 16/32 — the grader works); a native driver that
feeds only text and document addresses and reads the selected order-one
references and the writer's minted addresses; 13 contract tests (pass
here). The corpus follows the amendment's rules: singletons first, then
object support 1/2/3 with every kind × property × verb at every noun
position; a separate confounded stream; the two-candidate pronoun lesson
crossed on recency × introduction role × first position with refresh
mentions; a 2:1 role marginal in the three-candidate lesson reported, not
hidden; determiners right 90% (8/80 noisy); four held-out cue/content
conflicts; the two controls. This is the corpus-and-measurement-first step
the amendment asked for, and no mechanism changed.

**Measured:** nothing yet. Cold stream: 132/150 evaluation documents
complete, 0/122 bindings and 0/10 row pairs resolve, four order-one
reference requests in total, zero noun columns; then a production exception
stops it. Biased preflight: 16 + 16 complete, zero references, zero columns.
These are baselines, as the receipt says.

**Findings:**

- **M. A production defect blocks the measurement — repair it first.**
  `ValueError: compose trace binary operation underflows` in
  `_derivation_program`: an exploratory leaf admitted into the compose trace
  is absent from `source_leaf_mask` (saved before the trial's attention
  hand-off), so the derivation reconstruction drops one operand of a binary
  rule. Codex located the boundary (`SentenceField` candidates from the
  current trial vs the saved mask) and reproduced it on the prototype's
  passive input; the revised corpus hits the same class at evaluation
  132–135. This is unchanged-production correctness work, not an identity
  mechanism; it belongs before any part-2 run, with the other alongside
  items.
- **N. The same-kind lesson depends on morphology the model does not
  have.** `a cat runs . a cat sleeps . the running cat is black .` ties the
  later property to an individual through `running` ↔ `runs`, which is
  item 5.5's work; to this learner `running` is an unrelated word, so the
  later sentences cannot do what the lesson intends. Use the same word
  forms: `a cat runs . a cat sleeps . the black cat runs . the white cat
  sleeps .` — the verb is the distinguishing predicate and no morphology is
  needed. (Relative clauses are stage 6.)
- **O. Object topicalization is untested in the grammar.** 48 documents
  use `a dog , a cat saw .` to separate role from position. Whether the
  compose grammar can parse a fronted object (a comma rule, an attachment
  for the fronted NP) is not shown; if it cannot, the "role" factor is
  confounded with "unparseable". Either show the parse (one receipt row
  with the derivation) or drop the fronting and keep the refresh-mention
  crossing, documenting that introduction role and position are confounded
  at the first mention.
- **P. Stage 7 is thin.** The eight "document context" cases are
  property-follow-up pronouns whose probe text (`it is black .`) refers to
  different individuals in different documents; the per-batch resets mean
  nothing crosses a batch. Fine for now; the batch-differentiation,
  interruption and shared-truth stages of [stream-state §3](2026-10-08-stream-state.md)
  are not covered and should not be claimed.
- **Q. The real blocker the baseline exposes.** Four order-one reference
  requests in 132 documents: the cold chooser almost never selects `lower`,
  so no identity is ever minted or bound and the identity dictionary stays
  empty — admission is gated on the selected mint rule
  (`allow_mint = -1 in individual_references`). This gate is itself one of
  the amendment's retirement targets, and its lesson is now runnable: with
  the gate off, does factorial training mint the four kind columns and the
  confounded stream merge them, measured by *identification* on held-out
  rows (the toy's metric), not by max-cosine? That, after M, is the first
  part-2 experiment; the pronoun lessons need the staged warm-up (stages
  1–4) before they can show anything, as the receipt says.

Smaller: the noisy determiner items whose content is identical
(`the black cat runs . a black cat sleeps .` → 1 row) are graded against a
label the text cannot determine; reporting them separately, as the receipt
does, is right — keep them out of any gate. `candidate_coverage` reads
the journal, not the live bank; the inventory audit Codex names is needed
before a wrong choice can be told from a missing candidate.

### 5.3 Hand-off to Codex (Alec confirmed, 2026-10-10)

> Part-2 preparation reviewed (plan §5): corpus, grader and driver are
> accepted as the measurement; commit them. Before any learning run:
> (M) repair the compose-trace underflow — the saved `source_leaf_mask`
> must include every leaf the trial's field admitted, or the reconstruction
> must read the trial's own mask; regression on the retained failure
> context. (N) Same-kind lessons use the same word forms (`the black cat
> runs .`), no participles. (O) Show one parsed topicalized clause with its
> derivation, or drop the fronting. (I) Among pairs within the exact-fit
> threshold, rank by the chooser's learned score; residual decides only
> outside it; remeasure the bisect probe. Then the first experiment: the
> staged warm-up (stages 1–4), then factorial vs confounded recovery with
> the selected-mint admission gate on and off, scored by held-out
> identification; report columns minted, merges, and the gate's lesson
> outcome. Pronoun and determiner lessons follow from the same warm-up
> state with equal budgets on both branches. No rule is retired except by
> its lesson passing without it.

## 6. First part-2 experiment candidate (Codex, 2026-10-10)

The preparation accepted in §5 landed as `06f6e2b88`; WikiOracle is at
`56544f1`. The [single next receipt](../benchmarks/2026-10-10-item6/identity-first/README.md)
contains M's trial-owned leaf mask, N's same-form predicates, O's removal of
unverified fronting with the introduction-role/position confound reported,
and I's learned ranking within the exact-fit set. The cold bisect overlap
remains 0.000 with an untrained residual-only chooser; learned-score reachability
is separately covered by the permutation regression. A frozen-artifact teaching
probe trains only the chooser for 64 steps: CE falls 1.837530 → .374767 and
overlap reaches 1.000; the same learned weights with the old residual override
still score 0.000. This proves that repair's narrow teaching path, not native
curriculum learning. No fallback order is added.

The native experiment fixes stages 1–4 warm-up, then equal budgets on all
branches, active-column held-out identification, minted/merged-column reporting
and the mint-admission ablation. The receipt declares the budgets and threshold
before the learning run. Failed warm-up prerequisites remain visible; no rule
is retired absent its passing lesson. Native checkpoint reload drift is
recorded, with exact frozen tensor restoration used solely to match starts.

The first warm-up stopped at four-word presentation 8, after seven completions,
on a non-concept `sym` cue entering the taxonomy order reader. The full original
attempt is retained in the same receipt, including the first sweep's new b
failure. A narrow retrieval boundary repair preserves vector unfolding and
admits only positive concept handles to the order reader. A corrected-source
run uses the identical seed/budgets; it also retains native invalid-cue provenance
and stage checkpoints. No learning outcome is assigned to the unrun branches.

The corrected warm-up completes: final two-/four-word CE is .3465735912 /
.7945134640, with all sentence reconstruction, free readback and supplied-answer
checks at 0/4. Its eight branch starts then fail before training: the parent's
strict checkpoint preflight rejects 16 lazy comparison-reader keys before the
child's existing restoration hook runs. This failed start and its source remain
in the receipt. The experiment adapter now restores that child through its own
strict hook first. A replay of the retained checkpoint matches all 874 tensors
with no extra tensors; the checkpoint contains zero noun columns. Production
loading is unchanged and the newly observed retention finding stays open.

The branch-only continuation uses that exact checkpoint and adds no warm-up
presentations. Its guard verifies identical learner/corpus source; only the
experiment restoration adapter and its regression differ. Both source manifests
and the failed original starts are retained. Seven adapter/reader checks pass.
That source's 5,776-case sweep reports 5,488 passed, 286 skipped, one failed (`five`)
and one xfailed, with all 771 source files matching. The earlier new b failure
and the overlap XPASS-to-XFAIL remain open.

The continuation's four recovery branches completed with zero identification,
columns and pending prototypes. These are **invalid measurements**: the
word-chunk fixture used equal form and conceptual widths, leaving zero meaning
pole pairs. A read-only checkpoint audit found all 55 final chart basis rows
zero, and all 434 nonzero observations projected to zero. This setup cannot
test residual admission. The unfinished identity branches were stopped, with
their progress and every completed result preserved; no gate outcome follows
from those zeros.

The corrected measurement uses the production form/meaning separation:
104 form coordinates plus 128 meaning coordinates (64 pole pairs), 4,096
concept rows, 256 part rows and an 8,192-row LTM. Its ordinary-word preflight
reads back 4/4, supplies four nonzero basis rows and projects three of four
retained observations nontrivially. Constructor and post-lexical warm-up guards
now reject a missing meaning signal. The fresh staged experiment completes
with unchanged seed, data, presentation budgets and learning rate. All eight
branches exit zero; source and corpus hashes match. Its final
source-matched sweep is complete: 5,776 selected, 5,488 passed, 286 skipped,
one failed (`five`) and one xfailed, in 973.50 seconds; no new failing node.
The first attempt's new b failure and the adverse overlap outcome remain open.
The old checkpoint is not reused across the geometry change. No rule is retired.

The valid two-word warm-up completes 64 presentations and improves final
stored-derivation readback from 0/4 to 4/4 on those training phrases, evaluated
without the reading lesson. Free generation remains 0/4 and order CE stays
.3465735912. Its checkpoint contains ten noun columns and 72 pending prototypes;
those counts do not establish atom identification. The receipt's tensor audit
confirms updates to attention, pair choice and the generate policy, without
claiming that parameter changes establish successful order learning. The
four-word lesson also completes 64 presentations: initial and final unguided
trace readback are both 4/4, free readback is 0/4, and CE stays .7945134640.
All stage-4 lessons finish: final trace readback is 4/4 in each and free
readback is 0/4. Answer scores are 4/4, 4/4, 0/4 and 1/4; the first three
match a last-word baseline. The shared warm-up has 41 noun columns and
334 pending prototypes. All four recovery branches restore the same
checkpoint with exact matches on all 2,138 tensors and no extra tensors;
factorial/confounded × gate-on/off runs all finish at 0/240 held-out
identification, with zero new columns and zero merges. Every row selects
mint, so on/off eligibility is identical and the two checkpoints within each
corpus are byte-identical. All 192 newly added observations have nonzero full
vectors but zero meaning-band content. These are native pipeline misses,
not an informative test of ICA recovery from mixtures or of a binding
admission constraint. The numerical retirement gate fails; no rule retires.
The four identity branches restore the same warm-up exactly, each complete
136 training and 166 evaluation documents, and score 0/146 bindings and
0/20 row pairs. One-/two-/three-candidate scores are 0/4, 0/88 and 0/54;
neither-candidate and reversed-recency controls score 0/32 and 0/16. Every
binding and row pair is unresolved. Balanced training mints 61 columns
(102 total), biased training 62 (103 total); column counts do not establish
correct identification. All 940 observed commits per branch select mint,
including frozen evaluation, and the on/off checkpoints are byte-identical
within each corpus. There is no admission eligibility contrast in either
the recovery or identity experiment. Native banks are populated in all
146 probed cases, but none of 690 annotated mention references resolves;
the selected-mint-to-individual-reference seam remains open. These outcomes
do not establish prediction-based choice, a recency shortcut, successful
chance behavior, or a one-row/two-row distinction. No rule is retired.
Frozen replays of four held-out rows locate the semantic loss in compose:
adjective/noun/verb inputs have meaning norm .2, but selected `non` operations
clear it; binary `lower` keeps the newer operand, ultimately the period.
The chosen lossy readings and their cost/credit remain an open finding.
A read-only probe of the saved two-word checkpoint finds all 64 attention
tanh units saturated at exactly ±1 (preactivations about ±140). Candidate
differences vanish through tanh and every row has tied logits. This explains
the uniform-choice CE at that checkpoint; onset and cause remain unmeasured.
The finding is retained without changing the declared experiment's learner.
The code audit also bounds the reconstruction claim: pair teaching requires
operand identities in the word shortlist and skips a compound operand marked
`-1`. Its successful lexical-pair probe does not establish sufficient credit
for longer constituents. Identical chooser features cannot be separated by
training; information erased by `non` cannot be recovered from its value alone.

The completed receipt retains the raw measurements, matched checkpoints,
source/corpus archives and all earlier failed attempts. The §7 review below
accepts this pass for landing as preparation, with no learning claim and no
closing of item 6 part 2. R/S/T follow in one measured repair pass.

## 6. Parts of speech from operand participation (Alec, 2026-10-10)

*Where the code is (verified 2026-10-10).* No word carries a part of
speech; the grammar's operations carry **named** operand kinds —
`reference="I2:particular|pronoun|kind"` on `lower`/`definite`/`universal`/
`bind` (`data/complete.grammar` 170–177, 238–239) — with a fixed name→order
table (`Language.py:1681`: event 0; particular/name/pronoun 1; kind/generic
2), and the names are read at `ReferenceContext.binding_spec`/`resolve_operand`
(mode masks: a pronoun/bind operand may not be its own referent, mint names
no row, kind cancels), `ClauseJournal` 92/193, `ClauseScope` 26,
`Interpret.bind` (`explicit = mode in ('bind','pronoun')`) and
`selected_individual_references`. The learned part of speech exists beside
this: the MetaSymbol category learner (`Language.py:150`) accumulates each
word's role-participation profile, assigns it to the nearest centroid of a
**fixed-size** VQ codebook (`Spaces.assign_category`), EMAs the centroid,
commits the id on the word's order-1 META row, and the chooser reads it as
`cat_ctx`. It cannot start a new class for a word that fits none, never
merges classes, and nothing but the chooser's context reads it.

*Decision.*

1. **Operand names are positional** — `argin1`, `argin2`, `argout1` —
   never `pronoun`, `particular`, `kind`. What an operation declares is
   structural: its arity, which operand is its head, and the order change
   it performs (`order_delta`). An operand's reference order is read from
   the operand's own order stamp and that change, not from a name.
2. **Positions form classes.** Whatever words work in a position become
   that position's class. Classes collapse: if the words that serve as the
   adjective operation's operand and the words that serve as the verb's
   second operand are the same words, those two positions are one class
   (NP). A word may belong to several positions; the overlap is what
   collapses them.
3. **A small codebook of parts of speech, every word assigned to one,
   learned class by class** as a speaker learns: a word whose participation
   profile fits no existing class starts a new class vector; assignment and
   centroids are updated by online EM (k-means form: assign to the nearest
   class, move the class toward its members); classes whose centroids
   coincide merge. Count noun, proper noun, pronoun, determiner, adjective,
   verb are such classes if the data make them — the learner holds ids, not
   our names. The existing category learner is the seed: add growth on
   no-fit, merging on coincidence, and make its class the **only**
   part-of-speech information the chooser and the reference machinery read.
4. **Identity behaviour is learned per class**, by part 2's lessons: the
   class whose referent is always elsewhere is what we call pronouns; the
   mint/bind preference of determiners is a learned bias on their class
   ([§2](#2-identity-from-data-amendment-alec-2026-10-10)); no mode is
   declared on an operand.

*Retired by this decision* (each only when its lesson passes without it,
as in §2): the `reference=` name attributes and the name→order table; the
name-keyed sites listed above; `determiner_mode` on rules and part 1's
mode-keyed policy rows; `resolve_operand`'s mode masks and cosine argmax.
Kept: `order_delta`, `head`, arity, `clause_form` as declared structure.

*Lesson and measurement.* On the curriculum's stages 2–4 the learned
classes are scored against the generator's hidden word classes (purity and
completeness per class, number of classes over presentations, when each
class first appears); collapse is tested on a corpus whose two positions
share their words; the class ids must be stable across documents
(stage 7). Claude runs a toy first ([[simulate-before-specifying]]):
online k-means with growth and merge over participation profiles taken
from the identity corpus's parses, against its hidden classes. This pass
follows the §5.3 pass (M, N, O, I, the gate experiment), which proceeds
unchanged.

### 6.1 The learning rule (Alec, 2026-10-10; Claude's reading of it)

Each word points to exactly one class; nothing else is stored per word (no
participation profile — the existing learner's pending evidence rows go).
The class vectors live in position space (one coordinate per operand
position of each operation). When a word is used in a position:

- **E step:** the word's pointer may switch to the class whose vector best
  fits that position. The word's current class is its one memory slot, so
  the switch is sticky: it happens only when the fitting class beats the
  current one by a margin (a single odd use does not move a noun).
- **M step:** the class vectors move — the chosen class toward the
  observed position (ordinary online k-means), and on a *switch* the old
  and the new class are pulled toward each other. Since no word remembers
  that it has served in both, the switch itself is the only evidence that
  two classes overlap; the pull records it in the classes.
- **Collapse** follows from the pull: positions that share most of their
  words make their classes converge until they are within the merge
  distance, and the classes merge (the smaller class's words are re-pointed
  by a remap table, no per-word scan). A minority of ambiguous words
  (noun/verb `run`) pulls a little and keeps flipping, but the within-class
  M steps hold the two classes apart — the fraction decides, as in
  k-means, so noun and verb do not merge on ambiguity.
- **Growth:** a use that fits no class beyond a distance threshold starts
  a new class vector at that position, as the identity dictionary mints.

The toy tests exactly this on the identity corpus's parses: class count
and purity over presentations; the NP collapse when the adjective's and
the verb's operand share words; no noun/verb merge under ambiguous words;
stability of ids across documents.

*Participation is soft (Alec, 2026-10-10).* A word's class never gates a
position: if only class-c words could fill position p, no word could ever
be seen in a new position and nothing above could be learned. The class
matrix (classes × positions) read by column gives, for each position, a
**softmax over classes** — the share of participation at that location —
and the chooser adds the log-share of the word's class as a learned bias
on placing that word there, beside its other evidence. Other classes keep
nonzero probability, so exploration can try a word in a new position and
the sentence's credit decides whether the use is kept; a kept use is the
E step's observation. The same column softmax is the E step's criterion
(assign to the class with the largest share at p, with the sticky margin).
This is how `cat_ctx` already enters the chooser — as a feature, not a
mask — so the wiring changes only in what the feature is.

## 7. Review of the §5.3 pass: M/N/O/I and the first native experiment (Claude, 2026-10-10)

Reviewed the uncommitted tree against `06f6e2b88` and the
[receipt](../benchmarks/2026-10-10-item6/identity-first/README.md); read
the captured derivation programs and the scorer audit myself.

### 7.1 The repairs — correct, land them

- **M** — the trial's own admitted leaves become the grammar-leaf mask
  (`SentenceField.grammar_leaves`); the committed winner uses the mask of
  its scored trial; regression on the retained `[4,0,1,2,3,5,6,7]` context.
- **N** — corpus revision 3 uses the same word forms, no participles.
- **O** — fronting dropped; the role/position confound at first mention is
  stated; refresh mentions still cross recency with role.
- **I** — among exact recompositions the decomposition chooser's learned
  logits decide; residual no longer overrides them. Taught for 64
  presentations the probe reaches 1.000 overlap, and replaying those weights
  under the old override stays 0.000 — the obstruction is isolated. The
  cold model still reads 0.000 (its chooser is `[1,0,0,0,0]`, residual
  only), so the XFAIL stays adverse until the order lesson teaches.
- The sweep: 5,776 cases, 5,488 passed, 1 failed (c), 1 xfailed; no new
  failing node against 6.1.

### 7.2 The experiment — ran as declared, and every score is zero

Recovery 0/240 and identity 0/146, 0/20 in all eight branches; the
selected-mint gate on/off produces byte-identical checkpoints because every
one of the 192 (940) commits selects mint, so the ablation had nothing to
ablate. The value of the run is its diagnostics, which locate three
obstructions, all upstream of identity:

**R. The compose policy erases meaning.** The captured programs for the
held-out rows (`validation/recovery-program-audit`) read `the black cat
runs .` as: push `the` → `non`,`non`,`non`; push `black` → `lower` →
`non`,`non`; push `cat` → `lower` → `non`,`non`; … Every content word is
lowered onto the previous one and its meaning cleared twice; the closed
clause has meaning norm 0.0 while each word carried .2. `non` is the
tetralemma's non-affirming exclusion, declared lossy with no inverse. So
the rows handed to the identity dictionary contain nothing to unmix — the
zeros follow. Nothing charged this reading: derivation readback reads the
trace (4/4), the free reconstruction is reporting-only under the field
objective, and part 1's coverage charge counts leaves, which `non` does
not drop (finding J). This is the 6.2 observation — a kept reading that
discards content uncharged — in its purest form, and it is the first thing
to repair: a kept reading must pay for the meaning its row cannot return.
Concretely, the keep cost needs a content term: the meaning the clause's
own leaves carried that the closed row (through the derivation's lossy
steps) does not retain — the free reconstruction's meaning error or a
coverage over the leaves' meaning bands — so that `non` costs what it
discards. The chooser learned this reading, so a rule banning `non` is not
the fix; the objective is.

**S. The attention scorer saturates.** The order lesson's flat CE
(log 2 / 2, unchanged since 6.1) now has a cause: after the two-word stage
all 64 tanh units sit at exactly ±1 on every row and candidate —
preactivations span ±140, of which the shared-context block contributes
±140 and the candidates ±.28 — so the hidden derivative is zero and the
two candidates' logits are identical. The lexical-stage weights on the same
input spanned −.5 to .76 with no saturation; the growth happened during
stage 3 under Adam, which 6.1 §7.16 put on this scorer at my
recommendation. Repair: bound the scorer's preactivation (normalize the
shared-context input or the layer), and measure that the order lesson's CE
moves on the same checkpoint.

**T. The observer resolves no reference.** 0 of 690 annotated mention
references resolve: 445 are mint sentinels, 245 missing. A selected mint's
row address never reaches the grader, so even a correct mint→bind chain
could not be scored. Measurement plumbing; must be fixed before any
identity number is read. Relatedly, the grammar never selects
`definite`/`bind` — binding by column reuse is the part-2 mechanism and
cannot be tested before R.

Also recorded: the answer stages (identities 4/4, NPs 4/4, noun–verb 0/4,
relations 1/4) equal the free decoder's last word — not question-sensitive
retrieval; four-word trace readback already 4/4 before its stage; free
readback 0/4 everywhere. These stay in the alongside list.

### 7.3 Recommendation

Land this pass as it stands — repairs, revision-3 corpus, experiment driver,
receipt — as preparation, with no learning claim. Then one repair pass for
R, S and T with the identical experiment re-run (same seed, budgets,
corpus) as its measurement, one receipt. §6's parts-of-speech pass waits
for R: its observations are kept readings' operand positions, and readings
that are `lower`+`non` everywhere would teach classes of nothing. The toy
for §6 proceeds meanwhile.

### 7.4 Hand-off to Codex (Alec confirmed, 2026-10-11)

> §5.3 pass reviewed (plan §7): land it as preparation (commit, push,
> bump), no learning claim. Next pass, one receipt, same experiment
> re-run as the measurement:
> (R) The keep cost charges discarded meaning: add a content term — the
> meaning the clause's leaves carried that the closed row does not return
> (free reconstruction's meaning error, or coverage over the leaves'
> meaning bands) — so that `non`/lossy readings cost what they discard.
> No rule against `non`; the objective decides. Measure: the held-out
> rows' closed meaning norm is nonzero and the `lower`+`non` program is
> no longer the kept reading.
> (S) Bound the attention scorer's preactivation (normalize the
> shared-context input or the layer); regression: on the saved two-word
> checkpoint the candidates' logits differ and the order lesson's CE moves
> from log 2 / 2 within the stage.
> (T) The observer resolves a selected mint to the written row's address
> so mint→bind chains are gradable; regression on the retained identity
> run (690 references).
> Then re-run the eight branches unchanged and report factorial vs
> confounded identification, the gate contrast (or that it is still
> vacuous), bindings and row pairs. No rule retired except by its lesson.
