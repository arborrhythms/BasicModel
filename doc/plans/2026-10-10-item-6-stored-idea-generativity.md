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
