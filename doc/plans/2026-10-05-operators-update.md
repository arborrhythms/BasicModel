# The grammatical operators update: plan (Claude, 2026-10-05)

Status: item 6.8 accepted on the §22 receipt
([6.8 plan §23](2026-09-27-item-6-8-one-attention.md#23-review-of-the-22-measurement-acceptance-claude-2026-10-05));
the operators update is next, then 6.5 (decided order, 2026-10-03). Its
specification is the [operator catalogue](../specs/2026-09-29-operator-catalogue.md)
(§12 records what is already implemented); this plan orders what remains,
including what the 6.8 rounds carried to it (FutureWork, "Carried from the
6.8 §13–§15 rounds" and after).

## 1. Standing gate

No regression against the accepted 6.8 landing
(`doc/benchmarks/2026-10-03-operators-attention/README.md`, §22): XOR class
7/10 (conjunction runs at 0; disjunction runs at the affine floor ¼),
reconstruction 9/10, MM_xor 10/10, sum control 10/10, sweep green, zero
sentence-path gradient at perception, zero ownership conflicts, codes with
zero displacement. Every round: the full sweep once on the delivered source,
then the thirty trainings, read with the §20.5 bands. One change per round
where a change can move a gate; one text per round; stop for review before
any commit.

## 2. Rounds

### Round 1: what is decided and self-contained

1. **The attention chooser's estimator.** The input-attention walk's choices
   train by the score-function term with the greedy cost as baseline and the
   `K·R` correction, exactly as compose's (6.8 plan §16.3); attention's credit
   stays detached at the sentence handoff (no pathwise path). One writer for
   the shared chooser.
2. **The decomposition chooser.** An exactly recomposing pair (relative
   residual at float tolerance) is taken before context has a say;
   the activation features (projection coefficients) are standardized; the
   walk policy (undo / unary / STOP) trains by cross-entropy toward the
   compose derivation's structure, teacher-forced, free at inference (6.8
   plan §16.4, §23).
3. **`not` and `non` over the poles** (catalogue §3.8): `NonLayer` sets the
   expressed pole to zero (withdrawal), not `1 − x`; `ConjunctionLayer` reads
   the pair by the bilattice (∧ = (min c⁺, max c⁻)), not `max(c⁺, c⁻)`;
   `not` is the pole exchange only, never a code's sign. The binding kernel
   composes forms and is not asked to host negation.
4. **Footprints checked at load** (catalogue §1 rule 10): each operator
   declares its reads and writes over form, meaning and poles, checked
   against rule 2's declared writes.
5. **The idempotent intersection** (catalogue §4): the coded intersection is
   exact (`min`, a silent coordinate adding no restriction), not softened;
   the verb and the adverb may write on a dimension their operand is silent
   on (§4.3).

Gate: §1. Expected: unchanged counts; reconstruction's one §22 miss closed by
item 2's precedence.

### Round 2: the complement's bootstrap (the distributional item)

Order-0 meanings are zero until the conceptual wholes have locations: the
sets one order up, situation and document codes, properties as concepts,
trained by co-activation; a word's context mean seeded from them (6.8 plan
§13.4). **Design to settle with Alec before the round** (§3 below): the owner
of the complement's parameters, the co-activation rule, and what the gate
configurations carry in their complement.

### Round 3: connectives over meanings, the fold over forms

∧ = min, ∨ = max, `not = −d` over the conceptual coordinates (the towers'
Kleene algebra, 6.8 plan §13.4); forms composed by the located fold at every
order (anagrams separate; identity residue); the gate configurations given a
nonempty complement, since with the fold on the form band the connectives
otherwise never reach the root. What the XOR table then measures is restated
before the round (forms by the fold, meanings by the connectives).

### Round 4: the catalogue's remaining sections

Determiners (`a` mints, `the` binds, `every` stays high-order), `generic`,
`lift`/`lower` (catalogue §5); relations and the sentence that states a
definition (§6, an `equal`); operators tested by name (~140 places) become
declared properties; the item-7 residue (predicate identity as a rule
property; `GrammaticalQueryRegistry` retired). `surface`, tense, morphology,
aspect and `null` are item 5.5's.

## 3. Questions for Alec (before round 2)

1. **Who owns the complement's locations?** Co-activation is neither
   reconstruction, expectation nor output. A fourth owner ("association"),
   or reconstruction's, on the argument that a word's context is part of
   what reconstructs its row?
2. **The co-activation rule.** The successor-representation / diffusion
   family of 6.9 §25.5 (location trained from spreading activation), or a
   skip-gram-like objective over occurrence rows (the catalogue's
   distributional row)? The former follows the field the design already has.
3. **The gate configurations' complement.** Width (how many conceptual
   coordinates beside the fourteen form coordinates), and whether the gate
   corpora should carry contexts that differ (today every word shares every
   whole, so meanings coincide by design).

## 4. Carried elsewhere

Item 2: the negative image acts on the concept face. Item 1: the 2.6×
open-read cost; the decomposition teacher loss's host island. Perception,
when trained: form density (sparse presences). Later: REBAR/RELAX if a dense
chooser signal is wanted; the operator renames.

## 5. Review of round 1 (Claude, 2026-10-05)

**The landing.** `42daf96f4`'s tree matches all 696 runtime hashes of the
§22 frozen manifest (manifest sha `88e4df02…`, as recorded); the five
acceptance-document hashes match `review22-acceptance.json`; the two other
changed docs (`Architecture.md`, `GradientFlow.md`) match the §22
final-focused manifest. Pushed to `origin/main`; the WikiOracle bump
`3fc7802` points at it and is pushed.

**MM_xor 9/10: round 1 is inert on MM_xor.** MM_xor does run input attention
(stage_input/attend every epoch), so the round is on its path in code. But
with the same torch seed, the 200-epoch loss trajectories of the landing
source and the round-1 source are bit-identical for 33 seeds of 33 (first-step
gradients identical parameter by parameter). In that sample, the landing
itself fails 0 times out of 33. Run 5 is therefore a draw of the landing's
own unseeded failure rate, which ten runs cannot distinguish from zero.
The receipt was right not to waive it; the evidence now rules out the round
as its cause. **Recommendation:** accept, and amend §1: when a round is shown
to be trajectory-identical on a gate configuration, that gate's count is not
a regression test of the round. (Alec decides.)

**The attention term is wired but can receive no signal (revised after
Alec's reply).** Alec: "no direct-gradient credit" is the wrong name; the
decision is *chooser on policy credit only* — the choice is discrete, so the
credit is the comparison of trials, and a trial where attention is correct
should do better. Agreed; the finding is that today no trial can. In the
audited run all 1,600 departures tie (C_explore = C_greedy). Two causes:
(1) the cost is the byte reconstruction of the glossed values, which is the
same for every walk that glosses every word; (2) `and`/`or`/`not` inside the
walk change nothing that leaves it — only the next choice's pooled reading
and the budget meter (`narrow_words` returns table, values, accepted,
descended, actions; downstream reads `accepted`, the table's scope and the
detached credit). So no cost, upstream or downstream, could tell a correct
field action from a wrong one. The chooser's logit ranges are identical in
every epoch of the audit: it never moved.

The same blindness holds for compose: its 1,600 departures also tie, because
conjunction and disjunction compose forms identically and bytes cannot see
the meaning. The receipt's class results then split by which operator the
untrained chooser settled on: conjunction (alone or mixed) 8/8 pass; the two
disjunction-only runs (4, 6) miss. The class gate measures the chooser's
prior, not learning.

**Proposed (Alec decides):** (a) hand off the narrowed field — the poles and
values the walk computes at the selected bracket go with the scope into the
sentence (one attention's intent); (b) measure the trial at the owner step
with the costs that can see the choice: the word-level expectation's
prediction error (6.8-1's predictor: attend, then predict the next word — the
single-word learning Alec proposes for attention, in the form a trial can
score), reconstruction, and on supervised sentences the answer. Policy credit
only; no gradient through the choice; shared operators remain trained by
every objective. A one-word input offers the walk one legal action (gloss if
known, descend if unknown), so a single-word trial needs the word's context
to carry a choice.

**Decomposition precedence on the §22 miss: accepted.** The fixture equals
the landing's saved `xor-10` roots, codes and chooser weights exactly. The
replay with context weights ×1000 still recovers the true pairs. Rows 2–3
pick the commuted order of the true pair (product conjunction, same
residual); reconstruction is by multiset, 40/40. Tolerance
`(8·eps)²` on the squared relative residual is right.

**not / non ports: accepted.** The ported assertions follow catalogue §3.8
(code identity, pole exchange/withdrawal; closing polarity stays in
ClauseJournal's `polarity_effect`). The echoic-decoder walk test keeps its
unary coverage with a local reflection operator rather than losing it. Note
that in every shipped config (`representation='code'`), `not`/`non` are now
identities on the code path. Their only effect is the closing polarity flag,
which is why the free decoder needed the identity-unary guard.

**Scope beyond the plan.** Two changes were not in round 1: the LDUReadout
fold-identity initialization (its one construction site is the answer-percept
adapter, `Spaces.from_percepts`) and attention's negative pole no longer
signing the glossed form. Both are reported in the receipt and both should
be accepted with it. The receipt correctly does not attribute class 7→8 or
reconstruction 9→10 to any single change.

## 6. Round 2 direction (Alec, 2026-10-05; recorded by Claude)

**The trial cost is the reconstruction or output error** — the sentence's
error at the owner step, not the percept byte stand-in. Today compose's pair
sums `registry.total(objective='reconstruction')` only
(`Models.py` ≈20450) and attention's pair scores the glossed percepts before
admission (`percept_reconstruction_score`). Both move to the owner step's
registry total (reconstruction, expectation, and output on supervised
sentences). Consequence for attention: its explore walk must proceed through
the sentence to be scored there (one extra forward per training sentence), or
the baseline is the same row's cost at its previous presentation (one
forward, higher variance). Claude recommends the former: it keeps the decided
paired estimator and the open-read cost is already carried.

**Why reconstruction cannot score the operator choice.** Reconstruction
scores bytes. Conjunction and disjunction recover the same word pair exactly
(exact-fit precedence, exact inverses given the operator), so their
reconstruction errors tie to the float. What the choice means — what the
sentence asserts — shows only in the output error (the class) and in
expectation (what follows). The pair chooser is not the problem; it finds
the pair either way.

**Attention still needs a consequence.** Moving the cost does not by itself
give `and`/`or`/`not` inside the walk any effect outside it (§5). The narrowed
field (poles and values at the selected bracket) is handed off with the scope.

**The complement (Q2, Q3).** Part building identifies a word uniquely by its
adjacent parts; whole building identifies a word uniquely by the wholes it
belongs to. Co-activation flows from that shared-wholes representation: words
that share a whole activate each other. So the complement's locations are
the whole-membership representation, and the gate corpora must carry wholes
that differ (today every word shares every whole).

**Still open (Q1, narrowed).** If a whole's code reconstructs its members and
a word's code reconstructs its wholes, the owner is reconstruction — of
membership, not bytes — and no fourth owner is needed. Alec to confirm.
## 7. Architecture, popped up a level (Alec, 2026-10-06; verified against the code by Claude)

Alec's scheme, paragraph by paragraph, against the code as it is.

### 7.1 Attention is a filter: a mask over perception, trained by letting the mask participate in the loss

**Code.** One word at a time is the mode: XOR_grammar, MM_grammar and
BasicModel derive `word_brackets` (a grammar beyond the substrate folds plus
`conceptLayers` > 1) and run `_forward_body_per_word`; only model.xml runs the
whole slab in parallel; the `serial`/`modeSchedule` knobs are retired. Input
attention is a walk over one bracket (`Attention.narrow_words`):
divide / descend / gloss on the structure, `and` / `or` / `not` on the field's
poles. Its lasting outputs are a HARD mask (`accepted` → `_ar_grammar_leaf_mask`:
which words are admitted and composed), the scope and the budget spent. The
mask participates in the loss as a mask; the *choice* of mask participates
only through the trial comparison (score-function, §16.3), since the sentence
handoff detaches attention's credit (§20, round 1). Until 2026-10-05 the choice
also participated pathwise: `attend` returned `1 + (p − p.detach())`, the mask
weight in the loss. It was removed as a second writer into perception's codes
through the perception pullback, below float resolution at ties.
No soft mask touches perception anywhere; the one soft weighting in the
model is the answer's own `PrimedSymbolReader` over detached keys, answer-owned.
The budget (`QueryWorkBudget`) is a hard integer allowance that raises when
exhausted, not a graded cost.

**Reflection.** The field operations are continuous in the poles (`and` = min,
`or` = max, `not` = flip, `field_reduce`); only *which* operation and *where*
are discrete. An attention that is a filter can be soft: a weight in [0,1] per
word (or bracket) multiplying DETACHED percepts on the sentence path, trained
pathwise by the owner-step cost, with a graded budget cost replacing the
hard allowance; thresholded for admission. That keeps "codes are perception's"
(the mask's gradient reaches the chooser, not the codes) and gives Alec's
"mask participates in the loss" without the §20 writer. Compose cannot be
soft the same way: a derivation is one tree written to one row (§16.3, "one S
= one row"), so its choice stays a hard sample with score-function credit.
Either way, no estimator has signal while the cost cannot see the choice (§5):
the cost moves first (§6), the estimator second.

### 7.2 Parts aggregate: one code per word across its occurrences

**Code.** Coincides. Perception's row is content-addressed
(`RadixLayer.insert`, `hash_map[bytes]`): a known word returns its existing id;
the concept is minted on first sight (`_stage_serial_concept_rows` →
`definitions.word(form=key)`, `Interpret.lookup_word`), so there is one word
per surface form. One qualification: the row is seeded once and never
refreshed (no EMA, no re-seed on later sightings), and each occurrence's key
is recomputed from the byte rows rather than read from the admitted row. So
"represented by a single code" holds for identity; the code does not yet
*accumulate* anything from later occurrences except through the meaning
complement (context mean, §7.4), which is empty today.

### 7.3 Parts chunk up serially; overlapping parts plus length would identify a word without order

**Code.** The form is weaker than Alec's description. A word's form key is
the elementwise MAX over its byte rows (`synthesize_word_parts`,
`values.amax(dim=-2)`): order-free, multiplicity-free and length-free. So
"circus", "cursic" and "cirrus" share one form key (tested:
`test_partspace_word_code_is_max_and_anagrams_share_it`); the band encodes
only the start offset (the endpoint-sum bracket is retired), so length is
nowhere in the key. Only the string identity (`DefinitionIndex` on the exact
form) keeps them apart. There is no subword level: atoms are bytes, words are
maximal letter runs (`Meronomy.pin_word_spans` "cannot split a word"), and
the fold ladder's rung 0 is the max over bytes; the learned rungs are above
the word. Processing is serial per word (divide/descend/gloss rounds), as he
says.

**Reflection.** Alec's remedy is the right one and cheaper than the one on
record. Round 3's "located fold" (FutureWork: "the located fold also
separates anagrams, which the unlocated max does not") separates anagrams by
putting position back. Overlapping parts do it without position: make the
parts of a word its adjacent pairs with boundary markers (`#c, ci, ir, rc, cu,
us, s#`) and keep the fold as it is (L = the join of the parts). Then anagrams
separate (circus / cursic share letters, not pairs), length is implicit (a
word of n letters has n+1 boundary-marked pairs), "ends in s" becomes a part
(`s#`) rather than a whole, the derivation is one parallel max over the
slab instead of a serial walk, and the same rule serves every order (adjacent
word pairs at the sentence rung give "dog bites man" ≠ "man bites dog"). It is
the open-bigram account of visual word recognition (Grainger & Whitney 2004)
and fastText's subword bag (Bojanowski et al. 2017); collisions are rare
and are exactly the cases identity already handles. This is a change to the
part inventory at rung 0 (a config/inventory change for Codex), not to the
fold, and it would replace the located fold at the word order in round 3.

### 7.4 Wholes should identify a word too; today they are character classes

**Code.** Alec's diagnosis is exact. A word's property wholes are the
canonical character-class rows (`_CANONICAL_PROPERTY_ROWS`: letter, digit,
whitespace, punctuation, capital, control, high_byte, pad, word); "circus"
has {letter, word}. "Two c's" is not representable (`on_counts`: repeating a
primitive does not change membership) and "ends in s" is not (`counts_in_spans`
reduces an extent to a byte multiset). Sentence/document membership is not a
whole of the word: occurrence rows are context (the detached, recency-weighted
mean of the occurrence roots' meaning coordinates, `occurrence_terms`,
`@no_grad`), and in the XOR/MM gates that complement has zero width (22 − 22);
BasicModel has an 896-wide complement that has never been measured. The
membership itself is queryable in LTM (`refs` + inverted `_leaf_postings`,
`rows_for_code` / `leaf_terms`), so the shared-wholes representation Alec
wants co-activation to flow from already has its index; what it lacks is
locations for the wholes and a writer.

**Reflection.** Two kinds of whole identify a word, and the code has neither
yet. Form-wholes (types over the parts: "ends in s", "has a double letter")
are perception's, WS rows bifurcating the field; with boundary-marked pairs
as parts (§7.3) most of Alec's examples become parts and the remaining
form-wholes are counts and classes. Membership-wholes (the sentences,
situations and documents that contain the word) are the sets one order up,
and are what makes the meaning distributional — round 2. Order is unnecessary
for either once the parts are pairs.

### 7.5 The first-order concept: centroid of parts and wholes, or the join of the parts

**Code and record.** The centroid was Alec's (2026-10-04, 6.8 plan §14.3):
L = ∨ parts, U = ∧ wholes, c = the evidence-weighted centroid, with the room
rule `L + m ≤ U` as the back-pressure. The §14 measurement (plan §15.2) found
why it failed in the gates: the only wholes a word has are its character-class
types (§7.4), so U was the same vector for every word of one type; the
centroid blended each word with its type and words of one type lost their
identity (pairwise cosines .89 → .998–.9999; the room clamp made it worse,
raising the shared coordinate for every word). Alec accepted §15.2 "if it
ensures differences": position = L, the join of the parts; the wholes contain
it, the room rule acting on the wholes only (a type grows to hold its
members). That is what the code does (`MereologicalCodes.derive` writes
`lower`; `project_room` moves only the minimal whole, by the full violation;
parts are selected by d > 0, not scaled). `GradientFlow.md` lines 72–92 and
`Architecture.md` 1190–1198 still describe the §14 midpoint and the two-sided
v/2 move; stale, to correct.

So the back-pressure exists and was not the missing piece; the missing piece
is wholes that carry identity. The centroid returns the day the wholes
identify the word (§7.4): then U differs per word and the centroid places the
word between what it contains and what contains it, as Alec intended.

### 7.6 Concepts collected into the symbols of higher-order concepts

**Code.** Coincides: the fold ladder's learned rungs raise order above the
word; higher-order concepts are placed by the operators (identity residue,
composed meanings plus the context mean, 6.8 §14.3 "order 0 only"); one
sentence is one row; an identity is its occurrences tied by references.
Nothing here diverges from Alec's picture; it is where the pairs-as-parts
rule of §7.3 would apply at every rung.
### 7.7 The chooser: continuous and gradient-trained, or hard with trial credit

**Code and record.** The chooser is not continuous. Every choice is a hard
argmax or a sampled departure (`select_logits`); the recorded softmax
probability is credited by the score-function surrogate
`K·R·p(a_dep)·(C_explore − C_greedy)` (§16.3, Alec 2026-10-05: "So we are
doing SCG?"). That IS a gradient to the chooser's parameters in the same
backward pass; what it is not is a gradient *through* the choice. The soft
superposition over operators (the CKY-chart era) was retired by 7.5; the
straight-through at the sampled action (§16.2, the hard-choice MoE kind,
GDAS/SNAS) was superseded the same day for bias and for the §13 collapse (the
blend's mixture gradient pulled every candidate onto every target — the
documented DARTS failure). Sparsely-gated MoE's router gradient (gate
probability × expert output) is identically zero here because the readers are
scale-free (§16.1). Soft code survives only in tests (`execute_superposed`,
`soft_connective_compose`); `expectationPolicyWeight` is 0 everywhere. The
word selector that exists — the decomposition pair chooser — is a softmax
trained by cross-entropy to the true pair, hard at inference: continuous in
training, as Alec wants, because it has a teacher.

**Reflection.** Three estimators were tried on compose — soft mixture,
straight-through, score-function — and all three are blind to a choice the
loss cannot see, which is the gates' situation (§5). Attention can be a soft
filter (§7.1); compose stays hard; the pair chooser already is a supervised
soft selector. The open question is not the estimator but the cost's
visibility of the choice and the field's consequence.

### 7.8 Three objectives: expectation explains the input, reconstruction the residual, output only understanding → output

**Code and record.** Today reconstruction and expectation each see the whole
input: expectation's sources, routing and targets are detached and only the
predictors train (E's row of the ownership table); its negative image
(`Meaning.negative_image`, `@no_grad`) reaches only the selected-thought
chooser's context — not the compose chooser, which rejects an `expectation`
field (`GradientFlow.md` line 503 is stale on this) — and nothing is
subtracted from reconstruction's target; the word-level surprise
`words − gain·prediction` is stored but read only as a detached mean for the
answer's concept query. Reconstruction scores the raw input bytes. "Reconstruction explains the input less the
expectation" is the negative image (spec §2.6 rev 7; item 2), decided
2026-09-20 and carried to the operators update — not implemented. Output: the
Answer row owns the reader/head/retrieval and the concluded understanding is
detached (October 4: "the affine answer has its own owner and cannot write
these sources"); the 2026-10-01 amendment that let a supplied answer train
the whole path is retired (`GradientFlow.md` lines 10–11; `Models.py` ≈20009:
"the understanding is detached at its reader; reconstruction owns its
graph"); every reader is cut (`slots.detach()`, `reader_features()`), and a
parameter with two writers raises in `ObjectiveOwnership`. Alec's statement
of the output rule is the code.

**Reflection.** Alec's partition is predictive coding (Rao & Ballard 1999):
the prediction is removed before reconstruction, so the two objectives split
the signal rather than compete for it; no weighting is needed because they
do not see the same residual. One condition: the prediction subtracted must
be formed before the input arrives (context only), or the predictor learns
to copy the input and reconstruction has nothing left (the collapse channel
already noted for expectation.intra). With output cut at the concluded idea,
nothing trains the operator choice toward what the sentence asserts unless
the output error reaches the chooser as a scalar in the trial cost (§6) —
policy credit, not a gradient path, so Alec's rule "the output gradient is
only applied from understanding to output" stays literally true.


### 7.9 What follows (for Alec's decision)

1. **Attention as a soft filter.** A weight per word (or bracket) on DETACHED
   percepts, trained pathwise by the owner-step cost with a graded budget
   cost; admission by threshold; the field's `and`/`or`/`not` stay as the
   continuous pole operations they are. Compose stays hard with trial credit.
2. **Pairs as the parts at rung 0** (boundary-marked adjacent pairs), in
   place of round 3's located fold at the word order; the same rule at
   every rung.
3. **The output error in the chooser's trial cost** as a scalar (§6): policy
   credit, not a gradient path; "output trains only understanding → output"
   stays literally true, and the operator choice can learn what the sentence
   asserts.
4. **Item 2 into this update**: the negative image subtracted before
   reconstruction (predictive coding), with the image formed from context
   before the input arrives.
5. **Docs to correct in the next hand-off**: `GradientFlow.md` 72–92
   (midpoint, v/2), 106–108 (six/fourteen), 503 (image to the compose
   chooser); `Architecture.md` 1190–1198.

## 8. Alec's replies to §7 (2026-10-06) and what follows

### 8.1 Attention: the cost factors into attended and unattended regions (Alec)

Alec: the straight-through was a second writer into perception's codes, but
the cost separates into attended and unattended regions — two
responsibilities in one cost, which should factor. Read as a per-word gate
with two explainers:

    C = Σ_i  m_i · C_i^S(sentence path; percepts detached)  +  (1 − m_i) · C_i^P(perception)

so ∂C/∂m_i = C_i^S − C_i^P: a word is attended where the sentence path
explains it better than perception alone. Perception's codes receive
gradient only from C^P (and perception's own objective) — the factoring is
the cut, and "codes are perception's" holds. C^S is the owner-step cost
(reconstruction, expectation, output on supervised sentences; §6): with
lossless inverses the byte terms of C^S and C^P nearly tie, so the mask
learns from expectation and output. The budget becomes a graded term
(λ·Σ m_i) or the per-word cost of composing; the hard allowance goes. Soft in
training, hard at test (m_i > ½, or the top-k the budget allows). The field's
`and`/`or`/`not` stay continuous in the poles. **Decided in direction.**

### 8.2 Parts: pairs, and length for repetition (Alec: "bana / banana")

Boundary-marked pairs are a set under the max, so `bana` and `banana` share
{#b, ba, an, na, a#}. Trigrams separate that pair but not the next repetition
(`banana` / `bananana` share their trigram set); length separates every
repetition, which is Alec's own condition ("order is not necessary if length
is known"), and multiplicity is where the architecture already puts it
("multiplicity and order live on the witness"). **So: pairs as the rung-0
parts, plus the letter count as one form coordinate** (the band holds only
the start offset today). Collisions of pair-set-and-length are the rare
cases identity already handles.

### 8.3 Wholes defined by parts, and why the centroid still does not return

Alec: "words that end in s" will be a concept; let parts define wholes — the
set of words that have `s#` — to return to the centroid sooner.

Parts defining wholes is the Galois connection of formal concept analysis:
each part p has an extent E(p), the words containing it, and E(p) is cheap
(a query on the inverted postings). It is well motivated (orthographic and
morphological families prime each other) and it gives the complement
letter-based wholes now, before any sentence membership exists. **Take it.**

It does not bring the centroid back, for a structural reason. In the form
coordinates a whole must contain its members: its row is at or above every
member's L (the room rule: a type grows to hold its members), so the row of
"words ending in s" is the join of all those words' parts — nearly
everything for a common part. U = ∧ of such rows is near 1 for any word
whose families are large, and the centroid again pulls the word toward what
it shares with many words: the §14 collapse with a different U. Only rare
parts give specific wholes, and then the centroid is orthographic-neighbour
smoothing, not identity. Seen from the other side, the intent of E(s#) (what
all its members share) is `s#` itself — a part. So a part-defined whole adds
nothing to the code that the part did not; it adds a set, which is what
co-activation and priming need.

Where "the wholes identify the word" does work is where the code already
puts them: the complement, one coordinate per whole, concatenated with the
form (`[form | meaning]`), not averaged into it. A concept then sits between
its parts and its wholes in the lattice sense — above its parts in the form
block, below its wholes in the complement — which is the centroid's intent
without its failure. **Recommendation: let the centroid go; L for the form,
wholes (letter-defined now, membership later) in the complement, U and the
room rule for containment only.**

### 8.4 The chooser: soft superposition in training, hard at test (Alec, wrestling)

Alec signed off on SCG but prefers a soft superposition during training,
hard at test, approximated by two complete derivations per trial.

Two readings of "approximated by two derivations". If the two derivations are
costed separately and the costs are mixed by the policy's probabilities,
the gradient is Σ_k C_k ∇p_k — which, with the greedy cost as baseline, is
exactly the surrogate in the code (`K·R·p(a_dep)·(C_explore − C_greedy)`).
SCG *is* the two-derivation approximation of the mixture of costs. What it
cannot do is mix the *values*: superpose the two roots, `p·x_1 + (1−p)·x_2`,
and read once. On reconstruction that gives nothing (the hard inverse is
piecewise constant in the root — §16.1's finding stands). On the output and
expectation readers, which are affine in the root, it gives a dense pathwise
gradient to the policy: the class read from the superposition of the
conjunction root and the disjunction root trains p toward the root the
class prefers, every step, no tie. That is the single-network form Alec
wants, confined to the costs that are smooth. Its price: the chooser
acquires a second writer (output, through the mixture weights), which the
one-writer audit raises on and which the rule "output trains only
understanding → output" forbids in letter; and the reader trains on a
mixture it never sees at test until p sharpens.

**Proposed order.** First the cost (§6): with the output error in the trial
cost, SCG has a nonzero advantage on exactly the XOR choice (conjunction and
disjunction roots read differently), so the class gate should learn under
the estimator already in place; measure that. If it does, the superposition
is an efficiency question (variance) to take up at scale; if it does not,
add the superposed read on the smooth costs as the dense term, with the
ownership decision made explicitly. Word choice is already in Alec's
preferred form: the decoder's word is soft in training (a likelihood over
the bank) and hard in production, and the decomposition pair chooser is a
softmax trained by cross-entropy, hard at inference.

### 8.5 Three objectives: ok (Alec)

The output error enters the chooser's trial cost as a scalar (§6);
expectation explains the input and reconstruction the residual (item 2),
with the image formed from context before the input arrives.

### 8.6 Decisions still open

1. Round 1: accept as reviewed in §5 (Codex is holding the uncommitted
   candidate)? Recommend yes.
2. Centroid: let it go, `[form | wholes]` concatenated (§8.3)? Recommend yes.
3. Chooser: cost first and measure SCG, superposition on the smooth costs
   only if the class gate still fails (§8.4)? Recommend yes.
4. Item 2 (the negative image before reconstruction): which round? It needs
   meanings to exist, so after the bootstrap; recommend round 3 with the
   connectives.

Decided by this exchange: the attention gate and its factored cost (§8.1);
pairs plus length as rung-0 parts (§8.2); part-defined wholes into the
complement (§8.3); the owner-step trial cost with output and expectation
(§6, §8.5). The round sequence is rewritten once 1–4 are answered.

## 9. Alec's second replies (2026-10-06): the centroid in a property space, the chooser's shared buffer

### 9.1 Round 1 accepted

Alec, 2026-10-06: "Accept: sure, we can defer some things to the next
round." The §5 findings (the attention term's silence, the chooser's prior
deciding the class) carry to round 2. Codex: commit, push, bump the
WikiOracle submodule, with the acceptance recorded on the receipt as for 6.8
§22.

### 9.2 The centroid stays; WholeSpace is a property space

Alec: start with all the values; parts become A AND B, wholes A OR B; a whole
can be "the spans that contain a part", so wholes can be as narrow as parts;
and since wholes start with no atoms of their own, WholeSpace is better
defined as a property space. He will not let the centroid go if its only
failure was that the wholes could not identify anything.

He is right, and the reformulation makes the centroid work. In extent terms
WholeSpace starts at the top (all spans) and narrows by predicates; the
narrowest predicate is "contains p". In code terms a predicate's code is its
*intent* — what defines it — which is narrow: one coordinate (or a sparse
code) per property, in the same feature cube as the parts. Then:

- Parts, from the segmentation: P(w) = the boundary-marked pairs (+ length);
  L = ∨_{p∈P(w)} c_p, evidence-selected, as today.
- Properties, from predicates over the span: Q(w) ∋ "contains p" (one per
  part: the atoms of the property space, sharing p's coordinate), the
  classes (letter, digit, …), counts ("double letter", "two c's"), length
  classes, and later "occurs in S" — membership one order up is a predicate
  too, which is what the complement already holds. J = ∨_{q∈Q(w)} c_q.
- The symbol: c(w) = (W_P·L + W_Q·J)/(W_P + W_Q), Alec's evidence-weighted
  centroid of §14.3, now non-degenerate: J is specific to w because each
  property's code is narrow. The §14 collapse came from U being the dense
  *row of the containing type*, pushed by the room rule to dominate every
  member — a common mode shared by all words of the type. A narrow "letter"
  property is one shared coordinate among many; cosines are unaffected.
- Both joins are from below (presence with evidence). The "from everything
  down" scaling `1 − d(1 − c_w)` and the meet U = ∧ belonged to the
  interval [L, U] in one coordinate system; with narrow property codes the
  meet of two properties is empty and the scaling is all-ones, so they go.
  The containment order (part-extent ⊃ … ) lives in the index (postings and
  refs), where it is exact, not in coordinate-wise ≤; the room projection
  has nothing left to enforce and is retired.
- Continuity with today: with only "contains p" properties, J = L and
  c = L, the accepted form. Each added property moves c toward what contains
  the word. Nothing to re-measure until properties are added.
- Width: the content block must hold the part inventory (≈700 pairs plus
  properties) as a sparse superposition; the gates' 14 coordinates cannot
  (the join is already lossy at 14, §14.3). A config question for round 3.

### 9.3 The chooser: two derivations written to one buffer

Alec: write the output twice to the same buffer and backpropagate; if the two
answers are mixed, do we pull out the right one; is that soft superposition;
and can we present the more correct of the two?

Yes, with one condition. With `y = p·y₁ + (1−p)·y₂`, p the chooser's
probability of the first derivation (renormalized over the two), the
gradient on p is `∇L(y)·(y₁ − y₂)`: p moves toward the derivation whose
output lies in the descent direction. For a squared error that is exactly
`L(y₁) − L(y₂)` at p = ½ and to first order elsewhere — the same quantity
SCG credits, but pathwise and dense, no sampling, no silence at a tie of
the hard costs when the outputs differ. That is the mixture-of-experts gate
gradient, the soft superposition of outputs (`execute_superposed` has these
semantics, test-only today). The condition: the weights must be the
chooser's probabilities. Writing the two outputs unweighted into the buffer
puts no chooser parameter in the path, and nothing is pulled.

Presenting the better: both derivations are already costed separately
(§16.1 "the commit"); commit and present the argmin, train on the mixture.

Three consequences.
1. It works where the loss is smooth in the buffer: the answer reader
   (affine in the root) and expectation's readers. On reconstruction the
   hard inverse is piecewise constant in the root, so the mixture gives
   nothing there; SCG keeps that part. Both terms come from the same two
   trials.
2. The reader can learn to read the mixture, and then p never sharpens: the
   gate sits at ½ and the hard choice at test is a coin flip (the DARTS
   collapse, §16.1, in its mild form). Require sharpening — an entropy
   penalty or annealed temperature, or the SCG term itself, which prefers
   the better single derivation — and validate on the hard-choice class
   bar, never on the mixture's.
3. Ownership: the output's gradient reaches the chooser through p. The
   chooser is a map, and maps are trained by the objectives (2026-09-21:
   codes by distribution, maps by the objectives), so declare it shared
   like the operators and move it from reconstruction's list to the shared
   list in the ownership audit. Alec's rule "output trains only
   understanding → output" then holds for codes and perception, not for
   the chooser; said explicitly.

### 9.4 Item 2 needs no meanings (Alec)

The expectation's confidence gates the image: a prediction with no
confidence subtracts nothing, so the negative image can enter before
meanings exist and switches itself on as they appear. Item 2 is therefore
free to go in any round; proposed below with the credit changes, since both
are about what the trial cost sees.

### 9.5 Proposed round sequence (replaces §2)

- **Round 2, credit.** The owner-step trial cost (reconstruction,
  expectation, output on supervised sentences) for compose and attention;
  attention's explore walk through the sentence; the narrowed field handed
  off with the scope; the shared buffer (§9.3) with SCG retained on
  reconstruction, the chooser shared, the better derivation committed and
  presented; the negative image gated by confidence (§9.4). Expected: the
  class gate learns (conjunction chosen by cost, not prior); reconstruction,
  sum, MM_xor unchanged.
- **Round 3, identity.** Pairs plus length as the rung-0 parts; the content
  width; the has-a wholes and the narrowing rule (§12.1); the centroid, with
  the inherited-part containment order preserved after placement (§16 item
  7); the room projection retired. Expected: anagrams separate; gates
  unchanged.
- **Round 4, meaning.** Membership as properties ("occurs in S"), the
  wholes' locations by co-activation through shared wholes (owner:
  reconstruction of membership, §6), the gate corpora with distinct wholes,
  Kleene connectives over meanings.
- **Round 5, attention.** The soft filter with the factored cost (§8.1), the
  graded budget, hard at test.
- **Round 6.** The catalogue's remaining sections (determiners, relations,
  definitions, name-tested operators to declared properties, item-7 residue).

One text per round; the standing gate as §1, amended: a gate whose training
is shown trajectory-identical to the landing is not a regression test of the
round (§5).

## 10. Alec's third replies (2026-10-06): one bank, the attention-head chooser, the invertible path

### 10.1 One bank, two relations

Alec: see it as "is X" and "contains X" over one shared bank {X, …} that
grows only when necessary to separate two objects.

The shared-index invariant generalized. One bank; **contains** from the
segmentation (a word's pairs; a sentence's words), **is** from the predicates
(the thing's own row, its classes, later "occurs in S"); the symbol
`c = (W_C·∨contains + W_I·∨is)/(W_C + W_I)`; containment exact in the index.
Growth by collision: today identity rows mint on first sight (`RadixLayer.insert`,
content-addressed — which is "necessary to separate" for identities);
property rows split on dispersion (`maybe_split_property_row`: assignment
variance over ≥ `lbgMinCount` pulls above `lbgThreshold`), not on collision;
and a pair inventory would mint every pair seen. The collision rule replaces
the last two: a thing's code must differ from every other thing's; when a new
thing coincides with an old one, mint the cheapest distinguishing feature (an
unshared pair, a count, the length) and otherwise nothing. Round 3.

### 10.2 The chooser is an attention head; mix the values per round

Alec: with nonlinear weights the chooser makes a relatively hard choice, like
an attention head over locations; sold on soft superposition given such a
decision mechanism exposed to the gradient.

The chooser already is that head: anchor-dot or MLP scorer
(`score_unary`, `chooser='mlp'`), softmax over (location, operation)
candidates, peaked when confident. What differs from soft superposition is
only whether the branch *values* are mixed by the weights (pathwise) or
selected by argmax (SCG). **Round 2 design:** per-round mixture over the
operations — each round computes every eligible operation's value and the
next round sees their weighted sum (`execute_superposed` semantics, test-only
today); the committed tree is the per-round argmax, written to LTM and
presented; expectation's and the answer's readers read the mixed root in
training and the committed root at test; the SCG term stays for the hard
inverse's cost; sharpening required (entropy term or annealing); the
hard-choice class bar is the gate. The two-derivation buffer (§9.3) is the
two-sample approximation, kept as the cheap variant. The CKY-era
superposition was retired for the DP tiling (7.5), not for failing. Input
narrowing uses the same head; its per-word form is the filter of §8.1.

### 10.3 The invertible forward path: reconstruction as an audit

Alec: force the forward path up through percepts to be invertible; then
reconstruction is automatic, expectation and output integrate, and one
gradient path runs through the architecture, the chooser aside.

Already invertible: the SVD-factored projections (`invertible`: y = xVΣUᵀ,
x = yUΣ⁻¹Vᵀ), the butterflies, the LDU readout, the operators' tied inverses
(parameterless kernels; verb/adverb maps invertible given the modifier), and
the content-addressed bank as clean-up memory. Not invertible today:
(a) the form from bytes — the max loses order, multiplicity and length
(round 3: pairs plus length, injective at sufficient width); (b) binding —
`a·b·unit(u∘v)` on [0,1] directions kills silent coordinates and commutes, so
operands return only through the bank search, exact when forms are injective
and the width suffices, order from the roles; (c) the chooser (§10.2).

With (a) and (b) exact: reconstruction is a structural audit that must read
zero, not an objective; the decomposition chooser retires (the search is a
lookup); the trainable surfaces are the chooser, the maps (verb, adverb,
lift/lower), the predictors and the readers, trained by expectation and
output along one path; the collapse channel closes structurally (an
invertible path cannot discard information); the balance question reduces
to E against O on the shared maps, guarded by the dissonance diagnostic.
Price: width — a sparse superposition of the pair inventory and exact
clean-up over the vocabulary, far above the gates' 22. The gates change
meaning: the XOR reconstruction bar becomes the audit; class and expectation
are the learning gates. **Target of rounds 3–4; the reconstruction loss stays
until the audit reads zero, then retires.**

## 11. Reflections (Claude, 2026-10-06; Alec asked for thoughts)

1. **What it converges on.** A content-addressed bank as clean-up memory,
   structural binding with exact unbinding, an invertible path, and learning
   only where there are degrees of freedom (router, maps, predictors,
   readers): a holographic/VSA memory with a learned router; "is X /
   contains X" over one bank is the IS-A/HAS-A network (Collins & Quillian
   1969) on a lattice with evidence poles.
2. **The decision §10.3 quietly makes.** Invertibility is what makes
   "expectation trains the whole path" safe; the 2026-09-20 rule detaching
   expectation's sources existed against collapse, which an invertible path
   cannot do. It retires with the reconstruction objective, and the
   world-model signal is free to shape the verbs.
3. **Amendment to §10.2: hard forward, soft backward.** A per-round mixture
   makes the forward a weighted sum of trees — not a tree, not invertible,
   not writable. The form consistent with §10.3 is the committed tree in the
   forward at every round, the gradient to the chooser taken as if the
   branches had been mixed (the straight-through of §16.2, removed
   2026-10-05 for the blend collapse — since fixed by detaching codes — and
   for bias). Bring it back in that form; keep the two-derivation comparison
   as the unbiased check and the commit rule; measure the bias once.
4. **Commitment over variance.** A chooser at ½ has no reason to sharpen if
   the readers learn the mixture. Structural guard: LTM written from the
   argmax, readers at test on the argmax, the class bar on the argmax; plus a
   sharpening term (Switch's z-loss / annealing).
5. **The bank needs a merge rule.** Collision-minting only separates;
   abstraction comes from the sigma fold one order up, co-activation, and
   the forgetting spec's value pruning unused distinctions. Collision-minting
   is order-dependent (inventories differ run to run).
6. **Width, computed.** Exact clean-up of k bound items against N entries
   needs D ≈ c·k·log N — thousands for production; clean-up through the
   index, never brute force. Derive the number before round 3.
7. **Attention shows itself only at scale.** Round 5's gate must have more
   words than budget; choose it before building the filter.
8. **Milestone.** Reconstruction audit at zero at production width ⇒ item 0
   trains by expectation and output through one path.

## 12. Alec's fourth replies (2026-10-06): the interval kept; surprise throughout the chain

### 12.1 Centroid between max(parts) and min(wholes); has-a concepts one order up

Alec: keep centroids between max(parts) and min(wholes); the existing
WholeSpace's upper bound is poor, but leave it as cuts over the alphabet;
learn the concept "has-a" at the next order (an easy generalization over the
part), whose symbol — its centroid over its constituents — is the ceiling.
Locate the centroid better at a higher order; do not change WholeSpace.

Adopted; it supersedes §8.3 ("let the centroid go") and §9.2 ("both joins
from below"). The has-a concept for a part p is the sigma fold over p's
occurrences — the things containing p — one order up; its symbol is its
centroid over its constituents, above each of them by construction; a word's
ceiling is the meet of the has-a symbols it belongs to, together with the
alphabet cuts. **The rule that makes the ceiling informative: a whole bounds
in proportion to how much it narrows** — its distance from "thing", the
domain (spec §2.0.1): in `U = ∧ (1 − d_w(1 − c_w))` the edge evidence `d_w`
is multiplied by the whole's narrowing `s_w = 1 − |extent(w)| / |domain|`.
"Letter" has `s = 0` and bounds nothing (the §14 collapse was a zero-narrowing
whole given full weight); "has s#" bounds by about ¾; "has zq" by nearly 1.
The evidence-weighted centroid of §14.3 is otherwise unchanged; the room
rule becomes a consistency check (a has-a symbol is at or above its
members). Prerequisite: the wide sparse cube of round 3 — at fourteen dense
coordinates any join of more than a few items is near the top.

### 12.2 Two complete derivations, subtracted layer by layer

Alec: compute the two derivations independently, subtract one from the other
along the chain; expectation carries out a complete derivation and we learn
on surprise throughout the chain, not only at the output.

This is two-phase contrastive learning — contrastive Hebbian learning,
equilibrium propagation (Scellier & Bengio 2017: free vs. nudged phase,
layer-wise difference → the gradient in the limit), forward-forward (Hinton
2022), difference target propagation (Lee et al. 2015). The present trial
(greedy vs. explore, keep the better, readers on the kept rows) is its
skeleton. If parameters rather than activations are meant to differ between
the derivations, that is weight perturbation (SPSA), workable without
backprop but with variance growing in the parameter count; the activation
difference is the reading that scales.

Validity: the difference is a local signal for a layer only where that
layer's parameters could have produced the better state — the chooser at
the departure round (which is SCG: surprise at the choice; unchanged), the
readers (already the kept trial), and the maps downstream of the departure
for their share of the difference; a difference caused by a different
operator is not a map's error. **Under the invertible path (§10.3) that
share is exact:** the better row inverted through the committed operations
gives each layer its target — exact target propagation, no gradient through
the choice, no straight-through, no mixed forward. This supersedes §11 item
3 (hard forward, soft backward): the chooser keeps SCG; the maps take
inverted targets from round 3–4; the straight-through stays out.

Expectation in the same frame: the predicted row, inverted, is the expected
words — the word-level expectation follows from the row-level predictor by
inversion rather than by its own predictor — and the confidence-gated
surprise at every level is item 2 generalized from one face to the chain.

### 12.3 Round 2 as it now stands

The owner-step trial cost (reconstruction, expectation, output on
supervised sentences) for compose and attention; attention's explore walk
through the sentence; the narrowed field handed off with the scope; SCG
retained as the chooser's rule (the two-derivation buffer of §9.3 and the
mixture of §10.2 withdrawn in favour of §12.2); the chooser shared in the
ownership audit only if a second objective reaches it (none does under
§12.2 — it stays reconstruction's, with the cost it compares now including
expectation and output); the better derivation committed and presented; the
confidence-gated negative image at the row. Rounds 3–4 add the invertible
path, the targets by inversion, the has-a wholes and the narrowing rule.

### 12.4 SPSA, and the layer-wise comparison that replaces it (Alec, 2026-10-06)

Alec: SPSA is interesting; to keep variance bounded, compare the two
derivations layer by layer, so that expectation need not complete a full
derivation without access to any propagated input.

SPSA (Spall 1992): simultaneous random ±δ perturbation of all parameters,
two complete evaluations, every parameter credited with the one cost
difference over its own ±δ. No backprop; variance grows with the number of
parameters perturbed together; layer-wise perturbation bounds it per layer
at a derivation pair per layer. Here the non-differentiable places are the
choices (SCG is SPSA restricted to one action with the score function doing
the credit assignment — it knows which choice differed) and the hard search
(no signal once the search is exact, rounds 3–4). Not adopted; noted as the
tool for a black-box layer, of which there is none.

The layer-wise comparison is §12.2's inversion. The only unconditioned
prediction is the row-level one (the world-model: the next row from the
previous rows; conditioning it on the current input is the copy leak).
Inverted through the committed operations of the actual derivation it gives
an expected operand at every round, conditioned on the actual structure; the
actual operand against it is that layer's surprise. Expectation completes no
derivation of its own: one inverse pass on the existing reverse path. From
below, the teacher-forced next-step predictor (the word-level expectation
of 6.8-1 at the word order) conditions on the propagated input. Prior from
above, next step from below, the actual state against their combination at
each layer — predictive coding's two directions, neither free-running.

Variance budget: every layer's error is a deterministic local quantity (no
sampling in the parameter direction; exact toward the target for an
invertible layer); the expectation's own noise is bounded by the confidence
gate; the one stochastic estimate left is SCG at the departure round. Two
derivations for the chooser and the commit; one inverse pass for the
expectations at every layer; no parameter perturbation. Carried into rounds
3–4 as "expectation by inversion".

## 13. The scheme as confirmed (Alec, 2026-10-06)

Alec: "we are continuing with centroid for percepts/symbols, expectation and
two hard derivations propagated across the model, and an output criteria
for backprop learning (since reconstruction is handled by using 'mostly
invertible' transformations)." Confirmed, with three precisions:

1. **Centroid.** The symbol between max(parts) and min(wholes); wholes = the
   alphabet cuts plus the has-a concepts one order up, each weighted by its
   narrowing (§12.1). Informative only in the wide sparse cube — it arrives
   with pairs and width in round 3; until then the symbol is L.
2. **Expectation and two hard derivations.** The forward is always a tree:
   greedy and one departure, complete, costed at the owner step; the better
   committed and presented; the chooser by SCG on the cost difference. Round
   2: expectation in that cost and the confidence-gated image at the row.
   Expectation propagates across the model (expected operands at every
   layer by inversion, gradients into the maps) only once the path is
   invertible (rounds 3–4); before that its sources stay detached.
3. **Output and expectation as the criteria.** Output where supervised,
   expectation everywhere, one path once invertible. Reconstruction retires
   to an audit only where the transformations are exactly invertible
   (injective forms, exact unbinding through the bank, the committed tree);
   where "mostly" means approximately, the reconstruction term stays as a
   loss for that part. Round 2 keeps it; it goes when the audit reads zero
   at production width.

## 14. Round 2 hand-off to Codex (Claude, 2026-10-06)

Round 2, "credit": what the trial cost sees. Decided in §6, §8.1 (direction
only; not this round), §9.1, §12.3–§12.4, §13. Specification references:
[GradientFlow](../GradientFlow.md) ("The training step", "Codes are
perception's", "Attention's credit detached", "Operators update round 1"),
[accessible mind §2.6.1](../specs/2026-09-20-accessible-mind-subsystems.md#261-the-rule),
[catalogue §3.8 and §12.1](../specs/2026-09-29-operator-catalogue.md).
Start from the round-1 source as accepted.

### 14.0 First: land round 1

Alec accepted round 1 on 2026-10-06 ("sure, we can defer some things to the
next round"; the §5 findings carry here). As for 6.8 §22: record the
acceptance on the round-1 receipt README (status line) and an acceptance
record beside it (candidate manifest hash, acceptance date, documents
changed by the acceptance only); no source change; commit; push; bump the
WikiOracle submodule. Commit, push and bump are authorized by Alec.

### 14.1 Scope

One gate-moving change: the cost the two derivations are compared on and
credited by. It has three parts that only make sense together (14.2–14.4),
one mechanism that cannot move a gate in these configurations (14.5), and
the documentation corrections (14.7). Unchanged: SCG as the chooser's rule
and its registration; the pair search and the decomposition chooser; the
reconstruction objective and every owner list; the thirty trainings, their
budgets, the §20.5 bands; no seed, no retry, no tuning. Not this round:
pairs and length, width, the property space and the narrowing rule, the
invertible path and targets by inversion, the soft attention filter, the
catalogue's remaining sections.

### 14.2 The trial cost is the owner-step total

For each of the two derivations, `C_k = R_k + E_k + A_k`, as the Error
registry defines them (relative errors):

- `R_k`: the reconstruction total, as today.
- `E_k`: the expectation terms for this sentence (all-role relative error
  and presence/kind, against the trial's row), scaled by the same gate as
  the image, `g · κ` per role (§2.6.1), so an expectation that is not yet
  confident decides nothing. Targets remain detached; the predictors' own
  training is unchanged.
- `A_k`: the answer error where the sentence has a supplied answer, the
  reader applied to each trial's root for the comparison (a detached read,
  no optimizer step); the reader itself trains only on the kept rows, as
  today. Absent otherwise.

Keep rule: explore is kept only when `C_explore < C_greedy` strictly; a tie
keeps greedy. The score-function advantage is `C_explore − C_greedy` on these
totals; the surrogate, its reduction and its registration are unchanged.
Reconstruction and expectation train on both trials as now. Record the three
components per trial and which component decided the keep.

### 14.3 One departure per sentence, over both walks

The input-narrowing walk and the compose walk share the chooser; they now
share the sentence's single departure. `R` counts the eligible rounds of
both walks; the departure round is drawn uniformly over them; at that
round the departure action is drawn uniformly over the eligible
alternatives (`K`), as now. If the round lies in narrowing, the explore
derivation continues from the departed narrowing through admission and
compose greedily; if in compose, the narrowing is greedy's. Both derivations
are costed at the owner step by 14.2; one surrogate term per sentence,
`K·R·p(a_dep)·(C_explore − C_greedy)`, at the departure's walk. Retire the
percept-stage selection: `narrowing_pair`'s byte-cost comparison,
`percept_reconstruction_score` and the separate attention surrogate
(`attention_score_function`); the narrowing is selected by the sentence's
keep. The reconstruction consumer of the attention term goes with it.

### 14.4 The narrowed poles are handed off with the scope

Today the walk's field operations change nothing outside the walk (§5).
After the walk, each word's pole pair is the pair the walk computed — the
result of the `and`/`or`/`not` reductions over the brackets containing it,
`field_reduce`'s bilattice — or its native pair where no field operation
reached it. That pair is the word's evidence in the sentence path: the
explicit pair where the grammar uses the pole representation, the net
evidence of the leaf's features otherwise, and the clause's polarity at the
closing. `_attention_native_poles` gains its consumer. Values are not
handed off this round (connectives over meanings are round 4). Observable:
a `not` departure over a word flips the explore trial's leaf evidence and
closing polarity, and the two trials' costs can now differ.

### 14.5 The image at the closing, on the concept face (item 2, mechanism)

Implement §2.6.1 as written: `n = −g·(1 − m) ⊙ κ ⊙ ê`, `c = o + n` at the
closing, `o = c − n` exactly; `g` the gain element (default 1; 0 is
beginner's mind, §2.6.6), `κ` the predictor's per-role presence, `m` the
attention on the role (§2.6.7). The image acts on the concept face only;
the form block is never touched (FutureWork, item 2 amendment of
2026-10-04). The conceived `c` is what thought's context reads (replace the
ad-hoc `Meaning.negative_image` use); storage keeps `o`; the predictors'
targets stay `o`, detached. No new loss, optimizer or owner. In
XOR_grammar and MM_grammar the concept face has zero width, so the image is
identically zero there: report the face width and the zero. The mechanism
test runs on a fixture with a nonzero complement and checks the six rows of
the §2.6.1 table, `κ = 0 ⇒ c = o`, and the exact restoration.

### 14.6 Measurement and the gate

The standing gate is §1 as amended by §5: no regression against the 6.8
landing (class 7, reconstruction 9, MM_xor 10, sum 10, sweep green, zero
perception gradient on the sentence path, zero displacement, zero
ownership conflicts); a gate whose training is trajectory-identical to the
landing is not a regression test of the round (MM_xor is live again this
round: its narrowing now has a cost that can see it). The full sweep once
on the delivered source, then the thirty trainings, bands §20.5, the class
bar read on the committed derivation's root — never a mixture. The audit
adds: the cost components per trial and the deciding component; departures
with nonzero advantage, counted by walk and by action; the chooser's logit
ranges per epoch (expected to move); each run's final operators with the
cost that chose them; for MM_xor, whether its trajectory differs from the
landing's. Expected: class at or above the landing with conjunction chosen
by cost where disjunction reads worse; reconstruction 10/10; sum 10/10;
MM_xor 10/10. Receipt directory `doc/benchmarks/2026-10-06-operators-round2/`:
frozen source manifest and archive, diff at freeze, measurement-helper
hashes, complete old/new texts of every ported test, process logs, every
run including failures. No seed, no retry, no replacement run. Stop for
Claude's review before any commit.

### 14.7 Tests and documents

Focused tests: (i) the cost composition and keep rule — tie keeps greedy,
a lower total keeps explore, a zero-gated expectation contributes nothing,
the answer term exists only with a supplied answer; (ii) one departure over
both walks — `R` counts both, `K` at the round, a narrowing departure
continues through the sentence, one surrogate per sentence, its analytic
gradient against finite differences at a narrowing departure and at a
compose departure; (iii) the pole handoff — `not` over a word flips the
leaf's evidence and the closing polarity, `and`/`or` over a bracket give
the bilattice pair, costs differ between trials; (iv) the image per 14.5;
(v) the presented answer is the committed trial's. Port the existing tests
that assert the percept-stage selection or the attention term's separate
registration; complete old/new texts saved.

Documents: the stale passages of §7.9 item 5 are already corrected by
Claude (2026-10-06: GradientFlow's form/room/width/image lines,
Architecture's position paragraph) and the target scheme is stated in
[Architecture](../Architecture.md#the-scheme-confirmed-2026-10-06-expectation-and-surprise-through-the-architecture),
[GradientFlow](../GradientFlow.md#the-target-scheme-october-6-decided-scheduled-by-rounds)
and [Philosophy](../Philosophy.md#learning-is-surprise-at-every-layer-2026-10-06);
leave those as they are. Codex adds: a GradientFlow "Operators update round
2" section in the convention of round 1 (the total, the single departure,
the pole handoff, the image, with the ownership audit's reading), and
replaces the sentence "The answer and expectation never enter the
comparison" in "The training step"; catalogue §12.2 for this round;
FutureWork's status note and todo's round line marked landed when they are.
The plan is Claude's and is not edited.

## 15. Review of round 2 (Claude, 2026-10-06): not accepted; two of the causes are this plan's

**Round 1's landing is verified.** `73cd7b71b` matches all 698 runtime hashes of
the round-1 frozen manifest (`c49d6977…`); the acceptance record matches; the
WikiOracle bump `c9670b5` points at it; both are on `origin/main`.

**Round 2 fails the standing gate** (class 4/10, reconstruction 6/10, MM_xor
9/10 against 7 / 9 / 10) and is not accepted. The receipt
(`doc/benchmarks/2026-10-06-operators-round2/`) is complete and honest; the
implementation follows §14. The causes, from its audits:

1. **The keep by the per-row total lets the answer choose the operator per
   sentence.** Run 1's final training step: row 0 keeps the conjunction
   departure by the answer (A 1.21 → 0.007), row 1 keeps the disjunction
   greedy by the answer (A 0.006 against 1.28). Across the ten runs 3,676 of
   the 5,426 decided keeps were the answer's alone. The reader then trains on
   roots chosen to suit it, row by row, and at evaluation — no answer, the
   policy's greedy root — reads nothing (run 1's four answers all ≈ .90). The
   commit must be reproducible without the answer; §14.2 made it depend on
   the answer, and that was wrong. **Correction:** the keep is reconstruction's
   (strictly lower R keeps explore, a tie keeps greedy); the answer and the
   gated expectation enter only the chooser's advantage — the policy exists
   at test, the selection does not.
2. **Even in the advantage, the answer favours the incumbent operator,
   because the reader trains only on the kept roots.** A departure's root is
   one the reader never fitted, so it reads worse by unfamiliarity: run 1
   (disjunction policy) scored conjunction departures better 131 times and
   worse 187; run 10 (conjunction policy) scored disjunction departures
   better 169 and worse 338. The credit cannot flip a policy under a reader
   co-adapted to it. **Correction:** the reader trains on both trials' roots
   (the answer is the sentence's, not the derivation's; the reader stays
   answer-owned, the roots detached). Then conjunction roots, being
   XOR-separable, read well everywhere and disjunction's sit at the floor,
   and the advantage prefers conjunction in every row.
3. **The pole handoff scales the leaf's form and overwrites its certainty.**
   `_attention_sentence_payload` multiplies the leaf event by `net/old` and
   sets its activation to `net`. Disjunction roots no longer decompose:
   evaluation reconstruction cost .014–.029 against the landing's ~1e-15, one
   word recovered per sentence in all four disjunction runs ("hello world" →
   "hello"); conjunction runs recover 4/4. §14.4's "the net evidence of the
   leaf's features otherwise" is what licensed this, and it was wrong: the
   walk's pair is the leaf's *pole* — the sign of its activation, the
   explicit pair where the grammar uses the pole representation, the
   clause's polarity at the closing — and never a scale on the form or a
   replacement of the certainty ("certainty is the activation", October 4).
   **Correction** as stated; Codex to confirm the disjunction decomposition
   returns to the landing's cost under the corrected handoff, or state the
   cause.
4. **MM_xor is live, and the receipt's reason for saying otherwise is a gap
   in the observer** (*attribution to the reference slab withdrawn in §17;
   the trajectory difference stands, its path to be named in round 2c*). The narrowed pair also reaches
   `commit_word_reference_slab(evidence=_attention_poles)` on both the
   per-word and the whole-slab paths (`Models.py` ≈19798, ≈21585), where it
   replaces the pair derived from the signed activation as
   `_word_reference_evidence`. The observer counted only
   `_attention_sentence_payload`. Measured: the same seed gives identical
   parameters and RNG state after construction and a different first forward
   output between the landing and the round-2 source (2.009398 vs 2.009284,
   deterministic within each). So the §5 amendment does not apply: MM_xor's
   9/10 counts, and the round changed its path. The consumer list of the
   handoff must be declared and audited in full.
5. Observations, not findings: the expectation component is identically
   zero in these gates (no prior context), so the gated term is exercised by
   its focused tests only; the closing image is zero by width, as expected;
   at evaluation the failing runs' narrowing chose `descend` on the two-word
   bracket where the passing runs chose `divide` — *withdrawn in §17: the
   encoding I decoded was not the frozen one; all forty final sentences
   descend, and the failing runs' distinguishing mark is a `not`/`and`/`or`
   prefix over the bracket.*

**Round 2b** keeps §14 with four corrections — the keep rule (1), the reader
on both trials (2), the pole-only handoff (3), the handoff's declared and
audited consumers including the reference slab (4) — the same measurement
protocol, and the §5 amendment applied only where a paired replay shows the
trajectory identical. The text is §16.

## 16. Round 2b hand-off to Codex (Claude, 2026-10-06)

Start from the frozen round-2 candidate (`delivered-source/source.json`,
`63d68a9b…`). Everything in §14 stands except the following.

1. **The keep is reconstruction's.** Explore is kept only when its
   reconstruction total `R` is strictly lower; a tie keeps greedy. `E` (gated
   by `g·κ`) and `A` do not enter the keep. The advantage credited by the
   surrogate is `C_explore − C_greedy` on the total `R + E + A`, as in §14.2.
   Record, per trial, the three components, the keep's `R` decision and the
   advantage's sign, so the audit can show when the answer moved the policy
   against the keep.
2. **The reader trains on both trials' roots.** For a sentence with a supplied
   answer, the answer-owned reader takes its step on the greedy root and the
   explore root alike, both detached; its comparison read for the advantage
   is unchanged. Ownership lists are unchanged; the audit's "answer cost
   counts only the kept trial's rows" becomes "both trials' rows".
3. **The handoff is the pole only.** The walk's pair for a word sets the
   sign of the leaf's activation (positive if `for > against`, negative if
   `against > for`, unchanged magnitude), supplies the explicit pair where
   the grammar uses the pole representation, and sets the clause's polarity
   at the closing. It never scales the leaf event and never replaces the
   activation's magnitude. `_attention_sentence_payload`'s `event * scale`
   and `activation = net` go.
4. **The handoff's consumers are declared and audited.** List every reader of
   `_attention_poles` (today: `_attention_sentence_payload`,
   `_pushed_word_slab` under the pole representation,
   `commit_word_reference_slab` on both paths); the observer counts calls to
   each; the receipt states which are reached by each gate configuration.
   For `commit_word_reference_slab`, the evidence stored is the pole-only
   pair of item 3 (which, for a word no field operation reached, equals the
   pair derived from the signed activation, so the landing's behaviour is
   recovered exactly there).
5. **MM_xor.** It is on the handoff's path through the reference slab (§15
   item 4). Its count is live. In addition to the ten trainings, a paired
   replay of three seeds against the landing (same seed, same epochs,
   trajectory equality) is recorded; the §5 amendment applies only if the
   trajectories are identical.
6. **Disjunction reconstruction.** Confirm on the frozen round-2 fixture
   that the corrected handoff restores the disjunction root's decomposition
   (reconstruction cost at the landing's level, four multisets recovered)
   before the thirty trainings; if any residual remains, state its cause.

7. **Inherited-part containment in the form code (Alec, 2026-10-06).** For
   first-order concepts: if every part of X is also a part of Y (the parts
   with positive net evidence), require `c(X) ≤ c(Y)` coordinatewise in the
   perceptual/form code, and keep it so after any placement step. This is a
   partial order — `ab` above `a`; `ab` and `ac` incomparable — on forms only;
   contextual meanings (the complement) are unconstrained; a higher-order
   fold retains the guarantee only if it explicitly preserves the order
   (monotone in its inputs), and this round asks for no enforcement above
   order zero. Today the form is `L`, the join of the parts, so the order
   holds by construction wherever evidence selection is consistent: add the
   bank-wide audit now (every pair with `parts(X) ⊆ parts(Y)`, found through
   the postings as the intersection of the extents of X's parts; count and
   largest coordinatewise violation, expected zero) and a focused test
   (`a`, `ab`, `ac`: `L(a) ≤ L(ab)`, `L(a) ≤ L(ac)`, `ab` and `ac`
   incomparable). State for each higher-order fold whether it is monotone.
   When the centroid is placed (round 3), the placement step ends with a
   projection that caps each contained thing by its containers, processed in
   decreasing part count — `c(X) ← c(X) ∧ ⋀ c(Y)` over the Y containing X —
   which keeps `c(X) ≥ L(X)` (every `c(Y) ≥ L(Y) ≥ L(X)`) and never raises a
   container above its ceiling; the audit reports violations before and
   after, zero after.

Measurement, receipt, tests and documents as §14.6–§14.7, in
`doc/benchmarks/2026-10-06-operators-round2b/`; the round-2 directory is
preserved intact. Stop for Claude's review before any commit.

## 17. Review of round 2b as it stands (Claude, 2026-10-06; Codex not yet finished)

Receipt `doc/benchmarks/2026-10-06-operators-round2b/`: class 1/10,
reconstruction 6/10, MM_xor 10/10 (live), sum controls 10/10 but all ten
**above ¼** (.28–.31; the landing's were at ¼, .250–.260), sweep green. Not
acceptable; the causes are visible in its audits and two of §16's items need
tightening.

1. **Negation reaches the form through the activation.** Codex's diagnosis
   (`disjunction-result.json`): `InterpretLayer.forward` resolves a reference
   as `object_atoms × activation`, so the pole-only handoff's negative sign
   still negates the word's *form* at interpretation; `−u − v − u∘v` is not
   the negation of `u + v − u∘v`, so a negated disjunction cannot recompose
   from positive dictionary pairs (conjunction cancels two signs, hence the
   asymmetry). The evaluation traces (`round2-narrowing-recheck.json`) show
   every failing run's greedy narrowing applying `not`, `and`, `or` to the
   whole two-word bracket before descending; every passing run descends
   first. A diagnostic that interprets with the activation's magnitude
   recovers 4/4 (not installed, as §16 permitted). This is the spec: the cube
   has no additive inverse — negation exists for concepts, never for percepts
   (accessible mind §2.6.2); `not` is a pole exchange, never a code's sign
   (catalogue §3.8). The serial leaf `[form | meaning] × signed activation`
   applies the sign to both faces; it belongs to the meaning face only.
2. **The reader was trained on misreadings, twice per sentence.** §16 item 2
   said "both trials' roots"; the single departure falls in narrowing about
   three-quarters of the time (run 8: 1,184 of 1,600), so most explore roots
   were narrowing variants — `not` over the bracket, a negated root — paired
   with the sentence's true answer, and the reader took a separate step on
   each trial (8,000 reader steps for 4,000 sentences). In the sum control
   every departure is a narrowing one, and the control's reader drifted above
   the floor. The answer then rewarded the misreading: 1,237 credits where
   adding the answer reversed the keep's direction, which is how `not`
   entered four runs' greedy policies. Item 2 was right for compose
   departures (the same words, the same poles, composed differently) and
   wrong for narrowing departures (a different reading, to which the
   sentence's answer does not apply).
3. **MM_xor is live, but not through the reference slab.** The consumer census
   is explicit: `MM_xor` (`word_brackets=False`) reaches none of the four
   declared consumers, while the paired replays differ at the first forward.
   §15 item 4's attribution to `commit_word_reference_slab` is withdrawn; the
   path by which MM's forward changed is not yet named, and a live gate with
   an unnamed path is a hole to close.
4. **Withdrawn:** §15 item 5's `divide`/`descend` observation. The frozen
   encoding differs from the one I decoded; the recheck shows all forty
   final evaluation sentences descending on the two-word bracket.

Containment audit (§16 item 7): landed; vacuous on the gate banks (no
comparable pairs), nontrivial on the `a`/`ab`/`ac` fixture; fold
monotonicity tabulated in catalogue §12.3. Consumers declared and counted;
the image zero by width; paired MM replays recorded as asked.

## 18. Round 2c hand-off to Codex (Claude, 2026-10-06)

Start from the frozen round-2b candidate (`delivered-source/source.json`,
`d99b205a…`). §16 stands except the following.

1. **Negation never touches the form.** At interpretation the form face takes
   the activation's magnitude and the meaning face its signed value; the
   leaf's pole pair `(relu(a), relu(−a))` carries the sign for the
   pole-representation operators and the clause's polarity at the closing
   (accessible mind §2.6.2; catalogue §3.8; Philosophy, "the two symbols
   differ only in sign — one row holds the content"). Install the magnitude
   interpretation that the round-2b diagnostic verified; confirm on the
   frozen fixture that the forced negative-pole disjunction recovers 4/4 at
   the landing's reconstruction cost before the thirty trainings. In the two
   grammar gates the meaning face is empty, so a narrowing `not` then changes
   no root; its trials tie and teach nothing, which is correct until meanings
   exist.
2. **The reader: one step, compose departures only.** For a sentence with a
   supplied answer, when the departure lies in compose the reader takes one
   step on the mean of its losses over the greedy root and the explore root
   (both detached); when the departure lies in narrowing it trains on the
   kept root only, as in the landing. The comparison read for the advantage
   is unchanged for both kinds. Reader steps per sentence return to one;
   record them.
3. **Name MM_xor's path.** The paired replays differ at the first forward
   with identical parameters and RNG state and no declared consumer reached.
   Find the change on `MM_xor`'s path (candidates: the retired percept-stage
   selection — if the landing's explore walk ever won on MM — or the scope
   passback from the kept table) by bisecting the round-2 diff on the
   first-forward output at seed 0, and state it in the receipt. MM stays a
   live gate either way.
4. **Expectations.** Sum controls back at ¼ (the control must sit on the
   floor); disjunction roots recomposing at the landing's cost; narrowing
   departures tying in the gates; class at or above the landing with
   conjunction preferred by the answer term on compose departures;
   reconstruction and MM_xor at the landing's counts.

Measurement, receipt, tests and documents as §14.6–§14.7 and §16 (the
containment audit and consumer census retained), in
`doc/benchmarks/2026-10-06-operators-round2c/`; the 2 and 2b directories
preserved intact. Stop for Claude's review before any commit.

## 19. Review of round 2c (Claude, 2026-10-06): the mechanism is proven; the count is not yet back

Receipt `doc/benchmarks/2026-10-06-operators-round2c/` (manifest `f712768d…`,
702 files, matching the working tree): class **5/10**, reconstruction **10/10**,
sum **10/10 at ¼** (.24999998–.25, checkerboard contrast 6e-8 — cleaner than
the landing's .250–.260), MM_xor **9/10** (live), sweep green, zeros on
gradient, displacement and conflicts.

**What the round set out to do, it does.** Every one of the ten XOR runs ends
with conjunction chosen by cost — the landing's 7 and round 1's 8 were the
chooser's prior; here the policy flips from a disjunction start in six runs,
driven by the answer term on compose departures (3,374 rows favouring greedy
conjunction against 493 for disjunction departures). The three corrections
hold: narrowing departures tie exactly (10,650 of 10,650, including 1,801
`not`), so negation no longer reaches the form; the forced negative-pole
disjunction recovers 4/4 with zero pair residual on the frozen fixture and
every run reconstructs 4/4; the reader takes exactly one step per supplied
sentence (Adam counters at 400) and the control sits on its floor.

**Why the class count is 5.** Eight runs label all four rows correctly (the
landing: seven); only five also meet MSE < .05, and the table says why: runs
3, 5, 8, 9 started on conjunction (last greedy disjunction at epoch 0–20)
and pass; run 6 flipped at 121 and passes at .047; run 2 flipped at 150 and
misses at .0516; runs 1, 7, 10 flipped at 308, 290, 331, leaving the reader
69–110 epochs, and sit between; run 4 flipped at epoch 4 and sits at .071 —
the reader plateau the landing also showed and §16.3 forecast ("a flat
reader norm being the reader's convergence, output's"). The deficit is reader
time after late flips, not a mechanism. The flip is slow from a disjunction
start because the credit's sign is nearly balanced early (run 1: 218
conjunction departures read better, 202 worse) — the reader learns to read
conjunction roots only from the half-weighted departures, and compose
departures are only a third of all departures: the other two-thirds are
narrowing rounds whose alternatives cannot change the root in these gates
and teach nothing.

**MM_xor's path is named, and it is only the random stream.** The seed-zero
bisection: percept-stage costs tie on all four rows, no old explore walk ever
won, kept actions and scope match; retiring that exploration removes 33
global RNG draws, so `create_ir_mask`'s Bernoulli draw lands on 12 different
positions; restoring the old stage restores the landing's output, restoring
the RNG state after the greedy walk restores round 2's. Run 1's best is
.200930 against the strict .20 bar; the gate's bests sit at .17–.20 in every
campaign (the landing's .169–.198), so a draw at .2009 is the unchanged
mechanism's own tail. **Proposed amendment to §5:** a gate whose path differs
from the landing only in global-RNG consumption, shown by bisection, is not a
regression test of the round. For Alec: the MM gate's margin is structural
(bests within 10% of the bar); the bar is not changed here.

**Verdict.** Not accepted under §1 as written (class 5 against 7). The round's
purpose is achieved and verified; what remains is the count, and one
estimator change addresses its cause without tuning: the departure drawn
by walk first (§20). Recommend round 2d with that single change; if the
count is still short after it, the receipt's cost records will say whether
the remainder is the reader's plateau, and Alec decides.

## 20. Round 2d hand-off to Codex (Claude, 2026-10-06)

Start from the frozen round-2c candidate (`delivered-source/source.json`,
`f712768d…`). One change, and the §5 amendment.

1. **The departure is drawn by walk first.** Among the walks with an eligible
   round in this sentence (narrowing, compose — `W` of them), draw the walk
   uniformly, then the round uniformly over that walk's eligible rounds
   (`R_walk`), then the action uniformly over the round's eligible
   alternatives (`K`), as now. The surrogate's correction becomes
   `K · R_walk · W · p(a_dep) · (C_explore − C_greedy)`, which cancels the new
   proposal probability `1 / (W · R_walk · K)` exactly as `K·R` cancelled
   `1 / (R·K)`; the estimator stays unbiased with the greedy baseline. Nothing
   else changes: the keep, the reader rule, the advantage's components, the
   registration and reduction. Record, per sentence, the walk drawn, `W`,
   `R_walk` and `K`; the analytic-gradient and finite-difference checks
   cover a departure in each walk. (Why: in 2c two-thirds of departures fell
   on narrowing rounds whose alternatives cannot change the root in these
   gates, so the reader saw a conjunction root on only a third of sentences
   from a disjunction start and the flip took 150–330 epochs; drawing by walk
   first puts half the departures on compose without biasing the estimate.)
2. **The §5 amendment**, as §19 states it, applies to MM_xor with the
   bisection record carried forward (re-run the seed-zero first-forward
   bisection on the 2d source to confirm the path is unchanged).
3. **Expectations.** Flips from a disjunction start by about epoch 100;
   class at or above 7/10 with every final root conjunction; reconstruction
   10/10; sum at ¼; MM_xor at the landing's count, read under the amendment.

Measurement, receipt, tests and documents as §14.6–§14.7 and §16, in
`doc/benchmarks/2026-10-06-operators-round2d/`; the 2, 2b and 2c directories
preserved intact. Stop for Claude's review before any commit.

**6.1 maintenance, 2026-10-09.** The standalone
`test_xor_router_gradients_reach_all_three_ops` now prescribes the legal
NOT → AND → OR → AND path through `select_logits(replay_action=...)`, with
identity op matrices and a fixed positive input. It checks the resulting
operation journal before asserting gradients to all three real operations.
No seed selects the route. This repairs the assertion's former dependence
on whether unseeded routing happened to visit every op; the production
chooser is unchanged.

**6.1 landing record, 2026-10-10 (stream-state §7.15–§7.16).** Certificate
b, `test_ordinary_initial_binding_distribution_includes_every_retained_candidate`,
has a flaky unseeded .9–1.1 bound: the review records 3/3 at HEAD and 3/3
on the candidate in isolation, against the one sweep maximum of 1.129.
Tighten its coverage by construction (more menus), never by selecting a
seed or changing an observed failure into a pass. The landing receipt
retains that failure and reports the next source-matched outcome separately.

## 21. Review of round 2d (Claude, 2026-10-06): the flip is fixed; the reader is now the only miss, and I caused it

Receipt `doc/benchmarks/2026-10-06-operators-round2d/` (manifest `d7e2ead0…`):
class **2/10**, reconstruction 10/10, sum 10/10 at ¼, MM_xor 10/10 (path
RNG-only, re-bisected), sweep green, zeros held.

**The walk-first draw did what it was for.** Compose departures rose from a
third to half (7,978 of 16,000); every run is all-conjunction by epoch 106 at
the latest (2c: 331), four runs from the start, and all forty final roots are
conjunction with all forty labels right. The operator question is closed.

**The reader is the miss, and it is a dose–response on §16 item 2.** Eight runs
sit between .064 and .192 with correct labels — including runs 1, 2 and 7,
conjunction throughout, so late flips explain nothing here. The reader trains
on both trials' roots for every compose departure, and from a conjunction
policy every compose departure's explore root is a *disjunction* root: an
XOR-inseparable target under the same answer, which one affine map can only
fit at the floor. The share of sentences carrying a disjunction root into the reader's
step went 0 (landing, round 1: reader on the kept root) → one third (2c) →
one half (2d); since those sentences weight the two roots ½ each, the share
of the reader's *objective* on disjunction roots went 0 → 1/6 → 1/4 (Codex's
correction of my first statement), and the class count went 7–8 → 5 → 2. Under
the 2c/2d rule that share never decays, however committed the chooser is; the
chooser's `K·R_walk·W` is an importance weight for its own gradient and does
not touch the reader's distribution. The rule bought the flip (2c, 2d) at
the price of the fit; §22 sets the presented reader's share to exactly zero
and leaves the ¼ with the comparison reader, which only has to rank.

**The fix is to train the reader on what the policy will produce.** Weight
the two trials' reader losses by the policy's own probabilities — the greedy
root by `1 − p(a_dep)`, the explore root by `p(a_dep)`, the probability the
chooser recorded for the departure action. While the chooser is uncommitted
(`p ≈ ½`, the start) both roots train equally and the flip proceeds as in 2d;
as it commits to conjunction, `p(disjunction) → 0` and the reader trains on
the committed root alone, recovering the landing's fit. This is the mixture
Alec asked about in §9.3, applied where it belongs: not to the chooser's
gradient but to the reader's training distribution. Fallback if the audit
shows the chooser sharpening on disjunction before the reader has learned
conjunction roots (it did not in 2c or 2d): a separate answer-owned
comparison reader trained on both roots at full weight, used only for the
advantage's read, with the presented reader on the kept root.

Everything else carried: narrowing departures tie (8,022 of 8,022), one
reader step per sentence, consumers counted, containment vacuous on the
banks and nontrivial on the fixture, image zero by width. Not accepted
(class 2 against 7); round 2e is one change.

## 22. Round 2e hand-off to Codex (Claude, 2026-10-06; rewritten after the toy)

The first draft of this section weighted the reader's two losses by the
chooser's probabilities. Simulated before sending
(`doc/benchmarks/2026-10-06-credit-loop-toy/`), that rule falls into the
trap §21 named: with disjunction roots correlated with conjunction's, in 10 of
40 seeds the chooser commits to disjunction before the reader has learned
conjunction roots, the explore weight vanishes, and the policy never flips.
The two-reader rule flips 40/40 and reaches zero in 38/40 (the two misses
flipped after epoch 340). It is also the combination of two behaviours
already measured in the real model — 2d's reader, which flipped 10/10 by
epoch 106, kept as the judge; the landing's reader, which fit conjunction
roots to ~1e-3, restored as the one presented — with no new dynamics.

Start from the frozen round-2d candidate (`delivered-source/source.json`,
`d7e2ead0…`). One change.

1. **Two readers, one presented.** The reader that is presented, written
   from, and read by the class bar trains on the kept trial's root only — the
   landing's rule, one step per sentence. A second, answer-owned *comparison
   reader* of the same form trains on both trials' roots exactly as 2d's
   reader did (compose departure: the mean over the greedy and explore roots;
   narrowing departure: the kept root), one step per sentence, detached
   roots; it is used only for the advantage's answer read, never presented,
   never written from, never read by a gate. Both readers are in the answer's
   owner list; nothing else changes: the keep, the advantage's components, the
   walk-first draw, the registration and reduction. Record per run the
   comparison reader's and the presented reader's MSE on the committed roots
   over training; the audit reports the flip epoch as in 2d.
2. **Expectations.** Flips as in 2d (all-conjunction by about epoch 100);
   the presented reader at the landing's fit on conjunction roots — class at
   or above 7/10 at zero, the landing's reader plateau being the one known
   miss; reconstruction 10/10; sum at ¼; MM_xor 10/10 under the §5 amendment,
   bisection repeated.
3. **If the count is still short**, the receipt separates the misses into
   late flip (presented-reader time after the flip), reader plateau (correct
   labels, flat reader norm) and anything else, with the late cost records
   as in 2c/2d.

Measurement, receipt, tests and documents as before, in
`doc/benchmarks/2026-10-06-operators-round2e/`; the earlier directories
preserved intact. Stop for Claude's review before any commit.

## 23. Review of round 2e (Claude, 2026-10-06): accepted as the round-2 landing, pending Alec

Receipt `doc/benchmarks/2026-10-06-operators-round2e/` (manifest `3fd07f0c…`,
705 files, matching the working tree): class **9/10**, reconstruction
**10/10**, sum **10/10 at ¼**, MM_xor **10/10** (path RNG-only, bisection
repeated), sweep green, zeros held; 40/40 conjunction roots chosen by cost,
40/40 labels; the one miss is a late flip (run 6, stable conjunction from
epoch 140, 4/4 labels, MSE .080 and falling). Against the landing's 7 / 9 /
10 / 10 the standing gate is met with margin.

**Code, verified against §14–§22** (cumulative diff from `73cd7b71b`, 471+/163−
in `bin/`):
- `SentenceCredit.departure`: walk uniform among walks with eligible rounds,
  round uniform within the walk, action by the chooser; `comparison`: keep
  by `R` alone, strictly lower, advantage on `R + E + A`; `score_function`
  unchanged with `scale = W·R_walk·K` (verified on all 32,000 records);
  `reader_weights`: presented reader on the kept root, comparison reader
  `[½,½]` on compose departures and the kept root otherwise.
- `AnswerComparison`: a `ParameterDict` twin of every answer-owned parameter,
  cloned at first use without RNG, run through `torch.func.functional_call`
  on the model's own reader code with `tie_weights=False` and explicit
  aliases; in the answer owner list and the optimizer; checkpoint
  round-trip without an initialization draw. `_sentence_answer_error`
  returns the twin's cost as `A` and merges the presented reader's into the
  trial registry; both take their one step in the explore trial's backward.
- `Interpret.activate_code`: form × |a|, complement × a (§2.6.2);
  `pole_activation`: sign only where a field operation reached the word,
  ties keep the prior sign, magnitude preserved; `_attention_sentence_payload`
  no longer scales the event; the four consumers declared and counted.
- `Meaning.negative_image` on the concept face only, `ClosingImage.restore`
  exact; `MereologicalCodes.containment_audit` by postings intersection.
- Tests: 54 focused cases across rounds 1–2e pass on the working tree; 14
  assertions removed, 25 added, all ports with saved old/new texts.

**Observations, not blockers.**
1. The comparison reader twins the whole answer path (output space,
   answer attention, record reader, synthesis parameters), not only the
   affine head: the state dict grows by one answer path and the
   `functional_call` substitution is intricate (aliases across modules).
   Correct as delivered; a maintenance note for the day the answer path is
   large.
2. `test_runtime_split_ingestion`: a user-stored truth's row now carries
   `c_plus = 1` where it carried 0. This is the store's documented contract
   (the clause's poles are "input identification poles, never the source's
   authority"; trust is separate) restored by 2c's "a valid grammar word
   cannot lose its evidence to an absent serial row" — the 0 was an
   artifact — but it is visible: `support_true` of an ingested assertion is
   now 1. **Confirmed (Alec, 2026-10-06):** the pair is the representation
   and the single number for existence or truth is its signed sum
   `c⁺ − c⁻` (the unsigned sum is how much is known; `min` the dissonance),
   so an ingested assertion reading `c⁺ = 1` is right. **Decided for round 3 (Alec,
   2026-10-06):** the user's signed trust `t` becomes the row's pair at
   ingestion — `(|t|, 0)` for `t > 0` (trust), `(0, |t|)` for `t < 0`
   (distrust) — so the net `c⁺ − c⁻` recovers `t`, a negated sentence still
   exchanges the poles, and the separate provenance column is retired.
   Also from this exchange: expectation needs no twin (its comparison is a
   distance in row coordinates, not a learned read); the predictor's target
   should become the committed row only when meanings exist (round 4).
3. GradientFlow now carries five round sections (2, 2b, 2c, 2d, 2e). At the
   landing, Codex may collapse them into one "round 2: credit" section with
   one paragraph per rejected candidate's lesson; the receipts are the record.

**Landing.** Commit, push and bump as for round 1, with one difference: this
commit includes everything in the working tree that the round-1 landing left
uncommitted — this plan (untracked since 2026-10-05), the credit-loop toy,
and Claude's 2026-10-06 edits to Architecture, GradientFlow, Philosophy,
FutureWork and todo — alongside Codex's source, tests, documents and the
five receipts. Acceptance record on the 2e receipt as before. The sequence
of §9.5 then stands at: round 2 landed; round 3 (identity) next, its text to
be preceded by working its own loop to a fixed point (collision-minting
and the containment projection) before anything goes to Codex.

## 24. Round 3's fixed point (Claude, 2026-10-06): identity by construction

Worked before the text, per the practice adopted after 2d
(`doc/benchmarks/2026-10-06-identity-toy/`). Three results shape round 3a.

1. **Pairs plus length identify.** Over 20,000 dictionary words,
   boundary-marked adjacent pairs with a fixed sparse code per pair (3 ones
   of 64) and the length separate every anagram group but one and collide
   once — `calaba`/`cabala`, the same pair set and length — which is the
   genuine case collision-minting exists for. Width 64 suffices for that
   vocabulary; the gates need far less.
2. **The kernels must bind on a dense projection.** The product of two
   sparse codes that share no coordinate is zero, and two random short
   words share none about half the time. A fixed random projection of the
   sparse form (JL, seeded, not learned) gives every root a nonzero
   direction; on the gate words the projected products are affinely
   XOR-separable and unbind exactly by bank search. The sparse code remains
   the identity key; the projection is the kernels' input. Permutation
   bindings also work but are non-commutative, which conjunction is not.
3. **Length is a part.** If length were a separate coordinate, equal pair
   sets with different lengths (`bana`, `banana`) would violate the
   containment order of §16 item 7 in one direction. As cumulative atoms —
   `len≥1 … len≥n`, each a part with its own code — a word's length block is
   a thermometer by construction, `parts(bana) ⊂ parts(banana)` holds as a
   set inclusion, and the order is preserved without exception.

Round 3 is therefore two rounds: **3a, identity by construction** (the parts,
their codes, the projection, collision-minting, exact reconstruction at
rung 0, the trust-to-pair port) and **3b, the ceiling** (has-a wholes, the
narrowing rule, the centroid, the containment projection, the room pass
retired), each one gate-moving change.

## 25. Round 3a hand-off to Codex (Claude, 2026-10-06)

Start from the round-2 landing commit. One change: identity by construction
at rung 0. Specification: §7.3, §8.2, §10.1, §12.1 (prerequisite), §13,
§16 item 7, §24; catalogue §4 (idempotent intersection unchanged).

1. **The parts of a word.** At rung 0 the parts are the boundary-marked
   adjacent letter pairs (`#c ci ir rc cu us s#` for `circus`) and the
   cumulative length atoms `len≥1 … len≥n`. Each part is a bank row minted
   on first sight, content-addressed by its bytes (`RadixLayer.insert` as
   now), with a **fixed** sparse binary code: `s` ones of `D`, drawn by a
   generator seeded from the atom's bytes, so the code is a function of the
   atom alone, consumes no global RNG, and is identical across runs and
   configurations. Byte rows remain for the character-class wholes and for
   descent; they no longer form the word. `D` and `s` are configuration
   values: 64 and 3 for the gates and BasicModel's small vocabularies; the
   receipt reports collisions over each configuration's vocabulary.
2. **The form.** A word's form is the join (max) of its parts' codes, as
   now (`synthesize_word_parts` over pair rows instead of byte rows); the
   length block is the thermometer the cumulative atoms give. The form is
   the identity key and the content-addressed row's key; anagrams separate;
   `parts(X) ⊆ parts(Y) ⇒ form(X) ≤ form(Y)` holds by construction and the
   §16 item 7 audit now has nontrivial pairs on every vocabulary (`an`,
   `and`, `ant`; `bana`, `banana`).
3. **Collision-minting.** When a new word's form equals an existing row's
   with different bytes, mint the cheapest distinguishing atom — the first
   adjacent letter triple on which they differ — as a part of both words,
   and otherwise nothing. The receipt lists every mint.
4. **The kernels' input.** The binding kernels and the pair search take
   their code directions from a fixed dense projection of the sparse form
   — a seeded random matrix `D × d`, not learned, `d = 64` in the gates —
   so no root is empty; conjunction and disjunction are otherwise
   unchanged (product and probabilistic sum of directions, magnitude from
   the activation). The sparse form, not the projection, is what the index,
   the containment audit and the reconstruction audit read.
5. **Reconstruction at rung 0 is exact.** An identified word's bytes are read
   from its row through the index; its byte cost is zero and the
   reconstruction registry's rung-0 term becomes an audit that must read
   zero for every identified word. An unidentified word is assembled from
   its pairs and length (the sequence whose boundary-marked pairs are those
   pairs; ties by the index after minting). The trial cost `R` keeps its
   other terms; where `R` ties on every row the keep is greedy, as the
   scheme intends.
6. **Trust to the pair (Alec, 2026-10-06).** At ingestion a user's signed
   trust `t` becomes the row's pair, `(|t|, 0)` for `t > 0`, `(0, |t|)` for
   `t < 0`; a negated sentence exchanges the poles; the separate provenance
   column is retired and the ingestion tests ported to the pair.
7. **Not this round.** Has-a wholes, the narrowing rule, the centroid and
   the containment projection (3b); meanings; the attention filter.
8. **Expectations.** Forms: zero collisions on the gate vocabularies, every
   anagram and length pair separated, the containment audit nontrivial and
   at zero. Reconstruction 10/10 with the rung-0 audit at zero. Class at or
   above the landing (the projected roots are affinely independent). Sum at
   ¼. MM_xor (numeric input, no letters) unchanged but for RNG, bisection
   repeated. If class moves, the receipt states whether the roots' geometry
   or the reader's convergence moved it.

Tests: the form audit (collisions, anagram separation, length separation,
containment pairs) on each configuration's vocabulary; `circus`/`cursic`/
`cirrus` and `bana`/`banana` distinct; `calaba`/`cabala` minting a triple;
the projection fixed across runs; every root nonzero on the gate words; the
rung-0 reconstruction audit; the trust-to-pair port. Documents: catalogue
§12 (3a), GradientFlow "round 3a", Architecture's identity paragraph,
FutureWork's located-fold item retired in favour of pairs. Receipt in
`doc/benchmarks/2026-10-07-operators-round3a/`; measurement protocol as
before; stop for Claude's review before any commit.

## 26. Codex's construction check of round 3a (2026-10-07), and §25 amended

Codex probed §25 before touching the runtime
(`doc/benchmarks/2026-10-07-operators-round3a/construction-review.md`) and
found three defects, all real, two of them mine in construction:
(1) cumulative length atoms coded as three random bits are not an exact
thermometer — a join can fail to add a bit (`aaaaaa`/`aaaaaaa` collide), and a
binary join can grow strictly at most D times; (2) the "first differing
triple" can be masked by the existing join (`cal`/`cab` leave
`calaba`/`cabala` equal), so the mint must check the resulting forms and
continue; (3) `an`/`and`/`ant` are not part-set inclusions under boundary
pairs (`n#` is a part of `an` and of neither). Verified repairs
(`identity-toy/sim2_repairs.py`): a reserved thermometer block; a mint that
draws bits from each word's own positional atom until separation; the
witnesses `bana`/`banana`, `cat`/`concat`, `aba`/`ababa`.

**§25 is amended as follows** (everything else stands):

- *Item 1, length.* The length atoms `len≥k` are parts for the containment
  order (postings), but their codes are not random: the form block reserves
  a thermometer of `L` coordinates after the `D` pair coordinates (`L = 32`
  in every configuration), and a word of length `n` sets the first
  `min(n, L)` of them. Exact up to `L`; longer words saturate and rely on
  minting. The three-random-bits rule applies to pair and minted atoms only.
- *Item 3, the mint.* When two words' forms coincide, each receives its
  **own** atom at the first position where their boundary-marked letter
  triples differ (`cal@1` for `calaba`, `cab@1` for `cabala`); each atom's
  code is drawn from that atom's seeded bit stream, three bits first, then
  one more bit at a time, and the mint is complete when the two forms
  differ; if the position is exhausted without separation, the next
  differing position is used. A mint is recorded with its atoms and bit
  counts. Adding the same atom to both words is never a mint.
- *Item 2, witnesses.* The containment audit reports two things separately:
  the native vocabulary's census (vacuous where no part-set inclusion
  exists, as in the four gate words) and the witness fixtures run under
  each configuration — `bana`/`banana`, `cat`/`concat`, `aba`/`ababa` ordered;
  `an`/`and` and `an`/`ant` reported as not comparable.

## 27. Two findings during round 3a (Codex, 2026-10-07) and what the gates now mean

1. **Both projected conjunction and disjunction roots are affinely separable
   on the gate sentences.** A consequence of identity by construction: under
   a random dense projection the four pair-roots are generic for either
   kernel, so any four-row truth table is affinely readable. "Disjunction at
   the floor ¼" (6.8 §22, round 1) was an artifact of the dense max-joined
   codes, where `u + v` dominated and `u∘v` was nearly collinear. From 3a the
   XOR class gate certifies identity (anagram and length separation, exact
   reconstruction) and the reader's convergence on whatever roots the
   chooser produces — not the operator. The §25 expectation "conjunction
   preferred by the answer term" is withdrawn; correct behaviour is
   indifference (compose-departure credit near zero once both roots read).
   The sum control remains the certificate of the affine floor (the mean of
   two projections keeps the pairwise dependency) and must stay at ¼. The
   operator distinction is a semantic one and moves to round 4's meanings
   gate: a union answers membership of either operand, an intersection of
   both.
2. **BasicModel's corpus has 67,391 distinct words against a 32,768-row
   bank.** Forms collide zero times after two recorded mints at that scale —
   the construction's purpose. The bank holds admitted rows, not distinct
   words: the recurrence threshold stays, hapax words are assembled from
   pairs and length (§25 item 5), and the admitted vocabulary is reported
   against the capacity; if the recurring vocabulary exceeds the capacity,
   BasicModel's configured capacity is raised (a value, not a mechanism), and
   forgetting (sequence item 5) remains the standing answer. The storage
   report is kept separate from the gate results.

## 28. Round 3b's fixed point (Claude, 2026-10-07): the ceiling is empty at order 0

Computed on the 3a forms over the dictionary sample
(`identity-toy/sim3b_ceiling.py`). The has-a ceiling of §12.1, weighted
globally, lifts the mean pairwise cosine of the symbols from .52 to .90 — the
§14 collapse — and the narrowing rule does not prevent it (.908 → .902),
because wherever no whole narrows a coordinate `U` is 1 there and any
positive global weight on a ceiling of ones is a common offset. The
§14.3 clause "a concept with no wholes yet is the join of its parts" must
hold per coordinate: a whole weighs on a coordinate only by the narrowing
it does there. With that rule the collapse is gone (cosine .512 → .512) and
the centroid moves nothing (mean |c − L| = .001). Both are the same fact,
the one §8.3 stated in words: the extent of a word's own parts is a
function of those parts, so a letter-defined has-a whole can bound a word
with everything or with the word itself, never with new information.

Consequences: (1) the per-coordinate rule is the rule — it reduces exactly
to `L` where nothing narrows, so a ceiling can never collapse the symbols
when informative wholes arrive; (2) the containment projection works as
specified (cap the contained by its containers in decreasing part count:
violations 566 → 0, never below `L`); (3) the room pass has nothing to
enforce and is retired; (4) **round 3b has nothing to measure at order 0**
and folds into round 4, where the wholes that are not functions of a word's
parts live — membership one order up (the sentences, situations and
documents containing it), the character classes, counts — and the centroid
gets its first real test with Alec's condition met by construction.

## 29. Review of round 3a (Claude, 2026-10-07): accepted, pending Alec; the receipt to be finalized

Receipt `doc/benchmarks/2026-10-07-operators-round3a/` (frozen manifest
`84edc57e…`, 707 files; the README still marked "in progress" at review —
Codex finalizes it before landing). Sweep green (5,289/5,289). Thirty
trainings: class **10/10, all at zero** (MSE 0 to .021), reconstruction
**10/10**, sum **10/10 at ¼** (.25 to six digits, contrast ≤ 1e-6), MM_xor
**10/10** (bests .177–.198, RNG-only path re-bisected). Against the landing's
7 / 9 / 10 / 10 and 2e's 9 / 10 / 10 / 10, the standing gate is exceeded.

**Source, against §25 as amended by §26.** `bin/WordIdentity.py`: pair and
cumulative-length atoms minted on sight, content-addressed through the
RadixLayer with fixed codes — pair and mint atoms from each atom's
SHA-seeded bit stream (3 of 64), length atoms as the reserved 32-bit
thermometer; no global RNG; the form is the join over the word's atoms;
words keyed by their form; mints positional with recorded atoms and bit
counts, exhausting triples then quadruples; reconstruction by the index or
by assembly from pairs and length; the binding through a fixed projection
(`D+L → 64`, seeded once) with the event layout preserved; fixed rows masked
from learning. Configuration layout resolved before construction
(`resolve_identity_layout`): the gates' form block is 96 + the address
band. Form audit on every configuration: zero collisions, zero containment
violations, zero byte errors; the witnesses ordered and the invalid ones
reported not comparable; `calaba`/`cabala` minted at `cal@1`/`cab@1` with 3
bits; the dictionary sample zero collisions after one mint; BasicModel's
native census now has genuine containment pairs. Trust-to-pair ported.

**In training.** `R` is identically zero on both trials of every sentence
— reconstruction is an audit now, as §13 intended — and narrowing
departures tie (0 nonzero of ~800 per run). Compose departures all carry
nonzero advantage, mean |ΔC| ≈ .41, entirely from the answer: the
comparison reader reads conjunction roots at ~.003 and disjunction roots at
.26–.58 late in every run, so the policy settles on conjunction. This
differs from §27's static finding (both kernels' four roots affinely
separable, fit MSE ~1e-30). The likely reason: under a zero-mean projection
the disjunction kernel's product term is small against `Wu + Wv`, so the
four disjunction roots are separable only through a tiny, ill-conditioned
component that a reader trained by finite steps cannot exploit; in practice
they sit near the floor, and the gate still discriminates the operators.
Not verified: **Codex reports the singular values of the centered
disjunction roots** in the landing notes, and §27's reading of the class
gate is softened accordingly — it certifies identity, the reader's
convergence, and the operator in practice though not in principle. The
semantic operator gate on meanings (round 4) remains the proper certificate.

**Verdict.** Accept as the round-3a landing. One housekeeping item: the
receipt README is finalized with the tables, the storage report (67,391
distinct words, 69,566 atom/word rows in the audit bank against the
32,768-row production bank, kept separate from the gates), the two
findings of §27 and the singular values. Landing as for round 2 — the
commit includes the plan, the toys (`credit-loop`, `identity`) and all
uncommitted documents — then push and bump. Round 3b is folded into round
4 (§28); the round-4 text follows its own fixed-point work.

## 30. Round 4's fixed point (Claude, 2026-10-07): meanings by membership

Worked before the text (`doc/benchmarks/2026-10-07-meaning-toy/`). The
bootstrap problem (§3, §6: "a zero complement cannot bootstrap itself from
zero occurrence roots") dissolves if the rows a word occurs in contribute
their *identity* as wholes rather than their composed meaning: each stored
sentence row gets a fixed sparse code in the meaning complement, seeded from
its content key (no RNG, no parameter — 3a's construction one order up), and
a word's order-0 meaning is the existing recency-weighted context mean over
those codes. This is random indexing, and it is Alec's "co-activation flows
from the shared-wholes representation" literally: words share meaning to the
degree they share sentences. **It needs no owner** (§3 question 1): like the
identity codes it is an index, not a trained parameter — "codes by
distribution, maps by the objectives" (2026-09-21).

Composition on the meaning block is extensional: conjunction = min (zero is
false — a membership field, the catalogue's monotonic variant, not the
silence-preserving meet of the form face), disjunction = max, sum = mean.
On the XOR corpus the membership certificate is exact (the extent of `a∧b`
is the sentences containing both, of `a∨b` those containing either); both
operators' meaning roots are equally well conditioned, so the class gate
still cannot choose the operator — the certificate is the operator's test,
static, as the form audit is identity's; the sum control's mean stays at the
floor. At corpus scale the code is a Bloom-like sketch (false memberships
fall from ~10% at K = 64 to ~0.04% at K = 1024 for 400 sentences), so
membership is answered through the index and the code carries similarity
and composition.

Round 4 splits: **4a, meanings exist** (this construction, the per-block
composition, the certificate, expectation's target); **4b, the centroid**
with membership wholes in the form space — which needs Alec's ruling first
(§31 end); **4c** connectives' negation over meanings and the co-activation
priming, after 4b.

## 31. Round 4a hand-off to Codex (Claude, 2026-10-07)

Start from the round-3a landing (`cce3a4f7b`). One change: order-0 meanings
exist and compose extensionally. Specification: §6, §9.4, §13, §23 (the
expectation note), §27–§28, §30; GradientFlow "The training step" and the
context-mean paragraph; catalogue §4 (the monotonic membership variant).

1. **Sentence identity codes.** Every stored sentence row (REL_NONE,
   content-addressed) has a fixed sparse code in the meaning complement:
   `s` ones of `K`, from a generator seeded by the row's content key (the
   same key that content-addresses it; never the per-store occurrence
   namespace), no global RNG, recomputable, not a parameter. `K` is the
   configured complement width and `s` a configuration value: the grammar
   gates get `K = 64`, `s = 3` (their complement is zero today); BasicModel
   keeps its 896-wide complement with `s = 6`.
2. **Order-0 meaning.** `occurrence_terms` keeps its membership (leaf
   postings and references, DEF rows excluded), its recency weight and its
   detachment, but each containing row contributes its identity code, not
   its stored composed meaning. A word's meaning block is that mean; it is
   snapshotted before the forward as now. Composed sentence meanings are
   still stored as `o` and are what thought and expectation read.
3. **Per-block composition.** The form block composes as in 3a (the fixed
   projection, product / probabilistic sum of directions, normalized). The
   meaning block composes by its own rule, never normalized with the form:
   conjunction and intersection `min` (zero is false), disjunction `max`,
   `sum` the mean; operators that do not declare a meaning write (round 1's
   footprints) pass the meaning block through. Interpretation's
   `[form × |a| | meaning × a]` is unchanged.
4. **The semantic certificate (static, like the form audit).** For every
   pair of words in each configuration's vocabulary and every stored
   sentence row: code membership of the `min` root equals the index's
   "both", of the `max` root the index's "either". Exact on the grammar
   gates; on BasicModel's local corpus reported as a false-membership rate
   at its width (diagnostic, not a gate), membership itself being answered
   through the postings.
5. **Expectation's target.** The predictor trains on the committed (kept)
   trial's row only, not on both trials (§23). `E` remains zero in these
   gates (no prior context); the focused test covers it.
6. **Not this round.** The centroid and any form-space use of membership
   wholes (4b); negation over meanings and co-activation priming (4c).
7. **Expectations.** Meanings distinct for every gate word; the certificate
   exact; reconstruction 10/10 (the `R ≡ 0` audit unchanged); class 10/10 at
   zero; sum 10/10 at ¼ (the mean keeps the meaning block additive); MM_xor
   unchanged but for RNG. Both operators now read equally well, so final
   operators may be mixed and the compose credit near zero — correct, and
   reported; the operator's semantics is the certificate's to show.

Tests: identity codes fixed across runs and independent of the global RNG;
the context mean nonzero after one presentation and equal to the mean of
the containing rows' codes; per-block composition (min/max/mean on the
meaning block, the form block unchanged, no cross-block normalization); the
certificate on the gate vocabularies and on a synthetic fixture with a
known false membership at small `K`; expectation trained on the kept row
only. Documents: GradientFlow (the bootstrap resolved; no owner, an index),
Architecture's two-spaces section, catalogue §12 (4a), FutureWork (the
complement's bootstrap item closed). Receipt
`doc/benchmarks/2026-10-07-operators-round4a/`; measurement protocol as
before; stop for Claude's review before any commit.

**For Alec before 4b** (not blocking 4a): (i) in form space, the sentences
containing a word are its wholes only if a sentence's form is the join of its
words' forms; then a word's ceiling is "what always accompanies it", and the
centroid mixes co-occurrence into the form. Should the centroid's wholes be
membership wholes in the form, or should co-occurrence stay in the meaning
complement only (the form's centroid then remains `L`, §28)? (ii) Over
meanings, is a zero "false" (closed world: the extent's complement, `1 − x`)
or "unknown" (Kleene: `not = −x`)? It decides what `not` does to a meaning
in 4c.

## 32. Alec's rulings on §31's two questions (2026-10-07)

1. **The wholes of a word contain its adjacent words; its parts its
   constituents; both determine its context in mereological space.** So
   co-occurrence enters the form through the ceiling, by design. Technical
   consequence: a whole's form is the join of its constituents' forms (the
   whole is greater than its parts by construction, no room rule); the
   narrowest wholes are the adjacent-word pairs, so
   `U(w) = L(w) ∨ ⋀ L(neighbour)`, and the centroid is `c = L + α(U − L)`.
   Verified on the repository's documents (`meaning-toy/sim4b_adjacent_ceiling.py`):
   37% of words move, no collapse (.468 → .475), identity recoverable as
   `[c = 1]`, collocates drawn together, the containment projection
   repairing every violation the centroid causes. This is round 4b; it
   supersedes §28's conclusion that the centroid waits for membership
   meanings — §28 was right about letter-defined wholes only.
2. **Conceptual space is bipolar evidence.** `(0,0)` is complete unknown,
   `(0,1)` false, `(1,0)` true, `(1,1)` both; perceptual zero is nothing
   perceived. So the meaning block is a pole pair per coordinate (Belnap's
   bilattice), not a signed value: a word's meaning has its context mean on
   the *for* pole and nothing on the *against* pole (absence of evidence is
   unknown); conjunction is `(min⁺, max⁻)`, disjunction `(max⁺, min⁻)` —
   round 1's explicit-pole rule — and `not` exchanges the poles. 2c's
   interpretation rule `meaning × a` becomes: a negative activation exchanges
   the meaning's poles, scaled by `|a|`. This amends §31 items 1–4 (below)
   and absorbs 4c's negation item.

**§31 amended.** (1) `K` counts pole pairs: the gates' complement is
`2 × 64`; BasicModel's 896 holds 448 pairs (`s = 6`). (2) The context mean
fills the *for* pole; the *against* pole starts at zero. (3) The meaning
block composes by the bilattice: conjunction/intersection `(min⁺, max⁻)`,
disjunction `(max⁺, min⁻)`, `sum` the mean of each pole, `not` the pole
exchange; interpretation exchanges the meaning's poles for a negative
activation (form × `|a|` unchanged). (4) The certificate adds negation:
`not a` has `a`'s extent on the *against* pole and nothing *for*; `a ∧ not b`
has nothing *for* (open world) and `b`'s extent *against*. (7) A narrowing
`not` now changes the root's meaning, so narrowing departures carry credit;
with the 2e reader rule (kept root only on narrowing departures) the answer
is expected to teach the policy not to negate asserted sentences — reported.

## 33. Occurrence ids, content keys, and the store's growth (Codex's 4a question, 2026-10-07)

**What occurrence ids are.** A stored row is an *occurrence*: one sentence
read at one place in one document (Alec, 2026-09-30: an identity is nothing
but its occurrences in LTM tied by references; 2026-10-03: a row's address
is its own `.where`/`.when` bands, queried by content). The occurrence id
(`occurrence_id`, a per-store counter under a random per-store namespace) is
that episode's durable handle: references in slots, object permanence,
estimate/observation pairs and thought history point at occurrences, and
row indices move under compaction, so references cannot use them; the
namespace keeps one store's references from aliasing another's. A content
key is a different thing: what all occurrences of the same sentence share.
Nothing before 4a needed it, which is why there was no API; Codex is adding
it (`content_key`, `bind_sentence_content`, `rows_for_content`).

**The growth.** Every presentation of a sentence appends a new occurrence.
In the gates the same four sentences are re-stored every epoch until the
store is full — every gate word shows 510 occurrences at the end of every
campaign since 6.8, ~1,020 rows at the default capacity of 1,024 — after
which `append` returns −1 and every later sentence is silently not stored.
At that default, BasicModel's LTM stops recording after about a thousand
sentences. The cause is that a re-reading of the same sentence at the same
address (same document, same sentence index) is treated as a new episode,
though by the 2026-10-03 rule it has the same address.

**Proposed (Alec to confirm):** write by address — the store's write is an
upsert keyed by (content key, document, `.when`): a re-reading re-witnesses
the existing occurrence (recency and evidence refreshed), only a genuinely
new occurrence appends; a full store never drops silently — it raises, and
forgetting (sequence item 5) is the standing answer to a full store. For
4a itself the meaning code is seeded by the content key, so duplicate
occurrences do not fragment a meaning; the upsert is a correctness fix for
the store, and goes in as its own step before 4a's measurement (it changes
what the gates store, so it is measured on its own: four rows in the XOR
gate instead of ~1,020, gates otherwise unchanged).

## 34. Round 4a-0: the occurrence's address (Alec, 2026-10-07: "Yes, pull it forward")

The relative `.when` of the 2026-10-03 ruling (scheduled with 5.5) is pulled
forward as a prerequisite step before 4a. A stored sentence's identity is its
address — its document, its index in that document, its content — so a
re-reading is recognized and re-witnessed instead of appended; the counter
occurrence id and the per-store namespace go away, replaced by an integer
hash of the address that references use as its short name; a full store
raises instead of dropping writes. Measured on its own (it changes what the
gates store: four rows in the XOR gate instead of ~1,020), gates expected
unchanged; then 4a resumes on top, its identity codes seeded by the
sentence content key and its context means over one row per occurrence.
The text is the hand-off below (also given in chat).

## 35. One combined step to the end of the update (Alec, 2026-10-07)

Alec: combine the remaining rounds into a single step — "more work, [or]
testing overwhelms the implementation rate" — finish to round 6, then 6.5.
So 4a-0, 4a, 4b, the priming, round 5 and round 6 are one hand-off with one
measurement. Attribution is kept by three means instead of separate
campaigns: every part that can move a gate is a model.xml parameter (on by
default) so a miss can be bisected by switching parts off; every part has a
static certificate or focused test that holds or fails on its own; and the
cost records of 2c–2e stay in the receipt.

Round 5's fixed point (`doc/benchmarks/2026-10-07-filter-toy/`): the soft
filter has a degenerate fixed point — started undecided with the budget on,
it switches every word off before the reader can read and nothing pushes
back. Started from the hard mask's present behaviour (attend everything)
with a small budget (λ = .001), it keeps the content pair, removes the
fillers, and its hard choice at test agrees with the soft one (top-2 = the
content pair on every held-out sentence; accuracy 1.0 against .86 with
every word attended). With identity by construction an unattended word
costs perception nothing, so §8.1's factored cost reduces to the answer
and expectation through the root, plus the budget. The round-5 gate is that
toy's task in the model: `XOR_filler`, the XOR content pairs among filler
words.

## 36. The combined hand-off to Codex: the rest of the operators update (Claude, 2026-10-07)

One step, one measurement (§35). It supersedes the separate measurements of
4a-0 (§34) and 4a (§31–§32); their specifications stand and are parts A and
B. Start from the round-3a landing plus the uncommitted 4a/4a-0 work.

**A. The occurrence's address** — §34's text as given: relative `.when`
(the sentence's index in its document, absolute time in the timestamp
column only), document keys, the address key replacing the occurrence
counter and namespace, the write as an upsert by address (re-reading
re-witnesses), a full store raises. Not optional.

**B. Meanings** — §31 as amended by §32: sentence identity codes seeded by
the content key; order-0 meaning = the context mean of the containing rows'
codes on the *for* pole; the meaning block bipolar, composing by the
bilattice (`∧ = (min⁺, max⁻)`, `∨ = (max⁺, min⁻)`, `sum` = per-pole mean,
`not` = pole exchange); interpretation exchanges the meaning's poles for a
negative activation; the semantic certificate with its negation cases;
expectation trained on the kept row. Parameter: the meaning width
(`0` switches B off).

**C. The centroid** — §32 item 1, verified on text
(`meaning-toy/sim4b_adjacent_ceiling.py`): a whole's form is the join of
its constituents' forms; a word's wholes are the adjacent-word pairs it
occurs in (distinct occurrences, from the address-keyed rows); the ceiling
`U(w) = L(w) ∨ ⋀ L(neighbour)`; the symbol `c = L + α(U − L)` with
`α = W_U/(W_P + W_U)`, `W_P` the word's part atoms and `W_U` its distinct
adjacent occurrences; after placement the containment projection (cap each
contained word by its containers in decreasing part count; never below `L`,
§16 item 7); the binding kernels take the projection of `c`; the index keys
`L = [c = 1]`; the room pass is retired. Certificate (static): the share of
words moved, mean pairwise cosine before and after (no collapse), identity
recovered from `c`, containment violations before and after the projection
(zero after). Parameter: `symbolCentroid`.

**D. Priming through shared wholes.** The existing priming diffusion
(concept-store edges, `primingSpread`) gains the membership edges — word ↔
the address-keyed sentence rows containing it — with each node's outflow
normalized by its degree so frequent words do not dominate. Forward-only,
detached, as priming is now; it reaches the attention walk through the
existing codebook-retrieval prior. Report the priming mass on content and
filler words in the filler gate. Parameter: `membershipPriming`.

**E. The attention filter** (§8.1; fixed point §35,
`doc/benchmarks/2026-10-07-filter-toy/`). A weight per word,
`m = σ(head(detached word code) + prior)`, **initialized to attend
everything** (bias such that `m ≈ .95`; the hard mask's present behaviour —
started undecided, the filter collapses to attending nothing). The walk's
structural and field actions are unchanged; the filter replaces the hard
`accepted` decision in training. An unattended word drops out of
composition by interpolating its leaf toward the operation's identity:
the form block's product toward the all-ones direction, the meaning
block's `∧` toward `(1, 0)` and `∨` toward `(0, 1)`, `sum` by weight. The
filter trains pathwise on the owner-step cost through the root (answer and
expectation; reconstruction is exact and contributes nothing) plus a graded
budget `λ Σ m`, `λ = .001` relative to the answer's relative error (a
parameter); the hard `QueryWorkBudget` allowance stays for the walk's
rounds. At test the filter is hard: `m > ½` admits a word, with the budget's
top-k as a cap when configured. Parameter: `attentionFilter`.
**The gate, `XOR_filler.xml`:** XOR_grammar's four content pairs embedded
among two to four filler words drawn from a fixed eight-word filler
vocabulary, in random positions; training and held-out arrangements
generated deterministically from the dataset definition; bars: class 4/4
correct and MSE < .05 on held-out arrangements with the hard filter, and
the two highest-weighted words equal to the content pair on ≥ 95% of
held-out sentences.

**F. The catalogue's remaining sections** (catalogue §5–§6; plan §2 round
4): `lower` as the determiner — `a` mints a referent, `the` binds to an
earlier occurrence by its address key (A), `every` stays high-order —
`generic`, `lift` (§5.2–§5.6); the relations and the sentence that states a
definition as an `equal`/DEF row (§6.2–§6.3); operators tested by name
(about 140 places) become declared properties; the item-7 residue
(predicate identity as a rule property; `GrammaticalQueryRegistry`
retired). `surface`, tense, morphology, aspect and `null` stay 5.5's.
Focused tests per catalogue section; no gate parameter.

**G. Measurement.** One full sweep; then the thirty standing trainings
(sum, XOR class and reconstruction, MM_xor) and ten `XOR_filler`
trainings; the static certificates (form audit, semantic certificate,
centroid and containment audit, consumer census) and the store report (rows
used against capacity; the XOR gate's four rows). Expectations: class 10/10
at zero; reconstruction 10/10 (`R ≡ 0`); sum 10/10 at ¼ (every part keeps
the control additive); MM_xor 10/10 (path RNG-only, bisection repeated);
the filler gate at its bars. If a count misses, bisect by switching B–E off
one at a time on the failing gate and report which part moved it, with the
cost records. Receipt `doc/benchmarks/2026-10-07-operators-final/`; no seed,
retry or replacement; stop for Claude's review before any commit. Then item
6.5.


## 37. Attention is a mask over `.where`, not a filter (Alec, 2026-10-07); §36 part E withdrawn

Alec: "Attention is not a soft filter, it is a mask that operates on the
`.where`. It moves from an initial global scope per sentence to a mask on
each word, allowing serial processing (and various non-parallel grammar
operations) to process a word at a time. It incurs a loss in so far as it
filters out parts of the target; the part of the input that is not masked
has a similar cost function that is applied to the percept/concept chain."

My "soft filter" (§8.1 as I recorded it, §11 item 7, §35, §36 part E) was a
per-word *relevance* weight trained to drop words the answer does not need;
that is not attention. Attention is the narrowing walk that exists (6.8):
the open bracket over the sentence, divide/descend/gloss to each word, the
scope handed to conceptual space, every word visited (the walk reserves its
rounds). Its loss is what its mask leaves out of the target — under identity
by construction, a word missing from the reconstruction — which the
owner-step trial cost already charges and credits to the walk's choices
(round 2). **§36 part E and the `XOR_filler` gate are withdrawn**; §8.1's
"decided in direction" and §11 item 7 are superseded; the filter toy stays
as a record of the withdrawn design. Part D (priming) stays: it is the
walk's prior. Architecture, GradientFlow and FutureWork corrected the same
day.

## 38. The combined hand-off as amended (Claude, 2026-10-07)

§36 with part E and the `XOR_filler` gate removed (§37), parts A and B
written out in full so the text stands alone. Given to Alec for Codex the
same day; the chat text is the hand-off.

## 39. Review of the combined step (Claude, 2026-10-07): not accepted; three repairs

Receipt `doc/benchmarks/2026-10-07-operators-final/`. Delivered as specified
and documented with unusual care (including an invalid first bisection
harness, caught and replaced by verified switches). Results: class **10/10**
at zero; reconstruction **9/10**; sum additive 10/10 but at ¼ **9/10**;
MM_xor **10/10**; store 8/1,024 rows per grammar run (4 sentences re-witnessed
400 times, 4 DEF), addresses identical across runs; the semantic certificate
exact on the gates (BasicModel diagnostic: `a ∧ b` .03%, `a ∨ b` 3.7% false
membership, no false negatives); the centroid moves 44.7% of BasicModel's
words with no collapse (.467 → .472) and containment 3,114 → 0; no
name-dispatch left (72 rules, 46 sites). Not accepted, for these reasons:

1. **Negation reaches reconstruction again, now through the meaning block.**
   XOR-09's final greedy walk applies `not`, `and`, `or` over the two-word
   bracket before descending — 2b's signature — its `R` is .018 on every
   epoch, and its readback is one word per sentence. XOR-wide, 1,146 `not`
   departures change `R`. 2c kept negation off the form (`|a|`), but B
   exchanges a negated leaf's meaning poles, and the inverse searches a bank
   holding each word's positive meaning only, so a negated leaf cannot be
   recomposed. The inverse must offer both of a word's symbols — "every
   concept has two symbols; they differ only in sign" (Alec, 2026-09-23) —
   the stored meaning and its pole exchange, form identical. Reconstruction
   is then blind to polarity, as it must be; the answer alone judges `not`.
2. **A bootstrap inconsistency.** Every grammar and sum run has
   `R = .018` on both trials at epoch 2 and zero afterwards (C-off removes
   it): the first epoch at which meanings and centroids become nonzero. The
   composition, the inverse and the readback within one forward must read
   the same snapshot of codes; a code that moves between composing and
   inverting is reconstructed wrongly once.
3. **Sum-08's reader stalls at a common offset** (predictions .006–.008,
   MSE .49 instead of ¼, contrast 3e-8): the control stays additive but the
   presented reader never learns the mean. Each single switch-off passes,
   which shows trajectory sensitivity, not a cause. Needed: an all-on replay
   of sum-08 from its saved entry state (determinism), with the root's
   per-block norms and the reader's pre-activations, outputs and gradient
   norms over training, to say why the affine read cannot reach the mean.
4. **No green sweep on the measured source.** The one sweep ran on the
   pre-repair source (20 failures, repaired, focused contracts green); the
   measured source needs its own full sweep.

**MM_xor.** The bisection is decisive and honest: the path is not RNG-only —
the relative `.when` band enters the raw numeric forward through
`PartSpace._embed_radix`; restoring the old band at that one site restores
the landing's output. That is A working as specified (a numeric row is its
own document; its band is now constant where the absolute clock varied).
The gate counts as live, and holds at 10/10.

**Observation.** Membership priming produces boosts of 8.35 and 3.65 against
a neutral 1.0; final word reads are code winners without a priming
tie-break, so no effect is shown, but the boost's range should be bounded
(normalized to the neutral 1.0) before priming meets a corpus where it can
decide reads.

## 40. Repair pass for the combined step (Claude, 2026-10-07)

Start from the measured combined source
(`doc/benchmarks/2026-10-07-operators-final/delivered-source/`). Four
repairs, then the same measurement once.

1. **Polarity-blind inverse** (as restated in §41). The pair search, the
   decomposition chooser's shortlist and the readback identify each word by
   its signless form alone; the word's meaning is then read with the
   polarity its leaf carries. One bank entry per word. A negated leaf
   recomposes exactly and reads back as its word. Certificate
   before the trainings: the forced `not/and/or/descend` prefix on the
   frozen fixture, with B and C on, recovers 4/4 multisets at zero pair
   residual and `R = 0`; the narrowing `not` departures then tie on `R`,
   and the answer term alone credits them.
2. **One snapshot per forward.** The meaning context means and the
   centroids are snapshotted once before the forward, and composition,
   inverse search and readback all read that snapshot; nothing refreshes
   them until the forward's writes are committed. Expected: `R ≡ 0` on every
   trial of every run from epoch 1, including epoch 2.
3. **Sum-08 diagnosed.** Replay sum-08 all-on from its saved entry state
   (it must reproduce .4929), recording per epoch the root's form and
   meaning norms, the reader's pre-activation, output and gradient norms;
   state the cause. If it is a scale or saturation of the reader's input,
   fix it (per-block normalization of what the reader reads); if it is
   neither, report it without a fix.
4. **Bounded priming boosts.** Membership priming's boosts are normalized so
   the neutral boost is 1.0 and the largest is at most 2.0 (a parameter);
   report the resulting range.

Then one full sweep on the delivered source (must be green), the thirty
standing trainings, and the certificates and store report as before;
expectations as §38 (class 10/10, reconstruction 10/10 with `R ≡ 0`, sum
10/10 at ¼, MM_xor 10/10 live). Receipt
`doc/benchmarks/2026-10-07-operators-final-b/`; the first receipt preserved
intact; stop for Claude's review before any commit.

## 41. The order alternation and the "two symbols" (Alec and Claude, 2026-10-07)

Alec: every concept has positive and negative evidence (accounting for its
magnitude when composed of several concepts); the original tower alternated
perceptual parts/wholes → concept → two symbols → concepts over the field of
symbols; since 10-04 a zeroth-order concept's symbol is its location in
perceptual (mereological) space, leaving no subsequent symbol — is there a
reason to add one? Following perceptual parts/wholes with a symbol would not
respect the perceptual/conceptual alternation.

Claude's reading (for Alec's confirmation): symbols are perceptual objects
(forms, signifiers), concepts conceptual ones, and the alternation holds:
percepts (perceptual) → zeroth-order concept, the aggregation of parts and
wholes (conceptual) → its symbol, the resulting location (perceptual) →
first-order concept, composition over symbols (conceptual) → its symbol, the
sentence's location as the whole of its words (perceptual) → … No symbol
is added after the location; a second one would be perceptual after
perceptual. "Two symbols differing only in sign" (2026-09-23) does not
survive: perception has no sign, so a concept has one signless symbol; the
two signs belong to its evidence (for, against), which is conceptual.

Consequence for §40 item 1: the inverse identifies each word by its
signless form alone (the pair search and the decomposition shortlist on the
form block), then reads the word's meaning with the polarity its leaf
carries — polarity-blind by construction, one bank entry per word. If Alec
confirms, the 2026-09-23 wording in the accessible-mind spec (§2.6.2
amendment) and Philosophy is amended to: one signless symbol per concept, in
perceptual space; two evidence poles on its meaning, in conceptual space.

## 42. The tetralemma at the leaf (Alec, 2026-10-07); §41's reading retracted; repair item 0

Alec: "This does not respect the tetralemma. There is positive and negative
evidence, and they live in different swim lanes. Each contributes to one
positively-valued symbol." Two-truths §1.1 (from his 11c decision,
2026-09-24) already says so: every concept keeps `(c⁺, c⁻) ∈ [0,1]²`, two
positive symbols sharing one identity and one code; store the pair, never a
signed collapse, which loses *both* from *neither*. §41's "one signless
symbol, evidence as its magnitude" is withdrawn; the alternation it defended
holds without it: the zeroth-order concept's symbols are its location at
two magnitudes, perceptual; no further symbol follows.

**Where the code collapses the pair.** `ModelAttention.pole_activation`
forms `net = pair⁺ − pair⁻` and keeps only its sign on a single
"signed activation"; interpretation then scales the meaning by that scalar
(2c's `meaning × a`, the combined step's "poles exchanged when `a < 0`").
The attention walk itself is two-lane (`field_reduce`: and `(min⁺, max⁻)`,
or `(max⁺, min⁻)`, not = exchange; `narrowing_mask` reads pure/both/neither)
and the stored row keeps `(c_plus, c_minus)` by the spec's rule (min over
nonzero contributions per pole, `ClauseJournal.metadata`); the collapse is
confined to the handoff and the leaf.

**Repair item 0 (before §40's items).** The pair is carried to the leaf:
`[form × presence | code × c⁺ | code × c⁻]` — the form at its identification
presence (signless), the for lane at `c⁺`, the against lane at `c⁻`;
`pole_activation`'s net and sign and the single signed activation are
retired; interpretation applies the two magnitudes to the two lanes;
`_pushed_word_slab`, the reference slab and the closing read the pair.
Operators act lane by lane (catalogue §3.8, restated). Certificate: the
four corners at a leaf — `(1,0)` fills the for lane with the code, `(0,1)`
the against lane, `(1,1)` both, `(0,0)` neither; conjunction and
disjunction of corner pairs by the lane rules; the row's pair by the
required-evidence read (A-true and B-false gives both); and an AST audit
that no site between the walk's poles and the stored row subtracts the
lanes or takes a sign. §40 item 1 (the inverse identifies by the signless
form, then reads both lanes with the leaf's pair) stands and follows.

**One ruling still needed: the two zeros.** The spec's evidence pair treats
a zero pole as unknown ("never a veto": required evidence is min over
*nonzero* contributions). The meaning *code* of part B is an extent over
the read corpus, where a zero bit is a known absence (the corpus is complete
knowledge of what was read), so its conjunction is min over *all* bits
(the membership certificate). Both rules are stated; Alec to confirm that
the extent code's zero is a known absence while the evidence pair's zero is
unknown — or that the code too is min over nonzero, in which case `a ∧ b`'s
extent cannot be read from codes, only from postings.


## 43. Attention and the corners (Alec, 2026-10-07)

Alec: attention over a wide field generally accumulates *both*; narrowed to
a homogeneous object (or a symbol), the unipolar concept pervades it.
Confirmed, and already the walk's law: `narrowing_mask` reads the bracket's
pair — both permits divide, pure permits gloss, neither permits descend —
so the walk divides at both until each bracket is pure. The handed-off pair
of an identified word is therefore unipolar (or neither); *both* re-arises
by composition at the row (A-true and B-false), which is why the lanes must
reach the leaf. Union (max over an extent's positions) and pervasion (min)
coincide exactly on a pure extent. Recorded in Architecture's attention
paragraph.

Alec, same day: "So heterogeneity is a cue to narrow attention, or to learn
more." Yes, at two timescales: within a reading, *both* over a bracket cues
division until the parts are pure (the walk, now); across occurrences, a
*both* that persists at the narrowest extent cues refinement of the concept
into parts where the predicate is uniform (2026-09-23; at order 0 a
distinguishing mint or a property split, above it a sub-concept); *neither*
cues witnessing. The policy for the second cue — when to stop dividing the
reading and refine the concept instead — remains FutureWork's "four corners
as prompts".

## 44. The two zeros resolved by magnitude (Alec, 2026-10-07)

Alec: `(0,0)` is ignorance, but it may occur on a well-defined concept, where
it means the concept's absence; whether a concept has been seen before is
its vector's magnitude — which the 9-21 rule normalized so that a dot product
yields the input's uncertainty — so let the concept grow in magnitude as it
is learned, until it reaches 1.

Three quantities, kept apart: the **direction** (identity, on the sphere),
the **magnitude** (definedness, 0 → 1 with learning) and the **pair**
`(c⁺, c⁻)` (evidence in this reading). A read is `|input| × m × cos`; a
conjunction with an ill-defined concept is small, not false; `min` over all
coordinates stands; the membership certificate holds exactly at `m = 1`;
`(0,0)` is absence at `m = 1` and ignorance near `m = 0`, decided by nothing
but `m`. §42's open ruling is closed.

**Rule (Claude, for Codex unless Alec objects):** `m = n/(n + k)` with `n`
the concept's witnessed occurrences (a word's containing rows; a row's
re-witness count) and `k` the recurrence threshold already used for
admission (4). The meaning code used in composition and reads is
`m × unit(direction)`; a stored row's identity code likewise carries its
`m`. Forms stay at full presence (identity by construction). The pair and
the user's trust are evidence and are untouched. Certificate: `m` grows
monotonically with witnesses and reaches 1 to within `1/(n+k)`; a conjunction
with a concept at `m = ε` has for-lane mass at most `ε`; the membership
certificate is exact at `m = 1`.

## 45. Review of the repair pass (Claude, 2026-10-07): accepted as the operators-update landing

**Alec accepted, 2026-10-07.** Commit, push and the WikiOracle submodule bump
are authorized as one landing from round 3a. The [acceptance record](../benchmarks/2026-10-07-operators-final-b/acceptance.json)
preserves the measured source and labels the first combined receipt rejected.

Receipt `doc/benchmarks/2026-10-07-operators-final-b/` (measured source 716
files, matching the working tree). Sweep green on the measured source
(5,500 cases: 5,214 passed, 285 skipped, one non-strict XPASS). Thirty
trainings: class **10/10** at zero (MSE ≤ 1.3e-8), reconstruction
**10/10**, sum **10/10** at ¼ (.249999985–.25), MM_xor **10/10** (live; its
`.when` path intentional). **`R` and `E` identically zero on all 64,000
grammar trial rows**, including epoch 1 and the first nonzero-meaning
snapshot. No seed, retry, replacement or bisection. At review, the first
receipt's 2,119 files were unchanged; acceptance adds only its README status.

Against §39–§42: (0) the pair reaches the leaf — `[form × presence |
code × c⁺ | code × c⁻]` — with `pole_activation`'s net and sign gone;
the corner certificate is right at every corner, including *true ∧ both =
both*, *true ∧ neither = neither*, *true ∨ neither = true*, and A-true with
B-false stored as *both* by the required-evidence read; the AST census
lists 24 path sites and 11 direct accesses with zero cross-lane arithmetic,
sign or lane reduction, and mutation tests reject injected ones. (1) The
inverse identifies by the signless form, one entry per word, and reads the
lanes independently; the frozen `not/and/or/descend` prefix recovers 4/4 at
zero residual with B and C on. (2) One snapshot per forward, taken before
lexical staging: the epoch-2 artifact is gone. (3) Sum-08 reproduced to the
digit (.49290955) and explained: the reader had reached the mean by epoch
380 and then diverged as the record feature norm grew from ~16 to ~90 with
accumulating priming (reader gradient norm .097 → 57.7); the two fixes —
`readerBlockNormalization` (common per-block snapshot scales, affine in the
root, never the root's own norm) and `primingMaxBoost = 2.0` — hold the
control at ¼ in 10/10. (4) Priming bounded to [1.0, 2.0]; the four gate words
show 2.0 for a present word and 1.36 otherwise.

Also delivered: the `non` footprint corrected to declare its meaning write;
a legacy-width guard; the documentation census excluding executable
snapshot copies. Two earlier sweeps (three failures; one stopped by the
source guard) are retained and superseded by the green one.

**Observations, not blockers.** (a) `min`/`max` lose operand evidence by
construction; the inverse "recovers compatible magnitudes" — a choice among
the compatible ones, to be stated in GradientFlow. (b) One run ends
all-disjunction, reading correctly: with both operators separable on
meanings this is the indifference §27 predicted; the certificate, not the
gate, carries the operator's semantics. (c) The corpus diagnostic's 3.7%
false membership for `∨` at BasicModel's width is the sketch's limit;
postings answer exactly.

**Verdict.** Accept. Land as the operators-update landing (one commit over
rounds 4a-0 to final-b with the plan, the toys, the specs and the documents),
push, bump. §44 (magnitude as definedness) is the first item of the next
step, merged into 6.5's text per Alec.
