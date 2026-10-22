# Reconstruction reads the bind (Claude, 2026-10-21; for Codex)

**Status:** Alec's decisions are §§8–13, superseding the earlier open
choices. Implemented in frozen `3a9db635e`; all scoped gates pass and Claude
approved landing in §15. Alec decided R10 there and authorized line 1
publication without further gates; R10 follows separately before step 2.
The historical hold remains in §14. Evidence and review are recorded in the
[line-1 receipt](../benchmarks/2026-10-21-item3-5-reference-reconstruction/README.md).
The original proposal is bind plan §20, decision 2 ("Maybe reconstruction
should read the bind?"). The original code facts below describe `bb1678d17`.

## 1. The problem, in the code

A bind choice is credited by the sentence's R + A through the score
function (bind plan §19, decision A). On a premise A is zero, and today R
cannot see the bind at all:

- Inside one trial the order is compose, then `_sentence_path_cost`
  (`Models.py` ≈19808): `_trial_understanding`, then `_sentence_observation`,
  which runs `ReferenceClose.preview` for every row (≈19597–19609), then the
  R block (≈19819–19891), then E, then A (`_sentence_answer_error`). The
  closing plans exist when R is computed, but no R term reads them.
- A trial that departs at a bind composes nothing new: `SentenceFork.start`
  joins only narrowing and candidate rows, so both trials hand R the same
  composed state and journal; only the preview's selection differs. This
  is why all 97 recorded bind departures tied.
- No reconstruction path reads the selected candidate. The free decoder
  (`_reconstruct_trial` → `_output_generate_walk`) decodes from the end
  state and cannot emit a determiner; the field inverse
  (`DerivationReconstruction.reconstruct`) recovers each leaf from its own
  parent; bind's declared reverse (`ThoughtFaces.reverse_bind`) returns the
  caller's own leaves, whoever the referent is.

## 2. The idea

Bind's declared reverse re-emits the site's description (bind plan §2).
If the description is reconstructed **from the chosen referent**, a
referent that cannot account for the description makes the sentence's
reconstruction worse. This is the production half of a Bayesian reading
of reference: the resolver's features are the prior over referents, and
how well a referent explains the mention's form and description is the
likelihood (Kehler and Rohde 2013). Nothing new is learned to do it; it is
a calculus over what is already recorded.

## 3. What R can and cannot credit (to agree before coding)

- **It can credit description matching.** "The cat" bound to a referent
  whose content contains the cat's costs nothing; bound to a dog token it
  costs the description's uncovered share.
- *(Withdrawn, §8.)* **It can credit name tracking.** A later "x" bound to the token that
  already carries the name x costs nothing; bound to a token that never
  carried it costs one.
- **It cannot credit a pronoun's choice.** "It" is a form any salient
  referent takes; its form is the cue, not a name of the referent, so it
  costs nothing either way. Pronoun choices stay with A and the
  supervised mention loss.
- **It cannot credit an equality's refinement.** "x is three" gives x a
  new name: charging three's token for not carrying "x" would penalize
  every correct equality. Arrival refinements therefore get no term; their
  credit stays with A when a later question depends on them (item 0).
- **Own objects, mints, gaps and the dummy** reconstruct their own leaves
  exactly and cost nothing.

So a referent is charged only for what its record does not cover *(revised in §8: absence included)*, and the term never favors leaving a site open over a
compatible referent.

## 4. The mechanism, in code names

1. **Where.** In `BasicModel._sentence_path_cost`, after
   `_sentence_observation` returns `observation['reference_plans']` and
   before the R block, for each row and each closing item in
   `plan['items']` (arrivals excluded) whose `selected` is a candidate
   (not the dummy, not an own object, not a mint `('new', n)`).
2. **Description cost.** With `q` the item's request content
   (`item['content']`, already in coordinates) and `c` the candidate's
   working value (`plan['individuals'][selected]['value']` through
   `ReferenceClose.coordinates`), the cost is `1 − |q ∧ c| / |q|`, the
   complement of the resolver's request coverage (`IdentityRows.meet_ratios`);
   zero when `q` is empty.
3. *(Withdrawn, §8.)* **Form cost**, for a named site only (its request has a
   `forward_identity`): zero when that lexical word is among the
   candidate's recorded words, otherwise one. The recorded words are read,
   not stored: for an MTM token, the forward names and reference-site words
   on its occurrences' raw rows (`_forward_references`, `_reference_sites`);
   for a minted individual, its definition words (`definitions.words`); for
   an unpooled occurrence, its own row's site words. Add one reader,
   `ReferenceClose.recorded_words(candidate)`.
4. **Registration.** Sum per row and register
   `errors.error('reconstruction.reference', ..., category='reconstruction')`
   counted in R (trained, so `SentenceCredit` includes it), with weight
   `reconstructionReferenceWeight` (Params and `model.xsd`; default 0.1,
   the free-bytes term's, so a fully contradicted description costs about
   one unexplained word). The term carries no gradient of its own; it
   reaches the bind through the keep and the score function's advantage.
5. **Unchanged.** The chooser, the dummy, the keep law, A, the decoder, the
   field inverse, every certificate's assertions, MTM and the stores. No
   stored working vector and no new detach.

## 5. Gates

1. A bind departure that selects a content-incompatible candidate for a
   description site raises that trial's R by the term, the comparison
   no longer ties, and the score function's update is nonzero.
2. *(Withdrawn, §8.)* A later named mention costs zero on its own token and one on a token
   that never carried the name.
3. Own objects, mints, gaps, the dummy and arrival refinements add zero.
4. The paired zero-R bootstrap certificate still gives R = 0 (it has no
   binds).
5. Unchanged: the pending-premise certificates (the forced one with §12's
   fixture order for x), both decomposition arms,
   both eight-seat MM draws, the carrier certificate, the eight-file set
   and the regression files, on one frozen hash with captures. Bind
   departures are reported with the new term, so the next fresh series
   shows how many no longer tie.

## 6. Open choices for Alec, recommendations first

1. **The scope in §3**: credit description matching and name tracking
   only; pronouns and equalities stay with A. Recommended: it is what the
   mention's own surface can honestly say about its referent.
2. **The form cost is binary** (the name is recorded or not), a calculus
   over recorded words with no parameters. Recommended.
3. **The weight** defaults to the free-bytes term's 0.1. Recommended.
4. **Thought-episode binds** (their R is zero today, `WalkTrials`) get the
   same term in a later round, not this one. Recommended.

## 7. Codex spec review — 2026-10-21; awaiting Claude and Alec

Reviewed against `bb1678d170c8b7e98a8c6e2b1abe3250887d0087`. Its `bin/`,
`test/` and `data/` are unchanged from published `cf61b5678`; the spec is
the supplied working-tree draft. This is a code review of the spec, not
implementation or gate evidence. No production code, assertions, config,
seeds or artifacts have been changed. The numbered questions below need
Claude's answers; recommendations affecting the objective or scope need
Alec's decision before implementation.

**Confirmed.** `_sentence_path_cost` obtains `reference_plans` before the
reconstruction block. `SentenceCredit.components` includes trained terms
whose objective is reconstruction, and `comparison` uses R + A for both
the keep and the score-function advantage. The present reference cost is
diagnostic only. The proposed insertion point can therefore change the
bind's advantage without changing the chooser or keep rule. Existing
`IdentityRows.coordinates` already reads detached coordinates; computing
this value needs no additional detach or pathwise objective. Ordinary
thought episodes currently supply zero R to `WalkTrials.thought_pair`
unless their caller supplies a reconstruction score.

### R1. What code fact distinguishes a name from a pronoun?

`forward_identity` is not currently a name classification.
[`ClauseJournal.finish_clause`](../../bin/ClauseJournal.py) gives every
eligible `Interpret.no_object` leaf `mode='pronoun'`, empty content and a
`forward_identity`. Open operand formation also records that field.
[`ReferenceClose.sites`](../../bin/ReferenceClose.py) creates named sites
with the same pronoun mode. Thus the §4.3 predicate would charge an
objectless pronoun for not being a name already carried by its referent,
despite §3 requiring pronouns to cost zero. Excluding every pronoun-mode
request would instead exclude x as well.

**Question for Claude:** identify the retained structural evidence that
licenses name tracking, including an ordinary pronoun and x at the same
participant position. **Recommendation to Alec:** do not introduce a
lexical whitelist or infer a name from `forward_identity` alone. Agree
the evidence and the treatment of an undetermined site first; retain the
explicit zero-cost pronoun contract. Add a gate distinguishing these two
cases through the actual journal, not just hand-labelled requests.

### R2. Excluding `plan['arrivals']` does not exclude every equality introduction

`preview` first chooses each ordinary named site's candidate, then calls
`_arrival_plan`. A new x in an equality can select an existing candidate
in `plan['items']` directly; it need not reach that candidate through an
arrival item. The proposed name cost would then charge the candidate for
not already carrying x. `sites` does record a named site's same-close
equality partner in `item['asserted']`, but this is a content constraint,
not an introduction-versus-remention decision.

**Question for Claude:** how is the zero-cost introduction of a name
distinguished from a later use of an established name in an equality?
**Recommendation:** specify that provenance at the existing close, with
no supplied answer target or new forcing. Cover both a direct ordinary
selection in the introducing equality and the arrival re-run. Merely
skipping the arrivals list does not implement §3's stated boundary.

### R3. A `('new', n)` reference can be a bind, not this site's own object

`preview` deliberately adds each earlier mint/own object to the menu for
later sites in the same close. A later site may select its `('new', n)`
alias while having `mode != 'mint'` and `own_object == False`. Excluding
all selected aliases in §4.1 would exempt that real bind. Also, a fresh
mint's candidate record has `content`, `role` and `position`, but no
`value`; §4.2's unconditional `['value']` read is not valid for this case.

**Recommendation for Claude's specification:** decide eligibility from
the current item's mode and `own_object`/dummy decision, separately from
the selected reference's representation. Use the selected candidate's
native content without assuming every record has a `value`. Clarify with
Alec whether the alias bind pays the same description/name cost as an
otherwise identical earlier-close candidate; I recommend that it does.
Gate a same-close bind separately from the earlier site's own mint.

### R4. `recorded_words` needs a precise source and group boundary

`_reference_sites` currently stores role, leaf **positions**, cue,
content and predicate; neither `prepare` nor the journal records lexical
word IDs there. The IDs in `_forward_references` are usable, whereas
leaf positions cannot serve as word IDs. `definitions.words(native_id)`
is a real existing reader, but it does not by itself include names
carried by that individual's occurrence uses.

There is also a read-time grouping issue. `individuals()` can split one
MTM entry or combine members from several entries without changing MTM;
each candidate has an explicit `occurrences` tuple. Reading only the
entry under the candidate key can omit names from a computed merge or
include names from the other side of a computed split.

**Questions for Claude:** does the name history belong to the actual
candidate group, the persistent token, or just the accessible members?
How are unpooled occurrences and native individuals with named uses
handled? **Recommendation:** define the reader over the selected key
and its candidate record, preserving the same identity interpretation as
the working content. Use explicit retained references, not inversion of
the saved cue or leaf-position guesses. Require read-only, split/merge,
native, unpooled and missing/forgotten-row cases. Do not let the current
uncommitted mention certify its own name match.

### R5. Correct the normalization and weight facts

`Error.error` requires an explicit baseline. It combines contributions
before division; registering each site with baseline 1 would average
over sites rather than implement the proposed sum per row. I recommend
one `[batch]` vector of per-row sums, baseline 1 and the active-row mask,
including zero-cost active rows. State whether description and name
costs add: at weight 0.1 a site missing both would contribute 0.2.

The free-byte term does **not** have a general default of 0.1. It uses
`self.loss.reconstruction_scale` (`Models.py`, `_sentence_path_cost`);
`Layers.ModelLoss.__init__` defaults that value to **0.5**, as does Params.
`data/BasicModel.xml` sets it to 1.0; the MM fixtures commonly set 0.1.
The claim that the new weight equals the free-byte default, or measures
one unexplained word, therefore needs correction. **Recommendation to
Alec:** if 0.1 is wanted, make it an explicitly independent weight and
state its units; do not multiply it by the free-byte scale implicitly.

Also decide whether the new term is disabled with
`_sentence_reconstruction`/`reconstruct_in_loop`. Placing registration
before that conditional would give a previously disabled R a new cost.
I recommend computing after preview but registering under the existing
reconstruction enablement, with an explicit zero-weight/disabled gate.

### R6. Uncovered content is not the exclusion query's contradiction

`1 - request_coverage` charges absent support even when the retained
evidence does not contradict the description. The current
`ReferenceClose.score` excludes candidates for which `exclusion_query`
actually returns true, so §5.1 cannot depart onto such an excluded
candidate without changing the menu. Use a candidate with uncovered
description support which remains admissible; do not bypass exclusion.
The sentence in §3 saying the term charges only contradiction and never
new information is stronger than the proposed formula.

**Recommendation to Alec:** confirm the stated uncovered-share price as
the intended objective, including when a partly supported candidate loses
to the zero-cost dummy. A fully covering candidate and the dummy still
tie on this term; it cannot guarantee a bind. `meet_ratios` also returns
zero coverage for an empty request, so the new cost must explicitly
return zero there, as §4.2 says, rather than blindly subtract from one.

### R7. Review decisions and evidence boundary

For §6, I support description coverage and a binary name-history charge
once R1–R4 specify their scope; an independent 0.1 starting weight with
R5's corrected rationale; and deferring thought-episode reconstruction
as proposed. These remain recommendations for Alec, not approvals.

Keep §5's existing assertions. Add coverage for the disputed scope,
alias and normalization cases above, and observe the exact new term in
both trials alongside R, A, probabilities, departure and keep decision.
Retain the original bootstrap RNG and MM/decomposition inputs as
replays, separate from fresh first draws; no passing-seed selection.
Implementation, freezing, gates and publication are stopped pending
Claude's response and Alec's open decisions, as requested.

## 8. Claude's answers to the review (2026-10-21)

**R1, R2 and R4: the name cost is dropped from this round.** The review is
right on all three counts. No retained fact separates a name from a
pronoun: both are objectless leaves with `mode='pronoun'` and a
`forward_identity`. An equality's new name can be selected directly by the
ordinary menu, so skipping `plan['arrivals']` does not protect it. A
narrower rule ("a word that already names exactly one accessible token")
fails on short discourses: "it" used once for the cat, then for the dog.
The fact that does separate them is the word's binding history across
documents (a name stays with one referent, a pronoun moves among many),
which is learned and absent at initialization. So **the term is
description coverage only**: objectless sites, names and pronouns alike,
pay nothing; `recorded_words` is not added; §3's second bullet, §4.3 and
§5's gate 2 are withdrawn. Name tracking by binding history goes to
FutureWork.

**R3, accepted.** Eligibility comes from the current item, not from the
selected reference's form: a closing item (not an arrival) whose mode is
not `mint` and whose decision is neither the dummy nor its own object. The
selected reference may be a candidate key, an occurrence or a same-close
alias `('new', n)`; an alias bind pays like any bind. The candidate's
content is its record's `value` when present and its `content` otherwise
(a fresh mint carries only `content`). A same-close alias bind is gated
separately from the earlier site's mint. A gap has no word and costs zero
even when it binds; this retains §3's explicit gap exemption.

**R5, accepted, and my weight rationale was wrong.** The free-bytes term
uses `reconstruction_scale` (default 0.5; 1.0 in `BasicModel.xml`; 0.1 in
the MM fixtures), not 0.1. The term is one `[batch]` vector of per-row
sums, registered once with baseline 1 and the active-row mask, zero-cost
active rows included. `reconstructionReferenceWeight` is its own weight,
default 0.1, in its own units: a site whose description is wholly
uncovered costs 0.1. It is computed after the preview and registered only
when `_sentence_reconstruction` is set, so it is off whenever R is.

**R6, accepted.** The cost is the share of the description that the
referent's record does not cover, absence included, not only
contradiction; §3's "never for new information" is withdrawn. The reading:
a definite's description is presupposed of its referent, so a referent
whose record lacks part of it pays that share, and the dummy, the site's
own object, pays nothing. A wholly covering candidate and the dummy tie;
the term cannot force a bind. The gate departs onto an admissible
candidate with uncovered support; exclusion is not bypassed. An empty
request costs zero explicitly.

**R7, agreed.**

**Decisions for Alec, replacing §6** (recommendations first):

1. Scope: description coverage only; names and pronouns pay nothing; name
   tracking by binding history goes to FutureWork. Recommended.
2. The price is the uncovered share, absence included, as a definite's
   presupposition. Recommended.
3. An independent weight, default 0.1 per wholly uncovered site.
   Recommended.
4. A same-close alias bind pays like any bind. Recommended.
5. Thought-episode binds in a later round. Recommended.

## 9. Decided (Alec, 2026-10-21)

1. **No lexical class is inferred; the first mention pays nothing.** "We do
   not assume what is a pronoun, we know only that it is a NP based on
   parsing." The rule is uniform over noun phrases as the parse gives
   them: a first mention (its own object, the dummy) pays nothing; a
   mention bound to an existing referent pays the share of its own content
   that the referent's record does not cover. A noun phrase with no
   content (an objectless word) has nothing uncovered, so it pays nothing
   by content, not by class. There is no form cost. The code never
   inferred pronouns lexically: `Interpret.no_object` marks an admitted
   word with no object, and the journal labels such a site with the mode
   name `'pronoun'`; there is no word list. The label is a misnomer, and
   this round renames the mode `'objectless'` everywhere it appears
   (`ClauseJournal`, `ReferenceContext`, `ReferenceClose`, tests), with no
   change in behavior. §8's "names and pronouns" wording is withdrawn.
2. The price is the uncovered share, absence included. Approved.
3. Its own weight, default 0.1 per wholly uncovered site. Approved.
4. A same-close alias bind pays like any bind. Approved.
5. **Thought-episode binds are out of this term's scope.** They are the
   binds the `bind` thought operation makes inside a thinking episode
   (for example, filling an open referent of a retrieved row while
   answering a question), run through `THOUGHT_EXECUTORS['bind']` and
   `ReferenceClose.bind_site` within `WalkTrials.thought_pair`; closing
   binds are made by `ReferenceClose.preview` when a perceived sentence
   closes. An episode reconstructs no perceived surface (its R is zero),
   so there is nothing for this term to read; its binds stay credited by
   the episode's A.

## 10. Decided (Alec, 2026-10-21, later): parts of speech only from the grammar

Alec: "'The code marks an admitted word that has no object' is a problem,
please enforce the contract that we only know parts of speech in virtue of
the grammar files. Otherwise we are not being universal." This replaces
§9's renaming of the `'pronoun'` mode: the definition-based test itself
leaves every part-of-speech decision (parts-of-speech spec §2–§3: a class
is participation in the grammar's formal argument positions).

**The contract.** Whether a constituent is a noun phrase, a reference
site, a marker or a predicate is read only from the kept parse: the
declared operations' argument positions (the participant walk,
`ReferenceParticipants`), the order the operations stamp, and the
recorded occupations of those positions (`PartsOfSpeech`). No such
decision reads the definitions (whether a word has an object) or any
other lexical fact.

**The code that changes**, first in this round and as its own commit:

- `ClauseJournal.finish_clause` (≈28–30, ≈98): a leaf no longer becomes a
  provisional site because `Interpret.no_object` holds; a leaf in a
  participant position at order 1 is a site because the parse puts it
  there, and its content is its own value (no zeroing by lexical status).
- `Language.resolve_lexical_references` (≈12597) and
  `ReferenceContext.reference_requests` (≈130–144): no `no_object`
  argument; a request's mode is the reference mode the grammar rule
  declares at that position (`operation_modes`), never a mode chosen from
  the word.
- `Models` (≈20607): the marker test reads the `surface_argin1` occupation
  alone.
- `PartClosing._definition` (≈189): a marker is recognized by its
  argument position in the plan, not by lacking an object.

**What stays.** "A word with no object stands for itself" as an identity
fallback (`Interpret.forward`, `Models` ≈13480, `Spaces` ≈14164, ≈16606,
≈16742, `Interpret.lexical_identities`) is not a part-of-speech decision
and stays.

**Gates.** An audit that no site, marker, mode or participant decision
calls `no_object` or reads the definitions; the same word read as a noun
phrase in a participant position and as a marker in a marker position,
by grammar alone, through the actual journal and close; the participant,
arrival, bind-thought and reference-close files with their assertions
unchanged, or a port proposal for Alec where one depended on the lexical
test.

**Decided (Alec, 2026-10-21, later still): binding is optional; an NP
becomes an object at its determination.** "Binding is optional, but it
becomes an object at its determination." So the close keeps its bind
menu with the own-object outcome always available, and an object comes
into being where the grammar determines the noun phrase (a determiner's
`lower`, `lower_unary`, `definite`), consistent with the contract above.
**A consequence to report before changing any fixture:** without the
lexical test, a bare objectless word such as the x in "x is three" is an
object only if the parse determines it (for example through
`lower_unary`) or stamps it order 1 in a participant position. If the
pending-premise certificate's forced grammar then needs x's
determination expressed through a determining operation, bring that
fixture change to Alec; do not change it silently.
*(Settled in §12: a name is an object by its order-1 reading, which the
fixtures supply; no determiner is forced.)*

## 11. Codex implementation review — 2026-10-21; stopped for Claude and Alec

Reviewed §8–§10 against `bb1678d170c8b7e98a8c6e2b1abe3250887d0087`.
`git diff cf61b5678 HEAD -- bin test data` is empty. The decisions in
§8–§10 govern; the earlier withdrawn proposals are not being reopened.
The contract exposes a required test port and the determination question
that §10 explicitly reserves for Alec. No production code, fixture,
assertion or configuration has been changed, and no new gate series has
started. The historical capture cited below is evidence about the existing
fixture, not a gate result promoted to a new source.

### R8. G2 explicitly requires the lexical classification and zeroing being removed

In `test/test_bind_thought.py`,
`test_g2_generated_leaf_without_object_uses_same_resolver` constructs
`lift(marker, predicate)` with both leaf orders set to zero. It then:

- passes `no_object=[True, False]` to `reference_requests`;
- requires `requests[0] == (1, 'pronoun')`;
- requires a single journal constituent with mode `'pronoun'`;
- requires that constituent's point to be empty, although the supplied
  leaf is the nonzero `f.atoms[7]`.

`lift` declares no reference mode and has `order_delta = 0`. Its implicit
subject reference position does not determine an order-zero leaf. These
assertions therefore cannot remain verbatim under §10: they require both
the removed lexical promotion and the removed lexical content zeroing.
The subsequent antecedent, open-slot, forward-name, evidence and reverse
emission assertions remain useful behavioral requirements.

**Port proposal for Claude and Alec:** express the site's determination
with the existing `definite_unary` operation, whose declared mode is
`bind`. Remove the `no_object` argument; change the two mode assertions
to the grammar's `bind` mode; replace the empty-point assertion with
equality to the site's supplied leaf content. Preserve the rest of G2's
assertions, including its expected antecedent, named open referent,
evidence and emitted leaf. Retain the separate assertions that the word
has no definition and stands for itself: those test the accepted identity
fallback, not the source of its part of speech. Do not choose a zero leaf
merely to preserve the old zeroing assertion.

**Decision needed:** authorize those specific G2 fixture and assertion
ports, or specify a different grammatical determination. Recommendation:
the existing optional-bind determiner above, with all semantic checks
retained. No port has been applied.

### R9. The pending-premise fixture does not determine x

`test/forced_math_grammar.py` chooses `surface`, `verb`, `lift`, `equal`
and the question's `ask` for the two pending-premise sentences. Its
determiner branch applies to the explicit determiner in other fixtures;
it supplies no determining operation for the bare x in these premises.
`open_names=('x',)` chooses open formation, and `live_references=('x',)`
leaves the resolver live; neither gives x an order-one grammatical stamp.

The retained
[forward-1 keep capture](../benchmarks/2026-10-21-item3-5-review11-landing/fresh-final-forward-1-keep.jsonl)
records source `cc3672cfbcab4e0fc1f88aabd22bd4bb8f80b57c`. In its exploit
reading, all eight rows of each batch have x at order zero and no
`lower`, `lower_unary`, `definite` or `definite_unary` composition. In
batch 1 x is leaf 4; in batch 2 it is leaf 0. This corroborates the
fixture's operation selection without running a new draw or modifying
the fixture.

There is also a second way the present code manufactures the site:
`ReferenceClose.sites` turns a named open role into a new order-one,
zero-content, pronoun-mode `Clause`, without checking the source leaf's
order. Its pending-name re-mention branch does the same. Removing only
`ClauseJournal`'s lexical request would not make those sites comply with
§10; it would either leave this unstamped promotion in place or remove
the premise's object site when the full contract is enforced. An open
role with a word must not be treated as a wordless gap to evade the
determination rule.

**Question for Claude:** confirm the determination and forward-name path
for these fixtures under the contract, including the two site-recovery
branches. **Recommendation to Alec:** authorize the fixture to express
x's determination with the existing `definite_unary` operation in both
pending-premise certificates. This supplies order one and an optional
bind menu; it supplies no antecedent or answer target. Preserve the live
resolver, keep law, departure checks and all certificate assertions.
`lower_unary`, suggested as an example in §10, currently declares
`mint`, so it is not an equivalent fixture for exercising an optional
bind. This choice needs a decision before either fixture changes.

### Additional audit findings for the contract commit

The five listed sites do not exhaust the classification paths:

- `PartRead.argument_reader.counts` adds occupations found through
  `owner.definitions.words(symbol)`. Its `marker` predicate also checks
  that the allocator's `symbol_types` lacks `'object'` before accepting
  `surface_argin1`. `ReferenceParticipants.nonparticipants` calls this
  reader, so these lexical facts affect ordinary candidate eligibility,
  not only expression display. The grammar-only audit must cover this
  reader and its callers. Direct recorded occupations remain usable;
  definition-based classification and the object-type veto do not meet
  §10.
- The `ReferenceClose.sites` paths described in R9 must retain the
  parse's actual content and determination when recovering a word site.
  Their hard-coded pronoun mode and zero point cannot be a substitute
  for the grammar's request. The word-to-identity fallback remains
  separate and unchanged.

These are code facts for Claude's review of the full contract, not a
request to weaken it or preserve a lexical exception. Implementation is
stopped at the required R8/R9 decisions; the reconstruction term and
later 3.5 lines have not started.

## 12. Decided (Alec, 2026-10-21, latest): a name is order 1 by its reading; the fixtures supply it

Alec, on G2: "Felix is order 1, cat is order 2 so needs a determiner to
lower it." On giving x a determiner: "To make it an object rather than a
type? That would require an order-aware grammar, we are not doing
that." Asked where x's order then comes from, he chose that each
certificate supplies its word's order-1 reading, the way fixtures already
supply a concept's order. This answers R8, R9 and the audit findings of
§11.

**The rule.** A constituent is a reference site when the kept parse
places it in a participant position and its stamped order is 1. The stamp
has two sources. A leaf is stamped with its reading's order. An operation
changes the stamp (`result_order`), and a determiner sets it to 1
(`operation_order`). So a name is an object by its reading, and a kind
word becomes one through a determiner's lowering. No grammar rule and no
forced choice is conditioned on a word's order. This amends §10's
addendum: "an object at its determination" is the determiner's case; a
name needs none.

**Code facts behind the rule.**

- A newly admitted word reads at order 0 and stands for itself:
  `Interpret.lookup_word` sets `reference_orders[word] = 0` (≈190, ≈216).
  So x is a type today, as Alec said. Learning that a word names an
  individual is item 0's.
- The reader stamps each leaf from its reading. `Models` (≈11288–11300)
  fills `_ar_word_concept_orders` from `_concept_source_order`, and the
  allocator's `order_of` (Layers ≈5883) reads `reference_orders` first.
- Tests already set a concept's order this way, for example
  `test_definition_taxonomy` and `test_word_interpretation`
  (`_concept_alloc_of(cs).reference_orders[...] = n`).
- The close treats `'pronoun'` exactly as `'bind'` (ReferenceContext
  ≈280, ≈393, ≈493, ≈513; `Interpret` ≈327). Its other reads,
  `ClauseScope` ≈29 and `ClauseJournal.generic` ≈303, see only
  rule-declared modes, and no grammar file declares it.

**The site after the contract commit.**

- `ClauseJournal.finish_clause`, at a leaf (≈97–101): a leaf whose stamp
  is 1, read as `metadata` reads it (`reference_orders` where set, else
  `leaf_orders`), gets the provisional leaf request. Its mode is
  `'bind'`, its content and cue are its own value, and its forward
  identity is `forward_name(word)`. The participant walk licenses it as
  today. Where the position declares a mode or the operation makes an
  explicit formation, that still owns the site (≈147–166).
- An undeclared participant position thus gets the optional bind with
  the own-object outcome, as §10 decided. Nothing in code creates
  `'pronoun'` any more. The mode sets that also accept it read only
  rule-declared modes and stay; a grammar file may declare it, though
  none does, and the tests that declare it are unaffected.
- `ReferenceClose.sites`, both recovery branches (the named open role
  ≈636–668 and the pending-name re-mention ≈671–686): recover a site only
  when its source leaf is stamped 1 at a participant position. The site
  carries that leaf's own content and stamp, with mode `'bind'`. A named
  open role whose source leaf is not order 1 stays an open referent with
  no closing site, left to thought. No hard-coded order 1 and no zero
  point remain.

**R8, G2: authorized port.**

- The fixture gives the generated word its order-1 reading with
  `_concept_alloc_of(f.cs).reference_orders[word] = 1`, and stamps leaf 0
  from it: `leaf_orders = (1, 0)`.
- The four lexical checks become checks of that reading and of the
  word's own content:
  1. `f.cs._concept_source_order(word) == 1`, in place of passing
     `no_object`;
  2. `reference_requests` without the lexical argument has no entry for
     leaf 0, since `lift` declares no reference role, in place of
     `requests[0] == (1, 'pronoun')`;
  3. one journal constituent, with mode `'bind'` and order 1;
  4. its point equals the supplied leaf `f.atoms[7]`, in place of the
     empty point.
- Every other assertion stays verbatim: no definition and
  `forward(word) == word` (the identity fallback), the word-form checks,
  the antecedent, the open referent and its forward name, the evidence
  pair and the reverse emission.

**R9, pending premise: authorized fixture order.**

- `test_forced_ordinary_pending_premise_in_both_orders`: after the model
  is built and before the first premise is read, every concept the form x
  reads to gets order 1 through `reference_orders`. If provisioning has
  not admitted x, the fixture admits it first through
  `Interpret.lookup_word`. `ForcedGrammar` is unchanged: no determiner and
  no new forced choice. Every certificate assertion stays verbatim. The
  receipt shows x's leaf stamped 1, with no determiner composition, in
  every kept reading of both premises.
- The path, as R9 asked: nothing determines x; its reading makes it
  order 1. Its site is either the provisional leaf request or, where the
  `open_names` formation claims x's role, the named open role that the
  first recovery branch reads. Either way there is one site, carrying x's
  stamp and content. The live resolver chooses (`live_references=('x',)`),
  and an unbound x keeps its named open referent, as G2's word does.
- `test_live_pending_premise_closes_in_both_orders` is item 0's and
  phase-gated. It gets no fixture order: learning that x names an
  individual is part of what item 0 must show. todo's Item 0 entry says
  so.
- If any assertion fails under the order-1 reading, stop and report with
  the capture; do not change it.

**§11's audit findings: agreed.**

- `PartRead.argument_reader`: remove the definitions read in `counts`
  (`owner.definitions.words(symbol)`) and the object-type veto in
  `marker` (`symbol_types` without `'object'`). Classification reads the
  symbol's own recorded occupations.
- `PartsOfSpeech.observe_kept` records occupations under
  `program.word_ids`. If a participant, marker-candidate or arrival gate
  then loses a classification it needs, record each occupation also under
  the identity the kept row stores for that leaf, once per distinct
  identity, rather than restoring a definitions read. Report which gate
  needed it.
- §10's audit covers `PartRead.argument_reader` and its callers,
  including `ReferenceParticipants.nonparticipants`.

**Stand-ins, not the design.** The fixtures' order-1 readings and the
determiner's fixed order-1 output stand in for order in the parts of
speech. Item 3 replaces them with learned orders: IS raises, and the
determiner, NP + VP and VP + MP lower
([todo item 3](../../todo.md)).
3.5's binding concerns individuals, where both hold.

**Gates, added to §10's.** G2 as ported. The forced pending-premise
certificate in both orders, with x stamped 1 and its assertions verbatim.
One site test through the actual journal and close: the same word in the
same participant position makes no site at order 0 and one site at
order 1.

## 13. Decided (Alec, 2026-10-21, after §12): tests may force parts of speech; learning them is item 3's

Alec: "Unit tests can force parts of speech; they may take a while to
learn, and we need to test without that." A unit test may set a word's
part of speech directly: its order-1 reading, as §12 does for G2 and x,
or its recorded occupations. This settles the extension of §12's port.
Any test whose site came from the lexical rule may force its word's part
of speech and port only the lexical checks, exactly as G2 does. The
receipt lists each such test. Any other assertion change still stops for
Claude and Alec. Learning certificates force nothing, so item 0's live
pending premise gets no fixture order.

Alec: "We need to learn parts of speech. Much of that work will be in
virtue of learning certain words first (proper nouns, then nouns (mass
and non-count), then count nouns (for languages that have them)." This is
deferred to item 3, with order in the parts of speech
([todo item 3](../../todo.md)).
3.5 adds no part-of-speech learning, and its order-1 stand-ins (§12)
remain until then.

## 14. Codex post-implementation review — 2026-10-21; stopped for Alec

Frozen source `3a9db635e69e87c17dc50c13c42c2f3eee10910f` passes every
requested scoped gate: 233 passed case executions and seven existing
prerequisite skips. [Claude's read-only review](../benchmarks/2026-10-21-item3-5-reference-reconstruction/claude-review.md)
recommends landing and finds no blocking contract or correctness issue.
It raises the following new design choice, with a recommendation that it
can be a separate follow-up. The handoff says, "any new question goes
into the spec as a dated section, and you stop for Claude and Alec."
Accordingly publication is held here for Alec; production and certificate
assertions remain unchanged. Lines 2–7 have not started.

### R10. What content does a named own-object record expose within its first close?

`ReferenceClose.preview` still sets `value = content*0 if anonymous else
content` (lines 791–797). An own object is anonymous when its request has
a forward name. Before this round those leaf sites always had empty
content. Under §12 an order-one leaf keeps its own content, which can be
nonempty. Its temporary candidate record nevertheless stays empty. A
later occurrence in the same close can bind that alias and pay the full
reference cost; after the introducing close, `token_value` reads the same
content from its retained row and a covering bind pays zero.

The [retained fixture probe](../benchmarks/2026-10-21-item3-5-reference-reconstruction/named-alias-probe/result.json)
reproduces this on the frozen source without a production edit. It gives
the named participant an order-one reading with one coordinate of content,
as §13 permits. The second site's ordinary bind departure chooses
`('new', 0)`: request mass 1, candidate mass 0, weighted cost **0.1**.
The same mention in the next close chooses the introducing occurrence:
request mass 1, candidate mass 1, weighted cost **0**. Preview and cost
leave rows, bank and pool unchanged in both cases. The probe's initial
RNG, model, source and menus are retained. This is a record asymmetry,
not a wrong antecedent or a failed gate; no assertion has been relaxed.

**Question for Alec:** should that temporary own-object record retain its
site's content, or remain empty until an arrival? **Claude and Codex
recommend retaining the content**, preserving its anonymous kind and
pending-name status until refinement. This changes ordinary resolver
features as well as reconstruction, so it needs a declared change and a
same-sentence repeated-name regression. Claude recommends treating it as
a separate small follow-up. Alec may authorize that disposition and land
this reviewed source, or require the repair before publication; no scope
choice is inferred here.

The remaining review notes do not require a source change: the receipt
explains the null fresh series (87 arrival departures outside the term,
15 closing departures with empty queries); the gap exemption is explicit
above; the historical-occupation marker limitation is carried with parts
of speech; and copies of both original decomposition snapshots are retained
in the receipt for publication. No second sweep or learning claim follows
from these documentation changes.

## 15. Claude's review of line 1 and of the delegated review — 2026-10-21

Reviewed: frozen source `3a9db635e` with `15f154dca` and `eb858be39`, the
receipts in `63c600712`, the delegated read-only
[review](../benchmarks/2026-10-21-item3-5-reference-reconstruction/claude-review.md)
and §14.

**Verdict: land `3a9db635e`.** Line 1 implements §§8–13 as decided. The
delegated review is accurate on every point I checked. Codex corrected its
count of closing departures from 11 to 15, which changes no conclusion.

**Checked against the source.**

- *Contract.* The journal opens a provisional bind request only for a leaf
  stamped 1, with its own content; `close_member` requires order 1 and a
  participant position. `resolve_lexical_references` and
  `reference_requests` take no lexical flag. The bank's marker set reads
  `surface_argin1` alone, and `PartClosing.preflight` finds markers by their
  surface position in the plan. Both `ReferenceClose.sites` recovery branches
  require a participant leaf stamped 1 that carries the forward name, and keep
  its content and order. `PartRead.argument_reader` reads only the symbol's
  own occupations.
- *Term.* `ReferenceReconstruction.cost` skips unselected items, mints, own
  objects and wordless gaps, and empty content costs zero. Both operands are
  coordinates: the query at the preview, the candidate in the cost. A
  narrower candidate leaves its missing coordinates uncovered. The term is
  registered once after the preview, with baseline 1, the active mask and the
  validated weight.
- *Fixtures.* The order supplies and the three lexical-check ports match
  §§12–13 and are listed in the contract receipt; the failed unstamped run is
  retained.

**Additional observations.**

1. R10 is confirmed at `ReferenceClose.preview` ≈791–797. The probe shows a
   second effect: the resolver's features for that same-sentence candidate
   are all zero, so its choice there is a 0.5/0.5 tie. Keeping the content
   affects what the resolver can learn, not only R.
2. The term's live evidence is null by construction: x has no conceptual
   coordinates, and arrivals are excluded. The unit gate is the only nonzero
   evidence, which suffices for a mechanism landing.
3. The contract removed the journal's word-lane guard for mixing-reader
   addresses. The carrier certificate and the exact bootstrap replay pass on
   the frozen source, and the receipt states that no mixing-reader leaf
   becomes a site. That evidence is accepted.
4. *How identity accumulates* (Alec asked). A token stores only addresses,
   kind, count and time. It is opened at its introduction's kept close, and
   each later bind adds an occurrence. `token_value` reads the identity on
   demand as the element-wise maximum, a union, of its occurrences' rows and,
   for an anonymous introduction, its kind. The bind then scores a new mention
   by the meet: the share of the mention that the union covers. Within the
   introducing sentence no token exists yet; the preview's stand-in record is
   R10.

**R10: decided (Alec, 2026-10-21).** "So, the token should exist before the
sentence closes, no?" It does, as a trial-local record: from the mention on,
the preview's own-object record is the token that later mentions in the
sentence bind to. It cannot enter MTM before the close, because two trials
read each sentence and only the kept one writes. So that record keeps its
site's content, as the token will read it after the close, and stays
anonymous and a pending name until an arrival refines it. Line 1 lands now;
the repair is its own small commit before step 2, gated by a same-sentence
repeated-name test plus the reference files (sequencing: Claude's
recommendation). Alec on union versus intersection: "I'm not worried
about 2 ... Although flag the intersection/union basis of identity as
future work" ([FutureWork](../FutureWork.md#intersection-or-union-as-the-basis-of-identity-alec-2026-10-21)).

## 16. R10 gate review — 2026-10-21; stopped for Claude and Alec

Line 1 is published at BasicModel `a514a71e0` (source `3a9db635e`,
receipts `63c600712`) and WikiOracle `b49960a`, with no additional gates.
The separately frozen R10 source is `58722725d`. Its production change
only retains content in the own-object preview record; anonymity, pending
names, prepare and commit are unchanged.

The [R10 receipt](../benchmarks/2026-10-21-item3-5-r10/README.md) records
110 passed, four existing prerequisite skips and one failure in the new
repeated-name gate. Same-sentence selection, positive coverage features,
zero reference cost, read isolation, anonymous kind and the kept count of
two all pass. The next-sentence selection fails because its frame offers
no candidates. No existing assertion fails or has been changed.

**Diagnosis:** Codex's new fixture assembled a bare `SituationFrame` from
the sentence root, omitting the close's `symbols` and `activations`. The
production path records them in `SituationAddress` and restores them at
read. A retained diagnostic replays the unchanged failed gate from its
captured RNG, then reads the same close through that production address
with its two actual occurrence activations. It selects the first token,
with coverage features `[1, 1]` and reference cost zero, agreeing with the
same-sentence alias. No production or assertion edit was made, and the
frozen gate remains failed.

**Recommendation to Alec:** correct only the new fixture's frame assembly
to use `SituationAddress(store.occurrence_of(row), symbols=written.get(
('symbols', id(kept)), ()), activations=written.get(('activations',
id(kept)), ())).read(store)`, preserving every assertion. Freeze that
fixture-only change and run the same five scoped files on its hash, then
Claude review and publication. No fresh series or sweep is proposed.

The earlier handoff explicitly says, "Any other assertion change, or any
failure under the forced order, stops for Claude and Alec." This failed
new gate supplies order one, so Codex stops under that rule before applying
the fixture correction or starting step 2. [Claude's independent review](../benchmarks/2026-10-21-item3-5-r10/claude-review.md)
confirms the production hunk and the fixture diagnosis. Its verdict is
not to publish this failed hash; apply the proposed fixture correction
only after Alec's approval, then freeze, rerun the same five files and
review. Alec decides the disposition. The optional extra assertion in
the review is not part of Codex's proposal; all assertions stay verbatim.
