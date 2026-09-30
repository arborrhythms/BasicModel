# Two truths in LTM: ideas collapse, relations refer

> **Status:** specification, 2026-09-16, written by Claude from Alec's
> decisions in conversation on 2026-09-16, including the answers to the
> four questions the first draft left open and the fusion-over-references
> decision (§9 records them). For Codex to implement in a new session;
> Claude reviews the implementation afterwards. Governs the conceptual
> content of every LTM row, the grammatical closing that writes it, and the
> trust rule. It refines the two-truths passage in
> [Philosophy.md](../Philosophy.md) (satya-dvaya) and **supersedes** the
> nested-clause treatment in
> [the integrated plan §2.1](../plans/2026-09-15-next-sentence-as-the-production-objective.md#21-nested-clauses-and-phrases),
> where every embedded clause was a reference. Everything marked
> **(decided)** is Alec's decision. Nothing here claims learned behaviour.
> §3.5, object permanence by reference, was added on 2026-09-21.

## 1. Definitions (decided)

An **idea** is one point in the conceptual space: a noun phrase together
with its verb phrase, `NP VP`, where the VP may be compound, `VP = V NP`.
The second NP of "the cat chased the mouse" is a modifier on the VP, not a
second operand of the sentence. The sentence is **one idea** and an
**ultimate (absolute) truth**: it describes one situation, is evaluable
by coverage / luminosity, and licenses nothing beyond itself. It reduces
to the absolute start state `exist_O1`.

A **relation** holds between two rows: `row R row`. It is a **relative
truth**: a constraint between rows, not a region. It licenses inference
over other rows (parthood, subsumption, modus ponens, attribution) and is
evaluated relationally or by simulation, never by coverage. It reduces to
a relative start state. There are three relation kinds (§3.2): parthood,
implication, and the **operators that take a truth as an argument**
(said, knows, doubts, hopes, if). The last kind is what makes attribution
a relation rather than a scene.

**One grammatical S is one row.** The start state an S reduces to decides
the row's kind:

| S reduces to | Row | Slots | Truth | Example |
|---|---|---|---|---|
| `exist_O1` | idea | one fused point | ultimate | "the cat chased the mouse"; "the cat"; "he said she said it's so beautiful" |
| relative start | relation | three: `row`, `R`, `row` | relative | "cats are animals"; "cats breathe"; "he said cats are animals"; "if cats are animals then cats breathe" |

**Fusion is fair iff every operand has a fused point.** Concept rows and
idea rows have one; relation rows do not, being constraints rather than
regions. So an S fuses when all of its constituents are ideas or
concepts, however deeply nested, and stays three slots when any
constituent is a relation. A fused point over a relation reference would
differ from its neighbours only by an arbitrary row code, and coverage
over it would measure nothing. Relativity therefore propagates upward:
an S that references a relation is itself a relation (§2).

There is **no two-slot absolute form**. The conceptual space represents NP
and VP conjoined; the factoring into NP and VP is recovered from the
recorded derivation (§3.1), not stored as separate truth slots. The
reduce sweep already folds an absolute sentence to depth 1 and holds a
relative sentence at depth 3 by a per-row depth floor; §3.1 names that
existing reduction as the fusion.

**The generic subject decides, not the copula.** "Cats are animals" and
"cats breathe" are relations: their subject is a type, and the predicate
is nominalised as a type (*animal*, *breathing thing*), so both write the
same kind of row, the cat region part of the predicate's region. "This
cat breathes" and "the cat is an animal" (definite, a token) are ideas: a
property of one situation. §7 test 12 asserts this parse from rules
alone.

**Every S gets exactly one row.** Registration is about *reference*: the
row is written when the S endings, so anything that refers to it (a later
"that", an enclosing clause) finds it. Relation rows are referenceable
like any other row (§3.2). Trust is about *assertion* and is decided
separately (§4).

**A bare NP** has an implicit VP, the tense-only existence copula
(`exist`). As an LTM idea it is the concept row itself: eternal by
construction, since concepts carry one code and no location band. A
located occurrence ("cat!", here and now) is an idea with the tense-only
VP and a `.when`. The concept is the idea; its occurrences are located
rows. Neither nihilism nor eternalism, and no third form.

### 1.1 Both is a compositional fact (decided, Alec, 2026-09-23)

Truth in this system is defined over existence. The bare NP's VP is the
existence copula, and a predicate holds of a thing when the thing's region
lies in the predicate's region (§3.2: parthood is the primitive). A thing
has parts, and its parts may differ. A soccer ball is black and white:
some of its parts lie in the black region and others in the white. The
truth of "the ball is black" is then not true, not false and not unknown.
It is **both**, and both is a fact about composition, not a defect of
judgement.

A truth system that keeps one value per predicate — true, false, or a
signed scalar between them — forces every object to be simple: to take
exactly one value under each predicate, which is to be a symbol rather
than a thing with parts. That is the habit of symbolic logic in the
Western tradition, where the subject–predicate form makes predicates
singular. Whitehead named the error: the substance–quality reading of
every proposition, which he held responsible for misplacing concreteness
in the subject and could not survive a world of processes with parts
(*Process and Reality*, 1929, Part II; *Science and the Modern World*,
1925, ch. 3). It is also why mereology has to come before predicate logic,
as Peter Simons argues (*Parts: A Study in Ontology*, Oxford, 1987, whose
improper-parthood primitive §3.2 already adopts): parts are exactly where
one predicate can hold and fail of one thing at once, and identity,
overlap and proper part are derived from parthood rather than assumed.
Both is therefore a value the system **must** be able to state and keep,
because objects are not symbols.

The tetralemma's four corners are all first-class values of a predicate
over a thing: **true** (the predicate holds of the thing as a whole),
**false**, **both** (its parts differ under the predicate), **neither**
(unknown). Both is not a contradiction. A contradiction is two sources
asserting incompatible truths of the *same* part; it lives across LTM rows,
each with its source's trust, and is resolved by trust (§4). Both is one
source seeing a heterogeneous whole; it lives inside one row and must
survive the closing. Dharmakīrti's exclusion of the contradicted cognition
from valid cognition is untouched by this: a heterogeneous whole is
perceived, not inferred against itself.

Consequences (11c, Alec, September 24):

- Every concept keeps `(c⁺,c⁻) ∈ [0,1]²`, two positive symbols sharing one
  persistent identity and one distributed code. The min-based corners are
  `t=min(c⁺,1-c⁻)`, `f=min(c⁻,1-c⁺)`, `both=min(c⁺,c⁻)`, and
  `neither=min(1-c⁺,1-c⁻)`. Store the pair, never just the corners or
  signed collapse, which lose the distinction between both and neither.
- Native memberships supply the order-0 read. A present percept weighted
  positively contributes support; weighted negatively, counterevidence.
  Absence supplies neither. Required evidence uses min over nonzero
  contributions independently on each pole; alternatives and readout use
  max. A-true and B-false therefore give both for A and B. Negating B swaps
  its poles and makes A and not B true-only. Zero is uncertainty on that
  pole, never a veto. There is no projection read, admission floor,
  probabilistic accumulation, or De Morgan dual reducer.
- Pervasion applies to WholeSpace properties over observed positions.
  PartSpace reads ordered containment over canonical part ids within the
  subject extent. Runs are positions, words are extents, and max unions
  the retained occurrence evidence inside each extent. Precision belongs
  only to location. Concepts share the attentive field; none has its own
  `.where`.
- Witness each pole separately: a present percept co-active with the
  positive symbol writes a positive weight; with the negative symbol it
  writes a negative weight. Never write from an absent percept.
- Both can arise across occurrences or from conflicting required evidence
  within one occurrence. It prompts division at order 0; neither prompts
  attention. The action policy for those prompts is deferred in
  [FutureWork](../FutureWork.md#four-corners-as-prompts).
- Pi intersects located order-0 field readings. Sigma unions those cases
  or the previous order's symbols; symbolization raises order. Symbols
  cannot be split. The [Architecture argument](../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)
  explains why union commutes with pooling and intersection does not.
  XOR must be a learned row's positive pole over located mixed cases;
  reading P's pooled both corner is only a diagnosis.
- The taper keeps both poles together. Thought, checkpoints, priming and
  symbol addressing use persistent concept ids, not temporary field rows.
  Unrelated content must remain exactly zero at all scopes, with no limit
  on accumulation to calibrate. Trust remains on accepted LTM rows.

## 2. Collapse versus reference (decided)

An embedded clause whose content is absolute does **not** require a
reference. The grammar's `VP → V NP` derivation, with the NP itself an
`NP VP` idea, collapses recursively:

```text
"he said she said it's so beautiful"
  it_is_so_beautiful          : idea i0 = [it, (be, so-beautiful)]
  she said i0                 : idea i1 = [she, (said, i0)]
  he said i1                  : idea i2 = [he,  (said, i1)]
LTM: one fused idea row (i2), plus the rows of i0 and i1 (every S is a
row). No references pushed. Derivation retained.
```

One ultimate truth: a chain of sayings about a beautiful thing is one
scene. Understanding rests on the single fused representation (gist);
the verbatim nesting is available only through reconstruction of the
derivation.

A **relative truth in any clause forces a reference** to its row, and
makes every enclosing S relative:

```text
"he said cats are animals"
  r  = (cat,  part-of,  animal)        REL_PARTOF     trust 0 (not asserted)
  s  = (he,   said,     REF r)         REL_OPERATOR   trust = observation's

"she said he said cats are animals"
  r, s as above
  t  = (she,  said,     REF s)         REL_OPERATOR

"if cats are animals then cats breathe"          -- second order
  r1 = (cat, part-of, animal)
  r2 = (cat, part-of, breathing-thing)
  r3 = (REF r1, implies, REF r2)                 -- the sentence's own row
```

Rule: a sentence carries exactly the references its relative content
requires and no others. A reference is a row index in the shared
concept/symbol/idea index; the derivation is not a reference. The chain
`r → s → t` is the plan §2.1 `p0/p1/p2` example, now confined to the
relative case.

**Clause-level closing as grammar.** The closing is the reduction of an S to
its start state, with the row write as its side effect. Making that
reduction available inside a sentence is the clause-level closing:

- `NP → S` for an absolute clause: the clause fuses to a point and enters
  the enclosing VP as its modifier. No reference is pushed; the clause's
  row exists and the enclosing derivation records it.
- `NP → REF(S)` for a relative clause: the clause holds its three slots,
  writes its relation row, and pushes the row reference onto the STM as
  the NP. The enclosing S then cannot fuse (§1) and endings as an operator
  relation over that reference. The complementiser "that" is the surface
  form of this rule.

The depth floor that holds a relative sentence at depth 3 becomes
clause-scoped: it applies to the relative clause's own fold and to every
enclosing fold that received a reference. Embedded clauses closing
bottom-up, so a referenced row always exists before the referring S
endings. Mid-sentence row writes are host-side, like the existing
sentence-boundary hooks; the compiled reduce sweep carries only the
clause-scoped floor and the reference push.

The plan's §2.1 is rewritten to this rule (§8). The "compound-reference
prediction" gap in the plan's §8.3 is closed by definition: an NP slot
may hold any row, and because ideas, concepts, symbols and relations
share one index, no new reference mechanism exists.

## 3. Storage contract

### 3.1 Idea rows (decided)

*Amended (Alec, 2026-09-25, item 9b):* the row schema gains **`.where`
and `.when`**, the ended field's one bracket and one interval, written
once per row by the closing beside `refs`, the surprise column and the
`(c⁺, c⁻)` pair. A row's address is its `.when` alone; its `.where`
records what it was looking at, so rows may share a `.where`. No concept
inside the row carries either coordinate
([Architecture](../Architecture.md#one-where-one-when-many-whats-the-field-attention-and-the-two-modes-item-9b-september-25)).


An idea row in the unified `TernaryTruthStore` (`symbolSpace.ltm_store`)
has `rel_type = REL_NONE`, its fused vector in the NP1 slot, Null in VP
and NP2, scalar trust, timestamp, origin and optional source text as
today. Two additions:

- **Derivation.** The row records the compose derivation that produced
  the fused point (the leaf rows and forced forward actions the answer
  path already replays through the grammar's ops), so NP and VP are
  readable by replay. Cache the two top-level operand references (subject
  row, predicate row) in a new per-row reference column `refs
  [capacity, 3]` (int64, `-1` = none); the third entry is unused for
  ideas. The cache is a convenience for readers; the derivation is the
  record.
- **Fusion** is the grammar's existing reduction of an absolute S to
  depth 1 (the `exist_O1` start). It is not a mean, a concatenation or a
  learned projection outside the grammar. Every fused idea is a row in
  the shared index, allocated the way an admitted phrase's concept row is
  allocated (fold-ladder contract 2), so it can be referenced.

The current `REL_OTHER` writes at the packed drain and the non-packed
sink are retired: a sentence either reduces to `exist_O1` and fuses, or
reduces to a relative start and is one of the three relation kinds.

### 3.2 Relation rows (decided)

*Amended (Alec, 2026-09-29, §17):* a fourth kind, `REL_DEF`, ties a word
concept to its object concept. It is written by `interpret`, not by the
closing, and it implies nothing about order.

`rel_type ∈ {REL_PARTOF, REL_IMPLIES, REL_OPERATOR}`. There is no
`REL_EQUAL`, no `REL_WHOLE` and no `REL_OTHER`.

- **Parthood is the mereological primitive**, and it is improper: a
  thing is part of itself. "Whole" writes a part row with the operands
  swapped. "Equal" writes two part rows, A part-of B and B part-of A,
  each with its own trust; equality never merges rows, so a false
  "X is Y" costs one row's trust, not every fact about both. Subsumption
  between concept rows ("cats are animals") is a part row: the cat region
  is part of the animal region. Reference: Peter Simons, *Parts: A Study
  in Ontology*, Oxford University Press, 1987, for the choice of an
  improper-parthood primitive and the derivation of proper part, overlap
  and identity from it. This is the symbolic parthood between rows; the
  order-0 `.where` meronomy of percept extents is a different mechanism
  and is untouched.
- **Implication is between truths**: its operands are idea rows or
  relation rows, never concept rows. Modus ponens reads it.
- **Operator relations** are the verbs that take a truth as an
  argument: reporting (said), attitude (knows, doubts, hopes),
  conditional and similar scope-bearing predicates. `refs[0]` is the
  agent or antecedent row, the VP slot carries the verb atom, `refs[2]`
  is the truth's row. These are exactly the scoping operators of §4:
  the verbs that gate assertion are the verbs whose sentences cannot
  fuse. The lexicon marks a verb as an operator; the grammar's relative
  start for this kind is added alongside the part family (§3.3).

A relation row's three slots are `row, R, row`:

- `refs[0]` and `refs[2]` are the operand **rows**: concept rows, idea
  rows or relation rows, uniformly. Second order is licensed (§2).
- NP1 and NP2 carry the operands' fused vectors (copies of the referenced
  rows' NP1 slots) so vector readers (`consequents`, `evaluate`,
  recency) keep working without a join. A relation row has no fused
  point, so when an operand is a relation row its vector slot stays Null
  and the row-indexed reader form (§5) is the only reader for it. VP
  carries the relation predicate atom.
- Trust is the relation's own (§4), distinct from either operand's.

Deduplicate by `(rel_type, refs[0], VP atom, refs[2])`: writing an
existing relation again updates its trust and timestamp rather than
appending. For part and implies the VP atom is fixed by the kind.

### 3.3 The closing decides (decided)

At every S closing, sentence or clause:

1. Determine the start state the S reduces to. The classifier is the
   grammar (`sentence_relative_mask` over `TheGrammar.is_relative_rule`),
   extended in two ways: a **generic subject** (bare plural, indefinite
   generic) with any predicate parses under the part family with the
   predicate nominalised, while a token subject parses absolute; and a
   VP whose modifier NP is a reference to a relation row parses under
   the operator start, whatever the verb. This is a grammar and lexicon
   change with tests (§7), not a runtime heuristic over vectors. An
   unknown verb between two token NPs is an idea.
2. Absolute: fuse, write the idea row with its derivation, register the
   row. Inside a sentence the fused point is the enclosing VP's modifier.
3. Relative: resolve the two operands to rows (an embedded relative
   clause already has its row), write or update the relation row, push
   its reference if the S is embedded.

**One writer for relations.** The closing is the only relation writer. The
routing in `ConceptualSpace._route_learned_relation`, which sends
"reducible" relations into the WholeSpace META taxonomy
(`insert_relation`) and "ineffable" ones into a store, is deleted. The
distinction it served does not exist under §1: operands are rows and a
row exists for every S, so nothing is ineffable. The `truth_criterion`
learn-score gate is retired with it; rows are about reference, and trust
comes from provenance (§4).

### 3.4 META and the taxonomy (decided)

*Amended (Alec, 2026-09-29, §17):* the link between a word and its object
is a definition row, `word DEF object`, and no META is folded for it: "I
would not expect to create a fold just to unite word and object". What is
said below of the META as a concept, of its writer and of the META
bindings as a source of the taxonomy index is superseded by §17; the
direct index from a word to its object stands, and gains the reverse
direction.

The WholeSpace META taxonomy is vestigial and is retired. META is a
**concept-level** relation between the word-concept and the
object-concept: the meta-symbol is the generalisation over both, the
generalisation that gives symbolic understanding. META is **n-ary**: one
META node generalises over several words and several objects (synonymy and
polysemy), and the discrimination among its members happens at
interpretation time, from context, never at binding time (Alec,
2026-09-16; the binding table's one-row-per-word law is replaced by this;
detail in [FutureWork.md §5](../FutureWork.md#5-n-ary-meta-and-interpretation-time-discrimination)). Its one writer is the
existing word/object/meta path (`create_word_object_meta`, A = word,
B = object, C = meta), which this spec makes live. META nodes are created
only by word-object generalisation, never by a relation between two
objects.

The taxonomy readers (the decode reverse walks over parent and children,
the category-context builder, the autobind path) move to a concept-level
**taxonomy index** derived from two sources and rebuilt on load: the
part rows between concept rows (§3.2), and the META bindings. It is an
index, not a store; it holds no trust of its own. **Amended (Alec,
2026-09-22): that index is the concept store's own hierarchy of
higher-order concepts** — the taxonomy and the higher-order concepts are
one structure, read by union — and it has three writers: the closing's part
rows between concept rows, the META bindings, and discovery from witnessed
substitution — the same context on different occasions — for words and
percepts
([Architecture](../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)).
Trust stays on the LTM row; the edge carries none. *Amended (Alec,
2026-09-23, §1.1):* the row stores the evidence pair `(c⁺, c⁻)` and the
tetralemma tuple (t, f, both, neither) is derived from it at read time; the former
stored scalar `t − f` cannot hold **both**, which is a compositional fact.

*Noted (Alec, 2026-09-28).* An amendment written earlier the same day, which
separated the taxonomy from the order ladder on the ground that "cat and
animal are the same order", is withdrawn: "I was conflating order with part
of speech, which only sometimes correspond", and "the relation of the
taxonomy and the higher-order seems correct". The 2026-09-22 amendment above
stands. "Felix is a cat" and "cats are animals" are both taxonomic relations
and both can take part in the sigma fold; they differ in the parts of speech
of their terms, a proper noun and a count noun against two count nouns.
A set of discrete concepts is of a higher order than its members, "cats and
dogs, two discrete concepts, create animals, which is then necessarily of a
different order: a superset", while a part of one concept's extension,
"blue cats", keeps its order. Order is "determined by sigma or pi fold over
a class"
([accessible mind §2.0.1](2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention)).

**Word to object is a direct index.** Translation from a word to its
object never traverses the taxonomy. The word-keyed binding table
(`References`, `deref(word)`) is the fast path and stays the only one;
the taxonomy is for generalisation.

### 3.5 Object permanence: a word may translate to an earlier occurrence (decided, 2026-09-21)

Word forms may address concepts at several orders. The selected grammar
resolves particulars, names and pronouns at order 1, kinds at order 2, and
higher kinds at their symbolization depth. The canonical indexing and
resolution rule is in [Lexicon](../Lexicon.md#word-forms-and-concept-orders).


Translating a word to its object (§3.4) carries a **decision**: the object is
the type, as `deref(word)` gives it today, or it is a **token** — an earlier
noun, or an earlier sentence. In "The lion runs. The lion is tired." the
second *lion* is the lion that ran. Pronouns are the same decision with no
type of their own to fall back on.

It is **by reference**, the mechanism §2 already licenses ("an NP slot may
hold any row"); nothing new is stored:

- **An earlier noun in the same sentence** is the existing `bind`: the
  constituent reference to a live participant.
- **An earlier sentence** is that sentence's row. The second sentence's
  subject operand is the first row: `refs[0]` points to it, and the operand's
  vector is that row's fused point — the lion that ran, not the generic lion.
  An idea row has a fused point, so the referring sentence still fuses (§1).
  The word side still records *lion*, so reconstruction yields the words that
  were said; the derivation records the choice.
- **State is the chain; identity is imputed.** The latest row in a chain of
  such references is the individual's current state, and a second lion is a
  second chain. There is no referent table and no state-update rule. But a
  reference does not record a fact of sameness, because there is none to
  record: "identity has to be carried by expectation or prediction because
  (at least from a philosophical point of view) identity does not exist"
  (Alec, 2026-09-21). The reference records that the mind *took* the two as
  one.

**Who decides, and what carries it.** The grammar decides, at interpretation
time, as part of the n-ary META discrimination of §3.4: the candidates are
the type-level objects *and* the earlier occurrences. It is learned. No word
is wired to it — not "the", not "it" — since an operator has no predefined
surface. What *carries* an individual from one sentence to the next is the
predictor: each anchor in its situation
([accessible mind §2.7.3](2026-09-20-accessible-mind-subsystems.md#273-two-ways-in-what-is-still-active-and-what-is-cued))
is a standing prediction that an individual continues, and a word is tied to
an earlier occurrence when the situation expects one. Empty the situation and
the same words translate to their types. An anchored individual that fails to
recur leaves its expectation unmet — the surprise by which object permanence
is measured in infants.

**What keeps it honest.** Nothing in the input settles which individual a
word is about, so the input cannot correct a wrong imputation. Its
consequences can: what is *said* of the individual is composed from the input
alone, so a prediction that leaned on a wrong identity fails on content, and
that surprise is what revises the imputation.

**Bounds.** Candidates come from the recency buffer only: the live
constituents of the current sentence and the discourse chain of the last few
rows. `<compose>` gains no LTM access
([accessible mind §4](2026-09-20-accessible-mind-subsystems.md#4-which-grammar-may-touch-what));
an older individual becomes a candidate once a `what` has brought its frame
into STM. Identity is the one thing expectation contributes to composition;
what is said of the individual stays pure
([§2.6.3](2026-09-20-accessible-mind-subsystems.md#263-purity)). A row that a
later row refers to falls under the forgetting spec's reference rules like any
other referenced row.

**Why it matters beyond anaphora.** Words that share their contexts inside a
sentence — approximate antonyms such as *runs* and *does nothing* — differ in
what follows for the same individual: tired, or rested. With the chain in
place the expectation's source ideas carry the individual and what was done
to it, so consequences can tell such words apart. Without it the evidence is
there in the text and attaches to nothing.

## 4. Trust and scope (decided)

Rows are always written. Trust is decided at the top of the derivation
and propagated down through operator relations, under one rule: **the
sentence never sets its own trust.**

- The Cartesian default above (register at `0`, assert by provenance) is
  the `optimal` operation profile; the `human` profile asserts at the
  source's provenance trust on comprehension and unbelieves later. See
  [FutureWork.md §1](../FutureWork.md#1-operation-profile-optimal-versus-human).
- Trust magnitude comes from provenance (`origin`: conversation,
  provisioned, user, with the model's incoming-trust multiplier) and
  from later evidence. "It is certain that P" and "it is doubtful that P"
  leave P's trust untouched, because a liar can say either.
- An operator relation is a gate, not a value: it decides *whether* the
  embedded row is asserted at all. A bare assertion asserts. Reporting,
  conditionals and attitudes do not assert their argument; the embedded
  row is registered with trust `0` and only the operator row carries the
  observation's trust. A later direct assertion raises the embedded
  row's trust through the dedup path (§3.2). Forgetting or discounting
  the operator row can never raise the embedded row's trust: content
  does not inherit credibility from a decayed attribution.
- Negation is content, not modality: "cats are not animals" is asserted
  with the claimant's trust and negative sign. The liar's leverage is
  then exactly what it is for any plain assertion, which provenance
  already bounds.
- `origin` is unchanged and applies to all row kinds.

**Luminosity excludes relation rows.** `TruthLayer.sync_from_ltm` and
`_flatten_ltm_row` currently select provisioned and user rows of every
type and flatten them by the mean of live slots; under this contract they
select `REL_NONE` rows only. Relative truths never enter the coverage
measure (Philosophy.md); reasoning reads them through `relations()`.

## 5. Reading paths

- **Reconstruction and answer materialisation** read an idea row by
  replaying its derivation (the existing reverse chain and
  `_replay_program`); the `[B, 3, D]` factored idea the walk consumes is
  the replay's top three slots. A relation row is read as its three
  slots with operand rows resolved.
- **Reasoning** reads relation rows by `rel_type` and operand rows.
  `consequents` / `evaluate` keep their vector interface for concept and
  idea operands and gain a row-indexed form, which is the only form for
  relation-row operands. Parthood and subsumption over `REL_PARTOF`,
  modus ponens over `REL_IMPLIES`, substitution under equality as two
  part rows each weighted by its trust, attribution over
  `REL_OPERATOR` (who holds which truth, and whether it is asserted).
  Taxonomic subsumption remains inferential, never a `.where` meronomy.
- **Expectation** (the local-role objective) keeps its factored target,
  read as NP, V, modifier NP through the VP's derivation, with presence
  meaning "the VP is compound". Add one **kind** logit (idea / relation)
  so the expectation of a relative sentence is `[row, R, row]` and its
  discrepancy is scored against the relation row. Relation-to-relation
  expectation over rows is future work.
- **Recency** (`recent`) is over all rows; the discourse observation view
  is unaffected by this spec.

## 6. Migration (decided)

- The `refs` column rides the state_dict. A checkpoint without it loads
  with `-1` references.
- `REL_OTHER` conversation rows in an old checkpoint are dropped on load
  with a warning (training-time corpus, not user state). Provisioned rows
  are re-provisioned from their XML TruthSet, whose relation entries now
  resolve operands to rows and whose idea entries fuse. User rows follow
  the stateless rule already in force.
- The WholeSpace META taxonomy state is dropped on load; the concept-level
  index is rebuilt from rows and bindings.
- No optimizer state depends on the store.

## 7. Acceptance tests

Each test names the sentence, the rows written, and what must be readable.

1. **Idea, compound VP.** "the cat chased the mouse" → one `REL_NONE` row;
   replay yields `[cat, chase, mouse]`; no relation row; the row enters
   luminosity.
2. **Relation, generic subject.** "cats are animals" → one `REL_PARTOF`
   row with `refs` = concept rows *cat*, *animal*, excluded from
   luminosity; `consequents(cat)` yields *animal*; writing the sentence
   again appends nothing and updates trust.
3. **Token subject stays absolute.** "this cat is black", "the cat is an
   animal" → idea rows only.
4. **Recursive collapse.** "he said she said it's so beautiful" → one
   fused idea row for the sentence and one per embedded absolute S, no
   relation rows, no pushed references; replay reconstructs the full
   nesting.
5. **Forced reference makes the sentence relative.** "he said cats are
   animals" → `r = (cat, part-of, animal)` with trust `0`, and
   `s = (he, said, REF r)` as a `REL_OPERATOR` row with the observation's
   trust; no fused idea row exists for the sentence; a following "cats
   are animals" raises `r`'s trust; `r` is not asserted by `s` alone.
   "she said he said cats are animals" adds `t = (she, said, REF s)`.
6. **Second order.** "if cats are animals then cats breathe" → `r1`,
   `r2` (part rows over concept rows) and `r3 = (REF r1, implies,
   REF r2)`; `r3`'s vector slots are Null; the row-indexed reader
   resolves both operands; `r1` and `r2` have trust `0`.
7. **Equality as two part rows.** "the morning star is the evening star"
   → two `REL_PARTOF` rows in both directions with independent trust; no
   rows merged; substitution at read time weighted by each trust.
8. **Bare NP.** "cat" → the concept row is the idea (eternal); a located
   occurrence writes an idea row with the tense-only VP and a `.when`.
9. **Modality never sets trust.** "it is certain that cats are animals"
   and "it is doubtful that cats are animals" write the same part row
   with trust `0` under an operator row; "cats are not animals" asserts
   with negative sign and the claimant's trust; deleting or discounting
   the operator row leaves the part row's trust unchanged.
10. **Packed and single-sentence parity.** The packed drain and the
    non-packed sink write identical rows for the same sentences (ties to
    the plan's §11.1 and §11.4).
11. **Provisioning.** XML TruthSet ideas and relations map to the row
    kinds with resolved references; `store_truths` (user rows) likewise.
12. **Generic subject, not copula (parser).** From rules alone, with the
    same verbs on both sides: "cats breathe", "cats are animals", "a cat
    is an animal" parse relative; "this cat breathes", "the cat is black",
    "the cat is an animal" parse absolute.
13. **Clause-level closing and upward relativity.** Inside "he said cats
    are animals" the relative clause endings before the outer S, its
    reference is on the STM as the NP modifier of "said", the outer S is
    held at depth 3 and endings as `REL_OPERATOR`; inside "he said she
    said it's so beautiful" every clause fuses and the outer S endings at
    depth 1.
14. **META and translation.** *Superseded by tests 22 and 24 (§17.8).*
    A word bound to an object yields one META
    node in the concept-level index; `deref(word)` returns the object
    without touching the index; the WholeSpace taxonomy holds nothing.
15. **Expectation kind.** The kind logit trains on a corpus mixing ideas
    and relations; the existing sentence-expectation tests still pass.
16. **Migration.** An old checkpoint with `REL_OTHER` rows and WholeSpace
    taxonomy state loads with the warning, without those rows, and with
    the concept-level index rebuilt from rows and bindings.

Object permanence (§3.5). The choice is forced in tests 17–20; how well it
is *learned* is measured, not asserted.

17. **Same individual.** "The lion runs. The lion is tired." with the token
    choice: the second row's `refs[0]` is the first row and its subject
    operand equals the first row's fused point, not the type-level lion;
    reconstruction of the second sentence yields its own words.
18. **Pronoun.** A word with no type-level object resolves to an occurrence
    in the recency buffer, or fails explicitly; it never falls back silently
    to a type.
19. **Two individuals.** With the type choice on a second lion there is no
    reference to the first, and afterwards two chains coexist; a later
    reference reaches exactly one of them.
20. **Bounds and carrier.** Candidates are the current sentence's live
    constituents, the discourse chain, and frames a `what` brought into STM;
    resolution performs no LTM read. With the situation emptied, the same
    words translate to their types: identity is carried by the predictor's
    state, not by the rows.
21. **Consequences separate near-synonymous contexts (measurement).** On a
    small corpus where, for the same individual, *runs* is followed by
    *tired* and *rests* by *rested*, report the expectation residual and the
    separation of the two verbs' effects with references on and off; and,
    with a wrong identity forced, the rise in content surprise that follows.

## 8. Documentation required with implementation

- [Philosophy.md](../Philosophy.md) two-truths paragraph: idea = one fused
  point with a derivation; relation = three slots over rows; fusion fair
  iff every operand has a point; collapse versus reference; the generic
  subject; trust versus reference; parthood as the primitive with the
  Simons reference; the psychological grounding in §9.
- [Architecture.md](../Architecture.md) relation-table entry contract:
  the `refs` column, the three relation kinds, second-order operands,
  the concept-level taxonomy index, META as word-object generalisation.
- [STM.md](../STM.md) §11 and the LTM consolidation section: the
  clause-level closing, upward relativity, and the one write per S.
- [Reasoning.md](../Reasoning.md): reading relations by row; equality as
  two part rows; attribution over operator rows.
- The integrated plan: rewrite §2.1 to §2 of this spec; mark the §8.3
  compound-reference gap closed by definition and open by test; add this
  spec as a §10 milestone.
- Params.md: `truth_criterion` and any knob this retires; no new knob is
  expected.
- Code comments state technique only; the philosophy lives here and in
  Philosophy.md.

## 9. Decisions record (2026-09-16)

The first draft left four questions open. Alec's answers, and one
further decision, are folded into the sections above:

1. **Relations over relations: licensed.** "If cats are animals then cats
   breathe" is three relation rows, hierarchical not chained. Relation
   rows are operands like any other row; they have no fused point, so
   their readers are row-indexed. A sentence never sets its own trust;
   modality is gated, not valued, because liars exist.
2. **WholeSpace META taxonomy: vestigial, retired.** META is the
   concept-level word/object generalisation with one writer; the taxonomy
   is a derived index; word-to-object translation is the direct binding
   index. The tetralemma tuple is computation-time only (superseded
   2026-09-23 by §1.1: the row stores the pair); the learn-score gate is
   retired.
3. **No REL_EQUAL.** Improper parthood is the primitive (Simons 1987);
   whole is part with swapped operands; equality is two part rows with
   independent trust; subsumption between concepts is parthood;
   implication is between truths.
4. **Fusion exists; the closing moves.** The reduce sweep's depth-1 fold is
   the fusion. The open mechanism was the clause-level closing, decided as
   grammar: `NP → S` fuses an absolute clause, `NP → REF(S)` references a
   relative one, giving a one-to-one correspondence between grammatical S
   and rows.
5. **Fusing over a reference is not a fair operation.** A sentence that
   references a relation stays three slots and is itself a relation of
   the operator kind, so relativity propagates upward. Mechanically, a
   relation row has no point to fuse. Psychologically, attribution and
   content are separate traces that decay separately: source monitoring
   (Johnson, Hashtroudi and Lindsay 1993) shows content retained while
   its source is lost; the sleeper effect (Hovland and Weiss 1951;
   Kumkale and Albarracín 2004) shows a discredited source's message
   gaining force as the source memory decays faster than the content,
   which is only possible with separate traces and is also how liars
   win; propositional text memory (Kintsch and van Dijk 1978) stores an
   embedded proposition as an argument of the higher one, `SAY[he, P]`;
   situation models (Zwaan and Radvansky 1998) are the fused level and
   hold for absolute embedding, where a chain of sayings about one
   thing is one scene. The trust rule in §4 is designed so the model
   does not suffer the sleeper effect: forgetting the attribution row
   cannot raise the content's trust.
6. **Profile, forgetting, n-ary META (2026-09-16, from the psychological
   grounding review).** Optimal versus human operation is a `model.xml`
   profile; forgetting is urgent because every S is a row and the store
   has a capacity wall; META generalises over n concepts with
   interpretation-time discrimination; the reconstruction trace will be
   partially dropped so inversion is learned, not exact. All four are
   specified in [FutureWork.md](../FutureWork.md) and are not part of the
   Codex implementation of this spec, except that the store must not
   assume one word per META.
7. **Object permanence is by reference (2026-09-21).** Translating a word to
   its object may tie it to an earlier noun or an earlier sentence; state is
   the chain of such references and no referent table is added. Identity is
   imputed, not stored: the predictor carries it, and content surprise
   corrects it (§3.5).

## 10. Review round 1 (Claude, 2026-09-28, on the incomplete item 7 candidate)

Reviewed: the uncommitted tree on `1678ee1f`, receipt
[2026-09-27-item7](../benchmarks/2026-09-27-item7/README.md), source digest
`e0b217e0…`. Sweep: 5,016 cases, 4,611 passed, 320 skipped, 84 failed, one
expected failure. **Not accepted.** The clause closing, journal, concept index,
reference context and the 28 named §7 mechanism cases are sound as mechanisms
with forced derivations; the failures are integration, and three of the
families need a decision before more code is written.

**A. Two sentence boundaries exist; only one owns a clause (37 failures, and
the MM gate).** Probe: on the per-word eager boundary in
`_forward_body_per_word`, `_append_observed_meaning` is called with
`clause=None, program=None`; the store has its model. Raw `forward` and every
path where `_sentence_ends` is false never capture a program, so no row can
be ended. *Proposed:* the 7.5 closing driver is the only sentence boundary.
Raw forward and evaluation route through it (exploit only, no optimizer, as
`sentence_pair(training=False)` already does) and the per-word boundary
writer is deleted, not given a fallback.

**B. Autobind now requires learned primitive properties, which only
`propertyBasis` configurations have (13 failures, including
`test_fineweb_preflight` and the two-epoch graph-release gate).** Item 7
correctly deleted the legacy word/META WholeSpace autobind, leaving
`_autobind_property_concepts` as the one path; configurations without
`propertyBasis` now raise at the first Reset. *Proposed:* every WholeSpace is
a property basis — which is what "wholes are types" and §3.4 already say —
and the `<propertyBasis>` element is retired. This changes the behaviour of
configurations that omit the element, the protected XOR_grammar fixture
among them; that is a consequence of retiring WholeSpace word rows by
decision, not a change made to pass its gates, and the receipt must say so.

**C. Provisioning raises when the chooser's clause disagrees with the
TruthSet's declared kind (3 failures; the depth-3 campaign now dies in
provisioning).** A provisioned truth would then depend on an untrained
chooser. *Proposed:* a declared kind is a supplied grammar annotation. It
masks the start states that disagree in the closing's softmax, the same masking
the deadline uses, and trains the chooser as a grammar lesson. A declaration
the grammar cannot derive at all remains an error.

**D. Reconstruction regressed and must be explained before acceptance.**
Serial reconstruction is .1224 / .1192 / .1237 before, during and after
training against 7.5's .1176 / .1100 / .0989: training no longer improves
it. Packed/single mean byte cost is .384 against .180. Strict parity is lost
on one sentence by 2.4e-8. Bisect by component on the fixed protocol — the
identity straight-through factor, the situation context (its three weights
are zero by default and must then be exact no-ops), `interpret` order
resolution, the clause journal — and report which one moves the numbers. No
re-baseline until the cause is named. The parity difference is a
batch-layout dependence; find the operation, do not round it.

**E. Stale fixtures (about 20 failures): port, do not waive.** The category
codebook tests call the WholeSpace copy of an API whose owner is
ConceptualSpace; the thought-catalog test expects `true` to be absent; the
relative-end-state and role-collapse fixtures expect the previous row
contract. Each is updated to the item 7 contract with its assertion intact.

**F. For Alec's review, methods removed beyond the spec's list.** The spec
names `_route_learned_relation`, `insert_relation`, the learn-score gate,
the META taxonomy, `conceptualize_chain` and the JOINT concept. Also removed:
`order_ladder`, `apply_definition_constraint` (WholeSpace copy),
`record_lbg_pull`, `maybe_split_lbg`, `property_class_whole`,
`record_cross_tower_meronomy`, `_autobind_cross_tower`. None has a remaining
caller. They stay deleted unless Alec objects. Stale `truthCriterion`
comments remain in Models.py, Layers.py and Mereology.py and are to be
removed (comments state technique).

**G. Now visible, keep visible.** With WholeSpace word rows gone, both
XOR_grammar CLI gates get past the six-row capacity error and reach their
measurements: class accuracy 0.0 against ≥ .5, and 0/4 recovered against
≥ 50%. These are the first real measurements of those gates in several
landings. No tuning.

**Order of work after Alec's decisions on A, B and C:** A, B, C; then E;
then D's bisect and fix; rerun the explicit gates and the depth-3 campaign
(it must reach its measurement again); one source-matched full sweep; stop
for review.

## 11. Decisions (Alec, 2026-09-28) — these supersede §3.1 "Derivation", §5's reading by replay, and §10 finding A

### 11.1 A row holds structure, never a derivation (decided)

There are two grammatical start states and they are all a row needs:

| Start state | Slots | Reducible further? |
|---|---:|---|
| `S` | 1 | yes — `S` goes to `NP VP` and on from there |
| `S REL S` | 3 | no — which is why it keeps three slots |

What is stored is what the slots hold when the sentence ends: one slot or
three, each slot's content and its reference, the row's `.where` and
`.when`, and the evidence pair. **No derivation is stored, replayed or
required.** A new sentence is generated from the stored structure and the
grammar it entails, starting from `S` or `S REL S`.

Consequences, stated so they are chosen and not discovered:

- The record of which operations were chosen exists only while the sentence
  is being read. Training uses it there — the reconstruction objective
  inverts the chosen operations, and the explore derivation needs it to
  force one deviation — and it is discarded when the sentence ends.
- The row writer takes the end state, not a program. The row's kind is the
  slot count. A relation's kind (part, implies, operator) is read from the
  identity held in the REL slot. An idea row's two cached operand references
  (§3.1 `refs`) are taken from the state just before fusion; they are a
  cache, as before.
- The expectation's factored target `[NP, V, NP]` is the state just before
  fusion, taken while the sentence is still open. It is not recovered from
  the row.
- Reading a row back into words is **generation**, not replay. Item 6
  (stored-idea generativity) measures exactly this and today reports weak
  recovery; this decision makes item 6 load-bearing, and the forgetting
  spec's "drop the derivation before the row" step becomes vacuous.
- `clause_from_program`, the stored `derivation` field, its checkpoint
  state, and the writer's "requires its selected clause" error are removed.
- **The retrieval index's terms are derived by unfolding the row** (Alec,
  2026-09-28), not recorded as a list of the words the sentence contained.
  The unfolding is an educated guess made easy by activation: the symbols
  the sentence used are highly active when its row is written, so the
  choice of words is limited to a few candidates. This is the path
  accessible mind §2.7.3 already gives a row that has no derivation, now
  the only path. **The activation read is the symbolic activation that is
  the source of semantic priming** (accessible mind §2.5) — the one the
  model already holds. There is no separate store of the words used per
  sentence, under any name: no per-sentence word list, leaf list or
  activation snapshot is written beside the row.

### 11.2 Finding A, restated

The per-word boundary failed because the writer demanded a program. Under
§11.1 the writer needs only the end state, which that boundary has. Claude
still recommends one implementation of "a sentence ends" (no legacy paths),
with raw forward and evaluation running the single exploit derivation; this
is now a simplification, not a precondition.

### 11.3 Vocabulary (decided)

The retired sentence-boundary vocabulary is removed from identifiers,
comments, docstrings, configuration comments, tests, and the live documents
under `doc/` and `todo.md`. Receipts under `doc/benchmarks/` and commit
messages are historical records and are not rewritten. Sections 1–10 of this
document predate the decision and are reworded in the same pass.

The sentence (clause) **ends** and its **row is written**. Use **end state**
and **end root**, and **closing rounds**, **closing budget** and **closing
width**. The module, mixin and writer are `ClauseRow.py`, `ClauseRows` and
`write_clause`; the driver flag is `_sentence_ends`. The original mapping
and all additional identifier substitutions are preserved in the
[mechanical rename record](../benchmarks/2026-09-28-item7-review/rename/mapping.json).

Codex applies it as one mechanical change with no behaviour change, verified
by an unchanged sweep outcome, and records any identifier the table does not
cover. About 300 occurrences in 17 runtime files, 185 in 37 test files, and
300 in 30 live documents.

### 11.4 WholeSpace is the property basis, and nothing else (decided, Alec, 2026-09-28)

Wholes are properties: cuts in a set. That is the only form in which
WholeSpace exists. Word-concepts and object-concepts live in
ConceptualSpace. This resolves §10 finding B.

- Every WholeSpace creates its learned primitive properties. The
  `<propertyBasis>` element is deleted from the schema, from `model.xml` and
  from the thirteen configurations that set it; a configuration that still
  carries it fails to load with a message naming this section.
- The symbol-dictionary form of WholeSpace — word rows, META rows, and every
  branch that tests `property_basis` — is deleted, not gated.
- Sixty-one of the seventy-four shipped configurations inherited the old
  default and change behaviour, among them `MM_xor`, `MM_query_reasoning`
  (the depth-3 campaign), `MM_20M_fineweb` (item 0's run) and the protected
  `XOR_grammar` fixture. The fixture's file is not edited; the form it
  inherits is. This follows from retiring the dictionary form by decision,
  not from a wish to move its gates, and the receipt says so. Gate outcomes
  are recorded whatever they are.
- A checkpoint written under the dictionary form drops its WholeSpace word
  and META rows on load, with a warning, and its properties start from
  their a-priori examples; §6's rule for the META taxonomy state is extended
  to cover this.
- Mechanism tests: a configuration with no mention of the element builds a
  property WholeSpace and binds a new word at its first sentence boundary;
  the element's presence is a load error; no `property_basis` attribute or
  branch remains; an old checkpoint loads with the warning and no word rows
  in WholeSpace.

### 11.5 A given truth is plain text with a trust value (decided, Alec, 2026-09-28)

Given truths come from WikiOracle's TruthSet. Each is a sentence in plain
English with a trust value, and nothing says out of band what kind of
sentence it is. This resolves §10 finding C.

- The `kind` attribute of `<truth>` is removed from the schema and from the
  eight entries that carry it (`MM_query_reasoning`, `MM_qa`, and two
  consolidation fixtures). `kind_map`, the `relations` argument of
  `_ltm_ingest_truth_texts`, the `relation` field of the assertion context,
  and the error "TruthSet kind disagrees with the selected grammatical
  clause" are deleted. A configuration that still carries the attribute
  fails to load.
- The grammar alone decides whether a given truth is an idea or a relation
  (§3.3), exactly as for any other sentence. What provenance supplies is
  trust, origin and the source text (§4), nothing about structure.
- **Provisioning waits for training.** An untrained chooser cannot be
  expected to read a given truth in the right form, so provisioning before
  training is not a supported way to obtain relation rows. Tests that
  provision an untrained model and then require relation rows — the depth-3
  campaign, the reasoning fixtures — are learning results: they move under
  the mature-checkpoint prerequisite (item 9), or they are kept as mechanism
  fixtures with an explicitly forced derivation and say so. Neither is a
  waiver; the depth-3 assertion is kept as written.
- **The shipped texts are reworded into plain English** (decided, Alec,
  2026-09-28). They used an operator spelling as a corpus token.

  | Was | Now | Configurations |
  |---|---|---|
  | `socrates partOf human` | `socrates is a human` | `MM_query_reasoning` |
  | `human partOf mortal` | `humans are mortal` | `MM_query_reasoning` |
  | `paw partOf cat` | `a paw is part of a cat` | `MM_qa`, the three consolidation fixtures |

  `fire causes smoke` and `cat` are already plain and stay. The fixtures'
  lower-case convention is kept. Tests that quote the old texts
  (`test_reasoning_cde_model.py` among them) and the comment in
  `MM_query_reasoning.xml` that calls `partOf` a technical corpus token are
  updated with them. The grammars' `<part>partOf, partof, part-of</part>`
  spellings are left alone: they are corpus tokens, not meanings.
- **No disambiguation machinery now** (Alec): something may later be needed
  to choose between parsings of one sentence; it is not built here. One
  known case is recorded so it is not mistaken for a defect: by §3.3 a
  token subject parses absolute, so "socrates is a human" may be read as an
  idea while "humans are mortal", with its generic subject, is read as a
  part relation. Whether the syllogism then chains is a learning result.

## 12. Hand-off to Codex (Claude, 2026-09-28): what to change in the item 7 candidate

The candidate is not accepted. Work from this list, in this order. §11 holds
the decisions and their reasons; §10 holds the findings. Nothing is committed
until Claude has reviewed the result.

1. **Vocabulary first, alone (§11.3).** Remove the retired sentence-boundary vocabulary from
   identifiers, comments, docstrings, configuration comments, tests and the
   live documents, by the mapping table. One mechanical change with no
   change in behaviour: the sweep outcome before and after must be
   identical, failure for failure. Receipts under `doc/benchmarks/` and
   commit messages are not rewritten. Doing this first keeps every later
   diff readable.
2. **A row holds structure, never a derivation (§11.1).** The row writer
   takes the end state: one slot or three, each slot's content and
   reference, `.where`, `.when`, the evidence pair. Kind is the slot count;
   a relation's kind is read from the identity in the REL slot. Delete
   `clause_from_program`, the stored `derivation` field and its checkpoint
   state, and the "requires its selected clause" error. The record of
   chosen operations lives only while the sentence is open, for the
   reconstruction objective and the explore deviation. The expectation's
   factored target and an idea row's two cached operand references are
   taken from the state just before fusion.
3. **Index terms by unfolding (§11.1).** Derive the retrieval index's terms
   by unfolding the row, with candidates limited by the symbolic activation
   that drives semantic priming. No per-sentence word list, leaf list or
   activation snapshot is stored.
4. **One sentence boundary (§11.2).** Raw forward and evaluation end a
   sentence through the same driver as training, running the single exploit
   derivation with no optimizer. Delete the per-word boundary writer.
5. **WholeSpace is the property basis (§11.4).** Delete the
   `<propertyBasis>` element, the dictionary form and every
   `property_basis` branch. Old checkpoints drop their WholeSpace word and
   META rows on load with a warning. State in the receipt that sixty-one
   configurations change behaviour by decision.
6. **Given truths are plain text with trust (§11.5).** Delete the `kind`
   attribute, `kind_map`, the `relations` argument, the assertion context's
   `relation` field and the mismatch error. Apply the three rewordings.
   Move the depth-3 campaign and the reasoning fixtures that provision an
   untrained model under item 9's prerequisite, or keep them as mechanism
   fixtures with a forced derivation that says so; the depth-3 assertion
   stays as written.
7. **Port the stale fixtures (§10 E)** with their assertions intact, and
   remove the stale `truthCriterion` comments (§10 F).
8. **Explain the reconstruction regression (§10 D).** After steps 1–7,
   reissue the fixed serial and packed/single measurements. If
   reconstruction still fails to improve with training, or parity is not
   exact, bisect by component — the identity factor, the situation context
   (its three weights are zero by default and must be exact no-ops),
   `interpret` order resolution, the clause journal — and name the cause.
   No re-baseline until it is named.
9. **Then** the explicit gates, one source-matched full sweep, and stop for
   review. XOR_grammar's two gates now reach their measurements; record
   what they measure. No seed is pinned to pass and no threshold moves.

Not in this pass: a mechanism to disambiguate parsings; the nonzero training
temperature and the parsimony term deferred from 7.5.

## 13. Review round 2 (Claude, 2026-09-28, on the candidate after §12)

The candidate is **not accepted**. Its one source-matched sweep is red:
4,675 passed, 30 failed, 322 skipped, one expected failure, and 23 of the 30
passed before. The receipt is complete and honest about this
([receipt](../benchmarks/2026-09-28-item7-review/README.md),
[ledger](../benchmarks/2026-09-28-item7-review/closing-sweep/failure-ledger.json)).

**What was checked and stands.** The row keeps scalar source trust beside
the evidence pair, joins repeated evidence by the maximum of each pole, and
carries its order; a row without a stamp stays unknown. No `test_item7_*`
file sets a seed. `true` is a registered operator. The vocabulary change
left every case's outcome as it was. The reconstruction regression and the
parity failure each have a named cause.

**How the findings were reached.** Each failing case was read against the
candidate's source, and the cheap ones were run alone. *Measured* means a
probe was run and its numbers are given; *read* means the cause is what the
code says and is probable, not shown. A first version of this section
called three of these cases broken behaviour; two of them turned out to be
expectations that §11 itself makes stale, and the text below replaces it.

### 13.1 Findings, with the probable fix for each

* **A. Closed (Alec, 2026-09-28): the candidate's taxonomy rule stands.**
  Asked whether `cat` and `animal` are of one order once the model has
  been told that cats are animals: "No; cats and dogs, two discrete
  concepts, create animals, which is then necessarily of a different
  order: a superset." `index_part_row`, `ClauseTaxonomyPlan` and
  `test_part_testimony_symbols_the_parent_one_order_above_the_object` are
  right as they are. This letter was opened, withdrawn and held in the
  course of one day; nothing is to be changed under it.
* **B. Two configurations derive no word-level concepts, so they have no
  symbols** *(measured)*. "The symbolic state is the activation level of
  all symbols, pos and neg", and "all concepts get symbols" (Alec,
  2026-09-28). In `MM_xor` (parallel) and in `XOR_grammar` (serial) no
  concept is derived for any word: there are no concept identities, no
  concept activations (`_concept_activations` is `None`), and the symbolic
  state is exactly zero in every entry, while the parts and the conceptual
  state are not. Both run the default `conceptBinding`, `mixing`, which
  combines the two towers through a learned matrix and gives a conceptual
  state without identities; only `aligned` derives a concept for each
  word. They had their word symbols from WholeSpace's dictionary, which
  §11.4 removed. What is expected of them (Alec): "Those cases should have
  derived word-level concepts, and then had the grammatical option to
  create disjunction, and then to create conjunction of those." Two
  symptoms are certain: in `MM_xor` the parallel path "keeps `symbols`" as
  its answer seed (`_capture_understanding`), so the answer is zero before
  and after the update in
  `test_dedicated_synthesis_operators_start_at_identity_and_are_answer_only`;
  and `test_forward_keeps_continuous_symbols` reads the property tower as
  if it were the symbolic activation. XOR_grammar's constant answer is not
  explained by this alone, since its answer seed is the conceptual state,
  but its grammar has no symbols to disjoin or conjoin. *Fix (decided,
  Alec, 2026-09-28):* "Derive a concept for every word. Basically
  temporarily force interpret(), since a word is also an object, and not
  all objects should be interpreted (except in reading mode)." So every
  word of these configurations passes through `interpret`, whatever the
  binding, and `XOR_grammar.xml` is not changed. The forcing is temporary
  and is marked so where it is made: in general an object is interpreted
  only when reading.
* **C. Raw forward reaches compiled reconstruction with no grad anchor**
  *(read)*. `_ensure_grad_anchors` is called by `enable_compiled_step` and
  by `_run_batch_once` and by nothing else, and step 4 of §12 sent raw
  forward through the training driver, which compiles reconstruction.
  `test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks`
  calls `forward` directly. *Fix:* ensure the anchors where the sentence
  driver starts.
* **D. The compiled word step is specialised on `requires_grad`** *(read,
  probable)*. Each trial starts from the restored state, which is detached,
  and every later word's carry requires grad, so the step compiles for
  both; with the context pass and two trials the variants pass the limit of
  eight in `test_native_interleave_…[True]`. *Probable fix:* pass the
  restored state through `_carries_with_grad` at the start of each trial,
  so that under grad every call sees the same `requires_grad`.
* **E. The gradient report lost its output column** *(read)*. Before, the
  grammar lesson was computed at the end of the batch and its `generate`
  term was added to `gradient_objectives["output"]`. The lesson is now
  trained inside the sentence, in `_sentence_path_cost`, and only a
  detached total is reported at the end of the batch, so nothing feeds the
  output column, and
  `test_normal_batch_logs_named_shared_operator_gradients` — which has no
  supervised outputs, so the lesson was its only output objective — finds
  every `output_norm` zero. *Fix:* where the lesson is added to the
  sentence's cost, add its `generate` term to the sentence's gradient
  objectives under `output`, as the batch-end code did.
* **F. `test_pair_trains_twice_and_commits_once` is a stale expectation,
  not a broken commit** *(measured)*. The rows `select_rows` returns are
  exactly the rows the test expects. `_commit_sentence` then clears the
  record of chosen operations, every entry `−1`, which is what §11.1
  requires: the record lives only while the sentence is open. *Fix the
  test:* assert commit-once on the end state, the committed slots equal to
  the selected trial's, and assert that the record is empty after the
  closing.
* **G. `store_truths` stores its rows, and the view counts poles**
  *(measured)*. Two rows of user origin, trust 0.9 and 0.4, evidence
  `(0, 0)`, order 0; `TruthLayer.count` is 0 because the view admits a row
  by its poles and these have none. Nothing is lost, and by Alec's answer
  to question 2 nothing is wrong: truth cannot be expressed in a model that
  is not yet trained. *Fix the two tests* as mechanism tests, saying so:
  count the stored rows of user origin, check their trust, and assert
  *neither*. They claim nothing about what the rows mean.
* **H. `test_shared_operation_writes_the_caller_owned_stack` depends on the
  seed** *(measured)*. The chooser is untrained and nearly uniform, its
  winner at probability about 0.26. With seeds 0, 2 and 3 it chooses the
  unary `not` and the depth stays two; with 1, 4, 5, 6 and 7 it chooses a
  binary fold. The case passes alone, in its file and in its batch, and
  failed in the sweep on the state a worker had reached. *Fix the premise,
  not the seed.* What is required is that an absolute sentence collapse to
  a point by its closing (question 3), not that any one step reduce.
  Assert that: run the rounds to the deadline and check that the sequence
  fits its row, for whatever the chooser picks on the way. A single forced
  fold may be kept beside it as a check of the operation itself.
* **I. Fifteen tests still assert WholeSpace's retired roles.** Most of
  them are tests of the symbolic space that still reach it as `wholeSpace`;
  several are named `test_symbolic_space_*`. *Fix:* point them at the
  symbolic space. Where it no longer has what they assert — a `references`
  parameter, an `order` buffer, a regularizer factory, a lexicon API — the
  case is retired with the record, since a symbol's identity is its concept
  reference and allocates no row. The continuous-symbols assertion is
  ported, not retired. Reasoning methods are not removed without Alec's
  review.
* **J. The named cause of the reconstruction regression is a sensitivity.**
  Which initial atom an identity receives decides the result, because code
  positions are not trained. Measure seeds 0, 1 and 2 at HEAD and at the
  candidate, report all six, and re-baseline to that statistic only if the
  candidate lies inside the spread. This is reproducibility of a
  measurement, not a seed chosen to pass.
* **K. Fix the cause of the parity failure, not the tolerance.** Score the
  compacted candidates, so that the same logits occupy the same columns.
  The strict assertion stays.
* **L. The two-epoch graph release exceeds its 8 GiB guard** (8.47 GiB).
  Record whether HEAD stays under it. The guard is not raised.
* **M. Of the six that failed at HEAD, two now fail for another reason.**
  `test_tiny_canonical_detached_reverse_stops_at_root` raised "an
  observation requires its selected clause and shared index" and now fails
  an assertion on values; `test_training_step_uses_the_normal_policy_configuration`
  raised "TruthSet kind disagrees …" and now raises an `IndexError`, index 1
  in a tensor of one row. Name both. The other four are as they were.

### 13.2 Questions for Alec, and his answers (2026-09-28)

1. **The symbolic state.** "The symbolic state is the activation level of
   all symbols, pos and neg", and "we have adopted a convenience that all
   concepts get symbols". So no configuration is without symbols, and an
   earlier version of this question, which offered that as a choice, was
   ill formed. The two configurations of §13 B "should have derived
   word-level concepts, and then had the grammatical option to create
   disjunction, and then to create conjunction of those". **Decided:**
   "Derive a concept for every word. Basically temporarily force
   interpret(), since a word is also an object, and not all objects should
   be interpreted (except in reading mode)." A concept is derived for every
   word under every binding; `mixing` is not retired by this and
   `XOR_grammar.xml` stays as it is. *Open, and not blocking:* what ends
   the forcing. Claude's reading is that it ends when reading mode decides
   for itself when a word is interpreted (item 6.8).
2. **A given truth in an untrained model.** "We have to wait until a model
   is trained before we can express truth in that model, before LTM is
   meaningful in a consistent way. This suggests stages of learning, and
   perhaps they can act like gates. Object permanence has to be learned
   before objects. Words have to be learned as concepts before we can
   interpret them as references." So the stored rows are inert, and the
   tests of §13 G are mechanism tests. The stages are recorded as a
   proposal in
   [FutureWork](../FutureWork.md#stages-of-learning-as-gates-proposed-2026-09-28).
3. **Reduction in a single step.** "This relates to 2; I think we had said
   that all ultimate truths must collapse to a point in conceptual space;
   is that related?" It is, and it answers the question. The collapse is
   required of the sentence at its closing: an absolute S ends as one point
   in one slot (§1, §3.1), which is why STOP is not a candidate while the
   sequence is longer than its row. It is not required of any single step,
   so an untrained chooser may choose a unary on the way. It relates to 2
   in that the collapse is mechanical from the first day and means
   something only once the operators are trained.

## 14. Hand-off to Codex (Claude, 2026-09-28): what to change after review round 2

Work from this list, in this order. Every fix has a failing probe before
it. Nothing is committed until Claude has reviewed the result.

1. *Closed with §13 A.* The taxonomy is right as the candidate has it;
   there is nothing to do.
2. **Grad anchors and carries (§13 C, D).**
3. **The gradient report's output column (§13 E).**
4. **The ports: the pair driver (F), `store_truths` (G), the shared
   operation (H), the fifteen (I).** G and H as Alec's answers to questions
   2 and 3 have them.
5. **Word-level concepts in `MM_xor` and `XOR_grammar` (§13 B).** Derive a
   concept for every word by forcing `interpret` on it, under every
   binding. Mark the forcing temporary where it is made and in `todo.md`.
   Leave `XOR_grammar.xml` byte-identical. Then rerun XOR_grammar's gates
   and record what they measure; they are not expected to pass on this
   alone.
6. **Parity (K), the three-seed reconstruction measurement (J), the memory
   guard (L), and the two renamed causes (M).**
7. **Then** one source-matched full sweep, and stop for review.

Not in this pass: the `NonLayer` and `ConjunctionLayer` defects, the
modifier's selection among a head's taxonomic children, and everything else
in [accessible mind §2.0.1](2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention),
which is documentation of decisions and changes no code in item 7.

## 15. Review round 3 (Claude, 2026-09-29, on the candidate after §14)

The candidate is **not accepted**. Its sweep accounts for 5,026 cases:
4,678 passed, 22 failed, 322 skipped, one expected failure, and three
stopped by the memory guard
([receipt](../benchmarks/2026-09-28-item7-review-round2/README.md),
[ledger](../benchmarks/2026-09-28-item7-review-round2/full-sweep/failure-ledger.json)).
Sixteen cases that passed before now fail, and nine of them are proofs of
XOR. The receipt is complete and honest about what it ran.

**What was checked and stands.** Steps 2, 3, 4 and 6 of §14: thirteen
earlier failures pass, all 134 item 7 cases pass, parity is exact for every
sentence, the three seeds were measured in both trees and nothing was
re-baselined, and the eleven retired cases are recorded with their bodies.
No `test_item7_*` file sets a seed. `XOR_grammar.xml` and `MM_xor.xml` are
byte-identical to HEAD. No tolerance or threshold moved in the files this
pass changed (read by diff; the renamed cases of
`test_subspace_what_stm_contract.py` were read by name, not line by line).

**How the findings were reached.** Every measurement below was run on
2026-09-29, without a seed, on the candidate and on a clean archive of HEAD
`1678ee1f`. A *probe patch* replaces one method for the length of one test,
from a plugin outside the tree. Nothing in the repository was edited to
obtain any number here, and no patch is proposed as the final code.
*Measured* means a probe was run and its numbers are given; *read* means
the cause is what the code says.

### 15.1 The status of XOR

Alec, 2026-09-29: "it's a basic proof of nonlinear learning, and I don't
want it to regress."

| Proof | In the sweep | HEAD `1678ee1f` | Candidate | Candidate with the two probe patches of 15.3 |
|---|---|---|---|---|
| Grounded XOR, `test_grounded_xor.py`, six cases, exact (error below 1e-6) | yes | 6 pass | **6 fail**: no case is discovered (`0 == 4`) | 6 pass |
| XOR_exact through the lesson curriculum, `test_concept_output.py`, three cases | yes | 3 pass | **3 fail**: eleven case rows, four expected | 3 pass |
| XOR_exact from the command line, output crisp (error below .05) | no, marked `slow` | **crash** | **fail**: error 0.248, predictions .55 .58 .58 .58 | pass: error 0.0, predictions 0 1 1 0, three runs of three |
| XOR_exact from the command line, inputs reconstruct | no, marked `slow` | **crash** | pass, 4 of 4 | pass, 4 of 4 |
| MM_xor, `test_convergence` and `test_learns_xor_signal` | no, marked `slow` | pass; predictions −.01 .99 .99 −.01 | pass; predictions −.01 .99 .99 −.01 | pass |
| MM_grammar, `test_mm_grammar_learns_xor_signal` | explicit gates | pass | pass (U) | pass |
| XOR_grammar, class accuracy and reconstruction | explicit gates | fail: crash, "symbol codebook capacity exhausted" | fail: runs to the end; class 0 accuracy 0.0, 0 of 4 recovered | fail: error 0.2499, every prediction .49 to .50 |

Three things follow.

1. **This candidate regressed XOR**: nine cases that pass at HEAD fail.
   One function is the cause (N, O), and the instruction it implements
   was Claude's.
2. **XOR_exact was already broken at HEAD, and no receipt had measured
   it** (Q). It has crashed since the item 9b landing. The candidate does
   not crash, and does not learn.
3. **Both are repaired by two changes** (15.3), shown by probe: every row
   above except XOR_grammar passes, and XOR_exact learns XOR exactly.

Two tests prove less than their names say. `test_learns_xor_signal` asks
for an error below .26, and a constant answer of one half has error .25;
the row for MM_xor rests on `test_convergence` and on the predictions
measured here. `TestSPNN::test_xor_training` checks that a classical
network does not crash, and is not in the table.

### 15.2 Findings, with the probable fix for each

Letters continue from §13.

* **N. One function accounts for the sixteen regressions and the three
  new memory stops** *(measured)*. `_stage_reading_word_concepts` forces
  `interpret`'s word admission on every word of every reading except the
  canonical aligned serial one. With that one method replaced by a no-op,
  the fourteen cases of the first six groups of the receipt's table pass,
  14 of 14. The two compile-cache cases pass alone, 5 of 5 in their file;
  they failed as peers of a worker that was stopped for memory. Peak
  resident memory of the category codebook case falls from 10.80 GiB to
  0.79, and of the 64-word trace from 13.17 to 0.98 (HEAD: 0.52 and
  0.65). **The scope was Claude's error, not Codex's.** §13.2 (1) and §14
  item 5 said "under every binding". Alec's decision was for the two
  configurations of §13 B: "Those cases should have derived word-level
  concepts".
* **O. Under the aligned binding the forcing shuts out the model's own
  discovery** *(measured)*. The grounded models derive a concept for a
  word themselves: a fused unit that recurs is given a provisional row,
  and the row is promoted by use. `_observe_native_cases` gives the row
  only if no row already has that unit as its positive part. The forcing
  writes the word's row at the input boundary, before the field is read,
  so the unit is always taken and no case is ever assigned. XOR's union
  then has no cases to read. This is eleven of the regressions: grounded
  XOR (6), the curriculum (3), the attended-field reuse (a second concept
  for one word) and the membership boundary (nine rows for eight).
  *Fix, under question 1 of 15.4:* the forcing stands down under the
  aligned binding, which has its own derivation in both modes. Measured
  with that change alone: 11 of 11 pass, and the four cases of
  `test_item7_word_interpretation.py` still pass.
* **P. A word that is read again is not recognised as itself**
  *(measured; at HEAD since item 9b)*. `_populate_concept_weights` mints
  a "percept alternative" whenever the witness differs from the stored
  definition, and it differs for two reasons that are not differences of
  the word.
  1. *The parts have fused.* `hello` is stored as the fused unit 9. It is
     witnessed as its five byte parts `(1, 2, 3, 3, 4)`, whose canonical
     form is unit 9. The stored side is canonicalised before the
     comparison and the witness is not. Two words read eight times each
     placed seven concepts and then filled the inventory.
  2. *The property reading moved.* In XOR_exact the word `00` is
     witnessed with property rows `(1, 9)`, then `(1, 8, 9)`, `(8, 9)`,
     `(9,)` and `()`, as the learned memberships of its bytes change.
     Every change mints an alternative: 11 concepts become 23, and the
     next has no row. (Measured in the candidate with the forcing off,
     which takes HEAD's route and fails on HEAD's line.)

  At capacity the new alternative has no row, and
  `add_concept_edge(c_row, None)` raises `int(None)`; by then an identity
  has been allocated without a row. This is the word-router failure in
  the candidate and the XOR_exact crash at HEAD. *Fix:* the parts identify
  the word, as `lookup_word` already says ("Ordered native references
  identify the word"). Compare canonical parts with canonical parts. A
  reading with the same parts is the same word, whatever its properties
  read at that moment; an alternative is for different parts. At capacity
  refuse before anything is allocated, as the aligned route's contract has
  it ("an admission refused at capacity remains the exact −1 unknown").
  What a repeated reading writes to the property literals, if anything,
  is Codex's to propose; the probe wrote nothing.
* **Q. XOR_exact has been broken since the item 9b landing, and no
  receipt measured it** *(measured)*. One run of each revision on a clean
  archive, without a seed:

  | Revision | Landing | Inputs reconstruct | Output crisp | XOR_grammar |
  |---|---|---|---|---|
  | `d5f2e1a0` | 11c | fail, 0 of 4 | pass | fail, `W=6` |
  | `99207a38` | item 10 | pass | pass | fail, `W=6` |
  | `0e700012` | item 9 parity | pass | pass | fail, `W=6` |
  | `8bc710a8` | **item 9b** | **crash** | **crash** | fail, `W=6` |
  | `206a0146` | item 8 | crash | crash | fail, capacity |
  | `69067272` | item 7.5 | crash | crash | fail, capacity |
  | `1678ee1f` | HEAD | crash | crash | fail, capacity |

  The two gates are marked `slow`, so every sweep skips them, and they
  are not in the list of explicit gates; the last receipt that ran them
  is item 10's. Claude reviewed 9b, 8 and 7.5 and did not ask for them.
  *Fix:* P repairs the crash. The gates join the explicit list (question
  2).
* **R. The membership read does not scale to a sentence** *(measured)*.
  `PrimitiveProperties.evidence_on_counts` evaluates every property row
  on every native event before the few referenced rows are selected:
  1,288 events by 8,192 property rows by 256 byte values, 12.7 GiB in one
  call, from which the few referenced rows are kept. The grounded models never met this
  because their fields have a handful of events. *Fix:* select the
  referenced property rows first, and read only the events inside the
  word brackets.
* **S. A memory stop is not a diagnosis, and two of the four hide a
  failure** *(measured, each case run once without the guard)*.

  | Case | Candidate | Forcing off | HEAD |
  |---|---|---|---|
  | Two-epoch graph release | **fails**, 13.75 GiB | **fails**, 13.92 GiB | passes, 6.09 GiB |
  | Category codebook | **fails**: "codebook never enabled", 10.80 GiB | passes, 0.79 GiB | passes, 0.52 GiB |
  | 64-word trace | passes, 13.17 GiB | passes, 0.98 GiB | passes, 0.65 GiB |

  The two-epoch case raises "thought operation 'part' is unavailable:
  ConceptualSpace concept inventory exhausted while allocating 8 id(s)
  for grammatical thought VPs", reached through `_sentence_observation`,
  `finish_clause` and a long alternation of `recover` and `operand` in
  `ClauseJournal`. The forcing is not its cause, and it has been in the
  candidate since round 1. The guard stops the process first, so the
  receipt could only say "memory".
* **T. In `MM_xor` the forcing changes the optimizer's layout**
  *(measured)*. It creates `concept_parts_layer.features.values`, eight
  values, in the first batch. The live optimizer holds the new parameter
  as a group of its own, 91 / 1 / 3; a restored model has it from the
  start and builds 92 / 3. Hence "optimizer parameter-group layout
  differs". Without the forcing both are 91 / 3. *Fix:* a parameter that
  joins after construction lands in the group that construction gives it.
* **U. MM_grammar is not shown to differ, and its gate could not show
  it** *(measured, ten runs of each tree, no seed)*. The gate asks only
  that the error fall below .20 at some epoch of 900, so it passes on a
  dip. Trained for all 900, the error at the end was:

  | Tree | Runs | Median | Mean | Runs ending below .05 | Range |
  |---|---|---|---|---|---|
  | HEAD | 10 | 0.042 | 0.063 | 6 | 0.000 to 0.188 |
  | Candidate | 10 | 0.113 | 0.125 | 3 | 0.000 to 0.317 |
  | Candidate, forcing off | 6 | 0.048 | 0.066 | 3 | 0.000 to 0.191 |
  | Candidate with the patch of §15.5 | 10 | 0.097 | 0.103 | 4 | 0.000 to 0.239 |

  The candidate's median is higher, and ten runs do not separate the
  trees (a rank test gives p = .23). Nothing is claimed. *Fix:* the
  receipt records the error at the end of training, and the four
  predictions, for the runs of each tree.
* **V. In the configurations the forcing was decided for, the word
  symbols are the same for every sentence** *(measured)*. In `MM_xor`,
  `MM_grammar` and `XOR_grammar` the symbolic state has norm 2.0 and is
  identical for the four sentences, a distance of exactly zero between
  any two. Read in `XOR_grammar` and `MM_xor`, every one of the four
  word concepts is fully present at both word positions of every
  sentence: `there` and `loving` are present in "hello world". The percept evidence underneath is right (`hello`'s
  part is present at the first bracket of the first two sentences and
  nowhere else). The concept read is what loses it. A word is defined as
  its part *and* the property rows of its bytes, here row 0, which every
  one of these words has. An absent part is unknown, not absent, and the
  fold leaves an unknown out ("a zero pole supplies no evidence and
  cannot veto"), so the property alone satisfies the definition. The
  grounded cases are not affected: a case is defined by its part only.
  So §13 B is not yet repaired. The words have concepts and the symbols
  are not zero, which is what `test_every_read_word_has_a_concept_and_symbol`
  asserts, but a state that is the same for every input says nothing, and
  no disjunction or conjunction of it can be XOR. In `MM_xor` at HEAD the
  symbolic state did differ between sentences (distance .09). One more
  fact of `MM_xor`, which is the configuration's and not the forcing's:
  its perceptual input is eight bytes wide, so the second word is read
  as a fragment, and `wo`, `th`, `w` and `t` are admitted as words.
  *Fix:* a
  word's symbol is active where its parts are read, and not elsewhere.
  How is Codex's to propose; the fold's rule for field concepts (item 11)
  is not changed without Alec. The test gains the assertion that was
  missing: a word that is not in the sentence is not active, and two
  sentences of different words have different symbolic states.
* **W. XOR_grammar is red, as it was, and now runs to the end.** It has
  failed in every receipt back to 2026-09-21 and at every revision of Q.
  In the candidate it trains for its 400 epochs and answers one half to
  everything (error 0.2499). V is sufficient for that. Its reconstruction
  returns "world world" for "hello world" and nothing for the other
  three; that has not been diagnosed.
* **J stays open.** After training the candidate's mean is 0.1287 and
  HEAD's three seeds span 0.1085 to 0.1188, so nothing is re-baselined.
  The measure runs where the forcing is already off (the canonical
  aligned reading), so N does not explain it. Seven training batches move
  HEAD's own seeds in both directions (0.101 to 0.109, 0.122 to 0.119),
  so three seeds do not separate the trees. Declare eight seeds in
  advance and measure both.
* **X. What is forced is the admission of the word, not its
  interpretation** *(measured, after Alec's question of 15.4)*.
  `interpret` has two parts. `lookup_word` admits the word as a concept,
  defined by its parts. `forward` takes that word to its object, minting
  the object if the word has none, and writes the META that binds them
  ([item 9b §5](../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md#5-the-interpret-operator-decided-alec-2026-09-25)).
  The forcing calls the first and never the second. Four words were read
  twice in each reading:

  | Reading | Word concepts | Object concepts | METAs |
  |---|---|---|---|
  | Serial, aligned (the reading 9b built; not forced), HEAD and candidate | 4, order 0 | 4, order 1, one for each word | 4, order 2, each holding one word and one object |
  | Forced, `mixing`: `MM_xor`, `XOR_grammar` | 4, order 0 | none | none |
  | Grounded, aligned, forcing on | 8 for the four words (each admitted twice, P) | none | none; and no case is discovered |
  | Grounded, aligned, forcing off, as at HEAD | none | none | none; four cases are discovered at the second presentation |

  So where `interpret` runs whole, a word binds to one object through one
  META, the object's word is that word, and none of the regressions is in
  that reading. Where it was forced, no object and no META exists. The
  harm in the grounded models is done below interpretation. The word, as
  a thing perceived, and the case the field discovers are one order-0
  concept of one percept, reached by two routes: by decree at first
  sight, and by recurrence. The route by decree gets there first, and the
  field, finding its percept taken, assigns nothing (O).
  Run by probe in the three XOR configurations as they stand, the whole
  of `interpret` mints an object and a META for every word, and the
  objects have no row: `XOR_grammar` has six concept rows, three of order
  0 and one of order 1, and `MM_grammar` eight, four and two. Four words
  and their four objects do not fit, and the mint does not refuse
  (P's capacity defect again).
* **Y. At HEAD a word that is a concept before its object is discovered
  shuts the object out** *(measured; §17.7)*. With the sentence boundary
  run after the first presentation of the grounded test, HEAD and the
  candidate have four word concepts and discover no case. The grounded
  test passes at HEAD because it never runs the boundary between its
  presentations. This is O with the order of events reversed, and it is
  HEAD's. *Fix and test:* §17.7 and test 33.
* **A lead for S**, not a finding. In the serial aligned model the
  candidate adds one concept identity for every sentence read: the same
  four sentences read twice leave 25 concepts and then 29, where HEAD
  leaves 19 and 19. If each ended sentence is meant to have an identity,
  the inventory fills at one a sentence, and the exhausted inventory of S
  may be its first symptom.

### 15.3 The two probe patches

Both were applied from outside the tree, to test the diagnosis.

1. **The forcing stands down under the aligned binding** (O).
2. **A word read with the parts of its stored definition is recognised
   as itself**, and nothing is minted or written for it (P).

| Measured with both | Result |
|---|---|
| The fourteen regressed cases | 13 pass; the optimizer layout (T) remains |
| Those seven files in full, with `test_item7_word_interpretation.py` and `test_mm_xor.py` | 57 of 58 |
| XOR_exact, three runs | error 0.0, predictions 0 1 1 0, 4 of 4 reconstructed, each time |
| The sixty test files that concern concepts, XOR and item 7 (582 cases) | 556 pass, 26 skipped, none fails; the candidate as it stands fails 11 of them |
| XOR_grammar | error 0.2499; not repaired by these (V) |

### 15.4 Questions for Alec, and his answers (2026-09-29)

1. **Should the forced `interpret` stand down wherever the model already
   derives its word concepts itself?** Alec asked in return: "Interpret
   was introduced by me under the assumption that it would add meta
   symbols, as described previously, so that word-concepts bind
   unambiguously to object-concepts. Is that happening? If it is, why is
   interpretation of the word a problem?" Measured (X): it happens in
   the serial aligned reading and nowhere else, and nothing fails there;
   what was forced is only the admission of the word. Then, put again
   in two parts:
   * **1a. Should the forced admission stand down under the aligned
     binding?** **Superseded.** "See if the above reasoning restores XOR.
     We need that to pass." The reasoning is §15.5; it restores XOR, and
     nothing has to stand down.
   * **1b. Should the forcing run the whole of `interpret`?**
     **Decided:** "Yes, this was my first point."
2. **Should every proof of XOR run in every receipt?** **Decided:** "Yes."
3. **Is XOR_grammar's gate a condition of accepting item 7?** **Decided:**
   "No, but it would be nice to get it working soon; let's add it as a
   requirement gate for the grammatical operators update." It is recorded
   there in [the todo](../../todo.md). Item 7 is accepted when no other
   proof of XOR is worse than at the last revision where it passed.
4. **May `XOR_grammar.xml` and `MM_grammar.xml` be given more concept
   rows?** **Decided, no:** "They should not need more rows, we are
   basically replacing word-concepts with object-concepts, and had enough
   previously." Measured under §15.5: the four objects take rows 0, 1, 2
   and 4 of `XOR_grammar`'s six. The configurations stay as they are.
   (Claude's probe in X ran the operator as it is now written, which
   keeps a row for the word, a row for the object and a row for the META;
   that is what did not fit.) *Reopened the same day:* with Alec's later
   answer that "a word has to exist as a concept", a word and its object
   are two concepts and the count is eight again, without the META. Put
   to him again, he confirmed his first answer: "The interpret method is
   a unary that should do exactly that replacement, so we should not
   need more rows" (§17.7).
5. **Where does the grammatical operators update sit in the sequence?**
   **Decided:** after item 7 is accepted, after the conference freeze and
   before 6.8. "That would be fine, let's iron out the operators after
   getting 7 accepted."

### 15.5 What `interpret` was to be, and the test of it (Alec, 2026-09-29)

"So for interpret(), I wished it to be a replacement for the previous
two-step behavior: mint a word, and because we know the word is not the
object, we link those two concepts via a meta symbol. So it shouldn't have
digressed from previous behavior when forced (although we may optimize the
meta symbol process somehow: I would not expect to create a fold just to
unite word and object). Perhaps given that LTM now uses references, we can
leverage those to store the symbols (e.g. word REF object)." And, of the
rows: "we are basically replacing word-concepts with object-concepts". And
later the same day: "if we do use the references of LTM as the META
symbols, we will probably want to build a lookup table or some other way
to quickly find the referent for a reference."

**Where the forced step digressed** *(read against the two-step at
`0e700012`, `create_word_object_meta`)*.

| The two-step did | The forced step does |
|---|---|
| Fused the word's parts first (`fuse_parts`), so a word was defined by its unit once it had one | Takes the parts as they arrive; the stored definition is fused later and the next witness is not (P, 1) |
| Reused the word's triple when the word was known | Looks the word up by its unfused parts, so a word whose parts have fused is admitted a second time |
| Reserved word, object and META together, and refused all three at capacity | Admits the word alone; at capacity an identity is left without a row |
| Made the object and the link | Makes neither (X) |
| In the parallel reading, ran at the sentence boundary, after the field had been read | Runs before the field is read, and switches the boundary's other work off |

The last row has a cost of its own: with the forcing active the boundary
no longer enables the category codebook or registers recognised words.
That is why the category codebook case of S fails when it is run without
the memory guard, "codebook never enabled".

*Amended the same day (§17.7):* point 2 below was written before Alec's
answer that "a word has to exist as a concept". The row the probe kept
is, in his terms, the word concept's, defined by its parts; the object is
a second concept. The measurements stand for the patch they were made
with, and those made to his later answers are in §17.7. One of them does
not carry over: the memory below is that of words defined by their
parts alone, and with every word keeping its wholes it returns until R
is repaired (§17.7).

**The test.** A probe patch makes the forced step what the quotation
describes, with the forcing left on under every binding:

1. the word's parts are fused first;
2. one row is kept for a word, and it is the object, defined by the
   word's unit and not by the property rows of its bytes;
3. a word read again is recognised, and nothing is minted or written;
4. the link from word to object is a record and no fold is made;
5. where the field keeps a pool of provisional cases, the object of a
   read word is the row the field binds for its unit, and bytes that are
   not yet a unit are not yet a word;
6. the sentence boundary keeps its other work.

| Measured, candidate with that patch | Result |
|---|---|
| Grounded XOR | 6 of 6 |
| XOR_exact through the curriculum | 3 of 3 |
| XOR_exact end to end, three runs | error 0.0, predictions 0 1 1 0, 4 of 4 reconstructed, each time |
| The other regressed cases: word router, attended-field reuse, membership boundary | 4 of 4 |
| `test_item7_word_interpretation.py` | 4 of 4 |
| Category codebook, three cases | 3 of 3, at 0.82 GiB (one of them alone took 10.80 in the candidate) |
| 64-word trace | passes at 1.05 GiB (13.17) |
| Word symbols in `XOR_grammar`, `MM_grammar` and `MM_xor` | each sentence binds its own words and no others, each present at its own bracket only. In `XOR_grammar` and `MM_grammar` the four symbolic states differ, by 1.6 to 2.2. In `MM_xor` three do: its input is eight bytes wide, "loving world" and "loving there" are both read as `loving` and one further byte, and one byte is not a word |
| Rows used in `XOR_grammar` | four of six, no change to the configuration |
| `MM_xor` trained for its 600 epochs, three runs | error at the end 0.000, 0.000 and 0.004 |
| The 73 test files that concern concepts, XOR, item 7, the router and the category codebook (651 cases) | 623 pass, 27 skipped, one fails, which is T |
| Optimizer layout in `MM_xor` (T) | still fails; it is a separate repair |
| XOR_grammar's gates | still fail: error 0.25, every prediction near one half, 0 of 4 recovered, three runs |

So the reasoning restores XOR, and it repairs O, P, V and the memory of R
and S's two codebook cases with it, because a definition that holds the
word's unit alone has no property literal to read densely and none to
drift. It does not make XOR_grammar learn: its symbols now say which
words were read, and what composes them is the business of the
grammatical operators update.

Without point 6 the same patch fails three cases, the attended-field
reuse and two of the category codebook, all of which depend on the
boundary's work.

What XOR_exact holds after training under the patch: the four objects at
rows 24 to 27, each defined by its unit alone, `00`, `01`, `10`, `11`; and
the XOR row, of order 1, a union of two of them, `01` and `10`. **One
thing the probe left as HEAD has it.** In the grounded models the
sentence boundary still admits a word concept beside each object (rows 2
to 5, the same unit with the property rows of its bytes), and nothing
links the two: no interpretation is recorded. XOR does not read them.
Whether they go, so that the rows hold objects only as the quotation has
it, belongs with the reference design below.

**Claude's reading of the reference proposal, to confirm.** Nothing here
is decided.

* *What the fold costs today.* In the serial aligned reading one new
  word adds three identities and three rows: the word at order 0, the
  object at order 1 with the word as its only part, and the META at
  order 2 folded over the object. A reference in their place leaves one
  row, the object's.
* *The lookup exists in kind.* `ReferenceTable` is a word-keyed index
  from a word to its objects, rebuilt from META membership and cached by
  revision; and the row store has an inverted index from a code and its
  role to the rows that hold it. Finding the referent of a reference
  would be one keyed lookup, resolved at the eager boundary before the
  compiled word loop, as the word's row is now. The table would be
  rebuilt from the reference rows where it is now rebuilt from METAs.
* *For a word with one object the shared row may already be the link.*
  A symbol and its concept share a row index, and under point 2 the
  symbol a read word publishes is the code of its object's row.
  Reference rows would then be needed for a word's second object and for
  an earlier occurrence (§3.5), and not for every word.
* *Three things the spec would have to settle.* A naming row must
  outlive the forgetting of ordinary rows for as long as its word is in
  use. A relation row joins two rows today, and here its subject is a
  word, which is a perceived unit. And `ReferenceTable` is one-way by an
  earlier decision ("the reverse direction must stay a search"), while
  generation needs the word of an object; the row index would make that
  a lookup too, if Alec wants it so.
* *Which item.* Claude proposes that item 7 lands with the link as a
  plain record and the serial reading's META as it is, and that the
  reference design is written as a spec once item 7 is accepted.

## 16. Hand-off to Codex (Claude, 2026-09-29): what to change after review round 3

*Amended the same day:* steps 2 to 5 of this list were replaced by the
two stages of §18, after Alec's decision to fold definitions (§17) into
item 7, and the text of steps 2 to 4 is removed so that nothing
superseded is read as an instruction. Steps 1 and 6 to 9 stand.

Work from this list, in this order. Every fix has a failing probe before
it. No test pins a seed in order to pass, no threshold moves, and the
memory guard stays at 8 GiB. Nothing is committed until Claude has
reviewed the result.

1. **Measure XOR first (§15.1, Q; decided, question 2).** Add the two
   XOR_exact gates, and the `slow` cases of `test_mm_xor.py`, to the
   explicit gates, beside the grounded file and the curriculum, which the
   sweep already runs. Record HEAD's crash as the baseline. Every later
   step, and every receipt after this one, reports this table.
2. to 4. *See §18: stage A, restore XOR; stage B, definitions.*
   XOR_grammar's gates are rerun at the end of each stage and what they
   measure is recorded; they are not a condition of this pass, and they
   are the gate of the grammatical operators update (decided, question
   3).
5. **The membership read (R).** *Moved into stage A of §18, point 7:*
   every word keeps its wholes (§17.7), so every read of a word reaches
   it.
6. **The optimizer's layout (T).**
7. **The two-epoch graph release (S).** Diagnose the exhausted inventory
   in the clause journal's recovery, and say whether an identity for
   every sentence read is intended (the lead after X). Run any case the
   guard stops once more without it, as a diagnostic that is recorded
   apart from the gate, so that the receipt says what the case does.
8. **Reconstruction (J).** Eight seeds declared in advance, both trees.
9. **Then** the XOR table, MM_grammar's error at the end of training for
   ten runs of each tree (U), and one source-matched full sweep; and stop
   for review.

Not in this pass, as before: the `NonLayer` and `ConjunctionLayer`
defects, and everything in
[accessible mind §2.0.1](2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention).

## 17. Definitions: `word DEF object` (decided, Alec, 2026-09-29)

This section is part of item 7: "I'd like to fold this solution into the
current item 7 work, even though it is a big change, since it gets XOR
working." It amends §3.2, §3.3, §3.4 and test 14 of this spec,
[item 9b §5](../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md#5-the-interpret-operator-decided-alec-2026-09-25),
the [forgetting spec](2026-09-16-forgetting.md) and
[accessible mind §4](2026-09-20-accessible-mind-subsystems.md#4-which-grammar-may-touch-what).
The measurements behind it are in §15.5 and §17.7. Where a paragraph is
Claude's reading and not Alec's decision it says so. Alec answered the
questions of 17.9 in three rounds the same day, and the section is
written to his answers. Nothing that blocks the work is open.

### 17.1 What was decided

* **What `interpret` is.** "I wished it to be a replacement for the
  previous two-step behavior: mint a word, and because we know the word
  is not the object, we link those two concepts via a meta symbol. So it
  shouldn't have digressed from previous behavior when forced".
* **The link is a definition.** "So not relation: let's have a
  definition DEF relation, 3 slots, that tie word concept to object
  concept. We would need this already for English sentences (we had
  leveraged partOf, perhaps we need an Equals() or Def() for
  definitions)."
* **No fold.** "I would not expect to create a fold just to unite word
  and object."
* **The rows are the objects'.** "They should not need more rows, we are
  basically replacing word-concepts with object-concepts, and had enough
  previously."
* **Lookup in both directions.** "we will probably want to build a
  lookup table or some other way to quickly find the referent for a
  reference"; and, of finding the word of an object, "Again, a lookup
  table or similar for language would solve this."
* **A word is a concept.** Asked whether the word concept is only the
  word's unit in PartSpace: "No, a word has to exist as a concept. If
  the concern is redundancy, we can find a different way to resolve that
  (like not storing the full percept in part space)."
* **A definition is a row like any other.** "LTM has an associated
  .when … perhaps the way we achieve your objectives is to make
  definitions timeless/eternal. But the general point is a tension:
  timeless knowledge is conceptual, but LTM is both conceptual and has a
  time of occurrence (.when). let's keep LTM addressable by .when for
  consistency. I don't see any need to keep it out of recency or
  forgetting, but perhaps the forgetting algorithm can be altered so
  that it biases forgetting or relative/ultimate truth." This replaces
  his earlier "forgetting should probably not forget definitions".
* **The META goes everywhere.** Asked whether it is retired in every
  reading in this pass, the serial one included: "Yes".
* **No more rows, because `interpret` replaces.** Told that his "should
  not need more rows" had been given when objects were to replace words:
  "The interpret method is a unary that should do exactly that
  replacement, so we should not need more rows. Thanks for eliciting the
  correction."
* **A word keeps its parts and its wholes.** Asked whether a word's
  concept should be defined by its parts alone: "Words ought to have
  meaningful parts and wholes. We can predefine a word whole that is a
  cut of non-white space letters if that would help our certainty with
  Lexing words."
* **The definition names symbols, not content.** Of Claude's reading of
  the replacement (17.7): "That reading is correct. "WordC ref ObjectC"
  at LTM, where WordC and ObjectC are both symbols (that way if the
  content is being learned, the definition does not involve
  content-addressing)."
* **The word whole waits for 6.8.** "Waiting for 6.8 is fine, as long as
  it's written down. That's next anyway." It is written down in
  [the 6.8 plan](../plans/2026-09-27-item-6-8-one-attention.md#3a-the-word-whole-alec-2026-09-29-taken-up-with-this-item)
  and in the todo.

### 17.2 The row

A definition is a row of the store with `rel_type = REL_DEF`, a fourth
relation kind beside part, implies and operator. Its three slots are
`word, DEF, object`.

* `refs[0]` is the word concept and `refs[2]` the object concept, and
  **both are symbols**: the row names its two concepts by their
  identities. "That way if the content is being learned, the definition
  does not involve content-addressing": no reader finds either concept
  by comparing the content of a slot, and what a concept's code becomes
  in training cannot make a definition point elsewhere. This departs
  from §3.2, where NP1 and NP2 carry copies of the operands' vectors; a
  definition row carries no copy of learned content, which would go
  stale. What its vector slots do hold, the two symbols' own codes or
  nothing, is Codex's to propose. The VP slot carries the DEF atom,
  which the kind fixes. (Alec wrote the row once as "WordC ref ObjectC";
  the kind keeps the name he gave it, DEF.)
* **It implies nothing about order.** A word is not a part of its
  object. Until now the object was written as a sigma over its word and
  the META as a fold over the object, which set the object one order
  above the word and the META one above that, and cost three identities
  and three rows for every new word (§15.5).
* **It has a `.when`**, as every row of the store has, and is addressed
  by it. *Claude's reading:* the `.when` is the time the definition was
  written and does not move; reading the word again refreshes the row's
  timestamp through the dedup path, which is what recency reads.
* Trust, origin and the evidence pair are those of any row (§4). The
  sentence in which a word is first read does not set the trust of the
  word's definition.
* **Deduplicated** by `(REL_DEF, refs[0], refs[2])`. Reading a defined
  word again appends nothing.
* **Several objects, several words.** A word with two objects has two
  rows with one `refs[0]`; two words for one object have two rows with
  one `refs[2]`. The choice among a word's objects is made at
  interpretation time, from context, as §3.4 and §3.5 already have it.
  This is what the n-ary META held, held as rows.

### 17.3 The writer is `interpret`

When a word is read, at the eager boundary and before any graph:

1. **The word.** Its parts are fused first, as the two-step did
   (`fuse_parts`), and the word concept is found by its unit, or by its
   ordered parts in a configuration that forms no units (`MM_xor`). A
   word that is new is admitted as a concept, with its parts and its
   wholes.
2. **The object.** The word is looked up in the table of 17.4. With one
   object, that is its object. With several, the grammar selects. With
   none, `interpret` makes the object and writes `word DEF object`.
   `interpret` is a unary: the object takes the word's place, and no
   row is added for it (17.7).
3. **The symbol.** The symbol published for a read word is its
   object's. It is present at the word's bracket and nowhere else, and a
   word that is not in the sentence publishes nothing.

Four rules go with it.

* **One transaction.** What a new word needs, its row in the inventory
  and its definition's row in the store, is reserved before anything is
  written, as the two-step reserved its triple. A refusal at capacity
  leaves nothing behind, neither an identity without a row nor a word
  without its object.
* **Read again, recognised.** A word read again with the same parts is
  a lookup. Nothing is minted, and a changed property reading of its
  bytes changes nothing (§15.2 P). Different parts remain an alternative
  definition of the same word, as before.
* **Where the field discovers its own objects**, in the models that keep
  a pool of provisional cases, the object of a read word is the row the
  field binds for the word's unit, and the definition is written when
  that object has an identity, which for a provisional case is at its
  admission. **A word concept never shuts its object out** (17.7).
* **The boundary keeps its work.** `interpret` replaces the admission of
  the word. It does not switch off the rest of what the sentence
  boundary does (the category codebook, recognised words).

**Writers.** The closing stays the only writer of what a sentence
asserts: ideas, and the part, implies and operator relations (§3.3).
`interpret` is the only writer of definitions in item 7. *Claude's
reading:* a sentence that states a definition, "a bachelor is an
unmarried man", and the choice between `Equals` and `Def` for it, belong
to the grammatical operators update; until then "equal" stays two part
rows (§3.2, test 7).

### 17.4 The table

One derived index over the definition rows, rebuilt on load and after
every compaction of the store. It is an index and not a store: it holds
no trust, and nothing that the rows do not hold.

| Lookup | Gives |
|---|---|
| a form, or a unit | the word concept |
| a word concept | its objects |
| an object concept | its words |

No lookup scans. The third supersedes the one-way law of `References`
("the reverse direction must stay a search"): generation finds the word
of an object here. §3.4's "word to object is a direct index" stands, and
this is that index. The table replaces `ReferenceTable` as it is now
rebuilt from META membership, and the allocator's `interpretations` and
`word_obj_meta` records.

### 17.5 What is retired

By the no-legacy rule, and in the same pass: the META concept and its
fold (`bind_meta`, `meta_members`, the sigma singleton that raised a
member to the other's order), and the object's part edge to its word.
**Decided (Alec, 2026-09-29):** in every reading, the serial aligned one
included, so that one mechanism ties a word to its object.

**Migration.** A checkpoint's META bindings load as definition rows,
one for each pair of a word and an object, and the META identities are
retired. The table is rebuilt from the rows.

### 17.6 A definition is a row like any other

**Decided (Alec, 2026-09-29).** A definition is kept out of nothing. It
has its `.when`; it is in the recency view with every other row; and the
forgetting pass may delete it.

* **Forgetting.** No definition is protected. What keeps a word in use
  from losing its definition is the value the pass computes, and
  "perhaps the forgetting algorithm can be altered so that it biases
  forgetting or relative/ultimate truth". *Claude's reading, to confirm:*
  a weight in the value by the kind of truth, relative rows against
  ideas, with definitions among the relative. It is a proposal, it
  belongs to item 5 and the [forgetting spec](2026-09-16-forgetting.md),
  and item 7 builds none of it. A definition that is forgotten leaves a
  word that is new the next time it is read, and its object is minted
  afresh.
* **Luminosity.** A definition is a relation row, and §4 already leaves
  relation rows out of the coverage measure. Nothing changes.
* **What is measured, since nothing is excluded.** A sentence of new
  words writes a definition for each before it writes its own row, so
  early in training the rows nearest in time are mostly definitions. The
  receipt reports the share of definitions among the rows the
  predictor's situation reads, and the reconstruction measure of §15.2 J
  is taken with definitions on, so that an effect on expectation is seen
  and not assumed.
* **Capacity.** The store now holds the vocabulary as well. The receipt
  records how many of its rows are definitions.

### 17.7 A word is a concept, its object replaces it, and no row is added

**Decided (Alec, 2026-09-29), in the order he said it.** "A word has to
exist as a concept." "The interpret method is a unary that should do
exactly that replacement, so we should not need more rows." "Words ought
to have meaningful parts and wholes." No configuration is changed. If
the word's percept in PartSpace and the word's concept say the same
thing twice, that is to be resolved on PartSpace's side, "like not
storing the full percept in part space", and not in item 7.

**The replacement (Claude's reading; confirmed by Alec, 2026-09-29:
"That reading is correct").** A unary leaves as many slots as it found.
`XOR_grammar` has six concept rows and four words, so a row for each
word and a second for each object cannot both be meant. The reading
under which every statement above holds:

* a word that is read for the first time takes **one row** of the
  concept inventory, defined by its parts and its wholes, which is how
  the field knows the word when it meets it;
* `interpret` replaces the word by its object **in that row**: from then
  on the row is the object's, and what is learned of the object is
  written there;
* the word concept is kept as the **first slot of its definition row**
  in the store, with its own identity, so it exists as a concept and is
  tied to its object; the third slot names the object. Both slots hold
  symbols (17.2).

So a new word costs two identities, one row of the inventory and one row
of the store, where HEAD's serial reading costs three identities and
three rows of the inventory. It is the row the first probe of §15.5 kept,
and four words then use four of `XOR_grammar`'s six rows.

**Measured, all on 2026-09-29.**

* **At HEAD a word that is a concept first shuts its object out.** The
  grounded test never runs the sentence boundary between its two
  presentations, so its words are not yet concepts when the field
  discovers the four cases. With the boundary run after the first
  presentation, and nothing else changed, HEAD has four word concepts
  and discovers no case, and grounded XOR fails six of six at
  `0 == 4`; the candidate does the same. `_observe_native_cases` asks
  whether any row holds the unit, and the word's row does. *Fix:* the
  field gives a recurrent unit its case unless a case already holds it;
  a word concept that holds it is no reason to withhold one. Under the
  reading above the two are one row, and the question does not arise.
* **A word with its parts and its wholes, evidenced only where its part
  is read.** A probe patch with the six points of §15.5, every word
  keeping its wholes in its definition, and the read requiring the
  word's part: in `XOR_grammar` and `MM_grammar` each sentence evidences
  its own two words at their own brackets and no other, and the four
  symbolic states differ, by 1.4 to 2.0; XOR_exact end to end has error
  0.0 and answers 0 1 1 0 in three runs of three; and with the word a
  concept before its object the four cases are discovered and XOR is
  learned exactly, six cases of six. Over the 73 test files of §15.5,
  623 cases pass, 27 are skipped and one fails, the optimizer's layout
  (T). V is therefore a defect of the read and not of the definition,
  and the definition stays as the decision of 2026-09-24 has it.
* **With the wholes kept, the memory returns until R is repaired.**
  Under the same patch the category codebook cases pass at 11.08 GiB
  and the 64-word trace at 13.69, where the guard is 8: every word's
  definition now holds a property row, so every read of a word
  evaluates every property row on every event (R). The repair of R is
  therefore part of stage A and not an afterthought.
* **What the wholes in a definition still do.** In that last test an
  unrelated input ("AA", "22") raises twelve percept events, by the
  properties it shares with the words, where the test as written
  expects none. With wholes in a word's definition that is as it should
  be, and test 33 does not assert it. The learned property rows of a
  word still move in training (P, 2); point 3 of stage A is what keeps
  a move from minting anything.

**The word whole (Alec's offer; placed in item 6.8).** "We can predefine a
word whole that is a cut of non-white space letters if that would help
our certainty with Lexing words." *Claude's assessment:* it is not
needed to restore XOR. It would help in two ways. The wholes a word has
today are the character classes that hold on its surface, `letter` for
"hello", `letter` and `digit` for "abc123", and their memberships are
learned, so they move; a predefined word whole would give every word one
whole that does not. And WholeSpace cuts its field where a learned
property changes, while the reader cuts words by a fixed rule
(`Meronomy.word_spans`, maximal runs that are neither white space nor
punctuation); a predefined whole would make the two cuts agree. It does
not repair V, since every word would share it. **Decided (Alec,
2026-09-29):** it waits for item 6.8, "as long as it's written down",
and it is, in
[the 6.8 plan](../plans/2026-09-27-item-6-8-one-attention.md#3a-the-word-whole-alec-2026-09-29-taken-up-with-this-item);
and its extent is a word by the common definition, "punctuation and
digits also separate words". The reader's rule does not separate at
digits today, and the proofs of XOR read runs of digits; both are noted
in that plan.

### 17.8 Acceptance tests

They continue §7. Test 14 is superseded by 22 and 24.

22. **A new word.** Reading a word that has no definition admits the
    word concept, replaces it by its object, and writes one `REL_DEF`
    row, `word, DEF, object`. The inventory gains one row, the store
    one. No META concept exists, and the object has no part edge to its
    word. The definition row has a `.when`.
23. **Read again.** Reading it again writes nothing: the counts of
    rows, identities and concept rows are unchanged. The same holds
    after the word's parts have fused, and after the learned property
    rows of its bytes have changed.
24. **Both directions.** The table returns the object of a word and the
    word of an object, each without a scan. The table rebuilt from a
    saved store equals the live one.
25. **Two objects.** A word with two definition rows is ambiguous until
    the grammar selects; with a selection it returns that object; a
    second word for one object is returned by the reverse lookup beside
    the first.
26. **Capacity.** With no room for a new word's object or for its
    definition, the word is refused and nothing is left behind.
27. **A row like any other.** A definition row is addressed by its
    `.when`, appears in the recency view in the order it was written,
    and is refreshed there, not appended, when its word is read again.
    (Its forgetting is tested with item 5.)
28. **A forgotten definition.** With a word's definition row deleted
    from the store and the table rebuilt, the word is new when next
    read: it is interpreted afresh, and no lookup returns the deleted
    pair.
29. **The symbols say what was read.** In `MM_xor`, `MM_grammar` and
    `XOR_grammar`, on their own sentences: the symbol of a word that is
    not in the sentence is zero, and sentences that are read differently
    have different symbolic states.
30. **XOR.** Grounded XOR six of six; XOR_exact through the curriculum
    three of three; XOR_exact end to end, both gates; `MM_xor` and
    `MM_grammar`. No seed. XOR_grammar's two gates are run and recorded,
    and are the gate of the grammatical operators update.
31. **No more rows.** `XOR_grammar.xml`, `MM_xor.xml` and
    `MM_grammar.xml` are byte-identical to HEAD, apart from the retired
    `propertyBasis` key, and every word read in them has its word
    concept, its object and its definition.
32. **Migration.** A checkpoint with META bindings loads with one
    definition row for each binding, no META identity, and the table
    rebuilt.
33. **The word before its object.** The grounded test with the sentence
    boundary run after the first presentation, so that the four words
    are concepts before the field has seen their units recur: four cases
    are discovered and XOR is learned exactly. No seed. HEAD fails it at
    `0 == 4`. The test's last assertion, that an unrelated input raises
    no percept event, is not part of this test, since a word keeps its
    wholes (17.7); the count of events is recorded.
34. **By symbol, not by content.** With the code of a word's concept
    and the code of its object's row overwritten, as learning would
    change them, the table still returns that object for the word and
    that word for the object, and the definition row is unchanged. No
    reader of a definition row compares vectors.

### 17.9 Questions for Alec, and his answers (2026-09-29)

All answered, in three rounds, and written into the section above:

* *Is META retired in every reading in this pass?* "Yes" (17.5).
* *Is the word concept the word's unit in PartSpace, so that
  ConceptualSpace's rows hold objects only?* "No, a word has to exist
  as a concept" (17.7).
* *Are definitions kept out of discourse, and are the water marks
  reckoned over what can be forgotten?* No to both: a definition has
  its `.when`, and "I don't see any need to keep it out of recency or
  forgetting" (17.6).
* *May `XOR_grammar.xml` and `MM_grammar.xml` be given more rows?* No:
  "The interpret method is a unary that should do exactly that
  replacement, so we should not need more rows" (17.7).
* *Should a word's concept be defined by its parts alone?* No: "Words
  ought to have meaningful parts and wholes" (17.7).
* *Is the reading of the replacement in 17.7 right?* "That reading is
  correct", with both slots of the definition symbols (17.2).
* *Does the predefined word whole wait for item 6.8?* "Waiting for 6.8
  is fine, as long as it's written down" (17.7).

Claude's readings, which Alec may correct and which block nothing:

* a sentence that *states* a definition, and the choice between
  `Equals` and `Def` for it, belong to the grammatical operators update
  (17.3);
* a definition's `.when` is the time it was written, and reading the
  word again refreshes its timestamp and not its `.when` (17.2);
* the bias of forgetting by the kind of truth is a weight in the value
  of the forgetting spec, for item 5 (17.6);
* the kind is named DEF, as Alec named it, though he wrote the row once
  as "WordC ref ObjectC" (17.2).

## 18. Hand-off to Codex (Claude, 2026-09-29): the order of work, replacing §16's steps 2 to 5

§16's steps 1 and 6 to 9 stand. Its steps 2 to 5 are replaced by two
stages. Each stage has its failing probes first and ends with the XOR
table of §15.1 measured and no proof of XOR worse than at HEAD. No test
pins a seed in order to pass, no threshold moves, no configuration is
changed, and nothing is committed until Claude has reviewed the result.

* **Stage A. Restore XOR.** Every read word is interpreted, under every
  binding, and:
  1. the word's parts are fused first;
  2. the word is a concept with its parts and its wholes, and it is
     evidenced only where its part is read: a whole it shares with
     other words does not make it present where it is not (V, test 29).
     No word's definition is changed;
  3. a word read again with the same parts is recognised, and nothing
     is minted or written for it, whatever its properties read at that
     moment; different parts remain an alternative definition of the
     word, as `test_witnesses_write_alternatives_without_conjoining_their_literals`
     has it;
  4. where the field discovers its objects, the object of a word is the
     row the field binds for its unit, and the word's concept never
     shuts that discovery out (test 33);
  5. a refusal at capacity leaves nothing behind;
  6. the sentence boundary keeps its other work;
  7. the membership read evaluates the property rows a definition
     refers to, on the events inside the bracket, and no others (§15.2
     R). With every word keeping its wholes the category codebook and
     the 64-word trace take 11.08 and 13.69 GiB until this is done.

  The link from word to object is a plain record in this stage. Tests
  23, 26, 29, 30 and 33, and the category codebook and the 64-word trace
  under the 8 GiB guard. Shown by probe, with every word keeping its
  wholes: grounded XOR six of six as the test is written; with the word
  a concept first, four cases discovered and XOR learned exactly in six
  cases of six; the curriculum three of three; XOR_exact exact in three
  runs of three; the word symbols of `XOR_grammar` and `MM_grammar`
  different for every sentence.
* **Stage B. Definitions.** `interpret` replaces the word by its object
  in the word's row (17.7); `REL_DEF` rows, whose two operands are
  symbols (17.2), are written by `interpret`; the table of 17.4 takes
  the place of the record, of `ReferenceTable` and of the allocator's
  records; and the META and its fold are retired in every reading, with
  their migration. A definition is a row like any other (17.6). Tests
  22, 24, 25, 27, 28, 31, 32 and 34. The tests that read a word's own row
  in the concept inventory are ported to the row being the object's,
  each port recorded as §13 I required.

## 19. Review round 4 (Claude, 2026-09-29, on the candidate after §16 to §18)

Codex's receipt is `doc/benchmarks/2026-09-29-item7-review-round3/`,
"stopped for review, candidate red". Claude's runs used a copy of the
working tree taken at 19:07 on 2026-09-29, whose runtime is the one that
receipt measured, and the clean archive of HEAD `1678ee1f`.

**Not accepted yet.** Stage A and Stage B are in as §17 and §18 asked, and
the fourteen XOR proofs that are conditions of item 7 pass. Two things keep
it from acceptance: one defect makes an XOR proof crash and two item 7
acceptance tests fail now and then (Z); and twelve of the sweep's thirteen
failures pass at HEAD, so they are this candidate's, with the memory gate
still red. Separately, a slow XOR proof that no table named fails now and
then on both trees (AA).

### 19.1 The status of XOR

Run by Claude on both trees, unseeded, once each unless said otherwise:

| proof | HEAD | candidate |
|---|---|---|
| grounded XOR, six cases | 6 pass | 6 pass |
| XOR_exact curriculum, with the rest of its file (ten cases) | 10 pass | 10 pass |
| MM_xor and MM_grammar early-stop gates, with their file (seven cases) | 7 pass | 7 pass |
| XOR_exact CLI, crisp output and reconstruction | both crash, `int(None)` | both pass |
| XOR_grammar, class accuracy and reconstruction | both fail, capacity crash | both fail: 0.0, and 0 of 4 |
| SPNN XOR training | pass | pass |
| configuration matrix, XOR cases | 2 pass | 2 pass |
| reconstruction round trip, XOR cases (five, slow) | 5 pass | 4 pass; the MM_20M_xor exact round trip fails, exact match .5 where 1.0 is required |
| MM_20M_xor exact round trip, fifteen runs each | 14 pass, 1 fails (.5) | 12 pass, 3 fail (.5) |
| MM_grammar, ten full runs (Codex's receipt) | 10 of 10 complete | **8 of 10**: two crash on capacity (Z) |

### 19.2 What is right

* **The definitions of §17.** `REL_DEF` rows name the word in `refs[0]` and
  the object in `refs[2]`, carry a fixed DEF code in the relation slot, and
  are written once for each pair. `Definitions.py` is the derived table of
  §17.4: from a form or a unit to the word, from a word to its objects, from
  an object to its words, rebuilt from the rows and never scanned. META,
  `bind_meta` and `meta_members` are gone from the runtime and live only in
  the load-time migration. A new word costs two identities, one inventory
  row and one store row (test 22).
* **XOR is restored where the earlier candidate had broken it:** grounded
  six of six, the curriculum, and XOR_exact exact in Codex's runs (error
  0.0, answers 0 1 1 0) and passing in Claude's. The membership read of R is
  repaired: the category codebook runs at 0.90 GiB and the 64-word trace at
  0.93, where they had taken 11 and 14.
* **One identity owner.** Words, objects and ended clauses share the grammar
  registry's allocator, so a definition's operands no longer alias sentence
  rows.
* **The optimizer layout (T)** is 92/3 again.
* **Reconstruction (J) is not worse.** Over the eight declared seeds the
  candidate's mean after training, .1168, lies inside HEAD's range, .1013
  to .1238, with one seed above it. J is closed, apart from the memory stop
  of one seed, which is AG's.
* **MM_grammar's full runs (U) are better where they complete:** seven of
  eight end below .05, median .0035, against five of ten and .035 at HEAD.
  The two that do not complete are Z's.
* **The record.** Every port keeps its old and new body (97 entries); no
  protected assertion is weakened; no item 7 test sets a seed.

### 19.3 Findings, with the probable fix for each

* **Z. A sentence that ends as a relation made by a grammar operation
  crashes MM_grammar** *(measured)*. When a derivation puts `part` or
  `equal` under another operation, `disjunction` over a part relation for
  instance, the closing makes the sentence an operator relation whose
  predicate is that operation, and admits a concept for the predicate
  (`ClauseJournal.operation_concept`, a `grammar-predicate` term). The term
  needs an inventory row. MM_grammar's eight are in use, so
  `ClauseTaxonomyPlan.validate_rows` raises "native conceptual row capacity
  exhausted before clause admission". It stopped two of Codex's ten
  MM_grammar runs, and one of three of Claude's at the first epoch. It also
  makes two item 7 acceptance tests fail on MM_grammar now and then: test 29
  in 2 of 20 fresh processes and test 31 in 1 of 20, and one of the two
  failed in one of Claude's two runs of the item 7 files. *Fix, by Alec's
  answer (§19.5):* an identity is nothing but its occurrences in LTM, tied
  together by the references in the rows' slots. The predicate of a
  relation made by a grammar operation is such an identity: one for each
  operation, referred to by the relation slot of every row it occurs in,
  and given no inventory row. So is a phrase that a row refers to, whose
  point is read from its row. A cache from an identity to its rows may be
  kept where finding them needs it, as the table of §17.4 is kept for
  definitions. *Claude's recommendation, with it:* a symbolization that
  needs an inventory row when none is free is refused and leaves nothing
  behind, as a word's admission is (§17.3), and the reading goes on.
  *Test:* a derivation with `part` under `disjunction` ends in
  MM_grammar without taking a row; MM_grammar's ten full runs all complete;
  tests 29 and 31 pass in 20 of 20 fresh processes each.
* **AA. The slow MM_20M_xor exact round trip fails now and then, on both
  trees** *(measured)*. `test_mm20m_xor_exact_roundtrip` asks that every
  input be decoded exactly at 160 epochs. In fifteen unseeded runs it failed
  three times on the candidate and once at HEAD, each time decoding half the
  inputs. Fifteen runs do not establish a difference between the trees, so
  it is HEAD's as well, and not item 7's to repair. It is an XOR proof that
  was in no table: it is slow, so the sweep skips it, and the receipt's XOR
  table does not name it. *Fix:* every receipt's XOR table names it with its
  runs (§16 step 1); why it fails at all is recorded in the todo for the
  XOR baseline, item 6.9.
* **AB. Checkpoint restoration** (the synthesized-answer, old-clock and
  FineWeb checkpoint tests). The owner of the knowing codes is now the
  terminal space, and its allocator is restored later in the same loop than
  the knowing fields that refer to it. *Fix:* restore every allocator
  before any knowing field, in two passes. *Test:* the three pass.
* **AC. The order of a new object.** `interpret` gives a new word's object
  the word's order, 0, which §17.2 allows: a definition implies nothing
  about order. Two readers still assume that an object is one order above
  its word, as under the retired sigma: `Models._referent_representation`
  keeps only identities of order above 0, and the owned-answer test asks
  for `plus` at order 1. Alec's answer on Felix agrees with the object's
  order: what a word names in a reading is of its event's order, and the
  identity one order above is formed later, over the events (operator
  catalogue §5.5). *Fix:* a reader finds a word's objects through the
  definitions table and excludes words by identity, not by order; the test
  asks for the object at the order `interpret` gives it.
* **AD. Rows counted as sentences** (the reduction-pressure and
  provisioning tests). They count every row of the store, and DEF rows are
  rows. *Fix:* count by kind; the later assertions of both tests then run.
* **AE. The attended field** (one test). After twelve words, the identity
  returned for `lynx` does not occur in the published carrier, which
  carries objects (§17.3). *Fix:* a failing probe shows whether the lookup
  returns the word or its object; the test names the word's object through
  the table.
* **AF. A nested relation with fewer than three references** (the
  partition-isolation test) is a runtime defect of the closing. *Fix:*
  isolate it with a failing probe.
* **AG. The graph-release gate (S) is still red, and the memory is the
  finding.** HEAD passes at 5.91 GiB. The candidate stops at 8.30 GiB under
  the guard, and its one unguarded run grows to 17.01 GiB before it raises,
  because clause recovery asks the thought registry for `part`, which a
  small model cannot reserve. There are two defects. Clause recovery
  depends on the thought capability; the kinds of relation it reads, part
  and implies, are identities of Z's kind, defined by their rows, and do
  not wait on the thought registry. And the growth itself, to twice HEAD's peak, is not explained; the
  candidate's reconstruction also peaks higher, 7.72 GiB against 6.96, and
  one of its seeds stopped at 8.01. *Fix:* after the first, measure what
  grows from one sentence to the next, by owner, and remove it. The gate
  passes under the unchanged 8 GiB guard.
* **AH. An ended clause read through the concept inventory** (the native
  predictor-context observation). `program_meaning` asks the inventory for
  a reference that is an ended clause, which has an identity and a row in
  LTM and no inventory row. *Fix:* an ended clause's point is read from LTM
  (`point_of_row`), and a concept's from the inventory.
* **AI. Five failures carried from round 2** *(measured at HEAD on
  2026-09-30)*. Four of them pass at HEAD, so they are this candidate's:
  (a) the detached reverse, whose root has no gradient by the time the
  student is asked for; (b) the normal-policy path, which indexes stale
  one-row metadata; (c) the identity-candidate check, which reaches a field
  of neither one nor three slots; and (e) the part-width change, which
  compiles two graphs where one is required. The fifth, (d), provisioning's
  what-episode left unreset, fails at HEAD too. The FineWeb checkpoint test
  and the attended-field test, also carried, pass at HEAD, and are AB and
  AE. So twelve of the sweep's thirteen failures are this candidate's.
  Claude's earlier reading, that (a) was item 7.5's design and (e) compiler
  work for item 1, is withdrawn: both pass at HEAD, which has item 7.5.
  *Alec's answer (§19.5): all are fixed here.*
* **AJ.** Not item 7's, found meanwhile: the sentence pair costs its second
  trial after the optimizer has stepped on the first
  ([6.9 plan §3.12](../plans/2026-09-29-item-6-9-xor-grammar.md#312-what-the-answer-is-shown-in-training)).
  It is 6.9's question 6.

### 19.4 Order of work

1. Z, and with it the first half of AG; then AA's failing probe. The XOR
   proofs come first.
2. AB, AC, AD and AE, each with its failing probe or its port recorded.
3. AF, AH, and all five of AI.
4. The second half of AG, the memory.
5. The XOR table with every proof named, the slow ones included, on HEAD
   and on the candidate; MM_grammar's ten full runs; tests 29 and 31 twenty
   times each in fresh processes; the full sweep; stop for review.

### 19.5 Questions for Alec, and his answers (2026-09-30)

1. *The identity of a predicate made by a grammar operation (Z).* Alec:
   "The definition of a word gets an identity in LTM which is the
   utterance. Similarly, identity has as its definition its every
   occurrence in every sentence (references can be added to the slots to
   tie identity together). There is nothing other to it, but if expedient,
   you can build a cache to those elements in LTM to make finding them
   easier." Written into Z: such an identity takes no inventory row, it is
   referred to by the slots of the rows it occurs in, and a cache may find
   them. A DEF row's own identity is already its utterance, the store
   occurrence at which it was written.
2. *The failures carried from round 2 (AI).* "Let's fix everything. Unless
   there are good reasons why not." All five are fixed in this pass, and
   four of them are this candidate's in any case. Where a repair turns out
   to need work outside item 7, Codex stops and says why.

## 20. Hand-off to Codex (Claude, 2026-09-30): what to change after review round 4

Read §19 and Alec's answers in §19.5. The rules of §16 stand: each repair
has a failing probe saved before it; no seed, threshold, guard or
configuration is changed; no protected assertion is weakened; every port
keeps its old and new body; nothing is committed. XOR comes first: the XOR
table, every proof and the slow ones by name, on HEAD and on the candidate
before anything is changed and after each step below that touches
composition or the closing.

1. **Z, with the first half of AG.** A closing takes no inventory row for
   what it refers to. An identity is its occurrences in LTM, tied by the
   references in the rows' slots (§19.5). The predicate of a relation made
   by a grammar operation is one identity for each operation, referred to by
   the relation slot of every row it occurs in, found again through a cache
   from the operation to it, and given no inventory row. A phrase that a row
   refers to is the same, and its point is read from its row. The kinds of
   relation that clause recovery reads, part and implies, are identities of
   this kind and do not wait on the thought registry. A symbolization that
   needs an inventory row when none is free is refused and leaves nothing
   behind (§17.3), and the reading goes on. *Tests:* a derivation with
   `part` under `disjunction` ends in MM_grammar without taking a row;
   MM_grammar's ten full runs all complete; tests 29 and 31 pass in twenty
   fresh processes each; the graph-release gate reaches its assertions.
2. **AB.** Every allocator is restored before any knowing field. *Tests:*
   the synthesized-answer, old-clock and FineWeb checkpoint tests.
3. **AC.** A reader finds a word's objects through the table of §17.4 and
   excludes words by identity, not by order. The owned-answer test asks for
   the object at the order `interpret` gives it.
4. **AD.** The two tests count stored sentences by their kind, and their
   later assertions run.
5. **AE.** A failing probe shows whether the attended-field lookup returns
   the word or its object; the test names the word's object through the
   table, or the runtime is repaired.
6. **AF.** The nested relation with fewer than three references, isolated
   and repaired.
7. **AH.** An ended clause's point is read from LTM, and a concept's from
   the inventory. The native predictor-context observation completes.
8. **AI.** All five: the detached reverse, the normal-policy metadata, the
   identity-candidate field, the part-width graph count, and provisioning's
   what-episode (HEAD's as well). Where a repair needs work outside item 7,
   stop and say why.
9. **The second half of AG, the memory.** Measure what grows from one
   sentence to the next in the graph-release gate, by owner, and remove it.
   The gate passes under the unchanged 8 GiB guard, and the eight
   reconstruction seeds complete under it.
10. **Receipt.** The XOR table with every proof named and the slow ones
    included; fifteen runs of the MM_20M_xor exact round trip on each tree
    (§19 AA); MM_grammar's ten full runs; tests 29 and 31 twenty times each
    in fresh processes; the explicit gates; the full sweep. Then stop for
    review.

Not in this pass: item 6.9, which holds XOR_grammar, the comparison of a
sentence's two trials (awaiting Alec), and the cause of AA.


## 21. Review round 5 (Claude, 2026-09-30, on the candidate after §20)

Codex's receipt is `doc/benchmarks/2026-09-30-item7-review-round4/`, "handed
back for review with this failure cluster unresolved". Claude's runs used a
copy of the working tree taken at 10:34 on 2026-09-30, after Codex's sweep
had finished, and the clean archive of HEAD `1678ee1f`, which has not moved
since round 4, so round 4's HEAD runs stand.

**Not accepted yet, on one finding.** Every step of §20 is in and does what
it asked, and all thirteen of round 3's sweep failures now pass. One repair,
Z's, left the predicate `part` with two identities, and three cases that
pass at HEAD or in round 3 fail on it (AK). It is a small repair.

### 21.1 The status of XOR

Run by Claude on the candidate, unseeded, once each unless said otherwise;
HEAD's column is round 4's (§19.1), on the same commit:

| proof | HEAD | candidate |
|---|---|---|
| grounded XOR, six cases | 6 pass | 6 pass |
| XOR_exact curriculum, with the rest of its file (ten cases) | 10 pass | 10 pass |
| MM_xor and MM_grammar early-stop gates, with their file (seven cases) | 7 pass | 7 pass |
| XOR_exact CLI, crisp output and reconstruction | both crash, `int(None)` | both pass |
| XOR_grammar, class accuracy and reconstruction | both fail, capacity crash | both fail, item 6.9's |
| SPNN XOR training | pass | pass |
| configuration matrix, XOR cases | 2 pass | 2 pass |
| reconstruction round trip, XOR cases (five, slow) | 5 pass | 5 pass |
| MM_20M_xor exact round trip, fifteen runs | 14 pass, 1 fails (.5) | 13 pass, 2 fail (.5); peak 4.7 GiB |

Codex's own final tables give fourteen of fifteen on each tree.

### 21.2 What is right

* **Z.** A closed sentence's address is its LTM occurrence, as a
  definition's is; a grammar predicate and a referenced phrase take no
  inventory row, and a phrase's point is read from its row through a cache
  from identity to row. A full inventory refuses the optional symbolization
  all-or-nothing and the row is still written. The capacity crash is gone:
  MM_grammar completes ten of ten full runs on both trees, and tests 29 and
  31 pass 140 of 140 in Codex's fresh processes and 35 of 35 in Claude's
  (five for each of the seven cases).
* **AB to AF, AH and AI, as asked.** Every allocator is restored before any
  knowing field; a word's objects are found through the definitions table,
  with no order assumed; stored sentences are counted by their kind; the
  attended field names the word's object through the table; the nested
  relation's predicate gets its unasserted occurrence; an ended clause's
  point is read from LTM. All five of AI pass. The live root that AI (a)
  restores is exposed only when no training sentence run is active, so the
  cut at the concluded idea holds in training.
* **AG.** The support read is recomputed in backward, sharing ties as the
  dense reduction does; only the selected properties and the predicate
  columns present in the input are read. The graph-release gate passes at
  7.45 GiB under the unchanged 8 GiB guard (HEAD 5.91). All eight
  reconstruction seeds complete, the candidate peaking at 5.73 GiB against
  HEAD's 6.96, with mean after-training error .1183 against .1112.
* **The record.** A failing probe precedes every repair, every port keeps
  both bodies, no seed, threshold, guard or configuration changed, and the
  three failures are reported with their assertions unchanged.

### 21.3 Findings

* **AK. One predicate, two identities** *(measured)*. Z gives a grammar
  predicate an identity with no inventory row, `predicate_identity(name)`,
  and the closing writes it into the relation slot. The thought registry
  still binds `part` (and `clause:implies`) to a frozen concept it mints in
  the inventory when it is installed, and forms its meanings with that one:
  `form('part', ...)` and `Language.program_meaning` put `('sym', 1)` in the
  relation slot where the stored row has
  `('sym', predicate_identity('part'))`, and the two predicate points differ
  as well. So a stored relation and the same relation formed for a question
  or an expectation never agree in their predicate. Three cases fail:
  `test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm`
  in both its forms, which pass at HEAD (run by Claude), and
  `test_item9b_interpret.py::test_selected_generic_grammar_ends_the_interpreted_kinds`,
  which is item 7's and passed in round 3. Queries that find part rows by
  their kind still work (the reasoner climbs `REL_PARTOF` rows); what
  compares meanings does not. *Fix, by §19.5 ("There is nothing other to
  it"):* one identity for each predicate. The registry's meanings for part
  and implies carry the identity the closing writes, with the same point,
  and the registry takes no inventory row for them either, so the frozen
  concepts it mints for them go, and with them the second path by which
  `predicate_kind` recognises part through the registry's frozen names.
  *Test:* the three cases pass unchanged; a relation written by the closing
  and the same relation formed by the registry have equal references and
  equal predicate points; a model whose inventory is full still forms a part
  question.
* **AL. MM_grammar's learning** *(observation, not a finding)*. Median
  ending error is .070 at HEAD and .134 on the candidate; HEAD ends below .05
  in five runs of ten, four of them near zero, the candidate in two, none
  near zero. Ten runs a tree do not separate them (Mann–Whitney, p = .31).
  The candidate after Z alone did better (median .023, six below .05), but
  that campaign ran before AI (a), and the measurement trains through a raw
  forward, which AI (a) changed from a detached root to a live one, so the
  two candidate campaigns may not train the same path. Item 6.9 measures
  MM_grammar's ten runs again after its step 5; the receipt keeps the table.
* **AM. The exact round trip (AA)** *(observation)*. It stays item 6.9's.
  Its runs are in 21.1 and in every receipt's table.

*Not item 7's, for the items named:*

* Test 27 checks that a definition's `.when` is fixed and that reading it
  again refreshes recency without a new row; it looks nothing up by `.when`,
  and nothing in the runtime reads a row's `.when`. Item 5.5 replaces the
  row's `.when` by its document and sentence index and tests the lookup
  ([5.5 spec](2026-09-30-occurrence-tense-aspect.md)).
* `get_stm_chain`, a legacy adapter, reads the store's recency across all
  batch rows; the predictor already uses its per-row view. Item 5.5's
  recency within a document replaces it.
* `predicate_identity` derives the identity from the operation's name. When
  the catalogue's rule 8 lands (the name never consulted), it becomes a
  declared property of the rule.
* The graph-release gate passes 1.5 GiB above HEAD's peak.

## 22. Hand-off to Codex (Claude, 2026-09-30): what to change after review round 5

The rules of §16 stand. Nothing is committed.

1. **AK.** One identity for each grammar predicate, shared by the closing
   and the thought registry: the registry's meanings for part and implies
   carry the identity the closing writes, with the same point, and the
   registry mints no inventory concept for them. `predicate_kind` then has
   one path. The three failing cases are the saved probe. *Tests:* §21 AK.
2. **Receipt.** The three cases; every item 7 case; the thought and
   reasoning files; the XOR table with every proof named and the slow ones
   included, with fifteen exact round trips on the candidate; MM_grammar's
   ten full runs, since the registry's rows are freed; the graph-release
   gate; one source-matched full sweep. Then stop for review.

## 23. Review round 6 (Claude, 2026-09-30, on the candidate after §22)

Codex's receipt is `doc/benchmarks/2026-09-30-item7-review-round5/`.
Claude's runs used a copy of the working tree taken at 11:30 on
2026-09-30, after the AK repair was frozen, and HEAD's runs of §21.

**Accepted, provided Codex's full sweep is green.** AK is repaired as §22
asked, and nothing else changed. Green means that every case of the
source-matched sweep passes but the declared skips and the one expected
failure, the three AK cases among them, with no resource stop, and that
Codex's explicit gates agree with 23.2. Item 7 may then be committed and
pushed.

### 23.1 AK

The thought registry binds `part` with no inventory row: its identity is
the closing's `predicate_identity('part')` and its point the shared
`predicate_point`. `clause:implies` is no longer minted, `clause_reference`
returns that identity for part and implies, and `predicate_kind` has one
path. The three cases pass unchanged, and so do Codex's six new checks: the
reference and the point shared by the closing and the registry, for part
and for implies; no inventory row for either; a part question formed at
full inventory. The three ports keep their tests' intent: the shared
meaning, now with the identity's code where an inventory vector was; the
refusal of a missing VP, moved to `equal`, which keeps an inventory row;
the all-or-nothing capacity probe, with two inventory VPs beside the
row-free part.

### 23.2 The status of XOR and the measurements

Run by Claude on the candidate, unseeded; HEAD's column is §21.1's:

| proof | HEAD | candidate |
|---|---|---|
| grounded XOR, six cases | 6 pass | 6 pass |
| XOR_exact curriculum, with the rest of its file (ten cases) | 10 pass | 10 pass |
| MM_xor and MM_grammar early-stop gates, with their file (seven cases) | 7 pass | 7 pass |
| XOR_exact CLI, crisp output and reconstruction | both crash, `int(None)` | both pass |
| XOR_grammar, class accuracy and reconstruction | both fail, capacity crash | both fail, item 6.9's |
| SPNN XOR training | pass | pass |
| configuration matrix, XOR cases | 2 pass | 2 pass |
| reconstruction round trip, XOR cases (five, slow) | 5 pass | 5 pass |
| MM_20M_xor exact round trip, fifteen runs | 14 pass, 1 fails (.5) | 13 pass, 2 fail (.5); item 6.9's (AA) |

Tests 29 and 31 pass 35 of 35 in fresh processes; the five files that
exercise AK pass 81 of 81; documentation links pass. Codex's ten
MM_grammar runs have median ending error .058, five below .05, against
HEAD's .070 and five: AL is closed.

### 23.3 Left for later

* `equal` still has two identities: the closing writes an equality as two
  part rows under `part`'s, and the registry forms an `equal` question with
  an inventory VP of its own. Left to the grammatical operators update
  (Alec, 2026-09-30), with Claude's probe as its test
  ([catalogue §6.3](2026-09-29-operator-catalogue.md#63-the-sentence-that-states-a-definition)).
* `GrammaticalQueryRegistry` is a class nothing refers to, here and at
  HEAD; it goes under the no-legacy rule.
* The notes of §21 stand for the items named there: test 27 and the row's
  `.when`, `get_stm_chain`'s recency across batch rows, and the
  name-derived identity.

## 24. Review round 7 (Claude, 2026-09-30, on the swept candidate)

Codex's receipt is the same folder, now "complete red receipt for Claude's
review". The sweep completed all 5,122 cases: 4,793 passed, six failed, 322
skipped and one expected failure, with no resource stop; every failure of
rounds 3 and 4 passes. §23's condition, a green sweep, is not met, so §23's
acceptance waits on the six.

**Accepted once the six are ported (§25).** They are test code, not the
runtime, and nothing in the runtime has changed since the sweep.

### 24.1 The repair after §23

The first item-7 run found one more AK edge: a three-slot closing whose
predicate slot holds the row-free identity of part or implies still looked
it up through the thought registry and the inventory. The repair takes the
shared point for that identity, as the composed path does, and asks the
registry nothing; four probes and the unchanged failing test cover it,
with and without a registry. It is what §22 asked of part and implies, and
it is right.

### 24.2 The six failures

All six assume that every reference of a relation has an inventory row.
Four stop in `index_fixtures.terminal_model_index` and one in
`test_accessible_mind`'s own postings, calling `_existing_row` on the part
predicate, before any assertion of their own runs. The sixth,
`test_renamed_native_vocabulary_preserves_checked_relation_answers`,
asserts that all three references change with a renamed vocabulary and
then copies each value into its inventory row; the predicate is a grammar
operation, the same identity in every vocabulary, which is right, and has
no row. Claude ported the three sites in a scratch copy: a row-free
predicate primes no row and posts no term, and the renaming test expects
the operands renamed and the predicate shared, copying values only into the
operands' rows. All six then pass with their own assertions of evidence,
retrieval, ownership and oracle isolation, and the six files that use the
fixture or hold the ported tests pass 59 of 59.

### 24.3 Measurements

* The final XOR table has 44 passes of 49: XOR_grammar's two gates, item
  6.9's, and three of fifteen exact round trips (.75, .75, .25), AA, item
  6.9's. The three are consecutive trials and one decodes a quarter of the
  inputs, a new low; 6.9's receipt tracks it.
* MM_grammar's ten final runs have median .107; the twenty runs since AK
  against HEAD's twenty give p = .38 (medians .086 and .039). Not a finding;
  item 6.9 measures it again.
* The graph-release gate passes at 7.63 GiB, item 7's 202 cases and the
  thought and reasoning files' 315 pass.

## 25. Hand-off to Codex (Claude, 2026-09-30): what to change after review round 7

1. **Three test ports**, each with its old and new body in the ledger:
   `test/index_fixtures.py::terminal_model_index` primes only references
   that are not row-free predicates (`predicate_relation(reference[1]) ==
   'operator'`); `test_accessible_mind.py::test_normal_what_effect_enters_recency_and_detached_knowing`
   gives a row-free predicate no posting term; and
   `test_arithmetic_isolation.py::test_renamed_native_vocabulary_preserves_checked_relation_answers`
   asserts the two operands renamed and the predicate shared, and copies
   values only into the operands' rows. Every other assertion stays.
2. **Checks:** the six cases; `test_accessible_mind.py`,
   `test_query_work_budget.py`, `test_query_executors.py`,
   `test_replacement_kernel_contracts.py`, `test_taxonomy_policy.py` and
   `test_arithmetic_isolation.py`; every item 7 case; the documentation
   links. No new sweep: the runtime is the swept one.
3. **Then item 7 is accepted:** commit and push it, with the WikiOracle
   submodule bump, and begin item 6.9 from its hand-off
   ([6.9 plan §8](../plans/2026-09-29-item-6-9-xor-grammar.md#8-hand-off-to-codex-after-item-7-is-accepted)).
