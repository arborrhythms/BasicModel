# Two truths in LTM: ideas collapse, relations refer

> **Status:** specification, 2026-09-16, written by Claude from Alec's
> decisions in conversation on 2026-09-16, including the answers to the
> four questions the first draft left open and the fusion-over-references
> decision (§9 records them). For Codex to implement in a new session;
> Claude reviews the implementation afterwards. Governs the conceptual
> content of every LTM row, the grammatical seal that writes it, and the
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
row is written when the S seals, so anything that refers to it (a later
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
survive the seal. Dharmakīrti's exclusion of the contradicted cognition
from valid cognition is untouched by this: a heterogeneous whole is
perceived, not inferred against itself.

Consequences (decided):

- **Two symbols per concept.** Every concept has a positive symbol `c⁺`
  (the concept present) and a negative symbol `c⁻` (its negation present),
  both presences in `[0, 1]`, sharing one concept row: the symbol index is
  `i ↔ (i⁺, i⁻)` and the concept's distributed code is stored once. The
  four corners read directly from the pair — `t = c⁺(1 − c⁻)`,
  `f = c⁻(1 − c⁺)`, `both = c⁺c⁻`, `neither = (1 − c⁺)(1 − c⁻)` — so
  luminosity (catuṣkoṭi coverage) is measurable per symbol. **What is
  stored and carried is the evidence pair `(c⁺, c⁻)`, never the derived
  corners**: `t` and `f` send both `(1, 1)` and neither `(0, 0)` to
  `(0, 0)` and lose exactly the distinction being added (Codex,
  2026-09-23). Negation is a
  presence, so the concept pyramid is monotone over positive presences with
  non-negative exponents, and a negated part is an edge to `c⁻`
  ([accessible mind §2.2](2026-09-20-accessible-mind-subsystems.md#22-parallel-conceptual-knowing-zeroth-order);
  [Architecture, item 11](../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)).
- **The channels measure presence of supporting evidence among the
  parts, not the proportion of the whole.** The ball is both black and
  white whatever the share of each; how much is the meronomy's question
  (`.where` extents), not the symbol's. So the snap reads each channel by
  an **evidence fold over slots** of the rectified projection onto the
  atom (`relu p_n`) and onto its negation (`relu −p_n`): the **union over
  admitted slots**, `1 − ∏(1 − e_n)`, where a slot is admitted to a
  channel only above a noise floor `τ`, `e_n = relu(|p_n| − τ)/(1 − τ)`
  (implemented at `e917d9c`; **confirmed by Alec, 2026-09-23**, with the
  reservation that a calibrated floor is archaic: the admission should be
  a regularized weighting over the input space. Note that `relu(p − τ)` is
  the soft-thresholding operator, the proximal map of an L1 penalty of
  weight `τ` on the admitted evidence, so the floor already *is* an
  L1-regularized weighting; what is archaic is that its strength is
  measured offline. **Decided (Alec, 2026-09-23): keep the calibrated
  floor** — a rank-based or trained admission was judged more machinery
  than the objection warrants; `conceptEvidenceFloor` stays a parameter,
  recalibrated once in the corrected units below. *Superseded by todo item
  11b (decided, Alec 2026-09-23): order-0 concepts read the towers'
  feature memberships with their own signed weights by pervasion within an
  occurrence, so unrelated features read zero and no floor exists; the
  union across occurrences inside the subject is the readout, where both is
  born, and XOR is the both corner of one property.*) Every rectified fold is
  biased upward by noise on both channels — noise has a positive and a
  negative part, and rectification hands one to each — so admission comes
  first, as the taper does for the pyramid (Architecture, settled point 4);
  the union is then the existential "some part carries this", the same
  sigma the pyramid uses, so order 0 and order 1 read alike. Max is
  withdrawn (Alec): its extreme-value bias grows with the slot count, so
  the channel with more slots wins from noise, and one spurious slot in the
  other channel manufactures both. The slot-mean is ruled out: two slots at
  +1 and −1 average to `(.5, .5)`, a quarter in every corner, which is not
  both (Codex). `τ` is a `model.xml` parameter set from a measurement, the
  distribution of projection magnitudes for unrelated content in a trained
  model (item 10, finding 6), not guessed. The tests state the meaning:
  slots `+1, −1` → `(1, 1)`, both = 1; all slots inactive → `(0, 0)`,
  neither = 1; eight slots at .1 under `τ = .2` → `(0, 0)`.
- **The read is in the field's own chart** (decided, Alec 2026-09-23,
  from the landing review). A projection in hypercube-diagonal units
  divides by `√D`, which is right for cube-valued slots of norm up to `√D`
  and wrong for the unit-ball codes the settled field carries (measured
  slot norms .16–.97 at `D = 1024`): the presence ceiling for a perfectly
  aligned slot is then `1/√D = .03`, the union of eight slots at most .22,
  and the use floor .5 and discovery threshold .8 are unreachable by
  construction. The read is cosine times slot norm for unit-ball codes,
  with the `√D` divisor only where slots are cube-valued; the floor `τ`
  is recalibrated in those units.
- **An occurrence is an extent, not a tile** (decided, Alec 2026-09-23:
  "let's see how it pans out in practice"). The field's occurrence axis
  groups the positions inside one subject's extent — a word's or a
  whole's `.where` from the towers — with the pair kept per position
  inside it. Parts of a whole are at different positions of one extent at
  one time; at a single tile there is one code, so "co-present at a tile"
  is alternative readings of one code, not parts. Negation is scoped to
  the extent (pervasion over its parts).
- **Witnessing writes positive parts only** (Alec, 2026-09-23: "parts are
  percepts; absence of a percept is possible (0), but the opposite of a
  percept is not"). Co-presence discovery assigns the concepts present in
  an extent as parts; it never writes a negated part from counterevidence,
  since at the percept level there is nothing to witness but presence and
  absence. A negated part enters a definition only by learning or by
  testimony (the seal). The zeroth-order `c⁻` remains a computed reading —
  the definition contradicted by what is present — never a witnessable
  part.
- **Concepts that require both poles of one constituent exist, and the
  pair represents them.** A whole never needs `A` and `¬A` at one
  position, and a kind over both is a tautology; but across the positions
  of an extent a definition may require a constituent to be present and
  absent — striped, spotted, pied, "black and white" — and XOR requires it
  across its conjuncts. A conjunction addressing both poles of `A` computes
  `c⁺_A · c⁻_A`, the both corner itself; a union over both poles computes
  "A has been observed at all", the complement of neither. So the four
  corners of a constituent are all definable as parts, and heterogeneity
  can be a defining feature (the soccer ball's black and white; Alec's
  question of 2026-09-23).
- **A composed concept's `c⁻` is the De Morgan dual fold, never
  `1 − c⁺`** (which would erase both and neither). A literal over part `A`
  is the pair `(c⁺_A, c⁻_A)`, swapped for a negated part. A kind, a union
  over `W_σ`, has `c⁺ = union of the literals' positive channels` and
  `c⁻ = conjunction of their negative channels` (`¬(A ∨ B) = ¬A ∧ ¬B`); a
  whole, a conjunction over `W_π`, has `c⁺ = conjunction` and `c⁻ = union`
  of the respective channels. The same non-negative exponents serve both
  channels; only the chart is swapped (`log u` against `log(1 − u)`).
  Checks: all parts neither → `(0, 0)`; all parts both → `(1, 1)`; in the
  kind `A ∨ B` with `A` both and `B` absent, `(1, 1)`: the kind is present
  through `A` and its negation through `¬A ∧ ¬B`. The reverse is the
  transpose of each fold on its own channel; the taper ranks a row by its
  evidence `max(c⁺, c⁻)`, so a both-row ranks high; the symbolic readout
  carries the pair. Cost: two further scatter passes per rung.
- **All percepts are positive; evidence against comes through the
  definition** (Alec, 2026-09-23). A zeroth-order concept's atom has
  positive and negative components over the percept dimensions, and a
  white code projects negatively onto black's atom, so white at a position
  is evidence against black there. No negative percept exists. For the
  ball this reads black present and non-black present, which is the honest
  report of the presentation, and white counts as non-black without harm.
- **Evidence against is scoped to the subject's extent.** Read over the
  whole field, every concept's negative channel is fed by everything else
  present — the mat is non-cat — so at order 0, where the subject is the
  presentation itself, "both" is true of every concept that shares the
  scene and discriminates nothing. Wherever a subject exists, the negative
  channel is read only within its extent. In the terms of Tibetan debate
  this is pervasion (*khyab pa*, *vyāpti*): the cat does not pervade "cat
  and mat" (Alec, 2026-09-23). The corners are the four cases of
  pervasion over a subject's parts — the predicate pervades (true), its
  negation pervades (false), neither pervades because the parts differ
  (both), nothing is known of the parts (neither). A predicate over a whole
  reads its parts; a conjunction with a negated part in the pyramid reads the
  negated part at the positions of its positive parts, which is the
  positional grounding of item 11a. So the pair is kept **per position**
  through the zeroth-order read and the pyramid's rungs, and the union
  over positions is taken last, for the symbol's own activation. At the
  production tile count this is eight evaluations of a store of about
  fifteen rows, not a cost.
- **A row stores the pair `(c⁺, c⁻)`.** The stored scalar `t − f` (§3.4,
  §9 item 2 as first decided) cannot hold both; the corners are derived at
  read time. Trust stays on the row; the edge still carries none.
- **"Both" is asserted by composition, never by a sentence alone.** "The
  ball is black and white" seals as an idea whose object has parts under
  two predicates; a single assertion "the ball is black" from a source
  raises `c⁺` with that source's trust and leaves `c⁻` to the parts that
  differ or to another source.

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

**Clause-level seal as grammar.** The seal is the reduction of an S to
its start state, with the row write as its side effect. Making that
reduction available inside a sentence is the clause-level seal:

- `NP → S` for an absolute clause: the clause fuses to a point and enters
  the enclosing VP as its modifier. No reference is pushed; the clause's
  row exists and the enclosing derivation records it.
- `NP → REF(S)` for a relative clause: the clause holds its three slots,
  writes its relation row, and pushes the row reference onto the STM as
  the NP. The enclosing S then cannot fuse (§1) and seals as an operator
  relation over that reference. The complementiser "that" is the surface
  form of this rule.

The depth floor that holds a relative sentence at depth 3 becomes
clause-scoped: it applies to the relative clause's own fold and to every
enclosing fold that received a reference. Embedded clauses seal
bottom-up, so a referenced row always exists before the referring S
seals. Mid-sentence row writes are host-side, like the existing
sentence-boundary hooks; the compiled reduce sweep carries only the
clause-scoped floor and the reference push.

The plan's §2.1 is rewritten to this rule (§8). The "compound-reference
prediction" gap in the plan's §8.3 is closed by definition: an NP slot
may hold any row, and because ideas, concepts, symbols and relations
share one index, no new reference mechanism exists.

## 3. Storage contract

### 3.1 Idea rows (decided)

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

### 3.3 The seal decides (decided)

At every S seal, sentence or clause:

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

**One writer for relations.** The seal is the only relation writer. The
routing in `ConceptualSpace._route_learned_relation`, which sends
"reducible" relations into the WholeSpace META taxonomy
(`insert_relation`) and "ineffable" ones into a store, is deleted. The
distinction it served does not exist under §1: operands are rows and a
row exists for every S, so nothing is ineffable. The `truth_criterion`
learn-score gate is retired with it; rows are about reference, and trust
comes from provenance (§4).

### 3.4 META and the taxonomy (decided)

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
one structure, read by union — and it has three writers: the seal's part
rows between concept rows, the META bindings, and discovery from witnessed
substitution — the same context on different occasions — for words and
percepts
([Architecture](../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)).
Trust stays on the LTM row; the edge carries none. *Amended (Alec,
2026-09-23, §1.1):* the row stores the evidence pair `(c⁺, c⁻)` and the
tetralemma tuple (t, f, both, neither) is derived from it at read time; the former
stored scalar `t − f` cannot hold **both**, which is a compositional fact.

**Word to object is a direct index.** Translation from a word to its
object never traverses the taxonomy. The word-keyed binding table
(`References`, `deref(word)`) is the fast path and stays the only one;
the taxonomy is for generalisation.

### 3.5 Object permanence: a word may translate to an earlier occurrence (decided, 2026-09-21)

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
13. **Clause-level seal and upward relativity.** Inside "he said cats
    are animals" the relative clause seals before the outer S, its
    reference is on the STM as the NP modifier of "said", the outer S is
    held at depth 3 and seals as `REL_OPERATOR`; inside "he said she
    said it's so beautiful" every clause fuses and the outer S seals at
    depth 1.
14. **META and translation.** A word bound to an object yields one META
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
  clause-level seal, upward relativity, and the one write per S.
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
4. **Fusion exists; the seal moves.** The reduce sweep's depth-1 fold is
   the fusion. The open mechanism was the clause-level seal, decided as
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
