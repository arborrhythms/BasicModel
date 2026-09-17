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
index, not a store; it holds no trust of its own. The tetralemma trust
tuple (t, f, both, neither) is computation-time only; the stored scalar
is `t − f` as today.

**Word to object is a direct index.** Translation from a word to its
object never traverses the taxonomy. The word-keyed binding table
(`References`, `deref(word)`) is the fast path and stays the only one;
the taxonomy is for generalisation.

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
   index. The tetralemma tuple is computation-time only; the learn-score
   gate is retired.
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
