# Thought design notes: derivation over chunks with trust (Claude, 2026-10-26; exploration at Alec's invitation, nothing decided)

Alec's outline (2026-10-26): `intuit()` generates a guess, `ask()` pushes
a question through an unresolved reference, `query()` pulls an LTM or
concept reference into STM, all available to a chooser that has access to
STM; reasoning from A to B uses higher-order relations to reach logical
conclusions from the premises in STM; a few such forms formalized as
operators would force the chooser to make logical derivations over ideas
and relations that each carry a trust score. What follows explores the
solution space in code names. It feeds the thought design (item 3 design
list, item 1) and `intuit` (item 2), which Alec set aside together.

## 1. The workspace: premises are chunks, and the serial slot becomes one

The eight shared chunks (`STMChunks.py`: `chunk_refs`, `chunk_identity`,
`chunk_stack`, `chunk_read`, `allocate_chunk`) are the premise set. A
premise is a chunk whose reference is an LTM0 idea (a proposition with the
evidence pair its closing gave it) or an LTM1 relation (a rule, with its
two lanes). Trust is therefore already attached to everything a chunk can
hold; nothing new is stored for it. The one structural change is the
deferred one: the serial slot becomes the episode's *focus chunk*. An
episode allocates its focus with `allocate_chunk` and marks it on
`chunk_stack` so eviction cannot take it while the episode is open; every
operation reads its operands from chunks and writes its result into the
focus; `ThoughtRecord.address` names the chunk written. `descend` pushes a
new focus for a subgoal and `return` pops it, writing the returned binding
into the parent's open slot. Eight is the bound on live premises plus
focus chain; an intermediate result not on the stack can be evicted, which
is forgetting a lemma, recoverable by `query` or by deriving it again.

## 2. Two inference directions already exist in embryo

`intuit` (spec §5) is forward chaining in parallel: every rule whose
antecedent the knowing covers fires at that degree, the consequents are
unioned into the knowing, nothing serial is written. `ask` with an open
slot is backward chaining: a question is an open reference, the nested
context tries to fill it, and `descend`/`return` are the proof tree's
edges. The missing middle is a serial operator that *produces* a
conclusion from a premise and a rule. `isImplied` today is a judgment: it
takes both A and B and returns the pair `min(link, premise)`. Derivation
needs the production form: given A in a chunk and a rule in a chunk, write
B into the focus with that trust.

## 3. The operators worth formalizing (a small set)

- **`infer(premise, rule)`**, forward. The rule is an LTM1 binary row with
  lanes: an if-then row (temporal references), an IS row (symbol to
  symbol) or a kind-level statement. Its antecedent matches the premise by
  equality (`equal`/`isEqual`) or by membership: the premise's subject is a
  member of the rule's kind, read along the Part edges by `ConceptIndex`.
  The consequent is written into the focus with the member substituted for
  the kind, evidence kind `inference`, pair `(min(t+_premise, t+_rule),
  ...)` by the lane rule below, `sources` the two references. One operator
  covers modus ponens, Barbara (two IS rows) and instantiation, with the
  match kind recorded.
- **`refute(premise, rule)`**, the contrapositive: the premise matches the
  consequent under exchanged lanes, the antecedent is written with lanes
  exchanged. Kept separate from `infer` so the chooser learns when each
  pays and the records read plainly.
- **`substitute(idea, equality)`**: an `isEqual` row replaces one term by
  the other inside an idea; the cheapest form of Leibniz's law.
- Existing and unchanged: `not` (lane exchange), `isTrue` (stored
  evidence for a description or row), `query` (best match into a chunk),
  `ask` (open reference, descend), `bind`, `exist`, `gain`; `intuit` as
  specified, with §4's amendments.

Transitivity over `isPart` is two `infer` steps over IS rows; the
higher-order rule applied to a lower-order case is `infer` with a
membership match; relation-over-relation rules (a rule whose antecedent
is itself a relation) match by the kind and the two references' identity
or membership, no further unification. That restriction is the honest v1
boundary.

## 4. `intuit` as proposer, the surface as its filter

The spec's scan over "the store's leaf-code index" refers to an index step
1 removed. The replacement falls out of the priming design: the surface
after the sentence lights the symbols in play, so the rules `intuit`
considers are the LTM1 rows whose referenced symbols carry excess, read
through `_csw_row_of`; firing degree is the meet of the knowing with the
antecedent's idea (`LTM0` one slot). Its output becomes visible to the
serial path in the simplest way: the strongest fired consequents are
written into chunks with evidence kind `intuition` and the firing degree
as a provisional pair. A guess is then a chunk like any other, and the
serial operators can confirm it (an `infer` chain replaces the provisional
pair with a derived one), test it (`isTrue`) or ignore it. Intuition
proposes, derivation disposes. Two provenances already in the spec,
observed succession and stated if-then, gain a third, derived (§6).

## 5. Trust algebra, computed not learned

Leaves carry learned trust: the lanes on rows, the closing's pair on
ideas, accumulated by evidence. Derivations compute: conjunction along a
path is the minimum of the supports (the minimum t-norm, which `isImplied`
already uses), alternatives for the same conclusion combine by maximum
(the union `intuit` uses), negation is the lane exchange, and nothing is
netted, so a conclusion derived both ways keeps both lanes and prompts
refinement as the corner rule says. A derivation's pair is a function of
its records, so it is replayable from the history alone. The chooser
learns none of this; it learns the path, credited through the R+A
comparison (`SentenceCredit.comparison`) as today, which is to say it
learns to find short, high-trust derivations within the budget.

## 6. What "forcing" can mean, two ways

Weak forcing is the grammar: a thought section that offers, at a question,
only `query`, `ask`, `intuit`, `infer`, `refute`, `substitute`, `not`,
`isTrue`, `bind` and conclude. Every path is then a sequence of sound
steps, and the chooser decides which premises and rules, not whether to be
logical. Strong forcing is at the answer: the closing accepts as an answer
only focus content that carries a pair, which only the sound operators and
direct recall produce; a provisional `intuition` pair is not accepted
until confirmed. Recall is a derivation of length zero with the row's own
trust. The two can be combined; strong forcing alone leaves the chooser
free to wander and pay.

## 7. Derived conclusions as memory

A conclusion the episode keeps can be written at the close as an idea with
provenance derived and the derivation's pair, so a later thought recalls
it instead of deriving it again. That is learning by thinking, and
forgetting values it by its own worth as any row. The alternative keeps
derivations ephemeral, in the history only. Alec's call when the design is
written.

## 8. Decisions this would route to Alec

1. Derived conclusions written to LTM0 (§7) or history only.
2. `infer` and `refute` as two operators or one with a direction.
3. The unification boundary for relation-over-relation rules (§3).
4. Weak, strong or both forcings (§6).
5. Whether `intuit` writes its top consequents into chunks (§4) or stays a
   knowing-only effect read by `query`'s ranking.

A worked line, for the shape of it: "Is Felix furry?" opens a focus with
an open truth slot; `query` brings "cats are furry" (kind-level) into a
chunk; `infer` needs Felix a cat and `ask`s it, descending; `query` finds
the IS row and `return`s its pair; `infer` writes "Felix is furry" with
the minimum of the two pairs; conclude. Five records, two rows read, and a
trust that names them.

## 9. Two routes to truth (Alec, 2026-10-26; captured for implementation time)

Alec: "'Is Felix furry' will focus attention to Felix, and condition that
with 'furry'. Based on implicit reasoning, that asks over the parts and
wholes of Felix. A complete pervasion of the object will look like ones
everywhere, contradiction will look like minus one everywhere, and we are
likely to get an answer which is a bit mixed (the outside is furry). That
will have to happen in order-0 space, I think. We can also query the
logical relations (LTM1 and taxonomy), which gives the reasoning chain
that you described."

Reading, in code names:

- **The implicit route is pervasion in order-0 space.** The question
  focuses attention on Felix (his row bumped, the knowing restricted to
  his parts and wholes as his definition edges and Part edges give them)
  and conditions it with the property's code. For each part, the meet of
  "furry" with what is known of that part, in the two lanes: the for pole
  everywhere is complete pervasion, the against pole everywhere is
  contradiction, and the honest answer is mixed, the outside furry and the
  inside not, so both lanes carry something. The aggregate over the parts
  is the answer's pair. This is the conceptual face already in the
  catalogue, `implies` as containment of regions (`part` on the codes) and
  `exist` reading the two poles of a value through
  `ThoughtConceptualCapability.extent`, applied to a knowing that attention
  has narrowed to one object; `intuit` restricted to that object's
  mereology is its parallel form. No LTM1 row is read.
- **The explicit route is the chain of §3.** `query` over LTM1 and the
  taxonomy brings the rows, `infer` and `ask` follow them, and the
  derivation's pair is the minimum along the path.
- **The two are judged together**, as implicit and explicit definitions are
  (Orders §10): agreement fills one lane, disagreement fills both, and
  that corner prompts refinement. Neither route is privileged; the chooser
  can take either or both within the budget, and the records say which.

Implementation questions for that time: how attention narrows the knowing
to one object's mereology (the definition edges of `_definitions`, the
Part edges, or the percept `partOf` order of the sorting item); how the
per-part pairs aggregate (minimum for pervasion, with the mixed case kept
as both lanes rather than averaged); and whether the implicit answer is
written into a chunk with evidence kind `pervasion` so the explicit chain
can confirm or contest it.
