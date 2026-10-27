# Item 3.0 step 5: orders learned, not written (Claude, 2026-10-26; for Codex, after step 4's port is committed)

**Status.** The fifth and last round of item 3.0 ([hand-off](2026-10-24-handoff-3-0.md),
"Item 3.0, in order", 5). It builds [Orders §10](../Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)
and the todo item 3 bullets "The definition: IS and its raise", "Return to
S0 by lowering", "Every symbol's order; folds with definitions" and
"Learning parts of speech", whose design record is the "Design record:
order in the parts of speech" paragraphs of todo item 3. Step-2 spec
[§39](2026-10-21-step2-one-ltm.md#39-standing-rule-amended-alec-2026-10-22-codex-continues-documenting)
governs. Code names are read from the published source `8b523874f9` plus
step 4's port. The item's one full sweep follows this round.

## 1. Decided (Alec; the record this round builds)

1. **Orders live in the concepts, not in the grammar's categories.** "We
   need not create an ordered grammar, because we have the rule that
   raises order." Parts of speech stay order-free and come from the
   grammar's positions and occupations (Orders §10; reconstruction spec
   §10).
2. **Definitions raise, by equivalence class.** An equality writes an LTM1
   row and changes no order. When a second, distinct member joins the same
   side, the shared referent is raised one order and the word is given the
   raised concept as its object. "The same side" is the symbol the mentions
   read to; distinct means two events at different places or times, or two
   different identities in MTM; "y is three. x is three." raises nothing,
   since x and y are bound to three itself; "Paris is the capital of
   France" is identity and raises nothing. "This is Felix, that is Felix":
   0 to 1. "Felix is a cat, Ciao is a cat": 1 to 2 (Orders §10; todo).
3. **Only NP + VP lowers.** A predication lands one order below its
   subject ("Felix runs" an occurrence at 0; "dodos died out" a change
   across members at 1; "water flows" stays at 0). The modal withholds the
   lowering (2026-10-23). A determiner selects one of a plurality and the
   phrase's order is that of what it selects, by lookup (Orders §10,
   amended 2026-10-22 and 2026-10-23).
4. **Order assignment is by lookup throughout.** `lift` and `lower` keep
   their sigma and pi folds; the stamp machinery gives way to reading the
   denoted concept's order, which the equivalence classes set. Raises carry
   no price; binds are credited through the derivative (Orders §10).
5. **Every symbol has an order; both the explicit and the implicit
   definitions are kept.** Item 3 closes four gaps: 0 doubles as
   "unknown"; a recorded order overrides structure unchecked; a relation
   row's handle always reads 0; a test-only formation API keeps an outdated
   ladder (Orders §10, "Every symbol's order"; todo). The match or mismatch
   of explicit and implicit definitions is a judgement in the two truth
   lanes, never netted.
6. **Learning parts of speech** follows the orders: proper nouns, then mass
   nouns, then count nouns where the language has them; occupations are
   recorded by position and order; until the learning lands, unit tests
   force parts of speech (Alec: "Unit tests can force parts of speech").
   3.5's two order-1 stand-ins (a determiner always outputs order 1; a
   bare name's order supplied by fixtures) end here.

## 2. What exists, in code names

- **A concept's order.** `ConceptAllocator.order_of(cid)` returns the
  recorded `reference_orders[cid]` if any, else `1 + max` over its `part`
  references' orders, else 0; `settle(cid)` copies it into `placement`
  without checking the record against the structure.
  `ConceptualSpace._concept_source_order(cid)` reads `store.order[row]` for
  an LTM0 occurrence and `order_of` otherwise; an LTM1 handle has no
  `store.order` and reads 0. `_row_order(row)` maps a codebook row to its
  concept's order, else to its `_order_caps` pool. `_csw_concept_row(order,
  cid)` *writes* `reference_orders[cid] = order` whenever a row is requested
  at a higher order than the concept has (`Spaces.py` ≈15455), the
  unchecked override. `Interpret` records every word at order 0
  (`reference_orders[word] = 0`, ≈195, ≈222); `MemoryConsolidation.
  consolidate` records a promoted token at 1 and a promoted reading at its
  idea's order (≈161, ≈168).
- **The raise.** `ConceptIndex.index_part_row(cs, part, whole)` adds a
  `part` edge from the parent one order above `part` (creating or finding
  the parent through `alloc.relate_idx[('part-kind', whole, order)]` and
  `_symbol_at`, which climbs by `singleton_concept` copying the codebook
  row), records `reference_orders[parent]`, adds it to `singletons` and
  `raised`, settles it and inherits forms. Its only caller is
  `ConceptualSpace` ≈16873, a public wrapper with no live caller.
  `PartClosing._write` marks `alloc.singletons.add(right)` when an asserted
  `equal` relation is written (≈197); nothing reads the mark to raise.
- **The stamp machinery.** `ReferenceContext.shifted_order(stamp,
  delta)`, `operation_order(rule, stamp)` (determiner modes `mint`/`bind`
  return 1, `kind` returns 2, else `stamp + rule.order_delta`) and
  `result_order(rule, operands)` (the head operand's stamp or the joined
  max, then `operation_order`); `order_delta` declared on `LiftLayer` (0),
  `AdverbLayer` (0), `LowerLayer` and `LowerUnaryLayer` (−1), the base
  `Layers.py` ≈2473 (0), and in the grammar rule record (`Language.py`
  ≈627, ≈1582, "+1 for lift, −1 for lower"); `binding_spec` requests a
  reference at `operation_order`; `resolve_order(value, identity, order,
  types)` selects an existing typed concept of that order for the word
  from `bank.types`. Callers: `ClauseJournal` ≈207 (`Clause.order`, which
  `PartClosing.prepare` writes into `record['order']` and
  `store.append_meaning(order=...)` into `LTM0.order`), `FormCentroids`
  ≈100, ≈132, ≈142. Leaf orders are already by lookup:
  `_ar_word_concept_orders` holds `_concept_source_order(A)` per staged
  word (`Models.py` ≈11340), and the chunk state carries `orders` and
  `grammar_orders` per chunk.
- **Parts of speech.** `PartsOfSpeech` keeps `counts[word][argument]`
  over the grammar's formal arguments (`f'{name}_argin{i}'`), `labels`,
  `distribution(word)`, `probabilities(words, like)` (the chooser's
  occupation table, `Models.py` ≈17697, ≈20643) and `observe_kept(program,
  binary_rules, unary_rules)` at the kept close (≈20584), walking the
  journal's actions. `frozen` stops recording; `collapse()` reports
  argument similarity.
- **Forcing in tests.** Eleven test files write `reference_orders[word] =
  1` for a name; thirteen touch `parts_of_speech`.

## 3. Orders by lookup

`ReferenceContext.denoted_order(rule, operand_orders, operand_ids, alloc)`
replaces `shifted_order`, `operation_order` and `result_order`, and the
`order_delta` attribute leaves every layer and the grammar rule record
(`Language.py` ≈484, ≈501, ≈627, ≈633, ≈1335, ≈1582, ≈1588, ≈1595). It
reads orders, never adds a declared constant to them:

- A leaf's order is `alloc.order_of(concept)` (unchanged).
- `lift` (`clause_form 'S'`, `head_role` 1): the subject's order minus one,
  floor 0; when the VP operand carries a modal alternative (the journal's
  `timeline` flag of ≈219, the chunk's `chunk_timeline`), the subject's
  order unchanged (decision 3).
- `verb` (`clause_form 'VP'`): the verb's order (Orders §10: verbs and
  nouns are of the same order; the VP is a difference over the subject's
  members, placed with its verb).
- `lower`, `lower_unary`, `definite_unary` (a determiner): the noun's order
  minus one when the noun is a kind (order at least 2), else the noun's
  order; the `mint`/`bind`/`kind` constants go (decision 3; see §9 item 1
  for the order-0 noun).
- `equal`: the left operand's order; a `part`/`whole`/`kind` relation: the
  whole's order; `bind`: the identity's order (a token is 1; a promoted
  symbol its record); `sum`, `conjunction`, `disjunction` and the other
  joins: the maximum of the operands, unknown if any is unknown; unary
  markers: their operand's.

`binding_spec` requests at `denoted_order`; `resolve_order` keeps its
contract (an existing typed concept of the requested order for the word);
`ClauseJournal` ≈207 and `FormCentroids` read `denoted_order`;
`Clause.order`, `record['order']` and `LTM0.order` keep their columns with
the looked-up value. The comments at `Language.py` ≈627 and ≈1577 saying
`lift` raises order are rewritten to say what the lookup does.

## 4. The raise, from the equality close

In `PartClosing._write`, at the asserted `equal` relation (≈197): the
right reference `right` is the side the mentions read to (a `SYMBOL`, or
the token an `OCCURRENCE` reads to through `mtm.token_of`). The allocator
keeps, beside `singletons`, `singleton_members[right] = (member reference,
token, value snapshot)` for the first member that joined. When a second
member joins the same `right`:

- *Distinct* means a different MTM token (`mtm.token_of(left) !=
  mtm.token_of(first)`) whose published value is not equal to the first's
  (so "y is three. x is three.", where both are bound to three, raises
  nothing), or for occurrences without tokens, different `.where` or
  `.when` bands (two demonstrative events).
- The raise is `ConceptIndex.index_part_row(cs, member, right)` for both
  members: the parent at `order_of(right) + 1` gains both as parts,
  `reference_orders[parent]` is written, the parent is settled, and the
  word that reads to `right` is given the parent as its object
  (`Interpret.define(word, parent)`, so `Interpret.forward` reads the word
  to the raised concept from then on). The first raise of a name ("this is
  Felix, that is Felix") takes two order-0 events to an order-1
  individual; a later "Felix is a cat, Ciao is a cat" takes two
  individuals to an order-2 kind. Raises carry no price (decision 4).
- Threshold by strength of evidence (decision 5): both equalities must be
  asserted, `truth_plus` above `<raiseEvidence>` (ConceptualSpace, default
  0, meaning any asserted equality), written beside the read.

The implicit raise, promotion by recurrence in `consolidate`, keeps
recording order 1 for a token; it is the same class formation by a
different witness and is left as it is. The fold raise and training the
fold from DEF rows are the design record of decision 5, not this round's
code (§8).

## 5. Every symbol's order

- **Unknown is not 0.** `order_of` returns `-1` for a concept with no
  recorded order, no `part` references and no percept definition
  (`_definitions` empty); every reader that needs a number coerces
  explicitly with a comment at the site (`_row_order`, `denoted_order`'s
  unknown rule, `resolve_order`'s request). `Interpret`'s `reference_orders
  [word] = 0` stays: a word with a form is a witnessed order-0 type.
- **A record is checked against the structure.** `settle(cid)` raises
  `ValueError` when `reference_orders[cid]` exists, the concept has `part`
  references and the record is not `1 + max` of their orders; a raised
  parent satisfies it by construction. `_csw_concept_row`'s silent write of
  `reference_orders` at a higher order (≈15455) goes: a row is requested at
  the concept's own order, and asking for another is an error.
- **A relation handle has an order.** `_concept_source_order` on an LTM1
  handle reads `denoted_order` from the row's tagged references: `equal`
  the left's, taxonomy (`SYMBOL`, `SYMBOL`) the whole's, a predication chain
  (an `OCCURRENCE` subject with a `RELATION` or `SYMBOL` predicate) the
  subject's minus one, floor 0.
- **The test-only formation API** that keeps the outdated ladder (todo
  item 3, "Every symbol's order") is named in the receipt and retired with
  its tests listed; `_order_caps` stays the row-pool partition it is.

## 6. Parts of speech by position and order

`PartsOfSpeech` keys its counts by `(argument, order)`: `add(word,
argument, order)` with `order = alloc.order_of(concept)` for the word's
concept at the kept close (an unknown order counts under `-1`);
`distribution(word)` returns the argument distribution at the word's
current order, falling back to the sum over orders when the word's order
is unknown; `probabilities(words, like)` keeps its `[..., arguments]` shape
for the chooser's table, read at each word's order; `labels[word]` is the
argmax at the current order; the extras (`get_extra_state`/`set_extra_
state`) advance a version and reject the older layout. A noun occupying the
same position at order 1 and at order 2 is thereby two parts of speech,
learned from participation (the design record's third point). The learning
stages (proper nouns, mass nouns, count nouns) are item 0's curriculum.

**Forcing.** Until then unit tests force: the eleven files that write
`reference_orders[word] = 1` do so through one fixture helper,
`test/order_fixtures.py::force_order(cs, word, order)`, which writes the
record and `settle`s, so every forced order is declared and listed; a
fixture that supplied a determiner's output order through the retired
constant instead forces its noun's order.

## 7. Ports

Tests that assert `order_delta`, `operation_order`, `result_order`, a
determiner's constant 1 or 2, `needs`-style stamp arithmetic, or an LTM1
handle's order 0 are ported to the lookup with before and after text;
tests of `PartsOfSpeech.counts` by argument alone are ported to `(argument,
order)`; the thirteen `parts_of_speech` files and the eleven forcing files
are listed. A port that changes a decided behavior stops for Alec.

## 8. Not in this round; Claude's calls

**Not in this round:** training the fold that the symbolic order loop
uses from DEF rows, and judging explicit against implicit definitions in
the two truth lanes (decision 5's design record; item 0's learning with
the curriculum); the learning of parts of speech (item 0); `IdeaEvaluation`,
`begin_presentation`, `embed.py`.

**Claude's calls under §39:** the `denoted_order` table of §3 as the
reading of what each composition denotes; distinctness by token and value
with the `.where`/`.when` fallback; `<raiseEvidence>` default 0; `-1` as
the unknown order; the `(argument, order)` key and the `force_order`
helper.

## 9. Alec's decisions, with recommendations

1. *A determiner over an order-0 noun.* Orders §10 says "the water" is an
   individual at order 1 (a shaped portion), and the todo says "the
   explosion" should pick out one event at order 0. Both nouns are order-0
   types today, so the lookup alone cannot tell them apart; the difference
   is a mass noun against an event noun, which is parts-of-speech
   learning. Recommended: the determiner outputs the noun's order (0 for
   both) until a mass noun's record says otherwise, and the fixtures that
   need "the water" at 1 force it through `force_order`; the alternative is
   a declared list of mass nouns, which names lexical facts the grammar
   excludes.

## 10. Gates (sized to the change; one frozen hash)

1. **Raise by two members.** "this is Felix. that is Felix." (two
   demonstrative events at different `.where`) raises Felix 0 to 1: a
   parent with both events as parts, `reference_orders[parent] == 1`, the
   word reading to the parent. "Felix is a cat. Ciao is a cat." raises cat
   1 to 2 with two members. "y is three. x is three." and "Paris is the
   capital of France" raise nothing. Each raise consumes no work meter.
2. **Lookup.** With Felix at 1, "Felix runs" closes at order 0; with dodos
   at 2, "dodos died out" at 1; "water flows" at 0; a modal VP leaves the
   subject's order; a determiner over a kind gives one less, over an
   individual the same; `equal` keeps the left's; `bind` gives the
   identity's; no layer or rule record has `order_delta`; `operation_
   order`, `result_order` and `shifted_order` are not importable.
3. **Every symbol's order.** An unwitnessed concept's order is `-1` and no
   reader turns it into 0 silently; a recorded order inconsistent with the
   parts raises at `settle`; `_csw_concept_row` refuses a foreign order; an
   LTM1 handle's order follows its references; the test-only formation API
   is gone.
4. **Parts of speech.** After a lesson where one noun occupies the same
   position at orders 1 and 2, `counts` holds two keys and
   `probabilities` differs by the word's order; the extras round-trip and
   reject the older layout.
5. **Unchanged sets on the frozen hash:** G1–G3, the pending-premise
   certificates, the thought files, the reference, bind-thought and
   identity files, the layout, placement, chunk and retire-mode gates, the
   percept certificate, the XOR standing gate and both decomposition arms,
   with the forced orders declared through the helper.

One short native lesson smoke, one source freeze, the gates on that hash
in a receipt under the local `doc/benchmarks/2026-10-2X-item3-0-orders/`.
Then, with step 5 approved, **the item's one full sweep** on that same
hash (`make test` as the criterion, slow tests under `RUN_SLOW`), its
failures repaired in scope and nothing retired, Claude's review, and the
publication of item 3.0.

*§9 item 1 decided (Alec, 2026-10-26): no explicit machinery.* "The use of
the determiner in association with only some words will condition those
words such that they eventually form a part of speech which is distinct
from mass nouns. No explicit machinery should be necessary." So the
determiner outputs its noun's order by lookup, 0 for an order-0 noun
whether mass or event; the count-noun part of speech forms in
`PartsOfSpeech` from the determiner occupations themselves (§6, position
and order), and "the water" reads at order 1 when that occupation and the
definitions that follow it record the word there, not from a declared
class. Fixtures that need the order-1 reading before it is learned force
it through `force_order`. No list of mass nouns, no flag on the determiner.

## 11. Source review and implementation reconciliation — Codex, 2026-10-26

The local main and freshly fetched origin/main still identify `8b523874f9`.
The approved step-4 port is present in the working tree and matches every
byte of repair-05's after file; its §20 review and §21 result are present
too. This round starts from those approved contents without creating a
commit. The initial state and before files are retained under
`doc/benchmarks/2026-10-26-item3-0-orders/`.

The table of §3 also feeds the live `LanguageSpace.choose_operation` and
`ChunkJournal.choice` paths. Their orders must agree with `ClauseJournal`:
the staged native operand orders feed `denoted_order`; `OrderReference`
carries the existing timeline flag alongside an operand identity. The
off-stack bind reads its selected chunk's order, with the dummy retaining
the mention's own. The old modal floor of one is removed: a modal preserves
an order-0 subject at zero, as §9 requires. The reference contract selects
the operand's denotation; lowering the completed NP+VP does not lower the
subject's reference request. Determiners request their selected member's
looked-up order.

`SymbolSubSpace.conceptualize` is the obsolete formation API. Its only
caller is `test/test_conceptualize.py::test_conceptualize_dispatch`; the
three-order dispatcher is removed and that file becomes an explicit API
retirement certificate. Its complete old assertions are retained in the
port receipt. The underlying native relation and definition APIs remain.

Unknown concepts cannot acquire order through a row request. Fixtures
which requested a row before supplying their percept definition now admit
that witness first. The eleven direct order-writing files use the declared
`force_order` helper; no initialization is selected to obtain an outcome.
Order records are checked against structural members at `settle`, while
percept definitions witness order zero. The feature-row witness index is
rebuilt on topology revision, not by scanning the store at each lookup.

Occupation counts store `(argument, order)`, including `-1` when there is
no witnessed order. A read selects the word's current object's order;
unknown or ambiguous objects aggregate their order records. Labels are
read at that order rather than cached across a later definition. The
occupation extras schema advances to version 2 and grammar-catalogue
migration filters by the argument component of each pair.

The first equality member's value is held in a construction-sized slab,
one row per configured conceptual inventory seat and one native concept
vector per row. `singleton_members` holds its reference, MTM token and a
view of that snapshot. Checkpoint restore copies into the existing slab;
exhaustion and width mismatch are errors. Raises remain outside the work
meter and retain the first witness independently of subsequent token-value
updates. The implicit consolidation raise is unchanged.

## 12. Reader and fixture reconciliation — Codex, 2026-10-26

The strict row request exposes two previously conflated operations. A
constructed grammar primitive already has a declared order-zero role;
`mint_frozen_concept` records that role before requesting its row. A fixture
that supplies an otherwise unwitnessed native vector must declare its order
through `force_order`. Neither path changes the row allocator's new refusal
to invent a concept order. Padding and unavailable references return `-1`
before entering the tagged relation index. Unfolding excludes unknown-order
terminal candidates; it does not treat `-1` as a lower event order.

Occupation lookup is attached to the clause registry's conceptual owner.
A terminal stage's local allocator can have different IDs from the common
word/close owner; reading it produced `-1` counts for witnessed words. The
attachment now shares the same owner as clause admission. A requested
explicit order can still select a retained earlier definition, while an
unqualified word read follows its witnessed raised parent. Definitions are
not deleted.

Equality witnesses on an occurrence side canonicalize through its MTM token,
so different mentions of that token enter the same class. The first row's
positive evidence is rechecked before the raise. A raised class's minimum-
support exemption does not make a relation to that class an equality when
reading its order: the `raised` mark distinguishes it from a unit side.

Development exposed fixtures still using pre-chunk binary-bind shape and
pre-version-8 MTM restore expectations. They are ported to the already
landed stack arity and version rejection, preserving the original cases in
the before files. These are fixture repairs under §39, not changes to the
bind mechanism or the checkpoint decision. All ports, including the
retired formation-API certificate, are enumerated in this round's receipt.

## 13. Native determination and closing order — Codex, 2026-10-26

The inherited-set development run reached a native failure after the lookup
change: `ChunkJournal.apply` still allocated a transient identity for every
determiner, while `ClauseJournal` and `ReferenceClose` admit determined
introductions at order one. An order-zero noun therefore produced an
unwritten introduction which a later bind could select. Determination now
opens that identity only when the selected member order is one. Order-zero
and unknown results retain their own field without a hidden individual.
The new deterministic chunk case checks all three orders (0, 1, -1).
Identity fixtures which formerly relied on the determiner's constant supply
their kinds explicitly through `force_order`, retaining their recorded
walks, binds, derivatives and histories. No seed is changed.

A second closing-order calculation in `ClauseJournal.recover_content`
still selected a syntactic head or maximum after the journal had called
`denoted_order`. It now reads the journal node's looked-up order, including
its existing modal capture. A new case checks both `Clause.order` and the
written `LTM0.order` for subjects at zero, one and two. The sole zero
coercion there is explicit: a completed percept point can be stored at zero
when its source concept remains unknown; no concept record is assigned.

## 14. Retained classes and explicit fixture witnesses — Codex, 2026-10-26

Once a word denotes its raised class, another member of the same underlying
order joins that existing class. `relate_idx` retains its base alongside the
existing parent key; it is restored with that already checkpointed index.
The equality certificate now admits a third distinct event and checks that
the parent stays at one with all three parts, still without work charges.
This is continued membership of the equivalence class, not a further raise.

The wider fixture audit includes obsolete assertions outside step 4's
selected files. The definition/taxonomy cases now read relation entries in
LTM1 and individual references in LTM0, without expecting composition to
mint a lexical object. They retain deduplication, ownership and capacity
atomicity checks. The old unregistered-reference failure fixture now uses
actual LTM1 exhaustion: a missing symbol can retain its owned percept point
under the landed layout. The width arm explicitly checks the existing
`LTM0._fit` leading-coordinate adaptation instead of claiming width
adaptation is forbidden. Old vocabulary/MTM checkpoints are rejected at
their landed versions, not migrated by these tests. Before/after assertions
are retained in the port receipt; no production layout policy is changed.

The word-reading fixture constructs four stream seats before presenting
four rows, and its serial mask is applied after the MTM prefix. A later
symbolic definition is attached to a supplied object, not to the witnessed
order-zero word: the latter would correctly fail the new record/parts
consistency check. The unused-seat forgetting fixture withdraws its setup
order testimony before testing abandoned reservations; retained records
remain protected. The only order writer in these fixtures is `force_order`.

## 15. Buffer bound, remaining readers and fixture isolation — Codex, 2026-10-26

The first-witness slab's final construction bound is the LTM1 relation
capacity, superseding §11's conceptual-seat bound. Every first equality
witness requires a relation row, while its right reference can be an MTM
token or occurrence outside the codebook. The native width and this bound
are written beside the allocation. Exhaustion remains loud, restore copies
into existing seats, and the certificate checks both the default shape and
an explicitly bounded slab without pointer changes.

The sparse accessible-field reader explicitly maps an unknown order to its
physical base pool without assigning an order record. The conflicting-pole
alternative admits its existing percept constituents before requesting its
order-zero row. These are reader/admission reconciliations with §5's strict
row-request contract, not new concept-order authorities.

The remaining query/taxonomy fixtures declare their supplied primitive
orders before requesting rows; a structural parent requests its derived
order. The typed-reference fixtures supply a witnessed kind instead of
relying on the retired kind constant. Predication assertions now distinguish
the enclosing occurrence at zero from its owned individual at one, including
the stored row. Their reference, evidence and controller assertions remain.

The development run exposed a separate setup defect in
`test_what_training.py`: the bounded runner placed
`test_curriculum_drives_question_families_through_run` before
`test_temporal_question_content_is_causally_used` in the same worker, sharing
the module-scoped trainable `synth_model`. The latter's supposedly untrained
losses were 0.3729642629623413 and 0.37358012795448303. The fixture is now
function-scoped, giving both training cases the same existing constructor
initialization independently. No seed, tolerance, training count or causal
assertion changes. The failed result and exact before/after text are retained.

The dormant `embed.py` writer still accesses the removed signature field
`order_delta`; its artifact-writer/view/typed-stack tests retain their legacy
serialized schema. This is a source-inspection finding, not a measured
full-sweep result. §8 explicitly excludes that writer, so it remains
byte-identical and is called out for Claude's review before the item's full
sweep; no compatibility field is restored to the grammar and no test is
retired or hidden.

## 16. Equality-cache lifecycle and development record — Codex, 2026-10-26

A class's reverse base entry is a tagged reference in the already saved
`relate_idx`. The retirement reader no longer casts every cache value to an
integer; it removes the retired parent's forward and reverse entries and
preserves unrelated classes. The new lifecycle certificate exercises both
operations. If a remembered first equality is absent from LTM1 at the next
close (forgetting or the existing presentation clear), that new equality
renews the first-witness snapshot in its existing slab seat. A further
independent equality then raises from the two surviving witnesses. This
avoids a permanently stale cache without changing `begin_presentation`,
adding a memory reader, resizing a buffer or counting forgotten evidence.

Development results remain separate from the one measurement freeze.
`core-02` completed 1,621 selected cases and exposed the remaining ports;
`remaining-03` retains a collection indentation error fixed immediately.
`remaining-04` completed 239/241 cases, with one remaining stored-order
assertion (ported from one to zero) and a 600-second timeout in a worker
containing two supplied-text learning cases. Those two cases were included
by the broader file-level fixture check; they are not in the inherited
scoped selectors and remain in the full suite. The corrected causal test
passed with all its original assertions. No timed-out test is retired or
turned into a skip; the full sweep remains after Claude's approval.

## 17. First freeze's XOR reader repair — Codex, 2026-10-26

The first freeze, `5abf77991de2684821ef9861d717763ec9c38421fa52859edcf8a9945a4a6882`,
passed its native smoke and packing group. XOR stopped before learning in
all eight cases: its native fixture requested unwitnessed rows at zero and
one, and `ConceptLessons` did the same in the two CLI cases. These missed
callers require the same strict-row reconciliation as §12. Core was stopped
at 643/1641 completed cases to repair the source; its partial result remains.
The failed freeze, source archive, protocol, exact ports and all results are
retained intact under the receipt's `diagnostic-01/`. They are not counted
as final gates. Under §39's fix-and-continue rule, the repair receives a
replacement smoke and final freeze, and every required group is rerun on
that one final hash; the requested first-freeze plan was not achieved.

The native XOR fixture declares its supplied zero-order primitive and
one-order output through `force_order`; no learning, numerical, truth-table,
repeat-count or budget assertion changes. The production teacher receives
explicit native primitive memberships, which witness zero just as a supplied
word form does; it records that testimony before asking for a row, rejecting
an empty definition. Its output sigma remains unknown until its unlabelled
cases have been witnessed, then reads one above their orders before asking
for its row. The existing output edges still start at zero and learn through
the same loss. No candidate is asserted as a taxonomy member, no answer label
sets an order, and no fold-from-DEF training is added. `_csw_concept_row`
continues to refuse an unwitnessed or foreign order.

## 18. Final scoped measurement and review hand-off — Codex, 2026-10-26

The replacement native smoke passes, and all five final groups pass on
source SHA256
`c4a1d92e3fc27decc669322a507b362dd615556e89041331fe1259116584ab95`:
core 1,631 passed with ten existing prerequisite skips; packing 4 passed;
XOR 8 passed; decomposition zero and enabled 2 passed each. The total is
**1,647 passed, ten skipped, zero failures**, plus the smoke. The 36 new
order cases are included in core: gate 1 has seven, gate 2 twenty-three,
gate 3 five and gate 4 one. Gate 5 retains every inherited selector.
The two additionally selected occupation prerequisites account for the
increase from eight to ten skips; no skip was introduced.

The receipt at
[2026-10-26-item3-0-orders](../benchmarks/2026-10-26-item3-0-orders/README.md)
retains the source archive, all final results, exact before/after text for
57 test/helper ports, the eleven original forcing files and the named
formation-API retirement. All 1,482 inherited setup RNG states match exactly;
159 new or renamed cases have separately recorded states. Both decomposition
arms retain the original snapshots, seeds 4199722147 and 2428440506, and
question budgets 512 and 768. No initialization was selected to pass.

Companions remain **6/8**, matching the approved step-4 port, with document
0's length-1 question missed in both arms. All four proofs pass; 30/32
premise readings are full, none is empty, and both streams' first premises
are full. Every kept reading is recorded. The unchanged two-stream/predictor
file passes 2/2, the unported clock case passes, and the read-only percept
certificate passes its 40 atom edges and six nested-atom-set word pairs.
This lesson has no nested-part edges; the two tower-1 distances remain
diagnostics. Silence counts remain a mechanism record, not a learning claim.

The source, archive, protocols, ports and protected-file audits pass.
`IdeaEvaluation.py`, `BasicModel.begin_presentation` and `embed.py` are
unchanged. The failed first freeze and its incomplete results remain
separate under `diagnostic-01/`, as §17 records; the one-first-freeze plan
was not achieved. No result from it is counted in the final totals.

Return through Alec for Claude's review. The dormant exporter incompatibility
in §15 is a source-inspection finding to resolve before the full sweep;
its legacy tests remain. No full sweep, commit or publication has occurred.
On approval, the item's single full sweep begins on this hash, followed by
scoped repairs if needed, Claude's sweep review and publication.

## 19. Claude's review of step 5 (source `c4a1d92e3f`) — 2026-10-26: approved; one fix before the sweep

Reviewed: §§11–18, the [receipt](../benchmarks/2026-10-26-item3-0-orders/README.md),
`summary.json`, both `frozen-source.json` manifests against the tree, the
port manifest, the step 4 port's receipt (`repair-05`), and the new text of
`ReferenceContext.denoted_order`, `PartClosing._write`'s raise,
`ConceptAllocator.order_of`/`settle`/`remember_singleton`/`singleton_values`,
`_csw_concept_row`, `PartsOfSpeech`, `test/order_fixtures.py` and
`embed.grammar_signatures_to_serializable`.

**Verified.** The final source `c4a1d92e3f…` matches the tree on all 900
paths and its canonical map hashes to the declared value; the first freeze
(`5abf77991d`, the XOR row requests) is kept apart with its partial results.
Core 1,631 passed with ten known skips (the two added are declared
prerequisites in an additionally selected file), packing 4, XOR 8, both
decomposition arms 2 and 2 on the original seeds and snapshots; the 36 new
order cases sit inside core; 1,482 inherited initializations match exactly.
The step 4 port runs green on the committed source (two arms and its file).
`IdeaEvaluation.py`, `begin_presentation` and `embed.py` unchanged; no
language named; no seed added (the one `manual_seed` in the tree is the
recorded-initialization replay fixture from step 3); 57 ports with before
and after text; the one retirement is `SymbolSubSpace.conceptualize`, the
test-only ladder API todo named, with its old test kept as the retirement
certificate.

**The round is the spec.** `denoted_order` is the one order reader at the
journal, the live menu, the chunk journal and the form readers;
`operation_order`, `result_order` and `shifted_order` are gone and
`order_delta` is on no layer and in no rule record; the modal keeps an
order-0 subject at 0 (the old floor of one removed, as §9 requires). The
raise fires at the asserted `equal` in `PartClosing._write`: the first
witness is held in `singleton_members` with its value in a slab bounded by
the LTM1 relation capacity (`singleton_values`, construction-sized, loud on
exhaustion), distinctness is by MTM token and published value with the
`.where`/`.when` fallback, `index_part_row` raises both members and
`Interpret.define` gives the word the class, `raiseEvidence` defaults to 0,
and a forgotten first witness is renewed rather than left stale. Unknown is
`-1`, `settle` raises on a record that disagrees with the parts,
`_csw_concept_row` no longer writes an order, a relation handle reads its
references. `PartsOfSpeech` keys `(argument, order)` with versioned extras.
Thirty-eight files force through `force_order`; the only direct write left
in tests is the one that checks `settle`'s refusal. Companions stay 6/8 with
the same question as the step 4 port, no premise reading empty, both first
premises full; the percept certificate passes read-only.

**Finding, to fix before the sweep (Claude's call; no decision).**
`embed.grammar_signatures_to_serializable` still reads `sig.order_delta`
(≈971), a field `Grammar.RuleOrderSignature` no longer has. The live
artifact writer does not reach it (`save_artifact` never calls
`build_knowledge_section`), so no production path breaks, which is why the
scoped groups are green; but the knowledge-artifact tests
(`test_knowledge_artifact_writer`, `test_knowledge_view`,
`test_explicit_ordered_stm`, `test_activation_priming_multiply`,
`test_phase2_end_to_end`) call it and will fail at the sweep. The field is
the stamp machinery's serialization and goes with it: remove `order_delta`
from the serialized signature and from `KnowledgeView.rule_order_
signatures`' consumers, advance the artifact `grammar` section's version so
an older artifact is rejected by version, port the five files with before
and after text. §8's exclusion of `embed.py` was the SBOW pretraining, not
this; Codex was right to stop and flag it rather than touch a protected
file unasked.

**Noted.** Codex's reconciliations (`ChunkJournal.apply` opening an
identity only for an order-1 member, `recover_content` reading the
journal's looked-up order, the XOR teacher witnessing its primitives' order
before requesting rows, the `synth_model` fixture isolation) are within §39
and recorded. The one explicit zero coercion (a completed percept point
stored at 0 when its concept is unknown) carries its comment.

**Verdict.** Step 5 is approved as a mechanism round, and with it item
3.0's five steps are complete as mechanisms. One more short round on this
source: the serializer fix above, one smoke, one freeze, the five groups
plus the five artifact files. Then **the item's one full sweep** on that
hash (`make test`, slow tests under `RUN_SLOW`), every failure repaired in
scope or recorded, nothing retired, Claude's review of the sweep, and
publication: Alec commits the step 4 port, step 5 and the sweep's repairs
naming their frozen hashes, bumps WikiOracle, and item 3.0 lands.

## 20. Serializer boundary and sweep protocol — Codex, 2026-10-26

The short round starts from the approved `c4a1d92e3f` source. The grammar
section was unversioned; its replacement explicitly carries version 2.
`grammar_signatures_to_serializable` omits `order_delta`. Save, load and
direct `KnowledgeView` construction require that version before exposing
the signatures. Missing, old and unknown versions raise; no artifact is
migrated. The other artifact sections and the SBOW pretraining are unchanged.
An exact definition-level audit records the permitted serialization and
validation edits in `embed.py`.

The five requested artifact files retain their admissibility, taxonomy,
priming and round-trip assertions, replace the removed-field expectations
with absence checks, and certify the new section version. The view regression
rejects unversioned, version-1 and unknown-version artifacts without changing
their bytes, and rejects saving or directly viewing the same sections.
The additional `test_typed_stack.py` fixture also drops the retired field;
it is the remaining synthetic signature consumer. All six files retain
exact before/after text. Their development check passes 94/94.

The local serializer receipt is separate from the approved step-5 receipt:
`doc/benchmarks/2026-10-26-item3-0-serializer/`. The five inherited groups and
the artifact group will share its new freeze; core replays all 1,641 setup
states from the approved run, and decomposition retains the original seeds,
snapshots and budgets.

The repository's current `make test` recipe invokes a bounded worker
supervisor, rather than one pytest process. For the explicitly requested
single-process sweeps, a receipt-local makefile includes the repository
makefile and supplies `test` and `test_all` recipes that each invoke pytest
once over `test/`, with `RUN_SLOW=1` for the latter. The repository makefile
is unchanged. The receipt retains the exact commands, every report and
failure, setup RNG states and emitted certificates. The supervisor limits
the whole process tree while preserving that one pytest process. These
protocols are frozen before measurement; neither run replays chosen seeds.
The existing, measured virtual environment lacks Make's installation stamp;
the command marks that prerequisite old (`-o .venv/.installed-bin`) so Make
does not delete and rebuild it during the sweep.

Before starting either sweep, the runner self-tests exposed an observer
ownership concern: they launch nested pytest processes, so loading receipt
observers through inherited `PYTEST_ADDOPTS` would mix child reports into
the parent receipt. The sweep-local driver instead loads the same frozen
observers on the outer pytest command line only and records its own hash.
The serializer protocols and source freeze remain unchanged; no sweep run
used the inherited-observer invocation.

## 21. Serializer measurement complete; full sweep begun — Codex, 2026-10-26

The native smoke passes. Source SHA256
`caece43fc022c963889f61edf7ef3f786d5b424ef453ab02a44ce57bb28a4a6e`
passes core 1,631 with ten known prerequisite skips, packing 4, XOR 8,
decomposition zero 2 and enabled 2, and the six artifact/typed-stack files
94: **1,741 passed, ten skipped, zero failures**, plus the smoke. There is
one new source freeze for this round. All 1,641 inherited core setup states
match exactly, and the four decomposition runs retain their original
initializations and budgets. Companions remain 6/8. The unported clock,
two-stream and percept cases pass. The source, archive and protocol audits
pass; changes in `embed.py` are limited to the authorized artifact boundary.

The [serializer receipt](../benchmarks/2026-10-26-item3-0-serializer/README.md)
keeps all six test ports and the result separate from `c4a1d92e3f`'s record.
The full sweep has begun on this source, with its independent receipt under
`doc/benchmarks/2026-10-26-item3-0-sweep/`. No commit or publication.

## 22. Sweep diagnosis in progress (Codex, 2026-10-26)

The serializer source `caece43fc0` entered the requested one-process default
sweep. It stopped at the 28 GiB process-tree guard after 3,970 of 6,697
selected cases: 3,659 passed, 176 skipped, 135 failed and 2,727 unfinished.
Every failed report and setup RNG state is retained in the [sweep receipt](../benchmarks/2026-10-26-item3-0-sweep/README.md).
The slow-enabled target runs on the same unchanged source before applying repairs.

The checkpoint-order replay reproduces on its recorded initialization:
a chain references two bare words, and the allocator counts that semantic
edge as an additional constituent of the second word's form. A word's form
is already a witnessed order-0 type (§5); a chain reference is not a new
boundary-pair definition. The candidate repair reads a word's native form
constituents for its order, leaving structural order checks on concepts
intact. The recorded checkpoint replay passes with this candidate; it is
staged separately pending the initial sweeps and a dedicated regression.

Fixture ports use construction `batchSize` (the existing derivation of
`nStreams`), the native chunk tuple, the binary off-stack bind catalog and
one-idea storage. Complete-description evidence fixtures retain every
original independent content block and mismatch in a single supplied vector,
with no percept band in those content-only fixtures. Their evidence, scope,
provenance and corruption assertions remain; exact before/after text is
retained with each applied port. No initialization is chosen to pass.

## 23. Sweep reader repairs — Codex, 2026-10-26

Both initial single-process targets stopped at their 28 GiB process-tree
bound on serializer source `caece43fc0`. Default completed 3,970/6,697
(3,659 passed, 176 skipped, 135 failed); slow-enabled completed 1,982/6,697
(1,864 passed, eight skipped, 110 failed). Their complete phase reports,
failed traces and setup states remain under the sweep receipt's `run-01/`.
A diagnostic continuation covered the default run's unfinished 2,727:
2,463 passed, one xpassed, 148 skipped and 115 failed. This continuation
is not the requested completed sweep.

The scoped production repairs are recorded against that exact source,
with full before/after text in the sweep receipt's `repair-01/production/`:

1. `ConceptAllocator._part_order` reads a word's native boundary-pair
   constituents, rather than treating every semantic chain that names the
   word as a new part of its form. A word remains order zero; genuine
   concept-order contradictions still raise. The checkpoint failure
   reproduces on its saved initialization, then passes with this repair.
2. `_open_reference_mask` reads the kept clause's resolved journal
   references at the closing boundary. Temporary absent `role_refs` in a
   pre-write working meaning must not reopen a relation whose operands the
   journal has resolved. Explicit forward-reference annotations still mark
   open sites. This restores the exact native-gain expectation certificate.
3. `_reconstructionReport` bounds each evaluation batch by the constructed
   stream count while letting `runEpoch` visit every report row. A five-row,
   two-stream regression checks every row is visited and printed.
4. The noncanonical tensor-peer fallback retains existing `SymbolEntries`
   views. Republishing a slab view through its setter cleared the source
   before copying it. The fallback stages only absent metadata. The native
   regression checks both exact contents and data pointers across the call.
5. The peer-leg benchmark carries the complete thirteen-field chunk state,
   passes that tuple to the pure push, and reads lexical IDs through
   `word_view`. Its existing real compiled PS/WS/CS profiler certificate
   checks the corrected caller; no model kernel changes for this repair.
6. `recon_bench` also bounds evaluation batches by constructed streams.
   Its evaluation batch budget, training epochs and targets are unchanged.
   A real benchmark regression checks the exact stream bound and original
   one-batch limit before native evaluation.
7. `ConceptualSpace._iter_relation_rows` explicitly requests the numerical
   Part projection. An untyped chain's two references otherwise occupy
   working slots zero and one; reading slot two silently returned zero.
   The regression preserves exact references, evidence, vectors and row
   count and confirms the read does not write a predicate.
8. `TruthGroundedReasoner.query` reads the LTM0 `point` column, replacing its
   two accesses to the retired `np1` column. The regression checks the exact
   best idea, trust and row count.

The seven new cases in `test_sweep_repairs.py` cover items 1–4 and 6–8;
item 5 retains its existing real profiler case. `IdeaEvaluation.py`,
`BasicModel.begin_presentation`, and `embed.py` remain byte-identical to the
serializer freeze. No capacity is resized, chooser or comparison changed,
or learning assertion relaxed. These production repairs require a new
smoke, freeze and five-group measurement before the complete sweep resumes.

## 24. Sweep fixture ports and explicit retirements — Codex, 2026-10-26

The exact before and after text for every applied fixture is retained in
`doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/ports/`; the manifest also
lists renamed tests, never silently dropping a name. The initial failed
runs and all intermediate port checks remain. Final contract choices are:

- Supplied primitive, identity and parent orders use `force_order`; row
  lookup no longer silently supplies an order. The fixtures' original
  structural, evidence, capacity and propagation assertions remain.
- STM fixtures use the complete fixed tuple and distinguish physical
  chunks from the ordered stack. Full-stack shifts refuse without changing
  any field; off-stack eviction remains in the inherited chunk gates.
  Capacity-change tests now certify loud refusal and pointer stability.
  Closed fields leave recent chunks with an empty stack and the exact
  output field. Readers use the lexical views after the MTM prefix.
- Single-idea evidence fixtures preserve all independent content blocks in
  one explicitly sized vector, with no percept band in content-only data.
  Relation tests use tagged references, immutable chains and explicit
  numerical projections. Removed derivation edges, duplicated idea rows,
  old checkpoint columns and synthetic taxonomy predicates are not restored.
- Binding tests certify the binary off-stack composing face and its
  separate unary thought catalog. A fixture that needs a closed episode
  supplies the chooser-opened flag or the recorded walk. Nested refinement
  supplies a real free referent and resolves it with `query` before its
  equality refinement; cutoff requests retain the occurrence they read.
  Their original budgets, causal-source checks and zero-cost drainage stay.
- Percept tests use WordIdentity's boundary-pair atoms. The one-byte word
  padding test retains both its original single-part identity-padding check
  and the actual native three-atom word's exact padding invariance. The
  percept transitivity certificate itself is unchanged.
- Native spelling replaces the retired learned byte inverse in the mini
  production-width test: explicitly recovered leaves must spell the entire
  input exactly. Its supplied indexed-readout witness checks the same sparse
  gradient and optimizer update, now through the reconstruction owner and
  RowLocalSGD. The old `_shared_representation_parameters` fixture call is
  retired explicitly. The legacy affine numerical certificate declares its
  whole supplied vector content, preserving full-vector inverse equalities,
  bias perturbation and nonzero gradients without assigning numerical
  samples to time coordinates. Live content kernels still exclude time.
- The native static-body smoke now calls the actual fixed-state sentence
  body after the normal staging boundary. The old direct word cell lacks
  that native reference scope; the two dense-cell accumulator/recompile
  pins continue to test its supported mixing path. The native smoke adds
  exact result-tuple, chunk-tuple and fixed-extent checks.
- Six obsolete WholeSpace analysis/VQ assertions in
  `test_dual_input_contract.py` become explicit retirement certificates for
  the already-decided immutable property tower (operators-update §9.2).
  They assert absence of the old analysis store and reconstruction loss,
  native property ownership and exact carrier behavior. The independent
  painting blend test remains. No retired dictionary is restored.
- The two legacy relative-plan cases supply a plan-only owner: an actual
  empty clause or choice trace intentionally takes priority over an old
  `current_rules` plan. Typed reverse retrieval checks its passed numerical
  knobs, not an obsolete mode keyword. Expectation's first document cover
  and step 4's offered thought update are counted explicitly; compose's
  own exposure and optimizer ownership checks remain exact.
- The optional embedding text test supplies its own deterministic file at
  the original dimension with more than 1,000 unique entries. Its external
  file was absent. Archived document copies resolve relative links from
  their recorded original homes, without skipping or rewriting snapshots.
- An autouse fixture collects unreachable cyclic models before and after
  each independent test. Diagnostic samples reclaimed 2.1–2.7 GiB by
  collection alone. Live module fixtures and within-module compilation
  caches stay owned; the existing module teardown still resets Dynamo.
  The diagnostic tail peaked at 7.2 GiB. No RNG is changed by cleanup.

Recorded initializations are replayed only for diagnosis and the inherited
scoped gate protocol; the complete sweeps use their ordinary fresh streams.
The recorded blind-XOR failure remains a learning failure with its original
160-epoch budget and seed `3145616037`, alongside the 6/8 decomposition
companions at item 0. Its assertion is unchanged. The math-chain cube error
has not reproduced in the saved-state or whole-module probes; it remains
an unresolved initial failure until the complete repaired sweeps check it.

## 25. Final sweep diagnostics before the repair freeze — Codex, 2026-10-26

The replay of 277 original failure nodes on their captured initializations
passes 274; the three remaining cases are fixture ports. The ingestion
trace closes `union` over two unresolved SYMBOL names as an LTM1 chain,
exactly as layout step 1 requires. Its absolute-idea provenance fixture now
witnesses the four words' objects explicitly before its existing supplied
grammar runs. Both original ORIGIN_USER row counts, signed trust pairs,
replace-on-resubmit count and state-dict check remain unchanged. The two
checkpoint corruption payloads in the learning-readiness case remain
identical and rejected with ValueError; their regexes now name the tagged
reference and construction-capacity validators. The final five-case check
passes on the recorded initializations, including the two other ingestion
cases. No additional production repair was needed.

The replay, its three failures, their traces and the final check are retained
under the sweep receipt's `diagnosis/`. `repair-01/` records 97 fixture/helper
files and six production files against the serializer freeze, with full
before/after text and an assertion-level index. This count includes the
seven new repair regressions. The new smoke and freeze follow these checks;
the inherited groups and both complete single-process targets will use that
same source. No commit or publication.


## 26. Sweep repair freeze and scoped results — Codex, 2026-10-26

The native smoke passes. The repair freeze is source SHA256
`03fbbf2a7d9c67bbef7ccc0b77b65d40d89d220cc790f25a2bb1429080e25455`
over 901 measured paths. Core passes 1,631 with ten standing prerequisite
skips; packing 4, XOR 8, decomposition zero 2 and enabled 2, artifact files
94, and the seven new regressions plus the real profiler case 8. Total:
**1,749 passed, ten skipped, zero failures**, plus the smoke. The exact
archive, protocol, protected-file and before/after audits pass.

All 1,641 inherited core initializations match exactly. Both decomposition
arms retain their original seeds, snapshots and budgets. Companions stay
6/8; no premise is empty and both first premises are full in each of the
four evaluations. The unchanged two-stream, clock-independence and percept
certificates pass. The [repair receipt](../benchmarks/2026-10-26-item3-0-sweep/repair-01/README.md)
keeps these results separate from the initial sweeps and diagnostic runs.

The complete default and slow-enabled single-process sweeps now run on that
same source. Their failures, if any, remain to be assessed under §39; the
scoped result alone is not approval to commit or publish.


## 27. Complete default sweep; slow-enabled sweep running — Codex, 2026-10-26

The one-process `make test` target completes all 6,711 selected cases on
repair source `03fbbf2a7d9c`: **6,386 passed, 324 skipped, one permitted
XPASS, zero failures** (plus three passing subtests). Source and protocol
remain frozen. The process-tree peak is 11,384,467,120 bytes (10.6 GiB),
with 8,140.8 seconds elapsed; the initial run's 28 GiB interruption stays
in the receipt.

The XPASS is the existing cleared-cache top-k reconstruction criterion,
`test_stm_recon_from_cleared_cache::test_topk_recovered_words_overlap_input`.
Its unchanged marker records the historical 0.8/0.5 outcomes and permits
XPASS under §16.6; the result is recorded separately. The earlier math-chain
cube failure also passes in this complete run after its saved-state probes;
the original failure remains, and no unverified cause is claimed.

The selection audit accounts for every original node: twenty node renames,
no missing tests, seven new repair regressions and seven new receipt-link
checks. The read-only percept certificate and all seven new regressions
pass in the full-suite context. The default source check passes before
`RUN_SLOW=1 make test_all` starts on the same hash. The slow target and
Claude's sweep review remain; no commit or publication.


## 28. Slow-sweep blind reconstruction record — Codex, 2026-10-26

The unchanged 160-epoch blind `MM_20M_xor` round-trip assertion fails again
on repair source `03fbbf2a7d9c`: exact match 0.0 on fresh seed 3510236551,
location recovery 0.125 and output loss 0.169661. The initial slow sweep on
`caece43fc0` recorded exact match 0.0 on seed 3145616037, location recovery
0.0625 and output loss 0.166997 at the same budget. Both original benchmark
JSON files are copied byte-for-byte into the receipt, with hashes in
[the record](../benchmarks/2026-10-26-item3-0-sweep/repair-01/blind-learning-records.json).

As instructed for learning-side failures, no assertion, epoch count or seed
is changed and no production repair is made for this result. Both records
are carried to item 0 beside the 6/8 companions. Different fresh seeds do
not establish an improvement. The full slow-enabled sweep continues on the
same source, and all failed phase reports remain in its receipt.


## 29. Slow-sweep attention fixture ports staged — Codex, 2026-10-26

The full slow sweep on `03fbbf2a7d9c` exposes four additional fixture
mismatches in `test_global_attention.py` and `test_global_consume.py`:
two synthetic stores still use `[cap, 3, D]`, an input-forward test expects
the generation-owned reader's observation before calling that reader, and
a bypass stub lacks the caller's `identity` keyword. These are ports to
the landed layout and placement contracts, not changes to those decisions.

The candidates use one content row per stored idea, invoke generation's
reader after the same input-forward determinism checks, and accept the
keyword in the identity stub. Every original assertion stays. All 25 cases
in both files pass with the full slow sweep's recorded initializations;
all 25 replay paths have exactly one matching RNG snapshot. Exact file
faces and diffs are in the [pending port receipt](../benchmarks/2026-10-26-item3-0-sweep/repair-02/README.md).
The candidates remain separate until the complete frozen run finishes.

Two ordinary generation-gradient cases also fail their unchanged nonzero
conditioner-gradient assertion. Their recorded-state probes and original
failures are retained separately; their classification awaits the complete
walk trace. No production change, commit or publication in this staging step.


## 30. Unforced generation gradient remainder — Codex, 2026-10-26

Both variants of `test_normal_supervised_output_respects_gradient_contract`
fail the unchanged nonzero question-conditioner-gradient assertion in the
complete slow sweep. Read-only probes replay each recorded initialization.
The question contexts and prepared answer ideas are nonzero and remain in
the differentiable graph, but the selected generated concepts, percepts and
actual outputs are all zero. The retained output comparison costs are
0.0625 for each trial and each stream. The output reader cost observed at
the failing backward assertion is 1.4.

The detailed `output_loop=False` replay records the greedy `not` inverse
and the alternative `null` inverse in each stream, followed by no legal
continuation and zero emitted words; pending depth remains. Its valid
bank contains four terminal rows per stream. The prepared output walk has
a 48-trip construction bound (eight word seats and eight STM chunks).
Both variants request one two-stream `runBatch` at learning rate .001 and
fail before the answer-reader update; no additional training is supplied.
The selected paths therefore provide no conditioner gradient. The fixed
walk gradient mechanisms are measured separately in the unchanged groups.

This is retained as an unforced generation learning failure for item 0,
not converted into a passing assertion or a selected walk. No change to
the chooser, catalog, answer seed, R+A comparison, assertion or seed is
made. The original sweep failures, all failed diagnostic replays, exact RNG
snapshots and [walk/budget records](../benchmarks/2026-10-26-item3-0-sweep/repair-01/generation-learning-records.json)
remain. The full sweep continues on source `03fbbf2a7d9c`.


## 31. Slow-sweep word-cursor fixture port staged — Codex, 2026-10-26

`test_per_word_enabled_predicate_tracks_use_grammar` selects
`MM_20M_xor.xml` while asserting that it is a grammar-enabled configuration.
It also predates the live cursor back-reference: `create_from_config` mirrors
`word_brackets`, and `_has_word_objects` defines that structural contract.
The enabled fixture now builds retained `MM_ladder` with synthetic math
input; the disabled fixture remains `model.xml` with XOR input. The probe
reads `word_brackets` directly and asserts exact equality with the cursor
flag on both explicitly enabled and disabled sides. Grammar presence is
still checked independently in both chosen configurations.

The probe is renamed to `test_per_word_enabled_predicate_tracks_word_objects`
with its old identity retained in the receipt's replay alias. All seven
cases in the file pass on recorded setup states, including unchanged exact
slice, boundary and disabled-cursor checks. The [pending port receipt](../benchmarks/2026-10-26-item3-0-sweep/repair-02/README.md)
now has three fixture files and 32 passing candidate checks, with exact
before/after text. These changes remain staged while the complete slow
sweep continues on `03fbbf2a7d9c`; no production change or publication.


## 32. Reusable decomposition footprint port and fresh-initialization remainder — Codex, 2026-10-26

The full slow sweep reaches both decomposition certificates without the
receipt-only footprint adapter used in the approved scoped groups. It
therefore reports the nonpersistent form-validity buffer and the integer
form-derivation counter as semantic writes. The reusable fixture now makes
the same typed distinctions approved in placement §18 and the later chunk
and retire-mode receipts: nonpersistent form validity, detached centroid
caches, the bounded live chunk state, and empty constituent registrations
for actual chain rows. Every raw changed key and its before/after digest
is retained beside the classified comparison. Forms, placement, witnesses,
parameters, nonempty constituents, old idea rows and unexpected writes
remain checked. The proof, inference and descent/return assertions do not
change. Exact before/after text for the two fixture helpers is in
[repair-02](../benchmarks/2026-10-26-item3-0-sweep/repair-02/README.md).

On the full slow sweep's saved initializations, the ported one-step case
proves both documents at 246 of 512 work. The two-step case reaches 571 of
768 work, retrieves the final supporting sentence, but keeps an expression
reference instead of binding the expected final answer; both documents
fail the unchanged strict proof. Its first returned child binds correctly,
and all premise readings are full. Replaying both cases on the retained
serializer source `caece43fc0`, with only these same observer ports and the
same saved initializations, gives the same proof outcomes, operations and
work totals. Thus this remainder predates the sweep production repairs.
Both failed replays and both complete certificates remain in the
[comparison record](../benchmarks/2026-10-26-item3-0-sweep/repair-01/decomposition-learning-records.json).

The fresh two-step failure is carried to item 0 at its original budget,
without changing its assertion, chooser, evaluation or seed. It is separate
from the two approved decomposition arms on the original snapshots, which
still pass with companions 6/8. The fixture ports remain staged until the
frozen sweep ends; no production change or publication.


## 33. Equality-close eligibility before class mutation — Codex, 2026-10-26

The slow cold-boundary corpus exposes a partial-update defect. The saved
initialization writes two order-0 occurrences against one order-0 word,
then selects an order-1 bound occurrence against that same word. The close
calls `index_part_row` for the old member before calling it for the new
one; the second call raises its existing equal-order eligibility error.
The equality itself already carries both evidence lanes `(1, 1)`.
The complete read-only trace is retained in the sweep's diagnosis directory.

The candidate close now checks the current member and shared-side orders
before recording a class witness. Unknown or unlike-order testimony remains
an equality with both lanes, but is not a member of that next-order class.
A first witness whose referent has since changed order is renewed in the
same preallocated snapshot seat before a later compatible witness can
raise. The direct index's rejection and `settle`'s structural consistency
check stay unchanged. This applies the index's existing membership contract
before any partial class update; it neither rewrites an order nor discards
testimony. The decided raises for compatible distinct members are unchanged.

Two new cases cover the incompatible witness arriving before and after a
valid one: its evidence is retained, no partial parent is minted, the next
valid witness raises the expected class, and the snapshot slab does not
move. They and all 36 existing order cases pass in the isolated candidate,
with import paths and source hashes audited. Early diagnostic invocations
resolved helper imports to the main tree; those failed logs remain and are
not credited to the candidate. The original corpus replay is still running.

Separately, the recurring-pair and checkpoint fixtures exhaust the default
1,024 LTM rows during their six eight-stream batches. Their construction
bound is now `48 * (8 * (4 + 2*8) + 3) = 7,824`: journal identity capacity
per eight-word reading, plus cover, sentence and estimate. Both cases pass
at the original budgets with their admission and checkpoint assertions
unchanged. The capacity is fixed before either model is constructed.
These candidates remain separate from the running full sweep. A production
repair will receive a new freeze and the five scoped groups as required.


## 34. Prediction-export boundary and matched-state parity — Codex, 2026-10-26

The slow export gate traces public symbol publication after the prediction
has already been computed. That publication reads host-side boundary-pair
identity records and cannot be included in a tensor-only prediction graph.
The candidate adapter uses the ordinary model body, head and output
normalization, ending at that prediction boundary. Source staging, the
arrival tick and the sentence-code snapshot are done by its host staging
step. The export wrapper registers its model as an `nn.Module` child so
captured parameters have their normal ownership. No normal forward or
training path changes for this repair.

The old parity fixture compared consecutive calls on one live model. The
ported comparison starts the full and split paths from copied model state
and the same existing RNG state (through `torch.random.fork_rng`, with no
seed supplied): cold-word admission must see the same initialization as
well as the same arrival tick. The original maximum-error threshold stays
`1e-4`. Parity and Torch export pass in the candidate; a new two-input
exported-program regression and the remaining backend checks are running.
All failed probes, exact fixture faces and source/import records remain.


## 35. Export context and the native bridge reserve — Codex, 2026-10-26

The two-input export regression exposed a second issue beyond §34's adapter
boundary. The parallel body explicitly discarded `_staged_concepts_in` while
compiling. A read-only probe obtains exact exported predictions when it drops
that same input from the eager reference; keeping it reproduces the measured
difference. The candidate removes this compiler-only omission. Ordinary eager
behavior is unchanged; compiled prediction now receives the same evidence.
The export wrapper owns fixed STM seed buffers, copying them for each call
so a graph invocation cannot mutate lifted constants or retain the preceding
invocation's chunks. Both changed inputs and a repeated first input match the
ordinary prediction under the regression's original numerical tolerance.

The native production objective test stopped the slow sweep at 3,571 completed
cases and the 28 GiB process-tree ceiling. A fresh-process replay of that same
case also exceeded the ceiling in 17 seconds; both attempts are retained.
A read-only allocation trace identifies the first oversized allocation:
`_priming_bridge_src[2,148,073,472]`, filled with -1 at construction, followed
by an equally large destination reserve. The fixed bound itself is preserved.
Only prefixes below `_priming_bridge_counts` are ever read, and a revision
rebuild writes every element of those prefixes before publishing the counts.
The candidate therefore reserves these two buffers with `empty` instead of
initializing their inactive tails. It neither resizes nor changes a capacity,
projection, revision, overflow check, vocabulary or native batch.

A new regression poisons the inactive source tail with invalid indices and
valid destinations, verifies exact heat on both towers, repeats a cached read,
and then shrinks the bridge at a new revision. No stale or uninitialized edge
conducts, and both buffer addresses stay fixed. This case, the six placement
cases and the two-input export case pass 8/8. The unchanged native objective
workload now constructs and enters its epoch; its diagnostic run is pending.
The original sweep's other 3,139 unfinished cases continue in their original
order on `03fbbf2a7d9c`, with the interrupted native case recorded separately.
This is completion of interrupted coverage, not a claim that the original
single process finished. All source changes remain in isolated candidates
until that frozen-source coverage and its audit are complete.

The existing two MLX backend XFAILs are also run with `--runxfail`. They get
past Torch export, but the installed edge-lowering verifier rejects a mutated
symbol-basis parameter (`aten_mean_dim`). The original XFAIL markers and
assertions remain unchanged; the new observed diagnostic is retained beside
their historical multi-delegate limitation. It is not counted as a backend
pass or hidden behind the earlier lifted-STM-constant error.


## 36. Remaining corpus fixture bounds and learning evidence — Codex, 2026-10-26

The successor fixture trains eight rows but evaluates a fixed 32-row group.
It now constructs 32 streams. Its original eight concept seats cannot retain
the corpus: the construction reserve is 1,024 seats, from 256 presentations
times one cover and three word identities. Both derivations are beside the
fixture values. The existing initialization, 30 epochs, learning rate .001,
input grouping and success assertions remain. Its recorded-state replay is
running; the earlier stream-capacity and concept-capacity failures are kept.

After §33's equality-close repair, the cold-boundary corpus reaches its strict
learning assertion rather than the partial-class-update exception. At the
original six epochs, four-row batches, 48 corpus rows, .001 learning rate,
boundary rate 40 and update interval 16, both boundary weights remain -4.0.
The original seed 0 and corpus seed 7 were already in the fixture and remain
unchanged. This is item 0's learning remainder; the assertion is not ported.
The run, RNG replay, import audit and budget are retained in the sweep receipt.


## 37. Native inverse width and serial output observers — Codex, 2026-10-26

With §35's inactive reserve no longer filled, the unchanged native objective
workload stays below its memory ceiling (13.06 GiB peak in the first candidate
run) and reaches answer realization. Its inverse then tries to reshape a
native percept field using a zero width. `getEncodedOutputSize` computes the
raw input/output slab ratio; for this 8,192-position input that integer ratio
rounds to zero. `PartSpace.forwardBegin` already uses `_percept_width` after
embedding. `reverseBegin` now uses the same constructed native width when
present, retaining the existing encoded-width rule for other spaces. The
existing matching-call shape and meaning-complement removal remain intact.
Two regressions retain a three-event field despite a stale one-word descriptor
and raw encoded widths of zero and one. They and six dimensional checks pass;
three existing slow prerequisites are skipped in that diagnostic invocation.
The native objective workload is being replayed with the repair. The previous
run and its inverse traceback are retained, not reported as a pass.

The slow serial-output fixtures still read dense `Understanding` compatibility
carriers that are absent for completed fields. The variation check now reads
the completed roots and `AnswerDerivation.conceptual_answer`; the seed
precondition requires both completed fields to hold tensor roots. All three
variation thresholds remain, and that case passes on the saved initialization.
The memorization case completes its unchanged 40 epochs, .001 learning rate,
eight-row batches and ten-epoch evaluation schedule at .78125 best accuracy,
below its unchanged 1.0 assertion. This strict failure is item 0 learning work.
The ninth fixture port and its exact assertion differences are in `repair-02`.


## 38. Native diagnostic cosine storage — Codex, 2026-10-26

The native workload with the inverse repair reaches its first trial, then the
measurement observer itself exceeds the process-tree ceiling. The last emitted
event precedes the observer's broadcasted cosine calculation between all
padded decoder leaves and code rows. That calculation expands the batch, leaf,
code and content axes simultaneously, even when no leaf was emitted. This is
a receipt-probe allocation, not a training objective or another model update.

`decoder_margin_probe.decoder_cosines` now normalizes the two native slabs
with cosine_similarity's same `1e-8` norm clamp and multiplies them. The probe
retains the complete leaf-by-code matrix and the same reported slices; there
is no row sampling, vocabulary reduction or assertion removal. A deterministic
regression checks every result, zero rows, sub-clamp norms and unchanged input
tensors against the original broadcast. It passes. The failed resource run
and this probe port's exact before/after files are retained, and the original
native workload is running again with its batch, inventory and budget intact.

The serial gradient fixture's width assertions also read the constructed
native percept width (104 after identity resolution), instead of the legacy
136 declaration. It still requires realization gradient beyond that native
face and the exact native output width; the strict nonzero answer-gradient
assertion is unchanged. The full sweep's unforced zero-gradient failure is
being diagnosed separately without changing the chooser or adding a forced
walk. There are now eleven pending fixture/helper ports.


## 39. Native output learning and remaining observers — Codex, 2026-10-26

The slow sweep's two development-width gradient cases and the production-width
case without output-in-loop reach a nonzero answer error but a zero conditioner
gradient. The saved-state replay of the first case records .0466072597 answer
error, a nonzero gradient on the actual answer and no conditioner update.
Its final two output trials each emit one role, with two roles still pending
after the original 48 trips. Infix layout starts on the last role; conditioning
changes the first occupied role, which these walks never emit. Greedy emits
once and repeats `not`; the alternative emits once, chooses `null` and then
repeats `not`. Their equal output costs keep greedy. The earlier 28 walks in
the same run emit no roles. These are unforced learning results, retained for
item 0 with one requested batch of two rows and learning rate .001. No chooser,
objective, initialization, budget or nonzero-gradient assertion changes.

The fourth gradient case, production width with output-in-loop, gets beyond
the nonzero total-gradient assertion and fails the old 136-coordinate
realization assertion. It is being replayed under §38's native-width port;
it is not described as another zero-total-gradient failure.

Three output tests still expect a separate `output_policy` loss, which no
production caller writes. The output owner now compares the two complete
walks, while reconstruction trains the shared chooser. Their saved-state
diagnostic confirms the comparison and both chooser parameters' Adam state.
The observer now requires the two-by-two finite comparison and its output
owner trace. The Adam, target mask, nonzero supplied gradient, zero automatic
gradient, exact unchanged weights and target-only scoring checks remain.
The training probe also tests whether the reader registry contains `output`,
not whether that registry is empty: reconstruction can populate it on a row
with no supplied answer. No model loss or training path changes.

The legacy-memory isolation fixture assumed the unforced reading created a
DEF for its lexical referent. It now supplies that definition through the
existing definition fixture before looking it up. Both native-width cases
pass with their original exact root, referent and later-memory-isolation
assertions. This is fixture testimony, not an automatic definition on word
admission. All these edits belong to the same output test file; all failed
port attempts remain beside the exact before/after record.


## 40. Percept credit follows the landed form/meaning split — Codex, 2026-10-26

The fourth native gradient case is not a learning failure. Its saved-state
replay emits its single conditioned role and updates the conditioner: answer
error .0451222733, active gradient norm .0157021638, realization-gradient norm
3.87354565. All realization gradient lies in the native form; the meaning
complement is zero. The width-only port therefore still fails, as retained.

Iteration spec §42 already records the relevant decided swap: "Its
percept-only loss must not have a nonzero gradient to a meaning-only inverse."
The landed `test_percept_answer_keeps_meaning_inverse_and_state_out_of_credit`
checks that separation, and
`test_given_parent_trains_the_meaning_inverse_without_a_form_kernel` checks
nonzero credit from an actual meaning loss. The latter passes on the slow
sweep's exact saved setup state in this diagnostic round.

The native conditioner fixture now requires a nonzero realization gradient
inside its constructed native form and an exact zero in the complement,
instead of requiring nonzero complement credit. Its existing nonzero total
gradient, actual-answer gradient, parameter update, unique optimizer ownership
and exact source-root assertions remain. This ports the observer to the landed
split; it changes no kernel, detach, chooser or objective. The first three
unforced gradient failures remain learning-side with their strict assertions.
The fourth case passes on its original setup state (`slow-native-conditioner-04`,
225.69 seconds); the two existing independent credit certificates are included
in the final scoped repair group.

That replay passes every gradient assertion, then exposes the same obsolete
episode carrier in its final source check: `WhatAnswer.ltm_slot` can be absent
when a completed field declines the episode, as step 4 decides. The fixture
now reads the held `AnswerDerivation.sentence_states` source. It additionally
requires that this is the exact selected source field, and retains the original
concept-width and exact root-value comparisons. It neither forces an episode
nor substitutes a desired answer. The intermediate missing-slot failure is
retained; the complete case passes on the original setup state as recorded above.

The three output-owner ports in §39 pass on their original setup states,
including all no-credit and exact unchanged-weight checks after Adam. Together
with the two lexical-memory cases, five formerly blocked observer cases now
pass. No production change is needed for these ports.

## 41. Current word-cell and padding observers — Codex, 2026-10-26

The slow continuation exposes three capture fixtures still calling the legacy
`_per_word_body_step`. Its WholeLayer receives signed coordinates (minimum
-.9832566, maximum zero) and the native cube check rejects them. The diagnostic
retains the values and exact original setup state. No clamp or cube-assertion
port is proposed. The fixture instead captures the active pipeline's three
native callbacks after the ordinary host staging, and composes them into a
standalone word cell with the current explicit STM carries. Eager execution,
fullgraph execution and two consecutive word steps remain the contracts. The
first candidate passes three cases; a new explicit one-graph counter exposes
two initial/continuation specializations and is being diagnosed.

The padding fixture also observes the retired positional STM signature. It
now reads the current tuple and gate and compares every tuple field on an
inactive row. Its first replay has two real pushes and a closing call with a
false gate; the active source has three entries, while the selected lexical
leaf mask has two. The expected deposit count follows those two leaves,
including an independent count of the two words in the fixed input. The
former concept-tail assertion read a terminal concept with no W-wide tail;
its replacement checks the actual SymbolSpace word-reference handoff, requires
nonempty padding, and checks zero presence/evidence and absent row ids at
every padded position. All failed observer attempts are retained; no seed or
production change accompanies these ports.

The successor fixture's retained 30-epoch replay stops during its first epoch
at Dynamo's default eight-specialization limit, before its learning assertion.
This is recorded as a compiler execution failure, not a learned-successor
result or an accuracy failure. Its original 256-row corpus, batch eight,
learning rate .001 and 30-epoch budget remain recorded unchanged while the
compile boundary is reviewed.

The capture diagnostic identifies the second graph's sole first guard failure
as the initial chunk buffer's `requires_grad=False` versus the returned
buffer's `True`. A tensor-loop carry needs a stable autograd signature. The
fixture now derives that signature from one pure word step under `fork_rng`,
and attaches it to copies of the original zero state without changing values
or consuming the recorded random stream. All four capture cases pass,
including exactly one graph for both words. Both padding cases pass, with
nonempty padding checked on the actual symbolic handoff. These are observer
ports only; the original signed-path and intermediate graph-count failures
remain in the receipt.

## 42. Standalone reconstruction compiler policy — Codex, 2026-10-26

The successor replay's failure precedes learning: its standalone numerical
reconstruction brick compiles without visiting `util.compile`. That established
entry sets a 64-specialization policy; the standalone entry retains PyTorch's
default eight and fails on the next legitimate shape/greedy-replay variant.
The candidate adds the same existing policy at `_compiled_reconstruct`, keeping
fullgraph, static shapes, the selected backend and all decoder mathematics.
It changes no model capacity, training budget, catalog or objective.

`test_sweep_compile_repair` crosses the default limit with nine fixed input
shapes through the real reconstruction compiler, checking exact output and
backward values and reuse of the cached wrapper. It fails at the cache limit
on the audited frozen source, and passes on the audited candidate in 3.74
seconds. The candidate is staged for the next repair freeze. The 30-epoch
successor run's interrupted first-epoch result remains explicit; there is no
accuracy or learning claim from this compiler regression.

The first close-allocation observer wrapper imported Models before the native
probe set its eager backend. Its log shows Inductor/max-autotune instead of
the measured eager backend, so it was stopped and retained without validation
credit. The corrected wrapper establishes the exact probe environment before
any model import and traces only the kept close. This diagnostic does not
change source, tensors, optimizer steps or the production workload.

## 43. Reverse-rule fixture reads the kept sentence — Codex, 2026-10-26

`test_reconstruct_from_idea_rebuilds_after_clearing` passes the terminal CS lane
to the reverse-rule boundary. That lane is not the current owned sentence
idea and includes signed coordinates, so native cube kernels reject it. The
fixture now passes `_last_sentence_understanding.root` with its one-slot axis,
requiring that the kept reading exists. Its cache-clearing checks inside the
real generation call, call-count check and nonempty generated-rule check are
unchanged. The case passes on its exact original slow-sweep state in 22.94
seconds. No production change, cube clamp or alternative initialization is
used; the original failure and before/after file text are retained.

## 44. Native allocation diagnosis continues — Codex, 2026-10-26

The corrected eager close-only probe also hits the resource guard, this time
at 348.23 seconds and 30,106,738,472 bytes after the first trial's diagnostic
gradient snapshot, before any kept close. The varying last event is not enough
to assign the allocation to a close. The next read-only probe therefore
records large bound tensor calls and process high-water increases throughout
the factory, with the normal eager backend established before imports and the
initial RNG state saved without setting a seed. No result is discarded or
counted as a native pass. The compiler-policy candidate's original 30-epoch
successor replay is running separately, without shortening its budget.

The whole-factory Python profile imposed substantial tracing overhead and was
stopped without validation credit. The next periodic stack sampler retained
its exact initialization and reached the memory guard at 347.02 seconds and
30,594,490,656 bytes. Its last frame was a departure comparison, but that
comparison's tensor is about 41 MB and is not itself the multi-GB cause.
Read-only saved-value accounting instead attributes over 5.6 GB in the first
trial to retained dense Householder intermediates, split between object and
group bases. The first sampler's constructor defect, both stopped profiles,
and all resource-limited attempts remain in the receipt. Additional diagnostic
accounting runs use a declared 16 GiB ceiling solely to coexist with the
original slow sweep; they are not validation runs.

The successor replay was stopped before completing its first epoch when the
three concurrent processes together exceeded this 36 GiB machine's memory.
It remains pending at the original 30-epoch budget. This orchestration stop is
neither an accuracy result nor permission to shorten the test.

## 45. Unknown-group basis allocation — Codex, 2026-10-26

All 21 batched group-basis calls in the retained allocation probe have shape
`[28, 1]` and group `-1`; the other 25 calls are the modifier's scalar group
zero. `VerbGroups.forward` nevertheless constructs each unknown group's four
dense 448-by-448 reflection intermediates and retains them for backward before
masking their result to identity. The candidate directly returns the same
identity for an entirely unknown eager group. Mixed/known groups and compiled
tensor control flow retain the existing expression. An empty parameter sum
preserves a present zero gradient, including AdamW moment and decay updates;
omitting that edge would change the optimizer's behavior.

The new regression uses the observed native shape and fails on the frozen
source's discarded matrix construction. A second case checks exact outputs,
gradients, parameter updates and optimizer states after a mixed/known update
followed by an all-unknown update. Both and the 37 existing kernel contracts
pass on the candidate. No basis, order, candidate menu, training budget or
random initialization is changed. The native production certificate is being
rerun with this candidate; a unit allocation regression is not counted as its
completion.

## 46. Stored surface tiling observer — Codex, 2026-10-26

The slow tiling fixture interprets WordIdentity's internal atom and canonical
form keys as surface bytes. Its first counterexample is the metadata key
`\x00identity/atom/pair/#h`, not a percept spanning two words. The port retains
the original input-to-decoded-tile equality and the one-tile check for every
ordinary stored surface. It additionally verifies atom-key row ownership and
checks each canonical form's decoded word against the identity row map and the
same one-tile assertion. The case passes on the exact original sweep state;
the metadata counterexample and complete before/after text remain recorded.

The separate 160-epoch exact-reconstruction acceptance case still reports zero
exact matches on initialization 1506395199. Its strict assertion and budget
are unchanged and the learning result is carried to item 0. Three other
roundtrip failures stop at the already diagnosed standalone compiler limit;
they require replay with that repair, not an accuracy classification.

The unmodified native certificate replay with the unknown-group candidate still
hits the guard at 348.02 seconds and 30,739,833,984 bytes before completing its
first trial. That failure remains in `native-candidate-04`; no native pass is
inferred from the 39 kernel checks. Further allocation attribution reuses the
same retained diagnostic RNG state and leaves the original workload unchanged.

## 47. Forward reflection evaluation without dense bases — Codex, 2026-10-26

On that same diagnostic initialization, removing the unknown-group matrices
reduces the retained group intermediates, but object bases and their forward
matrix products still retain over 5 GB. The next candidate evaluates the same
orthogonal forward action directly as its existing sequence of reflections.
For `Q = H1 ... Hk`, `Q.T x` applies H1 through Hk, and `Q x` applies them in
reverse order. Each intermediate has two evidence vectors per operand instead
of every column of a dense square matrix. Parameters, reflection count,
spectrum, support masks, group/object selection and the restricted inverse
solve are unchanged. No cache, capacity change or additional retained buffer
is introduced.

The allocation regression uses the observed native `[28, 1, 896]` input and
fails on the frozen dense implementation. Nine new cases check both reflection
directions, zero vectors, known/unknown groups, object masks, modifier bases,
all input/parameter gradients and the absence of dense forward construction.
They and the 37 existing kernel contracts pass. Dense-reference comparisons
use double precision at 1e-10 or tighter; two additional fullgraph cases have
exact eager/compiled values and gradients. The native certificate is rerunning
on this candidate. Its earlier resource failures remain separate, and no
native completion is claimed yet.

## 48. Remaining native-loop fixture observers — Codex, 2026-10-26

The slow sweep reaches additional observers of the pre-chunk carriers. These
are fixture ports, with original setup states replayed and all failed attempts
retained. The main source remains the `03fbbf2a7d` freeze during collection.

* Reverse traversal now installs the complete STM tuple with the fixture's
  supplied depths. Its snapshot assertion reads the lookup view of
  `SymbolEntries`, excluding prepended identities; the original word/object
  half checks remain. Both cases pass (`slow-reverse-traversal-port-02`).
* The rule-gate fixture's first source sentence has 19 words but its declared
  bucket held eight. The test derives a construction bound from that same
  sentence and uses the current native staged callbacks. Its false-gate zero
  contribution and exact unchanged STM checks remain, extended to all chunk
  fields. All four gate cases and four capture cases pass on their original
  setup states (`slow-rule-gate-port-02`). The earlier config-only attempt did
  not survive `from_config`'s reload and is retained as a failed setup.
* Serial metadata now checks the full native atom spelling, including each
  word's exact row sequence: the six-character word has seven boundary pairs
  and six length atoms, so P is 13, not six byte rows. The push observer reads
  the gate argument of the tuple-based API. The unity observer hooks the
  instance's construction-installed compiled callable rather than the now
  shadowed class method. The existing exact commit and live-unity assertions
  stand. Their scoped replay is pending.

The review-17 audit fixture also lacked its imported decomposition observer.
The exact historical helper is restored from the retained 2026-10-03 receipt.
Its fixed W=2 assertion predates the actual nonempty walk menus: the current
unforced field has only narrowing eligible. The port independently reads all
menu eligibility masks, checks the reported walk count against their nonempty
count, and retains the exact proposal factor, finite-difference and gradient
checks. The storage observer counts stamped arrival entries (including covers),
not estimates or unstamped operand ideas; the exact four-entry and witness
assertions remain. Empty pre-close snapshots must be representable. These
observer changes are being replayed; no learning claim or chooser change is
made. Before/after faces and added helper provenance will be in the receipt.

## 49. Native post-update memory diagnostic — Codex, 2026-10-26

The matrix-free forward candidate completes both updates of the first native
trial pair, then still reaches the 28 GiB process-tree guard at close:
`native-candidate-05`, exit 137, 1,046.879 seconds, peak 30,278,411,032 bytes.
Its 18 events include both optimizer displacements and the selection. This is
an incomplete workload, not a passing native gate. The original 28-row batch,
configuration and one-epoch budget are unchanged. The retained diagnostic
initialization is now replayed with read-only stack/footprint sampling to
locate the remaining allocation (`native-memory-sampler-03`). The full slow
continuation remains independent on the main frozen source.

## 50. Observer replays and thought update accounting — Codex, 2026-10-26

The three serial metadata ports pass on the original slow-sweep setup states
(`slow-serial-meta-port-04`). Each native spelling is checked against
`WordIdentity.parts`, not the trie decoder's promoted whole-word percept. The
physical-push assertion excludes the already-decided host closing tick and
still requires exactly two lexical pushes in each stream. The live property
consumer is observed through the same eager loop bodies; graph capture has
its separate unchanged capture gate. Earlier incomplete isolated-fixture setup
runs and incorrect observer attempts remain in the receipt.

The restored review-17 observer passes all its remaining original semantic,
gradient, ownership, geometry and final-program assertions
(`slow-review17-observer-08`). Its additional stale assumption was that only
the two compose optimizer steps occur. Step 4 §15's ordinary thought-reader
exposures add one batch-end update: the captured output owner has no gradient
on exploit, a gradient on explore, then the `thought_answer.answer.what` and
`thought_comparison.answer.what` gradients at batch end. The sentence reader
update counter remains exactly one; both readers' Adam counters are exactly
two; the total optimizer events are exactly three. The port asserts each of
these boundaries, not just an enlarged count. No optimizer, thought chooser,
objective or exposure rule changes. The original failure and every intermediate
observer replay remain separately recorded. The native child runs use their
original ambient initialization rule; replaying the parent's setup state is
not represented as pinning the child's RNG.

## 51. Two-stream replay and nested audit harness — Codex, 2026-10-26

The slow continuation's two-stream failure is the same nonpersistent form-cache
observer mismatch already ported in §33. With that helper, the original seed
4199722147 and immutable initialization snapshot, the inherited proof and all
kept-reading assertions pass unchanged (`slow-two-stream-port-02`, 39.51 s).
The first isolated attempt lacked the snapshot at its expected path; its
failure remains, and the second reads a byte-identical copy of that file.

The review-17 subprocess passed standalone but failed under the final receipt's
`layout_capture` instrumentation: it replaced `PYTHONPATH` while inheriting the
parent's plugin options. Its fixture now removes only inherited
`PYTEST_ADDOPTS` before installing its own explicit observer. The actual child
assertions and its initialization rule are unchanged. The instrumented before
run fails to import `layout_capture`; the after run passes
(`slow-review17-instrumented-before-01` / `after-01`). This is the same nested
measurement-boundary correction recorded in step-2 §89, not a runtime change.

## 52. Thought-reader frozen values — Codex, 2026-10-26

The unchanged diagnostic initialization reaches both trial updates and then
answer generation, where the retained stack enters the ordinary output walk.
`native-memory-sampler-03` stops at the memory guard after 1,077.30 s, peak
30,221,115,176 bytes. It remains a failed diagnostic, alongside candidate 05.

`ThoughtAnswer.scorer` installs a saved-tensor hook that clones every use,
overriding the enclosing sentence's deduplication. The scoped candidate uses
`SentenceCompose.saved_sentence_values` for the same reader region: mutable
parameter/buffer views are frozen once per version; immutable intermediates
retain their storage with the existing version check. The score, graph edges,
reader exposure, optimizer timing and output objective are unchanged. No new
buffer, cache capacity or runtime state is introduced.

The deterministic regression repeats a parameter and buffer read sixteen
times. The original source copies each sixteen times and fails the allocation
assertion; the candidate copies each once and returns the exact same value and
both gradients, even after the live parameter/buffer are changed before
backward. The initial run passes 49 cases including the new regression; two existing lightweight
judge fixtures lacked the native module's empty `parameters`/`buffers`
iterators. Supplying those empty iterators preserves both original score
assertions. Their repaired replay and the new regression pass (three cases).
Logs: `slow-thought-save-before-01`, `after-01`, `after-02`. The full native
candidate 06 is running; no pass is claimed yet.

`native-saved-values-04` was stopped because its isolated source was edited
while that attribution probe was active. Its process result and explicit
stop reason are retained, and it supplies no validation or homogeneous-source
measurement. The main frozen source and the ongoing full sweep were untouched.

## 53. Streaming smoke's native dictionary bound — Codex, 2026-10-26

The 200-document stream smoke fails at its first native identity admission:
256 percept seats hold byte fallbacks but leave no room for the new word's
boundary/length atoms and canonical row. The candidate keeps all 200 documents
(8,670 sentences), the one-epoch budget, batch four and the existing 300-second
subprocess timeout. Before construction it counts all source units, their
input-width prefixes, base atoms and every possible triple/quadruple mint key.
The bound is 257 byte/padding seats + 19,582 canonical units + 67,011 atom keys
= 86,850 percept seats. No training trace or seed determines this capacity.

Only a temporary fixture XML receives the bound; its shard path is absolute
and its existing checkpoint destination is placed in the test's directory.
The subprocess uses the running interpreter, avoiding an accidental skip in
an isolated source tree. The production configuration is unchanged. The
original exit failure and the candidate replay are retained; a completed
epoch has not yet been certified.

## 54. Whole-sentence capture boundary — Codex, 2026-10-26

After the streaming fixture's native capacity is supplied, its first batch
reaches a production capture defect: the nonserial compiled `forward` enters
`_publish_reading_symbols` and `PerceptField.read_percepts`, whose native
membership dictionary performs data-dependent host comparisons. This public
symbol update follows the numerical head and is not an input to its prediction.
The candidate compiles the ordinary `_forward_body` and `_forward_head`
separately, leaving the existing sentence adapter and native publication on
the host. Its configured fullgraph policy and tensor computations are unchanged;
no state, capacity or objective is added. Serial capture paths are unchanged.

The new regression observes two full numerical graphs, one eager native
publication, and a bit-exact prediction against the same initialized eager
model. The frozen source fails by attempting that publication inside the graph;
the candidate passes. Initial regression setup attempted to deep-copy a staged
autograd tensor; staging under no-grad fixes that setup without altering the
assertions. Both failed setup runs are retained. Evidence:
`slow-parallel-boundary-before-01/02` and `after-01/02`.
The original 200-document, one-epoch stream replay remains pending.

The successor replay `slow-successor-port-04` exhausted its 8-GiB process-tree
ceiling after 2,776.53 seconds (peak 8,590,549,032 bytes). This is an incomplete
resource-limited run, not a learning verdict. Its original initialization and
30-epoch budget are retained for continuation by a fresh replay.

## 55. Native bridge index width and remaining resource cases — Codex, 2026-10-26

The slow continuation stopped at the server class's first setup with a
32,453,654,024-byte process-tree peak. Its remaining four class cases are
explicitly deferred with that shared setup; they are not counted as passes or
completed tests. The next continuation preserves the remaining collection
order and retains its fresh setup states. Its GPU placement case independently
reports a 16.50-GiB MPS allocation request, exceeding the device's 16.85-GiB
working-set allowance after the other model allocations.

Both requests are the bridge endpoint slabs: 65,536 concept seats times
(32,768 percept + nine property seats), twice, at eight bytes per endpoint.
Uninitialized allocation avoids touching their tails on CPU but does not avoid
the reservation on MPS. The candidate preserves exactly the same edge capacity
and uses signed integer offsets with a construction-selected width. Cardinality
up to 65,536 fits int16, up to 2**32 fits int32, otherwise int64. Destinations
also reserve the discarded -1 sentinel. Invalid destinations still conduct
nothing. Only the counted prefix is decoded for indexing; revision rebuilds,
edge multiplicity, projected values and cache invalidation are unchanged.
The native pair now occupies one quarter of the bytes, approximately 8 GiB.
There is no resize, new configuration or checkpoint payload (these caches are
nonpersistent). The derivation is beside the allocations.

Two regressions pass: projection with repeated, invalid and poisoned-tail
entries, revision reuse and shrinkage, and exact round trips at each integer
width boundary. The original GPU allocation failure remains in tail 03; its
candidate replay and the existing placement/priming contracts are pending.
Evidence: `slow-bridge-width-after-01` and `slow-gpu-bridge-port-01`.

The preceding native candidate 06 still exhausted its ceiling after 1,072.19
seconds, peak 30,171,292,488 bytes. The stream epoch's second replay exhausted
its smaller diagnostic ceiling after 275.14 seconds; the roundtrip's first
case exhausted 8 GiB after 1,100.83 seconds before the second case started.
These are retained incomplete resource measurements, not successful epochs
or learning failures. Their full budgets and numerical assertions remain.

## 56. Bridge masks preserve the full relation across five stages — Codex, 2026-10-26

The first width reduction passes 19 existing placement/priming contracts, but
`BasicModel.xml` constructs five conceptual stages. Its GPU replay still fails
while reserving the next 4.12-GiB endpoint array (14.42 GiB already allocated).
That failed run remains `slow-gpu-bridge-port-01`.

The cache now packs each source's destinations into 32-bit masks. Its fixed
block capacity is `nVectors * (ceil(partCap/32) + ceil(wholeCap/32))`, so it can
still represent every possible edge in either tower. Source and block indices
use the lossless offsets of §55; masks use signed int32 with exact bit-31
handling. At the native inventories the three arrays occupy 537,395,200 bytes
per stage. The 32 bit positions are a construction-allocated buffer. Only live
blocks are decoded; edges retain source/destination order and their original
addition order. Invalid-only groups still trigger the prior decay with no
conducting edge. Revision rebuilds and loud exhaustion remain. No vocabulary,
capacity of the represented relation, model value or objective is reduced.

The counted-prefix/shrinkage regression, integer-width boundaries and a full
64-destination case spanning bit 31 and two blocks all pass. Together with the
unchanged placement, energy and diffusion files this is 22 passes in
`slow-bridge-block-after-01`. The GPU replay is pending as
`slow-gpu-bridge-port-02`. The previous representations and their failed runs
remain as diagnostic history, not candidate passes for the full model.

## 57. Preserve double-precision form auditing on MPS — Codex, 2026-10-26

With the block cache, the five-stage GPU replay gets past allocation and
reaches `SymbolCentroid.place`: its diagnostic cosine converts an MPS tensor
to float64, which MPS does not support. This is a scalar report, not a learned
or returned form. The candidate copies the stacked report input to CPU before
the existing double-precision normalization and sum. The returned forms retain
their original device, dtype and exact coordinates; the report's arithmetic
and precision remain the same.

The original source passes the CPU regression and fails its MPS counterpart.
Both pass on the candidate, with the whole report equal to the CPU reference
and both original and returned forms unchanged (`slow-form-audit-before-01`,
`after-01`). The full GPU replay continues as `slow-gpu-bridge-port-03`.
The native saved-value attribution runs separately on an immutable candidate
and the retained diagnostic initialization; it remains diagnostic only.

## 58. Priming's double-precision energy report — Codex, 2026-10-26

The GPU replay next reaches `Priming.diffuse`'s before/after totals, which also
requested float64 on MPS. Only those diagnostic reductions now read a CPU
copy. Diffusion, its feasible budget, surface writes and degree remain on the
original device. Both original MPS audit regressions fail; CPU and MPS pass
with exact scalar totals and surface values after the repair. Together with
the three existing diffusion cases this is seven passes in
`slow-device-audits-after-01`. `slow-gpu-bridge-port-04` then reaches the
unchanged 120-second subprocess limit. That timeout remains a failed run.

The server class now starts and its health and empty-input checks pass. The
three inference checks exhaust their original 30-second request limits
(`slow-server-bridge-01`: two passes, three timeout failures). The interrupted
parent setup is replayed; the child retains the original ambient initialization
rule. The four siblings blocked by the original class setup have their first
complete candidate setup states recorded. Neither these timeouts nor the GPU
timeout is called a learning failure or a pass; no deadline is extended.

## 59. Retain candidate context once without changing the affine — Codex, 2026-10-26

The retained-state native attribution reaches 13,748,140,767 bytes of live
saved storage before its 24-GiB process ceiling. Eight saved candidate inputs
alone occupy 3,120,133,632 bytes: each repeats the shared context for three
candidates. Its memory-stop result remains `native-saved-values-05`.

The candidate scorer's first affine now retains the normalized detached
context and candidate keys separately. Its forward constructs the identical
dense input and calls the same affine; backward reconstructs that immutable
input for the same weight-gradient matrix product and bias reduction. It
retains no extra model state. Parameters, checkpoint keys, first-layer shape,
reader detachment, normalization, saturation behavior, and the existing
`hidden` module hooks are unchanged. No chooser objective or selection rule
changes. Tensor calls to the first layer keep the original linear path.

The original allocation regression fails at the recorded native input shape.
The candidate passes that regression, four exact eager/compiled float32/float64
score-and-every-parameter-gradient comparisons, and seven existing scorer
contracts (12 passes total). Input gradients remain absent. Evidence:
`slow-attention-storage-before-01` (four passes, allocation failure) and
`after-01`. Full native candidate 07 runs on a separate immutable copy.

## 60. Tied-reader fixture ownership and content bands — Codex, 2026-10-26

Three slow inverse fixtures populated the excluded percept band as content or
expected a content inverse to differentiate through that band. The verb and
repeated-occurrence fixtures now supply `content_operand` values before
composition; their inverse and recompose assertions are unchanged. The
selected-row gradient expectation is zero in the excluded band, while the
inactive passthrough row still has derivative two everywhere. All finiteness,
nontriviality and unused-inverse checks remain. The original seed 812 is
unchanged. `slow-tied-migration-port-01` passes all three original-state cases.

Two compiled backward fixtures used a later STM end as an independent input
to a packed reader that reads held sentence fields. The candidate gives the
last presented held field its own differentiable input, selected by presented
sentence ids, and assembles it with the other held roots. The direct reader
receives that same field's root and depth. Both original nonzero-gradient
assertions and repeated-backward equality/cache-count assertions remain.
The exact-state replay is pending as `slow-tied-compile-port-01`.

## 61. Preserve caller input reuse in the scorer storage repair — Codex, 2026-10-26

The original concatenation also insulated saved candidate keys from reuse of
a caller's input buffer. The first optimization retained a view of those keys;
a new mutation regression correctly rejects it. The final candidate clones
only the small key tensor. Normalization already owns the large context's
storage. This preserves the original input-reuse contract without retaining
repeated contexts. Scores and every parameter gradient remain exact; reader
inputs still receive no gradient. Six new cases plus the seven existing scorer
contracts pass in `slow-attention-storage-after-02`. The failed intermediate
input-reuse test is retained. Native candidate 07 uses its own immutable copy
from before this small key-copy correction and is not labeled as final-source
validation.

## 62. Explicit numerical fixtures for retained backward — Codex, 2026-10-26

The first ownership port in §60 still failed both positive-gradient checks.
The source explains why: fixed-identity `_byte_word_cost` is a discrete audit
with a zero numerical adjoint, and the legacy free inverse returns detached
dictionary children. Nonzero numerical gradients are not contracts of those
paths. No decoder or training objective is changed to obtain them.

The two compiler fixtures now declare a numerical reading with a supplied
co-operand, using the production `reverse_binary_step` tied affine inverse,
as `test_tied_affine_inverse_uses_the_retained_operand` already does. The
model's actual `_compiled_reconstruct` still owns full graph capture, buffer
donation and cached backward. The explicit fixture has no occurrence band;
fixed linear content supplies the numerical domain without any RNG draw.
It stages the same original surfaces, and independently differentiates the
last presented held field beside the other roots. The second case keeps
its three input offsets and tests the real compose parameters.

The first version of this supplied fixture omitted `_carries_with_grad`.
Its eager adjoint reached all fields, but the compiled loop credited only
the last trip: the already documented PyTorch checkpoint behavior that the
production helper handles. The fixture now uses that existing helper. No
production change or new workaround was needed. Both cases pass on their
original saved setup states in `slow-tied-compile-port-03`; the original
positive-gradient assertions, repeated-backward equalities and warmed-cache
count assertion are unchanged. Ports 01 and 02 remain failed records. The
before/after file faces and all assertions are in repair-02's port manifest.
The native free-reader tests remain in the unchanged sets; these two tests
make a compiler claim, not a gradient claim about discrete identity readback.

## 63. Completed sweep coverage and remaining budgets — Codex, 2026-10-26

On frozen source `03fbbf2a7d9c`, the default target completed all 6,711 cases:
6,386 passes, 324 skips, one permitted XPASS, no failures. The slow target
and its retained continuations completed 6,705 distinct cases on that same
source. The native objective case and the shared server setup hit memory
guards; four server siblings were consequently unexecuted on that source.
The repaired candidate measured all five server cases separately: two pass,
three exceed their original request deadlines. They are not frozen-source
passes. The GPU case reaches its original 120-second deadline after the
allocation and device repairs. Native candidate 07 still hits the memory
guard after both first-pair updates (1,063.10 seconds, 30,484,439,880 bytes);
no completed native close or successful epoch is inferred. Coverage, source
and protocol integrity are audited in repair-01/full-sweep-coverage-integrity.json.

Both supplied neighboring-sentence answer tests retain their strict 10%
improvement bar after 30 updates, B=2, lr=.001. Future error is
.0860480815 → .0965504199; past is .0799439251 → .0799437165. Their answer
masks and gradient checks pass. They are item-0 learning records, with the
original failed assertions and setup states retained.

The pending one- and three-epoch reconstruction cases retain
`run_config(seed=None)`: that function draws from `SystemRandom` and records
its own seed. Restoring Python/NumPy/Torch setup state is not a replay of
that OS-random draw. Receipts distinguish the parent-state replay from the
training initialization. A supplementary three-epoch XOR run on candidate
source records seed 1646304963 and exact match 0, whereas the original sweep
record used 2197030207 and reached .5; neither result is discarded or used
to select a seed. The original recorded seed is replayed once solely to
check the source comparison. The original 30-epoch successor case retains
its existing seed 0 and corpus seed 0. No budget or assertion is weakened.


## 64. Final repair freeze — Codex, 2026-10-26

After the coverage/source audit, the 27 fixture/support paths and nine
production repairs were applied to main's existing working tree, together
with twelve regression files. One native lesson smoke passed. The new content
freeze is `6fe273d2f12a70d49858137202a0edf8a4c192f3026c882b8f94855a7df589a8`;
its archive, immutable protocol hashes and 48 exact before/after file faces
are in `doc/benchmarks/2026-10-26-item3-0-sweep/repair-03/`. Protected
`IdeaEvaluation`, `begin_presentation`, and the serializer-fixed `embed.py`
match their baseline hashes. The seven scoped groups are measuring this hash.
The original source `03fbbf2a7d9c` and every sweep/diagnostic failure stay
separate. No commit or publication occurred.

The supplementary XOR source comparison at its original recorded seed
2197030207 reproduces the original output loss .17612890899181366,
reconstruction loss .0036852285265922546, exact match .5, and location recovery
1.0. The fresh-draw failure remains a distinct learning record; this comparison
is not substituted for a required gate. On the new freeze both original
512/768-budget decomposition arms pass their proof assertions, retaining the
6/8 companion total and full first premises in both streams.


## 65. Frozen repair gates and exact initialization audit — Codex, 2026-10-26

Source `6fe273d2f12a7` completes 1,812 scoped passes, ten standing prerequisite
skips and no failures, plus the one native smoke. Core contributes 1,631 passes
and ten skips, packing four, XOR eight, both decomposition arms two each,
artifact/serializer files 94, and scoped repairs/ports 71. All 1,641 inherited
core setup states and 26 inherited slow-port setup states match the recorded
initializations exactly. The source archive, all frozen protocols and protected
files match; no source edit followed the freeze.

The original four decomposition seed/snapshot/budget combinations retain 6/8
companions and full first premises in both streams. The read-only percept
certificate passes all 40 atom edges and six word-pair edges; whole entropy is
still only diagnostic. The silence receipt separately counts 247 spoken when
silent and one silent when asked. No output-learning claim follows.

A collection-only audit accounts for all 6,711 prior cases, with the documented
`test_input_word_cursor` predicate rename, 35 new regression cases and ten
new generated document-link cases. The first audit mistakenly put that rename
in the capture-gate file; its failed inventory result is retained. Correcting
the audit mapping required no source change. No test was retired and no second
full sweep was run. The long original-budget training replays remain pending;
these scoped passes are not a claim that their learning or runtime gates pass.


## 66. Runtime replay memory allowance — Codex, 2026-10-26

The original one-epoch reconstruction case still did not complete at the
16-GiB diagnostic ceiling: `slow-roundtrip-budget-replays-03` stops with exit
137 after 3,106.82 seconds at 17,182,292,008 bytes. Its following three-epoch
case never started. This lower-limit stop is retained without a learning
result or a full-budget runtime verdict. The requested full sweep used a
28-GiB allowance, verified as 30,064,771,072 bytes on this host.

The separately started three-epoch diagnostic recorded its unfiltered OS draw,
1303565034, but was stopped after 143.73 seconds to schedule a 28-GiB replay
without another large training process competing for memory. Its exit 143 and
explicit scheduling reason are retained. That same recorded draw will be
replayed; no result was observed or used to select an initialization. The
one-epoch case retains its original fresh-draw policy, with the draw now
recorded before model construction. Both retain their original epochs, data,
work limits and assertions. The seed-0 successor diagnostic subsequently
reaches its lower guard after 6,732.14 seconds at 17,179,949,120 bytes. Its
original-budget replay therefore starts first at the full sweep's 28-GiB
allowance; the interrupted diagnostic remains `slow-successor-port-05`.

These sequential replays use unchanged frozen source `6fe273d2f12a7` in the
main working tree, with their exact observer hashes recorded separately.
The frozen seven-group protocol remains unchanged. All ten newly collected
document-link cases pass. No source change, commit or publication follows from
this resource scheduling decision.

## 67. Halve optional slow work — Alec's direction, Codex, 2026-10-26

Alec requests roughly half the RUN_SLOW testing time, including fewer
stability repetitions. This supersedes the full-repetition requirement for
the optional slow workloads in this round. `RUN_SLOW=1` now uses a declared
`RUN_SLOW_FRACTION=0.5`; `RUN_SLOW_FRACTION=1` retains the historical budgets.
The test-only `slow_count` rounds upward and preserves structural minima.
No production/data file, assertion threshold, seed or test selection changes.
Single-shot contracts, explicit compilation checks, recurrence witnesses,
the required decomposition work budgets and the already measured five groups
keep their existing evidence.

The explicit ports halve long reconstruction/memorization epochs, repeated
answer/stability updates, and the remaining successor/ladder training batches
per epoch (32 to 16). Those three replays keep the original corpus, epochs
and evaluation; they use the existing setup states and seed policies. The
joint-cost fixture retains two disjoint five-observation comparison windows.
The receipt has every before/after face and verifies exact assertion syntax
after removing only the budget wrappers from call arguments.

The earlier full-budget queue was stopped at the user's request after 39:27
of its successor run. The observation, logs and explicit interruption remain;
neither reconstruction sibling had started. Three reduced replays are each
capped at 45 minutes, bounding their total to 135 minutes, with no automatic
retry. A timeout stays a recorded failure and provides no learning verdict.
This bounds the remaining diagnostics within roughly half the preceding
four-to-six-hour estimate, in addition to halving repeated work. Fixed model
construction and compilation costs need not halve with the iteration count.

Receipt: `doc/benchmarks/2026-10-26-item3-0-sweep/repair-04-slow-budget/`.
The ten budget-policy checks pass. A reduced priming stability case and the
three pending replays follow on a separate test-source freeze; the model and
data match `6fe273d2f12a7` exactly. No new full sweep, commit or publication.

The test-source freeze is `ee7ca4e920bc691ae65885228d72ec441c6be73564d9d911766b30e0774533d3`.
The three-epoch stability replay passes on its original recorded setup state
in 4.65 seconds including process startup. Its full-budget predecessor and
the ten passing policy cases remain separately identified in the receipt.

## 68. Reduced-budget results and sweep handoff — Codex, 2026-10-26

The test-only freeze `ee7ca4e920bc691ae65885228d72ec441c6be73564d9d911766b30e0774533d3`
passes all ten budget-policy checks and the reduced stability replay. Its
production and data files match `6fe273d2f12a7` exactly. All three final
training attempts reach their declared 45-minute ceilings: successor and
one-epoch reconstruction each stop after 2,701.02 seconds, and three-epoch
reconstruction after 2,701.01 seconds, all with exit 124. These are retained
runtime failures; no completed learning assertion or successful epoch is
inferred. No retry or further test remains queued.

Successor retains seed 0. The one-epoch reconstruction's unfiltered fresh draw
is 1168686015; the three-epoch reconstruction replays the interrupted attempt's
recorded draw 1303565034. No outcome selected an initialization. All 15 changed
existing function bodies have before/after text; names, decorators, assertions
and seed calls are unchanged. Source, archive, protocol, port-face and
initialization audits pass. Halving repeated work does not claim an exact
halving of wall time, because model construction and compilation still cost
time.

The receipt at `doc/benchmarks/2026-10-26-item3-0-sweep/repair-04-slow-budget/`
retains all 61 original failure/interruption records and adds these outcomes
without replacing prior evidence. The complete default sweep on `03fbbf2a7d9c`
remains 6,386 passes, 324 skips and one permitted XPASS. The repaired model at
`6fe273d2f12a7` retains its native smoke and 1,812 scoped passes with ten
standing skips. The slow sweep is not a complete passing run; runtime stops
and learning failures remain explicit, with the 6/8 companions and other
learning records carried to item 0 at their measured budgets. Return through
Alec for Claude's sweep review. No commit or publication occurred.

## 69. Claude's review of the serializer round and the item 3.0 sweep (test freeze `ee7ca4e920bc`, model `6fe273d2f12a7`) — 2026-10-26: approved for landing; three items carried

Reviewed: §§20–68; the [serializer receipt](../benchmarks/2026-10-26-item3-0-serializer/README.md)
with `embed.diff`, `embed-scope.json` and its six ports; the
[sweep receipt](../benchmarks/2026-10-26-item3-0-sweep/README.md) with
`run-01`, `repair-01` (its six production diffs and ports), `repair-03`
(`application.json`, the nine production patches, `integrity.json`,
`final-inventory-audit.json`, `slow-resolution.md`), and
`repair-04-slow-budget` (`frozen-source.json`, `slow_budget.py`,
`slow_tests.py`); every `frozen-source.json` of the round against the tree;
the new text of `PartClosing._observe_equality`, `ConceptIndex.index_part_row`,
`MeaningKernels.householder_action`/`_basis_action`/`VerbGroups.forward`,
`CandidateAttention._ReaderAffine`, `ThoughtAnswer.scorer`,
`SentenceCompose.saved_sentence_values`, the bridge storage in `Spaces`,
`Models._open_reference_mask`, the compile split and `_staged_concepts_in`,
`export_mlx`, and `test/test_orders_learned.py`'s two raise gates.

**Verified.** All four freezes recompute from their inline snapshots to the
declared values: `caece43fc022c` (900 paths, serializer), `03fbbf2a7d9c6`
(901, sweep repair), `6fe273d2f12a7` (914, final model repair),
`ee7ca4e920bc6` (916, test budgets). The working tree equals the last on
every path; it differs from `6fe273d2f12a7` on exactly the eleven declared
test paths and the Makefile, and `model_and_data_exact` holds, so the model
under review is `6fe273d2f12a7`. `integrity.json` records the three
protected files at their baseline hashes (I did not recompute the function
hash of `begin_presentation`). Results as stated: serializer 1,741 passed,
ten standing skips; default sweep on `03fbbf2a7d9c` complete, 6,386 passed,
324 skipped, one non-strict XPASS (the §16.6 marker), zero failed; final
scoped groups on `6fe273d2f12a7` 1,812 passed, ten skips; budget policy
10/10 and the stability replay on `ee7ca4e920bc`. Companions 6/8 on the
original four combinations with full first premises; percept certificate
40 atom and six word-pair edges. Inventory 6,711 to 6,756 with one explicit
rename and no unexplained missing case; my own diff of `test/` finds
eighteen test functions renamed, each with a successor that carries the
contract to the landed layout, one new case (grammar version rejection), no
test file deleted. No `manual_seed` or generator added in `bin/`. The
`embed.py` change is the version field, its validator at save, load and
view, and the removal of `order_delta`, nothing else; the five artifact
files and `test_typed_stack.py` carry before and after text. The nine
production repairs each have a regression that fails on the frozen source
and passes on the candidate, and the scoped groups were rerun on each new
freeze. Codex's "the repository makefile is unchanged" is true of the
recipes; repair-04 added a two-line comment, recorded here.

**The production repairs, read.** Within the standing rules: `_part_order`
reads a word's native constituents, so a chain naming the word does not
re-order its form; `_open_reference_mask` reads the kept clause's resolved
journal references, with explicit forward annotations still open;
`_iter_relation_rows` requests the Part projection explicitly;
`TruthGroundedReasoner.query` reads `point`; the two evaluation batch bounds
touch reporting only; the noncanonical path keeps the preallocated
`SymbolEntries` views. §33's equality eligibility applies the rule
`index_part_row` already enforced, before any partial class write, and
renews a witness in its fixed seat. The compile split keeps eager unchanged
and lets the compiled prediction read `_staged_concepts_in` at the first
stage as eager does, certified bit-exact; that is item 1's parity moved the
right way. The `recompile_limit` write in `_compiled_reconstruct` is
process-wide and equals `util.compile`'s policy. The bridge keeps the same
complete relation in counted prefixes with lossless signed offsets and
32-bit destination masks, one quarter of the bytes, with the poisoned-tail
and bit-31 regressions. `householder_action` applies the same reflections
without the dense matrix; its forward `allowed` reproduces `basis`'s
support rule case for case, the dense comparison is in double at 1e-10,
and the inverse still solves through `basis`. The eager identity shortcut
in `VerbGroups.forward` is bypassed under compilation and keeps the zero
parameter gradient edge. `_ReaderAffine` keeps the first layer's parameters
and checkpoint keys, detaches its inputs as before, and returns exact weight
and bias gradients. `saved_sentence_values` is the existing sentence
boundary helper applied to the reader region. The two CPU copies are scalar
diagnostics only. `export_mlx` is the opt-in script; its adapter owns the
STM seeds as buffers and mirrors the forward's arrival boundary.

**Findings.**

1. *The slow sweep is not a complete passing run, and nothing here asks for
   one.* Alec (2026-10-26): no `RUN_SLOW` before November at the earliest.
   Of the 61 ledger records, the fixture ports pass on their recorded
   states, the production repairs carry regressions and rerun groups, the
   learning failures keep their assertions and budgets for item 0 (blind
   XOR 0.0 at 160 epochs on two fresh seeds, the scaffold reconstruction
   0.0, the two generation and three native conditioner zero-gradient
   cases, the two-step decomposition 0/2 at 571/768, arithmetic .3125/.5234,
   serial output .78125, the cold boundary at -4.0, the two neighbouring-
   answer arms, the fresh three-epoch XOR draw), and four are runtime
   failures: the native production objective exceeds 28 GiB after both
   first-pair updates even after three allocation repairs, the GPU
   placement case reaches its 120-second deadline, three server requests
   exceed 30 seconds, and the three long replays reach 45 minutes at half
   work. These runtime failures are item 1's, beside the 2.6x slowdown;
   the native case needs the remaining allocation attributed before any
   repair (Householder intermediates and candidate contexts are found, the
   rest is not). None blocks the mechanisms.
2. *The 1-to-2 raise is certified only with the shared side forced.* Gate
   1's second clause ("Felix is a cat. Ciao is a cat." raises cat 1 to 2)
   is `test_only_distinct_published_identities_raise`, which calls
   `force_order(cs, word, 1)` on the shared word before the two order-1
   identities are equated to it. `index_part_row` requires the member's
   order to equal the shared side's, and §33 applies that same rule
   earlier. Whether unforced text brings the word's object to order 1 before
   the second member arrives is unmeasured. This is a gap in step 5's
   gates that §19 did not catch, not a regression of this round. It belongs
   to item 3's corpus round as a certificate (the raise measured, not
   forced), with one question to Codex below.
3. *The half-work default changes what a `RUN_SLOW` result means.* Recorded
   in Testing.md and todo; the rule for item 0 is that learning claims are
   measured at `RUN_SLOW_FRACTION=1`, and a half-work result states its
   work.

**Decisions for Alec, with recommendations.**

- *Land item 3.0 now.* Recommended. Commit the working tree as it stands
  (test freeze `ee7ca4e920bc`, model and data `6fe273d2f12a7`), naming in
  the message the step 5 source `c4a1d92e3f`, the serializer `caece43fc0`,
  the sweep repair `03fbbf2a7d9c`, the final repair `6fe273d2f12a7` and the
  budget freeze `ee7ca4e920bc`; bump the WikiOracle pointer. The
  alternative, holding for a complete slow pass, waits on November and on
  item 1's runtime work, and the mechanisms are measured.
- *Carry.* The learning records to item 0 as todo already lists them; the
  four runtime failures to item 1 as named cases; the 1-to-2 certificate to
  item 3's corpus round.

**Verdict.** The serializer round and the sweep are approved. With them
item 3.0's five mechanism steps, their one sweep and its scoped repairs are
complete and measured; publication is Alec's commit and the WikiOracle
bump. Codex's next text is one paragraph, below, and no new round starts
before the commit.

**Text for Codex (after Alec's commit).**

~~~
Item 3.0 is landed at Alec's commit. Record it in todo.md's item 3.0 entries
(the current state and the step ledger) with the review pointer (orders
spec §69) and the five hashes the commit names. Then answer one question in
a §70 of the orders spec, reading only, no source change: in unforced text,
"Felix is a cat. Ciao is a cat.", what order do the current readers give the
shared word's object at the first equality, and by which path (if any)
would it reach order 1 before the second member arrives, so that
index_part_row's equal-order rule admits the raise? Name the functions.
Do not run RUN_SLOW. No commit or publication.
~~~


## 70. Landing and the unforced shared-side order — Codex, 2026-10-26

Reading only, on the approved tree: for a newly admitted, unforced “cat”,
`InterpretLayer.lookup_word` creates a **word SYMBOL**, records
`reference_orders[word] = 0`, and leaves it pending without a DEF object;
`no_object` and scalar `forward` therefore return the word itself.
`ConceptualSpace._concept_source_order(word)` delegates to
`ConceptAllocator.order_of(word)` and reads **0**, not 1. If native case
admission has already supplied an object, `InterpretLayer.admit_discovered`
and `define` connect an **object SYMBOL** to that word with a **SYMBOL/SYMBOL
LTM1 DEF row**; a raw percept case is also order 0, while an object with no
record, parts or percept witness reads -1. The text alone does not specify an
already learned object's order. `ReferenceContext.denoted_order` preserves
this operand order: a determiner lowers only an existing order >= 2, equality
returns the left operand's order, and neither writes the right operand's
order. `binding_spec`, `resolve_order` and
`Language.resolve_lexical_references` select at that looked-up order; they do
not manufacture an order-1 object. Thus an order-1 Felix can give the
**equality's LTM1 RELATION handle** order 1 through
`_concept_source_order` -> `denoted_order(equal, ...)` (the right side is in
`singletons`), while the bare shared **word SYMBOL** still reads 0. An
**LTM0 OCCURRENCE** instead reads its stored `order`. `_write` adding the
first equality and marking `singletons` does not record an order for the
shared word; `_part_order` deliberately ignores semantic incoming edges for
a word and reads its native form constituents. There is therefore no
first-equality path that turns this fresh shared word into order 1 before
Ciao arrives: `_observe_equality` rejects a 1-versus-0 witness before
`remember_singleton`, and `index_part_row` requires equal known orders. For
an unrecorded non-word object, an incoming order-1 member is instead a
structural part read by `records`/`_part_order`, yielding 2, not the missing
intermediate 1; a recorded order takes precedence in `order_of`. An order-1
shared object must already have an independent witness: the existing
zero-to-one path is two distinct order-0 equalities through
`_observe_equality` -> `index_part_row` -> `InterpretLayer.define`. The
separate implicit path `promotion_observe` -> `maybe_raise_order` ->
`_assign_concept_parts` -> `promotion_pass` can form an order-1 concept, but
does not itself define “cat” as that fold: `admit_discovered` matches a
reserved word to its native percept-feature case. Neither path supplies an
order-1 shared object automatically at the first of these two sentences.
This is a source reading, not an unforced text measurement; the 1-to-2 corpus
certificate remains carried under §69. No source change or experiment is
proposed here.

Landing approval is §69. The single BasicModel landing commit names step 5
`c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model
repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The 916 measured source
paths still hash exactly to
`ee7ca4e920bc691ae65885228d72ec441c6be73564d9d911766b30e0774533d3`;
only `todo.md`, this spec and the handoff receive landing edits. The serializer
and sweep receipts, including repair-01, repair-03 and repair-04-slow-budget,
remain local and gitignored. Learning, runtime failures and the unforced
raise certificate carry to items 0, 1 and 3 respectively. No tests were run
for the landing.
