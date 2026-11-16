# No space moves another: the principle and where it is broken (Claude, 2026-11-15)

A specification for making one principle hold across the architecture.
It is built in round C of item 1.8, with the parts of speech
(`doc/plans/2026-11-15-parts-of-speech.md`). It has no text for Codex
yet. A rule marked *default* is Claude's choice where Alec has not
spoken.

## 1. Alec's words (2026-11-15)

""a class taking no centre of its members" is actually correct, let's
dig in. A sub should pull its doms together in dom space as a centroid
of the doms. A dom should pull its subs together in sub space. But the
doms location in dom space does not influence the subs location in sub
space. Does that make sense? The centroid defined by a dom or sub is
unrelated to its own position."

"Please ensure this design principle across the architecture by making
sure that the spaces do not have geometric influence on one another."

Standing: "the conceptual space is not shared with perceptual space ...
that bifurcation needs to be complete; we don't want leakage"
(2026-10-20); "I am more in favor of a single conceptual space (so
concrete and abstract concepts all intermingled)" (2026-10-22);
"symbols should not take the centroid of the percepts that they
collect; they are epistemic points" (2026-11-12).

## 2. The principle

**The spaces.** The percept space holds forms. The conceptual space
holds meanings, and each order of it has its own geometry. *Claude's
reading of 2026-10-22 with today's words:* one store and one coordinate
system hold every order's rows, intermingled, but a position of one
order is never moved by, nor measured against, a position of another.

**What may cross between spaces:**

- a membership or definition edge, which says which concepts belong
  together;
- activation and evidence: field values, presence, the two lanes, counts
  of views, trust as a scalar of certainty;
- a seed: a key hashed from content to start an unplaced row.

**What may not:**

- writing a row's position (its codebook row `W`, its placement, its
  direction, or a content vector that becomes its direction) from
  positions of another order or of the other space;
- comparing positions of two orders, or of the two spaces, by any
  likeness (coverage, cosine, distance, nearest);
- a placement step whose context or negatives hold another order's
  rows.

**What a cross-order relation does.** It groups. A word draws its
classes toward their centroid among classes, and a class draws its
words toward their centroid among words (parts of speech, §6). A
concept's centroid of the rows it groups is kept apart from its own
position and lies in their space; it is its extension.

**The form half of an order-zero row.** A word's row holds its form
(the percept space's coordinates) beside its meaning lanes, tied at the
one index (`doc/SymbolFirewall.md`). Writing a percept code into the
form half is the percept space's own position, stored at the shared
index (`PerceptCollections.code`, `MereologicalCodes.derive`); *Claude's
reading:* it is not a crossing, as long as no form is compared with a
meaning and no meaning is written from a form.

## 3. What already holds

Read by a sub-agent across `bin/` on the first machine's working tree
(rounds A and B, uncommitted); three of its findings were read again by
Claude.

- Placement: `MereologicalCodes.place_rows`, its negatives (`_negatives`)
  and its collapse audit all keep to the row's own order.
- Round B removed `ConceptHierarchy.refresh` and `condition`,
  `MereologicalCodes.derive_class_placements`, and the copies of form
  into higher-order rows (`ConceptIndex.index_part_row` and the rest).
- No minting, raising or admission code in `bin/Spaces.py` sets a new
  concept's position from its members.

## 4. Where it is broken, and the rule for each

**Positions written from another order.**

1. `PartsOfSpeech.observe_seat`: an operation's concept (order two) is
   given `W` as the mean of its operand words' points (order zero), and
   that mean goes to its trust. *Rule:* nothing is written to its
   position or its trust content from the words. Its membership over its
   arguments stays. It is placed by its context like any concept.
2. `ConceptHierarchy.observe_member` → `ConceptTrust.observe`: a member's
   direction is written into its class's trust content, and on a chart
   with no placement `ConceptTrust.project` takes that content as the
   class's direction. *Rule:* the trust content of a class is its
   extension, a vector among its members, and is never read as the
   class's direction. The agreement it measures is a scalar of evidence
   and may set the class's radius.
3. `FamiliarTier.promote_identity`: a new identity (order $m+1$) takes
   `W` from the value of its order-$m$ readings. *Rule:* a new concept's
   position starts unplaced, from its seed, and is placed by its context;
   the readings' value is kept as its extension. Counting its occurrences
   as views stays, with no content.
4. `FamiliarTier._resolve_null_token`, `ReferenceClose.commit` (a
   mention's point into an individual's trust) and
   `FamiliarTier.commit_read_words` (an object's value into a word's
   trust, where the word has no basic percept): *Rule:* each adds a view
   as a count; none adds content from another order.
5. `MemoryConsolidation.consolidate`: a concept minted from an LTM row
   (one order above that row's members) takes `W`, trust content and
   agreement from the row's slot. *Rule:* as rule 3.
6. `FamiliarTier.cover_members`, on a chart with no placement: a class's
   `W` and trust content are the fold of its member rows. *Rule:* not
   written from its members; it starts at zero, as a declared argument
   does. *Not determined:* how a row of such a chart is placed at all
   (§5).

**Positions compared across orders.**

7. `ConceptHierarchy.fit`, through `PartsOfSpeech.similarities` and
   `argument_scores` (the chooser's occupation term) and `operation`
   (definiteness): a word's direction against its class's. *Rule:* the
   word against the class's centroid of its words (parts of speech, rule
   5).
8. `PartsOfSpeech.prepare_values` and `value_scores` (the generate walk,
   `GenerateOccupation`): generated values against the arguments'
   directions. *Rule:* as rule 7, against each argument's centroid of its
   words.
9. `FamiliarTier.centroid`, `observe_field`, `witnessed_match`,
   `observe_candidate` and `MediumTermMemory.match_candidate`: centroids
   over support rows of several orders, compared with candidates of any
   order. *Rule:* a support's centroid is taken over rows of one order,
   and a candidate is compared only with candidates of that order.
10. `FieldProjection.stimulate`, used by `ReferenceClose` to recognise a
    percept and by `OutputOccurrences`: a point scored against every
    inventory row of every order. *Rule:* against the rows of one order,
    the order the reader asks for; *default* zero for a percept. The
    form half against the form half stays (both percept space).
11. `FieldProjection.key`, read as a meaning value by `IntuitionField`,
    `NumericPercepts` and `ExtendedRecall`: a centroid over a field's
    rows of every order. *Rule:* a key is taken over one order's rows;
    the reader names the order.
12. `ReferenceClose.score`: candidates are filtered by order only when
    the site names one. *Rule:* candidates are always compared at the
    request's own order (as `doc/Orders.md` §9 already says of `bind`);
    a site with no named order compares each order's candidates apart.
13. `BindingAnswers.cost`: the answer's content against the target
    word's concepts of any order. *Rule:* only against the target's
    concepts of the answer's order.

## 5. Not determined, for Codex to read and bring under §2

- Gradients into `similarity_codebook.W` where it is a parameter (a
  chart with no placement): whether a loss compares or pulls rows of two
  orders.
- `ClauseRow.consequents` and `evaluate`: the orders of the relation's
  endpoints that are compared.
- The key bank of every order built in `ModelAttention` near its read of
  the codebook: what reads it.
- `IndependentComponents.innovation`: whether it is on a live path.
- LTM role points filled from a kind's code (`ClauseRow`): whether they
  reach `consolidate`.

Each is reported with what it does; where it crosses, the rule of §2
applies and the change is listed with its test.

## 6. Certificates

1. *Invariance, for each site of §4.* Before the call, every row of the
   other orders (and the percept space) is moved by a random rotation
   that keeps its own order's geometry; the position written, or the
   score read, is unchanged (atol 1e-6). The same test fails before the
   change.
2. *The groupings within an order* of the parts of speech (their
   certificate 4).
3. *No reader of trust content as a direction:* a test reads the source
   for `ConceptTrust.project` and the rule 2 readers.
4. *Both stopping gates* hold. Learning results and the assessment's
   tables are read before and after and recorded as the baseline (§43
   of the 1.8 specification).

## 7. What it changes

Values change wherever a site of §4 is on the live path: the chooser's
and the generate walk's occupation terms, new concepts' starting
positions, MTM's matching of candidates, recognition in `bind`, the
answer cost, and every reader of a field's key. Nothing in the grammar
file or in the order blocks changes.
