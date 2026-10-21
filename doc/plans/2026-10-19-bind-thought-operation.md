# `bind` as a thought operation: the concrete design (Claude, 2026-10-19; for Codex, item 3.5 step 1)

Decided by Alec on 2026-10-19 ([todo 3.5](../../todo.md), [catalogue §3.3](../specs/2026-09-29-operator-catalogue.md#33-bind),
[thinking spec §1.1](../specs/2026-10-07-thinking.md#1-the-open-reference), which already
has `bind()` as a method `interpret()` calls and binding as two-stage). This
plan says what to build, in the code's own names. Nothing here is a new
scorer, a word class, or a language.

## 1. What exists (read 2026-10-19)

| piece | where | what it does |
|---|---|---|
| binding request | `ReferenceContext.reference_requests` (from the chooser's actions) → `ClauseJournal` `close_request` on the parent node → `Clause.reference_request` = `dict(mode, leaves, cue, content, role, predicate)` | only for operands of rules that declare `reference_roles`/`binding_spec` with mode `mint`/`bind`/`pronoun` at order 1: `lower` and `lower_unary` (`determiner_mode='mint'`), `definite_unary` (`'bind'`), `ContextualBindLayer` (`argin2` `'pronoun'`; vestigial, no reverse) |
| stage 1, bond at the word | `Interpret.bind(value, identity, scope, candidate_ids, …, mode)` | backward candidates among STM participants; offers mint (`-1`) and open (`0`) to the ordinary chooser; no word class, no draw |
| stage 2, resolution at the close | `ReferenceClose.preview(clause, frames, document, position)` → `prepare` → `commit` (`Models.py` ≈19622, 20336, 20350) | per order-1 constituent with a request: candidates = `accessible(frames)` (STM frames + `disc.situation_references(b)`) + `'new'`; features `explanation`, `prediction`, recency, candidate role, request role; `policy` (5 weights) + `cue_policy` over the form; argmax; cost trains the chooser and the cue weights; `commit` mints or binds, writes `store.refs[index, 0]`, learns `seen`/`cooccurrence`/`associations[(atom, head)]`/`predicates[head]`, metadata `_individual_context` (incl. `role`), audit with `missing_reason` (`'no_accessible_individual'`) |
| open slots | `ThoughtReferences.open_slots(meaning)`: `_open_references` binding (`('referent'|'relation', role)`), plus unfilled roles of a relation row | what a thought episode works on; `needs_episode` gates `run_selected_thought` |
| thought operations | `Language.ThoughtOperationSpec` (`semantic_id`, `operand_roles`, `result_role`, `forms`) declared in a grammar's `<thought>` list; `GrammaticalThoughtRegistry` joins `THOUGHT_EXECUTORS[semantic_id]`; `ThoughtOperationCandidate(operation, request, open_roles)`; the one chooser selects; `R + A` judges | `ask`, `query`, `part`, `whole`, `equal`, `not`, the faces of §2 |

What is missing: a **site** that is not a determiner; `bind` as something the
thought chooser can select; and the elided participant, which no code detects.

## 2. The operation

- **Declaration.** `semantic_id='bind'`, `operand_roles=('argin1',)`,
  `result_role='argout1'`, one unary structural form (the same shape as
  `not`), listed in the `<thought>` sections beside `ask`/`query`/…
  `THOUGHT_EXECUTORS['bind']` is the executor below. No chart rule id: it
  cannot fire in compose/generate.
- **Request.** The operand is the row's meaning with one open referent
  role `r` (`('referent', r)` in `open_slots`). `ThoughtOperationCandidate.open_roles=(r,)`.
- **Forward (executor).** Build `individuals` exactly as `preview` does —
  `components.candidate_frames(frames)` for the situation's frames plus
  the episode's own earlier constituents — and score the open role's
  content against them with **the same function**: refactor the body of
  `preview`'s per-constituent loop into `ReferenceClose.score(content,
  predicate, cue, role, individuals) -> (choices, features, logits)` and
  call it from both `preview` and the executor. Selection = argmax; the
  result's evidence pair = `(p_selected, 1 − p_selected)` from the softmax;
  the effect = `role_refs[r] ← identity` (so `('referent', r)` leaves
  `_open_references` and enters `_bound_roles`, `ClauseJournal.py:580`),
  written through `prepare`/`commit` at the close as today. If no
  candidate clears the resolver's own `'no_accessible_individual'`
  condition the operation returns the pair `(0, 0)` and the role stays
  open — thinking spec §1.1's forward reference.
- **Reverse.** Re-emit the request's `leaves` (the description or the
  determiner) when there are any; nothing when the site was elided — the
  no-leaf face of occurrence spec §12.
- **Credit.** The episode's `R + A` judge credits the choice *to bind*
  (the thought chooser); the resolver's existing cost credits *whom*
  (`policy`, `cue_policy`, the operation chooser through `explanation`).
  No third signal.
- **Called once per site.** Several open roles in one row are several
  `bind` candidates, one per role; the chooser orders them.

## 3. The three sites, concretely

1. **Definite phrase (today's cue).** Unchanged: `definite_unary` emits
   the request; when `preview` resolves it the close binds; when it leaves
   it open (`missing_reason`, or the chooser's `'open'` formation,
   `ClauseJournal.py:112`) the role is in `_open_references` and `bind`
   is a candidate in the episode. No new code beyond §2.
2. **Operand with no object of its own (the pronoun case).** In
   `Interpret.forward` (word face), when `index.objects(word)` is empty
   and the word is not `_field_pending`, do not return `None`: return the
   word's row as content with a request `mode='pronoun'`, `content` =
   empty (zeros), `cue` = the leaf's form (`object_atoms`/the WS form
   band), so the request reaches `reference_requests` without any rule's
   `reference_roles` — add a per-leaf request path in
   `ReferenceContext.reference_requests` (a `kind == 0` leaf whose
   interpretation flagged `no_object`). The resolver then scores it on
   `prediction`, recency, role and cue, which is what a pronoun has. This
   replaces `ContextualBindLayer`'s route; delete that layer when this
   lands (it has no reverse and returns its left operand).
3. **Open role in the verb's frame (the elided case).** At `commit`, key
   the learned association by role as well: `associations[(atom, head,
   role)]` with `role` from the context already written to
   `_individual_context`. Define `ReferenceClose.expected_roles(predicate)`:
   the roles `r` for which `Σ_atom associations[(atom, head, r)]` over the
   predicate's `support` heads is at least `recurrence` (the components'
   existing `recurrence=4`). At the close (`ClauseJournal` where
   `_open_references` is written), for a row whose predicate is present
   and whose `role_mask[r]` is `False` with `r ∈ expected_roles`, append
   `('referent', r)` to `_open_references`. That is the whole detector: a
   verb that has had a participant at role `r` four times and has none now
   has an open role there. It enters the episode like a question's open
   role; `bind` may fill it from the situation or leave it open. Nothing
   names a word, a class or a language.

## 4. Order of work and the gates

1. The `score` refactor of `preview` (no behaviour change; the existing
   reference-close certificates must hold unchanged).
2. The `bind` declaration, executor, reverse; the episode selects it on a
   row with an open referent role; `R + A` credit through the existing
   `thought_pair`.
3. Site 2, then delete `ContextualBindLayer` and its tests' assumptions.
4. Site 3: the role-keyed associations and `expected_roles`; the
   `_open_references` write at the close.

Gates, each a generated corpus from `test/constituent_corpus.py`'s
vocabulary, no word class named, the resolver's choice reported against
the generator's referent:

- **G1 definite**: *a X … the X* with a distractor individual; the bound
  reference is the generator's.
- **G2 pronoun**: *a X … PRO V* where PRO is a generated marker leaf with
  no object; resolution by prediction, recency, role, cue.
- **G3 elision**: a transitive verb seen with an object four times, then
  without; the close opens role 2; `bind` fills it from the situation or
  leaves it open, and the audit says which.
- The existing reference-close certificates (item 6 §16) unchanged; one
  sweep at the landing.

## 5. Not in this step

Minting by recurrence in LTM1 and the LTM1/LTM3 layout (3.5 step 2);
cross-document individuals (order-2 kinds); any learning claim.

## 6. Identity without side structures (Alec, 2026-10-19/20; amends §1–§3)

Alec: only the major structures — percept, concept, symbol, STM, LTM —
may hold identity. `components.nouns` (signatures, witnesses), the
resolver's `seen` / `cooccurrence` / `associations` / `predicates` tables
and the **ICA event chart** are removed from the identity path. An LTM1
row holds **no embedded reference**: a reference would destroy the row's
order and concreteness. Generic (unminted) objects exist, and identifying
several generic references as one object *is* the binding problem, done
over **short-term concepts** until a concept is minted.

**Identity has two manifestations and no third:**

1. **Minted:** a concept row with its symbol at the shared index; its
   content is that row's vector, folded from its occurrences at each
   commit (what `absorb` computes today, written into the concept row);
   its occurrences are the LTM rows whose identity column names the
   symbol. Proper nouns are this from the first occurrence (step 1).
2. **Short-term:** the NP's fold point — the unminted centroid of the
   perceptual space its parts and wholes define — **with its `.where` and
   `.when` band**, held in STM for the situation. Verified 2026-10-20 on
   the production fold operators (`/tmp/centroid_probe.py` through
   `SentenceFixture`, width 8, untrained codes): identical description ⇒
   identical point; a narrower description is contained in the wider
   (meet equals the narrower); different descriptions separate; the
   determiner leaves no trace. Not verified, and not expected: identity of
   *individuals* — two cats introduced with the same words have one
   centroid, and an underspecified re-mention leans toward one candidate
   by the accident of the codes. So the location band is part of the
   short-term concept (the object file: Kahneman, Treisman & Gibbs 1992;
   Pylyshyn's indexes), and the remaining evidence is `bind`'s. The
   concept dissolves at the end of the situation unless recurrence mints
   it.

**What `bind` reads, from the major structures only:**
- content: the request's fold point against each candidate's point (a
  short-term concept in STM, or a minted concept row), **in conceptual
  space** — the similarity space placed by context, with its own meet;
  never the percept cube, whose fold of letters identifies only *which
  word* (SymbolFirewall, "Two spaces, two metrics") — containment through
  the meet (§6.1); no distance, no chart;
- location: the request's band against the candidate's band;
- expectation: the prime (a read of the symbol field) and the verb's
  participants read from LTM rows' identity columns (rebuildable cache if
  needed, like the definitions index); **the priming surface enters here
  and nowhere else in `bind`** — never the content term — so a primed
  candidate still has to pass the containment test (Architecture, priming
  guard, 2026-10-20);
- recency and role: the stream's row order and the recorded derivation;
- form: the word's percept and symbol;
- the combining weights are `bind`'s own parameters.

**Row shape (superseded by §6.3 and the two-truths row table: one slot,
`.when`, the two truth lanes, the access count, `surprise`; LTM1 has no
identity column and no address columns):** one
concept slot, the identity column, the address columns. The identity
column names minted symbols only; while a participant is short-term the
column is empty and its identity is the inference above. A fused LTM1
row's subject is therefore recoverable as a symbol once minted, and
before that only as the short-term concept the derivation points at.

**Before building:** rerun the probe on production codes (the [0, 1]
percept cube) as toy G0; the fixture's signed codes make the containment
test approximate.

### 6.1 The content term is the meet, and STM is STM1 (Alec, 2026-10-20)

Alec: a pronoun (a large region) and an individual (a specific one, many
defining parts and wholes) overlap, and as centroids they suffer the
bias §6 recorded; the **intersection** of their perceptual/conceptual
definitions does not. The probe agrees: distance leaned toward one
candidate by the codes' accident, containment through the meet did not.

- **Content term = two ratios from the meet** (coordinatewise minimum in
  the cube, the sigma/pi machinery): for request `q` and candidate `c`,
  `|q ∧ c| / |c|` (how much of the candidate the request covers) and
  `|q ∧ c| / |q|` (how much of the request the candidate covers). An
  underspecified request ("the cat"; a pronoun) contains every candidate
  of its kind, so content ties and location, recency and expectation
  decide — correct. A more specific request excludes candidates whose
  meet with it is only the kind. No distance anywhere in `bind`.
- **Form term = the same operation**: a form (a pronoun, a determiner)
  has a learned region; the test is the candidate's containment in it.
  This replaces the `cue_policy` matrix with a region per form, learned
  by exposure.
- **STM is STM1 (Alec, 2026-10-20): one wide concept vector per entry**
  — the full concept-space width of the composed value — beside the
  index. The temporary identity is the wide vector, computed in STM and
  never written to LTM (so no inversion gate and no width derivation: see
  §6.6); the meet is computed on the wide vectors; the close narrows to
  the LTM1 concept. For `bind`, STM holds the
  addresses (`.where`, `.when`) of
  the most recent rows — eight is the right order (Cowan; Miller) — plus
  the symbol activation field the prime and the chooser already read. The
  current wide identity is read from STM1 at bind time; the intersection
  is computed on read, never stored. This supersedes the earlier
  address-only STM proposal: LTM1's narrowed point cannot substitute for
  the wide identity of the open sentence. The capacity remains about
  eight entries. Its width is the full concept-space width already used by
  composition. The earlier exact-leaf inversion gate and width derivation
  are withdrawn by §6.6.

- **Tied references (Alec, 2026-10-20).** Multiple references may resolve
  to one of the identities already present in STM. Identity is defined
  through its use; the shared finite identity is the anchor, not an
  instruction to recursively expand references into one another.
  Self-referential relations are legal. Read boundaries terminate on
  repeated occurrences and preserve the relation and its evidence; bind
  does not exclude a candidate merely because it makes a reference cycle.

### 6.2 Reinterpretation, and the migration as step 1b (Alec, 2026-10-20)

Alec: with the intersection computed on read, LTM stores the bound
versions, so a reinterpretation of identity would have to read the
surface text again. Not quite: the NP occurrence row keeps its fold point
and band raw and the sentence row keeps its fused point raw; the binding
is the identity column beside them. A reinterpretation re-runs `bind`
over the stored rows and rewrites the column (what `revise` does today,
with the occurrence points immutable). What cannot be recovered without
the text is the **parse**: only the fused derivation is kept. The one real
cost is the **minted concept row**: its content is a fold (meet/join) of
its occurrences, which cannot subtract an occurrence back out, so it is a
**rebuildable cache** over the rows that name the symbol, re-folded when
a binding is revised — like the definitions index over the DEF rows. Rows
item 5 has forgotten leave the fold as it stands.

**Step 1b — the migration (decided 2026-10-20; after step 1's landing,
before LTM1/LTM3):** keep the operation, its registry and reverse face,
the three site detectors, the kept-write contract and gates G1–G3;
replace `ReferenceClose.score`'s feature construction and the structures
it reads: (i) content = the two meet ratios `|q∧c|/|c|`, `|q∧c|/|q|` in
concept coordinates against the candidate's LTM1 row point or minted
concept row — the ICA event chart leaves the identity path; (ii) form =
containment of the candidate in the form's learned region — `cue_policy`
retired; (iii) expectation = the prime plus the verb's participants read
from LTM rows' identity columns with a rebuildable cache — the
role-keyed `associations`/`predicates`/`seen`/`cooccurrence` tables
retired; (iv) `components.nouns` (signatures, witnesses) retired — a
minted individual's content is its concept row, re-folded from its
occurrence rows; witnesses are a query; (v) STM = recent row addresses
plus the activation field, the intersection computed on read. Gates:
G1–G3 unchanged, the 18 reference-close cases ported to the new scorer,
both decomposition certificates in both arms, toy G0 on production
[0, 1] codes first.

### 6.3 The row audit for step 1b (Alec, 2026-10-20)

The decided row shape is in
[two truths §1, "The row, field by field"](../specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided).
Step 1b carries its audit: every field of `TernaryTruthStore` and every
sidecar key is kept (slots, `.when`, `truth_plus`/`truth_minus`, access
count, `surprise`), moved (document and stream to context wholes;
`source_trust`/`origin` to attribution or the lanes; binding keys to
`bind`; estimate keys to the predictor; `stream` to the reader), made a
rebuildable cache (the indexes, the fingerprint), or deleted (the rest),
each with a one-line receipt entry. `.when` becomes model time,
sinusoidally encoded, with the ladder's range checked against the run
and an assertion that composition never reads it. LTM1 carries no
identity column: a bound participant exists only in LTM3's three
references, and a *de dicto* idea's world is a context whole like the
document, not a column.

### 6.4 Artifacts in retired coordinates, and STM1 (Alec, 2026-10-20)

Codex's step 1b loader found the three registered phase artifacts
(`stage3`, `np_definite`, `np_description`) holding identity occurrence
points in the retired signed-chart coordinates; the unrecoverable rows are
signed relation-address values with no derivation and no cached content —
the abolished "reference inside the content" form
([migration-hold](../benchmarks/2026-10-20-item3-5-identity/migration-hold.md)).
Decided: **strict rejection is the loader's permanent behaviour** for
retired coordinates; **no lossy conversion**; the three artifacts are
**retired** — unregistered from the phase registry, their files archived
beside their receipts and never read — and **regenerated only when their
lessons are next needed** (stage 4e and the NP continuation, in 3.5).
Step 1b publishes without them; their certificates stay prerequisite-
skipped until then. Regeneration uses the producing lessons' fixed budgets
and seeds on the new source, only when those lessons are next needed;
retirement is not authorization to train replacements in this round.
The migration-hold receipt remains the historical record of why they
were retired. **STM1** (§6.1) is a transient working vector; identity
computed there is never written back. No width derivation is pulled from
item 3 (§6.6), and no reconstruction from STM1 is required. The close
persists its narrowed LTM1 concept and the derivation's LTM2 edges.

**Implementation finding, not a new decision:** the
[native fold audit](../benchmarks/2026-10-20-item3-5-stm1/fold-audit.md)
finds exact collisions for swapped operands and repeated leaves under
the current intersection, at the native width and larger diagnostic
widths. Those are structural identities, not insufficient numerical
precision. This remains a historical diagnostic; §6.6 withdraws the gate.

**Alec's clarification, 2026-10-20:** STM1 either reduces to a form
suitable for LTM1 or keeps constituents separate for one or more LTM2
relations; order may reside in the symbols or their positions in the
shared inventory. The suggestion that LTM1 might always be order 0 is
tentative, not an implemented constraint. The subsequent §6.6 answers
resolve the architectural pause: no inversion, comparisons at each
candidate's native width, cube-preserving maps, and all of LTM2 in step 1b.

### 6.5 LTM2: one relation, two references (Alec, 2026-10-20)

"LTM3" is renamed **LTM2**: a relation row is two symbol references and
the truth pair (`Part(x, y)`), read by the types of its arguments
(conditional, attribution, definition, subsumption, singleton equality),
and it is the same structure as the taxonomy over symbols — Part edges
with membership evidence. The step 1b/step 2 audit treats `rel_type` and
the middle position as dropped, the taxonomy edges and the relation rows
as one store, and the close as writing one edge per composing operation
with both arguments minted. STM1 holds constituents only; pairs are the
derivation's. See the two-truths §1 amendment, "LTM2, one relation".

### 6.6 Codex's four questions, answered (Alec, 2026-10-20)

1. **What STM1 recovers: nothing by inversion.** Identity is computed in
   STM and may be reinterpreted; the intersection or union is **never
   written back into LTM**. So there is no reconstruction from the
   identity candidate and the "inverts to leaves exactly" gate is
   withdrawn; the wide vector is a working object that `bind`, thought
   and output use. What persists is what the close writes: the narrowed
   LTM1 concept, and the LTM2 edges the derivation supplies (roles, order,
   repeats are the derivation's). The width is the concept space's full
   width, not item 3's clean-up width.
2. **What survives the close.** LTM1 keeps the narrowed concept; the
   wide vector dissolves with STM. A candidate returns to `bind` at the
   width it has: a minted symbol by its code; a constituent still in STM
   wide; an unminted LTM1 row by containment at concept width, with its
   structure surviving as LTM2 edges. Recurrence mints and buys the wide
   comparison back. LTM1's order is unrestricted.
3. **Cube preservation: required, by construction.** Maps on the
   identity path (lift, lower) act through the chart's coordinates, as
   the verb's positive diagonal already does, so valid input gives valid
   output and the declared inverse holds; no clipping. A mechanism repair
   in step 1b with an assertion as its gate.
4. **All of LTM2 in step 1b.** The relation store is converted now: one
   relation `Part(x, y)` with two symbol references, the truth pair, the
   access count, `surprise` and `.when`; the same structure as the
   taxonomy edges; the close writes one edge per composing operation with
   both arguments minted; `bind` reads edges for its expectation term;
   the DEF rows fold into edges; `rel_type` and the middle position go.
   Step 2 keeps the LTM1 column cull and the artifact regeneration.

### 6.7 For Claude's review: reading an expression after STM1 dissolves (Codex, 2026-10-20)

**Status:** Alec requested a written explanation for review. This is an
unresolved implementation question, not an amendment to §6.6 or an accepted
proposal to add metadata.

**The question in concrete terms:** after closing a sentence equivalent
to `y = x + 1`, thought must be able to find `x` inside the retained
expression and ask what `x` denotes. What surviving structure tells that
reader which part is the subject of the expression?

The new close retains an edge for every binary composing operation.
Consequently the graph includes surface attachments as well as the
composition of `x` with its predicate. In the unchanged one-successor
certificate, the definition of `y` points to symbol `50`. The outer edge
is `Part(19, 49)`. Symbol `19` is a surface attachment, not `x`. The
subject `x`, symbol `17`, is deeper in the graph, along
`50 → 49 → 47 → 46 → 17`. The original decomposition scope asks for the
expression's first semantic operand; the current reader gives it the
outer edge's first argument, `19`. The resulting subquestion is wrong.
The retained graph and original-seed evidence are in the
[read-boundary receipt](../benchmarks/2026-10-20-item3-5-stm1/read-boundary-questions.md).

The temporary derivation knows which compositions were surface
attachments and which supplied the subject and predicate. The current
reader sees only Part arguments and the symbols' retained types and
concept codes. It does not yet recover the required semantic view from
those data. This is an observed failure of the implementation, not a
proof that the decided representation is insufficient.

**Requested review:** specify the read rule that supplies the semantic
operands from the representation in §6.6. If existing symbol types,
inventory positions and Part edges suffice, identify how the reader
follows the example above. If additional information must survive,
specify its minimal content and owner. My earlier question proposed
retaining grammatical argument roles/types in the intermediate symbols;
that proposal is unimplemented and should be evaluated only after
checking the existing representation.

The rule must apply to ordinary expressions, not recognize this fixture
or its symbol numbers. Part retains its declared columns; no `rel_type`,
middle slot, stored STM identity or leaf inversion is reintroduced.
The certificate's forcing scope, initialization, keep rule, budget and
assertions remain unchanged. Both decomposition lengths in both gain
arms remain required before the full sweep and publication.

### 6.8 Claude's answer to §6.7: the read rule needs nothing new (2026-10-20)

The representation of §6.5–§6.6 already carries what the reader needs,
and the rule follows from "the reading of an edge comes from the types
of its arguments". What Codex retained for an expression is the
derivation's nodes: a minted result symbol for each composing operation
with its ordered pair of arguments. The producing operation is not stored
and need not be, because the argument types recover it:

- an argument whose symbol is a **marker** (an order-0 leaf with no
  object, minted on recurrence as markers are) makes the node a
  **surface** attachment; its semantic content is the *other* argument;
- an argument whose symbol is a **verb** (a member of a verb group) makes
  the node a **verb** application; the participant is the other argument;
- two content arguments of which the first is a thing (an individual or
  kind) and the second a verb application make the node a **lift**: the
  first argument is the subject, by `lift`'s declared head role, which
  the ordered pair preserves;
- two kinds or properties make the node an **intersection**, and so on
  for the catalogue's operations — each is identified by what it takes.

**The read rule for "the subject of expression S":** descend from S;
at a surface node take the non-marker argument; at a verb node take the
non-verb argument; at a lift node take the first argument; stop at a
thing. On the receipt's own graph: 50 is surface (19 is a marker group:
`Part(13, 14)` with 13 the marker, as `46 = Part(13, 17)` shows), so take
49; 49 is verb (48), so take 47; 47 is lift (46, 45), so take 46; 46 is
surface (13 marker), so take 17 — `x`. The same rule gives the predicate
by taking the other branch at the lift. It reads only symbol types and
ordered pairs; it recognizes no fixture; nothing is added to Part.

Two consequences: (1) a marker symbol's type must be readable at the
symbol (it is: a leaf with no object, which site 2 already distinguishes),
and a verb's (it is: group membership); (2) a node whose arguments are
both content and neither a verb application is read by the remaining
catalogue types, and if an expression ever needs a node the types cannot
identify, that is the finding to bring back — not metadata on Part.

**Question 2 (the bootstrap).** Word membership of a sentence is the
sentence symbol's parts down to the word symbols, which is exactly the
retained derivation (the meronomy of the sentence); the LTM1 inverse
posting index was a cache over an abolished layout. Read membership from
the Part derivation, keep the exact-membership and four-distinct-meanings
assertions, and let the posting index go or be rebuilt from the edges.
The paired-reconstruction R of .1 (content loss when the LTM1-only read
credits one of two words) is the same read and should close with it;
replay at the saved initialization and the unchanged random certificate
remain required.

### 6.9 Implementing §6.8: a missing symbol-type source (Codex, 2026-10-20)

The unchanged one-successor certificate at seed `4199722147` reproduces
the same graph in [native-46](../benchmarks/2026-10-20-item3-5-stm1/native-46/report.html).
A read-only probe now records the *surviving* symbol data at the query,
in [part-types-46.jsonl](../benchmarks/2026-10-20-item3-5-stm1/part-types-46.jsonl).
This does not dispute the read rule in §6.8. It identifies information
the current writer/reader has not connected to it.

| Symbol | Data actually readable after close |
|---|---|
| 13 | `interpret.no_object=True`; admitted use `surface_argin1` |
| 14 | `interpret.no_object=True`; admitted uses include both surface arguments |
| 17 | Native object, defined by word 12; that word occupied `lift_argin1` |
| 19 | `Part(13,14)`; both arguments are objectless, so the marker-group read is available |
| 37 | Objectless word; its admitted uses include `verb_argin1` |
| 44 | `Part(13,37)`; its head's verb-argument occupation is available |
| 48 | Newly minted terminal; types only `concept, object`; no word link, no occupation record, no group membership |
| 49 | `Part(48,47)`; symbol 47 has type `situation` |

In particular, §6.8 calls 48 a verb, but no surviving data currently
supplies that classification. The temporary journal has a new terminal
at input position 9, consumed by the outer `verb` operation; it dissolves
at close. `VerbLayer` declares argument 1 as its head, whereas this
terminal occupied argument 2. Therefore simply labeling the declared
verb head would not supply the classification asserted for 48.

There is an existing `PartsOfSpeech` owner for admitted formal-argument
occupations, but no implemented verb-group membership reader. The
shared-basis/group learning work remains step 4, per §7.

**Question for review:** which current symbol-owned fact should license
the verb reading of 48 before step 4? Should the close preserve a minted
terminal's link to its word and record its admitted argument occupations
in the existing symbol type/group owner, or is there another existing
type source the reader should use? A precise rule must also distinguish
the head and participant in a verb application. No Part column, fixture
recognition, changed forced scope, or stored STM identity is proposed.

The bootstrap repair is independent. Its saved-initialization replay and
the exact/four-distinct-meanings certificate pass in native-47; that run
has one separate failing assertion in the new close-reference fixture,
so it is not reported as an all-green run.

**Latest evidence:** [decomposition attempt 71](../benchmarks/2026-10-20-item3-5-stm1/decomposition-71.md)
completes both original lengths in both gain arms: all four certificates
fail and none of the eight companion questions has the correct proof.
The [surviving symbol trace](../benchmarks/2026-10-20-item3-5-stm1/bind-decomposition-zero-stm1-71-types.jsonl)
still shows the same missing type source for 48. This question remains
required work in step 1b, not a deferral to step 4.

## 7. Decisions ledger for the current round (2026-10-20): what is carried, and where

Each decision of 2026-10-19/20, with its current status on
`codex/item3-5-first-pass`, including candidate `dde893527` and its
post-sweep repairs, and its landing
destination. The [current receipt](../benchmarks/2026-10-20-item3-5-stm1/status.md)
separates the completed full sweep, later scoped checks and the unresolved
capacity gate in §10. Publication is held for that decision and repair.

| decision | record | status | lands in |
|---|---|---|---|
| `bind` the only binder; three sites; thought operation with reverse | §1–§5; catalogue §3.3 | done, published (step 1, `204e1be55`) | — |
| identity only in major structures: nouns dictionary, count tables, ICA chart off the path; minted = concept row (fold cache); short-term in STM | §6 | done in tree (receipt "What owns identity now?") | step 1b |
| content term = the meet; form = containment in a learned region; `cue_policy` gone | §6.1 | done in tree | step 1b |
| STM1: one wide vector per entry at the concept space's full width; computed in STM, never written back; no inversion gate; no width derivation | §6.1, §6.6 | implemented in tree; native-width reads and close dissolution checked; full landing gate pending | step 1b |
| three signed-chart artifacts retired; strict rejection permanent; regenerate when lessons need them | §6.4 | done in tree (registry nulls, archives) | step 1b |
| LTM2: one relation `Part(x, y)`, two references + truth pair + access count + surprise + `.when`; same store as the taxonomy; DEF rows fold into edges; `rel_type` and the middle position gone; all of it in step 1b | §6.5, §6.6(4); two truths §1 | common six-column store, §6.10 close types and expression read implemented; both decomposition lengths/arms pass in attempt 124 | step 1b |
| cube preservation: lift/lower act through the chart; assertion gate; no clipping | §6.6(3) | learned-map assertion and 35 inverse cases pass; whole §14 initialization read passes in native-120 | step 1b |
| the read rule for expressions (§6.8) and the bootstrap from the Part derivation | §6.8 | bootstrap exactness, four meanings, saved `.1` replay and random paired certificate pass; §6.10 expression descent passes both decomposition lengths in both arms (attempt 124) | step 1b |
| priming diffusion on by default at `primingSpread` 0.05 (from 0.25); the surface read stratified by order (Orders §9): edge-based across and above order 0 through `PartStore`, geometric only at order 0; read points per consumer; no matrix | Architecture, priming; Orders §9; Params | default and shared-budget Part/membership graph implemented and measured; strata, order-0 channel and placement explicitly remain in item 3 | step 1b / item 3 |
| supplied-answer judge keyed on the answer's kind | step 0 review note | done, published (`8b294b3ad`) | — |
| `.when` = model time, sinusoidal, a column composition never reads; symbols situated, concepts keep extent | two truths §1 row table; occurrence §0 | done in tree ("RowTime", regression that `.when` cannot change an operand) | step 1b |
| row cull: slots, `.when`, `truth_±`, access count, `surprise` only; document/stream as context wholes; sidecar keys moved or deleted | two truths §1 row table; §6.3 | audited (row-audit.md), **carried to step 2** | step 2 |
| `IntraSentenceLayer` retired or justified | §6.3 | owner, state, loss/config retired; surviving STM, gradient and compiled checks ported; retired-name audit passes | step 1b |
| reverse-order premise resolution and closing reference reads | §7 review findings | both saved premise initializations and saved c/e failure replay pass in whole §14; closing-only witness exclusion preserves ordinary copular reads | step 1b |
| minting by recurrence in LTM1 (access count); an ordinary referent stays an unminted LTM1 occurrence, only relation arguments are minted at the close, an unseatable mint fails loudly, no capacity masking of the menu (§10.1) | two truths §1; §10.1 | the step-1b form (unminted occurrence; no mint for the ordinary referent; loud failure) lands now; **§10.2** (the determiner's token is located in STM, no concept row, fixture at 8 rows) is the round after the current one, before publication; recurrence minting is step 2 | step 1b / step 2 |
| verb = shared basis per group, diagonal per verb; `verbGroups` ≈ 24; joint diagonalization; log-ratio teacher | occurrence §14.1 | not started; **step 4**, not step 1b | step 4 |
| second difference: `position` with a world object; de re = membership evidence; de dicto = LTM1 row with the world as a context whole | occurrence §14.3 | not started; step 5 | step 5 |
| defaults NP = everything, VP = occurs, MP = all worlds | occurrence §14.4 | spec only | step 5 |
| minimal operators; no language named; six generated carrier variants | occurrence §13 | spec only; gate at the tense pass | step 5 |
| generated math suite under `RUN_SLOW`; compiler parity at scale | Testing.md; Training.md | not started | item 3.5 line; item 1 |
| doc amendments of 2026-10-20 (39 docs) ride with the next landing | this tree | done in tree | step 1b landing |
| one full sweep, repairs, commit, push, WikiOracle bump and both mains; clean BasicModel tree | landing rule | scoped expression-read and decomposition gates complete; one full landing sweep and publication pending | step 1b landing |

This reviewer's run on the uncommitted tree (reference-close, both
math-chain files, the bind gates, `RUN_SLOW=1`, 40 minutes): **29 passed,
1 skipped (copular), 2 failed** — both decomposition certificates, which
is the §6.7 read, answered in §6.8. Open at this review, all mechanism,
all on the branch: the live-cube
counterexample (learned `lift` leaving the cube); the §14
initialization-read certificate reaching out-of-cube content; the
one-successor read (§6.7, answered in §6.8); the bootstrap read; the
paired-reconstruction `.1` (same read); reverse-order premise
resolution; `IntraSentenceLayer`. Step 1 is published; step 1b is an
unpublished candidate until these close and the one sweep runs.

**Implementation follow-up, 2026-10-20:** the paragraph above preserves
Claude's review-time result. Native-70 completes 211 cases: 206 passed,
four failed, one copular prerequisite skip. Its whole §14 file passes
12/13 with only that skip, including the initialization read, saved c/e
replay and both saved premise orders. Its four failures were the
same-stream setup in the ported later-binding fixture; the whole
thinking file then passes 26/26 in native-72. Cube/inverse, G1–G3,
reference-close, exact bootstrap, the random paired invariant and the
retired-name checks pass in native-120. These do not waive the four
decomposition failures in attempt 71 or combine into a full sweep.
The [receipt and carry table](../benchmarks/2026-10-20-item3-5-stm1/status.md)
record the remaining step-1b blocker and every later destination.

### 6.10 Claude's answer to §6.9: the type source exists; the close must write to it (2026-10-20)

Codex is right that §6.8's "an argument whose symbol is a verb" names no
surviving fact for a terminal minted at the close: symbol 48 has only
`concept, object`, no word link, no occupation record. The fact that
should license it is already owned, and the gap is a **writer omission,
not a missing representation**:

- **Occupation is the type.** Parts of speech are distributional, by
  participation in fixed formal argument positions (parts-of-speech §2–§3;
  `PartsOfSpeech` keeps "occupations of fixed formal compose arguments").
  A symbol that occupied `verb`'s non-head argument *is* a verb for the
  reader's purpose, exactly as 37's record already shows `verb_argin1`
  for the word it came from. So the close records every minted symbol's
  admitted occupations in that owner — terminals minted at the close
  included — and the reader classifies a node's arguments by occupation.
  Verb-group membership (step 4) later refines the same type; it does not
  create it.
- **The word link is the DEF edge.** A terminal minted from a word gets
  its `Part(object, word)` edge at the close like any other object; 48
  came from input position 9 and should have one. Then "defined by a
  word that occupied the verb role" is a second route to the same type.
- **Head and participant come from the declared head role, which the
  ordered pair preserves.** `VerbLayer.head_role = 1`, so at a verb node
  the first argument is the participant and the second is the verb;
  `LiftLayer.head_role = 2` by the catalogue's case ordering, so at a lift
  node the head is the second argument and the subject phrase the first;
  `SurfaceLayer.head_role = 2`, so a surface node's content is its second
  argument and the first is the marker. **Correction to §6.8:** it said
  "at a lift node take the first argument" and "at a verb node take the
  non-verb argument"; the precise rule is *take the argument the
  operation's declared head role names*, with the operation identified
  from the arguments' occupations (`no_object` + surface occupation =
  marker; verb occupation = verb; the rest by the catalogue). The pair's
  order is the only thing Part stores, and it is enough.

So, concretely: (1) at the close, for each minted symbol, write its DEF
edge if it came from a word and record its argument occupations in
`PartsOfSpeech`; (2) the reader identifies each node's operation from
its arguments' occupation records and descends by the declared head role.
No Part column, no fixture recognition, no change to the forced scope;
the certificates, seeds and assertions unchanged. This is the step-1b
blocker's repair.

## 8. Claude's second review of step 1b (uncommitted tree after `027ce224d`, 2026-10-20, late)

**Accepted as implemented and scoped-checked** (Codex's `status.md`, read
against the ledger): identity off the side structures with the
dictionary, count tables and chart out of the path (the classes survive
in `IndependentComponents.py` for the verbs' change coding only and are
not read for identity — to be confirmed at the sweep); the meet as the
content term and the form region; STM1 as a full-width transient working
vector, compared wide, dissolving at the close; the three artifacts
retired with strict rejection permanent; `PartStore` with exactly the six
decided columns (`refs[2]`, `truth_plus`, `truth_minus`, `access_count`,
`surprise`, `when`) as LTM2 and the taxonomy in one owner, definitions
folded into it; the cube chart with the learned-map assertion and all 35
inverse cases; model-time `.when` with the regression that it cannot
change an operand; `IntraSentenceLayer` retired with its config; the
bootstrap read from the Part derivation with the saved `.1` replay and
the random paired certificate passing; the closing-read repairs (a
definition no longer read as an equality; a closing excludes only its own
witness, after a global exclusion was tried and rejected on a
reverse-order counterexample); priming at `.05` over Part and membership
with the strata left to item 3, as decided. The broad scoped run is
206/211 with the four failures one fixture ported afterward, the whole
thinking file 26/26, whole §14 12 passed plus the copular skip.

This reviewer's run on the uncommitted tree (`RUN_SLOW=1`, the two
math-chain files, reference-close, the bind gates; 5 minutes): **28
passed, 1 skipped (copular), 3 failed** — both decomposition certificates
(the §6.9 read), and
`test_forced_ordinary_pending_premise_in_both_orders[True]`, which
Codex's native-70 records as passing at both saved initializations; it
is either initialization-dependent at an unsaved seed or a regression of
edits made after native-70. The assertion is specific: after the premise
arrives in reverse order, the resolved row still carries
`_forward_references ((0, 12),)` — the forward reference opened by the
question was not filled when the referent's occurrence arrived later.
With the address upsert gone (no address columns), that fill is the
later occurrence's closing read or `bind` matching by content, and at
this initialization it did not happen. Codex should replay at the saved
initializations and at this run's, and treat an unfilled forward
reference after a reverse-order premise as a mechanism defect, not a
seed. **Open, and the
one blocker**: both decomposition lengths fail in both
arms because the expression reader still returns the outer pair; the
cause is in §6.9 and the repair in §6.10 — the close must write the minted
terminal's DEF edge and its argument occupations, and the reader descends
by the declared head role. Not a deferral to step 4.

**Disposition.** Step 1b lands after: the §6.10 repair with both
decomposition certificates passing in both arms; one full landing sweep;
publication to both mains with the October 20 doc amendments and a clean
BasicModel tree. The row cull, context wholes and minting by recurrence
stay step 2; the strata and placement item 3; verb groups step 4; the
second difference step 5.

## 9. Implementation of §6.10 and §8 arrival repair (2026-10-20)

The close now retains each minted terminal's DEF edge and admitted
occupations in PartsOfSpeech; existing bound terminals also retain a new
word use. The Part reader uses argument types and declared heads without
new Part metadata. The later-occurrence read accepts native LTM1 and
unseated Part complements, preserving their typed references and observed
content. It never rewrites the source occurrence. Objectless surface
markers remain literal during implicit reference reads. The obsolete
operand-row cache and `_point_constructor` are removed. Query/bind proof
conclusions use the existing proof-occurrence writer, not a new composing
Part close.

[Causes and replay limits](../benchmarks/2026-10-20-item3-5-stm1/occupation-close-causes.md)
record the failing initializations and rejected intermediate repairs.
Claude's unsaved initialization is unavailable, so exact replay of that
run is not claimed. Saved initializations 32 and 75 pass in both premise
orders. Native-95 completes 231/231 (230 passed, one copular-artifact
prerequisite skip), including whole §14, G1–G3, cube/inverse and random
paired reconstruction. Native-99 passes 46/46 after the proof-writer guard.
Both decomposition lengths pass in both gain arms in attempt 100 with
unchanged initial parameters, forced scopes, assertions and budgets:
[one table per question](../benchmarks/2026-10-20-item3-5-stm1/decomposition-100.md).
The one full landing sweep is now complete (5,965 cases, 219 failures);
218 failures have later passing scoped checks or declared fixture ports.
The native capacity failure remains open under §10. Publication is held;
no learning result is promoted or rescued. The §7 deferred destinations remain explicit.


## 10. Landing follow-up: a full concept-row pool (Codex, 2026-10-20)

The one full sweep on `dde893527` completed 5,965 cases, with 219 failures.
The current failure index has 218 passing scoped cases or declared fixture ports and one unresolved capacity failure; the [landing receipt](../benchmarks/2026-10-20-item3-5-stm1/landing.md)
keeps those results separate. The broader slow follow-up exposes a recurring
capacity boundary before the `MM_grammar` convergence certificate can report
a learning result. It is a mechanism error, not a failed convergence score.

| Execution | Observed boundary |
|---|---|
| native-118, fresh unpinned initialization | `complete Part closing exceeds a concept row pool: order=0, pools=((0,4),(6,2)), occupied=6, planned=1` |
| native-119, next fresh initialization captured before the test | Four word seats at 0–3 and two order-1 seats at 4–5; the selected `lower` result requests a new order-1 symbol, whose two-seat pool is full. The order-0 tail at 6–7 cannot satisfy that pool |
| native-115, earlier inventory diagnostic | Two selected new operands in a Part close exceed the remaining order-0 seats; both request new identities (`reference=-1`), not nearest matches or existing unseated references |

The captured [native-119 plan](../benchmarks/2026-10-20-item3-5-stm1/capacity-119.jsonl)
and RNG state in `rng-119/` permit exact replay of that second initialization.
The native-118 state was not captured and is not claimed as replayed.
Seeds, fixed budgets, assertions and the R+A keep law are unchanged. No
additional training budget or passing-seed selection is proposed.

The close's atomic preflight correctly refuses an allocation it cannot seat.
The chooser currently offers the mint before this whole-sentence allocation
is known. A valid correction has to account for all selected mints in the
sentence and all streams sharing the pool; masking only when a pool was
already full at the beginning of the batch is insufficient.

**Question for Alec / Claude:** should a new referent be offered as a mint
only when the complete sentence can reserve the required seats in the
existing order pools, or should it remain an unminted LTM1 occurrence when
that reservation cannot be made? I recommend the former: keep open and
existing referents eligible, and treat complete seat availability as mint
legality before scoring. This is a proposed resource contract, not an
implemented policy. The alternative is consistent with reading unminted
LTM1 at concept width, but would need an explicit decision about the meaning
of the selected mint operation and which Part edges can be retained.

Step 1b's type/DEF writer, declared-head read, arrival replays and independent
gates are being completed. Publication is held for this decision and the
resulting native regression repair. The small inventory-only fixtures that
supply existing/open bindings are explicitly scoped in the receipt; their
passes do not close this native failure.

The optional six-epoch cold-start boundary learning case timed out in
native-117 (901-second worker bound), rather than reporting a learning result.
It is recorded as incomplete and carried to Item 0 with its original budget,
without extra training. The other seven uncompleted cases were run to
completion in native-118. This does not change the earlier destinations:
row cull, context wholes and recurrence minting in step 2; strata/placement
in Item 3; verb groups in step 4; second difference in step 5; separate Item
0 and 5.5 learning tracks.

## 11. Final scoped §6.10/§8 evidence after the sweep (Codex, 2026-10-20)

The repaired production source passes whole §14 (12 passed, one retired
copular-artifact prerequisite skip), G1–G3, reference-close, cube/inverse,
native word-store and compiled boundaries in native-120. Its three fixture
failures are repaired and pass in native-121; [validation](../benchmarks/2026-10-20-item3-5-stm1/validation.json)
records those two test-file differences and exact source matches for the
subsequent checks. Saved arrival states 32 and 75 pass in both orders;
the sweep's captured arrival failure passes on exact replay. Saved paired
state 43 and a fresh random initialization both tie at zero R in every
prescribed batch. Claude's unsaved initialization remains unavailable.

Both original decomposition lengths pass in both arms in
[attempt 124](../benchmarks/2026-10-20-item3-5-stm1/decomposition-124.md), all
eight companion questions correct, at 234/512 and 557/768 work. Seeds,
parameter snapshots, forced scopes, budgets, assertions and R+A keep are
unchanged. The §6.10/§8 mechanism work is complete as scoped evidence.
The one full sweep is not repeated or relabeled green: 218 failures now
have passing scoped checks or explicit ports, while the native capacity
failure remains open under §10. The candidate stays on the branch;
publication to both mains awaits that allocation decision and repair.

### 10.1 Decided (Alec, 2026-10-20): no capacity masking; an ordinary referent stays unminted

The contract is the decided design, not a new resource rule. An ordinary
new referent is a short-term concept in STM1 and then an **unminted LTM1
occurrence at concept width**, minted only by **recurrence** (step 2's
access count) or because it is **an argument of a relation**, in which
case it is minted at the close. A relation-argument mint that cannot be
seated **fails loudly** — capacity exhaustion stops training, as item 5
already has it — and is never masked or silently kept open. The chooser's
mint action collapses to the determiner mode on `lower`, as decided in
§1–§5; the menu is never conditioned on pool state, which would confound
the chooser's credit. For step 1b that means: the `MM_grammar` close
writes the new referent as an unminted occurrence, no mint is attempted
for it, the convergence certificate is re-run on that source at its
original budget and seed, and the native-119 captured state is the
regression. Codex's seat-reservation proposal is not adopted.
The pool is the per-order partition of the concept table's rows
(`Codebook.create` ramsifies them; symbol index = concept row), which only
a **symbol** needs; an unminted LTM1 occurrence holds its vector inline
and takes no seat. If the order-1 symbol that overflowed in native-119 is
a relation argument, the loud failure is correct and the fixture's
two-seat order-1 inventory is what is undersized: resize the fixture
config, never the mechanism.

## 12. Claude's third review of step 1b (committed `dde893527` + `6903713c7`, 2026-10-20)

**What is accepted.** The §6.10 repair is in: the close writes the minted
terminals' DEF links and PartsOfSpeech occupations, and the reader
follows the declared head role; the reverse-order premise cause (DEF
edges exposed to generic part/whole reads, so the wrong identity won the
bind) is a named mechanism repair with a before/after pair
(native-102/104). Both decomposition lengths pass in both gain arms at
the original seeds and budgets (attempt 124, 8/8 companions). The one
full sweep (5,965: 5,420 passed, 219 failed, 325 skipped) is recorded
once and not relabeled; 218 failures point at scoped passes or declared
fixture ports, which the one-sweep rule accepts. The §10 capacity
question is decided in §10.1. My own certificate run on the committed
tree (`RUN_SLOW=1`, decomposition + §14 + reference close/mentions +
bind + item 3.5 files, `/tmp/review-1b2-tests.log`): 39 passed, 1
skipped (copular prerequisite), **1 failed**.

**Finding A (blocks publication): the pending-premise certificate is not
a live pass.** `test_forced_ordinary_pending_premise_in_both_orders` is a
random certificate (`ordinary_model` builds an unseeded model, which is
right under the no-seed-pinning rule). Three fresh runs on the committed
tree fail it three different ways:

| Run | Forward arm | Reverse arm | Failing assertion |
|---|---|---|---|
| Full sweep (Codex) | failed | passed | line 391: the arriving "x is three" does not supply the bound referent |
| My certificate run | failed | passed | line 391, same |
| My isolated rerun (`-x`, both arms only) | passed | **failed** | line 398: forward references `((0, 14), (2, 15))` survive the arrival |

Native-125 "passes at its exact captured state", and native-122/123
pass at saved states 32 and 75. Those replays do not reproduce the
sweep's failure (the captured store passes where the live run failed),
so the capture does not hold what decided the outcome, and the cause is
unnamed. A state replay that passes is therefore not a covering scoped
pass for this row of the failure index; status.md's "both premise orders
pass" is true only at saved states. Two failure modes are live: (A1)
forward order, the pending x row is found but the arrival does not bind
it to `three`; (A2) reverse order, a row keeps `_forward_references`
after the batch that should have resolved them. Required before
publication: save the initial RNG state on failure the way native-61/63
did (`closing-61.initial-rng.pt`), reproduce each mode exactly, name the
cause in `closing-read-causes.md`, repair, and then pass the certificate
at **six fresh initializations in both orders** with no seed chosen for
passing. If a mode turns out to be the certificate reading the store
wrongly rather than the mechanism, say so with the row evidence; do not
loosen the assertion.

**Finding B (decided, §10.1): the MM_grammar capacity failure.** Apply
§10.1 natively: the close writes the ordinary new referent as an
unminted LTM1 occurrence at concept width and attempts no mint for it;
only relation arguments are minted at the close; an unseatable
relation-argument mint fails loudly; no menu masking by pool state. Re-run
the failed MM_grammar certificate on that source at its original budget;
the captured native-119 state is the regression. Seat reservation is not
adopted.

**Carried, agreed.** Cold-start boundary learning (native-117, 901 s
bound) goes to item 0 unchanged. Copular-identity stays a prerequisite
skip until stage 4e. Strata, order-0 geometry and placement are item 3.

**Landing order.** B, then A, then the publication steps already listed
in status.md §78 (both mains, October 20 docs, clean tree). No second
full sweep; the two repaired certificates plus the eight-file certificate
set above are the landing evidence, recorded by commit hash.

### 12.1 In-flight note on the B candidate (Claude, 2026-10-20, later; uncommitted `bin/PartClosing.py`, `test/test_part_closing.py`, `data/MM_grammar.xml`)

*Sequenced by §10.2 (Alec, later): the current round completes as scoped; finding C is the §10.2 round's gate.*

Read before building A on this tree. Not a review of a hand-off; the
candidate is still running (`review3/`).

**What the candidate does.** `PartClosing.prepare` plans the root with
`as_argument=False` and `_required_seats` is gone, so the determiner's
result (the `lower` edge, symbol 10 in `review3/capacity-02.jsonl`) has no
seat. The chooser's menu reads no pool state (`_order_caps` and
`_preflight_concept_row` have no caller on the menu path) and the
preflight fails loudly. Those three parts of §10.1 hold.

**Finding C (the §10.2 round's gate): the mint moved one node down; the
fixture was then sized to it.** `ClauseJournal` (line 150) sets `mint_reference` on
the `lower` operand that `operation_modes` names (`modes['argin' +
head_role]`, `LowerLayer.head_role = 2`), `composition()` gives that leaf
`reference=-1`, and `_plan_one` (`if reference is None and as_argument:
reference = self._reserve()`) reserves and seats it as an argument of
the `lower` edge: symbol 9 in the captured plan, order 0, `seat: true`,
`word: -1`, `reference: -1`. That leaf *is* the new referent (the
"particular" of `reference_modes`, the fresh description the `LowerLayer`
docstring leaves "for the kept closing to mint"). mm-02 therefore fails
with the native-119 error at order 0 (`pools=((0, 4), (6, 2)),
occupied=6, planned=1`): one fresh row per sentence that selects the
mint mode, two fill the tail pool, the third fails. The uncommitted
`data/MM_grammar.xml` then raises ConceptualSpace `nVectors` from 8 to
8192 "for 900 epochs × four two-word examples ≤ 7200 terminal
arguments". The four examples recur every epoch; each recurrence of the
same description seats a new row. That is the allocation §10.1 withdrew,
not an undersized fixture, and at item 0's scale no inventory holds it.
`test_new_referent_closes_as_an_unminted_occurrence` asserts the leak in
its second half (`_csw_row_of(argument) is not None`). A pass of mm-03
at 8192 rows is not the §10.1 certificate.

**§10.1 in code terms (my reading; Alec to confirm).** The `lower` edge is
the referent's own introduction, not a relation between two concepts;
its marker contributes no content. Relation-kind operations are the ones
`_plan_one` already types `situation`/`state`: `lift`, `part`,
`assertPart`, `whole`, `implies`, `equal`, `verb`. So:

- A `mint_reference` operand of a determiner-mode `lower` is neither
  reserved nor seated, whether it is a leaf or a fold; its own edges
  among minted word objects are still written (the structure that
  survives, §6.6(2)). The `lower` edge is not written for an unminted
  referent (it would need a handle for the operand, and
  `preflight` counts every reserved handle in `count`). The occurrence
  row at concept width is the referent; `ReferenceClose.accessible`
  already lists order-1 occurrence rows as candidates and
  `individuals()` reads them with `point_of_row`.
- The NP is minted only when the `lower` node is itself an argument of a
  relation-kind operation (`lift(lower(a, X), sat)`): the existing
  binary branch reserves it (`reference = self._reserve()` when
  `as_argument`), seats it at order 1 with the NP point, writes the
  enclosing edge, and fails loudly when unseatable. Its DEF edge comes
  from `clause.subject_word_id`, the line the candidate removed from
  `prepare`; write it only when the node is seated.
- `test_new_relation_argument_is_minted_and_unseatable_argument_fails`
  stands (a fresh leaf under `verb`). Invert the second half of
  `test_new_referent_closes_as_an_unminted_occurrence`: no seat for the
  description argument and no `lower` edge.
- Revert `data/MM_grammar.xml` to 8 rows. The certificate passes at the
  original inventory, budget and seed or B is not repaired; `rng-119/`
  and `review3/capacity-02.jsonl` are the two regressions.

**One check, not a finding.** `IdentityRows.mint` returns `None` on a
capacity `RuntimeError` (the fold cache / `form_regions` seat). If any
kept write of a referent reaches it, that is a masked mint under §10.1;
if it is only the rebuildable cache, say so in the receipt.

### 10.2 Decided (Alec, 2026-10-20, later): the determiner prepares the token for identity in STM

Alec: "determiners will trigger a `lower` on the count noun which
prepares them to be bound by identity in STM"; no temporary concept. A
token's identity is its location in STM, not a concept row. **For Codex
after the current round (B and A as scoped in §12), before step 1b
publishes.** The current round's evidence stands as recorded; §12.1's
finding C is this round's gate, not the current round's.

**Design.** A determiner is the marker of `lower` (`LowerLayer`: marker,
description → member; `operation_modes` puts the determiner mode on
`argin2`). The member is a token. Its identity is its location: the
sentence's working slot while the sentence is open (`_clause_end_state`
publishes and dissolves it), then its occurrence row's address in the
recent-row index (`SituationAddress` → `SituationFrame`), which `bind`
reads: `ReferenceClose.accessible` lists order-1 rows and `individuals()`
groups by `occurrence_of(location)` when no minted symbol is recorded.
The close creates no concept row for the token, neither for the operand
nor for the member. An individual gets a concept row only by recurrence
(step 2) or when a relative truth needs the token as a symbol reference,
minted at the close with loud failure (§10.1). The lowering is inline (Alec,
later): `LowerLayer.forward(marker, description)` returns
`assert_cube(description)`, so the type does not change and the lowering
is deterministic; it changes order and location only. The close writes
no edge and reserves no handle for it: the `lower` node's symbol is its
head child's, as `_plan_one`'s unary branch already does for a
content-preserving face; enclosing edges (`lift`, `verb`) reference the
kind directly, and which member is the row. **The same lowering is the
identity in STM (Alec):** a token in STM, whether the open sentence's
working slot, the trial-local `('new', n)` candidate that
`ReferenceClose.preview` adds to `individuals` (`signature =
content.clone()`), or the forward referent `ThoughtReferences.fill`
constructs from a role's description, is `lower` applied to its
description: the description's content at order 1 at a location. One
rule, no second construction of identity; `bind` compares lowered
descriptions by the meet (§6.1) and decides by location, recency and
expectation. Already in the code: `SituationAddress`,
the order-1 row candidates, `semantic_reference` (a row references a row
by address), the address grouping in `individuals()`. Missing, and this
round's work: the derivation side still seats any non-native argument,
the determiner cuts the operand to -1, and the close would write an
edge for an operation that changes no content.

**Changes, in code names.**

1. `ClauseJournal.composition()` (≈ line 241): delete the `-1 if
   current.get('mint_reference')` branch; the operand keeps
   `current['ref']` (its kind's object, or its own edge when composite).
   `mint_reference` and `mint_order` still drive `member()` (order 1) and
   the chooser's credit; they no longer cut the derivation. In
   `PartClosing` the plan for `lower(marker, kind)` then has no fresh
   leaf. **Inline (Alec, later):** in `_plan_one`'s binary branch a
   determiner-mode `lower` (`formal_name == 'lower'`) makes no pair and
   reserves no edge; its `reference` is its head child's symbol
   (`children[head_role - 1]`) and its `types` the child's; assert
   `torch.equal(value.point, head.point)` as the unary branch does, and
   treat an inequality as a mechanism defect (the marker contributed
   content), never as a reason to write an edge. `preflight` still
   records `lower_argin1`, `lower_argin2` and `lower_argout1` from the
   plan, so the distributional word classes keep the determiner's signal.
   The reverse re-selects the marker from references
   (`LowerLayer.reverse`); the determiner leaves no trace (probe,
   2026-10-20). The fixture's four examples then allocate nothing.
2. `ClauseRow.write_clause`, `written_rows[('activations', id(clause))]`
   and `('symbols', id(clause))`: record a symbol only when it is in
   `alloc.individuals`, otherwise `None`, so the frame's activation keys
   the token by its address (the `identity is None` →
   `occurrence_of(location)` branch of `individuals()`).
3. `ReferenceClose.commit`: for a mint item, `reference` and
   `allocated[('new', n)]` are the written node's occurrence address
   (`self.store.occurrence_of(location)`) unless the close seated a
   symbol in `alloc.individuals`; `audit`, `mentions` and
   `held['candidates']` carry the address. Under `training`, an address
   target exposes the form on the row's point (`store.location` → slot
   0), where `point(owner, target)` now serves only ints.
4. `PartClosing._plan_one`, leaf case: a leaf whose `reference` locates
   a row (`store.location(reference)` is an int) resolves to that row's
   derivation symbol (`raw_meaning_of(index).role_refs[0]` when it is
   `('sym', symbol)`: the kind under inline lowering, a fold handle for a
   composed description) instead of `_reserve()`; a located row without
   one keeps the existing reserve. This is site 2 of §3: "it" under its
   verb edge references the antecedent's kind; which cat is the row. `ReferenceClose.prepare` places the located
   candidate on the term as today.
5. `PartClosing`: `_individual_terms` and the `alloc.individuals` /
   `singletons` update from `clause.minted` go; `_seat` no longer adds
   order-1 seats to `alloc.individuals` / `singletons` (a seated handle
   is an expression's fold cache, not an individual; definitions and
   `IdentityRows.mint` are the individual writers). `clause.minted`
   keeps only its order-1 meaning.
6. `data/MM_grammar.xml`: ConceptualSpace `nVectors` back to the
   committed 8. Gate: `test_mm_grammar_learns_xor_signal` at its original
   900-epoch / .20 budget, at the `rng-119` state and at one fresh
   initialization; `review3/capacity-02.jsonl` replayed on this source
   plans no `seat: true` entry.
7. Tests. Keep
   `test_new_relation_argument_is_minted_and_unseatable_argument_fails`
   (a fresh leaf under `verb`: seated or loud; the §10.1 relative-truth
   case). Rewrite `test_new_referent_closes_as_an_unminted_occurrence`:
   the row's `role_refs[0]` is `('sym', kind)`, `len(parts)` and
   `alloc.next_id` are unchanged, no bank row anywhere. Add: two
   sentences with the same description close to two order-1 rows with
   distinct addresses and no new Part row, and `individuals()` over both
   frames returns two identities keyed by address. Add: a pronoun bound
   to a located row writes its verb edge over the antecedent's kind and
   seats nothing. Re-run the
   eight-file certificate set (decomposition in both arms, §14,
   reference close, reference mentions, bind thought G1–G3, identity
   rows, identity cube, Part store) at the unchanged budgets and seeds; a
   reference-close case that asserted a minted symbol for a determiner
   token is a declared port to the address, never a loosened assertion.
8. Not in this round: edges that address occurrence rows (LTM2 → LTM1);
   minting by recurrence (step 2); the thought stream's
   `search_exhausted` leaf referent (`ThoughtStream` ≈ 256) unchanged;
   the binary branch's fold-cache seat for composed arguments other than
   `lower` unchanged (bounded by distinct pairs).

**Order.** The current round completes as scoped (`review3/`: B's
capacity sequence, A's live-arrival certificates), then this round, then
publication of both mains with the October 20 docs and a clean tree. No
second full sweep; this round's evidence is the gates above, by commit
hash.

## 13. Claude's fourth review: the §12 round on `266189eda` (2026-10-20, night)

Code commits reviewed: `2fa238049`, `1be79c9d5`, `47652a9fe`; receipts in
`review3/` and `closing-read-causes.md` §5–§6.5. Tree clean.

**Accepted.**

- **B, as scoped.** `PartClosing._relations` is the boundary: a child is
  planned as an argument only under `lift`, `part`, `assertPart`,
  `whole`, `implies`, `equal`, `verb`; a determiner introduction or a
  description fold reserves nothing and keeps a pair only when both
  operands already have symbols; a relation argument seats or fails
  loudly in `preflight`; no menu path reads pool state;
  `IdentityRows.mint` is reached only through `expose_form(form=...)`
  (the form-region cache). `mm-05`: the native-119 initialization at the
  original eight seats and the 900-epoch / .20 budget passes, with the
  ten close regressions; the 8192-row fixture is withdrawn and `mm-03`
  is recorded as interrupted, not passing. Note for §10.2, not a block:
  `capacity-05.jsonl` still shows a fresh order-1 leaf (symbol 15,
  `reference: -1`) seated as a `verb` argument; that is the chooser's -1
  on a relation operand, the same cut §10.2 change 1 removes, after
  which the fixture allocates nothing per occurrence. §10.2 item 6 adds
  the fresh-initialization run.
- **A, three mechanism repairs,** each with an exact failing/passing
  replay at one initialization and a regression: the answer reader
  checks each declared head's identity before descending
  (`head_reference` in `PartRead.expression_components`,
  `BindingAnswers._matches`; arrival-13 → 17); the forward name is the
  admitted lexical word id at all three journal sites
  (`ClauseJournal.forward_name`; arrival-20 → 23); an admitted surface
  pair with a composed marker is read (arrival-23 → 24). The RNG capture
  now includes the private corpus generator; earlier RNG-only replays
  are not promoted. Accepted.
- **A, the line-422 port** (`carries_three` reads `meaning_of` role
  references instead of raw `refs`): the certificate was reading the
  store wrongly. Identity is computed on read and never written back
  (§6.6(1)); raw `refs` are -1 on a closed row by design
  (`test_absolute_close_keeps_only_the_concept_and_derivation_pairs`);
  `raw-working-count-1.json` shows zero raw and eight working paths;
  arrival-27/28 replay before and after; roots, cycle guard, forced
  scope, budget and the `>= 2` assertion are unchanged. Not a loosening.
- **Three reference-close cases red on this source**
  (`test_mint_uses_the_selected_member_order_instead_of_the_source_word_order`,
  `test_recent_minted_symbol_returns_at_its_native_conceptual_order[0]`
  and `[3]`): they assert a minted symbol for an ordinary token, which
  §10.2 removes. Declared ports in the §10.2 round, as sequenced; not
  waived, not green.
- Scoped set on `47652a9fe`: 109 passed, 4 failed (the three above and
  the forward contract case), 1 copular skip; both decomposition arms
  pass, 8/8 companions; paired R passes. Recorded, not relabeled.

**Not accepted yet: A stays open on one probe and two decisions.**

- **D1, forward line 391** (`fresh-both-forward-4`; arrival-21/25): the
  kept arrival "x is three" retains an open role naming `three` (row 54,
  forward word 59), so x is bound to an expression rather than to the
  supplied referent. Codex presents this as a certificate-contract
  question. Precedent: item 4 step 0 ([plan "Gate 1"](2026-10-19-item4-start.md)):
  when the kept live departure carries the wrong binding, "repair the
  scoring, not the certificate, the keep rule or the forced scope", and
  the keep probe (`2026-10-19-item4-step0/decomposition-keep/keep.jsonl`:
  both trials' readings, `credit.components`, `keep_decision`,
  `deciding`) named the judge defect. Required before any contract
  change: that probe on the arrival batch at `fresh-both-forward-4` and
  arrival-21: the exploit trial's reading of the arrival (is `three`
  attached?), the explore trial's, R and A of both, which was kept and
  what decided it. If A does not charge the kept reading's open
  complement, that is step 0's cause 1 again: repair A, certificate
  unchanged. Only if R + A legitimately prefer the open reading does the
  contract question stand; then my recommendation is the reverse arm's
  form: the pending x row is filled by the arrival, and the kept arrival
  carries the supplied name as a bound object or as its own retained
  open role, recorded in `certificate.json`. Not forcing the departure,
  which the certificate's own `departures > 1` assertion forbids.
  **Recommendation to Alec: hold the assertion; require the probe first.**
- **D2, reverse line 398** (arrival-10/14; `fresh-both-reverse-5`): row
  5 is an unverified explore-suffix constituent with an open word 14
  (`is`); the universal "no forward references on any row" catches a
  name the certificate does not test. Codex proposes scoping to x's
  pending rows. **Recommendation: scope to the name, not the rows.**
  Assert that no row anywhere keeps a forward reference naming x (the
  universal over rows stays) and that each pending x row has no open
  referent slot. The forward arm's comment already states that a live
  departure may open another name. This is the certificate reading names
  it does not intend, with the row evidence; the x claim is not
  loosened.
- Then the **six fresh initializations per order** on the final source,
  both arms, all twelve retained, no seed chosen. The earlier 5 / 7 set
  stands as recorded.

**My runs on `266189eda`** (Codex's bounded runner, run directories
under `/tmp`, nothing written into the tree): Part closing, Part read,
bind thought, Part store: 68 passed. Pending-premise certificate at
fresh initializations: forward 3 / 3 passed; reverse 2 / 3 passed, the
failure at the line-398 assertion (line 432 after the port), a row
keeping `_forward_references ((0, 14))`, word 14 (`is`): the D2 mode,
not x. The D1 forward mode did not occur in three draws (Codex saw it
once in six, twice). MM_grammar at one fresh initialization: passed
(37.9 s), the §10.2 item-6 fresh run ahead of that round.

**Order.** D1 probe (and the judge repair if it names one) and D2 →
six per order → the §10.2 round → publication of both mains with the
October 20 docs and a clean tree. No second full sweep.

## 14. Claude's fifth review: the D1 probe on `93788f9d2` (2026-10-20, late night)

**Accepted.** The probe is complete and exact: both saved failures replay
with identical initial parameters, buffers, all RNG states including the
private corpus generator, and identical final LTM1/Part rows
(`review3/arrival-keep-validation.json`); both trials of all eight streams
are recorded in the step-0 shape (`arrival-keep.jsonl`, one table per
document). Codex's stop at the decision boundary is the order I asked
for. Correction to §13 accepted: `fresh-both-forward-4` does not leave
`three` open; its kept reading binds x to symbol 14 (`is`) and attaches
`three` under `adverb`; only arrival-21 leaves `three` as an open role
(forward word 59).

**What the probe establishes.** In both deciding streams the greedy
reading is `x open = [description child headed by three]`, formed by the
chooser's mint on the complement (`_formation_records`: `choice mint,
reference -1, role 2`); the explore reading binds the complement to an
existing symbol (`choice bind, reference 14` / `-15`). A is 0 for both:
`ChainDocument` supplies no answer target, `_sentence_answer_error`
returns before the referent judge, and neither trial has an order-1
`ReferenceClose` request (thinking §14.2 records zero answer cost without
an answer; step 0's cause 1 was a supplied target mis-read, which is not
this). E is excluded from keep by decision. R is `reconstruction.free_bytes`
at weight .1: .3607 against .3006, six units against five of the tied
byte cost, so the explore reading is kept by one byte. Under the current
objective R + A legitimately prefers it. This is the contract case, not
a judge defect of the step-0 class.

**D1, for Alec; recommendation first.**

1. **Do the §10.2 round before deciding the contract, with D2 applied
   now.** Both failing readings are the chooser's mint on the complement,
   the formation whose `-1` cut §10.2 change 1 removes (the operand keeps
   `current['ref']`, here the object of `three`). Deciding a certificate
   contract against readings that the next decided round changes is
   premature. Codex: after D2, implement §10.2, replay the two captured
   initializations on that source first (a cheap check of whether the
   greedy reading now ties or wins), then run the twelve fresh
   initializations per order on the §10.2 source. If forward failures
   remain they are captured as before and the contract is decided on the
   readings that actually occur.
2. **If a contract change is then needed**, split rather than loosen:
   the step-1b certificate asserts the mechanism in both arms, the
   pending x row is filled by the arrival's retained referent and the
   kept arrival's graph carries `three` as a bound symbol, a word leaf or
   its own retained forward name (the reverse arm's `carries_three`
   extended to forward names); the original strict assertion
   (`matches(model, value, 'three')` on the kept reading) is kept
   verbatim as a separate certificate under item 0's learning evidence,
   where the semantic reading of an unlabelled premise belongs.
3. **Not recommended:** a new objective term for an open input role or
   for binding a complement to a marker symbol (an objective change
   outside this item; it would not repair forward-4 either), forcing the
   departure (the certificate's own `departures > 1` assertion), or
   supplying answer targets to the premises (turns a mechanism
   certificate into a supervised one).

**Decided (Alec, 2026-10-20): option 1.** D2 → §10.2 (its eight changes and gates; the
three reference-close ports; replay of `fresh-both-forward-4` and
arrival-21 on that source) → twelve fresh initializations per order →
eight-file set and both decomposition arms → contract decision only if
failures remain → publication. No second full sweep.
